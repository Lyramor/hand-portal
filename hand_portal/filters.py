"""Filter kamera yang dipasang di dalam portal.

Semua filter menerima potongan frame (roi) dan mengembalikan array baru.
roi dan ctx.frame tidak pernah diubah di tempat, karena pemanggil masih
memakai frame asli untuk menggambar tangan, portal, dan HUD.

Aturan yang dipegang di seluruh berkas ini:

- Keluaran selalu uint8 dan seukuran roi. `apply_filter` menutup semua
  kemungkinan meleset lewat `_finalize`, jadi filter di bawah boleh fokus
  pada tampilan saja.
- roi yang sangat kecil dikembalikan apa adanya. Di bawah 12 piksel sebuah
  efek tidak lagi terbaca, dan memaksakannya hanya memunculkan artefak.
- Animasi memakai `ctx.time_s` dengan pola deterministik (sinus dan hash
  koordinat), bukan acak per frame. Acak per frame terlihat seperti TV
  bersalju, dan itu justru yang paling merusak kesan mahal.
- Pola yang mahal (butir film, tekstur bintang, glyph hujan, medan jarak
  halftone) dibuat sekali lalu di-cache per ukuran. Portal berubah ukuran
  terus, jadi cache dibatasi jumlahnya dan dibuang seluruhnya kalau penuh.
- Setiap filter berakhir di `_polish`. Di situ ada dua hal yang sama untuk
  semua filter: vignette tepi dan butir film tipis. Dua ini yang membuat
  potongan portal terlihat menyatu dengan frame sekitarnya, bukan seperti
  gambar yang digunting dan ditempel. Filter yang butuh kadar berbeda
  memanggil `_polish` dengan angkanya sendiri.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from .contracts import FilterContext, FilterSpec

# --- batas umum -----------------------------------------------------------

# Di bawah ini efek apa pun jadi bubur piksel, lebih baik dikembalikan utuh.
MIN_EFFECT_SIDE = 12
# Batas jumlah entri cache pola. Portal berubah ukuran tiap frame saat
# tangan bergerak, jadi tanpa batas cache akan tumbuh terus.
CACHE_LIMIT = 24

# --- polesan bersama ------------------------------------------------------

# Vignette tepi: makin dekat tepi roi, makin gelap. Ini yang menyatukan
# potongan portal dengan frame aslinya. Tanpa ini, hasil filter terlihat
# seperti tempelan persegi di atas video.
EDGE_VIGNETTE = 0.30
# Lebar gradasi, relatif terhadap sisi pendek roi. Kalau terlalu lebar,
# isi portal jadi ikut gelap; kalau terlalu sempit, terlihat seperti bingkai.
EDGE_VIGNETTE_WIDTH = 0.26
# Vignette dihitung di kanvas kecil lalu diperbesar. Gradasinya lembut, jadi
# resolusi penuh hanya membuang waktu tanpa perbedaan yang kelihatan.
VIGNETTE_BASE_SIDE = 65
# Butir film tipis. Sedikit saja: tujuannya menghilangkan kesan gradasi
# digital yang terlalu bersih, bukan membuat gambar berbintik.
FILM_GRAIN = 5
GRAIN_TILE_SIDE = 256
GRAIN_SEED = 9137

# --- mono -----------------------------------------------------------------

MONO_CONTRAST = 1.24
MONO_BRIGHTNESS = 1
MONO_VIGNETTE = 0.32
MONO_GRAIN = 8

# --- dual-tone ------------------------------------------------------------

# Indigo pekat untuk bayangan, krem hangat untuk sorotan. Dua kutub yang
# cukup berjauhan supaya gradasinya terasa, tapi tidak sampai norak.
DUOTONE_SHADOW = (62, 18, 20)  # BGR indigo keplum
DUOTONE_LIGHT = (168, 228, 255)  # BGR krem dingin
DUOTONE_SHARPEN = 1.40
DUOTONE_CONTRAST = 1.20
DUOTONE_VIGNETTE = 0.34
DUOTONE_GRAIN = 7

# --- pixelate -------------------------------------------------------------

# Jumlah blok di sisi terpanjang. Dihitung dari ukuran roi, bukan angka
# piksel tetap, supaya blob-nya sama besar secara visual di semua ukuran.
PIXELATE_BLOCKS = 32
PIXELATE_CONTRAST = 1.08
# Sedikit dinaikkan supaya blok-bloknya berwarna, bukan abu-abu kusam.
PIXELATE_SATURATION = 1.22
PIXELATE_VIGNETTE = 0.24

# --- invert ---------------------------------------------------------------

INVERT_CONTRAST = 1.12
INVERT_VIGNETTE = 0.26
INVERT_GRAIN = 6

# --- sepia ----------------------------------------------------------------

# Matriks klasik, ditulis dalam urutan BGR karena input OpenCV selalu BGR.
SEPIA_MATRIX_BGR = np.array(
    [
        [0.131, 0.534, 0.272],
        [0.168, 0.686, 0.349],
        [0.189, 0.769, 0.393],
    ],
    dtype=np.float32,
)
SEPIA_CONTRAST = 1.14
SEPIA_VIGNETTE = 0.40
SEPIA_GRAIN = 10

# --- blur -----------------------------------------------------------------

BLUR_KERNEL_RATIO = 0.032
BLUR_KEEP_ORIGINAL = 0.10
BLUR_LIFT = 6
# Kabut hangat tipis di atas blur. Blur polos terlihat seperti kamera rusak;
# sedikit veil membuatnya terbaca sebagai foto soft focus yang disengaja.
BLUR_HAZE = (216, 226, 240)  # BGR
BLUR_HAZE_LEVEL = 0.10
BLUR_VIGNETTE = 0.32

# --- thermal --------------------------------------------------------------

THERMAL_CLIP = 2.6
THERMAL_SMOOTH = 3
THERMAL_CONTRAST = 1.18
THERMAL_SCANLINE_PERIOD = 3
THERMAL_SCANLINE_LEVEL = 246
THERMAL_GRAIN = 6

# --- sketch ---------------------------------------------------------------

SKETCH_KERNEL_RATIO = 0.026
SKETCH_PAPER = (206, 230, 240)  # BGR, kertas agak hangat
SKETCH_INK_DARKEN = 0.92
SKETCH_GRAIN = 7
SKETCH_CONTRAST = 1.45
# Arsir grafit dari nada asli. Versi lama hanya menggambar garis dodge,
# sehingga hasilnya nyaris kertas kosong dengan beberapa goresan tipis.
# Sedikit wash dari nada asli membuat gambarnya terbaca sebagai sketsa.
SKETCH_WASH = 0.28
SKETCH_LINE_GAMMA = 1.9

# --- glitch ---------------------------------------------------------------

GLITCH_RGB_SHIFT = 2
GLITCH_RGB_SWING = 3
GLITCH_BARS = 7
GLITCH_BAR_SWING = 26
GLITCH_SCANLINE_PERIOD = 3
GLITCH_SCANLINE_LEVEL = 232
GLITCH_TINT_A = (255, 90, 200)  # BGR magenta
GLITCH_TINT_B = (220, 255, 120)  # BGR sian
GLITCH_TINT_LEVEL = 0.16
GLITCH_VIGNETTE = 0.22

# --- neon -----------------------------------------------------------------

NEON_GAIN = 1.05
# Pengali normalisasi: batas kontras keras (misalnya tepi kotak) bisa
# menghasilkan magnitudo Sobel ribuan. Tanpa normalisasi, seluruh area tepi
# langsung mentok di 255 dan portal penuh garis putih tebal.
NEON_NORMALIZE = 22.0
# Lutut lunak: nilai tinggi ditekan mendekati 255 tanpa pernah mentok, jadi
# tabung neon tetap punya gradasi di bagian terangnya.
NEON_KNEE = 82.0
NEON_BASE_LEVEL = 0.17
NEON_CORE = (250, 252, 245)  # BGR, hampir putih dengan sedikit sian
NEON_INNER = (255, 214, 60)  # BGR sian
NEON_OUTER = (196, 60, 255)  # BGR magenta
NEON_GRADIENT = 0.14
NEON_VIGNETTE = 0.30

# --- galaxy ---------------------------------------------------------------

GALAXY_SEED = 20240517
# Nebula dihitung di kanvas kecil sekali seumur proses lalu diperbesar.
# Versi lama menghitungnya di resolusi roi setiap kali ukuran portal
# berubah, dan itu yang membuat filter ini 14 ms lebih.
GALAXY_NEBULA_SIDE = 128
GALAXY_STAR_DIVISOR = 260  # satu bintang tiap sekian piksel roi
GALAXY_STAR_MIN = 40
GALAXY_STAR_MAX = 1200
GALAXY_STAR_TWINKLE = 2.1
# Tint dingin untuk badan orang. Versi lama hampir tidak mengubah warna
# (1.14 / 1.02 / 0.96), jadi begitu mask menutupi seluruh portal hasilnya
# kembali terlihat seperti video asli. Sekarang orangnya jelas berwarna
# galaksi, dan bintang tetap ditumpuk di atasnya.
GALAXY_PERSON_HAZE = 0.26
GALAXY_PERSON_LIFT = 0
GALAXY_OVERLAY_STARS = 0.80
GALAXY_OVERLAY_HAZE = 0.20
GALAXY_PERSON_STARS = 0.30
GALAXY_RIM = (255, 196, 150)  # BGR sian pucat
GALAXY_RIM_LEVEL = 0.55
GALAXY_VIGNETTE = 0.30
GALAXY_GRAIN = 6
# Badan orang diwarnai lewat gradasi kecerahan, bukan perkalian matriks warna.
# Perkalian matriks mempertahankan warna asli (kulit tetap kemerahan), jadi
# hasilnya cuma terlihat seperti video yang diberi lapisan ungu. Gradasi
# memaksa bayangan jadi indigo pekat dan sorotan jadi lavender pucat, dan
# itulah yang membuat orangnya benar-benar terbaca sebagai bagian galaksi.
GALAXY_PERSON_SHADOW = (44, 8, 26)  # BGR indigo keunguan
GALAXY_PERSON_LIGHT = (176, 146, 190)  # BGR lavender pucat, sengaja bukan putih
GALAXY_PERSON_CONTRAST = 1.38
# Nebula dipakai sebagai pengali cahaya pada badan: bayangan nebula
# menggelapkan, bagian terangnya menyorot. Rentangnya dijaga di sekitar 1.0
# supaya wajah tidak ikut hilang.
GALAXY_NEBULA_LIGHT_MIN = 0.62
GALAXY_NEBULA_LIGHT_MAX = 1.34

# --- mirror ---------------------------------------------------------------

MIRROR_VIGNETTE = 0.26
MIRROR_SATURATION = 1.18
MIRROR_TINT = np.array(
    [
        [1.06, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.08],
    ],
    dtype=np.float32,
)
MIRROR_GRAIN = 5

# --- halftone -------------------------------------------------------------

# Jarak antar titik cetak, relatif terhadap sisi pendek roi.
HALFTONE_CELL_RATIO = 1.0 / 42.0
HALFTONE_ANGLE = math.radians(45.0)
HALFTONE_MAX_RADIUS = 0.56
HALFTONE_PAPER = (198, 228, 244)  # BGR kertas
HALFTONE_INK = (26, 22, 20)  # BGR tinta
# Nada ditekan sebelum dipetakan jadi radius. Tanpa ini, hampir semua
# wilayah jatuh di radius yang sama dan hasilnya terbaca seperti kain,
# bukan seperti cetakan koran yang punya beda gelap-terang jelas.
HALFTONE_TONE_GAMMA = 1.0
# Rentang nada direntangkan dulu ke 5..95 persentil. Tanpa perentangan,
# satu potongan wajah yang gelap rata berubah jadi satu bidang titik
# seukuran sama, dan gambarnya hilang sama sekali.
HALFTONE_STRETCH_LOW = 5
HALFTONE_STRETCH_HIGH = 95
HALFTONE_GRAIN = 8
HALFTONE_VIGNETTE = 0.22

# --- xray -----------------------------------------------------------------

XRAY_CLIP = 2.2
XRAY_TINT = (170, 225, 110)  # BGR biru-hijau pucat
XRAY_GLOW_TINT = (190, 250, 140)
XRAY_GLOW = 0.46
XRAY_CONTRAST = 1.75
XRAY_VIGNETTE = 0.34
XRAY_GRAIN = 6

# --- vaporwave ------------------------------------------------------------

VAPOR_SHADOW = (110, 22, 78)  # BGR ungu magenta gelap
VAPOR_LIGHT = (210, 246, 250)  # BGR sian pucat
VAPOR_HORIZON = 0.46
VAPOR_GRID_COLOR = (255, 236, 150)  # BGR sian terang
VAPOR_GRID_STRENGTH = 0.62
VAPOR_SUN_TINT = (170, 182, 255)  # BGR merah muda hangat
VAPOR_SUN_LEVEL = 0.42
# Jumlah garis dibatasi, bukan dihitung dari jarak piksel. Versi lama
# memakai fase 1/jarak sehingga untuk roi setinggi 100 piksel garisnya
# berjarak 1-2 piksel dan menutupi seluruh portal dengan pita putih.
VAPOR_COLS_MIN = 4
VAPOR_COLS_MAX = 14
VAPOR_ROWS_MIN = 3
VAPOR_ROWS_MAX = 10
VAPOR_SCROLL = 0.55
VAPOR_SCANLINE_PERIOD = 4
VAPOR_SCANLINE_LEVEL = 238
VAPOR_SPLIT_PX = 1
VAPOR_VIGNETTE = 0.28
VAPOR_GRAIN = 7

# --- edges ----------------------------------------------------------------

EDGES_GAIN = 3.6
EDGES_BASE_LEVEL = 0.12
EDGES_SOFT_COLOR = (200, 232, 255)  # BGR kuning-hangat
EDGES_CORE_COLOR = (255, 250, 238)
EDGES_KNEE = 96.0
# Ambang bawah sebelum diukur: noise sensor di area rata punya magnitudo
# kecil tapi bukan nol, dan tanpa gerbang ini seluruh latar jadi abu-abu
# berbintik seperti TV bersalju.
EDGES_FLOOR = 0.22
EDGES_VIGNETTE = 0.30

# --- posterize ------------------------------------------------------------

POSTERIZE_LEVELS = 5
# Blur agak lebar, bukan sekadar 1 piksel. Pada gradasi yang pelan (dinding
# ke lantai), noise sensor membalik nilai bolak-balik di sekitar batas
# tingkat, dan hasilnya bercak coklat seperti karat. Blur 2 piksel sudah
# cukup untuk menstabilkannya tanpa menghilangkan bentuk.
POSTERIZE_PREBLUR = 2.0
POSTERIZE_SATURATION = 1.25
POSTERIZE_EDGE_LEVEL = 0.30
# Ambang gradien morfologi pada hasil kuantisasi. Nilainya cukup rendah
# karena yang diukur hanya batas antar bidang warna, bukan noise sensor.
# Ambang rendah pada gambar asli menghasilkan bintik; di sini malah bersih.
POSTERIZE_EDGE_LOW = 12
POSTERIZE_VIGNETTE = 0.24
POSTERIZE_GRAIN = 5

# --- dream ----------------------------------------------------------------

DREAM_THRESHOLD = 112
DREAM_BLOOM = 0.70
DREAM_SOFTEN = 0.28
DREAM_VIGNETTE = 0.30
DREAM_WARM = np.array(
    [
        [0.94, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.10],
    ],
    dtype=np.float32,
)
DREAM_LEAK = (210, 190, 255)  # BGR merah muda hangat
DREAM_LEAK_LEVEL = 0.34
DREAM_GRAIN = 6

# --- noise ----------------------------------------------------------------

NOISE_AMOUNT = 18
NOISE_DRIFT = 11.0  # piksel per detik, cukup pelan supaya tidak berkedip
NOISE_DESATURATE = 0.20
NOISE_CONTRAST = 1.08
NOISE_VIGNETTE = 0.34

# --- matrix ---------------------------------------------------------------

MATRIX_SEED = 6161
MATRIX_CELL_W = 16
MATRIX_CELL_H = 22
MATRIX_COLS = 12
MATRIX_ROWS = 14
MATRIX_GLYPH_SCALE = 0.46
MATRIX_CHARS = "0123456789ABCDEFXYZ<>*+=-:."
MATRIX_TAIL = 26.0
MATRIX_GREEN = (120, 255, 96)  # BGR hijau terang
MATRIX_PERSON_GREEN = (86, 214, 96)
MATRIX_PERSON_CONTRAST = 1.18
# Hujan tetap ditumpuk tipis di atas badan, jadi efeknya terasa walau mask
# orang menutupi seluruh portal.
MATRIX_OVERLAY = 0.26
MATRIX_SCANLINE_PERIOD = 3
MATRIX_SCANLINE_LEVEL = 238
MATRIX_VIGNETTE = 0.30
MATRIX_GRAIN = 5

_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

_PATTERN_CACHE: dict[tuple, object] = {}


# --- pembantu umum --------------------------------------------------------


def _too_small(roi: np.ndarray) -> bool:
    """True kalau roi terlalu kecil untuk diberi efek apa pun."""
    return min(roi.shape[0], roi.shape[1]) < MIN_EFFECT_SIDE


def _bgr(roi: np.ndarray) -> np.ndarray:
    """Pastikan array tiga kanal dan contiguous.

    Beberapa fungsi OpenCV (LUT, applyColorMap) menolak array hasil
    pemotongan frame yang barisnya tidak rapat di memori.
    """
    out = roi if roi.ndim == 3 else cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    return out if out.flags["C_CONTIGUOUS"] else np.ascontiguousarray(out)


def _luma(roi: np.ndarray) -> np.ndarray:
    """Kecerahan 8-bit, kontigu, siap dipakai LUT dan colormap."""
    if roi.ndim == 2:
        gray = roi
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return gray if gray.flags["C_CONTIGUOUS"] else np.ascontiguousarray(gray)


def _odd(value: float, lo: int = 3, hi: int = 63) -> int:
    """Bulatkan ke bilangan ganjil di dalam rentang, syarat kernel Gaussian."""
    k = max(lo, min(hi, int(round(value))))
    if k % 2 == 0:
        k = k + 1 if k + 1 <= hi else k - 1
    return k


def _finalize(out: np.ndarray | None, roi: np.ndarray) -> np.ndarray:
    """Jaring pengaman terakhir: bentuk dan tipe keluaran selalu benar."""
    if out is None:
        return roi.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if out.shape != roi.shape:
        out = cv2.resize(out, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
    if out.dtype != np.uint8:
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    # Kalau sebuah filter tanpa sengaja mengembalikan roi itu sendiri,
    # pemanggil bisa jadi menulis di atas frame aslinya. Salin dulu.
    if out is roi or np.shares_memory(out, roi):
        out = out.copy()
    return out


def _cached(key: tuple, builder) -> object:
    """Cache pola mahal per ukuran, dengan pembersihan sederhana saat penuh."""
    try:
        return _PATTERN_CACHE[key]
    except KeyError:
        pass
    value = builder()
    if len(_PATTERN_CACHE) >= CACHE_LIMIT:
        _PATTERN_CACHE.clear()
    _PATTERN_CACHE[key] = value
    return value


@lru_cache(maxsize=32)
def _curve_lut(contrast: float, brightness: int = 0) -> np.ndarray:
    """Kurva kontras lembut (S tipis) dalam bentuk LUT 256.

    Garis lurus `x * contrast` membuat gambar terlihat murah karena
    sorotan langsung putih dan bayangan langsung hitam. S-curve kecil
    menahan kedua ujungnya.
    """
    x = np.arange(256, dtype=np.float32) / 255.0
    y = np.clip(0.5 + (x - 0.5) * contrast, 0.0, 1.0)
    y = 0.65 * y + 0.35 * (y * y * (3.0 - 2.0 * y))
    return np.clip(y * 255.0 + brightness, 0.0, 255.0).astype(np.uint8)


@lru_cache(maxsize=8)
def _knee_lut(knee: float) -> np.ndarray:
    """Kompresi sorotan yang tidak pernah mentok di 255.

    Tepi keras menghasilkan magnitudo Sobel ribuan. Kalau dipotong biasa,
    semua nilai di atas 255 jadi putih rata dan hasilnya pita putih tebal.
    Kurva eksponensial menahannya supaya masih ada gradasi di puncak.
    """
    x = np.arange(256, dtype=np.float32)
    return np.clip(255.0 * (1.0 - np.exp(-x / max(knee, 1.0))), 0.0, 255.0).astype(np.uint8)


@lru_cache(maxsize=32)
def _color_lut(color: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tiga LUT untuk mewarnai satu kanal abu-abu menjadi warna tertentu."""
    ramp = np.arange(256, dtype=np.float32)
    return tuple(np.clip(ramp * (c / 255.0), 0.0, 255.0).astype(np.uint8) for c in color)


def _tint(gray: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Warnai satu kanal abu-abu memakai gradasi linear ke `color`."""
    lb, lg, lr = _color_lut(color)
    return cv2.merge([cv2.LUT(gray, lb), cv2.LUT(gray, lg), cv2.LUT(gray, lr)])


@lru_cache(maxsize=8)
def _vignette_base(strength: float) -> np.ndarray:
    """Peta pengali 0..255 di kanvas kecil: terang di tengah, gelap di tepi.

    Disimpan satu kanal. `cv2.multiply` menerima peta satu kanal untuk
    gambar tiga kanal, jadi tidak perlu menggandakan memorinya.
    """
    side = VIGNETTE_BASE_SIDE
    t = np.linspace(0.0, 1.0, side, dtype=np.float32)
    edge = np.minimum(np.minimum(t, 1.0 - t)[:, None], np.minimum(t, 1.0 - t)[None, :])
    ramp = np.clip(edge / EDGE_VIGNETTE_WIDTH, 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    gain = 1.0 - strength * (1.0 - ramp)
    return np.clip(gain * 255.0, 0.0, 255.0).astype(np.uint8)


@lru_cache(maxsize=24)
def _edge_vignette(shape: tuple[int, int], strength: float) -> np.ndarray:
    """Vignette tepi siap pakai untuk satu ukuran roi.

    Nilai balikannya dipakai sebagai pengali dan tidak boleh diubah
    pemanggil, karena objek yang sama dipakai ulang antar frame.
    """
    base = _vignette_base(round(float(strength), 3))
    height, width = shape
    if base.shape == (height, width):
        return base
    return cv2.resize(base, (width, height), interpolation=cv2.INTER_LINEAR)


@lru_cache(maxsize=8)
def _grain_tile(amount: int) -> np.ndarray:
    """Satu petak butir film berpusat di 128, dibuat sekali per kekuatan."""
    rng = np.random.default_rng(GRAIN_SEED + int(amount))
    noise = rng.normal(0.0, amount * 0.5, (GRAIN_TILE_SIDE, GRAIN_TILE_SIDE))
    return np.clip(noise + 128.0, 0.0, 255.0).astype(np.uint8)


def _grain_layer(
    shape: tuple[int, int], amount: int, origin: tuple[int, int] = (0, 0)
) -> np.ndarray:
    """Butir film seukuran roi, dipatok ke koordinat frame.

    Dipatok ke frame, bukan ke roi, supaya butirnya tidak ikut bergeser
    saat portal bergerak. Butir yang bergerak terbaca sebagai noise sensor.
    """
    tile = _grain_tile(int(amount))
    height, width = shape
    off_y = int(origin[1]) % GRAIN_TILE_SIDE
    off_x = int(origin[0]) % GRAIN_TILE_SIDE
    reps_y = (height + off_y) // GRAIN_TILE_SIDE + 1
    reps_x = (width + off_x) // GRAIN_TILE_SIDE + 1
    big = np.tile(tile, (reps_y, reps_x))
    return big[off_y : off_y + height, off_x : off_x + width]


def _polish(
    out: np.ndarray,
    roi: np.ndarray,
    vignette: float = EDGE_VIGNETTE,
    grain: float = FILM_GRAIN,
    origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Polesan penutup yang dipakai semua filter.

    Urutannya sengaja begini: vignette dulu supaya tepi meredup, baru butir
    ditumpuk. Kalau dibalik, butir di tepi ikut digelapkan dan gradasinya
    terlihat bergerigi.
    """
    out = _finalize(out, roi)
    if vignette > 0.0:
        out = cv2.multiply(out, _edge_vignette(out.shape[:2], vignette), scale=1.0 / 255.0)
    if grain > 0.0:
        out = cv2.addWeighted(
            out, 1.0, _grain_layer(out.shape[:2], int(grain), origin), 1.0, -128.0
        )
    return out


@lru_cache(maxsize=12)
def _scanlines(shape: tuple[int, int], period: int, level: int) -> np.ndarray:
    """Pola garis gelap tiap beberapa baris, meniru layar CRT.

    Satu kanal saja; `cv2.multiply` menerimanya untuk gambar tiga kanal.
    """
    height, width = shape
    rows = np.full(height, 255, np.uint8)
    rows[:: max(1, period)] = level
    return np.repeat(rows[:, None], width, axis=1)


@lru_cache(maxsize=12)
def _vertical_gradient(
    shape: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]
) -> np.ndarray:
    """Gradasi warna dari atas ke bawah, dihitung di kanvas kecil."""
    height, width = shape
    small = np.zeros((2, 1, 3), np.uint8)
    small[0, 0] = top
    small[1, 0] = bottom
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


@lru_cache(maxsize=16)
def _radial_bump(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
    power: float = 2.0,
    strength: float | None = None,
) -> np.ndarray:
    """Cahaya lembut melingkar 0..255, dihitung di kanvas kecil lalu diperbesar.

    `power` mengatur seberapa cepat cahaya meredup dari pusat, `strength`
    mengatur tinggi puncaknya. Dua-duanya opsional supaya pemanggil bisa
    menyebut hanya yang ingin diubah.
    """
    height, width = shape
    side = 33
    peak = 1.0 if strength is None else max(0.0, min(1.0, float(strength)))
    yy = np.linspace(0.0, 1.0, side, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, side, dtype=np.float32)[None, :]
    distance = np.sqrt(
        ((xx - center[0]) / max(radius, 1e-3)) ** 2 + ((yy - center[1]) / max(radius, 1e-3)) ** 2
    )
    bump = np.clip(1.0 - distance, 0.0, 1.0) ** max(power, 0.1)
    small = (bump * peak * 255.0).astype(np.uint8)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def _person_mask_crop(ctx: FilterContext, roi: np.ndarray) -> np.ndarray | None:
    """Potong person_mask (seukuran frame) ke ukuran roi.

    Filter berlatar tidak tahu di mana roi berada di dalam frame, jadi
    posisinya diambil dari kotak pembatas portal.
    """
    if ctx.person_mask is None:
        return None
    mask = ctx.person_mask
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.size == 0:
        return None
    height, width = roi.shape[:2]
    if ctx.portal is not None and ctx.width > 0 and ctx.height > 0:
        x0, y0, bw, bh = ctx.portal.bounds(ctx.width, ctx.height)
    else:
        x0, y0, bw, bh = 0, 0, mask.shape[1], mask.shape[0]
    crop = mask[y0 : y0 + bh, x0 : x0 + bw]
    if crop.size == 0:
        crop = mask
    crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    if crop.dtype != np.uint8:
        crop = np.clip(crop, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(crop)


def _feather(mask: np.ndarray) -> np.ndarray:
    """Lembutkan tepi mask supaya orangnya tidak terlihat seperti guntingan."""
    sigma = max(1.0, min(mask.shape[:2]) * 0.014)
    return _wide_blur(mask, sigma)


def _wide_blur(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Blur lebar lewat penskalaan turun.

    Blur dengan sigma besar di resolusi penuh memakan beberapa milidetik
    untuk hasil yang secara visual sama saja setelah dikecilkan. Turunkan
    dulu ke seperempat, blur di sana, lalu naikkan lagi.
    """
    if sigma < 4.0:
        return cv2.GaussianBlur(gray, (0, 0), sigma)
    height, width = gray.shape[:2]
    factor = 4
    small = cv2.resize(
        gray, (max(1, width // factor), max(1, height // factor)), interpolation=cv2.INTER_AREA
    )
    small = cv2.GaussianBlur(small, (0, 0), sigma / factor)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def _blend_masked(front: np.ndarray, back: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tempel `front` ke `back` memakai mask 0..255 yang sudah dilembutkan."""
    m3 = cv2.merge([mask, mask, mask])
    inv3 = cv2.bitwise_not(m3)
    a = cv2.multiply(front, m3, scale=1.0 / 255.0)
    b = cv2.multiply(back, inv3, scale=1.0 / 255.0)
    return cv2.add(a, b)


def _add_scaled(base: np.ndarray, layer: np.ndarray, amount: float) -> np.ndarray:
    """Tambahkan `layer` ke `base` dengan bobot, tanpa normalisasi ulang.

    `addWeighted(base, 1.0, layer, amount, 0)` berarti penjumlahan, bukan
    campuran, jadi beberapa lapis cahaya bisa saling menumpuk seperti
    cahaya sungguhan.
    """
    return cv2.addWeighted(base, 1.0, layer, amount, 0.0)


def _saturate(bgr: np.ndarray, amount: float) -> np.ndarray:
    """Naikkan kepekatan warna tanpa konversi ke HSV.

    Mencampur balik kanal abu-abu ke warna aslinya cukup satu operasi dan
    hasilnya sama dengan menaikkan saturasi.
    """
    if abs(amount - 1.0) < 0.01:
        return bgr
    gray = cv2.cvtColor(_bgr(bgr), cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(bgr, amount, gray_bgr, 1.0 - amount, 0.0)


def _roi_origin(ctx: FilterContext) -> tuple[int, int]:
    """Posisi kiri-atas roi di dalam frame, kalau portalnya diketahui."""
    if ctx.portal is None or ctx.width <= 0 or ctx.height <= 0:
        return (0, 0)
    x0, y0, _, _ = ctx.portal.bounds(ctx.width, ctx.height)
    return (int(x0), int(y0))


# --- 1. mono --------------------------------------------------------------


def _filter_mono(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Hitam putih dengan kurva kontras film dan sudut yang diredupkan."""
    if _too_small(roi):
        return roi.copy()
    gray = cv2.LUT(_luma(roi), _curve_lut(MONO_CONTRAST, MONO_BRIGHTNESS))
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return _polish(out, roi, MONO_VIGNETTE, MONO_GRAIN, _roi_origin(ctx))


# --- 2. dual-tone ---------------------------------------------------------


@lru_cache(maxsize=8)
def _duotone_luts(shadow: tuple[int, int, int], light: tuple[int, int, int]):
    """Tiga LUT gradasi dua warna, dengan smoothstep supaya tidak bergaris."""
    t = np.arange(256, dtype=np.float32) / 255.0
    t = t * t * (3.0 - 2.0 * t)
    return tuple(
        np.clip(shadow[c] + (light[c] - shadow[c]) * t, 0.0, 255.0).astype(np.uint8)
        for c in range(3)
    )


def _filter_dual_tone(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Dua warna kontras: bayangan indigo, sorotan krem, gradasi halus."""
    if _too_small(roi):
        return roi.copy()
    gray = cv2.LUT(_luma(roi), _curve_lut(DUOTONE_CONTRAST))
    # Pertajam dulu secara lokal. Tanpa ini, area rata seperti dahi dan
    # dinding jadi satu warna saja dan efek duotone terlihat seperti
    # threshold kasar.
    soft = cv2.GaussianBlur(gray, (0, 0), 1.6)
    sharp = cv2.addWeighted(gray, DUOTONE_SHARPEN, soft, 1.0 - DUOTONE_SHARPEN, 0)
    lb, lg, lr = _duotone_luts(DUOTONE_SHADOW, DUOTONE_LIGHT)
    out = cv2.merge([cv2.LUT(sharp, lb), cv2.LUT(sharp, lg), cv2.LUT(sharp, lr)])
    return _polish(out, roi, DUOTONE_VIGNETTE, DUOTONE_GRAIN, _roi_origin(ctx))


# --- 3. pixelate ----------------------------------------------------------


def _filter_pixelate(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Balok besar: turunkan resolusi lalu naikkan lagi tanpa interpolasi."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]
    block = max(2, int(round(max(height, width) / PIXELATE_BLOCKS)))
    small_w = max(1, width // block)
    small_h = max(1, height // block)
    small = cv2.resize(_bgr(roi), (small_w, small_h), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    out = cv2.LUT(out, _curve_lut(PIXELATE_CONTRAST))
    out = _saturate(out, PIXELATE_SATURATION)
    return _polish(out, roi, PIXELATE_VIGNETTE, 0.0)


# --- 4. invert ------------------------------------------------------------


def _filter_invert(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Negatif film: semua kanal dibalik, kontras dirapikan sedikit."""
    if _too_small(roi):
        return roi.copy()
    base = np.arange(256, dtype=np.float32)
    lut = _curve_lut(INVERT_CONTRAST)
    # Balik dulu, baru kurva kontrasnya, supaya hitam tetap hitam.
    inverted = lut[np.clip(255.0 - base, 0, 255).astype(np.uint8)]
    return _polish(
        cv2.LUT(_bgr(roi), inverted), roi, INVERT_VIGNETTE, INVERT_GRAIN, _roi_origin(ctx)
    )


# --- 5. sepia -------------------------------------------------------------


def _filter_sepia(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Nada cokelat tua seperti cetakan foto lama, dengan sudut diredupkan."""
    if _too_small(roi):
        return roi.copy()
    out = cv2.transform(_bgr(roi), SEPIA_MATRIX_BGR)
    out = cv2.LUT(out, _curve_lut(SEPIA_CONTRAST, 4))
    return _polish(out, roi, SEPIA_VIGNETTE, SEPIA_GRAIN, _roi_origin(ctx))


# --- 6. blur --------------------------------------------------------------


def _filter_blur(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Fokus lembut: blur Gaussian kuat, disisakan sedikit detail aslinya."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]
    kernel = _odd(min(height, width) * BLUR_KERNEL_RATIO)
    blurred = cv2.GaussianBlur(_bgr(roi), (kernel, kernel), 0)
    out = cv2.addWeighted(blurred, 1.0 - BLUR_KEEP_ORIGINAL, roi, BLUR_KEEP_ORIGINAL, BLUR_LIFT)
    haze = np.tile(np.array([[[*BLUR_HAZE]]], np.uint8), (height, width, 1))
    out = cv2.addWeighted(out, 1.0, haze, BLUR_HAZE_LEVEL, 0)
    return _polish(out, roi, BLUR_VIGNETTE, FILM_GRAIN, _roi_origin(ctx))


# --- 7. thermal -----------------------------------------------------------


def _filter_thermal(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Peta panas infra merah: detail dinaikkan dulu, baru diberi colormap."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]
    gray = cv2.LUT(_luma(roi), _curve_lut(THERMAL_CONTRAST))
    detail = _CLAHE.apply(gray)
    # Sensor termal asli tidak pernah setajam kamera biasa; sedikit blur
    # membuat hasilnya terasa seperti alat, bukan seperti foto yang diwarnai.
    detail = cv2.GaussianBlur(detail, (THERMAL_SMOOTH, THERMAL_SMOOTH), 0)
    out = cv2.applyColorMap(np.ascontiguousarray(detail), cv2.COLORMAP_INFERNO)
    out = cv2.multiply(
        out, _scanlines((height, width), THERMAL_SCANLINE_PERIOD, THERMAL_SCANLINE_LEVEL),
        scale=1.0 / 255.0,
    )
    return _polish(out, roi, EDGE_VIGNETTE, THERMAL_GRAIN, _roi_origin(ctx))


# --- 8. sketch ------------------------------------------------------------


def _filter_sketch(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Pensil di atas kertas: garis color dodge plus arsir grafit."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]
    gray = cv2.LUT(_luma(roi), _curve_lut(SKETCH_CONTRAST))
    kernel = _odd(min(height, width) * SKETCH_KERNEL_RATIO, 3, 41)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (kernel, kernel), 0)
    # Angka 256 di sini adalah skala dodge, bukan angka ajaib warna.
    dodge = cv2.divide(gray, cv2.bitwise_not(blurred), scale=256)
    # Garis dodge dinaikkan kontrasnya dulu supaya tintanya benar-benar hitam
    # di bagian tergelap, bukan abu-abu pucat.
    lines = cv2.LUT(dodge, _curve_lut(SKETCH_LINE_GAMMA))
    # Arsir grafit diambil dari nada asli, dibalik lalu dikurangkan. Tanpa ini
    # area rata seperti pipi dan dahi cuma jadi kertas kosong, dan gambarnya
    # tidak terbaca sebagai sketsa.
    wash = cv2.convertScaleAbs(cv2.bitwise_not(gray), alpha=SKETCH_WASH, beta=0)
    out = _tint(cv2.subtract(lines, wash), SKETCH_PAPER)
    # Tinta jangan sampai putih bersih: kertas pensil selalu sedikit kotor.
    out = cv2.convertScaleAbs(out, alpha=SKETCH_INK_DARKEN, beta=6)
    return _polish(out, roi, 0.20, SKETCH_GRAIN, _roi_origin(ctx))


# --- 9. glitch ------------------------------------------------------------


def _filter_glitch(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Sinyal rusak: kanal RGB bergeser dan beberapa baris melompat."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]
    time_s = ctx.time_s
    out = _bgr(roi).copy()

    # Pergeseran kanal memakai sinus, jadi gerakannya mengalir bolak-balik
    # dan tidak pernah melompat tiap frame.
    shift = max(1, int(round(GLITCH_RGB_SHIFT + GLITCH_RGB_SWING * math.sin(time_s * 2.3))))
    out[:, :, 2] = np.roll(out[:, :, 2], shift, axis=1)
    out[:, :, 0] = np.roll(out[:, :, 0], -shift, axis=1)

    tint_a = np.tile(np.array([[[*GLITCH_TINT_A]]], np.uint8), (height, width, 1))
    tint_b = np.tile(np.array([[[*GLITCH_TINT_B]]], np.uint8), (height, width, 1))
    for i in range(GLITCH_BARS):
        phase = i * 1.7
        base = i / GLITCH_BARS
        drift = 0.11 * math.sin(time_s * 0.7 + phase)
        y0 = int(((base + drift) % 1.0) * height)
        bar_h = max(
            2,
            int(height / GLITCH_BARS * (0.35 + 0.45 * (0.5 + 0.5 * math.sin(time_s * 1.3 + phase)))),
        )
        y1 = min(height, y0 + bar_h)
        if y1 - y0 < 2:
            continue
        offset = int(round(GLITCH_BAR_SWING * math.sin(time_s * 1.9 + phase * 2.1)))
        if offset:
            out[y0:y1] = np.roll(out[y0:y1], offset, axis=1)
        # Sebagian baris diberi warna siaran rusak supaya bukan cuma geseran
        # abu-abu. Dipilih dari fase waktu, jadi polanya bergerak halus.
        if 0.5 + 0.5 * math.sin(time_s * 1.1 + phase * 3.3) > 0.72:
            tint = tint_a if i % 2 == 0 else tint_b
            band = out[y0:y1]
            out[y0:y1] = cv2.addWeighted(band, 1.0, tint[: y1 - y0], GLITCH_TINT_LEVEL, 0)

    out = cv2.multiply(
        out, _scanlines((height, width), GLITCH_SCANLINE_PERIOD, GLITCH_SCANLINE_LEVEL),
        scale=1.0 / 255.0,
    )
    return _polish(out, roi, GLITCH_VIGNETTE, 0.0)


# --- 10. neon -------------------------------------------------------------


def _filter_neon(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Tepi menyala: dua lapis glow (sian di dalam, magenta di luar)."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]
    side = min(height, width)
    gray = cv2.GaussianBlur(_luma(roi), (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    # Normalisasi dari rata-rata, lalu lutut lunak. Dua langkah ini yang
    # mengubah pita putih tebal menjadi tabung neon yang punya gradasi.
    mean = float(magnitude.mean())
    scale = NEON_GAIN * 255.0 / (mean * NEON_NORMALIZE + 1.0)
    edges = np.clip(magnitude * scale, 0.0, 255.0).astype(np.uint8)
    edges = cv2.LUT(edges, _knee_lut(NEON_KNEE))
    # Tabung neon punya ketebalan, bukan garis satu piksel.
    core = cv2.dilate(edges, np.ones((2, 2), np.uint8))

    wide = _wide_blur(edges, max(1.5, side * 0.022))
    mid = cv2.GaussianBlur(edges, (0, 0), max(1.0, side * 0.007))

    # Latar tetap memperlihatkan gambar aslinya, hanya digelapkan. Kalau
    # latarnya hitam total, hasilnya terasa seperti filter bawaan editor.
    out = cv2.convertScaleAbs(_bgr(roi), alpha=NEON_BASE_LEVEL, beta=0)
    out = _add_scaled(out, _tint(wide, NEON_OUTER), 0.55)
    out = _add_scaled(out, _tint(mid, NEON_INNER), 0.60)
    out = _add_scaled(out, _tint(core, NEON_CORE), 0.55)

    gradient = _vertical_gradient((height, width), (255, 150, 210), (210, 120, 255))
    out = cv2.addWeighted(out, 1.0, gradient, NEON_GRADIENT, 0.0)
    return _polish(out, roi, NEON_VIGNETTE, FILM_GRAIN, _roi_origin(ctx))


# --- 11. galaxy -----------------------------------------------------------


@lru_cache(maxsize=1)
def _galaxy_nebula() -> np.ndarray:
    """Nebula bir-ungu dengan pita diagonal dan inti terang.

    Dibuat sekali di kanvas kecil. Versi sebelumnya menghitung gradasi ini
    di resolusi roi setiap kali ukuran portal berubah, dan biayanya 15 ms
    untuk roi pendek sampai 60 ms untuk 720p.
    """
    side = GALAXY_NEBULA_SIDE
    rng = np.random.default_rng(GALAXY_SEED)
    coarse = rng.random((9, 9)).astype(np.float32)
    medium = rng.random((20, 20)).astype(np.float32)
    fine = rng.random((44, 44)).astype(np.float32)
    cloud = (
        cv2.resize(coarse, (side, side), interpolation=cv2.INTER_CUBIC) * 0.50
        + cv2.resize(medium, (side, side), interpolation=cv2.INTER_CUBIC) * 0.32
        + cv2.resize(fine, (side, side), interpolation=cv2.INTER_CUBIC) * 0.18
    )
    yy = np.linspace(-1.0, 1.0, side, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, side, dtype=np.float32)[None, :]
    core = np.exp(-((xx * 1.25) ** 2 + (yy * 1.7) ** 2) * 2.2)
    # Pita galaksi melintang diagonal: ini yang membuat latarnya langsung
    # terbaca sebagai galaksi, bukan sekadar kabut biru.
    band = np.exp(-(((xx * 0.62 + yy * 0.78) / 0.34) ** 2))
    # Nebula dijaga redup dan lebar. Versi yang terlalu terang dan sempit
    # terbaca sebagai noda ungu di satu sudut, bukan sebagai langit.
    density = cloud * (0.34 + 0.62 * band) + core * 0.34
    blue = 12.0 + density * 86.0 + band * 20.0
    green = 8.0 + density * 38.0 + core * 14.0
    red = 14.0 + density * 72.0 + core * 26.0
    return cv2.merge(
        [
            np.clip(blue, 0, 255).astype(np.uint8),
            np.clip(green, 0, 255).astype(np.uint8),
            np.clip(red, 0, 255).astype(np.uint8),
        ]
    )


def _galaxy_background(shape: tuple[int, int]) -> np.ndarray:
    """Nebula yang sudah diperbesar ke ukuran roi."""
    height, width = shape
    return cv2.resize(_galaxy_nebula(), (width, height), interpolation=cv2.INTER_LINEAR)


@lru_cache(maxsize=8)
def _nebula_light(shape: tuple[int, int]) -> np.ndarray:
    """Nebula yang dipetakan jadi pengali cahaya 0..255 di sekitar 1.0.

    Dipakai untuk menerangi badan orang mengikuti struktur nebula, bukan
    rata. Hasilnya seperti orang yang benar-benar berada di dalam awan
    galaksi: ada bagian yang tersorot dan ada yang tenggelam.
    """
    nebula = _galaxy_background(shape)
    gray = cv2.cvtColor(nebula, cv2.COLOR_BGR2GRAY)
    # Normalisasi ke 0..1 memakai rentang nebula itu sendiri, bukan 0..255.
    # Tanpa ini seluruh peta jatuh di ujung gelap dan wajahnya jadi hitam.
    low = float(gray.min())
    high = float(gray.max())
    spread = max(high - low, 1.0)
    level = (gray.astype(np.float32) - low) / spread
    gain = GALAXY_NEBULA_LIGHT_MIN + level * (GALAXY_NEBULA_LIGHT_MAX - GALAXY_NEBULA_LIGHT_MIN)
    return np.clip(gain * 255.0, 0.0, 255.0).astype(np.uint8)


def _galaxy_stars(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Posisi, kecerahan, fase kelap-kelip, dan tanda bintang besar.

    Disimpan sebagai daftar titik, bukan citra penuh: bintang cuma ratusan,
    dan menghitung sinus untuk seluruh roi tiap frame hanya membuang waktu.
    """
    height, width = shape
    rng = np.random.default_rng(GALAXY_SEED + 1)
    count = int(np.clip(height * width / GALAXY_STAR_DIVISOR, GALAXY_STAR_MIN, GALAXY_STAR_MAX))
    ys = rng.integers(0, height, count)
    xs = rng.integers(0, width, count)
    bright = rng.integers(80, 256, count).astype(np.float32)
    phase = rng.random(count).astype(np.float32) * math.tau
    big = rng.random(count) < 0.06
    return ys, xs, np.stack([bright, phase], axis=1), big


def _galaxy_star_layer(shape: tuple[int, int], time_s: float) -> np.ndarray:
    """Lapisan bintang satu kanal untuk satu frame."""
    ys, xs, star_data, big = _cached(("galaxy_stars", shape), lambda: _galaxy_stars(shape))
    twinkle = 0.60 + 0.40 * np.sin(time_s * GALAXY_STAR_TWINKLE + star_data[:, 1])
    # Dinaikkan karena halo di bawah ini memakan sebagian kecerahannya.
    level = np.clip(star_data[:, 0] * twinkle * 1.25, 0.0, 255.0).astype(np.uint8)
    layer = np.zeros(shape, np.uint8)
    layer[ys, xs] = level
    # Bintang paling terang diberi satu piksel tambahan supaya ada hierarki
    # ukuran; tanpa itu semua bintang terlihat sama besar dan terasa datar.
    for y, x, value in zip(ys[big], xs[big], level[big]):
        cv2.circle(layer, (int(x), int(y)), 1, int(value), -1, cv2.LINE_AA)
    # Halo ditambahkan, bukan menggantikan inti. Kalau seluruh lapisan
    # diburamkan, bintang satu piksel kehilangan hampir semua cahayanya dan
    # latarnya jadi kosong.
    halo = cv2.GaussianBlur(layer, (0, 0), 1.8)
    return cv2.addWeighted(layer, 1.0, halo, 0.75, 0.0)


def _filter_galaxy(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Bintang dan nebula di belakang orang, dengan tint dingin di badannya."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]
    shape = (height, width)
    background = _cached(("galaxy_bg", shape), lambda: _galaxy_background(shape))
    stars = _tint(_galaxy_star_layer(shape, ctx.time_s), (255, 244, 236))

    # Orangnya diwarnai lewat gradasi kecerahan. Di dalam portal biasanya
    # seluruh isinya badan orang, jadi tanpa langkah ini filter galaxy tidak
    # terlihat sama sekali. Ujung terang gradasi sengaja tidak dipasang di
    # 255: wajah yang kena cahaya kamera akan langsung putih dan detailnya
    # hilang, dan itu justru membuat hasilnya terlihat seperti video yang
    # kelebihan ekspos, bukan seperti potret galaksi.
    gray = cv2.LUT(_luma(roi), _curve_lut(GALAXY_PERSON_CONTRAST))
    lb, lg, lr = _duotone_luts(GALAXY_PERSON_SHADOW, GALAXY_PERSON_LIGHT)
    person = cv2.merge([cv2.LUT(gray, lb), cv2.LUT(gray, lg), cv2.LUT(gray, lr)])
    # Nebula dipakai sebagai pengali cahaya, bukan ditambahkan rata. Dengan
    # cara ini strukturnya ikut terbaca di atas badan, jadi orangnya terasa
    # berada di dalam galaksi dan bukan sekadar diberi lapisan ungu.
    person = cv2.multiply(person, _nebula_light(shape), scale=1.0 / 255.0)
    person = cv2.addWeighted(person, 1.0, background, GALAXY_PERSON_HAZE, 0.0)

    mask = _person_mask_crop(ctx, roi)
    if mask is None:
        # Tanpa segmenter, nebula dan bintang tetap ditumpuk di seluruh roi.
        out = cv2.addWeighted(person, 0.88, background, 0.42, 4.0)
        out = _add_scaled(out, stars, GALAXY_OVERLAY_STARS)
        return _polish(out, roi, GALAXY_VIGNETTE, GALAXY_GRAIN, _roi_origin(ctx))

    feathered = _feather(mask)
    scene = cv2.add(background, stars)
    # Tepi badan diberi rim tipis supaya orangnya terpisah dari latar.
    rim = cv2.subtract(feathered, mask)
    scene = _add_scaled(scene, _tint(rim, GALAXY_RIM), GALAXY_RIM_LEVEL)
    # Bintang tetap terlihat di atas badan, tapi lebih tipis daripada di
    # latar. Kalau ditumpuk rata, wajahnya jadi penuh titik terang dan
    # orangnya berubah jadi hantu berbintik.
    person_stars = cv2.multiply(
        stars, cv2.merge([feathered] * 3), scale=GALAXY_PERSON_STARS / 255.0
    )
    person = _add_scaled(person, person_stars, 1.0)
    out = _blend_masked(person, scene, feathered)
    out = cv2.addWeighted(out, 1.0, background, GALAXY_OVERLAY_HAZE, 0.0)
    return _polish(out, roi, GALAXY_VIGNETTE, GALAXY_GRAIN, _roi_origin(ctx))


# --- 12. mirror -----------------------------------------------------------


def _filter_mirror(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Kaleidoskop: seperempat kiri-atas dicerminkan ke empat arah."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]
    base = _bgr(roi)
    quad = base[: (height + 1) // 2, : (width + 1) // 2]
    top = np.concatenate([quad, quad[:, ::-1]], axis=1)[:, :width]
    out = np.concatenate([top, top[::-1]], axis=0)[:height]
    out = np.ascontiguousarray(cv2.transform(out, MIRROR_TINT))
    out = _saturate(out, MIRROR_SATURATION)
    return _polish(out, roi, MIRROR_VIGNETTE, MIRROR_GRAIN, _roi_origin(ctx))


# --- 13. halftone ---------------------------------------------------------


def _halftone_period(shape: tuple[int, int]) -> int:
    """Sisi petak pola titik, dalam piksel.

    Untuk kisi 45 derajat, pola berulang tepat setiap P piksel bila jarak
    antar titik = P / akar(2). Dengan begitu petaknya bisa diulang tanpa
    jahitan, dan medan jarak tidak perlu dihitung seukuran roi.
    """
    height, width = shape
    return max(6, int(round(min(height, width) * HALFTONE_CELL_RATIO * math.sqrt(2.0))))


@lru_cache(maxsize=16)
def _halftone_tile(period: int) -> np.ndarray:
    """Medan jarak ke pusat titik dalam satu petak yang bisa diulang."""
    yy, xx = np.mgrid[0:period, 0:period].astype(np.float32)
    # cos(45) = sin(45) = 1/akar(2), jadi (x + y) / period sudah berupa
    # koordinat ternormalisasi tanpa perkalian matriks.
    u = (xx + yy) / period
    v = (-xx + yy) / period
    du = u - np.round(u)
    dv = v - np.round(v)
    distance = np.sqrt(du * du + dv * dv) / 0.7071
    return np.clip(distance * 255.0, 0.0, 255.0).astype(np.uint8)


@lru_cache(maxsize=4)
def _halftone_radius_lut() -> np.ndarray:
    """Nada 0..255 menjadi radius titik 0..255, sekali seumur proses."""
    tone = np.arange(256, dtype=np.float32) / 255.0
    # Gamma menggeser rentang nada ke arah gelap, jadi perbedaan tipis di
    # area terang tetap terlihat sebagai beda ukuran titik.
    tone = np.power(np.clip(tone, 0.0, 1.0), HALFTONE_TONE_GAMMA)
    radius = np.sqrt(np.clip(1.0 - tone, 0.0, 1.0)) * HALFTONE_MAX_RADIUS
    return np.clip(radius * 255.0, 0.0, 255.0).astype(np.uint8)


def _filter_halftone(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Titik-titik cetak koran: gelap jadi titik besar, terang jadi titik kecil."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]
    period = _halftone_period((height, width))
    tile = _halftone_tile(period)
    field = np.tile(tile, (height // period + 1, width // period + 1))[:height, :width]

    gray = _luma(roi)
    # Nada dihaluskan seukuran sel supaya ukuran titik berubah mulus, lalu
    # direntangkan ke rentang penuh dan dipetakan lewat LUT.
    tone = cv2.GaussianBlur(gray, (0, 0), max(1.0, period * 0.5))
    low, high = np.percentile(tone, (HALFTONE_STRETCH_LOW, HALFTONE_STRETCH_HIGH))
    span = max(float(high) - float(low), 1.0)
    stretched = np.clip((tone.astype(np.float32) - low) * (255.0 / span), 0.0, 255.0)
    radius = cv2.LUT(stretched.astype(np.uint8), _halftone_radius_lut())
    dots = cv2.compare(field, radius, cv2.CMP_LE)

    out = np.empty((height, width, 3), np.uint8)
    out[:] = np.array(HALFTONE_PAPER, np.uint8)
    ink = np.tile(np.array([[[*HALFTONE_INK]]], np.uint8), (height, width, 1))
    cv2.copyTo(ink, dots, out)
    return _polish(out, roi, HALFTONE_VIGNETTE, HALFTONE_GRAIN, _roi_origin(ctx))


# --- 14. xray -------------------------------------------------------------


def _filter_xray(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Negatif biru-hijau seperti hasil rontgen, dengan glow tipis."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]
    inverted = cv2.bitwise_not(_luma(roi))
    # Kontras dinaikkan sebelum CLAHE. Tanpa ini negatif dari video kamera
    # jatuh di rentang abu-abu sempit dan hasilnya terlihat seperti foto yang
    # dicuci, bukan seperti lembaran rontgen.
    inverted = cv2.LUT(inverted, _curve_lut(XRAY_CONTRAST))
    detail = _CLAHE.apply(inverted)
    soft = cv2.GaussianBlur(detail, (0, 0), 1.1)
    # Unsharp ringan: rontgen asli punya tepi tulang yang tegas.
    core = cv2.addWeighted(detail, 1.3, soft, -0.3, 0)
    out = _tint(core, XRAY_TINT)
    glow = _wide_blur(core, max(2.0, min(height, width) * 0.018))
    out = _add_scaled(out, _tint(glow, XRAY_GLOW_TINT), XRAY_GLOW)
    return _polish(out, roi, XRAY_VIGNETTE, XRAY_GRAIN, _roi_origin(ctx))


# --- 15. vaporwave --------------------------------------------------------


def _vapor_layer(shape: tuple[int, int], time_s: float) -> np.ndarray:
    """Kisi perspektif dan garis horizon untuk satu frame.

    Semua jumlah garis dibatasi, bukan dihitung dari jarak piksel. Itu
    perbaikan inti dari versi lama: dengan roi setinggi 100 piksel, fase
    1/jarak menghasilkan garis berjarak 1-2 piksel yang menutupi seluruh
    portal dengan pita putih.

    Matahari tidak lagi digambar di sini. Piringan rata yang ditambahkan
    sebagai lapisan cahaya langsung menjenuhkan semua kanal dan wajahnya
    jadi putih polos; sekarang matahari berupa gradasi radial terpisah yang
    ditumpuk di bawah kisi.
    """
    height, width = shape
    layer = np.zeros((height, width), np.uint8)
    if height < 6 or width < 6:
        return layer

    horizon = int(round(height * VAPOR_HORIZON))
    horizon = max(1, min(height - 2, horizon))
    center_x = width // 2

    # Garis vertikal mengumpul ke titik hilang di horizon.
    columns = int(np.clip(width / 38.0, VAPOR_COLS_MIN, VAPOR_COLS_MAX))
    for i in range(-columns, columns + 1):
        x_bottom = center_x + int(round(i * (width * 0.72 / columns)))
        cv2.line(layer, (center_x, horizon), (x_bottom, height - 1), 235, 1, cv2.LINE_AA)

    # Garis horizontal makin rapat ke arah horizon, seperti lantai tak
    # berujung. Posisinya bergeser pelan dengan waktu.
    rows = int(np.clip((height - horizon) / 16.0, VAPOR_ROWS_MIN, VAPOR_ROWS_MAX))
    scroll = (time_s * VAPOR_SCROLL) % 1.0
    for i in range(rows):
        phase = (i + scroll) / rows
        y = horizon + int(round((height - horizon) * (phase ** 2.0)))
        if y >= height:
            continue
        thickness = 1 if phase < 0.6 else 2
        layer[y : min(height, y + thickness), :] = 235

    return cv2.GaussianBlur(layer, (0, 0), 0.6)


@lru_cache(maxsize=12)
def _vapor_sun(shape: tuple[int, int]) -> np.ndarray:
    """Matahari bergradasi dengan pita mendatar, ikon sampul kaset 80-an.

    Intensitasnya turun ke tepi, bukan piringan rata. Piringan rata terlihat
    seperti lampu sorot yang menempel di depan gambar.
    """
    height, width = shape
    layer = np.zeros((height, width), np.uint8)
    if height < 8 or width < 8:
        return layer
    horizon = int(round(height * VAPOR_HORIZON))
    center_x = width // 2
    # Radius dijaga di bawah setengah tinggi horizon. Begitu lebih besar,
    # piringannya menutupi seluruh wajah.
    radius = max(3, int(min(height * 0.26, width * 0.16)))
    sun_y = horizon - radius

    # Gradasi dari inti ke tepi, dibuat di kanvas kecil lalu diperbesar.
    side = 48
    yy = np.linspace(-1.0, 1.0, side, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, side, dtype=np.float32)[None, :]
    disc = np.clip(1.0 - np.sqrt(xx * xx + yy * yy), 0.0, 1.0) ** 0.65
    diameter = radius * 2
    small = np.clip(disc * 255.0, 0.0, 255.0).astype(np.uint8)
    sun = cv2.resize(small, (diameter, diameter), interpolation=cv2.INTER_LINEAR)

    # Pita mendatar: ciri khas matahari vaporwave. Potongannya makin tebal
    # ke bawah, jadi terlihat seperti piringan yang tenggelam.
    bands = 6
    for i in range(bands):
        offset = int(diameter * (0.42 + 0.10 * i))
        thickness = max(1, int(diameter * (0.028 + 0.022 * i)))
        y0, y1 = offset, min(diameter, offset + thickness)
        if y1 > y0:
            sun[y0:y1, :] = 0

    y0 = max(0, sun_y - radius)
    x0 = max(0, center_x - radius)
    y1 = min(height, y0 + sun.shape[0])
    x1 = min(width, x0 + sun.shape[1])
    if y1 > y0 and x1 > x0:
        layer[y0:y1, x0:x1] = sun[: y1 - y0, : x1 - x0]
    return layer


def _filter_vaporwave(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Magenta dan sian dengan kisi horizon bergulir ala sampul kaset 80-an."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]

    gray = cv2.LUT(_luma(roi), _curve_lut(1.10))
    lb, lg, lr = _duotone_luts(VAPOR_SHADOW, VAPOR_LIGHT)
    out = cv2.merge([cv2.LUT(gray, lb), cv2.LUT(gray, lg), cv2.LUT(gray, lr)])

    # Gradasi langit: sian di atas, magenta di bawah. Ini yang membedakan
    # vaporwave dari dual-tone biasa.
    sky = _vertical_gradient((height, width), (240, 250, 190), (190, 120, 255))
    out = cv2.addWeighted(out, 1.0, sky, 0.22, 0.0)

    # Matahari ditumpuk sebelum kisi, jadi garis-garis lantai melintas di
    # depannya seperti di sampul kaset aslinya.
    sun = _tint(_vapor_sun((height, width)), VAPOR_SUN_TINT)
    out = _add_scaled(out, sun, VAPOR_SUN_LEVEL)

    # Lapisan kisi digambar langsung tiap frame, bukan di-cache: posisinya
    # bergantung waktu, dan kunci cache yang berubah tiap frame justru
    # menghapus cache filter lain. Menggambar belasan garis jauh lebih murah
    # daripada satu kali membangun medan seukuran roi.
    grid_bgr = _tint(_vapor_layer((height, width), ctx.time_s), VAPOR_GRID_COLOR)
    out = _add_scaled(out, grid_bgr, VAPOR_GRID_STRENGTH)

    out = cv2.multiply(
        out, _scanlines((height, width), VAPOR_SCANLINE_PERIOD, VAPOR_SCANLINE_LEVEL),
        scale=1.0 / 255.0,
    )
    # Pemisahan kanal satu piksel: ciri khas pita VHS, dan cukup murah.
    if width > VAPOR_SPLIT_PX * 2:
        out[:, :, 2] = np.roll(out[:, :, 2], VAPOR_SPLIT_PX, axis=1)
        out[:, :, 0] = np.roll(out[:, :, 0], -VAPOR_SPLIT_PX, axis=1)
    return _polish(out, roi, VAPOR_VIGNETTE, VAPOR_GRAIN, _roi_origin(ctx))


# --- 16. edges ------------------------------------------------------------


def _filter_edges(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Hanya garis tepi: Sobel untuk tebal lembut, Canny untuk inti tajam."""
    if _too_small(roi):
        return roi.copy()
    gray = cv2.GaussianBlur(_luma(roi), (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    # Normalisasi memakai rata-rata, bukan nilai maksimum: satu kilau
    # tajam tidak boleh membuat seluruh gambar jadi hitam.
    mean = float(magnitude.mean())
    # Gerbang noise: nilai di bawah ambang dipotong ke nol dulu, baru
    # diskalakan. Tanpa langkah ini, latar yang rata tetap menghasilkan
    # magnitudo kecil dan ikut menyala sebagai bintik.
    gate = max(EDGES_FLOOR * mean, 1.0)
    magnitude = np.maximum(magnitude - gate, 0.0)
    scale = EDGES_GAIN * 255.0 / (mean * 8.0 + 1.0)
    soft = np.clip(magnitude * scale, 0.0, 255.0).astype(np.uint8)
    soft = cv2.LUT(soft, _knee_lut(EDGES_KNEE))

    mean_level = float(gray.mean())
    low = int(max(20.0, mean_level * 0.55))
    high = int(min(220.0, max(low + 25.0, mean_level * 1.35)))
    core = cv2.Canny(gray, low, high)

    base = cv2.convertScaleAbs(_bgr(roi), alpha=EDGES_BASE_LEVEL, beta=8)
    out = _add_scaled(base, _tint(soft, EDGES_SOFT_COLOR), 0.55)
    out = _add_scaled(out, _tint(core, EDGES_CORE_COLOR), 0.80)
    return _polish(out, roi, EDGES_VIGNETTE, 0.0)


# --- 17. posterize --------------------------------------------------------


@lru_cache(maxsize=8)
def _posterize_lut(levels: int) -> np.ndarray:
    """Tangga tingkat warna, tiap tingkat diwakili nilai tengahnya."""
    step = 256 // levels
    return np.clip(((np.arange(256) + step // 2) // step) * step, 0, 255).astype(np.uint8)


def _filter_posterize(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Warna disederhanakan jadi beberapa tingkat, dengan blur kecil dulu."""
    if _too_small(roi):
        return roi.copy()
    # Tanpa blur, noise sensor memecah satu bidang warna jadi bercak acak.
    smooth = cv2.GaussianBlur(_bgr(roi), (0, 0), POSTERIZE_PREBLUR)
    out = cv2.LUT(smooth, _posterize_lut(POSTERIZE_LEVELS))
    out = cv2.LUT(out, _curve_lut(1.05, 3))
    out = _saturate(out, POSTERIZE_SATURATION)
    # Garis tipis di batas bidang warna: ciri sablon layar, dan membuat
    # hasilnya tidak terlihat seperti foto yang rusak warnanya.
    # Tepinya diambil dari hasil kuantisasi, bukan dari gambar asli. Kalau
    # dari gambar asli, gradasi halus yang ber-noise menghasilkan ratusan
    # tepi pendek dan hasilnya jadi bintik, bukan garis.
    flat = cv2.LUT(out, _posterize_lut(POSTERIZE_LEVELS))
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    edge = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edge = cv2.threshold(edge, POSTERIZE_EDGE_LOW, 255, cv2.THRESH_BINARY)[1]
    # Bukaan membuang serpihan satu-dua piksel di batas bidang. Tanpa ini,
    # sisa ayunan nada yang terkuantisasi masih terlihat sebagai bintik.
    edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    edge = cv2.convertScaleAbs(edge, alpha=POSTERIZE_EDGE_LEVEL)
    out = cv2.multiply(out, cv2.bitwise_not(edge), scale=1.0 / 255.0)
    return _polish(out, roi, POSTERIZE_VIGNETTE, POSTERIZE_GRAIN, _roi_origin(ctx))


# --- 18. dream ------------------------------------------------------------


def _filter_dream(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Bloom lembut: sorotan mekar, warna menghangat, tepi mengabur."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]
    base = _bgr(roi)
    bright = cv2.subtract(base, (DREAM_THRESHOLD, DREAM_THRESHOLD, DREAM_THRESHOLD, 0))
    small_w = max(1, width // 4)
    small_h = max(1, height // 4)
    small = cv2.resize(bright, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3.0)
    bloom = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

    out = cv2.addWeighted(base, 1.0, bloom, DREAM_BLOOM, 8)
    soft = cv2.GaussianBlur(out, (0, 0), 1.4)
    out = cv2.addWeighted(out, 1.0 - DREAM_SOFTEN, soft, DREAM_SOFTEN, 0)
    out = cv2.transform(out, DREAM_WARM)
    # Kebocoran cahaya di sudut kanan atas: yang membuat hasilnya terasa
    # seperti foto lensa lama, bukan sekadar gambar yang diburamkan.
    leak = _radial_bump((height, width), (0.86, 0.10), 0.55, 2.2)
    out = _add_scaled(out, _tint(leak, DREAM_LEAK), DREAM_LEAK_LEVEL)
    return _polish(out, roi, DREAM_VIGNETTE, DREAM_GRAIN, _roi_origin(ctx))


# --- 19. noise ------------------------------------------------------------


def _filter_noise(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Butir film halus yang menggeser pelan, bukan berkedip tiap frame."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]
    grain = _grain_layer((height, width), NOISE_AMOUNT)
    # Butir digeser vertikal sangat pelan. Kalau diganti tiap frame, hasilnya
    # TV bersalju; kalau diam sama sekali, terlihat seperti tekstur statis.
    offset = int(ctx.time_s * NOISE_DRIFT) % height
    if offset:
        grain = np.roll(grain, offset, axis=0)
    # Kekuatan butirnya juga bernapas pelan, jadi mata tidak menangkap pola
    # yang sama terus walau geseran pikselnya belum berubah.
    strength = 0.80 + 0.20 * (0.5 + 0.5 * math.sin(ctx.time_s * 1.3))

    base = _bgr(roi)
    out = cv2.addWeighted(base, 1.0, grain, strength, -128.0 * strength)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    out = cv2.addWeighted(
        out, 1.0 - NOISE_DESATURATE, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), NOISE_DESATURATE, 0
    )
    out = cv2.LUT(out, _curve_lut(NOISE_CONTRAST))
    return _polish(out, roi, NOISE_VIGNETTE, 0.0, _roi_origin(ctx))


# --- 20. matrix -----------------------------------------------------------


@lru_cache(maxsize=1)
def _glyph_tile() -> np.ndarray:
    """Satu petak glyph yang bisa diulang, dibuat sekali seumur proses."""
    tile_h = MATRIX_ROWS * MATRIX_CELL_H
    tile_w = MATRIX_COLS * MATRIX_CELL_W
    tile = np.zeros((tile_h, tile_w), np.uint8)
    rng = np.random.default_rng(MATRIX_SEED)
    for row in range(MATRIX_ROWS):
        for col in range(MATRIX_COLS):
            char = MATRIX_CHARS[int(rng.integers(0, len(MATRIX_CHARS)))]
            x = col * MATRIX_CELL_W + 3
            y = row * MATRIX_CELL_H + MATRIX_CELL_H - 6
            cv2.putText(
                tile,
                char,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                MATRIX_GLYPH_SCALE,
                255,
                1,
                cv2.LINE_AA,
            )
    return tile


def _matrix_texture(shape: tuple[int, int]) -> np.ndarray:
    """Petak glyph yang sudah diulang selebar roi, di-cache per ukuran."""
    height, width = shape
    tile = _glyph_tile()
    reps_y = height // tile.shape[0] + 1
    reps_x = width // tile.shape[1] + 1
    big = np.tile(tile, (reps_y, reps_x))
    return np.ascontiguousarray(big[:height, :width])


def _matrix_speeds(count: int) -> np.ndarray:
    """Kecepatan jatuh per kolom, tetap untuk setiap proses.

    Dihitung dari hash kolom, bukan acak per frame, supaya hujan tidak
    berubah pola setiap kali frame berganti.
    """
    idx = np.arange(count, dtype=np.float32)
    return 0.55 + 0.9 * (0.5 + 0.5 * np.sin(idx * 12.9898) * np.cos(idx * 4.1414))


def _matrix_rain(shape: tuple[int, int], ctx: FilterContext) -> np.ndarray:
    """Hujan glyph satu kanal untuk satu frame."""
    height, width = shape
    texture = _cached(("matrix", shape), lambda: _matrix_texture(shape))
    columns = max(1, (width + MATRIX_CELL_W - 1) // MATRIX_CELL_W)
    speeds = _matrix_speeds(columns)
    origin_x, _ = _roi_origin(ctx)
    # Geser kolom sesuai posisi roi di frame supaya pola tidak melompat
    # saat portal bergerak.
    offsets = (np.arange(columns, dtype=np.float32) * MATRIX_CELL_H + origin_x) % (
        height + MATRIX_TAIL
    )
    heads = (ctx.time_s * speeds * 120.0 + offsets) % (height + MATRIX_TAIL)

    rain = np.empty((height, width), np.uint8)
    for i in range(columns):
        x0 = i * MATRIX_CELL_W
        x1 = min(width, x0 + MATRIX_CELL_W)
        if x1 <= x0:
            break
        rain[:, x0:x1] = np.roll(texture[:, x0:x1], int(heads[i]), axis=0)

    rows = np.arange(height, dtype=np.float32)[:, None]
    delta = heads[None, :] - rows
    fade = np.exp(-np.maximum(delta, 0.0) / MATRIX_TAIL)
    fade[delta < 0] = 0.0
    fade = np.clip(fade * 1.35 + 0.05, 0.0, 1.0)
    fade_u8 = (fade * 255.0).astype(np.uint8)
    fade_px = np.repeat(fade_u8, MATRIX_CELL_W, axis=1)[:, :width]
    return cv2.multiply(rain, fade_px, scale=1.0 / 255.0)


def _filter_matrix(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Hujan karakter hijau di belakang orang, plus lapisan tipis di atasnya."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]
    shape = (height, width)
    rain = _matrix_rain(shape, ctx)
    rain_bgr = _tint(rain, MATRIX_GREEN)

    person_gray = cv2.LUT(_luma(roi), _curve_lut(MATRIX_PERSON_CONTRAST))
    person = _tint(person_gray, MATRIX_PERSON_GREEN)
    person = cv2.convertScaleAbs(person, alpha=0.90, beta=12)
    # Sedikit hujan di atas badan: orangnya jadi terasa berada di dalam
    # hujan, bukan di depannya.
    person = _add_scaled(person, rain_bgr, MATRIX_OVERLAY * 0.55)

    mask = _person_mask_crop(ctx, roi)
    if mask is None:
        # Tanpa mask orang, hujan tetap ditumpuk di atas gambar. Masih
        # terbaca sebagai efek matrix, hanya bukan di belakang badan.
        out = cv2.addWeighted(person, 0.92, rain_bgr, 0.42, 0.0)
    else:
        feathered = _feather(mask)
        scene = cv2.add(rain_bgr, _tint(cv2.subtract(feathered, mask), (200, 255, 180)))
        out = _blend_masked(person, scene, feathered)

    out = _add_scaled(out, rain_bgr, MATRIX_OVERLAY)
    out = cv2.multiply(
        out, _scanlines((height, width), MATRIX_SCANLINE_PERIOD, MATRIX_SCANLINE_LEVEL),
        scale=1.0 / 255.0,
    )
    return _polish(out, roi, MATRIX_VIGNETTE, MATRIX_GRAIN, _roi_origin(ctx))


# --- daftar filter --------------------------------------------------------


# --- Filter tambahan -----------------------------------------------------
# Semuanya mengikuti aturan yang sama: kembalikan array baru, jangan sentuh
# roi asli, dan jangan pernah gagal pada roi kecil.

COMIC_LEVELS = 6
COMIC_SATURATION = 1.35
COMIC_EDGE_BLOCK = 9
COMIC_EDGE_C = 6
# Blur sebelum kuantisasi. Bidang warna komik harus rata; tanpa ini,
# gradasi pelan yang ber-noise terpecah jadi bercak coklat dan hasilnya
# terbaca seperti cetakan rusak, bukan seperti panel komik.
COMIC_PREBLUR = 2.2

WATERCOLOR_SHRINK = 2
WATERCOLOR_SIGMA_COLOR = 42
WATERCOLOR_SIGMA_SPACE = 12
WATERCOLOR_MEDIAN = 5
WATERCOLOR_EDGE_MIX = 0.22

NIGHTVISION_GAMMA = 0.62
NIGHTVISION_GREEN = (70, 255, 90)
NIGHTVISION_GRAIN = 16
NIGHTVISION_VIGNETTE = 0.62
NIGHTVISION_SCAN_PERIOD = 4
NIGHTVISION_SCAN_LEVEL = 236

BLUEPRINT_BLUE = (150, 96, 40)
BLUEPRINT_LINE = (236, 226, 208)
BLUEPRINT_GRID_PX = 46
BLUEPRINT_GRID_LEVEL = 42

HOLO_CYAN = (230, 240, 120)
HOLO_SCAN_PERIOD = 6
HOLO_SCAN_LEVEL = 214
HOLO_FLICKER = 0.045

SUNSET_TOP = (60, 120, 255)
SUNSET_BOTTOM = (200, 70, 150)
SUNSET_WARMTH = 0.34

EMBOSS_KERNEL = (
    (-2, -1, 0),
    (-1, 1, 1),
    (0, 1, 2),
)

CHROME_SILVER = (250, 246, 240)
CHROME_SHADOW = (150, 120, 96)
CHROME_CONTRAST = 1.05
# Kurva logam: nada dinaikkan dulu (gamma di bawah 1), lalu ditekan di
# kedua ujung. Chrome harus terbaca terang dan memantul; kalau nada
# ditekan seperti foto biasa, hasilnya cuma abu-abu gelap.
CHROME_GAMMA = 0.52
CHROME_LIFT = 0.10
CHROME_BANDS = 7

LOMO_SATURATION = 1.42
LOMO_WARM = (40, 120, 255)
LOMO_VIGNETTE = 0.72

INFRARED_TINT = (170, 150, 255)
INFRARED_CONTRAST = 1.22

CONFETTI_COUNT = 90
CONFETTI_RADIUS_PX = 5
CONFETTI_SEED = 7


def _filter_comic(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Buku komik: warna tebal dipisah garis tinta hitam."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]

    # Warna dulu: saturasi dinaikkan lalu dipotong jadi beberapa tingkat,
    # supaya terlihat seperti tinta cetak, bukan foto yang dikontraskan.
    smooth = cv2.GaussianBlur(_bgr(roi), (0, 0), COMIC_PREBLUR)
    colorful = _saturate(smooth, COMIC_SATURATION)
    step = max(1, 256 // COMIC_LEVELS)
    quantized = (colorful // step) * step + step // 2

    # Ambang adaptif diambil dari bidang yang sudah dihaluskan dan
    # dikuantisasi. Diambil dari gambar asli, gradasi pelan yang ber-noise
    # membuat ambang lokalnya bolak-balik dan hasilnya bercak tinta.
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), COMIC_PREBLUR)
    block = _odd(max(3, min(COMIC_EDGE_BLOCK, min(height, width) // 3 or 3)))
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, COMIC_EDGE_C
    )
    # Sisa bintik yang lolos disaring lewat bukaan: satu-dua piksel terpisah
    # bukan bidang bayangan komik, dan kalau dibiarkan terlihat seperti noda.
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ink_ratio = cv2.cvtColor(ink, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    out = (quantized.astype(np.float32) * ink_ratio).astype(np.uint8)
    return _polish(out, roi, vignette=0.30, grain=0.0)


def _filter_watercolor(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Cat air: warna melebur, detail hilang, tepi tetap terasa basah."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]

    # Bilateral di resolusi penuh terlalu mahal untuk tiap frame, jadi
    # dikerjakan di gambar kecil lalu dinaikkan lagi. Hasilnya justru lebih
    # terasa seperti cat air karena detailnya memang sengaja dibuang.
    small = cv2.resize(
        roi,
        (max(2, width // WATERCOLOR_SHRINK), max(2, height // WATERCOLOR_SHRINK)),
        interpolation=cv2.INTER_AREA,
    )
    smooth = cv2.bilateralFilter(
        small, 7, WATERCOLOR_SIGMA_COLOR, WATERCOLOR_SIGMA_SPACE
    )
    smooth = cv2.medianBlur(smooth, min(WATERCOLOR_MEDIAN, max(3, min(smooth.shape[:2]) // 2 * 2 + 1)))
    washed = cv2.resize(smooth, (width, height), interpolation=cv2.INTER_CUBIC)

    edges = cv2.Canny(cv2.GaussianBlur(_luma(roi), (5, 5), 0), 40, 110)
    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Tepi ditempel tipis saja: kalau terlalu tebal, hasilnya jadi sketsa,
    # bukan cat air.
    out = cv2.addWeighted(washed, 1.0, edge_bgr, -WATERCOLOR_EDGE_MIX, 0.0)
    out = _saturate(out, 1.18)
    return _polish(out, roi, vignette=0.22, grain=0.0)


def _filter_nightvision(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Kamera penglihatan malam: hijau, berbutir, dan gelap di sudut."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]

    gray = _luma(roi).astype(np.float32) / 255.0
    # Gamma di bawah satu menaikkan bayangan, seperti penguat cahaya sisa.
    boosted = np.power(np.clip(gray, 0.0, 1.0), NIGHTVISION_GAMMA) * 255.0
    out = _tint(boosted.astype(np.uint8), NIGHTVISION_GREEN)
    out = cv2.addWeighted(
        out, 1.0, _grain_layer((height, width), NIGHTVISION_GRAIN, _roi_origin(ctx)), 1.0, -128.0
    )
    out = cv2.multiply(
        out,
        _scanlines((height, width), NIGHTVISION_SCAN_PERIOD, NIGHTVISION_SCAN_LEVEL),
        scale=1.0 / 255.0,
    )
    out = cv2.multiply(
        out, _edge_vignette((height, width), NIGHTVISION_VIGNETTE), scale=1.0 / 255.0
    )
    return _finalize(out, roi)


def _filter_blueprint(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Cetak biru: garis putih tipis di atas kertas biru bergaris."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]

    gray = cv2.GaussianBlur(_luma(roi), (3, 3), 0)
    edges = cv2.Canny(gray, 55, 140)
    lines = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    paper = np.zeros((height, width, 3), np.float32)
    paper[:] = BLUEPRINT_BLUE

    # Kisi tipis supaya terasa seperti kertas gambar teknik. Garisnya
    # digambar dengan potongan array, bukan loop, supaya tetap murah.
    step = max(8, int(BLUEPRINT_GRID_PX * height / 720))
    grid = np.zeros((height, width), np.uint8)
    grid[::step, :] = BLUEPRINT_GRID_LEVEL
    grid[:, ::step] = BLUEPRINT_GRID_LEVEL
    paper += cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR).astype(np.float32) * 0.35

    line_color = np.array(BLUEPRINT_LINE, np.float32)
    out = paper * (1.0 - lines) + line_color * lines
    out = np.clip(out, 0, 255).astype(np.uint8)
    return _polish(out, roi, vignette=0.34, grain=0.0)


def _filter_holo(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Hologram: biru pucat, garis pindai bergerak, dan sedikit berkedip."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]

    gray = _luma(roi)
    out = _tint(gray, HOLO_CYAN)

    # Garis pindai digeser tiap frame, jadi terlihat seperti data yang
    # sedang diproyeksikan, bukan garis diam.
    offset = int(ctx.time_s * 46) % max(1, HOLO_SCAN_PERIOD)
    scan = np.full((height, width, 3), 255, np.uint8)
    rows = np.arange(height)
    mask_rows = ((rows + offset) % HOLO_SCAN_PERIOD) == 0
    scan[mask_rows] = HOLO_SCAN_LEVEL
    out = cv2.multiply(out, scan, scale=1.0 / 255.0)

    # Kilau bergerak diagonal, cukup tipis supaya tidak menutupi gambar.
    glow = cv2.cvtColor(
        _radial_bump((height, width), (0.5, 0.5), 0.9, 1.6), cv2.COLOR_GRAY2BGR
    )
    out = _add_scaled(out, glow, 0.20)

    flicker = 1.0 - HOLO_FLICKER * (0.5 + 0.5 * math.sin(ctx.time_s * 7.3))
    out = np.clip(out.astype(np.float32) * flicker, 0, 255).astype(np.uint8)
    return _polish(out, roi, vignette=0.30, grain=0.0)


def _filter_sunset(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Langit senja: jingga di atas, ungu di bawah, cahaya keemasan."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]

    gradient = _vertical_gradient((height, width), SUNSET_TOP, SUNSET_BOTTOM)
    warm = cv2.addWeighted(roi, 1.0, gradient, SUNSET_WARMTH, 0.0)
    warm = _saturate(warm, 1.16)

    # Sorotan lembut di bagian atas, seperti matahari rendah.
    glow = _radial_bump((height, width), (0.5, 0.12), 0.75, 1.4)
    warm = _add_scaled(warm, cv2.cvtColor(glow, cv2.COLOR_GRAY2BGR), 0.22)
    return _polish(warm, roi, vignette=0.34, grain=0.0)


def _filter_emboss(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Relief timbul seperti cetakan logam."""
    if _too_small(roi):
        return roi.copy()
    kernel = np.array(EMBOSS_KERNEL, np.float32)
    # Hasil filter punya nilai negatif, jadi dihitung di float dulu. Kalau
    # langsung ke CV_8U, semua bayangan terpotong ke nol dan reliefnya hilang.
    relief = cv2.filter2D(roi.astype(np.float32), cv2.CV_32F, kernel)
    relief = np.clip(relief + 128.0, 0, 255).astype(np.uint8)
    out = _tint(_luma(relief), (200, 214, 226))
    return _polish(out, roi, vignette=0.40, grain=0.0)


@lru_cache(maxsize=8)
def _chrome_lut() -> np.ndarray:
    """Kurva logam: bayangan pekat, lompatan tajam, puncak yang ditahan.

    Dipakai bersama perentaian band supaya gradasi halus berubah menjadi
    pantulan yang bertingkat, seperti krom yang memantulkan langit.
    """
    x = np.arange(256, dtype=np.float32) / 255.0
    y = np.power(x, CHROME_GAMMA)
    y = np.clip(CHROME_LIFT + y * (1.0 - CHROME_LIFT), 0.0, 1.0)
    return np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)


def _filter_chrome(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Logam mengkilap: pantulan bertingkat dari bayangan hangat ke perak."""
    if _too_small(roi):
        return roi.copy()
    gray = _luma(roi)
    curved = cv2.LUT(gray, _chrome_lut())
    # Sedikit banding: krom memantulkan lingkungan sebagai pita cahaya,
    # bukan sebagai gradasi mulus. Dibuat dari LUT tingkatan, jadi murah.
    levels = 256 // (CHROME_BANDS * 2)
    banded = (curved // max(1, levels)) * max(1, levels)
    curved = cv2.addWeighted(curved, 0.55, banded, 0.45, 0)
    curved = cv2.LUT(curved, _curve_lut(CHROME_CONTRAST))

    highlight = _tint(curved, CHROME_SILVER)
    shadow = _tint(curved, CHROME_SHADOW)
    # Bayangan diberi warna hangat, sorotan tetap perak: itu yang membuat
    # logam terlihat memantulkan langit, bukan sekadar abu-abu.
    ratio = (curved.astype(np.float32) / 255.0)[..., None]
    out = np.clip(
        shadow.astype(np.float32) * (1.0 - ratio) + highlight.astype(np.float32) * ratio,
        0,
        255,
    ).astype(np.uint8)
    return _polish(out, roi, vignette=0.28, grain=0.0)


def _filter_lomo(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Kamera film murah: warna pekat, hangat, sudut gelap."""
    if _too_small(roi):
        return roi.copy()
    out = _saturate(_bgr(roi), LOMO_SATURATION)
    warm = np.array(LOMO_WARM, np.float32) / 255.0
    out = np.clip(out.astype(np.float32) * (0.82 + 0.34 * warm), 0, 255).astype(np.uint8)
    out = cv2.LUT(out, _curve_lut(1.18))
    return _polish(out, roi, vignette=LOMO_VIGNETTE, grain=1.4)


def _filter_infrared(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Infra merah: daun dan kulit menyala merah muda, langit jadi gelap."""
    if _too_small(roi):
        return roi.copy()
    # Kanal merah ditukar dengan hijau. Pada foto infra merah asli, tanaman
    # yang memantulkan banyak inframerah justru tampak paling terang, dan
    # penukaran kanal meniru kesan itu tanpa sensor khusus.
    swapped = roi.copy()
    swapped[:, :, 2] = roi[:, :, 1]
    swapped[:, :, 1] = roi[:, :, 0]
    swapped = cv2.LUT(swapped, _curve_lut(INFRARED_CONTRAST))
    out = cv2.addWeighted(swapped, 0.78, _tint(_luma(roi), INFRARED_TINT), 0.30, 0.0)
    return _polish(out, roi, vignette=0.34, grain=0.0)


def _filter_confetti(roi: np.ndarray, ctx: FilterContext) -> np.ndarray:
    """Bintik warna meriah, seperti pesta yang tertangkap kamera."""
    if _too_small(roi):
        return roi.copy()
    height, width = roi.shape[:2]
    out = _saturate(_bgr(roi), 1.1)

    palette = (
        (90, 90, 245),
        (80, 220, 250),
        (120, 240, 140),
        (250, 200, 90),
        (230, 130, 250),
        (255, 255, 255),
    )
    rng = np.random.default_rng(CONFETTI_SEED)
    scale = max(1.0, height / 720.0)
    for index in range(CONFETTI_COUNT):
        # Posisi ditentukan benih tetap, tapi bergerak turun pelan mengikuti
        # waktu, jadi konfetinya jatuh alih-alih menempel diam.
        base_x = rng.random()
        base_y = rng.random()
        drift = (ctx.time_s * 0.06 + index * 0.013) % 1.0
        x = int((base_x * width + math.sin(ctx.time_s * 0.7 + index) * 6 * scale)) % max(1, width)
        y = int((base_y + drift) % 1.0 * height)
        color = palette[index % len(palette)]
        radius = max(1, int(round(CONFETTI_RADIUS_PX * scale * (0.6 + 0.7 * rng.random()))))
        cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)
    return _polish(out, roi, vignette=0.18, grain=0.0)


def build_filters() -> list[FilterSpec]:
    """Daftar filter berurutan; urutan inilah yang dilihat pemakai."""
    return [
        FilterSpec(
            key="mono",
            label="MONO",
            hint="Hitam putih dengan kontras film, bersih tanpa warna.",
            fn=_filter_mono,
        ),
        FilterSpec(
            key="dual-tone",
            label="DUO",
            hint="Dua warna kontras: bayangan indigo, sorotan krem.",
            fn=_filter_dual_tone,
        ),
        FilterSpec(
            key="pixelate",
            label="PIXEL",
            hint="Gambar dipecah jadi balok besar seperti game lama.",
            fn=_filter_pixelate,
        ),
        FilterSpec(
            key="invert",
            label="INVERT",
            hint="Semua warna dibalik seperti negatif film.",
            fn=_filter_invert,
        ),
        FilterSpec(
            key="sepia",
            label="SEPIA",
            hint="Nada cokelat tua seperti foto jadul di album.",
            fn=_filter_sepia,
        ),
        FilterSpec(
            key="blur",
            label="BLUR",
            hint="Fokus lembut, detail kecil melebur jadi halus.",
            fn=_filter_blur,
        ),
        FilterSpec(
            key="thermal",
            label="THERMAL",
            hint="Panas tubuh berubah jadi warna infra merah.",
            fn=_filter_thermal,
        ),
        FilterSpec(
            key="sketch",
            label="SKETCH",
            hint="Garis pensil di atas kertas, seperti digambar tangan.",
            fn=_filter_sketch,
        ),
        FilterSpec(
            key="glitch",
            label="GLITCH",
            hint="Warna bergeser dan baris gambar melompat seperti sinyal rusak.",
            fn=_filter_glitch,
        ),
        FilterSpec(
            key="neon",
            label="NEON",
            hint="Tepi objek menyala dengan glow sian dan magenta.",
            fn=_filter_neon,
        ),
        FilterSpec(
            key="galaxy",
            label="GALAXY",
            hint="Bintang dan nebula muncul di belakang badan kamu.",
            fn=_filter_galaxy,
        ),
        FilterSpec(
            key="mirror",
            label="MIRROR",
            hint="Kaleidoskop cermin empat arah dari sudut kiri atas.",
            fn=_filter_mirror,
        ),
        FilterSpec(
            key="halftone",
            label="HALFTONE",
            hint="Titik-titik cetak koran menyusun ulang gambarnya.",
            fn=_filter_halftone,
        ),
        FilterSpec(
            key="xray",
            label="XRAY",
            hint="Negatif biru-hijau seperti hasil rontgen di rumah sakit.",
            fn=_filter_xray,
        ),
        FilterSpec(
            key="vaporwave",
            label="VAPOR",
            hint="Magenta dan sian dengan kisi horizon ala tahun delapan puluhan.",
            fn=_filter_vaporwave,
        ),
        FilterSpec(
            key="edges",
            label="EDGES",
            hint="Hanya garis tepi objek, bersih dan tajam.",
            fn=_filter_edges,
        ),
        FilterSpec(
            key="posterize",
            label="POSTER",
            hint="Warna disederhanakan jadi beberapa tingkat seperti poster sablon.",
            fn=_filter_posterize,
        ),
        FilterSpec(
            key="dream",
            label="DREAM",
            hint="Cahaya mekar lembut dan warna menghangat seperti mimpi.",
            fn=_filter_dream,
        ),
        FilterSpec(
            key="noise",
            label="GRAIN",
            hint="Butir film halus di seluruh gambar seperti rekaman seluloid.",
            fn=_filter_noise,
        ),
        FilterSpec(
            key="matrix",
            label="MATRIX",
            hint="Hujan karakter hijau turun di belakang badan kamu.",
            fn=_filter_matrix,
        ),
        FilterSpec(
            key="comic",
            label="COMIC",
            hint="Buku komik: warna tebal, garis hitam, dan bintik bayangan.",
            fn=_filter_comic,
        ),
        FilterSpec(
            key="watercolor",
            label="AQUAREL",
            hint="Cat air: warna melebur lembut dengan tepi basah.",
            fn=_filter_watercolor,
        ),
        FilterSpec(
            key="nightvision",
            label="NIGHT",
            hint="Kamera penglihatan malam hijau dengan butir dan lingkaran lensa.",
            fn=_filter_nightvision,
        ),
        FilterSpec(
            key="blueprint",
            label="BLUEPRINT",
            hint="Cetak biru teknik: garis putih di atas kertas biru.",
            fn=_filter_blueprint,
        ),
        FilterSpec(
            key="neon-rain",
            label="HOLO",
            hint="Hologram biru dengan garis pindai yang bergerak.",
            fn=_filter_holo,
        ),
        FilterSpec(
            key="sunset",
            label="SUNSET",
            hint="Langit jingga keunguan dengan cahaya keemasan.",
            fn=_filter_sunset,
        ),
        FilterSpec(
            key="emboss",
            label="EMBOSS",
            hint="Gambar jadi relief timbul seperti cetakan logam.",
            fn=_filter_emboss,
        ),
        FilterSpec(
            key="chrome",
            label="CHROME",
            hint="Logam mengkilap dengan pantulan biru dan perak.",
            fn=_filter_chrome,
        ),
        FilterSpec(
            key="lomo",
            label="LOMO",
            hint="Warna pekat dan sudut gelap seperti kamera film murah.",
            fn=_filter_lomo,
        ),
        FilterSpec(
            key="infrared",
            label="INFRARED",
            hint="Dedaunan menyala merah muda seperti foto infra merah.",
            fn=_filter_infrared,
        ),
        FilterSpec(
            key="confetti",
            label="CONFETTI",
            hint="Bintik warna meriah menutupi gambar seperti pesta.",
            fn=_filter_confetti,
        ),
    ]


def apply_filter(roi: np.ndarray, spec: FilterSpec, ctx: FilterContext) -> np.ndarray:
    """Jalankan satu filter pada roi dan kembalikan array baru.

    Pemeriksaan bentuk dan tipe dilakukan sekali lagi di sini, jadi filter
    yang keliru sekalipun tidak akan merusak frame di pemanggil.
    """
    if roi is None or roi.size == 0:
        return np.zeros((0, 0, 3), np.uint8)
    out = spec.fn(roi, ctx)
    return _finalize(out, roi)
