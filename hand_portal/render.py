"""Semua yang digambar di atas frame kamera.

Modul ini sengaja tidak tahu apa-apa soal kamera, MediaPipe, atau filter. Dia
cuma menerima frame yang sudah jadi plus data dari modul lain, lalu menempelkan
lapisan visual di atasnya. Dengan begitu urutan penggambaran bisa diatur di
`app.py` tanpa mengubah isi modul ini.

Semua metode menggambar langsung ke array `frame` yang dikirim (in-place),
kecuali `bloom` yang mengembalikan array baru. Alasannya: frame kamera itu
buffer besar yang dipakai ulang tiap frame, jadi menyalinnya berkali-kali
hanya membuang waktu.
"""

from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np

from .config import Settings, Theme
from .contracts import INDEX_TIP, THUMB_TIP, HandFrame, Portal

# Skala semua ukuran mengikuti tinggi frame, supaya tampilan di 480p dan 1080p
# terasa sama besar secara proporsional. 720p dijadikan acuan karena itu
# setelan bawaan di config.
REFERENCE_HEIGHT = 720.0
MIN_SCALE = 0.45
MAX_SCALE = 2.4

# Sambungan tulang standar MediaPipe. Ditulis sebagai daftar pasangan supaya
# tidak ada yang perlu menghafal indeksnya di dalam fungsi gambar.
HAND_BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 13), (13, 17), (17, 0),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
)

BONE_WIDTH_PX = 2.0
BONE_OUTLINE_PX = 2.0
JOINT_RADIUS_PX = 3.0
TIP_RADIUS_PX = 7.0
TIP_HALO_FACTOR = 1.9
TIP_CORE_FACTOR = 0.42
MIN_SKELETON_SIDE = 8

# Tepi portal diblur sedikit supaya batas filter tidak bergerigi. Kalau terlalu
# tebal, portal terlihat berbayang dan bukan seperti jendela.
EDGE_SOFTNESS_PX = 2.6
EDGE_KERNEL_MAX = 13
# Gradasi tepi dibuat dengan mask yang dipotong lebih lebar dari kotak portal.
# Tanpa tambahan ini blur berhenti di batas potongan dan gradasinya hilang,
# jadi yang terlihat justru garis lurus 1 piksel.
EDGE_PAD_MAX = 8

PORTAL_LINE_PX = 2.0
CORNER_RADIUS_PX = 5.0
# Cincin luar sengaja diberi jarak dasar supaya tidak pernah menempel di tepi.
# Kalau dia bisa menyentuh tepi, hasilnya garis tipis dobel yang terlihat
# seperti gambar meleset, bukan halo.
RING_OFFSET_PX = 8.0
RING_PULSE_PX = 6.0
RING_SPEED = 0.85
# Batas atas geseran relatif terhadap ukuran portal, supaya portal kecil tidak
# dilingkari halo raksasa.
RING_MAX_FRACTION = 0.35
PARTICLE_MAX_FRACTION = 0.22
GLOW_PAD_PX = 18.0
GLOW_DOWNSCALE = 2
GLOW_KERNEL_PX = 9
GLOW_PEN = 2.0
MIN_PORTAL_RADIUS_PX = 1.0
# Garis tepi dibuat sedikit lebih redup dari aksen supaya partikel putih di
# atasnya tetap terbaca. Kalau garisnya sama terang, partikel hilang.
EDGE_LINE_MIX = 0.3

# Partikel menempuh sebagian keliling per detik, bukan piksel per detik, supaya
# kecepatannya terasa sama di portal besar maupun kecil.
PARTICLE_SPACING_PX = 42.0
PARTICLE_SPEED = 0.13
PARTICLE_TRAIL = 0.05
PARTICLE_TRAIL_SEGMENTS = 5
PARTICLE_MIN = 4
PARTICLE_MAX = 96
PARTICLE_RADIUS_PX = 2.6
# Jalur partikel digeser ke luar tepi supaya percikannya tidak tenggelam di
# garis aksen yang terang.
PARTICLE_OFFSET_PX = 5.0
GOLDEN_FRACTION = 0.6180339887498949

BLOOM_DOWNSCALE = 4
BLOOM_SIGMA = 6.0
BLOOM_THRESHOLD = 178
BLOOM_MIN_SIDE = 8


def _hash01(seed: int) -> float:
    """Angka 0..1 yang terlihat acak tapi selalu sama untuk seed yang sama.

    Dipakai partikel supaya tiap partikel punya sifat sendiri tanpa memanggil
    generator acak, karena generator acak membuat posisinya melompat tiap
    frame dan hasilnya jadi seperti salju TV.
    """
    x = (seed * 2654435761) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 2246822519) & 0xFFFFFFFF
    x ^= x >> 13
    return x / 4294967295.0


def _odd(value: int) -> int:
    """GaussianBlur menolak kernel genap, jadi bulatkan ke ganjil terdekat."""
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _mix(first: tuple[int, int, int], second: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    """Campur dua warna BGR. amount 0 = first, 1 = second."""
    amount = min(1.0, max(0.0, amount))
    return (
        int(first[0] + (second[0] - first[0]) * amount),
        int(first[1] + (second[1] - first[1]) * amount),
        int(first[2] + (second[2] - first[2]) * amount),
    )


def _polygon_edges(points: list[tuple[float, float]]) -> tuple[list[tuple[tuple[float, float], tuple[float, float], float]], float]:
    """Panjang tiap sisi poligon tertutup, plus totalnya.

    Dihitung sekali per frame lalu dipakai untuk menempatkan partikel
    berdasarkan panjang busur, bukan indeks sisi. Kalau berbasis indeks sisi,
    partikel akan menumpuk di sisi yang pendek.
    """
    edges: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    total = 0.0
    count = len(points)
    for index in range(count):
        start = points[index]
        end = points[(index + 1) % count]
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        edges.append((start, end, length))
        total += length
    return edges, total


def _offset_polygon(points: list[tuple[float, float]], distance: float) -> list[tuple[float, float]]:
    """Geser tiap sudut poligon menjauh dari titik pusatnya.

    Cukup memakai arah radial karena jarak gesernya kecil (beberapa piksel).
    Perhitungan normal sisi yang benar akan lebih tepat untuk sudut tajam,
    tapi hasilnya tidak terlihat beda dan jalurnya bisa melipat sendiri.
    """
    count = len(points)
    if count == 0:
        return []
    cx = sum(point[0] for point in points) / count
    cy = sum(point[1] for point in points) / count
    shifted = []
    for point in points:
        dx, dy = point[0] - cx, point[1] - cy
        length = math.hypot(dx, dy)
        if length < 1e-6:
            shifted.append(point)
        else:
            shifted.append((point[0] + dx / length * distance, point[1] + dy / length * distance))
    return shifted


def _point_along(
    edges: list[tuple[tuple[float, float], tuple[float, float], float]],
    total: float,
    position: float,
) -> tuple[float, float]:
    """Titik pada keliling poligon di posisi 0..1."""
    if total <= 0.0 or not edges:
        return edges[0][0] if edges else (0.0, 0.0)
    target = (position % 1.0) * total
    travelled = 0.0
    for start, end, length in edges:
        if length > 0.0 and travelled + length >= target:
            ratio = (target - travelled) / length
            return (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
        travelled += length
    return edges[-1][1]


class Renderer:
    """Menggambar kerangka tangan, tempelan filter, portal, dan glow."""

    def __init__(self, settings: Settings, theme: Theme) -> None:
        self._settings = settings
        self._theme = theme

    # ------------------------------------------------------------------
    # Kerangka tangan
    # ------------------------------------------------------------------

    def draw_skeleton(self, frame: np.ndarray, hands: HandFrame) -> None:
        """Gambar tulang dan sendi semua tangan yang terdeteksi."""
        if not self._settings.show_landmarks:
            return
        if not self._is_frame(frame):
            return

        height, width = frame.shape[:2]
        if height < MIN_SKELETON_SIDE or width < MIN_SKELETON_SIDE:
            return

        scale = self._scale(height)
        bone_width = max(1, int(round(BONE_WIDTH_PX * scale)))
        # Outline ikut mengecil bersama tulang. Kalau tetap 2 piksel di frame
        # kecil, garis gelapnya lebih tebal dari tulangnya sendiri dan kerangka
        # berubah jadi gumpalan hitam.
        outline_width = max(1, int(round(BONE_OUTLINE_PX * scale)))
        joint_radius = max(1, int(round(JOINT_RADIUS_PX * scale)))
        tip_radius = max(2, int(round(TIP_RADIUS_PX * scale)))
        outline = self._theme.panel
        bone_color = self._theme.text_dim
        joint_color = self._theme.text

        for hand in hands.hands:
            if len(hand.landmarks) < 21:
                # Landmark tidak lengkap lebih baik dilewati daripada
                # menggambar garis ke (0, 0).
                continue
            points = [
                (int(round(landmark.x * width)), int(round(landmark.y * height)))
                for landmark in hand.landmarks
            ]

            # Garis gelap tipis di bawah tulang terang: tanpa ini kerangka
            # menghilang di area frame yang terang.
            for start, end in HAND_BONES:
                cv2.line(frame, points[start], points[end], outline, bone_width + outline_width, cv2.LINE_AA)
            for start, end in HAND_BONES:
                cv2.line(frame, points[start], points[end], bone_color, bone_width, cv2.LINE_AA)

            for index, point in enumerate(points):
                if index in (THUMB_TIP, INDEX_TIP):
                    continue
                cv2.circle(frame, point, joint_radius, joint_color, -1, cv2.LINE_AA)

            # Ujung jempol dan telunjuk dibesarkan dan diberi warna aksen
            # karena dua titik itulah yang membentuk sudut portal.
            halo_radius = max(tip_radius, int(round(tip_radius * TIP_HALO_FACTOR)))
            core_radius = max(1, int(round(tip_radius * TIP_CORE_FACTOR)))
            for tip in (THUMB_TIP, INDEX_TIP):
                cv2.circle(frame, points[tip], halo_radius, self._theme.accent_soft, -1, cv2.LINE_AA)
                cv2.circle(frame, points[tip], tip_radius, self._theme.accent, -1, cv2.LINE_AA)
                cv2.circle(frame, points[tip], core_radius, self._theme.text, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Tempelan filter
    # ------------------------------------------------------------------

    def composite(self, frame: np.ndarray, portal: Portal, filtered: np.ndarray) -> None:
        """Tempel hasil filter ke dalam bentuk portal saja.

        Di luar bentuk portal frame tidak disentuh sama sekali, jadi pemakai
        tetap melihat kamera asli di sekeliling jendela filter.
        """
        if not self._is_frame(frame):
            return
        if not portal.valid or not portal.points:
            return
        if filtered is None or getattr(filtered, "size", 0) == 0:
            return

        height, width = frame.shape[:2]
        x, y, box_w, box_h = portal.bounds(width, height)
        if box_w < 2 or box_h < 2:
            # Portal degenerate atau seluruhnya di luar frame.
            return

        patch = self._prepare_patch(filtered, box_w, box_h)
        if patch is None:
            return

        mask = self._portal_alpha(portal, width, height, x, y, box_w, box_h)
        if mask is None:
            return

        region = frame[y : y + box_h, x : x + box_w]
        # Alpha dari mask dipakai sebagai bobot campuran, jadi tepi yang sudah
        # diblur otomatis menghasilkan gradasi beberapa piksel tanpa garis
        # keras. Dihitung dalam uint8 karena frame ini buffer besar; konversi
        # ke float32 per frame memakan beberapa milidetik tanpa beda yang
        # kelihatan.
        mask3 = cv2.merge([mask, mask, mask])
        inverse3 = cv2.bitwise_not(mask3)
        front = cv2.multiply(patch, mask3, scale=1.0 / 255.0)
        back = cv2.multiply(region, inverse3, scale=1.0 / 255.0)
        region[:] = cv2.add(front, back)

    def _portal_alpha(
        self,
        portal: Portal,
        width: int,
        height: int,
        x: int,
        y: int,
        box_w: int,
        box_h: int,
    ) -> np.ndarray | None:
        """Mask portal yang tepinya sudah dilembutkan, seukuran kotak portal.

        Mask dipotong lebih lebar dari kotak lalu dipotong balik sesudah
        diblur. Potongan yang pas dengan kotak membuat GaussianBlur berhenti
        di batas potongan, sehingga tepi lurus portal tidak punya gradasi
        sama sekali dan terlihat sebagai garis keras.
        """
        kernel = _odd(min(EDGE_KERNEL_MAX, round(EDGE_SOFTNESS_PX * self._scale(height) * 2.0) + 1))
        pad = min(EDGE_PAD_MAX, kernel)
        left = max(0, x - pad)
        top = max(0, y - pad)
        right = min(width, x + box_w + pad)
        bottom = min(height, y + box_h + pad)
        if right - left < 2 or bottom - top < 2:
            return None

        wide = portal.mask(width, height)[top:bottom, left:right]
        wide = cv2.GaussianBlur(wide, (kernel, kernel), 0)
        mask = wide[y - top : y - top + box_h, x - left : x - left + box_w]
        if mask.size == 0 or not mask.any():
            return None
        return mask

    def _prepare_patch(self, filtered: np.ndarray, box_w: int, box_h: int) -> np.ndarray | None:
        """Samakan bentuk, tipe, dan ukuran hasil filter dengan kotak portal."""
        patch = np.asarray(filtered)
        if patch.ndim == 2:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        elif patch.ndim == 3 and patch.shape[2] == 4:
            patch = patch[:, :, :3]
        elif patch.ndim != 3 or patch.shape[2] != 3:
            return None

        if patch.dtype != np.uint8:
            patch = np.clip(patch, 0, 255).astype(np.uint8)

        if patch.shape[0] != box_h or patch.shape[1] != box_w:
            # Selisih satu-dua piksel itu normal: kotak pembatas dibulatkan,
            # sedangkan filter menerima potongan frame yang bisa beda tipis.
            patch = cv2.resize(patch, (box_w, box_h), interpolation=cv2.INTER_LINEAR)

        return np.ascontiguousarray(patch)

    # ------------------------------------------------------------------
    # Portal
    # ------------------------------------------------------------------

    def draw_portal(self, frame: np.ndarray, portal: Portal, now_s: float) -> None:
        """Gambar tepi portal, sudut menyala, cincin berdenyut, dan partikel."""
        if not self._is_frame(frame):
            return
        if not portal.valid or not portal.points:
            return

        height, width = frame.shape[:2]
        if height < MIN_SKELETON_SIDE or width < MIN_SKELETON_SIDE:
            return

        scale = self._scale(height)
        quad: list[tuple[float, float]] | None = None
        center = (0.0, 0.0)
        radius = 0.0
        if portal.kind == "circle":
            center = (float(portal.points[0][0]), float(portal.points[0][1]))
            radius = portal.radius
            if radius < MIN_PORTAL_RADIUS_PX:
                return
        else:
            quad = [(float(point[0]), float(point[1])) for point in portal.points]
            if len(quad) < 3:
                return

        pad = int(round(GLOW_PAD_PX * scale)) + 2
        x0, y0, x1, y1 = self._glow_box(portal, width, height, pad)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return

        # Glow digambar di lapisan terpisah berukuran setengah kotak lalu
        # diblur dan dijumlahkan. Menambah cahaya seperti ini membuat tepi
        # portal terasa menyala, bukan cuma garis kuning di atas video.
        region = frame[y0:y1, x0:x1]
        glow_w = max(1, (x1 - x0) // GLOW_DOWNSCALE)
        glow_h = max(1, (y1 - y0) // GLOW_DOWNSCALE)
        factor = (glow_w / float(x1 - x0), glow_h / float(y1 - y0))
        layer = np.zeros((glow_h, glow_w, 3), np.uint8)
        self._stroke_portal(
            layer, quad, center, radius, (x0, y0), factor, scale, GLOW_PEN, now_s,
            corners=True, particles=False,
        )
        kernel = _odd(GLOW_KERNEL_PX * scale * min(factor))
        layer = cv2.GaussianBlur(layer, (kernel, kernel), 0)
        layer = cv2.resize(layer, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        region[:] = cv2.add(region, layer)

        self._stroke_portal(
            frame, quad, center, radius, (0, 0), (1.0, 1.0), scale, 1.0, now_s,
            corners=True, particles=True,
        )

    def _stroke_portal(
        self,
        canvas: np.ndarray,
        quad: list[tuple[float, float]] | None,
        center: tuple[float, float],
        radius: float,
        origin: tuple[int, int],
        factor: tuple[float, float],
        scale: float,
        pen: float,
        now_s: float,
        corners: bool,
        particles: bool,
    ) -> None:
        """Gambar seluruh isi portal ke satu kanvas dengan skala tertentu.

        Dipakai dua kali: sekali untuk lapisan glow (setengah ukuran, pena
        lebih tebal) dan sekali untuk garis tajam di frame asli. Karena
        geometrinya dari fungsi yang sama, glow mustahil meleset dari garisnya.

        Partikel dan sudut bisa dimatikan terpisah karena di lapisan glow
        keduanya sudah terwakili oleh blur garisnya sendiri; menggambar ulang
        di sana hanya membuat tepi portal jenuh jadi putih.
        """
        fx, fy = factor
        ox, oy = origin
        shrink = min(fx, fy)
        thickness = max(1, int(round(PORTAL_LINE_PX * scale * pen * shrink)))
        dot = max(1, int(round(PARTICLE_RADIUS_PX * scale * pen * shrink)))
        accent = self._theme.accent
        soft = self._theme.accent_soft
        edge = _mix(accent, soft, EDGE_LINE_MIX) if particles else accent

        def place(point: tuple[float, float]) -> tuple[int, int]:
            return (int(round((point[0] - ox) * fx)), int(round((point[1] - oy) * fy)))

        pulse = 0.5 + 0.5 * math.sin(now_s * math.tau * RING_SPEED)

        if quad is not None:
            polygon = np.array([place(point) for point in quad], np.int32)
            cx = sum(point[0] for point in quad) / len(quad)
            cy = sum(point[1] for point in quad) / len(quad)
            distances = [math.hypot(point[0] - cx, point[1] - cy) for point in quad]
            reach = min(distances) if distances else 0.0
            # Cincin luar yang mengembang-mengempis pelan, memberi kesan
            # tarikan energi tanpa perlu animasi lain. Tiap sudut digeser
            # sejauh jarak yang sama dari titik pusat, bukan diskalakan, supaya
            # bentuk quad yang tidak simetris tetap terlihat sama.
            ring_grow = self._offset(scale, pulse, reach, RING_MAX_FRACTION, RING_OFFSET_PX, RING_PULSE_PX)
            outer = _offset_polygon(quad, ring_grow)
            cv2.polylines(canvas, [np.array([place(point) for point in outer], np.int32)], True, soft, max(1, thickness // 2), cv2.LINE_AA)
            cv2.polylines(canvas, [polygon], True, edge, thickness, cv2.LINE_AA)
            if corners:
                self._draw_corners(canvas, quad, place, scale)
            if particles:
                # Partikel berjalan di jalur yang digeser sedikit ke luar tepi.
                # Kalau ditempel tepat di atas garis, warna putihnya tenggelam
                # di garis aksen yang sama terang; di luar tepi latar lebih
                # gelap sehingga percikannya terbaca.
                path = _offset_polygon(quad, self._offset(scale, 0.0, reach, PARTICLE_MAX_FRACTION, PARTICLE_OFFSET_PX, 0.0))
                edges, total = _polygon_edges(path)
                self._draw_particles(canvas, place, scale, now_s, dot=dot, edges=edges, total=total)
        else:
            ring_grow = self._offset(scale, pulse, radius, RING_MAX_FRACTION, RING_OFFSET_PX, RING_PULSE_PX)
            ring_radius = radius + ring_grow
            cv2.circle(canvas, place(center), max(1, int(round(ring_radius * shrink))), soft, max(1, thickness // 2), cv2.LINE_AA)
            cv2.circle(canvas, place(center), max(1, int(round(radius * shrink))), edge, thickness, cv2.LINE_AA)
            if corners:
                tick = max(2.0, CORNER_RADIUS_PX * scale * shrink)
                for step in range(4):
                    angle = math.pi / 2.0 * step
                    cos_a, sin_a = math.cos(angle), math.sin(angle)
                    inner = (center[0] + cos_a * (radius - tick), center[1] + sin_a * (radius - tick))
                    outer = (center[0] + cos_a * (radius + tick), center[1] + sin_a * (radius + tick))
                    cv2.line(canvas, place(inner), place(outer), self._theme.text, thickness, cv2.LINE_AA)
            if particles:
                path_radius = radius + self._offset(scale, 0.0, radius, PARTICLE_MAX_FRACTION, PARTICLE_OFFSET_PX, 0.0)
                self._draw_particles(canvas, place, scale, now_s, dot=dot, center=center, radius=path_radius)

    @staticmethod
    def _offset(scale: float, pulse: float, reach: float, limit: float, base: float, swing: float) -> float:
        """Jarak geser ke luar tepi, dibatasi ukuran portal.

        Tanpa batas ini, portal sekecil 15 piksel akan dikelilingi halo
        setinggi portalnya sendiri dan bentuknya hilang.
        """
        wanted = (base + swing * pulse) * scale
        if reach <= 0.0:
            return 0.0
        return min(wanted, reach * limit)

    def _draw_corners(
        self,
        canvas: np.ndarray,
        quad: list[tuple[float, float]],
        place: Callable[[tuple[float, float]], tuple[int, int]],
        scale: float,
    ) -> None:
        """Titik terang di tiap sudut quad, tempat ujung jari menyentuh portal."""
        corner = max(2, int(round(CORNER_RADIUS_PX * scale)))
        for point in quad:
            spot = place(point)
            cv2.circle(canvas, spot, max(corner, int(corner * 1.6)), self._theme.accent_soft, -1, cv2.LINE_AA)
            cv2.circle(canvas, spot, corner, self._theme.accent, -1, cv2.LINE_AA)
            cv2.circle(canvas, spot, max(1, corner // 2), self._theme.text, -1, cv2.LINE_AA)

    def _draw_particles(
        self,
        canvas: np.ndarray,
        place: Callable[[tuple[float, float]], tuple[int, int]],
        scale: float,
        now_s: float,
        dot: int = 2,
        edges: list[tuple[tuple[float, float], tuple[float, float], float]] | None = None,
        total: float = 0.0,
        center: tuple[float, float] | None = None,
        radius: float = 0.0,
    ) -> None:
        """Partikel yang mengalir di sepanjang tepi portal.

        Posisi tiap partikel hanya bergantung pada (waktu, indeksnya), jadi
        alirannya mulus dari frame ke frame. Tidak ada np.random di sini:
        keacakan yang berubah tiap frame terbaca sebagai noise, bukan energi.
        """
        if not self._settings.particles:
            return

        if edges is not None:
            if total <= 0.0:
                return
            count = int(total / (PARTICLE_SPACING_PX * scale))
        else:
            if center is None or radius <= 0.0:
                return
            count = int(math.tau * radius / (PARTICLE_SPACING_PX * scale))
        count = max(PARTICLE_MIN, min(PARTICLE_MAX, count))

        head_color = self._theme.text
        body_color = self._theme.accent

        for index in range(count):
            phase = _hash01(index * 7919 + 13)
            speed = PARTICLE_SPEED * (0.85 + 0.3 * phase)
            offset = (index * GOLDEN_FRACTION + phase * 0.2) % 1.0
            position = offset + now_s * speed
            # Kedipan pelan per partikel supaya alirannya punya ritme dan
            # tidak terlihat seperti titik-titik mati.
            glow = 0.55 + 0.45 * math.sin(now_s * 2.4 + phase * math.tau)

            def spot(at: float) -> tuple[float, float]:
                if edges is not None:
                    return _point_along(edges, total, at)
                angle = at * math.tau
                wobble = math.sin(now_s * 2.0 + phase * math.tau) * 2.0 * scale
                return (
                    center[0] + math.cos(angle) * (radius + wobble),
                    center[1] + math.sin(angle) * (radius + wobble),
                )

            # Ekor digambar sebagai beberapa potong yang makin tipis dan makin
            # redup ke belakang. Satu garis lurus sepanjang ekor terbaca
            # sebagai garis putus-putus di tepi portal, bukan sebagai percikan.
            # Warnanya digeser dari aksen ke putih, bukan ke accent_soft:
            # meredupkan ke warna gelap malah terlihat seperti noda di atas
            # garis tepi yang terang.
            for step in range(PARTICLE_TRAIL_SEGMENTS, 0, -1):
                fade = step / float(PARTICLE_TRAIL_SEGMENTS)
                back = position - PARTICLE_TRAIL * fade
                ahead = position - PARTICLE_TRAIL * (fade - 1.0 / PARTICLE_TRAIL_SEGMENTS)
                width = max(1, int(round(dot * (1.0 - 0.55 * fade))))
                color = _mix(body_color, head_color, glow * (1.0 - 0.7 * fade))
                cv2.line(canvas, place(spot(back)), place(spot(ahead)), color, width, cv2.LINE_AA)

            cv2.circle(canvas, place(spot(position)), dot, _mix(body_color, head_color, glow), -1, cv2.LINE_AA)

    def _glow_box(self, portal: Portal, width: int, height: int, pad: int) -> tuple[int, int, int, int]:
        x, y, box_w, box_h = portal.bounds(width, height)
        return (
            max(0, x - pad),
            max(0, y - pad),
            min(width, x + box_w + pad),
            min(height, y + box_h + pad),
        )

    # ------------------------------------------------------------------
    # Bloom
    # ------------------------------------------------------------------

    def bloom(self, frame: np.ndarray, strength: float = 0.35) -> np.ndarray:
        """Kembalikan salinan frame dengan cahaya area terang disebarkan.

        Blur dilakukan di seperempat ukuran karena glow itu frekuensi rendah;
        memblur 720p penuh dengan kernel besar cuma membuang waktu tanpa
        perbedaan yang kelihatan.
        """
        if not self._is_frame(frame):
            if isinstance(frame, np.ndarray):
                return np.array(frame, copy=True)
            return np.zeros((0, 0, 3), np.uint8)
        height, width = frame.shape[:2]
        if height < BLOOM_MIN_SIDE or width < BLOOM_MIN_SIDE:
            return np.array(frame, copy=True)

        small_w = max(1, width // BLOOM_DOWNSCALE)
        small_h = max(1, height // BLOOM_DOWNSCALE)
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

        # Ambang diambil dari kanal paling terang, bukan dari grayscale. Warna
        # aksen portal kuning punya nilai abu-abu sedang, jadi dengan grayscale
        # tepi portal tidak ikut menyala.
        peak = small.max(axis=2)
        excess = cv2.subtract(peak, BLOOM_THRESHOLD)
        if not excess.any():
            return np.array(frame, copy=True)
        weight = cv2.merge((excess, excess, excess))
        bright = cv2.multiply(small, weight, scale=1.0 / 255.0, dtype=cv2.CV_8U)

        glow = cv2.GaussianBlur(bright, (0, 0), sigmaX=BLOOM_SIGMA)
        glow = cv2.resize(glow, (width, height), interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(frame, 1.0, glow, float(strength), 0.0)

    # ------------------------------------------------------------------
    # Pembantu
    # ------------------------------------------------------------------

    def _scale(self, height: int) -> float:
        return max(MIN_SCALE, min(MAX_SCALE, height / REFERENCE_HEIGHT))

    @staticmethod
    def _is_frame(frame: np.ndarray) -> bool:
        return (
            isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape[2] == 3
            and frame.shape[0] > 0
            and frame.shape[1] > 0
        )
