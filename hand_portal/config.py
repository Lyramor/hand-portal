"""Setelan aplikasi dan token tampilan.

Nilai di sini sengaja dibuat longgar: kalau di laptop terasa berat, turunkan
resolusi dan matikan partikel lewat argumen CLI di main.py.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

Color = tuple[int, int, int]  # BGR

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
SEGMENTER_MODEL = MODELS_DIR / "selfie_segmenter.tflite"

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
SEGMENTER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/1/selfie_segmenter.tflite"
)


@dataclass(frozen=True)
class Theme:
    """Palet BGR. Semua modul ambil warna dari sini, jangan hardcode."""

    accent: Color = (250, 210, 30)
    accent_soft: Color = (140, 120, 20)
    # Warna garis tepi portal. Sengaja hampir hitam: garis berwarna terang
    # menarik perhatian ke tepinya, padahal yang ingin dilihat isi portalnya.
    portal_line: Color = (18, 18, 18)
    # Bola ketapel dan pecahannya. Hitam dengan tepi lebih gelap supaya
    # bentuknya tetap terbaca di atas latar apa pun.
    orb_fill: Color = (24, 24, 24)
    orb_edge: Color = (0, 0, 0)
    text: Color = (245, 245, 245)
    text_dim: Color = (160, 160, 160)
    panel: Color = (18, 18, 18)
    success: Color = (120, 220, 120)
    warning: Color = (60, 190, 250)
    danger: Color = (80, 80, 235)


THEME = Theme()


@dataclass(frozen=True)
class Settings:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    mirror: bool = True

    num_hands: int = 2
    detect_confidence: float = 0.6
    presence_confidence: float = 0.6
    track_confidence: float = 0.6

    # Semakin kecil, portal makin cepat mengikuti tangan tapi makin goyang.
    smoothing: float = 0.35
    # Portal harus bertahan beberapa frame sebelum digambar, biar tidak kedip.
    stabilize_frames: int = 3
    # Luas minimum portal relatif terhadap frame.
    min_portal_area: float = 0.015

    switch_cooldown_s: float = 0.45
    # Cukup untuk membedakan gestur yang disengaja dari tangan yang sekadar
    # lewat, tapi tetap terasa langsung. Lebih lama dari ini terasa seperti
    # gesturnya tidak berfungsi.
    hold_to_lock_s: float = 0.30

    # Filter berganti sendiri setiap sekian detik. 0 berarti mati.
    auto_cycle_s: float = 5.0

    # Ketapel: tarik cubitan lalu lepas untuk meluncurkan peluru.
    slingshot_min_pull: float = 0.045      # jarak tarik minimum, relatif lebar frame
    slingshot_max_pull: float = 0.30       # tarikan penuh, menentukan kekuatan
    slingshot_charge_s: float = 0.10       # waktu tarik sebelum dianggap siap
    slingshot_cooldown_s: float = 0.35
    # Garis karet ketapel dimatikan secara bawaan. Yang terlihat hanya bola
    # yang ditarik dan pelurunya, jadi layar tidak penuh garis.
    slingshot_bands: bool = False

    # Ketebalan garis, dikalikan skala tinggi frame.
    portal_line_scale: float = 0.35
    edge_softness_scale: float = 0.5

    # Hiasan tepi portal: cincin berdenyut, partikel yang mengalir, dan glow.
    # Bawaannya mati. Yang tersisa hanya garis tipis biasa, karena hiasan itu
    # membuat tepi terasa ramai dan mengalihkan perhatian dari gambarnya.
    portal_effects: bool = False

    particles: bool = True
    bloom: bool = True
    hud: bool = True
    # Garis kerangka tangan dimatikan secara bawaan: yang penting hanya titik
    # ujung jari, karena garisnya menutupi wajah dan terasa berantakan.
    show_landmarks: bool = False
    fullscreen: bool = False
    record_path: str | None = None
    fps_target: int = 60

    def with_overrides(self, **kwargs: object) -> "Settings":
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)
