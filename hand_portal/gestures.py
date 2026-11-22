"""Deteksi portal dari posisi tangan dan gestur untuk mengganti filter.

Modul ini sengaja tidak menyentuh kamera maupun MediaPipe: masukannya selalu
`HandFrame` yang sudah jadi, jadi seluruh perilakunya bisa diuji dengan tangan
sintetis tanpa membuka perangkat apa pun.
"""

from __future__ import annotations

import math

from .config import Settings
from .contracts import (
    INDEX_TIP,
    MIDDLE_MCP,
    PINKY_TIP,
    THUMB_TIP,
    WRIST,
    Hand,
    HandFrame,
    Point,
    Portal,
    SwitchEvent,
)

# --- Ambang portal -------------------------------------------------------
# Cubit dibaca dengan histeresis: masuk di bawah 0.30 span, baru dilepas di
# atas 0.45 span. Tanpa jarak antara dua angka itu, cubitan yang pas di ambang
# akan berkedip masuk-keluar tiap frame dan portalnya ikut kedip.
PINCH_ENTER_RATIO = 0.30
PINCH_EXIT_RATIO = 0.45

# Batas atas mutlak. Rasio terhadap span sudah menjaga gestur tetap terbaca
# dari jarak berapa pun, tapi rasio saja bisa lolos saat span-nya sendiri kecil
# karena noise. Batas piksel ini yang memastikan tangan jauh tidak memicu.
MIN_SPAN_FRAME_RATIO = 0.020
MAX_PINCH_FRAME_RATIO = 0.160

# --- Ambang gestur pengganti filter -------------------------------------
# Dua ujung telunjuk: paling gampang ketemu, jadi ambang masuknya sedang.
INDEX_ENTER_RATIO = 0.30
INDEX_EXIT_RATIO = 0.50

# Jempol ke ujung kelingking. Pada tangan terbuka jaraknya lebih dari satu
# span, jadi ambangnya dilonggarkan supaya tangan setengah menggenggam pun
# masih bisa memicunya.
PINKY_ENTER_RATIO = 0.35
PINKY_EXIT_RATIO = 0.55

MAX_INDEX_GAP_FRAME_RATIO = 0.250
MAX_PINKY_GAP_FRAME_RATIO = 0.250

# Urutan prioritas saat lebih dari satu gestur terbaca bersamaan. Telunjuk dua
# tangan didahulukan karena hanya bisa terjadi kalau memang dua tangan sengaja
# dirapatkan.
#
# Cubit jempol-telunjuk sengaja TIDAK ada di sini: gestur itu dipakai ketapel
# (lihat slingshot.py), jadi kalau ikut mengganti filter, setiap tarikan
# ketapel akan ikut melompatkan filter.
GESTURE_PRIORITY = ("index", "pinky")

# --- Penghalusan ---------------------------------------------------------
# Alpha dasar diambil dari settings.smoothing, tapi kalau titik bergerak lebih
# dari FAST_MOVE_RATIO tinggi frame dalam satu langkah, bobot frame baru naik
# ke FAST_MOVE_ALPHA. Tanpa ini, gerakan cepat membuat portal tertinggal di
# belakang tangan seperti bayangan.
SLOW_MOVE_RATIO = 0.010
FAST_MOVE_RATIO = 0.035
FAST_MOVE_ALPHA = 0.70
MIN_SMOOTHING_ALPHA = 0.05

# Lantai radius lensa. Setengah jarak cubitan asli hanya belasan piksel, dan
# portal sekecil itu tidak mungkin lolos gerbang luas di bawah. Radius lantai
# ini diturunkan dari gerbang luas itu sendiri, jadi lensa selalu cukup besar
# untuk digambar dan gerbang luas tetap terpenuhi.
CIRCLE_RADIUS_HEADROOM = 1.15


def _span_px(hand: Hand, width: int, height: int) -> float:
    """Lebar telapak dalam piksel, padanan `Hand.span()` yang tidak gepeng.

    `Hand.span()` menghitung jarak pada koordinat ternormalisasi, dan pada
    frame 16:9 satu satuan x tidak sama panjang dengan satu satuan y. Begitu
    jarak dipakai untuk membandingkan dengan titik piksel, skalanya harus
    disamakan per sumbu dulu.
    """
    wrist = hand.landmarks[WRIST]
    middle = hand.landmarks[MIDDLE_MCP]
    return math.hypot((middle.x - wrist.x) * width, (middle.y - wrist.y) * height)


def _distance(first: Point, second: Point) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _clamp(point: Point, width: int, height: int) -> Point:
    """Potong titik ke dalam frame.

    Landmark MediaPipe bisa sedikit keluar bidang gambar saat tangan di tepi
    layar. Titik di luar frame membuat mask dan kotak pembatas di renderer
    memotong ke ukuran nol, jadi dirapikan di sini sekali saja.
    """
    max_x = float(max(width - 1, 0))
    max_y = float(max(height - 1, 0))
    return (min(max(point[0], 0.0), max_x), min(max(point[1], 0.0), max_y))


def _polygon_area(points: tuple[Point, ...]) -> float:
    """Luas poligon dengan rumus shoelace, selalu positif."""
    total = 0.0
    count = len(points)
    for index in range(count):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % count]
        total += x0 * y1 - x1 * y0
    return abs(total) * 0.5


def _order_quad(points: list[Point]) -> tuple[Point, ...]:
    """Susun empat titik jadi poligon yang tidak menyilang.

    Dua titik paling atas jadi sisi atas dan dua sisanya sisi bawah, lalu
    masing-masing diurutkan kiri ke kanan. Urutan keluar: kiri-atas,
    kanan-atas, kanan-bawah, kiri-bawah.
    """
    by_y = sorted(points, key=lambda point: (point[1], point[0]))
    top = sorted(by_y[:2], key=lambda point: point[0])
    bottom = sorted(by_y[2:], key=lambda point: point[0])
    return (top[0], top[1], bottom[1], bottom[0])


class PortalTracker:
    """Melacak bentuk portal dari tangan dan menghaluskannya antar frame."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._kind: str | None = None
        self._points: tuple[Point, ...] = ()
        self._streak = 0
        self._pinch_locked = False

    def reset(self) -> None:
        """Hapus riwayat halus dan penghitung stabilisasi."""
        self._kind = None
        self._points = ()
        self._streak = 0
        self._pinch_locked = False

    def update(self, hands: HandFrame, width: int, height: int) -> Portal | None:
        """Kembalikan portal yang sudah dihaluskan, atau None kalau belum
        cukup stabil."""
        if width <= 0 or height <= 0:
            return None

        shape = self._detect_shape(hands, width, height)
        if shape is None:
            # Tangan hilang sesaat. Riwayat halus sengaja tidak dibuang supaya
            # portal tidak melompat saat deteksi berkedip satu frame; cukup
            # matikan stabilisasinya, jadi portal hilang sampai muncul lagi.
            self._streak = 0
            return None

        kind, raw = shape
        if kind != self._kind:
            # Bentuk berganti, jadi titik lama tidak ada pasangannya lagi.
            self._points = raw
            self._kind = kind
            self._streak = 1
        else:
            self._points = self._smooth(self._points, raw, width, height)
            self._streak += 1

        portal = self._build(kind, self._points, width, height)
        if portal is None:
            return None
        if self._streak < max(1, self._settings.stabilize_frames):
            return None
        return portal

    # -- deteksi bentuk ---------------------------------------------------

    def _detect_shape(
        self, hands: HandFrame, width: int, height: int
    ) -> tuple[str, tuple[Point, ...]] | None:
        # Hanya mode bingkai (dua tangan) yang dipakai. Mode lensa satu tangan
        # dulu dibentuk dengan cubitan, tapi gestur itu sekarang milik ketapel
        # (lihat slingshot.py). Kalau keduanya aktif, setiap tarikan ketapel
        # akan ikut memunculkan lensa di titik yang sama dan hasilnya kacau.
        if hands.count >= 2:
            quad = self._quad_points(hands.hands[0], hands.hands[1], width, height)
            if quad is not None:
                return ("quad", quad)
        return None

    def _quad_points(
        self, first: Hand, second: Hand, width: int, height: int
    ) -> tuple[Point, ...] | None:
        min_span = MIN_SPAN_FRAME_RATIO * min(width, height)
        raw: list[Point] = []
        for hand in (first, second):
            if _span_px(hand, width, height) < min_span:
                return None
            for index in (THUMB_TIP, INDEX_TIP):
                raw.append(_clamp(hand.point(index, width, height), width, height))
        return _order_quad(raw)

    def _circle_points(
        self, hand: Hand, width: int, height: int
    ) -> tuple[Point, Point] | None:
        scale = float(min(width, height))
        span = _span_px(hand, width, height)
        if span < MIN_SPAN_FRAME_RATIO * scale:
            self._pinch_locked = False
            return None

        thumb = hand.point(THUMB_TIP, width, height)
        index = hand.point(INDEX_TIP, width, height)
        gap = _distance(thumb, index)
        if gap > MAX_PINCH_FRAME_RATIO * scale:
            self._pinch_locked = False
            return None

        limit = PINCH_EXIT_RATIO if self._pinch_locked else PINCH_ENTER_RATIO
        if gap / span > limit:
            self._pinch_locked = False
            return None
        self._pinch_locked = True

        center = _clamp(((thumb[0] + index[0]) * 0.5, (thumb[1] + index[1]) * 0.5), width, height)
        radius = max(gap * 0.5, self._radius_floor(width, height))
        edge = _clamp((center[0] + radius, center[1]), width, height)
        return (center, edge)

    def _radius_floor(self, width: int, height: int) -> float:
        area_floor = self._settings.min_portal_area * width * height
        return math.sqrt(max(area_floor, 0.0) / math.pi) * CIRCLE_RADIUS_HEADROOM

    # -- penghalusan dan penyaringan --------------------------------------

    def _smooth(
        self,
        previous: tuple[Point, ...],
        current: tuple[Point, ...],
        width: int,
        height: int,
    ) -> tuple[Point, ...]:
        base = min(max(self._settings.smoothing, MIN_SMOOTHING_ALPHA), 1.0)
        scale = float(min(width, height))
        slow = SLOW_MOVE_RATIO * scale
        fast = FAST_MOVE_RATIO * scale
        fast_alpha = max(base, FAST_MOVE_ALPHA)

        smoothed: list[Point] = []
        for old, new in zip(previous, current):
            moved = _distance(old, new)
            if moved <= slow:
                alpha = base
            elif moved >= fast:
                alpha = fast_alpha
            else:
                blend = (moved - slow) / (fast - slow)
                alpha = base + (fast_alpha - base) * blend
            smoothed.append(
                (
                    old[0] + (new[0] - old[0]) * alpha,
                    old[1] + (new[1] - old[1]) * alpha,
                )
            )
        return tuple(smoothed)

    def _build(
        self, kind: str, points: tuple[Point, ...], width: int, height: int
    ) -> Portal | None:
        if kind == "circle":
            radius = _distance(points[0], points[1])
            if radius <= 0.0:
                return None
            area = math.pi * radius * radius
        else:
            if len(points) != 4:
                return None
            area = _polygon_area(points)
        if area < self._settings.min_portal_area * width * height:
            return None
        return Portal(kind=kind, points=points)


class FilterSwitcher:
    """Mengganti filter lewat gestur yang ditahan atau tombol keyboard."""

    def __init__(self, total: int, settings: Settings) -> None:
        self._settings = settings
        self._total = max(0, int(total))
        self._index = 0
        self._active = {name: False for name in GESTURE_PRIORITY}
        self._held: str | None = None
        self._hold_since: float | None = None
        self._cooldown_until = 0.0
        # Gestur yang sudah dipakai sekali harus dilepas dulu sebelum boleh
        # memicu lagi. Tanpa ini, tangan yang tetap ditahan akan berganti
        # filter terus tiap kali cooldown habis.
        self._spent: str | None = None
        # Tombol keyboard tidak membawa jam, jadi pakai waktu frame terakhir.
        self._now = 0.0

    @property
    def index(self) -> int:
        """Indeks filter yang sedang aktif."""
        return self._index

    def update(
        self, hands: HandFrame, width: int, height: int, now_s: float
    ) -> SwitchEvent | None:
        """Baca gestur; kembalikan SwitchEvent kalau filter berganti."""
        self._now = now_s
        if self._total <= 0:
            return None

        self._refresh(hands, width, height)
        gesture = self._held_gesture()
        if gesture is None:
            self._hold_since = None
            return None

        if now_s < self._cooldown_until:
            # Selama cooldown timer tahan juga tidak jalan. Kalau dibiarkan
            # jalan, filter berganti lagi tepat saat cooldown habis padahal
            # tangan pemakai tidak berubah sedikit pun.
            self._hold_since = None
            return None

        if self._hold_since is None:
            self._hold_since = now_s
            return None
        if now_s - self._hold_since < self._settings.hold_to_lock_s:
            return None

        self._spent = gesture
        self._hold_since = None
        return self._advance(1, gesture, now_s)

    def next(self, reason: str = "key") -> SwitchEvent:
        """Maju satu filter."""
        return self._advance(1, reason, self._now)

    def prev(self, reason: str = "key") -> SwitchEvent:
        """Mundur satu filter."""
        return self._advance(-1, reason, self._now)

    def select(self, index: int, reason: str = "key") -> SwitchEvent:
        """Lompat ke indeks tertentu, dibungkus modulo total."""
        if self._total > 0:
            self._index = int(index) % self._total
        self._start_cooldown()
        return SwitchEvent(self._index, reason)

    # -- gestur -----------------------------------------------------------

    def _refresh(self, hands: HandFrame, width: int, height: int) -> None:
        scale = float(min(width, height))
        min_span = MIN_SPAN_FRAME_RATIO * scale
        usable = [hand for hand in hands.hands if _span_px(hand, width, height) >= min_span]

        pinch_ratio = pinch_gap = None
        pinky_ratio = pinky_gap = None
        index_ratio = index_gap = None

        if usable:
            hand = usable[0]
            span = _span_px(hand, width, height)
            thumb = hand.point(THUMB_TIP, width, height)
            index = hand.point(INDEX_TIP, width, height)
            pinky = hand.point(PINKY_TIP, width, height)
            pinch_gap = _distance(thumb, index)
            pinky_gap = _distance(thumb, pinky)
            pinch_ratio = pinch_gap / span
            pinky_ratio = pinky_gap / span

        if len(usable) >= 2:
            first, second = usable[0], usable[1]
            span = 0.5 * (_span_px(first, width, height) + _span_px(second, width, height))
            index_gap = _distance(
                first.point(INDEX_TIP, width, height),
                second.point(INDEX_TIP, width, height),
            )
            index_ratio = index_gap / span

        self._set_active(
            "index", index_ratio, index_gap, INDEX_ENTER_RATIO, INDEX_EXIT_RATIO,
            MAX_INDEX_GAP_FRAME_RATIO * scale,
        )
        self._set_active(
            "pinky", pinky_ratio, pinky_gap, PINKY_ENTER_RATIO, PINKY_EXIT_RATIO,
            MAX_PINKY_GAP_FRAME_RATIO * scale,
        )

        if self._spent is not None and not self._active[self._spent]:
            self._spent = None

    def _set_active(
        self,
        name: str,
        ratio: float | None,
        gap: float | None,
        enter: float,
        leave: float,
        cap: float,
    ) -> None:
        """Perbarui status gestur dengan histeresis masuk/keluar."""
        if ratio is None or gap is None or gap > cap:
            self._active[name] = False
            return
        if self._active[name]:
            self._active[name] = ratio <= leave
        else:
            self._active[name] = ratio < enter

    def _held_gesture(self) -> str | None:
        """Gestur yang sedang ditahan, mengutamakan yang sudah berjalan.

        Mengunci pilihan pada gestur yang sudah aktif penting supaya gestur
        lain yang kebetulan ikut terbaca tidak memindahkan timer tahan dan
        membuat filter tidak pernah berganti.
        """
        if (
            self._held is not None
            and self._held != self._spent
            and self._active.get(self._held)
        ):
            return self._held
        for name in GESTURE_PRIORITY:
            if name == self._spent or not self._active[name]:
                continue
            if name != self._held:
                self._held = name
                self._hold_since = None
            return name
        self._held = None
        return None

    # -- pergantian indeks ------------------------------------------------

    def _advance(self, step: int, reason: str, now_s: float) -> SwitchEvent:
        if self._total > 0:
            self._index = (self._index + step) % self._total
        self._start_cooldown(now_s)
        return SwitchEvent(self._index, reason)

    def _start_cooldown(self, now_s: float | None = None) -> None:
        self._cooldown_until = (self._now if now_s is None else now_s) + (
            self._settings.switch_cooldown_s
        )
        self._hold_since = None
        # Gestur yang sedang ditahan dianggap terpakai, supaya pilihan tombol
        # tidak dibatalkan sepersekian detik kemudian oleh gestur yang sama.
        if self._held is not None:
            self._spent = self._held
