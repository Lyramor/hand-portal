"""Ketapel: cubit jempol dan telunjuk, tarik ke belakang, lalu lepas.

Perilakunya meniru ketapel sungguhan. Saat cubitan mulai, titik cubit dicatat
sebagai tumpuan. Selama tangan ditarik menjauh, jarak tarikan itu menjadi
kekuatan. Begitu cubitan dilepas, peluru melesat ke arah berlawanan dari
tarikan dan pecah berhamburan saat sampai.

Modul ini memegang logika sekaligus penggambarannya. Alasannya, ketapel punya
banyak potongan visual yang saling bergantung (karet, bola, jejak, pecahan),
dan menyebarkannya ke dua berkas justru membuat keduanya harus tahu terlalu
banyak hal yang sama.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Settings, Theme
from .contracts import INDEX_TIP, THUMB_TIP, Hand, HandFrame, Point

# Histeresis cubitan. Ambang masuk dan keluar dibedakan supaya cubitan yang
# pas di batas tidak berkedip masuk-keluar dan memicu tembakan berulang.
PINCH_ENTER_RATIO = 0.32
PINCH_EXIT_RATIO = 0.48
MAX_PINCH_FRAME_RATIO = 0.16

# Tangan yang terlalu jauh tidak dipakai: landmark-nya terlalu berisik untuk
# dijadikan dasar tarikan.
MIN_SPAN_FRAME_RATIO = 0.02

MAX_PROJECTILES = 12
MAX_BURSTS = 10

# Pecahan saat peluru buyar.
BURST_PARTICLES = 34
BURST_LIFE_S = 0.62
BURST_RING_LIFE_S = 0.34

# Bentuk visual ketapel, semuanya dalam piksel acuan 720p lalu diskalakan.
PRONG_SPREAD_PX = 17.0
BAND_WIDTH_PX = 3.0
ORB_RADIUS_PX = 9.0
CHARGE_RING_RADIUS_PX = 22.0
TRAIL_LENGTH = 9
GRAVITY_RATIO = 0.55
PROJECTILE_LIFE_S = 1.6

REFERENCE_HEIGHT = 720.0


@dataclass(frozen=True)
class Shot:
    """Satu tembakan yang baru dilepas."""

    origin: Point
    direction: tuple[float, float]
    power: float


@dataclass(frozen=True)
class SlingshotState:
    """Keadaan ketapel untuk digambar pada frame ini."""

    charging: bool
    anchor: Point
    pull: Point
    power: float


@dataclass
class Projectile:
    position: Point
    velocity: tuple[float, float]
    age: float
    power: float
    trail: list[Point] = field(default_factory=list)


@dataclass
class Burst:
    position: Point
    age: float
    power: float
    seed: int


def _distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _span_px(hand: Hand, width: int, height: int) -> float:
    """Lebar telapak dalam piksel, dengan skala per sumbu disamakan."""
    wrist = hand.landmarks[0]
    middle = hand.landmarks[9]
    return math.hypot((middle.x - wrist.x) * width, (middle.y - wrist.y) * height)


def _pinch_point(hand: Hand, width: int, height: int) -> Point:
    thumb = hand.point(THUMB_TIP, width, height)
    index = hand.point(INDEX_TIP, width, height)
    return ((thumb[0] + index[0]) * 0.5, (thumb[1] + index[1]) * 0.5)


class SlingshotTracker:
    """Membaca gestur ketapel dari satu tangan."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pinching = False
        self._anchor: Point | None = None
        self._pull: Point | None = None
        self._started_s = 0.0
        self._cooldown_until = 0.0

    def reset(self) -> None:
        self._pinching = False
        self._anchor = None
        self._pull = None

    def update(
        self, hands: HandFrame, width: int, height: int, now_s: float
    ) -> Shot | None:
        """Perbarui ketapel; kembalikan Shot kalau peluru baru dilepas."""
        if width <= 0 or height <= 0:
            return None

        scale = float(min(width, height))
        hand = self._pick_hand(hands, width, height)
        if hand is None:
            # Tangan hilang: cubitan dianggap lepas, tapi tanpa tembakan,
            # karena pemakai tidak mungkin menarik ketapel lalu pergi.
            self.reset()
            return None

        thumb = hand.point(THUMB_TIP, width, height)
        index = hand.point(INDEX_TIP, width, height)
        gap = _distance(thumb, index)
        span = max(_span_px(hand, width, height), 1.0)

        limit = PINCH_EXIT_RATIO if self._pinching else PINCH_ENTER_RATIO
        pinching = gap <= limit * span and gap <= MAX_PINCH_FRAME_RATIO * scale

        point = _pinch_point(hand, width, height)

        if pinching and not self._pinching:
            self._pinching = True
            self._anchor = point
            self._pull = point
            self._started_s = now_s
            return None

        if pinching:
            self._pull = point
            return None

        # Cubitan baru saja lepas.
        released = self._pinching
        self._pinching = False
        anchor = self._anchor
        pull = self._pull
        held_s = now_s - self._started_s
        self._anchor = None
        self._pull = None

        if not released or anchor is None or pull is None:
            return None
        if now_s < self._cooldown_until:
            return None
        if held_s < self._settings.slingshot_charge_s:
            return None

        min_pull = self._settings.slingshot_min_pull * width
        pull_distance = _distance(anchor, pull)
        if pull_distance < min_pull:
            return None

        max_pull = max(self._settings.slingshot_max_pull * width, min_pull + 1.0)
        power = min(1.0, (pull_distance - min_pull) / (max_pull - min_pull))
        power = max(power, 0.08)

        # Arah tembakan berlawanan dengan arah tarikan, seperti karet yang
        # dilepas. Tarikan nol akan menghasilkan arah acak, jadi dipaksa ke
        # atas layar sebagai gantinya.
        dx = anchor[0] - pull[0]
        dy = anchor[1] - pull[1]
        length = math.hypot(dx, dy)
        if length < 1e-3:
            dx, dy, length = 0.0, -1.0, 1.0

        self._cooldown_until = now_s + self._settings.slingshot_cooldown_s
        return Shot(origin=anchor, direction=(dx / length, dy / length), power=power)

    def state(self, width: int, height: int) -> SlingshotState | None:
        """Keadaan untuk digambar. None kalau tidak sedang menarik."""
        if not self._pinching or self._anchor is None or self._pull is None:
            return None
        min_pull = self._settings.slingshot_min_pull * width
        max_pull = max(self._settings.slingshot_max_pull * width, min_pull + 1.0)
        distance = _distance(self._anchor, self._pull)
        power = 0.0
        if distance > min_pull:
            power = min(1.0, (distance - min_pull) / (max_pull - min_pull))
        return SlingshotState(charging=True, anchor=self._anchor, pull=self._pull, power=power)

    def _pick_hand(self, hands: HandFrame, width: int, height: int) -> Hand | None:
        """Pilih tangan yang paling layak untuk ketapel.

        Kalau ada dua tangan, yang dipakai adalah yang jari-jarinya paling
        rapat, karena itulah yang sedang mencubit.
        """
        candidates = [
            hand
            for hand in hands.hands
            if _span_px(hand, width, height) >= MIN_SPAN_FRAME_RATIO * min(width, height)
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        def gap(hand: Hand) -> float:
            return _distance(
                hand.point(THUMB_TIP, width, height),
                hand.point(INDEX_TIP, width, height),
            )

        return min(candidates, key=gap)


class Projectiles:
    """Peluru yang sedang melayang beserta pecahannya."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._items: list[Projectile] = []
        self._bursts: list[Burst] = []
        self._seed = 1

    def clear(self) -> None:
        self._items.clear()
        self._bursts.clear()

    def fire(self, shot: Shot, width: int, height: int) -> None:
        scale = float(min(width, height))
        speed = (0.75 + 1.5 * shot.power) * scale
        velocity = (shot.direction[0] * speed, shot.direction[1] * speed)
        self._items.append(
            Projectile(
                position=shot.origin,
                velocity=velocity,
                age=0.0,
                power=shot.power,
                trail=[shot.origin],
            )
        )
        if len(self._items) > MAX_PROJECTILES:
            self._items.pop(0)

    def update(self, dt: float, width: int, height: int) -> None:
        """Majukan peluru dan pecahan; pecahkan yang habis masa hidupnya."""
        if dt <= 0:
            return
        gravity = GRAVITY_RATIO * min(width, height)

        alive: list[Projectile] = []
        for item in self._items:
            vx, vy = item.velocity
            vy += gravity * dt
            x = item.position[0] + vx * dt
            y = item.position[1] + vy * dt
            item.position = (x, y)
            item.velocity = (vx, vy)
            item.age += dt
            item.trail.append(item.position)
            if len(item.trail) > TRAIL_LENGTH:
                item.trail.pop(0)

            off_screen = x < -40 or x > width + 40 or y < -40 or y > height + 40
            if off_screen or item.age >= PROJECTILE_LIFE_S:
                self._spawn_burst(item.position, item.power)
                continue
            alive.append(item)
        self._items = alive

        live_bursts: list[Burst] = []
        for burst in self._bursts:
            burst.age += dt
            if burst.age < BURST_LIFE_S:
                live_bursts.append(burst)
        self._bursts = live_bursts

    def _spawn_burst(self, position: Point, power: float) -> None:
        self._seed += 1
        self._bursts.append(Burst(position=position, age=0.0, power=power, seed=self._seed))
        if len(self._bursts) > MAX_BURSTS:
            self._bursts.pop(0)

    @property
    def empty(self) -> bool:
        return not self._items and not self._bursts


# ----------------------------------------------------------------------
# Penggambaran
# ----------------------------------------------------------------------


def _scale_for(height: int) -> float:
    return max(0.4, min(2.6, height / REFERENCE_HEIGHT))


def _draw_band(
    frame: np.ndarray,
    anchor: Point,
    pull: Point,
    scale: float,
    color: tuple[int, int, int],
) -> None:
    """Gambar karet ketapel sebagai dua garis dari garpu ke titik tarik."""
    dx = anchor[0] - pull[0]
    dy = anchor[1] - pull[1]
    length = math.hypot(dx, dy)
    if length < 1e-3:
        return
    # Vektor tegak lurus dipakai untuk menentukan posisi dua ujung garpu.
    nx, ny = -dy / length, dx / length
    spread = PRONG_SPREAD_PX * scale
    left = (anchor[0] + nx * spread, anchor[1] + ny * spread)
    right = (anchor[0] - nx * spread, anchor[1] - ny * spread)

    width = max(1, int(round(BAND_WIDTH_PX * scale)))
    for prong in (left, right):
        cv2.line(
            frame,
            (int(round(prong[0])), int(round(prong[1]))),
            (int(round(pull[0])), int(round(pull[1]))),
            color,
            width,
            cv2.LINE_AA,
        )
        cv2.circle(frame, (int(round(prong[0])), int(round(prong[1]))), max(2, width + 1), color, -1, cv2.LINE_AA)


def draw_slingshot(
    frame: np.ndarray,
    state: SlingshotState | None,
    projectiles: Projectiles,
    theme: Theme,
    now_s: float,
) -> None:
    """Gambar ketapel yang sedang ditarik dan semua peluru di udara."""
    height = frame.shape[0]
    scale = _scale_for(height)

    if state is not None:
        pull = (int(round(state.pull[0])), int(round(state.pull[1])))
        anchor = (int(round(state.anchor[0])), int(round(state.anchor[1])))

        _draw_band(frame, anchor, state.pull, scale, theme.accent_soft)

        # Bola yang sedang ditarik membesar mengikuti kekuatan, jadi pemakai
        # bisa melihat kapan tarikannya sudah penuh.
        radius = max(3, int(round((ORB_RADIUS_PX + 7.0 * state.power) * scale)))
        glow = max(radius + 2, int(round(radius * 2.1)))
        cv2.circle(frame, pull, glow, theme.accent_soft, max(1, int(round(2 * scale))), cv2.LINE_AA)
        cv2.circle(frame, pull, radius, theme.accent, -1, cv2.LINE_AA)
        cv2.circle(frame, pull, max(1, radius // 2), (255, 255, 255), -1, cv2.LINE_AA)

        # Cincin pengisian: penuh berarti tarikan sudah maksimal.
        if state.power > 0.02:
            ring_radius = int(round(CHARGE_RING_RADIUS_PX * scale))
            cv2.ellipse(
                frame,
                pull,
                (ring_radius, ring_radius),
                0,
                -90,
                -90 + int(360 * state.power),
                theme.accent,
                max(1, int(round(2 * scale))),
                cv2.LINE_AA,
            )

    for item in projectiles._items:  # noqa: SLF001 - satu modul, sengaja langsung
        points = item.trail
        for index, point in enumerate(points):
            ratio = (index + 1) / len(points)
            radius = max(1, int(round((2.0 + 5.0 * ratio * item.power) * scale)))
            color = tuple(int(c * (0.25 + 0.75 * ratio)) for c in theme.accent)
            cv2.circle(
                frame,
                (int(round(point[0])), int(round(point[1]))),
                radius,
                color,
                -1,
                cv2.LINE_AA,
            )
        head = (int(round(item.position[0])), int(round(item.position[1])))
        core = max(2, int(round((ORB_RADIUS_PX * 0.8) * scale)))
        cv2.circle(frame, head, core + 3, theme.accent_soft, -1, cv2.LINE_AA)
        cv2.circle(frame, head, core, theme.accent, -1, cv2.LINE_AA)
        cv2.circle(frame, head, max(1, core // 2), (255, 255, 255), -1, cv2.LINE_AA)

    for burst in projectiles._bursts:  # noqa: SLF001
        progress = min(1.0, burst.age / BURST_LIFE_S)
        fade = 1.0 - progress
        cx, cy = burst.position
        travel = (0.35 + 0.9 * burst.power) * min(frame.shape[0], frame.shape[1]) * progress

        for index in range(BURST_PARTICLES):
            # Sudut disebar merata lalu digeser sedikit per pecahan, supaya
            # polanya berbeda tiap tembakan tanpa perlu angka acak.
            angle = (2 * math.pi * index) / BURST_PARTICLES + burst.seed * 0.7
            wobble = 0.65 + 0.35 * math.sin(burst.seed * 1.7 + index * 2.3)
            distance = travel * wobble
            px = int(round(cx + math.cos(angle) * distance))
            py = int(round(cy + math.sin(angle) * distance))
            radius = max(1, int(round((4.0 * fade + 1.0) * scale)))
            if index % 3 == 0:
                color = theme.accent
            elif index % 3 == 1:
                color = theme.text
            else:
                color = theme.accent_soft
            color = tuple(int(c * fade) for c in color)
            cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)

        if burst.age < BURST_RING_LIFE_S:
            ring_progress = burst.age / BURST_RING_LIFE_S
            ring_radius = int(round(travel * 1.15 + 6 * scale))
            ring_fade = 1.0 - ring_progress
            ring_color = tuple(int(c * ring_fade) for c in theme.accent)
            cv2.circle(
                frame,
                (int(round(cx)), int(round(cy))),
                max(2, ring_radius),
                ring_color,
                max(1, int(round(2 * scale))),
                cv2.LINE_AA,
            )
