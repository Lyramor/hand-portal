"""Kontrak bersama antar modul.

Semua modul lain impor tipe dari sini, jadi kalau ada yang perlu diubah,
ubahnya di satu tempat dan semuanya ikut. Isi berkas ini sengaja tidak
mengandung logika kamera atau efek apa pun.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cv2
import numpy as np

# Indeks landmark MediaPipe yang kita pakai. Angkanya tetap, jadi aman
# dijadikan konstanta daripada dihafal di tiap modul.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
PINKY_TIP = 20

Point = tuple[float, float]
Color = tuple[int, int, int]  # BGR, seperti yang dipakai OpenCV


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class Hand:
    """Satu tangan dengan 21 landmark, koordinat ternormalisasi 0..1."""

    label: str
    landmarks: tuple[Landmark, ...]

    def point(self, index: int, width: int, height: int) -> Point:
        lm = self.landmarks[index]
        return (lm.x * width, lm.y * height)

    @property
    def thumb_tip(self) -> Landmark:
        return self.landmarks[THUMB_TIP]

    @property
    def index_tip(self) -> Landmark:
        return self.landmarks[INDEX_TIP]

    @property
    def pinky_tip(self) -> Landmark:
        return self.landmarks[PINKY_TIP]

    @property
    def middle_tip(self) -> Landmark:
        return self.landmarks[MIDDLE_TIP]

    def span(self) -> float:
        """Lebar telapak dalam satuan ternormalisasi.

        Dipakai untuk mengukur jarak antar jari secara relatif, supaya gestur
        tetap terdeteksi walau tangan dekat atau jauh dari kamera.
        """
        wrist = self.landmarks[WRIST]
        middle = self.landmarks[MIDDLE_MCP]
        return math.hypot(middle.x - wrist.x, middle.y - wrist.y)


@dataclass(frozen=True)
class HandFrame:
    hands: tuple[Hand, ...]
    timestamp_ms: int

    @property
    def count(self) -> int:
        return len(self.hands)


@dataclass(frozen=True)
class Portal:
    """Bentuk area yang ditempel filter.

    kind "quad" untuk empat ujung jari (jempol dan telunjuk kedua tangan),
    kind "circle" untuk lensa dari satu tangan yang mencubit.
    """

    kind: str
    points: tuple[Point, ...]
    valid: bool = True

    @property
    def center(self) -> Point:
        if self.kind == "circle":
            return self.points[0]
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @property
    def radius(self) -> float:
        if self.kind != "circle":
            return 0.0
        (cx, cy), (rx, ry) = self.points[0], self.points[1]
        return math.hypot(rx - cx, ry - cy)

    def bounds(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Kotak pembatas yang sudah dipotong ke ukuran frame."""
        if self.kind == "circle":
            cx, cy = self.points[0]
            r = self.radius
            x0, y0 = cx - r, cy - r
            x1, y1 = cx + r, cy + r
        else:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

        x0 = max(0, int(round(x0)))
        y0 = max(0, int(round(y0)))
        x1 = min(width, int(round(x1)))
        y1 = min(height, int(round(y1)))
        return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))

    def mask(self, width: int, height: int) -> np.ndarray:
        """Mask penuh frame: 255 di dalam portal, 0 di luar."""
        mask = np.zeros((height, width), np.uint8)
        if not self.valid or not self.points:
            return mask
        if self.kind == "circle":
            cx, cy = self.points[0]
            radius = int(round(self.radius))
            if radius > 0:
                cv2.circle(mask, (int(cx), int(cy)), radius, 255, -1)
            return mask
        polygon = np.array([[int(round(x)), int(round(y))] for x, y in self.points], np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        return mask


@dataclass
class FilterContext:
    """Informasi yang boleh dibaca sebuah filter saat mengolah ROI.

    roi yang dikirim ke filter selalu potongan frame BGR berukuran uint8.
    Filter tidak boleh mengubah frame penuh, cukup kembalikan roi baru.
    """

    frame: np.ndarray
    person_mask: np.ndarray | None
    time_s: float
    portal: Portal | None
    width: int
    height: int


FilterFn = Callable[[np.ndarray, FilterContext], np.ndarray]


@dataclass(frozen=True)
class FilterSpec:
    key: str
    label: str
    hint: str
    fn: FilterFn


@dataclass(frozen=True)
class SwitchEvent:
    """Satu kali pergantian filter. reason dipakai HUD untuk toast."""

    index: int
    reason: str  # "pinch" | "index" | "pinky" | "key"


@dataclass(frozen=True)
class FrameStats:
    """Angka yang ditampilkan di HUD. Diisi renderer, dibaca hud."""

    fps: float = 0.0
    hands: int = 0
    portal_kind: str | None = None
    filter_key: str = ""
    filter_label: str = ""
    filter_index: int = 0
    filter_total: int = 0
    tracking: bool = False
    # Seberapa jauh gestur ganti filter sudah ditahan, 0..1. Nol berarti
    # tidak ada gestur yang sedang ditahan.
    hold_progress: float = 0.0
