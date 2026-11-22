"""Uji ketapel tanpa kamera: tarik, lepas, pecah.

Skrip ini menjalankan SlingshotTracker dengan tangan sintetis yang mencubit
lalu bergerak menjauh, melepasnya, dan merekam gambar di beberapa titik supaya
bentuk karet, bola, jejak, dan pecahannya bisa diperiksa langsung.

Pakai: .venv/bin/python tools/slingshot_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hand_portal.config import THEME, Settings  # noqa: E402
from hand_portal.contracts import Hand, HandFrame, Landmark  # noqa: E402
from hand_portal.slingshot import (  # noqa: E402
    Projectiles,
    SlingshotTracker,
    draw_slingshot,
)

WIDTH, HEIGHT = 960, 540
OUT = ROOT / "captures"


def scene() -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for y in range(HEIGHT):
        frame[y, :] = (int(35 + 90 * y / HEIGHT), 60, int(140 - 70 * y / HEIGHT))
    cv2.putText(frame, "KETAPEL", (40, HEIGHT - 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, (230, 230, 230), 2)
    return frame


def pinched_hand(center: tuple[float, float], gap: float) -> Hand:
    """Tangan dengan jempol dan telunjuk berjarak `gap` (satuan ternormalisasi).

    `gap` kecil berarti mencubit, `gap` besar berarti jari terbuka.
    """
    cx, cy = center
    span = 0.18
    points = [(cx, cy)] * 21
    points = list(points)
    points[0] = (cx, cy + span)
    points[4] = (cx - gap / 2, cy)
    points[8] = (cx + gap / 2, cy)
    points[9] = (cx, cy)
    points[20] = (cx + gap / 2 + 0.02, cy + 0.05)
    return Hand(label="right", landmarks=tuple(Landmark(x=float(x), y=float(y)) for x, y in points))


def main() -> int:
    OUT.mkdir(exist_ok=True)
    settings = Settings()
    tracker = SlingshotTracker(settings)
    shots = Projectiles(settings)

    frames: list[np.ndarray] = []
    log: list[str] = []

    def step(center: tuple[float, float], gap: float, now: float, dt: float = 1 / 60) -> None:
        hands = HandFrame(hands=(pinched_hand(center, gap),), timestamp_ms=int(now * 1000))
        shot = tracker.update(hands, WIDTH, HEIGHT, now)
        if shot is not None:
            shots.fire(shot, WIDTH, HEIGHT)
            log.append(f"  tembakan: power {shot.power:.2f} arah {shot.direction[0]:.2f},{shot.direction[1]:.2f}")
        shots.update(dt, WIDTH, HEIGHT)
        frame = scene()
        draw_slingshot(frame, tracker.state(WIDTH, HEIGHT), shots, THEME, now)
        frames.append(frame)

    # 1. cubit di tengah, tarik ke kiri bawah, tahan
    step((0.55, 0.45), 0.010, 0.0)
    for index in range(1, 14):
        ratio = index / 13
        step((0.55 - 0.13 * ratio, 0.45 + 0.09 * ratio), 0.010, index / 60)
    cv2.imwrite(str(OUT / "ketapel-tarik.png"), frames[-1])
    pull_state = tracker.state(WIDTH, HEIGHT)
    print(f"tarik: power {pull_state.power:.2f}" if pull_state else "tarik: tidak terbaca")

    # 2. lepas -> peluru melesat ke kanan atas
    release_frame_index = len(frames)
    step((0.42, 0.54), 0.10, 0.25)
    for index in range(1, 16):
        step((0.42, 0.54), 0.10, 0.25 + index / 60)
    cv2.imwrite(str(OUT / "ketapel-melesat.png"), frames[release_frame_index + 8])

    # 3. maju sampai pecah
    for index in range(16, 70):
        step((0.42, 0.54), 0.10, 0.25 + index / 60)
    cv2.imwrite(str(OUT / "ketapel-pecah.png"), frames[release_frame_index + 40])
    cv2.imwrite(str(OUT / "ketapel-pecah-akhir.png"), frames[-1])

    print("\n".join(log) if log else "  tidak ada tembakan")
    print(f"peluru aktif: {len(shots._items)}, pecahan aktif: {len(shots._bursts)}")
    print(f"gambar disimpan di {OUT}")
    return 0 if log else 1


if __name__ == "__main__":
    raise SystemExit(main())
