"""Uji cepat tanpa kamera.

Menjalankan seluruh rangkaian (deteksi, portal, filter, render, HUD) pada
frame buatan. Berguna untuk memastikan tidak ada yang rusak setelah mengubah
kode, tanpa harus duduk di depan webcam.

Pakai: .venv/bin/python tools/selftest.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hand_portal.config import THEME, Settings  # noqa: E402
from hand_portal.contracts import (  # noqa: E402
    FilterContext,
    FrameStats,
    Hand,
    HandFrame,
    Landmark,
    Portal,
)
from hand_portal.filters import apply_filter, build_filters  # noqa: E402
from hand_portal.gestures import FilterSwitcher, PortalTracker  # noqa: E402
from hand_portal.hud import Hud  # noqa: E402
from hand_portal.render import Renderer  # noqa: E402

WIDTH, HEIGHT = 960, 540
OUT = ROOT / "captures"


def synthetic_frame() -> np.ndarray:
    """Gradasi warna plus bentuk, cukup untuk melihat efek filter bekerja."""
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for y in range(HEIGHT):
        frame[y, :] = (int(40 + 120 * y / HEIGHT), 60, int(180 - 120 * y / HEIGHT))
    cv2.circle(frame, (WIDTH // 3, HEIGHT // 2), 110, (200, 180, 60), -1)
    cv2.rectangle(frame, (WIDTH // 2, 90), (WIDTH // 2 + 260, 380), (60, 200, 200), -1)
    cv2.putText(frame, "UJI", (60, HEIGHT - 60), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 4)
    return frame


def fake_hand(
    label: str,
    center: tuple[float, float],
    pinch: bool = False,
    middle_to_thumb: bool = False,
) -> Hand:
    """Tangan palsu dengan geometri yang masuk akal.

    Pergelangan di bawah, pangkal jari tengah di atas, jadi lebar telapak
    (span) sekitar 0.18 tinggi frame. Ujung jempol dan telunjuk diletakkan
    berdekatan kalau `pinch` True, atau terbuka lebar kalau False. Ujung jari
    tengah bisa didekatkan ke jempol lewat `middle_to_thumb` untuk menguji
    gestur ganti filter. Tanpa bentuk yang benar, uji gestur akan gagal karena
    rasio jaraknya ngawur, bukan karena kodenya salah.
    """
    cx, cy = center
    span = 0.18
    wrist = (cx, cy + span)
    middle_mcp = (cx, cy)
    gap = 0.012 if pinch else 0.14
    points = [middle_mcp] * 21
    points = list(points)
    points[0] = wrist
    points[4] = (cx - gap / 2, cy)              # ujung jempol
    points[8] = (cx + gap / 2, cy)              # ujung telunjuk
    points[9] = middle_mcp
    # Jari tengah: menempel ke jempol kalau gestur diminta, kalau tidak
    # dibiarkan jauh supaya tidak memicu apa pun.
    points[12] = (cx - gap / 2 + 0.004, cy) if middle_to_thumb else (cx + 0.06, cy)
    points[20] = (cx + gap / 2 + 0.02, cy + 0.05)  # ujung kelingking
    landmarks = tuple(Landmark(x=float(x), y=float(y)) for x, y in points)
    return Hand(label=label, landmarks=landmarks)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    settings = Settings()
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        status = "ok  " if condition else "GAGAL"
        print(f"[{status}] {name}{(' - ' + detail) if detail else ''}")
        if not condition:
            failures.append(name)

    specs = build_filters()
    check("jumlah filter", len(specs) >= 18, f"{len(specs)} filter")
    check("key unik", len({s.key for s in specs}) == len(specs))

    # -- portal hanya dari dua tangan --
    hands = HandFrame(hands=(fake_hand("left", (0.35, 0.4)), fake_hand("right", (0.65, 0.6))), timestamp_ms=0)
    tracker = PortalTracker(settings)
    portal = None
    for _ in range(settings.stabilize_frames + 2):
        portal = tracker.update(hands, WIDTH, HEIGHT) or portal
    check("portal quad terbentuk", portal is not None and portal.kind == "quad")

    # Satu tangan saja tidak membentuk portal: cubitan sekarang milik ketapel.
    single = HandFrame(hands=(fake_hand("right", (0.5, 0.5), pinch=True),), timestamp_ms=0)
    lens = PortalTracker(settings)
    lens_portal = None
    for _ in range(settings.stabilize_frames + 2):
        lens_portal = lens.update(single, WIDTH, HEIGHT) or lens_portal
    check("satu tangan tidak membentuk portal", lens_portal is None)

    # -- pengganti filter --
    switcher = FilterSwitcher(len(specs), settings)
    before = switcher.index
    switcher.next()
    check("tombol ganti filter", switcher.index == (before + 1) % len(specs))

    # Gestur jempol ke jari tengah, harus ditahan dulu sebelum berganti.
    gesture_switcher = FilterSwitcher(len(specs), settings)
    middle_hands = HandFrame(hands=(fake_hand("right", (0.5, 0.5), middle_to_thumb=True),), timestamp_ms=0)
    early = gesture_switcher.update(middle_hands, WIDTH, HEIGHT, 0.0)
    still_early = gesture_switcher.update(middle_hands, WIDTH, HEIGHT, 0.4)
    locked = gesture_switcher.update(middle_hands, WIDTH, HEIGHT, 1.0)
    check(
        "gestur jempol ke jari tengah harus ditahan",
        early is None and still_early is None and locked is not None and locked.reason == "middle",
    )

    # -- render tiap filter di dalam portal --
    renderer = Renderer(settings, THEME)
    hud = Hud(settings, THEME)
    base = synthetic_frame()

    for spec in specs:
        frame = base.copy()
        target = portal
        assert target is not None
        x, y, w, h = target.bounds(WIDTH, HEIGHT)
        roi = frame[y:y + h, x:x + w]
        ctx = FilterContext(
            frame=frame,
            person_mask=None,
            time_s=time.time(),
            portal=target,
            width=WIDTH,
            height=HEIGHT,
        )
        started = time.perf_counter()
        filtered = apply_filter(roi.copy(), spec, ctx)
        elapsed = (time.perf_counter() - started) * 1000
        ok = filtered.dtype == np.uint8 and filtered.shape[:2] == (h, w)
        check(f"filter {spec.key}", ok, f"{elapsed:.1f} ms")
        renderer.composite(frame, target, filtered)
        renderer.draw_portal(frame, target, time.time())
        renderer.draw_skeleton(frame, hands)
        stats = FrameStats(fps=60.0, hands=2, portal_kind=target.kind, filter_key=spec.key,
                           filter_label=spec.label, filter_index=0, filter_total=len(specs), tracking=True)
        hud.draw(frame, stats, time.time())
        cv2.imwrite(str(OUT / f"filter-{spec.key}.png"), frame)

    # -- filter dengan mask orang --
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.circle(mask, (WIDTH // 2, HEIGHT // 2), 200, 255, -1)
    for spec in specs:
        if spec.key not in {"galaxy", "matrix"}:
            continue
        frame = base.copy()
        target = portal
        assert target is not None
        x, y, w, h = target.bounds(WIDTH, HEIGHT)
        ctx = FilterContext(frame=frame, person_mask=mask, time_s=1.0, portal=target, width=WIDTH, height=HEIGHT)
        filtered = apply_filter(frame[y:y + h, x:x + w].copy(), spec, ctx)
        check(f"filter {spec.key} dengan mask", filtered.shape[:2] == (h, w))
        cv2.imwrite(str(OUT / f"mask-{spec.key}.png"), filtered)

    print(f"\nContoh gambar tersimpan di {OUT}")
    if failures:
        print(f"{len(failures)} uji gagal: {', '.join(failures)}")
        return 1
    print("Semua uji lolos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
