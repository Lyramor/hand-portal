"""Uji integrasi app.py tanpa kamera dan tanpa jendela.

Seluruh loop `App._step` dijalankan dengan frame buatan, sementara `imshow`
dan `waitKey` diganti tiruan. Tujuannya memastikan penyambungan antar modul
benar: portal terbentuk, filter ditempel, tombol bekerja, dan rekaman menulis
berkas. Kalau ada yang salah sambung, uji ini yang menangkap lebih dulu,
sebelum kita duduk di depan webcam.

Pakai: .venv/bin/python tools/appcheck.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hand_portal import app as app_module  # noqa: E402
from hand_portal.config import Settings  # noqa: E402
from hand_portal.contracts import Hand, HandFrame, Landmark  # noqa: E402

WIDTH, HEIGHT = 960, 540
OUT = ROOT / "captures"


def camera_frame(phase: float) -> np.ndarray:
    """Frame kamera tiruan dengan sesuatu yang bergerak."""
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for y in range(HEIGHT):
        frame[y, :] = (int(30 + 100 * y / HEIGHT), 55, int(150 - 80 * y / HEIGHT))
    x = int(WIDTH * 0.5 + np.sin(phase) * 120)
    cv2.circle(frame, (x, HEIGHT // 2), 90, (210, 190, 70), -1)
    cv2.putText(frame, "KAMERA TIRUAN", (40, HEIGHT - 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    return frame


def fake_hand(label: str, center: tuple[float, float], pinch: bool = False) -> Hand:
    """Tangan dengan bentuk wajar supaya portal benar-benar terbentuk.

    Ujung jempol dan telunjuk harus terbuka lebar untuk mode bingkai, karena
    kalau keduanya rapat keempat titik portal jadi segaris dan luasnya di
    bawah ambang, sehingga portal sengaja tidak ditampilkan.
    """
    cx, cy = center
    span = 0.18
    gap = 0.012 if pinch else 0.16
    points = [(cx, cy)] * 21
    points = list(points)
    points[0] = (cx, cy + span)
    points[4] = (cx - gap / 2, cy)
    points[8] = (cx + gap / 2, cy)
    points[9] = (cx, cy)
    points[20] = (cx + gap / 2 + 0.02, cy + 0.05)
    return Hand(label=label, landmarks=tuple(Landmark(x=float(x), y=float(y)) for x, y in points))


class FakeGrabber:
    def __init__(self, phase: float) -> None:
        self.phase = phase

    def latest(self) -> np.ndarray:
        return camera_frame(self.phase)

    def stop(self) -> None:
        pass


def main() -> int:
    OUT.mkdir(exist_ok=True)
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        print(f"[{'ok  ' if condition else 'GAGAL'}] {name}{(' - ' + detail) if detail else ''}")
        if not condition:
            failures.append(name)

    shown: list[np.ndarray] = []

    # imshow dan waitKey diganti tiruan: frame disimpan, tombol diambil dari
    # antrean yang sudah disiapkan supaya urutannya bisa diuji.
    keys: list[int] = []

    def fake_imshow(_title: str, image: np.ndarray) -> None:
        shown.append(image.copy())

    def fake_waitkey(_delay: int) -> int:
        return keys.pop(0) if keys else 255

    app_module.cv2.imshow = fake_imshow
    app_module.cv2.waitKey = fake_waitkey

    settings = Settings(width=WIDTH, height=HEIGHT)
    app = app_module.App(settings)
    app.grabber = FakeGrabber(0.0)  # type: ignore[assignment]

    # Tangan tiruan diumpankan langsung, menggantikan deteksi MediaPipe.
    hands = HandFrame(
        hands=(fake_hand("left", (0.36, 0.42)), fake_hand("right", (0.64, 0.58))),
        timestamp_ms=0,
    )
    app.tracker.detect = lambda frame, ts: hands  # type: ignore[method-assign]

    # -- beberapa frame biasa --
    for index in range(6):
        app.grabber.phase = index * 0.3  # type: ignore[attr-defined]
        app._step()

    check("frame tergambar", len(shown) == 6, f"{len(shown)} frame")
    check("ukuran frame utuh", all(f.shape[:2] == (HEIGHT, WIDTH) for f in shown))
    check("portal terbentuk", app.portal_tracker is not None)
    cv2.imwrite(str(OUT / "app-berjalan.png"), shown[-1])

    # -- tombol ganti filter --
    before = app.switcher.index
    keys.append(ord("]"))
    app._step()
    check("tombol ] ganti filter", app.switcher.index == (before + 1) % len(app.filters))
    keys.append(ord("["))
    app._step()
    check("tombol [ kembali", app.switcher.index == before)

    # -- lompat langsung --
    keys.append(ord("5"))
    app._step()
    check("tombol 5 lompat ke indeks 4", app.switcher.index == 4)

    # -- sakelar tampilan --
    hud_before = app.show_hud
    keys.append(ord("h"))
    app._step()
    check("tombol H sakelar HUD", app.show_hud is not hud_before)

    mirror_before = app.mirror
    keys.append(ord("m"))
    app._step()
    check("tombol M sakelar cermin", app.mirror is not mirror_before)

    # -- simpan foto --
    keys.append(ord(" "))
    app._step()
    photos = sorted(OUT.glob("handportal-*.png"), key=lambda p: p.stat().st_mtime)
    check("tombol spasi simpan foto", bool(photos), photos[-1].name if photos else "tidak ada")

    # -- rekam lalu hentikan --
    keys.append(ord("r"))
    app._step()
    check("tombol R mulai merekam", app.recording is True)
    for _ in range(10):
        app._step()
    keys.append(ord("r"))
    app._step()
    check("tombol R hentikan rekaman", app.recording is False)

    # -- tanpa tangan sama sekali --
    app.tracker.detect = lambda frame, ts: HandFrame(hands=(), timestamp_ms=ts)  # type: ignore[method-assign]
    for _ in range(4):
        app._step()
    check("tetap jalan tanpa tangan", len(shown) >= 20, f"{len(shown)} frame")

    # -- filter berlatar memakai segmenter --
    app.switcher.select([s.key for s in app.filters].index("galaxy"))
    app.tracker.detect = lambda frame, ts: hands  # type: ignore[method-assign]
    # HUD dinyalakan lagi supaya gambar contoh terakhir utuh seperti tampilan
    # yang dilihat pemakai.
    app.show_hud = True
    for _ in range(4):
        app._step()
    check("filter galaxy jalan", app.filters[app.switcher.index].key == "galaxy")
    cv2.imwrite(str(OUT / "app-galaxy.png"), shown[-1])

    # -- ketapel: cubit, tarik, lepas --
    def sling_hand(center: tuple[float, float], gap: float) -> Hand:
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

    clock = [100.0]

    def sling_step(center: tuple[float, float], gap: float) -> None:
        # Waktu dimajukan sendiri supaya uji tidak bergantung pada jam asli.
        clock[0] += 1 / 60
        hand = sling_hand(center, gap)
        frame_hands = HandFrame(hands=(hand,), timestamp_ms=int(clock[0] * 1000))
        app.tracker.detect = lambda f, ts, h=frame_hands: h  # type: ignore[method-assign]
        app.grabber.phase = clock[0]  # type: ignore[attr-defined]
        # _step memakai jam asli, jadi ketapel diuji langsung supaya waktunya
        # bisa dikendalikan.
        shot = app.slingshot.update(frame_hands, WIDTH, HEIGHT, clock[0])
        if shot is not None:
            app.projectiles.fire(shot, WIDTH, HEIGHT)
        app.projectiles.update(1 / 60, WIDTH, HEIGHT)
        app._step()

    # cubit di tengah
    sling_step((0.55, 0.45), 0.010)
    # tarik ke kiri bawah
    for index in range(1, 14):
        ratio = index / 13
        sling_step((0.55 - 0.13 * ratio, 0.45 + 0.09 * ratio), 0.010)
    check("ketapel sedang ditarik", app.slingshot.state(WIDTH, HEIGHT) is not None)

    # lepas
    sling_step((0.42, 0.54), 0.11)
    check("peluru meluncur setelah dilepas", len(app.projectiles._items) >= 1)  # noqa: SLF001

    for _ in range(45):
        sling_step((0.42, 0.54), 0.11)
    check("peluru pecah setelah sampai", len(app.projectiles._bursts) >= 1)  # noqa: SLF001
    cv2.imwrite(str(OUT / "app-ketapel.png"), shown[-1])

    # -- ganti filter otomatis --
    app.switcher.select(0)
    app.auto_cycle = True
    before_auto = app.switcher.index
    app._auto_next_s = 0.0
    app._auto_cycle(500.0)  # panggilan pertama hanya memulai hitungan
    app._auto_cycle(500.0 + app.settings.auto_cycle_s + 0.1)
    check("ganti otomatis memajukan filter", app.switcher.index != before_auto)
    app.auto_cycle = False

    # -- tombol keluar --
    keys.append(ord("q"))
    check("tombol Q keluar", app._step() is False)

    app.close()
    print(f"\nContoh gambar di {OUT}")
    if failures:
        print(f"{len(failures)} uji gagal: {', '.join(failures)}")
        return 1
    print("Semua uji integrasi lolos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
