"""Titik masuk aplikasi.

Contoh pemakaian:
    python main.py
    python main.py --camera 1 --width 1920 --height 1080
    python main.py --record demo.mp4
    python main.py --list-filters
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from hand_portal import cameras
from hand_portal.app import App
from hand_portal.assets import ensure_models
from hand_portal.config import HAND_MODEL, SEGMENTER_MODEL, Settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hand-portal",
        description="Filter kamera dengan portal dari jari tangan.",
    )
    parser.add_argument("--camera", type=int, default=None, help="Indeks kamera (default 0, biasanya kamera bawaan laptop)")
    parser.add_argument("--list-cameras", action="store_true", help="Tampilkan kamera yang terdeteksi lalu keluar")
    parser.add_argument("--auto-cycle", type=float, default=None, help="Ganti filter otomatis tiap N detik (0 mematikan)")
    parser.add_argument("--bones", action="store_true", help="Tampilkan garis kerangka tangan")
    parser.add_argument("--width", type=int, default=None, help="Lebar tangkapan (default 1280)")
    parser.add_argument("--height", type=int, default=None, help="Tinggi tangkapan (default 720)")
    parser.add_argument("--smooth", type=float, default=None, help="Kehalusan portal 0..1, makin kecil makin halus")
    parser.add_argument("--record", default=None, help="Rekam langsung ke berkas mp4 ini")
    parser.add_argument("--no-mirror", action="store_true", help="Matikan efek cermin")
    parser.add_argument("--no-hud", action="store_true", help="Sembunyikan panel informasi")
    parser.add_argument("--no-particles", action="store_true", help="Matikan partikel tepi portal")
    parser.add_argument("--no-bloom", action="store_true", help="Matikan glow lembut")
    parser.add_argument("--no-landmarks", action="store_true", help="Sembunyikan kerangka tangan (bawaan sudah sembunyi)")
    parser.add_argument("--fullscreen", action="store_true", help="Buka dalam layar penuh")
    parser.add_argument("--download-models", action="store_true", help="Unduh ulang model lalu keluar")
    parser.add_argument("--list-filters", action="store_true", help="Tampilkan daftar filter lalu keluar")
    parser.add_argument("--probe-camera", action="store_true", help="Cek kamera sebentar lalu keluar")
    parser.add_argument("--probe-all", action="store_true", help="Coba semua kamera dan simpan contoh gambarnya")
    parser.add_argument("--diagnose-cameras", action="store_true", help="Laporkan keadaan tiap kamera, termasuk yang gelap")
    parser.add_argument("--capture-frames", action="store_true", help="Simpan frame tiap kamera untuk diperiksa")
    return parser.parse_args(argv)


def build_settings(args: argparse.Namespace) -> Settings:
    base = Settings()
    # Kamera dipilih di sini, bukan di dalam App, supaya nilai yang dipakai
    # bisa dicetak ke log sebelum jendela dibuka.
    camera_index = cameras.resolve(args.camera)
    return base.with_overrides(
        camera_index=camera_index,
        width=args.width,
        height=args.height,
        smoothing=args.smooth,
        auto_cycle_s=args.auto_cycle,
        mirror=False if args.no_mirror else None,
        hud=False if args.no_hud else None,
        particles=False if args.no_particles else None,
        bloom=False if args.no_bloom else None,
        show_landmarks=True if args.bones else None,
        fullscreen=True if args.fullscreen else None,
        record_path=args.record,
    )


def list_cameras() -> int:
    """Tampilkan kamera yang terdeteksi beserta indeksnya.

    macOS sering punya lebih dari satu kamera (bawaan, Continuity dari iPhone,
    dan Desk View), dan urutannya bisa berubah. Perintah ini menunjukkan mana
    yang akan dipilih otomatis.
    """
    devices = cameras.list_devices()
    if not devices:
        print(
            "Tidak ada kamera yang terbaca.\n"
            "Kalau ffmpeg tidak terpasang, daftar nama tidak bisa dibaca. "
            "Coba indeks satu per satu dengan --camera 0, lalu 1, lalu 2.",
            file=sys.stderr,
        )
        return 1

    chosen = cameras.resolve(None)
    for device in devices:
        marks: list[str] = []
        if device.is_builtin:
            marks.append("bawaan")
        if not device.is_usable:
            marks.append("dilewati")
        if device.index == chosen:
            marks.append("dipakai")
        suffix = f"  ({', '.join(marks)})" if marks else ""
        print(f"  {device.index:>2}  {device.name}{suffix}")

    print("\nPakai: python main.py --camera <nomor>")
    return 0


def probe_camera(index: int) -> int:
    """Buka satu kamera, ambil satu frame, laporkan hasilnya.

    Dipakai untuk memisahkan dua masalah yang mirip di layar: kamera yang
    tidak diizinkan sistem, dan kamera yang dipakai aplikasi lain.
    """
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(index, getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY))
    if not capture.isOpened():
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        print(
            f"Kamera {index} tidak bisa dibuka.\n"
            "Kemungkinan penyebab:\n"
            "  1. Izin kamera belum diberikan untuk aplikasi ini.\n"
            "     Buka System Settings > Privacy & Security > Camera.\n"
            "  2. Kamera sedang dipakai aplikasi lain (Zoom, Meet, OBS).\n"
            "  3. Tidak ada kamera di perangkat ini.",
            file=sys.stderr,
        )
        return 1

    ok, frame = None, None
    # Kamera butuh beberapa ratus milidetik untuk bangun, dan frame pertama
    # sering masih gelap karena auto-exposure belum menyesuaikan. Tanpa
    # percobaan ulang, kamera yang sehat bisa dilaporkan rusak.
    for _ in range(30):
        ok, frame = capture.read()
        if ok and frame is not None and float(np.mean(frame)) > 18.0:
            break
        time.sleep(0.1)
    capture.release()
    if not ok or frame is None:
        print(
            "Kamera terbuka tetapi tidak memberi gambar. Kemungkinan sedang "
            "dipakai aplikasi lain (Zoom, Meet, OBS) atau kamera perangkat "
            "ini sedang bermasalah.",
            file=sys.stderr,
        )
        return 1

    height, width = frame.shape[:2]
    print(f"Kamera {index} siap: {width}x{height}.")

    # Satu frame disimpan supaya kamera yang terbuka bisa dipastikan memang
    # yang dimaksud, bukan kamera lain yang kebetulan satu indeks.
    directory = Path(__file__).resolve().parent / "captures"
    directory.mkdir(exist_ok=True)
    target = directory / f"camera-probe-{index}.png"
    cv2.imwrite(str(target), frame)
    print(f"Contoh frame disimpan di {target}")
    return 0


def diagnose_cameras(limit: int = 8) -> int:
    """Laporkan keadaan tiap indeks kamera apa adanya.

    Berbeda dari probe, frame gelap tetap dilaporkan beserta kecerahannya.
    Frame gelap yang konsisten biasanya berarti kamera itu sedang dipakai
    aplikasi lain atau sensornya belum siap pada resolusi yang diminta, jadi
    beberapa resolusi dicoba sebelum menyimpulkan.
    """
    import cv2
    import numpy as np

    resolutions = ((1280, 720), (640, 480), (1920, 1080))
    names = {device.index: device.name for device in cameras.list_devices()}
    print("indeks  nama                        resolusi      kecerahan")
    for index in range(limit):
        for width, height in resolutions:
            capture = cv2.VideoCapture(index, getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY))
            if not capture.isOpened():
                capture.release()
                break
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            samples: list[float] = []
            for _ in range(20):
                ok, frame = capture.read()
                if ok and frame is not None:
                    samples.append(round(float(np.mean(frame)), 1))
                time.sleep(0.1)
            capture.release()
            brightness = f"{min(samples):.0f}-{max(samples):.0f}" if samples else "tidak ada"
            print(f"{index:>6}  {names.get(index, '?'):<26}  {width}x{height:<7} {brightness}")
            if samples and max(samples) > 18:
                break

    print(
        "\nKecerahan 0 pada semua resolusi berarti kamera terbuka tetapi hanya\n"
        "memberi frame hitam. Penyebab paling sering: kamera sedang dipakai\n"
        "aplikasi lain, lensa tertutup, atau Continuity Camera mengambil alih\n"
        "kamera bawaan."
    )
    return 0


def capture_frames(limit: int = 8) -> int:
    """Simpan frame terbaik dari tiap kamera yang bisa dibuka.

    Dipakai untuk memastikan kamera mana yang benar-benar sedang tampil.
    Nama perangkat dan urutan indeks tidak bisa dipercaya sepenuhnya, jadi
    satu-satunya bukti yang sah adalah gambarnya sendiri.
    """
    import cv2
    import numpy as np

    from hand_portal.app import FALLBACK_RESOLUTIONS, _try_open

    names = {device.index: device.name for device in cameras.list_devices()}
    directory = Path(__file__).resolve().parent / "captures"
    directory.mkdir(exist_ok=True)

    found = 0
    for index in range(limit):
        chosen = None
        for width, height in ((1280, 720),) + FALLBACK_RESOLUTIONS:
            capture, size = _try_open(index, width, height)
            if capture is None:
                continue
            best = None
            best_mean = -1.0
            for _ in range(15):
                ok, frame = capture.read()
                if ok and frame is not None:
                    value = float(np.mean(frame))
                    if value > best_mean:
                        best, best_mean = frame, value
                time.sleep(0.08)
            capture.release()
            if best is not None and best_mean > 12:
                chosen = (best, size, best_mean)
                break
        if chosen is None:
            continue

        frame, size, brightness = chosen
        target = directory / f"kamera-{index}.png"
        cv2.imwrite(str(target), frame)
        print(
            f"  {index}  {names.get(index, '?'):<24} {size[0]}x{size[1]}  "
            f"terang {brightness:.0f}  {target}"
        )
        found += 1

    if found == 0:
        print("Tidak ada kamera yang memberi gambar.", file=sys.stderr)
        return 1
    return 0


def probe_all_cameras(limit: int = 8) -> int:
    """Coba semua indeks dan simpan satu frame dari masing-masing.

    Dipakai saat tidak jelas kamera mana yang sebenarnya dipakai OpenCV.
    Nama perangkat tidak bisa dipercaya sepenuhnya, jadi satu-satunya cara
    yang pasti adalah membuka tiap indeks dan melihat gambarnya.
    """
    import cv2
    import numpy as np

    found = 0
    for index in range(limit):
        capture = cv2.VideoCapture(index, getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY))
        if not capture.isOpened():
            capture.release()
            continue
        ok, frame = None, None
        for _ in range(25):
            ok, frame = capture.read()
            if ok and frame is not None and float(np.mean(frame)) > 18.0:
                break
            time.sleep(0.1)
        capture.release()
        if not ok or frame is None:
            continue
        # Frame gelap dilewati. Kamera yang baru dibuka sering memberi satu
        # frame hitam, dan frame itu tidak berguna untuk menilai kamera mana
        # yang sedang dibuka.
        if float(np.mean(frame)) <= 18.0:
            continue

        directory = Path(__file__).resolve().parent / "captures"
        directory.mkdir(exist_ok=True)
        target = directory / f"camera-{index}.png"
        cv2.imwrite(str(target), frame)
        height, width = frame.shape[:2]
        print(f"  {index}  {width}x{height}  {target}")
        found += 1

    if found == 0:
        print("Tidak ada kamera yang bisa dibuka.", file=sys.stderr)
        return 1
    print("\nBandingkan gambarnya, lalu pilih: python main.py --camera <nomor>")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_filters:
        from hand_portal.filters import build_filters

        for number, spec in enumerate(build_filters(), start=1):
            print(f"{number:2d}. {spec.label:<12} {spec.hint}")
        return 0

    if args.probe_camera:
        return probe_camera(cameras.resolve(args.camera))

    if args.probe_all:
        return probe_all_cameras()

    if args.diagnose_cameras:
        return diagnose_cameras()

    if args.capture_frames:
        return capture_frames()

    if args.list_cameras:
        return list_cameras()

    if args.download_models:
        hand, segmenter = ensure_models(force=True)
        print(f"Model siap: {hand}")
        print(f"Segmenter: {segmenter if segmenter else 'tidak tersedia'}")
        return 0

    try:
        ensure_models()
    except RuntimeError as exc:
        print(f"[gagal] {exc}", file=sys.stderr)
        return 1

    if not HAND_MODEL.exists():
        print(f"[gagal] Model tidak ditemukan di {HAND_MODEL}", file=sys.stderr)
        return 1
    if not SEGMENTER_MODEL.exists():
        print("[info] Segmenter tidak ada, filter berlatar akan memakai efek warna.")

    settings = build_settings(args)
    print(f"Memakai kamera {settings.camera_index}: {cameras.describe(settings.camera_index)}", flush=True)
    try:
        return App(settings).run()
    except RuntimeError as exc:
        print(f"[gagal] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
