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
    """Buka kamera, ambil satu frame, laporkan hasilnya.

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
    target = directory / "camera-probe.png"
    cv2.imwrite(str(target), frame)
    print(f"Contoh frame disimpan di {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_filters:
        from hand_portal.filters import build_filters

        for number, spec in enumerate(build_filters(), start=1):
            print(f"{number:2d}. {spec.label:<12} {spec.hint}")
        return 0

    if args.probe_camera:
        return probe_camera(args.camera or 0)

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
