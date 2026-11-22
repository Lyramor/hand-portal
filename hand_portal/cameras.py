"""Memilih kamera yang benar.

Di macOS, Continuity Camera membuat kamera iPhone muncul sebagai perangkat
tambahan. Urutannya bisa berubah-ubah, dan kalau iPhone kebetulan jadi indeks
0, aplikasi akan memakai kamera HP padahal yang dimaksud kamera laptop.

Modul ini membaca daftar perangkat lewat ffmpeg (yang memakai enumerasi
AVFoundation yang sama dengan OpenCV), lalu memilih kamera bawaan kalau ada.
Kalau ffmpeg tidak terpasang, semuanya jatuh ke indeks 0 seperti biasa.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Nama kamera bawaan di berbagai model Mac. Pencocokan dilakukan huruf kecil.
BUILTIN_HINTS = (
    "macbook",
    "built-in",
    "builtin",
    "facetime",
    "imac camera",
    "studio display",
    "isight",
)

# Perangkat yang jelas bukan kamera depan yang dipakai untuk gestur.
EXCLUDED_HINTS = (
    "desk view",
    "capture screen",
)

LIST_TIMEOUT_S = 8
DEVICE_PATTERN = re.compile(r"\[(\d+)\]\s+(.+?)\s*$")


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str

    @property
    def is_builtin(self) -> bool:
        lowered = self.name.lower()
        return any(hint in lowered for hint in BUILTIN_HINTS)

    @property
    def is_usable(self) -> bool:
        lowered = self.name.lower()
        return not any(hint in lowered for hint in EXCLUDED_HINTS)


def list_devices() -> list[CameraDevice]:
    """Daftar kamera video yang terdeteksi, urut sesuai indeks AVFoundation."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return []

    try:
        result = subprocess.run(
            [ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=LIST_TIMEOUT_S,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []

    devices: list[CameraDevice] = []
    inside = False
    for line in result.stderr.splitlines():
        if "AVFoundation video devices" in line:
            inside = True
            continue
        if "AVFoundation audio devices" in line:
            break
        if not inside:
            continue
        # Contoh baris: [AVFoundation indev @ 0x7e1010140] [0] MacBook Air Camera
        tail = line.rsplit("] [", 1)
        if len(tail) != 2:
            continue
        match = DEVICE_PATTERN.match("[" + tail[1])
        if match is None:
            continue
        index_text, name = match.groups()
        devices.append(CameraDevice(index=int(index_text), name=name.strip()))
    return devices


def resolve(preferred: int | None = None) -> int:
    """Tentukan indeks kamera yang dipakai.

    Urutan prioritasnya: pilihan pemakai, lalu pilihan yang terakhir disimpan,
    lalu kamera bawaan hasil deteksi nama, lalu indeks 0.

    Deteksi nama hanya jadi cadangan, bukan sumber utama. Alasannya, urutan
    perangkat yang dilaporkan ffmpeg tidak selalu sama dengan urutan yang
    dipakai OpenCV saat membuka kamera, jadi nama yang terbaca bisa menunjuk
    ke indeks yang salah. Karena itu aplikasi juga punya tombol untuk berganti
    kamera langsung sambil melihat hasilnya.
    """
    if preferred is not None:
        return preferred

    saved = load_preference()
    if saved is not None:
        return saved

    devices = list_devices()
    if not devices:
        return 0

    usable = [device for device in devices if device.is_usable]
    for device in usable:
        if device.is_builtin:
            return device.index
    if usable:
        return usable[0].index
    return 0


def preference_path() -> Path:
    return Path.home() / ".config" / "hand-portal" / "camera.txt"


def load_preference() -> int | None:
    """Indeks kamera yang terakhir dipilih pemakai, kalau ada."""
    try:
        text = preference_path().read_text().strip()
    except OSError:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value >= 0 else None


def save_preference(index: int) -> None:
    """Simpan pilihan kamera supaya tidak perlu diatur lagi tiap kali buka."""
    path = preference_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{int(index)}\n")
    except OSError:
        # Gagal menyimpan bukan alasan menghentikan aplikasi; pilihan hanya
        # tidak bertahan sampai sesi berikutnya.
        pass


def describe(index: int) -> str:
    """Nama kamera untuk ditampilkan di log, atau keterangan kosong."""
    for device in list_devices():
        if device.index == index:
            return device.name
    return f"kamera {index}"
