"""Perekam video dan penangkap gambar.

Kelas ini sengaja toleran terhadap kegagalan. Kalau penulis video tidak bisa
dibuka (codec mp4v tidak tersedia, folder tidak bisa ditulis), aplikasi tetap
jalan tanpa rekaman. Loop kamera tidak boleh mati hanya karena fitur rekam
bermasalah.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# mp4v paling luas dukungannya di OpenCV lintas platform; kalau ini pun gagal,
# masalahnya ada di build OpenCV, bukan di kode.
VIDEO_FOURCC = "mp4v"

SNAPSHOT_PREFIX = "handportal"
SNAPSHOT_EXTENSION = ".png"
# Timestamp sampai milidetik bisa bertabrakan kalau snapshot ditekan cepat,
# jadi nama diberi akhiran nomor sebelum menyerah.
MAX_NAME_ATTEMPTS = 100


class Recorder:
    """Penulis video latar belakang plus penyimpan gambar sesekali."""

    def __init__(self, path: str | None, width: int, height: int, fps: int) -> None:
        """Buka penulis video kalau path diisi.

        `path` None berarti mode tanpa rekaman: tidak ada berkas dibuat dan
        `write` tidak melakukan apa pun.
        """
        self._path = str(path) if path else None
        self._size = (max(1, int(width)), max(1, int(height)))
        self._fps = max(1, int(fps))
        self._writer: cv2.VideoWriter | None = None
        self.error: str | None = None

        if self._path:
            self._open_writer(self._path)

    # ------------------------------------------------------------------ publik

    @property
    def path(self) -> str | None:
        """Path berkas video, atau None kalau tidak merekam."""
        return self._path

    @property
    def active(self) -> bool:
        """True kalau penulis video benar-benar terbuka dan siap menerima frame."""
        return self._writer is not None

    def write(self, frame: np.ndarray) -> None:
        """Tulis satu frame; diam saja kalau tidak merekam atau writer gagal."""
        if self._writer is None or frame is None or frame.size == 0:
            return

        prepared = self._prepare_frame(frame)
        if prepared is None:
            return
        try:
            self._writer.write(prepared)
        except Exception as exc:  # noqa: BLE001 - kegagalan disk tidak boleh mematikan loop
            self.error = f"Gagal menulis frame ke rekaman: {exc}"
            self.close()

    def snapshot(self, frame: np.ndarray, directory: str = "captures") -> str:
        """Simpan frame sebagai PNG, kembalikan path lengkapnya."""
        if frame is None or frame.size == 0:
            raise ValueError("Frame kosong, tidak ada yang bisa disimpan.")

        folder = Path(directory).expanduser()
        folder.mkdir(parents=True, exist_ok=True)

        target = folder / f"{SNAPSHOT_PREFIX}-{self._timestamp()}{SNAPSHOT_EXTENSION}"
        attempt = 1
        while target.exists() and attempt < MAX_NAME_ATTEMPTS:
            target = folder / (
                f"{SNAPSHOT_PREFIX}-{self._timestamp()}-{attempt}{SNAPSHOT_EXTENSION}"
            )
            attempt += 1

        if not cv2.imwrite(str(target), frame):
            raise RuntimeError(f"Gagal menyimpan gambar ke {target}.")
        return str(target)

    def close(self) -> None:
        """Tutup penulis video. Aman dipanggil berkali-kali."""
        writer = self._writer
        self._writer = None
        if writer is None:
            return
        try:
            writer.release()
        except Exception:  # noqa: BLE001 - release yang gagal tidak perlu menggagalkan apa pun
            pass

    # ----------------------------------------------------------------- internal

    def _open_writer(self, path: str) -> None:
        parent: Path | None = None
        try:
            parent = Path(path).expanduser().parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.error = f"Folder rekaman tidak bisa dibuat ({parent}): {exc}"
            return

        try:
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
            writer = cv2.VideoWriter(path, fourcc, float(self._fps), self._size)
        except Exception as exc:  # noqa: BLE001 - beberapa build OpenCV melempar di sini
            self.error = f"Penulis video gagal dibuat untuk {path}: {exc}"
            return

        if not writer.isOpened():
            writer.release()
            self.error = (
                f"Penulis video tidak bisa dibuka untuk {path}. "
                "Codec mp4v mungkin tidak tersedia di build OpenCV ini."
            )
            return

        self._writer = writer

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """Samakan ukuran, jumlah kanal, dan tipe frame dengan yang diminta writer."""
        target_width, target_height = self._size
        prepared = frame

        if prepared.ndim == 2:
            prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)
        elif prepared.ndim == 3 and prepared.shape[2] == 4:
            prepared = cv2.cvtColor(prepared, cv2.COLOR_BGRA2BGR)
        elif prepared.ndim != 3:
            return None

        if prepared.dtype != np.uint8:
            prepared = np.clip(prepared, 0, 255).astype(np.uint8)

        if prepared.shape[1] != target_width or prepared.shape[0] != target_height:
            prepared = cv2.resize(
                prepared, (target_width, target_height), interpolation=cv2.INTER_AREA
            )
        return np.ascontiguousarray(prepared)

    @staticmethod
    def _timestamp() -> str:
        """Stempel waktu sampai milidetik, aman dipakai di nama berkas."""
        return datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
