"""Unduhan model MediaPipe.

Model tidak ikut di dalam repo karena ukurannya besar, jadi diambil sekali
saat pertama dijalankan. Kalau jaringan mati, pesan errornya menyebutkan
URL-nya langsung supaya bisa diunduh manual.
"""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from pathlib import Path

from .config import HAND_MODEL, HAND_MODEL_URL, MODELS_DIR, SEGMENTER_MODEL, SEGMENTER_MODEL_URL

# Model cukup diunduh sekali. Menyimpan ulang berkas 7 MB setiap start
# bikin aplikasi terasa lambat tanpa alasan.
CHUNK_SIZE = 1 << 16
TIMEOUT_S = 120


def _download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "hand-portal/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, CHUNK_SIZE)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Gagal mengunduh {target.name} dari {url}. "
            "Periksa koneksi internet, atau unduh manual lalu letakkan di "
            f"{target}."
        ) from exc
    # Baru dianggap ada setelah unduhan selesai, jadi unduhan yang terputus
    # tidak meninggalkan berkas setengah jadi.
    partial.replace(target)
    return target


def ensure_models(force: bool = False) -> tuple[Path, Path | None]:
    """Pastikan kedua model tersedia.

    Hand landmarker wajib ada karena tanpa itu tidak ada deteksi sama sekali.
    Segmenter hanya dipakai filter yang butuh memisahkan orang dari latar,
    jadi kegagalannya tidak boleh menghentikan aplikasi.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if force or not HAND_MODEL.exists():
        _download(HAND_MODEL_URL, HAND_MODEL)

    segmenter: Path | None = SEGMENTER_MODEL
    if force or not SEGMENTER_MODEL.exists():
        try:
            _download(SEGMENTER_MODEL_URL, SEGMENTER_MODEL)
        except RuntimeError as exc:
            print(f"[peringatan] {exc}")
            print("[peringatan] Filter berlatar (galaxy, matrix) akan memakai efek warna biasa.")
            segmenter = None

    return HAND_MODEL, segmenter
