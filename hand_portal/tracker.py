"""Pembungkus MediaPipe Tasks untuk mendeteksi tangan dan memisahkan badan.

Dua model dipakai di sini dan sengaja dipisah:

- HandLandmarker dijalankan setiap frame, karena hasilnya yang menggerakkan
  portal dan gestur.
- ImageSegmenter jauh lebih berat, jadi hanya dipanggil saat pemanggil benar
  benar minta mask badan (filter berlatar saja yang butuh). Kalau segmenter
  gagal dimuat, aplikasi tetap jalan dengan filter warna biasa.

Keduanya memakai RunningMode.VIDEO, yang artinya MediaPipe melacak antar
frame dan menuntut timestamp naik monoton. Itu sebabnya kelas ini menyimpan
timestamp terakhir dan menaikkannya sendiri kalau pemanggil mengirim angka
yang sama atau lebih kecil.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from . import config
from .config import Settings
from .contracts import Hand, HandFrame, Landmark

# Nilai piksel pada mask yang dikembalikan person_mask.
MASK_PERSON = 255
MASK_BACKGROUND = 0

# Pada selfie_segmenter, kategori 0 adalah latar dan sisanya badan orang. Jadi
# cukup "bukan nol" daripada menghafal nomor kategori yang bisa berubah antar
# versi model.
PERSON_CATEGORY_MIN = 1

# MediaPipe menuntut timestamp yang benar benar lebih besar dari sebelumnya.
TIMESTAMP_STEP_MS = 1

# Segmenter bisa rusak di tengah jalan (misal frame rusak atau backend gagal).
# Setelah beberapa kegagalan beruntun, modelnya dimatikan supaya tiap frame
# tidak membayar biaya inferensi yang pasti gagal.
MAX_SEGMENTER_FAILURES = 5

# Pesan ini sengaja menyebut nama berkas dan perintah unduhnya, karena
# pengguna yang baru clone repo tidak tahu apa yang hilang.
MISSING_HAND_MODEL_MESSAGE = (
    "Berkas model tangan '{name}' tidak ditemukan di '{path}'. "
    "Jalankan `python main.py --download-models` untuk mengunduhnya, "
    "lalu jalankan aplikasi lagi."
)


class HandTracker:
    """Deteksi tangan dan mask badan dari frame kamera BGR."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        # Timestamp terakhir per aliran. Nilai -1 menandakan belum pernah
        # dipanggil, jadi timestamp pertama dari pemanggil selalu diterima.
        self._last_hand_ts = -1
        self._last_mask_ts = -1

        # Alasan segmenter mati, kalau ada. Aplikasi boleh menampilkannya di
        # HUD, tapi tidak wajib. Kosong berarti segmenter siap dipakai.
        # Diisi lebih dulu karena _create_segmenter menulis ke sini.
        self.segmenter_error: str | None = None
        self._segmenter_failures = 0

        self._hand_landmarker = self._create_hand_landmarker()
        self._segmenter = self._create_segmenter()

    # ------------------------------------------------------------------
    # Pembuatan model
    # ------------------------------------------------------------------

    def _create_hand_landmarker(self) -> vision.HandLandmarker:
        path = config.HAND_MODEL
        if not path.exists():
            # Model tangan wajib: tanpa ini tidak ada portal sama sekali,
            # jadi lebih baik gagal keras dengan petunjuk yang jelas daripada
            # berjalan dengan layar kosong tanpa penjelasan.
            raise FileNotFoundError(
                MISSING_HAND_MODEL_MESSAGE.format(name=path.name, path=path.parent)
            )

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self._settings.num_hands,
            min_hand_detection_confidence=self._settings.detect_confidence,
            min_hand_presence_confidence=self._settings.presence_confidence,
            min_tracking_confidence=self._settings.track_confidence,
        )
        return vision.HandLandmarker.create_from_options(options)

    def _create_segmenter(self) -> vision.ImageSegmenter | None:
        path = config.SEGMENTER_MODEL
        if not path.exists():
            # Berbeda dari model tangan, segmenter hanya menambah kualitas
            # filter. Tidak adanya berkas bukan alasan untuk menghentikan
            # aplikasi.
            self.segmenter_error = (
                f"Berkas model segmenter '{path.name}' tidak ditemukan di "
                f"'{path.parent}'. Filter berlatar akan memakai efek warna biasa."
            )
            return None

        try:
            options = vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.VIDEO,
                # Hanya category mask yang dipakai, confidence mask akan
                # membuang waktu dan memori tanpa dipakai siapa pun.
                output_confidence_masks=False,
                output_category_mask=True,
            )
            return vision.ImageSegmenter.create_from_options(options)
        except Exception as error:  # noqa: BLE001 - kegagalan model tidak boleh mematikan app
            # Berkas bisa ada tapi rusak atau tidak cocok dengan versi
            # MediaPipe. Pesannya disimpan supaya bisa ditampilkan.
            self.segmenter_error = f"Segmenter gagal dimuat: {error}"
            return None

    # ------------------------------------------------------------------
    # Deteksi tangan
    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HandFrame:
        """Deteksi tangan pada frame BGR dan kembalikan kontrak HandFrame.

        Frame kosong tetap mengembalikan HandFrame tanpa tangan, bukan
        exception, karena kamera sesekali memberi frame gagal baca.
        """
        if not self._is_usable_frame(frame_bgr):
            return HandFrame(hands=(), timestamp_ms=int(timestamp_ms))

        timestamp = self._next_hand_timestamp(timestamp_ms)
        image = self._to_mp_image(frame_bgr)
        result = self._hand_landmarker.detect_for_video(image, timestamp)

        hands: list[Hand] = []
        for index, raw_landmarks in enumerate(result.hand_landmarks):
            if not raw_landmarks:
                continue
            hands.append(
                Hand(
                    label=self._label_for(result, index),
                    landmarks=tuple(
                        Landmark(
                            x=float(lm.x),
                            y=float(lm.y),
                            z=float(getattr(lm, "z", 0.0) or 0.0),
                        )
                        for lm in raw_landmarks
                    ),
                )
            )

        return HandFrame(hands=tuple(hands), timestamp_ms=timestamp)

    def _label_for(self, result: vision.HandLandmarkerResult, index: int) -> str:
        """Ambil label kiri/kanan, atau nomor urut kalau model tidak yakin.

        Label mengikuti MediaPipe terhadap gambar yang diterima, jadi saat
        pemanggil sudah mem-flip frame, hasilnya ikut tercermin. Itu memang
        yang diinginkan untuk tampilan cermin.
        """
        handedness = getattr(result, "handedness", None) or []
        if index < len(handedness):
            categories = handedness[index] or []
            if categories:
                name = (getattr(categories[0], "category_name", "") or "").strip().lower()
                if name in ("left", "right"):
                    return name
        return f"hand{index}"

    # ------------------------------------------------------------------
    # Mask badan
    # ------------------------------------------------------------------

    def person_mask(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        """Mask uint8 seukuran frame: 255 di badan orang, 0 di latar.

        None kalau segmenter tidak tersedia atau inferensi gagal.
        """
        if self._segmenter is None:
            return None
        if not self._is_usable_frame(frame_bgr):
            return None

        timestamp = self._next_mask_timestamp(timestamp_ms)
        image = self._to_mp_image(frame_bgr)

        try:
            result = self._segmenter.segment_for_video(image, timestamp)
        except Exception as error:  # noqa: BLE001 - satu frame gagal bukan alasan crash
            self._note_segmenter_failure(error)
            return None

        self._segmenter_failures = 0
        return self._mask_from_result(result, frame_bgr.shape[:2])

    def _mask_from_result(
        self, result: vision.ImageSegmenterResult, size: tuple[int, int]
    ) -> np.ndarray | None:
        category_mask = getattr(result, "category_mask", None)
        if category_mask is None:
            return None

        categories = np.asarray(category_mask.numpy_view())
        if categories.ndim == 3:
            # Sebagian build mengembalikan (h, w, 1); ambil satu kanal saja.
            categories = categories[..., 0]

        person = np.where(categories >= PERSON_CATEGORY_MIN, MASK_PERSON, MASK_BACKGROUND)
        person = person.astype(np.uint8)

        height, width = size
        if person.shape[:2] != (height, width):
            # Ukuran keluaran segmenter bisa sedikit berbeda dari frame.
            # NEAREST dipakai supaya mask tetap tegas, tidak jadi gradasi abu.
            person = cv2.resize(person, (width, height), interpolation=cv2.INTER_NEAREST)

        return person

    def _note_segmenter_failure(self, error: Exception) -> None:
        self._segmenter_failures += 1
        self.segmenter_error = f"Segmenter gagal saat memproses frame: {error}"
        if self._segmenter_failures >= MAX_SEGMENTER_FAILURES:
            self.close_segmenter()
            self.segmenter_error = (
                f"Segmenter dimatikan setelah {MAX_SEGMENTER_FAILURES} kegagalan "
                f"beruntun: {error}"
            )

    def close_segmenter(self) -> None:
        """Matikan segmenter tanpa menyentuh hand landmarker."""
        if self._segmenter is not None:
            try:
                self._segmenter.close()
            except Exception:  # noqa: BLE001 - close ganda atau sudah mati
                pass
            self._segmenter = None

    @property
    def has_segmenter(self) -> bool:
        """True kalau mask badan bisa diminta."""
        return self._segmenter is not None

    # ------------------------------------------------------------------
    # Utilitas
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Lepas kedua model MediaPipe. Aman dipanggil berkali-kali."""
        self.close_segmenter()
        if self._hand_landmarker is not None:
            try:
                self._hand_landmarker.close()
            except Exception:  # noqa: BLE001 - close ganda saat keluar aplikasi
                pass
            self._hand_landmarker = None

    def _is_usable_frame(self, frame_bgr: np.ndarray | None) -> bool:
        if frame_bgr is None:
            return False
        frame = np.asarray(frame_bgr)
        return frame.ndim >= 2 and frame.size > 0

    def _to_mp_image(self, frame_bgr: np.ndarray) -> mp.Image:
        """Ubah frame BGR OpenCV jadi mp.Image RGB yang contiguous."""
        frame = np.asarray(frame_bgr)
        if frame.ndim == 2:
            # Kamera abu abu sesekali muncul di perangkat tertentu.
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # MediaPipe membaca buffer tanpa menyalin, jadi array harus benar benar
        # rapat di memori dan tidak boleh jadi view dari frame pemanggil.
        rgb = np.ascontiguousarray(rgb)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def _next_hand_timestamp(self, timestamp_ms: int) -> int:
        timestamp = int(timestamp_ms)
        if timestamp <= self._last_hand_ts:
            timestamp = self._last_hand_ts + TIMESTAMP_STEP_MS
        self._last_hand_ts = timestamp
        return timestamp

    def _next_mask_timestamp(self, timestamp_ms: int) -> int:
        """Timestamp segmenter harus naik di atas timestamp tangan juga.

        Kedua task berbagi sumbu waktu yang sama di aplikasi ini, dan mask
        diminta setelah tangan pada frame yang sama. Menyamakan acuannya
        menghindari segmenter tertinggal di masa lalu saat pemanggil mengirim
        timestamp lama.
        """
        timestamp = max(
            int(timestamp_ms),
            self._last_hand_ts + TIMESTAMP_STEP_MS,
            self._last_mask_ts + TIMESTAMP_STEP_MS,
        )
        self._last_mask_ts = timestamp
        return timestamp
