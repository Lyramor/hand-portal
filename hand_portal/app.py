"""Loop utama: kamera masuk, deteksi tangan, filter ditempel di portal.

Modul ini satu-satunya tempat yang boleh menyentuh `cv2.imshow` dan kamera.
Semua bagian lain dibuat agar bisa diuji tanpa perangkat keras.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import cameras
from .config import THEME, Settings
from .contracts import FilterContext, FrameStats, Portal, SwitchEvent
from .filters import apply_filter, build_filters
from .gestures import FilterSwitcher, PortalTracker
from .hud import Hud
from .recorder import Recorder
from .render import Renderer
from .slingshot import Projectiles, SlingshotTracker, draw_slingshot
from .tracker import HandTracker

WINDOW_TITLE = "Hand Portal"

# Filter yang butuh pemisahan orang dari latar. Segmenter mahal, jadi hanya
# dijalankan saat filter ini aktif, bukan setiap frame.
MASK_FILTERS = frozenset({"galaxy", "matrix"})

# Mask yang menutupi hampir seluruh frame berarti segmenter salah membaca
# (frame gelap dianggap orang semua), jadi tidak dipakai.
MAX_PERSON_COVERAGE = 0.95

# Seberapa banyak indeks yang dicoba saat pemakai menekan tombol ganti kamera.
# Delapan sudah lebih dari cukup untuk kamera bawaan, Continuity, dan layar.
MAX_CAMERA_PROBE = 8

SWITCH_REASON_TEXT = {
    "pinch": "Filter diganti (cubit)",
    "index": "Filter diganti (telunjuk bertemu)",
    "middle": "Filter diganti (jempol dan jari tengah)",
    "pinky": "Filter diganti (jempol dan kelingking)",
    "auto": "Filter berganti otomatis",
    "key": "Filter diganti",
}

HELP_LINES = (
    "Q keluar   [ ] ganti filter   1-9 pilih cepat   A ganti otomatis   C kamera",
    "SPACE simpan foto   R rekam   H sembunyikan HUD",
    "P partikel   L kerangka tangan   F layar penuh   M cermin",
)


@dataclass
class _FrameGrabber:
    """Membaca kamera di thread terpisah.

    Tanpa ini, frame yang diproses bisa tertinggal beberapa ratus milidetik
    karena buffer kamera menumpuk saat loop kita sibuk menggambar.
    """

    capture: cv2.VideoCapture
    frame: np.ndarray | None = None
    running: bool = True

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self.frame = frame

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        self.running = False
        self._thread.join(timeout=1.0)


# Resolusi yang dicoba berurutan saat membuka kamera. Sebagian kamera
# bawaan Mac memberi frame hitam pada resolusi tertinggi walaupun melaporkan
# ukuran itu, jadi resolusi yang lebih kecil perlu dicoba sebelum menyerah.
FALLBACK_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (1280, 720),
    (960, 540),
    (640, 480),
)

# Jumlah frame yang dibaca untuk memastikan kamera benar-benar memberi gambar.
WARMUP_FRAMES = 12
MIN_MEAN_BRIGHTNESS = 12.0


def _frame_is_usable(frame: np.ndarray | None) -> bool:
    """True kalau frame berisi sesuatu, bukan hitam total."""
    if frame is None or frame.size == 0:
        return False
    return float(np.mean(frame)) > MIN_MEAN_BRIGHTNESS


def _try_open(index: int, width: int, height: int) -> tuple[cv2.VideoCapture | None, tuple[int, int]]:
    """Buka kamera pada satu resolusi dan pastikan gambarnya tidak hitam."""
    backend = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        capture.release()
        return None, (0, 0)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Kamera butuh beberapa frame untuk menyesuaikan exposure. Yang dinilai
    # adalah frame terakhir, bukan frame pertama.
    last = None
    for _ in range(WARMUP_FRAMES):
        ok, frame = capture.read()
        if ok and frame is not None:
            last = frame
    if not _frame_is_usable(last):
        capture.release()
        return None, (0, 0)

    assert last is not None
    return capture, (last.shape[1], last.shape[0])


def open_camera(settings: Settings) -> cv2.VideoCapture:
    """Buka kamera dan pastikan ia benar-benar memberi gambar.

    Resolusi yang diminta dicoba lebih dulu. Kalau hasilnya hitam, resolusi
    lain dicoba. Ini penting karena kamera bawaan sebagian Mac mengembalikan
    frame hitam pada 1080p tetapi normal pada 720p, dan tanpa pemeriksaan ini
    aplikasi akan tampak "tidak jalan" padahal kameranya sehat.
    """
    wanted = (settings.width, settings.height)
    attempts: list[tuple[int, int]] = [wanted]
    for resolution in FALLBACK_RESOLUTIONS:
        if resolution not in attempts:
            attempts.append(resolution)

    for width, height in attempts:
        capture, actual = _try_open(settings.camera_index, width, height)
        if capture is not None:
            if (width, height) != wanted:
                print(
                    f"Kamera {settings.camera_index} hanya memberi gambar pada "
                    f"{actual[0]}x{actual[1]}, bukan {wanted[0]}x{wanted[1]}. "
                    "Resolusi itu yang dipakai.",
                    flush=True,
                )
            return capture

    raise RuntimeError(
        f"Kamera {settings.camera_index} ({cameras.describe(settings.camera_index)}) "
        "tidak bisa dibuka atau hanya memberi gambar hitam. Kemungkinan "
        "penyebab: kamera sedang dipakai aplikasi lain (QuickTime, Photo Booth, "
        "panggilan video), lensa tertutup, atau izin kamera belum diberikan. "
        "Di dalam aplikasi, tekan C untuk mencoba kamera berikutnya."
    )


LOCK_PATH = Path("/tmp/hand-portal.lock")


def _acquire_single_instance() -> int | None:
    """Cegah dua jendela Hand Portal terbuka bersamaan.

    Ini bukan sekadar soal rapi. Kamera bawaan macOS hanya bisa dipakai satu
    proses. Kalau instance kedua tetap dibuka, permintaan ke kamera laptop
    gagal karena sedang dipakai, dan macOS diam-diam memberi kamera iPhone
    (Continuity Camera) sebagai gantinya. Itu yang membuat aplikasi tiba-tiba
    memakai kamera HP.

    Mengembalikan file descriptor pengunci, atau None kalau sudah ada instance
    lain yang berjalan.
    """
    handle = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (ImportError, OSError):
        # Di sistem tanpa fcntl, penguncian dilewati. Lebih baik tetap jalan
        # daripada menolak membuka aplikasi.
        os.close(handle)
        return None if _lock_owner_alive() else -1
    os.ftruncate(handle, 0)
    os.write(handle, str(os.getpid()).encode())
    return handle


def _lock_owner_alive() -> bool:
    """True kalau pemilik berkas kunci masih hidup."""
    try:
        pid = int(LOCK_PATH.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _usable_person_mask(mask: np.ndarray | None) -> np.ndarray | None:
    """Tolak mask yang sebenarnya tidak berguna.

    Selfie segmenter menganggap frame gelap total sebagai orang, jadi saat
    kamera tertutup atau ruangan mati lampu hasilnya mask penuh. Kalau mask
    seperti itu diteruskan, filter galaxy akan menggambar latar bintang di
    seluruh layar dan efeknya hilang. Lebih baik kembali ke efek warna biasa.
    """
    if mask is None:
        return None
    coverage = float(np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0
    if coverage > MAX_PERSON_COVERAGE:
        return None
    return mask


class App:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.theme = THEME
        self.filters = build_filters()
        self.tracker = HandTracker(settings)
        self.portal_tracker = PortalTracker(settings)
        self.switcher = FilterSwitcher(len(self.filters), settings)
        self.slingshot = SlingshotTracker(settings)
        self.projectiles = Projectiles(settings)
        self.renderer = Renderer(settings, self.theme)
        # HUD butuh seluruh nama filter untuk korsel di bawah layar, bukan
        # hanya yang sedang aktif, jadi daftarnya dikirim dari sini.
        self.hud = Hud(settings, self.theme, filter_labels=[spec.label for spec in self.filters])
        self.recorder = Recorder(settings.record_path, settings.width, settings.height, settings.fps_target)
        self.capture: cv2.VideoCapture | None = None
        self.grabber: _FrameGrabber | None = None
        self.mirror = settings.mirror
        self.recording = settings.record_path is not None
        self.show_hud = settings.hud
        self.show_particles = settings.particles
        self.show_landmarks = settings.show_landmarks
        self._fps = 0.0
        self._last_tick = time.perf_counter()
        self._frame_ms = 0.0
        # Ganti filter otomatis: menyala kalau auto_cycle_s lebih dari nol.
        self.auto_cycle = settings.auto_cycle_s > 0
        self._auto_next_s = 0.0
        self._last_frame_s = 0.0
        # Pesan yang ditampilkan sekali di frame pertama.
        self._pending_toast: str | None = None

    # -- siklus hidup ---------------------------------------------------

    def run(self) -> int:
        lock = _acquire_single_instance()
        if lock is None:
            raise RuntimeError(
                "Hand Portal sudah berjalan di jendela lain. "
                "Tutup jendela yang lama dulu, karena kamera laptop hanya bisa "
                "dipakai satu aplikasi. Kalau tidak ada jendela yang terlihat, "
                f"hapus {LOCK_PATH} lalu coba lagi."
            )
        try:
            return self._run_loop()
        finally:
            if lock not in (None, -1):
                os.close(lock)

    def _run_loop(self) -> int:
        self.capture = open_camera(self.settings)
        self.grabber = _FrameGrabber(self.capture)
        name = cameras.describe(self.settings.camera_index)
        print(
            f"Kamera {self.settings.camera_index}: {name}. "
            "Tekan C untuk ganti kamera, Q untuk keluar.",
            flush=True,
        )
        # Nama kamera ditampilkan sekilas di layar, jadi kalau ternyata yang
        # terpakai bukan kamera yang dimaksud, itu langsung kelihatan tanpa
        # harus membuka log.
        self._pending_toast = f"Kamera {self.settings.camera_index}: {name}"
        try:
            while True:
                if not self._step():
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
        return 0

    def close(self) -> None:
        if self.grabber is not None:
            self.grabber.stop()
        if self.capture is not None:
            self.capture.release()
        self.recorder.close()
        self.tracker.close()
        cv2.destroyAllWindows()

    # -- satu frame -----------------------------------------------------

    def _step(self) -> bool:
        started = time.perf_counter()
        frame = None if self.grabber is None else self.grabber.latest()
        if frame is None:
            time.sleep(0.005)
            return True

        if self.mirror:
            frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]
        timestamp_ms = int(time.time() * 1000)
        now_s = timestamp_ms / 1000.0

        hands = self.tracker.detect(frame, timestamp_ms)
        portal = self.portal_tracker.update(hands, width, height)

        event = self.switcher.update(hands, width, height, now_s)
        if event is not None:
            self._announce(event, now_s)
        else:
            self._auto_cycle(now_s)

        # Ketapel: cubit, tarik, lepas.
        shot = self.slingshot.update(hands, width, height, now_s)
        if shot is not None:
            self.projectiles.fire(shot, width, height)
        dt = 0.0 if self._last_frame_s == 0.0 else min(0.1, now_s - self._last_frame_s)
        self._last_frame_s = now_s
        self.projectiles.update(dt, width, height)

        spec = self.filters[self.switcher.index]
        if portal is not None:
            filtered = self._render_filter(frame, portal, spec, now_s)
            if filtered is not None:
                self.renderer.composite(frame, portal, filtered)
            self.renderer.draw_portal(frame, portal, now_s)

        if self.show_landmarks:
            self.renderer.draw_skeleton(frame, hands)

        # Ketapel digambar setelah portal supaya bola dan pecahannya berada di
        # atas semua efek, lalu bloom membuat semuanya ikut bercahaya.
        draw_slingshot(
            frame,
            self.slingshot.state(width, height),
            self.projectiles,
            self.theme,
            now_s,
            bands=self.settings.slingshot_bands,
        )

        if self.settings.bloom:
            frame = self.renderer.bloom(frame)

        self._update_fps(started)
        stats = FrameStats(
            fps=self._fps,
            hands=hands.count,
            portal_kind=None if portal is None else portal.kind,
            filter_key=spec.key,
            filter_label=spec.label,
            filter_index=self.switcher.index,
            filter_total=len(self.filters),
            tracking=portal is not None,
        )
        if self.show_hud:
            self.hud.draw(frame, stats, now_s)
        if self._pending_toast is not None:
            self.hud.toast(self._pending_toast, now_s)
            self._pending_toast = None

        if self.recording:
            self.recorder.write(frame)

        cv2.imshow(WINDOW_TITLE, frame)
        return self._handle_keys(cv2.waitKey(1) & 0xFF, frame, now_s)

    def _render_filter(self, frame: np.ndarray, portal: Portal, spec, now_s: float) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x, y, w, h = portal.bounds(width, height)
        if w < 4 or h < 4:
            return None

        roi = frame[y:y + h, x:x + w]
        person_mask = None
        if spec.key in MASK_FILTERS and self.tracker.has_segmenter:
            person_mask = self.tracker.person_mask(frame, int(now_s * 1000))
            person_mask = _usable_person_mask(person_mask)

        ctx = FilterContext(
            frame=frame,
            person_mask=person_mask,
            time_s=now_s,
            portal=portal,
            width=width,
            height=height,
        )
        filtered = apply_filter(roi.copy(), spec, ctx)
        if filtered.shape[:2] != (h, w):
            filtered = cv2.resize(filtered, (w, h), interpolation=cv2.INTER_LINEAR)
        return filtered

    def _update_fps(self, started: float) -> None:
        elapsed = max(time.perf_counter() - started, 1e-6)
        self._frame_ms = elapsed * 1000.0
        instant = 1.0 / elapsed
        # Rata-rata bergerak supaya angka di HUD tidak melompat-lompat.
        self._fps = instant if self._fps == 0.0 else self._fps * 0.9 + instant * 0.1

    def _announce(self, event: SwitchEvent, now_s: float) -> None:
        label = self.filters[event.index].label
        reason = SWITCH_REASON_TEXT.get(event.reason, SWITCH_REASON_TEXT["key"])
        self.hud.toast(f"{reason}: {label}", now_s)
        # Setiap pergantian filter mengulang hitungan, jadi filter yang baru
        # dipilih tidak langsung ditinggal sepersekian detik kemudian.
        self._auto_next_s = now_s + self.settings.auto_cycle_s

    def _auto_cycle(self, now_s: float) -> None:
        """Ganti filter sendiri setiap beberapa detik.

        Hitungannya baru dimulai pada frame pertama, bukan sejak aplikasi
        dibuka, supaya filter pertama tetap sempat dilihat.
        """
        if not self.auto_cycle or self.settings.auto_cycle_s <= 0:
            return
        if self._auto_next_s == 0.0:
            self._auto_next_s = now_s + self.settings.auto_cycle_s
            return
        if now_s < self._auto_next_s:
            return
        self._announce(self.switcher.next("auto"), now_s)

    # -- papan tombol ---------------------------------------------------

    def _handle_keys(self, key: int, frame: np.ndarray, now_s: float) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == ord("["):
            self._announce(self.switcher.prev(), now_s)
        elif key == ord("]"):
            self._announce(self.switcher.next(), now_s)
        elif ord("1") <= key <= ord("9"):
            index = key - ord("1")
            if index < len(self.filters):
                self._announce(self.switcher.select(index), now_s)
        elif key == ord(" "):
            path = self.recorder.snapshot(frame)
            self.hud.toast(f"Foto disimpan: {path}", now_s)
        elif key == ord("r"):
            self._toggle_recording(now_s)
        elif key == ord("h"):
            self.show_hud = not self.show_hud
        elif key == ord("p"):
            self.show_particles = not self.show_particles
        elif key == ord("l"):
            self.show_landmarks = not self.show_landmarks
        elif key == ord("m"):
            self.mirror = not self.mirror
            self.portal_tracker.reset()
            self.slingshot.reset()
        elif key == ord("a"):
            self.auto_cycle = not self.auto_cycle
            self._auto_next_s = now_s + self.settings.auto_cycle_s
            self.hud.toast(
                "Ganti otomatis menyala" if self.auto_cycle else "Ganti otomatis mati",
                now_s,
            )
        elif key == ord("f"):
            self._toggle_fullscreen()
        elif key == ord("c"):
            self._switch_camera(now_s)
        elif key == ord("?"):
            for line in HELP_LINES:
                print(line)
        return True

    def _toggle_recording(self, now_s: float) -> None:
        if self.recording:
            self.recorder.close()
            self.recording = False
            self.hud.toast("Rekaman dihentikan", now_s)
            return
        path = time.strftime("handportal-%Y%m%d-%H%M%S.mp4")
        self.recorder = Recorder(path, self.settings.width, self.settings.height, self.settings.fps_target)
        self.recording = True
        self.hud.toast(f"Merekam ke {path}", now_s)

    def _switch_camera(self, now_s: float) -> None:
        """Pindah ke kamera berikutnya tanpa menutup aplikasi.

        Ini cara paling andal memilih kamera di macOS: pemakai melihat
        langsung gambarnya, jadi tidak perlu menebak dari nama atau urutan
        indeks yang bisa berubah. Pilihan terakhir disimpan untuk sesi
        berikutnya.
        """
        start = self.settings.camera_index
        for step in range(1, MAX_CAMERA_PROBE + 1):
            candidate = (start + step) % MAX_CAMERA_PROBE
            capture, actual = _try_open(candidate, self.settings.width, self.settings.height)
            if capture is None:
                # Resolusi yang diminta gagal, coba yang lebih kecil sebelum
                # menyimpulkan kamera ini tidak bisa dipakai.
                for width, height in FALLBACK_RESOLUTIONS:
                    capture, actual = _try_open(candidate, width, height)
                    if capture is not None:
                        break
            if capture is None:
                continue

            # Kamera lama dilepas dulu supaya kamera yang baru benar-benar
            # bisa dipakai, karena satu kamera hanya bisa dibuka satu proses.
            if self.grabber is not None:
                self.grabber.stop()
            if self.capture is not None:
                self.capture.release()

            self.capture = capture
            self.grabber = _FrameGrabber(capture)
            self.settings = self.settings.with_overrides(camera_index=candidate)
            self.portal_tracker.reset()
            self.slingshot.reset()
            self.projectiles.clear()
            cameras.save_preference(candidate)
            label = cameras.describe(candidate)
            self.hud.toast(f"Kamera {candidate}: {label} ({actual[0]}x{actual[1]})", now_s)
            print(
                f"Pindah ke kamera {candidate}: {label} pada {actual[0]}x{actual[1]}",
                flush=True,
            )
            return

        self.hud.toast("Tidak ada kamera lain yang bisa dibuka", now_s)

    def _toggle_fullscreen(self) -> None:
        try:
            current = cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN)
            target = cv2.WINDOW_NORMAL if current == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, target)
        except cv2.error:
            # Jendela belum ada di frame pertama; tidak perlu diapa-apakan.
            pass
