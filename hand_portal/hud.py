"""HUD: panel info, korsel filter, dan toast di atas video.

Semua ukuran diturunkan dari tinggi frame. Alasannya: aplikasi ini dipakai
mulai dari webcam 480p sampai layar penuh 1080p, dan angka piksel tetap akan
terlihat kekecilan di satu sisi lalu kebesaran di sisi lain. Kalau frame
terlalu kecil untuk sebuah bagian, bagian itu dilewati, bukan dipaksa masuk
sampai teksnya keluar layar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .config import Settings, Theme
from .contracts import FrameStats

# Tinggi acuan untuk semua ukuran huruf dan jarak. Frame lain diskalakan
# relatif ke sini supaya proporsi HUD terasa sama di semua resolusi.
REFERENCE_HEIGHT = 720.0
# Skala tidak dibiarkan turun terlalu jauh: di bawah 0.5 huruf Hershey sudah
# tidak terbaca, lebih baik bagiannya dilewati daripada jadi bubur.
MIN_SCALE = 0.5
MAX_SCALE = 1.8

# Di bawah ukuran ini HUD tidak ada gunanya dan hanya menutupi video.
MIN_FRAME_WIDTH = 160
MIN_FRAME_HEIGHT = 120

TITLE_FONT = cv2.FONT_HERSHEY_DUPLEX
BODY_FONT = cv2.FONT_HERSHEY_SIMPLEX

TITLE_SCALE = 0.62
BODY_SCALE = 0.46
SMALL_SCALE = 0.40
TOAST_SCALE = 0.52

MARGIN_RATIO = 0.030
PAD_RATIO = 0.016
GAP_RATIO = 0.010
CORNER_RATIO = 0.012

# Panel dibuat tembus pandang: kalau terlalu pekat, video di belakangnya
# hilang dan HUD terasa seperti menu, bukan lapisan di atas kamera.
PANEL_ALPHA = 0.55
TOAST_ALPHA = 0.62

TOAST_LIFETIME_S = 1.6
TOAST_FADE_S = 0.5

HINT_TEXT = "Cubit lalu tarik untuk ketapel - Tahan jempol ke jari tengah untuk ganti efek - [ ] pilih - C kamera - Q keluar"

Segment = tuple[str, tuple[int, int, int]]


@dataclass(frozen=True)
class _Metrics:
    """Ukuran turunan frame yang dipakai bersama semua bagian HUD."""

    width: int
    height: int
    scale: float
    margin: int
    pad: int
    gap: int
    radius: int
    thickness: int


def _text_metrics(
    text: str, font: int, scale: float, thickness: int
) -> tuple[int, int, int]:
    """Kembalikan (lebar, tinggi huruf, baseline) sebuah teks."""
    if not text:
        return (0, 0, 0)
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    return (int(width), int(height), int(baseline))


def _text_width(text: str, font: int, scale: float, thickness: int) -> int:
    return _text_metrics(text, font, scale, thickness)[0]


def _shorten_to_width(
    text: str, max_width: int, font: int, scale: float, thickness: int
) -> str:
    """Potong teks dengan elipsis kalau tidak muat.

    Dipakai hanya sebagai jaring pengaman terakhir, misalnya saat nama filter
    sangat panjang di frame sempit.
    """
    if not text or max_width <= 0:
        return ""
    if _text_width(text, font, scale, thickness) <= max_width:
        return text
    ellipsis = "..."
    trimmed = text
    while trimmed and _text_width(trimmed + ellipsis, font, scale, thickness) > max_width:
        trimmed = trimmed[:-1]
    if not trimmed:
        return ""
    return trimmed.rstrip() + ellipsis


def _rounded_mask(height: int, width: int, radius: int) -> np.ndarray:
    """Mask kotak sudut membulat, dipakai supaya panel tidak terlihat kaku."""
    mask = np.zeros((height, width), np.uint8)
    if height <= 0 or width <= 0:
        return mask
    r = max(0, min(int(radius), width // 2, height // 2))
    if r == 0:
        mask[:] = 255
        return mask
    cv2.rectangle(mask, (r, 0), (width - r, height), 255, -1)
    cv2.rectangle(mask, (0, r), (width, height - r), 255, -1)
    for cx, cy in ((r, r), (width - r - 1, r), (r, height - r - 1), (width - r - 1, height - r - 1)):
        cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def _soft_panel(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    alpha: float,
    radius: int,
) -> None:
    """Gambar kotak setengah transparan, hanya area kotak yang di-blend.

    Blend dilakukan per potongan (bukan seluruh frame) supaya ongkosnya tetap
    kecil walau dipanggil beberapa kali tiap frame.
    """
    height, width = frame.shape[:2]
    x0, x1 = max(0, int(x0)), min(width, int(x1))
    y0, y1 = max(0, int(y0)), min(height, int(y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    alpha = float(min(max(alpha, 0.0), 1.0))
    if alpha <= 0.0:
        return

    roi = frame[y0:y1, x0:x1]
    overlay = np.empty_like(roi)
    overlay[:] = color
    blended = cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0)
    mask = _rounded_mask(y1 - y0, x1 - x0, radius)
    frame[y0:y1, x0:x1] = np.where(mask[:, :, None] > 0, blended, roi)


def _put_text_soft(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    font: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
    alpha: float,
) -> None:
    """Tulis teks dengan transparansi, untuk toast yang sedang memudar."""
    if not text or alpha <= 0.0:
        return
    alpha = float(min(alpha, 1.0))
    height, width = frame.shape[:2]
    text_w, text_h, baseline = _text_metrics(text, font, scale, thickness)
    x0 = max(0, org[0] - 2)
    y0 = max(0, org[1] - text_h - 2)
    x1 = min(width, org[0] + text_w + 2)
    y1 = min(height, org[1] + baseline + 2)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return

    roi = frame[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.putText(
        overlay,
        text,
        (org[0] - x0, org[1] - y0),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    if alpha >= 1.0:
        frame[y0:y1, x0:x1] = overlay
        return
    frame[y0:y1, x0:x1] = cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0)


def _put_text_shadowed(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    font: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
    shadow: tuple[int, int, int],
) -> None:
    """Teks dengan bayangan satu piksel supaya tetap terbaca di video terang."""
    if not text:
        return
    cv2.putText(frame, text, (org[0] + 1, org[1] + 1), font, scale, shadow, thickness, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


def _load_filter_labels() -> list[str]:
    """Ambil daftar nama filter dari modul filters kalau tersedia.

    FrameStats hanya membawa label filter yang aktif, sedangkan korsel butuh
    seluruh daftar. Impor sengaja ditunda dan dibungkus try: HUD harus tetap
    bisa dipakai (dan diuji) walau modul filters belum ada.
    """
    try:
        from .filters import build_filters

        return [spec.label for spec in build_filters()]
    except Exception:
        return []


class Hud:
    """Lapisan teks di atas frame: panel info, korsel filter, petunjuk, toast."""

    def __init__(
        self,
        settings: Settings,
        theme: Theme,
        filter_labels: Sequence[str] | None = None,
    ) -> None:
        """Siapkan HUD.

        `filter_labels` opsional; kalau tidak diisi, HUD mencoba membaca daftar
        nama filter dari modul filters saat pertama kali menggambar korsel.
        """
        self._settings = settings
        self._theme = theme
        self._labels: list[str] = [str(label) for label in (filter_labels or [])]
        self._labels_loaded = bool(self._labels)
        self._toast_message = ""
        self._toast_expire_s = 0.0

    # ------------------------------------------------------------------ publik

    def draw(self, frame: np.ndarray, stats: FrameStats, now_s: float) -> None:
        """Gambar seluruh HUD ke frame. Tidak melakukan apa pun kalau dimatikan."""
        if not self._settings.hud:
            return
        if frame is None or frame.size == 0 or frame.ndim != 3:
            return
        height, width = frame.shape[:2]
        if width < MIN_FRAME_WIDTH or height < MIN_FRAME_HEIGHT:
            return

        metrics = self._metrics(width, height)
        panel_bottom = self._draw_panel(frame, stats, metrics)
        hint_height = self._hint_height(metrics)
        # Korsel nama filter dan batang kemajuan gestur sengaja tidak
        # digambar. Keduanya menutupi bagian bawah layar, dan nama filter
        # sudah terbaca di panel kiri atas.
        del panel_bottom
        self._draw_hint(frame, metrics)
        self._draw_toast(frame, now_s, metrics)

    def toast(self, message: str, now_s: float) -> None:
        """Tampilkan pesan singkat di tengah atas; memudar sendiri setelah 1,6 detik."""
        self._toast_message = str(message or "")
        self._toast_expire_s = float(now_s) + TOAST_LIFETIME_S

    # ------------------------------------------------------------------ panel

    def _metrics(self, width: int, height: int) -> _Metrics:
        scale = min(max(height / REFERENCE_HEIGHT, MIN_SCALE), MAX_SCALE)
        return _Metrics(
            width=width,
            height=height,
            scale=scale,
            margin=max(4, int(round(height * MARGIN_RATIO))),
            pad=max(4, int(round(height * PAD_RATIO))),
            gap=max(2, int(round(height * GAP_RATIO))),
            radius=max(3, int(round(height * CORNER_RATIO))),
            thickness=1 if scale < 1.15 else 2,
        )

    def _hand_status(self, stats: FrameStats) -> tuple[str, tuple[int, int, int]]:
        theme = self._theme
        if stats.hands <= 0:
            return "cari tangan...", theme.warning
        if stats.hands == 1:
            text = "1 tangan terdeteksi"
        else:
            text = f"{stats.hands} tangan terdeteksi"
        if stats.portal_kind:
            text = f"{text} - {stats.portal_kind}"
        return text, theme.success

    def _draw_panel(self, frame: np.ndarray, stats: FrameStats, m: _Metrics) -> int:
        """Gambar panel kiri atas, kembalikan koordinat y bawahnya."""
        theme = self._theme
        body_scale = BODY_SCALE * m.scale
        title_scale = TITLE_SCALE * m.scale

        hand_text, hand_color = self._hand_status(stats)
        if stats.filter_total > 0:
            counter = f"Filter {stats.filter_index + 1}/{stats.filter_total} - "
        else:
            counter = "Filter - "
        label = stats.filter_label or "-"

        lines: list[tuple[list[Segment], int, float, int]] = [
            ([("HAND PORTAL", theme.accent)], TITLE_FONT, title_scale, m.thickness),
            ([(f"FPS {stats.fps:.1f}", theme.text)], BODY_FONT, body_scale, 1),
            ([(hand_text, hand_color)], BODY_FONT, body_scale, 1),
            ([(counter, theme.text_dim), (label, theme.accent)], BODY_FONT, body_scale, 1),
        ]

        widths: list[int] = []
        heights: list[int] = []
        for segments, font, scale, thickness in lines:
            widths.append(sum(_text_width(text, font, scale, thickness) for text, _ in segments))
            heights.append(max(_text_metrics(text, font, scale, thickness)[1] for text, _ in segments))

        inner_width = max(widths) if widths else 0
        panel_width = min(inner_width + 2 * m.pad, m.width - 2 * m.margin)
        content_height = sum(heights) + m.gap * (len(lines) - 1)
        panel_height = content_height + 2 * m.pad
        if panel_width < 2 * m.pad or panel_height < 2 * m.pad:
            return m.margin

        x0 = m.margin
        y0 = m.margin
        x1 = x0 + panel_width
        y1 = y0 + panel_height
        if x1 > m.width - m.margin or y1 > m.height - m.margin:
            return y0

        _soft_panel(frame, x0, y0, x1, y1, theme.panel, PANEL_ALPHA, m.radius)

        text_x = x0 + m.pad
        text_y = y0 + m.pad
        max_text_width = panel_width - 2 * m.pad
        for (segments, font, scale, thickness), line_width, line_height in zip(lines, widths, heights):
            # Baris yang tetap tidak muat dilewati utuh; memotongnya di tengah
            # hanya membuat panel terlihat rusak.
            if line_width <= max_text_width:
                cursor = text_x
                for text, color in segments:
                    cv2.putText(
                        frame,
                        text,
                        (cursor, text_y + line_height),
                        font,
                        scale,
                        color,
                        thickness,
                        cv2.LINE_AA,
                    )
                    cursor += _text_width(text, font, scale, thickness)
            text_y += line_height + m.gap

        return y1

    # --------------------------------------------------------------- korsel


    def _hint_height(self, m: _Metrics) -> int:
        """Tinggi teks petunjuk, atau 0 kalau tidak muat di frame ini."""
        scale = SMALL_SCALE * m.scale
        width, height, _ = _text_metrics(HINT_TEXT, BODY_FONT, scale, 1)
        if width > m.width - 2 * m.margin:
            return 0
        return height

    def _draw_hint(self, frame: np.ndarray, m: _Metrics) -> None:
        """Petunjuk gestur redup di kanan bawah."""
        if self._hint_height(m) <= 0:
            return
        scale = SMALL_SCALE * m.scale
        width, _, _ = _text_metrics(HINT_TEXT, BODY_FONT, scale, 1)
        org = (m.width - m.margin - width, m.height - m.margin)
        _put_text_shadowed(
            frame,
            HINT_TEXT,
            org,
            BODY_FONT,
            scale,
            self._theme.text_dim,
            1,
            self._theme.panel,
        )

    # ------------------------------------------------------------------ toast

    def _draw_toast(self, frame: np.ndarray, now_s: float, m: _Metrics) -> None:
        """Pesan singkat di tengah atas yang memudar di sisa waktunya."""
        if not self._toast_message:
            return
        remaining = self._toast_expire_s - now_s
        if remaining <= 0:
            return

        # Toast tampil penuh dulu, baru memudar di detik terakhir supaya
        # pesan sempat terbaca.
        fade = min(1.0, remaining / TOAST_FADE_S)
        theme = self._theme
        scale = TOAST_SCALE * m.scale
        thickness = m.thickness
        max_text_width = m.width - 2 * m.margin - 2 * m.pad
        text = _shorten_to_width(self._toast_message, max_text_width, TITLE_FONT, scale, thickness)
        if not text:
            return

        text_width, text_height, _ = _text_metrics(text, TITLE_FONT, scale, thickness)
        box_width = text_width + 2 * m.pad
        box_height = text_height + 2 * m.pad
        x0 = (m.width - box_width) // 2
        y0 = m.margin
        x1 = x0 + box_width
        y1 = y0 + box_height
        if x1 > m.width or y1 > m.height:
            return

        _soft_panel(frame, x0, y0, x1, y1, theme.panel, TOAST_ALPHA * fade, m.radius)
        _put_text_soft(
            frame,
            text,
            (x0 + m.pad, y0 + m.pad + text_height),
            TITLE_FONT,
            scale,
            theme.text,
            thickness,
            fade,
        )
