"""
ebsviewer.py — the classic live viewer: a second, deliberately-different player tab.

This is the EBS Tools v2 ``Player`` (from ``ebs_tools/qtgui.py``) vendored verbatim — every
view mode and the Band/Single/Stack column-mode expression render exactly as they were tuned
here. The eleven live view modes are:

    Events · Panorama · Radar · AT Waterfall · Polarity Div · Phase-Locked Pano ·
    Space-Time Vol · IAT Surface · Dual-Cam Diff · Elev-Time Sweep · Freq Map

Only two things changed in the port: (1) the data layer is rewired to GottLUX's
EBS-compatible backends — ``gottlux.rotation.io_evt21.load`` (the same memmapped event dict)
and ``gottlux.io.telemetry.Telemetry`` (API-identical); (2) the EBS Dashboard/Run window and
its inline video export were dropped in favour of GottLUX's own exporters. The
:class:`EBSViewer` subclass at the bottom adds the GottLUX panel protocol (set_recording /
sync / capture_frame / sensor_size / capture_clock) so the recording the main window loads
flows in and GottLUX's Capture (video + poster) / Export (figures · cubes · tables) / frame-PNG
work on these views. The viewer keeps its own playhead/transport — a standalone tab.

The ``FFTDialog`` (a detachable spectrum analyser the Player launches) is kept; the
``Player``/``EventView``/``RangeSlider``/``DensityStrip``/``AnnotationBar`` classes are the
original EBS implementations.
"""
from __future__ import annotations
import os, sys, glob, json, time
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib import colormaps as _mpl_colormaps

from gottlux.app import icons
from gottlux.app import style

# Ported into GottLUX: only the data layer is rewired to GottLUX's compatible backends.
from gottlux.rotation import io_evt21           # returns the same memmapped ev-dict the Player expects
from gottlux.io.telemetry import Telemetry, estimate_spin_period_s

VIEW_MODES = ["Events", "Panorama", "Radar",
              "AT Waterfall", "Polarity Div", "Phase-Locked Pano",
              "Space-Time Vol", "IAT Surface", "Dual-Cam Diff", "Elev-Time Sweep",
              "Freq Map"]
# Views that de-rotate events against the sensor's spin. They need an azimuth(t) track; when the
# clip ships no telemetry FILE one is synthesized from the event-rate periodicity (see
# _ensure_rotation_telemetry) so the views still render — bearings are phase-relative (absolute
# North uncalibrated). Only a truly empty/aperiodic clip falls back to the explicit placeholder.
ROTATION_ONLY_MODES = frozenset({"Panorama", "Radar", "AT Waterfall",
                                 "Phase-Locked Pano", "Dual-Cam Diff"})
PLAYBACK_WALL_FPS = 60          # display refresh cap; request more fps than this -> smooth slow-motion


def _fov_for(path):
    n = os.path.basename(path).lower()
    if n.startswith("cam0") or "76_fov" in n: return 76.0
    if n.startswith("cam1") or "50_fov" in n: return 50.0
    return 50.0

ANALYSES = ["detect", "pointcloud", "rate_surface", "validation", "radar", "panorama", "mti", "masksweep"]
ANALYSIS_TOOLTIPS = {
    "detect": "Isolate the target and summarize bearing / elevation / range tracks; export .mat + .csv.",
    "pointcloud": "Clean isolated-target 3D spatio-temporal point cloud.",
    "rate_surface": "Elevated point-cloud video: X/Y = sensor, Z = local event rate (events/sec).",
    "validation": "3-panel proof video: raw | isolated target + bbox + callout | bearing radar.",
    "radar": "360° polar target map (bearing / range / altitude) + animated tactical sweep.",
    "panorama": "360° de-rotated environmental map (vertical axis = sensor Y px).",
    "mti": "Moving-target indication: de-rotated azimuth/elevation vs time (static = horizontal bands).",
    "masksweep": "Compare successive masks N=0..levels and quantify the event-rate drop-off.",
}
PLAYER_THEMES = {"Inferno": "inferno", "Viridis": "viridis", "Tactical Green": "_green",
                 "Ice (cyan)": "_ice", "Grayscale": "gray"}

# What each tracker is actually good for (shown under the Tracking selector).
TRACKER_USES = {
    "single": "A single dominant target on a quiet background (one drone, or a distant airliner). A robust "
              "median/MAD gate around bearing & elevation rejects outliers — the simplest, most stable single-object choice.",
    "nearest": "A lightweight baseline: greedily links the strongest detection per frame into one track (gated on "
               "bearing/elevation + max time gap). Good for a few well-separated targets in low clutter, or as a sanity check.",
    "kalman": "Smooth, predictive MULTI-target tracking. Constant-velocity predict/gate/smooth with COASTING bridges "
              "brief dropouts/occlusions — use it for several steadily-moving targets where you want clean, continuous trajectories.",
    "hummingbird": "Flapping-wing flyers — birds, insects, hummingbirds (~10-120 Hz wingbeat). Confirms each candidate by its "
                   "wingbeat frequency on the event time-surface, rejecting rigid/static clutter, and links a swarm via nearest-neighbour.",
    "drone_fft": "Multirotor / UAS detection (~80-800 Hz rotor signature). Verifies targets by their rotor harmonics — strongly "
                 "rejecting birds and clutter that lack a rotor tone — and coasts on velocity. Best bet for spotting drones, including faint/distant ones.",
    "cmax": "Contrast-Maximization: solves for the image-plane VELOCITY that sharpens the motion-compensated event image. "
            "Use it to measure how fast and which way a target moves (and to deblur fast motion), not just where it is.",
}

# ── Theming ──────────────────────────────────────────────────────────────────
# Each theme is a flat color palette; build_qss() turns a palette + a UI-scale
# factor into a stylesheet, so every theme automatically supports scaling.
#   "Blade Runner" — the signature neon dark look (premium / cinematic)
#   "Graphite"     — a calmer, simplified neutral-dark palette (low chroma)
#   "Nord"         — a soft, simplified slate/blue palette (easy on the eyes)
#   "Light"        — a clean, simplified day mode
THEMES = {
    "Blade Runner": {
        "is_dark": True,
        "bg": "#0a0e14", "fg": "#cfe9f2", "panel": "#0e151b", "border": "#16323a",
        "accent": "#00e5ff", "accent_fg": "#e0ffff", "title": "#00e5ff",
        "btn_bg": "#0f1b22", "btn_border": "#1f6f7d", "btn_fg": "#9bf3ff",
        "run_bg": "#13262b", "run_border": "#00e5ff", "run_fg": "#00e5ff", "run_hover": "#0a3a42",
        "sel_bg": "#13343c", "groove": "#16323a", "handle": "#ff2bd6", "progress": "#00e5ff",
    },
    "Graphite": {
        "is_dark": True,
        "bg": "#1e1f22", "fg": "#dfe3e8", "panel": "#26282c", "border": "#3a3d42",
        "accent": "#7fb4e6", "accent_fg": "#ffffff", "title": "#a9c4dd",
        "btn_bg": "#2b2e33", "btn_border": "#454a51", "btn_fg": "#dfe3e8",
        "run_bg": "#3b6ea5", "run_border": "#3b6ea5", "run_fg": "#ffffff", "run_hover": "#4f82bb",
        "sel_bg": "#3b6ea5", "groove": "#3a3d42", "handle": "#7fb4e6", "progress": "#7fb4e6",
    },
    "Nord": {
        "is_dark": True,
        "bg": "#2e3440", "fg": "#e5e9f0", "panel": "#3b4252", "border": "#434c5e",
        "accent": "#88c0d0", "accent_fg": "#eceff4", "title": "#88c0d0",
        "btn_bg": "#3b4252", "btn_border": "#4c566a", "btn_fg": "#e5e9f0",
        "run_bg": "#5e81ac", "run_border": "#5e81ac", "run_fg": "#eceff4", "run_hover": "#81a1c1",
        "sel_bg": "#4c566a", "groove": "#434c5e", "handle": "#88c0d0", "progress": "#a3be8c",
    },
    "Light": {
        "is_dark": False,
        "bg": "#f4f6f8", "fg": "#1b242b", "panel": "#ffffff", "border": "#c8d2da",
        "accent": "#0a6ebd", "accent_fg": "#0a6ebd", "title": "#0a6ebd",
        "btn_bg": "#ffffff", "btn_border": "#9fb3c0", "btn_fg": "#1b242b",
        "run_bg": "#0a6ebd", "run_border": "#0a6ebd", "run_fg": "#ffffff", "run_hover": "#0b7fd6",
        "sel_bg": "#d6e7f5", "groove": "#c8d2da", "handle": "#0a6ebd", "progress": "#0a6ebd",
    },
}
DEFAULT_THEME = "Blade Runner"


def build_qss(theme_name=DEFAULT_THEME, scale=1.0):
    """Render a full stylesheet for `theme_name`, with every font/padding/spacing
    metric multiplied by `scale` so the whole UI can be sized up or down."""
    p = THEMES.get(theme_name, THEMES[DEFAULT_THEME])
    def s(px):           # scale a px metric (never below 1)
        return max(1, int(round(px * scale)))
    return f"""
QWidget {{ background:{p['bg']}; color:{p['fg']}; font-family:'Segoe UI','Noto Sans','DejaVu Sans',sans-serif; font-size:{s(12)}px; }}
QToolTip {{ background:{p['panel']}; color:{p['fg']}; border:1px solid {p['border']}; padding:{s(3)}px; }}
QGroupBox {{ border:1px solid {p['border']}; border-radius:6px; margin-top:{s(16)}px; padding-top:{s(6)}px; }}
QGroupBox::title {{ subcontrol-origin:margin; subcontrol-position:top left; left:10px; top:1px; padding:0 4px; color:{p['accent']}; }}
QGroupBox::indicator {{ width:{s(14)}px; height:{s(14)}px; }}
QPushButton {{ background:{p['btn_bg']}; border:1px solid {p['btn_border']}; border-radius:4px; padding:{s(6)}px {s(10)}px; color:{p['btn_fg']}; }}
QPushButton:hover {{ border-color:{p['accent']}; color:{p['accent_fg']}; }}
QPushButton:checked {{ background:{p['run_bg']}; border-color:{p['run_border']}; color:{p['run_fg']}; }}
QPushButton#run {{ background:{p['run_bg']}; border:1px solid {p['run_border']}; color:{p['run_fg']}; font-weight:bold; }}
QPushButton#run:hover {{ background:{p['run_hover']}; }}
QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QPlainTextEdit {{ background:{p['panel']}; border:1px solid {p['border']}; border-radius:4px; padding:{s(3)}px; color:{p['fg']}; }}
QComboBox QAbstractItemView {{ background:{p['panel']}; color:{p['fg']}; selection-background-color:{p['sel_bg']}; }}
QCheckBox::indicator,QRadioButton::indicator {{ width:{s(14)}px; height:{s(14)}px; }}
QCheckBox::indicator:checked {{ background:{p['accent']}; }}
QSlider::groove:horizontal {{ height:{s(5)}px; background:{p['groove']}; border-radius:2px; }}
QSlider::handle:horizontal {{ background:{p['handle']}; width:{s(12)}px; margin:-{s(5)}px 0; border-radius:{s(6)}px; }}
QLabel#title {{ color:{p['title']}; font-size:{s(18)}px; font-weight:bold; }}
QProgressBar {{ border:1px solid {p['border']}; border-radius:4px; text-align:center; }}
QProgressBar::chunk {{ background:{p['progress']}; }}
QScrollBar:vertical {{ background:{p['bg']}; width:{s(12)}px; margin:0; }}
QScrollBar::handle:vertical {{ background:{p['border']}; border-radius:{s(5)}px; min-height:{s(24)}px; }}
QScrollBar::handle:vertical:hover {{ background:{p['accent']}; }}
QScrollBar:horizontal {{ background:{p['bg']}; height:{s(12)}px; margin:0; }}
QScrollBar::handle:horizontal {{ background:{p['border']}; border-radius:{s(5)}px; min-width:{s(24)}px; }}
QScrollBar::add-line,QScrollBar::sub-line {{ width:0; height:0; }}
QSplitter::handle {{ background:{p['border']}; }}
"""


# UI scale presets offered in the toolbar (label -> factor).
UI_SCALES = ["75%", "90%", "100%", "110%", "125%", "150%", "175%", "200%"]
# Live-viewer image size presets (label -> EventView minimum size in px).
VIEWER_SIZES = {"Compact": (280, 240), "Standard": (360, 340),
                "Large": (520, 470), "Huge": (700, 620)}
DEFAULT_VIEWER_SIZE = "Standard"


def _colormap(name):
    if name == "_green":
        g = np.linspace(0, 1, 256); return np.stack([g * 0.2, g, g * 0.3], 1)
    if name == "_ice":
        g = np.linspace(0, 1, 256); return np.stack([g * 0.2, g * 0.8, np.ones_like(g)], 1)
    return _mpl_colormaps[name](np.linspace(0, 1, 256))[:, :3]


# ── Inline chrome ────────────────────────────────────────────────────────────
# This panel predates the shared stylesheet and dresses a few widgets by hand: the plates
# the rendered frames and strips sit on, and the monospace readouts. Both are built from
# the instrument palette so the panel follows the app's light/dark theme; every widget
# holding one is refreshed by :meth:`Player.apply_theme` when the theme changes.

def _plate_qss():
    """The recessed plate a rendered image / strip is drawn on."""
    return f"background:{style.BG2}; border:1px solid {style.BORDER};"


def _mono_qss(font_size_px=None):
    """A monospace accent readout (frame counters, status lines)."""
    size = f" font-size:{font_size_px}px;" if font_size_px else ""
    return (f"color:{style.ACCENT}; "
            f"font-family:Consolas,'DejaVu Sans Mono',monospace;{size}")


class RangeSlider(QtWidgets.QWidget):
    """iMovie-style dual-handle trim slider (emits normalized 0..1 lo/hi)."""
    rangeChanged = QtCore.Signal(float, float)

    def __init__(self):
        super().__init__()
        self.lo, self.hi, self._drag = 0.0, 1.0, None
        self.setMinimumHeight(26); self.setEnabled(False)

    def set_range(self, lo, hi):
        self.lo, self.hi = max(0.0, min(lo, hi)), min(1.0, max(lo, hi)); self.update()

    def _px(self, v): return int(9 + v * (self.width() - 18))
    def _val(self, x): return min(1.0, max(0.0, (x - 9) / max(1, self.width() - 18)))

    def paintEvent(self, _):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, y = self.width(), self.height() // 2
        p.setPen(QtGui.QPen(QtGui.QColor(style.BORDER), 4)); p.drawLine(9, y, w - 9, y)
        x0, x1 = self._px(self.lo), self._px(self.hi)
        p.setPen(QtGui.QPen(QtGui.QColor(style.SELECT), 6)); p.drawLine(x0, y, x1, y)
        p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(style.HANDLE))
        for x in (x0, x1):
            p.drawRoundedRect(QtCore.QRectF(x - 5, y - 9, 10, 18), 3, 3)

    def mousePressEvent(self, e):
        if not self.isEnabled(): return
        v = self._val(e.position().x())
        self._drag = "lo" if abs(v - self.lo) <= abs(v - self.hi) else "hi"; self._move(v)

    def mouseMoveEvent(self, e):
        if self._drag: self._move(self._val(e.position().x()))

    def mouseReleaseEvent(self, e): self._drag = None

    def _move(self, v):
        if self._drag == "lo": self.lo = min(v, self.hi)
        elif self._drag == "hi": self.hi = max(v, self.lo)
        self.update(); self.rangeChanged.emit(self.lo, self.hi)


class EventView(QtWidgets.QLabel):
    """Image view with rubber-band ROI selection (emits ROI in sensor px)."""
    roiSelected = QtCore.Signal(object)

    def __init__(self):
        super().__init__("Load a .raw file to preview", alignment=QtCore.Qt.AlignCenter)
        self.setMinimumSize(360, 340); self.setStyleSheet(_plate_qss())
        self._rb = None; self._origin = None
        self._geom = (1.0, 0, 0, 320, 320)      # scale, ox, oy, W, H (events mode only)
        self.roi_enabled = True

    def set_geom(self, scale, ox, oy, W, H):
        self._geom = (scale, ox, oy, W, H)

    def mousePressEvent(self, e):
        if not self.roi_enabled:
            return
        self._origin = e.position().toPoint()
        if self._rb is None:
            self._rb = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._rb.setGeometry(QtCore.QRect(self._origin, QtCore.QSize())); self._rb.show()

    def mouseMoveEvent(self, e):
        if self._origin is not None:
            self._rb.setGeometry(QtCore.QRect(self._origin, e.position().toPoint()).normalized())

    def mouseReleaseEvent(self, e):
        if self._origin is None:
            return
        r = self._rb.geometry(); self._origin = None; self._rb.hide()
        scale, ox, oy, W, H = self._geom
        if scale <= 0:
            return
        xs = sorted((max(0, min(W - 1, (r.left() - ox) / scale)), max(0, min(W - 1, (r.right() - ox) / scale))))
        ys = sorted((max(0, min(H - 1, (r.top() - oy) / scale)), max(0, min(H - 1, (r.bottom() - oy) / scale))))
        if xs[1] - xs[0] >= 3 and ys[1] - ys[0] >= 3:
            self.roiSelected.emit((int(xs[0]), int(ys[0]), int(xs[1]), int(ys[1])))


_PANO_AXIS_CACHE: dict = {}   # width -> uint8 [BAR_H, width, 3] — rendered once, reused every frame


def _pano_axis_overlay(rgb):
    """Composite the pre-rendered degree-axis strip onto the bottom of a panorama frame.
    The strip itself is built with Pillow once per unique width then cached — zero PIL cost
    per frame so interactive scrubbing stays fast even with two panorama viewers open."""
    h, w = rgb.shape[:2]; BAR_H = 14
    if w not in _PANO_AXIS_CACHE:
        from PIL import Image as _PILImg, ImageDraw as _PILDraw
        strip_img = _PILImg.new("RGB", (w, BAR_H), (0, 0, 0))
        draw = _PILDraw.Draw(strip_img)
        color = (80, 180, 200)
        for deg in (0, 45, 90, 135, 180, 225, 270, 315, 360):
            px = int(round(deg * (w - 1) / 360.0))
            draw.line([(px, 0), (px, 3)], fill=color)
            label = f"{deg}°"
            try:
                bb = draw.textbbox((0, 0), label); tw = bb[2] - bb[0]
            except AttributeError:
                tw = len(label) * 6
            tx = int(np.clip(px - tw // 2, 0, w - 1))
            draw.text((tx, 4), label, fill=color)
        _PANO_AXIS_CACHE[w] = np.asarray(strip_img).copy()
    out = rgb.copy()
    out[-BAR_H:] = _PANO_AXIS_CACHE[w]
    return out


class DensityStrip(QtWidgets.QLabel):
    """Pre-rendered event-density waveform for the full recording.
    Built once at file load from a coarse sample of the timestamp memmap; click to jump."""
    jumped = QtCore.Signal(float)   # normalized position 0..1

    _H = 28

    def __init__(self):
        super().__init__()
        self.setFixedHeight(self._H)
        self.setStyleSheet(_plate_qss())
        self._dur = 0.0
        self._norm = None                     # the built waveform, kept so a theme switch
        self.setEnabled(False)                # can re-render it without the memmap
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.setToolTip("Event-density timeline — click to jump to that position.")

    def build(self, t_memmap, n, dur):
        self._dur = max(dur, 1e-9)
        W = max(self.width(), 512)
        stride = max(1, n // 1_000_000)
        ts = np.asarray(t_memmap[::stride], np.float64) * 1e-6
        counts, _ = np.histogram(ts, bins=W, range=(0.0, dur))
        counts = counts.astype(np.float32)
        vmax = float(counts.max()) if counts.max() > 0 else 1.0
        self._norm = counts / vmax
        self._render_bars()
        self.setEnabled(True)

    def _render_bars(self):
        """Draw the cached waveform in the palette's plate + accent (also what a theme
        switch re-runs, so the strip never stays on the outgoing theme's colours)."""
        if self._norm is None:
            return
        norm = self._norm
        W = norm.size
        plate = QtGui.QColor(style.BG2).getRgb()[:3]
        bar = QtGui.QColor(style.ACCENT).getRgb()[:3]
        img = np.empty((self._H, W, 3), np.uint8)
        img[:, :] = plate
        bar_hs = (norm * self._H).astype(int)
        for x in range(W):
            if bar_hs[x] > 0:
                img[self._H - bar_hs[x]:, x] = bar
        qi = QtGui.QImage(img.data, W, self._H, 3 * W, QtGui.QImage.Format_RGB888)
        self.setPixmap(QtGui.QPixmap.fromImage(qi).scaled(
            self.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.FastTransformation))

    def apply_theme(self, *_):
        """Re-take the plate and re-render the waveform after a light/dark switch."""
        self.setStyleSheet(_plate_qss())
        self._render_bars()

    def mousePressEvent(self, e):
        if self._dur <= 0: return
        pos = float(np.clip(e.position().x() / max(self.width() - 1, 1), 0.0, 1.0))
        self.jumped.emit(pos)


class AnnotationBar(QtWidgets.QWidget):
    """Thin strip below the scrub bar — paints named time-marker flags; click to jump."""
    jumped = QtCore.Signal(float)   # emits the annotation time in seconds

    def __init__(self):
        super().__init__()
        self.setFixedHeight(10)
        self.setEnabled(False)
        self.setToolTip("Annotation markers — click a flag to jump to it.")
        self._annotations = []
        self._dur = 1.0

    def set_dur(self, dur, annotations):
        self._dur = max(dur, 1e-9)
        self._annotations = annotations
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(12, 20, 28))
        if not self._dur: return
        for ann in self._annotations:
            x = int(ann['t'] / self._dur * (self.width() - 1))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 200, 0), 2))
            p.drawLine(x, 0, x, self.height())

    def mousePressEvent(self, e):
        if not self.isEnabled() or not self._dur: return
        best, dist = None, float('inf')
        for ann in self._annotations:
            x = ann['t'] / self._dur * (self.width() - 1)
            d = abs(e.position().x() - x)
            if d < dist:
                dist = d; best = ann
        if best and dist < 12:
            self.jumped.emit(best['t'])


class Player(QtWidgets.QWidget):
    """Live EBS frame viewer: scrub, trim (range slider), accumulation, themes,
    raw/scaled rendering, ROI, live view modes (events/panorama/radar), readouts."""
    scrubbed = QtCore.Signal(int)               # for playhead sync across viewers

    def __init__(self):
        super().__init__()
        self.ev = None; self.t = None; self.p = None; self.dur = 0.0
        self._tkey = np.zeros(0, np.int64); self._tstride = 1; self._n = 0   # RAM time index (fast seeking)
        self.W = self.H = 320; self.accum = 0.03; self.cmap = _colormap("inferno")
        self.raw_mode = False; self.roi = None; self.tel = None; self.fov = 50.0
        self.pos_color = np.array([1.0, 1.0, 1.0]); self.neg_color = np.array([1.0, 0.32, 0.0])
        self.bg_color = np.array([0.02, 0.03, 0.09])     # canvas background (distinct from events)
        self.map_window = 2.0                            # Panorama/Radar accumulation window [s] (when not 'All')
        self.play_fps = 60.0; self._pt = 0.0             # frame-generation fps + float playhead cursor [s]
        self.sync_offset = 0.0                           # time offset for multi-view slate alignment [s]
        self.n_on = self.n_off = 0
        # novel view parameters (V1-V7)
        self.at_win = 10.0; self.at_rows = 200          # V1: AT waterfall
        self.pdiv_radius = 1.5; self.pdiv_scale = 1.0   # V2: polarity divergence
        self.plock_n = 2                                 # V3: phase-locked pano
        self.stv_theta = 30.0; self.stv_phi = 20.0; self.stv_tdepth = 0.4  # V4: space-time vol
        self.iat_min_ev = 3; self.iat_pct_lo = 5; self.iat_pct_hi = 95     # V5: IAT surface
        self.et_win = 5.0                                # V7: elevation-time sweep
        self._diff_snapshot = None                       # V6: reference pano for diff mode
        # Freq Map state (V8)
        self.fmap_win = 0.5; self.fmap_grid = 16        # window (s), spatial grid NxN
        self.fmap_fs  = 1000                             # time-axis sample rate (bins/s)
        self.fmap_flo = 80.0; self.fmap_fhi = 800.0     # frequency range [Hz]
        # Live FFT panel config
        self.fft_win  = 2.0                              # FFT window duration [s]
        self.fft_flo  = 20.0                             # lower frequency cutoff (exclude 1 Hz rotation)
        self._src_path = ""                              # source file path (for FFTDialog metadata)
        # count-based accumulation (U5)
        self.accum_count_mode = False; self.accum_n = 5000
        # annotations (U3)
        self.annotations = []; self._ann_path = None
        # result overlay (U4)
        self._result_track = None
        # FFT frame counter (U2)
        self._fft_tick = 0
        self.col_mode = "band"            # column selection mode: "band" | "single" | "stack"
        self.col_stack = []               # list of int column indices (stack mode)
        self._col_lut = None              # bool [W] or None (all columns pass)
        v = QtWidgets.QVBoxLayout(self)
        self.view = EventView(); self.view.roiSelected.connect(self.set_roi)
        v.addWidget(self.view, 1)
        # file-level readout
        self.readout = QtWidgets.QLabel("—"); self.readout.setStyleSheet(_mono_qss())
        v.addWidget(self.readout)
        # trim (range) slider + playhead
        v.addWidget(QtWidgets.QLabel("trim ◁ ▷  (drag handles to select In/Out)"))
        self.range = RangeSlider(); self.range.rangeChanged.connect(self._range_changed); v.addWidget(self.range)
        self.scrub = QtWidgets.QSlider(QtCore.Qt.Horizontal, minimum=0, maximum=1000, enabled=False)
        self.scrub.setToolTip("Playhead — scrub through the recording.")
        self.scrub.valueChanged.connect(self._scrubbed); v.addWidget(self.scrub)
        self.tlabel = QtWidgets.QLabel("t = -- s"); v.addWidget(self.tlabel)
        # U3: annotation marker strip
        self._ann_bar = AnnotationBar(); self._ann_bar.jumped.connect(self._jump_to_annotation)
        v.addWidget(self._ann_bar)
        # U1: event-density timeline waveform
        self._density_strip = DensityStrip()
        self._density_strip.jumped.connect(lambda pos: self.scrub.setValue(int(pos * 1000)) if self.dur else None)
        v.addWidget(self._density_strip)
        # In/Out numeric inputs
        io = QtWidgets.QHBoxLayout()
        self.in_s = QtWidgets.QDoubleSpinBox(); self.in_s.setDecimals(3); self.in_s.setRange(0, 1e6)
        self.out_s = QtWidgets.QDoubleSpinBox(); self.out_s.setDecimals(3); self.out_s.setRange(0, 1e6)
        self.in_s.setToolTip("In point (s) — start of the selection.")
        self.out_s.setToolTip("Out point (s) — end of the selection.")
        self.in_s.editingFinished.connect(self._spin_changed); self.out_s.editingFinished.connect(self._spin_changed)
        io.addWidget(QtWidgets.QLabel("In")); io.addWidget(self.in_s)
        io.addWidget(QtWidgets.QLabel("Out")); io.addWidget(self.out_s); io.addStretch(1)
        v.addLayout(io)
        # controls
        row = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Play", enabled=False); self.play_btn.clicked.connect(self._toggle)
        self.play_btn.setIcon(icons.icon("play")); self.play_btn.setIconSize(QtCore.QSize(14, 14))
        icons.freeze_width(self.play_btn, ["Play", "Pause"])   # constant geometry across the toggle
        row.addWidget(self.play_btn)
        row.addWidget(QtWidgets.QLabel("fps:"))
        self.fps_cb = QtWidgets.QComboBox(); self.fps_cb.setEditable(True)
        self.fps_cb.addItems(["2", "5", "10", "24", "30", "60", "120", "240", "1000", "5000", "10000"])
        self.fps_cb.setCurrentText("60")
        self.fps_cb.setValidator(QtGui.QDoubleValidator(0.1, 1000000.0, 2))
        self.fps_cb.setToolTip("Frames generated per second of RECORDING time (Metavision-style).\n"
                               "Each frame advances 1/fps of recording time, so a HIGH fps gives tiny,\n"
                               "overlapping steps = ultra-smooth playback. <=60 plays in real time;\n"
                               "above 60 the display caps at 60 fps, so it becomes smooth SLOW-MOTION\n"
                               "(e.g. 5000-10000 fps). Accumulation time is independent (trail length).")
        self.fps_cb.currentTextChanged.connect(self._set_fps); row.addWidget(self.fps_cb)
        self.speed_lbl = QtWidgets.QLabel(); self.speed_lbl.setStyleSheet(f"color:{style.ACCENT};"); row.addWidget(self.speed_lbl)
        row.addWidget(QtWidgets.QLabel("accum (ms):"))
        self.accum_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal, minimum=1, maximum=200, value=30)
        self.accum_sl.setToolTip("Accumulation window: how many ms of events are integrated into each displayed frame.")
        self.accum_sl.valueChanged.connect(self._accum_sl); row.addWidget(self.accum_sl, 1)
        self.accum_sp = QtWidgets.QSpinBox(); self.accum_sp.setRange(1, 200); self.accum_sp.setValue(30)
        self.accum_sp.setToolTip("Accumulation window in ms (numeric).")
        self.accum_sp.valueChanged.connect(self._accum_sp); row.addWidget(self.accum_sp)
        self.raw_chk = QtWidgets.QCheckBox("Raw +/-")
        self.raw_chk.setToolTip("Raw: render events at full brightness with distinct ON/OFF colors (faithful).\n"
                                "Unchecked: density-scaled single colormap (better for dim scenes).")
        self.raw_chk.toggled.connect(self._raw_toggled); row.addWidget(self.raw_chk)
        row.addWidget(QtWidgets.QLabel("theme:"))
        self.theme = QtWidgets.QComboBox(); self.theme.addItems(PLAYER_THEMES.keys())
        self.theme.setToolTip("Colormap for the scaled view.")
        self.theme.currentTextChanged.connect(self._theme_changed); row.addWidget(self.theme)
        v.addLayout(row)
        # U5: count-based accumulation toggle
        row_cnt = QtWidgets.QHBoxLayout()
        self.accum_count_chk = QtWidgets.QCheckBox("count mode  N:")
        self.accum_count_chk.setToolTip("Accumulate exactly N events per frame instead of a fixed time window.\n"
                                        "Stabilises image density when the event rate varies across the recording.")
        self.accum_count_chk.toggled.connect(self._accum_count_toggled); row_cnt.addWidget(self.accum_count_chk)
        self.accum_n_sp = QtWidgets.QSpinBox(); self.accum_n_sp.setRange(100, 500_000)
        self.accum_n_sp.setValue(5000); self.accum_n_sp.setSingleStep(500); self.accum_n_sp.setEnabled(False)
        self.accum_n_sp.setToolTip("Number of events per accumulated frame (count mode).")
        self.accum_n_sp.valueChanged.connect(lambda v: self._set_and_render('accum_n', v))
        row_cnt.addWidget(self.accum_n_sp); row_cnt.addStretch(1)
        v.addLayout(row_cnt)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("view:"))
        self.vmode = QtWidgets.QComboBox(); self.vmode.addItems(VIEW_MODES)
        self.vmode.setToolTip("Live view: Events, Panorama, Radar, and 7 novel analytical modes.\n"
                              "Panorama/Radar/AT-Waterfall/Phase-Locked/Dual-Diff need rotation telemetry.")
        self.vmode.currentTextChanged.connect(self._on_vmode_changed); row2.addWidget(self.vmode)
        # Assumed rotation period for the spin-dependent views when a clip has NO telemetry file.
        # 0 = auto-estimate from the event-rate periodicity. Lets Panorama/Radar/etc. render
        # straight from the events (phase-relative bearings; absolute North uncalibrated).
        row2.addWidget(QtWidgets.QLabel("spin:"))
        self.spin_sp = QtWidgets.QDoubleSpinBox()
        self.spin_sp.setRange(0.0, 10.0); self.spin_sp.setDecimals(3); self.spin_sp.setSingleStep(0.05)
        self.spin_sp.setSuffix(" s"); self.spin_sp.setSpecialValueText("auto"); self.spin_sp.setFixedWidth(86)
        self.spin_sp.setToolTip(
            "Rotation period assumed for the Panorama / Radar / AT-Waterfall / Phase-Locked / "
            "Dual-Diff views when this clip has NO telemetry file.\n"
            "0 = auto-estimate the period from the event-rate periodicity. The synthesized "
            "bearings are rotation-phase-relative — absolute North is uncalibrated.\n"
            "A real telemetry CSV, when present, always overrides this.")
        self.spin_sp.valueChanged.connect(self._on_spin_changed)
        row2.addWidget(self.spin_sp)
        self._spin_period = 0.0          # 0 = auto-estimate; >0 = forced assumed period (s)
        self._synth_key = None           # (period_request) the current synthesized track was built for
        row2.addStretch(1)
        row2.addWidget(QtWidgets.QLabel("ON"))
        self.on_btn = QtWidgets.QPushButton(); self.on_btn.setFixedWidth(30)
        self.on_btn.setToolTip("ON-event color (raw view) — click to choose."); self.on_btn.clicked.connect(lambda: self._pick_color("on"))
        row2.addWidget(self.on_btn)
        row2.addWidget(QtWidgets.QLabel("OFF"))
        self.off_btn = QtWidgets.QPushButton(); self.off_btn.setFixedWidth(30)
        self.off_btn.setToolTip("OFF-event color (raw view) — click to choose."); self.off_btn.clicked.connect(lambda: self._pick_color("off"))
        row2.addWidget(self.off_btn)
        row2.addWidget(QtWidgets.QLabel("BG"))
        self.bg_btn = QtWidgets.QPushButton(); self.bg_btn.setFixedWidth(30)
        self.bg_btn.setToolTip("Canvas color for the Panorama / Radar views.\nThe EBS Events scene stays black.")
        self.bg_btn.clicked.connect(lambda: self._pick_color("bg")); row2.addWidget(self.bg_btn)
        v.addLayout(row2); self._refresh_color_btns()
        # Novel-view parameter controls (one page per mode; pages 0-2 are empty for classic modes)
        self._novel_pages = QtWidgets.QStackedWidget(); self._novel_pages.setMaximumHeight(44)
        for _ in range(3): self._novel_pages.addWidget(QtWidgets.QWidget())   # Events/Pano/Radar: no extra controls
        # page 3: AT Waterfall
        _atw = QtWidgets.QWidget(); _atl = QtWidgets.QHBoxLayout(_atw); _atl.setContentsMargins(0, 0, 0, 0)
        _atl.addWidget(QtWidgets.QLabel("window (s):")); self.at_win_sp = QtWidgets.QDoubleSpinBox()
        self.at_win_sp.setRange(1, 120); self.at_win_sp.setValue(10.0); self.at_win_sp.setSuffix(" s")
        self.at_win_sp.valueChanged.connect(lambda v: self._set_and_render('at_win', v)); _atl.addWidget(self.at_win_sp)
        _atl.addWidget(QtWidgets.QLabel("rows:")); self.at_rows_sp = QtWidgets.QSpinBox()
        self.at_rows_sp.setRange(50, 600); self.at_rows_sp.setValue(200)
        self.at_rows_sp.valueChanged.connect(lambda v: self._set_and_render('at_rows', v)); _atl.addWidget(self.at_rows_sp); _atl.addStretch(1)
        self._novel_pages.addWidget(_atw)
        # page 4: Polarity Div
        _pdw = QtWidgets.QWidget(); _pdl = QtWidgets.QHBoxLayout(_pdw); _pdl.setContentsMargins(0, 0, 0, 0)
        _pdl.addWidget(QtWidgets.QLabel("smooth r:")); self.pdiv_r_sp = QtWidgets.QDoubleSpinBox()
        self.pdiv_r_sp.setRange(0, 10); self.pdiv_r_sp.setSingleStep(0.5); self.pdiv_r_sp.setValue(1.5)
        self.pdiv_r_sp.valueChanged.connect(lambda v: self._set_and_render('pdiv_radius', v)); _pdl.addWidget(self.pdiv_r_sp)
        _pdl.addWidget(QtWidgets.QLabel("scale:")); self.pdiv_sc_sp = QtWidgets.QDoubleSpinBox()
        self.pdiv_sc_sp.setRange(0.1, 10); self.pdiv_sc_sp.setSingleStep(0.1); self.pdiv_sc_sp.setValue(1.0)
        self.pdiv_sc_sp.valueChanged.connect(lambda v: self._set_and_render('pdiv_scale', v)); _pdl.addWidget(self.pdiv_sc_sp); _pdl.addStretch(1)
        self._novel_pages.addWidget(_pdw)
        # page 5: Phase-Locked Pano
        _plw = QtWidgets.QWidget(); _pll = QtWidgets.QHBoxLayout(_plw); _pll.setContentsMargins(0, 0, 0, 0)
        _pll.addWidget(QtWidgets.QLabel("rotations:")); self.plock_n_sp = QtWidgets.QSpinBox()
        self.plock_n_sp.setRange(1, 8); self.plock_n_sp.setValue(2)
        self.plock_n_sp.setToolTip("Accumulate this many complete sensor rotations for a ghost-free 360° map.")
        self.plock_n_sp.valueChanged.connect(lambda v: self._set_and_render('plock_n', v)); _pll.addWidget(self.plock_n_sp); _pll.addStretch(1)
        self._novel_pages.addWidget(_plw)
        # page 6: Space-Time Volume
        _stvw = QtWidgets.QWidget(); _stvl = QtWidgets.QHBoxLayout(_stvw); _stvl.setContentsMargins(0, 0, 0, 0)
        _stvl.addWidget(QtWidgets.QLabel("θ:")); self.stv_theta_sp = QtWidgets.QSpinBox()
        self.stv_theta_sp.setRange(0, 89); self.stv_theta_sp.setValue(30); self.stv_theta_sp.setSuffix("°")
        self.stv_theta_sp.valueChanged.connect(lambda v: self._set_and_render('stv_theta', float(v))); _stvl.addWidget(self.stv_theta_sp)
        _stvl.addWidget(QtWidgets.QLabel("φ:")); self.stv_phi_sp = QtWidgets.QSpinBox()
        self.stv_phi_sp.setRange(0, 89); self.stv_phi_sp.setValue(20); self.stv_phi_sp.setSuffix("°")
        self.stv_phi_sp.valueChanged.connect(lambda v: self._set_and_render('stv_phi', float(v))); _stvl.addWidget(self.stv_phi_sp)
        _stvl.addWidget(QtWidgets.QLabel("t-depth:")); self.stv_td_sp = QtWidgets.QDoubleSpinBox()
        self.stv_td_sp.setRange(0.1, 2.0); self.stv_td_sp.setSingleStep(0.1); self.stv_td_sp.setValue(0.4)
        self.stv_td_sp.valueChanged.connect(lambda v: self._set_and_render('stv_tdepth', v)); _stvl.addWidget(self.stv_td_sp); _stvl.addStretch(1)
        self._novel_pages.addWidget(_stvw)
        # page 7: IAT Surface
        _iatw = QtWidgets.QWidget(); _iatl = QtWidgets.QHBoxLayout(_iatw); _iatl.setContentsMargins(0, 0, 0, 0)
        _iatl.addWidget(QtWidgets.QLabel("pct clip:")); self.iat_plo_sp = QtWidgets.QSpinBox()
        self.iat_plo_sp.setRange(0, 49); self.iat_plo_sp.setValue(5); self.iat_plo_sp.setSuffix("%")
        self.iat_plo_sp.valueChanged.connect(lambda v: self._set_and_render('iat_pct_lo', v)); _iatl.addWidget(self.iat_plo_sp)
        _iatl.addWidget(QtWidgets.QLabel("–")); self.iat_phi_sp = QtWidgets.QSpinBox()
        self.iat_phi_sp.setRange(51, 100); self.iat_phi_sp.setValue(95); self.iat_phi_sp.setSuffix("%")
        self.iat_phi_sp.valueChanged.connect(lambda v: self._set_and_render('iat_pct_hi', v)); _iatl.addWidget(self.iat_phi_sp)
        _iatl.addWidget(QtWidgets.QLabel("min ev:")); self.iat_min_sp = QtWidgets.QSpinBox()
        self.iat_min_sp.setRange(2, 20); self.iat_min_sp.setValue(3)
        self.iat_min_sp.valueChanged.connect(lambda v: self._set_and_render('iat_min_ev', v)); _iatl.addWidget(self.iat_min_sp); _iatl.addStretch(1)
        self._novel_pages.addWidget(_iatw)
        # page 8: Dual-Cam Diff
        _dcdw = QtWidgets.QWidget(); _dcdl = QtWidgets.QHBoxLayout(_dcdw); _dcdl.setContentsMargins(0, 0, 0, 0)
        self.diff_snap_btn = QtWidgets.QPushButton("📸 Snapshot ref")
        self.diff_snap_btn.setToolTip("Capture the current panorama as the reference; differential = live − ref.")
        self.diff_snap_btn.clicked.connect(self._diff_take_snapshot); _dcdl.addWidget(self.diff_snap_btn)
        _dcdl.addWidget(QtWidgets.QLabel("blend:")); self.diff_blend_cb = QtWidgets.QComboBox()
        self.diff_blend_cb.addItems(["subtract", "ratio", "abs-diff"])
        self.diff_blend_cb.setToolTip("subtract: live-ref+128 (signed). ratio: live/ref×128. abs-diff: |live-ref|.")
        self.diff_blend_cb.currentTextChanged.connect(lambda _: self._render(self.current_t()))
        _dcdl.addWidget(self.diff_blend_cb); _dcdl.addStretch(1)
        self._novel_pages.addWidget(_dcdw)
        # page 9: Elev-Time Sweep
        _etw = QtWidgets.QWidget(); _etl = QtWidgets.QHBoxLayout(_etw); _etl.setContentsMargins(0, 0, 0, 0)
        _etl.addWidget(QtWidgets.QLabel("window (s):")); self.et_win_sp = QtWidgets.QDoubleSpinBox()
        self.et_win_sp.setRange(0.5, 120); self.et_win_sp.setValue(5.0); self.et_win_sp.setSuffix(" s")
        self.et_win_sp.setToolTip("Time window scrolled across the X axis. Y axis = sensor elevation (row).")
        self.et_win_sp.valueChanged.connect(lambda v: self._set_and_render('et_win', v)); _etl.addWidget(self.et_win_sp); _etl.addStretch(1)
        self._novel_pages.addWidget(_etw)
        # page 10: Freq Map
        _fmw = QtWidgets.QWidget(); _fml = QtWidgets.QHBoxLayout(_fmw); _fml.setContentsMargins(0, 0, 0, 0)
        _fml.addWidget(QtWidgets.QLabel("win (s):")); self.fmap_win_sp = QtWidgets.QDoubleSpinBox()
        self.fmap_win_sp.setRange(0.1, 5.0); self.fmap_win_sp.setSingleStep(0.1); self.fmap_win_sp.setValue(0.5); self.fmap_win_sp.setSuffix(" s")
        self.fmap_win_sp.setToolTip("Time window for per-pixel FFT. Longer = better frequency resolution; shorter = more responsive.")
        self.fmap_win_sp.valueChanged.connect(lambda v: self._set_and_render('fmap_win', v)); _fml.addWidget(self.fmap_win_sp)
        _fml.addWidget(QtWidgets.QLabel("grid:")); self.fmap_grid_cb = QtWidgets.QComboBox()
        self.fmap_grid_cb.addItems(["8", "16", "32", "48"])
        self.fmap_grid_cb.setCurrentText("16")
        self.fmap_grid_cb.setToolTip("Spatial grid resolution (N×N cells). Higher = finer spatial detail but slower.")
        self.fmap_grid_cb.currentTextChanged.connect(lambda t: self._set_and_render('fmap_grid', int(t))); _fml.addWidget(self.fmap_grid_cb)
        _fml.addWidget(QtWidgets.QLabel("f:")); self.fmap_flo_sp = QtWidgets.QDoubleSpinBox()
        self.fmap_flo_sp.setRange(1, 2000); self.fmap_flo_sp.setValue(80); self.fmap_flo_sp.setSuffix(" Hz")
        self.fmap_flo_sp.setToolTip("Lower bound of the target frequency range.")
        self.fmap_flo_sp.valueChanged.connect(lambda v: self._set_and_render('fmap_flo', v)); _fml.addWidget(self.fmap_flo_sp)
        _fml.addWidget(QtWidgets.QLabel("–")); self.fmap_fhi_sp = QtWidgets.QDoubleSpinBox()
        self.fmap_fhi_sp.setRange(10, 5000); self.fmap_fhi_sp.setValue(800); self.fmap_fhi_sp.setSuffix(" Hz")
        self.fmap_fhi_sp.setToolTip("Upper bound of the target frequency range.")
        self.fmap_fhi_sp.valueChanged.connect(lambda v: self._set_and_render('fmap_fhi', v)); _fml.addWidget(self.fmap_fhi_sp); _fml.addStretch(1)
        self._novel_pages.addWidget(_fmw)
        v.addWidget(self._novel_pages)
        # U2: live FFT spectrum panel (toggle-able)
        fft_row = QtWidgets.QHBoxLayout()
        self.fft_chk = QtWidgets.QCheckBox("Live FFT")
        self.fft_chk.setToolTip("Show a live event-rate frequency spectrum (ROI-aware).\nReveals drone rotor harmonics, wingbeat, and other periodic signatures.\nDC removed via mean subtraction + Hanning window.")
        self.fft_chk.toggled.connect(self._toggle_fft); fft_row.addWidget(self.fft_chk)
        fft_row.addWidget(QtWidgets.QLabel("win:")); self.fft_win_sp = QtWidgets.QDoubleSpinBox()
        self.fft_win_sp.setRange(0.1, 10.0); self.fft_win_sp.setSingleStep(0.5); self.fft_win_sp.setValue(2.0); self.fft_win_sp.setSuffix(" s")
        self.fft_win_sp.setToolTip("FFT time window. Longer = finer frequency resolution (0.5 Hz at 2s).")
        self.fft_win_sp.valueChanged.connect(lambda v: self._set_and_render('fft_win', v)); fft_row.addWidget(self.fft_win_sp)
        self.fft_expand_btn = QtWidgets.QPushButton("Expand")
        self.fft_expand_btn.setIcon(icons.icon("split"))
        self.fft_expand_btn.setToolTip("Open a full-size FFT analyser with parameter tuning and save options.")
        self.fft_expand_btn.clicked.connect(self._open_fft_dialog); fft_row.addWidget(self.fft_expand_btn)
        fft_row.addStretch(1); v.addLayout(fft_row)
        self._fft_panel = QtWidgets.QWidget()
        _ffl = QtWidgets.QVBoxLayout(self._fft_panel); _ffl.setContentsMargins(0, 0, 0, 0)
        self._fft_lbl = QtWidgets.QLabel()
        self._fft_lbl.setFixedHeight(58); self._fft_lbl.setStyleSheet(_plate_qss())
        _ffl.addWidget(self._fft_lbl)
        self._fft_panel.setVisible(False); v.addWidget(self._fft_panel)
        # U3: annotation controls row
        ann_row = QtWidgets.QHBoxLayout()
        ann_add_btn = QtWidgets.QPushButton("Annotation")
        ann_add_btn.setIcon(icons.icon("add"))
        ann_add_btn.setToolTip("Add a named time marker at the current playhead. Saved as <file>_annotations.json.")
        ann_add_btn.clicked.connect(self._add_annotation); ann_row.addWidget(ann_add_btn)
        ann_del_btn = QtWidgets.QPushButton("× Del nearest")
        ann_del_btn.setToolTip("Remove the annotation closest to the current playhead position.")
        ann_del_btn.clicked.connect(self._del_nearest_annotation); ann_row.addWidget(ann_del_btn)
        ann_row.addStretch(1); v.addLayout(ann_row)
        # Panorama/Radar accumulation ("map window") — the map-view equivalent of accum time
        row3 = QtWidgets.QHBoxLayout()
        row3.addWidget(QtWidgets.QLabel("map window (s):"))
        self.mapwin_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal, minimum=1, maximum=600, value=20)
        self.mapwin_sl.setToolTip("Panorama / Radar accumulation: seconds of past events integrated into the map\n"
                                  "(ending at the playhead). Lower = recent-only live sweep; higher = fuller map.")
        self.mapwin_sl.valueChanged.connect(self._mapwin_sl); row3.addWidget(self.mapwin_sl, 1)
        self.mapwin_sp = QtWidgets.QDoubleSpinBox(); self.mapwin_sp.setDecimals(1)
        self.mapwin_sp.setRange(0.1, 100000.0); self.mapwin_sp.setValue(2.0); self.mapwin_sp.setSuffix(" s")
        self.mapwin_sp.setToolTip("Map accumulation window in seconds (numeric).")
        self.mapwin_sp.valueChanged.connect(self._mapwin_sp); row3.addWidget(self.mapwin_sp)
        self.mapall_chk = QtWidgets.QCheckBox("All"); self.mapall_chk.setChecked(True)
        self.mapall_chk.setToolTip("Accumulate ALL events from the start (cumulative map). Uncheck to use the window.")
        self.mapall_chk.toggled.connect(self._mapall_toggled); row3.addWidget(self.mapall_chk)
        self.mapwin_sl.setEnabled(False); self.mapwin_sp.setEnabled(False)   # 'All' is default
        v.addLayout(row3)
        # Column (azimuth-slice) selector — three modes: Band (range), Single (one column), Stack (arbitrary set).
        row4a = QtWidgets.QHBoxLayout()
        row4a.addWidget(QtWidgets.QLabel("col mode:"))
        self.col_mode_band = QtWidgets.QPushButton("Band"); self.col_mode_single = QtWidgets.QPushButton("Single")
        self.col_mode_stack = QtWidgets.QPushButton("Stack")
        for btn in (self.col_mode_band, self.col_mode_single, self.col_mode_stack):
            btn.setCheckable(True); row4a.addWidget(btn)
        self.col_mode_band.setChecked(True)
        self._col_mode_grp = QtWidgets.QButtonGroup(self); self._col_mode_grp.setExclusive(True)
        for btn in (self.col_mode_band, self.col_mode_single, self.col_mode_stack):
            self._col_mode_grp.addButton(btn)
        self.col_mode_band.clicked.connect(lambda: self._col_mode_switch("band", 0))
        self.col_mode_single.clicked.connect(lambda: self._col_mode_switch("single", 1))
        self.col_mode_stack.clicked.connect(lambda: self._col_mode_switch("stack", 2))
        row4a.addStretch(1)
        self.col_lbl = QtWidgets.QLabel("all 320"); self.col_lbl.setStyleSheet(f"color:{style.ACCENT};")
        row4a.addWidget(self.col_lbl); v.addLayout(row4a)
        self._col_pages = QtWidgets.QStackedWidget()
        # page 0: Band — range slider + from/to spinboxes
        _bw = QtWidgets.QWidget(); _bl = QtWidgets.QHBoxLayout(_bw); _bl.setContentsMargins(0, 0, 0, 0)
        self.col_range = RangeSlider(); self.col_range.set_range(0.0, 1.0)
        self.col_range.setToolTip("Azimuth band: drag handles to select a column range (all views).")
        self.col_range.rangeChanged.connect(self._col_band_changed); _bl.addWidget(self.col_range, 1)
        self.col_from = QtWidgets.QSpinBox(); self.col_from.setRange(0, 320); self.col_from.setValue(0)
        self.col_to = QtWidgets.QSpinBox(); self.col_to.setRange(0, 320); self.col_to.setValue(320)
        self.col_from.setToolTip("First column, inclusive."); self.col_to.setToolTip("Last column, exclusive.")
        self.col_from.editingFinished.connect(self._col_spin); self.col_to.editingFinished.connect(self._col_spin)
        _bl.addWidget(QtWidgets.QLabel("from")); _bl.addWidget(self.col_from)
        _bl.addWidget(QtWidgets.QLabel("to")); _bl.addWidget(self.col_to)
        self._col_pages.addWidget(_bw)
        # page 1: Single — one column, slider to navigate
        _sw = QtWidgets.QWidget(); _sl = QtWidgets.QHBoxLayout(_sw); _sl.setContentsMargins(0, 0, 0, 0)
        _sl.addWidget(QtWidgets.QLabel("col:"))
        self.col_single_sp = QtWidgets.QSpinBox(); self.col_single_sp.setRange(0, 319); self.col_single_sp.setValue(160)
        self.col_single_sp.setToolTip("Single sensor column to visualize (all views).")
        self.col_single_sp.editingFinished.connect(self._col_single_changed); _sl.addWidget(self.col_single_sp)
        self.col_single_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal, minimum=0, maximum=319, value=160)
        self.col_single_sl.setToolTip("Drag to navigate single column.")
        self.col_single_sl.valueChanged.connect(self._col_single_sl_changed); _sl.addWidget(self.col_single_sl, 1)
        self._col_pages.addWidget(_sw)
        # page 2: Stack — add non-adjacent columns by index
        _stw = QtWidgets.QWidget(); _stl = QtWidgets.QHBoxLayout(_stw); _stl.setContentsMargins(0, 0, 0, 0)
        self.col_stack_lbl = QtWidgets.QLabel("(empty)")
        self.col_stack_lbl.setStyleSheet(f"color:{style.ACCENT}; background:{style.PANEL}; "
                                         f"border:1px solid {style.BORDER}; border-radius:3px; "
                                         "padding:2px 6px;")
        _stl.addWidget(self.col_stack_lbl, 1)
        self.col_add_sp = QtWidgets.QSpinBox(); self.col_add_sp.setRange(0, 319); self.col_add_sp.setValue(0)
        self.col_add_btn = QtWidgets.QPushButton("+ Add"); self.col_add_btn.clicked.connect(self._col_stack_add)
        self.col_clr_btn = QtWidgets.QPushButton("x Clear"); self.col_clr_btn.clicked.connect(self._col_stack_clear)
        _stl.addWidget(self.col_add_sp); _stl.addWidget(self.col_add_btn); _stl.addWidget(self.col_clr_btn)
        self._col_pages.addWidget(_stw)
        v.addWidget(self._col_pages)
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self._tick)
        self._update_speed_lbl()
        style.notifier().themeChanged.connect(self.apply_theme)

    def apply_theme(self, *_):
        """Re-dress the hand-styled widgets after an app light/dark switch.

        Everything else in this panel is plain Qt and follows the application stylesheet;
        these few carry inline sheets (plates and monospace readouts) that Qt has no reason
        to revisit on its own.
        """
        for w in (self.view, self._fft_lbl):
            if w is not None:
                w.setStyleSheet(_plate_qss())
        self._density_strip.apply_theme()          # plate + a re-render of its waveform
        self.readout.setStyleSheet(_mono_qss())
        for w in (self.speed_lbl, self.col_lbl):
            w.setStyleSheet(f"color:{style.ACCENT};")
        self.col_stack_lbl.setStyleSheet(f"color:{style.ACCENT}; background:{style.PANEL}; "
                                         f"border:1px solid {style.BORDER}; border-radius:3px; "
                                         "padding:2px 6px;")
        self._refresh_color_btns()

    # ---- ROI + colors ----
    def set_roi(self, roi): self.roi = roi; self._render(self.current_t())
    def clear_roi(self): self.roi = None; self._render(self.current_t())

    def _pick_color(self, which):
        cur = {"on": self.pos_color, "off": self.neg_color, "bg": self.bg_color}[which] * 255
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(*cur.astype(int)), self, "Pick color")
        if c.isValid():
            arr = np.array([c.redF(), c.greenF(), c.blueF()])
            if which == "on": self.pos_color = arr; self.raw_chk.setChecked(True)
            elif which == "off": self.neg_color = arr; self.raw_chk.setChecked(True)
            else: self.bg_color = arr
            self._refresh_color_btns(); self._render(self.current_t())

    def _refresh_color_btns(self):
        for btn, col in ((self.on_btn, self.pos_color), (self.off_btn, self.neg_color),
                         (self.bg_btn, self.bg_color)):
            r, g, b = (col * 255).astype(int)
            btn.setStyleSheet(f"background:rgb({r},{g},{b}); border:1px solid {style.BORDER};")

    # ---- selection accessors (seconds) ----
    def sel_t0(self): return self.range.lo * self.dur
    def sel_t1(self): return self.range.hi * self.dur

    def load(self, path):
        """Synchronous load (tests / non-GUI). The GUI uses a background thread + set_data."""
        self.set_data(path, io_evt21.load(path))

    def set_data(self, path, ev):
        """Populate the player from an already-decoded (memmapped) event dict. The event
        arrays stay memmapped (x/y/p/t) so a billion-event file stays light; alongside them we
        build a compact RAM-resident time index (see _build_time_index) so per-frame seeking is
        fast and scrubbing stays smooth. Counts come from the cache, not a full reduction."""
        self.ev = ev
        self.t = ev["t"]                                 # int64 MICROSECONDS, memmapped (zero-copy)
        self.p = ev["p"]                                 # uint8 polarity, memmapped
        self.W, self.H = ev["width"], ev["height"]
        n = int(ev.get("n", len(self.t)))
        self._n = n; self._build_time_index()            # RAM time index for smooth seeking (bounded; coarse only for billion-event files)
        self.dur = float(self.t[-1]) * 1e-6 if n else 0.0
        self.n_on = int(ev.get("n_on", 0)); self.n_off = max(0, n - self.n_on)
        self.roi = None
        self.fov = _fov_for(path); self.tel = None       # telemetry for panorama/radar views
        self._synth_key = None                            # force a fresh assumed-spin synth for this clip
        try:
            csv = (glob.glob(os.path.join(os.path.dirname(path), "data_*.csv"))
                   or glob.glob(os.path.join(os.path.dirname(path), "*.csv")))
            if csv:
                step = max(1, n // 200_000)              # subsample for offset refine (avoid materializing t)
                tsec = np.asarray(self.t[::step], np.float64) * 1e-6
                self.tel = Telemetry(csv[0]); self.tel.refine_offset_to_events(tsec)
        except Exception:
            self.tel = None
        rate = n / self.dur if self.dur else 0
        fmt = str(ev.get("fmt", "evt21")).upper()
        self.readout.setText(
            f"[{fmt}]  {n:,} ev  |  ON {self.n_on:,} ({100*self.n_on/max(n,1):.0f}%)  "
            f"OFF {self.n_off:,} ({100*self.n_off/max(n,1):.0f}%)  |  {self.dur:.2f}s  "
            f"{self.W}x{self.H}  |  mean {rate/1e3:.0f}k ev/s")
        self.range.setEnabled(True); self.range.set_range(0.0, 1.0)
        self.in_s.setValue(0.0); self.out_s.setValue(self.dur)
        # reset column selector to full width
        self.col_from.setRange(0, self.W); self.col_to.setRange(0, self.W)
        self.col_single_sp.setRange(0, self.W - 1); self.col_single_sl.setRange(0, self.W - 1)
        self.col_add_sp.setRange(0, self.W - 1)
        for sp, val in ((self.col_from, 0), (self.col_to, self.W)):
            sp.blockSignals(True); sp.setValue(val); sp.blockSignals(False)
        for sp, val in ((self.col_single_sp, self.W // 2), (self.col_single_sl, self.W // 2)):
            sp.blockSignals(True); sp.setValue(val); sp.blockSignals(False)
        self.col_range.blockSignals(True); self.col_range.set_range(0.0, 1.0); self.col_range.blockSignals(False)
        self.col_stack.clear(); self.col_stack_lbl.setText("(empty)")
        self._col_lut = None; self._update_col_lbl()
        self._pt = 0.0
        self.scrub.setEnabled(True); self.play_btn.setEnabled(True)
        # U1: build density timeline strip (coarse sample, done in set_data to avoid blocking play)
        QtCore.QTimer.singleShot(50, lambda: self._density_strip.build(self.t, n, self.dur))
        # U3: load annotations from sidecar JSON
        self._load_annotations(path)
        self._result_track = None   # clear any prior overlay
        self._src_path = path
        self.scrub.setValue(0); self._render(0.0)

    # Keep the WHOLE timestamp column in RAM up to ~100M events (~800 MB int64) so per-frame
    # seeking never touches the memmap (cold page-faults are what stutter on this platform).
    # Only genuinely huge (billion-event) files fall back to the coarse index + bounded refine.
    _TINDEX_MAX = 100_000_000

    def _build_time_index(self):
        """RAM-resident timestamp index for fast per-frame seeking (see _ss). Normal files keep
        their whole timestamp column in RAM (exact, fastest); only genuinely huge (> _TINDEX_MAX
        event) files fall back to a strided coarse index, so RAM stays bounded without
        re-introducing the multi-GB full load the streaming decoder was built to avoid."""
        n = self._n
        if n <= 0:
            self._tkey = np.zeros(0, np.int64); self._tstride = 1; return
        self._tstride = 1 if n <= self._TINDEX_MAX else -(-n // self._TINDEX_MAX)   # ceil division
        self._tkey = np.array(self.t[::self._tstride], dtype=np.int64)              # force a RAM copy

    def _ss(self, t_us):
        """Index into the timestamps as np.searchsorted(self.t, t_us) would, but fast.

        Two things make per-frame seeking cheap:
          1. The query is cast to int64 FIRST. Searching an int64 array with a *float* query
             makes NumPy upcast the WHOLE array to float64 on every call (O(n) per frame); run
             twice per frame that was the dominant playback-stutter regression (self.t became
             int64 µs in the ingestion rewrite while the query stayed float seconds·1e6).
          2. We binary-search the small RAM key array, then (big files only) refine inside a
             bounded memmap slice — so seeking is O(log n) on RAM and never re-scans the array."""
        k = self._tkey
        if k.size == 0:
            return 0
        v = np.int64(np.ceil(t_us))                  # int64 query (ceil => identical to float side='left')
        j = int(np.searchsorted(k, v))
        if self._tstride == 1:                       # key IS the full timestamp column -> exact answer
            return j
        lo = max(0, (j - 1) * self._tstride)
        hi = min(self._n, j * self._tstride + 1)     # bracket is guaranteed to contain the true index
        return lo + int(np.searchsorted(np.asarray(self.t[lo:hi]), v))

    def current_t(self):
        return self.scrub.value() / 1000.0 * self.dur if self.dur else 0.0

    def _accum_sl(self, v): self.accum_sp.blockSignals(True); self.accum_sp.setValue(v); self.accum_sp.blockSignals(False); self._set_accum(v)
    def _accum_sp(self, v): self.accum_sl.blockSignals(True); self.accum_sl.setValue(v); self.accum_sl.blockSignals(False); self._set_accum(v)
    def _set_accum(self, v): self.accum = v / 1000.0; self._render(self.current_t())

    def _wall_ms(self):
        """Wall-clock timer interval. Capped at PLAYBACK_WALL_FPS so high fps requests
        turn into smooth slow-motion instead of an impossible refresh rate."""
        return max(1, int(round(1000.0 / min(self.play_fps, PLAYBACK_WALL_FPS))))

    def _set_fps(self, text):
        try: f = float(text)
        except ValueError: return
        if f <= 0: return
        self.play_fps = f
        if self.timer.isActive(): self.timer.start(self._wall_ms())
        self._update_speed_lbl()

    def _update_speed_lbl(self):
        spd = min(self.play_fps, PLAYBACK_WALL_FPS) / self.play_fps     # playback rate vs real time
        self.speed_lbl.setText("realtime" if spd >= 0.999 else f"{1.0 / spd:.0f}x slow")

    # ---- column (azimuth-slice) selection — Band / Single / Stack ----
    def _col_mode_switch(self, mode, page):
        self.col_mode = mode
        self._col_pages.setCurrentIndex(page)
        self._rebuild_col_lut(); self._render(self.current_t())

    def _col_band_changed(self, lo, hi):
        x0 = int(round(lo * self.W)); x1 = int(round(hi * self.W))
        for sp, val in ((self.col_from, x0), (self.col_to, x1)):
            sp.blockSignals(True); sp.setValue(val); sp.blockSignals(False)
        self._rebuild_col_lut(); self._render(self.current_t())

    def _col_spin(self):
        x0, x1 = self.col_from.value(), self.col_to.value()
        if x1 < x0: x1 = x0; self.col_to.blockSignals(True); self.col_to.setValue(x1); self.col_to.blockSignals(False)
        self.col_range.blockSignals(True); self.col_range.set_range(x0 / max(self.W, 1), x1 / max(self.W, 1)); self.col_range.blockSignals(False)
        self._rebuild_col_lut(); self._render(self.current_t())

    def _col_single_changed(self):
        v = self.col_single_sp.value()
        self.col_single_sl.blockSignals(True); self.col_single_sl.setValue(v); self.col_single_sl.blockSignals(False)
        self._rebuild_col_lut(); self._render(self.current_t())

    def _col_single_sl_changed(self, v):
        self.col_single_sp.blockSignals(True); self.col_single_sp.setValue(v); self.col_single_sp.blockSignals(False)
        self._rebuild_col_lut(); self._render(self.current_t())

    def _col_stack_add(self):
        c = self.col_add_sp.value()
        if c not in self.col_stack:
            self.col_stack.append(c); self.col_stack.sort()
            self.col_stack_lbl.setText(", ".join(str(x) for x in self.col_stack))
        self._rebuild_col_lut(); self._render(self.current_t())

    def _col_stack_clear(self):
        self.col_stack.clear(); self.col_stack_lbl.setText("(empty)")
        self._rebuild_col_lut(); self._render(self.current_t())

    def _rebuild_col_lut(self):
        W = self.W
        if self.col_mode == "band":
            x0, x1 = self.col_from.value(), self.col_to.value()
            if x0 <= 0 and x1 >= W:
                self._col_lut = None
            else:
                lut = np.zeros(W, bool); lut[max(0, x0):min(W, x1)] = True
                self._col_lut = lut
        elif self.col_mode == "single":
            c = self.col_single_sp.value()
            lut = np.zeros(W, bool)
            if 0 <= c < W: lut[c] = True
            self._col_lut = lut
        else:  # stack
            if not self.col_stack:
                self._col_lut = None
            else:
                lut = np.zeros(W, bool)
                for c in self.col_stack:
                    if 0 <= c < W: lut[c] = True
                self._col_lut = lut
        self._update_col_lbl()

    def _update_col_lbl(self):
        lut = self._col_lut
        if lut is None:
            self.col_lbl.setText(f"all {self.W}")
        else:
            n = int(lut.sum())
            self.col_lbl.setText("none" if n == 0 else f"{n} col{'s' if n != 1 else ''}")

    def _col_filter(self):
        return self._col_lut

    def _mapwin_sl(self, v): self.mapwin_sp.blockSignals(True); self.mapwin_sp.setValue(v / 10.0); self.mapwin_sp.blockSignals(False); self._set_mapwin(v / 10.0)
    def _mapwin_sp(self, v): self.mapwin_sl.blockSignals(True); self.mapwin_sl.setValue(int(round(v * 10))); self.mapwin_sl.blockSignals(False); self._set_mapwin(v)
    def _set_mapwin(self, secs):
        self.map_window = float(secs)
        if not self.mapall_chk.isChecked(): self._render(self.current_t())
    def _mapall_toggled(self, on):
        self.mapwin_sl.setEnabled(not on); self.mapwin_sp.setEnabled(not on); self._render(self.current_t())
    def _theme_changed(self, name): self.cmap = _colormap(PLAYER_THEMES[name]); self._render(self.current_t())
    def _raw_toggled(self, on): self.raw_mode = on; self._render(self.current_t())
    def _scrubbed(self, v): self._pt = self.current_t(); self.scrubbed.emit(v); self._render(self.current_t())

    def _range_changed(self, lo, hi):
        for s, val in ((self.in_s, lo * self.dur), (self.out_s, hi * self.dur)):
            s.blockSignals(True); s.setValue(val); s.blockSignals(False)

    def _spin_changed(self):
        if self.dur:
            self.range.set_range(self.in_s.value() / self.dur, self.out_s.value() / self.dur)

    def _toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play"); self.play_btn.setIcon(icons.icon("play"))
        else:
            self._pt = self.current_t()
            self.timer.start(self._wall_ms())
            self.play_btn.setText("Pause"); self.play_btn.setIcon(icons.icon("pause"))

    def _tick(self):
        if not self.dur: return
        t0 = time.perf_counter()
        self._pt += 1.0 / self.play_fps                      # advance ONE frame of recording time (1/fps)
        if self._pt >= self.dur: self._pt = 0.0
        v = int(self._pt / self.dur * 1000)
        self.scrub.blockSignals(True); self.scrub.setValue(v); self.scrub.blockSignals(False)
        self.scrubbed.emit(v)                                # keep multi-viewer playheads synced
        self._render(self._pt)
        render_ms = int((time.perf_counter() - t0) * 1000)
        target_ms = self._wall_ms()
        new_ms = max(1, render_ms + 1)                       # aim to fire just after render completes
        if abs(self.timer.interval() - new_ms) > 2 and self.timer.isActive():
            self.timer.start(new_ms)                         # adapt interval; drains any queued backlog

    def _cmap_rgb(self, img2d, empty=None):
        empty = self.bg_color if empty is None else empty          # Panorama/Radar -> BG; Events -> black
        vmax = max(np.percentile(img2d[img2d > 0], 99), 1.0) if (img2d > 0).any() else 1.0
        idx = np.clip(img2d / vmax * 255, 0, 255).astype(np.uint8)
        rgb = (self.cmap[idx] * 255).astype(np.uint8)
        rgb[img2d <= 0] = (np.asarray(empty) * 255).astype(np.uint8)
        return rgb

    def _derot_az(self, x, t):
        pan = np.rad2deg(np.interp(t - self.tel.offset, self.tel.t, self.tel.azimuth_unwrapped()))
        return np.mod(pan - (x - self.W / 2) * (self.fov / self.W), 360.0)

    def _boresight_az(self, tc):
        """World azimuth (deg) the sensor centre points at time tc — the live sweep bearing."""
        if self.tel is None: return None
        pan = np.rad2deg(np.interp(tc - self.tel.offset, self.tel.t, self.tel.azimuth_unwrapped()))
        return float(np.mod(pan, 360.0))

    _LEAD = (0, 229, 255); _TRAIL = (150, 90, 60)    # sweep markers: new data / data aging out

    def _draw_sweep_pano(self, rgb, tc):
        """Vertical sweep lines on the panorama: leading edge = current bearing (new data),
        trailing edge = bearing of the window start (data leaving the sliding window)."""
        az = self._boresight_az(tc)
        if az is None: return
        w = rgb.shape[1]; scale = w / 360.0              # px per degree (2.0 for 720-wide panorama)
        rgb[:, int(az * scale) % w] = self._LEAD
        if not self.mapall_chk.isChecked():
            az0 = self._boresight_az(max(tc - self.map_window, 0.0))
            if az0 is not None: rgb[::2, int(az0 * scale) % w] = self._TRAIL   # dashed = trailing edge

    def _draw_sweep_radar(self, rgb, tc):
        """Radial sweep 'needle' on the radar: leading edge = current bearing, plus a dashed
        trailing needle at the window start when a finite map-window is active."""
        az = self._boresight_az(tc)
        if az is None: return
        S = rgb.shape[0]; c = (S - 1) / 2.0; R = S * 0.46
        rr = np.linspace(self.R_INNER, 1.0, S)
        def needle(a, col, step=1):
            th = np.deg2rad(a)
            px = (c + rr * R * np.cos(th)).astype(np.int64)[::step]
            py = (c + rr * R * np.sin(th)).astype(np.int64)[::step]
            ok = (px >= 0) & (px < S) & (py >= 0) & (py < S)
            rgb[py[ok], px[ok]] = col
        needle(az, self._LEAD)
        if not self.mapall_chk.isChecked():
            az0 = self._boresight_az(max(tc - self.map_window, 0.0))
            if az0 is not None: needle(az0, self._TRAIL, step=2)

    def _map_lo(self, hi):
        """Lower index for Panorama/Radar accumulation. 'All' -> 0 (cumulative from start);
        otherwise the start of a `map_window`-second window ending at the playhead."""
        if self.mapall_chk.isChecked() or hi <= 0:
            return 0
        tc = float(self.t[min(hi, len(self.t)) - 1]) * 1e-6           # self.t is µs
        return self._ss((tc - self.map_window) * 1e6)

    def _sub(self, lo, hi, cap, cols=None):
        """Sample up to `cap` events from [lo, hi) via a stride read — reads only cap events from
        the memmap instead of all n, then discarding 98% with Fisher-Yates. 30-50× less I/O on
        large All-mode windows. Temporal order is preserved; slight temporal bias is fine for
        histogram-based views (panorama/radar/waterfall)."""
        n_raw = hi - lo
        if n_raw <= 0:
            return np.zeros(0, np.float64), np.zeros(0, np.float64), np.zeros(0, np.float64)
        stride = max(1, -(-n_raw // cap))                              # ceil(n_raw / cap)
        sl = slice(lo, hi, stride)
        x = np.asarray(self.ev["x"][sl])
        y = np.asarray(self.ev["y"][sl])
        t = np.asarray(self.t[sl], np.float64) * 1e-6
        if cols is not None:
            m = cols[np.clip(x, 0, len(cols) - 1)]; x, y, t = x[m], y[m], t[m]
        return x.astype(np.float64), y.astype(np.float64), t

    def _render_events(self, lo, hi, ps):
        xs = np.asarray(self.ev["x"][lo:hi]); ys = np.asarray(self.ev["y"][lo:hi])
        lut = self._col_filter()
        if lut is not None:                                         # apply column mask (all three modes)
            m = lut[np.clip(xs, 0, len(lut) - 1)]; xs, ys, ps = xs[m], ys[m], ps[m]
        BLACK = (0.0, 0.0, 0.0)                                    # EBS event scene = black background
        if self.raw_mode:
            rgb = np.zeros((self.H, self.W, 3), np.float32)
            if (ps == 0).any(): rgb[ys[ps == 0], xs[ps == 0]] = self.neg_color
            if (ps == 1).any(): rgb[ys[ps == 1], xs[ps == 1]] = self.pos_color
            rgb = (rgb * 255).astype(np.uint8)
        else:
            img = np.zeros((self.H, self.W), np.float32)
            if hi > lo: np.add.at(img, (ys, xs), 1.0)
            rgb = self._cmap_rgb(img, empty=BLACK)
        if self.roi:
            x0, y0, x1, y1 = self.roi
            rgb[y0:y1 + 1, [x0, min(x1, self.W - 1)]] = (0, 255, 0)
            rgb[[y0, min(y1, self.H - 1)], x0:x1 + 1] = (0, 255, 0)
        return np.ascontiguousarray(rgb)

    def _render_panorama(self, hi):
        if self.tel is None or hi < 2: return None
        PAN_W = 720                                          # 2px per degree: 360° × 2 = 720 columns (less vertically compressed)
        x, y, t = self._sub(self._map_lo(hi), hi, 600_000, cols=self._col_filter())
        if len(x) == 0:
            rgb = self._cmap_rgb(np.zeros((self.H, PAN_W), np.float32))
        else:
            az = self._derot_az(x, t)
            Hh = np.histogram2d(y, az, bins=[self.H, PAN_W], range=[[0, self.H], [0, 360]])[0]
            rgb = self._cmap_rgb(np.log1p(Hh))
        rgb = _pano_axis_overlay(rgb)
        self._draw_sweep_pano(rgb, float(self.t[min(hi, len(self.t)) - 1]) * 1e-6)
        return np.ascontiguousarray(rgb)

    R_INNER = 0.20                                       # blank inner circle (donut hole), in disk units

    def _radar_canvas(self, S):
        """Two-tone radar background (cached): BG everywhere except the donut annulus,
        which is shifted slightly lighter/darker. Inner and outer circle boundaries are
        anti-aliased over ~2px so the disc looks smooth instead of pixelated."""
        key = (S, tuple(np.round(self.bg_color, 4)))
        if getattr(self, "_radar_key", None) == key:
            return self._radar_bg
        c = (S - 1) / 2.0; R = S * 0.46
        yy, xx = np.mgrid[0:S, 0:S]
        rad = np.hypot(xx - c, yy - c) / R
        lum = float(self.bg_color @ np.array([0.299, 0.587, 0.114]))
        sgn = 1.0 if lum < 0.5 else -1.0
        bg_f    = (self.bg_color * 255).astype(np.float32)
        field_f = (np.clip(self.bg_color + sgn * 0.06, 0, 1) * 255).astype(np.float32)
        # ~2-pixel soft transition at each circle boundary (alpha 0→1 over 2/R in rad units)
        alpha_in  = np.clip((rad - self.R_INNER) * (R * 0.5), 0.0, 1.0)
        alpha_out = np.clip((1.0 - rad) * (R * 0.5), 0.0, 1.0)
        alpha = (alpha_in * alpha_out)[:, :, np.newaxis].astype(np.float32)
        rgb = np.clip(alpha * field_f + (1.0 - alpha) * bg_f, 0, 255).astype(np.uint8)
        self._radar_key = key; self._radar_bg = rgb
        return rgb

    def _render_radar(self, hi):
        if self.tel is None or hi < 2: return None
        S = self.H; tc = float(self.t[min(hi, len(self.t)) - 1]) * 1e-6
        rgb = self._radar_canvas(S).copy()                        # two-tone background (donut field stands out)
        x, y, t = self._sub(self._map_lo(hi), hi, 400_000, cols=self._col_filter())
        if len(x):
            th = np.deg2rad(self._derot_az(x, t))
            rr = self.R_INNER + (1.0 - self.R_INNER) * (y / self.H)  # map elevation into [r_inner, 1]
            c = (S - 1) / 2.0; R = S * 0.46
            px = (c + rr * R * np.cos(th)).astype(np.int64); py = (c + rr * R * np.sin(th)).astype(np.int64)
            ok = (px >= 0) & (px < S) & (py >= 0) & (py < S)
            cv = np.zeros((S, S), np.float32); np.add.at(cv, (py[ok], px[ok]), 1.0)
            if (cv > 0).any():
                ev = self._cmap_rgb(np.log1p(cv)); has = cv > 0; rgb[has] = ev[has]
        self._draw_sweep_radar(rgb, tc)
        return np.ascontiguousarray(rgb)

    # ── Novel view renderers (V1-V7) ──────────────────────────────────────────

    def _rotation_period(self):
        """Estimate sensor rotation period [s] from the telemetry azimuth trace."""
        if self.tel is None: return None
        az = self.tel.azimuth_unwrapped(); t = self.tel.t
        if len(t) < 2: return None
        total_rad = abs(float(az[-1]) - float(az[0]))
        total_t   = abs(float(t[-1]) - float(t[0]))
        if total_rad < 0.1 or total_t < 0.1: return None
        return float((2 * np.pi) / (total_rad / total_t))

    def _render_at_waterfall(self, hi):
        """V1: Azimuth-Time waterfall — X=0..360° azimuth, Y=time rows (newest top), colour=density."""
        if self.tel is None or hi < 2: return None
        PAN_W = 720; ROWS = self.at_rows
        tc = float(self.t[min(hi, len(self.t)) - 1]) * 1e-6
        lo = self._ss((tc - self.at_win) * 1e6)
        x, y, t = self._sub(lo, hi, 800_000, cols=self._col_filter())
        if len(x) == 0:
            return self._cmap_rgb(np.zeros((ROWS, PAN_W), np.float32))
        az = self._derot_az(x, t)
        t_row = np.clip(((t - (tc - self.at_win)) / self.at_win * ROWS).astype(int), 0, ROWS - 1)
        t_row = ROWS - 1 - t_row                          # newest at top
        az_col = np.clip((az / 360.0 * PAN_W).astype(int), 0, PAN_W - 1)
        H = np.zeros((ROWS, PAN_W), np.float32)
        np.add.at(H, (t_row, az_col), 1.0)
        rgb = self._cmap_rgb(np.log1p(H))
        rgb = _pano_axis_overlay(rgb)
        return np.ascontiguousarray(rgb)

    def _render_polarity_div(self, lo, hi, ps):
        """V2: Polarity divergence — signed (ON-OFF) map; RdBu colormap; optional Gaussian smooth."""
        xs = np.asarray(self.ev["x"][lo:hi], np.int32)
        ys = np.asarray(self.ev["y"][lo:hi], np.int32)
        lut = self._col_filter()
        if lut is not None:
            m = lut[np.clip(xs, 0, len(lut) - 1)]; xs, ys, ps = xs[m], ys[m], ps[m]
        on_img = np.zeros((self.H, self.W), np.float32); off_img = np.zeros((self.H, self.W), np.float32)
        if len(xs):
            on_m = (ps == 1); off_m = (ps == 0)
            if on_m.any():  np.add.at(on_img,  (ys[on_m],  xs[on_m]),  1.0)
            if off_m.any(): np.add.at(off_img, (ys[off_m], xs[off_m]), 1.0)
        r = self.pdiv_radius
        if r > 0:
            from scipy.ndimage import gaussian_filter
            on_img = gaussian_filter(on_img, r).astype(np.float32)
            off_img = gaussian_filter(off_img, r).astype(np.float32)
        div = on_img - off_img
        vmax = max(float(np.abs(div).max()) * self.pdiv_scale, 1e-6)
        idx = np.clip((div / vmax * 127.5 + 127.5), 0, 255).astype(np.uint8)
        if not hasattr(self, '_pdiv_cmap'):
            self._pdiv_cmap = (_mpl_colormaps['RdBu'](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        return np.ascontiguousarray(self._pdiv_cmap[idx])

    def _render_phase_locked_pano(self, hi):
        """V3: Phase-locked panorama — accumulates exactly N sensor rotations for a ghost-free 360° map."""
        if self.tel is None or hi < 2: return None
        period = self._rotation_period()
        if period is None: return self._render_panorama(hi)
        tc = float(self.t[min(hi, len(self.t)) - 1]) * 1e-6
        lo = self._ss(max(0.0, tc - period * self.plock_n) * 1e6)
        PAN_W = 720
        x, y, t = self._sub(lo, hi, 800_000, cols=self._col_filter())
        if len(x) == 0:
            return self._cmap_rgb(np.zeros((self.H, PAN_W), np.float32))
        az = self._derot_az(x, t)
        Hh = np.histogram2d(y, az, bins=[self.H, PAN_W], range=[[0, self.H], [0, 360]])[0]
        rgb = self._cmap_rgb(np.log1p(Hh.astype(np.float32)))
        rgb = _pano_axis_overlay(rgb)
        return np.ascontiguousarray(rgb)

    def _render_st_volume(self, lo, hi):
        """V4: Space-time volume — orthographic projection of (x, y, t) events onto a 2D canvas."""
        x, y, t_r = self._sub(lo, hi, 40_000, cols=self._col_filter())
        if len(x) == 0:
            return np.zeros((self.H, self.W, 3), np.uint8)
        th = np.deg2rad(self.stv_theta); ph = np.deg2rad(self.stv_phi)
        t_n = ((t_r - t_r.min()) / max(t_r.max() - t_r.min(), 1e-9)) * self.W * self.stv_tdepth
        u = x * np.cos(th) - t_n * np.sin(th)
        v_p = y * np.cos(ph) + (x * np.sin(th) + t_n * np.cos(th)) * np.sin(ph)
        u -= u.min(); v_p -= v_p.min()
        sc = min(self.W / max(u.max(), 1.0), self.H / max(v_p.max(), 1.0)) * 0.92
        pu = (u * sc + (self.W - u.max() * sc) / 2).astype(np.int64)
        pv = (v_p * sc + (self.H - v_p.max() * sc) / 2).astype(np.int64)
        ok = (pu >= 0) & (pu < self.W) & (pv >= 0) & (pv < self.H)
        img = np.zeros((self.H, self.W), np.float32)
        np.add.at(img, (pv[ok], pu[ok]), 1.0)
        return np.ascontiguousarray(self._cmap_rgb(np.log1p(img), empty=(0.02, 0.03, 0.09)))

    def _render_iat_surface(self, lo, hi):
        """V5: Inter-arrival-time surface — per-pixel median IAT; short IAT = high activity = bright."""
        if hi <= lo: return np.zeros((self.H, self.W, 3), np.uint8)
        xs = np.asarray(self.ev["x"][lo:hi], np.int32)
        ys = np.asarray(self.ev["y"][lo:hi], np.int32)
        ts = np.asarray(self.t[lo:hi], np.float64) * 1e-6
        lut = self._col_filter()
        if lut is not None:
            m = lut[np.clip(xs, 0, len(lut) - 1)]; xs, ys, ts = xs[m], ys[m], ts[m]
        if len(xs) < 2: return np.zeros((self.H, self.W, 3), np.uint8)
        order = np.lexsort((ts, ys.astype(np.int64), xs.astype(np.int64)))
        xs_s = xs[order]; ys_s = ys[order]; ts_s = ts[order]
        same = (xs_s[:-1] == xs_s[1:]) & (ys_s[:-1] == ys_s[1:])
        dts  = np.diff(ts_s)[same]
        pix  = ys_s[:-1][same].astype(np.int64) * self.W + xs_s[:-1][same].astype(np.int64)
        if not len(dts): return np.zeros((self.H, self.W, 3), np.uint8)
        iat_sum = np.bincount(pix, weights=dts,  minlength=self.H * self.W).reshape(self.H, self.W).astype(np.float32)
        iat_cnt = np.bincount(pix, minlength=self.H * self.W).reshape(self.H, self.W).astype(np.float32)
        valid   = iat_cnt >= self.iat_min_ev
        iat = np.where(valid, iat_sum / np.maximum(iat_cnt, 1), 0.0)
        if valid.any():
            vals = iat[valid]
            lo_p = np.percentile(vals, self.iat_pct_lo); hi_p = np.percentile(vals, self.iat_pct_hi)
            iat_n = np.where(valid, np.clip((iat - lo_p) / max(hi_p - lo_p, 1e-9), 0, 1), 0.0)
            iat_n = 1.0 - iat_n.astype(np.float32)   # invert: short IAT = high activity = bright
        else:
            iat_n = np.zeros((self.H, self.W), np.float32)
        return np.ascontiguousarray(self._cmap_rgb(iat_n, empty=(0.0, 0.0, 0.0)))

    def _render_freq_map(self, hi):
        """V8/Freq Map: per-cell dominant-frequency heatmap.
        Divides the sensor (or current ROI) into an NxN grid. For each cell, bins events
        into time slots → FFT → dominant frequency in [f_lo, f_hi]. Colour = frequency (HSV),
        brightness = spectral power. DC removed via mean-subtraction + Hanning window
        (match the reference viewer: subtract the per-cell mean before the FFT)."""
        if self.ev is None or hi < 2: return None
        tc  = float(self.t[min(hi, len(self.t)) - 1]) * 1e-6
        WIN = self.fmap_win; FS = self.fmap_fs; GN = self.fmap_grid
        f_lo, f_hi = self.fmap_flo, self.fmap_fhi
        lo  = self._ss((tc - WIN) * 1e6)
        n_tbins = max(int(WIN * FS), 16)

        # Load events — stride-cap to keep computation fast
        n_raw = hi - lo
        stride = max(1, -(-n_raw // 500_000))
        sl = slice(lo, hi, stride)
        xs = np.asarray(self.ev["x"][sl], np.int32)
        ys = np.asarray(self.ev["y"][sl], np.int32)
        ts = np.asarray(self.t[sl], np.float64) * 1e-6

        # Apply column LUT if set
        lut = self._col_filter()
        if lut is not None:
            m = lut[np.clip(xs, 0, len(lut) - 1)]; xs, ys, ts = xs[m], ys[m], ts[m]

        # Restrict to ROI if drawn — map to ROI-local grid coords
        if self.roi:
            rx0, ry0, rx1, ry1 = self.roi
            roi_mask = (xs >= rx0) & (xs <= rx1) & (ys >= ry0) & (ys <= ry1)
            xs, ys, ts = xs[roi_mask], ys[roi_mask], ts[roi_mask]
            roi_w = max(rx1 - rx0, 1); roi_h = max(ry1 - ry0, 1)
            gx = np.clip(((xs - rx0) * GN / roi_w).astype(int), 0, GN - 1)
            gy = np.clip(((ys - ry0) * GN / roi_h).astype(int), 0, GN - 1)
        else:
            gx = np.clip((xs * GN / self.W).astype(int), 0, GN - 1)
            gy = np.clip((ys * GN / self.H).astype(int), 0, GN - 1)

        bt = np.clip(((ts - (tc - WIN)) * FS).astype(int), 0, n_tbins - 1)

        # Build 3D histogram [grid_y, grid_x, time_bin] — vectorised via add.at
        vol = np.zeros((GN, GN, n_tbins), np.float32)
        if len(gx): np.add.at(vol, (gy, gx, bt), 1.0)

        # DC removal: subtract the per-cell mean before the FFT (match the reference viewer)
        vol -= vol.mean(axis=2, keepdims=True)
        # Spectral leakage reduction: Hanning window
        hann = np.hanning(n_tbins).astype(np.float32)
        vol *= hann                            # broadcast over [GN, GN, n_tbins]

        # FFT → one-sided magnitude spectrum [GN, GN, n_freq]
        spec   = np.abs(np.fft.rfft(vol, axis=2))
        freqs  = np.fft.rfftfreq(n_tbins, d=1.0 / FS)

        # Exclude f=0 (DC bin) and restrict to target range
        f_mask = (freqs >= max(f_lo, freqs[1])) & (freqs <= f_hi)
        if not f_mask.any():
            return np.zeros((self.H, self.W, 3), np.uint8)

        spec_m   = spec[:, :, f_mask]          # [GN, GN, n_in_range]
        freqs_m  = freqs[f_mask]

        peak_pow = spec_m.max(axis=2)          # [GN, GN]
        peak_idx = spec_m.argmax(axis=2)       # [GN, GN]
        dom_freq = freqs_m[peak_idx]           # [GN, GN]

        # Significance threshold: suppress cells below 5% of global max or 30th percentile
        pow_thresh = max(float(np.percentile(peak_pow, 30)), float(peak_pow.max()) * 0.05)
        sig = peak_pow > pow_thresh

        # Map dominant frequency → hue (HSV colormap, 0.85 span avoids red wrap-around)
        if not hasattr(self, '_fmap_cmap'):
            self._fmap_cmap = (_mpl_colormaps['hsv'](np.linspace(0, 0.85, 256))[:, :3] * 255).astype(np.uint8)
        freq_n = np.clip((dom_freq - f_lo) / max(f_hi - f_lo, 1.0), 0, 1)
        cmap_idx = (freq_n * 255).astype(np.uint8)
        rgb_grid = self._fmap_cmap[cmap_idx].astype(np.float32)   # [GN, GN, 3]

        # Alpha-blend: significant cells bright, insignificant dark
        pw_norm = np.clip(peak_pow / max(float(peak_pow.max()), 1e-9), 0, 1)[:, :, np.newaxis]
        alpha   = np.where(sig[:, :, np.newaxis], pw_norm, 0.0).astype(np.float32)
        bg      = np.array([10, 10, 20], np.float32)
        rgb_out = (alpha * rgb_grid + (1.0 - alpha) * bg).clip(0, 255).astype(np.uint8)

        # Upsample grid to sensor resolution
        from PIL import Image as _PILImg
        if self.roi:
            rx0, ry0, rx1, ry1 = self.roi
            ow = rx1 - rx0; oh = ry1 - ry0
            up = np.array(_PILImg.fromarray(rgb_out).resize((max(ow, 1), max(oh, 1)), resample=_PILImg.NEAREST))
            canvas = np.zeros((self.H, self.W, 3), np.uint8)
            canvas[ry0:ry0 + oh, rx0:rx0 + ow] = up[:min(oh, self.H - ry0), :min(ow, self.W - rx0)]
            return np.ascontiguousarray(canvas)
        return np.ascontiguousarray(
            np.array(_PILImg.fromarray(rgb_out).resize((self.W, self.H), resample=_PILImg.NEAREST)))

    def _diff_take_snapshot(self):
        """V6: Capture the current panorama as the reference for the differential view."""
        hi = self._ss(self.current_t() * 1e6)
        snap = self._render_panorama(hi)
        if snap is not None:
            self._diff_snapshot = snap.copy()

    def _render_dual_diff(self, hi):
        """V6: Differential dual-camera view — live panorama minus a saved reference snapshot."""
        if self.tel is None or hi < 2: return None
        pano = self._render_panorama(hi)
        if pano is None: return None
        if self._diff_snapshot is None: return pano       # no ref yet — show plain pano as guide
        h = min(pano.shape[0], self._diff_snapshot.shape[0])
        w = min(pano.shape[1], self._diff_snapshot.shape[1])
        a = pano[:h, :w].astype(np.float32); b = self._diff_snapshot[:h, :w].astype(np.float32)
        blend = self.diff_blend_cb.currentText()
        if blend == "subtract":
            out = np.clip(a - b + 128, 0, 255).astype(np.uint8)
        elif blend == "ratio":
            out = np.clip(a / np.maximum(b, 1.0) * 128, 0, 255).astype(np.uint8)
        else:
            out = np.clip(np.abs(a - b), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(out)

    def _render_elev_time(self, hi):
        """V7: Elevation-time sweep — Y=sensor elevation (row), X=time; rolling window spectrogram."""
        if hi < 2: return None
        tc  = float(self.t[min(hi, len(self.t)) - 1]) * 1e-6
        lo  = self._ss((tc - self.et_win) * 1e6)
        xs  = np.asarray(self.ev["x"][lo:hi], np.int32)
        ys  = np.asarray(self.ev["y"][lo:hi], np.int32)
        ts  = np.asarray(self.t[lo:hi], np.float64) * 1e-6
        lut = self._col_filter()
        if lut is not None:
            m = lut[np.clip(xs, 0, len(lut) - 1)]; xs, ys, ts = xs[m], ys[m], ts[m]
        OUT_W = max(self.W, 200)
        if len(xs) == 0:
            return self._cmap_rgb(np.zeros((self.H, OUT_W), np.float32))
        t_col = np.clip(((ts - (tc - self.et_win)) / self.et_win * OUT_W).astype(int), 0, OUT_W - 1)
        H_img = np.zeros((self.H, OUT_W), np.float32)
        valid = (ys >= 0) & (ys < self.H)
        np.add.at(H_img, (ys[valid], t_col[valid]), 1.0)
        return np.ascontiguousarray(self._cmap_rgb(np.log1p(H_img)))

    def _on_spin_changed(self, v):
        """The assumed-spin period changed — drop any synthesized track so the new period takes
        effect (a real telemetry CSV is left untouched) and re-render the current view."""
        self._spin_period = float(v)
        if self.tel is not None and getattr(self.tel, "synthesized", False):
            self.tel = None
            self._synth_key = None
        self._render(self.current_t())

    def _ensure_rotation_telemetry(self):
        """Make an azimuth(t) track available for the spin-dependent views even when the clip
        shipped NO telemetry file: synthesize one from the recording itself.

        The period is taken from the 'spin' control, or auto-estimated from the event-rate
        periodicity when that is 0 (a spinning sensor re-images the scene once per revolution, so
        its rate is periodic at the rotation period). A clip with no detectable periodicity falls
        back to a single sweep across the whole clip. The synthesized bearings are
        rotation-phase-relative — absolute North is unknown — so the track is flagged estimated and
        the views annotate themselves accordingly. A real logged telemetry CSV always wins and is
        never overwritten."""
        if self.tel is not None and not getattr(self.tel, "synthesized", False):
            return True                                        # real logged telemetry — keep it
        if self.ev is None or self.dur <= 0:
            return self.tel is not None
        want = float(self._spin_period or 0.0)
        if self.tel is not None and self._synth_key == want:   # already synthesized for this setting
            return True
        period, conf = want, None
        if period <= 0:                                        # auto-estimate from the events
            step = max(1, self._n // 300_000)
            tsec = np.asarray(self.t[::step], np.float64) * 1e-6
            period, conf = estimate_spin_period_s(tsec)
            if not period or period <= 0:
                period = float(self.dur)                       # aperiodic clip → one sweep over the clip
        tel = Telemetry.from_spin(self.dur, float(period))
        tel.synthesized = True
        self.tel = tel
        self._synth_key = want
        self._synth_period_est = float(period)
        self._synth_conf = conf
        return True

    def _tag_font(self, px):
        """A cached PIL font at pixel size *px* (truetype() does disk I/O, so don't reload it on
        every playback frame)."""
        cache = self.__dict__.setdefault("_font_cache", {})
        if px not in cache:
            from PIL import ImageFont as _PILFont
            f = None
            for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
                try:
                    f = _PILFont.truetype(name, px); break
                except Exception:
                    continue
            if f is None:
                try:
                    f = _PILFont.load_default(px)
                except Exception:
                    f = _PILFont.load_default()
            cache[px] = f
        return cache[px]

    def _mark_estimated(self, rgb):
        """Stamp a small 'est. spin · North uncal.' tag onto a synthesized rotation frame so the
        caveat travels with on-screen views and exported frames alike."""
        try:
            from PIL import Image as _PILImg, ImageDraw as _PILDraw
            period = getattr(self, "_synth_period_est", 0.0)
            txt = f"est. spin {period:.3f}s · North uncal."
            im = _PILImg.fromarray(rgb); draw = _PILDraw.Draw(im)
            draw.text((4, 3), txt, fill=(120, 200, 220), font=self._tag_font(max(10, rgb.shape[0] // 36)))
            return np.ascontiguousarray(np.asarray(im, dtype=np.uint8))
        except Exception:
            return rgb

    def _render_needs_rotation(self, mode):
        """Placeholder card for the rotation-only views (Panorama / Radar / AT Waterfall /
        Phase-Locked Pano / Dual-Cam Diff) when no azimuth telemetry is available.

        Those modes de-rotate events against the sensor's spin, which is impossible without a
        telemetry track. Rather than silently re-showing the Events frame — which makes the view
        selector look broken ("stuck on Events") — draw an explicit, mode-named notice so the
        switch is visibly registered and the requirement is clear."""
        from PIL import Image as _PILImg, ImageDraw as _PILDraw, ImageFont as _PILFont
        W, H = max(int(self.W), 80), max(int(self.H), 60)
        img = _PILImg.new("RGB", (W, H), (14, 18, 26))           # dark slate, distinct from event black
        draw = _PILDraw.Draw(img)

        def _font(px):
            for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
                try:
                    return _PILFont.truetype(name, px)
                except Exception:
                    continue
            try:
                return _PILFont.load_default(px)                 # Pillow >= 10.1 accepts a size
            except Exception:
                return _PILFont.load_default()

        title_f = _font(max(14, H // 12))
        body_f  = _font(max(11, H // 22))
        accent, muted = (90, 190, 215), (150, 165, 180)
        lines = [(mode, title_f, accent),
                 ("needs rotation telemetry", body_f, muted),
                 ("", body_f, muted),
                 ("No azimuth track for this clip.", body_f, muted),
                 ("Load a rotating capture with a data_*.csv", body_f, muted),
                 ("sidecar so the events can be de-rotated.", body_f, muted)]

        def _tw_th(text, font):
            try:
                b = draw.textbbox((0, 0), text, font=font); return b[2] - b[0], b[3] - b[1]
            except Exception:
                return len(text) * 6, 11

        heights = [(_tw_th(t, f)[1] if t else max(6, H // 40)) for t, f, _ in lines]
        gap = max(4, H // 60)
        total = sum(heights) + gap * (len(lines) - 1)
        y = max(4, (H - total) // 2)
        for (text, font, color), th in zip(lines, heights):
            if text:
                tw, _ = _tw_th(text, font)
                draw.text(((W - tw) // 2, y), text, fill=color, font=font)
            y += th + gap
        return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))

    # ── Analysis result overlay ────────────────────────────────────────────────

    def _draw_result_overlay(self, rgb, tc, pano_mode=False):
        """U4: Draw a yellow bearing crosshair from the last analysis result onto rgb (in-place)."""
        if self._result_track is None: return
        track = self._result_track
        bearing = float(np.interp(tc, track['t'], track['bearing']))
        if np.isnan(bearing): return
        h, w = rgb.shape[:2]
        if pano_mode:
            col = int(bearing / 360.0 * w) % w
            rgb[:, col]               = [255, 255,   0]
            rgb[:, (col - 1) % w]     = [180, 180,   0]
        else:
            az = self._boresight_az(tc)
            if az is None: return
            col = int(self.W / 2 + (bearing - az) * (self.W / max(self.fov, 1.0)))
            if 0 <= col < w:
                rgb[:, col] = [255, 255, 0]

    # ── Utility helpers ────────────────────────────────────────────────────────

    def _set_and_render(self, attr, v):
        setattr(self, attr, v); self._render(self.current_t())

    def _on_vmode_changed(self, mode):
        _VM = {"Events": 0, "Panorama": 1, "Radar": 2,
               "AT Waterfall": 3, "Polarity Div": 4, "Phase-Locked Pano": 5,
               "Space-Time Vol": 6, "IAT Surface": 7, "Dual-Cam Diff": 8,
               "Elev-Time Sweep": 9, "Freq Map": 10}
        if hasattr(self, '_novel_pages'):
            self._novel_pages.setCurrentIndex(_VM.get(mode, 0))
        self._render(self.current_t())

    def _accum_count_toggled(self, on):
        self.accum_count_mode = on
        self.accum_sl.setEnabled(not on); self.accum_sp.setEnabled(not on)
        self.accum_n_sp.setEnabled(on); self._render(self.current_t())

    def _jump_to_annotation(self, t):
        if self.dur:
            self.scrub.setValue(int(t / self.dur * 1000))

    def _add_annotation(self, label=None):
        if not self.dur: return
        if label is None:
            text, ok = QtWidgets.QInputDialog.getText(self, "Add annotation", "Label (press OK to confirm):")
            if not ok or not text.strip(): return
            label = text.strip()
        self.annotations.append({'t': self.current_t(), 'label': label})
        self.annotations.sort(key=lambda a: a['t'])
        self._ann_bar.set_dur(self.dur, self.annotations); self._ann_bar.setEnabled(True)
        self._save_annotations()

    def _del_nearest_annotation(self):
        if not self.annotations or not self.dur: return
        tc = self.current_t()
        idx = min(range(len(self.annotations)), key=lambda i: abs(self.annotations[i]['t'] - tc))
        self.annotations.pop(idx)
        self._ann_bar.set_dur(self.dur, self.annotations)
        self._save_annotations()

    def _save_annotations(self):
        if not self._ann_path: return
        try:
            with open(self._ann_path, 'w', encoding='utf-8') as f:
                json.dump(self.annotations, f, indent=2)
        except Exception:
            pass

    def _load_annotations(self, raw_path):
        self._ann_path = os.path.splitext(raw_path)[0] + '_annotations.json'
        self.annotations = []
        if os.path.isfile(self._ann_path):
            try:
                with open(self._ann_path, encoding='utf-8') as f:
                    self.annotations = json.load(f)
            except Exception:
                self.annotations = []
        if self.dur:
            self._ann_bar.set_dur(self.dur, self.annotations); self._ann_bar.setEnabled(True)

    def _toggle_fft(self, on):
        self._fft_panel.setVisible(on)
        if on: self._update_fft()

    def _update_fft(self):
        """U2: Live ROI-aware event-rate FFT spectrum.
        Approach mirrors the reference viewer: bin event times → subtract mean (DC removal) →
        Hanning window → rfft. The strong 0-Hz DC component from mean event rate is
        eliminated before the FFT so it does not dominate the spectrum.
        ROI is respected if drawn; otherwise the full sensor is used."""
        if not self._fft_panel.isVisible() or self.ev is None: return
        tc  = self.current_t()
        WIN = self.fft_win
        BIN_S = 0.0005  # 0.5 ms bins → 1 kHz Nyquist
        n_bins = max(int(WIN / BIN_S), 32)
        lo = self._ss(max(0.0, tc - WIN) * 1e6); hi = self._ss(tc * 1e6)
        if hi <= lo: return

        # Load events — apply ROI filter if set
        xs  = np.asarray(self.ev["x"][lo:hi], np.int32)
        ys  = np.asarray(self.ev["y"][lo:hi], np.int32)
        ts  = np.asarray(self.t[lo:hi], np.float64) * 1e-6
        if self.roi:
            rx0, ry0, rx1, ry1 = self.roi
            roi_m = (xs >= rx0) & (xs <= rx1) & (ys >= ry0) & (ys <= ry1)
            ts = ts[roi_m]
        lut = self._col_filter()
        if lut is not None and not self.roi:
            m = lut[np.clip(xs, 0, len(lut) - 1)]; ts = ts[m]

        if len(ts) < 8: return
        bidx = np.clip(((ts - (tc - WIN)) / BIN_S).astype(int), 0, n_bins - 1)
        rate = np.bincount(bidx, minlength=n_bins).astype(np.float32)

        # DC removal: subtract the mean before the FFT (match the reference viewer)
        rate -= rate.mean()
        # Hanning window to reduce spectral leakage
        rate *= np.hanning(n_bins).astype(np.float32)

        spec  = np.abs(np.fft.rfft(rate))
        freqs = np.fft.rfftfreq(n_bins, d=BIN_S)
        # Exclude low-frequency region (rotation ~1 Hz) — use configurable fft_flo (default 20 Hz)
        keep  = (freqs >= self.fft_flo) & (freqs <= 1000.0)
        spec_m = spec[keep]; freqs_m = freqs[keep]
        if not len(spec_m): return

        W = max(self._fft_lbl.width(), 300); H = 62
        img = np.zeros((H, W, 3), np.uint8)
        # Render spectrum bars (green)
        n_f = len(spec_m)
        xs_px = (np.arange(n_f) / n_f * W).astype(int)
        bhs   = (spec_m / max(spec_m.max(), 1e-9) * (H - 10)).astype(int)
        for xi, bh in zip(xs_px, bhs):
            if bh > 0 and xi < W:
                img[H - 10 - bh: H - 10, xi] = [0, 200, 100]
        # Baseline (axis)
        img[H - 10, :] = [60, 80, 70]
        # Frequency reference markers: drone (80/400/800 Hz) and hummingbird (200 Hz)
        ref_freqs = [(80,  [255, 80,  0],  "80"),
                     (200, [0,   200, 255], "200"),
                     (400, [255, 200, 0],  "400"),
                     (800, [255, 80,  0],  "800")]
        f_max = 1000.0
        for rf, col, _ in ref_freqs:
            px = int(rf / f_max * W)
            if 0 <= px < W:
                img[H - 10 - int(H * 0.6): H - 10, px] = col
        # ROI label strip
        roi_lbl = "  ROI" if self.roi else "  full"
        # Render
        qi = QtGui.QImage(img.data, W, H, 3 * W, QtGui.QImage.Format_RGB888)
        pm = QtGui.QPixmap.fromImage(qi).scaled(
            self._fft_lbl.size(), QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.FastTransformation)
        self._fft_lbl.setPixmap(pm)
        # Update tooltip with dominant frequency
        if spec_m.max() > 0:
            dom = float(freqs_m[spec_m.argmax()])
            self._fft_lbl.setToolTip(f"Dominant: {dom:.0f} Hz{roi_lbl}  |  DC removed  |  Hanning window")

    def _open_fft_dialog(self):
        if self.ev is None:
            QtWidgets.QMessageBox.information(self, "FFT Analyser", "Load a file first.")
            return
        dlg = FFTDialog(self, src_path=self._src_path, parent=self)
        dlg.show()

    def _show(self, rgb, sensor_mode):
        h, w = rgb.shape[:2]
        qi = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pm = QtGui.QPixmap.fromImage(qi).scaled(self.view.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        self.view.setPixmap(pm)
        if sensor_mode and pm.width():
            scale = pm.width() / w
            ox = (self.view.width() - pm.width()) // 2; oy = (self.view.height() - pm.height()) // 2
            self.view.set_geom(scale, ox, oy, self.W, self.H)

    def _render(self, tc):
        if self.ev is None: return
        mode = self.vmode.currentText()
        hi = self._ss(tc * 1e6)
        if self.accum_count_mode:
            lo = max(0, hi - self.accum_n)
        else:
            lo = self._ss((tc - self.accum) * 1e6)
        ps = self.p[lo:hi]; non = int((ps == 1).sum()); noff = int((ps == 0).sum())
        if mode in ROTATION_ONLY_MODES:           # render straight from the events when no telemetry file
            self._ensure_rotation_telemetry()
        rgb, sensor = None, False
        if   mode == "Panorama":          rgb = self._render_panorama(hi)
        elif mode == "Radar":             rgb = self._render_radar(hi)
        elif mode == "AT Waterfall":      rgb = self._render_at_waterfall(hi)
        elif mode == "Polarity Div":      rgb = self._render_polarity_div(lo, hi, np.asarray(ps)); sensor = True
        elif mode == "Phase-Locked Pano": rgb = self._render_phase_locked_pano(hi)
        elif mode == "Space-Time Vol":    rgb = self._render_st_volume(lo, hi); sensor = True
        elif mode == "IAT Surface":       rgb = self._render_iat_surface(lo, hi); sensor = True
        elif mode == "Dual-Cam Diff":     rgb = self._render_dual_diff(hi)
        elif mode == "Elev-Time Sweep":   rgb = self._render_elev_time(hi)
        elif mode == "Freq Map":          rgb = self._render_freq_map(hi); sensor = True
        if rgb is None:
            if mode in ROTATION_ONLY_MODES and self.tel is None:
                rgb = self._render_needs_rotation(mode)   # honest notice, not a silent Events fallback
            else:
                rgb = self._render_events(lo, hi, ps); sensor = True
        elif mode in ROTATION_ONLY_MODES and getattr(self.tel, "synthesized", False):
            rgb = self._mark_estimated(rgb)               # bearings are estimated — say so on the frame
        # U4: bearing overlay from last analysis run
        if self._result_track is not None:
            self._draw_result_overlay(rgb, tc, pano_mode=(mode in ("Panorama", "Phase-Locked Pano", "Dual-Cam Diff")))
        self.view.roi_enabled = sensor
        self._show(rgb, sensor)
        n_win = hi - lo
        rate_str = (f"{n_win/self.accum/1e3:.0f}k ev/s" if not self.accum_count_mode else f"{n_win} ev")
        roistr = f"  ROI {self.roi}" if self.roi else ""
        spinstr = ""
        if mode in ROTATION_ONLY_MODES and getattr(self.tel, "synthesized", False):
            spinstr = f"  |  est. spin {getattr(self, '_synth_period_est', 0.0):.3f}s (North uncal.)"
        self.tlabel.setText(f"t = {tc:6.3f} / {self.dur:.2f} s   |   window: {n_win} ev  "
                            f"{rate_str}   ON {non}  OFF {noff}{roistr}{spinstr}")
        self._fft_tick = (self._fft_tick + 1) % 4
        if self._fft_tick == 0: self._update_fft()

    def _render_to_rgb(self, tc):
        """Render the current view at time `tc` and return an RGB uint8 array.
        Safe to call from a background thread — reads memmapped event data but does not
        touch any Qt widgets. Used by VideoExportThread to render frames off-screen."""
        if self.ev is None: return None
        mode = self.vmode.currentText()
        hi = self._ss(tc * 1e6)
        lo = max(0, hi - self.accum_n) if self.accum_count_mode else self._ss((tc - self.accum) * 1e6)
        ps = self.p[lo:hi]
        if mode in ROTATION_ONLY_MODES:
            self._ensure_rotation_telemetry()             # synthesize from events when no telemetry file
        rgb = None
        if   mode == "Panorama":          rgb = self._render_panorama(hi)
        elif mode == "Radar":             rgb = self._render_radar(hi)
        elif mode == "AT Waterfall":      rgb = self._render_at_waterfall(hi)
        elif mode == "Polarity Div":      rgb = self._render_polarity_div(lo, hi, np.asarray(ps))
        elif mode == "Phase-Locked Pano": rgb = self._render_phase_locked_pano(hi)
        elif mode == "Space-Time Vol":    rgb = self._render_st_volume(lo, hi)
        elif mode == "IAT Surface":       rgb = self._render_iat_surface(lo, hi)
        elif mode == "Dual-Cam Diff":     rgb = self._render_dual_diff(hi)
        elif mode == "Elev-Time Sweep":   rgb = self._render_elev_time(hi)
        elif mode == "Freq Map":          rgb = self._render_freq_map(hi)
        elif mode == "Events":            rgb = self._render_events(lo, hi, np.asarray(ps))
        if rgb is None:
            if mode in ROTATION_ONLY_MODES and self.tel is None:
                rgb = self._render_needs_rotation(mode)   # keep capture/export in step with the on-screen view
            else:
                rgb = self._render_events(lo, hi, np.asarray(ps))
        elif mode in ROTATION_ONLY_MODES and getattr(self.tel, "synthesized", False):
            rgb = self._mark_estimated(rgb)
        if self._result_track is not None:
            self._draw_result_overlay(rgb, tc, pano_mode=(mode in ("Panorama", "Phase-Locked Pano", "Dual-Cam Diff")))
        return rgb


class FFTDialog(QtWidgets.QDialog):
    """Full-size FFT spectrum analyser with parameter tuning, annotation, and export."""

    _WIN_FUNCS = {
        "Hanning":  np.hanning,
        "Hamming":  np.hamming,
        "Blackman": np.blackman,
        "Rect":     lambda n: np.ones(n),
    }

    def __init__(self, player: "Player", src_path: str = "", parent=None):
        super().__init__(parent)
        self.player = player
        self.src_path = src_path
        self.setWindowTitle("FFT Spectrum Analyser")
        self.resize(900, 560)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window)  # non-modal, stays on top of parent

        # ── matplotlib canvas ──
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self._fig = Figure(figsize=(8, 3.5), facecolor="#05080b")
            self._ax  = self._fig.add_subplot(111)
            self._canvas = FigureCanvasQTAgg(self._fig)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "FFT Analyser",
                f"matplotlib backend unavailable: {exc}\n\nInstall matplotlib>=3.8.")
            QtCore.QTimer.singleShot(0, self.close); return

        # ── parameter controls ──
        ctrl = QtWidgets.QHBoxLayout()

        ctrl.addWidget(QtWidgets.QLabel("f min:"))
        self.flo_sp = QtWidgets.QDoubleSpinBox()
        self.flo_sp.setRange(1.0, 2000.0); self.flo_sp.setValue(player.fft_flo)
        self.flo_sp.setSuffix(" Hz"); self.flo_sp.setSingleStep(5.0)
        self.flo_sp.setToolTip("Lower frequency cutoff (Hz). Default 20 Hz excludes the ~1 Hz sensor rotation.")
        ctrl.addWidget(self.flo_sp)

        ctrl.addWidget(QtWidgets.QLabel("f max:"))
        self.fhi_sp = QtWidgets.QDoubleSpinBox()
        self.fhi_sp.setRange(10.0, 5000.0); self.fhi_sp.setValue(1000.0)
        self.fhi_sp.setSuffix(" Hz"); self.fhi_sp.setSingleStep(50.0)
        ctrl.addWidget(self.fhi_sp)

        ctrl.addWidget(QtWidgets.QLabel("window:"))
        self.win_sp = QtWidgets.QDoubleSpinBox()
        self.win_sp.setRange(0.1, 30.0); self.win_sp.setValue(player.fft_win)
        self.win_sp.setSuffix(" s"); self.win_sp.setSingleStep(0.5)
        self.win_sp.setToolTip("FFT time window duration. Longer = finer frequency resolution.")
        ctrl.addWidget(self.win_sp)

        ctrl.addWidget(QtWidgets.QLabel("taper:"))
        self.taper_cb = QtWidgets.QComboBox()
        self.taper_cb.addItems(list(self._WIN_FUNCS.keys()))
        self.taper_cb.setCurrentText("Hanning")
        ctrl.addWidget(self.taper_cb)

        ctrl.addWidget(QtWidgets.QLabel("y scale:"))
        self.yscale_cb = QtWidgets.QComboBox()
        self.yscale_cb.addItems(["Linear", "Log"])
        ctrl.addWidget(self.yscale_cb)

        self.live_chk = QtWidgets.QCheckBox("Live update")
        self.live_chk.setChecked(True)
        self.live_chk.setToolTip("Update spectrum automatically as the playhead moves.")
        ctrl.addWidget(self.live_chk)

        upd_btn = QtWidgets.QPushButton("Update")
        upd_btn.setIcon(icons.icon("sync"))
        upd_btn.setToolTip("Recompute the spectrum at the current playhead position.")
        upd_btn.clicked.connect(self._update)
        ctrl.addWidget(upd_btn)

        ctrl.addStretch(1)

        # ── save controls ──
        save_row = QtWidgets.QHBoxLayout()
        png_btn = QtWidgets.QPushButton("Save PNG")
        png_btn.setIcon(icons.icon("save"))
        png_btn.setToolTip("Save this spectrum as a PNG with embedded metadata.")
        png_btn.clicked.connect(self._save_png)
        csv_btn = QtWidgets.QPushButton("Save CSV")
        csv_btn.setIcon(icons.icon("export"))
        csv_btn.setToolTip("Export frequency / amplitude data as CSV with metadata header.")
        csv_btn.clicked.connect(self._save_csv)
        save_row.addWidget(png_btn); save_row.addWidget(csv_btn); save_row.addStretch(1)

        # ── status label ──
        self._status = QtWidgets.QLabel("—")
        self._status.setStyleSheet(_mono_qss(font_size_px=11))

        # ── layout ──
        vl = QtWidgets.QVBoxLayout(self)
        vl.addWidget(self._canvas, 1)
        vl.addLayout(ctrl)
        vl.addLayout(save_row)
        vl.addWidget(self._status)

        # ── connect signals ──
        for w in (self.flo_sp, self.fhi_sp, self.win_sp):
            w.valueChanged.connect(self._update)
        for w in (self.taper_cb, self.yscale_cb):
            w.currentTextChanged.connect(self._update)
        player.scrubbed.connect(self._on_scrub)

        # ── initial draw ──
        self._freqs = None; self._spec = None; self._meta = {}
        self._update()

    def _on_scrub(self, _):
        if self.live_chk.isChecked():
            self._update()

    def _compute_spectrum(self):
        """Return (freqs_hz, amplitude, meta_dict) or None if not enough data."""
        pl = self.player
        if pl.ev is None: return None
        tc  = pl.current_t()
        WIN = self.win_sp.value()
        BIN_S = 0.0005          # 0.5 ms → 1 kHz Nyquist
        n_bins = max(int(WIN / BIN_S), 32)

        lo = pl._ss(max(0.0, tc - WIN) * 1e6); hi = pl._ss(tc * 1e6)
        if hi <= lo: return None

        xs = np.asarray(pl.ev["x"][lo:hi], np.int32)
        ys = np.asarray(pl.ev["y"][lo:hi], np.int32)
        ts = np.asarray(pl.t[lo:hi], np.float64) * 1e-6

        roi_label = "full sensor"
        if pl.roi:
            rx0, ry0, rx1, ry1 = pl.roi
            m = (xs >= rx0) & (xs <= rx1) & (ys >= ry0) & (ys <= ry1)
            ts = ts[m]; xs = xs[m]; ys = ys[m]
            roi_label = f"ROI [{rx0},{ry0}–{rx1},{ry1}]"
        else:
            lut = pl._col_filter()
            if lut is not None:
                m = lut[np.clip(xs, 0, len(lut) - 1)]
                ts = ts[m]; xs = xs[m]; ys = ys[m]

        n_ev = len(ts)
        if n_ev < 8: return None

        bidx = np.clip(((ts - (tc - WIN)) / BIN_S).astype(int), 0, n_bins - 1)
        rate = np.bincount(bidx, minlength=n_bins).astype(np.float64)

        rate -= rate.mean()                                  # DC removal
        taper_fn = self._WIN_FUNCS[self.taper_cb.currentText()]
        rate *= taper_fn(n_bins)                             # spectral taper

        spec  = np.abs(np.fft.rfft(rate))
        freqs = np.fft.rfftfreq(n_bins, d=BIN_S)

        flo = self.flo_sp.value(); fhi = self.fhi_sp.value()
        keep = (freqs >= flo) & (freqs <= fhi)
        if not keep.any(): return None

        meta = {
            "file":       os.path.basename(self.src_path) if self.src_path else "unknown",
            "path":       self.src_path,
            "t_center_s": f"{tc:.4f}",
            "window_s":   f"{WIN:.2f}",
            "flo_hz":     f"{flo:.1f}",
            "fhi_hz":     f"{fhi:.1f}",
            "taper":      self.taper_cb.currentText(),
            "n_events":   str(n_ev),
            "sensor":     f"{pl.W}x{pl.H}",
            "view_mode":  pl.vmode.currentText(),
            "accum_s":    f"{pl.accum:.4f}",
            "roi":        roi_label,
            "freq_res_hz":f"{1.0/WIN:.3f}",
        }

        return freqs[keep], spec[keep], meta

    def _update(self, *_):
        result = self._compute_spectrum()
        ax = self._ax; ax.clear()
        ax.set_facecolor("#05080b")
        for spine in ax.spines.values(): spine.set_edgecolor("#2a4a50")
        ax.tick_params(colors="#7fe", labelsize=9)
        ax.set_xlabel("Frequency (Hz)", color="#7fe", fontsize=10)
        ax.set_ylabel("Amplitude", color="#7fe", fontsize=10)

        if result is None:
            ax.set_title("Insufficient events — scrub to an active region", color="#7fe", fontsize=10)
            self._canvas.draw_idle(); return

        freqs, spec, meta = result
        self._freqs = freqs; self._spec = spec; self._meta = meta

        # Normalise
        spec_n = spec / max(spec.max(), 1e-12)

        if self.yscale_cb.currentText() == "Log":
            ax.set_yscale("log")
            plot_spec = np.clip(spec_n, 1e-6, None)
        else:
            ax.set_yscale("linear")
            plot_spec = spec_n

        ax.fill_between(freqs, plot_spec, alpha=0.55, color="#00c878", linewidth=0)
        ax.plot(freqs, plot_spec, color="#00e890", linewidth=0.8)

        # Reference lines
        for rf, col, label in [(80, "#ff5000", "80 Hz"), (200, "#00c8ff", "200 Hz"),
                                (400, "#ffc800", "400 Hz"), (800, "#ff5000", "800 Hz")]:
            if self.flo_sp.value() <= rf <= self.fhi_sp.value():
                ax.axvline(rf, color=col, linewidth=1.0, alpha=0.7, linestyle="--")
                ax.text(rf + 2, ax.get_ylim()[1] * 0.92, label, color=col, fontsize=8, va="top")

        # Dominant frequency annotation
        dom_idx = spec_n.argmax()
        dom_f   = float(freqs[dom_idx])
        dom_a   = float(plot_spec[dom_idx])
        ax.annotate(f"  {dom_f:.0f} Hz",
                    xy=(dom_f, dom_a), xytext=(dom_f + (freqs[-1] - freqs[0]) * 0.03, dom_a * 0.85),
                    color="#fff", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8))

        ax.set_title(
            f"{meta['file']}  |  t={meta['t_center_s']}s  win={meta['window_s']}s  "
            f"ROI={meta['roi']}  |  dominant: {dom_f:.0f} Hz",
            color="#cfe9f2", fontsize=9)
        ax.set_xlim(self.flo_sp.value(), self.fhi_sp.value())

        self._status.setText(
            f"Dominant: {dom_f:.1f} Hz  |  {meta['n_events']} events  |  "
            f"res={meta['freq_res_hz']} Hz  |  {meta['taper']} window  |  DC removed")
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _build_meta_text(self):
        m = self._meta
        if not m: return ""
        lines = [
            f"File:      {m.get('file','')}",
            f"Path:      {m.get('path','')}",
            f"t_center:  {m.get('t_center_s','')} s",
            f"Window:    {m.get('window_s','')} s",
            f"Freq range:{m.get('flo_hz','')} – {m.get('fhi_hz','')} Hz",
            f"Taper:     {m.get('taper','')}",
            f"N events:  {m.get('n_events','')}",
            f"Sensor:    {m.get('sensor','')}",
            f"View mode: {m.get('view_mode','')}",
            f"Accum:     {m.get('accum_s','')} s",
            f"ROI:       {m.get('roi','')}",
            f"Freq res:  {m.get('freq_res_hz','')} Hz",
        ]
        return "\n".join(lines)

    def _save_png(self):
        if self._freqs is None:
            QtWidgets.QMessageBox.information(self, "Save PNG", "No spectrum computed yet."); return

        default_name = ""
        if self.src_path:
            base = os.path.splitext(self.src_path)[0]
            tc   = self.player.current_t()
            default_name = f"{base}_fft_{tc:.2f}s.png"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save FFT spectrum PNG", default_name, "PNG Images (*.png)")
        if not path: return

        # Add metadata annotation box to a temporary copy of the figure
        meta_text = self._build_meta_text()
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig2 = Figure(figsize=(10, 5), facecolor="#05080b")
        FigureCanvasAgg(fig2)
        ax2 = fig2.add_axes([0.08, 0.12, 0.60, 0.78])   # spectrum on left 60%
        ax2.set_facecolor("#05080b")
        for spine in ax2.spines.values(): spine.set_edgecolor("#2a4a50")
        ax2.tick_params(colors="#7fe", labelsize=9)
        ax2.set_xlabel("Frequency (Hz)", color="#7fe"); ax2.set_ylabel("Amplitude", color="#7fe")

        spec_n = self._spec / max(self._spec.max(), 1e-12)
        if self.yscale_cb.currentText() == "Log":
            ax2.set_yscale("log"); plot_spec = np.clip(spec_n, 1e-6, None)
        else:
            plot_spec = spec_n
        ax2.fill_between(self._freqs, plot_spec, alpha=0.55, color="#00c878", linewidth=0)
        ax2.plot(self._freqs, plot_spec, color="#00e890", linewidth=0.8)
        for rf, col, lbl in [(80, "#ff5000", "80 Hz"), (200, "#00c8ff", "200 Hz"),
                              (400, "#ffc800", "400 Hz"), (800, "#ff5000", "800 Hz")]:
            if self.flo_sp.value() <= rf <= self.fhi_sp.value():
                ax2.axvline(rf, color=col, linewidth=1.0, alpha=0.7, linestyle="--")
        ax2.set_xlim(self.flo_sp.value(), self.fhi_sp.value())

        dom_f = float(self._freqs[self._spec.argmax()])
        m = self._meta
        ax2.set_title(f"dominant: {dom_f:.0f} Hz  |  {m.get('file','')}", color="#cfe9f2", fontsize=9)

        # Metadata panel on right
        fig2.text(0.72, 0.92, "Metadata", color="#00e5ff", fontsize=10, fontweight="bold",
                  transform=fig2.transFigure)
        fig2.text(0.72, 0.88, meta_text, color="#cfe9f2", fontsize=8,
                  transform=fig2.transFigure, verticalalignment="top",
                  fontfamily="monospace")

        try:
            fig2.savefig(path, dpi=150, facecolor="#05080b", bbox_inches="tight")
            self._status.setText(f"Saved PNG: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save PNG", str(exc))

    def _save_csv(self):
        if self._freqs is None:
            QtWidgets.QMessageBox.information(self, "Save CSV", "No spectrum computed yet."); return

        default_name = ""
        if self.src_path:
            base = os.path.splitext(self.src_path)[0]
            tc   = self.player.current_t()
            default_name = f"{base}_fft_{tc:.2f}s.csv"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save FFT data CSV", default_name, "CSV files (*.csv)")
        if not path: return

        meta_text = self._build_meta_text()
        try:
            with open(path, "w", encoding="utf-8") as f:
                for line in meta_text.splitlines():
                    f.write(f"# {line}\n")
                f.write("#\n# frequency_hz,amplitude\n")
                spec_n = self._spec / max(self._spec.max(), 1e-12)
                for freq, amp in zip(self._freqs, spec_n):
                    f.write(f"{freq:.4f},{amp:.8f}\n")
            self._status.setText(f"Saved CSV: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save CSV", str(exc))

    def closeEvent(self, event):
        try:
            self.player.scrubbed.disconnect(self._on_scrub)
        except Exception:
            pass
        super().closeEvent(event)


# ── Dashboard ────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
# GottLUX integration — the EBS viewer as a GottLUX tab panel
# ════════════════════════════════════════════════════════════════════════════
class _PlayerClock:
    """Adapt the standalone Player's playhead + trim to the small TimeController-like interface
    GottLUX's Capture / Export dialogs drive — so they can record video, grab PNGs, and build
    figures/cubes/tables from the EBS views without the viewer adopting the shared clock."""

    def __init__(self, player):
        self._p = player

    @property
    def cursor(self):
        return self._p.current_t()

    @property
    def accum(self):
        return self._p.accum

    @property
    def fps(self):
        return self._p.play_fps

    @property
    def t0(self):
        return 0.0

    @property
    def t1(self):
        return self._p.dur

    def has_selection(self):
        r = self._p.range
        return not (abs(r.lo) < 1e-6 and abs(r.hi - 1.0) < 1e-6)

    def sel_t0(self):
        return self._p.sel_t0()

    def sel_t1(self):
        return self._p.sel_t1()

    def set_cursor(self, t):
        self._p._seek_seconds(float(t))

    def pause(self):
        if self._p.timer.isActive():
            self._p._toggle()

    def play(self):
        if not self._p.timer.isActive():
            self._p._toggle()

    def toggle(self):
        self._p._toggle()


class EBSViewer(Player):
    """The classic viewer tab: an alternate player with its own playhead.

    The render pipeline, every view mode, and the column-mode expression are the vendored EBS
    :class:`Player` verbatim; this subclass only adds the GottLUX panel protocol so the
    recording loaded by the main window flows in and GottLUX's own exporters work on these
    views. The viewer keeps its own playhead/transport (a standalone tab)."""

    def __init__(self, clock=None, filters=None, parent=None):
        # GottLUX builds panels as Panel(clock, filters); the EBS Player is standalone and runs
        # on its own playhead, so those are accepted and ignored (keeps the call site uniform).
        super().__init__()
        self._gott_clock = _PlayerClock(self)

    # ---------------------------------------------------------------- panel protocol
    def set_recording(self, rec):
        if rec is None:
            return
        # Feed the live Recording's arrays straight in (no second decode); pass its real path so
        # the Player's own telemetry-sidecar glob and filename-FOV heuristic run exactly as in the
        # classic path. If no sidecar CSV is found, fall back to the attached telemetry.
        path = getattr(rec, "source_path", None) or (getattr(rec, "name", "") or "clip")
        ev = {"x": np.asarray(rec.x), "y": np.asarray(rec.y), "p": np.asarray(rec.p),
              "t": np.asarray(rec.t), "width": int(rec.width), "height": int(rec.height),
              "n": int(len(rec)), "n_on": int(getattr(rec, "n_on", 0)),
              "fmt": getattr(rec, "fmt", "evt21")}
        self.set_data(path, ev)
        if self.tel is None and getattr(rec, "telemetry", None) is not None:
            self.tel = rec.telemetry
            self._render(self.current_t())

    def sync(self):
        if self.ev is not None:
            self._render(self.current_t())

    def _seek_seconds(self, t):
        """Move the playhead to absolute time *t* (s) and render — used by the capture clock."""
        if self.dur > 0:
            self.scrub.setValue(int(round(min(max(t, 0.0), self.dur) / self.dur * 1000)))
        else:
            self._render(0.0)

    # ---------------------------------------------------------------- export hooks
    def capture_frame(self, t, dt=None, size=None):
        """Faithful off-screen RGB of the current view at time *t* (any resolution) — the path
        GottLUX's video/PNG exporters sample. *dt* overrides accumulation; *size* resizes."""
        if self.ev is None:
            return None
        old = self.accum
        if dt:
            self.accum = float(dt)
        try:
            rgb = self._render_to_rgb(float(t))
        finally:
            self.accum = old
        if rgb is not None and size:
            try:
                from PIL import Image
                rgb = np.asarray(Image.fromarray(rgb).resize((int(size[0]), int(size[1]))))
            except Exception:
                pass
        return rgb

    def sensor_size(self):
        rgb = self._render_to_rgb(self.current_t()) if self.ev is not None else None
        if rgb is not None:
            return (int(rgb.shape[1]), int(rgb.shape[0]))
        return (int(self.W), int(self.H))

    def capture_clock(self):
        return self._gott_clock
