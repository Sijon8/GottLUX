"""
spacetime.py — the 3-D space-time event-cloud explorer (live, seekable, and measurable).

Events are plotted as a point cloud with **time as depth**, so a fluttering target appears as
a *striped column* of periodic bursts. Beyond looking, this tab lets you **measure** the cloud:

* **Temporal corridor (streaming).** Independently of the program-wide accumulation, dial the
  depth of the shown slab from **5 ms to 60 s**, or flip on **Full ∞** for the entire stream —
  a *trailing corridor* in which the newest events sit at the front plane and history recedes and
  fades into the distance, flowing continuously toward "now" as you play. Anchor it Trailing
  (stream), Forward, or Centered on the cursor.
* **Articulate.** Re-orient which axis carries time (Z / X / Y) and flip its direction, to view
  the corridor from the most legible angle.
* **Style.** Colour the cloud by **polarity**, **time** (recency) or **event density**; for the
  continuous modes choose the **colormap**, a tone **expression** (linear … log / asinh to lift
  faint structure) and a **dynamic / static** white-point; and set the background **theme** — the
  sensor-plane frame and axis markers recolour to stay legible against any canvas.
* **Interactive FFT box.** A movable 3-D box you position (in x, y and time) and *analyse*: it
  pulls the events inside it and reports the dominant frequency three ways — a binned FFT, a
  non-uniform FFT (no Nyquist ceiling), and a near-zero-compute inter-event-interval estimate —
  plus SNR, harmonic comb, and a live spectrum plot.
* **Two-point measure.** Park the box on one flutter stripe (**Set A**), move to the next
  (**Set B**), and read the implied frequency, period and apparent speed straight off the cloud.

Built on pyqtgraph's OpenGL scatter; points are subsampled to a budget so even tens of millions
of events stay interactive. Degrades to an explanatory message without OpenGL.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from gottlux.app import icons
from gottlux.app import style
from gottlux.app.legend import ColorKey
from gottlux.app.navcube import navcube_container
from gottlux.app.transport import TransportBar
from gottlux.core import frequency as fq
from gottlux.core import tonemap

try:
    import pyqtgraph.opengl as gl
    _HAVE_GL = True
except Exception:                                  # pragma: no cover
    _HAVE_GL = False

_COLOR_BY = ["polarity", "time", "density"]   # what the point colour *encodes* (the "mode")
# Colormaps for the continuous modes (time / density). Polarity is a fixed categorical ON/OFF pair.
_CMAPS = ["inferno", "viridis", "magma", "plasma", "cividis", "gray", "turbo",
          "coolwarm", "RdBu", "bwr", "seismic", "PiYG", "Spectral"]
# Background canvas themes (same palette as the Event-rate tower, so the suite reads
# consistently). The default entry is not a fixed colour: it follows the app's light/dark
# instrument theme, so the cloud sits on the same canvas as the rest of the window.
_APP_THEME = "App theme"
_THEMES = {"Charcoal": "#0e1116", "Black": "#000000", "Midnight": "#05070d",
           "Graphite": "#15171a", "Slate": "#26313f", "Steel": "#2c3e4d",
           "Navy": "#15285e", "Cobalt": "#11407e", "Deep teal": "#0d5057",
           "Forest": "#16401f", "Indigo": "#281c63", "Plum": "#3b1d4d",
           "Oxblood": "#4a1620", "Sepia": "#3a2b12", "White": "#eef1f5"}
_TIME_AXIS = ["Z (up)", "X", "Y"]
_METHODS = ["FFT (binned)", "NUFFT (non-uniform)", "ISI (low-compute)"]
_NORMS = ["none", "median", "zscore"]
_CORRIDOR_LO, _CORRIDOR_HI = 0.005, 60.0           # seconds (log slider; "Full ∞" goes deeper)
_ANCHORS = ["Trailing (stream)", "Forward", "Centered"]
_HANDLE_PICK_PX = 20.0                              # click radius to grab a box corner handle
_EVENT_PICK_PX = 16.0                               # click radius to snap a measure point to an event


def _relative_luminance(hexc):
    c = hexc.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _theme_label_color(hexc):
    """Axis-label RGBA that reads against the canvas: dark labels on a light theme, light on dark."""
    return (28, 36, 46, 255) if _relative_luminance(hexc) > 0.55 else (185, 200, 215, 255)


def _theme_bg(name):
    """The canvas colour a background preset means — the app theme's own for 'App theme'."""
    return style.BG if name == _APP_THEME else _THEMES.get(name, style.BG)


def _compute_slab(t_start, t_stop, cursor, depth_s, anchor, full):
    """Resolve the shown time slab → ``(t0, t1, newest_front)``.

    * ``Trailing`` — ``[cursor - depth, cursor]``: the stream flows toward the cursor, newest
      events at the front plane and history receding (the "infinite corridor"). ``full`` opens
      it all the way back to the recording start (a continuous, ever-deepening stream).
    * ``Forward`` — ``[cursor, cursor + depth]`` (the original behaviour), cursor at the front.
    * ``Centered`` — ``[cursor - depth/2, cursor + depth/2]``.

    ``newest_front`` is True when the newest events should sit at the front plane (trailing/full)
    so the renderer and the FFT box agree on which end of the corridor is "now".
    """
    cur = min(max(cursor, t_start), t_stop)
    if full:
        return t_start, max(cur, t_start + 1e-6), True
    if anchor.startswith("Forward"):
        t0, t1, newest_front = cur, min(t_stop, cur + depth_s), False
    elif anchor.startswith("Centered"):
        t0 = max(t_start, cur - depth_s / 2)
        t1 = min(t_stop, cur + depth_s / 2)
        newest_front = False
    else:                                          # Trailing (stream) — the default
        t0, t1, newest_front = max(t_start, cur - depth_s), cur, True
    if t1 - t0 < 1e-6:
        t1 = t0 + 1e-6
    return t0, t1, newest_front


def _project_points(world, mvp, w, h):
    """Project world ``(N,3)`` points to screen pixels with a 4×4 row-major MVP matrix.

    Returns ``(screen (N,2), w_clip (N,))``; ``w_clip <= 0`` marks points behind the camera.
    """
    world = np.asarray(world, float)
    if world.size == 0:
        return np.zeros((0, 2)), np.zeros(0)
    homo = np.empty((world.shape[0], 4))
    homo[:, :3] = world
    homo[:, 3] = 1.0
    clip = homo @ mvp.T
    wv = clip[:, 3]
    safe = np.where(np.abs(wv) < 1e-9, 1e-9, wv)
    ndc = clip[:, :3] / safe[:, None]
    sx = (ndc[:, 0] * 0.5 + 0.5) * w
    sy = (0.5 - 0.5 * ndc[:, 1]) * h
    return np.column_stack([sx, sy]), wv


def _nearest_screen(screen, w_clip, click, max_dist):
    """Index of the nearest in-front point within ``max_dist`` px of ``click`` → ``(idx, dist)``.

    ``idx`` is -1 if the closest point is farther than ``max_dist`` (or none are in front).
    """
    screen = np.asarray(screen, float)
    if screen.shape[0] == 0:
        return -1, float("inf")
    d = np.hypot(screen[:, 0] - click[0], screen[:, 1] - click[1])
    d = np.where(np.asarray(w_clip) > 0, d, np.inf)
    i = int(np.argmin(d))
    return (i, float(d[i])) if d[i] <= max_dist else (-1, float(d[i]))


def _aabb_from_opposite(o, c):
    """Axis-aligned box (data x, y, t) spanning two opposite corners → ``(x0,x1,y0,y1,t0,t1)``."""
    return (min(o[0], c[0]), max(o[0], c[0]),
            min(o[1], c[1]), max(o[1], c[1]),
            min(o[2], c[2]), max(o[2], c[2]))


def _measure_stats(points, cycles=1.0):
    """Distance/frequency readout for a polyline of picked ``(x_px, y_px, t_s)`` points.

    * **2 points** — Δx, Δy, Δr (px), Δt, implied frequency (``cycles / Δt``), apparent speed.
    * **3+ points** (recurring events) — average frequency ``(N-1)/span``, mean interval and its
      jitter (std), plus the spatial path length. Points are ordered by time first.
    """
    pts = sorted(points, key=lambda p: p[2])
    n = len(pts)
    out = {"n": n}
    if n < 2:
        return out
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    ts = np.array([p[2] for p in pts], float)
    seg = np.hypot(np.diff(xs), np.diff(ys))
    span = float(ts[-1] - ts[0])
    out["span_s"] = span
    out["path_px"] = float(seg.sum())
    if n == 2:
        dt = max(span, 1e-12)
        out.update(dx=float(xs[1] - xs[0]), dy=float(ys[1] - ys[0]), dr_px=float(seg[0]),
                   dt_s=span, freq_hz=cycles / dt, period_s=dt / max(cycles, 1e-9),
                   speed_px_s=float(seg[0]) / dt)
        return out
    intervals = np.diff(ts)
    out.update(avg_freq_hz=((n - 1) / span if span > 0 else float("inf")),
               avg_period_s=(span / (n - 1)), interval_mean_s=float(np.mean(intervals)),
               jitter_s=float(np.std(intervals)))
    return out


class _PopoutWindow(QtWidgets.QWidget):
    """A floating window for a detached panel; calls ``on_close`` so the host can re-dock it."""

    def __init__(self, on_close):
        super().__init__()
        self._on_close = on_close
        self._closing = False

    def closeEvent(self, ev):
        self._closing = True
        self._on_close()
        super().closeEvent(ev)


class SpaceTimeView(QtWidgets.QWidget):
    """Interactive, live, seekable, *measurable* 3-D (x, y, t) event cloud."""

    def __init__(self, controller, filters=None, parent=None):
        super().__init__(parent)
        self.ctl = controller
        self.filters = filters        # shared live noise-filter suite (FilterController | None)
        self.rec = None
        self._last_render = 0.0
        self._map = None              # last render's mapping (t0, t1, span, depth, …) for the box
        self._newest_front = True     # which end of the corridor is "now" (trailing stream)
        self._drag_last = None        # last mouse pos while dragging the box
        self._mode = "orbit"          # sticky interaction mode: orbit | box | measure
        self._grab = None             # active box grab: ("corner", idx) | ("move", None) | None
        self._pts_world = None        # rendered points in plot coords (N,3) for event picking
        self._pts_xyt = None          # matching (x_px, y_px, t_s) for picked events
        self._handles_world = None    # box corner handles in plot coords (8,3)
        self._handles_data = None     # box corner handles in data coords (8, [x,y,t])
        self._measure_pts = []        # CAD measure points, each (x_px, y_px, t_s), click order

        root = QtWidgets.QHBoxLayout(self)

        if not _HAVE_GL:
            root.addWidget(QtWidgets.QLabel(
                "3-D view needs PyOpenGL (pip install PyOpenGL). "
                "The rest of gottlux works without it."))
            self.view = None
            return

        # ---- left: the GL view + transport + render controls ----
        left = QtWidgets.QVBoxLayout()
        from gottlux.app.glview import GLView
        self.view = GLView() if GLView is not None else gl.GLViewWidget()
        self._static_vmax = None                      # frozen white-point for Scale=static
        self._label_rgba = _theme_label_color(style.BG)
        self.view.setBackgroundColor(style.BG)
        self.view.opts["distance"] = 420
        self.scatter = gl.GLScatterPlotItem()
        self.scatter.setGLOptions("additive")
        self.view.addItem(self.scatter)
        # the "front frame": a thin bezel outlining the sensor plane at the corridor's near edge —
        # a clean rectangular frame the event stream flows out of, replacing the old ugly filled
        # translucent plane. Colour / bezel-width / visibility are all user-controllable.
        self.frame_color = (0.36, 0.86, 0.94, 0.95)
        self.frame_width = 3
        self.front_frame = gl.GLLinePlotItem(mode="line_strip", antialias=True)
        self.view.addItem(self.front_frame)
        # axis annotations: sensor pixel extents on the frame + the time/accumulation scale along
        # the time axis (GLTextItem markers, toggled together).
        self._decor_labels = []
        for _ in range(7):
            t = gl.GLTextItem(text="", color=tuple(self._label_rgba))
            self.view.addItem(t)
            self._decor_labels.append(t)
        self.fft_box = gl.GLBoxItem(color=(247, 129, 102, 255))
        self.view.addItem(self.fft_box)
        # draggable corner handles for CAD-style resize (placed on the box in _update_box)
        self.handle_scatter = gl.GLScatterPlotItem(color=(1.0, 0.85, 0.2, 1.0), size=11)
        self.handle_scatter.setGLOptions("translucent")
        self.view.addItem(self.handle_scatter)
        # CAD measure: picked points + the connecting path
        self.measure_scatter = gl.GLScatterPlotItem(color=(0.22, 0.95, 0.85, 1.0), size=13)
        self.measure_scatter.setGLOptions("translucent")
        self.view.addItem(self.measure_scatter)
        self.measure_line = gl.GLLinePlotItem(color=(0.22, 0.95, 0.85, 0.9), width=2,
                                              antialias=True, mode="line_strip")
        self.view.addItem(self.measure_line)
        # drag the box / drop measure points directly in the scene (intercept mouse + keys)
        self.view.installEventFilter(self)
        self.view.setFocusPolicy(QtCore.Qt.StrongFocus)
        holder, self.navcube = navcube_container(self.view)
        left.addWidget(holder, 1)
        # the corridor's own Anchor control (Trailing / Forward / Centered) governs direction here,
        # so the transport's accumulation-direction toggle is hidden to avoid two competing controls.
        self.transport = TransportBar(self.ctl, host=self, show_accum_dir=False)
        left.addWidget(self.transport)
        left.addLayout(self._build_render_row())
        left.addLayout(self._build_color_row())
        left.addLayout(self._build_corridor_row())
        left.addLayout(self._build_scene_row())
        left_w = QtWidgets.QWidget(); left_w.setLayout(left)

        # ---- right: the measurement deck, in a draggable min-width splitter (was a 340 px
        # cap) so the 3-D view stays usable on a laptop and in the split view ----
        from gottlux.app.uikit import plot_with_deck
        root.addWidget(plot_with_deck(left_w, self._build_measure_panel(),
                                      min_deck=300, init_deck=340, scroll=False))

        self.maxpts.valueChanged.connect(self._render)
        self.psize.valueChanged.connect(self._render)
        self.ctl.cursorChanged.connect(self._render_throttled)
        self.ctl.accumChanged.connect(self._on_accum)
        if self.filters is not None:
            self.filters.changed.connect(self._render)
        self._on_colorby()
        self._sync_box_ranges()

    # ================================================================== build: rows
    def _build_render_row(self):
        self.maxpts = QtWidgets.QSpinBox(); self.maxpts.setRange(10_000, 2_000_000)
        self.maxpts.setSingleStep(50_000); self.maxpts.setValue(250_000)
        self.maxpts.setToolTip("Render budget: events are subsampled to this many points.")
        self.psize = QtWidgets.QDoubleSpinBox(); self.psize.setRange(1, 8); self.psize.setValue(2.0)
        self.psize.setToolTip("Point size (px).")
        self.zscale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.zscale.setRange(2, 80); self.zscale.setValue(10); self.zscale.setFixedWidth(100)
        self.zscale.setToolTip("Stretch the time axis taller — easier to read flutter stripes.")
        self.zlabel = QtWidgets.QLabel("1.0×"); self.zlabel.setMinimumWidth(34)
        self.zscale.valueChanged.connect(self._on_zscale)
        self.key = ColorKey("color")
        self.export_btn = QtWidgets.QToolButton(); self.export_btn.setText("Export")
        self.export_btn.setIcon(icons.icon("export"))
        self.export_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.export_btn.setToolTip("Save a 3-D snapshot, or export the space-time event cube.")
        self.export_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self._build_export_menu()
        row = QtWidgets.QHBoxLayout()
        for lbl, w in (("Max pts", self.maxpts), ("Size", self.psize)):
            row.addWidget(QtWidgets.QLabel(lbl)); row.addWidget(w)
        row.addWidget(QtWidgets.QLabel("Z-stretch")); row.addWidget(self.zscale); row.addWidget(self.zlabel)
        row.addWidget(self.export_btn)
        row.addSpacing(8); row.addWidget(self.key, 1)
        return row

    def _build_color_row(self):
        """Colour pipeline for the point cloud: Mode (what the colour encodes) → Color (colormap) →
        Expr (tone curve) → Scale (white-point). Mirrors the Live/Multi-clip controls; polarity is a
        fixed categorical pair, so colormap/expr/scale apply only to the continuous time/density modes."""
        self.color_by = QtWidgets.QComboBox(); self.color_by.addItems(_COLOR_BY)
        self.color_by.setToolTip("What the point colour encodes: polarity (ON/OFF), time (recency "
                                 "along the corridor), or local event density.")
        self.color_by.currentIndexChanged.connect(self._on_colorby)
        self.cmap = QtWidgets.QComboBox(); self.cmap.addItems(_CMAPS)
        self.cmap.setToolTip("Colormap for the continuous modes (time / density). The key updates to match.")
        self.cmap.currentIndexChanged.connect(self._on_cmap)
        self.expr = QtWidgets.QComboBox(); self.expr.addItems(tonemap.EXPRESSIONS)
        self.expr.setToolTip("Tone curve applied before the colormap — log/sqrt lift faint structure "
                             "(see the value distribution); linear keeps magnitude.")
        self.expr.currentIndexChanged.connect(self._on_expr)
        self.scale = QtWidgets.QComboBox(); self.scale.addItems(["dynamic", "static"])
        self.scale.setToolTip("dynamic: recompute the white-point each frame (best instantaneous "
                              "contrast). static: hold it fixed so colours stay comparable as you seek.")
        self.scale.currentIndexChanged.connect(self._on_scale)
        row = QtWidgets.QHBoxLayout()
        for lbl, w in (("Mode", self.color_by), ("Color", self.cmap),
                       ("Expr", self.expr), ("Scale", self.scale)):
            row.addWidget(QtWidgets.QLabel(lbl)); row.addWidget(w)
        row.addStretch(1)
        return row

    def _build_corridor_row(self):
        self.corridor_chk = QtWidgets.QCheckBox("Corridor")
        self.corridor_chk.setToolTip("Override the program accumulation and set the shown time "
                                     "slab depth here (5 ms … 60 s), independent of the other "
                                     "tabs. Use 'Full ∞' for the whole stream. When OFF (default), "
                                     "the shared Accum control sets the slab depth.")
        # Default OFF so the program-wide accumulation drives the slab depth (the 3-D cloud
        # responds to the Accum control like every other tab); tick it to override independently.
        self.corridor_chk.setChecked(False)
        self.corridor_chk.toggled.connect(self._render)
        self.corridor = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.corridor.setRange(0, 1000); self.corridor.setFixedWidth(140)
        self.corridor.setValue(self._corr_to_slider(0.2))
        self.corridor.setToolTip("Corridor depth (log scale, 5 ms … 60 s).")
        self.corridor.valueChanged.connect(self._on_corridor)
        self.corridor_lbl = QtWidgets.QLabel("200 ms"); self.corridor_lbl.setMinimumWidth(56)
        self.full_chk = QtWidgets.QCheckBox("Full ∞")
        self.full_chk.setToolTip("Infinite corridor: show the entire stream from the start up to "
                                 "the cursor, deepening continuously as you play.")
        self.full_chk.toggled.connect(self._on_full)
        self.anchor = QtWidgets.QComboBox(); self.anchor.addItems(_ANCHORS)
        self.anchor.setToolTip("Where the slab sits around the cursor. Trailing = a stream "
                               "flowing toward 'now' (newest at the front, history receding); "
                               "Forward looks ahead; Centered straddles the cursor.")
        self.anchor.currentIndexChanged.connect(self._render)
        self.time_axis = QtWidgets.QComboBox(); self.time_axis.addItems(_TIME_AXIS)
        self.time_axis.setToolTip("Which axis carries time (articulate the corridor).")
        self.time_axis.currentIndexChanged.connect(self._render)
        self.flip_chk = QtWidgets.QCheckBox("Flip time")
        self.flip_chk.setToolTip("Reverse the time direction along its axis.")
        self.flip_chk.toggled.connect(self._render)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.corridor_chk); row.addWidget(self.corridor)
        row.addWidget(self.corridor_lbl); row.addWidget(self.full_chk)
        row.addSpacing(8)
        row.addWidget(QtWidgets.QLabel("Anchor")); row.addWidget(self.anchor)
        row.addSpacing(8)
        row.addWidget(QtWidgets.QLabel("Time axis")); row.addWidget(self.time_axis)
        row.addWidget(self.flip_chk)
        row.addStretch(1)
        return row

    def _build_scene_row(self):
        """Scene furniture: the sensor-plane frame (bezel) and the axis markers — both optional."""
        self.frame_chk = QtWidgets.QCheckBox("Frame")
        self.frame_chk.setToolTip("Show a thin bezel framing the sensor plane at the corridor's near "
                                  "edge (replaces the old filled plane). Uncheck to hide it.")
        self.frame_chk.setChecked(True)
        self.frame_chk.toggled.connect(self._apply_decor)
        self.bezel_sp = QtWidgets.QSpinBox(); self.bezel_sp.setRange(1, 12)
        self.bezel_sp.setValue(self.frame_width); self.bezel_sp.setSuffix(" px")
        self.bezel_sp.setToolTip("Bezel width of the frame (a few pixels by default).")
        self.bezel_sp.valueChanged.connect(self._on_bezel)
        self.frame_color_btn = QtWidgets.QToolButton(); self.frame_color_btn.setFixedWidth(30)
        self.frame_color_btn.setToolTip("Frame / bezel colour — click to choose.")
        self.frame_color_btn.clicked.connect(self._pick_frame_color)
        self.markers_chk = QtWidgets.QCheckBox("Markers")
        self.markers_chk.setToolTip("Show sensor pixel extents on the frame and the time / "
                                    "accumulation scale along the time axis.")
        self.markers_chk.setChecked(True)
        self.markers_chk.toggled.connect(self._apply_decor)
        self.theme = QtWidgets.QComboBox(); self.theme.addItems([_APP_THEME] + list(_THEMES))
        self.theme.setCurrentText(_APP_THEME)
        self.theme.setToolTip("Background canvas theme; the frame/axis markers recolour to stay "
                              "legible. 'App theme' follows the window's light/dark theme.")
        self.theme.currentTextChanged.connect(self._on_theme)
        # a light/dark switch moves the 'App theme' canvas with it
        style.notifier().themeChanged.connect(self.apply_theme)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.frame_chk)
        row.addWidget(QtWidgets.QLabel("Bezel")); row.addWidget(self.bezel_sp)
        row.addWidget(QtWidgets.QLabel("Colour")); row.addWidget(self.frame_color_btn)
        row.addSpacing(12)
        row.addWidget(self.markers_chk)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("Theme")); row.addWidget(self.theme)
        row.addStretch(1)
        self._refresh_frame_color_btn()
        return row

    def _on_bezel(self, v):
        self.frame_width = int(v)
        self._apply_decor()

    def _refresh_frame_color_btn(self):
        c = QtGui.QColor.fromRgbF(*self.frame_color)
        self.frame_color_btn.setStyleSheet(
            f"background:{c.name()}; border:1px solid #888; border-radius:3px;")

    def _pick_frame_color(self):
        c = QtWidgets.QColorDialog.getColor(
            QtGui.QColor.fromRgbF(*self.frame_color), self, "Frame / bezel colour",
            QtWidgets.QColorDialog.ShowAlphaChannel)
        if c.isValid():
            self.frame_color = (c.redF(), c.greenF(), c.blueF(), c.alphaF())
            self._refresh_frame_color_btn()
            self._apply_decor()

    # ================================================================== build: measure deck
    def _build_measure_panel(self):
        """The right-hand controls, tabbed (Box · Spectrum · Measure) to cut congestion. The
        Spectrum plot can be popped out into its own window and saved."""
        panel = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(panel); outer.setContentsMargins(0, 0, 0, 0)
        self._spec_window = None         # floating window when the spectrum is popped out
        self._box_sp = None              # last computed box spectrum (for saving)
        self.measure_deck = QtWidgets.QTabWidget()
        self.measure_deck.addTab(self._box_tab(), "Box")
        self.measure_deck.addTab(self._spectrum_tab(), "Spectrum")
        self.measure_deck.addTab(self._measure_tab(), "Measure")
        outer.addWidget(self.measure_deck)
        self._set_mode("orbit")
        self._update_measure_readout()
        return panel

    def _box_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        # interaction mode (hybrid: sticky buttons + Shift/M shortcuts)
        im = QtWidgets.QGroupBox("Interaction")
        iv = QtWidgets.QVBoxLayout(im)
        self.btn_orbit = QtWidgets.QToolButton(); self.btn_orbit.setText("Orbit")
        self.btn_box = QtWidgets.QToolButton(); self.btn_box.setText("Edit box")
        self.btn_measure = QtWidgets.QToolButton(); self.btn_measure.setText("Measure")
        self.mode_group = QtWidgets.QButtonGroup(self); self.mode_group.setExclusive(True)
        hb = QtWidgets.QHBoxLayout()
        for b, mm, tip in (
                (self.btn_orbit, "orbit", "Left-drag orbits the camera, wheel zooms."),
                (self.btn_box, "box", "Left-drag moves the box; drag a yellow corner handle to "
                                      "resize; Shift+wheel scales. (Hold Shift in any mode.)"),
                (self.btn_measure, "measure", "Left-click drops a measurement point on the "
                                              "nearest event. (Or press M in any mode.)")):
            b.setCheckable(True); b.setToolTip(tip)
            b.clicked.connect(lambda _=False, m=mm: self._set_mode(m))
            self.mode_group.addButton(b); hb.addWidget(b)
        iv.addLayout(hb)
        self.mode_status = QtWidgets.QLabel(); self.mode_status.setWordWrap(True)
        self.mode_status.setObjectName("muted")
        iv.addWidget(self.mode_status)
        v.addWidget(im)

        # analysis box position (precise sliders; drag/handles edit it in-scene)
        box = QtWidgets.QGroupBox("Analysis box")
        bf = QtWidgets.QFormLayout(box)
        self.box_show = QtWidgets.QCheckBox("Show box"); self.box_show.setChecked(True)
        self.box_show.toggled.connect(self._render)
        self.bx = self._mk_slider("X centre (px)")
        self.by = self._mk_slider("Y centre (px)")
        self.bt = self._mk_slider("Time centre")
        # independent extents on all three axes — the box can be elongated along any of them
        self.bsx = self._mk_slider("X size (px)", default=180)
        self.bsy = self._mk_slider("Y size (px)", default=180)
        self.bdt = self._mk_slider("Time depth")          # mapped to seconds in _box_window
        bf.addRow(self.box_show)
        for lbl, s in (("X centre", self.bx), ("Y centre", self.by), ("Time centre", self.bt),
                       ("X size", self.bsx), ("Y size", self.bsy), ("Time depth", self.bdt)):
            bf.addRow(lbl, s)
        for s in (self.bx, self.by, self.bt, self.bsx, self.bsy, self.bdt):
            s.valueChanged.connect(self._render)
        v.addWidget(box)
        v.addStretch(1)
        return w

    def _spectrum_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        # the detachable panel: controls + plot + readout (moves into a window when popped out)
        self.spec_panel = QtWidgets.QWidget()
        af = QtWidgets.QVBoxLayout(self.spec_panel); af.setContentsMargins(0, 0, 0, 0)
        h = QtWidgets.QHBoxLayout()
        self.method = QtWidgets.QComboBox(); self.method.addItems(_METHODS)
        self.method.setToolTip("FFT: fast, binned. NUFFT: exact non-uniform transform (no Nyquist "
                               "ceiling). ISI: near-zero-compute inter-event-interval estimate.")
        self.norm = QtWidgets.QComboBox(); self.norm.addItems(_NORMS)
        self.norm.setToolTip("Spectral whitening to emphasize the peak over colored noise.")
        h.addWidget(QtWidgets.QLabel("Method")); h.addWidget(self.method, 1)
        h.addWidget(QtWidgets.QLabel("Norm")); h.addWidget(self.norm)
        af.addLayout(h)
        h2 = QtWidgets.QHBoxLayout()
        self.flo = QtWidgets.QSpinBox(); self.flo.setRange(1, 4000); self.flo.setValue(80)
        self.fhi = QtWidgets.QSpinBox(); self.fhi.setRange(2, 4000); self.fhi.setValue(800)
        self.flo.setSuffix(" Hz"); self.fhi.setSuffix(" Hz")
        h2.addWidget(QtWidgets.QLabel("Band")); h2.addWidget(self.flo); h2.addWidget(self.fhi)
        af.addLayout(h2)
        self.analyze_btn = QtWidgets.QPushButton("Analyze box")
        self.analyze_btn.setObjectName("primary")
        self.analyze_btn.clicked.connect(self._analyze_box)
        af.addWidget(self.analyze_btn)
        self.box_spec = pg.PlotWidget()
        self.box_spec.setLabel("bottom", "Hz"); self.box_spec.setLogMode(y=True)
        self.box_curve = self.box_spec.plot(pen=pg.mkPen(style.ACCENT, width=1.5))
        self.box_peak = pg.ScatterPlotItem(symbol="t", size=10, brush=style.ACCENT2)
        self.box_spec.addItem(self.box_peak)
        af.addWidget(self.box_spec, 1)
        self.box_readout = QtWidgets.QLabel("Move the box and press Analyze.")
        self.box_readout.setWordWrap(True); self.box_readout.setObjectName("muted")
        af.addWidget(self.box_readout)

        self.spec_host = QtWidgets.QVBoxLayout()      # where spec_panel lives when docked
        self.spec_host.addWidget(self.spec_panel)
        v.addLayout(self.spec_host, 1)
        self.spec_placeholder = QtWidgets.QPushButton("Spectrum is in a separate window — "
                                                      "click to re-dock")
        self.spec_placeholder.clicked.connect(self._dock_spectrum)
        self.spec_placeholder.setVisible(False)
        v.addWidget(self.spec_placeholder)

        row = QtWidgets.QHBoxLayout()
        self.popout_btn = QtWidgets.QPushButton("Pop out ⧉")
        self.popout_btn.setToolTip("Detach the spectrum into its own resizable window.")
        self.popout_btn.clicked.connect(self._popout_spectrum)
        self.savespec_btn = QtWidgets.QPushButton("Save plot…")
        self.savespec_btn.setToolTip("Save the current box spectrum as a figure (PNG/PDF).")
        self.savespec_btn.clicked.connect(self._save_box_spectrum)
        row.addWidget(self.popout_btn); row.addWidget(self.savespec_btn)
        v.addLayout(row)
        return w

    def _measure_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        mp = QtWidgets.QGroupBox("Measure (CAD)")
        mf = QtWidgets.QVBoxLayout(mp)
        hb = QtWidgets.QHBoxLayout()
        self.clear_btn = QtWidgets.QPushButton("Clear (Esc)")
        self.clear_btn.clicked.connect(self._clear_measure)
        self.undo_btn = QtWidgets.QPushButton("Remove last (⌫)")
        self.undo_btn.clicked.connect(self._remove_last_measure)
        hb.addWidget(self.clear_btn); hb.addWidget(self.undo_btn)
        mf.addLayout(hb)
        hc = QtWidgets.QHBoxLayout()
        self.cycles = QtWidgets.QDoubleSpinBox(); self.cycles.setRange(0.25, 64); self.cycles.setValue(1.0)
        self.cycles.setToolTip("For a 2-point measure: how many flutter cycles span the two "
                               "points (1 = adjacent stripes).")
        self.cycles.valueChanged.connect(self._update_measure_readout)
        hc.addWidget(QtWidgets.QLabel("Cycles (2-pt)")); hc.addWidget(self.cycles); hc.addStretch(1)
        mf.addLayout(hc)
        self.meas_readout = QtWidgets.QLabel()
        self.meas_readout.setWordWrap(True); self.meas_readout.setObjectName("muted")
        mf.addWidget(self.meas_readout)
        help2 = QtWidgets.QLabel(
            "Enter Measure mode (button or <b>M</b>), then click points on events in the cloud. "
            "<b>2 points</b> → distance, Δt, frequency and apparent speed. <b>3+ points</b> on "
            "recurring bursts → average frequency, with interval jitter and path length.")
        help2.setWordWrap(True); help2.setObjectName("muted")
        mf.addWidget(help2)
        v.addWidget(mp)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------------ pop-out / save spectrum
    def _popout_spectrum(self):
        if self._spec_window is not None:
            self._spec_window.raise_(); self._spec_window.activateWindow(); return
        self.spec_host.removeWidget(self.spec_panel)
        self._spec_window = _PopoutWindow(self._dock_spectrum)
        self._spec_window.setWindowTitle("gottlux — frequency in box")
        self._spec_window.resize(560, 380)
        lay = QtWidgets.QVBoxLayout(self._spec_window)
        self.spec_panel.setParent(self._spec_window)
        lay.addWidget(self.spec_panel)
        self.spec_panel.show()
        self._spec_window.show()
        self.spec_placeholder.setVisible(True)
        self.popout_btn.setEnabled(False)

    def _dock_spectrum(self):
        if self._spec_window is None:
            return
        w = self._spec_window
        self._spec_window = None
        w.layout().removeWidget(self.spec_panel)
        self.spec_panel.setParent(None)
        self.spec_host.addWidget(self.spec_panel)
        self.spec_panel.show()
        self.spec_placeholder.setVisible(False)
        self.popout_btn.setEnabled(True)
        if not getattr(w, "_closing", False):
            w.close()
        w.deleteLater()

    def _save_box_spectrum(self):
        sp = self._box_sp
        if sp is None or getattr(sp, "freqs", None) is None or not sp.freqs.size:
            QtWidgets.QMessageBox.information(self, "Save plot",
                                             "Run Analyze (FFT or NUFFT) first — ISI has no curve.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save box spectrum",
                                                        "box_spectrum.png", "PNG image (*.png)")
        if not path:
            return
        from gottlux.io import export
        from gottlux.viz import spectral
        self._notify(export.save_figure(spectral.spectrum_figure(sp),
                                        os.path.splitext(path)[0], close=True))

    def _mk_slider(self, tip, default=500):
        s = QtWidgets.QSlider(QtCore.Qt.Horizontal); s.setRange(0, 1000); s.setValue(default)
        s.setToolTip(tip)
        return s

    # ================================================================== corridor / accum helpers
    def _corr_to_slider(self, sec):
        f = (math.log10(min(max(sec, _CORRIDOR_LO), _CORRIDOR_HI)) - math.log10(_CORRIDOR_LO)) / \
            (math.log10(_CORRIDOR_HI) - math.log10(_CORRIDOR_LO))
        return int(round(f * 1000))

    def _slider_to_corr(self, v):
        f = v / 1000.0
        return 10 ** (math.log10(_CORRIDOR_LO) + f * (math.log10(_CORRIDOR_HI) - math.log10(_CORRIDOR_LO)))

    def _corridor_s(self):
        if self.corridor_chk.isChecked():
            return self._slider_to_corr(self.corridor.value())
        return self.ctl.accum

    def _anchor(self):
        return self.anchor.currentText()

    def _on_corridor(self, v):
        s = self._slider_to_corr(v)
        self.corridor_lbl.setText(f"{s*1e3:.0f} ms" if s < 1 else f"{s:.2f} s")
        if self.corridor_chk.isChecked() and not self.full_chk.isChecked():
            self._render()

    def _on_full(self, on):
        self.corridor.setEnabled(not on)
        self.corridor_lbl.setText("∞ (full)" if on else
                                  self._fmt_depth(self._slider_to_corr(self.corridor.value())))
        self._render()

    @staticmethod
    def _fmt_depth(s):
        return f"{s*1e3:.0f} ms" if s < 1 else f"{s:.2f} s"

    def _on_accum(self, *_):
        if not self.corridor_chk.isChecked():
            self._render()

    def _on_zscale(self, v):
        self.zlabel.setText(f"{v / 10.0:.1f}×")
        self._render()

    # ================================================================== data
    def set_recording(self, rec):
        self.rec = rec
        if self.view is None:
            return
        self._reset_scale()          # a new clip → re-derive the static white-point
        self._sync_box_ranges()
        self._render()

    def _sync_box_ranges(self):
        """Default the box to the centre of the sensor / current slab."""
        for s, val in ((self.bx, 500), (self.by, 500), (self.bt, 500)):
            blocker = QtCore.QSignalBlocker(s); s.setValue(val); del blocker

    def showEvent(self, ev):
        super().showEvent(ev)
        self._render()

    def sync(self):
        self._render(force=True)

    # ------------------------------------------------------------------ faithful capture
    def sensor_size(self):
        s = self.view.size()
        return (max(s.width(), 16), max(s.height(), 16))

    def capture_frame(self, t, dt=None, size=None):
        """Offscreen-render the 3-D event cloud at time *t* to RGB at *size* (high-res GL grab)."""
        from gottlux.app.capture import gl_to_rgb
        self.ctl.set_cursor(float(t))
        QtWidgets.QApplication.processEvents()
        w, h = size if size else self.sensor_size()
        try:
            return gl_to_rgb(self.view.renderToArray((int(w), int(h))))
        except Exception:
            return None

    # ================================================================== mapping
    def _slab(self):
        t0, t1, newest_front = _compute_slab(
            self.rec.t_start_s, self.rec.t_stop_s, self.ctl.cursor,
            self._corridor_s(), self._anchor(), self.full_chk.isChecked())
        self._newest_front = newest_front
        return t0, t1

    def _axis_map(self, xc, yc, zt):
        """Place (image-x, image-y, time-z) onto plot axes per the time-axis selector."""
        a = self.time_axis.currentText()
        if a.startswith("X"):
            return np.column_stack([zt, yc, xc])
        if a.startswith("Y"):
            return np.column_stack([xc, zt, yc])
        return np.column_stack([xc, yc, zt])

    def _render_throttled(self, *_):
        if time.perf_counter() - self._last_render >= 0.04:
            self._render()

    def _render(self, *_, force=False):
        if self.rec is None or self.view is None or (not force and not self.isVisible()):
            return
        self._last_render = time.perf_counter()
        t0, t1 = self._slab()
        win = self.rec.window(t0, t1)
        if self.filters is not None:
            win = self.filters.apply(win)
        n = win.n
        if n == 0:
            self.scatter.setData(pos=np.zeros((0, 3)))
            self._map = None
            self._pts_world = None
            self._pts_xyt = None
            self._apply_decor()
            self._update_box()
            self._draw_measure()
            return
        step = max(1, n // self.maxpts.value())
        x = np.asarray(win.x[::step]).astype(np.float32)
        y = np.asarray(win.y[::step]).astype(np.float32)
        t_s = np.asarray(win.t[::step]).astype(np.float64) * 1e-6
        p = np.asarray(win.p[::step])
        flip = self.flip_chk.isChecked()
        span_s = max(t1 - t0, 1e-6)
        zf = self.zscale.value() / 10.0
        depth = max(self.rec.width, self.rec.height) * zf
        # pos: 0 at the oldest event, 1 at the newest. For a trailing stream the newest sits at
        # the front plane (frac 0) and history recedes; Forward/Centered keep the cursor at front.
        pos = np.clip((t_s - t0) / span_s, 0.0, 1.0)
        frac = (1.0 - pos) if self._newest_front else pos
        if flip:
            frac = 1.0 - frac
        zt = frac * depth - depth / 2
        xc = x - self.rec.width / 2
        yc = y - self.rec.height / 2
        posarr = self._axis_map(xc, yc, zt)
        colors = self._colors(p, zt, x, y)
        colors[:, 3] = colors[:, 3] * (0.30 + 0.70 * pos).astype(np.float32)  # fade old events
        self.scatter.setData(pos=posarr, color=colors,
                             size=float(self.psize.value()), pxMode=True)
        # keep the rendered points (plot coords) and their (x, y, t) for event picking
        self._pts_world = np.asarray(posarr, np.float64)
        self._pts_xyt = np.column_stack([x.astype(np.float64), y.astype(np.float64), t_s])
        self._map = dict(t0=t0, t1=t1, span=span_s, depth=depth,
                         newest_front=self._newest_front, flip=flip)
        self._apply_decor()
        self._update_box()
        self._draw_measure()

    def _apply_decor(self, *_):
        """Refresh the optional scene furniture (sensor-plane frame + axis markers) from the last
        render's mapping. Cheap, so the toggles/colour/bezel call it without a full re-render."""
        if self.view is None:
            return
        m = self._map
        if self.rec is None or m is None:
            self.front_frame.setVisible(False)
            for t in self._decor_labels:
                t.setVisible(False)
            return
        self._update_front_frame(m["depth"])
        self._update_axis_labels(m)

    def _front_z(self, depth):
        """Plot-z of the near frame plane (the corridor edge the events flow out of)."""
        frac = 1.0 if self.flip_chk.isChecked() else 0.0
        return frac * depth - depth / 2

    def _time_at_zfrac(self, frac, m):
        """Inverse of the render's time→z map: the absolute time (s) at a normalized z (0..1)."""
        f = (1.0 - frac) if m["flip"] else frac
        pos = (1.0 - f) if m["newest_front"] else f
        return m["t0"] + pos * m["span"]

    def _update_front_frame(self, depth):
        """A thin rectangular bezel framing the sensor plane at the corridor's near edge."""
        W, H = self.rec.width, self.rec.height
        z = np.array([self._front_z(depth)], np.float64)
        corners = [(0, 0), (W, 0), (W, H), (0, H), (0, 0)]      # closed loop
        pts = np.array([self._axis_map(np.array([cx - W / 2.0]), np.array([cy - H / 2.0]), z)[0]
                        for cx, cy in corners], np.float32)
        self.front_frame.setData(pos=pts, color=self.frame_color,
                                 width=float(self.frame_width), mode="line_strip", antialias=True)
        self.front_frame.setVisible(self.frame_chk.isChecked())

    def _update_axis_labels(self, m):
        """Place the sensor-pixel-extent markers (on the frame) and the time/accumulation scale
        (along the time axis). All follow the current time-axis permutation."""
        vis = self.markers_chk.isChecked()
        W, H = self.rec.width, self.rec.height
        depth = m["depth"]
        zf = self._front_z(depth)

        def P(cx, cy, z):
            return self._axis_map(np.array([cx - W / 2.0]), np.array([cy - H / 2.0]),
                                  np.array([z]))[0]

        span = m["span"]
        span_txt = (f"Δt {span * 1e3:.0f} ms" if span < 1.0 else f"Δt {span:.2f} s")
        specs = [
            # sensor pixel extents, on the frame plane
            (P(0, 0, zf), "(0,0)"),
            (P(W, 0, zf), f"x {W}px"),
            (P(0, H, zf), f"y {H}px"),
            (P(W, H, zf), f"{W}×{H}"),
            # time / accumulation scale, along the time axis at the far corner
            (P(W * 1.04, H, -depth / 2), f"t {self._time_at_zfrac(0.0, m):.3f}s"),
            (P(W * 1.04, H, depth / 2), f"t {self._time_at_zfrac(1.0, m):.3f}s"),
            (P(W * 1.04, H, 0.0), span_txt),
        ]
        for t, (pos, text) in zip(self._decor_labels, specs):
            t.setData(pos=np.asarray(pos, np.float32), text=text, color=tuple(self._label_rgba))
            t.setVisible(vis)

    def _colors(self, p, z, x, y):
        mode = self.color_by.currentText()
        n = len(p)
        if mode == "polarity":
            c = np.empty((n, 4), np.float32)
            on = p == 1
            c[on] = (1.0, 0.18, 0.27, 0.85)       # ON: red-ish
            c[~on] = (0.18, 0.38, 1.0, 0.85)      # OFF: blue-ish
            return c
        # Continuous modes: build a scalar per point, tone-map it (Expr + Scale), then colormap (Color).
        if mode == "time":
            val = (z - z.min()).astype(np.float32)          # recency along the corridor
        else:                                                # density: local 2-D event count
            H, _, _ = np.histogram2d(x, y, bins=64)
            xi = np.clip((x / (x.max() + 1e-9) * 63), 0, 63).astype(int)
            yi = np.clip((y / (y.max() + 1e-9) * 63), 0, 63).astype(int)
            val = H[xi, yi].astype(np.float32)
        static = self.scale.currentText() == "static"
        disp, vmax_used = tonemap.compress(val, expr=self.expr.currentText(),
                                           vmax=self._static_vmax if static else None)
        if static and self._static_vmax is None:
            self._static_vmax = vmax_used                    # freeze the white-point on first static frame
        cmap = tonemap.colormap(self.cmap.currentText())
        c = cmap(np.asarray(disp, np.float64)).astype(np.float32)
        c[:, 3] = 0.82
        return c

    # ================================================================== FFT box
    def _box_window(self):
        """Resolve the FFT box's absolute (x0,x1,y0,y1, t0,t1) from the sliders."""
        W, H = self.rec.width, self.rec.height
        t0, t1 = self._slab()
        cx = self.bx.value() / 1000.0 * W
        cy = self.by.value() / 1000.0 * H
        ct = t0 + self.bt.value() / 1000.0 * max(t1 - t0, 1e-6)
        sx = max(4.0, self.bsx.value() / 1000.0 * W)
        sy = max(4.0, self.bsy.value() / 1000.0 * H)
        sdt = max(1e-3, self.bdt.value() / 1000.0 * max(t1 - t0, 1e-3))
        x0 = max(0, cx - sx / 2); x1 = min(W, cx + sx / 2)
        y0 = max(0, cy - sy / 2); y1 = min(H, cy + sy / 2)
        bt0 = max(self.rec.t_start_s, ct - sdt / 2)
        bt1 = min(self.rec.t_stop_s, ct + sdt / 2)
        return (x0, x1, y0, y1, bt0, bt1, cx, cy, ct)

    @staticmethod
    def _z_of_time(t_s, m):
        """Map an absolute time (s) to its plot-z using a render's stored mapping ``m``.

        The single source of truth shared by the point cloud, the front plane and the FFT box,
        so they always agree on which end of the corridor is "now".
        """
        pos = min(max((t_s - m["t0"]) / m["span"], 0.0), 1.0)
        frac = (1.0 - pos) if m["newest_front"] else pos
        if m["flip"]:
            frac = 1.0 - frac
        return frac * m["depth"] - m["depth"] / 2

    def _update_box(self):
        if self.view is None or self.rec is None:
            return
        self.fft_box.setVisible(self.box_show.isChecked())
        if not self.box_show.isChecked() or self._map is None:
            self.handle_scatter.setVisible(False)
            return
        x0, x1, y0, y1, bt0, bt1, cx, cy, ct = self._box_window()
        m = self._map
        sx = (x1 - x0); sy = (y1 - y0)
        zlo, zhi = sorted((self._z_of_time(bt0, m), self._z_of_time(bt1, m)))
        sz = max(zhi - zlo, 1.0)
        self.fft_box.setSize(sx, sy, sz)
        tr = pg.Transform3D()
        # GLBoxItem origin is a corner; place at mapped (x0-W/2, y0-H/2, zlo) honoring axis swap
        corner = self._axis_map(np.array([x0 - self.rec.width / 2]),
                                np.array([y0 - self.rec.height / 2]),
                                np.array([zlo]))[0]
        # size must follow the same axis permutation
        a = self.time_axis.currentText()
        if a.startswith("X"):
            size = (sz, sy, sx)
        elif a.startswith("Y"):
            size = (sx, sz, sy)
        else:
            size = (sx, sy, sz)
        self.fft_box.setSize(*size)
        tr.translate(float(corner[0]), float(corner[1]), float(corner[2]))
        self.fft_box.setTransform(tr)
        self._update_handles(x0, x1, y0, y1, bt0, bt1, m)

    # ------------------------------------------------------------------ in-scene interaction
    def eventFilter(self, obj, ev):
        if obj is not self.view or self.view is None:
            return super().eventFilter(obj, ev)
        et = ev.type()
        if et == QtCore.QEvent.KeyPress:
            return self._handle_key(ev)
        if et == QtCore.QEvent.Wheel:
            shift = bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
            if (shift or self._mode == "box") and self._map is not None:
                self._scale_box(ev.angleDelta().y())
                return True
            return super().eventFilter(obj, ev)            # let the camera zoom
        if et == QtCore.QEvent.MouseButtonPress and ev.button() == QtCore.Qt.LeftButton:
            shift = bool(ev.modifiers() & QtCore.Qt.ShiftModifier)
            mode = "box" if shift else self._mode
            if mode == "measure":
                self._add_measure_point(ev.position())
                return True
            if mode == "box" and self._map is not None:
                idx, _ = self._pick_handle(ev.position())
                self._grab = ("corner", idx) if idx >= 0 else ("move", None)
                self._drag_last = ev.position()
                return True
            return super().eventFilter(obj, ev)            # orbit
        if et == QtCore.QEvent.MouseMove and self._grab is not None and self._drag_last is not None:
            if self._grab[0] == "corner":
                self._resize_from_corner(self._grab[1], ev.position())
            else:
                self._drag_box(ev.position())
            self._drag_last = ev.position()
            return True
        if et == QtCore.QEvent.MouseButtonRelease and ev.button() == QtCore.Qt.LeftButton:
            if self._grab is not None:
                self._grab = None
                self._drag_last = None
                return True
        return super().eventFilter(obj, ev)

    def _handle_key(self, ev):
        k = ev.key()
        if k == QtCore.Qt.Key_M:
            self._set_mode("orbit" if self._mode == "measure" else "measure")
            return True
        if k == QtCore.Qt.Key_Escape:
            self._clear_measure()
            return True
        if k in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
            self._remove_last_measure()
            return True
        return super().eventFilter(self.view, ev)

    def _camera_basis(self):
        """Camera right/up world vectors and the world-units-per-screen-pixel at the centre."""
        o = self.view.opts
        az = math.radians(o["azimuth"]); el = math.radians(o["elevation"])
        cam = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        forward = -cam
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        nr = np.linalg.norm(right)
        right = np.array([1.0, 0.0, 0.0]) if nr < 1e-6 else right / nr
        up = np.cross(right, forward); up = up / (np.linalg.norm(up) or 1.0)
        dist = o["distance"]; fov = o.get("fov", 60)
        wpp = (2 * dist * math.tan(math.radians(fov) / 2)) / max(self.view.height(), 1)
        return right, up, wpp

    def _inv_axis(self, w):
        """Invert :meth:`_axis_map`: a world (plot) delta → (x, y, time) deltas."""
        a = self.time_axis.currentText()
        pX, pY, pZ = float(w[0]), float(w[1]), float(w[2])
        if a.startswith("X"):
            return pZ, pY, pX
        if a.startswith("Y"):
            return pX, pZ, pY
        return pX, pY, pZ

    def _drag_box(self, pos):
        last = self._drag_last
        dx = pos.x() - last.x(); dy = pos.y() - last.y()
        right, up, wpp = self._camera_basis()
        world = right * (dx * wpp) - up * (dy * wpp)        # screen y is downward
        dxc, dyc, dzt = self._inv_axis(world)
        W, H = self.rec.width, self.rec.height
        m = self._map
        # invert z→time: dz/dpos = depth · (newest_front ? -1 : +1) · (flip ? -1 : +1)
        s = (-1.0 if m["newest_front"] else 1.0) * (-1.0 if m["flip"] else 1.0)
        dpos = dzt / (max(m["depth"], 1e-6) * s)
        new = {self.bx: self.bx.value() + dxc / W * 1000,
               self.by: self.by.value() + dyc / H * 1000,
               self.bt: self.bt.value() + dpos * 1000}
        for s, val in new.items():
            blk = QtCore.QSignalBlocker(s)
            s.setValue(int(np.clip(val, 0, 1000)))
            del blk
        self._render()

    # ------------------------------------------------------------------ modes
    def _set_mode(self, mode):
        self._mode = mode
        for b, mm in ((self.btn_orbit, "orbit"), (self.btn_box, "box"),
                      (self.btn_measure, "measure")):
            blk = QtCore.QSignalBlocker(b); b.setChecked(mm == mode); del blk
        hints = {
            "orbit": "ORBIT — left-drag rotates · wheel zooms.  Hold Shift to edit the box; "
                     "press M to measure.",
            "box": "EDIT BOX — left-drag moves the box · drag a yellow corner to resize · "
                   "Shift+wheel scales.",
            "measure": "MEASURE — left-click drops a point on the nearest event · Esc clears · "
                       "⌫ removes last.",
        }
        self.mode_status.setText(hints.get(mode, ""))
        self._update_handle_visibility()

    # ------------------------------------------------------------------ corner handles
    def _update_handles(self, x0, x1, y0, y1, bt0, bt1, m):
        """Place the 8 box-corner handles; index = ix*4 + iy*2 + it (opposite corner = 7-idx)."""
        W, H = self.rec.width, self.rec.height
        xs, ys, ts = (x0, x1), (y0, y1), (bt0, bt1)
        world = np.empty((8, 3)); data = np.empty((8, 3)); i = 0
        for ix in (0, 1):
            for iy in (0, 1):
                for it in (0, 1):
                    xx, yy, tt = xs[ix], ys[iy], ts[it]
                    world[i] = self._axis_map(np.array([xx - W / 2]), np.array([yy - H / 2]),
                                              np.array([self._z_of_time(tt, m)]))[0]
                    data[i] = (xx, yy, tt)
                    i += 1
        self._handles_world = world
        self._handles_data = data
        self.handle_scatter.setData(pos=world.astype(np.float32))
        self._update_handle_visibility()

    def _update_handle_visibility(self):
        vis = (self.view is not None and self.box_show.isChecked() and self._mode == "box"
               and self._handles_world is not None)
        self.handle_scatter.setVisible(bool(vis))

    # ------------------------------------------------------------------ projection / picking
    def _mvp_np(self):
        """The current model-view-projection as a 4×4 row-major numpy matrix.

        ``projectionMatrix`` takes a (region, viewport) pair in current pyqtgraph (each an
        ``(x, y, w, h)`` tuple; region == viewport = the full view for a plain projection),
        with fallbacks for older single-arg / zero-arg signatures.
        """
        v = self.view
        vp = list(v.getViewport())
        vp[2] = max(vp[2], 1); vp[3] = max(vp[3], 1)     # avoid divide-by-zero before first layout
        vp = tuple(vp)
        try:
            proj = v.projectionMatrix(vp, vp)
        except TypeError:
            try:
                proj = v.projectionMatrix(vp)
            except TypeError:
                proj = v.projectionMatrix()
        mvp = proj * v.viewMatrix()
        return np.array(mvp.data(), float).reshape(4, 4).T

    def _pick_handle(self, pos):
        if self._handles_world is None:
            return -1, float("inf")
        screen, wv = _project_points(self._handles_world, self._mvp_np(),
                                     self.view.width(), self.view.height())
        return _nearest_screen(screen, wv, (pos.x(), pos.y()), _HANDLE_PICK_PX)

    def _pick_event(self, pos):
        if self._pts_world is None or self._pts_world.shape[0] == 0:
            return None
        screen, wv = _project_points(self._pts_world, self._mvp_np(),
                                     self.view.width(), self.view.height())
        idx, _ = _nearest_screen(screen, wv, (pos.x(), pos.y()), _EVENT_PICK_PX)
        if idx < 0:
            return None
        x, y, t_s = self._pts_xyt[idx]
        return float(x), float(y), float(t_s)

    # ------------------------------------------------------------------ resize / scale
    def _resize_from_corner(self, idx, pos):
        if idx < 0 or self._handles_data is None or self._map is None:
            return
        last = self._drag_last
        dx = pos.x() - last.x(); dy = pos.y() - last.y()
        right, up, wpp = self._camera_basis()
        world = right * (dx * wpp) - up * (dy * wpp)
        ddx, ddy, ddz = self._inv_axis(world)
        m = self._map
        s = (-1.0 if m["newest_front"] else 1.0) * (-1.0 if m["flip"] else 1.0)
        dt = (ddz / (max(m["depth"], 1e-6) * s)) * m["span"]
        cdat = self._handles_data[idx]
        odat = self._handles_data[7 - idx]                 # opposite corner stays fixed
        nc = (cdat[0] + ddx, cdat[1] + ddy, cdat[2] + dt)
        x0, x1, y0, y1, t0b, t1b = _aabb_from_opposite(odat, nc)
        self._set_box_data(x0, x1, y0, y1, t0b, t1b)
        self._render()

    def _set_box_data(self, x0, x1, y0, y1, t0b, t1b):
        """Write a data-space box (x,y px; t s) back to the centre/size sliders."""
        W, H = self.rec.width, self.rec.height
        t0, t1 = self._slab()
        span = max(t1 - t0, 1e-6)
        vals = {self.bx: 0.5 * (x0 + x1) / W * 1000, self.by: 0.5 * (y0 + y1) / H * 1000,
                self.bt: (0.5 * (t0b + t1b) - t0) / span * 1000,
                self.bsx: max(0.0, x1 - x0) / W * 1000, self.bsy: max(0.0, y1 - y0) / H * 1000,
                self.bdt: max(0.0, t1b - t0b) / span * 1000}
        for s, val in vals.items():
            blk = QtCore.QSignalBlocker(s)
            s.setValue(int(np.clip(val, 0, 1000)))
            del blk

    def _scale_box(self, delta):
        factor = 1.1 if delta > 0 else (1.0 / 1.1)
        for s in (self.bsx, self.bsy, self.bdt):
            blk = QtCore.QSignalBlocker(s)
            s.setValue(int(np.clip(round(s.value() * factor), 1, 1000)))
            del blk
        self._render()

    # ------------------------------------------------------------------ CAD measure
    def _add_measure_point(self, pos):
        pt = self._pick_event(pos)
        if pt is None:
            self.mode_status.setText("MEASURE — no event near the click; aim at a point in the "
                                     "cloud.")
            return
        self._measure_pts.append(pt)
        self._draw_measure()
        self._update_measure_readout()

    def _clear_measure(self):
        self._measure_pts = []
        self._draw_measure()
        self._update_measure_readout()

    def _remove_last_measure(self):
        if self._measure_pts:
            self._measure_pts.pop()
            self._draw_measure()
            self._update_measure_readout()

    def _draw_measure(self):
        if self.view is None:
            return
        if not self._measure_pts or self._map is None:
            self.measure_scatter.setData(pos=np.zeros((0, 3)))
            self.measure_line.setData(pos=np.zeros((0, 3)))
            return
        W, H = self.rec.width, self.rec.height
        m = self._map
        pos = np.array([self._axis_map(np.array([px - W / 2]), np.array([py - H / 2]),
                                       np.array([self._z_of_time(t, m)]))[0]
                        for px, py, t in self._measure_pts], np.float32)
        self.measure_scatter.setData(pos=pos)
        self.measure_line.setData(pos=pos if len(pos) >= 2 else np.zeros((0, 3)))

    def _update_measure_readout(self):
        st = _measure_stats(self._measure_pts, cycles=float(self.cycles.value()))
        n = st["n"]
        if n == 0:
            self.meas_readout.setText("No points. In Measure mode (button or M), click events in "
                                      "the cloud.")
        elif n == 1:
            p = self._measure_pts[0]
            self.meas_readout.setText(f"Point 1 · x {p[0]:.0f}, y {p[1]:.0f}, t {p[2]*1e3:.2f} ms"
                                      " · click another point.")
        elif n == 2:
            self.meas_readout.setText(
                f"<b>2 points</b> · Δxy <b>{st['dr_px']:.0f} px</b> (Δx {st['dx']:.0f}, "
                f"Δy {st['dy']:.0f}) · Δt <b>{st['dt_s']*1e3:.2f} ms</b><br>implied "
                f"<b>{st['freq_hz']:.1f} Hz</b> (period {st['period_s']*1e3:.2f} ms) · apparent "
                f"speed {st['speed_px_s']:.0f} px/s")
        else:
            self.meas_readout.setText(
                f"<b>{n} points</b> over {st['span_s']*1e3:.1f} ms · average "
                f"<b>{st['avg_freq_hz']:.1f} Hz</b> (period {st['avg_period_s']*1e3:.2f} ms)<br>"
                f"interval {st['interval_mean_s']*1e3:.2f} ± {st['jitter_s']*1e3:.2f} ms · "
                f"path {st['path_px']:.0f} px")

    def _box_events(self):
        x0, x1, y0, y1, bt0, bt1, cx, cy, ct = self._box_window()
        win = self.rec.window(bt0, bt1, roi=(int(x0), int(y0), int(x1), int(y1)))
        return win, (cx, cy, ct)

    def _analyze_box(self):
        if self.rec is None:
            return
        win, _ = self._box_events()
        lo, hi = float(self.flo.value()), float(self.fhi.value())
        if hi <= lo:
            hi = lo + 1
        method = self.method.currentText()
        norm = self.norm.currentText()
        if win.n < 8:
            self.box_readout.setText(f"Only {win.n} events in box — enlarge it.")
            self.box_curve.setData([], []); self.box_peak.setData([], [])
            self._box_sp = None
            return
        if method.startswith("ISI"):
            f, strength = fq.isi_frequency(win.t, fmin=lo, fmax=hi)
            self.box_curve.setData([], []); self.box_peak.setData([], [])
            self._box_sp = None                          # ISI has no curve to save
            self.box_readout.setText(
                f"<b>ISI estimate</b> · {win.n:,} events<br>dominant ≈ "
                f"<b>{f:.0f} Hz</b> · concentration {strength:.2f} "
                f"(higher = more periodic)")
            return
        fs = max(2.2 * hi, 2000.0)
        if method.startswith("NUFFT"):
            sp = fq.nufft_spectrum(win.t, fmin=lo, fmax=hi, normalize=norm)
        else:
            sp = fq.region_spectrum(win.t, fs=fs, fmin=lo, fmax=hi, normalize=norm)
        self._box_sp = sp                                # remember for "Save plot…"
        if sp.freqs.size:
            self.box_curve.setData(sp.freqs, np.maximum(sp.power, 1e-12))
            self.box_spec.setXRange(0, min(hi * 1.3, sp.freqs[-1]))
        if np.isfinite(sp.peak_freq):
            self.box_peak.setData([sp.peak_freq], [max(sp.peak_power, 1e-12)])
            self.box_readout.setText(
                f"<b>{method.split()[0]}</b> · {win.n:,} events<br>peak <b>{sp.peak_freq:.0f} Hz</b>"
                f" · SNR <b>{sp.snr:.1f}</b> · harmonic {sp.harmonic_score:.2f}")
        else:
            self.box_peak.setData([], [])
            self.box_readout.setText(f"No in-band peak · {win.n:,} events.")
        self._append_ladder_readout(win, lo, hi)

    def _append_ladder_readout(self, win, f_lo, f_hi):
        """Live rotor-ladder search on the box's (x, t): the spinning-sensor stair-step that is a
        drone telltale. The sweep rate is estimated from the box's event drift (or telemetry)."""
        try:
            from gottlux.rotation.rotor_ladder import ladder_signature
            r = ladder_signature(np.asarray(win.x, float), np.asarray(win.t_s, float),
                                 f_lo=f_lo, f_hi=f_hi, min_events=120)
            if r.detected:
                msg = (f"<span style='color:{style.ACCENT}'><b>rotor-ladder: DRONE</b> — f≈"
                       f"{r.blade_hz:.0f} Hz · Δx {r.step_px:g} px · sweep {r.drift_px_s:.0f} px/s "
                       f"· comb {r.comb_strength:.2f}</span>")
            elif r.step_px:
                msg = (f"<span style='color:{style.MUTED}'>rotor-ladder: comb "
                       f"{r.comb_strength:.2f} (below threshold — not a clear rotor)</span>")
            else:
                msg = ""
            if msg:
                self.box_readout.setText(self.box_readout.text() + "<br>" + msg)
        except Exception:
            pass

    # ================================================================== export
    def _build_export_menu(self):
        m = QtWidgets.QMenu(self)
        m.addAction("Save 3-D snapshot (PNG)…", self._save_snapshot)
        m.addAction("Export space-time event cube (x, y, t)…", self._export_cube)
        m.addSeparator()
        a = m.addAction("Export rotor-ladder study (scenes + plots + LaTeX)…",
                        self._export_ladder_study)
        a.setToolTip("Full-resolution scene renders from several angles + the rotor-ladder and "
                     "spectrum plots + a compilable LaTeX report. Put the analysis box over the "
                     "swept drone first (tick 'Show box').")
        self.export_btn.setMenu(m)

    def _save_snapshot(self):
        if self.view is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save 3-D snapshot", "spacetime.png", "PNG image (*.png)")
        if not path:
            return
        from gottlux.app.exporting import save_gl_snapshot
        self._notify(save_gl_snapshot(self.view, path))

    def _export_cube(self):
        if self.rec is None:
            return
        base, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export space-time event cube", "spacetime", "Data block (*.npz *.h5)")
        if not base:
            return
        from gottlux.app.exporting import save_event_cube
        t0, t1 = self._slab()
        self._notify(save_event_cube(os.path.splitext(base)[0], self.rec, t0, t1, nt=64),
                     extra=f"window [{t0:.3f}, {t1:.3f}] s, 64 time slices")

    #: Camera angles rendered for the rotor-ladder study (label, elevation°, azimuth°) —
    #: four quarter-turn isometrics that show the swept ladder from around, plus the orthos.
    _STUDY_ANGLES = (("iso", 30, 45), ("iso-90", 30, 135), ("iso-180", 30, 225),
                     ("iso-270", 30, 315), ("top", 90, -90), ("front", 0, 90), ("side", 0, 0))
    _STUDY_RES = (1920, 1440)        # full-resolution offscreen render size per scene

    def _export_ladder_study(self):
        """Export the rotor-ladder study: full-res scene renders from several angles, the
        rotor-ladder + spectrum plots, a measurements table, and a labelled LaTeX report."""
        if self.view is None or self.rec is None:
            QtWidgets.QMessageBox.information(self, "Rotor-ladder study", "Load a recording first.")
            return
        parent = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Export rotor-ladder study to folder")
        if not parent:
            return
        from gottlux.app.capture import gl_to_rgb
        from gottlux.io.paths import open_in_file_browser, unique_export_dir
        from gottlux.rotation import ladder_report
        from gottlux.rotation.rotor_ladder import ladder_signature
        name = self.rec.name or "recording"
        out_dir = unique_export_dir(parent, name, "ladder")

        saved_cam = dict(self.view.opts)             # restore the camera afterwards
        scenes = {}
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for label, el, az in self._STUDY_ANGLES:
                self.view.setCameraPosition(elevation=float(el), azimuth=float(az))
                QtWidgets.QApplication.processEvents()
                try:
                    rgb = gl_to_rgb(self.view.renderToArray(self._STUDY_RES))
                except Exception:
                    rgb = None
                if rgb is not None and rgb.size:
                    scenes[label] = rgb
            self.view.setCameraPosition(elevation=saved_cam.get("elevation", 30),
                                        azimuth=saved_cam.get("azimuth", 45),
                                        distance=saved_cam.get("distance", 420))
            QtWidgets.QApplication.processEvents()

            # the analysis box drives the ladder measurement + the spectrum
            win, _ = self._box_events()
            lo, hi = float(self.flo.value()), float(self.fhi.value())
            if hi <= lo:
                hi = lo + 1
            x = np.asarray(win.x, float); t = np.asarray(win.t_s, float)
            result = (ladder_signature(x, t, f_lo=lo, f_hi=hi, min_events=120)
                      if win.n else None)
            spectrum = self._box_sp                  # reuse the last "Analyze box" spectrum…
            if spectrum is None and win.n >= 8:      # …or compute one now
                try:
                    spectrum = fq.region_spectrum(win.t, fs=max(2.2 * hi, 2000.0),
                                                  fmin=lo, fmax=hi)
                except Exception:
                    spectrum = None
            meta = {"recording": name, "sensor_px": f"{self.rec.width}x{self.rec.height}",
                    "window_s": ([round(float(t.min()), 4), round(float(t.max()), 4)]
                                 if win.n else None),
                    "box_events": int(win.n)}
            written = ladder_report.save_ladder_study(
                out_dir, scenes=scenes, x=(x if win.n else None), t=(t if win.n else None),
                result=result, spectrum=spectrum, band=(lo, hi), meta=meta,
                title=f"Rotor ladder — {name}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        open_in_file_browser(out_dir)
        self._notify(written,
                     extra=f"{len(scenes)} scene angles + plots + LaTeX "
                           f"(compile rotor-ladder-report.tex with pdflatex)")

    def _notify(self, written, extra=""):
        if written:
            from gottlux.io.paths import open_in_file_browser
            open_in_file_browser(os.path.dirname(os.path.abspath(written[0])))   # reveal on export
            msg = "Saved:\n" + "\n".join(os.path.basename(p) for p in written)
            if extra:
                msg += f"\n\n{extra}"
            QtWidgets.QMessageBox.information(self, "Export", msg)
        else:
            QtWidgets.QMessageBox.warning(self, "Export", "Nothing was written.")

    def _on_colorby(self, *_):
        mode = self.color_by.currentText()
        continuous = mode != "polarity"
        # colormap/expr/scale only mean something for the continuous (time/density) modes
        for w in (self.cmap, self.expr, self.scale):
            w.setEnabled(continuous)
        self._reset_scale()
        self._refresh_key()
        self._render()

    def _refresh_key(self):
        """Update the legend strip to reflect the current mode + colormap."""
        mode = self.color_by.currentText()
        if mode == "polarity":
            self.key.set_discrete([("#ff2e45", "ON"), ("#2e61ff", "OFF")], title="polarity")
            return
        lo, hi = ("early", "late") if mode == "time" else ("sparse", "dense")
        self.key.set_gradient(self.cmap.currentText(), lo, hi, title=mode)

    def _on_cmap(self, *_):
        self._refresh_key()
        self._render()

    def _on_expr(self, *_):
        self._reset_scale()
        self._render()

    def _on_scale(self, *_):
        self._reset_scale()      # re-freeze the white-point from the next frame when switching to static
        self._render()

    def _reset_scale(self):
        self._static_vmax = None

    def _on_theme(self, name):
        if self.view is None:
            return
        hexc = _theme_bg(name)
        self.view.setBackgroundColor(hexc)
        self._label_rgba = _theme_label_color(hexc)
        self._apply_decor()      # re-applies the axis markers in the legible colour
        self.view.update()

    def apply_theme(self, *_):
        """Re-resolve the background preset after an app light/dark switch (the
        ``themeChanged`` slot): only 'App theme' moves, a chosen colour stays put."""
        self._on_theme(self.theme.currentText())
