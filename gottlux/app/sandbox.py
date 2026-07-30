"""
sandbox.py — the low-level data-manipulation lab ("build it from the ground up").

The other tabs are opinionated instruments; this one is the opposite — a bench where you
work directly with the raw event arrays and compose primitive operations yourself, to
develop new representations and detection ideas before they become a detector. It has two
faces, organised into a small tabbed deck on the right:

* **Build** — select a slice (time window at the cursor, an ROI box, a polarity) and compose
  an op-chain from the core primitives (hot-pixel removal, refractory, static-background
  suppression, subsampling), watching how many events each stage keeps; plus a raw
  ``(x, y, p, t)`` inspector — a real look at the data.
* **Track** — the live algorithm lab: write a ``track(ev, state)`` function and it runs on
  **every frame as you seek/play**, on the post-op-chain events, overlaying its detections
  (boxes + centroids + fading trails) directly on the frame. ``state`` persists across frames
  so you can build a real tracker; the project's own blob clusterer and ``MultiTracker`` are
  injected so a working tracker is a few lines. Preset algorithms are one click away.
* **Performance** — measures the algorithm as it runs: per-frame compute time, max sustainable
  FPS, event throughput, active-track count and trail length, and a real-time verdict against
  the current playback-FPS budget, with a rolling history plot.
* **Analyze** — the selection's temporal spectrum (FFT / NUFFT / ISI) and a one-click export
  of exactly this selection (event table + NPZ sub-stream).

Everything here is the same vectorized NumPy the detectors use, exposed without ceremony —
so the path from "I wonder if…" to a saved, reproducible artifact is as short as possible.
"""
from __future__ import annotations

import inspect
import math
import os
import time
import traceback
from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from gottlux.app import style
from gottlux.app.transport import TransportBar
from gottlux.core import frequency as fq
from gottlux.core import tonemap
from gottlux.core.accumulate import accumulate_frame
from gottlux.core.detect import cluster_frame
from gottlux.core.filters import hot_pixel_mask, refractory_filter
from gottlux.detectors.tracking import MultiTracker
from gottlux.io.recording import EventWindow

_POL = ["all", "ON only", "OFF only"]
_METHODS = ["FFT (binned)", "NUFFT (non-uniform)", "ISI (low-compute)"]

_DEFAULT_BOX = 14.0       # px box drawn around a detection that gives no bbox
_TRAIL_LEN = 32           # frames of centroid history kept per track for the fading trail
_PERF_HIST = 240          # frames of compute-time history kept for the performance plot

# --------------------------------------------------------------------------- algorithm presets
# Each preset is a self-contained `track(ev, state)`. `ev` exposes the current frame's events
# and helpers; `state` is a dict that survives across frames (your tracker's memory). Return a
# list of detection dicts to track, a boolean keep-mask to filter, a dict(x,y,p,t) to replace,
# or None for no change. Injected names: np, ndimage, MultiTracker, cluster_frame, math.
_PRESET_NN_TRACKER = '''\
def track(ev, state):
    """Greedy nearest-neighbour tracker over per-frame blobs.

    `state` persists between frames, so we keep ONE MultiTracker in it and feed it this
    frame's blob centroids; it associates them into stable, id'd tracks (velocity-predicted,
    with coasting). We return only the tracks confirmed this frame — the sandbox draws each
    with its own colour and a fading trail, and the Performance tab times the whole thing.
    """
    if "trk" not in state:
        state["trk"] = MultiTracker(max_match_dist=40, max_missed=6, max_tracks=8)
    trk = state["trk"]

    # per-frame detection: connected-component blobs on the current events
    cands = [dict(cx=b["cx"], cy=b["cy"], bbox=b["bbox"]) for b in ev.blobs(min_pixels=40)]
    trk.update(ev.t0, cands)                       # advance the tracker one step

    out = []
    for tid, rec in trk.tracks(min_len=1).items():
        last = rec[-1]
        if last["t"] != ev.t0:                     # skip coasting tracks (no hit this frame)
            continue
        x0, y0, x1, y1 = last["bbox"]
        out.append(dict(id=tid, cx=last["cx"], cy=last["cy"],
                        bbox=(x0, y0, x1, y1), label=f"#{tid}"))
    return out
'''

_PRESET_BLOBS = '''\
def track(ev, state):
    """Per-frame blob detector — no memory, just connected components each frame.

    `ev.blobs()` rasterizes the current events, morphologically closes them, labels
    connected components and gates by area (the same primitive the detectors use).
    Returns one detection per blob; ids are assigned by position so trails may flicker —
    compare this with the NN-tracker preset to see what association buys you.
    """
    return ev.blobs(min_pixels=40, dilation=2, erode=1)
'''

_PRESET_CENTROID = '''\
def track(ev, state):
    """Single-target smoothed centroid — the simplest tracker.

    Take the strongest blob, exponentially smooth its centroid using the position kept in
    `state`, and follow it. Good for one dominant target on a quiet background.
    """
    blobs = ev.blobs(min_pixels=30, dilation=2)
    if not blobs:
        return []
    b = max(blobs, key=lambda d: d["area"])
    cx, cy = b["cx"], b["cy"]
    px, py = state.get("pos", (cx, cy))
    a = 0.5                                          # smoothing (0 = raw, →1 = sticky)
    cx, cy = a * px + (1 - a) * cx, a * py + (1 - a) * cy
    state["pos"] = (cx, cy)
    return [dict(id=0, cx=cx, cy=cy, bbox=b["bbox"], label="target")]
'''

_PRESET_FILTER = '''\
def track(ev, state):
    """Not a tracker — a live FILTER, to show the other return type.

    Return a boolean keep-mask (length ev.n) and the sandbox re-renders the frame from just
    those events. Here: ON events in the right half of the sensor.
    """
    np = ev.np
    return (ev.p == 1) & (ev.x > ev.W // 2)
'''

_PRESET_CLASSIFY = '''\
def track(ev, state):
    """Tracker that also CLASSIFIES and emits a custom efficacy metric.

    Builds on the NN tracker, then labels each track by apparent size (a stand-in for a real
    classifier) with a confidence `score`, and returns `(detections, metrics)` so the extra
    numbers show up on the Efficacy tab. Tune the size thresholds and watch the labels settle.
    """
    np = ev.np
    if "trk" not in state:
        state["trk"] = MultiTracker(max_match_dist=40, max_missed=6, max_tracks=8)
    trk = state["trk"]
    blobs = ev.blobs(min_pixels=40)
    trk.update(ev.t0, [dict(cx=b["cx"], cy=b["cy"], bbox=b["bbox"]) for b in blobs])

    out = []
    for tid, rec in trk.tracks(min_len=1).items():
        last = rec[-1]
        if last["t"] != ev.t0:
            continue
        x0, y0, x1, y1 = last["bbox"]
        diag = float(np.hypot(x1 - x0, y1 - y0))
        label = "drone" if diag > 28 else ("insect" if diag > 14 else "speck")
        score = float(np.clip(diag / 40.0, 0.0, 1.0))
        out.append(dict(id=tid, cx=last["cx"], cy=last["cy"], bbox=(x0, y0, x1, y1),
                        label=label, score=score))
    return out, {"n_tracks": len(out),
                 "mean_size_px": float(np.mean([d["bbox"][2] - d["bbox"][0] for d in out]))
                 if out else 0.0}
'''

_PRESETS = {
    "Greedy NN tracker (MultiTracker)": _PRESET_NN_TRACKER,
    "Classifying tracker (+ metrics)": _PRESET_CLASSIFY,
    "Per-frame blobs (no memory)": _PRESET_BLOBS,
    "Single smoothed centroid": _PRESET_CENTROID,
    "Live filter (keep-mask demo)": _PRESET_FILTER,
}


class _AlgoEnv:
    """The ``ev`` object handed to a user algorithm: this frame's events plus helpers.

    Attributes
    ----------
    x, y : int arrays     pixel coordinates of the events in this frame
    p    : int array      polarity (1 = ON, 0 = OFF)
    t    : int array      event time in µs (zero-based to the recording)
    t_s  : float array    event time in seconds
    n    : int            number of events
    W, H : int            sensor width / height
    t0, t1, dt : float    window start / stop / length (s) at the seek cursor
    np   : module         numpy, so algorithms need no imports
    frame : (H, W) array  the accumulated event-count image (computed on first access)
    """

    def __init__(self, win, t0, t1):
        self.x = np.asarray(win.x)
        self.y = np.asarray(win.y)
        self.p = np.asarray(win.p)
        self.t = np.asarray(win.t)
        self.t_s = win.t_s
        self.n = int(win.n)
        self.W = int(win.width)
        self.H = int(win.height)
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.dt = float(t1 - t0)
        self.np = np
        self._win = win
        self._frame = None

    @property
    def frame(self):
        if self._frame is None:
            self._frame = (accumulate_frame(self._win, mode="count") if self.n
                           else np.zeros((self.H, self.W), np.float32))
        return self._frame

    def blobs(self, min_pixels: int = 40, dilation: int = 2, erode: int = 1):
        """Connected-component blobs on this frame's events (strongest first).

        Returns a list of ``dict(cx, cy, bbox=(x0,y0,x1,y1), area)`` — feed the centroids to a
        tracker, or return the list directly to overlay raw per-frame detections.
        """
        out = cluster_frame(self.x, self.y, self.W, self.H,
                            min_pixels=int(min_pixels), dilation=int(dilation), erode=int(erode))
        return [dict(cx=c[0], cy=c[1], bbox=(c[2], c[3], c[4], c[5]), area=c[6]) for c in out]


class _CodeEditor(QtWidgets.QPlainTextEdit):
    """A tiny code editor: monospace, 4-space soft tabs (Python-friendly)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 12px;")
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Tab:
            self.insertPlainText("    ")
            return
        super().keyPressEvent(e)


class Sandbox(QtWidgets.QWidget):
    """A bench for direct, low-level manipulation, live tracking, and inspection of events."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.ctl = controller
        self.rec = None
        self._win = None                 # last filtered EventWindow
        self._algo_fn = None             # compiled user algorithm (or None)
        self._algo_nargs = 1             # 1 -> track(ev) · 2 -> track(ev, state)
        self._algo_state = {}            # persists across frames (the tracker's memory)
        self._play_throttle = 0.0

        # tracking overlay / trails / performance history
        self._trails = {}                # id -> deque[(cx, cy)]
        self._trail_seen = {}            # id -> frame index last updated
        self._frame_idx = 0
        self._perf_ms = deque(maxlen=_PERF_HIST)
        self._perf_in = deque(maxlen=_PERF_HIST)

        # baseline (A/B) algorithm — a frozen snapshot run alongside the candidate
        self._baseline_fn = None
        self._baseline_nargs = 1
        self._baseline_state = {}
        self._trails_b = {}
        self._trail_seen_b = {}

        # efficacy (real-data proxies) history
        self._eff_hist = deque(maxlen=_PERF_HIST)     # selected metric over time (candidate)
        self._cov_hist = deque(maxlen=60)             # candidate: 1 if a track this frame
        self._cov_hist_b = deque(maxlen=60)           # baseline coverage
        self._last_metrics = {}
        self._last_metrics_b = None

        root = QtWidgets.QHBoxLayout(self)

        # ---- left: frame (+overlays) + transport + stats ----
        left = QtWidgets.QVBoxLayout()
        self.glw = pg.GraphicsLayoutWidget()
        self.vb = self.glw.addViewBox(lockAspect=True, invertY=True)
        self.img = pg.ImageItem(axisOrder="row-major"); self.vb.addItem(self.img)
        self.roi = pg.RectROI([40, 40], [120, 120], pen=pg.mkPen(style.ACCENT, width=2))
        self.vb.addItem(self.roi)
        self.roi.sigRegionChangeFinished.connect(self._on_roi)
        try:
            self.img.setColorMap(pg.colormap.get("inferno", source="matplotlib"))
        except Exception:
            pass
        # baseline overlay layer (grey, drawn behind; only when A/B is on)
        self.trail_scatter_b = pg.ScatterPlotItem(pen=None, size=3)
        self.head_scatter_b = pg.ScatterPlotItem(pen=pg.mkPen(style.MUTED, width=1), size=8)
        self.trail_scatter_b.setZValue(-1); self.head_scatter_b.setZValue(-1)
        self.vb.addItem(self.trail_scatter_b)
        self.vb.addItem(self.head_scatter_b)
        self._track_boxes_b = []
        self._track_texts_b = []
        # candidate tracking overlay items (created once, updated each frame; drawn on top)
        self.trail_scatter = pg.ScatterPlotItem(pen=None, size=4)
        self.head_scatter = pg.ScatterPlotItem(pen=pg.mkPen(style.FG, width=1), size=11)
        self.trail_scatter.setZValue(5); self.head_scatter.setZValue(5)
        self.vb.addItem(self.trail_scatter)
        self.vb.addItem(self.head_scatter)
        self._track_boxes = []
        self._track_texts = []
        left.addWidget(self.glw, 1)
        # the op-chain preview uses its own window spinbox, so the transport's accumulation-
        # direction toggle would be a no-op here — hide it.
        self.transport = TransportBar(self.ctl, host=self, show_accum_dir=False)
        left.addWidget(self.transport)
        self.stats = QtWidgets.QLabel("—"); self.stats.setObjectName("muted")
        from gottlux.app.uikit import reserve_lines      # fixed height: live rate re-wraps without
        reserve_lines(self.stats, 3)                     # twitching the view above it

        left.addWidget(self.stats)
        lw = QtWidgets.QWidget(); lw.setLayout(left)

        # ---- right: tabbed control deck, in a draggable min-width splitter (was a 480 px
        # cap) so the view and deck both stay usable on small and split-view layouts ----
        from gottlux.app.uikit import plot_with_deck
        root.addWidget(plot_with_deck(lw, self._build_deck(), min_deck=360, init_deck=460,
                                      scroll=False))

        self.ctl.cursorChanged.connect(self._on_cursor)
        self.ctl.accumChanged.connect(self._debounced)
        self.ctl.accumChanged.connect(lambda *_: self._refresh_budget_line())
        self._debounce = QtCore.QTimer(self, singleShot=True, interval=140)
        self._debounce.timeout.connect(self.apply)

    # ================================================================== deck
    def _build_deck(self):
        self.deck = QtWidgets.QTabWidget()
        self.deck.addTab(self._build_tab(), "Build")
        self.deck.addTab(self._track_tab(), "Track")
        self.deck.addTab(self._efficacy_tab(), "Efficacy")
        self.deck.addTab(self._perf_tab(), "Performance")
        self.deck.addTab(self._analyze_tab(), "Analyze")
        return self.deck

    # ------------------------------------------------------------- Build tab
    def _build_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)

        sel = QtWidgets.QGroupBox("1 · Select")
        sf = QtWidgets.QFormLayout(sel)
        self.window = QtWidgets.QDoubleSpinBox(); self.window.setRange(0.001, 30.0)
        self.window.setValue(0.2); self.window.setDecimals(3); self.window.setSuffix(" s")
        self.window.setToolTip("Selection length starting at the seek cursor — the per-frame "
                               "window the op-chain and your tracking algorithm see.")
        self.window.valueChanged.connect(self._debounced)
        self.pol = QtWidgets.QComboBox(); self.pol.addItems(_POL)
        self.pol.setToolTip("Keep all events, or only ON / only OFF.")
        self.pol.currentIndexChanged.connect(self.apply)
        self.use_roi = QtWidgets.QCheckBox("Restrict to ROI box"); self.use_roi.setChecked(False)
        self.use_roi.setToolTip("Limit the selection to the cyan ROI box on the image.")
        self.use_roi.toggled.connect(self.apply)
        sf.addRow("Window", self.window)
        sf.addRow("Polarity", self.pol)
        sf.addRow(self.use_roi)
        v.addWidget(sel)

        ops = QtWidgets.QGroupBox("2 · Op-chain (applied in order, then fed to the algorithm)")
        of = QtWidgets.QFormLayout(ops)
        self.op_hot = QtWidgets.QCheckBox("Hot-pixel remove")
        self.hot_pct = QtWidgets.QDoubleSpinBox(); self.hot_pct.setRange(90, 100); self.hot_pct.setValue(99.9)
        self.hot_pct.setDecimals(2); self.hot_pct.setSuffix(" pct")
        self.op_refr = QtWidgets.QCheckBox("Refractory")
        self.refr_us = QtWidgets.QSpinBox(); self.refr_us.setRange(0, 100000); self.refr_us.setValue(1000)
        self.refr_us.setSuffix(" µs")
        self.op_bg = QtWidgets.QCheckBox("Suppress static bg")
        self.op_sub = QtWidgets.QCheckBox("Subsample")
        self.sub_n = QtWidgets.QSpinBox(); self.sub_n.setRange(1, 64); self.sub_n.setValue(1)
        self.sub_n.setPrefix("1/")
        for cb in (self.op_hot, self.op_refr, self.op_bg, self.op_sub):
            cb.toggled.connect(self.apply)
        for sp in (self.hot_pct, self.refr_us, self.sub_n):
            sp.valueChanged.connect(self.apply)
        self.op_hot.setToolTip("Drop top-percentile firing (stuck) pixels.")
        self.op_refr.setToolTip("Drop events arriving < N µs after the previous at the same pixel.")
        self.op_bg.setToolTip("Subtract the persistent-pixel static background (staring scenes).")
        self.op_sub.setToolTip("Keep every Nth event (thin dense streams for quick looks).")
        of.addRow(self.op_hot, self.hot_pct)
        of.addRow(self.op_refr, self.refr_us)
        of.addRow(self.op_bg)
        of.addRow(self.op_sub, self.sub_n)
        v.addWidget(ops)

        insp = QtWidgets.QGroupBox("3 · Raw events (first 24)")
        inf = QtWidgets.QVBoxLayout(insp)
        self.raw = QtWidgets.QPlainTextEdit(); self.raw.setReadOnly(True)
        self.raw.setMaximumHeight(150)
        self.raw.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        inf.addWidget(self.raw)
        v.addWidget(insp)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------- Track tab
    def _track_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)

        top = QtWidgets.QHBoxLayout()
        self.algo_on = QtWidgets.QCheckBox("Apply live")
        self.algo_on.setToolTip("Run your algorithm on every frame as you seek/play, overlaying "
                                "its output on the image. Off = plain op-chain bench.")
        self.algo_on.toggled.connect(self._on_algo_toggle)
        self.preset = QtWidgets.QComboBox(); self.preset.addItems(list(_PRESETS))
        self.preset.setToolTip("Load a worked example into the editor (replaces its contents).")
        self.load_btn = QtWidgets.QToolButton(); self.load_btn.setText("Load")
        self.load_btn.setToolTip("Load the selected preset into the editor and compile it.")
        self.load_btn.clicked.connect(self._load_preset)
        top.addWidget(self.algo_on)
        top.addStretch(1)
        top.addWidget(self.preset, 1)
        top.addWidget(self.load_btn)
        v.addLayout(top)

        self.editor = _CodeEditor()
        self.editor.setPlainText(_PRESET_NN_TRACKER)
        self.editor.setMinimumHeight(280)
        v.addWidget(self.editor, 1)

        btns = QtWidgets.QHBoxLayout()
        self.compile_btn = QtWidgets.QPushButton("Compile"); self.compile_btn.setObjectName("primary")
        self.compile_btn.setToolTip("Compile the editor's code (Ctrl+Enter). Defines track(ev, state).")
        self.compile_btn.clicked.connect(self._compile)
        self.runone_btn = QtWidgets.QPushButton("Run once")
        self.runone_btn.setToolTip("Run the algorithm a single time at the current cursor.")
        self.runone_btn.clicked.connect(self.apply)
        self.reset_btn = QtWidgets.QPushButton("Reset state")
        self.reset_btn.setToolTip("Clear the algorithm's persistent state and all trails — start "
                                  "the tracker fresh.")
        self.reset_btn.clicked.connect(self._reset_state)
        btns.addWidget(self.compile_btn); btns.addWidget(self.runone_btn); btns.addWidget(self.reset_btn)
        v.addLayout(btns)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self.editor, self._compile)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Enter"), self.editor, self._compile)

        # display options
        opt = QtWidgets.QHBoxLayout()
        self.show_trails = QtWidgets.QCheckBox("Trails"); self.show_trails.setChecked(True)
        self.show_boxes = QtWidgets.QCheckBox("Boxes"); self.show_boxes.setChecked(True)
        self.show_labels = QtWidgets.QCheckBox("Labels"); self.show_labels.setChecked(True)
        for c in (self.show_trails, self.show_boxes, self.show_labels):
            c.toggled.connect(self.apply)
        opt.addWidget(QtWidgets.QLabel("Show:"))
        opt.addWidget(self.show_trails); opt.addWidget(self.show_boxes); opt.addWidget(self.show_labels)
        opt.addStretch(1)
        v.addLayout(opt)

        self.console = QtWidgets.QLabel("Compile to begin. Define track(ev, state).")
        self.console.setObjectName("muted"); self.console.setWordWrap(True)
        self.console.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        v.addWidget(self.console)

        help_lbl = QtWidgets.QLabel(
            "<b>API.</b> <code>track(ev, state)</code> runs once per frame on the post-op-chain "
            "events. <code>ev</code>: x,y,p,t,t_s,n,W,H,t0,t1,dt,np,frame,blobs(). "
            "<code>state</code> is a dict that persists across frames (your tracker's memory). "
            "Injected: np, ndimage, MultiTracker, cluster_frame, math.<br>"
            "<b>Return</b> a list of <code>dict(cx,cy,bbox=(x0,y0,x1,y1),id,label,score)</code> "
            "to track · a bool mask (len n) to filter · dict(x,y,p,t) to replace · None for no "
            "change. Append a dict to emit live metrics: <code>return out, {\"name\": value}</code>. "
            "The <b>Efficacy</b> tab scores how well the box holds/classifies the target.")
        help_lbl.setWordWrap(True); help_lbl.setObjectName("muted")
        v.addWidget(help_lbl)
        return w

    # ------------------------------------------------------------- Efficacy tab
    _EFF_METRICS = ["lock", "on-target SNR", "jitter (px)", "coverage", "tracks"]

    def _efficacy_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)

        # headline lock score + one-line breakdown
        self.lock_lbl = QtWidgets.QLabel("—")
        self.lock_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.lock_lbl.setStyleSheet("font-size: 26px; font-weight: 700;")
        self.lock_lbl.setToolTip("Composite lock quality 0–1 = 0.5·SNR-term + 0.3·steadiness + "
                                 "0.2·coverage. Green ≥ 0.66, amber ≥ 0.33, red below. Your eyes "
                                 "are the real judge — this just quantifies what you see.")
        v.addWidget(self.lock_lbl)
        self.eff_sub = QtWidgets.QLabel("Enable Apply live on the Track tab and play.")
        self.eff_sub.setAlignment(QtCore.Qt.AlignCenter)
        self.eff_sub.setObjectName("muted"); self.eff_sub.setWordWrap(True)
        v.addWidget(self.eff_sub)

        # on-target SNR band + history-plot metric selector
        row = QtWidgets.QHBoxLayout()
        self.eff_lo = QtWidgets.QSpinBox(); self.eff_lo.setRange(1, 4000); self.eff_lo.setValue(80)
        self.eff_lo.setSuffix(" Hz")
        self.eff_hi = QtWidgets.QSpinBox(); self.eff_hi.setRange(2, 4000); self.eff_hi.setValue(800)
        self.eff_hi.setSuffix(" Hz")
        for s in (self.eff_lo, self.eff_hi):
            s.setToolTip("Frequency band for the on-target spectral SNR (the flutter tone you "
                         "expect inside the box).")
        self.eff_metric = QtWidgets.QComboBox(); self.eff_metric.addItems(self._EFF_METRICS)
        self.eff_metric.setToolTip("Which metric the history plot below traces over time.")
        row.addWidget(QtWidgets.QLabel("On-target band")); row.addWidget(self.eff_lo)
        row.addWidget(self.eff_hi); row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("Plot")); row.addWidget(self.eff_metric)
        v.addLayout(row)

        self.eff_plot = pg.PlotWidget(); self.eff_plot.setMaximumHeight(150)
        self.eff_plot.setLabel("bottom", "recent frames")
        self.eff_curve = self.eff_plot.plot(pen=pg.mkPen(style.ACCENT, width=1.5))
        self.eff_curve_b = self.eff_plot.plot(pen=pg.mkPen(style.MUTED, width=1,
                                                           style=QtCore.Qt.DashLine))
        v.addWidget(self.eff_plot)

        # A/B comparison
        ab = QtWidgets.QGroupBox("A/B vs baseline")
        av = QtWidgets.QVBoxLayout(ab)
        hb = QtWidgets.QHBoxLayout()
        self.ab_chk = QtWidgets.QCheckBox("Compare")
        self.ab_chk.setToolTip("Run a frozen baseline algorithm alongside the candidate and "
                               "overlay it (grey) so you can see whether your tuning helped.")
        self.ab_chk.toggled.connect(self._on_ab_toggle)
        self.set_baseline_btn = QtWidgets.QPushButton("Set baseline = current")
        self.set_baseline_btn.setToolTip("Freeze the currently compiled algorithm as the "
                                         "baseline, then keep editing — the candidate is compared "
                                         "against this snapshot.")
        self.set_baseline_btn.clicked.connect(self._set_baseline)
        hb.addWidget(self.ab_chk); hb.addWidget(self.set_baseline_btn, 1)
        av.addLayout(hb)
        self.baseline_lbl = QtWidgets.QLabel("No baseline set.")
        self.baseline_lbl.setObjectName("muted"); self.baseline_lbl.setWordWrap(True)
        av.addWidget(self.baseline_lbl)
        self.ab_table = QtWidgets.QTableWidget(5, 3)
        self.ab_table.setHorizontalHeaderLabels(["baseline", "candidate", "Δ"])
        self.ab_table.setVerticalHeaderLabels(["Lock", "SNR", "Jitter px", "Coverage", "Tracks"])
        self.ab_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ab_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.ab_table.setMaximumHeight(190)
        av.addWidget(self.ab_table)
        v.addWidget(ab)

        self.custom_lbl = QtWidgets.QLabel(""); self.custom_lbl.setObjectName("muted")
        self.custom_lbl.setWordWrap(True)
        v.addWidget(self.custom_lbl)

        note = QtWidgets.QLabel(
            "Your eyes are the ground truth — these proxies just quantify what you watch. "
            "<b>On-target SNR</b>: is the box sitting on a real periodic flutter tone? "
            "<b>Jitter</b>: how steady the box centroid is. <b>Coverage</b>: fraction of recent "
            "frames the target was held. Have your algorithm emit extra numbers with "
            "<code>return out, {\"name\": value}</code> and they appear here too.")
        note.setWordWrap(True); note.setObjectName("muted")
        v.addWidget(note)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------- Performance tab
    def _perf_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        self.perf_lbl = QtWidgets.QLabel("Enable <b>Apply live</b> on the Track tab and play to "
                                         "measure the algorithm.")
        self.perf_lbl.setWordWrap(True)
        v.addWidget(self.perf_lbl)

        self.perf_plot = pg.PlotWidget()
        self.perf_plot.setLabel("left", "compute", "ms")
        self.perf_plot.setLabel("bottom", "recent frames")
        self.perf_plot.setToolTip("Per-frame algorithm compute time. The dashed line is the "
                                  "live-data budget (the accumulation window's own duration) — "
                                  "stay under it to keep up with a live sensor at this exposure.")
        self.perf_curve = self.perf_plot.plot(pen=pg.mkPen(style.ACCENT, width=1.5))
        self.budget_line = pg.InfiniteLine(angle=0, movable=False,
                                           pen=pg.mkPen(style.ACCENT2, width=1, style=QtCore.Qt.DashLine))
        self.perf_plot.addItem(self.budget_line)
        self.budget_text = pg.TextItem(color=style.ACCENT2, anchor=(0, 1))
        self.perf_plot.addItem(self.budget_text)
        v.addWidget(self.perf_plot, 1)

        note = QtWidgets.QLabel(
            "Compute time is measured around your <code>track()</code> call only. "
            "<b>Max FPS</b> = 1000 / ms is the frame rate the algorithm could sustain. "
            "<b>Throughput</b> is events processed per second. The <b>real-time</b> verdict checks "
            "p95 compute time against the live-data budget — the accumulation window's own "
            "duration — so 'YES' means the algorithm could keep up with a live sensor at this "
            "exposure, regardless of how fast or slow you play it back.")
        note.setWordWrap(True); note.setObjectName("muted")
        v.addWidget(note)
        return w

    # ------------------------------------------------------------- Analyze tab
    def _analyze_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        an = QtWidgets.QGroupBox("Spectrum of selection")
        af = QtWidgets.QVBoxLayout(an)
        h = QtWidgets.QHBoxLayout()
        self.method = QtWidgets.QComboBox(); self.method.addItems(_METHODS)
        self.flo = QtWidgets.QSpinBox(); self.flo.setRange(1, 4000); self.flo.setValue(20); self.flo.setSuffix(" Hz")
        self.fhi = QtWidgets.QSpinBox(); self.fhi.setRange(2, 4000); self.fhi.setValue(800); self.fhi.setSuffix(" Hz")
        h.addWidget(self.method, 1); h.addWidget(self.flo); h.addWidget(self.fhi)
        af.addLayout(h)
        self.spec = pg.PlotWidget(); self.spec.setMaximumHeight(160)
        self.spec.setLabel("bottom", "Hz"); self.spec.setLogMode(y=True)
        self.spec_curve = self.spec.plot(pen=pg.mkPen(style.ACCENT, width=1.5))
        af.addWidget(self.spec)
        self.spec_btn = QtWidgets.QPushButton("Compute spectrum"); self.spec_btn.clicked.connect(self._spectrum)
        af.addWidget(self.spec_btn)
        self.spec_lbl = QtWidgets.QLabel("—"); self.spec_lbl.setObjectName("muted"); self.spec_lbl.setWordWrap(True)
        af.addWidget(self.spec_lbl)
        v.addWidget(an)

        self.export_btn = QtWidgets.QPushButton("Export this selection…")
        self.export_btn.clicked.connect(self._export)
        v.addWidget(self.export_btn)
        v.addStretch(1)
        return w

    # ================================================================== data
    def set_recording(self, rec):
        self.rec = rec
        self.vb.setRange(xRange=(0, rec.width), yRange=(0, rec.height), padding=0)
        self._reset_state()
        self.apply()

    def showEvent(self, ev):
        super().showEvent(ev)
        self.apply()

    def sync(self):
        self.apply()

    def _on_cursor(self, *_):
        if self.rec is None or not self.isVisible():
            return
        if self.ctl.playing:
            now = time.perf_counter()
            if now - self._play_throttle >= 0.03:        # cap live work at ~33 Hz
                self._play_throttle = now
                self.apply()
        else:
            self._debounced()

    def _debounced(self, *_):
        if self.isVisible():
            self._debounce.start()

    def _on_roi(self, *_):
        if self.use_roi.isChecked():
            self.apply()

    def _roi_tuple(self):
        pos = self.roi.pos(); size = self.roi.size()
        x0 = int(max(0, pos.x())); y0 = int(max(0, pos.y()))
        x1 = int(min(self.rec.width, pos.x() + size.x()))
        y1 = int(min(self.rec.height, pos.y() + size.y()))
        return (x0, y0, x1, y1) if (x1 > x0 and y1 > y0) else None

    # ================================================================== the pipeline
    def apply(self, *_):
        if self.rec is None or not self.isVisible():
            return
        t0 = self.ctl.cursor
        t1 = min(t0 + self.window.value(), self.rec.t_stop_s)
        roi = self._roi_tuple() if self.use_roi.isChecked() else None
        win = self.rec.window(t0, t1, roi=roi)
        n_in = win.n
        stages = [("select", n_in)]
        if win.n:
            keep = np.ones(win.n, bool)
            pol = self.pol.currentText()
            if pol == "ON only":
                keep &= (np.asarray(win.p) == 1)
            elif pol == "OFF only":
                keep &= (np.asarray(win.p) == 0)
            if pol != "all":
                stages.append(("polarity", int(keep.sum())))
            if self.op_hot.isChecked():
                keep &= hot_pixel_mask(win, self.hot_pct.value())
                stages.append(("hot-pixel", int(keep.sum())))
            if self.op_refr.isChecked():
                keep &= refractory_filter(win, self.refr_us.value())
                stages.append(("refractory", int(keep.sum())))
            if self.op_bg.isChecked():
                keep &= self._bg_mask(win)
                stages.append(("bg-suppress", int(keep.sum())))
            win = _subset(win, keep)
            if self.op_sub.isChecked() and self.sub_n.value() > 1:
                win = _subset(win, _stride_mask(win.n, self.sub_n.value()))
                stages.append(("subsample", win.n))

        win0 = win                              # the events both candidate and baseline see
        dets_c = dets_b = None
        custom_c = {}
        ms_c = None
        win_disp = win0
        if self.algo_on.isChecked() and self._algo_fn is not None:
            rc = self._run_one(self._algo_fn, self._algo_nargs, self._algo_state, win0,
                               t0, t1, is_candidate=True)
            win_disp, dets_c, custom_c, ms_c = rc["win"], rc["dets"], rc["custom"], rc["ms"]
            if rc["stage"]:
                stages.append(rc["stage"])
        if self.ab_chk.isChecked() and self._baseline_fn is not None:
            rb = self._run_one(self._baseline_fn, self._baseline_nargs, self._baseline_state,
                               win0, t0, t1, is_candidate=False)
            dets_b = rb["dets"]

        self._win = win_disp
        self._render(win_disp, t0, t1, stages)

        # advance one frame and update both overlay layers
        if dets_c is not None or dets_b is not None:
            self._frame_idx += 1
        if dets_c is not None:
            self._update_trails(dets_c, self._trails, self._trail_seen, self._frame_idx)
            self._track_boxes, self._track_texts = self._render_layer(
                dets_c, self._trails, self.head_scatter, self.trail_scatter,
                self._track_boxes, self._track_texts, grey=False)
        else:
            self._clear_candidate()
        if dets_b is not None:
            self._update_trails(dets_b, self._trails_b, self._trail_seen_b, self._frame_idx)
            self._track_boxes_b, self._track_texts_b = self._render_layer(
                dets_b, self._trails_b, self.head_scatter_b, self.trail_scatter_b,
                self._track_boxes_b, self._track_texts_b, grey=True)
        else:
            self._clear_baseline()

        # performance + efficacy (whenever the candidate ran)
        if ms_c is not None:
            n_out = len(dets_c) if dets_c is not None else win_disp.n
            self._record_perf(ms_c, win0.n, n_out, dets_c)
            mc = self._efficacy(dets_c, win0, self._trails, self._cov_hist, custom_c)
            mb = (self._efficacy(dets_b, win0, self._trails_b, self._cov_hist_b, {})
                  if dets_b is not None else None)
            self._last_metrics, self._last_metrics_b = mc, mb
            self._update_efficacy(mc, mb)

    # -------------------------------------------------- the live algorithm
    def _run_one(self, fn, nargs, state, win0, t0, t1, is_candidate):
        """Run one compiled algorithm on *win0*; return a dict of its outcome.

        ``{win, dets, custom, ms, ok, stage}``. A ``(output, metrics_dict)`` return is split
        into the output and a custom-metrics dict. Folds a returned mask/replace into ``win``.
        """
        out = {"win": win0, "dets": None, "custom": {}, "ms": 0.0, "ok": False, "stage": None}
        env = _AlgoEnv(win0, t0, t1)
        try:
            tic = time.perf_counter()
            ret = fn(env, state) if nargs >= 2 else fn(env)
            out["ms"] = (time.perf_counter() - tic) * 1e3
        except Exception:
            if is_candidate:
                self._algo_error()
            return out
        if is_candidate:
            self._console_ok()
        out["ok"] = True
        ret, custom = _split_metrics(ret)
        out["custom"] = custom or {}
        kind, payload = _classify_return(ret, win0.n)
        if kind == "mask":
            out["win"] = _subset(win0, payload); out["stage"] = ("algo-filter", out["win"].n)
        elif kind == "replace":
            out["win"] = _win_from_dict(payload, win0); out["stage"] = ("algo-replace", out["win"].n)
        elif kind == "dets":
            out["dets"] = payload; out["stage"] = ("algo-track", len(payload))
        return out

    # -------------------------------------------------- compile / presets / state
    def _compile(self):
        src = self.editor.toPlainText()
        g = {"np": np, "math": math, "MultiTracker": MultiTracker,
             "cluster_frame": cluster_frame}
        try:
            from scipy import ndimage
            g["ndimage"] = ndimage
        except Exception:
            pass
        ns = {}
        try:
            exec(compile(src, "<sandbox-algorithm>", "exec"), g, ns)
        except Exception:
            self._algo_fn = None
            self._algo_error("compile")
            return
        fn = ns.get("track") or ns.get("algorithm")
        if not callable(fn):
            self._algo_fn = None
            self.console.setText(f"<span style='color:{style.ACCENT2}'>No <b>track(ev, state)</b> "
                                 "(or algorithm(ev)) function defined.</span>")
            return
        self._algo_fn = fn
        try:
            self._algo_nargs = len(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            self._algo_nargs = 1
        self._reset_state()
        self.console.setText(f"<span style='color:{style.GOOD}'>Compiled ✓</span> — "
                             f"track takes {self._algo_nargs} arg(s). "
                             "Enable <b>Apply live</b> and play.")
        if self.algo_on.isChecked():
            self.apply()

    def _load_preset(self):
        self.editor.setPlainText(_PRESETS[self.preset.currentText()])
        self._compile()

    def _reset_state(self):
        self._algo_state = {}
        self._baseline_state = {}                 # restart baseline runtime too, for a fair A/B
        self._trails.clear(); self._trail_seen.clear()
        self._trails_b.clear(); self._trail_seen_b.clear()
        self._frame_idx = 0
        self._perf_ms.clear(); self._perf_in.clear()
        self._eff_hist.clear(); self._cov_hist.clear(); self._cov_hist_b.clear()
        self._clear_tracks()

    def _on_algo_toggle(self, on):
        if on and self._algo_fn is None:
            self._compile()
        self.apply()

    def _algo_error(self, where="run"):
        tb = traceback.format_exc(limit=3).strip().replace("<", "&lt;").replace(">", "&gt;")
        last = tb.splitlines()[-1] if tb else "error"
        self.console.setText(
            f"<span style='color:{style.ACCENT2}'><b>{where} error:</b> {last}</span>"
            f"<pre style='color:{style.MUTED};font-size:10px'>{tb}</pre>")

    def _console_ok(self):
        # only overwrite an error message once things are healthy again
        if self.console.text().startswith(f"<span style='color:{style.ACCENT2}'") or \
                "error" in self.console.text():
            self.console.setText(f"<span style='color:{style.GOOD}'>Running ✓</span>")

    # ================================================================== render + overlay
    def _render(self, win, t0, t1, stages):
        frame = accumulate_frame(win, mode="count") if win.n else np.zeros(self.rec.sensor_shape, np.float32)
        disp, vmax = tonemap.compress(frame, expr="sqrt")
        self.img.setImage(disp, levels=(0, 1), autoLevels=False)
        chain = "  →  ".join(f"{k}:{v:,}" for k, v in stages)
        on = int((np.asarray(win.p) == 1).sum()) if win.n else 0
        dur = win.duration_s
        rate = win.n / dur if dur > 0 else 0
        self.stats.setText(
            f"<b>{win.n:,}</b> events kept · {on:,}↑ {win.n-on:,}↓ · "
            f"{dur*1e3:.1f} ms · {rate/1e3:.1f} kev/s<br><span style='color:{style.MUTED}'>{chain}</span>")
        self._fill_raw(win)

    @staticmethod
    def _update_trails(dets, trails, trail_seen, fi):
        for d in dets:
            trails.setdefault(d["id"], deque(maxlen=_TRAIL_LEN)).append((d["cx"], d["cy"]))
            trail_seen[d["id"]] = fi
        stale = [tid for tid, seen in trail_seen.items() if fi - seen > _TRAIL_LEN]
        for tid in stale:
            trails.pop(tid, None); trail_seen.pop(tid, None)

    def _render_layer(self, dets, trails, head_scatter, trail_scatter, prev_boxes, prev_texts, grey):
        """Draw one overlay layer (candidate = id-coloured, baseline = grey). Returns its
        (boxes, texts) item lists so the caller can clear them next frame."""
        for it in prev_boxes + prev_texts:
            self.vb.removeItem(it)
        boxes, texts = [], []
        show_box = self.show_boxes.isChecked()
        show_lab = self.show_labels.isChecked()

        def col(tid):
            return pg.mkColor(150, 150, 150) if grey else pg.intColor(tid, hues=12)

        for d in dets[:32]:
            color = col(d["id"])
            if show_box:
                x0, y0, x1, y1 = d["bbox"]
                r = pg.RectROI([x0, y0], [max(1.0, x1 - x0), max(1.0, y1 - y0)],
                               pen=pg.mkPen(color, width=1 if grey else 2),
                               movable=False, resizable=False)
                for h in r.getHandles():
                    r.removeHandle(h)
                self.vb.addItem(r); boxes.append(r)
            if show_lab:
                parts = []
                if d.get("label"):
                    parts.append(str(d["label"]))
                if d.get("score") is not None:
                    parts.append(f"{float(d['score']):.2f}")
                if parts:
                    txt = pg.TextItem(" ".join(parts), color=color, anchor=(0, 1))
                    txt.setPos(d["bbox"][0], d["bbox"][1])
                    self.vb.addItem(txt); texts.append(txt)

        head_scatter.setData([{"pos": (d["cx"], d["cy"]), "brush": col(d["id"]),
                               "size": 8 if grey else 11} for d in dets])
        if self.show_trails.isChecked():
            spots = []
            for tid, pts in trails.items():
                base = col(tid); m = len(pts)
                for j, (cx, cy) in enumerate(pts):
                    a = int(40 + 180 * (j + 1) / m)
                    c = pg.mkColor(base); c.setAlpha(min(a, 120) if grey else a)
                    spots.append({"pos": (cx, cy), "brush": c, "size": 3 if grey else 4})
            trail_scatter.setData(spots)
        else:
            trail_scatter.setData([])
        return boxes, texts

    def _clear_candidate(self):
        for it in self._track_boxes + self._track_texts:
            self.vb.removeItem(it)
        self._track_boxes = []; self._track_texts = []
        self.head_scatter.setData([]); self.trail_scatter.setData([])

    def _clear_baseline(self):
        for it in self._track_boxes_b + self._track_texts_b:
            self.vb.removeItem(it)
        self._track_boxes_b = []; self._track_texts_b = []
        self.head_scatter_b.setData([]); self.trail_scatter_b.setData([])

    def _clear_tracks(self):
        self._clear_candidate()
        self._clear_baseline()

    # ================================================================== performance
    def _record_perf(self, compute_ms, n_in, n_out, dets):
        self._perf_ms.append(compute_ms)
        self._perf_in.append(n_in)
        ms = np.array(self._perf_ms, float)
        # real-time = process one accumulation window faster than the window's own duration
        # (i.e. keep up with a live sensor at this exposure), independent of playback speed.
        budget = self.ctl.accum * 1000.0
        p50 = float(np.percentile(ms, 50)); p95 = float(np.percentile(ms, 95))
        rt = p95 < budget
        thru = (n_in / (compute_ms / 1e3)) if compute_ms > 0 else 0.0
        active = len(self._trails) if dets is not None else 0
        mean_trail = (np.mean([len(t) for t in self._trails.values()])
                      if self._trails else 0.0)
        out_kind = "detections" if dets is not None else "events kept"
        verdict = (f"<span style='color:{style.GOOD}'><b>YES</b></span>" if rt
                   else f"<span style='color:{style.ACCENT2}'><b>NO</b></span>")
        self.perf_lbl.setText(
            f"compute <b>{compute_ms:.2f} ms</b>  (max <b>{1000.0/max(compute_ms,1e-6):.0f} fps</b>)"
            f"   ·   throughput <b>{thru/1e6:.2f} Mev/s</b>  ({n_in:,} ev in)<br>"
            f"output <b>{n_out}</b> {out_kind}   ·   active tracks <b>{active}</b>   ·   "
            f"mean trail <b>{mean_trail:.0f}</b> frames<br>"
            f"last {len(ms)} frames:  p50 <b>{p50:.2f}</b>  ·  p95 <b>{p95:.2f}</b> ms   ·   "
            f"real-time vs live data (accum {budget:.1f} ms/frame): {verdict}")
        self.perf_curve.setData(np.arange(ms.size), ms)
        self._refresh_budget_line()

    def _refresh_budget_line(self):
        if not hasattr(self, "budget_line"):
            return
        budget = self.ctl.accum * 1000.0
        self.budget_line.setPos(budget)
        self.budget_text.setText(f"live-data budget (accum) = {budget:.1f} ms")
        self.budget_text.setPos(0, budget)

    # ================================================================== efficacy (real-data proxies)
    def _efficacy(self, dets, win0, trails, cov_hist, custom):
        """Quantify how well the box holds/classifies the target — proxies, no ground truth.

        on-target SNR (is the box on a real flutter tone), centroid jitter (steadiness),
        coverage (fraction of recent frames held), lifetime, class label/score, and a composite
        ``lock`` score. ``custom`` carries whatever the algorithm emitted via ``return out, {…}``.
        """
        m = {"_custom": dict(custom or {})}
        n = len(dets) if dets else 0
        m["tracks"] = n
        cov_hist.append(1 if n > 0 else 0)
        m["coverage"] = float(np.mean(cov_hist)) if cov_hist else 0.0
        # centroid jitter: wobble AROUND smooth motion, not absolute spread — so a fast but
        # steadily-moving target scores low. Use the spread of frame-to-frame steps: for
        # constant velocity the steps are equal, so their std → 0.
        jit = []
        for d in (dets or []):
            tr = trails.get(d["id"])
            if tr and len(tr) >= 3:
                da = np.diff(np.asarray(tr, float), axis=0)        # successive steps
                jit.append(float(np.hypot(da[:, 0].std(), da[:, 1].std())))
        m["jitter"] = float(np.mean(jit)) if jit else float("nan")
        m["lifetime"] = float(np.mean([len(t) for t in trails.values()])) if trails else 0.0
        # on-target spectral SNR: per track, the events inside its bbox over the window
        lo = float(self.eff_lo.value()); hi = max(float(self.eff_hi.value()), lo + 1)
        snrs, freqs = [], []
        if dets and win0.n:
            wx = np.asarray(win0.x); wy = np.asarray(win0.y); wt = np.asarray(win0.t)
            for d in dets[:6]:
                x0, y0, x1, y1 = d["bbox"]
                inb = (wx >= x0) & (wx < x1) & (wy >= y0) & (wy < y1)
                tt = wt[inb]
                if tt.size >= 16:
                    sp = fq.region_spectrum(tt, fs=max(2.2 * hi, 2000.0), fmin=lo, fmax=hi)
                    if np.isfinite(sp.peak_freq):
                        snrs.append(sp.snr); freqs.append(sp.peak_freq)
        m["snr"] = float(np.mean(snrs)) if snrs else float("nan")
        m["freq"] = float(np.median(freqs)) if freqs else float("nan")
        # classification (the algorithm's own labels/scores — eyes judge correctness)
        m["labels"] = [str(d.get("label")) for d in (dets or []) if d.get("label")]
        scores = [d.get("score") for d in (dets or []) if d.get("score") is not None]
        m["score"] = float(np.mean(scores)) if scores else float("nan")
        # composite lock score 0–1
        if n > 0:
            snr_term = float(np.clip(m["snr"] / 8.0, 0, 1)) if np.isfinite(m["snr"]) else 0.0
            jit_term = 1.0 / (1.0 + (m["jitter"] if np.isfinite(m["jitter"]) else 0.0) / 5.0)
            m["lock"] = 0.5 * snr_term + 0.3 * jit_term + 0.2 * m["coverage"]
        else:
            m["lock"] = 0.0
        return m

    @staticmethod
    def _metric_value(m, key):
        return {"lock": m["lock"], "on-target SNR": m["snr"], "jitter (px)": m["jitter"],
                "coverage": m["coverage"], "tracks": float(m["tracks"])}.get(key, float("nan"))

    def _update_efficacy(self, mc, mb):
        lock = mc["lock"]
        color = style.GOOD if lock >= 0.66 else (style.WARN if lock >= 0.33 else style.ACCENT2)
        self.lock_lbl.setText(f"<span style='color:{color}'>LOCK {lock:.2f}</span>")
        snr = mc["snr"]; jit = mc["jitter"]; freq = mc["freq"]
        snr_s = f"{snr:.1f}" if np.isfinite(snr) else "—"
        jit_s = f"{jit:.1f}" if np.isfinite(jit) else "—"
        freq_s = f" · {freq:.0f} Hz" if np.isfinite(freq) else ""
        cls = ""
        if mc["labels"]:
            cls = " · class " + ",".join(sorted(set(mc["labels"])))
            if np.isfinite(mc["score"]):
                cls += f" {mc['score']:.2f}"
        self.eff_sub.setText(f"SNR {snr_s} · jitter {jit_s} px · cover {mc['coverage']*100:.0f}% · "
                             f"{mc['tracks']} track(s){freq_s}{cls}")

        # history plot of the selected metric (candidate solid, baseline dashed)
        key = self.eff_metric.currentText()
        self._eff_hist.append(self._metric_value(mc, key))
        ev = np.array([x if x is not None else np.nan for x in self._eff_hist], float)
        self.eff_curve.setData(np.arange(ev.size), np.nan_to_num(ev, nan=0.0))
        self.eff_plot.setLabel("left", key)

        # A/B table
        rows = [("Lock", "lock", "{:.2f}"), ("SNR", "snr", "{:.1f}"),
                ("Jitter px", "jitter", "{:.1f}"), ("Coverage", "coverage", "{:.0%}"),
                ("Tracks", "tracks", "{:.0f}")]
        for i, (_, k, fmt) in enumerate(rows):
            cv = mc.get(k, float("nan"))
            self._set_cell(i, 1, self._fmt(cv, fmt))
            if mb is not None:
                bv = mb.get(k, float("nan"))
                self._set_cell(i, 0, self._fmt(bv, fmt))
                d = (cv - bv) if (np.isfinite(cv) and np.isfinite(bv)) else float("nan")
                self._set_cell(i, 2, self._fmt(d, fmt, signed=True))
            else:
                self._set_cell(i, 0, "—"); self._set_cell(i, 2, "—")

        # custom metrics
        cust = mc.get("_custom") or {}
        if cust:
            self.custom_lbl.setText("custom: " + " · ".join(
                f"<b>{k}</b> {v:.4g}" if isinstance(v, (int, float)) else f"<b>{k}</b> {v}"
                for k, v in cust.items()))
        else:
            self.custom_lbl.setText("")

    @staticmethod
    def _fmt(v, fmt, signed=False):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        s = fmt.format(v)
        return ("+" + s) if (signed and v >= 0) else s

    def _set_cell(self, r, c, text):
        it = QtWidgets.QTableWidgetItem(text)
        it.setTextAlignment(QtCore.Qt.AlignCenter)
        self.ab_table.setItem(r, c, it)

    # ================================================================== A/B baseline
    def _set_baseline(self):
        if self._algo_fn is None:
            self._compile()
        if self._algo_fn is None:
            self.baseline_lbl.setText(f"<span style='color:{style.ACCENT2}'>Compile a working algorithm "
                                      "first.</span>")
            return
        self._baseline_fn = self._algo_fn
        self._baseline_nargs = self._algo_nargs
        self._baseline_state = {}
        self._trails_b.clear(); self._trail_seen_b.clear(); self._cov_hist_b.clear()
        self.baseline_lbl.setText(f"<span style='color:{style.GOOD}'>Baseline frozen ✓</span> — edit "
                                  "the algorithm; A/B overlays it (grey) and compares.")
        if not self.ab_chk.isChecked():
            blk = QtCore.QSignalBlocker(self.ab_chk); self.ab_chk.setChecked(True); del blk
        self.apply()

    def _on_ab_toggle(self, on):
        if on and self._baseline_fn is None:
            self.baseline_lbl.setText("Set a baseline first (button above).")
        if not on:
            self._clear_baseline()
        self.apply()

    # ================================================================== raw inspector
    def _fill_raw(self, win):
        n = min(24, win.n)
        lines = [f"{'idx':>4} {'x':>4} {'y':>4} {'p':>2} {'t (ms)':>12}"]
        x = np.asarray(win.x); y = np.asarray(win.y); p = np.asarray(win.p); ts = win.t_s
        for i in range(n):
            lines.append(f"{i:>4} {int(x[i]):>4} {int(y[i]):>4} {int(p[i]):>2} {ts[i]*1e3:>12.4f}")
        self.raw.setPlainText("\n".join(lines))

    # ================================================================== analysis / export
    def _bg_mask(self, win):
        try:
            from gottlux.core.background import staring_foreground_mask
            return staring_foreground_mask(win, bg_window_s=min(1.0, max(win.duration_s * 0.3, 0.05)))
        except Exception as e:
            print(f"[sandbox] background suppression unavailable: {e}")
            return np.ones(win.n, bool)

    def _spectrum(self):
        if self._win is None or self._win.n < 8:
            self.spec_lbl.setText("Too few events; widen the selection.")
            return
        lo, hi = float(self.flo.value()), float(self.fhi.value())
        if hi <= lo:
            hi = lo + 1
        meth = self.method.currentText()
        if meth.startswith("ISI"):
            f, s = fq.isi_frequency(self._win.t, fmin=lo, fmax=hi)
            self.spec_curve.setData([], [])
            self.spec_lbl.setText(f"ISI dominant ≈ <b>{f:.0f} Hz</b> · concentration {s:.2f}")
            return
        if meth.startswith("NUFFT"):
            sp = fq.nufft_spectrum(self._win.t, fmin=lo, fmax=hi)
        else:
            sp = fq.region_spectrum(self._win.t, fs=max(2.2 * hi, 2000), fmin=lo, fmax=hi)
        if sp.freqs.size:
            self.spec_curve.setData(sp.freqs, np.maximum(sp.power, 1e-12))
            self.spec.setXRange(0, min(hi * 1.3, sp.freqs[-1]))
        self.spec_lbl.setText(
            f"peak <b>{sp.peak_freq:.0f} Hz</b> · SNR <b>{sp.snr:.1f}</b> · "
            f"harmonic {sp.harmonic_score:.2f}" if np.isfinite(sp.peak_freq)
            else "no in-band peak")

    def _export(self):
        if self._win is None or self._win.n == 0:
            QtWidgets.QMessageBox.information(self, "Export", "Nothing selected.")
            return
        base, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export selection", "selection",
                                                        "Data (*.npz)")
        if not base:
            return
        base = os.path.splitext(base)[0]
        from gottlux.io import export
        w = self._win
        written = export.save_arrays(base + "_events", x=np.asarray(w.x), y=np.asarray(w.y),
                                     p=np.asarray(w.p), t_us=np.asarray(w.t))
        written += export.save_table({"x": np.asarray(w.x), "y": np.asarray(w.y),
                                      "p": np.asarray(w.p), "t_us": np.asarray(w.t)},
                                     base + "_events")
        if written:
            from gottlux.io.paths import open_in_file_browser
            open_in_file_browser(os.path.dirname(os.path.abspath(written[0])))   # reveal on export
        QtWidgets.QMessageBox.information(
            self, "Export", "Saved:\n" + "\n".join(os.path.basename(p) for p in written))


# ====================================================================================
# helpers
# ====================================================================================
def _subset(win, mask):
    """Return a new EventWindow keeping only events where mask is True."""
    m = np.asarray(mask, bool)
    return EventWindow(np.asarray(win.x)[m], np.asarray(win.y)[m], np.asarray(win.p)[m],
                       np.asarray(win.t)[m], win.width, win.height, win.t0_us)


def _stride_mask(n, stride):
    m = np.zeros(n, bool)
    m[::stride] = True
    return m


def _win_from_dict(d, ref):
    """Build an EventWindow from a user-returned dict(x, y, p, t) (t in µs)."""
    x = np.asarray(d["x"]); y = np.asarray(d["y"])
    p = np.asarray(d.get("p", np.ones(len(x), np.uint8)))
    t = np.asarray(d.get("t", d.get("t_us")))
    return EventWindow(x, y, p, t, ref.width, ref.height, ref.t0_us)


def _looks_like_metrics(d):
    """A plain metrics dict has none of the detection keys (so it isn't a detection itself)."""
    return isinstance(d, dict) and not any(k in d for k in ("cx", "cy", "x", "y", "bbox"))


def _split_metrics(ret):
    """Split a ``(output, metrics_dict)`` return into ``(output, metrics_or_None)``."""
    if isinstance(ret, tuple) and len(ret) == 2 and _looks_like_metrics(ret[1]):
        return ret[0], ret[1]
    return ret, None


def _classify_return(ret, n):
    """Map a user algorithm's return value to ``(kind, payload)``.

    kind ∈ {"none", "mask", "replace", "dets"}. A ``(output, metrics_dict)`` pair is reduced to
    its output here too, so the classifier is safe to call on a raw algorithm return.
    """
    ret, _ = _split_metrics(ret)
    if ret is None:
        return "none", None
    # boolean keep-mask
    if isinstance(ret, np.ndarray):
        if ret.dtype == bool and ret.shape == (n,):
            return "mask", ret
        return "none", None
    # a single detection or a replacement stream
    if isinstance(ret, dict):
        if any(k in ret for k in ("cx", "cy", "bbox")):
            d = _norm_det(ret, 0)
            return ("dets", [d]) if d else ("none", None)
        if "x" in ret and "y" in ret and ("t" in ret or "t_us" in ret):
            return "replace", ret
        return "none", None
    # a list of detections
    if isinstance(ret, (list, tuple)):
        out = []
        for i, item in enumerate(ret):
            d = _norm_det(item, i)
            if d:
                out.append(d)
        return "dets", out
    return "none", None


def _norm_det(item, i):
    """Normalize one detection (dict or (cx, cy[, bbox]) tuple) to a canonical dict."""
    cx = cy = None
    bbox = None
    label = None
    score = None
    if isinstance(item, dict):
        cx = item.get("cx", item.get("x"))
        cy = item.get("cy", item.get("y"))
        bbox = item.get("bbox")
        label = item.get("label")
        score = item.get("score", item.get("conf", item.get("confidence")))
        tid = int(item.get("id", i))
        if bbox is None and ("w" in item or "h" in item):
            w = float(item.get("w", _DEFAULT_BOX)); h = float(item.get("h", w))
            if cx is not None and cy is not None:
                bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        cx, cy = item[0], item[1]
        if len(item) >= 6:
            bbox = (item[2], item[3], item[4], item[5])
        elif len(item) == 3:
            bbox = item[2]
        tid = i
    else:
        return None
    if bbox is not None and (cx is None or cy is None):
        cx = 0.5 * (bbox[0] + bbox[2]); cy = 0.5 * (bbox[1] + bbox[3])
    if cx is None or cy is None:
        return None
    if bbox is None:
        half = _DEFAULT_BOX / 2
        bbox = (cx - half, cy - half, cx + half, cy + half)
    return dict(id=int(tid), cx=float(cx), cy=float(cy),
                bbox=tuple(float(b) for b in bbox), label=label,
                score=(float(score) if score is not None else None))
