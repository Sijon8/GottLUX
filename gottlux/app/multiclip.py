"""
multiclip.py — the multi-clip slate: several clips side-by-side, every function available.

This loads **several independent recordings (clips) at once**, all driven by **one shared
clock** so they play, scrub, and accumulate in lock-step, with a per-clip **slate offset** to
align clips that started at different moments (the EBS dashboard's slate, rebuilt on the
staring substrate).

Each clip pane can be rendered through **any function tab** via its **View** selector — an
event frame (the fast built-in), the Live viewer, Space-time 3-D, the Event-rate tower, the
Range lab, the Flutter workbench, or the Sandbox. A single **View** selector in the toolbar
switches every clip at once, so the slate becomes a side-by-side *multi-clip Space-time*,
*multi-clip event-rate tower*, *multi-clip range lab*, and so on. Each non-frame view runs on a
per-pane **child clock** (master cursor + the clip's slate offset), reusing the real panel
widgets unchanged — the shared live noise-filter suite applies throughout.

Any Prophesee ``.raw`` decodes regardless of sensor geometry (GenX320 320×320, IMX636
1280×720, mixed). Layout is **stacked** by default (Side-by-side / Grid also available); the
slate **loops** at the end of the clip.
"""
from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from gottlux.app import icons
from gottlux.app.loader import RecordingLoader
from gottlux.app.transport import TimeController, TransportBar
from gottlux.core import tonemap
from gottlux.core.accumulate import accumulate_frame

_MODES = ["count", "time_surface", "polarity", "polarity_ratio", "on", "off", "binary"]
_CMAPS = ["inferno", "viridis", "magma", "plasma", "cividis", "gray", "turbo",
          "coolwarm", "RdBu", "bwr", "seismic", "PiYG", "Spectral"]
_DIVERGING = {"coolwarm", "RdBu", "bwr", "seismic", "PiYG", "Spectral"}
_POLARITY_MODES = {"polarity", "polarity_ratio"}
_CTRL_W = 196        # width of the right-hand per-clip control column (px)

#: The function views a clip pane can be rendered through ("Event frame" is the built-in fast path).
VIEWS = ["Event frame", "Live viewer", "Space-time", "Event-rate", "Range lab",
         "Flutter workbench", "Sandbox"]


def _make_panel(name, clock, filters, compact=False):
    """Instantiate a function panel bound to *clock* (the pane's child clock).

    ``compact`` trims a panel's chrome for the cramped multi-clip slate (e.g. the Range lab
    drops its per-pane export/converge controls, which are redundant once you fuse the panes).
    """
    if name == "Live viewer":
        from gottlux.app.viewer import LiveViewer
        return LiveViewer(clock, filters)
    if name == "Space-time":
        from gottlux.app.spacetime import SpaceTimeView
        return SpaceTimeView(clock, filters)
    if name == "Event-rate":
        from gottlux.app.tower import EventRateTower
        return EventRateTower(clock, filters)
    if name == "Range lab":
        from gottlux.app.rangelab import RangeLab
        return RangeLab(clock, filters, compact=compact)
    if name == "Flutter workbench":
        from gottlux.app.workbench import FlutterWorkbench
        return FlutterWorkbench(clock)
    if name == "Sandbox":
        from gottlux.app.sandbox import Sandbox
        return Sandbox(clock)
    return None


def _labeled(label, widget):
    row = QtWidgets.QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    lab = QtWidgets.QLabel(label); lab.setFixedWidth(52)
    row.addWidget(lab); row.addWidget(widget, 1)
    return row


# ====================================================================================
# One clip in the slate — a recording, a slate offset, and a selectable function view
# ====================================================================================
class ClipPane(QtWidgets.QFrame):
    """A single clip: built-in event-frame view, or any function panel via the View selector."""

    removeRequested = QtCore.Signal(object)
    durationChanged = QtCore.Signal()
    offsetChanged = QtCore.Signal()

    def __init__(self, controller: TimeController, default_mode="count",
                 default_cmap="inferno", filters=None, default_view="Event frame", parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        # Grow to fill the slot the slate's splitter gives us, but never collapse so small a
        # pane reads as "missing" — pair with the explicit setSizes() in the slate's _relayout.
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(240, 200)
        self.ctl = controller            # the master clock (shared across the slate)
        self.filters = filters
        self.rec = None
        self.offset = 0.0
        self._static_vmax = None
        self._loader = None
        self._last_rgb = None
        self._child = None               # the function panel when View != "Event frame"
        self._child_type = None
        self._child_clk = TimeController(self)   # per-clip child clock (master cursor + offset)

        # ----- header: title · View · offset · remove -----
        self.title = QtWidgets.QLabel("(empty)")
        self.title.setObjectName("h2"); self.title.setWordWrap(True)
        self.view_sel = QtWidgets.QComboBox(); self.view_sel.addItems(VIEWS)
        self.view_sel.setCurrentText(default_view)
        self.view_sel.setToolTip("Render this clip through any function: an event frame, the "
                                 "Live viewer, Space-time 3-D, Event-rate tower, Range lab, "
                                 "Flutter workbench, or Sandbox.")
        self.view_sel.currentIndexChanged.connect(self._on_view)
        self.offset_sp = QtWidgets.QDoubleSpinBox()
        self.offset_sp.setRange(-600.0, 600.0); self.offset_sp.setDecimals(3)
        self.offset_sp.setSingleStep(0.01); self.offset_sp.setSuffix(" s"); self.offset_sp.setFixedWidth(88)
        self.offset_sp.setToolTip("Slate offset: shift this clip on the shared timeline so clips "
                                  "that started at different moments line up (local = cursor + offset).")
        self.offset_sp.valueChanged.connect(self._on_offset)
        self.remove_btn = QtWidgets.QToolButton()
        self.remove_btn.setIcon(icons.icon("close")); self.remove_btn.setIconSize(QtCore.QSize(12, 12))
        self.remove_btn.setToolTip("Remove this clip from the slate.")
        self.remove_btn.clicked.connect(lambda: self.removeRequested.emit(self))
        head = QtWidgets.QHBoxLayout(); head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self.title, 1)
        head.addWidget(QtWidgets.QLabel("View")); head.addWidget(self.view_sel)
        head.addWidget(QtWidgets.QLabel("off")); head.addWidget(self.offset_sp)
        head.addWidget(self.remove_btn)

        # ----- page 0: the built-in event-frame view (image + right control column) -----
        self.glw = pg.GraphicsLayoutWidget(); self.glw.setMinimumHeight(160)
        self.vb = self.glw.addViewBox(lockAspect=True, invertY=True)
        self.img = pg.ImageItem(axisOrder="row-major"); self.vb.addItem(self.img)
        self.busy = QtWidgets.QLabel("", self.glw)
        self.busy.setObjectName("muted")          # themed by the app stylesheet
        self.busy.move(8, 6)

        self.mode = QtWidgets.QComboBox(); self.mode.addItems(_MODES); self.mode.setCurrentText(default_mode)
        self.mode.setToolTip("Accumulation mode: count / time-surface / polarity (raw ON−OFF) / "
                             "polarity_ratio ((ON−OFF)/(ON+OFF) ∈ [−1,1], bounded) / on / off / binary.")
        self.mode.currentIndexChanged.connect(self._on_mode)
        self.cmap = QtWidgets.QComboBox(); self.cmap.addItems(_CMAPS); self.cmap.setCurrentText(default_cmap)
        self.cmap.currentIndexChanged.connect(self._apply_cmap)
        self.expr = QtWidgets.QComboBox(); self.expr.addItems(tonemap.EXPRESSIONS); self.expr.setCurrentText("sqrt")
        self.expr.currentIndexChanged.connect(self._render)
        self.scale = QtWidgets.QComboBox(); self.scale.addItems(["dynamic", "static"])
        self.scale.currentIndexChanged.connect(self._on_scale)
        self.readout = QtWidgets.QLabel("—"); self.readout.setObjectName("muted")
        # Per-frame metrics on fixed lines in a monospace font with word-wrap OFF: when a value's
        # width changes (the event rate spiking, counts jumping) the digits update in place within
        # fixed-width columns — a metric can never bump onto a new line or shove the others. That
        # kills the inline-vs-new-line "twitch". reserve_lines pins the height to 3 lines so the
        # pane footprint never reflows either.
        _mf = self.readout.font(); _mf.setFamily("Consolas"); _mf.setStyleHint(QtGui.QFont.Monospace)
        self.readout.setFont(_mf)
        from gottlux.app.uikit import reserve_lines
        reserve_lines(self.readout, 3)
        self.readout.setWordWrap(False)

        ctrl = QtWidgets.QWidget(); ctrl.setMinimumWidth(_CTRL_W); ctrl.setMaximumWidth(2 * _CTRL_W)
        cv = QtWidgets.QVBoxLayout(ctrl); cv.setContentsMargins(8, 4, 4, 4)
        cv.addLayout(_labeled("Mode", self.mode)); cv.addLayout(_labeled("Color", self.cmap))
        cv.addLayout(_labeled("Expr", self.expr)); cv.addLayout(_labeled("Scale", self.scale))
        cv.addStretch(1); cv.addWidget(self.readout)
        self.page_frame = QtWidgets.QWidget()
        pf = QtWidgets.QHBoxLayout(self.page_frame); pf.setContentsMargins(0, 0, 0, 0)
        pf.addWidget(self.glw, 1); pf.addWidget(ctrl, 0)

        # ----- page 1: a hosted function panel -----
        self.page_child = QtWidgets.QWidget()
        self.page_child_lay = QtWidgets.QVBoxLayout(self.page_child)
        self.page_child_lay.setContentsMargins(0, 0, 0, 0)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.page_frame); self.stack.addWidget(self.page_child)

        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(head); lay.addWidget(self.stack, 1)

        self._apply_cmap()
        self.ctl.cursorChanged.connect(self._render)
        self.ctl.accumChanged.connect(self._render)
        if self.filters is not None:
            self.filters.changed.connect(self._render)
        if default_view != "Event frame":
            self._on_view()

    # ------------------------------------------------------------------ loading
    def load_path(self, path, camera="cam0"):
        self.title.setText(f"Loading {os.path.basename(path)} …")
        self.busy.setText("decoding…"); self.busy.adjustSize()
        self._loader = RecordingLoader(path, camera=camera)
        self._loader.loaded.connect(self._on_loaded)
        self._loader.failed.connect(self._on_failed)
        self._loader.start()

    def set_recording(self, rec, label=None):
        self._on_loaded(rec, label=label)

    def _on_loaded(self, rec, label=None):
        self.rec = rec
        name = label or getattr(rec, "name", "clip")
        tag = "ROT" if getattr(rec, "is_rotating", False) else "STARE"
        self.title.setText(f"{name}\n{rec.width}×{rec.height} · {rec.duration_s:.2f}s · {rec.fmt} · {tag}")
        self.busy.setText("")
        self.vb.setRange(xRange=(0, rec.width), yRange=(0, rec.height), padding=0)
        self._child_clk.set_range(rec.t_start_s, rec.t_stop_s)
        if self._child is not None:
            try:
                self._child.set_recording(rec)
            except Exception:
                pass
        self.durationChanged.emit()
        self._render()

    def _on_failed(self, msg):
        self.title.setText("Load failed"); self.busy.setText("load failed"); self.busy.adjustSize()
        QtWidgets.QMessageBox.critical(self, "Clip load failed", msg)

    # ------------------------------------------------------------------ view selection
    def set_view(self, name):
        if name not in VIEWS:
            return
        b = QtCore.QSignalBlocker(self.view_sel); self.view_sel.setCurrentText(name); del b
        self._on_view()

    def _ensure_child(self, name):
        if self._child_type == name and self._child is not None:
            return
        if self._child is not None:
            self._child.setParent(None); self._child.deleteLater(); self._child = None
        self._child = _make_panel(name, self._child_clk, self.filters, compact=True)
        self._child_type = name
        if self._child is not None:
            if hasattr(self._child, "transport"):
                self._child.transport.hide()       # the slate's master transport drives time
            self.page_child_lay.addWidget(self._child)
            if self.rec is not None:
                self._child_clk.set_range(self.rec.t_start_s, self.rec.t_stop_s)
                try:
                    self._child.set_recording(self.rec)
                except Exception:
                    pass

    def _on_view(self, *_):
        name = self.view_sel.currentText()
        is_frame = name == "Event frame"
        if is_frame:
            self.stack.setCurrentIndex(0)
        else:
            self._ensure_child(name)
            self.stack.setCurrentIndex(1)
        self._render()
        if not is_frame and self._child is not None and hasattr(self._child, "sync"):
            self._child.sync()

    # ------------------------------------------------------------------ controls
    def _on_offset(self, v):
        self.offset = float(v); self.offsetChanged.emit(); self._render()

    def set_offset(self, v):
        b = QtCore.QSignalBlocker(self.offset_sp); self.offset_sp.setValue(float(v)); del b
        self.offset = float(v); self._render()

    def _on_scale(self, *_):
        if self.scale.currentText() == "static":
            self._static_vmax = None
        self._render()

    def _on_mode(self, *_):
        mode = self.mode.currentText(); cur = self.cmap.currentText()
        if mode in _POLARITY_MODES and cur not in _DIVERGING:
            b = QtCore.QSignalBlocker(self.cmap); self.cmap.setCurrentText("coolwarm"); del b
            self._apply_cmap(); return
        if mode not in _POLARITY_MODES and cur in _DIVERGING:
            b = QtCore.QSignalBlocker(self.cmap); self.cmap.setCurrentText("inferno"); del b
            self._apply_cmap(); return
        self._render()

    def _apply_cmap(self, *_):
        try:
            cm = pg.colormap.get(self.cmap.currentText(), source="matplotlib")
            self.img.setColorMap(cm)
        except Exception:
            pass
        self._render()

    @property
    def duration_s(self) -> float:
        return self.rec.duration_s if self.rec is not None else 0.0

    # ------------------------------------------------------------------ render
    def current_rgb(self):
        return self._last_rgb

    def sync(self):
        self._render(force=True)
        if self._child is not None and self.view_sel.currentText() != "Event frame" \
                and hasattr(self._child, "sync"):
            self._child.sync()

    def _render(self, *_, force=False):
        if self.rec is None or (not force and not self.isVisible()):
            return
        if self.view_sel.currentText() != "Event frame":
            # drive the hosted panel's child clock = master cursor + this clip's slate offset
            if self._child is None:
                return
            self._child_clk.set_accum(self.ctl.accum)
            desired = min(max(self.ctl.cursor + self.offset, self.rec.t_start_s), self.rec.t_stop_s)
            before = self._child_clk.cursor
            self._child_clk.set_cursor(desired)
            if self._child_clk.cursor == before and hasattr(self._child, "sync"):
                self._child.sync()        # cursor unchanged → still refresh (e.g. offset nudge)
            return
        # --- built-in event-frame view ---
        dt = self.ctl.accum
        local = self.ctl.cursor + self.offset
        dur = self.rec.t_stop_s
        if local < self.rec.t_start_s - 1e-9 or local > dur + 1e-9:
            self.busy.setText("— outside this clip —"); self.busy.adjustSize()
            self.img.setOpacity(0.25)
            self.readout.setText(f"local {local:.3f}s · (no events in window)")
            return
        self.busy.setText(""); self.img.setOpacity(1.0)
        from gottlux.core.render import render_frame
        t0 = min(max(local, self.rec.t_start_s), dur)
        mode = self.mode.currentText()
        static = self.scale.currentText() == "static"
        disp, levels, vmax, win = render_frame(
            self.rec, t0, dt, mode=mode, expr=self.expr.currentText(),
            filters=self.filters, vmax_ref=(self._static_vmax if static else None),
            back=self.ctl.accum_back)
        if static and vmax is not None:
            self._static_vmax = vmax
        self.img.setImage(disp, levels=levels, autoLevels=False)
        self._last_rgb = self._to_rgb(disp, levels)
        on, off = win.polarity_split()
        rate = win.n / dt if dt > 0 else 0.0
        # Fixed-width columns + one metric group per line (see __init__): values change in place,
        # never re-wrapping. Right-justified so the units/arrows stay put as the digits grow.
        self.readout.setText(
            f"t={t0:.3f}/{dur:.3f}s\n"
            f"{win.n:>10,} ev   {rate/1e6:6.2f} Mev/s\n"
            f"{int(on.sum()):>9,} ↑  {int(off.sum()):>9,} ↓")

    def _to_rgb(self, disp, levels):
        try:
            cm = pg.colormap.get(self.cmap.currentText(), source="matplotlib")
            lut = cm.getLookupTable(nPts=256, alpha=False)
            lo, hi = levels
            idx = np.clip((np.asarray(disp) - lo) / (hi - lo + 1e-12) * 255, 0, 255).astype(np.uint8)
            return lut[idx]
        except Exception:
            return None

    def intensity_frame(self, dt=None):
        """Normalized ``[0, 1]`` event-count frame at the current cursor + this clip's offset, for
        the Overlay layout (computed on demand, independent of which mode/colormap the pane shows
        and of whether the pane is visible). ``None`` if the cursor is outside this clip."""
        if self.rec is None:
            return None
        from gottlux.core.render import render_frame
        dt = self.ctl.accum if dt is None else float(dt)
        local = self.ctl.cursor + self.offset
        if local < self.rec.t_start_s - 1e-9 or local > self.rec.t_stop_s + 1e-9:
            return None
        t0 = min(max(local, self.rec.t_start_s), self.rec.t_stop_s)
        disp, _levels, _v, _w = render_frame(self.rec, t0, dt, mode="count", expr="sqrt",
                                             filters=self.filters, back=self.ctl.accum_back)
        return np.clip(disp, 0.0, 1.0)


# ====================================================================================
# The slate: many clips on one shared clock, each renderable through any function
# ====================================================================================
class MultiClipViewer(QtWidgets.QWidget):
    """Load several recordings and view them side-by-side through any function, on one clock."""

    roiChanged = QtCore.Signal(object)

    def __init__(self, app_clock=None, filters=None, parent=None):
        super().__init__(parent)
        self.clock = TimeController(self)
        self.clock.set_loop(True)
        self.filters = filters
        self.panes: list[ClipPane] = []
        self._default_view = "Event frame"

        bar = QtWidgets.QHBoxLayout()
        self.add_file_btn = QtWidgets.QPushButton("Add clip…")
        self.add_file_btn.setIcon(icons.icon("add"))
        self.add_file_btn.clicked.connect(self.add_file_dialog)
        self.add_folder_btn = QtWidgets.QPushButton("Add folder…")
        self.add_folder_btn.setIcon(icons.icon("add"))
        self.add_folder_btn.clicked.connect(self.add_folder_dialog)
        self.add_capture_btn = QtWidgets.QPushButton("Add capture (cam0+cam1)")
        self.add_capture_btn.setIcon(icons.icon("add"))
        self.add_capture_btn.clicked.connect(self.add_capture_dialog)

        self.view_cb = QtWidgets.QComboBox(); self.view_cb.addItems(VIEWS)
        self.view_cb.setToolTip("Render EVERY clip through this function (multi-clip event frame / "
                                "Space-time / Event-rate / Range lab / Workbench / Sandbox). Each "
                                "pane can still be overridden individually with its own View selector.")
        self.view_cb.currentTextChanged.connect(self._apply_view_all)

        self.layout_cb = QtWidgets.QComboBox()
        self.layout_cb.addItems(["Stacked", "Side-by-side", "Grid", "Overlay"])
        self.layout_cb.setToolTip("Stacked (default), Side-by-side (one row), a near-square Grid, "
                                  "or Overlay — superimpose every clip's events in one image, each "
                                  "clip a distinct colour (e.g. wide vs narrow sensor).")
        self.layout_cb.currentIndexChanged.connect(self._relayout)
        self.align_btn = QtWidgets.QPushButton("⇆ Align starts")
        self.align_btn.clicked.connect(self._align_starts)
        self.loop_chk = QtWidgets.QCheckBox("Loop"); self.loop_chk.setChecked(True)
        self.loop_chk.setToolTip("Loop playback: restart from the beginning at the end of the clip.")
        self.loop_chk.toggled.connect(self.clock.set_loop)
        self.fuse_btn = QtWidgets.QPushButton("⊕ Fuse keyframes…")
        self.fuse_btn.setToolTip("Converge a pixels-on-target / perception-range study across the "
                                 "clips: set ≥2 panes' View to 'Range lab', keyframe the target in "
                                 "each (with known ranges), then fuse the two sensor spaces.")
        self.fuse_btn.clicked.connect(self._fuse_keyframes)
        self.save_btn = QtWidgets.QPushButton("Save slate…")
        self.save_btn.setIcon(icons.icon("export"))
        self.save_btn.setToolTip("Save the current event-frame composite as a PNG figure.")
        self.save_btn.clicked.connect(self._save_composite)

        bar.addWidget(self.add_file_btn); bar.addWidget(self.add_folder_btn)
        bar.addWidget(self.add_capture_btn)
        bar.addSpacing(10)
        bar.addWidget(QtWidgets.QLabel("View")); bar.addWidget(self.view_cb)
        bar.addWidget(QtWidgets.QLabel("Layout")); bar.addWidget(self.layout_cb)
        bar.addWidget(self.align_btn); bar.addWidget(self.loop_chk)
        bar.addStretch(1); bar.addWidget(self.fuse_btn); bar.addWidget(self.save_btn)

        self.container = QtWidgets.QWidget()
        self.container_lay = QtWidgets.QVBoxLayout(self.container)
        self.container_lay.setContentsMargins(0, 0, 0, 0)
        self.hint = QtWidgets.QLabel(
            "No clips yet.\n\nUse  Add clip…  to load recordings side-by-side. Pick a View "
            "(event frame, Space-time, Event-rate, Range lab, …) to render every clip through that "
            "function; they share one clock, so they scrub and accumulate together. Give a clip a "
            "slate Offset to line it up. Press Play to run it (loops by default).")
        self.hint.setAlignment(QtCore.Qt.AlignCenter); self.hint.setObjectName("muted")
        self.container_lay.addWidget(self.hint)

        # --- Overlay layout: one image blending every clip's events in distinct colours ---
        self._overlay_glw = pg.GraphicsLayoutWidget()
        self._overlay_vb = self._overlay_glw.addViewBox(lockAspect=True, invertY=True)
        self._overlay_img = pg.ImageItem(axisOrder="row-major")
        self._overlay_vb.addItem(self._overlay_img)
        self._overlay_legend = QtWidgets.QLabel(""); self._overlay_legend.setTextFormat(QtCore.Qt.RichText)
        self._overlay_wrap = QtWidgets.QWidget()
        _owl = QtWidgets.QVBoxLayout(self._overlay_wrap); _owl.setContentsMargins(0, 0, 0, 0)
        _owl.addWidget(self._overlay_glw, 1); _owl.addWidget(self._overlay_legend)
        self._overlay_last_rgb = None
        self.clock.cursorChanged.connect(self._maybe_overlay)
        self.clock.accumChanged.connect(self._maybe_overlay)
        if self.filters is not None:
            self.filters.changed.connect(self._maybe_overlay)

        self.transport = TransportBar(self.clock, host=self)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(bar); lay.addWidget(self.container, 1); lay.addWidget(self.transport)

    # ------------------------------------------------------------------ panel protocol
    def set_recording(self, rec):
        if rec is None:
            return
        if self.panes:
            self.panes[0].set_recording(rec)
        else:
            self._add_pane_with(rec)
        self._recompute_range()

    def sync(self):
        for p in self.panes:
            p.sync()

    def showEvent(self, ev):
        super().showEvent(ev)
        for p in self.panes:
            p.sync()

    def hideEvent(self, ev):
        super().hideEvent(ev)
        self.clock.pause()

    # ------------------------------------------------------------------ add clips
    def _new_pane(self) -> ClipPane:
        pane = ClipPane(self.clock, filters=self.filters, default_view=self._default_view)
        pane.removeRequested.connect(self._remove_pane)
        pane.durationChanged.connect(self._recompute_range)
        pane.offsetChanged.connect(self._recompute_range)
        self.panes.append(pane)
        self._relayout()
        return pane

    def _add_pane_with(self, rec, label=None):
        pane = self._new_pane(); pane.set_recording(rec, label=label)
        self._recompute_range(); return pane

    def add_path(self, path, camera="cam0"):
        pane = self._new_pane(); pane.load_path(path, camera=camera); return pane

    def add_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Add clip", "", "Event recordings (*.raw *.h5 *.hdf5 *.meta.json);;All files (*)")
        if path:
            self.add_path(path)

    def add_folder_dialog(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Add capture folder (one camera)")
        if path:
            self.add_path(path, camera="cam0")

    def add_capture_dialog(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Add capture folder (both cameras)")
        if path:
            self.add_path(path, camera="cam0")
            self.add_path(path, camera="cam1")

    def _remove_pane(self, pane):
        if pane in self.panes:
            self.panes.remove(pane)
            pane.setParent(None); pane.deleteLater()
            self._relayout(); self._recompute_range()

    # ------------------------------------------------------------------ view / timeline / layout
    def _apply_view_all(self, name):
        self._default_view = name
        for p in self.panes:
            p.set_view(name)

    def _recompute_range(self):
        spans = [p.duration_s - p.offset for p in self.panes if p.rec is not None]
        t1 = max(spans) if spans else 1.0
        self.clock.set_range(0.0, max(t1, 1e-3))

    def _align_starts(self):
        for p in self.panes:
            p.set_offset(0.0)
        self._recompute_range()

    def _cols(self) -> int:
        n = max(1, len(self.panes))
        which = self.layout_cb.currentText()
        if which == "Stacked":
            return 1
        if which == "Side-by-side":
            return n
        return max(1, int(np.ceil(np.sqrt(n))))

    def _relayout(self, *_):
        while self.container_lay.count():
            item = self.container_lay.takeAt(0)
            w = item.widget()
            if w is not None and w is not self.hint:
                w.setParent(None)
        if not self.panes:
            self.hint.setParent(self.container); self.container_lay.addWidget(self.hint); self.hint.show()
            return
        self.hint.hide()
        if self.layout_cb.currentText() == "Overlay":
            self.container_lay.addWidget(self._overlay_wrap); self._overlay_wrap.show()
            self._render_overlay()
            return
        cols = self._cols()
        rows_split = QtWidgets.QSplitter(QtCore.Qt.Vertical); rows_split.setChildrenCollapsible(False)
        row_splits = []
        row_split = None
        for i, pane in enumerate(self.panes):
            if i % cols == 0:
                row_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal); row_split.setChildrenCollapsible(False)
                rows_split.addWidget(row_split); row_splits.append(row_split)
            row_split.addWidget(pane); pane.show()
        # Hand every pane an equal share *explicitly*. A freshly-built nested QSplitter sizes
        # its children from their size hints against its own current width — which is ~0 at
        # build time — so a just-added pane was being given near-zero height/width and only
        # popped back after a resize/maximize forced a redistribution. This was the
        # "added clips disappear until I shrink then full-screen the window" bug.
        for rs in row_splits:
            n = max(rs.count(), 1)
            rs.setSizes([10_000 // n] * n)
            for j in range(rs.count()):
                rs.setStretchFactor(j, 1)
        nrows = max(rows_split.count(), 1)
        rows_split.setSizes([10_000 // nrows] * nrows)
        for r in range(rows_split.count()):
            rows_split.setStretchFactor(r, 1)
        self.container_lay.addWidget(rows_split)

    # ------------------------------------------------------------------ overlay layout
    def _maybe_overlay(self, *_):
        if self.layout_cb.currentText() == "Overlay" and self.panes:
            self._render_overlay()

    def _render_overlay(self):
        """Blend every clip's current event frame into one image, each clip a distinct colour."""
        from gottlux.viz.video import overlay_frames, OVERLAY_COLORS
        frames, names = [], []
        for p in self.panes:
            f = p.intensity_frame()
            if f is not None:
                frames.append(f); names.append(p.rec.name if p.rec else "clip")
        if not frames:
            return
        rgb = overlay_frames(frames)
        self._overlay_last_rgb = rgb
        self._overlay_img.setImage(rgb, autoLevels=False)
        self._overlay_legend.setText("  ".join(
            f"<span style='color:rgb{OVERLAY_COLORS[i % len(OVERLAY_COLORS)]}'>■</span> {n}"
            for i, n in enumerate(names)))

    # ------------------------------------------------------------------ faithful capture
    def capture_clock(self):
        """The Multi-clip slate runs on its own clock (not the app clock) — Capture uses this."""
        return self.clock

    def sensor_size(self):
        sizes = [(p.rec.width, p.rec.height) for p in self.panes if p.rec is not None]
        if not sizes:
            return None
        return (max(w for w, _ in sizes), max(h for _, h in sizes))

    def capture_frame(self, t, dt=None, size=None):
        """Faithful RGB of the **overlay composite** (every clip's events, distinct colours) at
        time *t*, at any resolution — the compelling 'combined sensor spaces' high-res capture."""
        from gottlux.viz.video import overlay_frames
        self.clock.set_cursor(float(t))                  # drive the slate's own clock to t
        frames = [p.intensity_frame(dt) for p in self.panes]
        frames = [f for f in frames if f is not None]
        if not frames:
            return None
        return overlay_frames(frames, size=size)

    # ------------------------------------------------------------------ keyframe fusion
    def _range_lab_panes(self):
        """Panes whose View is 'Range lab' and that have keyframes — ``[(name, RangeLab)]``."""
        out = []
        for p in self.panes:
            child = getattr(p, "_child", None)
            if (p.view_sel.currentText() == "Range lab" and child is not None
                    and getattr(child, "keyframes", None)):
                out.append((p.rec.name if p.rec else "clip", child))
        return out

    def _fuse_keyframes(self):
        """Converge a pixels-on-target study across ≥2 Range-lab panes (the 'add-clip' route)."""
        labs = self._range_lab_panes()
        if len(labs) < 2:
            QtWidgets.QMessageBox.information(
                self, "Fuse keyframes",
                "Set at least two panes' View to 'Range lab', keyframe the target in each "
                "(with known ranges), then fuse the two sensor spaces.")
            return
        parent = QtWidgets.QFileDialog.getExistingDirectory(self, "Export converged study to folder")
        if not parent:
            return
        from gottlux import sensors
        from gottlux.core.dualview import ConvergedStudy
        from gottlux.io import export as _export
        from gottlux.io.paths import open_in_file_browser, unique_export_dir
        from gottlux.run.resolution_report import save_resolution_study
        d = unique_export_dir(parent, "converged_study", "fuse")
        try:
            written, studies = [], []
            for name, lab in labs:                          # each clip → its own bundle, one folder
                study = lab._study()
                studies.append((name, study))
                written += save_resolution_study(os.path.join(d, name), study,
                                                 title=f"Pixels on target — {name}",
                                                 profile=sensors.get(sensors.DEFAULT_PROFILE))
            (na, sa), (nb, sb) = studies[0], studies[1]      # converge the wide + narrow pair
            cs = ConvergedStudy(na, sa, nb, sb)
            written += _export.save_json(cs.summary(), os.path.join(d, "converged_study.json"))
            open_in_file_browser(d)
            QtWidgets.QMessageBox.information(
                self, "Fuse keyframes",
                f"Converged {len(studies)} clips ({na} + {nb}) → {len(written)} file(s):\n{d}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Fuse keyframes", str(e))

    # ------------------------------------------------------------------ composite export
    def _save_composite(self):
        rgbs = [p.current_rgb() for p in self.panes if p.current_rgb() is not None]
        if not rgbs:
            QtWidgets.QMessageBox.information(self, "Save slate",
                                             "Nothing rendered to save yet (event-frame view only).")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save slate composite", "slate.png",
                                                        "PNG (*.png)")
        if not path:
            return
        stacked = self.layout_cb.currentText() != "Side-by-side"
        comp = self._compose(rgbs, vertical=stacked)
        try:
            from PIL import Image
            Image.fromarray(comp).save(path)
            from gottlux.io.paths import open_in_file_browser
            open_in_file_browser(os.path.dirname(os.path.abspath(path)))         # reveal on export
            QtWidgets.QMessageBox.information(self, "Save slate", f"Saved:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save slate", str(e))

    @staticmethod
    def _compose(rgbs, vertical=True):
        if vertical:
            wmax = max(r.shape[1] for r in rgbs)
            parts = []
            for r in rgbs:
                if r.shape[1] < wmax:
                    r = np.hstack([r, np.zeros((r.shape[0], wmax - r.shape[1], 3), np.uint8)])
                parts.append(r); parts.append(np.zeros((4, wmax, 3), np.uint8))
            return np.vstack(parts[:-1]) if len(parts) > 1 else parts[0]
        hmax = max(r.shape[0] for r in rgbs)
        parts = []
        for r in rgbs:
            if r.shape[0] < hmax:
                r = np.vstack([r, np.zeros((hmax - r.shape[0], r.shape[1], 3), np.uint8)])
            parts.append(r); parts.append(np.zeros((hmax, 4, 3), np.uint8))
        return np.hstack(parts[:-1]) if len(parts) > 1 else parts[0]
