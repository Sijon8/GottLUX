"""
transport.py — one shared clock for the whole program.

Every visualization in gottlux looks at the *same recording at the same moment*, so the
cursor time, the accumulation window, and the play/pause state are held in a single
:class:`TimeController` that every panel subscribes to. Seek (or change the accumulation
time, or hit play) on any tab and every other tab follows — that is what makes "seek a
moment" and "accumulation time across the entire program" consistent rather than per-widget.

:class:`TransportBar` is the reusable strip of controls (play · seek · time · speed ·
accumulation) that each visualization embeds; many bars can bind to one controller and they
all stay in lock-step.
"""
from __future__ import annotations

import time

from PySide6 import QtCore, QtGui, QtWidgets

from gottlux.app import icons

# Quick-pick playback frame rates offered in the FPS dropdown.
# The field stays editable, so any value up to 10000 can also be typed.
FPS_PRESETS = [0.5, 1, 2, 5, 10, 15, 24, 30, 48, 60, 120, 240, 500, 1000, 2000, 5000, 10000,
               20000, 50000, 100000]

# The FPS control models an *equivalent* (slow-motion-camera) frame rate, not an on-screen
# refresh: footage "shot" at this many fps and watched at a normal viewing cadence. So FPS is
# the inverse of speed — REALTIME_FPS plays at real time, higher FPS plays slower (slow-motion),
# lower FPS plays faster than real time. The recording advances REALTIME_FPS / FPS recording-
# seconds per wall-second. Accumulation (exposure) is a separate, independent control.
REALTIME_FPS = 30.0


def _speed_factor_text(fps: float) -> str:
    """Human label for the playback speed implied by an equivalent FPS (30 fps = real time)."""
    sm = fps / REALTIME_FPS
    if abs(sm - 1.0) < 1e-3:
        return "real-time"
    if sm > 1.0:                                  # slower than real time
        return f"{sm:.0f}× slow" if sm >= 10 else f"{sm:.1f}× slow"
    fast = 1.0 / sm                               # faster than real time
    return f"{fast:.0f}× fast" if fast >= 10 else f"{fast:.1f}× fast"


class TimeController(QtCore.QObject):
    """Shared playback clock: cursor time, accumulation window, speed, and play state."""

    cursorChanged = QtCore.Signal(float)      # current time (s)
    accumChanged = QtCore.Signal(float)       # accumulation window (s)
    accumDirChanged = QtCore.Signal(bool)     # accumulation direction (True = behind the cursor)
    fpsChanged = QtCore.Signal(float)         # playback frame rate (frames / wall-second)
    rangeChanged = QtCore.Signal(float, float)
    playStateChanged = QtCore.Signal(bool)
    selectionChanged = QtCore.Signal(float, float)   # In/Out selection (s) — for cut/capture/export
    highlightsChanged = QtCore.Signal()              # the list of highlight ranges changed

    def __init__(self, parent=None):
        super().__init__(parent)
        self._t0 = 0.0
        self._t1 = 1.0
        self._cursor = 0.0
        self._accum = 0.02
        self._accum_back = False              # False = integrate ahead [t, t+Δ]; True = behind [t−Δ, t]
        self._fps = 30.0
        self._playing = False
        self._loop = True                     # when True, playback wraps to t0 instead of stopping
        self._sel_lo = 0.0                    # In/Out selection as fractions of the range (0..1)
        self._sel_hi = 1.0
        self._highlights = []                 # extra In/Out ranges [[lo, hi], ...] to merge into one .raw
        self._last_wall = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(20)           # ~50 Hz wall clock
        self._timer.timeout.connect(self._tick)

    # ----- range -----
    def set_range(self, t0, t1):
        self._t0, self._t1 = float(t0), float(t1)
        self._cursor = min(max(self._cursor, t0), t1)
        self.rangeChanged.emit(self._t0, self._t1)
        self.cursorChanged.emit(self._cursor)

    @property
    def t0(self): return self._t0

    @property
    def t1(self): return self._t1

    @property
    def duration(self): return max(self._t1 - self._t0, 1e-9)

    # ----- cursor -----
    @property
    def cursor(self): return self._cursor

    def set_cursor(self, t, *, emit=True):
        t = min(max(float(t), self._t0), self._t1)
        if t != self._cursor:
            self._cursor = t
            if emit:
                self.cursorChanged.emit(t)

    def set_fraction(self, frac):
        self.set_cursor(self._t0 + float(frac) * self.duration)

    def fraction(self):
        return (self._cursor - self._t0) / self.duration

    # ----- In/Out selection (a sub-range of the recording for cut / capture / export) -----
    @property
    def sel_lo(self): return self._sel_lo

    @property
    def sel_hi(self): return self._sel_hi

    def set_selection(self, lo, hi):
        """Set the In/Out selection as fractions (0..1) of the recording range."""
        lo, hi = max(0.0, min(float(lo), float(hi))), min(1.0, max(float(lo), float(hi)))
        if (lo, hi) != (self._sel_lo, self._sel_hi):
            self._sel_lo, self._sel_hi = lo, hi
            self.selectionChanged.emit(self.sel_t0(), self.sel_t1())

    def sel_t0(self) -> float:
        """In point in seconds."""
        return self._t0 + self._sel_lo * self.duration

    def sel_t1(self) -> float:
        """Out point in seconds."""
        return self._t0 + self._sel_hi * self.duration

    def has_selection(self) -> bool:
        """True if the selection is a strict sub-range (not the whole recording)."""
        return self._sel_lo > 1e-4 or self._sel_hi < 1.0 - 1e-4

    # ----- highlights (extra In/Out ranges, e.g. two moments to merge into one .raw) -----
    @property
    def highlights(self):
        """The highlight ranges as ``(lo, hi)`` fraction tuples (copies)."""
        return [(lo, hi) for lo, hi in self._highlights]

    def highlight_times(self):
        """Highlight ranges as ``(t0, t1)`` seconds, in playback (time) order."""
        spans = [(self._t0 + lo * self.duration, self._t0 + hi * self.duration)
                 for lo, hi in self._highlights]
        return sorted(spans)

    def add_highlight(self, lo=None, hi=None) -> int:
        """Add a highlight (defaults to the current In/Out selection). Returns its index."""
        if lo is None or hi is None:
            lo, hi = self._sel_lo, self._sel_hi
        lo, hi = max(0.0, min(float(lo), float(hi))), min(1.0, max(float(lo), float(hi)))
        self._highlights.append([lo, hi])
        self.highlightsChanged.emit()
        return len(self._highlights) - 1

    def set_highlight(self, i, lo, hi):
        if 0 <= i < len(self._highlights):
            lo, hi = max(0.0, min(float(lo), float(hi))), min(1.0, max(float(lo), float(hi)))
            self._highlights[i] = [lo, hi]
            self.highlightsChanged.emit()

    def remove_highlight(self, i):
        if 0 <= i < len(self._highlights):
            del self._highlights[i]
            self.highlightsChanged.emit()

    def clear_highlights(self):
        if self._highlights:
            self._highlights = []
            self.highlightsChanged.emit()

    # ----- accumulation -----
    @property
    def accum(self): return self._accum

    def set_accum(self, dt):
        dt = max(float(dt), 1e-6)            # down to 1 µs (high-speed-camera exposure)
        if dt != self._accum:
            self._accum = dt
            self.accumChanged.emit(dt)

    # ----- accumulation direction -----
    # Forward (default): the window opens AHEAD of the cursor, [t, t+Δ] — the cursor marks the
    # start ("now") and events integrate into the future. Backward: the window trails BEHIND the
    # cursor, [t−Δ, t] — the cursor is the leading edge and the exposure reaches into the past.
    # The integration time Δ is the same either way; only which side of the cursor it covers flips.
    @property
    def accum_back(self): return self._accum_back

    def set_accum_back(self, back):
        back = bool(back)
        if back != self._accum_back:
            self._accum_back = back
            self.accumDirChanged.emit(back)
            self.accumChanged.emit(self._accum)     # re-render every bound view at the new window

    def accum_window(self, t=None, dt=None, back=None):
        """Resolve the accumulation window ``(t0, t1)`` at time *t* (default: the cursor), honouring
        the integration direction. ``dt``/``back`` default to the controller's current values.
        Clamped to the controller's ``[t0, t1]`` range so it never reads past the recording."""
        t = self._cursor if t is None else float(t)
        dt = self._accum if dt is None else max(float(dt), 1e-9)
        back = self._accum_back if back is None else bool(back)
        if back:
            return max(self._t0, t - dt), min(self._t1, t)
        return max(self._t0, t), min(self._t1, t + dt)

    # ----- playback rate (equivalent FPS) -----
    # The FPS value is an *equivalent capture frame rate*, like a high-speed camera: footage
    # shot at FPS frames/second and watched at a normal viewing cadence. So it is the inverse
    # of playback speed — at REALTIME_FPS it plays at real time; a higher FPS plays *slower*
    # (slow-motion: 10000 fps ≈ 333× slow), a lower FPS plays faster than real time. The
    # accumulation window (exposure) is independent and does not affect playback speed.
    @property
    def fps(self): return self._fps

    def set_fps(self, f):
        f = float(max(f, 0.1))
        if f != self._fps:
            self._fps = f
            self.fpsChanged.emit(f)

    @property
    def speed(self):
        """Recording-seconds advanced per wall-second (= REALTIME_FPS / equivalent FPS)."""
        return REALTIME_FPS / self._fps

    @property
    def slowmo(self):
        """Slow-motion factor (>1 = slower than real time, <1 = faster). = FPS / REALTIME_FPS."""
        return self._fps / REALTIME_FPS

    # ----- play / pause -----
    @property
    def playing(self): return self._playing

    def play(self):
        if not self._playing:
            self._playing = True
            self._last_wall = time.perf_counter()
            self._timer.start()
            self.playStateChanged.emit(True)

    def pause(self):
        if self._playing:
            self._playing = False
            self._timer.stop()
            self.playStateChanged.emit(False)

    def toggle(self):
        self.pause() if self._playing else self.play()

    # ----- loop -----
    @property
    def loop(self): return self._loop

    def set_loop(self, on):
        """When on, playback wraps back to the start at the end of the clip instead of stopping."""
        self._loop = bool(on)

    def _tick(self):
        now = time.perf_counter()
        dt = now - (self._last_wall or now)
        self._last_wall = now
        nxt = self._cursor + self.speed * dt
        if nxt >= self._t1:
            if self._loop:
                span = self.duration                       # wrap, carrying any overshoot
                over = (nxt - self._t1) % span if span > 0 else 0.0
                self.set_cursor(self._t0 + over)
            else:
                self.set_cursor(self._t1)
                self.pause()
        else:
            self.set_cursor(nxt)


class TransportBar(QtWidgets.QWidget):
    """Play · seek · time · speed · accumulation, bound to a shared :class:`TimeController`.

    Embed one in every visualization. ``show_accum``/``show_speed`` can hide those fields for
    a view that does not use them, but by default both are shown so accumulation time is
    adjustable from anywhere in the program.
    """

    _ACC_LO, _ACC_HI = 1e-5, 2.0         # accumulation slider range (s), log-scaled (10 µs … 2 s)

    def _acc_to_slider(self, dt):
        import math
        dt = min(max(float(dt), self._ACC_LO), self._ACC_HI)
        f = (math.log10(dt) - math.log10(self._ACC_LO)) / \
            (math.log10(self._ACC_HI) - math.log10(self._ACC_LO))
        return int(round(f * 1000))

    def _slider_to_acc(self, v):
        import math
        f = v / 1000.0
        return 10 ** (math.log10(self._ACC_LO) +
                      f * (math.log10(self._ACC_HI) - math.log10(self._ACC_LO)))

    def __init__(self, controller: TimeController, show_accum=True, show_speed=True,
                 show_selection=True, show_accum_dir=True, host=None, parent=None):
        super().__init__(parent)
        self.ctl = controller
        self.host = host                 # the panel that embeds this bar (for rec + capture path)

        self.play_btn = QtWidgets.QToolButton()
        self.play_btn.setIcon(icons.icon("play"))
        self.play_btn.setIconSize(QtCore.QSize(16, 16))
        self.play_btn.setToolTip("Play / pause (live view)")
        # a FIXED size so toggling play <-> pause never changes the button's geometry
        self.play_btn.setFixedSize(36, 28)
        self.play_btn.clicked.connect(self.ctl.toggle)

        self.seek = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.seek.setRange(0, 1000)
        self.seek.setToolTip("Seek to any moment in the recording")
        self.seek.sliderMoved.connect(self._on_seek)
        self.seek.valueChanged.connect(self._on_seek_value)

        self.time_lbl = QtWidgets.QLabel("0.000 / 0.000 s")
        self.time_lbl.setMinimumWidth(120)
        self.time_lbl.setAlignment(QtCore.Qt.AlignCenter)

        # Two co-aligned rows in a grid: the In/Out + highlight bar sits directly ABOVE the seek
        # slider and shares its track column, so the cut/crop marks and the playhead line up by
        # eye with the timeline — no need to read the time numbers to position them.
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2); outer.setSpacing(2)
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6); grid.setVerticalSpacing(2)
        grid.setColumnStretch(1, 1)                  # the shared slider column stretches

        self.selbar = None
        if show_selection:
            from gottlux.app.timeline import HighlightRangeBar
            self.selbar = HighlightRangeBar(); self.selbar.setEnabled(True)
            self.selbar.set_primary(self.ctl.sel_lo, self.ctl.sel_hi)
            self.selbar.set_highlights(self.ctl.highlights)
            self.selbar.set_cursor_frac(self.ctl.fraction())
            self.selbar.primaryChanged.connect(self._on_selection)
            self.selbar.highlightChanged.connect(
                lambda i, lo, hi: self.ctl.set_highlight(i, lo, hi))
            self.selbar.highlightRemoved.connect(self.ctl.remove_highlight)
            self.sel_lbl = QtWidgets.QLabel("[ full ]"); self.sel_lbl.setObjectName("muted")
            self.sel_lbl.setMinimumWidth(150)
            sel_right = QtWidgets.QHBoxLayout(); sel_right.setContentsMargins(0, 0, 0, 0)
            sel_right.addWidget(self.sel_lbl); sel_right.addWidget(self._build_edit_controls())
            sel_right.addStretch(1)
            sel_right_w = QtWidgets.QWidget(); sel_right_w.setLayout(sel_right)
            in_out = QtWidgets.QLabel("In/Out"); in_out.setToolTip(
                "Drag the handles to select a portion of the recording (the In/Out range) for "
                "Save → .raw, MP4, or Capture.")
            grid.addWidget(in_out, 0, 0)
            grid.addWidget(self.selbar, 0, 1)
            grid.addWidget(sel_right_w, 0, 2)
            self.ctl.selectionChanged.connect(self._reflect_selection)
            self.ctl.highlightsChanged.connect(self._reflect_highlights)

        # right-hand cluster of the playback row: time · FPS · accumulation
        play_right = QtWidgets.QHBoxLayout(); play_right.setContentsMargins(0, 0, 0, 0)
        play_right.addWidget(self.time_lbl)

        if show_speed:
            self.fps = QtWidgets.QComboBox()
            self.fps.setEditable(True)
            self.fps.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
            self.fps.addItems([f"{v:g}" for v in FPS_PRESETS])
            self.fps.setValidator(QtGui.QDoubleValidator(0.1, 100000.0, 3, self.fps))
            self.fps.setCurrentText(f"{self.ctl.fps:g}")
            self.fps.setFixedWidth(88)
            self.fps.setToolTip(
                "Equivalent FPS — like a slow-motion camera's capture rate. Pick a preset or "
                "type any value up to 100000. 30 fps plays at real time; HIGHER fps plays "
                "SLOWER (100000 fps ≈ 3333× slow-motion), lower fps plays faster than real time. "
                "The recording advances 30 ÷ FPS recording-seconds per real second. "
                "Accumulation (exposure) is set separately — to match a high-speed camera's time "
                "resolution, set Accum to 1/FPS (e.g. 100000 fps → 10 µs).")
            self.fps.activated.connect(self._fps_from_combo)
            self.fps.lineEdit().editingFinished.connect(self._fps_from_combo)
            self.ctl.fpsChanged.connect(self._reflect_fps)
            self.fps_factor = QtWidgets.QLabel("")
            self.fps_factor.setObjectName("muted")
            self.fps_factor.setMinimumWidth(64)
            self.fps_factor.setToolTip("Resulting playback speed (30 fps = real time; "
                                       "higher = slower slow-motion).")
            play_right.addWidget(QtWidgets.QLabel("FPS"))
            play_right.addWidget(self.fps)
            play_right.addWidget(self.fps_factor)
            self._reflect_fps(self.ctl.fps)
        if show_accum:
            tip = ("Accumulation window (integration time / exposure) in seconds — how much time "
                   "is summed into one frame/slab. Shared across the whole program. "
                   "Log slider spans 10 µs … 2 s (set to 1/FPS to match a high-speed camera's "
                   "time resolution).")
            self.accum_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.accum_slider.setRange(0, 1000)
            self.accum_slider.setFixedWidth(110)
            self.accum_slider.setToolTip(tip)
            self.accum = QtWidgets.QDoubleSpinBox()
            self.accum.setRange(self._ACC_LO, self._ACC_HI)
            self.accum.setDecimals(5)
            self.accum.setSingleStep(0.001)
            self.accum.setSuffix(" s")
            self.accum.setValue(self.ctl.accum)
            self.accum.setToolTip(tip)
            self.accum_slider.setValue(self._acc_to_slider(self.ctl.accum))
            self.accum_slider.valueChanged.connect(
                lambda v: self.ctl.set_accum(self._slider_to_acc(v)))
            self.accum.valueChanged.connect(self.ctl.set_accum)
            lab = QtWidgets.QLabel("Accum"); lab.setToolTip(tip)
            play_right.addWidget(lab)
            if show_accum_dir:
                self.accum_dir_btn = QtWidgets.QToolButton()
                self.accum_dir_btn.setCheckable(True)
                self.accum_dir_btn.setAutoRaise(True)
                self.accum_dir_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
                self.accum_dir_btn.setIcon(icons.icon("arrow-right"))
                # fixed width across both labels so toggling never jitters the row
                icons.freeze_width(self.accum_dir_btn, ["behind", "ahead"])
                self.accum_dir_btn.setChecked(self.ctl.accum_back)
                self.accum_dir_btn.toggled.connect(self.ctl.set_accum_back)
                self.ctl.accumDirChanged.connect(self._reflect_accum_dir)
                self._reflect_accum_dir(self.ctl.accum_back)
                play_right.addWidget(self.accum_dir_btn)
            play_right.addWidget(self.accum_slider)
            play_right.addWidget(self.accum)
        play_right.addStretch(1)
        play_right_w = QtWidgets.QWidget(); play_right_w.setLayout(play_right)

        grid.addWidget(self.play_btn, 1, 0)
        grid.addWidget(self.seek, 1, 1)
        grid.addWidget(play_right_w, 1, 2)
        outer.addLayout(grid)

        # match the In/Out bar's track inset to the seek slider's half-handle, and mirror the
        # playhead onto it, so marks and the live cursor line up across both bars.
        if self.selbar is not None:
            hw = self.seek.style().pixelMetric(
                QtWidgets.QStyle.PixelMetric.PM_SliderLength, None, self.seek)
            self.selbar.set_inset(max(2, hw // 2))
            self.ctl.cursorChanged.connect(
                lambda *_: self.selbar.set_cursor_frac(self.ctl.fraction()))
            self.ctl.rangeChanged.connect(
                lambda *_: self.selbar.set_cursor_frac(self.ctl.fraction()))

        # reflect controller -> widgets (keep many bars in sync)
        self.ctl.cursorChanged.connect(self._reflect_cursor)
        self.ctl.rangeChanged.connect(lambda *_: self._reflect_cursor(self.ctl.cursor))
        self.ctl.accumChanged.connect(self._reflect_accum)
        self.ctl.playStateChanged.connect(self._reflect_play)

    # ----- selection (In/Out) + highlights -----
    def _build_edit_controls(self):
        """The inline edit/export strip on the selection row: Add · Save · MP4 · Clear.

        These live in the transport itself so they work in *whatever* viewing tab you are in,
        with no trip up to the top toolbar. The recording and the faithful-render path come
        from the host panel (set via ``host=``)."""
        box = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(box); row.setContentsMargins(6, 0, 0, 0); row.setSpacing(3)

        self.add_hl_btn = QtWidgets.QToolButton(); self.add_hl_btn.setText("Add")
        self.add_hl_btn.setIcon(icons.icon("add"))
        self.add_hl_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.add_hl_btn.setToolTip("Freeze the current In/Out range as a colored highlight band. "
                                   "Add as many as you like, then Save → Merge into one .raw.")
        self.add_hl_btn.clicked.connect(self._add_highlight)

        self.save_btn = QtWidgets.QToolButton(); self.save_btn.setText("Save")
        self.save_btn.setIcon(icons.icon("save"))
        self.save_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.save_btn.setToolTip("Save the selection or the merged highlights as a new .raw.")
        self.save_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self._save_menu = QtWidgets.QMenu(self.save_btn)
        self._save_menu.addAction("Selection → .raw…", self._save_selection_raw)
        self._merge_action = self._save_menu.addAction(
            "Merge highlights → one .raw…", self._save_merge_raw)
        self._save_menu.addSeparator()
        self._save_menu.addAction("Clear highlights", self.ctl.clear_highlights)
        self._save_menu.aboutToShow.connect(self._sync_save_menu)
        self.save_btn.setMenu(self._save_menu)

        self.mp4_btn = QtWidgets.QToolButton(); self.mp4_btn.setText("MP4")
        self.mp4_btn.setIcon(icons.icon("film"))
        self.mp4_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.mp4_btn.setToolTip("Export the current view over the In/Out range to an MP4 "
                                "(faithful to the tuned mode / color / scale) with a context banner.")
        self.mp4_btn.clicked.connect(self._export_mp4)

        self.clear_hl_btn = QtWidgets.QToolButton()
        self.clear_hl_btn.setIcon(icons.icon("close"))
        self.clear_hl_btn.setIconSize(QtCore.QSize(12, 12))
        self.clear_hl_btn.setToolTip("Clear all highlight bands.")
        self.clear_hl_btn.clicked.connect(self.ctl.clear_highlights)

        for b in (self.add_hl_btn, self.save_btn, self.mp4_btn, self.clear_hl_btn):
            row.addWidget(b)
        return box

    def _on_selection(self, lo, hi):
        self.ctl.set_selection(lo, hi)
        self._update_sel_label()

    def _reflect_selection(self, *_):
        if self.selbar is not None:
            b = QtCore.QSignalBlocker(self.selbar)
            self.selbar.set_primary(self.ctl.sel_lo, self.ctl.sel_hi); del b
        self._update_sel_label()

    def _reflect_highlights(self, *_):
        if self.selbar is not None:
            b = QtCore.QSignalBlocker(self.selbar)
            self.selbar.set_highlights(self.ctl.highlights); del b
        self._update_sel_label()

    def _update_sel_label(self):
        if self.selbar is None:
            return
        base = (f"[ {self.ctl.sel_t0():.3f}–{self.ctl.sel_t1():.3f} s ]"
                if self.ctl.has_selection() else "[ full ]")
        n = len(self.ctl.highlights)
        if n:
            base += f"  ·  {n} marked"
        self.sel_lbl.setText(base)

    def _add_highlight(self):
        self.ctl.add_highlight()                    # freeze the current In/Out as a band

    def _sync_save_menu(self):
        n = len(self.ctl.highlights)
        self._merge_action.setText(f"Merge {n} highlight{'' if n == 1 else 's'} → one .raw…")
        self._merge_action.setEnabled(n >= 1)

    # ----- export from the bar (uses the host panel's recording + render path) -----
    def _rec(self):
        return getattr(self.host, "rec", None)

    def _save_selection_raw(self):
        rec = self._rec()
        if rec is None:
            QtWidgets.QMessageBox.information(self, "Save .raw", "Load a recording first.")
            return
        t0, t1 = self.ctl.sel_t0(), self.ctl.sel_t1()
        if t1 - t0 < 1e-4:
            QtWidgets.QMessageBox.information(self, "Save .raw",
                                             "Select a non-empty In/Out range first.")
            return
        self._write_raw(rec, [(t0, t1)], f"_clip_{t0:.2f}-{t1:.2f}")

    def _save_merge_raw(self):
        rec = self._rec()
        if rec is None:
            QtWidgets.QMessageBox.information(self, "Merge .raw", "Load a recording first.")
            return
        spans = self.ctl.highlight_times()
        if not spans:
            QtWidgets.QMessageBox.information(
                self, "Merge .raw",
                "Add at least one highlight first: drag the In/Out handles, then click 'Add'.")
            return
        self._write_raw(rec, spans, f"_merged_{len(spans)}")

    def _write_raw(self, rec, spans, suffix):
        import os
        base = (os.path.splitext(rec.source_path)[0] if rec.source_path
                else os.path.join(os.getcwd(), rec.name))
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save → .raw", f"{base}{suffix}.raw", "EVT raw (*.raw)")
        if not out:
            return
        if not out.lower().endswith(".raw"):
            out += ".raw"
        from gottlux.io import writer
        from gottlux.io.paths import open_in_file_browser
        from gottlux.app.uikit import with_progress
        try:
            if len(spans) == 1:
                t0, t1 = spans[0]
                n = with_progress(self.window(), "Saving .raw clip",
                                  lambda cb: writer.cut_clip(rec, out, t0=t0, t1=t1, progress=cb),
                                  label="Writing the clip…")
                msg = f"Wrote {n:,} events ([{t0:.3f}, {t1:.3f}] s) →\n{out}"
            else:
                res = with_progress(
                    self.window(), "Merging highlights → .raw",
                    lambda cb: writer.stitch_clips(out, [(rec, t0, t1) for t0, t1 in spans],
                                                   progress=cb),
                    label="Stitching the highlights…")
                msg = (f"Merged {len(spans)} highlights → {os.path.basename(out)}\n"
                       f"{res['n_events']:,} events · {res['duration_s']:.2f} s")
            open_in_file_browser(os.path.dirname(os.path.abspath(out)))
            QtWidgets.QMessageBox.information(self, "Saved .raw", msg)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save .raw", str(e))

    def _export_mp4(self):
        rec = self._rec()
        if rec is None:
            QtWidgets.QMessageBox.information(self, "Export MP4", "Load a recording first.")
            return
        self.ctl.pause()
        host = self.host
        t0, t1 = ((self.ctl.sel_t0(), self.ctl.sel_t1()) if self.ctl.has_selection()
                  else (self.ctl.t0, self.ctl.t1))
        fields = {
            "Recording": rec.name,
            "Sensor": f"{rec.width}×{rec.height} px · {rec.fmt}",
            "Geometry": "rotation" if rec.is_rotating else "staring",
            "Events": f"{rec.n:,}",
            "Duration": f"{rec.duration_s:.2f} s",
            "Accum (exposure)": f"{self.ctl.accum * 1e3:.1f} ms",
        }
        sensor_wh = None
        if hasattr(host, "sensor_size"):
            sensor_wh = host.sensor_size()
        ctx = dict(rec=rec, target=getattr(host, "glw", host), set_cursor=self.ctl.set_cursor,
                   view=type(host).__name__ if host is not None else "view",
                   t0=t0, t1=t1, accum=self.ctl.accum, fps=self.ctl.fps, fields=fields,
                   render=getattr(host, "capture_frame", None),
                   sensor_wh=sensor_wh or (rec.width, rec.height))
        from gottlux.app.capture import ScreenCaptureDialog
        ScreenCaptureDialog(self.window(), ctx).exec()

    # ----- user -> controller -----
    def _on_seek(self, v):
        self.ctl.set_fraction(v / 1000.0)

    def _on_seek_value(self, v):
        # only react to programmatic/keyboard changes when not dragging (sliderMoved covers drag)
        if not self.seek.isSliderDown():
            self.ctl.set_fraction(v / 1000.0)

    # ----- controller -> user -----
    def _reflect_cursor(self, t):
        block = QtCore.QSignalBlocker(self.seek)
        self.seek.setValue(int(round(self.ctl.fraction() * 1000)))
        del block
        self.time_lbl.setText(f"{t:.3f} / {self.ctl.t1:.3f} s")

    def _reflect_accum(self, dt):
        if hasattr(self, "accum"):
            b1 = QtCore.QSignalBlocker(self.accum)
            self.accum.setValue(dt)
            del b1
            b2 = QtCore.QSignalBlocker(self.accum_slider)
            self.accum_slider.setValue(self._acc_to_slider(dt))
            del b2

    def _reflect_accum_dir(self, back):
        if not hasattr(self, "accum_dir_btn"):
            return
        b = QtCore.QSignalBlocker(self.accum_dir_btn)
        self.accum_dir_btn.setChecked(bool(back))
        del b
        if back:
            self.accum_dir_btn.setText("behind")
            self.accum_dir_btn.setIcon(icons.icon("arrow-left"))
            self.accum_dir_btn.setToolTip(
                "Accumulation reaches BEHIND the cursor: the window is [t − Δ, t], so the cursor is "
                "the leading edge and the exposure trails into the past.\nClick to integrate AHEAD "
                "of the cursor instead.")
        else:
            self.accum_dir_btn.setText("ahead")
            self.accum_dir_btn.setIcon(icons.icon("arrow-right"))
            self.accum_dir_btn.setToolTip(
                "Accumulation reaches AHEAD of the cursor: the window is [t, t + Δ], so the cursor "
                "marks the start and the exposure extends into the future.\nClick to integrate "
                "BEHIND the cursor instead.")

    def _fps_from_combo(self, *_):
        try:
            v = float(self.fps.currentText())
        except (TypeError, ValueError):
            return
        self.ctl.set_fps(v)

    def _reflect_fps(self, f):
        if hasattr(self, "fps"):
            blocker = QtCore.QSignalBlocker(self.fps)
            self.fps.setCurrentText(f"{f:g}")
            del blocker
            self.fps_factor.setText(_speed_factor_text(f))

    def _reflect_play(self, playing):
        self.play_btn.setIcon(icons.icon("pause" if playing else "play"))
