"""
workbench.py — the flutter/flicker detection workbench (the tuning instrument).

Build and tune a detector with instant visual feedback. The left side shows the **flicker
map** (hue = dominant frequency, opacity = SNR, with a frequency legend) and, below it, the
**live temporal spectrum** of a draggable region plus a plain-language stats readout. The
right side is organised top-to-bottom into clear stages — pick a **Detector**, set the
**Analysis window**, **Tune** its parameters (with a **Reset**), **Run**, and read the
**Targets** table — every control carrying a hover description.

It is seekable and live like the other tabs (the analysis window is anchored at the shared
cursor); the flicker map recompute is throttled while playing and debounced while scrubbing.
**Export** saves the flicker map / spectrum as figures, or the flicker-map cube / detections
as data blocks.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from gottlux.app import icons
from gottlux.app import style
from gottlux.app.legend import FrequencyLegend
from gottlux.app.loader import DetectorWorker
from gottlux.app.transport import TransportBar
from gottlux.app.widgets import ParamPanel
from gottlux.config import Config
from gottlux.core import frequency as fq
from gottlux.core.accumulate import accumulate_frame
from gottlux.detectors import get_detector, list_detectors
from gottlux.viz import theme as viz_theme

_RESULT_COLS = ["ID", "Freq (Hz)", "Conf", "SNR", "Dets", "Dur (s)"]
_COL_TIPS = {
    "ID": "Track identifier.",
    "Freq (Hz)": "Median verified flutter frequency of the track (rotor/wingbeat tone).",
    "Conf": "0–1 confidence: blends track persistence, mean SNR, frequency stability and "
            "harmonic support.",
    "SNR": "Mean in-band spectral peak / noise-floor over the track (higher = cleaner tone).",
    "Dets": "Number of verified detections linked into the track.",
    "Dur (s)": "Time span the track was followed.",
}


class FlutterWorkbench(QtWidgets.QWidget):
    """Interactive flicker-map + region-spectrum + tunable-detector panel (seekable/live)."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.ctl = controller
        self.rec = None
        self.cfg = Config(mode="staring")
        self._result = None
        self._worker = None
        self._param_panel = None
        self._last_refresh = 0.0
        self._fm = None
        self._sp = None
        self._live_worker = None
        self._live_last = 0.0

        self._build_left()
        self._build_right()

        from gottlux.app.uikit import plot_with_deck
        main = QtWidgets.QHBoxLayout(self)
        # The tabbed control deck is a draggable, min-width column (was a fixed 400 px cap),
        # so the flicker map keeps usable width on a laptop and in the split view.
        main.addWidget(plot_with_deck(self._leftcol, self._right, min_deck=320, init_deck=400,
                                      scroll=False))

        self._debounce = QtCore.QTimer(self, singleShot=True, interval=180)
        self._debounce.timeout.connect(self.refresh)
        self.ctl.cursorChanged.connect(self._on_cursor)
        self._rebuild_params("drone")

    # ================================================================== left
    def _build_left(self):
        self.glw = pg.GraphicsLayoutWidget()
        self.vb = self.glw.addViewBox(lockAspect=True, invertY=True, row=0, col=0)
        self.bg_img = pg.ImageItem(axisOrder="row-major")
        self.flk_img = pg.ImageItem(axisOrder="row-major")
        self.vb.addItem(self.bg_img)
        self.vb.addItem(self.flk_img)
        self.overlay = pg.ScatterPlotItem(pen=None)
        self.vb.addItem(self.overlay)
        self.boxes = []
        self.region = pg.RectROI([40, 140], [60, 40], pen=pg.mkPen(style.ACCENT2, width=2))
        self.region.setToolTip("Drag/resize to pick a region; its live spectrum is shown below.")
        self.vb.addItem(self.region)
        self.region.sigRegionChanged.connect(self._update_spectrum)
        self.freq_legend = FrequencyLegend("turbo")
        self.freq_legend.setToolTip("Flicker-map colour key: hue = dominant frequency in the "
                                    "band; a cell's opacity grows with its SNR (confidence).")

        map_box = QtWidgets.QWidget()
        mb = QtWidgets.QVBoxLayout(map_box); mb.setContentsMargins(0, 0, 0, 0)
        mb.addWidget(QtWidgets.QLabel("Flicker map — where the scene flickers and how fast",
                                      objectName="h2"))
        mb.addWidget(self.glw, 1)
        mb.addWidget(self.freq_legend)

        self.spec = pg.PlotWidget()
        self.spec.setLabel("bottom", "frequency", "Hz")
        self.spec.setLabel("left", "power")
        self.spec.setLogMode(y=True)
        self.spec.setToolTip("Temporal power spectrum of the selected region over the analysis "
                             "window. The shaded band is the detector's search range.")
        self.spec_curve = self.spec.plot(pen=pg.mkPen(style.ACCENT, width=1.5))
        self.spec_peak = pg.ScatterPlotItem(symbol="t", size=11, brush=style.ACCENT2)
        self.spec.addItem(self.spec_peak)
        self.band_region = pg.LinearRegionItem(brush=(57, 197, 207, 30), movable=False)
        self.spec.addItem(self.band_region)

        self.region_readout = QtWidgets.QLabel("Region: —")
        self.region_readout.setToolTip(
            "Selected region's spectrum: peak frequency, SNR (peak ÷ noise floor), harmonic-comb "
            "score (0–1, how rotor-like the overtones are), and event count.")

        spec_box = QtWidgets.QWidget()
        sb = QtWidgets.QVBoxLayout(spec_box); sb.setContentsMargins(0, 0, 0, 0)
        sb.addWidget(QtWidgets.QLabel("Region spectrum", objectName="h2"))
        sb.addWidget(self.spec, 1)
        sb.addWidget(self.region_readout)

        left_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left_split.addWidget(map_box)
        left_split.addWidget(spec_box)
        left_split.setSizes([470, 210])

        # the workbench integrates over its own analysis window, so hide the transport's
        # accumulation-direction toggle (it would not affect this view).
        self.transport = TransportBar(self.ctl, host=self, show_accum_dir=False)
        self._leftcol = QtWidgets.QWidget()
        lc = QtWidgets.QVBoxLayout(self._leftcol); lc.setContentsMargins(0, 0, 0, 0)
        lc.addWidget(left_split, 1)
        lc.addWidget(self.transport)

    # ================================================================== right (tabbed deck)
    def _build_right(self):
        """The control deck — organised into TABS (no long scrolling): Detect · Tune ·
        Spectrum · Targets — so every option is visible without hunting."""
        self._build_detect_widgets()
        self._build_spectrum_widgets()
        self._build_tune_widgets()
        self._build_targets_widgets()
        self._build_export_menu()

        self.deck = QtWidgets.QTabWidget()
        self.deck.addTab(self._detect_tab(), "Detect")
        self.deck.addTab(self._tune_tab(), "Tune")
        self.deck.addTab(self._spectrum_tab(), "Spectrum")
        self.deck.addTab(self._targets_tab(), "Targets")

        self._right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(self._right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.deck, 1)

    # ---- widgets ----
    def _build_detect_widgets(self):
        self.det_combo = QtWidgets.QComboBox()
        self.det_combo.addItems(sorted(list_detectors()))
        self.det_combo.setCurrentText("drone")
        self.det_combo.setToolTip("Target class to detect. Sets the default frequency band and "
                                  "gates; tune them in the Tune tab.")
        self.det_combo.currentTextChanged.connect(self._rebuild_params)
        self.det_info = QtWidgets.QLabel(); self.det_info.setObjectName("muted")
        self.det_info.setWordWrap(True)

        self.win = QtWidgets.QDoubleSpinBox(); self.win.setRange(0.05, 60); self.win.setValue(1.0)
        self.win.setSuffix(" s"); self.win.setDecimals(2)
        self.win.setToolTip("Length of the flicker/FFT analysis window, starting at the seek "
                            "cursor. Longer = finer frequency resolution and higher SNR, but "
                            "blurs fast motion.")
        self.win.valueChanged.connect(self.refresh)
        self.cell = QtWidgets.QSpinBox(); self.cell.setRange(2, 32); self.cell.setValue(8)
        self.cell.setSuffix(" px")
        self.cell.setToolTip("Flicker-map spatial cell size. Smaller = finer map, slower.")
        self.cell.valueChanged.connect(self.refresh)
        self.map_btn = QtWidgets.QPushButton("Recompute flicker map")
        self.map_btn.setToolTip("Recompute the flicker map for the current window/cell now.")
        self.map_btn.clicked.connect(self.refresh)
        self.export_btn = QtWidgets.QToolButton(); self.export_btn.setText("Export")
        self.export_btn.setIcon(icons.icon("export"))
        self.export_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.export_btn.setToolTip("Save the flicker map / spectrum as figures, or export the "
                                   "flicker-map cube / detections / detection report as data.")
        self.export_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        self.run_btn = QtWidgets.QPushButton("Run detector")
        self.run_btn.setObjectName("primary")
        self.run_btn.setToolTip("Run the tuned detector over the analysis window (off-thread).")
        self.run_btn.clicked.connect(self.run_detector)
        self.progress = QtWidgets.QProgressBar(); self.progress.setVisible(False)

        # live tracking (ticket 6) — run the detector continuously as you seek/play
        self.live_chk = QtWidgets.QCheckBox("Live track while seeking/playing")
        self.live_chk.setToolTip("Continuously run the detector on a trailing window at the "
                                 "cursor and overlay targets — tune a tracker in real time.")
        self.live_chk.toggled.connect(self._on_live_toggle)
        self.live_win = QtWidgets.QDoubleSpinBox(); self.live_win.setRange(0.05, 3.0)
        self.live_win.setValue(0.4); self.live_win.setSuffix(" s"); self.live_win.setDecimals(2)
        self.live_win.setToolTip("Trailing window length used for each live-tracking pass "
                                 "(shorter = more responsive, less frequency resolution).")

    def _build_spectrum_widgets(self):
        self.spec_method = QtWidgets.QComboBox()
        self.spec_method.addItems(["FFT (binned)", "NUFFT (non-uniform)"])
        self.spec_method.setToolTip("Region-spectrum transform. FFT bins the stream (fast). "
                                    "NUFFT evaluates a non-uniform transform straight from event "
                                    "times — no Nyquist ceiling, good for sparse/odd sampling.")
        self.spec_method.currentIndexChanged.connect(self._update_spectrum)
        self.spec_norm = QtWidgets.QComboBox()
        self.spec_norm.addItems(["none", "median", "zscore"])
        self.spec_norm.setToolTip("Spectral normalization to EMPHASIZE PEAKING over colored "
                                  "noise: 'median' whitens by a sliding-median floor; 'zscore' "
                                  "reports sigmas above local noise.")
        self.spec_norm.currentIndexChanged.connect(self._update_spectrum)
        # --- rotational-data compensation ---
        self._spin_hz = None
        self.derotate_chk = QtWidgets.QCheckBox("Rotational: remove spin envelope")
        self.derotate_chk.setToolTip(
            "Spinning sensor: a region's events arrive in once-per-revolution bursts, so the raw "
            "spectrum is dominated by the rotation frequency (~1 Hz) and its harmonics — the 'FFT "
            "gravity'. This high-passes away that slow envelope (subtracts a moving-average "
            "low-pass) so the high-frequency flutter shows. Click 'Find rotation rate' first to set "
            "the cutoff from the measured spin. (FFT mode only.)")
        self.derotate_chk.toggled.connect(self._update_spectrum)
        self.spin_btn = QtWidgets.QPushButton("Find rotation rate")
        self.spin_btn.setToolTip("Estimate the sensor's rotation rate from the autocorrelation of "
                                 "the event-rate-vs-time, and show the event-rate + autocorrelation "
                                 "plot. Sets the de-rotation cutoff.")
        self.spin_btn.clicked.connect(self._find_rotation_rate)
        self.spin_lbl = QtWidgets.QLabel("rotation rate: —"); self.spin_lbl.setObjectName("muted")

    def _build_tune_widgets(self):
        self._param_holder = QtWidgets.QScrollArea()
        self._param_holder.setWidgetResizable(True)
        self._param_holder.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.reset_btn = QtWidgets.QPushButton("Reset to defaults")
        self.reset_btn.setToolTip("Restore this detector's parameters (and the window/cell) to "
                                  "their default values.")
        self.reset_btn.clicked.connect(self._reset)
        self.suggest_lbl = QtWidgets.QLabel(); self.suggest_lbl.setWordWrap(True)
        self.suggest_lbl.setObjectName("muted")
        self.suggest_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

    def _build_targets_widgets(self):
        self.summary_lbl = QtWidgets.QLabel("No run yet."); self.summary_lbl.setObjectName("muted")
        self.summary_lbl.setWordWrap(True)
        self.table = QtWidgets.QTableWidget(0, len(_RESULT_COLS))
        self.table.setHorizontalHeaderLabels(_RESULT_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        for i, c in enumerate(_RESULT_COLS):
            self.table.horizontalHeaderItem(i).setToolTip(_COL_TIPS[c])
        self.table.itemSelectionChanged.connect(self._highlight_selected)

    # ---- tab pages ----
    def _detect_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        det_box = QtWidgets.QGroupBox("Detector")
        db = QtWidgets.QVBoxLayout(det_box); db.addWidget(self.det_combo); db.addWidget(self.det_info)
        win_box = QtWidgets.QGroupBox("Analysis window")
        wf = QtWidgets.QFormLayout(win_box)
        wf.addRow("Window", self.win); wf.addRow("Cell size", self.cell)
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.map_btn, 1); row.addWidget(self.export_btn)
        wf.addRow(row)
        run_box = QtWidgets.QGroupBox("Run")
        rb = QtWidgets.QVBoxLayout(run_box)
        rb.addWidget(self.run_btn); rb.addWidget(self.progress)
        lr = QtWidgets.QHBoxLayout(); lr.addWidget(self.live_chk, 1)
        lr.addWidget(QtWidgets.QLabel("win")); lr.addWidget(self.live_win)
        rb.addLayout(lr)
        for b in (det_box, win_box, run_box):
            v.addWidget(b)
        v.addStretch(1)
        return w

    def _tune_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.addWidget(self._param_holder, 1)
        v.addWidget(self.reset_btn)
        sug = QtWidgets.QGroupBox("Suggested settings")
        sv = QtWidgets.QVBoxLayout(sug)
        sa = QtWidgets.QScrollArea(); sa.setWidgetResizable(True); sa.setMaximumHeight(180)
        sa.setFrameShape(QtWidgets.QFrame.NoFrame); sa.setWidget(self.suggest_lbl)
        sv.addWidget(sa)
        v.addWidget(sug)
        return w

    def _spectrum_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QFormLayout(w)
        v.addRow("Transform", self.spec_method)
        v.addRow("Normalize", self.spec_norm)
        rot = QtWidgets.QGroupBox("Rotational data")
        rv = QtWidgets.QVBoxLayout(rot)
        rv.addWidget(self.spin_btn); rv.addWidget(self.spin_lbl); rv.addWidget(self.derotate_chk)
        v.addRow(rot)
        note = QtWidgets.QLabel(
            "These apply to the region spectrum (lower-left). Drag the orange box on the map "
            "to pick a region; normalization sharpens a faint tone, and NUFFT probes frequencies "
            "directly from event times. For a SPINNING sensor, 'Find rotation rate' then 'remove "
            "spin envelope' lifts the rotor flutter out from under the ~1 Hz rotation gravity.")
        note.setWordWrap(True); note.setObjectName("muted")
        v.addRow(note)
        return w

    def _targets_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.addWidget(self.summary_lbl)
        v.addWidget(self.table, 1)
        return w

    # ================================================================== data
    def set_recording(self, rec):
        self.rec = rec
        self.cfg = Config(mode="rotation" if rec.is_rotating else "staring",
                          sensor=self.cfg.sensor, fov_deg=self.cfg.fov_deg)
        self.win.setValue(min(1.0, max(0.1, rec.duration_s)))
        self.vb.setRange(xRange=(0, rec.width), yRange=(0, rec.height), padding=0)
        self.refresh(force=True)

    def showEvent(self, ev):
        super().showEvent(ev)
        self.refresh()

    def sync(self):
        self.refresh(force=True)

    # ================================================================== params
    def _rebuild_params(self, name):
        det = get_detector(name)
        self.det_info.setText(f"{det.description}\nUse for: {det.use_for}")
        self._param_panel = ParamPanel(det.PARAMS)
        self._param_panel.changed.connect(self._on_params_changed)
        self._param_holder.setWidget(self._param_panel)
        self.suggest_lbl.setText(self._suggestions_html(det))
        self._update_band_region()

    def _suggestions_html(self, det):
        """Auto-build a plain-language tuning guide from the detector's Param specs."""
        rows = ["<b>How to fine-tune this detector</b> (start with the band & SNR gate):"]
        for p in det.PARAMS:
            default = det.params.get(p.key, p.default)
            unit = f" {p.unit}" if p.unit else ""
            if p.kind in ("float", "int"):
                rng = f"range {p.lo:g}–{p.hi:g}{unit}, default {default:g}{unit}"
            elif p.kind == "bool":
                rng = f"default {'on' if default else 'off'}"
            else:
                rng = f"choices {', '.join(map(str, p.choices))}"
            tip = f" — {p.help}" if p.help else ""
            rows.append(f"• <b>{p.label}</b> ({rng}){tip}")
        rows.append("<br><i>Workflow:</i> box a candidate on the map → read its peak in the "
                    "Spectrum tab → set the band around it → raise the SNR gate until only the "
                    "target survives → add a harmonic gate for rotors → Run.")
        return "<br>".join(rows)

    def _reset(self):
        """Restore the current detector's params and the window/cell to defaults."""
        self._rebuild_params(self.det_combo.currentText())
        if self.rec is not None:
            self.win.setValue(min(1.0, max(0.1, self.rec.duration_s)))
        self.cell.setValue(8)
        self.refresh()

    def _on_params_changed(self, _vals):
        self._update_band_region()
        self.refresh()

    def _current_detector(self):
        overrides = self._param_panel.values() if self._param_panel else {}
        return get_detector(self.det_combo.currentText(), **overrides)

    def _band(self):
        v = self._param_panel.values() if self._param_panel else {}
        return v.get("freq_lo", 80.0), v.get("freq_hi", 800.0), v.get("fft_fs", 2000.0)

    def _update_band_region(self):
        lo, hi, _ = self._band()
        self.band_region.setRegion([lo, hi])
        self.freq_legend.set_band(lo, hi)

    def _window(self):
        t0 = self.ctl.cursor
        t1 = min(t0 + self.win.value(), self.rec.t_stop_s) if self.rec else t0 + self.win.value()
        return t0, t1

    # ================================================================== live / seek
    def _on_cursor(self, *_):
        if self.rec is None or not self.isVisible():
            return
        if self.live_chk.isChecked():
            self._maybe_live_track()
        if self.ctl.playing:
            if time.perf_counter() - self._last_refresh > 0.3:
                self.refresh()
        else:
            self._debounce.start()

    # ================================================================== live tracking
    def _on_live_toggle(self, on):
        if on and self.rec is not None:
            self._maybe_live_track(force=True)

    def _maybe_live_track(self, force=False):
        """Run the detector on a short trailing window at the cursor (throttled, off-thread)."""
        if self.rec is None:
            return
        if self._live_worker and self._live_worker.isRunning():
            return                       # don't pile up passes
        now = time.perf_counter()
        if not force and now - self._live_last < 0.35:
            return
        self._live_last = now
        det = self._current_detector()
        lw = self.live_win.value()
        t1 = self.ctl.cursor
        t0 = max(self.rec.t_start_s, t1 - lw)
        self.cfg.mode = "rotation" if self.rec.is_rotating else "staring"
        self._live_worker = DetectorWorker(det, self.rec, self.cfg, t0, t1)
        self._live_worker.done.connect(self._on_live_result)
        self._live_worker.failed.connect(lambda *_: None)
        self._live_worker.start()

    def _on_live_result(self, res):
        self._result = res
        self._draw_targets(res)
        n = res.n_targets
        self.summary_lbl.setText(f"<b>LIVE</b> · {n} target(s) at t={self.ctl.cursor:.3f}s "
                                 f"(window {self.live_win.value():.2f}s)")
        self._fill_table(res)

    # ================================================================== flicker map
    def refresh(self, *_, force=False):
        if self.rec is None or (not force and not self.isVisible()):
            return
        self._last_refresh = time.perf_counter()
        lo, hi, fs = self._band()
        t0, t1 = self._window()
        self._fm = fq.flicker_map(self.rec, fmin=lo, fmax=hi, fs=fs,
                                  cell=self.cell.value(), t0=t0, t1=t1)
        rgba = viz_theme.flicker_rgba(self._fm.dominant_freq, self._fm.snr, lo, hi)
        up = np.kron(rgba, np.ones((self._fm.cell, self._fm.cell, 1)))[: self.rec.height,
                                                                       : self.rec.width]
        bg = accumulate_frame(self.rec.window(t0, t1), mode="count")
        bgn = bg / (np.percentile(bg[bg > 0], 99) if np.any(bg > 0) else 1.0)
        self.bg_img.setImage(np.clip(bgn, 0, 1) * 0.5, levels=(0, 1))
        try:
            self.bg_img.setColorMap(pg.colormap.get("gray", source="matplotlib"))
        except Exception:
            pass
        self.flk_img.setImage(up)
        self.freq_legend.set_band(lo, hi)
        self._update_spectrum()

    # ================================================================== region spectrum
    def _update_spectrum(self):
        if self.rec is None:
            return
        pos = self.region.pos(); size = self.region.size()
        x0 = int(max(0, pos.x())); y0 = int(max(0, pos.y()))
        x1 = int(min(self.rec.width, pos.x() + size.x()))
        y1 = int(min(self.rec.height, pos.y() + size.y()))
        if x1 <= x0 or y1 <= y0:
            return
        lo, hi, fs = self._band()
        t0, t1 = self._window()
        win = self.rec.window(t0, t1, roi=(x0, y0, x1, y1))
        norm = self.spec_norm.currentText()
        if self.spec_method.currentText().startswith("NUFFT"):
            sp = fq.nufft_spectrum(win.t, fmin=lo, fmax=hi, normalize=norm)
        else:
            # rotational de-rotation: high-pass away the spin envelope (cutoff ~8x the spin rate,
            # well below the flutter band) so the ~1 Hz rotation gravity stops dominating
            dr = 0.0
            if self.derotate_chk.isChecked():
                dr = max(30.0, 8.0 * self._spin_hz) if self._spin_hz else 50.0
            sp = fq.region_spectrum(win.t, fs=max(fs, 2.2 * hi), fmin=lo, fmax=hi, normalize=norm,
                                    derotate_hz=dr)
        self._sp = sp
        if sp.freqs.size:
            self.spec_curve.setData(sp.freqs, np.maximum(sp.power, 1e-12))
            if np.isfinite(sp.peak_freq):
                self.spec_peak.setData([sp.peak_freq], [max(sp.peak_power, 1e-12)])
                self.region_readout.setText(
                    f"Region:  peak <b>{sp.peak_freq:.0f} Hz</b>   ·   SNR <b>{sp.snr:.1f}</b>   "
                    f"·   harmonic <b>{sp.harmonic_score:.2f}</b>   ·   {sp.n_events:,} events")
            else:
                self.spec_peak.setData([], [])
                self.region_readout.setText(f"Region:  no peak in band   ·   {win.n:,} events")
            self.spec.setXRange(0, min(hi * 1.4, sp.freqs[-1]))

    def _find_rotation_rate(self):
        """Estimate the spin rate from the event-rate autocorrelation, enable de-rotation, and
        write/reveal the high-quality event-rate + autocorrelation figure."""
        if self.rec is None:
            return
        from gottlux.rotation import rate_analysis as ra
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            res = ra.find_rotation_rate(self.rec)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        hz, per = res.get("hz"), res.get("period_s")
        if hz:
            self._spin_hz = hz
            self.spin_lbl.setText(f"rotation rate: <b>{hz:.3f} Hz</b>  (T={per:.4f}s, "
                                  f"conf {res['confidence']:.2f})")
            self.derotate_chk.setChecked(True)          # auto-enable de-rotation for the FFT
            self._update_spectrum()
        else:
            self.spin_lbl.setText("rotation rate: not found (no clear periodicity)")
        try:
            from gottlux.io.paths import analysis_subdir, open_in_file_browser
            out = analysis_subdir(getattr(self.rec, "source_path", "") or ".", "rotation_rate")
            ra.save_rotation_rate_report(self.rec, out)
            open_in_file_browser(out)
        except Exception:
            pass

    # ================================================================== run
    def run_detector(self):
        if self.rec is None or (self._worker and self._worker.isRunning()):
            return
        det = self._current_detector()
        t0, t1 = self._window()
        self.cfg.mode = "rotation" if self.rec.is_rotating else "staring"
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True); self.progress.setValue(0)
        self.summary_lbl.setText("Running…")
        self._worker = DetectorWorker(det, self.rec, self.cfg, t0, t1)
        self._worker.progress.connect(lambda f: self.progress.setValue(int(f * 100)))
        self._worker.done.connect(self._on_result)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_result(self, res):
        self._result = res
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        conf = res.confident(0.5)
        self.summary_lbl.setText(
            f"{res.n_targets} target(s); {len(conf)} confident (≥0.5). "
            f"diag: {res.diagnostics.get('n_verified', 0)} verified / "
            f"{res.diagnostics.get('n_candidates', 0)} candidates")
        self._fill_table(res)
        self._draw_targets(res)

    def _on_fail(self, msg):
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.summary_lbl.setText("ERROR — see console.")
        print(msg)

    def _fill_table(self, res):
        targets = sorted(res.targets, key=lambda t: -t.confidence)
        self.table.setRowCount(len(targets))
        self._table_targets = targets
        for r, t in enumerate(targets):
            vals = [str(t.id), f"{t.median_freq:.0f}", f"{t.confidence:.2f}",
                    f"{np.nanmean(t.snr):.1f}", str(t.n), f"{t.duration_s:.2f}"]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(r, c, it)

    def _highlight_selected(self):
        rows = {i.row() for i in self.table.selectedItems()}
        if not rows or not getattr(self, "_table_targets", None):
            return
        # redraw boxes, emphasizing the selected target
        self._draw_targets(self._result, emphasize={self._table_targets[r].id for r in rows})

    # ================================================================== overlay
    def _draw_targets(self, res, emphasize=None):
        emphasize = emphasize or set()
        for b in self.boxes:
            self.vb.removeItem(b)
        self.boxes = []
        spots = []
        for i, t in enumerate(res.targets):
            color = pg.intColor(i, hues=max(len(res.targets), 6))
            bb = t.bbox[-1]
            wide = 3 if t.id in emphasize else 2
            r = pg.RectROI([bb[0], bb[1]], [bb[2] - bb[0], bb[3] - bb[1]],
                           pen=pg.mkPen(color, width=wide), movable=False, resizable=False)
            for h in r.getHandles():
                r.removeHandle(h)
            self.vb.addItem(r)
            self.boxes.append(r)
            spots += [{"pos": (cx, cy), "brush": color, "size": 5}
                      for cx, cy in zip(t.cx, t.cy)]
        self.overlay.setData(spots)

    def set_region(self, roi):
        if roi is None:
            return
        x0, y0, x1, y1 = roi
        self.region.setPos((x0, y0))
        self.region.setSize((max(2, x1 - x0), max(2, y1 - y0)))

    def result(self):
        return self._result

    # ================================================================== export
    def _build_export_menu(self):
        m = QtWidgets.QMenu(self)
        m.addAction("Save flicker-map figure…", self._save_map_fig)
        m.addAction("Save region-spectrum figure…", self._save_spec_fig)
        m.addSeparator()
        m.addAction("Export flicker-map cube…", self._export_flicker_cube)
        m.addAction("Export detections (CSV/Parquet)…", self._export_detections)
        m.addAction("Export detection report (first-principles)…", self._export_report)
        self.export_btn.setMenu(m)

    def _export_report(self):
        if not self._result:
            QtWidgets.QMessageBox.information(self, "Export", "Run the detector first.")
            return
        base, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export detection report",
                                                        "detection", "Report (*.md)")
        if not base:
            return
        from gottlux.run.report import save_detection_report
        self._notify(save_detection_report(os.path.splitext(base)[0], self._result, self.rec,
                                            cfg=self.cfg, window=self._window()))

    def _save_map_fig(self):
        if self._fm is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save flicker map", "flicker_map.png",
                                                        "PNG (*.png)")
        if not path:
            return
        from gottlux.io import export
        from gottlux.viz import spectral
        t0, t1 = self._window()
        bg = accumulate_frame(self.rec.window(t0, t1), mode="count")
        fig = spectral.flicker_map_figure(self._fm, background=bg,
                                          title=f"Flicker map — {self.rec.name}")
        self._notify(export.save_figure(fig, os.path.splitext(path)[0], close=True))

    def _save_spec_fig(self):
        if self._sp is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save spectrum", "spectrum.png",
                                                        "PNG (*.png)")
        if not path:
            return
        from gottlux.io import export
        from gottlux.viz import spectral
        self._notify(export.save_figure(spectral.spectrum_figure(self._sp),
                                        os.path.splitext(path)[0], close=True))

    def _export_flicker_cube(self):
        if self._fm is None:
            return
        base, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export flicker-map cube",
                                                        "flicker", "Data block (*.npz *.h5)")
        if not base:
            return
        from gottlux.app.exporting import save_flicker_cube
        self._notify(save_flicker_cube(os.path.splitext(base)[0], self._fm))

    def _export_detections(self):
        if not self._result or not self._result.targets:
            QtWidgets.QMessageBox.information(self, "Export", "Run the detector first.")
            return
        base, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export detections",
                                                        "detections", "Table (*.parquet *.csv)")
        if not base:
            return
        from gottlux.io import export
        rows = {k: [] for k in ("target_id", "t_s", "cx", "cy", "freq_hz", "snr", "harmonic",
                                "azimuth_deg", "elev_deg", "range_m", "confidence")}
        for t in self._result.targets:
            for i in range(t.n):
                rows["target_id"].append(t.id); rows["t_s"].append(t.t[i])
                rows["cx"].append(t.cx[i]); rows["cy"].append(t.cy[i])
                rows["freq_hz"].append(t.freq_hz[i]); rows["snr"].append(t.snr[i])
                rows["harmonic"].append(t.harmonic[i])
                rows["azimuth_deg"].append(t.azimuth_deg[i] if t.azimuth_deg is not None else np.nan)
                rows["elev_deg"].append(t.elev_deg[i] if t.elev_deg is not None else np.nan)
                rows["range_m"].append(t.range_m[i] if t.range_m is not None else np.nan)
                rows["confidence"].append(t.confidence)
        self._notify(export.save_table(rows, os.path.splitext(base)[0]))

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
