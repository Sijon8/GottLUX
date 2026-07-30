"""
fusionlab.py — the Fusion Lab tab: co-register an EBS recording with an audio ``.wav``.

Two sensors (the loaded EBS recording and an audio recorder) filmed the same scene on
independent clocks. This tab brings them onto **one timeline**: it plots the EBS **event-rate**
envelope above the audio **RMS** envelope (X-linked), recovers their temporal **offset** by
cross-correlation (or lets you nudge it by hand, like a video editor's audio-sync slider), then
**exports the aligned pair** (an EBS ``.raw`` + a ``.wav`` re-zeroed to a shared ``t = 0``) and
can run the full **fusion study** (cross-domain spectra + report) into a folder.

It is the GUI front-end to :mod:`gottlux.io.fusion` and :mod:`gottlux.run.fusion_study`; nothing
here is project-specific. Degrades to a message if pyqtgraph is unavailable.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6 import QtWidgets

from gottlux.app import style
from gottlux.app.uikit import plot_with_deck, reserve_lines, with_progress
from gottlux.io import fusion

try:
    import pyqtgraph as pg
    _HAVE_PG = True
except Exception:                                  # pragma: no cover
    _HAVE_PG = False


class FusionLab(QtWidgets.QWidget):
    """Align the loaded EBS recording to an audio ``.wav`` and export/analyze the fused pair."""

    def __init__(self, controller, filters=None, parent=None):
        super().__init__(parent)
        self.ctl = controller
        self.filters = filters
        self.rec = None
        self.audio = None
        self.result = None
        self._bin_s = 0.010

        if not _HAVE_PG:
            lay = QtWidgets.QVBoxLayout(self)
            lay.addWidget(QtWidgets.QLabel(
                "The Fusion Lab needs pyqtgraph (pip install pyqtgraph)."))
            self.glw = None
            return

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(style.BG)
        self.p_ebs = self.glw.addPlot(row=0, col=0, title="EBS event rate (events/s)")
        self.p_ebs.setLabel("left", "ev/s")
        self.p_ebs.showGrid(x=True, y=True, alpha=0.25)
        self.c_ebs = self.p_ebs.plot(pen=pg.mkPen("#e8820c", width=1))
        self.p_aud = self.glw.addPlot(row=1, col=0, title="Audio RMS (load a .wav)")
        self.p_aud.setLabel("left", "rms"); self.p_aud.setLabel("bottom", "EBS time", units="s")
        self.p_aud.showGrid(x=True, y=True, alpha=0.25)
        self.c_aud = self.p_aud.plot(pen=pg.mkPen("#1f77b4", width=1))
        self.p_aud.setXLink(self.p_ebs)
        self._playheads = [
            pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(style.MUTED, width=1)),
            pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(style.MUTED, width=1))]
        self.p_ebs.addItem(self._playheads[0]); self.p_aud.addItem(self._playheads[1])

        main = QtWidgets.QHBoxLayout(self)
        main.addWidget(plot_with_deck(self.glw, self._build_deck(), init_deck=320))

        if hasattr(self.ctl, "cursorChanged"):
            self.ctl.cursorChanged.connect(self._on_cursor)

    # ------------------------------------------------------------------ controls deck
    def _build_deck(self):
        deck = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(deck)

        self.load_btn = QtWidgets.QPushButton("Load audio (.wav)…")
        self.load_btn.setToolTip("Load the time-synchronized audio recording of this scene.")
        self.load_btn.clicked.connect(self._load_audio)
        v.addWidget(self.load_btn)
        self.audio_lbl = reserve_lines(QtWidgets.QLabel("No audio loaded."), 2)
        v.addWidget(self.audio_lbl)

        box = QtWidgets.QGroupBox("Temporal alignment")
        bl = QtWidgets.QVBoxLayout(box)
        self.auto_btn = QtWidgets.QPushButton("Auto-align (cross-correlate)")
        self.auto_btn.setToolTip("Recover the offset by cross-correlating the EBS event-rate "
                                 "envelope against the audio RMS envelope.")
        self.auto_btn.clicked.connect(self._auto_align)
        bl.addWidget(self.auto_btn)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Offset"))
        self.offset_sp = QtWidgets.QDoubleSpinBox()
        self.offset_sp.setRange(-60.0, 60.0); self.offset_sp.setDecimals(3)
        self.offset_sp.setSingleStep(0.05); self.offset_sp.setSuffix(" s")
        self.offset_sp.setToolTip("Add to audio timestamps to land on the EBS clock. Nudge to "
                                  "fine-tune the auto estimate.")
        self.offset_sp.valueChanged.connect(self._on_offset_changed)
        row.addWidget(self.offset_sp, 1)
        bl.addLayout(row)
        self.align_lbl = reserve_lines(QtWidgets.QLabel("Offset not set."), 2)
        bl.addWidget(self.align_lbl)
        v.addWidget(box)

        box2 = QtWidgets.QGroupBox("Export & analyze")
        b2 = QtWidgets.QVBoxLayout(box2)
        self.export_btn = QtWidgets.QPushButton("Export aligned pair (.raw + .wav)…")
        self.export_btn.setToolTip("Write the EBS .raw and the .wav cropped to their overlap and "
                                   "re-zeroed to a shared t = 0 (+ a fusion manifest).")
        self.export_btn.clicked.connect(self._export_aligned)
        b2.addWidget(self.export_btn)
        self.study_btn = QtWidgets.QPushButton("Run fusion study (figures + report)…")
        self.study_btn.setToolTip("Align, export the pair, and write the cross-domain spectra, "
                                  "spectrogram, summary JSON and report into a folder.")
        self.study_btn.clicked.connect(self._run_study)
        b2.addWidget(self.study_btn)
        v.addWidget(box2)

        v.addStretch(1)
        self._set_enabled(False)
        return deck

    def _set_enabled(self, on):
        for w in (getattr(self, n, None) for n in
                  ("auto_btn", "offset_sp", "export_btn", "study_btn")):
            if w is not None:
                w.setEnabled(on)

    # ------------------------------------------------------------------ data binding
    def set_recording(self, rec):
        self.rec = rec
        self.result = None
        if self.glw is None:
            return
        ce, re = rec.event_rate(self._bin_s)
        self.c_ebs.setData(ce, re)
        self.p_ebs.setTitle(f"EBS event rate — {rec.name} ({rec.duration_s:.1f}s)")
        self._set_enabled(self.audio is not None)
        self._redraw_audio()

    def sync(self):
        pass

    def _on_cursor(self, *_):
        if self.glw is None:
            return
        try:
            x = float(self.ctl.cursor)
        except Exception:
            return
        for ph in self._playheads:
            ph.setPos(x)

    # ------------------------------------------------------------------ audio
    def _load_audio(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load audio recording", "", "Audio (*.wav);;All files (*)")
        if not path:
            return
        try:
            self.audio = fusion.read_wav(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load audio", str(e))
            return
        a = self.audio
        self.audio_lbl.setText(f"{os.path.basename(path)}\n"
                               f"{a.sample_rate/1000:g} kHz · {a.subtype} · {a.duration_s:.1f} s")
        self.p_aud.setTitle(f"Audio RMS — {os.path.basename(path)}")
        self._set_enabled(self.rec is not None)
        if self.rec is not None:
            self._auto_align()
        else:
            self._redraw_audio()

    def _redraw_audio(self):
        if self.glw is None or self.audio is None:
            return
        ca, ra = self.audio.rms_envelope(self._bin_s)
        off = self.offset_sp.value()
        self.c_aud.setData(ca + off, ra)

    # ------------------------------------------------------------------ alignment
    def _auto_align(self):
        if self.rec is None or self.audio is None:
            return
        self.result = fusion.plan_alignment(self.rec, self.audio, bin_s=self._bin_s)
        self.offset_sp.blockSignals(True)
        self.offset_sp.setValue(self.result.offset_s)
        self.offset_sp.blockSignals(False)
        self._redraw_audio()
        self._update_align_label()

    def _on_offset_changed(self, *_):
        # a manual nudge supersedes the auto estimate; recompute overlap for the new offset
        if self.rec is not None and self.audio is not None:
            self.result = fusion.plan_alignment(self.rec, self.audio,
                                                offset_s=self.offset_sp.value(), bin_s=self._bin_s)
        self._redraw_audio()
        self._update_align_label()

    def _update_align_label(self):
        r = self.result
        if r is None:
            self.align_lbl.setText("Offset not set.")
            return
        pk = "manual" if not np.isfinite(r.peak_corr) else f"peak corr {r.peak_corr:.2f}"
        self.align_lbl.setText(f"offset {r.offset_s:+.3f} s · {pk}\noverlap {r.overlap_s:.2f} s")

    # ------------------------------------------------------------------ export / study
    def _ensure_result(self):
        if self.result is None and self.rec is not None and self.audio is not None:
            self.result = fusion.plan_alignment(self.rec, self.audio,
                                                offset_s=self.offset_sp.value(), bin_s=self._bin_s)
        return self.result

    def _export_aligned(self):
        if self.rec is None or self.audio is None:
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder for the aligned pair")
        if not out_dir:
            return
        res = self._ensure_result()
        base = f"{self.rec.name}_aligned"
        bias = os.path.splitext(self.rec.source_path)[0] + ".bias" \
            if str(getattr(self.rec, "source_path", "")).lower().endswith(".raw") else None
        try:
            man = with_progress(self, "Exporting aligned pair",
                                lambda cb: fusion.export_aligned(self.rec, self.audio, res, out_dir,
                                                                 base_name=base, bias_src=bias,
                                                                 progress=cb),
                                label="Writing aligned .raw + .wav…")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export aligned pair", str(e))
            return
        from gottlux.io.paths import open_in_file_browser
        open_in_file_browser(out_dir)
        QtWidgets.QMessageBox.information(
            self, "Export aligned pair",
            f"Wrote {man['ebs']['n_events']:,} events + {man['aligned_duration_s']:.1f}s audio\n"
            f"(offset {res.offset_s:+.3f}s) →\n{out_dir}")

    def _run_study(self):
        if self.rec is None or self.audio is None:
            return
        raw = getattr(self.rec, "source_path", "")
        if not str(raw).lower().endswith(".raw"):
            QtWidgets.QMessageBox.information(
                self, "Run fusion study",
                "The fusion study reads the EBS recording from its .raw on disk. Load a .raw "
                "recording (not a cache/folder) to run the full study, or use 'Export aligned "
                "pair' here.")
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder for the fusion study")
        if not out_dir:
            return
        res = self._ensure_result()
        from gottlux.run.fusion_study import run_fusion_study
        try:
            summary = with_progress(
                self, "Running fusion study",
                lambda cb: run_fusion_study(raw, self.audio.source_path, out_dir,
                                            offset_s=res.offset_s, label=self.rec.name,
                                            progress=cb),
                label="Aligning, exporting, computing spectra…")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Run fusion study", str(e))
            return
        from gottlux.io.paths import open_in_file_browser
        open_in_file_browser(out_dir)
        ac = summary["acoustic"]; ebs = summary["ebs"]; fz = summary["fusion"]
        QtWidgets.QMessageBox.information(
            self, "Fusion study",
            f"offset {summary['alignment']['offset_s']:+.3f}s · "
            f"overlap {summary['alignment']['overlap_s']:.1f}s\n"
            f"acoustic f0 {ac['f0_hz']:.0f} Hz (C={ac['confidence']:.2f}) · "
            f"EBS in-box f0 {ebs['f0_hz']:.0f} Hz (C={ebs['confidence']:.2f})\n"
            f"fused P(drone)={fz['p_fused']:.2f}\n→ {out_dir}")
