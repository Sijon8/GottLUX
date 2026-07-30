"""
export_dialog.py — the program-wide "export exactly what I choose" dialog.

The per-tab ``Export`` buttons are quick, single-artifact saves. This dialog is the
*overall* export: one place to tick precisely which artifacts to write — event-frame figure,
space-time cube, event-rate series, flicker-map figure/cube, region spectrum, detections
table, a first-principles detection report, and the run config/provenance — and drop them all
into one folder with a manifest. Items whose inputs are not available in the current session
(no flicker map computed, no detector run) are shown disabled with the reason, so the
selection always reflects what can actually be produced right now.
"""
from __future__ import annotations

import os

from PySide6 import QtWidgets

from gottlux.app.exporting import BUNDLE_ITEMS, PURPOSES, export_bundle

# Items that require an extra input beyond the recording, and what gates them.
_NEEDS = {
    "flicker_fig": ("flicker_map", "compute a flicker map in the workbench first"),
    "flicker_cube": ("flicker_map", "compute a flicker map in the workbench first"),
    "spectrum": ("spectrum", "select a region in the workbench first"),
    "detections": ("result", "run a detector in the workbench first"),
    "report": ("result", "run a detector in the workbench first"),
    "video": ("render", "this view has no faithful renderer — use Capture to grab the screen"),
}
# Sensible default ticks.
_DEFAULT_ON = {"frame_fig", "event_cube", "event_rate", "config", "infographic"}


class GlobalExportDialog(QtWidgets.QDialog):
    """Pick artifacts + an output folder, then write a manifest'd bundle."""

    def __init__(self, parent, context: dict):
        super().__init__(parent)
        self.ctx = context
        self.setWindowTitle("Export — choose what to save")
        self.setMinimumWidth(520)
        v = QtWidgets.QVBoxLayout(self)

        rec = context.get("rec")
        t0, t1 = context.get("t0", 0.0), context.get("t1", 0.0)
        head = QtWidgets.QLabel(
            f"<b>{rec.name if rec else '—'}</b> · window [{t0:.3f}, {t1:.3f}] s · "
            f"mode {context.get('mode','count')} · expr {context.get('expr','sqrt')}")
        head.setWordWrap(True)
        v.addWidget(head)

        self.checks = {}
        grid = QtWidgets.QGridLayout()
        for i, (key, label) in enumerate(BUNDLE_ITEMS.items()):
            cb = QtWidgets.QCheckBox(label)
            gate = _NEEDS.get(key)
            available = True
            if gate is not None:
                ctx_key, reason = gate
                available = context.get(ctx_key) is not None
                if not available:
                    cb.setEnabled(False)
                    cb.setToolTip(f"Unavailable — {reason}.")
            cb.setChecked(available and key in _DEFAULT_ON)
            self.checks[key] = cb
            grid.addWidget(cb, i // 2, i % 2)
        v.addLayout(grid)

        # output folder
        h = QtWidgets.QHBoxLayout()
        # the chosen folder is the PARENT; export_bundle creates a uniquely-named subfolder in it
        default_dir = (os.path.dirname(rec.source_path) if rec and rec.source_path else os.getcwd())
        self.out_edit = QtWidgets.QLineEdit(default_dir)
        browse = QtWidgets.QPushButton("Browse…"); browse.clicked.connect(self._browse)
        h.addWidget(QtWidgets.QLabel("Folder")); h.addWidget(self.out_edit, 1); h.addWidget(browse)
        v.addLayout(h)

        # purpose ("class") + free-text note — recorded in the manifest for your own tracking
        pn = QtWidgets.QHBoxLayout()
        self.purpose = QtWidgets.QComboBox()
        self.purpose.addItems([p for p in PURPOSES])
        self.purpose.setToolTip("Output class — a label saved in the manifest (research / demo / "
                                "graphic). Organizes the export; doesn't change what's produced.")
        self.note = QtWidgets.QLineEdit()
        self.note.setPlaceholderText("note (optional) — saved in the manifest")
        pn.addWidget(QtWidgets.QLabel("Purpose")); pn.addWidget(self.purpose)
        pn.addWidget(QtWidgets.QLabel("Note")); pn.addWidget(self.note, 1)
        v.addLayout(pn)

        # video output options (used when 'video' is ticked) — inherit the view's tuned settings,
        # tunable here: fps, resolution, and whether to burn in the infographic banner
        vg = QtWidgets.QHBoxLayout()
        self.vid_fps = QtWidgets.QSpinBox(); self.vid_fps.setRange(1, 1000)
        _vf = context.get("fps")
        self.vid_fps.setValue(int(min(max(round(float(_vf)), 1), 1000)) if _vf else 25)
        self.vid_fps.setToolTip("Output video frame rate — pre-filled from the view's FPS.")
        self.vid_accum = QtWidgets.QDoubleSpinBox(); self.vid_accum.setRange(1e-5, 2.0)
        self.vid_accum.setDecimals(5); self.vid_accum.setSingleStep(0.001); self.vid_accum.setSuffix(" s")
        self.vid_accum.setValue(float(context.get("accum") or 0.02))
        self.vid_accum.setToolTip("Accumulation (exposure) per rendered frame — pre-filled from the view.")
        self.vid_res = QtWidgets.QComboBox()
        self.vid_res.addItems(["1080p", "720p", "native", "2x", "4x"])
        self.vid_banner = QtWidgets.QCheckBox("infographic banner"); self.vid_banner.setChecked(True)
        has_render = context.get("render") is not None
        for w in (self.vid_fps, self.vid_accum, self.vid_res, self.vid_banner):
            w.setEnabled(has_render)
        vg.addWidget(QtWidgets.QLabel("Video:")); vg.addWidget(QtWidgets.QLabel("fps"))
        vg.addWidget(self.vid_fps); vg.addWidget(QtWidgets.QLabel("accum")); vg.addWidget(self.vid_accum)
        vg.addWidget(QtWidgets.QLabel("res")); vg.addWidget(self.vid_res)
        vg.addWidget(self.vid_banner); vg.addStretch(1)
        v.addLayout(vg)

        # select-all / none
        hb = QtWidgets.QHBoxLayout()
        all_btn = QtWidgets.QPushButton("All available"); all_btn.clicked.connect(lambda: self._set_all(True))
        none_btn = QtWidgets.QPushButton("None"); none_btn.clicked.connect(lambda: self._set_all(False))
        hb.addWidget(all_btn); hb.addWidget(none_btn); hb.addStretch(1)
        v.addLayout(hb)

        self.status = QtWidgets.QLabel(""); self.status.setObjectName("muted"); self.status.setWordWrap(True)
        v.addWidget(self.status)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Export")
        bb.accepted.connect(self._do_export)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder", self.out_edit.text())
        if d:
            self.out_edit.setText(d)

    def _set_all(self, on):
        for cb in self.checks.values():
            if cb.isEnabled():
                cb.setChecked(on)

    def _do_export(self):
        want = [k for k, cb in self.checks.items() if cb.isChecked() and cb.isEnabled()]
        if not want:
            self.status.setText("Nothing selected.")
            return
        rec = self.ctx.get("rec")
        if rec is None:
            self.status.setText("No recording loaded.")
            return
        out_dir = self.out_edit.text().strip()
        self.status.setText("Exporting…")
        QtWidgets.QApplication.processEvents()
        try:
            written, manifest = export_bundle(
                out_dir, rec, self.ctx.get("t0", 0.0), self.ctx.get("t1", 0.0),
                want=want, mode=self.ctx.get("mode", "count"), cmap=self.ctx.get("cmap", "inferno"),
                expr=self.ctx.get("expr", "sqrt"), flicker_map=self.ctx.get("flicker_map"),
                spectrum=self.ctx.get("spectrum"), result=self.ctx.get("result"),
                cfg=self.ctx.get("cfg"),
                purpose=self.purpose.currentText(), note=self.note.text().strip(),
                render=self.ctx.get("render"), sensor_wh=self.ctx.get("sensor_wh"),
                accum=self.vid_accum.value(), fps=self.vid_fps.value(),
                video_res=self.vid_res.currentText(), video_banner=self.vid_banner.isChecked())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))
            self.status.setText("Failed.")
            return
        from gottlux.io.paths import open_in_file_browser
        folder = manifest.get("out_dir", out_dir)     # the unique subfolder the files landed in
        open_in_file_browser(folder)
        QtWidgets.QMessageBox.information(
            self, "Export complete",
            f"Wrote {len(written)} file(s) to:\n{folder}\n\n"
            f"Produced: {', '.join(manifest['produced']) or '(none)'}")
        self.accept()
