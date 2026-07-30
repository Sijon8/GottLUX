"""
batchtrim.py — a small dialog to shorten every ``.raw`` in a folder by one common time window.

Drone captures often start with a few seconds of junk (rig settling / ego-motion) that breeds
false tracks. This trims a left (and optional right) bound off **every** clip in a folder at once
and writes them to a ``trimmed/`` subfolder — re-based to a *common* origin (via
:func:`gottlux.io.writer.trim_folder`) so a synchronized cam0/cam1 pair stays slate-aligned.
The decode/trim runs on a worker thread so the UI stays responsive.
"""
from __future__ import annotations

import glob
import os

from PySide6 import QtCore, QtWidgets

from gottlux.app import icons


class _TrimWorker(QtCore.QThread):
    done = QtCore.Signal(dict)
    failed = QtCore.Signal(str)
    progressed = QtCore.Signal(str)

    def __init__(self, folder, t0, t1, out_subdir):
        super().__init__()
        self.folder, self.t0, self.t1, self.out_subdir = folder, t0, t1, out_subdir

    def run(self):
        try:
            from gottlux.io import writer
            m = writer.trim_folder(self.folder, t0=self.t0, t1=self.t1, out_subdir=self.out_subdir,
                                   progress=lambda raw, n: self.progressed.emit(os.path.basename(raw)))
            self.done.emit(m)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class BatchTrimDialog(QtWidgets.QDialog):
    """Shorten every ``.raw`` in a folder to a common [start, end] window (synced clips stay aligned)."""

    def __init__(self, parent=None, folder=None):
        super().__init__(parent)
        self.setWindowTitle("Batch-trim clips")
        self.setMinimumWidth(580)
        self._worker = None
        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(QtWidgets.QLabel(
            "Shorten every clip in a folder by one shared time window — written to a subfolder, "
            "re-based to a common origin so synchronized cameras stay aligned."))

        fr = QtWidgets.QHBoxLayout()
        self.folder_edit = QtWidgets.QLineEdit(folder or "")
        browse = QtWidgets.QPushButton("Browse…"); browse.clicked.connect(self._browse)
        fr.addWidget(QtWidgets.QLabel("Folder:")); fr.addWidget(self.folder_edit, 1); fr.addWidget(browse)
        v.addLayout(fr)
        self.clips_lbl = QtWidgets.QLabel("—")
        self.clips_lbl.setObjectName("muted"); self.clips_lbl.setWordWrap(True)
        v.addWidget(self.clips_lbl)

        form = QtWidgets.QFormLayout()
        self.t0 = QtWidgets.QDoubleSpinBox()
        self.t0.setRange(0.0, 100000.0); self.t0.setDecimals(3); self.t0.setSuffix(" s"); self.t0.setValue(3.0)
        self.t0.setToolTip("Left bound — drop everything before this time (e.g. the ego-motion start).")
        form.addRow("Start (left bound)", self.t0)
        endrow = QtWidgets.QHBoxLayout()
        self.to_end = QtWidgets.QCheckBox("to end"); self.to_end.setChecked(True)
        self.t1 = QtWidgets.QDoubleSpinBox()
        self.t1.setRange(0.0, 100000.0); self.t1.setDecimals(3); self.t1.setSuffix(" s"); self.t1.setEnabled(False)
        self.t1.setToolTip("Right bound — drop everything after this time.")
        self.to_end.toggled.connect(lambda on: self.t1.setEnabled(not on))
        endrow.addWidget(self.to_end); endrow.addWidget(self.t1, 1)
        form.addRow("End (right bound)", endrow)
        self.subdir = QtWidgets.QLineEdit("trimmed")
        form.addRow("Output subfolder", self.subdir)
        v.addLayout(form)

        self.status = QtWidgets.QLabel(""); self.status.setObjectName("muted")
        v.addWidget(self.status)
        brow = QtWidgets.QHBoxLayout(); brow.addStretch(1)
        self.trim_btn = QtWidgets.QPushButton("Trim all"); self.trim_btn.setObjectName("primary")
        self.trim_btn.setIcon(icons.icon("cut", color="ACCENT_TEXT"))
        self.trim_btn.clicked.connect(self._trim)
        close_btn = QtWidgets.QPushButton("Close"); close_btn.clicked.connect(self.close)
        brow.addWidget(self.trim_btn); brow.addWidget(close_btn)
        v.addLayout(brow)

        self.folder_edit.textChanged.connect(self._refresh_clips)
        self._refresh_clips()

    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Pick a capture folder",
                                                       self.folder_edit.text() or "")
        if d:
            self.folder_edit.setText(d)

    def _refresh_clips(self):
        folder = self.folder_edit.text().strip()
        raws = sorted(glob.glob(os.path.join(folder, "*.raw"))) if folder and os.path.isdir(folder) else []
        if raws:
            self.clips_lbl.setText(f"{len(raws)} clip(s): " + ", ".join(os.path.basename(r) for r in raws))
        else:
            self.clips_lbl.setText("No .raw clips found in this folder.")
        self.trim_btn.setEnabled(bool(raws))

    def _trim(self):
        folder = self.folder_edit.text().strip()
        if not (folder and os.path.isdir(folder)):
            return
        t1 = None if self.to_end.isChecked() else self.t1.value()
        self.trim_btn.setEnabled(False)
        self.status.setText("Trimming… (decoding each clip)")
        self._worker = _TrimWorker(folder, self.t0.value(), t1, self.subdir.text().strip() or "trimmed")
        self._worker.progressed.connect(lambda nm: self.status.setText(f"Trimming… {nm}"))
        self._worker.done.connect(self._done)
        self._worker.failed.connect(self._fail)
        self._worker.start()

    def _done(self, m):
        self.trim_btn.setEnabled(True)
        self.status.setText(f"Done — {m['n_clips']} clip(s) → {m['out_dir']}")
        try:
            from gottlux.io.paths import open_in_file_browser
            open_in_file_browser(m["out_dir"])
        except Exception:
            pass
        lines = "\n".join(f"  {c['out']}  ({c['n_events']:,} ev, kept {c['kept_s']:g} s)"
                          for c in m["clips"])
        QtWidgets.QMessageBox.information(
            self, "Batch-trim", f"Trimmed {m['n_clips']} clip(s) → {m['out_dir']}\n\n{lines}")

    def _fail(self, msg):
        self.trim_btn.setEnabled(True)
        self.status.setText("Failed.")
        QtWidgets.QMessageBox.critical(self, "Batch-trim", msg)
