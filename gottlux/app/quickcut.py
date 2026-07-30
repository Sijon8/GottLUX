"""
quickcut.py — crop a ``.raw`` to a time window **without opening (decoding) it**.

For a multi-GB recording the slow part is the one-time decode that a normal *Open* pays before you
can even see the timeline. When all you want is to shorten the file, that decode is wasted. This
dialog indexes the EVT2.1 byte stream (a fast I/O pass, no event expansion) to learn the duration,
lets you pick In/Out, and writes the cropped ``.raw`` directly on the bytes — no decode, bounded
RAM. The engine is :mod:`gottlux.io.rawcut`; this is just the small Qt front-end.

Reached from **File → Quick-cut a .raw (no decode)…**. Non-EVT2.1 files (EVT2.0 / EVT3) can't be
byte-cut, so the dialog points you at opening them normally.
"""
from __future__ import annotations

import os

from PySide6 import QtWidgets

from gottlux.app import icons
from gottlux.app.uikit import with_progress


class QuickCutDialog(QtWidgets.QDialog):
    """Pick a ``.raw``, index it (no decode), choose In/Out, and write the cropped clip."""

    def __init__(self, parent=None, path=None):
        super().__init__(parent)
        self.setWindowTitle("Quick-cut a .raw — crop without decoding")
        self.setMinimumWidth(560)
        self._index = None
        self._dur = 0.0

        self.in_edit = QtWidgets.QLineEdit(path or "")
        self.in_edit.setReadOnly(True)
        browse = QtWidgets.QPushButton("Choose .raw…"); browse.clicked.connect(self._browse_in)
        self.info = QtWidgets.QLabel("Choose a .raw file to index it."); self.info.setObjectName("muted")

        self.in_s = QtWidgets.QDoubleSpinBox(); self.out_s = QtWidgets.QDoubleSpinBox()
        for sp in (self.in_s, self.out_s):
            sp.setDecimals(3); sp.setRange(0, 0); sp.setSuffix(" s"); sp.setEnabled(False)
        self.in_s.setToolTip("In point (s) — start of the kept clip.")
        self.out_s.setToolTip("Out point (s) — end of the kept clip.")
        self.in_s.valueChanged.connect(self._update_out_default)

        self.out_edit = QtWidgets.QLineEdit()
        out_browse = QtWidgets.QPushButton("Browse…"); out_browse.clicked.connect(self._browse_out)

        form = QtWidgets.QFormLayout()
        irow = QtWidgets.QHBoxLayout(); irow.addWidget(self.in_edit, 1); irow.addWidget(browse)
        form.addRow("Source", irow)
        form.addRow("", self.info)
        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(QtWidgets.QLabel("In")); trow.addWidget(self.in_s)
        trow.addWidget(QtWidgets.QLabel("Out")); trow.addWidget(self.out_s); trow.addStretch(1)
        form.addRow("Keep", trow)
        orow = QtWidgets.QHBoxLayout(); orow.addWidget(self.out_edit, 1); orow.addWidget(out_browse)
        form.addRow("Output (.mp4→.raw)", orow)

        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(form)
        hint = QtWidgets.QLabel("Cuts an EVT2.1 file straight on the bytes — no decode, low RAM — so "
                                "you can crop a multi-GB recording without the slow open.")
        hint.setObjectName("muted"); hint.setWordWrap(True); v.addWidget(hint)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        self.cut_btn = bb.addButton("Cut", QtWidgets.QDialogButtonBox.AcceptRole)
        self.cut_btn.setIcon(icons.icon("cut"))
        self.cut_btn.setEnabled(False); self.cut_btn.clicked.connect(self._cut)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

        if path and os.path.exists(path):
            self._scan(path)

    # ----- input -----
    def _browse_in(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose a .raw to cut", "",
                                                        "EVT raw (*.raw)")
        if path:
            self._scan(path)

    def _scan(self, path):
        from gottlux.io import rawcut
        try:
            ix = with_progress(self, "Indexing .raw (no decode)",
                               lambda cb: rawcut.scan(path, progress=cb),
                               label="Scanning the file…")
        except rawcut.UnsupportedRaw:
            self.info.setText("Not an EVT2.1 file — open it normally (File → Open) to cut.")
            self.cut_btn.setEnabled(False)
            return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Quick-cut", str(e))
            return
        self._index = ix
        self._dur = float(ix["duration_s"])
        self.in_edit.setText(path)
        self.info.setText(f"{ix['width']}×{ix['height']} px · {self._dur:.3f} s · "
                          f"{ix['n_time_high']:,} time-marks  (indexed, not decoded)")
        for sp in (self.in_s, self.out_s):
            sp.setRange(0.0, self._dur); sp.setEnabled(True)
        self.in_s.setValue(0.0); self.out_s.setValue(self._dur)
        stem = os.path.splitext(path)[0]
        self.out_edit.setText(f"{stem}_cut.raw")
        self.cut_btn.setEnabled(True)

    def _update_out_default(self, v):
        if self.out_s.value() <= v:
            self.out_s.setValue(min(self._dur, v + min(1.0, self._dur)))

    def _browse_out(self):
        out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Cut output", self.out_edit.text(),
                                                       "EVT raw (*.raw)")
        if out:
            self.out_edit.setText(out)

    # ----- run -----
    def _cut(self):
        if self._index is None:
            return
        path = self.in_edit.text().strip()
        out = self.out_edit.text().strip()
        if not out:
            QtWidgets.QMessageBox.warning(self, "Quick-cut", "Choose an output path."); return
        if not out.lower().endswith(".raw"):
            out += ".raw"
        t0, t1 = self.in_s.value(), self.out_s.value()
        if t1 - t0 < 1e-3:
            QtWidgets.QMessageBox.warning(self, "Quick-cut", "Keep a non-empty range."); return
        from gottlux.io import rawcut
        try:
            res = with_progress(
                self, "Cutting .raw (no decode)",
                lambda cb: rawcut.cut_evt21(path, out, t0=t0, t1=t1, index=self._index, progress=cb),
                label="Writing the clip…")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Quick-cut", str(e)); return
        from gottlux.io.paths import open_in_file_browser
        open_in_file_browser(os.path.dirname(os.path.abspath(out)))
        QtWidgets.QMessageBox.information(
            self, "Quick-cut",
            f"Cut [{t0:.3f}, {t1:.3f}] s → {os.path.basename(out)}\n"
            f"{res['n_events']:,} events · {res['duration_s']:.3f} s  (no decode)")
        self.accept()
