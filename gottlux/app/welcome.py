"""
welcome.py — the first-launch example picker.

When the app opens with nothing loaded it offers the bundled demo recordings (discovered by
:mod:`gottlux.examples`) as one-click cards, so a new user has something to explore immediately.
A "Show demos at startup" checkbox (persisted via :class:`QtCore.QSettings`) lets it be turned off.
The dialog only *chooses* a path; the caller does the actual (background) load.
"""
from __future__ import annotations

import os
from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from gottlux import examples as ex

_SETTINGS_KEY = "welcome/show_examples_on_start"


def show_examples_on_start() -> bool:
    """Whether the startup example picker should be shown (user preference, default on)."""
    s = QtCore.QSettings()
    return s.value(_SETTINGS_KEY, True, type=bool)


class WelcomeDialog(QtWidgets.QDialog):
    """A startup chooser listing the bundled example recordings. Sets :attr:`chosen_path` when the
    user picks one (then ``accept()``); the caller loads it."""

    def __init__(self, parent=None, items: Optional[List[ex.Example]] = None):
        super().__init__(parent)
        self.setWindowTitle("GottLUX — open an example")
        self.setModal(True)
        self.chosen_path: Optional[str] = None
        self._items = items if items is not None else ex.list_examples()
        self.resize(620, 460)
        self._build()

    # ------------------------------------------------------------------ UI
    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 14)
        lay.setSpacing(10)

        title = QtWidgets.QLabel("Welcome to GottLUX")
        title.setStyleSheet("font-size: 17px; font-weight: 600;")
        lay.addWidget(title)

        src = ex.examples_dir()
        sub = QtWidgets.QLabel(
            "Pick a bundled example recording to explore, or close this and open your own clip."
            + (f"\nExamples folder: {src}" if src else ""))
        sub.setWordWrap(True)
        sub.setObjectName("muted")          # themed by the app stylesheet, both light and dark
        lay.addWidget(sub)

        self.list = QtWidgets.QListWidget()
        self.list.setAlternatingRowColors(True)
        self.list.setUniformItemSizes(False)
        self.list.itemDoubleClicked.connect(lambda *_: self._open_selected())
        for e in self._items:
            it = QtWidgets.QListWidgetItem()
            badge = "◑ rotating" if e.is_rotating else "▣ staring"
            it.setText(f"{e.title}\n     {badge}   ·   {e.detail}")
            it.setData(QtCore.Qt.UserRole, e.path)
            it.setToolTip(e.path)
            self.list.addItem(it)
        if self._items:
            self.list.setCurrentRow(0)
        else:
            empty = QtWidgets.QListWidgetItem("No example recordings found.\n     "
                                              "Drop .raw files into a RawExamples/ folder.")
            empty.setFlags(QtCore.Qt.NoItemFlags)
            self.list.addItem(empty)
        lay.addWidget(self.list, 1)

        self.remember = QtWidgets.QCheckBox("Show demos at startup")
        self.remember.setChecked(show_examples_on_start())
        lay.addWidget(self.remember)

        btns = QtWidgets.QHBoxLayout()
        browse = QtWidgets.QPushButton("Open file…")
        browse.setToolTip("Open your own .raw recording instead.")
        browse.clicked.connect(self._browse)
        btns.addWidget(browse)
        btns.addStretch(1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.reject)
        self.open_btn = QtWidgets.QPushButton("Open example")
        self.open_btn.setDefault(True)
        self.open_btn.setEnabled(bool(self._items))
        self.open_btn.clicked.connect(self._open_selected)
        btns.addWidget(close)
        btns.addWidget(self.open_btn)
        lay.addLayout(btns)

    # ------------------------------------------------------------------ actions
    def _persist(self):
        QtCore.QSettings().setValue(_SETTINGS_KEY, self.remember.isChecked())

    def _open_selected(self):
        it = self.list.currentItem()
        path = it.data(QtCore.Qt.UserRole) if it is not None else None
        if not path:
            return
        self.chosen_path = path
        self._persist()
        self.accept()

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open event recording", ex.examples_dir() or "",
            "Event recordings (*.raw *.h5 *.hdf5 *.meta.json);;All files (*)")
        if path:
            self.chosen_path = path
            self._persist()
            self.accept()

    def reject(self):
        self._persist()
        super().reject()
