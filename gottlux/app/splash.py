"""
splash.py — a lightweight boot splash with real, staged progress.

The main GUI needs a few seconds (and on a cold start — first launch of the day,
with the project living on a synced/AV-scanned folder — up to a minute) before its
window appears: heavy scientific imports (scipy/pyqtgraph/numba) plus per-panel
construction all happen before the first ``show()``. This splash paints within a
couple hundred milliseconds — *before* any of that work — and is advanced through
genuine milestones, so the user can see that the app is booting and how far along
it is rather than staring at nothing.

It is deliberately app-agnostic: hand it a title/subtitle (the default title names
the running build — "GottLUX <version>") and drive it with :meth:`step`. The only
dependencies are PySide6, the palette constants (plain strings) from
:mod:`gottlux.app.style`, and the version string, so importing it never drags in
the heavy stack — which is the whole point of showing it first.

The card is styled from the palette *as loaded when it is constructed*, so it comes up in
the user's chosen theme: the entry point calls :func:`gottlux.app.style.load_theme` (which
needs no ``QApplication`` and no pyqtgraph) before the splash is built.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from gottlux import __version__  # the parent package is already imported; adds no weight
from gottlux.app import style


class BootSplash(QtWidgets.QWidget):
    """A small, centered, frameless progress card shown while the app boots."""

    def __init__(self, title: str = f"GottLUX {__version__}",
                 subtitle: str = "event-based-sensor analysis instrument"):
        super().__init__(None, QtCore.Qt.SplashScreen | QtCore.Qt.FramelessWindowHint
                         | QtCore.Qt.WindowStaysOnTopHint)
        # Translucent window so the card can have rounded corners.
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        card = QtWidgets.QFrame(self)
        card.setObjectName("card")
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(card)

        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(30, 26, 30, 22)
        lay.setSpacing(4)
        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setObjectName("title")
        sub_lbl = QtWidgets.QLabel(subtitle)
        sub_lbl.setObjectName("subtitle")
        lay.addWidget(title_lbl)
        lay.addWidget(sub_lbl)
        lay.addSpacing(16)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        lay.addWidget(self.bar)
        self.status = QtWidgets.QLabel("Starting up…")
        self.status.setObjectName("status")
        lay.addWidget(self.status)

        self.setStyleSheet(f"""
            #card {{ background: {style.BG2}; border: 1px solid {style.BORDER};
                     border-radius: 12px; }}
            #title {{ color: {style.FG}; font-size: 23px; font-weight: 700;
                      letter-spacing: 1px; }}
            #subtitle {{ color: {style.MUTED}; font-size: 12px; }}
            #status {{ color: {style.ACCENT}; font-size: 11px; }}
            QProgressBar {{ background: {style.PANEL}; border: none; border-radius: 3px; }}
            QProgressBar::chunk {{ background: {style.ACCENT}; border-radius: 3px; }}
        """)
        self.setFixedSize(470, 200)
        self._center()

    def _center(self):
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.move(geo.center().x() - self.width() // 2,
                      geo.center().y() - self.height() // 2)

    def step(self, frac: float, msg: str | None = None):
        """Advance the bar to ``frac`` (0..1) with an optional status line, and repaint now.

        Called from the main thread between blocking construction steps; the explicit
        ``processEvents`` is what lets the splash actually paint its new state before the
        next (blocking) step begins.
        """
        self.bar.setValue(max(0, min(100, int(round(frac * 100)))))
        if msg is not None:
            self.status.setText(msg)
        QtWidgets.QApplication.processEvents()

    def finish(self, window: QtWidgets.QWidget | None = None):
        """Take the splash down once the real window is up."""
        self.close()
