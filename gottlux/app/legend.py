"""
legend.py — every colour map in the GUI gets a legend, so colours are never a mystery.

Three reusable pieces:

* :func:`make_colorbar` — a standalone pyqtgraph ``ColorBarItem`` for a scalar image (the live
  viewer), showing the colormap and the current value range with a units label.
* :class:`FrequencyLegend` — the flicker-map legend: a frequency gradient over ``[fmin, fmax]``
  plus a note that opacity encodes SNR/confidence.
* :class:`ColorKey` — a compact widget for the 3-D view explaining the current *color-by* mode
  (discrete polarity swatches, or a continuous time/density gradient).

The *colormaps* on show are data (an inferno ramp means the same thing in either theme); the
lettering and the bar's frame are chrome and are read from :mod:`gottlux.app.style` at paint
time, so both legends follow a live light/dark switch.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from gottlux.app import style


def _cmap(name):
    """A pyqtgraph ColorMap from a matplotlib name (with a safe fallback)."""
    try:
        return pg.colormap.get(name, source="matplotlib")
    except Exception:
        return pg.colormap.get("viridis")


def make_colorbar(label="events / pixel", cmap="inferno", levels=(0, 1)):
    """A standalone, non-interactive ColorBarItem for embedding next to an ImageItem.

    The bar's frame is drawn in the theme's border colour — pyqtgraph's own default is a
    hardcoded black, which vanishes on a dark canvas and shouts on a light one.
    """
    cb = pg.ColorBarItem(values=tuple(levels), colorMap=_cmap(cmap), label=label,
                         interactive=False, width=14, pen=style.BORDER)
    return cb


def _qcolor_stops(cmap_name, n=16):
    """Sample a matplotlib colormap into (pos, QColor) gradient stops."""
    import matplotlib.cm as cm
    m = cm.get_cmap(cmap_name)
    stops = []
    for i in range(n):
        f = i / (n - 1)
        r, g, b, _ = m(f)
        stops.append((f, QtGui.QColor(int(r * 255), int(g * 255), int(b * 255))))
    return stops


class ColorKey(QtWidgets.QWidget):
    """A small legend strip: either discrete colour swatches or a continuous gradient.

    Configure with :meth:`set_discrete` (``[(QColor|str, label), …]``) or
    :meth:`set_gradient` (``cmap_name, lo_label, hi_label``).
    """

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self._title = title
        self._mode = "discrete"
        self._entries = []
        self._cmap = "viridis"
        self._lo = ""
        self._hi = ""
        self.setMinimumHeight(22)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_discrete(self, entries, title=None):
        self._mode = "discrete"
        self._entries = [(QtGui.QColor(c) if not isinstance(c, QtGui.QColor) else c, lbl)
                         for c, lbl in entries]
        if title is not None:
            self._title = title
        self.update()

    def set_gradient(self, cmap_name, lo_label, hi_label, title=None):
        self._mode = "gradient"
        self._cmap = cmap_name
        self._lo, self._hi = lo_label, hi_label
        if title is not None:
            self._title = title
        self.update()

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtGui.QColor(style.MUTED))
        x = 4
        if self._title:
            p.drawText(x, self.height() - 6, self._title + ":")
            x += p.fontMetrics().horizontalAdvance(self._title + ": ") + 6
        if self._mode == "discrete":
            for color, lbl in self._entries:
                p.fillRect(x, 5, 12, 12, color)
                p.setPen(QtGui.QColor(style.FG))
                p.drawText(x + 16, self.height() - 6, lbl)
                x += 16 + p.fontMetrics().horizontalAdvance(lbl) + 14
                p.setPen(QtGui.QColor(style.MUTED))
        else:
            # lo label, gradient bar, hi label — laid out left to right
            p.setPen(QtGui.QColor(style.FG))
            lo_w = p.fontMetrics().horizontalAdvance(self._lo)
            p.drawText(x, self.height() - 6, self._lo)
            x += lo_w + 6
            grad = QtGui.QLinearGradient(x, 0, x + 120, 0)
            for pos, c in _qcolor_stops(self._cmap):
                grad.setColorAt(pos, c)
            p.fillRect(x, 5, 120, 12, QtGui.QBrush(grad))
            x += 126
            p.drawText(x, self.height() - 6, self._hi)
        p.end()


class FrequencyLegend(QtWidgets.QWidget):
    """Flicker-map legend: a frequency gradient over [fmin, fmax] + an 'opacity = SNR' note."""

    def __init__(self, cmap="turbo", parent=None):
        super().__init__(parent)
        self._cmap = cmap
        self._fmin = 0.0
        self._fmax = 1.0
        self.setMinimumHeight(40)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_band(self, fmin, fmax):
        self._fmin, self._fmax = float(fmin), float(fmax)
        self.update()

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w = self.width()
        bar_w = max(80, w - 120)
        x0 = 70
        grad = QtGui.QLinearGradient(x0, 0, x0 + bar_w, 0)
        for pos, c in _qcolor_stops(self._cmap):
            grad.setColorAt(pos, c)
        p.fillRect(x0, 4, bar_w, 12, QtGui.QBrush(grad))
        p.setPen(QtGui.QColor(style.FG))
        p.drawText(0, 14, "freq (Hz)")
        p.drawText(x0 - 2, 30, f"{self._fmin:.0f}")
        hi = f"{self._fmax:.0f}"
        p.drawText(x0 + bar_w - p.fontMetrics().horizontalAdvance(hi), 30, hi)
        p.setPen(QtGui.QColor(style.MUTED))
        p.drawText(x0 + bar_w + 8, 14, "opacity = SNR")
        p.end()
