"""
filters.py — one live noise-filter suite, shared by every tab.

:class:`FilterController` holds the program-wide :class:`~gottlux.core.denoise.FilterSettings`
and a ``changed`` signal; every view binds to the *same* controller and re-renders when it
fires, so toggling a filter updates the live viewer, the multi-clip slate, the range lab, the
space-time cloud and the event-rate tower together — applied live, with no re-decode.

:class:`FilterBar` is the compact strip of controls (master enable · polarity · hot-pixel ·
refractory · background-activity) embedded once in the main window's toolbar area.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from gottlux.core.denoise import FilterSettings


class FilterController(QtCore.QObject):
    """Program-wide live denoise settings + a change signal every view subscribes to."""

    changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = FilterSettings()

    def apply(self, win):
        """Filter an EventWindow with the current settings (no-op when inactive)."""
        from gottlux.core.denoise import filter_window
        return filter_window(win, self.settings)

    def update(self, **kw):
        for k, v in kw.items():
            setattr(self.settings, k, v)
        self.changed.emit()


class FilterBar(QtWidgets.QWidget):
    """Live denoise controls bound to a :class:`FilterController` (one per program)."""

    def __init__(self, controller: FilterController, parent=None):
        super().__init__(parent)
        self.ctl = controller
        s = controller.settings

        self.enable = QtWidgets.QCheckBox("Filters")
        self.enable.setToolTip("Master switch for the live noise-reduction suite (applies to "
                               "every tab).")
        self.enable.setChecked(s.enabled)
        self.enable.toggled.connect(lambda v: self.ctl.update(enabled=v))

        self.pol = QtWidgets.QComboBox(); self.pol.addItems(["both", "on", "off"])
        self.pol.setToolTip("Keep both polarities, or only ON / only OFF events.")
        self.pol.currentTextChanged.connect(lambda v: self.ctl.update(polarity=v))

        self.hot = QtWidgets.QCheckBox("Hot-px")
        self.hot.setToolTip("Drop the hottest-firing pixels (stuck / flickering).")
        self.hot.toggled.connect(lambda v: self.ctl.update(hot_pixel=v))
        self.hot_pct = QtWidgets.QDoubleSpinBox(); self.hot_pct.setRange(90.0, 100.0)
        self.hot_pct.setDecimals(2); self.hot_pct.setValue(s.hot_pct); self.hot_pct.setSuffix(" %")
        self.hot_pct.setFixedWidth(78)
        self.hot_pct.setToolTip("Pixels above this count-percentile (within the window) are dropped.")
        self.hot_pct.valueChanged.connect(lambda v: self.ctl.update(hot_pct=v))

        self.refr = QtWidgets.QCheckBox("Refractory")
        self.refr.setToolTip("Per-pixel dead time — drop events arriving too soon after the "
                             "previous kept event at the same pixel.")
        self.refr.toggled.connect(lambda v: self.ctl.update(refractory=v))
        self.refr_us = QtWidgets.QSpinBox(); self.refr_us.setRange(1, 1_000_000)
        self.refr_us.setValue(int(s.refractory_us)); self.refr_us.setSuffix(" µs")
        self.refr_us.setFixedWidth(96)
        self.refr_us.valueChanged.connect(lambda v: self.ctl.update(refractory_us=float(v)))

        self.baf = QtWidgets.QCheckBox("Denoise (BAF)")
        self.baf.setToolTip("Background-activity filter: keep an event only if a neighbouring "
                            "pixel fired within the correlation window — removes isolated shot "
                            "noise. The strongest general denoiser.")
        self.baf.toggled.connect(lambda v: self.ctl.update(baf=v))
        self.baf_us = QtWidgets.QSpinBox(); self.baf_us.setRange(1, 1_000_000)
        self.baf_us.setValue(int(s.baf_dt_us)); self.baf_us.setSuffix(" µs")
        self.baf_us.setFixedWidth(96)
        self.baf_us.setToolTip("Neighbour correlation window for the background-activity filter.")
        self.baf_us.valueChanged.connect(lambda v: self.ctl.update(baf_dt_us=float(v)))

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 6, 0)
        lay.addWidget(self.enable)
        lay.addSpacing(6)
        lay.addWidget(QtWidgets.QLabel("Pol")); lay.addWidget(self.pol)
        lay.addSpacing(6)
        lay.addWidget(self.hot); lay.addWidget(self.hot_pct)
        lay.addSpacing(6)
        lay.addWidget(self.refr); lay.addWidget(self.refr_us)
        lay.addSpacing(6)
        lay.addWidget(self.baf); lay.addWidget(self.baf_us)
        lay.addStretch(1)
