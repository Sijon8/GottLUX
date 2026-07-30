"""
widgets.py — reusable Qt widgets for the gottlux GUI.

The centerpiece is :class:`ParamPanel`: it reads a detector's self-describing
:class:`~gottlux.detectors.base.Param` list and *auto-builds* a grouped panel of sliders /
spin-boxes / check-boxes / combos, each with the parameter's tooltip. Tuning a detector is
then immediate and discoverable — no per-detector UI code, and any new detector you write
gets a full tuning panel for free.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class ParamPanel(QtWidgets.QWidget):
    """A live, grouped editor auto-built from a list of :class:`Param` specs.

    Emits :attr:`changed` (debounced) whenever any value changes; :meth:`values` returns the
    current ``{key: value}`` dict ready to pass as detector overrides.
    """
    changed = QtCore.Signal(dict)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self._specs = {p.key: p for p in params}
        self._editors = {}
        self._debounce = QtCore.QTimer(self, singleShot=True, interval=180)
        self._debounce.timeout.connect(lambda: self.changed.emit(self.values()))

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        groups: dict[str, list] = {}
        for p in params:
            groups.setdefault(p.group, []).append(p)
        for group, plist in groups.items():
            box = QtWidgets.QGroupBox(group)
            form = QtWidgets.QFormLayout(box)
            form.setLabelAlignment(QtCore.Qt.AlignRight)
            for p in plist:
                ed = self._make_editor(p)
                self._editors[p.key] = ed
                lab = QtWidgets.QLabel(p.label + (f" ({p.unit})" if p.unit else ""))
                if p.help:
                    lab.setToolTip(p.help)
                    ed.setToolTip(p.help)
                form.addRow(lab, ed)
            outer.addWidget(box)
        outer.addStretch(1)

    def _make_editor(self, p):
        if p.kind == "bool":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(p.default))
            w.toggled.connect(self._debounce.start)
            return w
        if p.kind == "choice":
            w = QtWidgets.QComboBox()
            w.addItems([str(c) for c in p.choices])
            if str(p.default) in [str(c) for c in p.choices]:
                w.setCurrentText(str(p.default))
            w.currentIndexChanged.connect(self._debounce.start)
            return w
        # numeric: slider + spinbox combo
        return _SliderSpin(p, self._debounce.start)

    def values(self) -> dict:
        out = {}
        for k, ed in self._editors.items():
            p = self._specs[k]
            if p.kind == "bool":
                out[k] = ed.isChecked()
            elif p.kind == "choice":
                out[k] = ed.currentText()
            else:
                out[k] = ed.value()
        return out

    def set_values(self, d: dict):
        for k, v in d.items():
            ed = self._editors.get(k)
            if ed is None:
                continue
            p = self._specs[k]
            blocker = QtCore.QSignalBlocker(ed)
            if p.kind == "bool":
                ed.setChecked(bool(v))
            elif p.kind == "choice":
                ed.setCurrentText(str(v))
            else:
                ed.set_value(v)
            del blocker


class _SliderSpin(QtWidgets.QWidget):
    """A horizontal slider tied to a spin-box, sharing one parameter's range/step/type."""

    def __init__(self, p, on_change):
        super().__init__()
        self._p = p
        self._is_int = p.kind == "int"
        step = p.step or ((p.hi - p.lo) / 100.0)
        self._scale = 1.0 if self._is_int else (1.0 / step if step else 100.0)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(int(round(p.lo * self._scale)))
        self.slider.setMaximum(int(round(p.hi * self._scale)))
        if self._is_int:
            self.spin = QtWidgets.QSpinBox()
            self.spin.setRange(int(p.lo), int(p.hi))
            self.spin.setSingleStep(max(1, int(step)))
        else:
            self.spin = QtWidgets.QDoubleSpinBox()
            self.spin.setRange(p.lo, p.hi)
            self.spin.setDecimals(max(0, len(str(step).split(".")[-1])) if "." in str(step) else 3)
            self.spin.setSingleStep(step)
        self.spin.setMaximumWidth(86)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.set_value(p.default)
        self._on_change = on_change
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _from_slider(self, v):
        val = v / self._scale
        blocker = QtCore.QSignalBlocker(self.spin)
        self.spin.setValue(int(val) if self._is_int else val)
        del blocker
        self._on_change()

    def _from_spin(self, v):
        blocker = QtCore.QSignalBlocker(self.slider)
        self.slider.setValue(int(round(v * self._scale)))
        del blocker
        self._on_change()

    def value(self):
        return int(self.spin.value()) if self._is_int else float(self.spin.value())

    def set_value(self, v):
        self.spin.setValue(int(v) if self._is_int else float(v))
        self.slider.setValue(int(round(float(v) * self._scale)))


def hline():
    """A thin horizontal separator."""
    f = QtWidgets.QFrame()
    f.setFrameShape(QtWidgets.QFrame.HLine)
    f.setObjectName("muted")
    return f
