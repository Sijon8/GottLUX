"""
cellstyle.py — the "how is this cell drawn" panel shared by the Canvas composer and the Timeline.

One :class:`CellStyleEditor` edits the visualization of a :class:`~gottlux.core.canvas.CanvasClip`:

* **Visualization** — *Event frame* (the classic accumulate → tone-map → colormap picture,
  with its mode, colormap, tone map + γ and accumulation window) or any **Studio look**
  (trails, accumulated frame, time surface, light painting, space-time cloud, stroboscope,
  waterfall, flicker hue, radar sweep), whose own parameters appear below it, auto-built like
  the Studio's, with its ON/OFF palette, denoise and bloom;
* **Label** and **Time stamp** — burned into the cell in the preview and in the video.

Both composers used to build the same rows by hand; now they embed this widget, so a cell
in a canvas block and a plain timeline clip offer exactly the same choices. The classic
widgets keep their old attribute names (``cb_mode``, ``cb_cmap``, ``cb_tone``, ``sp_gamma``,
``sp_accum``), which the composers re-export.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from gottlux.app.colorschemes import ColormapCombo, PaletteCombo
from gottlux.app.widgets import ParamPanel
from gottlux.core import tonemap

#: Accumulation modes of the classic event frame.
MODES = ["count", "time_surface", "polarity", "polarity_ratio", "on", "off", "binary"]

#: The Visualization entry for the classic event frame.
FRAME_TITLE = "Event frame (mode · colormap)"


def look_choices() -> list:
    """``[(name, title)]``: the classic frame (name ``""``) then every Studio look."""
    from gottlux.studio.looks import list_looks
    return [("", FRAME_TITLE)] + [(n, lk.title) for n, lk in list_looks().items()]


class CellStyleEditor(QtWidgets.QWidget):
    """Edits a cell's visualization; :attr:`changed` fires on every edit (live apply)."""

    changed = QtCore.Signal()

    def __init__(self, parent=None, scope="cell"):
        super().__init__(parent)
        self._loading = False
        self._look_params: dict = {}
        self.panel = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        top = QtWidgets.QFormLayout()
        self.cb_look = QtWidgets.QComboBox()
        for name, title in look_choices():
            self.cb_look.addItem(title, name)
        self.cb_look.setToolTip(
            f"How this {scope} is drawn: the classic event frame, or a Studio look. "
            "Different cells of one canvas can use different visualizations; the video "
            "export draws each one as previewed.")
        self.cb_look.currentIndexChanged.connect(self._on_look)
        top.addRow("Visualization", self.cb_look)
        self.look_desc = QtWidgets.QLabel("")
        self.look_desc.setObjectName("muted"); self.look_desc.setWordWrap(True)
        top.addRow(self.look_desc)
        lay.addLayout(top)

        # --- classic event frame ---
        self.classic = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.classic)
        form.setContentsMargins(0, 0, 0, 0)
        self.cb_mode = QtWidgets.QComboBox(); self.cb_mode.addItems(MODES)
        self.cb_mode.setToolTip(f"Accumulation mode for this {scope} (count, polarity, …).")
        self.cb_mode.currentIndexChanged.connect(self._emit)
        form.addRow("Mode", self.cb_mode)
        self.cb_cmap = ColormapCombo()
        self.cb_cmap.setToolTip(f"Colormap for this {scope}. 'Add a colour scheme…' at the "
                                "end of the list makes your own.")
        self.cb_cmap.currentIndexChanged.connect(self._emit)
        form.addRow("Color", self.cb_cmap)
        self.cb_tone = QtWidgets.QComboBox(); self.cb_tone.addItems(tonemap.EXPRESSIONS)
        self.cb_tone.setCurrentText("sqrt")
        self.cb_tone.setToolTip(f"Tone-map expression for this {scope} (dynamic-range "
                                "compression before the colormap).")
        self.cb_tone.currentIndexChanged.connect(self._emit)
        self.sp_gamma = QtWidgets.QDoubleSpinBox()
        self.sp_gamma.setRange(0.1, 3.0); self.sp_gamma.setSingleStep(0.05)
        self.sp_gamma.setValue(0.5); self.sp_gamma.setPrefix("γ ")
        self.sp_gamma.setToolTip("Exponent for the 'gamma' expression.")
        self.sp_gamma.valueChanged.connect(self._emit)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.cb_tone); row.addWidget(self.sp_gamma)
        form.addRow("Tone map", row)
        self.sp_accum = QtWidgets.QDoubleSpinBox()
        self.sp_accum.setRange(1e-5, 2.0); self.sp_accum.setDecimals(5)
        self.sp_accum.setValue(0.02); self.sp_accum.setSingleStep(0.005)
        self.sp_accum.setSuffix(" s")
        self.sp_accum.setToolTip(f"Accumulation window (exposure) for this {scope} only.")
        self.sp_accum.valueChanged.connect(self._emit)
        form.addRow("Accum", self.sp_accum)
        lay.addWidget(self.classic)

        # --- a Studio look ---
        self.lookbox = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(self.lookbox)
        lv.setContentsMargins(0, 0, 0, 0)
        self._panel_holder = QtWidgets.QVBoxLayout()
        lv.addLayout(self._panel_holder)
        sform = QtWidgets.QFormLayout()
        self.cb_palette = PaletteCombo()
        self.cb_palette.setToolTip("ON / OFF / background colours of the look.")
        self.cb_palette.currentIndexChanged.connect(self._emit)
        sform.addRow("Palette", self.cb_palette)
        self.chk_denoise = QtWidgets.QCheckBox("Denoise (hot pixels + background activity)")
        self.chk_denoise.setChecked(True)
        self.chk_denoise.toggled.connect(self._emit)
        sform.addRow("", self.chk_denoise)
        self.sp_bloom = QtWidgets.QDoubleSpinBox()
        self.sp_bloom.setRange(0.0, 3.0); self.sp_bloom.setSingleStep(0.1)
        self.sp_bloom.setToolTip("Glow around bright events (0 = off).")
        self.sp_bloom.valueChanged.connect(self._emit)
        sform.addRow("Bloom", self.sp_bloom)
        lv.addLayout(sform)
        lay.addWidget(self.lookbox)

        # --- labels (both kinds) ---
        lform = QtWidgets.QFormLayout()
        self.ed_label = QtWidgets.QLineEdit()
        self.ed_label.setPlaceholderText("optional title drawn in the corner")
        self.ed_label.setToolTip("A title burned into this cell's top-left corner — in the "
                                 "preview and in the exported video.")
        self.ed_label.textChanged.connect(self._emit)
        lform.addRow("Label", self.ed_label)
        self.chk_stamp = QtWidgets.QCheckBox("Time stamp")
        self.chk_stamp.setToolTip("Draw the recording time this cell shows (a look draws "
                                  "its own readout, e.g. the stroboscope's f0 and phase).")
        self.chk_stamp.toggled.connect(self._emit)
        lform.addRow("", self.chk_stamp)
        self.cb_fit = QtWidgets.QComboBox()
        for key, label in (("auto", "Auto (frame fills, look keeps aspect)"),
                           ("fit", "Keep aspect (letterbox)"), ("stretch", "Stretch to cell")):
            self.cb_fit.addItem(label, key)
        self.cb_fit.setToolTip("How the picture fills its cell. The .raw export places the "
                               "events the same way.")
        self.cb_fit.currentIndexChanged.connect(self._emit)
        lform.addRow("Fit", self.cb_fit)
        lay.addLayout(lform)
        self._sync_visibility()

    # ------------------------------------------------------------------ state
    def look(self) -> str:
        return str(self.cb_look.currentData() or "")

    def _emit(self, *_):
        if not self._loading:
            self.changed.emit()

    def _sync_visibility(self):
        look = self.look()
        self.classic.setVisible(not look)
        self.lookbox.setVisible(bool(look))
        if look:
            from gottlux.studio.looks import get_look
            self.look_desc.setText(get_look(look).description)
        else:
            self.look_desc.setText("Events per pixel in the accumulation window, "
                                   "tone-mapped through a colormap.")

    def _build_panel(self, look, values=None):
        if self.panel is not None:
            self._panel_holder.removeWidget(self.panel)
            self.panel.deleteLater()
            self.panel = None
        if not look:
            return
        from gottlux.studio.looks import get_look
        lk = get_look(look)
        self.panel = ParamPanel(lk.PARAMS)
        if values:
            self.panel.set_values(lk.resolve(values))
        self.panel.changed.connect(self._on_params)
        self._panel_holder.addWidget(self.panel)

    def _on_look(self, *_):
        look = self.look()
        if look and not self._loading:            # a fresh look starts as the Studio does
            b = QtCore.QSignalBlocker(self.chk_denoise)
            self.chk_denoise.setChecked(True)
            del b
        self._build_panel(look, self._look_params if self._loading else None)
        self._sync_visibility()
        self._emit()

    def _on_params(self, _values):
        self._emit()

    # ------------------------------------------------------------------ load / store
    def load(self, clip) -> None:
        """Show *clip*'s visualization settings (without emitting :attr:`changed`)."""
        self._loading = True
        try:
            self.cb_mode.setCurrentText(clip.mode)
            self.cb_cmap.setCurrentText(clip.colormap)
            self.cb_tone.setCurrentText(clip.tonemap)
            self.sp_gamma.setValue(clip.gamma)
            self.sp_accum.setValue(clip.accumulation_s)
            self._look_params = dict(clip.look_params or {})
            i = self.cb_look.findData(clip.look or "")
            prev = self.cb_look.currentIndex()
            self.cb_look.setCurrentIndex(i if i >= 0 else 0)
            if self.cb_look.currentIndex() == prev:       # same look: refresh its values
                self._build_panel(self.look(), self._look_params)
                self._sync_visibility()
            self.cb_palette.setCurrentText(clip.palette)
            self.chk_denoise.setChecked(bool(clip.denoise))
            self.sp_bloom.setValue(float(clip.bloom))
            self.ed_label.setText(clip.label or "")
            self.chk_stamp.setChecked(bool(clip.stamp))
            j = self.cb_fit.findData(getattr(clip, "fit", "auto"))
            self.cb_fit.setCurrentIndex(j if j >= 0 else 0)
        finally:
            self._loading = False

    def store(self, clip) -> None:
        """Write the panel into *clip*."""
        clip.mode = self.cb_mode.currentText()
        clip.colormap = self.cb_cmap.currentText()
        clip.tonemap = self.cb_tone.currentText()
        clip.gamma = float(self.sp_gamma.value())
        clip.accumulation_s = float(self.sp_accum.value())
        clip.look = self.look()
        clip.look_params = dict(self.panel.values()) if (self.panel and clip.look) else {}
        clip.palette = self.cb_palette.currentText()
        clip.denoise = self.chk_denoise.isChecked()
        clip.bloom = float(self.sp_bloom.value())
        clip.label = self.ed_label.text()
        clip.stamp = self.chk_stamp.isChecked()
        clip.fit = str(self.cb_fit.currentData() or "auto")
