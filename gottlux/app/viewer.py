"""
viewer.py — the live event viewer: a fast, scrubbable, playable window into the stream.

A pyqtgraph :class:`ImageItem` shows an accumulated event frame at the shared cursor time,
with a **colorbar legend** so the mapping from colour to value is always explicit. The
transport (play / seek / speed / accumulation) is the shared :class:`~gottlux.app.transport`
bar, so seeking and accumulation stay consistent with every other view. Switch accumulation
mode (count / time-surface / polarity / ON / OFF / binary) and colormap live.
"""
from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from gottlux.app import icons
from gottlux.app import style
from gottlux.app.legend import make_colorbar
from gottlux.app.transport import TransportBar
from gottlux.core import tonemap
from gottlux.core.accumulate import accumulate_frame

_MODES = ["count", "time_surface", "polarity", "polarity_ratio", "on", "off", "binary"]
# Sequential maps for magnitude views; diverging maps for the polarity (signed) views.
_CMAPS = ["inferno", "viridis", "magma", "plasma", "cividis", "gray", "turbo",
          "coolwarm", "RdBu", "bwr", "seismic", "PiYG", "Spectral"]
_DIVERGING = {"coolwarm", "RdBu", "bwr", "seismic", "PiYG", "Spectral"}
_POLARITY_MODES = {"polarity", "polarity_ratio"}
_CLABEL = {"count": "events / pixel", "on": "ON events / px", "off": "OFF events / px",
           "polarity": "ON − OFF", "polarity_ratio": "(ON−OFF)/(ON+OFF)",
           "time_surface": "recency (decayed)", "binary": "occupancy"}


class LiveViewer(QtWidgets.QWidget):
    """Scrub/play a :class:`~gottlux.io.recording.Recording` as accumulated event frames."""

    roiChanged = QtCore.Signal(object)        # (x0, y0, x1, y1) or None

    def __init__(self, controller, filters=None, parent=None):
        super().__init__(parent)
        self.ctl = controller
        self.filters = filters          # shared live noise-filter suite (FilterController | None)
        self.rec = None
        self._static_vmax = None        # frozen white-point for 'static' scale mode

        # --- image canvas + colorbar ---
        self.glw = pg.GraphicsLayoutWidget()
        self.vb = self.glw.addViewBox(lockAspect=True, invertY=True, row=0, col=0)
        self.img = pg.ImageItem(axisOrder="row-major")
        self.vb.addItem(self.img)
        self.roi = pg.RectROI([20, 20], [80, 80], pen=pg.mkPen(style.ACCENT, width=2))
        self.roi.setVisible(False)
        self.vb.addItem(self.roi)
        self.roi.sigRegionChangeFinished.connect(self._emit_roi)
        self.cbar = make_colorbar("events / pixel", "inferno", (0, 1))
        self.glw.addItem(self.cbar, row=0, col=1)

        # --- controls ---
        self.transport = TransportBar(self.ctl, host=self)
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(_MODES)
        self.mode.currentIndexChanged.connect(self._on_mode)
        self.cmap = QtWidgets.QComboBox(); self.cmap.addItems(_CMAPS)
        self.cmap.currentIndexChanged.connect(self._apply_cmap)
        self.mode.setToolTip("How events map to pixel value: count, time-surface (sharp motion), "
                             "polarity (ON−OFF, raw signed count), polarity_ratio "
                             "((ON−OFF)/(ON+OFF) ∈ [−1,1] — bounded, won't blow out in dense "
                             "regions), ON only, OFF only, or binary.")
        self.cmap.setToolTip("Colormap for the image (the colorbar legend updates to match).")
        # --- dynamic range: scale (static/dynamic) + tone-map expression ---
        self.scale = QtWidgets.QComboBox(); self.scale.addItems(["dynamic", "static"])
        self.scale.setToolTip("White-point source. " + tonemap.SCALE_HELP["dynamic"] + " | "
                              + tonemap.SCALE_HELP["static"])
        self.scale.currentIndexChanged.connect(self._on_scale)
        self.expr = QtWidgets.QComboBox(); self.expr.addItems(tonemap.EXPRESSIONS)
        self.expr.setCurrentText("sqrt")
        self.expr.setToolTip("Map expression — compresses the dynamic range so bright regions "
                             "(rotor disks, glints) stop diluting faint structure. Hover options.")
        self.expr.currentIndexChanged.connect(self._on_expr)
        self.gamma = QtWidgets.QDoubleSpinBox(); self.gamma.setRange(0.1, 3.0)
        self.gamma.setSingleStep(0.05); self.gamma.setValue(0.5); self.gamma.setPrefix("γ ")
        self.gamma.setToolTip("Exponent for the 'gamma' expression (γ<1 lifts faint regions).")
        self.gamma.valueChanged.connect(self._render)
        self.freeze_btn = QtWidgets.QToolButton(); self.freeze_btn.setText("Freeze scale")
        self.freeze_btn.setToolTip("Snap the static white-point to the current frame "
                                   "(only meaningful in 'static' scale mode).")
        self.freeze_btn.clicked.connect(self._freeze_scale)
        self.roi_chk = QtWidgets.QCheckBox("ROI")
        self.roi_chk.setToolTip("Drag a box to analyze a region (mirrored into the workbench).")
        self.roi_chk.toggled.connect(self._toggle_roi)
        self.export_btn = QtWidgets.QToolButton(); self.export_btn.setText("Export")
        self.export_btn.setIcon(icons.icon("export"))
        self.export_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.export_btn.setToolTip("Save this frame as a journal figure, or export an event cube.")
        self.export_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self._build_export_menu()
        self.readout = QtWidgets.QLabel("—"); self.readout.setObjectName("muted")
        self.readout.setToolTip("Events in the current window · ON↑ / OFF↓ counts · mean event rate.")

        opts = QtWidgets.QHBoxLayout()
        for lbl, w in (("Mode", self.mode), ("Color", self.cmap), ("Scale", self.scale),
                       ("Expr", self.expr)):
            opts.addWidget(QtWidgets.QLabel(lbl)); opts.addWidget(w)
        opts.addWidget(self.gamma)
        opts.addWidget(self.freeze_btn)
        opts.addWidget(self.roi_chk)
        opts.addWidget(self.export_btn)
        opts.addStretch(1)
        opts.addWidget(self.readout)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.glw, 1)
        lay.addWidget(self.transport)
        lay.addLayout(opts)

        self._apply_cmap()
        self.ctl.cursorChanged.connect(self._render)
        self.ctl.accumChanged.connect(self._render)
        if self.filters is not None:
            self.filters.changed.connect(self._render)

    # ------------------------------------------------------------------ data
    def set_recording(self, rec):
        self.rec = rec
        self.vb.setRange(xRange=(0, rec.width), yRange=(0, rec.height), padding=0)
        self._render()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._render()

    # ------------------------------------------------------------------ export
    def _build_export_menu(self):
        m = QtWidgets.QMenu(self)
        m.addAction("Save current frame as figure…", self._save_figure)
        m.addAction("Export event cube (x, y, t)…", self._export_cube)
        m.addSeparator()
        act_tl = m.addAction("Timeline editor — stitch clips into one .raw…", self._open_timeline)
        act_tl.setIcon(icons.icon("film"))
        self.export_btn.setMenu(m)

    def _open_timeline(self):
        """Open the clip timeline editor, pre-loaded with the current recording (if any)."""
        from gottlux.app.timeline import TimelineEditorDialog
        TimelineEditorDialog(self, recordings=[self.rec] if self.rec is not None else None).exec()

    def _save_figure(self):
        if self.rec is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save frame figure", "event_frame.png", "PNG (*.png);;PDF (*.pdf)")
        if not path:
            return
        from gottlux.io import export
        from gottlux.viz import frames
        tc = self.ctl.cursor
        t0, t1 = self.ctl.accum_window()
        frame = accumulate_frame(self.rec.window(t0, t1), mode=self.mode.currentText())
        fig = frames.event_frame_figure(frame, mode=self.mode.currentText(),
                                        cmap=self.cmap.currentText(),
                                        title=f"{self.rec.name} @ {tc:.3f}s ({self.mode.currentText()})")
        base = os.path.splitext(path)[0]
        w = export.save_figure(fig, base, dpi=300, formats=("png", "pdf"), close=True)
        self._notify(w)

    def _export_cube(self):
        if self.rec is None:
            return
        base, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export event cube", "events", "Data block (*.npz *.h5)")
        if not base:
            return
        base = os.path.splitext(base)[0]
        from gottlux.app.exporting import save_event_cube
        t0, t1 = self.ctl.accum_window()
        mode = "polarity" if self.mode.currentText() == "polarity" else "count"
        w = save_event_cube(base, self.rec, t0, t1, nt=64, mode=mode)
        self._notify(w, f"window [{t0:.3f}, {t1:.3f}] s")

    def _notify(self, written, extra=""):
        if written:
            from gottlux.io.paths import open_in_file_browser
            open_in_file_browser(os.path.dirname(os.path.abspath(written[0])))   # reveal on export
            msg = "Saved:\n" + "\n".join(os.path.basename(p) for p in written)
            if extra:
                msg += f"\n\n{extra}"
            QtWidgets.QMessageBox.information(self, "Export", msg)
        else:
            QtWidgets.QMessageBox.warning(self, "Export", "Nothing was written.")

    # ------------------------------------------------------------------ mode
    def _on_mode(self, *_):
        """Switch accumulation mode; auto-pick a diverging colormap for the polarity views."""
        mode = self.mode.currentText()
        cur = self.cmap.currentText()
        if mode in _POLARITY_MODES and cur not in _DIVERGING:
            b = QtCore.QSignalBlocker(self.cmap)
            self.cmap.setCurrentText("coolwarm")
            del b
            self._apply_cmap()
            return
        if mode not in _POLARITY_MODES and cur in _DIVERGING:
            b = QtCore.QSignalBlocker(self.cmap)
            self.cmap.setCurrentText("inferno")
            del b
            self._apply_cmap()
            return
        self._render()

    # ------------------------------------------------------------------ colormap
    def _apply_cmap(self, *_):
        try:
            cm = pg.colormap.get(self.cmap.currentText(), source="matplotlib")
            self.img.setColorMap(cm)
            self.cbar.setColorMap(cm)
        except Exception:
            pass
        self._render()

    # ------------------------------------------------------------------ ROI
    def _toggle_roi(self, on):
        self.roi.setVisible(on)
        self._emit_roi() if on else self.roiChanged.emit(None)

    def _emit_roi(self):
        if not self.roi.isVisible() or self.rec is None:
            return
        pos = self.roi.pos(); size = self.roi.size()
        self.roiChanged.emit((int(max(0, pos.x())), int(max(0, pos.y())),
                              int(min(self.rec.width, pos.x() + size.x())),
                              int(min(self.rec.height, pos.y() + size.y()))))

    # ------------------------------------------------------------------ dynamic range
    def _on_scale(self, *_):
        # entering static mode freezes the current white-point as the reference
        if self.scale.currentText() == "static":
            self._static_vmax = None        # recompute once on the next render, then hold
        self._render()

    def _on_expr(self, *_):
        self.expr.setToolTip(tonemap.EXPR_HELP.get(self.expr.currentText(), ""))
        self._render()

    def _freeze_scale(self):
        self._static_vmax = None            # force a fresh capture on next render
        self.scale.setCurrentText("static")
        self._render()

    # ------------------------------------------------------------------ render
    def sync(self):
        """Force a render at the current cursor regardless of visibility (used by Sync views)."""
        self._render(force=True)

    def _render(self, *_, force=False):
        if self.rec is None or (not force and not self.isVisible()):
            return
        from gottlux.core.render import render_frame
        dt, t0 = self.ctl.accum, self.ctl.cursor
        mode, expr = self.mode.currentText(), self.expr.currentText()
        static = self.scale.currentText() == "static"
        disp, levels, vmax, win = render_frame(
            self.rec, t0, dt, mode=mode, expr=expr, gamma=float(self.gamma.value()),
            filters=self.filters, vmax_ref=(self._static_vmax if static else None),
            back=self.ctl.accum_back)
        if static and vmax is not None:
            self._static_vmax = vmax
        self.img.setImage(disp, levels=levels, autoLevels=False)
        if mode in ("time_surface", "binary"):
            clabel = _CLABEL.get(mode, "")
        elif mode == "polarity_ratio":
            clabel = _CLABEL["polarity_ratio"]
        elif mode == "polarity":
            clabel = f"ON−OFF · {expr} (±{vmax:.0f})"
        else:
            clabel = f"{_CLABEL.get(mode, '')} · {expr} (max {vmax:.0f})"
        try:
            self.cbar.setLevels(levels)
            self.cbar.setLabel(clabel)
        except Exception:
            pass
        on, off = win.polarity_split()
        rate = win.n / dt if dt > 0 else 0
        self.readout.setText(f"{win.n:,} ev · {on.sum():,}↑ {off.sum():,}↓ · {rate/1e6:.2f} Mev/s")

    # ------------------------------------------------------------------ faithful capture
    def sensor_size(self):
        """Native sensor (width, height) in px, or ``None`` if no recording."""
        return (self.rec.width, self.rec.height) if self.rec is not None else None

    def capture_frame(self, t, dt=None, size=None):
        """A faithful RGB of the configured view at time *t* — the **exact** tuned mode / colormap /
        expression / gamma / scale / live filters, rendered from the events at native sensor
        resolution and resized to *size* ``(w, h)``. This is what makes Capture a high-res,
        settings-accurate video instead of a screen grab.
        """
        if self.rec is None:
            return None
        from gottlux.core.render import render_frame
        from gottlux.viz import video as _vid
        dt = self.ctl.accum if dt is None else float(dt)
        static = self.scale.currentText() == "static"
        disp, levels, _v, _w = render_frame(
            self.rec, t, dt, mode=self.mode.currentText(), expr=self.expr.currentText(),
            gamma=float(self.gamma.value()), filters=self.filters,
            vmax_ref=(self._static_vmax if static else None), back=self.ctl.accum_back)
        rgb = _vid.disp_to_rgb(disp, levels, self.cmap.currentText())
        return _vid.resize_rgb(rgb, size) if size else rgb
