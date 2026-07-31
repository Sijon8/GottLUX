"""
tower.py — the live "event-rate tower" map: a 3-D relief of activity over the sensor.

Where the live viewer shows event density as *colour* on a flat image, this tab shows it as
*height*: the sensor's x/y pixels lie on the ground plane (labelled with the sensor
dimensions) and each cell rises to its **event rate (events/second)**. A buzzing rotor or
wingbeat region becomes a sharp tower standing out of the noise floor.

It is seekable and live on the shared clock, offers two **styles** (a smooth **surface** or
discrete **bars/bins** — one cubic column per cell, with adjustable grid resolution), the same
static/dynamic white-point and map-expression dynamic-range controls as the live viewer, a
selectable **background theme** and **brightness**, a colormap and height exaggeration, and a
CAD-style **orientation cube** in the corner. Degrades to a message without OpenGL.
"""
from __future__ import annotations

import os
import time

import numpy as np
from PySide6 import QtCore, QtWidgets

from gottlux.app import icons
from gottlux.app import style
from gottlux.app.legend import ColorKey
from gottlux.app.navcube import navcube_container
from gottlux.app.transport import TransportBar
from gottlux.core import tonemap
from gottlux.core.accumulate import count_image

try:
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import MeshData
    _HAVE_GL = True
except Exception:                                  # pragma: no cover
    _HAVE_GL = False

_CMAPS = ["inferno", "viridis", "magma", "plasma", "turbo", "cividis"]
_STYLES = ["bars", "surface"]
# Background canvas presets. The neutrals stay near-black; the *coloured* presets are lifted
# enough to actually read as their hue (the earlier values were so dark every choice looked like
# the same charcoal), so the dropdown changes the environment, not just a faint tone. The
# default entry has no fixed colour — it follows the app's light/dark instrument theme.
_APP_THEME = "App theme"
_BG = {"Charcoal": "#0e1116", "Black": "#000000", "Midnight": "#05070d",
       "Graphite": "#15171a", "Slate": "#26313f", "Steel": "#2c3e4d",
       "Navy": "#15285e", "Cobalt": "#11407e", "Deep teal": "#0d5057",
       "Forest": "#16401f", "Indigo": "#281c63", "Plum": "#3b1d4d",
       "Oxblood": "#4a1620", "Sepia": "#3a2b12", "White": "#eef1f5"}


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _relative_luminance(h):
    r, g, b = _hex_to_rgb(h)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def _scene_furniture(hexc):
    """Grid + axis-label RGBA that read clearly against a canvas of the given colour.

    A light canvas gets dark grid lines / labels; a dark canvas gets light ones — so changing
    the background recolours the whole environment instead of leaving fixed-colour furniture."""
    if _relative_luminance(hexc) > 0.55:               # light canvas → dark furniture
        return (40, 48, 58, 160), (28, 36, 46, 255)
    return (150, 170, 190, 90), (180, 190, 205, 255)   # dark canvas → light furniture


def _bg_hex(name):
    """The canvas colour a background preset means — the app theme's own for 'App theme'."""
    return style.BG if name == _APP_THEME else _BG.get(name, style.BG)


# Unit cube corners / faces (for building the bar mesh; matches GLBarGraphItem's layout).
_CUBE = np.mgrid[0:2, 0:2, 0:2].reshape(3, 8).T.astype(np.float32)
_CUBE_FACES = np.array([[0, 1, 2], [3, 2, 1], [4, 5, 6], [7, 6, 5], [0, 1, 4], [5, 4, 1],
                        [2, 3, 6], [7, 6, 3], [0, 2, 4], [6, 4, 2], [1, 3, 5], [7, 5, 3]])


def _downsample_sum(frame, cell):
    """Sum a (H, W) frame into (Gh, Gw) cells of ``cell×cell`` (zero-padded at the edges)."""
    H, W = frame.shape
    Gh = (H + cell - 1) // cell
    Gw = (W + cell - 1) // cell
    pad = np.zeros((Gh * cell, Gw * cell), frame.dtype)
    pad[:H, :W] = frame
    return pad.reshape(Gh, cell, Gw, cell).sum(axis=(1, 3))


class EventRateTower(QtWidgets.QWidget):
    """Interactive, live, seekable 3-D event-rate (events/s) relief over the sensor."""

    def __init__(self, controller, filters=None, parent=None):
        super().__init__(parent)
        self.ctl = controller
        self.filters = filters        # shared live noise-filter suite (FilterController | None)
        self.rec = None
        self._static_vmax = None
        self._last_render = 0.0
        lay = QtWidgets.QVBoxLayout(self)

        if not _HAVE_GL:
            lay.addWidget(QtWidgets.QLabel(
                "The event-rate tower needs PyOpenGL (pip install PyOpenGL). "
                "The rest of gottlux works without it."))
            self.view = None
            return

        from gottlux.app.glview import GLView
        self.view = GLView() if GLView is not None else gl.GLViewWidget()
        self.view.setBackgroundColor(style.BG)
        self.view.opts["distance"] = 520
        self.surf = gl.GLSurfacePlotItem(shader="shaded", computeNormals=False, smooth=True)
        self.view.addItem(self.surf)
        self.bars = gl.GLMeshItem(smooth=False, shader="shaded", glOptions="opaque")
        self.bars.setVisible(False)
        self.view.addItem(self.bars)
        self.grid = gl.GLGridItem(); self.grid.setSize(320, 320); self.grid.setSpacing(40, 40)
        self.view.addItem(self.grid)
        self._axis_labels = []
        self._make_axis_labels()
        holder, self.navcube = navcube_container(self.view)
        lay.addWidget(holder, 1)

        self.transport = TransportBar(self.ctl, host=self)
        lay.addWidget(self.transport)
        lay.addLayout(self._build_controls())

        self.ctl.cursorChanged.connect(self._render_throttled)
        self.ctl.accumChanged.connect(self._render)
        if self.filters is not None:
            self.filters.changed.connect(self._render)
        self.key.set_gradient(self.cmap.currentText(), "low", "high", title="event rate")
        self._on_bg(self.bg.currentText())            # match grid/label furniture to the default bg

    # ------------------------------------------------------------------ controls
    def _build_controls(self):
        self.cell = QtWidgets.QSpinBox(); self.cell.setRange(1, 16); self.cell.setValue(4)
        self.cell.setSuffix(" px")
        self.cell.setToolTip("Grid resolution: each cell is one tower/bar. Smaller = finer "
                             "relief, slower (use a larger cell for the bars style).")
        self.cell.valueChanged.connect(self._render)
        self.style = QtWidgets.QComboBox(); self.style.addItems(_STYLES)
        self.style.setToolTip("Surface (smooth relief) or bars (a cubic bin per cell, height = "
                              "events).")
        self.style.currentIndexChanged.connect(self._on_style)
        self.cmap = QtWidgets.QComboBox(); self.cmap.addItems(_CMAPS)
        self.cmap.setToolTip("Colormap applied by height.")
        self.cmap.currentIndexChanged.connect(self._render)
        self.scale = QtWidgets.QComboBox(); self.scale.addItems(["dynamic", "static"])
        self.scale.setToolTip("Height reference. " + tonemap.SCALE_HELP["dynamic"] + " | "
                              + tonemap.SCALE_HELP["static"])
        self.scale.currentIndexChanged.connect(self._on_scale)
        self.expr = QtWidgets.QComboBox(); self.expr.addItems(tonemap.EXPRESSIONS)
        self.expr.setCurrentText("sqrt")
        self.expr.setToolTip("Height expression — compress the rate so one hot cell does not "
                             "dwarf everything (same options as the live viewer).")
        self.expr.currentIndexChanged.connect(self._on_expr)
        self.zscale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.zscale.setRange(10, 400); self.zscale.setValue(120); self.zscale.setFixedWidth(90)
        self.zscale.setToolTip("Tower height exaggeration.")
        self.zscale.valueChanged.connect(self._render)
        self.bg = QtWidgets.QComboBox(); self.bg.addItems([_APP_THEME] + list(_BG))
        self.bg.setToolTip("Background canvas theme. 'App theme' follows the window's "
                           "light/dark theme.")
        self.bg.currentTextChanged.connect(self._on_bg)
        # a light/dark switch moves the 'App theme' canvas with it
        style.notifier().themeChanged.connect(self.apply_theme)
        self.bright = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bright.setRange(20, 160); self.bright.setValue(160); self.bright.setFixedWidth(80)
        self.bright.setToolTip("Scene brightness (scales the rendered colours).")
        self.bright.valueChanged.connect(self._render)
        self.key = ColorKey("event rate")
        self.export_btn = QtWidgets.QToolButton(); self.export_btn.setText("Export")
        self.export_btn.setIcon(icons.icon("export"))
        self.export_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.export_btn.setToolTip("Save a 3-D snapshot, or export the event cube.")
        self.export_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self._build_export_menu()

        # two compact rows so everything is visible without scrolling
        outer = QtWidgets.QVBoxLayout()
        r1 = QtWidgets.QHBoxLayout()
        for lbl, w in (("Cell", self.cell), ("Style", self.style), ("Color", self.cmap),
                       ("Scale", self.scale), ("Expr", self.expr)):
            r1.addWidget(QtWidgets.QLabel(lbl)); r1.addWidget(w)
        r1.addWidget(QtWidgets.QLabel("Height")); r1.addWidget(self.zscale)
        r1.addStretch(1)
        r2 = QtWidgets.QHBoxLayout()
        r2.addWidget(QtWidgets.QLabel("Background")); r2.addWidget(self.bg)
        r2.addWidget(QtWidgets.QLabel("Brightness")); r2.addWidget(self.bright)
        r2.addWidget(self.export_btn)
        r2.addSpacing(8); r2.addWidget(self.key, 1)
        outer.addLayout(r1); outer.addLayout(r2)
        return outer

    def _make_axis_labels(self):
        """GLTextItems labelling the base plane with the sensor pixel extents."""
        for _ in range(5):
            t = gl.GLTextItem(text="", color=(180, 190, 205, 255))
            self.view.addItem(t)
            self._axis_labels.append(t)

    def _update_axis_labels(self):
        if self.rec is None:
            return
        W, H = self.rec.width, self.rec.height
        hw, hh = W / 2.0, H / 2.0
        specs = [((-hw, -hh, 0), "(0,0)"),
                 ((hw, -hh, 0), f"x={W}"),
                 ((-hw, hh, 0), f"y={H}"),
                 ((0, -hh - 14, 0), f"x  ({W} px)"),
                 ((-hw - 24, 0, 0), f"y  ({H} px)")]
        for t, (pos, text) in zip(self._axis_labels, specs):
            t.setData(pos=np.array(pos, np.float32), text=text)

    # ------------------------------------------------------------------ data
    def set_recording(self, rec):
        self.rec = rec
        if self.view is None:
            return
        self.grid.setSize(rec.width, rec.height)
        self._update_axis_labels()
        self._static_vmax = None
        self._render()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._render()

    def sync(self):
        self._render(force=True)

    # ------------------------------------------------------------------ faithful capture
    def sensor_size(self):
        # Even width/height (and ≥16): the faithful-capture path derives the export resolution from
        # this, and the H.264 encoder rejects odd dimensions.
        s = self.view.size()
        w = max(s.width() - s.width() % 2, 16)
        h = max(s.height() - s.height() % 2, 16)
        return (w, h)

    def capture_frame(self, t, dt=None, size=None):
        """Offscreen-render the 3-D event-rate relief at time *t* to RGB at *size*."""
        from gottlux.app.capture import gl_to_rgb
        self.ctl.set_cursor(float(t))
        QtWidgets.QApplication.processEvents()
        w, h = size if size else self.sensor_size()
        try:
            return gl_to_rgb(self.view.renderToArray((int(w), int(h))))
        except Exception:
            return None

    def _on_style(self, *_):
        self._render()

    def _on_scale(self, *_):
        if self.scale.currentText() == "static":
            self._static_vmax = None
        self._render()

    def _on_expr(self, *_):
        self.expr.setToolTip(tonemap.EXPR_HELP.get(self.expr.currentText(), ""))
        self._render()

    def apply_theme(self, *_):
        """Re-resolve the background preset after an app light/dark switch (the
        ``themeChanged`` slot): only 'App theme' moves, a chosen colour stays put."""
        self._on_bg(self.bg.currentText())

    def _on_bg(self, name):
        if self.view is None:
            return
        hexc = _bg_hex(name)
        self.view.setBackgroundColor(hexc)
        # recolour the grid + axis labels so the whole environment re-reads against the new canvas
        grid_rgba, label_rgba = _scene_furniture(hexc)
        try:
            self.grid.setColor(grid_rgba)
        except Exception:
            pass
        for t in self._axis_labels:
            try:
                t.setData(color=label_rgba)
            except Exception:
                pass
        self.view.update()

    # ------------------------------------------------------------------ export
    def _build_export_menu(self):
        m = QtWidgets.QMenu(self)
        m.addAction("Save 3-D snapshot (PNG)…", self._save_snapshot)
        m.addAction("Export event cube (x, y, t)…", self._export_cube)
        self.export_btn.setMenu(m)

    def _save_snapshot(self):
        if self.view is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save tower snapshot",
                                                        "event_tower.png", "PNG image (*.png)")
        if not path:
            return
        from gottlux.app.exporting import save_gl_snapshot
        self._notify(save_gl_snapshot(self.view, path))

    def _export_cube(self):
        if self.rec is None:
            return
        base, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export event cube", "events",
                                                        "Data block (*.npz *.h5)")
        if not base:
            return
        from gottlux.app.exporting import save_event_cube
        t0, t1 = self.ctl.accum_window()
        self._notify(save_event_cube(os.path.splitext(base)[0], self.rec, t0, t1, nt=64),
                     f"window [{t0:.3f}, {t1:.3f}] s")

    def _notify(self, written, extra=""):
        if written:
            from gottlux.io.paths import open_in_file_browser
            open_in_file_browser(os.path.dirname(os.path.abspath(written[0])))   # reveal on export
            msg = "Saved:\n" + "\n".join(os.path.basename(p) for p in written)
            QtWidgets.QMessageBox.information(self, "Export", msg + (f"\n\n{extra}" if extra else ""))
        else:
            QtWidgets.QMessageBox.warning(self, "Export", "Nothing was written.")

    # ------------------------------------------------------------------ render
    def _render_throttled(self, *_):
        if time.perf_counter() - self._last_render >= 0.05:
            self._render()

    def _render(self, *_, force=False):
        if self.rec is None or self.view is None or (not force and not self.isVisible()):
            return
        self._last_render = time.perf_counter()
        dt = max(self.ctl.accum, 1e-6)
        t0, t1 = self.ctl.accum_window()                     # honours the accumulation direction
        cell = self.cell.value()
        win = self.rec.window(t0, t1)
        if self.filters is not None:
            win = self.filters.apply(win)
        counts = count_image(win).astype(np.float64)
        cell_counts = _downsample_sum(counts, cell)
        rate = cell_counts / dt                              # events / second per cell
        static = self.scale.currentText() == "static"
        ref = self._static_vmax if static else None
        disp, vmax = tonemap.compress(rate, expr=self.expr.currentText(), vmax=ref, gamma=0.5)
        if static:
            self._static_vmax = vmax
        zh = self.zscale.value()
        bright = self.bright.value() / 100.0
        cmap = tonemap.colormap(self.cmap.currentText())
        if self.style.currentText() == "bars":
            self._render_bars(disp, cell, zh, cmap, bright)
        else:
            self._render_surface(disp, cell, zh, cmap, bright)
        self.key.set_gradient(self.cmap.currentText(), "0",
                              f"{vmax/1e3:.0f}k ev/s" if vmax >= 1e3 else f"{vmax:.0f} ev/s",
                              title="event rate")

    def _render_surface(self, disp, cell, zh, cmap, bright):
        self.bars.setVisible(False); self.surf.setVisible(True)
        Gh, Gw = disp.shape
        z = (disp.T.astype(np.float32)) * zh
        xs = (np.arange(Gw) * cell - self.rec.width / 2).astype(np.float32)
        ys = (np.arange(Gh) * cell - self.rec.height / 2).astype(np.float32)
        colors = cmap(disp.T).astype(np.float32)
        colors[..., :3] = np.clip(colors[..., :3] * bright, 0, 1)
        self.surf.setData(x=xs, y=ys, z=z, colors=colors.reshape(Gw, Gh, 4))

    def _render_bars(self, disp, cell, zh, cmap, bright):
        self.surf.setVisible(False); self.bars.setVisible(True)
        Gh, Gw = disp.shape
        gap = max(1.0, cell * 0.12)
        bw = cell - gap
        ys, xs = np.meshgrid(np.arange(Gh), np.arange(Gw), indexing="ij")
        xs = xs.ravel(); ys = ys.ravel()
        d = disp.ravel()
        keep = d > 1e-4                                      # skip empty cells (huge speedup)
        xs, ys, d = xs[keep], ys[keep], d[keep]
        n = xs.size
        if n == 0:
            self.bars.setMeshData(meshdata=MeshData(vertexes=np.zeros((0, 3)), faces=np.zeros((0, 3), int)))
            return
        base = np.stack([xs * cell - self.rec.width / 2, ys * cell - self.rec.height / 2,
                         np.zeros(n)], axis=1).astype(np.float32)
        size = np.stack([np.full(n, bw), np.full(n, bw), d * zh], axis=1).astype(np.float32)
        verts = (_CUBE[None, :, :] * size[:, None, :] + base[:, None, :]).reshape(-1, 3)
        faces = (_CUBE_FACES[None, :, :] + (np.arange(n) * 8)[:, None, None]).reshape(-1, 3)
        cols = cmap(d).astype(np.float32)
        cols[:, :3] = np.clip(cols[:, :3] * bright, 0, 1)
        vcols = np.repeat(cols, 8, axis=0)
        self.bars.setMeshData(meshdata=MeshData(vertexes=verts, faces=faces, vertexColors=vcols))
