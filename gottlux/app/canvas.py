"""
canvas.py — the Canvas composer: place several recordings on one canvas and play them.

A self-contained :class:`QMainWindow` over the engine in :mod:`gottlux.core.canvas`. Each
loaded clip becomes a draggable, resizable cell on a fixed-size canvas; a settings deck
edits the *selected* cell's placement, source ROI, clock mapping (offset / time scale /
loop), and look (accumulation mode + window, tone-map, colormap) — applied live. A shared
transport plays and scrubs the canvas timeline (the longest clip extent). Compositions
save/load as ``.gottlux-canvas.json`` and export as an MP4 (rendered frames) or as a
single composited EVT2.1 ``.raw`` (re-encoded events; the spec rides along as a sidecar).

Recordings arrive through the 'Add clips…' dialog **or by OS drag-and-drop** — dropped
``.raw``/``.h5``/``.hdf5`` files and capture folders become new cells through the same
add-clip path. 'Add title…' (:class:`TitleDialog`) attaches the engine's
:class:`~gottlux.core.canvas.CanvasText` items — title slides and running overlays,
listed after the cells with a ``T`` prefix; text renders in **video** export only, and
the ``.raw`` export's completion dialog notes the omission.

The cell stage itself is :class:`CanvasArrangeView` — a standalone widget holding the
scene, the movable/resizable cells, and the live per-cell rendering. The composer window
embeds one; so does the **Timeline** tab's inline arrange mode
(:mod:`gottlux.app.timeline`), which is why editing a mosaic never requires this pop-out
window while this window keeps working exactly as before.

The window is constructible standalone (offscreen tests build it directly);
``add_clip_recording`` accepts in-memory recordings so no file dialog is required.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from gottlux.app import icons
from gottlux.app import style
from gottlux.app.transport import TimeController, TransportBar
from gottlux.app.uikit import plot_with_deck, with_progress
from gottlux.core import canvas as engine
from gottlux.core import tonemap

_MODES = ["count", "time_surface", "polarity", "polarity_ratio", "on", "off", "binary"]
_CMAPS = ["inferno", "viridis", "magma", "plasma", "cividis", "gray", "turbo",
          "coolwarm", "RdBu", "bwr", "seismic", "PiYG", "Spectral"]
_GRIP = 12          # resize-grip square, in canvas pixels


def to_pixmap(rgb) -> QtGui.QPixmap:
    """An ``(H, W, 3)`` uint8 array as a QPixmap (copied, so the array may be freed).

    Shared by every widget that shows an engine-rendered frame (the cells here, the
    Timeline's preview viewport and its lane thumbnails)."""
    rgb = np.ascontiguousarray(rgb, np.uint8)
    h, w = rgb.shape[:2]
    img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(img.copy())


#: File suffixes accepted from an OS drag (folders always qualify) — the gottlux.load set.
_DROP_SUFFIXES = (".raw", ".h5", ".hdf5")


def droppable_paths(md) -> list:
    """The local recording paths in a drag payload — ``.raw``/``.h5``/``.hdf5`` files or
    folders (anything :func:`gottlux.load` accepts) — in name order; empty when the drag
    carries none, so the caller refuses the drop. Shared by the composer window and the
    timeline editor."""
    paths = []
    if md is None or not md.hasUrls():
        return paths
    for url in md.urls():
        p = url.toLocalFile()
        if p and os.path.exists(p) and (os.path.isdir(p)
                                        or p.lower().endswith(_DROP_SUFFIXES)):
            paths.append(p)
    return sorted(paths, key=lambda p: os.path.basename(p).lower())


class TitleDialog(QtWidgets.QDialog):
    """Create or edit one text item: a full-frame title *slide* or a *running* overlay.

    The state travels as a plain ``values()`` dict — the timeline editor stores it on
    its title items, and :func:`text_item_from_values` turns it into the engine's
    :class:`~gottlux.core.canvas.CanvasText`. The colour defaults are taken from the
    active instrument theme (both are freely editable per title). Text renders in
    **video** exports only; event (``.raw``/``.h5``) exports skip it and surface the
    one-line omission note instead.
    """

    KINDS = ("slide", "overlay")
    _ANCHORS = ("s", "n", "center", "w", "e")

    def __init__(self, parent=None, values=None):
        super().__init__(parent)
        self.setWindowTitle("Title — text over the composition")
        self.setMinimumWidth(400)
        self._t0_s = 0.0                       # preserved span start (edits keep it)
        self._margin_px = 24                   # preserved overlay margin, likewise
        self._color = QtGui.QColor(style.FG)
        self._bg = QtGui.QColor(style.BG)

        self.text_edit = QtWidgets.QPlainTextEdit("Title")
        self.text_edit.setFixedHeight(76)
        self.text_edit.setToolTip("The text — line breaks are kept (multi-line titles "
                                  "render centered).")
        self.kind_cb = QtWidgets.QComboBox()
        self.kind_cb.addItems(["Title slide (fills the frame)",
                               "Running title (overlaid on the video)"])
        self.kind_cb.setToolTip(
            "A title slide occupies its duration like a clip and fills the frame; a "
            "running title overlays the video at its anchor for its span.")
        self.kind_cb.currentIndexChanged.connect(self._on_kind)
        self.dur_sp = QtWidgets.QDoubleSpinBox()
        self.dur_sp.setRange(0.1, 3600.0); self.dur_sp.setDecimals(2)
        self.dur_sp.setValue(3.0); self.dur_sp.setSuffix(" s")
        self.whole_chk = QtWidgets.QCheckBox("Whole video")
        self.whole_chk.setToolTip("Run the title over the entire video rather than a "
                                  "fixed span.")
        self.whole_chk.toggled.connect(self._sync_enabled)
        self.font_sp = QtWidgets.QSpinBox()
        self.font_sp.setRange(8, 256); self.font_sp.setValue(32)
        self.font_sp.setSuffix(" px")
        self.anchor_cb = QtWidgets.QComboBox()
        self.anchor_cb.addItems(self._ANCHORS)
        self.anchor_cb.setToolTip("Overlay placement: a compass edge (n/s/e/w) or center.")
        self.color_btn = QtWidgets.QPushButton("Text color…")
        self.color_btn.clicked.connect(lambda: self._pick("_color"))
        self.bg_btn = QtWidgets.QPushButton("Background…")
        self.bg_btn.setToolTip("The slide's fill / the overlay's translucent backing bar.")
        self.bg_btn.clicked.connect(lambda: self._pick("_bg"))

        form = QtWidgets.QFormLayout(self)
        form.addRow("Text", self.text_edit)
        form.addRow("Kind", self.kind_cb)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.dur_sp); row.addWidget(self.whole_chk); row.addStretch(1)
        form.addRow("Duration", row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.font_sp); row.addWidget(QtWidgets.QLabel("Anchor"))
        row.addWidget(self.anchor_cb); row.addStretch(1)
        form.addRow("Font", row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.color_btn); row.addWidget(self.bg_btn); row.addStretch(1)
        form.addRow("Colors", row)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                        QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        form.addRow(bb)

        if values:
            self.set_values(values)
        self._sync_enabled()
        self._swatches()

    # ----- state in / out -----
    def set_values(self, vals: dict):
        """Prefill every field from a ``values()``-shaped dict (the edit path)."""
        self.text_edit.setPlainText(str(vals.get("text", "")))
        b = QtCore.QSignalBlocker(self.kind_cb)
        self.kind_cb.setCurrentIndex(0 if vals.get("kind") == "slide" else 1)
        del b
        self.dur_sp.setValue(float(vals.get("duration_s", 3.0)))
        b = QtCore.QSignalBlocker(self.whole_chk)
        self.whole_chk.setChecked(bool(vals.get("whole", False)))
        del b
        self.font_sp.setValue(int(vals.get("font_size_px", 32)))
        self.anchor_cb.setCurrentText(str(vals.get("anchor", "s")))
        if vals.get("color") is not None:
            self._color = QtGui.QColor(*(int(v) for v in vals["color"]))
        if vals.get("bg_color") is not None:
            self._bg = QtGui.QColor(*(int(v) for v in vals["bg_color"]))
        self._t0_s = float(vals.get("t0_s", 0.0))
        self._margin_px = int(vals.get("margin_px", 24))
        self._sync_enabled()
        self._swatches()

    def values(self) -> dict:
        """The dialog's state as a plain dict (see :func:`text_item_from_values`)."""
        kind = self.KINDS[self.kind_cb.currentIndex()]
        return {"text": self.text_edit.toPlainText().rstrip() or "Title",
                "kind": kind,
                "duration_s": float(self.dur_sp.value()),
                "whole": bool(kind == "overlay" and self.whole_chk.isChecked()),
                "anchor": self.anchor_cb.currentText(),
                "font_size_px": int(self.font_sp.value()),
                "color": (self._color.red(), self._color.green(), self._color.blue()),
                "bg_color": (self._bg.red(), self._bg.green(), self._bg.blue()),
                "t0_s": float(self._t0_s), "margin_px": int(self._margin_px)}

    # ----- wiring -----
    def _on_kind(self, *_):
        b = QtCore.QSignalBlocker(self.whole_chk)
        self.whole_chk.setChecked(self.kind_cb.currentIndex() == 1)   # running → whole
        del b
        self._sync_enabled()

    def _sync_enabled(self, *_):
        slide = self.kind_cb.currentIndex() == 0
        self.whole_chk.setEnabled(not slide)
        self.anchor_cb.setEnabled(not slide)
        self.dur_sp.setEnabled(slide or not self.whole_chk.isChecked())

    def _pick(self, attr):
        col = QtWidgets.QColorDialog.getColor(getattr(self, attr), self, "Title color")
        if col.isValid():
            setattr(self, attr, col)
            self._swatches()

    def _swatches(self):
        for btn, col in ((self.color_btn, self._color), (self.bg_btn, self._bg)):
            btn.setStyleSheet(f"QPushButton {{ border-left: 12px solid {col.name()}; }}")


def text_item_from_values(vals: dict) -> engine.CanvasText:
    """The engine text item a :class:`TitleDialog` ``values()`` dict describes.

    A slide always spans ``[t0, t0+duration)``; a running title spans the whole video
    (``span=None``) unless a fixed span was chosen."""
    t0 = float(vals.get("t0_s", 0.0))
    whole = vals.get("kind") == "overlay" and bool(vals.get("whole", True))
    bg = vals.get("bg_color")
    return engine.CanvasText(
        text=str(vals.get("text", "")), kind=str(vals.get("kind", "overlay")),
        span=None if whole else (t0, t0 + float(vals.get("duration_s", 3.0))),
        anchor=str(vals.get("anchor", "s")),
        margin_px=int(vals.get("margin_px", 24)),
        font_size_px=int(vals.get("font_size_px", 32)),
        color=tuple(int(v) for v in vals.get("color", (215, 221, 231))),
        bg_color=None if bg is None else tuple(int(v) for v in bg))


def values_from_text_item(txt: engine.CanvasText) -> dict:
    """A :class:`TitleDialog` ``values()`` dict prefilled from an engine text item (the
    double-click edit path; the span start is preserved through the round-trip)."""
    t0, dur = (0.0, 3.0) if txt.span is None else \
        (float(txt.span[0]), float(txt.span[1]) - float(txt.span[0]))
    return {"text": txt.text, "kind": txt.kind, "duration_s": dur,
            "whole": txt.span is None, "anchor": txt.anchor,
            "font_size_px": int(txt.font_size_px),
            "color": tuple(int(v) for v in txt.color),
            "bg_color": None if txt.bg_color is None
            else tuple(int(v) for v in txt.bg_color),
            "t0_s": t0, "margin_px": int(txt.margin_px)}


class _CellItem(QtWidgets.QGraphicsItem):
    """One clip's cell on the canvas scene: movable by drag, resizable by its corner grip.

    Item coordinates are canvas pixels; the item's scene position is the cell's ``(x, y)``
    and its local size the cell's ``(w, h)``. Geometry edits call back into the window
    (``on_geometry(index)``), which writes them into the spec and re-renders the cell.
    """

    def __init__(self, index: int, name: str, w: int, h: int, on_geometry):
        super().__init__()
        self.index = index
        self.name = name
        self.cell_w = max(int(w), 4)
        self.cell_h = max(int(h), 4)
        self._on_geometry = on_geometry
        self._pix = None
        self._resizing = False
        self.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable |
                      QtWidgets.QGraphicsItem.ItemIsSelectable |
                      QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self.setToolTip(f"{name} — drag to move; drag the bottom-right corner to resize.")

    # ----- geometry -----
    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, 0, self.cell_w, self.cell_h)

    def set_cell_size(self, w: int, h: int):
        w, h = max(int(w), 4), max(int(h), 4)
        if (w, h) != (self.cell_w, self.cell_h):
            self.prepareGeometryChange()
            self.cell_w, self.cell_h = w, h
            self.update()

    def set_image(self, rgb):
        """Set the rendered cell content (``None`` shows the dark 'outside extent' fill)."""
        self._pix = None if rgb is None else to_pixmap(rgb)
        self.update()

    # ----- painting -----
    def paint(self, p: QtGui.QPainter, _opt, _widget=None):
        r = QtCore.QRectF(0, 0, self.cell_w, self.cell_h)
        p.fillRect(r, QtGui.QColor(style.BG2))
        if self._pix is not None:
            p.drawPixmap(r.toRect(), self._pix)
        chrome = QtGui.QColor(style.ACCENT if self.isSelected() else style.BORDER)
        pen = QtGui.QPen(chrome, 1)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.drawRect(r)
        grip = QtCore.QRectF(self.cell_w - _GRIP, self.cell_h - _GRIP, _GRIP, _GRIP)
        p.fillRect(grip, chrome)
        # The cell's name is chrome sitting on *content* (a rendered event frame), so it
        # gets its own panel-coloured plate — theme foreground straight onto the frame
        # would be unreadable on whichever of the two themes fights the imagery.
        f = p.font(); f.setPointSizeF(8.0); p.setFont(f)
        plate = QtCore.QRectF(2, 2, p.fontMetrics().horizontalAdvance(self.name) + 8, 13)
        bg = QtGui.QColor(style.PANEL); bg.setAlpha(220)
        p.fillRect(plate, bg)
        p.setPen(QtGui.QPen(QtGui.QColor(style.FG)))
        p.drawText(plate.adjusted(4, 0, 0, 0), QtCore.Qt.AlignVCenter, self.name)

    # ----- interaction -----
    def _in_grip(self, pos: QtCore.QPointF) -> bool:
        return pos.x() >= self.cell_w - _GRIP and pos.y() >= self.cell_h - _GRIP

    def mousePressEvent(self, ev):
        if self._in_grip(ev.pos()):
            self._resizing = True
            self.setSelected(True)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._resizing:
            self.set_cell_size(int(round(ev.pos().x())), int(round(ev.pos().y())))
            self._on_geometry(self.index)
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._resizing:
            self._resizing = False
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            return QtCore.QPointF(round(value.x()), round(value.y()))   # snap to whole px
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self._on_geometry(self.index)
        return super().itemChange(change, value)


class CanvasArrangeView(QtWidgets.QGraphicsView):
    """The draggable/resizable cell stage over a :class:`~gottlux.core.canvas.CanvasSpec`.

    The composition's *spatial* editor, and nothing else: it owns the scene, the canvas
    frame, and one movable/resizable :class:`_CellItem` per cell; it writes every drag /
    resize straight back into the spec (optionally snapped to a fraction grid) and renders
    each cell live through the engine. Hosts bind to ``cellSelected`` and
    ``cellGeometryChanged`` and drive the clock with :meth:`render_at`, so the same widget
    serves the pop-out :class:`CanvasComposerWindow` *and* the Timeline tab's inline
    arrange mode — one implementation of the cell machinery, two places to use it.

    Drops are deliberately refused here: the hosting window handles them, so a file
    dragged onto the stage lands as a new cell rather than being eaten by the scene.
    """

    cellSelected = QtCore.Signal(int)          # the selected cell's index, or -1 for none
    cellGeometryChanged = QtCore.Signal(int)   # cell i was dragged or resized

    def __init__(self, spec=None, recs=None, parent=None, snap_divisions=0, margin=16):
        super().__init__(parent)
        self.spec = spec if spec is not None else engine.CanvasSpec(width=640, height=480)
        self.recs: dict = recs if recs is not None else {}
        self.cells: list = []              # _CellItem per clip (parallel to spec.clips)
        self.snap_divisions = int(snap_divisions)
        self.margin = int(margin)
        self._t = 0.0                      # the canvas time the cells were last rendered at

        self.gscene = QtWidgets.QGraphicsScene(self)
        self.gscene.setBackgroundBrush(QtGui.QColor(style.BG))
        # The frame is the *video* canvas, so it stays black in either theme — that is what
        # an exported MP4 shows behind the cells. Only the surround is chrome.
        self.frame_item = self.gscene.addRect(
            0, 0, self.spec.width, self.spec.height,
            QtGui.QPen(QtGui.QColor(style.BORDER)), QtGui.QBrush(QtGui.QColor(0, 0, 0)))
        self.frame_item.setZValue(-10)
        self.setScene(self.gscene)
        self.setRenderHint(QtGui.QPainter.Antialiasing, False)
        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self.gscene.selectionChanged.connect(self._on_scene_selection)
        self.setAcceptDrops(False)
        self.viewport().setAcceptDrops(False)
        # follow a live light/dark switch wherever this stage is embedded (the composer
        # window, the Timeline tab's inline arrange mode) without the host wiring it up
        style.notifier().themeChanged.connect(self.apply_theme)
        self.sync_canvas()

    # ----- composition in -----
    def set_composition(self, spec, recs):
        """Adopt *spec* and *recs* **by reference** (edits land in the caller's objects)."""
        self.spec, self.recs = spec, recs
        self.sync_canvas()
        self.rebuild()

    def set_snap(self, divisions):
        """Snap dragged/resized cells to a *divisions*-step grid (``0`` disables it)."""
        self.snap_divisions = int(divisions)

    def apply_theme(self, *_):
        """Re-take the chrome colours after a light/dark switch (also the ``themeChanged``
        slot this stage connects itself to).

        The scene's background brush and the canvas frame's pen are set once, at
        construction; unlike the cells (which repaint from the palette) Qt has no reason
        to revisit them.
        """
        self.gscene.setBackgroundBrush(QtGui.QColor(style.BG))
        self.frame_item.setPen(QtGui.QPen(QtGui.QColor(style.BORDER)))
        self.gscene.update()

    # ----- geometry -----
    def sync_canvas(self):
        """Match the canvas frame and the visible area to the spec's current size."""
        self.frame_item.setRect(0, 0, int(self.spec.width), int(self.spec.height))
        self.fit()

    def fit(self):
        m = self.margin
        self.setSceneRect(-m, -m, int(self.spec.width) + 2 * m,
                          int(self.spec.height) + 2 * m)
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.fit()

    # ----- items -----
    def rebuild(self):
        """Rebuild one scene item per cell from the spec, then re-render them."""
        blocker = QtCore.QSignalBlocker(self.gscene)      # removals fire selectionChanged
        for it in self.cells:
            self.gscene.removeItem(it)
        self.cells = []
        for i, clip in enumerate(self.spec.clips):
            rec = self.recs.get(clip.source)
            name = getattr(rec, "name", clip.source)
            item = _CellItem(i, name, clip.rect[2], clip.rect[3], self._on_geometry)
            item.setPos(clip.rect[0], clip.rect[1])
            self.gscene.addItem(item)
            self.cells.append(item)
        del blocker
        self.render_at(self._t)

    def sync_cell(self, i):
        """Push ``spec.clips[i].rect`` back onto its scene item (after a panel edit)."""
        if not (0 <= i < len(self.cells) and i < len(self.spec.clips)):
            return
        rect = self.spec.clips[i].rect
        b = QtCore.QSignalBlocker(self.gscene)
        self.cells[i].setPos(rect[0], rect[1])
        self.cells[i].set_cell_size(rect[2], rect[3])
        del b

    def _on_geometry(self, i):
        """A cell was dragged/resized on the scene → snap it, mirror it into the spec."""
        if not (0 <= i < len(self.spec.clips) and i < len(self.cells)):
            return                       # fires during rebuild, before the item is listed
        item = self.cells[i]
        pos = item.pos()
        rect = (int(round(pos.x())), int(round(pos.y())),
                int(item.cell_w), int(item.cell_h))
        if self.snap_divisions > 0:
            rect = engine.snap_rect(rect, (self.spec.width, self.spec.height),
                                    self.snap_divisions)
            b = QtCore.QSignalBlocker(self.gscene)
            item.setPos(rect[0], rect[1])
            item.set_cell_size(rect[2], rect[3])
            del b
        self.spec.clips[i].rect = rect
        self.render_cell(i)
        self.cellGeometryChanged.emit(i)

    # ----- selection -----
    def _on_scene_selection(self):
        sel = [it for it in self.cells if it.isSelected()]
        self.cellSelected.emit(sel[0].index if sel else -1)

    def select(self, i):
        """Select cell *i* (``-1`` clears) without echoing a selection signal back."""
        b = QtCore.QSignalBlocker(self.gscene)
        for it in self.cells:
            it.setSelected(it.index == i)
        del b

    def selected(self) -> int:
        for it in self.cells:
            if it.isSelected():
                return it.index
        return -1

    # ----- rendering -----
    def render_cell(self, i, t=None):
        """Re-render one cell — at *t*, or at the last time the view was given."""
        if t is not None:
            self._t = float(t)
        if not (0 <= i < len(self.spec.clips) and i < len(self.cells)):
            return
        clip = self.spec.clips[i]
        try:
            rgb = engine.render_cell(clip, self.recs[clip.source], self._t)
        except Exception:
            rgb = None
        self.cells[i].set_image(rgb)

    def render_at(self, t=None):
        """Render every cell at canvas time *t* (default: the last one used)."""
        if t is not None:
            self._t = float(t)
        for i in range(min(len(self.spec.clips), len(self.cells))):
            self.render_cell(i)

    def showEvent(self, ev):
        super().showEvent(ev)
        self.fit()
        self.render_at()


class CanvasComposerWindow(QtWidgets.QMainWindow):
    """Compose recordings from different collects onto one canvas — play and export it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Canvas composer — GottLUX")
        self.setWindowIcon(icons.app_icon())
        self.resize(1100, 720)

        self.spec = engine.CanvasSpec(width=640, height=480)
        self.recs: dict = {}                 # source key -> Recording
        self._loading = False                # guards panel <-> spec feedback loops

        self.clock = TimeController(self)
        self.clock.set_range(0.0, 1.0)

        # --- the canvas stage (the shared arrange widget) ---
        self.view = CanvasArrangeView(self.spec, self.recs)
        self.view.cellSelected.connect(self._on_scene_selection)
        self.view.cellGeometryChanged.connect(self._on_cell_geometry)
        self.setAcceptDrops(True)            # OS drops become new cells — see dropEvent

        self.transport = TransportBar(self.clock, show_accum=False, show_selection=False,
                                      host=self)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.view, 1)
        lv.addWidget(self.transport)

        self.setCentralWidget(plot_with_deck(left, self._build_deck()))

        self.clock.cursorChanged.connect(self._render_cells)
        self.statusBar().showMessage(
            "Add clips — or drop recordings here — to start composing.")

    @property
    def _items(self) -> list:
        """The stage's cell items (parallel to ``spec.clips``) — the arrange view owns them."""
        return self.view.cells

    # ================================================================== deck (controls)
    def _build_deck(self) -> QtWidgets.QWidget:
        deck = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(deck)

        # --- canvas geometry ---
        g_canvas = QtWidgets.QGroupBox("Canvas")
        self.canvas_w = QtWidgets.QSpinBox(); self.canvas_w.setRange(16, 8192)
        self.canvas_h = QtWidgets.QSpinBox(); self.canvas_h.setRange(16, 8192)
        self.canvas_w.setValue(self.spec.width); self.canvas_h.setValue(self.spec.height)
        for sp in (self.canvas_w, self.canvas_h):
            sp.setToolTip("Canvas size in pixels. A .raw export needs ≤ "
                          f"{engine.MAX_RAW_DIM} px per side (EVT2.1 coordinate range).")
            sp.valueChanged.connect(self._on_canvas_size)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("W")); row.addWidget(self.canvas_w)
        row.addWidget(QtWidgets.QLabel("H")); row.addWidget(self.canvas_h)
        row.addStretch(1)
        g_canvas.setLayout(row)
        v.addWidget(g_canvas)

        # --- clip list + add/remove ---
        g_clips = QtWidgets.QGroupBox("Clips")
        self.clip_list = QtWidgets.QListWidget()
        self.clip_list.currentRowChanged.connect(self._on_list_selection)
        self.clip_list.itemDoubleClicked.connect(self._edit_text_item)
        add_btn = QtWidgets.QPushButton("Add clips…"); add_btn.setIcon(icons.icon("add"))
        add_btn.setToolTip("Load one or more recordings (.raw / .h5 / cache) as new cells "
                           "— or just drop the files onto the window.")
        add_btn.clicked.connect(self._add_clips_dialog)
        title_btn = QtWidgets.QPushButton("Add title…")
        title_btn.setIcon(icons.icon("add"))
        title_btn.setToolTip("Add a text item — a full-frame title slide or a running "
                             "overlay line (double-click it to edit). Text renders in "
                             "video export only; a .raw export notes the omission.")
        title_btn.clicked.connect(self._add_title_dialog)
        rm_btn = QtWidgets.QPushButton("Remove"); rm_btn.setIcon(icons.icon("close"))
        rm_btn.setToolTip("Remove the selected clip or title from the canvas.")
        rm_btn.clicked.connect(self._remove_selected)
        cl = QtWidgets.QVBoxLayout()
        cl.addWidget(self.clip_list)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(add_btn); h.addWidget(title_btn); h.addWidget(rm_btn); h.addStretch(1)
        cl.addLayout(h)
        g_clips.setLayout(cl)
        v.addWidget(g_clips)

        # --- per-clip settings (auto-applied live) ---
        g_sel = QtWidgets.QGroupBox("Selected clip")
        form = QtWidgets.QFormLayout()

        self.sp_rect = [QtWidgets.QSpinBox() for _ in range(4)]
        for sp, hi in zip(self.sp_rect, (8192, 8192, 8192, 8192)):
            sp.setRange(-8192, hi); sp.valueChanged.connect(self._apply_panel)
        rr = QtWidgets.QHBoxLayout()
        for lbl, sp in zip(("x", "y", "w", "h"), self.sp_rect):
            rr.addWidget(QtWidgets.QLabel(lbl)); rr.addWidget(sp)
        form.addRow("Cell", rr)

        self.sp_roi = [QtWidgets.QSpinBox() for _ in range(4)]
        for sp in self.sp_roi:
            sp.setRange(0, 100000); sp.valueChanged.connect(self._apply_panel)
            sp.setToolTip("Source crop x0,y0 → x1,y1 in sensor pixels; the full sensor "
                          "means no crop.")
        rr = QtWidgets.QHBoxLayout()
        for sp in self.sp_roi:
            rr.addWidget(sp)
        form.addRow("ROI", rr)

        self.sp_offset = QtWidgets.QDoubleSpinBox()
        self.sp_offset.setRange(-3600.0, 3600.0); self.sp_offset.setDecimals(3)
        self.sp_offset.setSuffix(" s")
        self.sp_offset.setToolTip("Canvas time at which this clip starts playing.")
        self.sp_offset.valueChanged.connect(self._apply_panel)
        form.addRow("Offset", self.sp_offset)

        self.sp_scale = QtWidgets.QDoubleSpinBox()
        self.sp_scale.setRange(0.0001, 1000.0); self.sp_scale.setDecimals(4)
        self.sp_scale.setValue(1.0); self.sp_scale.setSingleStep(0.1)
        self.sp_scale.setToolTip("Clip-seconds per canvas-second: 1.0 = real time, "
                                 "0.1 = 10× slow-motion, 2.0 = 2× fast.")
        self.sp_scale.valueChanged.connect(self._apply_panel)
        form.addRow("Time scale", self.sp_scale)

        self.sp_accum = QtWidgets.QDoubleSpinBox()
        self.sp_accum.setRange(1e-5, 2.0); self.sp_accum.setDecimals(5)
        self.sp_accum.setValue(0.02); self.sp_accum.setSingleStep(0.005)
        self.sp_accum.setSuffix(" s")
        self.sp_accum.setToolTip("Accumulation window (exposure) for this cell only.")
        self.sp_accum.valueChanged.connect(self._apply_panel)
        form.addRow("Accum", self.sp_accum)

        self.cb_mode = QtWidgets.QComboBox(); self.cb_mode.addItems(_MODES)
        self.cb_mode.setToolTip("Accumulation mode for this cell (count, polarity, …).")
        self.cb_mode.currentIndexChanged.connect(self._apply_panel)
        form.addRow("Mode", self.cb_mode)

        self.cb_cmap = QtWidgets.QComboBox(); self.cb_cmap.addItems(_CMAPS)
        self.cb_cmap.setToolTip("Colormap for this cell.")
        self.cb_cmap.currentIndexChanged.connect(self._apply_panel)
        form.addRow("Color", self.cb_cmap)

        self.cb_tone = QtWidgets.QComboBox(); self.cb_tone.addItems(tonemap.EXPRESSIONS)
        self.cb_tone.setCurrentText("sqrt")
        self.cb_tone.setToolTip("Tone-map expression for this cell (dynamic-range "
                                "compression before the colormap).")
        self.cb_tone.currentIndexChanged.connect(self._apply_panel)
        self.sp_gamma = QtWidgets.QDoubleSpinBox()
        self.sp_gamma.setRange(0.1, 3.0); self.sp_gamma.setSingleStep(0.05)
        self.sp_gamma.setValue(0.5); self.sp_gamma.setPrefix("γ ")
        self.sp_gamma.setToolTip("Exponent for the 'gamma' expression.")
        self.sp_gamma.valueChanged.connect(self._apply_panel)
        rr = QtWidgets.QHBoxLayout()
        rr.addWidget(self.cb_tone); rr.addWidget(self.sp_gamma)
        form.addRow("Tone map", rr)

        self.chk_loop = QtWidgets.QCheckBox("Loop")
        self.chk_loop.setChecked(True)
        self.chk_loop.setToolTip("Repeat the clip from its start when its extent ends; "
                                 "unchecked, the cell goes dark instead.")
        self.chk_loop.toggled.connect(self._apply_panel)
        form.addRow("", self.chk_loop)

        g_sel.setLayout(form)
        v.addWidget(g_sel)
        self._sel_group = g_sel
        g_sel.setEnabled(False)

        # --- composition + exports ---
        g_out = QtWidgets.QGroupBox("Composition")
        save_btn = QtWidgets.QPushButton("Save…"); save_btn.setIcon(icons.icon("save"))
        save_btn.setToolTip("Save the composition as a .gottlux-canvas.json spec.")
        save_btn.clicked.connect(self._save_composition)
        load_btn = QtWidgets.QPushButton("Load…"); load_btn.setIcon(icons.icon("export"))
        load_btn.setToolTip("Load a saved composition and its source recordings.")
        load_btn.clicked.connect(self._load_composition)
        vid_btn = QtWidgets.QPushButton("Export video…"); vid_btn.setIcon(icons.icon("film"))
        vid_btn.setToolTip("Render the composition to an MP4 over the canvas timeline.")
        vid_btn.clicked.connect(self._export_video)
        raw_btn = QtWidgets.QPushButton("Export .raw…"); raw_btn.setIcon(icons.icon("export"))
        raw_btn.setToolTip("Re-encode the composited EVENTS as one EVT2.1 .raw whose "
                           "geometry is the canvas size. Per-clip colors/tone-maps do not "
                           "apply (a .raw carries events, not rendering); the canvas JSON "
                           "is written alongside as a sidecar.")
        raw_btn.clicked.connect(self._export_raw)
        gl = QtWidgets.QGridLayout()
        gl.addWidget(save_btn, 0, 0); gl.addWidget(load_btn, 0, 1)
        gl.addWidget(vid_btn, 1, 0); gl.addWidget(raw_btn, 1, 1)
        g_out.setLayout(gl)
        v.addWidget(g_out)
        v.addStretch(1)
        return deck

    # ================================================================== clips
    def _unique_source(self, base: str) -> str:
        if base not in self.recs:
            return base
        k = 2
        while f"{base}_{k}" in self.recs:
            k += 1
        return f"{base}_{k}"

    def add_clip_recording(self, rec, source: str | None = None,
                           rect: tuple | None = None) -> engine.CanvasClip:
        """Add *rec* as a new cell. *source* defaults to the recording's path or name
        (uniquified); *rect* defaults to a cascaded placement at the sensor's size,
        scaled down if it would not fit the canvas. Returns the new :class:`CanvasClip`."""
        key = self._unique_source(source or rec.source_path or rec.name)
        self.recs[key] = rec
        if rect is None:
            n = len(self.spec.clips)
            w = min(int(rec.width), self.spec.width)
            h = min(int(rec.height), self.spec.height)
            if w < rec.width or h < rec.height:            # keep the sensor aspect
                s = min(w / rec.width, h / rec.height)
                w, h = max(int(rec.width * s), 8), max(int(rec.height * s), 8)
            x = min(24 * n, max(self.spec.width - w, 0))
            y = min(24 * n, max(self.spec.height - h, 0))
            rect = (x, y, w, h)
        clip = engine.CanvasClip(source=key, rect=tuple(int(v) for v in rect))
        self.spec.clips.append(clip)
        self._rebuild()
        self.clip_list.setCurrentRow(len(self.spec.clips) - 1)
        self.statusBar().showMessage(
            f"Added {rec.name} — {len(self.spec.clips)} clip(s) on the canvas.")
        return clip

    def _add_clips_dialog(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add clips", "",
            "Event recordings (*.raw *.h5 *.hdf5 *.meta.json);;All files (*)")
        if paths:
            self._ingest_paths(paths)

    def _ingest_paths(self, paths):
        """Load *paths* and add each as a cell; a per-file failure lands in the status
        bar and never aborts the rest (the dialog and drop paths both come through)."""
        import gottlux as eb
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for p in paths:
                try:
                    self.add_clip_recording(eb.load(p, progress=lambda f: None), source=p)
                except Exception as e:
                    self.statusBar().showMessage(
                        f"failed to load {os.path.basename(p)}: {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _remove_selected(self):
        r = self.clip_list.currentRow()
        if 0 <= r < len(self.spec.clips):
            del self.spec.clips[r]
            self._rebuild()
        elif 0 <= r - len(self.spec.clips) < len(self.spec.texts):
            del self.spec.texts[r - len(self.spec.clips)]
            self._rebuild()

    # ================================================================== drag & drop
    def dragEnterEvent(self, ev):
        """Accept dragged recordings — the same set the 'Add clips…' dialog loads."""
        if droppable_paths(ev.mimeData()):
            ev.acceptProposedAction()
        else:
            super().dragEnterEvent(ev)

    def dropEvent(self, ev):
        """Every dropped recording becomes a new cell (name order; errors per file)."""
        paths = droppable_paths(ev.mimeData())
        if paths:
            ev.acceptProposedAction()
            self._ingest_paths(paths)
        else:
            super().dropEvent(ev)

    # ================================================================== text items
    def _add_title_dialog(self):
        dlg = TitleDialog(self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.add_text_item(text_item_from_values(dlg.values()))

    def add_text_item(self, txt: engine.CanvasText) -> engine.CanvasText:
        """Append *txt* to the composition. Listed after the cells with a ``T`` prefix;
        rendered by video export (and :meth:`current_frame`), never by a .raw export."""
        self.spec.texts.append(txt)
        self._rebuild()
        self.clip_list.setCurrentRow(len(self.spec.clips) + len(self.spec.texts) - 1)
        self.statusBar().showMessage(
            f"Added a {'title slide' if txt.kind == 'slide' else 'running title'} — "
            "text renders in video export only.")
        return txt

    def _edit_text_item(self, *_):
        """Double-click on a title row edits it in place (cells use the settings deck)."""
        r = self.clip_list.currentRow() - len(self.spec.clips)
        if not (0 <= r < len(self.spec.texts)):
            return
        dlg = TitleDialog(self, values=values_from_text_item(self.spec.texts[r]))
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.spec.texts[r] = text_item_from_values(dlg.values())
            self._rebuild()

    @staticmethod
    def _text_row(txt) -> str:
        """A text item's list line — ``T`` prefix, kind, span (the muted-tag style)."""
        head = (txt.text.strip().splitlines() or ["Title"])[0]
        span = ("whole video" if txt.span is None
                else f"{txt.span[0]:.2f}–{txt.span[1]:.2f}s")
        kind = "title slide" if txt.kind == "slide" else "running title"
        return f"T  {head}   [{kind}, {span}]"

    # ================================================================== scene plumbing
    def _rebuild(self):
        """Rebuild scene items + list from the spec, refresh the timeline, re-render."""
        self.view.rebuild()
        row = self.clip_list.currentRow()
        self.clip_list.clear()
        for clip in self.spec.clips:
            rec = self.recs[clip.source]
            t0, t1 = engine.clip_extent_s(clip, rec)
            self.clip_list.addItem(f"{rec.name}   [{t0:.2f}–{t1:.2f}s @ {clip.rect[0]},"
                                   f"{clip.rect[1]} {clip.rect[2]}×{clip.rect[3]}]")
        for txt in self.spec.texts:                    # titles list after the cells
            self.clip_list.addItem(self._text_row(txt))
        n_rows = len(self.spec.clips) + len(self.spec.texts)
        if n_rows:
            self.clip_list.setCurrentRow(min(max(row, 0), n_rows - 1))
        self._sel_group.setEnabled(bool(self.spec.clips))
        self._update_range()
        self._render_cells()

    def _update_range(self):
        dur = 1.0
        if self.spec.clips or self.spec.texts:
            dur = max(engine.canvas_duration(self.spec, self.recs),
                      engine.texts_extent_s(self.spec))
        self.clock.set_range(0.0, max(dur, 1e-3))

    def _on_canvas_size(self, *_):
        self.spec.width = int(self.canvas_w.value())
        self.spec.height = int(self.canvas_h.value())
        self.view.sync_canvas()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._render_cells()

    # ================================================================== selection sync
    def _on_scene_selection(self, index):
        if index < 0:
            return
        b = QtCore.QSignalBlocker(self.clip_list)
        self.clip_list.setCurrentRow(index)
        del b
        self._load_panel(index)

    def _on_list_selection(self, row):
        if not (0 <= row < len(self.spec.clips)):
            self._sel_group.setEnabled(False)
            return
        self._sel_group.setEnabled(True)
        self.view.select(row)
        self._load_panel(row)

    # ================================================================== settings panel
    def _load_panel(self, i):
        if not (0 <= i < len(self.spec.clips)):
            return
        clip = self.spec.clips[i]
        rec = self.recs[clip.source]
        self._loading = True
        try:
            for sp, val in zip(self.sp_rect, clip.rect):
                sp.setValue(int(val))
            roi = clip.roi or (0, 0, rec.width, rec.height)
            for sp, val in zip(self.sp_roi, roi):
                sp.setValue(int(val))
            self.sp_offset.setValue(clip.t_offset_s)
            self.sp_scale.setValue(clip.time_scale)
            self.sp_accum.setValue(clip.accumulation_s)
            self.cb_mode.setCurrentText(clip.mode)
            self.cb_cmap.setCurrentText(clip.colormap)
            self.cb_tone.setCurrentText(clip.tonemap)
            self.sp_gamma.setValue(clip.gamma)
            self.chk_loop.setChecked(clip.loop)
        finally:
            self._loading = False

    def _apply_panel(self, *_):
        """Write the panel back into the selected clip and re-render (live apply)."""
        if self._loading:
            return
        i = self.clip_list.currentRow()
        if not (0 <= i < len(self.spec.clips)):
            return
        clip = self.spec.clips[i]
        rec = self.recs[clip.source]
        clip.rect = tuple(int(sp.value()) for sp in self.sp_rect)
        x0, y0, x1, y1 = (int(sp.value()) for sp in self.sp_roi)
        full = (x0 <= 0 and y0 <= 0 and x1 >= rec.width and y1 >= rec.height)
        clip.roi = None if (full or x1 <= x0 or y1 <= y0) else (x0, y0, x1, y1)
        clip.t_offset_s = float(self.sp_offset.value())
        clip.time_scale = float(self.sp_scale.value())
        clip.accumulation_s = float(self.sp_accum.value())
        clip.mode = self.cb_mode.currentText()
        clip.colormap = self.cb_cmap.currentText()
        clip.tonemap = self.cb_tone.currentText()
        clip.gamma = float(self.sp_gamma.value())
        clip.loop = self.chk_loop.isChecked()
        self.view.sync_cell(i)
        self._update_range()
        self._render_one(i)

    def _on_cell_geometry(self, i):
        """A cell was dragged/resized on the stage → mirror its new rect into the panel."""
        if not (0 <= i < len(self.spec.clips)) or self.clip_list.currentRow() != i:
            return
        self._loading = True
        try:
            for sp, val in zip(self.sp_rect, self.spec.clips[i].rect):
                sp.setValue(int(val))
        finally:
            self._loading = False

    # ================================================================== rendering
    def _render_one(self, i):
        self.view.render_cell(i, self.clock.cursor)

    def _render_cells(self, *_):
        self.view.render_at(self.clock.cursor)

    def current_frame(self) -> np.ndarray:
        """The full composited canvas frame at the current cursor (engine-rendered)."""
        return engine.render_frame(self.spec, self.recs, self.clock.cursor)

    # ================================================================== compositions
    def _save_composition(self):
        if not (self.spec.clips or self.spec.texts):
            QtWidgets.QMessageBox.information(self, "Save composition", "Add clips first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save composition", "composition" + engine.SPEC_SUFFIX,
            f"Canvas composition (*{engine.SPEC_SUFFIX})")
        if not path:
            return
        if not path.endswith(".json"):
            path += engine.SPEC_SUFFIX
        engine.save_spec(self.spec, path)
        self.statusBar().showMessage(f"Saved composition → {path}")

    def _load_composition(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load composition", "",
            f"Canvas composition (*{engine.SPEC_SUFFIX} *.json)")
        if not path:
            return
        try:
            self.load_composition(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load composition", str(e))

    def load_composition(self, path: str):
        """Load a spec and its source recordings; clips whose sources fail to load are
        dropped (reported in the status bar) rather than blocking the rest."""
        import gottlux as eb
        spec = engine.load_spec(path)
        recs, kept, missing = {}, [], []
        for clip in spec.clips:
            if clip.source in recs:
                kept.append(clip)
                continue
            try:
                recs[clip.source] = eb.load(clip.source, progress=lambda f: None)
                kept.append(clip)
            except Exception:
                missing.append(clip.source)
        spec.clips = kept
        self.spec, self.recs = spec, recs
        b = QtCore.QSignalBlocker(self.canvas_w); self.canvas_w.setValue(spec.width); del b
        b = QtCore.QSignalBlocker(self.canvas_h); self.canvas_h.setValue(spec.height); del b
        self.view.set_composition(spec, recs)
        self._rebuild()
        msg = f"Loaded {len(kept)} clip(s) from {os.path.basename(path)}"
        if missing:
            msg += f" — {len(missing)} source(s) not found: " + ", ".join(missing)
        self.statusBar().showMessage(msg)

    # ================================================================== exports
    def _export_video(self):
        if not (self.spec.clips or self.spec.texts):
            QtWidgets.QMessageBox.information(self, "Export video", "Add clips first.")
            return
        self.clock.pause()
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export composition video", "canvas.mp4", "MP4 video (*.mp4)")
        if not out:
            return
        res = with_progress(self, "Exporting canvas video",
                            lambda cb: engine.export_video(self.spec, self.recs, out,
                                                           fps=30.0, progress=cb),
                            label="Rendering frames…")
        if res:
            from gottlux.io.paths import open_in_file_browser
            open_in_file_browser(os.path.dirname(os.path.abspath(res)))
            QtWidgets.QMessageBox.information(self, "Export video", f"Wrote {res}")
        else:
            QtWidgets.QMessageBox.warning(
                self, "Export video",
                "Encoding unavailable — install imageio-ffmpeg for MP4 export.")

    def _export_raw(self):
        if not self.spec.clips:
            QtWidgets.QMessageBox.information(self, "Export .raw", "Add clips first.")
            return
        self.clock.pause()
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export composited .raw", "canvas.raw", "EVT raw (*.raw)")
        if not out:
            return
        if not out.lower().endswith(".raw"):
            out += ".raw"
        try:
            res = with_progress(self, "Exporting composited .raw",
                                lambda cb: engine.export_raw(self.spec, self.recs, out,
                                                             progress=cb),
                                label="Re-encoding events…")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export .raw", str(e))
            return
        from gottlux.io.paths import open_in_file_browser
        open_in_file_browser(os.path.dirname(os.path.abspath(out)))
        note = ("\n".join(res["warnings"]) + "\n") if res.get("warnings") else ""
        QtWidgets.QMessageBox.information(
            self, "Export .raw",
            f"Wrote {res['n_events']:,} events ({res['width']}×{res['height']}, "
            f"{res['duration_s']:.2f} s) →\n{res['path']}\n\n"
            f"{note}Spec sidecar: {os.path.basename(res['sidecar'])}\n"
            "Note: the .raw carries events only — per-clip colormaps/tone-maps apply to "
            "rendering, not to the event stream.")
