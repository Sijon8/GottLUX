"""
timeline.py — the Timeline tab: a video editor for event recordings.

The tab is laid out like a simple video editor, top to bottom:

1. an **embedded preview viewport** showing the whole timeline rendered at the playhead;
2. a **transport** — play/pause/scrub on the timeline's own clock, spanning the program;
3. **track lanes** — a *sequence* lane and an *overlay* lane, drawn as blocks whose width
   is their duration, each clip block carrying a cached midpoint-frame thumbnail. Click a
   block to select it, drag to reorder, double-click to edit; click or drag the ruler to
   seek.

Everything renders through **one** path. The timeline compiles into a
:class:`~gottlux.core.canvas.Program` (:func:`~gottlux.core.canvas.compile_program`) in
which every sequential item is a :class:`~gottlux.core.canvas.CanvasSpec` segment: a plain
clip becomes a single cell covering the whole canvas carrying that clip's own
visualization settings, a **canvas block** (a mosaic) contributes its multi-cell spec
verbatim, and a title slide becomes a text item — with running titles and overlay clips
applied across every segment. The preview and 'Export video…' both walk that program
through :func:`gottlux.core.canvas.render_program_frame`, so what you scrub is what you
export.

Mosaics are edited **in place**: selecting a canvas block turns the preview viewport into
an arrange stage (:class:`~gottlux.app.canvas.CanvasArrangeView`, the same widget the
pop-out Canvas composer uses) where cells are dragged and resized, while the right-hand
deck edits the selected cell's EBS settings. The deck edits a plain clip the same way —
its one full-frame cell — so different clips on one timeline can carry different colormaps,
tone-maps, accumulation windows and modes, exactly as mosaic cells do.

Sizes are explicit rather than guessed: a **project canvas** preset (Native / 640×640 /
1280×720 / 1920×1080 / 1024×1024 / Custom) sets the preview and export geometry, and a
selected cell takes exact **fractions** of it (Full, 1/2, 1/3, 1/4 quadrant, 2/3), with
optional 1/12-grid snapping and an auto-tile button. All of that geometry math lives in
:mod:`gottlux.core.canvas` where it is pure and tested.

Exports: **Export video…** renders the whole program (segments + overlays + titles);
**Export .raw…** stays events-only — a plain sequence stitches through
:func:`gottlux.io.writer.stitch_clips` exactly as before (trim + crop + gap, one monotonic
clock), a timeline containing canvas blocks composites events into the canvas geometry
instead, and titles plus visualization settings are omitted with the usual one-line note.
Either export writes a **provenance folder** rather than a loose file
(:mod:`gottlux.run.export_provenance`): the artifact, a README naming every source clip
with its directory and SHA-256 and what was done to it, the machine-readable
``provenance.json``, and the composition spec — so a program built from fifteen collects
stays traceable to all fifteen files. The completion dialog reports the folder.

Clips arrive through 'Add clips…', 'Add canvas block…', **or by OS drag-and-drop** —
dropping several files at once offers 'as sequence clips' or 'as one mosaic block'.

:class:`RangeSlider` is a compact dual-handle In/Out trim widget and
:class:`HighlightRangeBar` its multi-band superset (the transport's selection strip);
:class:`TimelineEditor` is the tab itself, and :class:`TimelineEditorDialog` its thin modal
wrapper for the historical dialog entry points.
"""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from gottlux.app import icons
from gottlux.app import style
from gottlux.app.canvas import (CanvasArrangeView, TitleDialog, droppable_paths,
                                text_item_from_values, to_pixmap)
from gottlux.app.transport import TimeController, TransportBar
from gottlux.app.uikit import plot_with_deck, with_progress
from gottlux.core import canvas as engine
from gottlux.core import tonemap

#: The empty timeline's placeholder line — the drag-and-drop affordance.
_DROP_HINT = "Drop recordings here, or Add clips…"

#: Accumulation modes and colormaps offered per clip / per cell (the composer's set).
_MODES = ["count", "time_surface", "polarity", "polarity_ratio", "on", "off", "binary"]
_CMAPS = ["inferno", "viridis", "magma", "plasma", "cividis", "gray", "turbo",
          "coolwarm", "RdBu", "bwr", "seismic", "PiYG", "Spectral"]

#: Lane-block thumbnail height, in pixels (rendered once per clip, then cached).
_THUMB_H = 64


class RangeSlider(QtWidgets.QWidget):
    """Dual-handle In/Out trim slider; emits normalized ``(lo, hi)`` in 0..1."""
    rangeChanged = QtCore.Signal(float, float)

    def __init__(self):
        super().__init__()
        self.lo, self.hi, self._drag = 0.0, 1.0, None
        self.setMinimumHeight(26); self.setEnabled(False)

    def set_range(self, lo, hi):
        self.lo, self.hi = max(0.0, min(lo, hi)), min(1.0, max(lo, hi)); self.update()

    def _px(self, v):
        return int(9 + v * (self.width() - 18))

    def _val(self, x):
        return min(1.0, max(0.0, (x - 9) / max(1, self.width() - 18)))

    def paintEvent(self, _):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, y = self.width(), self.height() // 2
        p.setPen(QtGui.QPen(QtGui.QColor(style.BORDER), 4)); p.drawLine(9, y, w - 9, y)
        x0, x1 = self._px(self.lo), self._px(self.hi)
        p.setPen(QtGui.QPen(QtGui.QColor(style.SELECT), 6)); p.drawLine(x0, y, x1, y)
        p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(style.HANDLE))
        for x in (x0, x1):
            p.drawRoundedRect(QtCore.QRectF(x - 5, y - 9, 10, 18), 3, 3)

    def mousePressEvent(self, e):
        if not self.isEnabled():
            return
        v = self._val(e.position().x())
        self._drag = "lo" if abs(v - self.lo) <= abs(v - self.hi) else "hi"; self._move(v)

    def mouseMoveEvent(self, e):
        if self._drag:
            self._move(self._val(e.position().x()))

    def mouseReleaseEvent(self, e):
        self._drag = None

    def _move(self, v):
        if self._drag == "lo":
            self.lo = min(v, self.hi)
        elif self._drag == "hi":
            self.hi = max(v, self.lo)
        self.update(); self.rangeChanged.emit(self.lo, self.hi)


class HighlightRangeBar(QtWidgets.QWidget):
    """In/Out bar with a primary dual handle **plus** N draggable, colored *highlight* ranges.

    The primary range (cyan, magenta handles) is the live In/Out selection. Each committed
    highlight is a distinctly-colored band you can drag by either end; double-click a band to
    remove it. All values are normalized 0..1. The transport bar turns the highlights into a
    one-click "merge these moments into a single .raw". A drop-in superset of :class:`RangeSlider`
    for the transport's selection strip (the timeline/capture dialogs keep the simpler slider).
    """
    primaryChanged = QtCore.Signal(float, float)         # (lo, hi)
    highlightChanged = QtCore.Signal(int, float, float)  # (index, lo, hi)
    highlightRemoved = QtCore.Signal(int)                # (index)

    _PALETTE = [(255, 176, 32), (63, 185, 80), (165, 113, 220), (88, 166, 255),
                (247, 129, 102), (255, 99, 164), (86, 211, 194), (210, 153, 34)]
    _HIT_PX = 10

    def __init__(self):
        super().__init__()
        self.primary = [0.0, 1.0]
        self.highlights = []             # [[lo, hi], ...]
        self.cursor = None               # playhead fraction (0..1) drawn for visual alignment
        self.inset = 9                   # px track margin; matched to the seek slider's handle
        self._drag = None                # ("primary"|"hl", index, "lo"|"hi")
        self.setMinimumHeight(30)
        self.setEnabled(False)
        self.setToolTip(
            "Drag the cyan In/Out handles to select a range. 'Add' freezes it as a colored "
            "highlight band; drag a band's ends to adjust, double-click a band to remove it. "
            "'Save → Merge highlights' stitches every band into one .raw. The bright line is "
            "the playhead — it lines up with the seek slider directly below.")

    # ----- state in (from the controller) -----
    def set_primary(self, lo, hi):
        self.primary = [max(0.0, min(lo, hi)), min(1.0, max(lo, hi))]
        self.update()

    def set_highlights(self, ranges):
        self.highlights = [[max(0.0, min(a, b)), min(1.0, max(a, b))] for a, b in ranges]
        self.update()

    def set_cursor_frac(self, frac):
        """Place the playhead marker (0..1) so cut marks line up with the current time."""
        self.cursor = None if frac is None else min(1.0, max(0.0, float(frac)))
        self.update()

    def set_inset(self, px):
        """Track margin in px — set to the seek slider's half-handle so the two bars co-align."""
        self.inset = max(1, int(px))
        self.update()

    # ----- geometry -----
    def _px(self, v):
        return int(self.inset + v * (self.width() - 2 * self.inset))

    def _val(self, x):
        return min(1.0, max(0.0, (x - self.inset) / max(1, self.width() - 2 * self.inset)))

    def _handles(self):
        """(kind, index, which, px) for every draggable handle (primary + all highlights)."""
        yield ("primary", -1, "lo", self._px(self.primary[0]))
        yield ("primary", -1, "hi", self._px(self.primary[1]))
        for i, (lo, hi) in enumerate(self.highlights):
            yield ("hl", i, "lo", self._px(lo))
            yield ("hl", i, "hi", self._px(hi))

    @classmethod
    def _color(cls, i):
        return QtGui.QColor(*cls._PALETTE[i % len(cls._PALETTE)])

    # ----- paint -----
    def paintEvent(self, _):
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, y = self.width(), self.height() // 2
        p.setPen(QtGui.QPen(QtGui.QColor(style.BORDER), 4)); p.drawLine(9, y, w - 9, y)
        # committed highlight bands (each a distinct color, numbered)
        for i, (lo, hi) in enumerate(self.highlights):
            c = self._color(i); x0, x1 = self._px(lo), self._px(hi)
            fill = QtGui.QColor(c); fill.setAlpha(90)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(fill)
            p.drawRoundedRect(QtCore.QRectF(x0, y - 10, max(2, x1 - x0), 20), 3, 3)
            p.setPen(QtGui.QPen(c, 3))
            p.drawLine(x0, y - 10, x0, y + 10); p.drawLine(x1, y - 10, x1, y + 10)
            p.setPen(QtGui.QPen(QtGui.QColor(style.PLAYHEAD), 1))
            p.drawText(QtCore.QRectF(x0, y - 10, max(12, x1 - x0), 20),
                       QtCore.Qt.AlignCenter, str(i + 1))
        # the live primary In/Out on top
        x0, x1 = self._px(self.primary[0]), self._px(self.primary[1])
        p.setPen(QtGui.QPen(QtGui.QColor(style.SELECT), 6)); p.drawLine(x0, y, x1, y)
        p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(style.HANDLE))
        for x in (x0, x1):
            p.drawRoundedRect(QtCore.QRectF(x - 5, y - 9, 10, 18), 3, 3)
        # playhead (current time) — a bright vertical line aligned with the seek slider below
        if self.cursor is not None:
            xc = self._px(self.cursor)
            p.setPen(QtGui.QPen(QtGui.QColor(style.PLAYHEAD), 2))
            p.drawLine(xc, 1, xc, self.height() - 1)

    # ----- mouse -----
    def mousePressEvent(self, e):
        if not self.isEnabled():
            return
        x = e.position().x()
        best, bestd = None, 1e9
        for h in self._handles():
            d = abs(x - h[3])
            if d < bestd:
                best, bestd = h, d
        if best is not None and bestd <= self._HIT_PX:
            self._drag = (best[0], best[1], best[2])
        else:                                          # empty track: grab nearest primary handle
            v = self._val(x)
            self._drag = ("primary", -1,
                          "lo" if abs(v - self.primary[0]) <= abs(v - self.primary[1]) else "hi")
        self._move(self._val(x))

    def mouseMoveEvent(self, e):
        if self._drag:
            self._move(self._val(e.position().x()))

    def mouseReleaseEvent(self, _e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        if not self.isEnabled():
            return
        x = e.position().x()
        for i, (lo, hi) in enumerate(self.highlights):
            if self._px(lo) - 6 <= x <= self._px(hi) + 6:
                self.highlightRemoved.emit(i)
                return

    def _move(self, v):
        if not self._drag:
            return
        kind, i, which = self._drag
        if kind == "primary":
            lo, hi = self.primary
            lo, hi = (min(v, hi), hi) if which == "lo" else (lo, max(v, lo))
            self.primary = [lo, hi]; self.update()
            self.primaryChanged.emit(lo, hi)
        elif 0 <= i < len(self.highlights):
            lo, hi = self.highlights[i]
            lo, hi = (min(v, hi), hi) if which == "lo" else (lo, max(v, lo))
            self.highlights[i] = [lo, hi]; self.update()
            self.highlightChanged.emit(i, lo, hi)


# ====================================================================================
# The preview viewport
# ====================================================================================
class PreviewView(QtWidgets.QWidget):
    """The Timeline's embedded viewport: one engine-rendered program frame, letterboxed.

    A display surface and nothing more — the editor renders the frame through
    :func:`gottlux.core.canvas.render_program_frame` and hands it over, so the preview
    cannot drift away from what the video export writes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix = None
        self.hint = ("The timeline preview appears here.\n"
                     "Add clips — or drop recordings on this tab — then press play.")
        self.setMinimumHeight(180)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_frame(self, rgb):
        """Show one ``(H, W, 3)`` uint8 frame (``None`` falls back to the hint text)."""
        self._pix = None if rgb is None else to_pixmap(rgb)
        self.update()

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(style.BG))
        if self._pix is None:
            p.setPen(QtGui.QColor(style.MUTED))
            p.drawText(self.rect().adjusted(12, 12, -12, -12),
                       QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, self.hint)
            return
        size = self._pix.size().scaled(self.size(), QtCore.Qt.KeepAspectRatio)
        target = QtCore.QRect(QtCore.QPoint(0, 0), size)
        target.moveCenter(self.rect().center())
        p.drawPixmap(target, self._pix)


# ====================================================================================
# Lane-block thumbnails
# ====================================================================================
class ThumbCache:
    """Midpoint-frame thumbnails for the lane blocks — one render per clip + settings.

    A clip's thumbnail is rendered once through the engine at :data:`_THUMB_H` px with that
    clip's own visualization settings and kept; the cache key carries those settings (plus
    the trim and crop), so changing a colormap or dragging a trim handle refreshes exactly
    the thumbnails that changed and leaves the rest alone.
    """

    def __init__(self, height=_THUMB_H):
        self.height = int(height)
        self._cache = {}                      # id(item) -> (signature, QPixmap | None)

    @staticmethod
    def signature(item) -> tuple:
        """What a thumbnail depends on — recompute only when one of these moves."""
        cell = item.get("cell")
        look = None if cell is None else (cell.mode, cell.colormap, cell.tonemap,
                                          round(float(cell.gamma), 4),
                                          round(float(cell.accumulation_s), 6))
        return (id(item.get("rec")), round(float(item.get("t0", 0.0)), 4),
                round(float(item.get("t1", 0.0)), 4), item.get("roi"), look)

    def get(self, item):
        """The cached thumbnail for *item* (rendered on first use); ``None`` for non-clips."""
        if item.get("kind") != "clip" or item.get("rec") is None:
            return None
        sig = self.signature(item)
        hit = self._cache.get(id(item))
        if hit is not None and hit[0] == sig:
            return hit[1]
        pix = self._render(item)
        self._cache[id(item)] = (sig, pix)
        return pix

    def purge(self, items):
        """Drop entries for items that are no longer on the timeline."""
        live = {id(c) for c in items}
        for key in [k for k in self._cache if k not in live]:
            del self._cache[key]

    def clear(self):
        self._cache.clear()

    def _render(self, item):
        rec = item["rec"]
        roi = item.get("roi")
        src_w = (roi[2] - roi[0]) if roi else int(rec.width)
        src_h = (roi[3] - roi[1]) if roi else int(rec.height)
        w = max(8, int(round(self.height * max(src_w, 1) / max(src_h, 1))))
        cell = engine.CanvasClip.from_dict(item["cell"].to_dict())
        cell.rect, cell.roi, cell.loop = (0, 0, w, self.height), roi, False
        cell.t_offset_s, cell.time_scale = 0.0, 1.0     # canvas time == recording time
        mid = 0.5 * (float(item.get("t0", 0.0)) + float(item.get("t1", 0.0)))
        try:
            rgb = engine.render_cell(cell, rec, mid)
        except Exception:
            rgb = None
        return None if rgb is None else to_pixmap(rgb)


# ====================================================================================
# The track lanes
# ====================================================================================
class TrackLanes(QtWidgets.QWidget):
    """The sequence and overlay lanes: duration-proportional blocks under a time ruler.

    A dumb painter driven by :meth:`set_blocks` — the editor hands it a list of block
    descriptions (index, name, span, kind, thumbnail, selected) and it reports back what
    the user did: which block was clicked, where one was dragged to, which was
    double-clicked, and where the ruler was seeked to.
    """

    selected = QtCore.Signal(int)            # the clicked block's item index (-1 = none)
    reordered = QtCore.Signal(int, int)      # (item index, new position among sequence items)
    activated = QtCore.Signal(int)           # double-clicked block's item index
    seeked = QtCore.Signal(float)            # program seconds

    RULER_H = 20                             # the ruler strip's height, px
    SEQ_H = 66                               # the sequence lane's height, px
    OVER_H = 30                              # the overlay lane's height, px
    GUTTER = 52                              # left column holding the lane names, px
    PAD = 8                                  # right-hand track margin, px
    LANE_GAP = 6
    _MIN_DRAG = 5                            # px before a press becomes a reorder drag

    def __init__(self, parent=None):
        super().__init__(parent)
        self.seq: list = []                  # [{index, name, t0, dur, kind, pixmap, selected}]
        self.over: list = []
        self.duration = 1.0
        self.cursor = 0.0
        self._press = None                   # (item index, x) while a press is live
        self._drag_x = None                  # cursor x once the press became a drag
        self._scrub = False
        self.setMinimumHeight(self.RULER_H + self.SEQ_H + self.LANE_GAP + self.OVER_H + 8)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(False)
        self.setToolTip(
            "The sequence lane (top) plays left to right; the overlay lane (bottom) rides "
            "over all of it. Click a block to select it, drag to reorder, double-click to "
            "edit. Click or drag the ruler to seek.")

    # ----- model in -----
    def set_blocks(self, seq, over, duration_s):
        self.seq, self.over = list(seq), list(over)
        self.duration = max(float(duration_s), 1e-6)
        self.update()

    def set_cursor(self, t):
        self.cursor = float(t)
        self.update()

    # ----- geometry -----
    def _track_w(self):
        return max(self.width() - self.GUTTER - self.PAD, 1)

    def _x(self, t):
        return self.GUTTER + (float(t) / self.duration) * self._track_w()

    def _t(self, x):
        return max(0.0, min(self.duration,
                            (float(x) - self.GUTTER) / self._track_w() * self.duration))

    def _seq_y(self):
        return self.RULER_H + 2

    def _over_y(self):
        return self._seq_y() + self.SEQ_H + self.LANE_GAP

    def _block_rect(self, b, y, h):
        x0, x1 = self._x(b["t0"]), self._x(b["t0"] + b["dur"])
        return QtCore.QRectF(x0, y, max(x1 - x0, 3.0), h)

    def _hit(self, pos):
        """``(item index, 'seq'|'over')`` under *pos*, or ``(-1, None)``."""
        for blocks, y, h, lane in ((self.seq, self._seq_y(), self.SEQ_H, "seq"),
                                   (self.over, self._over_y(), self.OVER_H, "over")):
            for b in blocks:
                if self._block_rect(b, y, h).contains(pos):
                    return b["index"], lane
        return -1, None

    def _drop_position(self, x):
        """Which slot among the sequence blocks an x lands in (0 … len(seq))."""
        for i, b in enumerate(self.seq):
            r = self._block_rect(b, self._seq_y(), self.SEQ_H)
            if x < r.center().x():
                return i
        return len(self.seq)

    # ----- paint -----
    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(style.BG))
        self._paint_ruler(p)
        for y, h, name in ((self._seq_y(), self.SEQ_H, "Sequence"),
                           (self._over_y(), self.OVER_H, "Overlay")):
            p.fillRect(QtCore.QRectF(self.GUTTER, y, self._track_w(), h),
                       QtGui.QColor(style.BG2))
            p.setPen(QtGui.QColor(style.MUTED))
            f = p.font(); f.setPointSizeF(7.5); p.setFont(f)
            p.drawText(QtCore.QRectF(2, y, self.GUTTER - 6, h),
                       QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, name)
        if not self.seq and not self.over:
            p.setPen(QtGui.QColor(style.MUTED))
            f = p.font(); f.setPointSizeF(9.0); p.setFont(f)
            p.drawText(QtCore.QRectF(self.GUTTER, self._seq_y(), self._track_w(), self.SEQ_H),
                       QtCore.Qt.AlignCenter, _DROP_HINT)
        for b in self.seq:
            self._paint_block(p, b, self._seq_y(), self.SEQ_H)
        for b in self.over:
            self._paint_block(p, b, self._over_y(), self.OVER_H, compact=True)
        if self._drag_x is not None:                      # the reorder insertion marker
            x = self._x(0.0) if not self.seq else self._insert_x()
            p.setPen(QtGui.QPen(QtGui.QColor(style.ACCENT), 3))
            p.drawLine(QtCore.QPointF(x, self._seq_y()),
                       QtCore.QPointF(x, self._seq_y() + self.SEQ_H))
        x = self._x(self.cursor)                          # the playhead, across every lane
        p.setPen(QtGui.QPen(QtGui.QColor(style.ACCENT2), 2))
        p.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, self.height()))

    def _insert_x(self):
        pos = self._drop_position(self._drag_x)
        if pos >= len(self.seq):
            b = self.seq[-1]
            return self._x(b["t0"] + b["dur"])
        return self._x(self.seq[pos]["t0"])

    def _paint_ruler(self, p):
        p.fillRect(QtCore.QRectF(0, 0, self.width(), self.RULER_H), QtGui.QColor(style.PANEL))
        p.setPen(QtGui.QColor(style.MUTED))
        f = p.font(); f.setPointSizeF(7.0); p.setFont(f)
        step = _nice_step(self.duration)
        t = 0.0
        while t <= self.duration + 1e-9:
            x = self._x(t)
            p.drawLine(QtCore.QPointF(x, self.RULER_H - 6), QtCore.QPointF(x, self.RULER_H))
            p.drawText(QtCore.QRectF(x + 2, 1, 56, self.RULER_H - 6),
                       QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, f"{t:g}s")
            t += step

    @staticmethod
    def _fill(kind):
        """A block's body colour — each kind one step further off the lane's panel.

        :func:`gottlux.app.style.step` moves *away* from the page in either theme, so the
        three kinds stay distinguishable on a light lane instead of collapsing into white.
        """
        if kind == "block":
            return style.step(style.PANEL, 150)
        if kind == "title":
            return style.step(style.PANEL, 120)
        return QtGui.QColor(style.PANEL)

    def _paint_block(self, p, b, y, h, compact=False):
        r = self._block_rect(b, y, h)
        p.setPen(QtCore.Qt.NoPen); p.setBrush(self._fill(b["kind"]))
        p.drawRoundedRect(r, 3, 3)
        p.save()
        p.setClipRect(r)
        tx = r.left() + 5
        pix = b.get("pixmap")
        if pix is not None and not compact and r.width() > 24:
            th = int(h - 8)
            scaled = pix.scaledToHeight(th, QtCore.Qt.SmoothTransformation)
            p.drawPixmap(QtCore.QPoint(int(r.left()) + 3, int(y) + 4), scaled)
            tx = r.left() + 6 + scaled.width()
        p.setPen(QtGui.QColor(style.FG))
        f = p.font(); f.setPointSizeF(8.0); p.setFont(f)
        p.drawText(QtCore.QRectF(tx, y + 3, max(r.right() - tx - 4, 4), 14),
                   QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, b["name"])
        p.setPen(QtGui.QColor(style.MUTED))
        f.setPointSizeF(7.0); p.setFont(f)
        p.drawText(QtCore.QRectF(tx, y + h - 17, max(r.right() - tx - 4, 4), 14),
                   QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, b["detail"])
        p.restore()
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(style.ACCENT if b["selected"] else style.BORDER),
                            2 if b["selected"] else 1))
        p.drawRoundedRect(r, 3, 3)

    # ----- mouse -----
    def mousePressEvent(self, ev):
        x = ev.position().x()
        if ev.position().y() < self.RULER_H:
            self._scrub = True
            self.seeked.emit(self._t(x))
            return
        index, lane = self._hit(ev.position())
        self.selected.emit(index)
        self._press = (index, x) if (index >= 0 and lane == "seq") else None

    def mouseMoveEvent(self, ev):
        x = ev.position().x()
        if self._scrub:
            self.seeked.emit(self._t(x))
            return
        if self._press is not None and abs(x - self._press[1]) >= self._MIN_DRAG:
            self._drag_x = x
            self.update()

    def mouseReleaseEvent(self, ev):
        if self._scrub:
            self._scrub = False
            return
        if self._press is not None and self._drag_x is not None:
            index = self._press[0]
            self.reordered.emit(index, self._drop_position(self._drag_x))
        self._press, self._drag_x = None, None
        self.update()

    def mouseDoubleClickEvent(self, ev):
        if ev.position().y() < self.RULER_H:
            return
        index, _lane = self._hit(ev.position())
        if index >= 0:
            self.activated.emit(index)


def _nice_step(duration_s: float) -> float:
    """A ruler tick step that keeps roughly 4–10 labels across *duration_s*."""
    for step in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0):
        if duration_s / step <= 10:
            return step
    return max(duration_s / 10.0, 1e-3)


# ====================================================================================
# The editor
# ====================================================================================
class TimelineEditor(QtWidgets.QWidget):
    """The Timeline tab — preview, transport, track lanes, and the per-item settings deck.

    ``clips`` is the ordered model, one dict per item, all three kinds sharing the trim /
    lane fields so the lanes and the compiler can treat them uniformly:

    ``{"kind": "clip"}``
        ``rec`` (the recording), ``dur``, ``t0``/``t1`` (the In/Out trim), ``roi`` (the
        crop, or ``None``), and ``cell`` — a :class:`~gottlux.core.canvas.CanvasClip`
        carrying *this clip's own* look and clock (mode, accumulation, tone-map, gamma,
        colormap, time scale, loop). The compiler turns it into one full-canvas cell.
    ``{"kind": "block"}``
        ``spec`` + ``recs`` — a mosaic edited inline on the arrange stage and compiled
        verbatim.
    ``{"kind": "title"}``
        ``title`` — a :class:`~gottlux.app.canvas.TitleDialog` values dict; a slide takes a
        slot on the sequence, a running title rides the overlay lane.

    ``overlay`` moves an item off the sequence and onto the overlay lane. ``done`` fires
    after a completed hand-off (a finished export) — the dialog wrapper closes on it, the
    tab stays put.
    """

    done = QtCore.Signal()

    def __init__(self, parent=None, recordings=None):
        super().__init__(parent)
        self.clips = []                  # the ordered items (see the class docstring)
        self._sel = -1                   # selected item index (-1 = nothing selected)
        self._auto_seed = False          # the current content is an untouched auto-seed
        self._loading = False            # guards deck <-> model feedback loops
        self._program = None             # the compiled program (None = needs recompiling)
        self._segments = {}              # item index -> its ProgramSegment
        self._lane_order = []            # item indices drawn on the sequence lane, in order
        self._rendering = False          # re-entrancy guard for the preview render
        self.thumbs = ThumbCache()
        self.setAcceptDrops(True)        # OS file drops append clips (see dropEvent)

        self.clock = TimeController(self)
        self.clock.set_loop(True)
        self.clock.set_range(0.0, 1.0)

        self.preview = PreviewView()
        self.arrange = CanvasArrangeView(snap_divisions=engine.SNAP_DIVISIONS)
        self.arrange.cellSelected.connect(self._on_cell_selected)
        self.arrange.cellGeometryChanged.connect(self._on_cell_geometry)
        self.stage = QtWidgets.QStackedWidget()
        self.stage.addWidget(self.preview)
        self.stage.addWidget(self.arrange)

        self.transport = TransportBar(self.clock, show_accum=False, show_selection=False,
                                      host=self)
        self.lanes = TrackLanes()
        self.lanes.selected.connect(self.select)
        self.lanes.reordered.connect(self._reorder)
        self.lanes.activated.connect(self._activate)
        self.lanes.seeked.connect(self.clock.set_cursor)

        # the editor stack: preview viewport over transport over the track lanes
        self.split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.split.addWidget(self.stage)
        self.split.addWidget(self.transport)
        self.split.addWidget(self.lanes)
        self.transport.setMaximumHeight(self.transport.sizeHint().height())
        for i in range(3):
            self.split.setCollapsible(i, False)
        self.split.setStretchFactor(0, 3); self.split.setStretchFactor(2, 1)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addLayout(self._build_toolbar())
        lv.addWidget(self.split, 1)
        lv.addWidget(self.status_row())

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(plot_with_deck(left, self._build_deck()))

        self.clock.cursorChanged.connect(self._on_cursor)
        for rec in (recordings or []):
            self._append(rec)
        self._refresh()

    # ------------------------------------------------------------------ construction
    def _build_toolbar(self):
        add = QtWidgets.QPushButton("Add clips…"); add.setIcon(icons.icon("add"))
        add.setToolTip("Load recordings as sequence clips — or just drop the files on "
                       "this tab.")
        add.clicked.connect(self._add)
        addb = QtWidgets.QPushButton("Add canvas block…"); addb.setIcon(icons.icon("split"))
        addb.setToolTip("Insert a mosaic as ONE sequence item: several recordings tiled on "
                        "the project canvas. Select it to arrange the cells right here in "
                        "the preview — no pop-out window needed.")
        addb.clicked.connect(self._add_block)
        addt = QtWidgets.QPushButton("Add title…"); addt.setIcon(icons.icon("add"))
        addt.setToolTip("Insert a text item: a title slide (a sequence item occupying its "
                        "duration) or a running title on the overlay lane. Text renders in "
                        "video export only — the .raw export notes the omission. "
                        "Double-click a title block to edit it.")
        addt.clicked.connect(self._add_title)
        rm = QtWidgets.QPushButton("Remove"); rm.setIcon(icons.icon("close"))
        rm.setToolTip("Remove the selected item from the timeline.")
        rm.clicked.connect(self._remove)
        up = QtWidgets.QPushButton(); up.setIcon(icons.icon("arrow-left"))
        up.setToolTip("Move the selected item earlier (dragging a block does the same).")
        up.clicked.connect(lambda: self._move(-1))
        dn = QtWidgets.QPushButton(); dn.setIcon(icons.icon("arrow-right"))
        dn.setToolTip("Move the selected item later (dragging a block does the same).")
        dn.clicked.connect(lambda: self._move(+1))
        self.gap = QtWidgets.QDoubleSpinBox(); self.gap.setRange(0.0, 60.0)
        self.gap.setDecimals(3); self.gap.setSuffix(" s")
        self.gap.setToolTip("Blank gap inserted between consecutive sequence items — in "
                            "the preview, the video, and the .raw export alike.")
        self.gap.valueChanged.connect(lambda *_: self._refresh())
        row = QtWidgets.QHBoxLayout()
        for w in (add, addb, addt, rm, up, dn):
            row.addWidget(w)
        row.addSpacing(10)
        row.addWidget(QtWidgets.QLabel("Gap")); row.addWidget(self.gap)
        row.addStretch(1)
        return row

    def status_row(self):
        self.status = QtWidgets.QLabel(""); self.status.setObjectName("muted")
        return self.status

    def _build_deck(self):
        deck = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(deck)

        # --- project canvas: the preview's and the export's geometry ---
        g_canvas = QtWidgets.QGroupBox("Project canvas")
        self.canvas_cb = QtWidgets.QComboBox()
        self.canvas_cb.addItems([label for label, _ in engine.CANVAS_PRESETS])
        self.canvas_cb.setToolTip(
            "The pixel stage the timeline renders and exports on. 'Native' adopts the "
            "first clip's sensor geometry; changing this rescales the preview, every "
            "canvas block's arrangement, and the exported video.")
        self.canvas_cb.currentIndexChanged.connect(self._on_canvas_preset)
        self.canvas_w = QtWidgets.QSpinBox(); self.canvas_w.setRange(16, 8192)
        self.canvas_h = QtWidgets.QSpinBox(); self.canvas_h.setRange(16, 8192)
        self.canvas_w.setValue(1280); self.canvas_h.setValue(720)
        for sp in (self.canvas_w, self.canvas_h):
            sp.setEnabled(False)
            sp.setToolTip(f"Custom canvas size. A .raw export needs ≤ {engine.MAX_RAW_DIM} "
                          "px per side (the EVT2.1 coordinate range).")
            sp.valueChanged.connect(self._on_canvas_preset)
        cv = QtWidgets.QVBoxLayout()
        cv.addWidget(self.canvas_cb)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("W")); row.addWidget(self.canvas_w)
        row.addWidget(QtWidgets.QLabel("H")); row.addWidget(self.canvas_h)
        row.addStretch(1)
        cv.addLayout(row)
        g_canvas.setLayout(cv)
        v.addWidget(g_canvas)

        # --- trim / crop: the selected sequence clip ---
        g_clip = QtWidgets.QGroupBox("Selected clip")
        form = QtWidgets.QFormLayout()
        self.trim = RangeSlider(); self.trim.rangeChanged.connect(self._on_trim)
        form.addRow(self.trim)
        self.in_s = QtWidgets.QDoubleSpinBox(); self.in_s.setDecimals(3); self.in_s.setSuffix(" s")
        self.out_s = QtWidgets.QDoubleSpinBox(); self.out_s.setDecimals(3); self.out_s.setSuffix(" s")
        self.in_s.valueChanged.connect(self._on_spin); self.out_s.valueChanged.connect(self._on_spin)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("In")); row.addWidget(self.in_s)
        row.addWidget(QtWidgets.QLabel("Out")); row.addWidget(self.out_s)
        form.addRow("Trim", row)
        self.overlay_chk = QtWidgets.QCheckBox("Overlay lane")
        self.overlay_chk.setToolTip(
            "Move the selected clip onto the overlay lane: it renders over every segment "
            "for the whole program instead of taking a slot in the sequence (and stays "
            "out of the .raw export, which carries the sequential lane).")
        self.overlay_chk.toggled.connect(self._on_overlay)
        self.cut_sel_btn = QtWidgets.QPushButton("Cut selected → .raw…")
        self.cut_sel_btn.setToolTip("Write just the selected clip (its In/Out + crop) to a new .raw.")
        self.cut_sel_btn.clicked.connect(self._cut_selected)
        form.addRow(self.overlay_chk)
        form.addRow(self.cut_sel_btn)
        g_clip.setLayout(form)
        self._clip_group = g_clip
        v.addWidget(g_clip)

        # --- the selected cell: geometry inside a canvas block ---
        g_cell = QtWidgets.QGroupBox("Selected cell (canvas block)")
        form = QtWidgets.QFormLayout()
        self.sp_rect = [QtWidgets.QSpinBox() for _ in range(4)]
        row = QtWidgets.QHBoxLayout()
        for lbl, sp in zip(("x", "y", "w", "h"), self.sp_rect):
            sp.setRange(-8192, 8192); sp.valueChanged.connect(self._apply_settings)
            row.addWidget(QtWidgets.QLabel(lbl)); row.addWidget(sp)
        form.addRow("Cell", row)
        self.cell_preset_cb = QtWidgets.QComboBox()
        self.cell_preset_cb.addItem("—")
        self.cell_preset_cb.addItems([label for label, _ in engine.CELL_PRESETS])
        self.cell_preset_cb.setToolTip("Size the selected cell as an exact fraction of the "
                                       "project canvas.")
        self.cell_preset_cb.activated.connect(self._apply_cell_preset)
        form.addRow("Size", self.cell_preset_cb)
        self.sp_offset = QtWidgets.QDoubleSpinBox()
        self.sp_offset.setRange(-3600.0, 3600.0); self.sp_offset.setDecimals(3)
        self.sp_offset.setSuffix(" s")
        self.sp_offset.setToolTip("When this cell starts playing inside its block (a plain "
                                  "sequence clip takes its offset from the In point instead).")
        self.sp_offset.valueChanged.connect(self._apply_settings)
        form.addRow("Offset", self.sp_offset)
        self.snap_chk = QtWidgets.QCheckBox(f"Snap to a 1/{engine.SNAP_DIVISIONS} grid")
        self.snap_chk.setChecked(True)
        self.snap_chk.setToolTip("Snap dragged and resized cells to a twelfth of the canvas.")
        self.snap_chk.toggled.connect(
            lambda on: self.arrange.set_snap(engine.SNAP_DIVISIONS if on else 0))
        self.tile_btn = QtWidgets.QPushButton("Auto-tile cells")
        self.tile_btn.setToolTip("Lay every cell of the selected canvas block into the "
                                 "best-fit grid over the canvas.")
        self.tile_btn.clicked.connect(self._autotile)
        self.done_btn = QtWidgets.QPushButton("Done arranging")
        self.done_btn.setToolTip("Leave arrange mode and go back to the timeline preview.")
        self.done_btn.clicked.connect(lambda: self.select(-1))
        form.addRow(self.snap_chk)
        form.addRow(self.tile_btn)
        form.addRow(self.done_btn)
        g_cell.setLayout(form)
        self._cell_group = g_cell
        v.addWidget(g_cell)

        # --- the visualization settings of whatever cell is selected ---
        g_look = QtWidgets.QGroupBox("Visualization")
        form = QtWidgets.QFormLayout()
        self.cb_mode = QtWidgets.QComboBox(); self.cb_mode.addItems(_MODES)
        self.cb_mode.setToolTip("Accumulation mode for this clip/cell (count, polarity, …).")
        self.cb_mode.currentIndexChanged.connect(self._apply_settings)
        form.addRow("Mode", self.cb_mode)
        self.cb_cmap = QtWidgets.QComboBox(); self.cb_cmap.addItems(_CMAPS)
        self.cb_cmap.setToolTip("Colormap for this clip/cell.")
        self.cb_cmap.currentIndexChanged.connect(self._apply_settings)
        form.addRow("Color", self.cb_cmap)
        self.cb_tone = QtWidgets.QComboBox(); self.cb_tone.addItems(tonemap.EXPRESSIONS)
        self.cb_tone.setCurrentText("sqrt")
        self.cb_tone.currentIndexChanged.connect(self._apply_settings)
        self.sp_gamma = QtWidgets.QDoubleSpinBox()
        self.sp_gamma.setRange(0.1, 3.0); self.sp_gamma.setSingleStep(0.05)
        self.sp_gamma.setValue(0.5); self.sp_gamma.setPrefix("γ ")
        self.sp_gamma.setToolTip("Exponent for the 'gamma' expression.")
        self.sp_gamma.valueChanged.connect(self._apply_settings)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.cb_tone); row.addWidget(self.sp_gamma)
        form.addRow("Tone map", row)
        self.sp_accum = QtWidgets.QDoubleSpinBox()
        self.sp_accum.setRange(1e-5, 2.0); self.sp_accum.setDecimals(5)
        self.sp_accum.setValue(0.02); self.sp_accum.setSingleStep(0.005)
        self.sp_accum.setSuffix(" s")
        self.sp_accum.setToolTip("Accumulation window (exposure) for this clip/cell only.")
        self.sp_accum.valueChanged.connect(self._apply_settings)
        form.addRow("Accum", self.sp_accum)
        self.sp_scale = QtWidgets.QDoubleSpinBox()
        self.sp_scale.setRange(0.0001, 1000.0); self.sp_scale.setDecimals(4)
        self.sp_scale.setValue(1.0); self.sp_scale.setSingleStep(0.1)
        self.sp_scale.setToolTip("Clip-seconds per program-second: 1.0 = real time, "
                                 "0.1 = 10× slow-motion (and the block takes 10× longer).")
        self.sp_scale.valueChanged.connect(self._apply_settings)
        form.addRow("Time scale", self.sp_scale)
        self.chk_loop = QtWidgets.QCheckBox("Loop")
        self.chk_loop.setToolTip("Repeat from the start once the source runs out — for a "
                                 "cell that has to fill a longer slot.")
        self.chk_loop.toggled.connect(self._apply_settings)
        form.addRow("", self.chk_loop)
        self.roi = [QtWidgets.QSpinBox() for _ in range(4)]
        row = QtWidgets.QHBoxLayout()
        for sp in self.roi:
            sp.setRange(0, 100000); sp.valueChanged.connect(self._on_roi); row.addWidget(sp)
            sp.setToolTip("Source crop x0,y0 → x1,y1 in sensor pixels; the full sensor "
                          "means no crop. Applies to whichever clip/cell is selected — and "
                          "for a plain clip it rides the .raw export too.")
        form.addRow("Crop x0,y0→x1,y1", row)
        g_look.setLayout(form)
        self._look_group = g_look
        v.addWidget(g_look)

        # --- exports ---
        g_out = QtWidgets.QGroupBox("Export")
        self.out_edit = QtWidgets.QLineEdit()
        self.out_edit.setPlaceholderText("output .raw path")
        browse = QtWidgets.QPushButton("Browse…"); browse.clicked.connect(self._browse)
        vid_btn = QtWidgets.QPushButton("Export video…"); vid_btn.setIcon(icons.icon("film"))
        vid_btn.setToolTip("Render the WHOLE program — every segment, the overlay lane and "
                           "the titles — frame by frame to an MP4, through the same render "
                           "path the preview uses.")
        vid_btn.clicked.connect(self._export_video)
        raw_btn = QtWidgets.QPushButton("Export .raw…"); raw_btn.setIcon(icons.icon("export"))
        raw_btn.setToolTip(
            "Write the sequential lane as EVENTS in one valid EVT2.1 .raw: plain clips "
            "stitch end-to-end with their trim + crop on a single monotonic clock; a "
            "timeline holding canvas blocks composites the events into the canvas "
            "geometry instead. Titles and per-clip visualization settings are render-only "
            "and are noted, not written.")
        raw_btn.clicked.connect(self._export_raw)
        ov = QtWidgets.QVBoxLayout()
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.out_edit, 1); row.addWidget(browse)
        ov.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(vid_btn); row.addWidget(raw_btn)
        ov.addLayout(row)
        g_out.setLayout(ov)
        v.addWidget(g_out)
        v.addStretch(1)
        return deck

    # ------------------------------------------------------------------ panel protocol
    @staticmethod
    def _auto_out_path(rec):
        """The output path :meth:`_append` derives for a first clip ('' when path-less)."""
        return (os.path.splitext(rec.source_path)[0] + "_stitched.raw"
                if rec.source_path else "")

    def set_recording(self, rec):
        """Adopt the app's shared recording as the timeline's first clip.

        Called through the same wiring as every other tab whenever a recording loads
        (including the preview → full-decode swap). Only an empty timeline is seeded —
        and a previous auto-seed that is still untouched (a single clip with no trim/crop/
        overlay edits) is refreshed to the new recording — so a timeline the user is
        building is never clobbered.
        """
        if rec is None:
            return
        if self.clips:
            c = self.clips[0]
            untouched = (self._auto_seed and len(self.clips) == 1
                         and c["kind"] == "clip" and c["t0"] == 0.0 and c["t1"] == c["dur"]
                         and not c.get("roi") and not c.get("overlay"))
            if not untouched or c["rec"] is rec:
                return
            if self.out_edit.text() == self._auto_out_path(c["rec"]):
                self.out_edit.clear()          # re-derive the output from the new clip
            self.clips.clear()
        self._append(rec)
        self._auto_seed = True

    def capture_clock(self):
        """The Timeline runs on its own clock (the program's), like the Multi-clip slate."""
        return self.clock

    def sync(self):
        """Re-render the viewport at the current playhead ('Sync views')."""
        if self.stage.currentWidget() is self.arrange:
            self._render_arrange()
        else:
            self._render_preview(force=True)

    def showEvent(self, ev):
        super().showEvent(ev)
        self.sync()

    def hideEvent(self, ev):
        super().hideEvent(ev)
        self.clock.pause()

    def keyPressEvent(self, ev):
        """Spacebar plays/pauses the timeline's own clock when this tab has focus."""
        if ev.key() == QtCore.Qt.Key_Space and not ev.isAutoRepeat():
            self.clock.toggle()
            ev.accept()
            return
        super().keyPressEvent(ev)

    # ------------------------------------------------------------------ drag & drop
    def dragEnterEvent(self, ev):
        """Accept dragged recordings — the same set 'Add clips…' loads (gottlux.load)."""
        if droppable_paths(ev.mimeData()):
            ev.acceptProposedAction()
        else:
            super().dragEnterEvent(ev)

    def dropEvent(self, ev):
        """Append every dropped recording; several at once may become one mosaic block."""
        paths = droppable_paths(ev.mimeData())
        if not paths:
            super().dropEvent(ev)
            return
        ev.acceptProposedAction()
        how = self._ask_multi_drop(len(paths)) if len(paths) > 1 else "clips"
        if how == "block":
            self._append_block(paths)
        elif how == "clips":
            self._ingest_paths(paths)

    def _ask_multi_drop(self, n):
        """How a multi-file drop should land: ``'clips'``, ``'block'``, or ``None``."""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Add dropped recordings")
        box.setText(f"{n} recordings dropped — add them as separate clips one after "
                    "another, or tile them into one mosaic block on the sequence?")
        seq_btn = box.addButton("As sequence clips", QtWidgets.QMessageBox.AcceptRole)
        mos_btn = box.addButton("As one mosaic block", QtWidgets.QMessageBox.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.exec()
        clicked = box.clickedButton()
        return "clips" if clicked is seq_btn else ("block" if clicked is mos_btn else None)

    # ------------------------------------------------------------------ the model
    @staticmethod
    def _default_cell():
        """A fresh per-clip settings template (a timeline slot plays once, so no loop)."""
        return engine.CanvasClip(source="", loop=False)

    def _append(self, rec):
        dur = float(rec.duration_s)
        self.clips.append({"kind": "clip", "rec": rec, "name": rec.name, "dur": dur,
                           "t0": 0.0, "t1": dur, "roi": None, "overlay": False,
                           "cell": self._default_cell()})
        self._refresh()
        if not self.out_edit.text() and rec.source_path:
            self.out_edit.setText(os.path.splitext(rec.source_path)[0] + "_stitched.raw")

    def _add(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add clips", "", "Event recordings (*.raw *.h5 *.hdf5 *.meta.json);;All files (*)")
        if paths:
            self._ingest_paths(paths)

    def _load_paths(self, paths):
        """Load *paths* → ``[Recording]``; a per-file failure lands in the status label
        and never aborts the rest (the file dialogs and OS drops all come through here)."""
        import gottlux as eb
        out = []
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for p in paths:
                try:
                    out.append(eb.load(p, progress=lambda f: None))
                except Exception as e:
                    self.status.setText(f"failed to load {os.path.basename(p)}: {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        return out

    def _ingest_paths(self, paths):
        """Append each loadable path as its own sequence clip."""
        self._auto_seed = False           # user-built content: never auto-refreshed away
        for rec in self._load_paths(paths):
            self._append(rec)

    def _remove(self):
        if 0 <= self._sel < len(self.clips):
            self._auto_seed = False       # a deliberate edit: the seed contract is over
            del self.clips[self._sel]
            self._sel = min(self._sel, len(self.clips) - 1)
            self._refresh()

    def _move(self, d):
        """Shift the selected item one slot earlier/later (the buttons' reorder)."""
        r, j = self._sel, self._sel + d
        if 0 <= r < len(self.clips) and 0 <= j < len(self.clips):
            self.clips[r], self.clips[j] = self.clips[j], self.clips[r]
            self._sel = j
            self._refresh()

    def _reorder(self, index, position):
        """Drop the dragged sequence block into slot *position* among the sequence blocks.

        *position* indexes the blocks the lane actually drew (``_lane_order``), so the two
        can never disagree about what "third block" means.
        """
        if not (0 <= index < len(self.clips)) or index not in self._lane_order:
            return
        rest = [i for i in self._lane_order if i != index]
        item = self.clips.pop(index)
        rest = [i - 1 if i > index else i for i in rest]     # indices after the removal
        at = rest[position] if position < len(rest) else len(self.clips)
        self.clips.insert(at, item)
        self._sel = at
        self._refresh()

    def _activate(self, index):
        """Double-click: titles open their dialog, canvas blocks enter arrange mode."""
        if not (0 <= index < len(self.clips)):
            return
        self.select(index)
        if self.clips[index]["kind"] == "title":
            self._edit_title()

    # ------------------------------------------------------------------ canvas blocks
    def _add_block(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add canvas block", "",
            "Event recordings (*.raw *.h5 *.hdf5 *.meta.json);;All files (*)")
        if paths:
            self._append_block(paths)

    def _append_block(self, paths):
        """Tile the loadable *paths* into one mosaic and insert it as a sequence item."""
        self._auto_seed = False
        recs = self._load_paths(paths)
        if recs:
            self.append_block_recordings(recs)

    def append_block_recordings(self, recs, name=None):
        """Insert *recs* as one canvas block, auto-tiled over the project canvas.

        The public seam the 'Add canvas block…' action, the multi-file drop, and tests all
        use — a block is a whole :class:`~gottlux.core.canvas.CanvasSpec` occupying a single
        slot on the sequence, arranged inline once selected.
        """
        wh = self._canvas_wh()
        spec = engine.CanvasSpec(width=wh[0], height=wh[1])
        mapping, rects = {}, engine.autotile(len(recs), wh)
        for i, (rec, rect) in enumerate(zip(recs, rects)):
            key = f"{i}:{rec.name}"
            mapping[key] = rec
            spec.clips.append(engine.CanvasClip(source=key, rect=rect, loop=False))
        item = {"kind": "block", "rec": None,
                "name": name or f"Mosaic ({len(recs)} clips)",
                "spec": spec, "recs": mapping, "roi": None, "overlay": False,
                "t0": 0.0, "t1": 0.0, "dur": 0.0}
        item["dur"] = engine.item_duration_s(self._compile_item(item))
        item["t1"] = item["dur"]
        self.clips.append(item)
        self._refresh()
        self.select(len(self.clips) - 1)      # a new block opens straight into arrange mode
        return item

    def _autotile(self):
        c = self.current()
        if c is None or c["kind"] != "block":
            return
        for clip, rect in zip(c["spec"].clips, engine.autotile(len(c["spec"].clips),
                                                               self._canvas_wh())):
            clip.rect = rect
        self.arrange.rebuild()
        self._load_settings()
        self._refresh()

    def _apply_cell_preset(self, *_):
        """Size the selected cell as an exact fraction of the project canvas."""
        cell, in_block = self._settings_cell()
        label = self.cell_preset_cb.currentText()
        if cell is None or not in_block or label == "—":
            return
        cell.rect = engine.cell_preset_rect(label, self._canvas_wh(), origin=cell.rect[:2])
        self.arrange.sync_cell(self.arrange.selected())
        self._load_settings()
        self._refresh()

    # ------------------------------------------------------------------ title items
    @staticmethod
    def _title_name(vals):
        """A title item's display name — the first non-empty line of its text."""
        return (str(vals.get("text", "")).strip().splitlines() or ["Title"])[0]

    def _append_title(self, vals: dict):
        """Append a title item from a :class:`~gottlux.app.canvas.TitleDialog` values dict:
        a *slide* takes a slot on the sequence, a *running* title rides the overlay lane."""
        self._auto_seed = False
        dur = float(vals["duration_s"]) if vals["kind"] == "slide" else 0.0
        self.clips.append({"kind": "title", "rec": None, "name": self._title_name(vals),
                           "dur": dur, "t0": 0.0, "t1": dur, "roi": None,
                           "overlay": vals["kind"] == "overlay", "title": dict(vals)})
        self._refresh()

    def _update_title(self, r: int, vals: dict):
        """Rewrite title item *r* from an edited values dict (the double-click path)."""
        c = self.clips[r]
        dur = float(vals["duration_s"]) if vals["kind"] == "slide" else 0.0
        c.update(name=self._title_name(vals), dur=dur, t0=0.0, t1=dur,
                 overlay=vals["kind"] == "overlay", title=dict(vals))
        self._refresh()

    def _add_title(self):
        dlg = TitleDialog(self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self._append_title(dlg.values())

    def _edit_title(self, *_):
        """Edit the selected title item in place (the lane's double-click path)."""
        r = self._sel
        if not (0 <= r < len(self.clips)) or self.clips[r]["kind"] != "title":
            return
        dlg = TitleDialog(self, values=self.clips[r]["title"])
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self._update_title(r, dlg.values())

    # ------------------------------------------------------------------ the program
    def _native_wh(self):
        """The 'Native' canvas: the first clip's sensor (or the first block's canvas)."""
        for c in self.clips:
            if c["kind"] == "clip" and c["rec"] is not None:
                return (int(c["rec"].width), int(c["rec"].height))
            if c["kind"] == "block":
                return (int(c["spec"].width), int(c["spec"].height))
        return None

    def _canvas_wh(self):
        """The project canvas the preview and the exports render on."""
        return engine.canvas_preset_size(
            self.canvas_cb.currentText(), native_wh=self._native_wh(),
            custom_wh=(self.canvas_w.value(), self.canvas_h.value()))

    def _on_canvas_preset(self, *_):
        """A new project canvas: rescale every block's arrangement onto the new stage."""
        custom = self.canvas_cb.currentText() == engine.CUSTOM_CANVAS
        for sp in (self.canvas_w, self.canvas_h):
            sp.setEnabled(custom)
        wh = self._canvas_wh()
        for c in self.clips:
            if c["kind"] != "block":
                continue
            scaled = engine.rescale_spec(c["spec"], wh)      # in place: the stage holds it
            c["spec"].width, c["spec"].height = scaled.width, scaled.height
            for clip, ref in zip(c["spec"].clips, scaled.clips):
                clip.rect = ref.rect
        self.arrange.sync_canvas()
        self.arrange.rebuild()
        self._refresh()

    def _compile_item(self, c) -> dict:
        """One editor item as the program compiler's input (see :func:`compile_program`)."""
        key = id(c)
        if c["kind"] == "title":
            return {"kind": "title", "key": key, "name": c["name"], "dur": c["dur"],
                    "overlay": bool(c.get("overlay")),
                    "text": text_item_from_values(c["title"])}
        if c["kind"] == "block":
            return {"kind": "block", "key": key, "name": c["name"], "spec": c["spec"],
                    "recs": c["recs"], "dur": None, "overlay": False}
        return {"kind": "clip", "key": key, "name": c["name"], "rec": c["rec"],
                "t0": c["t0"], "t1": c["t1"], "roi": c.get("roi"), "cell": c["cell"],
                "overlay": bool(c.get("overlay"))}

    def program(self):
        """The compiled :class:`~gottlux.core.canvas.Program` for the whole timeline.

        Rebuilt on demand after any edit, so the preview, the lanes, the clock's range and
        the video export always describe the same thing.
        """
        if self._program is None:
            items = [self._compile_item(c) for c in self.clips]
            self._program = engine.compile_program(items, self._canvas_wh(),
                                                   gap_s=self.gap.value())
            by_key = {seg.key: seg for seg in self._program.segments if seg.key is not None}
            self._segments = {i: by_key[id(c)] for i, c in enumerate(self.clips)
                              if id(c) in by_key}
        return self._program

    def _invalidate(self):
        self._program, self._segments = None, {}

    # ------------------------------------------------------------------ selection
    def current(self):
        """The selected item, or ``None``."""
        return self.clips[self._sel] if 0 <= self._sel < len(self.clips) else None

    def selected_index(self) -> int:
        return self._sel

    def select(self, index):
        """Select item *index* (``-1`` clears): syncs the deck and the preview mode."""
        self._sel = int(index) if 0 <= int(index) < len(self.clips) else -1
        c = self.current()
        if c is not None and c["kind"] == "block":
            self.arrange.set_composition(c["spec"], c["recs"])
            self.arrange.set_snap(engine.SNAP_DIVISIONS if self.snap_chk.isChecked() else 0)
            self.stage.setCurrentWidget(self.arrange)
            self._render_arrange()
        else:
            self.stage.setCurrentWidget(self.preview)
            self._render_preview(force=True)
        self._load_settings()
        self._paint_lanes()

    def _on_cell_selected(self, _index):
        self._load_settings()

    def _on_cell_geometry(self, _index):
        """A cell was dragged/resized inline: mirror it into the deck and mark the program
        stale. The stage already re-rendered that one cell, so nothing else has to run."""
        self._load_settings()
        self._invalidate()

    # ------------------------------------------------------------------ the settings deck
    def _settings_cell(self):
        """``(cell, in_block)`` — the :class:`CanvasClip` the Visualization rows edit.

        A plain clip edits its own single full-frame cell; a selected canvas block edits
        whichever of its cells is selected on the arrange stage.
        """
        c = self.current()
        if c is None:
            return None, False
        if c["kind"] == "block":
            i = self.arrange.selected()
            return (c["spec"].clips[i] if 0 <= i < len(c["spec"].clips) else None), True
        if c["kind"] == "clip":
            return c["cell"], False
        return None, False

    def _settings_rec(self):
        """The recording behind the settings cell — the crop's full-frame reference."""
        c = self.current()
        if c is None:
            return None
        if c["kind"] == "block":
            cell, _ = self._settings_cell()
            return None if cell is None else c["recs"].get(cell.source)
        return c.get("rec")

    def _load_settings(self):
        """Reflect the selection into the deck (never the other way — see ``_loading``)."""
        c = self.current()
        is_clip = c is not None and c["kind"] == "clip"
        cell, in_block = self._settings_cell()
        rec = self._settings_rec()
        self._clip_group.setEnabled(is_clip)
        self._cell_group.setEnabled(in_block)
        self._look_group.setEnabled(cell is not None)
        self._loading = True
        try:
            if is_clip:
                for w, val in ((self.in_s, c["t0"]), (self.out_s, c["t1"])):
                    w.setRange(0.0, c["dur"]); w.setValue(val)
                if c["dur"]:
                    self.trim.set_range(c["t0"] / c["dur"], c["t1"] / c["dur"])
                self.trim.setEnabled(True)
            else:
                self.trim.setEnabled(False)
            self.overlay_chk.setChecked(bool(c is not None and c.get("overlay")))
            if cell is not None and rec is not None:
                roi = (cell.roi if in_block else c.get("roi")) or \
                    (0, 0, rec.width, rec.height)
                for sp, val in zip(self.roi, roi):
                    sp.setValue(int(val))
            if in_block:
                for sp, val in zip(self.sp_rect, cell.rect if cell else (0, 0, 0, 0)):
                    sp.setValue(int(val))
                self.sp_offset.setValue(cell.t_offset_s if cell else 0.0)
            if cell is not None:
                self.cb_mode.setCurrentText(cell.mode)
                self.cb_cmap.setCurrentText(cell.colormap)
                self.cb_tone.setCurrentText(cell.tonemap)
                self.sp_gamma.setValue(cell.gamma)
                self.sp_accum.setValue(cell.accumulation_s)
                self.sp_scale.setValue(cell.time_scale)
                self.chk_loop.setChecked(cell.loop)
        finally:
            self._loading = False

    def _apply_settings(self, *_):
        """Write the deck back into the selected clip/cell and re-render (live apply)."""
        if self._loading:
            return
        cell, in_block = self._settings_cell()
        if cell is None:
            return
        cell.mode = self.cb_mode.currentText()
        cell.colormap = self.cb_cmap.currentText()
        cell.tonemap = self.cb_tone.currentText()
        cell.gamma = float(self.sp_gamma.value())
        cell.accumulation_s = float(self.sp_accum.value())
        cell.time_scale = float(self.sp_scale.value())
        cell.loop = self.chk_loop.isChecked()
        if in_block:
            cell.rect = tuple(int(sp.value()) for sp in self.sp_rect)
            cell.t_offset_s = float(self.sp_offset.value())
            i = self.arrange.selected()
            self.arrange.sync_cell(i)
            self.arrange.render_cell(i)
        self._refresh()

    # ------------------------------------------------------------------ trim / crop / lane
    def _on_trim(self, lo, hi):
        c = self.current()
        if c is None or c["kind"] != "clip":
            return
        c["t0"], c["t1"] = lo * c["dur"], hi * c["dur"]
        b = QtCore.QSignalBlocker(self.in_s); self.in_s.setValue(c["t0"]); del b
        b = QtCore.QSignalBlocker(self.out_s); self.out_s.setValue(c["t1"]); del b
        self._refresh()

    def _on_spin(self, *_):
        if self._loading:
            return
        c = self.current()
        if c is None or c["kind"] != "clip":
            return
        c["t0"] = min(self.in_s.value(), self.out_s.value())
        c["t1"] = max(self.in_s.value(), self.out_s.value())
        if c["dur"]:
            b = QtCore.QSignalBlocker(self.trim)
            self.trim.set_range(c["t0"] / c["dur"], c["t1"] / c["dur"]); del b
        self._refresh()

    def _on_roi(self, *_):
        """The source crop of whatever is selected — a plain clip, or a block's cell.

        For a plain clip it is also the crop the ``.raw`` export applies; a full-frame or
        degenerate box means 'no crop' either way.
        """
        if self._loading:
            return
        cell, in_block = self._settings_cell()
        rec = self._settings_rec()
        if cell is None or rec is None:
            return
        x0, y0, x1, y1 = (sp.value() for sp in self.roi)
        full = (x0 <= 0 and y0 <= 0 and x1 >= rec.width and y1 >= rec.height)
        roi = None if (full or x1 <= x0 or y1 <= y0) else (x0, y0, x1, y1)
        if in_block:
            cell.roi = roi
            self.arrange.render_cell(self.arrange.selected())
        else:
            self.current()["roi"] = roi
        self._refresh()

    def _on_overlay(self, on):
        """Move the selected clip onto / off the overlay lane."""
        if self._loading:
            return
        c = self.current()
        if c is not None and c["kind"] == "clip":
            c["overlay"] = bool(on)       # a title's lane is fixed by its kind
            self._refresh()

    # ------------------------------------------------------------------ lanes + preview
    def _detail(self, c, dur) -> str:
        """The muted second line on a lane block — the item's span and its tags."""
        if c["kind"] == "title":
            kind = "title slide" if c["title"]["kind"] == "slide" else "running title"
            return f"{kind} · {dur:.2f}s" if c["dur"] else kind
        if c["kind"] == "block":
            return f"mosaic · {len(c['spec'].clips)} cells · {dur:.2f}s"
        return (f"{c['t0']:.2f}–{c['t1']:.2f}s · {dur:.2f}s"
                + (" · crop" if c.get("roi") else ""))

    def _paint_lanes(self):
        prog = self.program()
        total = max(prog.duration_s, 1e-3)
        seq, over = [], []
        self._lane_order = []
        for i, c in enumerate(self.clips):
            seg = self._segments.get(i)
            t0 = seg.t0_s if seg is not None else 0.0
            dur = seg.duration_s if seg is not None else total
            if c.get("overlay") and c["kind"] == "clip":
                dur = min(engine.item_duration_s(self._compile_item(c)), total)
            block = {"index": i, "name": c["name"], "kind": c["kind"], "t0": t0,
                     "dur": dur, "detail": self._detail(c, dur),
                     "selected": i == self._sel, "pixmap": self.thumbs.get(c)}
            if c.get("overlay"):
                over.append(block)
            elif seg is not None:
                seq.append(block)
                self._lane_order.append(i)
        self.lanes.set_blocks(seq, over, total)
        self.lanes.set_cursor(self.clock.cursor)

    def _on_cursor(self, t):
        self.lanes.set_cursor(t)
        if self.stage.currentWidget() is self.arrange:
            self._render_arrange()
        else:
            self._render_preview()

    def _render_arrange(self):
        """Drive the arrange stage's cells at the selected block's own local time."""
        self.program()
        seg = self._segments.get(self._sel)
        t = 0.0 if seg is None else min(max(self.clock.cursor - seg.t0_s, 0.0),
                                        seg.duration_s)
        self.arrange.render_at(t)

    def current_frame(self):
        """The whole program rendered at the playhead — exactly what the viewport shows
        and what 'Export video…' writes, from the one engine render path."""
        return engine.render_program_frame(self.program(), self.clock.cursor)

    def _render_preview(self, force=False):
        """Render the program at the playhead into the viewport — the one render path."""
        if self._rendering or (not force and not self.isVisible()):
            return
        if not self.program().segments:
            self.preview.set_frame(None)
            return
        self._rendering = True
        try:
            self.preview.set_frame(self.current_frame())
        except Exception as e:
            self.preview.set_frame(None)
            self.status.setText(f"preview render failed: {e}")
        finally:
            self._rendering = False

    def _refresh(self):
        """Recompile, re-range the clock, repaint the lanes and the viewport, restate."""
        self._invalidate()
        self.thumbs.purge(self.clips)
        prog = self.program()
        self.clock.set_range(0.0, max(prog.duration_s, 1e-3))
        self._paint_lanes()
        if self.stage.currentWidget() is self.arrange:
            self._render_arrange()
        else:
            self._render_preview()
        n_clip = sum(1 for c in self.clips if c["kind"] == "clip")
        n_block = sum(1 for c in self.clips if c["kind"] == "block")
        n_title = sum(1 for c in self.clips if c["kind"] == "title")
        n_over = len(self._overlay_set())
        wh = self._canvas_wh()
        self.status.setText(
            f"{n_clip} clip(s) · {prog.duration_s:.2f}s total · canvas {wh[0]}×{wh[1]}"
            + (f" · {n_block} canvas block(s)" if n_block else "")
            + (f" · {n_title} title(s)" if n_title else "")
            + (f" · {n_over} on the overlay lane" if n_over else ""))

    # ------------------------------------------------------------------ the lanes' sets
    def _overlay_set(self):
        """The overlay lane, in order — overlay-marked clips plus every running title."""
        return [c for c in self.clips if c.get("overlay")]

    def _sequential_set(self):
        """The sequence lane: every item not marked overlay (title slides ride here too)."""
        return [c for c in self.clips if not c.get("overlay")]

    # ------------------------------------------------------------------ exports
    def _browse(self):
        out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Timeline output",
                                                       self.out_edit.text(),
                                                       "EVT raw (*.raw)")
        if out:
            self.out_edit.setText(out)

    # ------------------------------------------------------------------ provenance
    def _export_spec(self, render=False):
        """``(spec, recs)`` for this timeline, with real file paths as the cell sources.

        :func:`~gottlux.core.canvas.program_spec` namespaces its cell sources per segment
        (they are lookup keys, not paths); rewriting each one to its recording's file path
        is what makes the saved spec reloadable — the Canvas composer's
        :func:`~gottlux.core.canvas.load_recordings` can find the clips again. A recording
        with no file on disk keeps its key, and the provenance document records why.
        *render* asks for the full rendering flatten (the overlay lane and the titles too).
        """
        spec, recs = engine.program_spec(self.program(), overlays=render, texts=render)
        mapped = {}
        for clip in spec.clips:
            rec = recs.get(clip.source)
            path = str(getattr(rec, "source_path", "") or "")
            clip.source = path or clip.source
            mapped[clip.source] = rec
        return spec, mapped

    def _write_spec(self, folder, out, render=True):
        """Save the composition spec into *folder* beside the artifact; returns its name."""
        spec, _recs = self._export_spec(render=render)
        name = os.path.splitext(os.path.basename(out))[0] + engine.SPEC_SUFFIX
        engine.save_spec(spec, os.path.join(folder, name))
        return name

    def _export_sources(self):
        """Every distinct recording behind this timeline, with one usage row per placement.

        Returns ``(sources, usage)``. ``sources`` holds one
        :func:`~gottlux.run.export_provenance.source_facts` dict per distinct recording (a
        file placed twice is listed once, identified by its absolute path); ``usage`` holds
        one row per placement — every clip on the sequence lane, every clip on the overlay
        lane, and **every cell inside every canvas block** — so a fifteen-clip timeline
        accounts for all fifteen. The rows read the cells the compiler actually rendered,
        so the trims, crops and rects recorded are the ones the export used.
        """
        from gottlux.run import export_provenance as prov
        prog = self.program()
        sources, usage, index = [], [], {}

        def source_of(rec):
            # A recording with no file on disk is keyed by identity: an empty
            # ``source_path`` must not become ``abspath('')`` — the working directory —
            # which would collapse every in-memory recording into one bogus source row.
            raw = str(getattr(rec, "source_path", "") or "")
            path = os.path.abspath(raw) if raw else ""
            key = os.path.normcase(path) if path else id(rec)
            if key not in index:
                index[key] = len(sources) + 1
                sources.append(prov.source_facts(path, rec=rec))
            return index[key]

        gap = float(self.gap.value())
        lane = [i for i, c in enumerate(self.clips)
                if not c.get("overlay") and i in self._segments]
        over_items = [c for c in self.clips if c.get("overlay") and c["kind"] == "clip"
                      and c.get("rec") is not None]
        over_cells = dict(zip((id(c) for c in over_items), prog.overlay_clips))
        for i, c in enumerate(self.clips):
            if c["kind"] == "title":
                continue
            seg = self._segments.get(i)
            span = [seg.t0_s, seg.t1_s] if seg is not None else None
            after = gap if (gap > 0 and lane and i in lane and i != lane[-1]) else None
            if c["kind"] == "block":
                cells = seg.spec.clips[:seg.n_own] if seg is not None else c["spec"].clips
                lookup = seg.recs if seg is not None else c["recs"]
                for cell in cells:
                    rec = lookup.get(cell.source)
                    if rec is None:
                        continue
                    usage.append(prov.cell_usage(
                        cell, source_of(rec), name=cell.source.split(":", 1)[-1],
                        lane="sequence", block=c["name"], program_span_s=span,
                        gap_after_s=after))
                continue
            rec = c.get("rec")
            if rec is None:
                continue
            cell = (over_cells.get(id(c)) if c.get("overlay")
                    else (seg.spec.clips[0] if (seg is not None and seg.n_own) else None))
            usage.append(prov.cell_usage(
                cell if cell is not None else c["cell"], source_of(rec), name=c["name"],
                lane="overlay" if c.get("overlay") else "sequence",
                trim_in_s=c["t0"], trim_out_s=c["t1"], program_span_s=span,
                gap_after_s=None if c.get("overlay") else after))
        return sources, usage

    def _export_texts(self):
        """The title items, described for the provenance document (they render in video)."""
        out = []
        for c in self.clips:
            if c["kind"] != "title":
                continue
            vals = c["title"]
            out.append({"text": str(vals.get("text", "")),
                        "kind": ("title slide (takes a sequence slot)"
                                 if vals.get("kind") == "slide"
                                 else "running title (overlay lane)"),
                        "duration_s": round(float(c["dur"]), 3),
                        "anchor": str(vals.get("anchor", "")),
                        "font_size_px": int(vals.get("font_size_px", 0))})
        return out

    def _export_settings(self, **extra):
        """The full parameter list the provenance document prints."""
        prog, wh = self.program(), self._canvas_wh()
        settings = {
            "Project canvas": f"{self.canvas_cb.currentText()} → {wh[0]} × {wh[1]} px",
            "Program duration": f"{prog.duration_s:.3f} s",
            "Program segments": len(prog.segments),
            "Gap between sequence items": f"{self.gap.value():g} s",
            "Sequence clips": sum(1 for c in self.clips
                                  if c["kind"] == "clip" and not c.get("overlay")),
            "Canvas blocks": sum(1 for c in self.clips if c["kind"] == "block"),
            "Overlay-lane clips": sum(1 for c in self.clips
                                      if c["kind"] == "clip" and c.get("overlay")),
            "Title items": sum(1 for c in self.clips if c["kind"] == "title"),
        }
        settings.update({k: v for k, v in extra.items() if v is not None})
        return settings

    def _write_provenance(self, folder, kind, artifact, settings, spec=None, steps=(),
                          warnings=()):
        """Write the folder's README.md + provenance.json for one finished export."""
        from gottlux.run import export_provenance as prov
        sources, usage = self._export_sources()
        return prov.write_provenance(
            folder, kind, artifact, sources, settings,
            extra={"usage": usage, "texts": self._export_texts(),
                   "warnings": list(warnings),
                   "reproduce": {"spec": spec, "steps": list(steps)}})

    # ------------------------------------------------------------------ exports
    def _export_video(self):
        """Render the whole program — segments, overlays, titles — into an export folder."""
        prog = self.program()
        if not prog.segments:
            self.status.setText("Add at least one clip.")
            return
        self.clock.pause()
        base = os.path.splitext(self.out_edit.text().strip())[0] or "timeline"
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export timeline video", base + ".mp4", "MP4 video (*.mp4)")
        if not out:
            return
        from gottlux.io.paths import open_in_file_browser
        from gottlux.run import export_provenance as prov
        folder = prov.export_folder(out)
        target = prov.artifact_path(folder, out)
        try:
            res = with_progress(self, "Exporting timeline video",
                                lambda cb: engine.export_program_video(prog, target,
                                                                       fps=30.0,
                                                                       progress=cb),
                                label="Rendering frames…")
        except Exception as e:
            prov.discard_folder(folder)
            QtWidgets.QMessageBox.critical(self, "Export video", str(e))
            return
        if not res:
            prov.discard_folder(folder)
            QtWidgets.QMessageBox.warning(
                self, "Export video",
                "Encoding unavailable — install imageio-ffmpeg for MP4 export.")
            return
        spec = self._write_spec(folder, out, render=True)
        self._write_provenance(
            folder, "Timeline video (MP4)",
            prov.artifact_facts(res["path"], frames=res["frames"], fps=res["fps"],
                                duration_s=res["duration_s"], width=res["width"],
                                height=res["height"], canvas=res["canvas"],
                                codec=res["codec"]),
            self._export_settings(**{"Frame rate": f"{res['fps']:g} fps",
                                     "Codec": res["codec"]}),
            spec=spec,
            steps=[f"Open the Canvas composer, load `{spec}`, and use 'Export video…' — "
                   "the spec carries every cell, its clock and its look.",
                   "Or rebuild the timeline: add the sources in the order listed above, "
                   "apply each row's trim, crop and settings, set the project canvas and "
                   "gap from the settings table, then use 'Export video…'."])
        open_in_file_browser(folder)
        QtWidgets.QMessageBox.information(
            self, "Export video",
            f"Rendered {len(prog.segments)} segment(s) · {prog.duration_s:.2f} s "
            f"({prog.width}×{prog.height}) →\n{folder}\n\n"
            f"The folder holds {os.path.basename(target)}, README.md, provenance.json "
            f"and {spec}.")
        self.done.emit()

    def _export_raw(self):
        """Write the sequential lane as events — a stitch, or a canvas composite."""
        if not self.clips:
            self.status.setText("Add at least one clip."); return
        seq = self._sequential_set()
        slides = [c for c in seq if c["kind"] == "title"]   # a .raw carries events, not text
        blocks = [c for c in seq if c["kind"] == "block"]
        plain = [c for c in seq if c["kind"] == "clip"]
        if not plain and not blocks:
            self.status.setText(
                "Nothing to stitch — the sequential lane holds no clips (title slides "
                "render in video export only; move a clip off the overlay lane to "
                "include it).")
            return
        out = self.out_edit.text().strip()
        if not out:
            self.status.setText("Choose an output path."); return
        if not out.lower().endswith(".raw"):
            out += ".raw"
        n_over = sum(1 for c in self.clips if c.get("overlay") and c["kind"] != "title")
        notes = ([f"{n_over} overlay clip(s) not included — a .raw carries the "
                  "sequential lane"] if n_over else [])
        notes += [engine.text_omission_note(len(slides))] if slides else []
        note = "".join(f"\n({n})" for n in notes)
        from gottlux.run import export_provenance as prov
        folder = prov.export_folder(out)
        target = prov.artifact_path(folder, out)
        try:
            msg = (self._composite_raw(target, folder, notes) if blocks
                   else self._stitch_raw(target, folder, plain, notes))
        except Exception as e:
            prov.discard_folder(folder)
            QtWidgets.QMessageBox.critical(self, "Export .raw", str(e)); return
        from gottlux.io.paths import open_in_file_browser
        open_in_file_browser(folder)
        QtWidgets.QMessageBox.information(
            self, "Export .raw", msg + note
            + f"\n\nExport folder (artifact + README.md + provenance.json):\n{folder}")
        self.done.emit()

    def _stitch_raw(self, out, folder, plain, notes=()):
        """The plain-sequence path: clips stitched end-to-end on one monotonic clock."""
        from gottlux.io import writer
        from gottlux.run import export_provenance as prov
        specs = [(c["rec"], c["t0"], c["t1"], c.get("roi")) for c in plain]
        res = with_progress(self, "Stitching clips → .raw",
                            lambda cb: writer.stitch_clips(out, specs,
                                                           gap_s=self.gap.value(),
                                                           progress=cb),
                            label="Stitching the clips…")
        rec0 = plain[0]["rec"]
        spec = self._write_spec(folder, out, render=True)
        self._write_provenance(
            folder, "Timeline .raw (stitched sequence)",
            prov.artifact_facts(out, events=res["n_events"],
                                duration_s=res["duration_s"],
                                width=int(rec0.width), height=int(rec0.height)),
            self._export_settings(**{
                "Event export path": "stitch — clips laid end to end on one monotonic clock",
                "Sensor geometry": f"{rec0.width} × {rec0.height} px (unchanged)",
                "Stitched segments": len(res["segments"])}),
            spec=spec,
            steps=["Load the sources listed above, set each clip's trim and crop from the "
                   "usage rows and the gap from the settings table, then use "
                   "'Export .raw…' on the Timeline tab.",
                   f"The saved `{spec}` describes the same composition for **rendering** "
                   "(the .raw itself carries events, not the per-clip look)."],
            warnings=notes)
        return (f"Stitched {len(specs)} clip(s) → {os.path.basename(out)}\n"
                f"{res['n_events']:,} events · {res['duration_s']:.2f} s")

    def _composite_raw(self, out, folder, notes=()):
        """The canvas path: a timeline holding mosaics re-encodes into canvas geometry."""
        from gottlux.run import export_provenance as prov
        spec, recs = self._export_spec()
        res = with_progress(self, "Compositing the timeline → .raw",
                            lambda cb: engine.export_raw(spec, recs, out, progress=cb),
                            label="Re-encoding events…")
        # export_raw already wrote the composition beside the .raw — that one sidecar,
        # now inside the export folder, is the spec; no second copy is written.
        sidecar = os.path.basename(res["sidecar"])
        self._write_provenance(
            folder, "Timeline .raw (canvas composite)",
            prov.artifact_facts(res["path"], events=res["n_events"],
                                duration_s=res["duration_s"], width=res["width"],
                                height=res["height"],
                                canvas=(res["width"], res["height"])),
            self._export_settings(**{
                "Event export path": "canvas composite — events re-encoded into the "
                                     "canvas geometry",
                "Composited cells": len(spec.clips)}),
            spec=sidecar,
            steps=[f"Open the Canvas composer and load `{sidecar}` to re-render the "
                   "styled composition.",
                   "Re-encode the events again with 'Export .raw…' from the same "
                   "composition — the cell rects and clocks above are the full recipe."],
            warnings=list(notes) + list(res.get("warnings") or []))
        return (f"Composited {len(spec.clips)} cell(s) → {os.path.basename(out)}\n"
                f"{res['n_events']:,} events ({res['width']}×{res['height']}, "
                f"{res['duration_s']:.2f} s)\n"
                f"Spec sidecar: {sidecar}\n"
                "Note: the .raw carries events only — per-clip colormaps/tone-maps apply "
                "to rendering, not to the event stream.")

    def _cut_selected(self):
        """Write just the selected clip (its In/Out trim + crop ROI) to a new .raw."""
        c = self.current()
        if c is None or c["kind"] != "clip":
            return
        base = (os.path.splitext(c["rec"].source_path)[0] if c["rec"].source_path
                else os.path.join(os.getcwd(), c["name"]))
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Cut selected clip → .raw", f"{base}_clip.raw", "EVT raw (*.raw)")
        if not out:
            return
        if not out.lower().endswith(".raw"):
            out += ".raw"
        from gottlux.io import writer
        from gottlux.io.paths import open_in_file_browser
        try:
            n = with_progress(
                self, "Cutting clip → .raw",
                lambda cb: writer.cut_clip(c["rec"], out, t0=c["t0"], t1=c["t1"],
                                           roi=c.get("roi"), progress=cb),
                label="Writing the clip…")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Cut", str(e)); return
        open_in_file_browser(os.path.dirname(os.path.abspath(out)))
        QtWidgets.QMessageBox.information(self, "Cut", f"Wrote {n:,} events →\n{out}")


class TimelineEditorDialog(QtWidgets.QDialog):
    """The timeline editor as a modal dialog — a thin wrapper embedding :class:`TimelineEditor`.

    Kept for the historical entry points (the toolbar's 'Clip editor' action and the Live
    viewer's Export menu); the same editor lives permanently on the main window's
    **Timeline** tab. Attribute access falls through to the embedded editor, so the
    dialog exposes the editor's full surface (``clips``, ``lanes``, the ``_on_*`` /
    ``_export_*`` handlers) unchanged; the editor's ``done`` signal accepts the dialog.
    """

    def __init__(self, parent=None, recordings=None):
        super().__init__(parent)
        self.setWindowTitle("Timeline — edit and export a program")
        self.setMinimumSize(900, 620)
        self.editor = TimelineEditor(self, recordings=recordings)
        self.editor.done.connect(self.accept)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.editor, 1)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(9, 0, 9, 9)
        row.addWidget(bb)
        v.addLayout(row)

    def __getattr__(self, name):
        # Fires only for names QDialog itself lacks — everything editor-owned resolves here.
        try:
            editor = self.__dict__["editor"]
        except KeyError:                       # during construction, before the editor exists
            raise AttributeError(name) from None
        return getattr(editor, name)
