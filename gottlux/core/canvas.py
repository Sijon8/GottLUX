"""
canvas.py — the EBS canvas composer engine: many recordings, one composited frame.

A *canvas* is a fixed pixel stage onto which several recordings — possibly from different
collects, sensors, and time bases — are placed as independently positioned, sized, and
styled cells. Each :class:`CanvasClip` carries its own visualization settings (accumulation
mode and window, tone-map expression, colormap, an optional source ROI crop) **and its own
clock mapping** (a canvas-time offset, a time scale for slow-motion/speed-up, and looping),
so one canvas can hold a real-time wide view beside a 10× slow-motion crop of the same
moment. The whole composition is a small, JSON-serializable :class:`CanvasSpec`, which makes
a composition reproducible: save it, reload it, re-render it.

Three consumers share this engine:

* :func:`render_frame` — one composited ``(H, W, 3)`` frame at a canvas time, for live
  playback (the GUI in :mod:`gottlux.app.canvas`) and for video export;
* :func:`export_video` — the composition rendered frame-by-frame to an MP4 via
  :class:`gottlux.viz.video.VideoWriter`;
* :func:`export_raw` — the composition re-encoded as **events**: each clip's windowed,
  ROI-cropped events are spatially mapped into its destination cell, their timestamps
  rescaled onto the canvas clock, and the merged stream written as one valid EVT2.1
  ``.raw`` whose sensor geometry is the canvas size.

Programs — a timeline of canvases
---------------------------------
A *program* (:class:`Program`) is the sequential form of the same idea and the model behind
the **Timeline** tab: an ordered list of :class:`ProgramSegment`, each one a whole
:class:`CanvasSpec` playing over its own slice of the program clock. A plain clip compiles
to a segment holding a single cell that covers the canvas, a mosaic/canvas block passes its
multi-cell spec through verbatim, and a title slide becomes a segment carrying only a text
item; running titles and overlay clips ride *across* every segment.
:func:`compile_program` builds one from a list of timeline items and
:func:`render_program_frame` renders it at a program time — through :func:`render_frame`, so
the Timeline's embedded preview and its video export are the same one render path.

Standard sizes and layout math
------------------------------
:data:`CANVAS_PRESETS` / :data:`CELL_PRESETS`, :func:`snap`, :func:`autotile` and
:func:`rescale_spec` are the pure geometry helpers behind the GUI's size pickers, its
snap-to-grid dragging, and its auto-tile button — deliberately here (tested, Qt-free)
rather than buried in a widget.

Per-cell rendering deliberately reuses the one canonical pipeline
(:func:`gottlux.core.render.render_frame` → *window → filter → accumulate → tone-map*),
so a cell on the canvas looks exactly like the same settings in the Live viewer; the only
additions here are the clock mapping, the ROI crop, and the nearest-neighbour placement
into the destination rect.

What the ``.raw`` export does — and honestly does not — carry
-------------------------------------------------------------
An event file carries **events**, not rendering. The exported ``.raw`` preserves each
event's polarity and applies the *geometric* (ROI crop, cell placement/scale) and
*temporal* (offset, time scale, loop) parts of the composition; the per-clip visualization
settings — colormap, tone-map, accumulation window — do **not** apply, because they only
exist at render time. To keep the full composition reproducible, the canvas JSON is written
as a sidecar next to the exported ``.raw``, so any consumer can re-render the styled view
from the spec.

Titles
------
A composition can also carry :class:`CanvasText` items — full-frame title *slides* and
running *overlay* lines. Text exists only at render time: :func:`render_frame` (and so
:func:`export_video`) draws the active items after every cell, while :func:`export_raw`
ignores them **by design** (a ``.raw`` carries events) and surfaces a one-line omission
note in its returned ``warnings``; the JSON sidecar still records the texts, so a titled
composition round-trips.

Pure NumPy + the existing render/tonemap/colormap machinery — no Qt anywhere in this module.
"""
from __future__ import annotations

import json
import math
import os
import warnings as _warnings
from dataclasses import dataclass, field

import numpy as np

from gottlux.core.render import render_frame as _render_cell_frame
from gottlux.io.recording import EventWindow

#: Canonical file suffix for a saved composition.
SPEC_SUFFIX = ".gottlux-canvas.json"

#: EVT2.1 CD words carry 11-bit x/y fields, so a composited ``.raw`` cannot exceed this
#: canvas dimension (coordinates must stay below 2048). Rendering has no such limit.
MAX_RAW_DIM = 2048

_SPEC_VERSION = 1


# ====================================================================================
# The spec — a JSON-serializable description of the whole composition
# ====================================================================================
@dataclass
class CanvasClip:
    """One placed cell: a source recording plus its geometry, clock, and look.

    Attributes
    ----------
    source : str
        Path (or label) identifying the recording; the key used to look it up in the
        mapping handed to :func:`render_frame` / :func:`export_raw`.
    rect : tuple[int, int, int, int]
        Destination cell ``(x, y, w, h)`` in canvas pixels.
    roi : tuple[int, int, int, int] | None
        Optional source crop ``(x0, y0, x1, y1)`` in sensor pixels (inclusive-exclusive);
        ``None`` uses the full sensor.
    t_offset_s : float
        Canvas time at which the clip starts playing.
    time_scale : float
        Clip-seconds advanced per canvas-second: ``1.0`` is real time, ``0.1`` plays the
        clip 10× slow (a 1 s clip occupies 10 s of canvas time).
    accumulation_s : float
        Integration window (exposure) for this cell's frames, in seconds.
    mode : str
        Accumulation mode (``count``, ``polarity``, ``polarity_ratio``, ``on``, ``off``,
        ``time_surface``, ``binary`` — see :mod:`gottlux.core.accumulate`).
    colormap : str
        Matplotlib colormap name for the cell.
    tonemap : str
        Tone-map expression (one of :data:`gottlux.core.tonemap.EXPRESSIONS`).
    gamma : float
        Exponent for the ``gamma`` tone-map expression.
    loop : bool
        When True the clip repeats from its start after its extent ends; when False the
        cell goes dark outside its extent.
    """
    source: str
    rect: tuple = (0, 0, 320, 320)
    roi: tuple | None = None
    t_offset_s: float = 0.0
    time_scale: float = 1.0
    accumulation_s: float = 0.02
    mode: str = "count"
    colormap: str = "inferno"
    tonemap: str = "sqrt"
    gamma: float = 0.5
    loop: bool = True

    def to_dict(self) -> dict:
        return {"source": self.source, "rect": [int(v) for v in self.rect],
                "roi": None if self.roi is None else [int(v) for v in self.roi],
                "t_offset_s": float(self.t_offset_s),
                "time_scale": float(self.time_scale),
                "accumulation_s": float(self.accumulation_s),
                "mode": self.mode, "colormap": self.colormap, "tonemap": self.tonemap,
                "gamma": float(self.gamma), "loop": bool(self.loop)}

    @classmethod
    def from_dict(cls, d: dict) -> CanvasClip:
        roi = d.get("roi")
        return cls(source=str(d["source"]),
                   rect=tuple(int(v) for v in d.get("rect", (0, 0, 320, 320))),
                   roi=None if roi is None else tuple(int(v) for v in roi),
                   t_offset_s=float(d.get("t_offset_s", 0.0)),
                   time_scale=float(d.get("time_scale", 1.0)),
                   accumulation_s=float(d.get("accumulation_s", 0.02)),
                   mode=str(d.get("mode", "count")),
                   colormap=str(d.get("colormap", "inferno")),
                   tonemap=str(d.get("tonemap", "sqrt")),
                   gamma=float(d.get("gamma", 0.5)),
                   loop=bool(d.get("loop", True)))


@dataclass
class CanvasText:
    """One rendered text item: a full-frame title *slide* or a running *overlay* line.

    Text exists only at render time — video export draws it, the ``.raw`` export carries
    events and cannot (the omission is surfaced through the export's ``warnings``). Items
    draw **after** the cells, in list order.

    Attributes
    ----------
    text : str
        The text; may span multiple lines (rendered centered).
    kind : str
        ``'slide'`` fills the whole frame with ``bg_color`` and centers the text on it;
        ``'overlay'`` draws the text over the cells at ``anchor``, on a translucent
        backing bar when ``bg_color`` is set.
    span : tuple | None
        Canvas-clock ``(t0, t1)`` seconds during which the item is visible
        (inclusive-exclusive); ``None`` means the whole duration.
    anchor : str
        Overlay placement: a compass edge ``'n'``/``'s'``/``'e'``/``'w'`` or ``'center'``.
    margin_px : int
        Distance from the anchored edge(s), in canvas pixels.
    font_size_px : int
        Text height in pixels (a real TrueType face — see :func:`_load_font`).
    color : tuple
        Text ink ``(r, g, b)``; defaults to the instrument theme's foreground.
    bg_color : tuple | None
        Slide background fill / overlay backing-bar color; ``None`` draws an overlay
        with no bar (a slide falls back to black).
    """
    text: str = ""
    kind: str = "overlay"
    span: tuple | None = None
    anchor: str = "s"
    margin_px: int = 24
    font_size_px: int = 32
    color: tuple = (215, 221, 231)
    bg_color: tuple | None = (14, 17, 22)

    def to_dict(self) -> dict:
        return {"text": self.text, "kind": self.kind,
                "span": None if self.span is None else [float(v) for v in self.span],
                "anchor": self.anchor, "margin_px": int(self.margin_px),
                "font_size_px": int(self.font_size_px),
                "color": [int(v) for v in self.color],
                "bg_color": None if self.bg_color is None
                else [int(v) for v in self.bg_color]}

    @classmethod
    def from_dict(cls, d: dict) -> CanvasText:
        span, bg = d.get("span"), d.get("bg_color")
        return cls(text=str(d.get("text", "")), kind=str(d.get("kind", "overlay")),
                   span=None if span is None else tuple(float(v) for v in span),
                   anchor=str(d.get("anchor", "s")),
                   margin_px=int(d.get("margin_px", 24)),
                   font_size_px=int(d.get("font_size_px", 32)),
                   color=tuple(int(v) for v in d.get("color", (215, 221, 231))),
                   bg_color=None if bg is None else tuple(int(v) for v in bg))


@dataclass
class CanvasSpec:
    """The whole composition: geometry, background, the placed clips, and the text items."""
    width: int = 640
    height: int = 480
    background: tuple = (0, 0, 0)
    clips: list = field(default_factory=list)
    texts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"version": _SPEC_VERSION, "width": int(self.width), "height": int(self.height),
                "background": [int(v) for v in self.background],
                "clips": [c.to_dict() for c in self.clips],
                "texts": [t.to_dict() for t in self.texts]}

    @classmethod
    def from_dict(cls, d: dict) -> CanvasSpec:
        return cls(width=int(d.get("width", 640)), height=int(d.get("height", 480)),
                   background=tuple(int(v) for v in d.get("background", (0, 0, 0))),
                   clips=[CanvasClip.from_dict(c) for c in d.get("clips", [])],
                   texts=[CanvasText.from_dict(t) for t in d.get("texts", [])])


def save_spec(spec: CanvasSpec, path: str) -> str:
    """Write *spec* to *path* as pretty-printed JSON. Returns the path written."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f, indent=2)
    return path


def load_spec(path: str) -> CanvasSpec:
    """Load a :class:`CanvasSpec` back from a ``.gottlux-canvas.json`` file."""
    with open(path, encoding="utf-8") as f:
        return CanvasSpec.from_dict(json.load(f))


def load_recordings(spec: CanvasSpec, progress=None) -> dict:
    """Load every distinct clip source via :func:`gottlux.load` → ``{source: Recording}``.

    Convenience for scripted use; the GUI keeps its own mapping so in-memory recordings
    (with no on-disk source) work too.
    """
    import gottlux as eb
    recs = {}
    for clip in spec.clips:
        if clip.source not in recs:
            recs[clip.source] = eb.load(clip.source, progress=progress)
    return recs


def _rec_for(recs, clip: CanvasClip, i: int):
    """Resolve the Recording for clip *i*: a mapping is keyed by ``clip.source``, a
    sequence is parallel to ``spec.clips``."""
    if hasattr(recs, "get"):
        return recs[clip.source]
    return recs[i]


# ====================================================================================
# Standard sizes and layout math — pure geometry, shared by every composition GUI
# ====================================================================================
#: The preset whose size is "whatever the first clip's sensor is".
NATIVE_CANVAS = "Native (first clip)"

#: The preset that takes the width/height the caller supplies verbatim.
CUSTOM_CANVAS = "Custom W×H"

#: Project-canvas size presets, in menu order — ``(label, size)`` where the size is a
#: ``(w, h)`` pair, ``None`` for :data:`NATIVE_CANVAS`, or ``"custom"`` for
#: :data:`CUSTOM_CANVAS`. Resolve one with :func:`canvas_preset_size`.
CANVAS_PRESETS = (
    (NATIVE_CANVAS, None),
    ("640 × 640", (640, 640)),
    ("1280 × 720", (1280, 720)),
    ("1920 × 1080", (1920, 1080)),
    ("1024 × 1024", (1024, 1024)),
    (CUSTOM_CANVAS, "custom"),
)

#: Cell size presets as exact ``(width, height)`` fractions of the canvas — 'horizontal'
#: and 'vertical' name the direction the canvas is *split*, so "1/2 horizontal" is a
#: half-width, full-height cell (a left/right split) and "1/2 vertical" a full-width,
#: half-height one (a top/bottom split). Resolve one with :func:`cell_preset_rect`.
CELL_PRESETS = (
    ("Full", (1.0, 1.0)),
    ("1/2 horizontal", (1.0 / 2, 1.0)),
    ("1/2 vertical", (1.0, 1.0 / 2)),
    ("1/3", (1.0 / 3, 1.0 / 3)),
    ("1/4 quadrant", (1.0 / 2, 1.0 / 2)),
    ("2/3", (2.0 / 3, 2.0 / 3)),
)

#: Drag snapping divides each canvas edge into this many steps (a 1/12 grid).
SNAP_DIVISIONS = 12


def canvas_preset_size(label, native_wh=None, custom_wh=None):
    """Resolve a :data:`CANVAS_PRESETS` label to a ``(w, h)`` pair.

    :data:`NATIVE_CANVAS` yields *native_wh* and :data:`CUSTOM_CANVAS` yields *custom_wh*
    (both fall back to ``(640, 480)`` when the caller has nothing to offer); an unknown
    label is treated as native.
    """
    fallback = (640, 480)
    size = dict(CANVAS_PRESETS).get(str(label), None)
    if size == "custom":
        size = custom_wh
    elif size is None:
        size = native_wh
    return tuple(max(int(v), 1) for v in (size or fallback))


def cell_preset_rect(label, canvas_wh, origin=(0, 0)):
    """The cell rect ``(x, y, w, h)`` a :data:`CELL_PRESETS` label gives on *canvas_wh*.

    The size is an exact fraction of the canvas; *origin* keeps the cell where it already
    sits, shifted back only as far as needed to keep it fully on the canvas. An unknown
    label falls back to the full canvas.
    """
    W, H = (max(int(v), 1) for v in canvas_wh)
    fw, fh = dict(CELL_PRESETS).get(str(label), (1.0, 1.0))
    w = max(1, min(int(round(W * float(fw))), W))
    h = max(1, min(int(round(H * float(fh))), H))
    x = max(0, min(int(origin[0]), W - w))
    y = max(0, min(int(origin[1]), H - h))
    return (x, y, w, h)


def snap(v, extent, divisions=SNAP_DIVISIONS):
    """Snap the pixel coordinate *v* to the nearest gridline of a *divisions*-step grid
    spanning ``[0, extent]``. ``divisions <= 0`` snaps nothing (returns ``int(round(v))``)."""
    v = float(v)
    divisions, extent = int(divisions), float(extent)
    if divisions <= 0 or extent <= 0:
        return int(round(v))
    step = extent / divisions
    return int(round(round(v / step) * step))


def snap_rect(rect, canvas_wh, divisions=SNAP_DIVISIONS):
    """Snap a whole cell rect to the grid — position *and* size, the size kept to at
    least one grid step so a snapped cell can never collapse to nothing."""
    W, H = (max(int(v), 1) for v in canvas_wh)
    x, y, w, h = (float(v) for v in rect)
    if int(divisions) <= 0:
        return tuple(int(round(v)) for v in (x, y, w, h))
    return (snap(x, W, divisions), snap(y, H, divisions),
            max(snap(w, W, divisions), int(round(W / int(divisions)))),
            max(snap(h, H, divisions), int(round(H / int(divisions)))))


def autotile(n, canvas_wh):
    """Lay *n* cells into the best-fit grid over *canvas_wh* → a list of ``(x, y, w, h)``.

    The grid is the near-square ``ceil(sqrt(n))`` columns by as many rows as that needs;
    boundaries are computed by integer division from the canvas edges, so the cells tile
    the canvas exactly — no gaps, no overlaps, no rounding drift on the last column/row.
    """
    W, H = (max(int(v), 1) for v in canvas_wh)
    n = max(int(n), 0)
    if n == 0:
        return []
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    rects = []
    for i in range(n):
        r, c = divmod(i, cols)
        x0, x1 = c * W // cols, (c + 1) * W // cols
        y0, y1 = r * H // rows, (r + 1) * H // rows
        rects.append((x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)))
    return rects


def rescale_spec(spec: CanvasSpec, canvas_wh) -> CanvasSpec:
    """A copy of *spec* on a new canvas size, every cell rect scaled by the same factors.

    What the GUI's project-canvas picker does: change the stage, keep the arrangement.
    """
    W, H = (max(int(v), 1) for v in canvas_wh)
    out = CanvasSpec.from_dict(spec.to_dict())
    sx, sy = W / max(int(spec.width), 1), H / max(int(spec.height), 1)
    out.width, out.height = W, H
    for clip in out.clips:
        x, y, w, h = clip.rect
        clip.rect = (int(round(x * sx)), int(round(y * sy)),
                     max(1, int(round(w * sx))), max(1, int(round(h * sy))))
    return out


# ====================================================================================
# The clock mapping — canvas time → clip time
# ====================================================================================
def clip_extent_s(clip: CanvasClip, rec) -> tuple:
    """The canvas-time span ``(t0, t1)`` one pass of *clip* occupies (loop ignored)."""
    dur = float(rec.duration_s)
    span = dur / clip.time_scale if clip.time_scale > 0 else 0.0
    return (clip.t_offset_s, clip.t_offset_s + span)


def clip_time(clip: CanvasClip, rec, t_canvas: float):
    """Map a canvas time to the clip's recording time, or ``None`` when the cell is dark.

    ``t_clip = (t_canvas − t_offset_s) · time_scale``. A looping clip wraps modulo its
    duration once the offset has passed; a non-looping clip is only live inside its one
    pass. The returned value is absolute recording time (seconds), ready for the render
    pipeline's windowing.
    """
    dur = float(rec.duration_s)
    if dur <= 0 or clip.time_scale <= 0:
        return None
    t_rel = (float(t_canvas) - clip.t_offset_s) * clip.time_scale
    if t_rel < 0:
        return None                                    # before the clip first starts
    if clip.loop:
        t_rel = t_rel % dur
    elif t_rel > dur:
        return None                                    # past the single pass
    return rec.t_start_s + t_rel


def canvas_duration(spec: CanvasSpec, recs) -> float:
    """The canvas timeline extent: the latest end among all clips' single passes.

    Looping clips simply repeat inside this span, so the timeline is the *max clip
    extent*, matching what a shared transport should scrub over.
    """
    end = 0.0
    for i, clip in enumerate(spec.clips):
        end = max(end, clip_extent_s(clip, _rec_for(recs, clip, i))[1])
    return end


# ====================================================================================
# Text items — visibility, the font, and the PIL draw
# ====================================================================================
#: Alpha (0–255) of the translucent backing bar behind an anchored overlay line.
_TEXT_BAR_ALPHA = 150

_font_warned = False


def text_active(txt: CanvasText, t_canvas: float) -> bool:
    """Whether *txt* is visible at *t_canvas* (``span=None`` → the whole duration)."""
    if txt.span is None:
        return True
    return float(txt.span[0]) <= float(t_canvas) < float(txt.span[1])


def texts_extent_s(spec: CanvasSpec) -> float:
    """The latest end among finite text spans (``0.0`` when none).

    Video timelines extend to cover it, so a title slide past the last clip still plays;
    :func:`canvas_duration` deliberately ignores texts, keeping the event (``.raw``)
    export untouched by render-only items.
    """
    end = 0.0
    for txt in spec.texts:
        if txt.span is not None:
            end = max(end, float(txt.span[1]))
    return end


def text_omission_note(n: int) -> str:
    """The one-line note event exports surface when render-only text items are present."""
    return f"{int(n)} text item(s) omitted — text renders in video export only"


def _load_font(size_px: int):
    """A DejaVuSans TrueType face at *size_px*, resolved through matplotlib's bundled
    fonts (matplotlib is a core dependency, so the face exists wherever gottlux runs);
    PIL's fixed-size bitmap font — with a one-time size warning — only if that fails."""
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), int(size_px))
    except Exception:
        global _font_warned
        if not _font_warned:
            _font_warned = True
            _warnings.warn("no TrueType font could be resolved — text renders at the "
                           "bitmap fallback's fixed size, ignoring font_size_px",
                           RuntimeWarning, stacklevel=2)
        return ImageFont.load_default()


def draw_texts(frame: np.ndarray, texts, t_canvas: float) -> np.ndarray:
    """Draw every text item active at *t_canvas* onto *frame* → ``(H, W, 3)`` uint8.

    A *slide* repaints the whole frame (its ``bg_color`` fill, the text centered on it);
    an *overlay* alpha-composites its anchored line — and its optional translucent
    backing bar — over whatever the cells rendered. Items draw in list order, after the
    cells (callers pass the already-composited frame).
    """
    from PIL import Image, ImageDraw
    active = [t for t in texts if text_active(t, t_canvas)]
    if not active:
        return frame
    H, W = frame.shape[:2]
    im = Image.fromarray(np.ascontiguousarray(frame, np.uint8))
    for txt in active:
        font = _load_font(txt.font_size_px)
        ink = tuple(int(v) for v in txt.color)
        if txt.kind == "slide":
            im = Image.new("RGB", (W, H),
                           tuple(int(v) for v in (txt.bg_color or (0, 0, 0))))
            d = ImageDraw.Draw(im)
            box = d.multiline_textbbox((0, 0), txt.text, font=font, align="center")
            d.multiline_text(((W - box[2] - box[0]) / 2, (H - box[3] - box[1]) / 2),
                             txt.text, fill=ink, font=font, align="center")
            continue
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        box = d.multiline_textbbox((0, 0), txt.text, font=font, align="center")
        tw, th = box[2] - box[0], box[3] - box[1]
        m = int(txt.margin_px)
        x = {"w": m, "e": W - tw - m}.get(txt.anchor, (W - tw) // 2)
        y = {"n": m, "s": H - th - m}.get(txt.anchor, (H - th) // 2)
        if txt.bg_color is not None:
            pad = max(4, int(txt.font_size_px) // 4)
            d.rectangle((x - pad, y - pad, x + tw + pad, y + th + pad),
                        fill=tuple(int(v) for v in txt.bg_color) + (_TEXT_BAR_ALPHA,))
        d.multiline_text((x - box[0], y - box[1]), txt.text, fill=ink + (255,),
                         font=font, align="center")
        im = Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")
    return np.asarray(im, np.uint8)


# ====================================================================================
# Rendering — one cell, then the whole canvas
# ====================================================================================
class _RoiFilter:
    """A minimal live-filter shim (the ``filters.apply(win)`` protocol the canonical
    render pipeline already accepts) that keeps only events inside an ROI — so the ROI
    crop rides through :func:`gottlux.core.render.render_frame` unchanged."""

    def __init__(self, roi):
        self.roi = roi

    def apply(self, win: EventWindow) -> EventWindow:
        x0, y0, x1, y1 = self.roi
        m = (win.x >= x0) & (win.x < x1) & (win.y >= y0) & (win.y < y1)
        return EventWindow(win.x[m], win.y[m], win.p[m], win.t[m],
                           win.width, win.height, win.t0_us)


def _norm_roi(clip: CanvasClip, rec):
    """Clamp the clip's ROI to the sensor; ``None`` (or a degenerate box) → full frame."""
    if clip.roi is None:
        return (0, 0, int(rec.width), int(rec.height))
    x0, y0, x1, y1 = (int(v) for v in clip.roi)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(int(rec.width), x1), min(int(rec.height), y1)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, int(rec.width), int(rec.height))
    return (x0, y0, x1, y1)


def render_cell(clip: CanvasClip, rec, t_canvas: float):
    """Render one clip's cell at a canvas time → ``(h, w, 3)`` uint8, or ``None`` (dark).

    Runs the canonical pipeline (:func:`gottlux.core.render.render_frame`) at the mapped
    clip time with the clip's own accumulation/tone-map settings, colourizes with the
    clip's colormap, crops to the source ROI, and nearest-resizes into the destination
    rect — so events stay crisp blocks rather than smearing.
    """
    from gottlux.viz import video as _vid
    t_rec = clip_time(clip, rec, t_canvas)
    w, h = int(clip.rect[2]), int(clip.rect[3])
    if t_rec is None or w <= 0 or h <= 0:
        return None
    roi = _norm_roi(clip, rec)
    filt = None if roi == (0, 0, int(rec.width), int(rec.height)) else _RoiFilter(roi)
    disp, levels, _vmax, _win = _render_cell_frame(
        rec, t_rec, float(clip.accumulation_s), mode=clip.mode, expr=clip.tonemap,
        gamma=float(clip.gamma), filters=filt)
    rgb = _vid.disp_to_rgb(disp, levels, clip.colormap)
    x0, y0, x1, y1 = roi
    rgb = rgb[y0:y1, x0:x1]
    return _vid.resize_rgb(rgb, (w, h))


def render_frame(spec: CanvasSpec, recs, t_canvas: float) -> np.ndarray:
    """Composite every live cell onto the canvas at *t_canvas* → ``(H, W, 3)`` uint8.

    *recs* is a mapping ``{source: Recording}`` (or a list parallel to ``spec.clips``).
    Cells are painted in list order (later clips draw on top), clipped to the canvas
    bounds; a cell outside its time extent leaves the background showing. Text items
    active at *t_canvas* draw last, over every cell (:func:`draw_texts`).
    """
    H, W = int(spec.height), int(spec.width)
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = np.asarray(spec.background, np.uint8)
    for i, clip in enumerate(spec.clips):
        rec = _rec_for(recs, clip, i)
        cell = render_cell(clip, rec, t_canvas)
        if cell is None:
            continue
        x, y = int(clip.rect[0]), int(clip.rect[1])
        h, w = cell.shape[:2]
        x0c, y0c = max(x, 0), max(y, 0)
        x1c, y1c = min(x + w, W), min(y + h, H)
        if x1c <= x0c or y1c <= y0c:
            continue                                   # entirely off-canvas
        canvas[y0c:y1c, x0c:x1c] = cell[y0c - y:y1c - y, x0c - x:x1c - x]
    if spec.texts:
        canvas = draw_texts(canvas, spec.texts, t_canvas)
    return canvas


# ====================================================================================
# Programs — a sequence of canvases on one clock (the Timeline tab's model)
# ====================================================================================
#: The shortest segment the compiler will emit; keeps a zero-length item from creating a
#: segment no cursor can ever land on.
MIN_SEGMENT_S = 1e-3


@dataclass
class ProgramSegment:
    """One sequential slice of a :class:`Program`: a whole composition and its span.

    ``spec``/``recs`` are exactly what :func:`render_frame` takes, evaluated at
    *segment-local* time (``t_program − t0_s``). ``n_own`` is how many of ``spec.clips``
    are the segment's own cells — any beyond that are the program's overlay clips,
    appended by :func:`compile_program` with their offsets re-based onto this segment, so
    event exports can flatten the program without emitting an overlay once per segment.
    """
    spec: CanvasSpec = field(default_factory=CanvasSpec)
    recs: dict = field(default_factory=dict)
    t0_s: float = 0.0
    duration_s: float = 0.0
    kind: str = "clip"                 # 'clip' | 'block' | 'title' | 'gap'
    label: str = ""
    n_own: int = 0
    key: object = None                 # the source item's ``key``, so a GUI can map back

    @property
    def t1_s(self) -> float:
        return self.t0_s + self.duration_s


@dataclass
class Program:
    """An ordered run of :class:`ProgramSegment` plus the items that span all of them.

    ``overlays`` are :class:`CanvasText` items on the **program** clock (running titles);
    ``overlay_clips``/``overlay_recs`` are the overlay lane's cells, also on the program
    clock — :func:`compile_program` copies them into every segment for rendering and keeps
    the originals here so an event export can emit them once (or leave them out).
    """
    width: int = 640
    height: int = 480
    segments: list = field(default_factory=list)
    overlays: list = field(default_factory=list)
    overlay_clips: list = field(default_factory=list)
    overlay_recs: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """The whole program's extent — the end of its last segment (``0.0`` when empty)."""
        return self.segments[-1].t1_s if self.segments else 0.0

    def segment_at(self, t_program: float):
        """The segment covering *t_program* (the last one past the end); ``None`` if empty."""
        if not self.segments:
            return None
        t = float(t_program)
        for seg in self.segments:
            if t < seg.t1_s:
                return seg
        return self.segments[-1]


def _clip_cell(item: dict, canvas_wh) -> CanvasClip:
    """The single full-canvas cell a plain timeline clip renders through.

    The item's ``cell`` (a :class:`CanvasClip` carrying just that clip's *look and clock*
    — colormap, tone-map, accumulation, mode, time scale, loop) is copied, then the parts
    the timeline owns are written over it: the rect is the whole canvas, the ROI is the
    clip's crop, and the In point becomes a clock offset (``t_offset = −t0 / time_scale``)
    so segment-local time zero lands exactly on the trimmed clip's first frame — no
    sub-recording has to be materialized to honour a trim.
    """
    tmpl = item.get("cell")
    cell = CanvasClip.from_dict(tmpl.to_dict()) if tmpl is not None else CanvasClip(source="")
    cell.rect = (0, 0, int(canvas_wh[0]), int(canvas_wh[1]))
    cell.roi = item.get("roi")
    scale = cell.time_scale if cell.time_scale > 0 else 1.0
    cell.t_offset_s = -float(item.get("t0", 0.0)) / scale
    return cell


def item_duration_s(item: dict) -> float:
    """How much program time one timeline item occupies (its own ``dur`` wins if given)."""
    kind = item.get("kind", "clip")
    if kind == "clip":
        rec = item["rec"]
        t0 = float(item.get("t0", 0.0))
        t1 = float(item["t1"]) if item.get("t1") is not None else float(rec.duration_s)
        tmpl = item.get("cell")
        scale = tmpl.time_scale if (tmpl is not None and tmpl.time_scale > 0) else 1.0
        return max(t1 - t0, 0.0) / scale
    if kind == "block":
        if item.get("dur"):
            return float(item["dur"])
        spec, recs = item["spec"], item.get("recs") or {}
        return max(canvas_duration(spec, recs), texts_extent_s(spec))
    return float(item.get("dur", 0.0))


def program_canvas_size(items, canvas_wh=None) -> tuple:
    """The canvas a program compiles onto: *canvas_wh* if given, else the first clip's
    sensor (or the first block's canvas) — 'Native' — else ``(640, 480)``."""
    if canvas_wh is not None:
        return tuple(max(int(v), 1) for v in canvas_wh)
    for item in items:
        kind = item.get("kind", "clip")
        if kind == "clip" and item.get("rec") is not None:
            return (int(item["rec"].width), int(item["rec"].height))
        if kind == "block" and item.get("spec") is not None:
            return (int(item["spec"].width), int(item["spec"].height))
    return (640, 480)


def compile_program(items, canvas_wh=None, gap_s=0.0) -> Program:
    """Compile an ordered list of timeline *items* into a :class:`Program`.

    Each item is a plain dict; ``kind`` selects the shape:

    ``'clip'``
        ``rec`` (a :class:`~gottlux.io.recording.Recording`), ``t0``/``t1`` (the In/Out
        trim, seconds), ``roi`` (the crop, or ``None``) and ``cell`` (a
        :class:`CanvasClip` carrying this clip's own visualization settings). Compiles to
        a segment holding **one cell covering the whole canvas** — so a plain clip and a
        mosaic are the same kind of thing to the renderer.
    ``'block'``
        ``spec`` (a :class:`CanvasSpec`) and ``recs`` — a mosaic/canvas block, whose
        multi-cell composition is used **verbatim** (rescaled onto the program canvas).
    ``'title'``
        ``text`` (a :class:`CanvasText`). A slide compiles to a segment of its own
        occupying ``dur`` seconds; a running title (``overlay=True``) is hoisted onto the
        program clock and drawn across every segment instead.

    ``overlay=True`` on a clip likewise lifts it out of the sequence and onto the overlay
    lane: its cell is copied into every segment with its offset re-based, so it plays over
    the whole program on the program clock. *gap_s* inserts a blank segment between
    consecutive sequence items, matching the gap the ``.raw`` stitch writes.

    An optional ``key`` on an item is copied onto the segment it produced, so a caller can
    map a compiled segment back to the row it came from without re-deriving the layout.
    """
    W, H = program_canvas_size(items, canvas_wh)
    prog = Program(width=W, height=H)

    # ----- the overlay lane: items that span the whole program, not a slot in it -----
    for i, item in enumerate(items):
        if not item.get("overlay"):
            continue
        kind = item.get("kind", "clip")
        if kind == "title" and item.get("text") is not None:
            prog.overlays.append(CanvasText.from_dict(item["text"].to_dict()))
        elif kind == "clip" and item.get("rec") is not None:
            key = f"overlay{i}:{item.get('name', 'clip')}"
            cell = _clip_cell(item, (W, H))
            cell.source = key
            prog.overlay_clips.append(cell)
            prog.overlay_recs[key] = item["rec"]

    # ----- the sequence lane: one segment per item, laid end to end -----
    seq = [it for it in items if not it.get("overlay")]
    cursor = 0.0
    for i, item in enumerate(seq):
        kind = item.get("kind", "clip")
        dur = item_duration_s(item)
        if dur <= 0.0:
            continue
        dur = max(dur, MIN_SEGMENT_S)
        if kind == "block" and item.get("spec") is not None:
            spec = rescale_spec(item["spec"], (W, H))
            recs = dict(item.get("recs") or {})
        elif kind == "title" and item.get("text") is not None:
            text = CanvasText.from_dict(item["text"].to_dict())
            text.span = (0.0, dur)                 # active for the whole slide segment
            spec, recs = CanvasSpec(width=W, height=H, texts=[text]), {}
        elif kind == "clip" and item.get("rec") is not None:
            key = f"{i}:{item.get('name', 'clip')}"
            cell = _clip_cell(item, (W, H))
            cell.source = key
            spec, recs = CanvasSpec(width=W, height=H, clips=[cell]), {key: item["rec"]}
        else:
            continue
        prog.segments.append(ProgramSegment(
            spec=spec, recs=recs, t0_s=cursor, duration_s=dur, kind=kind,
            label=str(item.get("name", "")), n_own=len(spec.clips),
            key=item.get("key")))
        cursor += dur
        if gap_s > 0 and i < len(seq) - 1:
            prog.segments.append(ProgramSegment(
                spec=CanvasSpec(width=W, height=H), t0_s=cursor,
                duration_s=float(gap_s), kind="gap", label="gap"))
            cursor += float(gap_s)

    # ----- the overlay cells ride every segment, re-based onto the segment's local clock -----
    for seg in prog.segments:
        for cell in prog.overlay_clips:
            copy = CanvasClip.from_dict(cell.to_dict())
            copy.t_offset_s = cell.t_offset_s - seg.t0_s
            seg.spec.clips.append(copy)
        seg.recs.update(prog.overlay_recs)
    return prog


def render_program_frame(program: Program, t_program: float) -> np.ndarray:
    """One ``(H, W, 3)`` uint8 frame of *program* at a program time.

    Finds the live segment, renders it through the one canonical :func:`render_frame` at
    segment-local time, then draws the program-wide running titles over it on the program
    clock. This is the single path the Timeline's embedded preview and its video export
    both go through.
    """
    seg = program.segment_at(t_program)
    if seg is None:
        frame = np.zeros((int(program.height), int(program.width), 3), np.uint8)
    else:
        frame = render_frame(seg.spec, seg.recs, float(t_program) - seg.t0_s)
    if program.overlays:
        frame = draw_texts(frame, program.overlays, float(t_program))
    return frame


def program_spec(program: Program) -> tuple:
    """Flatten *program* into one ``(spec, recs)`` pair for the event (``.raw``) export.

    Every segment's **own** cells are re-based onto the program clock
    (``t_offset_s += t0_s``) and their source keys namespaced per segment, so the whole
    sequence becomes a single composition :func:`export_raw` can re-encode. Overlay-lane
    cells are left out — the sequential lane is what an event export carries — and text
    items are dropped, exactly as :func:`export_raw` drops them anyway.
    """
    spec = CanvasSpec(width=int(program.width), height=int(program.height))
    recs = {}
    for i, seg in enumerate(program.segments):
        for clip in seg.spec.clips[:seg.n_own]:
            copy = CanvasClip.from_dict(clip.to_dict())
            copy.source = f"{i}/{clip.source}"
            copy.t_offset_s = clip.t_offset_s + seg.t0_s
            copy.loop = False                      # a program slot plays exactly once
            spec.clips.append(copy)
            recs[copy.source] = seg.recs[clip.source]
    return spec, recs


def export_program_video(program: Program, out_mp4: str, fps: float = 30.0,
                         duration_s: float | None = None, progress=None):
    """Render the whole program to an MP4 at *fps* — segments, overlays, and titles.

    Frame *i* samples program time ``i / fps`` through :func:`render_program_frame`, so
    the file is frame-for-frame what the Timeline's preview shows. Fails soft like
    :func:`export_video`: returns the written path, or ``None`` when the muxer is missing.
    """
    from gottlux.viz.video import VideoWriter
    duration = program.duration_s if duration_s is None else float(duration_s)
    n = max(1, int(round(duration * float(fps))))
    w = VideoWriter(out_mp4, fps=fps)
    if not w.ok:
        w.close()
        return None
    for i in range(n):
        w.append(render_program_frame(program, i / float(fps)))
        if progress:
            try:
                progress((i + 1) / n)
            except Exception:
                pass
    return w.close()


# ====================================================================================
# Exports — MP4 (rendered) and .raw (re-encoded events)
# ====================================================================================
def export_video(spec: CanvasSpec, recs, out_mp4: str, fps: float = 30.0,
                 duration_s: float | None = None, progress=None):
    """Render the composition to an MP4 at *fps* over ``[0, duration_s]``.

    *duration_s* defaults to :func:`canvas_duration` extended over the text spans
    (:func:`texts_extent_s`), so a title slide past the last clip still plays. Frame *i*
    samples canvas time ``i / fps``. Uses :class:`gottlux.viz.video.VideoWriter`
    (imageio-ffmpeg) and fails soft: returns the written path, or ``None`` when the
    muxer is unavailable.
    """
    from gottlux.viz.video import VideoWriter
    duration = (max(canvas_duration(spec, recs), texts_extent_s(spec))
                if duration_s is None else float(duration_s))
    n = max(1, int(round(duration * float(fps))))
    w = VideoWriter(out_mp4, fps=fps)
    if not w.ok:
        w.close()
        return None
    for i in range(n):
        w.append(render_frame(spec, recs, i / float(fps)))
        if progress:
            try:
                progress((i + 1) / n)
            except Exception:
                pass
    return w.close()


def export_raw(spec: CanvasSpec, recs, out_raw: str, duration_s: float | None = None,
               progress=None, block: int = 500_000) -> dict:
    """Re-encode the composition as one EVT2.1 ``.raw`` whose geometry is the canvas size.

    Per clip (and per loop pass inside ``[0, duration_s]``): the covered span of events is
    taken, the ROI crop applied, ``x``/``y`` mapped into the destination rect by integer
    nearest scaling, and timestamps placed on the canvas clock
    (``t_canvas = t_offset + t_clip / time_scale``, in µs). The clips' streams are then
    merged, time-sorted, and encoded block-by-block through the streamed EVT2.1 writer
    (:func:`gottlux.io.writer.write_raw`). Polarity is preserved event-for-event.

    Per-clip *visualization* settings (colormap, tone-map, accumulation) do **not** apply
    here — a ``.raw`` carries events, not rendering — so the canvas spec is written as a
    JSON sidecar next to the output, keeping the styled composition reproducible. Text
    items are ignored the same way and for the same reason: when any exist, the returned
    ``warnings`` carries the one-line :func:`text_omission_note` (the GUI shows it), and
    the sidecar still records them so a titled composition round-trips.

    Memory: each clip's span is extracted in bounded blocks off the memmap, but the merged
    composition is held in RAM once for the global time sort; the encode itself streams.
    Returns ``{"path", "sidecar", "n_events", "duration_s", "width", "height",
    "warnings"}``.
    """
    from gottlux.io import writer as _writer
    W, H = int(spec.width), int(spec.height)
    if W > MAX_RAW_DIM or H > MAX_RAW_DIM:
        raise ValueError(f"canvas {W}×{H} exceeds the EVT2.1 coordinate range "
                         f"(max {MAX_RAW_DIM}×{MAX_RAW_DIM} for a .raw export)")
    duration = canvas_duration(spec, recs) if duration_s is None else float(duration_s)

    # Enumerate every (clip, loop-pass) whose canvas span intersects [0, duration].
    passes = []                     # (rec, clip, pass_offset_s, lo_clip_s, hi_clip_s)
    for i, clip in enumerate(spec.clips):
        rec = _rec_for(recs, clip, i)
        dur = float(rec.duration_s)
        rx, ry, rw, rh = (int(v) for v in clip.rect)
        if dur <= 0 or clip.time_scale <= 0 or rw <= 0 or rh <= 0:
            continue
        span = dur / clip.time_scale
        n_pass = 1
        if clip.loop and span > 0:
            n_pass = max(1, int(math.ceil((duration - clip.t_offset_s) / span)))
        for k in range(n_pass):
            off = clip.t_offset_s + k * span
            lo = max(0.0, (0.0 - off) * clip.time_scale)
            hi = min(dur, (duration - off) * clip.time_scale)
            if hi > lo:
                passes.append((rec, clip, off, lo, hi))

    # Each pass keeps events in [lo, hi) clip-seconds — except that a pass reaching the
    # clip's end closes the interval, so a fully-covered clip contributes its final event.
    spans = []
    for rec, _clip, _off, lo, hi in passes:
        i0 = rec.index_at(rec.t_start_s + lo, "left")
        i1 = rec.n if hi >= float(rec.duration_s) else rec.index_at(rec.t_start_s + hi, "left")
        spans.append((i0, i1))
    total = max(sum(i1 - i0 for i0, i1 in spans), 1)
    done = 0
    xs_all, ys_all, ps_all, ts_all = [], [], [], []
    for (rec, clip, off, _lo, _hi), (i0, i1) in zip(passes, spans):
        rx, ry, rw, rh = (int(v) for v in clip.rect)
        x0r, y0r, x1r, y1r = _norm_roi(clip, rec)
        roi_w, roi_h = x1r - x0r, y1r - y0r
        t0_us = np.int64(round(rec.t_start_s * 1e6))
        for s in range(i0, i1, block):
            e = min(s + block, i1)
            xs = np.asarray(rec.x[s:e]).astype(np.int64)
            ys = np.asarray(rec.y[s:e]).astype(np.int64)
            ps = np.asarray(rec.p[s:e])
            ts = np.asarray(rec.t[s:e]).astype(np.int64)
            m = (xs >= x0r) & (xs < x1r) & (ys >= y0r) & (ys < y1r)
            xs, ys, ps, ts = xs[m], ys[m], ps[m], ts[m]
            # integer nearest mapping into the destination rect, then canvas clipping
            xs = rx + (xs - x0r) * rw // roi_w
            ys = ry + (ys - y0r) * rh // roi_h
            keep = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            xs, ys, ps, ts = xs[keep], ys[keep], ps[keep], ts[keep]
            tc = np.round(off * 1e6 + (ts - t0_us) / clip.time_scale).astype(np.int64)
            xs_all.append(xs.astype(np.uint16)); ys_all.append(ys.astype(np.uint16))
            ps_all.append(ps.astype(np.uint8)); ts_all.append(np.clip(tc, 0, None))
            done += (e - s)
            if progress:
                try:
                    progress(min(done / total, 1.0))
                except Exception:
                    pass

    if xs_all:
        X = np.concatenate(xs_all); Y = np.concatenate(ys_all)
        P = np.concatenate(ps_all); T = np.concatenate(ts_all)
    else:
        X = np.zeros(0, np.uint16); Y = np.zeros(0, np.uint16)
        P = np.zeros(0, np.uint8); T = np.zeros(0, np.int64)
    n = _writer.write_raw(out_raw, X, Y, P, T, width=W, height=H)

    sidecar = os.path.splitext(out_raw)[0] + SPEC_SUFFIX
    save_spec(spec, sidecar)
    return {"path": out_raw, "sidecar": sidecar, "n_events": int(n),
            "duration_s": duration, "width": W, "height": H,
            "warnings": [text_omission_note(len(spec.texts))] if spec.texts else []}
