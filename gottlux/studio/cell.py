"""
cell.py — a Studio look as a canvas cell: the same picture, on the canvas clock.

The Canvas composer and the Timeline place recordings as *cells* on one canvas
(:mod:`gottlux.core.canvas`). A cell normally shows the classic event frame (accumulate →
tone-map → colormap). With ``CanvasClip.look`` set to a Studio look — trails, heat, time
surface, light painting, space-time cloud, stroboscope, waterfall, flicker hue, radar — the
cell is drawn by that look instead, so one canvas can hold several visualizations of the
same moment side by side and export them as one video.

How a look maps onto a cell
---------------------------
A Studio shot renders frame *k* at recording time ``tau_k``; a cell is asked for the picture
at an arbitrary recording time ``tau`` (the canvas clock mapped through the cell's offset,
time scale and loop). :class:`LookCell` therefore treats the cell's recording as one long
shot spanning the whole recording and renders the look at ``tau`` directly
(:meth:`gottlux.studio.render.ShotRenderer.frame_at`). Looks that also use the frame index —
the space-time camera orbit and the stroboscope's phase bin — take it from how far ``tau``
is through the recording, so an orbit turns once over the clip and the strobe plays its
*Loops* over one pass, exactly as in a Studio shot of the whole clip.

Events are held in **chunks** (:data:`EVENT_BUDGET` events at most, never shorter than
:data:`MIN_CHUNK_S`): a short clip is one chunk, identical to the Studio; a long one is
sliced, each slice carrying its look-back. Exposure stays **locked across chunks**: the
scale a look measures (and the stroboscope's f0, the radar's period and column sign) is
taken from the clip's first chunk and reused for every other, so brightness does not jump
at a chunk boundary and the preview and the export agree whichever moment is shown first.
Light painting without a fade paints from the start of the chunk it is in.

The picture keeps the look's aspect ratio inside the cell (letterboxed with the palette
background) — a radar dial stays round and a space-time box keeps its proportions — unless
the cell asks to be stretched (``CanvasClip.fit = "stretch"``).

Cells are cached (:func:`cell_for`) by recording and settings, so playback re-renders
without re-slicing; a changed setting builds a new cell. Thread-safe: an export worker and
the GUI preview can render the same cell.
"""
from __future__ import annotations

import json
import threading
import weakref
from collections import OrderedDict
from dataclasses import replace

import numpy as np

from gottlux.studio.looks import get_look
from gottlux.studio.render import ShotRenderer, clamp_crop, prepare_events
from gottlux.studio.shot import Shot

#: The most events one cell holds at a time (≈ 34 bytes each in RAM).
EVENT_BUDGET = 2_000_000

#: A chunk never covers less recording time than this.
MIN_CHUNK_S = 0.25

#: The nominal frame rate a cell's frame index is counted at (orbit / strobe progress).
_INDEX_FPS = 25.0

#: State a look measures once and every later chunk of the clip reuses.
_CARRY = ("scale", "hscale", "f0", "period", "mirror", "direction", "t_ref")

_CACHE_MAX = 24
_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()


class LookCell:
    """One cell's look over one recording: ``frame(tau)`` → ``(h, w, 3)`` uint8."""

    def __init__(self, rec, look: str, params=None, palette: str = "fire-ice", roi=None,
                 denoise: bool = False, bloom: float = 0.0, size=(320, 240),
                 stretch: bool = False):
        self._rec = weakref.ref(rec)
        self.stretch = bool(stretch)
        self.look = get_look(look)
        self.P = self.look.resolve(params or {})
        self.size = (max(2, int(size[0])), max(2, int(size[1])))
        self.t_lo, self.t_hi = float(rec.t_start_s), float(rec.t_stop_s)
        dur = max(self.t_hi - self.t_lo, 1e-6)
        crop = clamp_crop(roi, rec.width, rec.height)
        x0, y0, x1, y1 = crop or (0, 0, rec.width, rec.height)
        cw, ch = self.look.canvas_size(x1 - x0, y1 - y0, self.P)
        # the look's own aspect, fitted inside the cell
        w, h = self.size
        fit_w = int(min(w, round(h * cw / max(ch, 1e-9))))
        self._shot = Shot(look=self.look.name, params=dict(self.P), crop=crop, speed=1.0,
                          fps=_INDEX_FPS, width=max(2, fit_w), palette=palette,
                          denoise=bool(denoise), bloom=float(bloom), stamp=False,
                          speed_label=False, title="", legend=False, rate_strip=False,
                          pingpong=False)
        n = int(rec.n)
        self.chunk_s = dur if n <= EVENT_BUDGET else max(MIN_CHUNK_S,
                                                          dur * EVENT_BUDGET / max(n, 1))
        self.n_chunks = max(1, int(np.ceil(dur / self.chunk_s - 1e-9)))
        whole = Shot(look=self.look.name, params=dict(self.P), speed=1.0, fps=_INDEX_FPS)
        self.n_index = int(self.look.n_frames(None, self.P)
                           or whole.n_frames(self.t_lo, self.t_hi))
        self._chunk = None                 # (index, ShotRenderer)
        self._carry = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ chunks
    @property
    def rec(self):
        rec = self._rec()
        if rec is None:
            raise ReferenceError("the cell's recording no longer exists")
        return rec

    def _bounds(self, j):
        a = self.t_lo + j * self.chunk_s
        b = self.t_hi if j >= self.n_chunks - 1 else min(self.t_hi, a + self.chunk_s)
        return a, b

    def _build(self, j) -> ShotRenderer:
        a, b = self._bounds(j)
        shot = replace(self._shot, t0=a, t1=b)
        rec = self.rec
        lookback = self.look.memory_s(self.P)
        ev = prepare_events(rec, shot, a, b, lookback)
        r = ShotRenderer(rec, shot, events=ev)
        if "frames" in r.st:                 # orbit progress counts over the whole clip
            r.st["frames"] = self.n_index
        return r

    def _renderer(self, tau):
        j = int(np.clip(np.floor((tau - self.t_lo) / self.chunk_s), 0, self.n_chunks - 1))
        if self._chunk is not None and self._chunk[0] == j:
            return self._chunk[1]
        if self._carry is None:
            first = self._build(0) if j != 0 else None
            r0 = first if first is not None else self._build(0)
            self._carry = {k: r0.st[k] for k in _CARRY if k in r0.st}
            if j == 0:
                self._chunk = (0, r0)
                return r0
        r = self._build(j)
        r.st.update(self._carry)
        self._chunk = (j, r)
        return r

    def index_at(self, tau) -> int:
        """The frame index *tau* corresponds to over the whole clip (orbit / strobe)."""
        frac = (float(tau) - self.t_lo) / max(self.t_hi - self.t_lo, 1e-9)
        return int(np.clip(np.floor(frac * self.n_index), 0, self.n_index - 1))

    # ------------------------------------------------------------------ frames
    def frame(self, tau) -> np.ndarray:
        """The look at recording time *tau*, letterboxed into the cell → ``(h, w, 3)``."""
        tau = float(np.clip(tau, self.t_lo, self.t_hi))
        with self._lock:
            r = self._renderer(tau)
            img = r.frame_at(tau, self.index_at(tau))
            bg = tuple(int(round(c * 255)) for c in r.colors["bg"])
        if self.stretch and img.shape[1::-1] != self.size:
            from PIL import Image
            return np.asarray(Image.fromarray(img).resize(self.size, Image.BILINEAR))
        return letterbox(img, self.size, bg)

    def stamp(self, tau) -> str:
        """The look's own time label at *tau* (what the Studio prints in its corner)."""
        tau = float(np.clip(tau, self.t_lo, self.t_hi))
        with self._lock:
            r = self._renderer(tau)
            return self.look.stamp(self.index_at(tau), tau, self.P, r.st)

    def background(self) -> tuple:
        from gottlux.core import colors
        return tuple(int(round(c * 255)) for c in colors.hex_to_rgb01(
            colors.palette(self._shot.palette)["bg"]))


def letterbox(img, size, bg=(0, 0, 0)) -> np.ndarray:
    """Center *img* in a ``size = (w, h)`` frame filled with *bg* (cropping any excess)."""
    w, h = int(size[0]), int(size[1])
    img = np.asarray(img, np.uint8)
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img
    out = np.empty((h, w, 3), np.uint8)
    out[:] = np.asarray(bg, np.uint8)
    cw, ch = min(iw, w), min(ih, h)
    sx, sy = (iw - cw) // 2, (ih - ch) // 2
    dx, dy = (w - cw) // 2, (h - ch) // 2
    out[dy:dy + ch, dx:dx + cw] = img[sy:sy + ch, sx:sx + cw]
    return out


def _key(rec, look, params, palette, roi, denoise, bloom, size, stretch):
    return (id(rec), int(rec.n), float(rec.t_start_s), look,
            json.dumps(params or {}, sort_keys=True, default=str), str(palette),
            None if roi is None else tuple(int(v) for v in roi), bool(denoise),
            round(float(bloom), 4), (int(size[0]), int(size[1])), bool(stretch))


def cell_for(rec, look, params=None, palette="fire-ice", roi=None, denoise=False, bloom=0.0,
             size=(320, 240), stretch=False) -> LookCell:
    """The cached :class:`LookCell` for these settings (built on first use)."""
    key = _key(rec, look, params, palette, roi, denoise, bloom, size, stretch)
    with _cache_lock:
        cell = _cache.get(key)
        if cell is not None and cell._rec() is rec:
            _cache.move_to_end(key)
            return cell
    cell = LookCell(rec, look, params, palette, roi, denoise, bloom, size, stretch)
    with _cache_lock:
        _cache[key] = cell
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return cell


def clear_cache() -> None:
    """Forget every cached cell (frees their event chunks)."""
    with _cache_lock:
        _cache.clear()
