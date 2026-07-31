"""
accumulate.py — turn a window of events into a 2-D frame.

An event stream is sparse and asynchronous; almost every visualization and detector needs
it rendered onto the sensor grid first. This module is the single, fast place that happens.
All functions take an :class:`~gottlux.io.recording.EventWindow` (or raw arrays) and return
a ``(H, W)`` float array.

Accumulation modes
------------------
``count``        events per pixel (ON+OFF).                      Good general view.
``polarity``     signed ON−OFF per pixel.                        Shows contrast direction.
``polarity_ratio`` (ON−OFF)/(ON+OFF) per pixel, ∈ [-1, 1].        Bounded — never blows out
                 in dense regions (unlike the raw signed count). Diverging colormap.
``on`` / ``off`` one polarity only.                              Asymmetric detection.
``time_surface`` exponentially-decayed time of the most recent   The "SAE" — sharp motion,
                 event at each pixel (a.k.a. Surface of Active    used by trackers/flutter.
                 Events).
``binary``       1 where any event landed, else 0.               Occupancy / masks.

The time-surface scatter (last-event-time-wins, then exponential decay) is the one loop
worth JIT-compiling; everything else is a vectorized :func:`numpy.bincount`.
"""
from __future__ import annotations

import numpy as np

from gottlux.core.accel import njit


@njit(cache=True)
def _time_surface_kernel(x, y, t, H, W):
    """Last-event-time per pixel (events assumed time-sorted ascending). Returns (H, W)
    float array of event times; pixels never hit hold -inf."""
    sae = np.full((H, W), -np.inf)
    for i in range(x.shape[0]):
        sae[y[i], x[i]] = t[i]
    return sae


def _as_arrays(win_or_x, y=None, p=None):
    """Accept either an EventWindow or explicit (x, y, p) arrays."""
    if y is None:                         # an EventWindow-like object
        w = win_or_x
        return (np.asarray(w.x), np.asarray(w.y), np.asarray(w.p),
                np.asarray(w.t), int(w.width), int(w.height))
    raise ValueError("pass an EventWindow, not bare arrays, to accumulate_frame")


def accumulate_frame(win, mode: str = "count", tau: float = 0.02,
                     ref_time_us: float | None = None,
                     normalize: bool = False) -> np.ndarray:
    """Accumulate an :class:`EventWindow` into a ``(H, W)`` frame.

    Parameters
    ----------
    win : EventWindow
        The events to render (already sliced to a time window / ROI).
    mode : str
        One of ``count, polarity, on, off, time_surface, binary`` (see module docs).
    tau : float
        Time-surface decay constant in **seconds** (only used for ``time_surface``).
    ref_time_us : float | None
        Reference "now" (µs) the time-surface decays toward; defaults to the last event.
    normalize : bool
        If True, scale the result into [0, 1] (per-frame) — convenient for display.
    """
    x, y, p, t, W, H = _as_arrays(win)
    if x.shape[0] == 0:
        return np.zeros((H, W), np.float32)
    xi = x.astype(np.int64)
    yi = y.astype(np.int64)
    flat = yi * W + xi

    if mode == "count":
        frame = np.bincount(flat, minlength=H * W).astype(np.float32)
    elif mode == "binary":
        frame = (np.bincount(flat, minlength=H * W) > 0).astype(np.float32)
    elif mode == "on":
        frame = np.bincount(flat[p == 1], minlength=H * W).astype(np.float32)
    elif mode == "off":
        frame = np.bincount(flat[p == 0], minlength=H * W).astype(np.float32)
    elif mode == "polarity":
        w = np.where(p == 1, 1.0, -1.0)
        frame = np.bincount(flat, weights=w, minlength=H * W).astype(np.float32)
    elif mode == "polarity_ratio":
        # Per-pixel signed balance (ON−OFF)/(ON+OFF) ∈ [-1, 1] — bounded regardless of how many
        # events land, so dense regions never "blow out" the way the raw signed count does.
        on = np.bincount(flat[p == 1], minlength=H * W).astype(np.float32)
        off = np.bincount(flat[p == 0], minlength=H * W).astype(np.float32)
        tot = on + off
        frame = np.zeros(H * W, np.float32)
        nz = tot > 0
        frame[nz] = (on[nz] - off[nz]) / tot[nz]
        return frame.reshape(H, W)          # already in [-1, 1]
    elif mode == "time_surface":
        ts = t.astype(np.float64) / 1e6
        ref = (ref_time_us / 1e6) if ref_time_us is not None else float(ts[-1])
        sae = _time_surface_kernel(xi, yi, ts, H, W)
        frame = np.exp((sae - ref) / max(tau, 1e-9)).astype(np.float32)
        frame[~np.isfinite(sae)] = 0.0
        return frame                      # already in [0, 1]
    else:
        raise ValueError(f"unknown accumulation mode {mode!r}")

    frame = frame.reshape(H, W)
    if normalize:
        m = frame.max() if mode != "polarity" else np.abs(frame).max()
        if m > 0:
            frame = frame / m
    return frame


def count_image(win) -> np.ndarray:
    """Shorthand for an integer event-count image (``(H, W)`` int32)."""
    x, y, p, t, W, H = _as_arrays(win)
    if x.shape[0] == 0:
        return np.zeros((H, W), np.int32)
    return np.bincount((y.astype(np.int64) * W + x.astype(np.int64)),
                       minlength=H * W).reshape(H, W).astype(np.int32)


def render_rgb(frame: np.ndarray, cmap: str = "inferno",
               polarity: bool = False, gamma: float = 1.0) -> np.ndarray:
    """Map a scalar frame to an ``(H, W, 3)`` uint8 RGB image via a matplotlib colormap.

    For ``polarity`` frames pass ``polarity=True`` to use a diverging map centered at 0.
    Imports matplotlib lazily (kept out of the hot path).
    """
    from matplotlib.colors import Normalize, TwoSlopeNorm

    from gottlux.core.tonemap import colormap
    f = frame.astype(np.float32)
    if polarity:
        a = float(np.abs(f).max()) or 1.0
        norm = TwoSlopeNorm(vmin=-a, vcenter=0.0, vmax=a)
        mapper = colormap("coolwarm")
    else:
        lo, hi = float(f.min()), float(np.percentile(f, 99.5)) if f.max() > 0 else (0.0, 1.0)
        if gamma != 1.0 and hi > lo:
            f = ((f - lo) / (hi - lo)).clip(0, 1) ** gamma
            norm = Normalize(0, 1)
        else:
            norm = Normalize(lo, hi if hi > lo else lo + 1)
        mapper = colormap(cmap)
    rgba = mapper(norm(f))
    return (rgba[..., :3] * 255).astype(np.uint8)
