"""
denoise.py — a live, composable noise-reduction suite for event windows.

Event-camera streams carry several distinct nuisances; this module is the one place that
removes them, operating on any :class:`~gottlux.io.recording.EventWindow` so the *same* filter
chain can be applied **live in every tab** (the GUI binds one
:class:`~gottlux.app.filters.FilterController` to every view).

Filters (each independently toggleable, applied in this order)
-------------------------------------------------------------
* **polarity**       keep only ON or only OFF events (or both).
* **hot-pixel**      drop pixels whose event count is in the top percentile of the window
                     (stuck / flickering pixels that fire far more than the scene).
* **refractory**     per pixel, drop an event arriving < ``refractory_us`` after the previous
                     *kept* event at that pixel (suppresses high-rate pixel chatter).
* **background-activity (BAF)**  the classic event denoiser: keep an event only if a
                     spatio-temporal neighbour (8-connected) fired within ``baf_dt_us`` —
                     isolated shot noise has no correlated neighbour and is removed.

Everything is vectorized; the two per-event passes (refractory, BAF) are Numba-JIT with a
pure-NumPy fallback. Filtering a display window is cheap (a window is one accumulation slab).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gottlux.core.accel import njit
from gottlux.io.recording import EventWindow


@dataclass
class FilterSettings:
    """The live denoise configuration shared across the program."""
    enabled: bool = False
    polarity: str = "both"            # "both" | "on" | "off"
    hot_pixel: bool = False
    hot_pct: float = 99.9             # pixels above this count-percentile are dropped
    refractory: bool = False
    refractory_us: float = 1000.0     # per-pixel dead time (µs)
    baf: bool = False                 # background-activity (nearest-neighbour) filter
    baf_dt_us: float = 5000.0         # neighbour correlation window (µs)

    def active(self) -> bool:
        """True iff filtering would change anything (master switch + at least one filter)."""
        return self.enabled and (self.hot_pixel or self.refractory or self.baf
                                 or self.polarity != "both")

    def describe(self) -> str:
        if not self.active():
            return "off"
        bits = []
        if self.polarity != "both":
            bits.append(self.polarity)
        if self.hot_pixel:
            bits.append(f"hot≥{self.hot_pct:g}%")
        if self.refractory:
            bits.append(f"refr {self.refractory_us:g}µs")
        if self.baf:
            bits.append(f"BAF {self.baf_dt_us:g}µs")
        return " · ".join(bits)


# ------------------------------------------------------------------ per-pixel passes (JIT)
@njit(cache=True)
def _refractory_keep(flat, t, npix, refr):
    last = np.full(npix, -(2 ** 62), np.int64)
    keep = np.zeros(t.shape[0], np.uint8)
    for i in range(t.shape[0]):
        f = flat[i]
        if t[i] - last[f] >= refr:
            keep[i] = 1
            last[f] = t[i]            # measure the dead time from the last *kept* event
    return keep


@njit(cache=True)
def _baf_keep(x, y, t, H, W, dt):
    T = np.full((H, W), -(2 ** 62), np.int64)
    keep = np.zeros(t.shape[0], np.uint8)
    for i in range(t.shape[0]):
        xi = x[i]; yi = y[i]; ti = t[i]
        ok = False
        for dyk in range(-1, 2):
            yy = yi + dyk
            if yy < 0 or yy >= H:
                continue
            for dxk in range(-1, 2):
                if dxk == 0 and dyk == 0:
                    continue
                xx = xi + dxk
                if xx < 0 or xx >= W:
                    continue
                if ti - T[yy, xx] <= dt:
                    ok = True
        if ok:
            keep[i] = 1
        T[yi, xi] = ti                # causal: only past neighbours
    return keep


def hot_pixel_keep(x, y, W, H, pct) -> np.ndarray:
    """Boolean keep-mask dropping events at the hottest-firing pixels in the window."""
    flat = y.astype(np.int64) * W + x.astype(np.int64)
    cnt = np.bincount(flat, minlength=W * H)
    nz = cnt[cnt > 0]
    if nz.size == 0:
        return np.ones(x.shape[0], bool)
    thr = max(np.percentile(nz, pct), 1)
    hot = cnt >= thr
    return ~hot[flat]


# ------------------------------------------------------------------ the chain
def filter_window(win, settings: FilterSettings) -> EventWindow:
    """Return a denoised :class:`EventWindow` per *settings* (the input is returned unchanged
    when filtering is inactive or the window is empty)."""
    if settings is None or not settings.active() or len(win) == 0:
        return win
    x = np.asarray(win.x); y = np.asarray(win.y)
    p = np.asarray(win.p); t = np.asarray(win.t)
    W, H = int(win.width), int(win.height)
    keep = np.ones(x.shape[0], bool)

    if settings.polarity == "on":
        keep &= (p == 1)
    elif settings.polarity == "off":
        keep &= (p == 0)
    if settings.hot_pixel:
        keep &= hot_pixel_keep(x, y, W, H, settings.hot_pct)
    if settings.refractory and settings.refractory_us > 0:
        flat = y.astype(np.int64) * W + x.astype(np.int64)
        keep &= _refractory_keep(flat, t.astype(np.int64), W * H,
                                 np.int64(settings.refractory_us)).astype(bool)
    if settings.baf:
        keep &= _baf_keep(x.astype(np.int64), y.astype(np.int64), t.astype(np.int64),
                          H, W, np.int64(settings.baf_dt_us)).astype(bool)

    if keep.all():
        return win
    return EventWindow(x[keep], y[keep], p[keep], t[keep], W, H, getattr(win, "t0_us", 0))
