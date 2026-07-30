"""
filters.py — event-stream denoising pre-filters.

Each filter returns a boolean **keep mask** over the events of an
:class:`~gottlux.io.recording.EventWindow` (``True`` = keep). They compose by AND-ing.

* :func:`hot_pixel_mask`        drop stuck/defective pixels (top-percentile firing rate)
* :func:`refractory_filter`     drop per-pixel chatter (events too soon after the last)
* :func:`rotation_phase_filter` ROTATION-only: keep only events that break the rotation-
  locked phase signature — i.e. moving/anomalous targets, not the static swept background.
"""
from __future__ import annotations

import numpy as np

from gottlux.core.accel import njit


def hot_pixel_mask(win, pct: float = 99.95) -> np.ndarray:
    """Keep mask removing pixels that fire in the top ``(100-pct)%`` by total count."""
    x = np.asarray(win.x); y = np.asarray(win.y)
    W, H = win.width, win.height
    cnt = np.bincount(y.astype(np.int64) * W + x.astype(np.int64),
                      minlength=H * W).reshape(H, W)
    nz = cnt[cnt > 0]
    thr = np.percentile(nz, pct) if nz.size else 1
    hot = cnt >= max(thr, 1)
    return ~hot[y, x]


@njit(cache=True)
def _refractory_kernel(pix, t, npix, period_us):
    """Drop events arriving < period_us after the previous event at the same pixel
    (events assumed time-sorted ascending). Returns a uint8 keep mask."""
    last = np.full(npix, -1.0e18)
    keep = np.ones(pix.shape[0], np.uint8)
    for i in range(pix.shape[0]):
        pp = pix[i]
        if t[i] - last[pp] < period_us:
            keep[i] = 0
        else:
            last[pp] = t[i]
    return keep


def refractory_filter(win, period_us: float) -> np.ndarray:
    """Keep mask enforcing a per-pixel refractory period (µs). 0 disables (keep all)."""
    n = len(win)
    if period_us <= 0 or n == 0:
        return np.ones(n, bool)
    x = np.asarray(win.x).astype(np.int64); y = np.asarray(win.y).astype(np.int64)
    pix = y * win.width + x
    t = np.asarray(win.t).astype(np.float64)
    return _refractory_kernel(pix, t, win.width * win.height, float(period_us)).astype(bool)


def rotation_phase_filter(win, telemetry, r_min: float = 0.5, recur_frac: float = 0.34,
                          min_events: int = 4, phase_tol: float = 0.12) -> np.ndarray:
    """Rotation-phase anomaly denoise: keep only events that *break* the rotation lock.

    On a continuously rotating sensor a static scene point is swept past a pixel once per
    revolution, always at the **same** phase. Such pixels are phase-concentrated (high
    circular resultant ``R``) and recur across many revolutions. A moving target is not
    phase-locked. Per pixel we accumulate circular statistics of the event phases and the
    revolution recurrence (both vectorized with :func:`numpy.bincount`, O(N)), then keep an
    event if its pixel is **not** locked background, or its phase deviates from the pixel
    mean by more than ``phase_tol`` revolutions.

    Returns an all-keep mask (no-op) if no telemetry or too few revolutions to judge.
    """
    n = len(win)
    if telemetry is None or n < 2:
        return np.ones(n, bool)
    x = np.asarray(win.x).astype(np.int64); y = np.asarray(win.y).astype(np.int64)
    t = np.asarray(win.t).astype(np.float64) / 1e6
    W, H = win.width, win.height
    n_rev = max(int(telemetry.n_revolutions), int(telemetry.revolution_at(t).max()) + 1)
    if n_rev < 2:
        return np.ones(n, bool)
    recur_min = max(2, int(recur_frac * n_rev))

    pix = y * W + x
    npix = W * H
    theta = 2.0 * np.pi * telemetry.phase_at(t)
    C = np.bincount(pix, weights=np.cos(theta), minlength=npix)
    S = np.bincount(pix, weights=np.sin(theta), minlength=npix)
    N = np.bincount(pix, minlength=npix).astype(np.float64)
    R = np.hypot(C, S) / np.maximum(N, 1.0)
    mu = np.arctan2(S, C)

    rev = telemetry.revolution_at(t).astype(np.int64)
    key = pix * n_rev + rev
    occ = np.bincount(key, minlength=npix * n_rev).reshape(npix, n_rev) > 0
    n_revs_pix = occ.sum(axis=1)

    static = (N >= min_events) & (n_revs_pix >= recur_min) & (R >= r_min)
    keep = ~static[pix]
    at_static = static[pix]
    if at_static.any():
        dphi = np.abs(np.angle(np.exp(1j * (theta - mu[pix]))))    # [0, pi]
        keep |= at_static & (dphi > 2.0 * np.pi * phase_tol)
    return keep
