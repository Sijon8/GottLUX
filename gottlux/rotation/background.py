"""
background.py  --  Static-background modeling for BOTH capture modes.

ROTATION mode: a FROZEN first-rotation phase-space reference mask. The static
rotating background is learned once (rotation 0) in voxels (phase, x, y) and never
updated, so a target that appears later is never eroded (unlike a cumulative model).

STARING mode: a per-pixel persistent-background mask. On a fixed sensor the static
scene fires rarely; pixels with a high BASELINE event rate over a learning window
are flicker/defect/background and are suppressed, leaving moving foreground.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage


def hot_pixel_mask(ev, pct=99.95):
    """(H,W) bool: pixels firing in the top `pct` percentile of total counts."""
    W, H = ev["width"], ev["height"]
    cnt = np.zeros((H, W), np.int64)
    np.add.at(cnt, (np.asarray(ev["y"]), np.asarray(ev["x"])), 1)
    nz = cnt[cnt > 0]
    thr = np.percentile(nz, pct) if nz.size else 1
    return cnt >= max(thr, 1)


# --------------------------- ROTATION ---------------------------
def reference_end_time(tel, n_rotations):
    """End time (event-time base) of a reference window spanning the first
    `n_rotations` revolutions: the n-th Hall pulse, clamped to what's available."""
    n = max(1, int(n_rotations))
    i = min(n - 1, len(tel.hall_t) - 1)
    return float(tel.hall_t[i] + tel.offset)


def build_reference(ev, tel, ref_end_s, n_phase=360, dil_phase=1, dil_xy=1):
    """Frozen (n_phase,H,W) bool occupancy from the first rotation [0, ref_end_s]."""
    W, H = ev["width"], ev["height"]
    t = np.asarray(ev["t"]) / 1e6
    sel = t < ref_end_s
    xs, ys = np.asarray(ev["x"])[sel], np.asarray(ev["y"])[sel]
    ph = (tel.phase_at(t[sel]) * n_phase).astype(np.int64) % n_phase
    ref = np.zeros((n_phase, H, W), bool)
    ref[ph, ys, xs] = True
    if dil_phase or dil_xy:
        st = np.ones((2 * dil_phase + 1, 2 * dil_xy + 1, 2 * dil_xy + 1), bool)
        ref = ndimage.binary_dilation(ref, structure=st)
    return ref


def rotation_drop_mask(ev, tel, ref, n_phase, hot):
    """Bool over events: True = suppress (in frozen reference OR hot pixel)."""
    t = np.asarray(ev["t"]) / 1e6
    x, y = np.asarray(ev["x"]), np.asarray(ev["y"])
    ph = (tel.phase_at(t) * n_phase).astype(np.int64) % n_phase
    return ref[ph, y, x] | hot[y, x]


# --------------------------- STARING ----------------------------
def staring_drop_mask(ev, hot, bg_window_s=1.0, baseline_pct=99.5):
    """Bool over events: True = suppress (hot OR persistent-background pixel).

    Background pixels are those whose event rate in the first `bg_window_s`
    exceeds `baseline_pct` of active-pixel rates (flicker, swaying clutter, etc.)."""
    W, H = ev["width"], ev["height"]
    t = np.asarray(ev["t"]) / 1e6
    x, y = np.asarray(ev["x"]), np.asarray(ev["y"])
    learn = t < bg_window_s
    base = np.zeros((H, W), np.int64)
    if learn.any():
        np.add.at(base, (y[learn], x[learn]), 1)
    nz = base[base > 0]
    thr = np.percentile(nz, baseline_pct) if nz.size else np.inf
    bg = (base >= max(thr, 1)) | hot
    return bg[y, x]
