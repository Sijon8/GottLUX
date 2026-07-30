"""
background.py — static-clutter suppression for both capture regimes.

The detector only wants the *moving foreground*. What counts as static differs by regime:

* **Rotation** — a frozen, first-N-revolutions phase-space reference. The static rotating
  scene is learned once in ``(phase, y, x)`` voxels and **never updated**, so a target that
  enters later is never eroded (the failure mode of a cumulative running model, which keeps
  absorbing — and then masking — a target once it appears).
* **Staring** — a per-pixel persistent-background mask. On a fixed sensor the static scene
  fires rarely; pixels with a high baseline rate over a learning window are flicker / defect
  / swaying clutter and are suppressed.

Every function returns a boolean **keep mask** over events (``True`` = foreground to keep),
so background suppression composes with the denoise filters by simple AND.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


# ============================ ROTATION ============================
def reference_end_time(telemetry, n_rotations: int) -> float:
    """Event-time (s) at the end of the first *n_rotations* revolutions (clamped)."""
    n = max(1, int(n_rotations))
    i = min(n - 1, len(telemetry.hall_t) - 1)
    return float(telemetry.hall_t[i] + telemetry.offset)


def build_rotation_reference(rec, telemetry, n_rotations: int = 1, n_phase: int = 360,
                             dil_phase: int = 1, dil_xy: int = 1) -> np.ndarray:
    """Build the frozen ``(n_phase, H, W)`` boolean occupancy from the first N revolutions.

    A voxel is set if any event landed at that ``(phase, y, x)`` during the reference window;
    a small morphological dilation tolerates jitter. Pass this to
    :func:`rotation_foreground_mask` to suppress the static rotating background.
    """
    ref_end = reference_end_time(telemetry, n_rotations)
    win = rec.window(None, ref_end)
    W, H = rec.width, rec.height
    t = win.t_s
    xs = np.asarray(win.x); ys = np.asarray(win.y)
    ph = (telemetry.phase_at(t) * n_phase).astype(np.int64) % n_phase
    ref = np.zeros((n_phase, H, W), bool)
    ref[ph, ys, xs] = True
    if dil_phase or dil_xy:
        st = np.ones((2 * dil_phase + 1, 2 * dil_xy + 1, 2 * dil_xy + 1), bool)
        ref = ndimage.binary_dilation(ref, structure=st)
    return ref


def rotation_foreground_mask(win, telemetry, reference: np.ndarray) -> np.ndarray:
    """Keep mask: ``True`` where an event is NOT in the frozen rotation reference voxel."""
    n_phase = reference.shape[0]
    t = win.t_s
    x = np.asarray(win.x); y = np.asarray(win.y)
    ph = (telemetry.phase_at(t) * n_phase).astype(np.int64) % n_phase
    return ~reference[ph, y, x]


# ============================ STARING ============================
def staring_foreground_mask(win, bg_window_s: float = 1.0, baseline_pct: float = 99.5,
                            learn_from=None) -> np.ndarray:
    """Keep mask suppressing persistent-background pixels on a fixed sensor.

    Background pixels are those whose event count in the learning window (the first
    ``bg_window_s`` of *learn_from*, defaulting to *win* itself) exceeds ``baseline_pct`` of
    active-pixel counts. Returns ``True`` for foreground events to keep.
    """
    W, H = win.width, win.height
    src = learn_from if learn_from is not None else win
    lt = src.t_s
    lx = np.asarray(src.x); ly = np.asarray(src.y)
    learn = lt < (lt[0] + bg_window_s) if lt.size else np.zeros(0, bool)
    base = np.zeros((H, W), np.int64)
    if np.any(learn):
        np.add.at(base, (ly[learn], lx[learn]), 1)
    nz = base[base > 0]
    thr = np.percentile(nz, baseline_pct) if nz.size else np.inf
    bg = base >= max(thr, 1)
    x = np.asarray(win.x); y = np.asarray(win.y)
    return ~bg[y, x]
