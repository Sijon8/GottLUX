"""
dualview.py — co-register and converge a tracking/range study across two clips.

When the same target is observed by two sensors at once — a **wide** acquisition FOV and a
**narrow** precision FOV on a short-baseline rig — this module maps image-plane geometry from
one view into the other so their track boxes can be **superimposed**, and pools their keyframes
into one **converged** pixels-on-target / perception-range solve (the wide sensor for reach and
continuity, the narrow for precision).

Two pieces, both pure NumPy (no Qt):

* :class:`CoRegistration` + :func:`map_point` / :func:`map_box` — transform a point/box from a
  *source* view to a *target* view under one of four modes:
    - ``"none"``      — don't superimpose.
    - ``"fov_scale"`` — pinhole angular mapping by the two FOVs (the narrow view is the central
      crop of the wide view). Parallax-free; exact when the boresights are co-aligned.
    - ``"parallax"``  — ``fov_scale`` plus the range-dependent stereo disparity from the rig
      baseline (``disparity_px = baseline · f_px / range``) and a fixed boresight offset.
    - ``"manual"``    — a stored affine nudge ``(dx, dy, scale)`` you align by eye.
* :class:`ConvergedStudy` — pools keyframes from both clips into one target-size fit and reports
  each clip's Johnson perception ranges plus the fused estimate.

The mapping is angular, so it is correct regardless of which view is wider. At the rig's ~25 mm
baseline the parallax is sub-pixel beyond ~10 m, so ``fov_scale`` is usually sufficient; the
``parallax`` mode is there for short range or when the calibration is known.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gottlux.core import photogrammetry as pg


# --------------------------------------------------------------------- view geometry
@dataclass
class ViewGeom:
    """One view's image geometry: horizontal/vertical FOV (deg) across W×H pixels."""
    fov_h_deg: float
    width_px: int
    height_px: int
    fov_v_deg: Optional[float] = None      # None → derived from the pinhole aspect ratio

    def __post_init__(self):
        if self.fov_v_deg is None:
            half = math.tan(math.radians(self.fov_h_deg) / 2.0) * (self.height_px / self.width_px)
            self.fov_v_deg = math.degrees(2.0 * math.atan(half))

    @property
    def dpp_x(self) -> float:
        """Degrees per pixel along the width (the bearing scale)."""
        return self.fov_h_deg / self.width_px

    @property
    def dpp_y(self) -> float:
        """Degrees per pixel along the height."""
        return self.fov_v_deg / self.height_px

    def focal_px(self) -> float:
        return pg.focal_px(self.fov_h_deg, self.width_px)


# --------------------------------------------------------------------- co-registration
@dataclass
class CoRegistration:
    """How to map a point/box from a source view into a target view."""
    mode: str = "fov_scale"                 # 'none' | 'fov_scale' | 'parallax' | 'manual'
    baseline_m: float = 0.025               # inter-sensor baseline (m) for the parallax disparity
    offset_deg_x: float = 0.0               # fixed boresight misalignment (deg, horizontal)
    offset_deg_y: float = 0.0               # fixed boresight misalignment (deg, vertical)
    parallax_sign: float = 1.0              # +1 / -1: which way the disparity shifts
    manual_dx: float = 0.0                  # manual mode: pixel nudge + scale about the centre
    manual_dy: float = 0.0
    manual_scale: float = 1.0

    @property
    def superimpose(self) -> bool:
        return self.mode != "none"


def map_point(x, y, src: ViewGeom, dst: ViewGeom, coreg: CoRegistration,
              range_m: Optional[float] = None):
    """Map image point ``(x, y)`` in *src* to *dst* coordinates. ``None`` if mode is ``"none"``.

    *range_m* is only used by the ``"parallax"`` mode (the disparity scales as ``1/range``).
    """
    if coreg.mode == "none":
        return None
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if coreg.mode == "manual":
        xp = dst.width_px / 2.0 + (x - src.width_px / 2.0) * coreg.manual_scale + coreg.manual_dx
        yp = dst.height_px / 2.0 + (y - src.height_px / 2.0) * coreg.manual_scale + coreg.manual_dy
        return xp, yp
    # angular (fov_scale / parallax): pixel → bearing in src, bearing → pixel in dst
    ang_x = (x - src.width_px / 2.0) * src.dpp_x + coreg.offset_deg_x
    ang_y = (y - src.height_px / 2.0) * src.dpp_y + coreg.offset_deg_y
    xp = dst.width_px / 2.0 + ang_x / dst.dpp_x
    yp = dst.height_px / 2.0 + ang_y / dst.dpp_y
    if coreg.mode == "parallax" and range_m and range_m > 0:
        xp = xp + coreg.parallax_sign * coreg.baseline_m * dst.focal_px() / float(range_m)
    return xp, yp


def map_box(bbox, src: ViewGeom, dst: ViewGeom, coreg: CoRegistration,
            range_m: Optional[float] = None):
    """Map a box ``(x0, y0, x1, y1)`` from *src* to *dst*. ``None`` if not superimposing."""
    if coreg.mode == "none":
        return None
    x0, y0, x1, y1 = bbox
    cx = map_point(np.array([x0, x1]), np.array([y0, y1]), src, dst, coreg, range_m)
    if cx is None:
        return None
    xs, ys = cx
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


# --------------------------------------------------------------------- converged study
def pooled_target_size_fit(points) -> dict:
    """Least-squares target size pooled across views.

    *points* is an iterable of ``(measured_px, range_m, focal_px)``. From ``N = L·f/D``, the
    weighted slope-through-origin is ``L = Σ N·(f/D) / Σ (f/D)²``. Pools keyframes from any
    number of clips with different optics into one estimate, with an R² across all of them.
    """
    pts = [(float(n), float(d), float(f)) for n, d, f in points if d and d > 0 and f > 0]
    if not pts:
        return {"n": 0}
    N = np.array([p[0] for p in pts]); D = np.array([p[1] for p in pts])
    F = np.array([p[2] for p in pts])
    x = F / D
    L = float(np.sum(N * x) / np.sum(x * x)) if np.sum(x * x) > 0 else float("nan")
    out = {"n": len(pts), "fitted_target_size_m": round(L, 4)}
    if len(pts) >= 2:
        pred = L * x
        ss_res = float(np.sum((N - pred) ** 2)); ss_tot = float(np.sum((N - N.mean()) ** 2))
        out["r2"] = round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else None
    return out


@dataclass
class ConvergedStudy:
    """A two-clip converged pixels-on-target / perception-range study.

    Each clip is a :class:`~gottlux.core.photogrammetry.ResolutionStudy` (its own FOV/geometry +
    keyframes with known ranges). The converged study pools every ranged keyframe into one
    target-size fit and reports the perception ranges each clip's optics would achieve for the
    fused size — so the wide and narrow sensors validate one common target model.
    """
    label_a: str
    study_a: "pg.ResolutionStudy"
    label_b: str
    study_b: "pg.ResolutionStudy"

    def _points(self):
        for st in (self.study_a, self.study_b):
            f = st.focal_px()
            for k in st.keyframes:
                if k.distance_m is not None and k.distance_m > 0:
                    yield (max(k.w_px, k.h_px), k.distance_m, f)

    def fit(self) -> dict:
        return pooled_target_size_fit(self._points())

    def summary(self) -> dict:
        fit = self.fit()
        L = fit.get("fitted_target_size_m") or self.study_a.target_size_m
        return {
            "converged": True,
            "clips": {
                self.label_a: {"fov_deg": self.study_a.fov_deg, "width_px": self.study_a.width_px,
                               "n_keyframes": len(self.study_a.keyframes),
                               "perception_ranges_m": pg.perception_ranges(
                                   L, self.study_a.fov_deg, self.study_a.width_px)},
                self.label_b: {"fov_deg": self.study_b.fov_deg, "width_px": self.study_b.width_px,
                               "n_keyframes": len(self.study_b.keyframes),
                               "perception_ranges_m": pg.perception_ranges(
                                   L, self.study_b.fov_deg, self.study_b.width_px)},
            },
            "pooled_fit": fit,
            "fused_target_size_m": L,
        }
