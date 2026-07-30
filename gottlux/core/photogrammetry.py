"""
photogrammetry.py — pinhole pixels-on-target / perception-range solver.

Given the sensor geometry (horizontal field of view across ``width`` pixels), a target's
physical size, and a range, this module answers the question the range lab is built around:
**how many pixels does the target span, and how far away can it still be perceived?**

The pinhole model
-----------------
Focal length in pixels for a horizontal FOV ``θ`` across ``W`` px::

    f_px = (W / 2) / tan(θ / 2)

A target of physical size ``L`` (m, its critical/largest dimension) at range ``D`` (m) images
across::

    N(D) = L · f_px / D          (pixels on target)

which inverts to the range at which it spans ``N`` pixels::

    D(N) = L · f_px / N

Per-pixel angular resolution (IFOV) and the ground-sample distance (metres/pixel) at range D::

    IFOV = θ / W                 GSD(D) = D · IFOV

Perception thresholds (Johnson's criteria)
------------------------------------------
Johnson's criteria express a discrimination task as a number of resolved line-pairs (*cycles*)
across the target's critical dimension. Sampling a cycle needs ≥ 2 pixels (Nyquist), so the
common pixels-across thresholds are:

==============  ======  ==========
task            cycles  pixels (≈)
==============  ======  ==========
detection       1.0     2.0
orientation     1.4     2.8
recognition     4.0     8.0
identification  6.4     12.8
==============  ======  ==========

These are 50%-probability rules of thumb; the cycle→pixel factor (2 px/cycle) and the task
definitions are stated explicitly so a paper can cite the exact assumption.

Everything here is pure NumPy (scalar or array ``D``/``N``); no Qt, no matplotlib.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

#: Johnson's criteria as **pixels across the critical dimension** (2 px per resolved cycle).
JOHNSON_PIXELS = {
    "detection": 2.0,
    "orientation": 2.8,
    "recognition": 8.0,
    "identification": 12.8,
}
#: The underlying line-pair (cycle) counts, for citation.
JOHNSON_CYCLES = {"detection": 1.0, "orientation": 1.4, "recognition": 4.0, "identification": 6.4}
PIXELS_PER_CYCLE = 2.0


# ------------------------------------------------------------------ primitives
def focal_px(fov_deg: float, width_px: float) -> float:
    """Focal length in pixels for a horizontal field of view across ``width_px`` pixels."""
    return (width_px / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def ifov_mrad(fov_deg: float, width_px: float) -> float:
    """Instantaneous field of view per pixel (milliradians) — the angular resolution."""
    return math.radians(fov_deg / width_px) * 1e3


def gsd_m(distance_m, fov_deg: float, width_px: float):
    """Ground-sample distance: metres on target per pixel at range ``distance_m``."""
    return np.asarray(distance_m, float) * math.radians(fov_deg / width_px)


def pixels_on_target(size_m, distance_m, fov_deg: float, width_px: float):
    """Expected pixels spanning a target of physical size ``size_m`` (m) at ``distance_m`` (m)."""
    f = focal_px(fov_deg, width_px)
    d = np.asarray(distance_m, float)
    return size_m * f / np.maximum(d, 1e-9)


def range_for_pixels(size_m, n_px, fov_deg: float, width_px: float):
    """Range (m) at which a target of size ``size_m`` (m) spans ``n_px`` pixels."""
    f = focal_px(fov_deg, width_px)
    n = np.asarray(n_px, float)
    return size_m * f / np.maximum(n, 1e-9)


def angular_size_deg(size_px, fov_deg: float, width_px: float):
    """Angle (deg) subtended by ``size_px`` pixels at the sensor's deg/px scale."""
    return np.asarray(size_px, float) * (fov_deg / width_px)


def size_from_pixels(n_px, distance_m, fov_deg: float, width_px: float):
    """Implied physical size (m) of a target that spans ``n_px`` px at known ``distance_m``.

    Inverts the pinhole model — the empirical calibration of target size from a measured box.
    """
    f = focal_px(fov_deg, width_px)
    return np.asarray(n_px, float) * np.asarray(distance_m, float) / f


def perception_ranges(size_m: float, fov_deg: float, width_px: float,
                      criteria: dict | None = None) -> dict:
    """Max range (m) for each Johnson task for a target of physical size ``size_m`` (m)."""
    crit = criteria or JOHNSON_PIXELS
    return {k: float(range_for_pixels(size_m, n, fov_deg, width_px)) for k, n in crit.items()}


# ------------------------------------------------------------------ optics (sensor + lens)
# The authoritative sensor/camera datasheet registry is :mod:`gottlux.sensors`. These presets
# are derived from it so the geometry helpers, the range-lab picker and the run manifest all
# agree on one set of numbers; pixel pitch is what ties pixels to a metric angle.
from gottlux import sensors as _sensors


def _sensor_preset(profile) -> dict:
    return dict(width_px=profile.width_px, height_px=profile.height_px,
                pixel_pitch_um=profile.pixel_pitch_um, note=profile.note or profile.name)


#: Built-in sensor presets (display name → geometry), derived from :data:`gottlux.sensors.PROFILES`.
SENSORS = {p.name: _sensor_preset(p) for p in _sensors.list_profiles().values()}
SENSORS["Custom"] = dict(width_px=320, height_px=320, pixel_pitch_um=6.3, note="user-defined sensor")

#: Built-in M12 / S-mount lens focal lengths (mm). 1.8 mm is the GenX320 default rig (~76° DFoV);
#: the default rig's focal length is taken from the registry so the two never drift apart.
LENSES = {
    f"{_sensors.GENX320.focal_length_mm:g} mm (M12, ~{_sensors.GENX320.fov_diagonal_deg:g}° DFoV)":
        _sensors.GENX320.focal_length_mm,
    "6 mm": 6.0,
    "8 mm": 8.0,
    "12 mm (narrow)": 12.0,
    "16 mm": 16.0,
    "25 mm": 25.0,
}
FOV_AXES = ("diagonal", "horizontal", "vertical")


def n_along_axis(width_px, height_px, axis="horizontal") -> float:
    """Pixel count along the chosen FOV axis (diagonal uses the array diagonal in pixels)."""
    if axis == "vertical":
        return float(height_px)
    if axis == "diagonal":
        return float(math.hypot(width_px, height_px))
    return float(width_px)


def focal_px_from_optics(focal_length_mm, pixel_pitch_um) -> float:
    """Focal length in **pixels** = focal length / pixel pitch (the most fundamental scale)."""
    return float(focal_length_mm) / (float(pixel_pitch_um) * 1e-3)


def focal_px_from_fov(fov_deg, n_px) -> float:
    """Focal length in pixels from a field of view spanning *n_px* pixels along that axis."""
    return (float(n_px) / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def fov_from_optics(focal_length_mm, pixel_pitch_um, n_px) -> float:
    """FOV (deg) along an axis of *n_px* pixels for a lens focal length + pixel pitch."""
    half_mm = float(n_px) * float(pixel_pitch_um) * 1e-3 / 2.0
    return float(math.degrees(2.0 * math.atan2(half_mm, float(focal_length_mm))))


# focal_px-centric primitives (axis-independent — focal_px already encodes the optics)
def pix_on_target(L, D, fpx):
    return np.asarray(L, float) * fpx / np.maximum(np.asarray(D, float), 1e-9)


def range_for_pix(L, N, fpx):
    return np.asarray(L, float) * fpx / np.maximum(np.asarray(N, float), 1e-9)


def size_from_pix(N, D, fpx):
    return np.asarray(N, float) * np.asarray(D, float) / fpx


def ang_size_deg(px, fpx):
    return np.degrees(2.0 * np.arctan(np.asarray(px, float) / (2.0 * fpx)))


def ifov_mrad_f(fpx) -> float:
    return float(2.0 * math.atan(1.0 / (2.0 * fpx)) * 1e3)


def gsd_m_f(D, fpx):
    return np.asarray(D, float) / fpx


def perception_ranges_f(L, fpx) -> dict:
    return {k: float(range_for_pix(L, n, fpx)) for k, n in JOHNSON_PIXELS.items()}


# ------------------------------------------------------------------ keyframes
@dataclass
class Keyframe:
    """One annotated bounding box at a moment in time, with an (optional) known range."""
    t_s: float
    bbox: tuple                       # (x0, y0, x1, y1) in pixels
    distance_m: Optional[float] = None
    label: str = ""

    @property
    def w_px(self) -> float:
        return abs(self.bbox[2] - self.bbox[0])

    @property
    def h_px(self) -> float:
        return abs(self.bbox[3] - self.bbox[1])

    @property
    def diag_px(self) -> float:
        return float(math.hypot(self.w_px, self.h_px))


def solve_keyframe(kf: Keyframe, target_size_m: float, fov_deg: float,
                   width_px: float, height_px: float) -> dict:
    """Solve every derived quantity for one keyframe.

    Returns a flat dict suitable for a results table: measured pixels-on-target (w/h/diag),
    the angular size, and — when the keyframe's range is known — the expected pixels at that
    range, the implied physical size (calibration), the IFOV and the GSD at that range.
    """
    w, h, diag = kf.w_px, kf.h_px, kf.diag_px
    measured = max(w, h)            # critical dimension ≈ the largest apparent extent
    out = {
        "t_s": round(float(kf.t_s), 4),
        "label": kf.label,
        "x0": float(kf.bbox[0]), "y0": float(kf.bbox[1]),
        "x1": float(kf.bbox[2]), "y1": float(kf.bbox[3]),
        "width_px": round(w, 2), "height_px": round(h, 2), "diag_px": round(diag, 2),
        "measured_px_on_target": round(float(measured), 2),
        "angular_size_deg": round(float(angular_size_deg(measured, fov_deg, width_px)), 4),
        "ifov_mrad": round(ifov_mrad(fov_deg, width_px), 5),
        "distance_m": (round(float(kf.distance_m), 3) if kf.distance_m is not None else None),
    }
    if kf.distance_m is not None and kf.distance_m > 0:
        D = float(kf.distance_m)
        out["expected_px_on_target"] = round(float(pixels_on_target(target_size_m, D, fov_deg, width_px)), 2)
        out["implied_target_size_m"] = round(float(size_from_pixels(measured, D, fov_deg, width_px)), 4)
        out["gsd_m_per_px"] = round(float(gsd_m(D, fov_deg, width_px)), 5)
        # residual: how far the measured box is from the pinhole prediction (sanity / SNR of fit)
        out["px_residual"] = round(out["measured_px_on_target"] - out["expected_px_on_target"], 2)
    return out


@dataclass
class ResolutionStudy:
    """A full pixels-on-target study: inputs + per-keyframe solves + fits + perception ranges."""
    target_size_m: float
    fov_deg: float
    width_px: int
    height_px: int
    keyframes: list = field(default_factory=list)        # list[Keyframe]

    # ----- derived -----
    def focal_px(self) -> float:
        return focal_px(self.fov_deg, self.width_px)

    def solved(self) -> list:
        return [solve_keyframe(k, self.target_size_m, self.fov_deg, self.width_px, self.height_px)
                for k in self.keyframes]

    def fit_target_size(self) -> dict:
        """Least-squares effective target size from keyframes that carry a known range.

        From ``N = L·(f/D)`` a weighted least-squares slope through the origin gives
        ``L = Σ N·(f/D) / Σ (f/D)²``; also reports the per-keyframe implied sizes' spread.
        """
        f = self.focal_px()
        pts = [(max(k.w_px, k.h_px), float(k.distance_m)) for k in self.keyframes
               if k.distance_m is not None and k.distance_m > 0]
        if len(pts) < 1:
            return {"n": 0}
        N = np.array([p[0] for p in pts], float)
        D = np.array([p[1] for p in pts], float)
        x = f / D
        L_fit = float(np.sum(N * x) / np.sum(x * x)) if np.sum(x * x) > 0 else float("nan")
        implied = N * D / f
        res = {
            "n": len(pts),
            "fitted_target_size_m": round(L_fit, 4),
            "implied_size_mean_m": round(float(np.mean(implied)), 4),
            "implied_size_std_m": round(float(np.std(implied)), 4),
        }
        if len(pts) >= 2:                                  # coefficient of determination
            pred = L_fit * x
            ss_res = float(np.sum((N - pred) ** 2))
            ss_tot = float(np.sum((N - N.mean()) ** 2))
            res["r2"] = round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else None
        return res

    def perception_ranges(self, size_m: Optional[float] = None) -> dict:
        """Johnson max-range table for the configured (or fitted) target size."""
        L = self.target_size_m if size_m is None else size_m
        return perception_ranges(L, self.fov_deg, self.width_px)

    def summary(self, use_fitted: bool = True) -> dict:
        """A machine-readable summary: inputs, fit, perception ranges, per-keyframe solves."""
        fit = self.fit_target_size()
        L = fit.get("fitted_target_size_m") if (use_fitted and fit.get("n", 0)) else self.target_size_m
        return {
            "inputs": {
                "target_size_m": self.target_size_m,
                "fov_deg": self.fov_deg,
                "width_px": self.width_px,
                "height_px": self.height_px,
                "focal_px": round(self.focal_px(), 2),
                "ifov_mrad": round(ifov_mrad(self.fov_deg, self.width_px), 5),
            },
            "johnson_pixels": JOHNSON_PIXELS,
            "johnson_cycles": JOHNSON_CYCLES,
            "fit": fit,
            "size_used_for_ranges_m": L,
            "perception_ranges_m": self.perception_ranges(L),
            "n_keyframes": len(self.keyframes),
            "keyframes": self.solved(),
        }
