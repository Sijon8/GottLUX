"""
geometry.py — the pinhole projection that turns sensor pixels into angles, world bearings,
and metric range. One place for all of gottlux's optical geometry.

Two regimes share this math:

* **Staring** — the boresight is fixed, so a pixel's horizontal offset from centre is a
  *relative bearing* and its vertical offset an *elevation* (both via the pinhole FOV).
* **Rotation** — the payload pans, so the *world* azimuth of a target is the boresight
  azimuth ``azimuth(t)`` (from telemetry) plus the intra-FOV pixel correction. A beautiful
  invariant falls out: a fixed target's world bearing is constant across a sweep, because as
  the FOV pans, ``azimuth(t)`` rises while the pixel offset falls and they cancel.

Range comes from the pinhole size relation ``D = L · f_px / s_px`` (physical size ``L``,
focal length in pixels ``f_px``, apparent size ``s_px``).
"""
from __future__ import annotations

import numpy as np


def focal_px(fov_deg: float, sensor_px: int) -> float:
    """Focal length in pixels for a horizontal field of view across ``sensor_px`` pixels."""
    return (sensor_px / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)


def pixel_to_bearing(x, fov_deg: float, width: int, az_sign: float = -1.0) -> np.ndarray:
    """Relative bearing (deg) of sensor column *x* from the boresight (centre = 0)."""
    return az_sign * (np.asarray(x, float) - width / 2.0) * (fov_deg / width)


def pixel_to_elevation(y, fov_deg: float, width: int, height: int) -> np.ndarray:
    """Elevation angle (deg) of sensor row *y* about the **height** centre (centre = 0).

    Uses the same per-pixel angular scale as bearing (``fov_deg/width``, the sensor's
    intrinsic deg/px); elevation is measured about ``height/2`` so it is correct on
    non-square sensors (a target at the vertical centre reads exactly 0°).
    """
    return (height / 2.0 - np.asarray(y, float)) * (fov_deg / width)


def world_azimuth(x, t_s, telemetry, fov_deg: float, width: int,
                  az_sign: float = -1.0) -> np.ndarray:
    """De-rotated **world** azimuth (deg, 0..360) of events/detections at columns *x*,
    times *t_s* (seconds), using rotation *telemetry*. Without telemetry this is just the
    relative bearing wrapped to 0..360."""
    rel = pixel_to_bearing(x, fov_deg, width, az_sign)
    if telemetry is None:
        return np.mod(rel, 360.0)
    pan = np.rad2deg(np.interp(np.asarray(t_s) - telemetry.offset,
                               telemetry.t, telemetry.azimuth_unwrapped()))
    return np.mod(pan + rel, 360.0)


def estimate_range_m(size_px, fov_deg: float, phys_m: float, sensor_px: int) -> np.ndarray:
    """Metric range (m) from apparent size via the pinhole model. ``phys_m<=0`` → NaN
    (absolute ranging disabled; use the relative-distance proxy instead)."""
    size_px = np.asarray(size_px, float)
    if phys_m <= 0:
        return np.full_like(size_px, np.nan)
    return phys_m * focal_px(fov_deg, sensor_px) / np.maximum(size_px, 1.0)


def relative_distance_proxy(size_px) -> np.ndarray:
    """Unitless, monotonic-with-range distance proxy ``1/apparent_size`` (no object size
    needed). Calibrate to metres per flight from a known near/far via a linear fit."""
    return 1.0 / np.maximum(np.asarray(size_px, float), 1.0)
