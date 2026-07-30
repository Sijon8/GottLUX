"""
panorama.py — figures unique to the rotating payload: the de-rotated 360° world map and a
polar radar of localized targets.

A spinning EBS sees only a narrow slice at any instant, but telemetry lets us project every
event into a stabilized world frame. The **panorama** is a 2-D histogram of world azimuth
(0–360°) against elevation, so the entire surrounding scene reconstructs into one image — the
core demonstration that a single rotating sensor is a panoramic, volumetric instrument. The
**radar** places each detected target at its bearing and range, the operator's-eye view.
"""
from __future__ import annotations

import numpy as np

from gottlux import sensors
from gottlux.core import geometry as geo
from gottlux.viz import theme


def panorama_figure(rec, fov_deg=None, az_bins=720, el_bins=240, max_events=4_000_000,
                    cmap="gottlux_events", title=None, width=theme.COL_DOUBLE):
    """De-rotated 360° azimuth×elevation panorama of a rotating recording.

    Requires telemetry (falls back to a single-FOV relative-bearing strip if absent).
    Subsamples to *max_events* for responsiveness. Returns the figure.
    """
    import matplotlib.pyplot as plt
    theme.apply()
    fov = fov_deg or sensors.DEFAULT_FOV_DEG
    n = rec.n
    step = max(1, n // max_events)
    x = np.asarray(rec.x[::step]).astype(np.float64)
    y = np.asarray(rec.y[::step]).astype(np.float64)
    t_s = np.asarray(rec.t[::step]).astype(np.float64) / 1e6
    az = geo.world_azimuth(x, t_s, rec.telemetry, fov, rec.width)
    el = geo.pixel_to_elevation(y, fov, rec.width, rec.height)
    el_lo, el_hi = np.percentile(el, [0.5, 99.5])
    H = np.histogram2d(az, el, bins=[az_bins, el_bins],
                       range=[[0, 360], [el_lo, el_hi]])[0]
    fig = theme.figure(width, width * 0.42)
    ax = fig.add_subplot(111)
    hi = np.percentile(H[H > 0], 99) if np.any(H > 0) else 1.0
    im = ax.imshow(np.clip(H.T / max(hi, 1e-9), 0, 1), origin="lower", aspect="auto",
                   extent=[0, 360, el_lo, el_hi], cmap=cmap, interpolation="nearest")
    ax.set_xlabel("world azimuth (deg)")
    ax.set_ylabel("elevation (deg)")
    ax.set_title(title or ("De-rotated 360° panorama" if rec.is_rotating
                           else "Relative-bearing strip (no telemetry)"))
    ax.set_xticks(np.arange(0, 361, 45))
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.025)
    cb.set_label("event density (norm.)")
    fig.tight_layout()
    return fig


def radar_figure(targets, use_range=True, title=None, width=theme.COL_SINGLE):
    """Polar radar of localized targets: bearing (angle) × range or relative distance.

    *targets* is a list of :class:`~gottlux.detectors.base.Target`. Each target is drawn as a
    track in polar coordinates, colored by its flutter frequency.
    """
    import matplotlib.pyplot as plt
    theme.apply()
    fig = theme.figure(width, width)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    any_pts = False
    for t in targets:
        az = t.azimuth_deg if t.azimuth_deg is not None else None
        if az is None:
            continue
        if use_range and t.range_m is not None and np.isfinite(t.range_m).any():
            r = t.range_m
            rlabel = "range (m)"
        else:
            r = t.rel_distance if t.rel_distance is not None else np.ones_like(az)
            rlabel = "relative distance"
        c = t.median_freq
        sc = ax.scatter(np.deg2rad(az), r, c=np.full_like(az, c), s=14,
                        cmap="turbo", vmin=0, vmax=max(c * 1.5, 1), alpha=0.85)
        any_pts = True
    ax.set_title(title or "Target radar", pad=14)
    if not any_pts:
        ax.text(0.5, 0.5, "no localized targets", ha="center", va="center",
                transform=ax.transAxes)
    return fig
