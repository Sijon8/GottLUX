"""
metrics.py — quantitative figures of merit (the numbers that belong in a paper).

Two families:

* **Coverage** — what volume a *single rotating* EBS surveils versus a static one: the swept
  solid angle (steradians), the fraction of the full sphere, the gain over a fixed sensor,
  and the revisit interval / update rate.
* **Localization** — how well targets are pinned down: bearing standard error, elevation
  spread, and range statistics from a detection/track table.

Everything returns a plain dict so it drops straight into a run manifest, and
:func:`to_markdown` renders a tidy report.
"""
from __future__ import annotations

import numpy as np


def _solid_angle(az_span_deg, elev_lo_deg, elev_hi_deg) -> float:
    """Solid angle (sr) of an azimuth wedge over an elevation band:
    ``Ω = Δaz · (sin elev_hi − sin elev_lo)``."""
    return float(np.deg2rad(az_span_deg) *
                 (np.sin(np.deg2rad(elev_hi_deg)) - np.sin(np.deg2rad(elev_lo_deg))))


def coverage_metrics(rec, fov_deg: float, sample_events: int = 2_000_000) -> dict:
    """Coverage figures of merit for *rec* (rotating if telemetry attached, else staring)."""
    H, W = rec.height, rec.width
    dpp = fov_deg / W
    n = rec.n
    if n == 0:
        return {}
    step = max(1, n // sample_events)
    y = np.asarray(rec.y[::step])
    elev = (H / 2.0 - y) * dpp
    e_lo, e_hi = np.percentile(elev, [1, 99])
    az_span = 360.0 if rec.is_rotating else fov_deg
    omega = _solid_angle(az_span, e_lo, e_hi)
    omega_static = _solid_angle(fov_deg, -fov_deg / 2, fov_deg / 2)
    res = {
        "coverage_azimuth_deg": round(az_span, 1),
        "coverage_elev_band_deg": round(float(e_hi - e_lo), 1),
        "swept_solid_angle_sr": round(omega, 3),
        "sphere_fraction_pct": round(100 * omega / (4 * np.pi), 1),
        "static_ebs_solid_angle_sr": round(omega_static, 3),
        "coverage_gain_vs_static": round(omega / omega_static, 1) if omega_static else None,
    }
    if rec.is_rotating:
        tel = rec.telemetry
        res["revisit_interval_s"] = round(tel.T_rot, 3)
        res["update_rate_hz"] = round(1.0 / tel.T_rot, 3) if tel.T_rot else None
        res["n_revolutions"] = int(tel.n_revolutions)
    return res


def localization_metrics(track_table: dict) -> dict:
    """Localization figures of merit from a packed track/detection table.

    Expects optional keys ``azimuth_deg``, ``elev_deg``, ``range_m``. Missing keys are
    skipped gracefully.
    """
    res = {}
    if not track_table:
        return res
    az = track_table.get("azimuth_deg")
    if az is not None and len(az):
        res["n_points"] = int(len(az))
        # bearing standard error from the scatter about a slowly-varying mean
        res["bearing_std_deg"] = round(float(np.std(np.asarray(az))), 3)
    el = track_table.get("elev_deg")
    if el is not None and len(el):
        res["elevation_median_deg"] = round(float(np.median(el)), 2)
        res["elevation_std_deg"] = round(float(np.std(el)), 2)
    rng = track_table.get("range_m")
    if rng is not None and len(rng) and np.isfinite(rng).any():
        res["range_median_m"] = round(float(np.nanmedian(rng)), 2)
        res["range_iqr_m"] = (f"{np.nanpercentile(rng, 25):.2f}-"
                              f"{np.nanpercentile(rng, 75):.2f}")
    return res


def to_markdown(res: dict, title: str = "gottlux — Quantitative Metrics") -> str:
    """Render a metrics dict as a grouped Markdown report."""
    groups = [
        ("Coverage (single rotating sensor vs static EBS)",
         ["coverage_azimuth_deg", "coverage_elev_band_deg", "swept_solid_angle_sr",
          "sphere_fraction_pct", "static_ebs_solid_angle_sr", "coverage_gain_vs_static"]),
        ("Temporal sampling",
         ["revisit_interval_s", "update_rate_hz", "n_revolutions"]),
        ("Target detection & localization",
         ["n_points", "bearing_std_deg", "elevation_median_deg", "elevation_std_deg",
          "range_median_m", "range_iqr_m"]),
    ]
    lines = [f"# {title}", ""]
    for header, keys in groups:
        present = [k for k in keys if k in res and res[k] is not None]
        if not present:
            continue
        lines.append(f"## {header}")
        for k in present:
            lines.append(f"- **{k}**: {res[k]}")
        lines.append("")
    return "\n".join(lines) + "\n"
