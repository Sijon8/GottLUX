"""
metrics.py  --  Quantitative figures of merit for a rotating-EBS single-sensor
volumetric sensing study.

Reports:
  - Angular COVERAGE: swept solid angle (sr) vs a static EBS, and fraction of sphere.
  - REVISIT interval (temporal sampling of the volume).
  - LOCALIZATION accuracy: bearing standard error, elevation/range track stats.
  - DETECTION rate across revolutions.
"""
from __future__ import annotations
import numpy as np


def _solid_angle(az_span_deg, elev_lo_deg, elev_hi_deg):
    """Solid angle (sr) of an azimuth wedge over an elevation band:
       Omega = Delta_az * (sin(elev_hi) - sin(elev_lo))."""
    return np.deg2rad(az_span_deg) * (np.sin(np.deg2rad(elev_hi_deg)) - np.sin(np.deg2rad(elev_lo_deg)))


def compute(ev, keep, traj, P, cfg, tel):
    W, H = cfg.sensor_w, cfg.sensor_h
    dpp = cfg.fov_deg / W
    y = np.asarray(ev["y"])
    elev = (H / 2 - y) * dpp
    e_lo, e_hi = np.percentile(elev, [1, 99])

    az_span = 360.0 if (cfg.mode == "rotation" and tel is not None) else cfg.fov_deg
    omega = _solid_angle(az_span, e_lo, e_hi)
    omega_static = _solid_angle(cfg.fov_deg, -cfg.fov_deg / 2, cfg.fov_deg / 2)
    res = {
        "coverage_azimuth_deg": round(float(az_span), 1),
        "coverage_elev_band_deg": round(float(e_hi - e_lo), 1),
        "swept_solid_angle_sr": round(float(omega), 3),
        "sphere_fraction_pct": round(float(100 * omega / (4 * np.pi)), 1),
        "static_ebs_solid_angle_sr": round(float(omega_static), 3),
        "coverage_gain_vs_static": round(float(omega / omega_static), 1),
    }
    if tel is not None:
        res["revisit_interval_s"] = round(float(tel.T_rot), 3)
        res["update_rate_hz"] = round(float(1 / tel.T_rot), 3)
        res["n_revolutions"] = int(tel.n_revolutions)

    if len(P):
        rng = P[:, 5]
        res.update({
            "target_passes_detected": int(len(P)),
            "bearing_standard_error_deg": round(float(np.median(P[:, 2])), 3),
            "elevation_track_deg": f"{np.median(P[:,3]):.1f} (sd {np.std(P[:,3]):.1f})",
            "range_median_m": round(float(np.nanmedian(rng)), 2) if np.isfinite(rng).any() else None,
            "range_iqr_m": (f"{np.nanpercentile(rng,25):.2f}-{np.nanpercentile(rng,75):.2f}"
                            if np.isfinite(rng).any() else None),
        })
        if tel is not None:
            # detection rate over revolutions the target could be seen
            res["detection_rate_passes_per_rev"] = round(len(P) / max(tel.n_revolutions, 1), 2)
    return res


def to_markdown(res, tag):
    lines = [f"# EBS Tools — Quantitative Metrics  ({tag})", "",
             "## Coverage (single rotating sensor vs static EBS)"]
    cov = ["coverage_azimuth_deg", "coverage_elev_band_deg", "swept_solid_angle_sr",
           "sphere_fraction_pct", "static_ebs_solid_angle_sr", "coverage_gain_vs_static"]
    for k in cov:
        if k in res:
            lines.append(f"- **{k}**: {res[k]}")
    lines += ["", "## Temporal sampling"]
    for k in ("revisit_interval_s", "update_rate_hz", "n_revolutions"):
        if k in res:
            lines.append(f"- **{k}**: {res[k]}")
    lines += ["", "## Target detection & 3D localization"]
    for k in ("target_passes_detected", "detection_rate_passes_per_rev",
              "bearing_standard_error_deg", "elevation_track_deg", "range_median_m", "range_iqr_m"):
        if k in res:
            lines.append(f"- **{k}**: {res[k]}")
    return "\n".join(lines) + "\n"
