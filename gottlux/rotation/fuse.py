"""
fuse.py  --  Dual-EBS fusion + mixed-FOV range cross-check (rotation mode).
Both cameras share epoch/Hall-sync/azimuth, so their world bearings are directly
comparable. Gate the wider/noisier cam to the cleaner cam's bearing track.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fuse_trajectories(traj_a, traj_b, out_path, name_a="cam1 narrow", name_b="cam0 wide",
                      gate_deg=15.0, bearing_offset_deg=0.0):
    """traj_a = reference (cleaner) track; traj_b gated to it. `bearing_offset_deg`
    is the measured cam_b-cam_a co-registration constant, subtracted from cam_b
    bearings to align the two sensors. Returns summary dict."""
    if not traj_a or not traj_b:
        return {}
    at, aaz, arng, ael = traj_a["t"], traj_a["azimuth_deg"], traj_a["range_m"], traj_a["elev_deg"]
    bt, brng, bel = traj_b["t"], traj_b["range_m"], traj_b["elev_deg"]
    baz = np.mod(np.asarray(traj_b["azimuth_deg"]) - bearing_offset_deg, 360.0)  # apply calibration
    o = np.argsort(at)
    aaz_u = np.unwrap(np.deg2rad(aaz[o]))
    ref = np.interp(bt, at[o], aaz_u)
    d = np.angle(np.exp(1j * (np.deg2rad(baz) - ref)))
    gate = np.abs(np.rad2deg(d)) < gate_deg

    rng_a = float(np.nanmedian(arng)) if np.isfinite(arng).any() else np.nan
    rng_b = float(np.nanmedian(brng[gate])) if gate.any() and np.isfinite(brng[gate]).any() else np.nan

    fig, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    ax[0].scatter(at, aaz, s=16, c="b", label=name_a)
    ax[0].scatter(bt[gate], baz[gate], s=16, c="orange", marker="x", label=f"{name_b} (gated)")
    ax[0].set_ylabel("bearing [deg]")
    ax[0].set_title(f"Dual-EBS bearing agreement (co-registration offset {bearing_offset_deg:+.2f} deg applied)")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].scatter(at, ael, s=16, c="g", label=name_a)
    ax[1].scatter(bt[gate], bel[gate], s=16, c="olive", marker="x", label=f"{name_b} (gated)")
    ax[1].set_ylabel("elevation [deg]"); ax[1].set_title("Elevation"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].scatter(at, arng, s=16, c="m", label=f"{name_a} (med {rng_a:.1f} m)")
    ax[2].scatter(bt[gate], brng[gate], s=16, c="purple", marker="x", label=f"{name_b} gated (med {rng_b:.1f} m)")
    ax[2].set_ylabel("range [m]"); ax[2].set_xlabel("time [s]"); ax[2].set_title("Mixed-FOV range cross-check"); ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    # median SIGNED residual (systematic alignment after calibration; ~0 if corrected)
    resid_sys = float(np.median(np.rad2deg(d[gate]))) if gate.any() else float("nan")
    return {"gated_kept": int(gate.sum()), "gated_total": int(len(bt)),
            "applied_offset_deg": round(float(bearing_offset_deg), 3),
            "residual_systematic_deg": round(resid_sys, 3),
            "range_a_m": rng_a, "range_b_m": rng_b}
