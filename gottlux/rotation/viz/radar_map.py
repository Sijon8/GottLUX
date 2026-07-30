"""
radar_map.py  --  360-degree radar visualizations for a rotating sensor.

Two products:
  render_radar_map()    static high-res polar TARGET ACQUISITION MAP:
                        theta = bearing, r = range (m), color = altitude Z (m).
                        Tactical green-on-black, range rings, colorbar. (V26 Fig6)
  render_radar_sweep()  animated tactical radar: a green "wiper" sweeps with the
                        payload azimuth and target detections light up / persist
                        at their (bearing, range). (V26 in-video radar panel)
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def _polar_axes(fig, rmax):
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor((0.05, 0.05, 0.05))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)               # bearing increases clockwise
    ax.set_ylim(0, rmax)
    ax.grid(color=(0.3, 0.8, 0.3), alpha=0.5)
    ax.tick_params(colors="w")
    ax.spines["polar"].set_color((0.3, 0.8, 0.3))
    return ax


def render_radar_map(traj, cfg, out_path, title="360° Target Acquisition Map"):
    if not traj:
        return None
    az = np.asarray(traj["azimuth_deg"]); rng = np.asarray(traj["range_m"])
    z = np.asarray(traj.get("altitude_z_m", np.full_like(az, np.nan)))
    finite = np.isfinite(rng)
    if not finite.any():           # no ranging -> use detection order as pseudo-range
        rng = np.arange(len(az), dtype=float) + 1.0; finite = np.ones_like(az, bool)
    az, rng, z = az[finite], rng[finite], z[finite]
    rmax = float(np.nanpercentile(rng, 95) * 1.15) if len(rng) else 1.0

    fig = plt.figure(figsize=(8, 8), facecolor="k")
    ax = _polar_axes(fig, max(rmax, 1.0))
    cvals = z if np.isfinite(z).any() else traj["t"][finite]
    sc = ax.scatter(np.deg2rad(az), rng, c=cvals, cmap="jet", s=90,
                    edgecolors="w", linewidths=0.5)
    cb = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.8)
    cb.ax.yaxis.set_tick_params(color="w"); plt.setp(cb.ax.get_yticklabels(), color="w")
    cb.set_label("Estimated Altitude Z [m]" if np.isfinite(z).any() else "time [s]", color="w")
    ax.set_title(title, color="w", fontsize=14, pad=18)
    fig.text(0.5, 0.02, "Optomechanical Bearing & Kinematic Ranging Fusion",
             color="c", ha="center", fontsize=10)
    fig.savefig(out_path, dpi=200, facecolor="k"); plt.close(fig)
    return out_path


def render_radar_sweep(traj, cfg, out_path, tel=None, title="Radar Sweep"):
    if not traj:
        return None
    az = np.asarray(traj["azimuth_deg"]); rng = np.asarray(traj["range_m"]).copy()
    tt = np.asarray(traj["t"]); z = np.asarray(traj.get("altitude_z_m", np.full_like(az, np.nan)))
    if not np.isfinite(rng).any():
        rng = np.full_like(az, 1.0)
    rmax = float(np.nanpercentile(rng[np.isfinite(rng)], 95) * 1.2) if np.isfinite(rng).any() else 1.0
    rmax = max(rmax, 1.0)

    dt = cfg.vr_accum_dt
    t0 = 0.0 if cfg.rs_t0 is None else cfg.rs_t0
    t1 = float(tt.max()) if cfg.rs_t1 is None else cfg.rs_t1
    edges = np.arange(t0, t1, dt)

    fig = plt.figure(figsize=(7, 7), facecolor="k")
    w = imageio.get_writer(out_path, fps=cfg.vr_fps, codec="libx264", quality=8, ffmpeg_log_level="error")
    try:
        for tc in edges + dt / 2:
            sweep = (tel.azimuth_at(tc) if tel is not None
                     else (az[np.argmin(np.abs(tt - tc))] if len(az) else 0.0))
            fig.clf()
            ax = _polar_axes(fig, rmax)
            seen = tt <= tc
            if seen.any():
                fin = seen & np.isfinite(rng)
                cvals = z[fin] if np.isfinite(z[fin]).any() else tt[fin]
                ax.scatter(np.deg2rad(az[fin]), rng[fin], c=cvals, cmap="jet",
                           s=60, edgecolors="w", linewidths=0.4)
            # green sweep wiper
            ax.plot([np.deg2rad(sweep), np.deg2rad(sweep)], [0, rmax], color=(0, 0.9, 0), lw=2.5)
            ax.set_title(f"{title}   t={tc:5.2f}s   BRG {sweep:05.1f}", color="w", fontsize=12, pad=14)
            fig.canvas.draw()
            w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    finally:
        w.close(); plt.close(fig)
    return out_path
