"""
diagnostics.py  --  Static diagnostic figures: detection summary + clean target
3D point cloud.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def tracks_plot(tracks, out_path, tel=None, title=""):
    """Bearing & elevation vs time, colored by track id."""
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    cmap = plt.cm.tab10
    for tr in tracks:
        c = cmap(tr["id"] % 10)
        ax[0].scatter(tr["t"], tr["azimuth_deg"], s=14, color=c, label=f"track {tr['id']}")
        ax[1].scatter(tr["t"], tr["elev_deg"], s=14, color=c)
    ax[0].set_ylabel("bearing [deg]"); ax[0].set_title(f"{title} tracks"); ax[0].grid(alpha=.3)
    if len(tracks) <= 10:
        ax[0].legend(fontsize=8, ncol=2)
    ax[1].set_ylabel("elevation [deg]"); ax[1].set_xlabel("time [s]"); ax[1].grid(alpha=.3)
    if tel is not None:
        for a in ax:
            for hl in tel.hall_t + tel.offset:
                a.axvline(hl, color="r", ls="--", lw=0.3, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def event_rate_plot(ev, keep, cfg, out_path, tel=None, title="", bin_s=0.02):
    """Standalone event-rate vs time: raw stream + isolated/kept stream."""
    t = np.asarray(ev["t"]) / 1e6
    edges = np.arange(0, t.max() + bin_s, bin_s); ctr = 0.5 * (edges[:-1] + edges[1:])
    raw = np.histogram(t, bins=edges)[0] / bin_s
    iso = np.histogram(t[keep], bins=edges)[0] / bin_s
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(ctr, raw, "k", lw=0.7, label=f"raw ({np.asarray(ev['t']).size:,} ev)")
    ax.plot(ctr, iso, "r", lw=0.9, label=f"isolated ({int(keep.sum()):,} ev)")
    if tel is not None:
        for hl in tel.hall_t + tel.offset:
            ax.axvline(hl, color="b", ls="--", lw=0.4, alpha=0.35)
    ax.set_xlabel("time [s]"); ax.set_ylabel("events / s")
    ax.set_title(f"{title} event rate"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def summary_plot(ev, keep, traj, cfg, out_path, tel=None, title=""):
    t = np.asarray(ev["t"]) / 1e6
    kt = t[keep]
    bin_s = 0.02
    edges = np.arange(0, t.max() + bin_s, bin_s); ctr = 0.5 * (edges[:-1] + edges[1:])
    raw = np.histogram(t, bins=edges)[0] / bin_s
    iso = np.histogram(kt, bins=edges)[0] / bin_s

    has = len(traj) > 0
    fig, ax = plt.subplots(4, 1, figsize=(12, 11))
    ax[0].plot(ctr, raw, "k", lw=0.6, label="raw")
    ax[0].plot(ctr, iso, "r", lw=0.9, label="isolated target")
    ax[0].set_ylabel("ev/s"); ax[0].legend(); ax[0].set_title(f"{title} event rate"); ax[0].grid(alpha=.3)
    if has:
        ax[1].scatter(traj["t"], traj["azimuth_deg"], s=12, c="b")
        ax[1].set_ylabel("bearing [deg]"); ax[1].set_title("bearing track"); ax[1].grid(alpha=.3)
        ax[2].scatter(traj["t"], traj["elev_deg"], s=12, c="g")
        ax[2].set_ylabel("elevation [deg]"); ax[2].set_title("elevation track"); ax[2].grid(alpha=.3)
        if np.isfinite(traj["range_m"]).any():
            rm = float(np.nanmedian(traj["range_m"]))
            ax[3].scatter(traj["t"], traj["range_m"], s=12, c="m")
            ax[3].axhline(rm, color="k", ls=":", lw=1, label=f"median {rm:.1f} m"); ax[3].legend()
            ax[3].set_ylabel("range [m]"); ax[3].set_title(f"pinhole range ({cfg.target_diag_m*1e3:.0f} mm, FOV {cfg.fov_deg:.0f} deg)")
        ax[3].set_xlabel("time [s]"); ax[3].grid(alpha=.3)
    if tel is not None:
        for a in ax:
            for hl in tel.hall_t + tel.offset:
                a.axvline(hl, color="r", ls="--", lw=0.4, alpha=0.4)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path


def target_pointcloud(kx, ky, kt, kaz, cfg, out_path, title=""):
    R = 380 - ky * ((380 - 100) / cfg.sensor_h)
    th = np.deg2rad(kaz) - np.pi / 2
    X = R * np.cos(th); Y = R * np.sin(th); Z = kx - cfg.sensor_w / 2
    fig = plt.figure(figsize=(11, 9)); ax = fig.add_subplot(111, projection="3d")
    s = ax.scatter(X, Y, Z, c=kt, cmap="cool", s=2)
    ax.set_xlabel("Polar X"); ax.set_ylabel("Polar Y"); ax.set_zlabel("Sensor Breadth (Z)")
    ax.set_title(f"{title} isolated target 3D spatio-temporal cloud")
    fig.colorbar(s, ax=ax, label="t [s]", shrink=0.6)
    fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path
