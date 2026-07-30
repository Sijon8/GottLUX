"""
panorama_video.py  --  Time-filling panorama sweep video.

Renders the 360 de-rotated panorama being PAINTED IN over time: a vertical sweep
line (like a radar pointer) tracks the current payload azimuth and streaks across
the world-azimuth axis, and the panorama fills in cumulatively with the events
captured up to that instant. A fun + informative way to see the environment build
up rotation by rotation. (Rotation mode; in staring mode it fills without a sweep.)
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from gottlux.rotation.viz.panorama import _world_coords


def render_panorama_video(ev, cfg, out_path, tel=None, az_bins=720, y_bins=None,
                          title="Panorama sweep"):
    H = cfg.sensor_h
    if y_bins is None: y_bins = min(int(H), 720)         # match the sensor height (any size)
    az, y = _world_coords(ev, cfg, tel)
    t = np.asarray(ev["t"]) / 1e6
    col = np.clip((az / 360.0 * az_bins).astype(np.int64), 0, az_bins - 1)
    row = np.clip((y / H * y_bins).astype(np.int64), 0, y_bins - 1)

    t0 = 0.0 if cfg.rs_t0 is None else cfg.rs_t0
    t1 = float(t.max()) if cfg.rs_t1 is None else cfg.rs_t1
    frame_dt = max(cfg.pv_frame_dt, 1e-3)
    edges = np.arange(t0, t1, frame_dt)

    # final-frame color scale (stable across the animation)
    full = np.zeros((y_bins, az_bins), np.float32)
    np.add.at(full, (row, col), 1.0)
    fimg = np.log1p(full)
    vmax = np.percentile(fimg[fimg > 0], 99) if (fimg > 0).any() else 1.0

    fig = plt.figure(figsize=(15, 5.2), facecolor="w")
    ax = fig.add_subplot(111)
    writer = imageio.get_writer(out_path, fps=cfg.pv_fps, codec="libx264",
                                quality=8, ffmpeg_log_level="error")
    buf = np.zeros((y_bins, az_bins), np.float32)
    lo = np.searchsorted(t, edges)
    hi = np.searchsorted(t, edges + frame_dt)
    try:
        for i in range(len(edges)):
            tc = edges[i] + frame_dt / 2
            a, b = lo[i], hi[i]
            if b > a:
                np.add.at(buf, (row[a:b], col[a:b]), 1.0)
            ax.cla()
            ax.imshow(np.log1p(buf), origin="lower", aspect="auto", cmap="inferno",
                      extent=[0, 360, 0, H], vmax=vmax)
            if tel is not None:
                sweep = float(tel.azimuth_at(tc))
                ax.axvline(sweep, color=(0, 1, 0), lw=2.0, alpha=0.9)
            ax.set_xlim(0, 360); ax.set_xticks(np.arange(0, 361, 30))
            ax.set_ylim(0, H); ax.invert_yaxis()
            ax.set_xlabel("world azimuth [deg]"); ax.set_ylabel("sensor Y [px]")
            ax.set_title(f"{title}   t = {tc:6.2f} s")
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    finally:
        writer.close(); plt.close(fig)
    return out_path
