"""
rate_surface.py  --  "Activity terrain" video of an event stream.

Renders the event stream as an ELEVATED POINT CLOUD: X,Y = the physical sensor
layout, Z = local event RATE (events/second) at each point's neighbourhood, color
also = rate. A decaying buffer gives a smooth, evolving terrain. Mode-agnostic:
works identically for rotating or staring captures (it is just rate-over-sensor).

Output: an MP4. Reuses one Matplotlib 3D axes across frames for speed; frames are
grabbed from the Agg canvas and encoded with imageio's bundled ffmpeg.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def run_rate_surface(ev, cfg, out_path, title="EBS rate terrain"):
    W, H = ev["width"], ev["height"]
    G = int(cfg.rs_grid)
    sx, sy = W / G, H / G
    x = np.asarray(ev["x"]); y = np.asarray(ev["y"]); t = np.asarray(ev["t"]) / 1e6
    dt = cfg.rs_accum_dt
    t0 = 0.0 if cfg.rs_t0 is None else cfg.rs_t0
    t1 = float(t.max()) if cfg.rs_t1 is None else cfg.rs_t1

    # cell index per event
    gx = np.clip((x / sx).astype(np.int64), 0, G - 1)
    gy = np.clip((y / sy).astype(np.int64), 0, G - 1)

    tail = max(cfg.rs_tail_sec, dt)
    edges = np.arange(t0, t1, dt)
    rng = np.random.default_rng(0)

    def cell_rate(lo, hi):
        """events/sec per GxG cell over the trailing window [lo:hi]."""
        c = np.zeros((G, G), np.float32)
        if hi > lo:
            np.add.at(c, (gy[lo:hi], gx[lo:hi]), 1.0)
        return c / tail

    # global Z scale from a high percentile of trailing-window cell rates (sampled)
    samp = []
    for i in range(0, len(edges), max(1, len(edges) // 60)):
        tc = edges[i] + dt / 2
        lo = int(np.searchsorted(t, tc - tail)); hi = int(np.searchsorted(t, tc))
        r = cell_rate(lo, hi)
        if r.max() > 0:
            samp.append(r.max())
    zmax = max(float(np.percentile(samp, 92)) if samp else 1.0, 1.0)

    fig = plt.figure(figsize=(9, 6.5), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    writer = imageio.get_writer(out_path, fps=cfg.rs_fps, codec="libx264",
                                quality=8, ffmpeg_log_level="error")
    try:
        for i in range(len(edges)):
            tc = edges[i] + dt / 2
            lo = int(np.searchsorted(t, tc - tail)); hi = int(np.searchsorted(t, tc))
            rate = cell_rate(lo, hi)
            # elevated point cloud: events in the trailing window, lifted to cell rate
            ex, ey = x[lo:hi], y[lo:hi]
            if len(ex) > cfg.rs_max_points:
                pick = rng.choice(len(ex), cfg.rs_max_points, replace=False)
                ex, ey = ex[pick], ey[pick]
            cx = np.clip((ex / sx).astype(np.int64), 0, G - 1)
            cy = np.clip((ey / sy).astype(np.int64), 0, G - 1)
            ez = rate[cy, cx]

            ax.cla()
            if len(ex):
                ax.scatter(ex, ey, ez, c=ez, cmap="turbo", s=6, vmin=0, vmax=zmax,
                           depthshade=False, linewidths=0)
            ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_zlim(0, zmax)
            ax.set_xlabel("sensor X [px]"); ax.set_ylabel("sensor Y [px]")
            ax.set_zlabel("events / sec")
            ax.view_init(elev=30, azim=-58)
            ax.set_title(f"{title}\nt = {tc:6.2f} s   (Z = local event rate, ev/s; {tail*1e3:.0f} ms tail)")
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            writer.append_data(frame)
    finally:
        writer.close(); plt.close(fig)
    return out_path
