"""
panorama.py  --  360-degree de-rotated environmental map (the volumetric-sensing
centerpiece).

A single rotating EBS, de-rotated with the azimuth telemetry, reconstructs the
whole surrounding scene: every event is mapped to a WORLD coordinate
    world_azimuth = azimuth(t) + sign*(x - W/2)*deg_per_px      (0..360)
    elevation     = (H/2 - y)*deg_per_px                         (deg off boresight)
and accumulated into an (elevation x azimuth) image. Static structure recurs at
the same world azimuth every revolution and accumulates sharply; transient
targets smear. Optionally overlays the target track (per-pass centroids) so the
environment map and the 3D target localization share one spherical frame.

Works in rotation mode (true 360 panorama) and degrades gracefully in staring
mode (relative bearing strip).
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _world_coords(ev, cfg, tel, idx=None):
    W, H = cfg.sensor_w, cfg.sensor_h
    dpp = cfg.fov_deg / W
    x = np.asarray(ev["x"]); y = np.asarray(ev["y"]); t = np.asarray(ev["t"]) / 1e6
    if idx is not None:
        x, y, t = x[idx], y[idx], t[idx]
    x = x.astype(np.float64); y = y.astype(np.float64)
    if cfg.mode == "rotation" and tel is not None:
        pan = np.rad2deg(np.interp(t - tel.offset, tel.t, tel.azimuth_unwrapped()))
        az = np.mod(pan + cfg.az_sign * (x - W / 2) * dpp, 360.0)
    else:
        az = np.mod(cfg.az_sign * (x - W / 2) * dpp, 360.0)
    return az, y          # vertical axis = raw sensor row (px), not elevation deg


def render_panorama(ev, cfg, out_path, tel=None, az_bins=1440, y_bins=None,
                    title="360 De-rotated Environmental Map", subsample=4_000_000):
    H = cfg.sensor_h
    if y_bins is None: y_bins = min(int(H), 1080)        # match the sensor height (any size)
    az, y = _world_coords(ev, cfg, tel)
    if len(az) > subsample:                      # cap for speed; uniform subsample
        sel = np.random.default_rng(0).choice(len(az), subsample, replace=False)
        az, y = az[sel], y[sel]
    H2d, ye, xe = np.histogram2d(y, az, bins=[y_bins, az_bins], range=[[0, H], [0, 360]])
    img = np.log1p(H2d)

    fig, ax = plt.subplots(figsize=(15, 5.2), facecolor="w")
    im = ax.imshow(img, origin="lower", aspect="auto", cmap="inferno",
                   extent=[0, 360, 0, H],
                   vmax=np.percentile(img[img > 0], 99) if (img > 0).any() else 1)
    cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("log(1 + event count)")
    ax.set_xlabel("world azimuth [deg]  (full 360 sweep)")
    ax.set_ylabel("sensor Y [px]")
    ax.set_title(title)
    ax.set_xlim(0, 360); ax.set_xticks(np.arange(0, 361, 30))
    ax.set_ylim(0, H); ax.invert_yaxis()        # row 0 at top, like the sensor
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path
