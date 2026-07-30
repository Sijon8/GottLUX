"""
mti.py  --  Moving-Target-Indication maps for faint, slow targets.

After de-rotation, a STATIC scene feature sits at a fixed world azimuth/elevation
for all time; a MOVING target traces a smooth track. Plotting the background-
subtracted event density as (time x world_azimuth) and (time x elevation) turns
the static residual into horizontal bands and the moving target into a diagonal
streak -- the classic MTI view. This is the key diagnostic for the faint airliner
where per-frame blob detection fails.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def derotated(ev, keep, cfg, tel):
    W, H = cfg.sensor_w, cfg.sensor_h
    dpp = cfg.fov_deg / W
    x = np.asarray(ev["x"])[keep].astype(np.float64)
    y = np.asarray(ev["y"])[keep].astype(np.float64)
    t = np.asarray(ev["t"])[keep] / 1e6
    if cfg.mode == "rotation" and tel is not None:
        pan = np.rad2deg(np.interp(t - tel.offset, tel.t, tel.azimuth_unwrapped()))
        az = np.mod(pan + cfg.az_sign * (x - W / 2) * dpp, 360.0)
    else:
        az = np.mod(cfg.az_sign * (x - W / 2) * dpp, 360.0)
    elev = (H / 2 - y) * dpp
    return t, az, elev


def render_mti(ev, keep, cfg, out_path, tel=None, t_bins=600, az_bins=720,
               title="MTI: de-rotated azimuth vs time"):
    t, az, elev = derotated(ev, keep, cfg, tel)
    tmax = float(t.max()) if len(t) else 1.0
    # (time x azimuth)
    Hta, te, ae = np.histogram2d(t, az, bins=[t_bins, az_bins], range=[[0, tmax], [0, 360]])
    # subtract each azimuth column's time-median => removes the static horizontal
    # bands, leaving moving (time-localized) structure = the target track.
    Ht_res = Hta - np.median(Hta, axis=0, keepdims=True)
    e_lo, e_hi = (np.percentile(elev, [1, 99]) if len(elev) else (-30, 30))
    Hte, _, ee = np.histogram2d(t, elev, bins=[t_bins, 200], range=[[0, tmax], [e_lo, e_hi]])
    Hte_res = Hte - np.median(Hte, axis=0, keepdims=True)

    fig, ax = plt.subplots(2, 1, figsize=(14, 9))
    im0 = ax[0].imshow(np.clip(Ht_res, 0, None).T, origin="lower", aspect="auto",
                       cmap="inferno", extent=[0, tmax, 0, 360],
                       vmax=np.percentile(Ht_res[Ht_res > 0], 99) if (Ht_res > 0).any() else 1)
    ax[0].set_ylabel("world azimuth [deg]"); ax[0].set_yticks(np.arange(0, 361, 45))
    ax[0].set_title(f"{title}  (static bands removed; moving target = diagonal streak)")
    fig.colorbar(im0, ax=ax[0], pad=0.01, label="excess events")
    im1 = ax[1].imshow(np.clip(Hte_res, 0, None).T, origin="lower", aspect="auto",
                       cmap="inferno", extent=[0, tmax, e_lo, e_hi],
                       vmax=np.percentile(Hte_res[Hte_res > 0], 99) if (Hte_res > 0).any() else 1)
    ax[1].set_ylabel("elevation [deg]"); ax[1].set_xlabel("time [s]")
    ax[1].set_title("MTI: elevation vs time")
    fig.colorbar(im1, ax=ax[1], pad=0.01, label="excess events")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    return out_path


if __name__ == "__main__":
    import sys, glob, os
    from gottlux.config import Config, CAMERA_FOV_DEG
    from gottlux.rotation import io_evt21, background, detect
    from gottlux.io.telemetry import Telemetry

    base = sys.argv[1]; prefix = sys.argv[2] if len(sys.argv) > 2 else "cam1"
    mp = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    cfg = Config(mode="rotation"); cfg.fov_deg = CAMERA_FOV_DEG[prefix]; cfg.min_pixels = mp
    raw = glob.glob(os.path.join(base, f"{prefix}*.raw"))[0]
    csv = (glob.glob(os.path.join(base, "data_*.csv")) or glob.glob(os.path.join(base, "*.csv")))[0]
    ev = io_evt21.load(raw); tel = Telemetry(csv)
    t = np.asarray(ev["t"]) / 1e6; tel.refine_offset_to_events(t)
    hot = background.hot_pixel_mask(ev, cfg.hot_pixel_pct)
    ref = background.build_reference(ev, tel, float(tel.hall_t[0] + tel.offset), n_phase=cfg.n_phase)
    drop = background.rotation_drop_mask(ev, tel, ref, cfg.n_phase, hot)
    keep, _ = detect.isolate_target(ev, drop, cfg.accum_dt, cfg.min_pixels)
    out = os.path.join(base, f"{prefix}_mti.png")
    render_mti(ev, keep, cfg, out, tel, title=f"{prefix} MTI")
    print("saved", out, "| kept", int(keep.sum()))
