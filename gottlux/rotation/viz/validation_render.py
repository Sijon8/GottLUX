"""
validation_render.py  --  3-panel proof video:
  [RAW] all events | [TARGET] isolated events + tracking bbox + callout | [RADAR] bearing track
Pure numpy + Pillow drawing, imageio(+ffmpeg) encoding. Decaying tails so target
streaks persist. Works for rotation (bearing radar = true azimuth) and staring
(radar = relative bearing within FOV).
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

try:
    FONT = ImageFont.truetype("arial.ttf", 13); FONT_S = ImageFont.truetype("arial.ttf", 11)
except Exception:
    FONT = ImageFont.load_default(); FONT_S = FONT


def _radar(S, az, tarr, rng, tnow, sweep_az, rmax):
    im = Image.fromarray(np.zeros((S, S, 3), np.uint8)); dr = ImageDraw.Draw(im); c = S // 2
    for rr in (0.33, 0.66, 1.0):
        rad = rr * (S * 0.46)
        dr.ellipse([c - rad, c - rad, c + rad, c + rad], outline=(40, 80, 40))
        dr.text((c + 2, c - rad), f"{rr*rmax:.0f}m", fill=(60, 110, 60), font=FONT_S)
    past = tarr <= tnow
    for a, rg in zip(az[past], np.clip(rng[past], 0, rmax)):
        th = np.deg2rad(a) - np.pi / 2; rad = (rg / rmax) * (S * 0.46)
        px, py = c + rad * np.cos(th), c + rad * np.sin(th)
        dr.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 60, 200))
    th = np.deg2rad(sweep_az) - np.pi / 2
    dr.line([c, c, c + (S * 0.46) * np.cos(th), c + (S * 0.46) * np.sin(th)], fill=(0, 230, 0), width=2)
    return np.asarray(im)


def run_validation(ev, keep, traj, cfg, out_path, tel=None, title="EBS"):
    W, H = ev["width"], ev["height"]
    x = np.asarray(ev["x"]); y = np.asarray(ev["y"]); t = np.asarray(ev["t"]) / 1e6
    dt = cfg.vr_accum_dt
    t0 = 0.0 if cfg.rs_t0 is None else cfg.rs_t0
    t1 = float(t.max()) if cfg.rs_t1 is None else cfg.rs_t1
    has = len(traj) > 0
    az = traj["azimuth_deg"] if has else np.array([])
    tt = traj["t"] if has else np.array([])
    rng = traj["range_m"] if has else np.array([])
    rmax = float(np.nanpercentile(rng, 90) * 1.2) if has and np.isfinite(rng).any() else 10.0
    deg_per_px = cfg.fov_deg / W
    T_rot = tel.T_rot if tel is not None else 1.0
    omega = tel.omega_deg_s if tel is not None else 0.0

    edges = np.arange(t0, t1, dt)
    lo = np.searchsorted(t, edges); hi = np.searchsorted(t, edges + dt)
    decay = max(0.0, 1.0 - dt / max(cfg.vr_tail_sec, dt))
    rbuf = np.zeros((H, W), np.float32); tbuf = np.zeros((H, W), np.float32)
    gap, S = 8, H
    w = imageio.get_writer(out_path, fps=cfg.vr_fps, codec="libx264", quality=8, ffmpeg_log_level="error")
    try:
        for i in range(len(edges)):
            l, h = lo[i], hi[i]; tc = edges[i] + dt / 2
            sweep = (tel.azimuth_at(tc) if tel is not None else 0.0)
            rbuf *= decay; tbuf *= decay
            rbuf[y[l:h], x[l:h]] = 255.0
            km = keep[l:h]
            tbuf[y[l:h][km], x[l:h][km]] = 255.0
            g = rbuf.astype(np.uint8); raw_p = np.stack([g, g, g], -1)
            c = tbuf.astype(np.uint8); tgt_p = np.stack([np.zeros_like(c), c, c], -1)
            info = "SCANNING"
            if has:
                j = int(np.argmin(np.abs(tt - tc)))
                if abs(tt[j] - tc) < T_rot * 0.6:
                    cx, cy = traj["cx"][j], traj["cy"][j]
                    dx, dy = traj["dx"][j], traj["dy"][j]
                    im = Image.fromarray(tgt_p); d = ImageDraw.Draw(im)
                    d.rectangle([cx - max(dx, 8), cy - max(dy, 8), cx + max(dx, 8), cy + max(dy, 8)], outline=(0, 255, 0))
                    tgt_p = np.asarray(im)
                    rs = f"{rng[j]:4.1f}m" if np.isfinite(rng[j]) else "  -- "
                    info = f"LOCK  BRG {az[j]:05.1f}  RNG {rs}  EL {traj['elev_deg'][j]:+.1f}"
            rad_p = _radar(S, az, tt, rng, tc, sweep, rmax)
            spacer = np.zeros((H, gap, 3), np.uint8)
            row = np.concatenate([raw_p, spacer, tgt_p, spacer, rad_p], axis=1)
            frame = np.concatenate([np.zeros((34, row.shape[1], 3), np.uint8), row], axis=0)
            im = Image.fromarray(frame); d = ImageDraw.Draw(im)
            d.text((6, 4), f"EBS-TOOLS  {title}  t={tc:6.2f}s  omega={omega:.0f}deg/s  {info}", fill=(0, 255, 255), font=FONT)
            d.text((6, 38), "RAW NEUROMORPHIC", fill=(180, 180, 180), font=FONT_S)
            d.text((W + gap + 6, 38), "ISOLATED TARGET", fill=(0, 230, 230), font=FONT_S)
            d.text((2 * (W + gap) + 6, 38), "BEARING RADAR", fill=(0, 230, 0), font=FONT_S)
            w.append_data(np.asarray(im))
    finally:
        w.close()
    return out_path
