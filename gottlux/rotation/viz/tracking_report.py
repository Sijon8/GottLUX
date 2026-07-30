"""
tracking_report.py  --  consistent, paper-ready output for each tracker, per regime.

Measurements are intentionally minimal and mode-specific:

  ROTATION : bearing-to-target, relative-distance proxy, elevation angle.
  STARING  : relative-distance proxy, radial velocity (closer/farther), blade-flutter FFT.

Always emitted: a video with the tracker boxes overlaid, an enriched per-point CSV (which
STORES the box sizes + relative distance), and a short effectiveness report. Distance is a
unitless proxy — calibrate it to metres per flight with scripts/calibrate_range.py.

Works on any tracker's output: missing centroids / boxes are recovered from the detector
trajectory by nearest-in-time association.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gottlux.rotation import track_analysis as ta

_C = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _assoc(track, traj, key, tol_s=0.05):
    t = np.asarray(track["t"], float); n = len(t)
    if key in track:
        return np.asarray(track[key], float)
    if not traj or key not in traj or not len(traj.get("t", [])):
        return np.full(n, np.nan)
    tt = np.asarray(traj["t"], float); vv = np.asarray(traj[key], float)
    out = np.full(n, np.nan)
    for i, ti in enumerate(t):
        j = int(np.argmin(np.abs(tt - ti)))
        if abs(tt[j] - ti) <= tol_s:
            out[i] = vv[j]
    return out


def _analyze(tracks, traj, ev, cfg, mode):
    recs = []
    for tr in tracks:
        t = np.asarray(tr["t"], float)
        az = np.asarray(tr.get("azimuth_deg", np.full_like(t, np.nan)), float)
        el = np.asarray(tr.get("elev_deg", np.full_like(t, np.nan)), float)
        dx, dy = ta.track_bbox(tr, traj)
        cx = _assoc(tr, traj, "cx"); cy = _assoc(tr, traj, "cy")
        size = ta.apparent_size_px(dx, dy, mode)
        rel = ta.relative_distance(size)
        vrad = ta.radial_velocity(t, rel) if mode != "rotation" else np.full_like(t, np.nan)
        rec = dict(id=int(tr.get("id", len(recs))), t=t, az=az, el=el, cx=cx, cy=cy,
                   dx=dx, dy=dy, size_px=size, rel_distance=rel, radial_velocity=vrad,
                   vitality=np.asarray(tr.get("vitality", np.full_like(t, np.nan)), float),
                   duration_s=float(t[-1] - t[0]) if len(t) > 1 else 0.0, n_points=len(t),
                   blade_hz=np.nan, blade_snr=0.0, blade_freqs=np.zeros(0), blade_power=np.zeros(0))
        if mode != "rotation":
            bf = ta.blade_fft(ev, dict(t=t, cx=cx, cy=cy, dx=dx, dy=dy))
            rec.update(blade_hz=bf["peak_hz"], blade_snr=bf["snr"],
                       blade_freqs=bf["freqs"], blade_power=bf["power"])
        recs.append(rec)
    return recs


def _grid(ax):
    ax.grid(True, ls="--", alpha=0.35)


def _figure_rotation(recs, out_path, tag):
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True, facecolor="w")
    fig.suptitle(f"{tag} — rotating tracking", fontsize=13, fontweight="bold")
    for r in recs:
        c = _C[r["id"] % len(_C)]
        ax[0].plot(r["t"], r["az"], "-", color=c, lw=1.3, label=f"#{r['id']}")
        ax[1].plot(r["t"], r["rel_distance"], "-", color=c, lw=1.3)
        ax[2].plot(r["t"], r["el"], "-", color=c, lw=1.3)
    ax[0].set_ylabel("bearing [deg]"); ax[0].set_title("Bearing to target"); _grid(ax[0]); ax[0].legend(fontsize=8)
    ax[1].set_ylabel("relative distance"); ax[1].set_title("Relative distance proxy (calibrate to m)"); _grid(ax[1])
    ax[2].set_ylabel("elevation [deg]"); ax[2].set_title("Elevation angle"); ax[2].set_xlabel("time [s]"); _grid(ax[2])
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


def _figure_staring(recs, out_path, tag):
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), facecolor="w")
    fig.suptitle(f"{tag} — staring tracking", fontsize=13, fontweight="bold")
    for r in recs:
        c = _C[r["id"] % len(_C)]
        ax[0].plot(r["t"], r["rel_distance"], "-", color=c, lw=1.3, label=f"#{r['id']}")
        ax[1].plot(r["t"], r["radial_velocity"], "-", color=c, lw=1.2)
    ax[0].set_ylabel("relative distance"); ax[0].set_title("Relative distance proxy (calibrate to m)")
    ax[0].set_xlabel("time [s]"); _grid(ax[0]); ax[0].legend(fontsize=8)
    ax[1].axhline(0, color="gray", lw=1); ax[1].set_ylabel("d(rel)/dt"); ax[1].set_xlabel("time [s]")
    ax[1].set_title("Radial velocity  (+ moving away,  − approaching)"); _grid(ax[1])
    ax[2].set_title("Blade-flutter FFT inside tracked box"); ax[2].set_xlabel("frequency [Hz]"); ax[2].set_ylabel("power (norm.)")
    plotted = False
    for r in recs:
        if r["blade_freqs"].size and r["blade_power"].size:
            band = (r["blade_freqs"] >= 20) & (r["blade_freqs"] <= 600)
            if band.any() and r["blade_power"][band].max() > 0:
                c = _C[r["id"] % len(_C)]
                ax[2].plot(r["blade_freqs"][band], r["blade_power"][band] / r["blade_power"][band].max(),
                           color=c, lw=1.0)
                if np.isfinite(r["blade_hz"]):
                    ax[2].axvline(r["blade_hz"], color=c, ls="--", lw=1.0,
                                  label=f"#{r['id']}: {r['blade_hz']:.0f} Hz (snr {r['blade_snr']:.1f})")
                plotted = True
    if plotted:
        ax[2].legend(fontsize=8)
    else:
        ax[2].text(0.5, 0.5, "no blade signature detected", ha="center", va="center",
                   transform=ax[2].transAxes, color="gray")
    _grid(ax[2])
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path


def _overlay_video(ev, recs, cfg, mode, out_path):
    """Event frames with tracker boxes + IDs overlaid (boxes interpolated to each frame)."""
    try:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw
    except Exception:
        return None
    t = np.asarray(ev["t"], float) / 1e6
    if not len(t) or not recs:
        return None
    W, H = int(ev["width"]), int(ev["height"])
    x = np.asarray(ev["x"]); y = np.asarray(ev["y"])
    tmax = float(t.max())
    fps = int(getattr(cfg, "vr_fps", 25) or 25)
    accum = float(getattr(cfg, "viz_accum_dt", 0.03) or 0.03)
    step = max(1.0 / fps, tmax / 750.0)               # cap ~750 frames for long captures
    scale = max(1, int(round(640 / max(W, 1))))
    Wd, Hd = (W * scale) - (W * scale) % 2, (H * scale) - (H * scale) % 2
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=None, ffmpeg_log_level="error")
    try:
        tc = accum
        while tc <= tmax + 1e-9:
            lo = int(np.searchsorted(t, tc - accum)); hi = int(np.searchsorted(t, tc))
            img = np.zeros((H, W), np.uint8)
            if hi > lo:
                np.add.at(img, (y[lo:hi], x[lo:hi]), np.uint8(60))
            im = Image.fromarray(img).convert("RGB").resize((Wd, Hd), Image.NEAREST)
            dr = ImageDraw.Draw(im)
            nactive = 0
            for r in recs:
                if len(r["t"]) < 2 or not (r["t"][0] <= tc <= r["t"][-1]):
                    continue
                nactive += 1
                cx = np.interp(tc, r["t"], r["cx"]); cy = np.interp(tc, r["t"], r["cy"])
                if not (np.isfinite(cx) and np.isfinite(cy)):
                    continue
                dxw = np.interp(tc, r["t"], np.nan_to_num(r["dx"], nan=8.0))
                dyh = np.interp(tc, r["t"], np.nan_to_num(r["dy"], nan=8.0))
                col = tuple(int(255 * v) for v in matplotlib.colors.to_rgb(_C[r["id"] % len(_C)]))
                box = [(cx - dxw / 2) * scale, (cy - dyh / 2) * scale,
                       (cx + dxw / 2) * scale, (cy + dyh / 2) * scale]
                dr.rectangle(box, outline=col, width=2)
                rel = np.interp(tc, r["t"], r["rel_distance"])
                lbl = f"#{r['id']} d~{rel:.0f}" if np.isfinite(rel) else f"#{r['id']}"
                if mode != "rotation" and np.isfinite(r["blade_hz"]):
                    lbl += f" {r['blade_hz']:.0f}Hz"
                dr.text((box[0], max(0, box[1] - 11)), lbl, fill=col)
            dr.rectangle([0, 0, Wd, 16], fill=(0, 0, 0))
            dr.text((4, 3), f"t={tc:5.2f}s  active:{nactive}  [{mode}]", fill=(255, 255, 255))
            writer.append_data(np.asarray(im))
            tc += step
    finally:
        writer.close()
    return out_path


def _write_csv(recs, mode, out_path):
    hdr = ("track_id,t_s,bearing_deg,elev_deg,apparent_size_px,rel_distance,"
           "radial_velocity,cx,cy,dx_px,dy_px,blade_hz")
    rows = []
    for r in recs:
        for j in range(r["n_points"]):
            rows.append([r["id"], r["t"][j], r["az"][j], r["el"][j], r["size_px"][j],
                         r["rel_distance"][j], r["radial_velocity"][j], r["cx"][j], r["cy"][j],
                         r["dx"][j], r["dy"][j], r["blade_hz"]])
    np.savetxt(out_path, np.array(rows) if rows else np.zeros((0, 12)),
               delimiter=",", header=hdr, comments="", fmt="%.4f")
    return out_path


def _write_metrics(summary, recs, cfg, mode, tracker, out_path, tag):
    with open(out_path, "w") as f:
        f.write(f"# Tracking effectiveness — {tag}  [{tracker}]\n\n")
        f.write(f"- regime: **{'rotation' if mode=='rotation' else 'staring'}**\n")
        f.write(f"- sensor: {cfg.sensor_w}x{cfg.sensor_h} px, FOV {cfg.fov_deg:g} deg\n")
        f.write("- distance is a RELATIVE proxy; calibrate to metres with scripts/calibrate_range.py\n\n")
        f.write("## Summary\n\n")
        for k, v in summary.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Per-track\n\n")
        cols = "| id | pts | dur [s] | rel-dist min..max |" + (" blade [Hz] |" if mode != "rotation" else "")
        f.write(cols + "\n" + "|---" * cols.count("|") + "|\n")
        for r in recs:
            rel = r["rel_distance"][np.isfinite(r["rel_distance"])]
            rr = f"{rel.min():.0f}..{rel.max():.0f}" if rel.size else "n/a"
            line = f"| {r['id']} | {r['n_points']} | {r['duration_s']:.2f} | {rr} |"
            if mode != "rotation":
                line += f" {r['blade_hz']:.0f} |" if np.isfinite(r["blade_hz"]) else " — |"
            f.write(line + "\n")
    return out_path


def render_tracking_report(tracks, traj, ev, keep, cfg, tel, out_dir, tag, tracker="tracker", video=True):
    """Build the per-tracker report. Returns (summary_dict, [(path, description), ...])."""
    mode = getattr(cfg, "mode", "staring")
    os.makedirs(out_dir, exist_ok=True)
    if not tracks:
        return {"n_tracks": 0}, []
    recs = _analyze(tracks, traj, ev, cfg, mode)
    summary = ta.effectiveness_summary(recs, mode)
    P = lambda n: os.path.join(out_dir, n)
    arts = []
    fig = _figure_rotation(recs, P(f"{tag}_tracking.png"), tag) if mode == "rotation" \
        else _figure_staring(recs, P(f"{tag}_tracking.png"), tag)
    arts.append((fig, f"tracking measurements ({'rotation' if mode=='rotation' else 'staring'})"))
    arts.append((_write_csv(recs, mode, P(f"{tag}_tracks.csv")), "enriched per-point CSV (boxes + relative distance)"))
    arts.append((_write_metrics(summary, recs, cfg, mode, tracker, P(f"{tag}_tracking_metrics.md"), tag),
                 "tracking effectiveness report"))
    if video:
        v = _overlay_video(ev, recs, cfg, mode, P(f"{tag}_tracker_overlay.mp4"))
        if v:
            arts.append((v, "tracker overlay video (boxes + IDs)"))
    return summary, arts
