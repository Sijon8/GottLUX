"""
centroid.py  --  Accumulation-INDEPENDENT per-pass target centroiding.

Why not centroid the accumulated image?
  The drone is only in-FOV for ~FOV/omega seconds per revolution. An accumulated
  frame of that transit is a streak whose shape (and image centroid) depends on
  the accumulation window. So we must NOT centroid the binned image.

Principle (de-rotation invariance):
  As the sensor pans, a fixed target at true bearing B satisfies
      azimuth(t) + (x(t) - W/2)*deg_per_px = B   = constant
  for every event during the transit. So the *de-rotated world bearing* of each
  event equals B regardless of WHEN it fired. The event-weighted centroid of the
  de-rotated bearings over one pass is therefore an accumulation-independent
  estimate of B; its dispersion gives the target's angular size (-> range).

per_pass_centroids() groups the isolated target events by revolution and returns,
for each pass: t_centroid, bearing, elevation, range(angular-width), and the
standard error of the bearing (localization precision).
"""
from __future__ import annotations
import numpy as np


def _robust_center(v):
    """Trimmed-mean center + robust std (MAD) of a 1-D sample."""
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-9
    keep = np.abs(v - med) < 3 * 1.4826 * mad
    c = float(np.mean(v[keep])) if keep.any() else float(med)
    s = float(np.std(v[keep])) if keep.sum() > 1 else float(1.4826 * mad)
    return c, s, int(keep.sum())


def per_pass_centroids(ev, keep, cfg, tel, min_events=50):
    """Return a structured array of per-pass (per-revolution) target centroids.

    Columns: t, bearing_deg, bearing_se_deg, elev_deg, ang_size_deg, range_m, n_events
    Range uses a uniform-disk model: angular_size ~= sqrt(12)*std(bearing)."""
    W, H = cfg.sensor_w, cfg.sensor_h
    deg_per_px = cfg.fov_deg / W
    t = np.asarray(ev["t"])[keep] / 1e6
    x = np.asarray(ev["x"])[keep].astype(np.float64)
    y = np.asarray(ev["y"])[keep].astype(np.float64)

    # per-EVENT de-rotated world bearing (accumulation-independent)
    if cfg.mode == "rotation" and tel is not None:
        pan = np.rad2deg(np.interp(t - tel.offset, tel.t, tel.azimuth_unwrapped()))
        bearing = np.mod(pan + cfg.az_sign * (x - W / 2) * deg_per_px, 360.0)
        rev = tel.revolution_at(t)
    else:
        bearing = cfg.az_sign * (x - W / 2) * deg_per_px
        # staring: segment by temporal gaps instead of revolutions
        rev = _segment_by_gaps(t, gap_s=0.2)
    elev = (H / 2 - y) * deg_per_px                  # vertical -> elevation about the HEIGHT centre

    rows = []
    for r in np.unique(rev):
        m = rev == r
        if m.sum() < min_events:
            continue
        b = bearing[m]
        # unwrap bearings that straddle 0/360 for a meaningful mean
        bu = np.rad2deg(np.unwrap(np.deg2rad(b)))
        bc, bs, nkeep = _robust_center(bu)
        ec, _, _ = _robust_center(elev[m])
        tc = float(np.median(t[m]))
        n = int(m.sum())
        se = bs / np.sqrt(max(nkeep, 1))               # bearing standard error
        ang_size = float(np.sqrt(12.0) * bs)           # uniform-disk angular diameter (deg)
        rng = (cfg.target_diag_m / np.deg2rad(ang_size)) if (cfg.target_diag_m > 0 and ang_size > 1e-6) else np.nan
        rows.append((tc, np.mod(bc, 360.0), se, ec, ang_size, rng, n))
    if not rows:
        return np.zeros((0, 7))
    return np.array(rows)


def _segment_by_gaps(t, gap_s=0.2):
    """Assign a group id to events, incrementing whenever a temporal gap > gap_s."""
    seg = np.zeros(len(t), np.int64)
    if len(t) > 1:
        brk = np.where(np.diff(t) > gap_s)[0]
        for b in brk:
            seg[b + 1:] += 1
    return seg


def image_centroid_per_pass(ev, keep, cfg, tel, accum_dt, min_events=50):
    """For comparison ONLY: bearing centroid via per-FRAME image centroids binned
    at `accum_dt` (the accumulation-DEPENDENT method). Returns (t, bearing) per pass
    using, per pass, the frame with the most events (mimics 'pick the best frame')."""
    W = cfg.sensor_w
    deg_per_px = cfg.fov_deg / W
    t = np.asarray(ev["t"])[keep] / 1e6
    x = np.asarray(ev["x"])[keep].astype(np.float64)
    rev = tel.revolution_at(t) if (cfg.mode == "rotation" and tel is not None) else _segment_by_gaps(t)
    out = []
    for r in np.unique(rev):
        m = rev == r
        if m.sum() < min_events:
            continue
        tt, xx = t[m], x[m]
        edges = np.arange(tt.min(), tt.max() + accum_dt, accum_dt)
        if len(edges) < 2:
            continue
        idx = np.clip(np.searchsorted(edges, tt) - 1, 0, len(edges) - 2)
        # pick the densest frame, take its image x-centroid
        counts = np.bincount(idx, minlength=len(edges) - 1)
        fi = int(np.argmax(counts))
        sel = idx == fi
        cx = xx[sel].mean(); tc = 0.5 * (edges[fi] + edges[fi + 1])
        if cfg.mode == "rotation" and tel is not None:
            pan = float(np.interp(tc - tel.offset, tel.t, tel.azimuth_unwrapped()))
            bearing = np.mod(np.rad2deg(pan) if False else
                             np.rad2deg(np.interp(tc - tel.offset, tel.t, tel.azimuth_unwrapped()))
                             + cfg.az_sign * (cx - W / 2) * deg_per_px, 360.0)
        else:
            bearing = cfg.az_sign * (cx - W / 2) * deg_per_px
        out.append((tc, bearing))
    return np.array(out) if out else np.zeros((0, 2))


if __name__ == "__main__":
    import sys, glob, os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gottlux.config import Config, CAMERA_FOV_DEG
    from gottlux.rotation import io_evt21, background, detect
    from gottlux.io.telemetry import Telemetry

    base = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "cam1"
    cfg = Config(mode="rotation"); cfg.fov_deg = CAMERA_FOV_DEG[prefix]
    raw = glob.glob(os.path.join(base, f"{prefix}*.raw"))[0]
    csv = (glob.glob(os.path.join(base, "data_*.csv")) or glob.glob(os.path.join(base, "*.csv")))[0]
    ev = io_evt21.load(raw); tel = Telemetry(csv)
    t = np.asarray(ev["t"]) / 1e6; tel.refine_offset_to_events(t)
    hot = background.hot_pixel_mask(ev, cfg.hot_pixel_pct)
    ref = background.build_reference(ev, tel, float(tel.hall_t[0] + tel.offset), n_phase=cfg.n_phase)
    drop = background.rotation_drop_mask(ev, tel, ref, cfg.n_phase, hot)
    keep, _ = detect.isolate_target(ev, drop, cfg.accum_dt, cfg.min_pixels)

    P = per_pass_centroids(ev, keep, cfg, tel)
    print(f"{prefix}: {len(P)} passes")
    if len(P):
        print("  median bearing SE = %.3f deg | median ang.size = %.2f deg | median range = %.2f m"
              % (np.median(P[:, 2]), np.median(P[:, 4]), np.nanmedian(P[:, 5])))

    # accumulation-independence study on the densest pass
    accs = [0.005, 0.01, 0.02, 0.05, 0.1]
    img_curves = {a: image_centroid_per_pass(ev, keep, cfg, tel, a) for a in accs}

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    # A: per-event centroid track with SE error bars
    if len(P):
        ax[0].errorbar(P[:, 0], P[:, 1], yerr=P[:, 2], fmt="o", ms=4, color="b", ecolor="c", capsize=2)
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("bearing [deg]")
    ax[0].set_title("Per-pass de-rotated centroid (±SE)\naccumulation-INDEPENDENT"); ax[0].grid(alpha=.3)

    # B: azimuthal sweep profile for the densest pass
    rev = tel.revolution_at(t[keep])
    counts = np.bincount(rev - rev.min()) if len(rev) else np.array([0])
    rstar = (rev.min() + int(np.argmax(counts))) if len(rev) else 0
    m = rev == rstar
    deg_per_px = cfg.fov_deg / cfg.sensor_w
    pan = np.rad2deg(np.interp(t[keep][m] - tel.offset, tel.t, tel.azimuth_unwrapped()))
    bb = np.mod(pan + cfg.az_sign * (np.asarray(ev["x"])[keep][m] - cfg.sensor_w / 2) * deg_per_px, 360.0)
    bbu = np.rad2deg(np.unwrap(np.deg2rad(bb)))
    ax[1].hist(bbu, bins=120, color="steelblue")
    cstar, sstar, _ = _robust_center(bbu)
    ax[1].axvline(cstar, color="r", lw=2, label=f"centroid {np.mod(cstar,360):.2f} deg")
    ax[1].set_xlabel("de-rotated world bearing [deg]"); ax[1].set_ylabel("event count")
    ax[1].set_title(f"Sweep profile, pass #{rstar}\n(width->angular size->range)"); ax[1].legend(); ax[1].grid(alpha=.3)

    # C: bearing estimate vs accumulation time for that pass (image method) vs per-event (flat)
    for a, cv in img_curves.items():
        if len(cv):
            j = int(np.argmin(np.abs(cv[:, 0] - np.median(t[keep][m]))))
            ax[2].plot(a * 1e3, cv[j, 1], "s", ms=8, label=f"image @ {int(a*1e3)}ms")
    ax[2].axhline(np.mod(cstar, 360), color="b", lw=2, label="per-event centroid (fixed)")
    ax[2].set_xlabel("accumulation time [ms]"); ax[2].set_ylabel("estimated bearing [deg]")
    ax[2].set_title("Image-centroid drifts with accumulation;\nper-event centroid does not"); ax[2].legend(); ax[2].grid(alpha=.3)

    fig.tight_layout(); out = os.path.join(base, f"{prefix}_centroid_study.png")
    fig.savefig(out, dpi=130); print("saved", out)
