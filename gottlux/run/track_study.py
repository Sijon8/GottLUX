"""
track_study.py — the single-clip "track study": detect → track → KPI → figures → videos.

This is the reusable, in-library pipeline behind the staring drone results (it was prototyped in
``scripts/staring_results.py``; the plot/video builders now live here so any clip can be processed
the same way, from the CLI (``gottlux CLIP --track-study``) or from Python).

For one recording it runs a tracker (default ``single_centroid`` — the sandbox preset: strongest
connected-component blob per frame, EMA-smoothed centroid, all polarity), the three KPIs (range /
prop-frequency / time-to-contact), and writes a bundle into *out_dir*:

* the KPI figures + ``detections.csv`` + ``kpi_report.md`` (via :mod:`gottlux.run.performance_report`),
* ``range_vs_time_full`` — full-flight D(t) with the velocity-zero / out-of-FOV dropouts shaded,
* ``lock_score`` — the composite track-lock quality over time,
* ``track_dashboard.png`` — a contact-sheet of the tracked target,
* ``track_fft_dashboard`` — the boxed target beside its rotor spectrum at several moments,
* ``<label>_<det>_<accum>ms_overlay.mp4`` — the tracked-overlay video (red "target" box + tail),
* ``<label>_track_fft_live.mp4`` — the live track-and-classify demo (frame + live spectrum).

Everything is parameterised (FOV, target size, accumulation, band, detector), so updating a plot is
a parameter change + a re-run. Pure NumPy/matplotlib/PIL + the gottlux library; no Qt.
"""
from __future__ import annotations

import os

import numpy as np

import gottlux as eb
from gottlux.config import Config
from gottlux.core import frequency as fq
from gottlux.core.photogrammetry import focal_px
from gottlux.core.render import render_frame
from gottlux.io import export
from gottlux.run import performance_report as pr
from gottlux.viz.video import disp_to_rgb, infographic_frame, write_video


# --------------------------------------------------------------------- loading
def _load(clip, cache_local=False):
    """Load *clip*. *cache_local* is kept for API compatibility but no longer copies the
    file to %TEMP%: :func:`gottlux.io.cache.load` now falls back by itself to a per-process
    temp cache dir when another process (e.g. a running GUI session) holds the memmapped
    bins — the collision this flag used to work around."""
    return eb.load(clip, progress=lambda f: None)


# --------------------------------------------------------------------- track + spectrum helpers
def track_of(det):
    """The single (primary) track as plain arrays, or None."""
    if not det.targets:
        return None
    tg = max(det.targets, key=lambda k: (getattr(k, "confidence", 0.0), k.n))
    return dict(t=np.asarray(tg.t, float), cx=np.asarray(tg.cx, float), cy=np.asarray(tg.cy, float),
                bbox=np.asarray(tg.bbox, float), rng=np.asarray(tg.range_m, float),
                snr=np.asarray(tg.snr, float))


def box_spectrum(full, fx, fy, fts, ftus, ct, bbox, band, fwin=0.3):
    """In-band spectrum of the events inside *bbox* over the trailing ``[ct-fwin, ct]`` window."""
    lo, hi = band
    x0, y0, x1, y1 = bbox
    a0 = np.searchsorted(fts, ct - fwin); a1 = np.searchsorted(fts, ct)
    mb = (fx[a0:a1] >= x0) & (fx[a0:a1] < x1) & (fy[a0:a1] >= y0) & (fy[a0:a1] < y1)
    vt = ftus[a0:a1][mb]
    return fq.region_spectrum(vt, fs=max(2.2 * hi, 2000.0), fmin=lo, fmax=hi) if vt.size > 16 else None


def _draw_rotor_spectrum(ax, sp, band, *, line, floor, peak):
    """Draw an in-box rotor spectrum, **normalized to its noise floor** and with the sub-band
    near-DC leakage excluded.

    The raw spectrum has a large low-frequency (≈0 Hz) component from the slow event-rate envelope
    that, on a linear axis, dwarfs the rotor line and crushes the scale. We therefore (1) plot the
    power in units of the median noise floor (so the y-axis reads directly in "× over noise" and
    panels are comparable), (2) restrict the displayed band to the rotor band so the DC spike is
    off-screen, and (3) set the y-limit from the **in-band** content only, so nothing below the
    rotor band can drive the scale. Returns the noise floor used.
    """
    lo, hi = band
    freqs = np.asarray(sp.freqs, float)
    power = np.maximum(np.asarray(sp.power, float), 0.0)
    noise = sp.peak_power / max(sp.snr, 1e-6)                       # median in-band noise floor
    if not np.isfinite(noise) or noise <= 0:
        pos = power[power > 0]
        noise = float(np.median(pos)) if pos.size else 1.0
    yn = power / noise                                             # power in multiples of the floor
    disp = (freqs >= lo * 0.85) & (freqs <= hi * 1.02)             # display window: no near-DC
    inb = (freqs >= lo) & (freqs <= hi)                            # in-band: sets the y-scale
    if disp.any():
        ax.plot(freqs[disp], yn[disp], color=line, lw=1.1)
    ax.axhline(1.0, color=floor, ls=":", lw=0.9)                  # noise floor = 1×
    ymax = float(yn[inb].max()) if inb.any() else (float(yn[disp].max()) if disp.any() else 2.0)
    ax.set_ylim(0, max(ymax * 1.15, 2.0))
    ax.set_xlim(lo * 0.85, hi * 1.02)
    if np.isfinite(sp.peak_freq):
        ax.axvline(sp.peak_freq, color=peak, ls="--", lw=1.1)
    return noise


# --------------------------------------------------------------------- figures
def range_vs_time_full(det, accum_dt, out_dir, label):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    tk = track_of(det)
    if tk is None:
        return
    t, rng = tk["t"], tk["rng"]
    m = np.isfinite(t) & np.isfinite(rng); t, rng = t[m], rng[m]
    if t.size < 2:
        return
    gap = accum_dt * 1.8
    tt, rr, gaps = [float(t[0])], [float(rng[0])], []
    for i in range(1, t.size):
        if t[i] - t[i - 1] > gap:
            tt.append(np.nan); rr.append(np.nan); gaps.append((float(t[i - 1]), float(t[i])))
        tt.append(float(t[i])); rr.append(float(rng[i]))
    fig, ax = plt.subplots(figsize=(9.2, 4.3), facecolor="w")
    ax.plot(tt, rr, "-o", ms=3, lw=1.4, color="#1f4e8c")
    for g0, g1 in gaps:
        ax.axvspan(g0, g1, color="#e8a200", alpha=0.18)
    ax.set_xlabel("time [s]"); ax.set_ylabel("range to drone [m]")
    ax.set_title(f"Range vs time — full flight — {label}")
    handles = [plt.Line2D([0], [0], color="#1f4e8c", marker="o", ms=3, label="range to drone (single track)")]
    if gaps:
        handles.append(Patch(facecolor="#e8a200", alpha=0.3,
                             label=f"{len(gaps)} dropout(s): out-of-FOV or rel. velocity → 0 (no events)"))
    ax.legend(handles=handles, fontsize=8); ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    export.save_figure(fig, os.path.join(out_dir, "range_vs_time_full"), formats=("png", "pdf"), close=True)


def lock_score(det, accum_dt, out_dir, label):
    """The sandbox composite lock score over time: 0.5·SNR-term + 0.3·steadiness + 0.2·coverage."""
    import matplotlib.pyplot as plt
    tk = track_of(det)
    if tk is None or tk["t"].size < 4:
        return
    t, cx, cy, snr = tk["t"], tk["cx"], tk["cy"], tk["snr"]
    n_steps = int((det.diagnostics or {}).get("n_steps", t.size)); span = max(t[-1] - t[0], 1e-6)
    expected_per_s = n_steps / span
    lock, cov, jit_t, snr_t, last_snr, win_s = [], [], [], [], 0.0, 1.0
    for i, ti in enumerate(t):
        m = (t >= ti - win_s) & (t <= ti)
        coverage = float(np.clip(m.sum() / max(expected_per_s * win_s, 1.0), 0, 1))
        pts = np.stack([cx[m], cy[m]], 1)
        jitter = float(np.hypot(np.diff(pts[:, 0]).std(), np.diff(pts[:, 1]).std())) if pts.shape[0] >= 3 else 0.0
        if np.isfinite(snr[i]) and snr[i] > 0:
            last_snr = snr[i]
        snr_term = float(np.clip(last_snr / 8.0, 0, 1)); jit_term = 1.0 / (1.0 + jitter / 5.0)
        lock.append(0.5 * snr_term + 0.3 * jit_term + 0.2 * coverage)
        cov.append(coverage); jit_t.append(jit_term); snr_t.append(snr_term)
    fig, ax = plt.subplots(figsize=(9.2, 4.0), facecolor="w")
    ax.axhspan(0.66, 1, color="#3fb950", alpha=0.10); ax.axhspan(0.33, 0.66, color="#d29922", alpha=0.10)
    ax.axhspan(0, 0.33, color="#f85149", alpha=0.10)
    ax.plot(t, lock, "-", lw=1.8, color="#1f4e8c", label="lock score")
    ax.plot(t, snr_t, ":", lw=1, color="#c62828", label="0.5·SNR term")
    ax.plot(t, jit_t, ":", lw=1, color="#2e7d32", label="0.3·steadiness")
    ax.plot(t, cov, ":", lw=1, color="#6a1b9a", label="0.2·coverage")
    ax.set_ylim(0, 1.02); ax.set_xlabel("time [s]"); ax.set_ylabel("lock score (0–1)")
    ax.set_title(f"Track lock score — {label}")
    ax.grid(True, ls="--", alpha=0.3); ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    export.save_figure(fig, os.path.join(out_dir, "lock_score"), formats=("png", "pdf"), close=True)


def track_dashboard(rec, det, accum_dt, out_dir, label, n=12, thumb=120, pad=10):
    from PIL import Image, ImageDraw
    tk = track_of(det)
    if tk is None or tk["t"].size == 0:
        return
    t, bb, rng = tk["t"], tk["bbox"], tk["rng"]
    idx = np.linspace(0, t.size - 1, min(n, t.size)).round().astype(int)
    cols = 4; rows = int(np.ceil(len(idx) / cols)); cellw, cellh = thumb, thumb + 16
    sheet = Image.new("RGB", (cols * cellw, rows * cellh), (12, 16, 22)); sd = ImageDraw.Draw(sheet)
    for k, i in enumerate(idx):
        disp, levels, _v, _w = render_frame(rec, float(t[i]), accum_dt, mode="count", expr="sqrt", back=True)
        full = Image.fromarray(disp_to_rgb(disp, levels, "inferno")).convert("RGB")
        x0, y0, x1, y1 = bb[i]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) / 2 + pad
        crop = full.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))).resize((thumb, thumb))
        cd = ImageDraw.Draw(crop); fr = (x1 - x0) / (2 * half) * thumb / 2; fh = (y1 - y0) / (2 * half) * thumb / 2
        cd.rectangle([thumb / 2 - fr, thumb / 2 - fh, thumb / 2 + fr, thumb / 2 + fh], outline=(235, 40, 40), width=2)
        cx0, cy0 = (k % cols) * cellw, (k // cols) * cellh
        sheet.paste(crop, (cx0, cy0))
        rs = f"{rng[i]:.1f} m" if np.isfinite(rng[i]) else "-"
        sd.text((cx0 + 3, cy0 + thumb + 1), f"t={t[i]:.1f}s  {rs}", fill=(220, 230, 240))
    out = Image.new("RGB", (sheet.width, sheet.height + 22), (12, 16, 22))
    ImageDraw.Draw(out).text((6, 4), f"Tracked target across the flight - {label}", fill=(255, 255, 255))
    out.paste(sheet, (0, 22)); out.save(os.path.join(out_dir, "track_dashboard.png"))


def track_fft_dashboard(rec, det, accum_dt, band, out_dir, label, n=6):
    import matplotlib.pyplot as plt
    tk = track_of(det)
    if tk is None or tk["t"].size == 0:
        return
    full = rec.window(rec.t_start_s, rec.t_stop_s)
    fx = np.asarray(full.x); fy = np.asarray(full.y); fts = full.t_s; ftus = np.asarray(full.t)
    t, bb, rng = tk["t"], tk["bbox"], tk["rng"]; lo, hi = band
    idx = np.linspace(0, t.size - 1, min(n, t.size)).round().astype(int)
    fig, axes = plt.subplots(len(idx), 2, figsize=(8.6, 1.9 * len(idx)), facecolor="w",
                             gridspec_kw={"width_ratios": [1, 1.5]}, squeeze=False)
    for r, i in enumerate(idx):
        ct = float(t[i]); x0, y0, x1, y1 = bb[i]
        disp, levels, _v, _w = render_frame(rec, ct, accum_dt, mode="count", expr="sqrt", back=True)
        ax0 = axes[r][0]; ax0.imshow(disp_to_rgb(disp, levels, "inferno"), origin="upper")
        ax0.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#e82828", lw=1.5))
        ax0.set_xlim(0, rec.width); ax0.set_ylim(rec.height, 0); ax0.set_xticks([]); ax0.set_yticks([])
        rs = f"{rng[i]:.1f} m" if np.isfinite(rng[i]) else "-"
        ax0.set_ylabel(f"t={ct:.1f}s\nD={rs}", fontsize=8, rotation=0, ha="right", va="center")
        ax1 = axes[r][1]; sp = box_spectrum(full, fx, fy, fts, ftus, ct, bb[i], band)
        if sp is not None:
            # LINEAR y, normalized to the noise floor (= 1×) with the near-DC leakage excluded.
            _draw_rotor_spectrum(ax1, sp, band, line="#1f4e8c", floor="#888", peak="#c62828")
            if np.isfinite(sp.peak_freq):
                ax1.set_title(f"peak {sp.peak_freq:.0f} Hz · {sp.snr:.0f}× over noise · "
                              f"harm {sp.harmonic_score:.2f}", fontsize=8)
        else:
            ax1.set_title("insufficient events", fontsize=8)
            ax1.set_xlim(lo * 0.85, hi * 1.02)
        ax1.tick_params(labelsize=7)
        ax1.set_ylabel("× noise", fontsize=7)
        if r == len(idx) - 1:
            ax1.set_xlabel("frequency [Hz]", fontsize=8)
    fig.suptitle(f"Track + rotor-FFT across the flight - {label}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    export.save_figure(fig, os.path.join(out_dir, "track_fft_dashboard"), formats=("png", "pdf"), close=True)


def extra_figs(det, rec, out_dir, label):
    for name, fn in (("track_timeseries", lambda: _ts(det, rec, label)),
                     ("radar", lambda: _radar(det, label))):
        try:
            fig = fn()
            if fig is not None:
                export.save_figure(fig, os.path.join(out_dir, name), formats=("png", "pdf"), close=True)
        except Exception as e:
            print(f"   [warn] {name}: {e}")


def _ts(det, rec, label):
    from gottlux.viz import tracks as vt
    return vt.track_timeseries_figure(det, rotating=rec.is_rotating, title=f"Track time-series — {label}")


def _radar(det, label):
    from gottlux.viz import panorama as vp
    return vp.radar_figure(getattr(det, "targets", []), title=f"Radar — {label}")


# --------------------------------------------------------------------- videos
def tracked_overlay_video(rec, det, accum_dt, target_size_m, drone, fov_deg, out_dir, label,
                          role, tracking_m, scale=2):
    from PIL import Image, ImageDraw
    tk = track_of(det)
    if tk is None:
        return None
    t, bb, cx, cy, rng = tk["t"], tk["bbox"], tk["cx"], tk["cy"], tk["rng"]
    fps = max(1, int(round(1.0 / accum_dt)))
    order = range(t.size) if t.size <= 1100 else range(0, t.size, int(np.ceil(t.size / 1100)))
    diag = drone.get("diag_mm") if drone else None
    title = f"GottLUX — {label} — single-centroid track @ {int(accum_dt*1000)} ms"
    subtitle = (f'drone {drone.get("prop","")} {diag} mm diag · ' if diag else "") + \
        f"L={target_size_m} m · FOV {fov_deg}° · {role}"
    footer = ["red box = target (raw blob) · dotted = track tail · all polarity · "
              f"trailing {int(accum_dt*1000)} ms window (no resampling lag)",
              f"max tracking range {tracking_m:g} m" if tracking_m else ""]
    frames = []
    for i in order:
        ft = float(t[i])
        disp, levels, _v, _w = render_frame(rec, ft, accum_dt, mode="count", expr="sqrt", back=True)
        im = Image.fromarray(disp_to_rgb(disp, levels, "inferno")).convert("RGB")
        if scale != 1:
            im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
        d = ImageDraw.Draw(im); s = scale
        for px, py in zip(cx[(t <= ft) & (t > ft - 1.2)], cy[(t <= ft) & (t > ft - 1.2)]):
            d.ellipse([px * s - 1, py * s - 1, px * s + 1, py * s + 1], fill=(255, 215, 70))
        x0, y0, x1, y1 = bb[i]
        d.rectangle([x0 * s, y0 * s, x1 * s, y1 * s], outline=(235, 40, 40), width=2)
        lab = "target" + (f"  D={rng[i]:.1f} m" if np.isfinite(rng[i]) else "")
        d.text((x0 * s, max(0, y0 * s - 12)), lab, fill=(255, 90, 90))
        frames.append(infographic_frame(np.asarray(im), title=title, subtitle=subtitle, footer_lines=footer))
    out = os.path.join(out_dir, f"{label}_single_centroid_{int(accum_dt*1000)}ms_overlay.mp4")
    return write_video(out, frames, fps=fps)


def track_fft_video(rec, det, accum_dt, band, out_dir, label, role, tracking_m, max_frames=200):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    tk = track_of(det)
    if tk is None:
        return None
    full = rec.window(rec.t_start_s, rec.t_stop_s)
    fx = np.asarray(full.x); fy = np.asarray(full.y); fts = full.t_s; ftus = np.asarray(full.t)
    t, bb, cx, cy, rng = tk["t"], tk["bbox"], tk["cx"], tk["cy"], tk["rng"]; lo, hi = band
    order = range(t.size) if t.size <= max_frames else range(0, t.size, int(np.ceil(t.size / max_frames)))
    frames = []
    for i in order:
        ct = float(t[i]); x0, y0, x1, y1 = bb[i]
        disp, levels, _v, _w = render_frame(rec, ct, accum_dt, mode="count", expr="sqrt", back=True)
        rgb = disp_to_rgb(disp, levels, "inferno"); sp = box_spectrum(full, fx, fy, fts, ftus, ct, bb[i], band)
        fig = Figure(figsize=(9.2, 3.7), facecolor="#0e1116")
        axL = fig.add_subplot(1, 2, 1); axR = fig.add_subplot(1, 2, 2)
        axL.imshow(rgb, origin="upper")
        axL.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#ff3030", lw=1.8))
        m = (t <= ct) & (t > ct - 1.2); axL.scatter(cx[m], cy[m], s=4, c="#ffd246")
        axL.text(x0, max(0, y0 - 6), "target", color="#ff7070", fontsize=9)
        axL.set_xlim(0, rec.width); axL.set_ylim(rec.height, 0); axL.set_xticks([]); axL.set_yticks([])
        rs = f"{rng[i]:.1f} m" if np.isfinite(rng[i]) else "-"
        axL.set_title(f"{label}  ·  t={ct:5.2f}s  ·  D={rs}", color="w", fontsize=10)
        if sp is not None:
            # LINEAR y, normalized to the noise floor (= 1×), near-DC leakage excluded.
            _draw_rotor_spectrum(axR, sp, band, line="#39c5cf", floor="#8a93a3", peak="#ff6a3d")
            ttl = "rotor FFT (in box)"
            if np.isfinite(sp.peak_freq):
                ttl = f"rotor FFT — {sp.peak_freq:.0f} Hz · {sp.snr:.0f}× over noise"
            axR.set_title(ttl, color="w", fontsize=10)
        else:
            axR.set_title("rotor FFT — (few events)", color="w", fontsize=10)
            axR.set_xlim(lo * 0.85, hi * 1.02)
        axR.set_ylabel("× noise floor", color="w", fontsize=9)
        axR.set_xlabel("frequency [Hz]", color="w", fontsize=9)
        axR.set_facecolor("#0e1116"); axR.tick_params(colors="#c9d4e0", labelsize=8)
        for sp_ in axR.spines.values():
            sp_.set_color("#33405a")
        fig.tight_layout()
        c = FigureCanvasAgg(fig); c.draw()
        frames.append(np.asarray(c.buffer_rgba())[..., :3].copy())
    out = os.path.join(out_dir, f"{label}_track_fft_live.mp4")
    return write_video(out, frames, fps=max(1, int(round(1.0 / accum_dt))))


# --------------------------------------------------------------------- the orchestrator
def run_track_study(clip, *, fov_deg, target_size_m, out_dir, detector="single_centroid",
                    accum_dt=0.085, band=(80.0, 800.0), drone=None, role="", label=None,
                    approach_speed_mps=15.0, t_start=None, t_stop=None, make_overlay=True,
                    make_fft_video=False, cache_local=False) -> dict:
    """Run the full single-clip track study into *out_dir*; return ``{result, det, headline,
    diagnostics, rec}``. See the module docstring for the artifacts produced."""
    os.makedirs(out_dir, exist_ok=True)
    label = label or os.path.splitext(os.path.basename(clip))[0]
    rec = _load(clip, cache_local)
    cfg = Config(mode="staring")
    cfg.fov_deg = fov_deg; cfg.target_size_m = target_size_m; cfg.detector = detector
    cfg.accum_dt = accum_dt; cfg.open_when_done = False
    cfg.sensor_w = rec.width; cfg.sensor_h = rec.height
    cfg.t_start = t_start; cfg.t_stop = t_stop
    cfg.freq_lo, cfg.freq_hi = band; cfg.approach_speed_mps = approach_speed_mps
    det = pr._run_detector(rec, cfg)
    result = pr.compute_performance(rec, cfg, det_result=det, approach_speed=approach_speed_mps)
    pr.save_performance(result, rec, cfg, out_dir=out_dir)
    h = result.headline()
    range_vs_time_full(det, accum_dt, out_dir, label)
    lock_score(det, accum_dt, out_dir, label)
    track_dashboard(rec, det, accum_dt, out_dir, label)
    track_fft_dashboard(rec, det, accum_dt, band, out_dir, label)
    extra_figs(det, rec, out_dir, label)
    if make_overlay:
        tracked_overlay_video(rec, det, accum_dt, target_size_m, drone or {}, fov_deg, out_dir,
                              label, role, h.get("tracking_range_m"))
    if make_fft_video:
        track_fft_video(rec, det, accum_dt, band, out_dir, label, role, h.get("tracking_range_m"))
    return {"result": result, "det": det, "headline": h, "diagnostics": det.diagnostics or {}, "rec": rec}
