"""
radarlab.py — GottLUX Radar/Box Lab (standalone).

A self-contained window for the rotating-EBS workflow: view the **de-rotated panorama** (the 360°
"radar" of the environment, the most intuitive way to see this data — see `figure_context`), suppress
static clutter (background masking), then **box each drone instance** and read out its **bearing**
(de-rotated per-pass centroid) and **range** (apparent y-pixel cross-section). New in this version:

  * **walk the time series revolution-by-revolution** (rev back / rev forward) and **track one box across every rev** →
    a bearing/range-vs-time timeline of the drone's behaviour;
  * **FFT of the boxed section** (rotor-flutter band, with the spin-envelope high-pass);
  * **column-subset analysis** — render/solve with the whole sensor, a few central columns, or a
    single column, and export the side-by-side **visualization difference** as publication figures.

Each box becomes a row in the table and a point on the tactical bearing×range radar. *Export* writes a
publication-ready figure bundle into `gottlux_runs/box_track/<tag>/`.

Standalone (own window, own time controls; no coupling to the main GUI):

    python -m gottlux.app.radarlab path/to/capture_folder --camera cam0 --fov 58
    python -m gottlux.app.radarlab <cam1_20.raw> --fov 20            # a .raw path works directly

Headless checks (no Qt):

    python -m gottlux.app.radarlab <capture> --fov 20 --solve 180,225,140,240 --t 5,13
    python -m gottlux.app.radarlab <capture> --fov 20 --track 180,225,140,240   # per-rev timeline

The de-rotation and box-solve below are the SAME math as scripts/rotational_box_solve.py.
NOTE on the azimuth sign: for the field rig's spin direction the correct convention is `+1` (verified
visually — it makes static structure vertical and the drone path a clean diagonal; the wrong sign
zig-zags). It is the default here; the GUI "flip azimuth" toggle exposes the other sign.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from gottlux.rotation.detect import focal_px

#: Correct de-rotation sign for the field rig (verified visually — see module docstring).
AZ_SIGN_DEFAULT = 1.0


# --------------------------------------------------------------------------- pure math (testable)
def world_azimuth(tel, x, t, fov_deg, width, az_sign=AZ_SIGN_DEFAULT):
    """De-rotate sensor (x, t) → world azimuth [deg). Same convention as rotational_box_solve."""
    dpp = fov_deg / width
    return np.mod(tel.azimuth_at(t) + az_sign * (x - width / 2) * dpp, 360.0)


def solve_box(azw, y, t, *, az, ywin, twin=None, sel=None, fov_deg, width, target_size_m=0.225,
              gap_s=0.30, min_pass=120):
    """Per-pass bearing + apparent-size range for events inside a panorama box (or, if ``sel`` is a
    precomputed boolean mask — e.g. from a lasso polygon — those events)."""
    azc = (az[0] + az[1]) / 2.0
    azu = ((azw - azc + 180) % 360) - 180 + azc                 # unwrap around the box centre
    if sel is not None:
        m = np.asarray(sel, bool).copy()
    else:
        m = (azu >= az[0]) & (azu <= az[1]) & (y >= ywin[0]) & (y <= ywin[1])
    if twin is not None:
        m = m & (t >= twin[0]) & (t <= twin[1])
    n_in = int(m.sum())
    empty = {"n_events": n_in, "n_passes": 0, "track": [], "bearing_deg": None,
             "bearing_span_deg": None, "range_m": None, "target_size_m": target_size_m}
    if n_in < min_pass:
        return empty
    tg = t[m]; o = np.argsort(tg); ts = tg[o]; ag = azu[m][o]; yg = y[m][o]
    fpx = focal_px(fov_deg, width)
    segs = np.split(np.arange(ts.size), np.where(np.diff(ts) > gap_s)[0] + 1)
    track = []
    for s in segs:
        if s.size < min_pass:
            continue
        b = float(np.median(ag[s])); se = float(np.std(ag[s]) / np.sqrt(s.size))
        ye = float(np.percentile(yg[s], 90) - np.percentile(yg[s], 10))
        rng = float(target_size_m * fpx / max(ye, 1.0))
        track.append({"t_s": round(float(np.median(ts[s])), 3), "bearing_deg": round(b, 2),
                      "bearing_SE_deg": round(se, 3), "y_extent_px": round(ye, 1),
                      "range_m": round(rng, 2), "n_events": int(s.size)})
    if not track:
        return empty
    bearings = [d["bearing_deg"] for d in track]; ranges = [d["range_m"] for d in track]
    return {"n_events": n_in, "n_passes": len(track), "track": track,
            "bearing_deg": round(float(np.median(bearings)), 2),
            "bearing_span_deg": round(float(max(bearings) - min(bearings)), 2),
            "range_m": round(float(np.median(ranges)), 2),
            "range_span_m": [round(float(min(ranges)), 2), round(float(max(ranges)), 2)],
            "target_size_m": target_size_m}


def static_keep_mask(tel, x, y, t, width, height, n_rev):
    """Rotational background mask: events whose (rotation-phase, x, y) voxel was occupied in the first
    n_rev revolutions are static clutter → dropped. Returns a boolean keep mask."""
    if n_rev <= 0:
        return np.ones(t.size, bool)
    T = tel.T_rot or (float(t.max() - t.min()) / 4.0)
    t_ref = float(t.min()) + n_rev * T
    ph = np.mod(tel.azimuth_at(t), 360.0)
    pb = np.clip((ph / 360.0 * 180).astype(np.int64), 0, 179)
    xb = np.clip((x / width * 160).astype(np.int64), 0, 159)
    yb = np.clip((y / height * 160).astype(np.int64), 0, 159)
    key = (pb * 160 + xb) * 160 + yb
    ref = np.unique(key[t < t_ref])
    return ~np.isin(key, ref)


def revolution_windows(tel, t_min=None, t_max=None):
    """(t_start, t_end) for each full revolution, from the telemetry rotation boundaries."""
    rb = np.asarray(tel.rotation_bounds(), float)
    wins = [(float(rb[i]), float(rb[i + 1])) for i in range(len(rb) - 1)]
    lo = -np.inf if t_min is None else float(t_min)
    hi = np.inf if t_max is None else float(t_max)
    return [(max(a, lo), min(b, hi)) for a, b in wins if b > lo and a < hi]


def column_subset_keep(x, width, mode="all", n_cols=8):
    """Boolean mask selecting which sensor columns' events to use: the whole sensor (``all``), a few
    central columns (``few``), or a single central column (``single``) — the line-scanner knob."""
    if mode == "all":
        return np.ones(np.shape(x), bool)
    c = width / 2.0
    half = 0.5 if mode == "single" else max(0.5, n_cols / 2.0)
    return np.abs(np.asarray(x, float) - c) <= half


def densest_box(azw, y, *, height, az_pad=4.0, y_pad=28.0, az_bins=360, y_bins=80, min_events=60):
    """Suggest a box around the densest (azimuth, Y) cluster in the given events — used to pre-place
    the box on the drone when stepping to a revolution, so the operator just verifies and accepts."""
    if np.size(azw) < min_events:
        return None
    Hh, ae, ye = np.histogram2d(azw, y, bins=[az_bins, y_bins], range=[[0, 360], [0, height]])
    from scipy import ndimage
    Hs = ndimage.gaussian_filter(Hh, 1.0)
    ai, yi = np.unravel_index(int(np.argmax(Hs)), Hs.shape)
    az_c = (ai + 0.5) / az_bins * 360.0
    y_c = (yi + 0.5) / y_bins * height
    return (max(az_c - az_pad, 0.0), min(az_c + az_pad, 360.0),
            max(y_c - y_pad, 0.0), min(y_c + y_pad, height))


def polygon_keep(azw, y, verts):
    """Boolean mask of events inside a polygon (lasso) drawn in (azimuth, Y) space. The all-column
    drone is a **diagonal streak**, so a rectangle leaves the corners empty and clips the ends — a
    lasso outlines the actual shape tightly. (The single/few-central-column view is compact, so a
    box is fine there.)"""
    from matplotlib.path import Path
    v = np.asarray(verts, float) if verts is not None else None
    if v is None or len(v) < 3:
        return np.zeros(np.shape(azw), bool)
    pts = np.column_stack([np.asarray(azw, float), np.asarray(y, float)])
    return Path(v).contains_points(pts)


def masking_metrics(tel, x, y, t, *, width, height, drone_box=None, n_list=(0, 1, 2, 3, 4, 5)):
    """Quantify successive (rotation-phase, x, y) background masking vs reference depth N: event-rate
    reduction, absolute rate, and — given a drone box (az0,az1,y0,y1,az_sign,fov) — the drone events
    retained, background retained, and the drone:background contrast (the self-masking tradeoff)."""
    dur = float(t.max() - t.min()) or 1.0
    tot = int(t.size)
    dsel = None
    if drone_box:
        az0, az1, y0, y1, az_sign, fov = drone_box
        azw = world_azimuth(tel, x, t, fov, width, az_sign)
        azc = (az0 + az1) / 2.0; azu = ((azw - azc + 180) % 360) - 180 + azc
        dsel = (azu >= az0) & (azu <= az1) & (y >= y0) & (y <= y1)
        n_d = int(dsel.sum()); n_bg = tot - n_d
    rows = []
    for n in n_list:
        keep = static_keep_mask(tel, x, y, t, width, height, n)
        kept = int(keep.sum())
        row = {"N": int(n), "events_kept": kept, "reduction_pct": round(100 * (1 - kept / max(tot, 1)), 2),
               "event_rate_Mev_s": round(kept / dur / 1e6, 3)}
        if dsel is not None:
            dk = int((keep & dsel).sum()); bk = kept - dk
            dret = dk / max(n_d, 1); bret = bk / max(n_bg, 1)
            row.update({"drone_events_kept": dk, "drone_retained_pct": round(100 * dret, 1),
                        "background_retained_pct": round(100 * bret, 1),
                        "drone_to_bg_contrast": round(dret / max(bret, 1e-6), 2)})
        rows.append(row)
    return {"total_events": tot, "duration_s": round(dur, 3), "n_drone_box": (int(dsel.sum()) if dsel is not None else None),
            "rows": rows}


def figure_masking_quant(metrics, tag, out_dir, style=None):
    """Quantified successive-masking study: reduction %, absolute event rate, drone vs background
    retention (the self-masking tradeoff), and drone:background contrast — all vs reference depth N."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = style or {}
    knee = style.get("knee_N", 2)
    os.makedirs(out_dir, exist_ok=True)
    R = metrics["rows"]; N = [r["N"] for r in R]
    fig, axs = plt.subplots(2, 2, figsize=tuple(style.get("figsize", (12, 8))))
    a = axs[0, 0]; a.plot(N, [r["reduction_pct"] for r in R], "s-", color="k")
    a.set_title("(a) event-rate reduction vs reference depth N"); a.set_xlabel("N revolutions"); a.set_ylabel("% events dropped"); a.grid(alpha=.3)
    b = axs[0, 1]; b.plot(N, [r["event_rate_Mev_s"] for r in R], "o-", color="tab:purple")
    b.set_title("(b) absolute kept event rate"); b.set_xlabel("N revolutions"); b.set_ylabel("Mev/s"); b.grid(alpha=.3)
    c = axs[1, 0]
    if any("drone_retained_pct" in r for r in R):
        c.plot(N, [r.get("drone_retained_pct") for r in R], "o-", color="tab:green", label="drone retained")
        c.plot(N, [r.get("background_retained_pct") for r in R], "s-", color="tab:red", label="background retained")
        c.axvline(knee, color="gray", ls=":", lw=1, label=f"self-masking knee (N≈{knee})"); c.legend(fontsize=8)
    c.set_title("(c) target preservation vs clutter rejection"); c.set_xlabel("N revolutions"); c.set_ylabel("% retained"); c.grid(alpha=.3)
    d = axs[1, 1]
    if any("drone_to_bg_contrast" in r for r in R):
        d.plot(N, [r.get("drone_to_bg_contrast") for r in R], "^-", color="tab:blue")
    d.set_title("(d) drone : background contrast"); d.set_xlabel("N revolutions"); d.set_ylabel("contrast (×)"); d.grid(alpha=.3)
    fig.suptitle(style.get("suptitle", f"{tag}: successive rotational background masking — quantified "
                 f"(in-tool (phase,x,y) voxel mask; {metrics['total_events']:,} events, "
                 f"{metrics['duration_s']:.1f}s)"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, os.path.join(out_dir, "masking_quantification"), style)


def box_spectrum(t_box_s, *, fs=2000.0, fmin=200.0, fmax=800.0, derotate_hz=None, spin_hz=None):
    """Temporal power spectrum (FFT) of the boxed events' arrival times, searched over the **rotor
    band (default 200–800 Hz)** for a propeller tone. On a spinning sensor the raw spectrum is
    dominated by the once-per-rev burst envelope; ``derotate_hz`` (default 8× the spin rate)
    high-passes it away, and the 200 Hz floor excludes the low-frequency comb entirely."""
    from gottlux.core.frequency import region_spectrum
    if derotate_hz is None:
        derotate_hz = (8.0 * spin_hz) if spin_hz else 0.0
    return region_spectrum(np.asarray(t_box_s, float) * 1e6, fs=fs, fmin=fmin, fmax=fmax,
                           derotate_hz=derotate_hz)


def per_rev_spectra(azw, y, t, tel, instances, *, fmin=200.0, fmax=800.0, spin_hz=None, min_events=200):
    """FFT of the boxed drone **per revolution, all columns** (the high-SNR choice). For each boxed
    instance, gate its (azimuth, Y) box within that revolution's time window and spectrum it. Returns a
    per-rev list with peak frequency + SNR — the test of whether a rotor tone persists with revolution
    (i.e. with range), the rotational analogue of the staring 'prop-frequency vs range' result."""
    revs = revolution_windows(tel)
    out = []
    for b in sorted(instances, key=lambda d: d.get("rev", 0)):
        k = b.get("rev")
        if k is None or k < 0 or k >= len(revs):
            continue
        (a0, a1), (y0, y1) = b["az_window"], b["y_window"]
        t0, t1 = revs[k]
        azc = (a0 + a1) / 2.0; azu = ((azw - azc + 180) % 360) - 180 + azc
        m = (azu >= a0) & (azu <= a1) & (y >= y0) & (y <= y1) & (t >= t0) & (t <= t1)
        if int(m.sum()) < min_events:
            continue
        spec = box_spectrum(t[m], fmin=fmin, fmax=fmax, spin_hz=spin_hz)
        out.append({"rev": int(k), "t_s": round((t0 + t1) / 2.0, 3), "n_events": int(m.sum()),
                    "peak_freq": round(float(spec.peak_freq), 1) if np.isfinite(spec.peak_freq) else None,
                    "snr": round(float(spec.snr), 1), "freqs": spec.freqs, "power": spec.power,
                    "band": spec.band, "range_m": b.get("range_m")})
    return out


def figure_fft_vs_rev(spectra, tag, out_dir, style=None, snr_gate=4.0):
    """Per-revolution rotor-FFT: (a) spectrum montage (freq × revolution; a vertical stripe = a tone
    that persists with revolution/range), (b) peak frequency per revolution, (c) in-band SNR per
    revolution vs the gate. Confirms whether a clean rotor tone is recoverable under rotation at close
    range with all columns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = style or {}
    os.makedirs(out_dir, exist_ok=True)
    if not spectra:
        return None
    lo, hi = spectra[0]["band"]
    grid = np.linspace(lo, hi, 320)
    revs = [s["rev"] for s in spectra]
    M = np.zeros((len(spectra), grid.size))
    for i, s in enumerate(spectra):
        f = np.asarray(s["freqs"]); p = np.asarray(s["power"])
        sel = (f >= lo) & (f <= hi)
        pp = np.interp(grid, f[sel], p[sel]) if sel.any() else np.zeros_like(grid)
        M[i] = pp / (pp.max() + 1e-20)                  # normalize each revolution to its own peak
    fig, axs = plt.subplots(1, 3, figsize=tuple(style.get("figsize", (16, 4.8))),
                            gridspec_kw={"width_ratios": [2.0, 1, 1]})
    im = axs[0].imshow(M, origin="lower", aspect="auto", cmap=style.get("cmap", "magma"),
                       extent=[lo, hi, revs[0] - 0.5, revs[-1] + 0.5])
    axs[0].set_xlabel("frequency [Hz]"); axs[0].set_ylabel("revolution")
    axs[0].set_title("(a) per-rev rotor spectrum (each row normalized)\na persistent vertical stripe = a real tone")
    fig.colorbar(im, ax=axs[0], pad=0.01)
    axs[1].plot([s["peak_freq"] for s in spectra], revs, "o-", color="tab:blue")
    axs[1].set_xlim(lo, hi); axs[1].set_xlabel("peak freq [Hz]"); axs[1].set_ylabel("revolution")
    axs[1].set_title("(b) in-band peak frequency\n(constant ⇒ rotor; scattered ⇒ noise)"); axs[1].grid(alpha=.3)
    axs[2].plot([s["snr"] for s in spectra], revs, "s-", color="tab:red")
    axs[2].axvline(snr_gate, color="gray", ls="--", label=f"gate {snr_gate:g}×")
    axs[2].set_xlabel("in-band SNR (× noise)"); axs[2].set_ylabel("revolution")
    axs[2].set_title("(c) tone SNR vs revolution\n(drone recedes with revolution)"); axs[2].grid(alpha=.3); axs[2].legend(fontsize=8)
    fig.suptitle(style.get("suptitle", f"{tag}: rotor tone vs revolution (boxed drone, ALL columns) — "
                 f"is the blade-pass recoverable under rotation at close range?"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, os.path.join(out_dir, "fft_vs_rev"), style)


def _box_events_mask(azw, y, t, tel, box):
    """All-column event mask inside a box (az, Y) within its revolution's time window."""
    (a0, a1), (y0, y1) = box["az_window"], box["y_window"]
    revs = revolution_windows(tel); k = box.get("rev")
    if k is not None and 0 <= k < len(revs):
        t0, t1 = revs[k]
    else:
        t0, t1 = box.get("t_window", [float(t.min()), float(t.max())])
    azc = (a0 + a1) / 2.0; azu = ((azw - azc + 180) % 360) - 180 + azc
    return (azu >= a0) & (azu <= a1) & (y >= y0) & (y <= y1) & (t >= t0) & (t <= t1), (a0, a1, y0, y1)


def drone_band_snr(t_box_s, *, fs=4000.0, band=(300.0, 1500.0), spin_hz=None, lo_exclude=50.0):
    """Drone/no-drone statistic: peak power in the rotor band (300–1500 Hz) vs the median power of the
    **rest of the signal space** (out-of-band, above the ego-motion floor). Returns (stat, spectrum)."""
    from gottlux.core.frequency import region_spectrum
    sp = region_spectrum(np.asarray(t_box_s, float) * 1e6, fs=fs, fmin=band[0], fmax=band[1],
                         derotate_hz=(8.0 * spin_hz if spin_hz else 0.0))
    f, p = sp.freqs, sp.power
    inb = (f >= band[0]) & (f <= band[1])
    rest = (f > lo_exclude) & ~inb                      # 50 Hz..band and band..Nyquist (exclude ego-motion <50)
    if not inb.any() or not rest.any():
        return None, sp
    return float(p[inb].max() / (np.median(p[rest]) + 1e-30)), sp


def figure_box_fft_dashboard(azw, y, t, tel, instances, height, *, tag, out_dir, fs=4000.0,
                             band=(300.0, 1500.0), spin_hz=None, style=None):
    """Master dashboard — one row per boxed instance: the de-rotated panorama crop (left) and its
    rotor-band FFT (right, 300–1500 Hz, all columns, spin-envelope removed) with the band-SNR statistic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = style or {}
    os.makedirs(out_dir, exist_ok=True)
    inst = list(instances)
    n = len(inst)
    fig, axs = plt.subplots(n, 2, figsize=tuple(style.get("figsize", (13, 1.9 * n))),
                            gridspec_kw={"width_ratios": [1.0, 1.5]}, squeeze=False)
    for i, b in enumerate(inst):
        m, (a0, a1, y0, y1) = _box_events_mask(azw, y, t, tel, b)
        axl, axr = axs[i, 0], axs[i, 1]
        pad = max(5.0, 0.2 * (a1 - a0))
        _panorama_imshow(axl, azw[m], y[m], height, az_range=(a0 - pad, a1 + pad), az_bins=160, y_bins=120,
                         smooth=True, cmap=style.get("panorama_cmap", "inferno"))
        axl.add_patch(plt.Rectangle((a0, y0), a1 - a0, y1 - y0, fill=False, edgecolor="#39c5cf", lw=1.2))
        axl.set_ylabel(f"rev {b.get('rev','?')}\n{b['bearing_deg']:.0f}°/{b['range_m']:.1f}m", fontsize=8)
        axl.tick_params(labelsize=6)
        stat, sp = drone_band_snr(t[m], fs=fs, band=band, spin_hz=spin_hz)
        fb = (sp.freqs >= band[0]) & (sp.freqs <= band[1])
        axr.plot(sp.freqs[fb], sp.power[fb], color="k", lw=0.7)
        if np.isfinite(sp.peak_freq):
            axr.axvline(sp.peak_freq, color="tab:red", ls="--", lw=1)
        axr.set_xlim(band); axr.set_ylim(bottom=0)
        axr.text(0.98, 0.86, f"peak {sp.peak_freq:.0f}Hz\nband/rest {stat:.1f}× · {int(m.sum())} ev",
                 transform=axr.transAxes, ha="right", va="top", fontsize=7,
                 bbox=dict(fc="white", ec="none", alpha=.7))
        axr.tick_params(labelsize=6)
        if i < n - 1:
            axl.set_xticklabels([]); axr.set_xticklabels([])
    axs[0, 0].set_title("de-rotated panorama (boxed drone)", fontsize=9)
    axs[0, 1].set_title(f"rotor-band FFT {band[0]:.0f}–{band[1]:.0f} Hz (all columns, linear)", fontsize=9)
    axs[-1, 0].set_xlabel("world azimuth [deg]"); axs[-1, 1].set_xlabel("frequency [Hz]")
    fig.suptitle(style.get("suptitle", f"{tag}: per-instance panorama + rotor-band FFT dashboard "
                 f"(spin-envelope removed; rotor band 300–1500 Hz)"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return _save(fig, os.path.join(out_dir, "box_fft_dashboard"), style)


def figure_egomotion_correction(t_box_s, *, spin_hz, tag, out_dir, fs=4000.0, style=None):
    """The ego-motion (slew) base tone and the attempt to back it out: the raw spectrum is dominated by
    the once-per-revolution (~spin Hz) burst comb; a moving-average high-pass (de-rotation) removes that
    envelope. Before/after, linear power (no log axis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gottlux.core.frequency import region_spectrum
    style = style or {}
    os.makedirs(out_dir, exist_ok=True)
    tb = np.asarray(t_box_s, float) * 1e6
    raw = region_spectrum(tb, fs=fs, fmin=1.0, fmax=2000.0, derotate_hz=0.0)
    cor = region_spectrum(tb, fs=fs, fmin=300.0, fmax=1500.0, derotate_hz=8.0 * spin_hz)
    fig, axs = plt.subplots(1, 3, figsize=tuple(style.get("figsize", (16, 4.4))))
    z = raw.freqs <= 15
    axs[0].plot(raw.freqs[z], raw.power[z], color="tab:purple")
    if spin_hz:
        for h in range(1, 13):
            axs[0].axvline(h * spin_hz, color="gray", lw=.4, alpha=.5)
    axs[0].set_title(f"(a) RAW spectrum, 0–15 Hz — the ego-motion base tone\nat the spin rate "
                     f"({spin_hz:.2f} Hz) + harmonics"); axs[0].set_xlabel("freq [Hz]"); axs[0].set_ylabel("power"); axs[0].grid(alpha=.3)
    full = raw.freqs <= 2000
    axs[1].plot(raw.freqs[full], raw.power[full], color="tab:purple")
    axs[1].set_title("(b) RAW, full band — the slew tone dwarfs the rotor band\n(why the rotational FFT is hard)")
    axs[1].set_xlabel("freq [Hz]"); axs[1].set_ylabel("power"); axs[1].set_ylim(bottom=0); axs[1].grid(alpha=.3)
    cf = cor.freqs <= 2000
    axs[2].plot(cor.freqs[cf], cor.power[cf], color="k")
    axs[2].axvspan(300, 1500, color="tab:green", alpha=.08, label="rotor band 300–1500")
    axs[2].set_title("(c) AFTER de-rotation (high-pass) — slew removed,\nrotor band revealed (still a forest, no single line)")
    axs[2].set_xlabel("freq [Hz]"); axs[2].set_ylabel("power"); axs[2].set_ylim(bottom=0); axs[2].grid(alpha=.3); axs[2].legend(fontsize=8)
    fig.suptitle(style.get("suptitle", f"{tag}: ego-motion / slew correction — back-propagating the "
                 f"once-per-revolution base tone out of the FFT"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, os.path.join(out_dir, "egomotion_correction"), style)


def figure_drone_detector(azw, y, t, tel, instances, height, *, tag, out_dir, fs=4000.0,
                          band=(300.0, 1500.0), spin_hz=None, background_az=(300.0, 340.0),
                          background_y=(0.0, 200.0), style=None):
    """Drone/no-drone via the rotor-band SNR vs the rest of the spectrum: each boxed instance's
    band/rest statistic against an empty (no-drone) reference region."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = style or {}
    os.makedirs(out_dir, exist_ok=True)
    stats = []
    for b in instances:
        m, _ = _box_events_mask(azw, y, t, tel, b)
        s, _sp = drone_band_snr(t[m], fs=fs, band=band, spin_hz=spin_hz)
        stats.append((b.get("rev"), s if s else 0.0, int(m.sum())))
    azc = (background_az[0] + background_az[1]) / 2.0; azu = ((azw - azc + 180) % 360) - 180 + azc
    bgm = (azu >= background_az[0]) & (azu <= background_az[1]) & (y >= background_y[0]) & (y <= background_y[1])
    bg_stat, _ = drone_band_snr(t[bgm], fs=fs, band=band, spin_hz=spin_hz) if int(bgm.sum()) > 200 else (0.0, None)
    thr = style.get("threshold") or max(2.0 * (bg_stat or 1.0), 4.0)
    fig, ax = plt.subplots(figsize=tuple(style.get("figsize", (11, 4.6))))
    revs = [f"rev {r}" for r, _, _ in stats]; vals = [v for _, v, _ in stats]
    colors = ["tab:green" if v >= thr else "tab:orange" for v in vals]
    ax.bar(revs, vals, color=colors)
    ax.axhline(bg_stat or 0.0, color="tab:red", ls="-", lw=1.5, label=f"no-drone reference (az {background_az[0]:.0f}–{background_az[1]:.0f}°): {bg_stat:.1f}×")
    ax.axhline(thr, color="gray", ls="--", lw=1.2, label=f"detect threshold {thr:.1f}×")
    ax.set_ylabel("rotor-band / rest-of-spectrum  (×)"); ax.set_xlabel("boxed instance")
    ax.set_title(style.get("title", f"{tag}: drone / no-drone by rotor-band SNR (300–1500 Hz) vs the rest "
                 f"of the signal space")); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    n_det = sum(1 for v in vals if v >= thr)
    return _save(fig, os.path.join(out_dir, "drone_detector"), style), {"background_stat": round(bg_stat or 0.0, 2),
            "threshold": round(thr, 2), "instances_detected": n_det, "n_instances": len(stats),
            "per_instance": [{"rev": r, "band_rest_snr": round(v, 2), "n_events": n} for r, v, n in stats]}


def track_over_revs(azw, y, t, tel, *, az, ywin, fov_deg, width, target_size_m=0.225,
                    t_min=None, t_max=None, min_pass=60):
    """Solve the SAME box on every revolution → a per-rev bearing/range time series (the drone's
    behaviour over the timeline)."""
    out = []
    for k, (a, b) in enumerate(revolution_windows(tel, t_min, t_max)):
        r = solve_box(azw, y, t, az=az, ywin=ywin, twin=(a, b), fov_deg=fov_deg, width=width,
                      target_size_m=target_size_m, min_pass=min_pass)
        if r["n_passes"]:
            out.append({"rev": k, "t_s": round((a + b) / 2.0, 3), "bearing_deg": r["bearing_deg"],
                        "range_m": r["range_m"], "n_events": r["n_events"]})
    return out


# --------------------------------------------------------------------------- figures (Agg, testable)
def _save(fig, base, style=None):
    from gottlux.io import export
    style = style or {}
    return export.save_figure(fig, base, dpi=style.get("dpi", 140),
                              formats=tuple(style.get("formats", ("png", "pdf"))), close=True)


def _panorama_imshow(ax, azw, y, height, *, az_range=(0, 360), az_bins=720, y_bins=None, smooth=False,
                     cmap="inferno", vmax=None):
    if y_bins is None:
        y_bins = min(int(height), 320)
    Hh, _, _ = np.histogram2d(azw, y, bins=[az_bins, y_bins], range=[list(az_range), [0, height]])
    img = np.log1p(Hh).T                                    # rows = Y, cols = az
    if smooth:
        from scipy import ndimage
        img = ndimage.gaussian_filter(img, 0.7)
    if vmax is None:                                        # pass a shared vmax to compare panels fairly
        vmax = np.percentile(img[img > 0], 99.5) if (img > 0).any() else 1.0
    # origin="upper" + flipped extent ⇒ sensor row 0 (chip top, high elevation) at the TOP
    ax.imshow(img, origin="upper", aspect="auto", cmap=cmap,
              extent=[az_range[0], az_range[1], height, 0], vmax=vmax)
    return vmax


def figure_context(azw, y, height, boxes, tag, out_dir, style=None):
    """The canonical rotational visualization: the de-rotated 360° panorama + the tactical radar,
    side by side, with the boxed detections — the most intuitive way to see the stitched event space."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = style or {}
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=tuple(style.get("figsize", (16, 6))))
    ax = fig.add_subplot(1, 2, 1)
    _panorama_imshow(ax, azw, y, height, smooth=True, cmap=style.get("panorama_cmap", "inferno"))
    for i, b in enumerate(boxes or []):
        a, yv = b["az_window"], b["y_window"]
        ax.add_patch(plt.Rectangle((a[0], yv[0]), a[1] - a[0], yv[1] - yv[0], fill=False,
                                   edgecolor=style.get("box_color", "#39c5cf"), lw=2))
        ax.text(a[0], yv[0] - 4, f"#{i+1} {b['bearing_deg']:.0f}°", color=style.get("box_color", "#39c5cf"), fontsize=8)
    ax.set_xlabel("world azimuth [deg]  (full 360° sweep, de-rotated)")
    ax.set_ylabel("sensor Y [px]  (chip top = up)")
    ax.set_title(style.get("panorama_title", "(a) de-rotated panorama — every column stitched into one 360° scene"))
    ax.set_xticks(np.arange(0, 361, 45))

    axr = fig.add_subplot(1, 2, 2, projection="polar")
    axr.set_theta_zero_location("N"); axr.set_theta_direction(-1); axr.set_facecolor((.04, .06, .04))
    if boxes:
        B = [np.deg2rad(b["bearing_deg"]) for b in boxes]; R = [b["range_m"] for b in boxes]
        axr.scatter(B, R, c=range(len(boxes)), cmap=style.get("radar_cmap", "cool"), s=90, edgecolors="w")
        for i, b in enumerate(boxes):
            axr.annotate(f"#{i+1}", (np.deg2rad(b["bearing_deg"]), b["range_m"]), color="w", fontsize=8)
    axr.set_title(style.get("radar_title", "(b) tactical radar — bearing × range to each boxed instance"))
    fig.suptitle(style.get("suptitle", f"{tag}: the de-rotated panorama + radar are the intuitive view "
                                       f"of rotational EBS data"), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, os.path.join(out_dir, "context_panorama_radar"), style)


def figure_panorama(azw, y, height, boxes, tag, out_dir, style=None):
    """Standalone de-rotated 360° panorama (the context view) with the boxed drone instances."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = style or {}
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=tuple(style.get("figsize", (15, 5.2))))
    _panorama_imshow(ax, azw, y, height, smooth=True, cmap=style.get("panorama_cmap", "inferno"))
    for i, b in enumerate(boxes or []):
        a, yv = b["az_window"], b["y_window"]
        ax.add_patch(plt.Rectangle((a[0], yv[0]), a[1] - a[0], yv[1] - yv[0], fill=False,
                                   edgecolor=style.get("box_color", "#39c5cf"), lw=1.6))
        ax.text(a[0], yv[0] - 4, f"#{i+1} {b['bearing_deg']:.0f}°",
                color=style.get("box_color", "#39c5cf"), fontsize=8)
    ax.set_xlabel("world azimuth [deg]  (full 360° sweep, de-rotated)")
    ax.set_ylabel("sensor Y [px]  (chip top = up)")
    ax.set_xticks(np.arange(0, 361, 30)); ax.set_xlim(0, 360)
    ax.set_title(style.get("title", f"{tag}: de-rotated 360° environmental panorama "
                 f"(one spinning EBS, telemetry-de-rotated) + operator-boxed drone"))
    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "panorama"), style)


def figure_radar(instances, track, tag, out_dir, style=None):
    """Standalone tactical radar — bearing × range to the drone, with the per-revolution track."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = style or {}
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=tuple(style.get("figsize", (7.2, 7.2))))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_facecolor(style.get("facecolor", (.04, .06, .04)))
    if instances:
        B = [np.deg2rad(b["bearing_deg"]) for b in instances]; R = [b["range_m"] for b in instances]
        ax.scatter(B, R, c=range(len(instances)), cmap=style.get("radar_cmap", "viridis"),
                   s=170, edgecolors="w", linewidths=1.2, zorder=3)
        for i, b in enumerate(instances):           # large, high-contrast, offset labels
            ax.annotate(str(i + 1), (np.deg2rad(b["bearing_deg"]), b["range_m"]),
                        textcoords="offset points", xytext=(9, 5), color="white", fontsize=12,
                        fontweight="bold", zorder=4,
                        bbox=dict(boxstyle="circle,pad=0.12", fc=(0, 0, 0, 0.55), ec="white", lw=0.6))
    rmax = style.get("rmax") or (max([b["range_m"] for b in (instances or [])] + [1]) * 1.15)
    ax.set_ylim(0, rmax); ax.set_rlabel_position(135)
    ax.set_title(style.get("title", f"{tag}: tactical radar — bearing × range (range = apparent size, coarse)"),
                 pad=18)
    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "radar"), style)


def figure_ebs_radar_map(azw, y, t, height, instances, *, tag, out_dir, style=None,
                         az_bins=360, r_bins=90, r_inner=0.12):
    """The radar map from GottLUX's EBSviewer (Radar mode), reproduced headlessly: a **PPI scope** where
    θ = world bearing and the radius = elevation (chip-top/up at the centre, horizon at the rim), with
    the de-rotated event density painted green-on-black. The whole 360° static environment becomes the
    radar 'landscape'; the drone is the off-landscape return — outlined here with its boxed track."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = style or {}
    os.makedirs(out_dir, exist_ok=True)
    rr = r_inner + (1.0 - r_inner) * (np.asarray(y, float) / height)
    Hh, ae, re = np.histogram2d(np.asarray(azw, float), rr, bins=[az_bins, r_bins], range=[[0, 360], [0, 1]])
    img = np.log1p(Hh)
    fig = plt.figure(figsize=tuple(style.get("figsize", (8.6, 8.6))), facecolor="k")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1); ax.set_facecolor("black")
    TH, RR = np.meshgrid(np.deg2rad(ae), re)
    vmax = np.percentile(img[img > 0], 99.5) if (img > 0).any() else 1.0
    ax.pcolormesh(TH, RR, img.T, cmap=style.get("cmap", "Greens"), vmax=vmax, shading="auto")
    ax.set_ylim(0, 1.0)
    ax.grid(color=(0.3, 0.8, 0.3), alpha=0.4); ax.tick_params(colors=(0.6, 1.0, 0.6))
    ax.spines["polar"].set_color((0.3, 0.8, 0.3)); ax.set_yticklabels([])
    if instances:
        bd = [np.deg2rad(b["bearing_deg"]) for b in instances]
        rd = [r_inner + (1.0 - r_inner) * ((b["y_window"][0] + b["y_window"][1]) / 2.0 / height) for b in instances]
        azs = [b["bearing_deg"] for b in instances]; a0, a1 = min(azs), max(azs)
        ax.fill_between(np.deg2rad(np.linspace(a0, a1, 60)), 0, 1, color="#39c5cf", alpha=0.08, zorder=2)
        ax.scatter(bd, rd, s=150, facecolors="none", edgecolors="#39c5cf", linewidths=2.0, zorder=5)
        for i, (aa, r) in enumerate(zip(bd, rd)):
            ax.annotate(str(i + 1), (aa, r), color="#39c5cf", fontsize=11, fontweight="bold", zorder=6,
                        textcoords="offset points", xytext=(8, 5))
        ax.annotate("drone track", (np.deg2rad((a0 + a1) / 2), 1.07), color="#39c5cf", fontsize=10,
                    ha="center", fontweight="bold")
    ax.set_title(style.get("title", f"{tag}: EBSviewer radar map (Radar mode) — θ=bearing, r=elevation, "
                 f"event density; the de-rotated 360° scene as a PPI, drone track outlined"),
                 color="w", pad=22)
    return _save(fig, os.path.join(out_dir, "ebs_radar_map"), style)


def figure_timeline(per_rev, tag, out_dir, style=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = style or {}
    os.makedirs(out_dir, exist_ok=True)
    T = [d["t_s"] for d in per_rev]; B = [d["bearing_deg"] for d in per_rev]; R = [d["range_m"] for d in per_rev]
    fig, axs = plt.subplots(1, 2, figsize=tuple(style.get("figsize", (13, 4.4))))
    axs[0].plot(T, B, "o-", color=style.get("bearing_color", "tab:blue"))
    axs[0].set_title(style.get("bearing_title", f"{tag}: drone bearing vs time (per revolution)"))
    axs[0].set_xlabel("t [s]"); axs[0].set_ylabel("bearing [deg]"); axs[0].grid(alpha=.3)
    if style.get("bearing_ylim"):
        axs[0].set_ylim(style["bearing_ylim"])
    axs[1].plot(T, R, "o-", color=style.get("range_color", "tab:red"))
    axs[1].set_title(style.get("range_title", "range vs time (apparent size — coarse)"))
    axs[1].set_xlabel("t [s]"); axs[1].set_ylabel("range [m]"); axs[1].grid(alpha=.3)
    if style.get("range_ylim"):
        axs[1].set_ylim(style["range_ylim"])
    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "box_timeline"), style)


def figure_fft(spec, tag, out_dir, box=None, spin_hz=None, style=None):
    """FFT of the boxed section over the rotor band, **linear** power axis (no log-y)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = style or {}
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = spec.band
    fig, ax = plt.subplots(figsize=tuple(style.get("figsize", (9.5, 4.6))))
    band = (spec.freqs >= lo) & (spec.freqs <= hi)
    ax.plot(spec.freqs[band], spec.power[band], color=style.get("line_color", "k"), lw=0.9)
    if np.isfinite(spec.peak_freq) and spec.peak_freq > 0:
        ax.axvline(spec.peak_freq, color=style.get("peak_color", "tab:red"), ls="--",
                   label=f"strongest tone in band: {spec.peak_freq:.0f} Hz (SNR {spec.snr:.0f})")
        ax.legend(fontsize=9)
    ax.set_xlim(lo, hi); ax.set_ylim(bottom=0)                       # linear y, from 0
    if style.get("ylim"):
        ax.set_ylim(style["ylim"])
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("power (linear)")
    bx = f"  az {box[0]:.0f}–{box[1]:.0f}°, Y {box[2]:.0f}–{box[3]:.0f}px" if box else ""
    sp = f"  (spin {spin_hz:.2f} Hz)" if spin_hz else ""
    ax.set_title(style.get("title", f"{tag}: FFT of the boxed section{bx} — rotor-band search "
                 f"{lo:.0f}–{hi:.0f} Hz{sp}\nper-rev burst comb (≤~{12*(spin_hz or 1):.0f} Hz) excluded "
                 f"by the {lo:.0f} Hz floor; a clean multi-rotor tone may still need a staring dwell (§6)"))
    ax.grid(alpha=.3); fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "box_fft"), style)


def _per_pass_bearing(au, t, gap_s=0.30, min_n=20):
    """Bearing = the de-rotated centroid taken PER REVOLUTION (so the drone's motion across revs is
    not averaged into it). Returns (median-of-per-pass-bearings, pass-to-pass scatter, n_passes)."""
    o = np.argsort(t); ts = t[o]; aus = au[o]
    segs = np.split(np.arange(ts.size), np.where(np.diff(ts) > gap_s)[0] + 1)
    bs = [float(np.median(aus[s])) for s in segs if s.size >= min_n]
    if not bs:
        return None, None, 0
    return float(np.median(bs)), (float(np.std(bs)) if len(bs) > 1 else 0.0), len(bs)


def figure_column_comparison(azw, y, x, t, width, height, *, box, tag, out_dir, n_few=8, style=None):
    """Side-by-side panorama of the boxed region — whole sensor vs a few central columns vs a single
    central column — annotated with the PER-PASS centroid bearing (identical across the three; only the
    azimuthal spread, the smear, differs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = style or {}
    n_few = int(style.get("n_few", n_few))
    cmap = style.get("cmap", "inferno"); bline = style.get("bearing_line_color", "#39ff14")
    os.makedirs(out_dir, exist_ok=True)
    az0, az1, y0, y1 = box
    azc = (az0 + az1) / 2.0
    maz = max(5.0, 0.15 * (az1 - az0))
    azr = (az0 - maz, az1 + maz)
    sel_box = (azw >= azr[0]) & (azw <= azr[1])
    in_box = (y >= y0) & (y <= y1) & (azw >= az0) & (azw <= az1)
    modes = [("all", "all 320 columns (full sensor)"), ("few", f"{n_few} central columns"),
             ("single", "1 central column")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.9), sharey=True)
    for ax, (mode, title) in zip(axs, modes):
        col = column_subset_keep(x, width, mode, n_few)
        ck = col & sel_box
        n = int(ck.sum())
        if n > 5:
            _panorama_imshow(ax, azw[ck], y[ck], height, az_range=azr, az_bins=240, y_bins=160,
                             smooth=(mode != "single"), cmap=cmap)
        ax.add_patch(plt.Rectangle((az0, y0), az1 - az0, y1 - y0, fill=False,
                                   edgecolor=style.get("box_color", "#39c5cf"), lw=1.5))
        cb = col & in_box
        bearing = scat = None; npass = 0
        if cb.sum() > 10:
            au = ((azw[cb] - azc + 180) % 360) - 180 + azc
            bearing, scat, npass = _per_pass_bearing(au, t[cb])
            if bearing is not None:
                ax.axvline(bearing, color=bline, ls="--", lw=1.5)
        bt = (f"\nbearing {bearing:.2f}° (±{scat:.2f}° over {npass} revs)" if bearing is not None
              else "\n(too sparse for a bearing)")
        ax.set_title(f"{title}\n{n:,} events{bt}", fontsize=9); ax.set_xlabel("world azimuth [deg]")
    axs[0].set_ylabel("sensor Y [px]")
    fig.suptitle(style.get("suptitle", f"{tag}: bearing = the de-rotated PER-REVOLUTION centroid (green "
                 f"dashed) — the same for the full sensor (a smeared streak) and a single column (a sharp "
                 f"line); only the spread (the smear) differs"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, os.path.join(out_dir, "column_comparison"), style)


def figure_event_rate(t, *, spin_hz, tag, out_dir, style=None, bin_ms=20.0):
    """Global event rate vs time (cyclical at the spin frequency) + its autocorrelation, from which the
    rotation rate is back-calculated (the period of the cyclical pattern = the autocorrelation's first
    lag peak). Telemetry-free spin estimate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = style or {}
    os.makedirs(out_dir, exist_ok=True)
    t = np.asarray(t, float); dt = bin_ms / 1000.0
    edges = np.arange(t.min(), t.max(), dt); tc = edges[:-1] + dt / 2
    rate = np.histogram(t, edges)[0] / dt
    r = rate - rate.mean()
    ac = np.correlate(r, r, "full")[r.size - 1:]; ac = ac / (ac[0] + 1e-20)
    lags = np.arange(ac.size) * dt
    lo = max(1, int(0.3 / dt))                          # ignore the zero-lag lobe (<0.3 s)
    pk = lo + int(np.argmax(ac[lo:int(2.0 / dt)])) if ac.size > int(2.0 / dt) else lo + int(np.argmax(ac[lo:]))
    T_est = lags[pk]; f_est = 1.0 / T_est if T_est > 0 else float("nan")
    fig, axs = plt.subplots(1, 2, figsize=tuple(style.get("figsize", (14, 4.4))))
    axs[0].plot(tc, rate / 1e6, color=style.get("rate_color", "k"), lw=0.8)
    axs[0].set_xlabel("t [s]"); axs[0].set_ylabel("event rate [Mev/s]")
    axs[0].set_title("(a) global event rate vs time — cyclical at the spin frequency"); axs[0].grid(alpha=.3)
    axs[1].plot(lags, ac, color="tab:purple"); axs[1].axvline(T_est, color="tab:red", ls="--",
                label=f"period {T_est*1000:.0f} ms → {f_est:.3f} Hz")
    if spin_hz:
        axs[1].axvline(1.0 / spin_hz, color="tab:green", ls=":", label=f"telemetry {spin_hz:.3f} Hz")
    axs[1].set_xlim(0, 2.0); axs[1].set_xlabel("lag [s]"); axs[1].set_ylabel("autocorrelation")
    axs[1].set_title("(b) autocorrelation → rotation rate (first lag peak)"); axs[1].grid(alpha=.3); axs[1].legend(fontsize=8)
    fig.suptitle(style.get("suptitle", f"{tag}: the spin rate is back-calculated from the event rate's "
                 f"once-per-revolution cycle"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, os.path.join(out_dir, "event_rate_spin"), style)


def figure_mask_reduction_over_time(tel, x, y, t, *, n_list=(0, 1, 2, 3, 4), tag, out_dir, style=None, bin_ms=50.0):
    """Event rate vs time for successive mask depths N — the rate visibly drops as more revolutions are
    folded into the static reference (the operational data-rate reduction, shown over time)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = style or {}
    os.makedirs(out_dir, exist_ok=True)
    t = np.asarray(t, float); dt = bin_ms / 1000.0
    edges = np.arange(t.min(), t.max(), dt); tc = edges[:-1] + dt / 2
    fig, ax = plt.subplots(figsize=tuple(style.get("figsize", (12, 4.6))))
    cmap = plt.get_cmap(style.get("cmap", "viridis"))
    for i, n in enumerate(n_list):
        keep = static_keep_mask(tel, x, y, t, 320, 320, n)
        rate = np.histogram(t[keep], edges)[0] / dt / 1e6
        red = 100 * (1 - keep.sum() / t.size)
        ax.plot(tc, rate, color=cmap(i / max(len(n_list) - 1, 1)), lw=1.0, label=f"N={n}  (−{red:.0f}%)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("kept event rate [Mev/s]")
    ax.set_title(style.get("title", f"{tag}: event rate over time falls with each successive background "
                 f"mask (reference depth N)")); ax.grid(alpha=.3); ax.legend(fontsize=8, title="mask revs")
    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "mask_reduction_over_time"), style)


def figure_mask_panoramas(azw, y, t, tel, x, height, *, n_list=(0, 2, 4), tag, out_dir, style=None):
    """De-rotated panorama at increasing mask depth N — a visual of the static scene being progressively
    removed while the moving drone survives."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = style or {}
    os.makedirs(out_dir, exist_ok=True)
    fig, axs = plt.subplots(1, len(n_list), figsize=tuple(style.get("figsize", (5.2 * len(n_list), 4.2))), sharey=True)
    if len(n_list) == 1:
        axs = [axs]
    vmax0 = None
    for ax, n in zip(axs, n_list):
        keep = static_keep_mask(tel, x, y, t, 320, height, n)
        red = 100 * (1 - keep.sum() / t.size)
        vm = _panorama_imshow(ax, azw[keep], y[keep], height, smooth=True,
                              cmap=style.get("panorama_cmap", "inferno"), vmax=vmax0)
        if vmax0 is None:
            vmax0 = vm                                      # lock all panels to the N=0 brightness
        ax.set_title(f"N={n}  (−{red:.0f}% events)"); ax.set_xlabel("world azimuth [deg]"); ax.set_xticks(np.arange(0, 361, 90))
    axs[0].set_ylabel("sensor Y [px]")
    fig.suptitle(style.get("suptitle", f"{tag}: successive background masking removes the static scene "
                 f"while the moving drone survives"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, os.path.join(out_dir, "mask_panoramas"), style)


def export_result(tag, fov_deg, target_size_m, boxes, azw_kept, y_kept, height, out_root):
    """Write the panorama+boxes figure, the tactical radar, and a JSON of the boxed instances."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gottlux.io import export
    os.makedirs(out_root, exist_ok=True)
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(2, 1, 1)
    _panorama_imshow(ax, azw_kept, y_kept, height, smooth=True)
    for i, b in enumerate(boxes):
        a, yv = b["az_window"], b["y_window"]
        ax.add_patch(plt.Rectangle((a[0], yv[0]), a[1] - a[0], yv[1] - yv[0], fill=False,
                                   edgecolor="#39c5cf", lw=2))
        ax.text(a[0], yv[0] - 4, f"#{i+1} {b['bearing_deg']:.1f}° {b['range_m']:.0f}m",
                color="#39c5cf", fontsize=8)
    ax.set_xlabel("world azimuth [deg]"); ax.set_ylabel("sensor Y [px] (chip top = up)")
    ax.set_title(f"{tag}: de-rotated panorama + operator-boxed drone instances")
    axr = fig.add_subplot(2, 1, 2, projection="polar")
    axr.set_theta_zero_location("N"); axr.set_theta_direction(-1); axr.set_facecolor((.04, .06, .04))
    B = [np.deg2rad(b["bearing_deg"]) for b in boxes]; R = [b["range_m"] for b in boxes]
    axr.scatter(B, R, c=range(len(boxes)), cmap="cool", s=90, edgecolors="w")
    for i, b in enumerate(boxes):
        axr.annotate(f"#{i+1}", (np.deg2rad(b["bearing_deg"]), b["range_m"]), color="w", fontsize=8)
    axr.set_title("tactical radar (bearing × range)")
    fig.tight_layout()
    _save(fig, os.path.join(out_root, "radarlab_result"))
    meta = {"tag": tag, "fov_deg": fov_deg, "target_size_m": target_size_m,
            "n_instances": len(boxes), "instances": boxes}
    export.save_json(meta, os.path.join(out_root, "radarlab_result.json"))
    return out_root


def _figure_experiment_summary(azw, y, height, instances, track, reduction, tag, out_dir, style=None):
    """The converged centerpiece figure: panorama+boxes | tactical radar | bearing vs time |
    range vs time + the data-rate reduction inset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = style or {}
    bc = style.get("bearing_color", "tab:blue"); rc = style.get("range_color", "tab:red")
    boxc = style.get("box_color", "#39c5cf")
    fig = plt.figure(figsize=tuple(style.get("figsize", (15, 8.5))))
    ax = fig.add_subplot(2, 2, 1)
    _panorama_imshow(ax, azw, y, height, smooth=True, cmap=style.get("panorama_cmap", "inferno"))
    for i, b in enumerate(instances):
        a, yv = b["az_window"], b["y_window"]
        ax.add_patch(plt.Rectangle((a[0], yv[0]), a[1] - a[0], yv[1] - yv[0], fill=False, edgecolor=boxc, lw=1.6))
        ax.text(a[0], yv[0] - 4, f"#{i+1}", color=boxc, fontsize=8)
    ax.set_title("(a) de-rotated panorama + boxed drone instances")
    ax.set_xlabel("world azimuth [deg]"); ax.set_ylabel("sensor Y [px] (up)"); ax.set_xticks(np.arange(0, 361, 45))

    axr = fig.add_subplot(2, 2, 2, projection="polar")
    axr.set_theta_zero_location("N"); axr.set_theta_direction(-1); axr.set_facecolor((.04, .06, .04))
    if instances:
        B = [np.deg2rad(b["bearing_deg"]) for b in instances]; R = [b["range_m"] for b in instances]
        axr.scatter(B, R, c=range(len(instances)), cmap=style.get("radar_cmap", "cool"), s=80, edgecolors="w")
    axr.set_title("(b) tactical radar — bearing × range")

    ax2 = fig.add_subplot(2, 2, 3)
    if track:
        ax2.plot([d["t_s"] for d in track], [d["bearing_deg"] for d in track], "-", color=bc,
                 alpha=.6, label="per-rev track")
    if instances:
        ti = [np.mean(b.get("t_window", [0, 0])) for b in instances]
        ax2.plot(ti, [b["bearing_deg"] for b in instances], "o", color=bc, label="boxed instances")
    ax2.set_title("(c) drone bearing vs time"); ax2.set_xlabel("t [s]"); ax2.set_ylabel("bearing [deg]")
    ax2.grid(alpha=.3); ax2.legend(fontsize=8)
    if style.get("bearing_ylim"):
        ax2.set_ylim(style["bearing_ylim"])

    ax3 = fig.add_subplot(2, 2, 4)
    if track:
        ax3.plot([d["t_s"] for d in track], [d["range_m"] for d in track], "-", color=rc, alpha=.6)
    if instances:
        ti = [np.mean(b.get("t_window", [0, 0])) for b in instances]
        ax3.plot(ti, [b["range_m"] for b in instances], "o", color=rc)
    ax3.set_title("(d) range vs time (apparent size — coarse)"); ax3.set_xlabel("t [s]"); ax3.set_ylabel("range [m]")
    ax3.grid(alpha=.3)
    if reduction:
        ins = ax3.inset_axes([0.58, 0.62, 0.4, 0.34])
        ins.plot([n for n, _ in reduction], [r for _, r in reduction], "s-", color="k", ms=3)
        ins.set_title("data-rate reduction", fontsize=7); ins.set_xlabel("mask revs N", fontsize=6)
        ins.set_ylabel("% dropped", fontsize=6); ins.tick_params(labelsize=6); ins.grid(alpha=.3)

    fig.suptitle(style.get("suptitle", f"{tag}: rotational single-EBS — detect · mask · operator-box → "
                 f"bearing & range (converged result)"), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, os.path.join(out_dir, "experiment_summary"), style)


#: The plot **ontology** — every report figure with its editable ``style`` and an ``enabled`` flag.
#: Edit ``report_manifest.json`` (titles, colours, cmaps, axis limits, dpi, which plots) and re-render
#: with ``rotational_experiment.py --manifest <file> [--only id1,id2]`` — only the listed/enabled plots
#: are regenerated, everything else is left untouched. ``style: {}`` means "use the built-in default".
def default_manifest(tag, *, fov_deg, target_size_m, az_sign, fft_box=None):
    return {
        "meta": {"tag": tag, "fov_deg": fov_deg, "target_size_m": target_size_m, "az_sign": az_sign,
                 "mask_n_list": [0, 1, 2, 3, 4], "fft_box": fft_box,
                 "_note": "edit a plot's 'style' or 'enabled' and re-render with --manifest/--only; "
                          "style keys per plot are shown pre-filled below."},
        "plots": {
            "experiment_summary": {"enabled": True, "style": {"suptitle": None, "panorama_cmap": "inferno",
                                   "bearing_color": "tab:blue", "range_color": "tab:red", "box_color": "#39c5cf",
                                   "radar_cmap": "cool", "bearing_ylim": None, "dpi": 140}},
            "context": {"enabled": True, "style": {"suptitle": None, "panorama_cmap": "inferno",
                        "box_color": "#39c5cf", "radar_cmap": "cool", "dpi": 140}},
            "panorama": {"enabled": True, "style": {"title": None, "panorama_cmap": "inferno",
                         "box_color": "#39c5cf", "dpi": 140}},
            "radar": {"enabled": True, "style": {"title": None, "radar_cmap": "viridis",
                      "track_color": "#2b8cbe", "rmax": None, "dpi": 140}},
            # headless approximation of EBSviewer's Radar-mode PPI; disabled by default — the study export
            # uses a placeholder until EBSviewer's view-mode rendering is fixed and the real map is exported.
            "ebs_radar_map": {"enabled": False, "style": {"title": None, "cmap": "Greens", "dpi": 140}},
            "timeline": {"enabled": True, "style": {"bearing_color": "tab:blue", "range_color": "tab:red",
                         "bearing_ylim": None, "range_ylim": None, "bearing_title": None, "range_title": None, "dpi": 140}},
            "masking_quant": {"enabled": True, "style": {"suptitle": None, "knee_N": 2, "dpi": 140}},
            "event_rate": {"enabled": True, "style": {"suptitle": None, "rate_color": "k", "dpi": 140}},
            "mask_reduction": {"enabled": True, "style": {"title": None, "cmap": "viridis", "dpi": 140}},
            "mask_panoramas": {"enabled": True, "style": {"suptitle": None, "panorama_cmap": "inferno", "dpi": 140}},
            "fft_vs_rev": {"enabled": True, "style": {"suptitle": None, "cmap": "magma", "fmin": 150, "fmax": 950, "dpi": 140}},
            "box_fft_dashboard": {"enabled": True, "style": {"suptitle": None, "panorama_cmap": "inferno", "fs": 4000, "band": [300, 1500], "dpi": 140}},
            "egomotion_correction": {"enabled": True, "style": {"suptitle": None, "fs": 4000, "dpi": 140}},
            "drone_detector": {"enabled": True, "style": {"title": None, "fs": 4000, "band": [300, 1500], "background_az": [300, 340], "background_y": [0, 200], "threshold": None, "dpi": 140}},
            "column_comparison": {"enabled": True, "style": {"suptitle": None, "cmap": "inferno",
                                  "n_few": 8, "box_color": "#39c5cf", "bearing_line_color": "#39ff14", "dpi": 140}},
            "fft": {"enabled": True, "style": {"fmin": 200, "fmax": 800, "line_color": "k",
                    "peak_color": "tab:red", "ylim": None, "title": None, "dpi": 140}},
        },
    }


def _load_json(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def run_experiment(tel, x, y, t, *, fov_deg, width, height, target_size_m, boxes, tag, out_dir,
                   az_sign=AZ_SIGN_DEFAULT, mask_n_list=(0, 1, 2, 3, 4), manifest=None, only=None):
    """Converge the rotational result into one publication figure set — study summary, panorama+radar
    context, per-rev timeline, column-subset comparison, masking quantification, FFT — plus a JSON +
    README. Manifest-driven (the plot ontology): each plot's ``style``/``enabled`` is editable, and
    ``only`` re-renders just the listed plots, leaving the rest untouched."""
    from gottlux.io import export
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "report_manifest.json")
    if manifest is None:
        manifest = _load_json(mpath) or default_manifest(tag, fov_deg=fov_deg, target_size_m=target_size_m,
                                                          az_sign=az_sign)
    # merge in any plot keys added since the manifest was written (preserves the user's edits)
    dm = default_manifest(tag, fov_deg=fov_deg, target_size_m=target_size_m, az_sign=az_sign)
    manifest.setdefault("plots", {})
    for pid, spec in dm["plots"].items():
        manifest["plots"].setdefault(pid, spec)
    meta = manifest.get("meta", {}); plots = manifest.get("plots", {})
    mask_n_list = tuple(meta.get("mask_n_list", mask_n_list))
    only = set(only) if only else None

    def on(pid):
        return plots.get(pid, {}).get("enabled", True) and (only is None or pid in only)

    def st(pid):
        return plots.get(pid, {}).get("style", {}) or {}

    azw = world_azimuth(tel, x, t, fov_deg, width, az_sign)
    instances = list(boxes or [])
    base = static_keep_mask(tel, x, y, t, width, height, 0)            # full environment for context
    track = []
    written = {}; det_meta = None
    drone_box = box0 = None
    if instances:
        az0 = min(b["az_window"][0] for b in instances); az1 = max(b["az_window"][1] for b in instances)
        y0 = min(b["y_window"][0] for b in instances); y1 = max(b["y_window"][1] for b in instances)
        drone_box = (az0, az1, y0, y1, az_sign, fov_deg); box0 = (az0, az1, y0, y1)
        track = track_over_revs(azw, y, t, tel, az=(az0, az1), ywin=(y0, y1), fov_deg=fov_deg,
                                width=width, target_size_m=target_size_m)
    mm = masking_metrics(tel, x, y, t, width=width, height=height, drone_box=drone_box, n_list=mask_n_list)
    reduction = [(r["N"], r["reduction_pct"]) for r in mm["rows"]]

    if on("masking_quant"):
        written["masking_quant"] = figure_masking_quant(mm, tag, out_dir, style=st("masking_quant"))[0]
    if box0 is not None and on("column_comparison"):
        written["column_comparison"] = figure_column_comparison(azw[base], y[base], x[base], t[base], width,
                                                                height, box=box0, tag=tag, out_dir=out_dir,
                                                                style=st("column_comparison"))[0]
    if on("experiment_summary"):
        written["experiment_summary"] = _figure_experiment_summary(azw[base], y[base], height, instances, track,
                                                                   reduction, tag, out_dir, style=st("experiment_summary"))[0]
    if on("context"):
        written["context"] = figure_context(azw[base], y[base], height, instances, tag, out_dir, style=st("context"))[0]
    if on("panorama"):
        written["panorama"] = figure_panorama(azw[base], y[base], height, instances, tag, out_dir, style=st("panorama"))[0]
    if on("radar"):
        written["radar"] = figure_radar(instances, track, tag, out_dir, style=st("radar"))[0]
    if on("ebs_radar_map"):
        written["ebs_radar_map"] = figure_ebs_radar_map(azw[base], y[base], t[base], height, instances,
                                                        tag=tag, out_dir=out_dir, style=st("ebs_radar_map"))[0]
    if track and on("timeline"):
        written["timeline"] = figure_timeline(track, tag, out_dir, style=st("timeline"))[0]
    if on("fft"):
        fb = meta.get("fft_box") or box0
        if fb:
            fs = st("fft"); azc = (fb[0] + fb[1]) / 2.0
            azu = ((azw - azc + 180) % 360) - 180 + azc
            sel = base & (azu >= fb[0]) & (azu <= fb[1]) & (y >= fb[2]) & (y <= fb[3])
            spin = (1.0 / tel.T_rot) if tel.T_rot else None
            if int(sel.sum()) >= 64:
                spec = box_spectrum(t[sel], fmin=fs.get("fmin", 200), fmax=fs.get("fmax", 800), spin_hz=spin)
                written["fft"] = figure_fft(spec, tag, out_dir, box=tuple(fb[:4]), spin_hz=spin, style=fs)[0]
    spin_hz = (1.0 / tel.T_rot) if tel.T_rot else None
    if on("event_rate"):
        written["event_rate"] = figure_event_rate(t, spin_hz=spin_hz, tag=tag, out_dir=out_dir, style=st("event_rate"))[0]
    if on("mask_reduction"):
        written["mask_reduction"] = figure_mask_reduction_over_time(tel, x, y, t, n_list=mask_n_list, tag=tag,
                                                                    out_dir=out_dir, style=st("mask_reduction"))[0]
    if on("mask_panoramas"):
        written["mask_panoramas"] = figure_mask_panoramas(azw, y, t, tel, x, height, tag=tag, out_dir=out_dir,
                                                          style=st("mask_panoramas"))[0]
    if instances and on("fft_vs_rev"):
        fr = st("fft_vs_rev")
        sp = per_rev_spectra(azw, y, t, tel, instances, fmin=fr.get("fmin", 150), fmax=fr.get("fmax", 950), spin_hz=spin_hz)
        if sp:
            written["fft_vs_rev"] = figure_fft_vs_rev(sp, tag, out_dir, style=fr)[0]
    if instances and on("box_fft_dashboard"):
        s = st("box_fft_dashboard")
        written["box_fft_dashboard"] = figure_box_fft_dashboard(azw, y, t, tel, instances, height, tag=tag,
            out_dir=out_dir, fs=s.get("fs", 4000), band=tuple(s.get("band", (300, 1500))), spin_hz=spin_hz, style=s)[0]
    if instances and spin_hz and on("egomotion_correction"):
        bb = max(instances, key=lambda b: b.get("n_events", 0))
        em, _ = _box_events_mask(azw, y, t, tel, bb)
        if int(em.sum()) >= 256:
            s = st("egomotion_correction")
            written["egomotion_correction"] = figure_egomotion_correction(t[em], spin_hz=spin_hz, tag=tag,
                out_dir=out_dir, fs=s.get("fs", 4000), style=s)[0]
    if instances and on("drone_detector"):
        s = st("drone_detector")
        dd, det_meta = figure_drone_detector(azw, y, t, tel, instances, height, tag=tag, out_dir=out_dir,
            fs=s.get("fs", 4000), band=tuple(s.get("band", (300, 1500))), spin_hz=spin_hz,
            background_az=tuple(s.get("background_az", (300, 340))), background_y=tuple(s.get("background_y", (0, 200))), style=s)
        written["drone_detector"] = dd[0]

    export.save_json(manifest, mpath)                   # merge-preserving (setdefault never overwrites edits)

    summary = {"tag": tag, "fov_deg": fov_deg, "target_size_m": target_size_m, "az_sign": az_sign,
               "n_instances": len(instances), "n_track_revs": len(track),
               "data_rate_reduction_pct": dict(reduction), "masking_metrics": mm,
               "bearing_deg_median": (round(float(np.median([b["bearing_deg"] for b in instances])), 2) if instances else None),
               "range_m_median": (round(float(np.median([b["range_m"] for b in instances])), 2) if instances else None),
               "drone_detector": det_meta,
               "instances": instances, "per_rev_track": track, "figures": written}
    export.save_json(summary, os.path.join(out_dir, "experiment.json"))
    _write_readme(summary, out_dir)
    return out_dir


def _write_readme(s, out_dir):
    bm = s["bearing_deg_median"]; rm = s["range_m_median"]
    red = s["data_rate_reduction_pct"]
    lines = [
        f"# {s['tag']} — rotational single-EBS experiment (converged)\n",
        "Operator-in-the-loop result: detect the drone, suppress static clutter (background masking), "
        "and **box each drone instance across the revolution sequence** → bearing + range. The cue for "
        "a staring confirm/range dwell (cue-and-confirm).\n",
        "## Result",
        f"- **{s['n_instances']} boxed drone instances**, tracked across **{s['n_track_revs']} revolutions**.",
        (f"- **Bearing {bm:.1f}°** (median of instances); **range {rm:.1f} m** (apparent-size, coarse)."
         if bm is not None else "- (no boxes yet — box instances in the Radar/Box Lab, then re-run.)"),
        f"- **Background masking data-rate reduction:** " + ", ".join(f"N={n}→{r:.0f}%" for n, r in red.items()) + ".",
        f"- De-rotation azimuth sign = {s['az_sign']:+.0f} (rig convention).\n",
        "## Figures",
        "- `experiment_summary.png` — the converged centerpiece (panorama+boxes | radar | bearing vs t | range vs t).",
        "- `context_panorama_radar.png` — the intuitive panorama + radar view.",
        "- `box_timeline.png` — per-revolution bearing/range time series.",
        "- `column_comparison.png` — full sensor vs few/single central column (line-scan; the bearing is "
        "the per-pass centroid, identical across views).",
        "- `masking_quantification.png` — successive masking quantified: reduction %, kept event rate, "
        "drone vs background retention (self-masking knee N≈2), drone:background contrast.",
        "- `box_fft.png` — FFT of the boxed section (rotor band, linear y).",
        "- `experiment.json` — all numbers (incl. `masking_metrics`).\n",
        "## Editing / regenerating plots (the ontology)",
        "Every figure is described in **`report_manifest.json`** (the plot ontology): each plot has an "
        "`enabled` flag and a `style` block (title, colours, cmap, axis limits, dpi, …). To tweak one "
        "plot, edit its `style` and re-render *just that one* — the rest are left untouched:",
        "```",
        "python scripts/rotational_experiment.py <capture> --fov 20 --tag <tag> \\",
        "    --boxes box_track/radarlab_result.json \\",
        "    --manifest experiment/report_manifest.json --only timeline",
        "```",
        "Omit `--only` to re-render all enabled plots; set a plot's `enabled:false` to drop it.\n",
        "Methods & honest limits: `docs/ROTATIONAL_EBS_METHODS.md`. Range is the weak axis (no "
        "triangulation baseline on a pure-rotation sensor); rotor classification needs the staring dwell.",
    ]
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _open_path(p):
    from gottlux.io.paths import open_in_file_browser
    open_in_file_browser(p)                          # cross-platform (Explorer / open / xdg-open)


# --------------------------------------------------------------------------- Qt application
def _build_gui():
    import pyqtgraph as pg
    from PySide6 import QtCore, QtWidgets

    from gottlux.app import icons

    class RadarBoxLab(QtWidgets.QWidget):
        def __init__(self, rec, *, fov_deg, target_size_m=0.225, tag=None, out_root=None, parent=None):
            super().__init__(parent)
            self.rec = rec
            self.W, self.Hh = rec.width, rec.height
            self.fov_deg = float(fov_deg)
            self.az_sign = AZ_SIGN_DEFAULT
            self.tag = tag or (rec.name or "capture")
            self.out_root = out_root or os.path.join("gottlux_runs", "box_track", self.tag)
            self.boxes = []

            w = rec.all()
            self.x = np.asarray(w.x, float); self.y = np.asarray(w.y, float)
            self.t = np.asarray(w.t_s, float)
            self.tel = rec.telemetry
            self.spin_hz = (1.0 / self.tel.T_rot) if self.tel.T_rot else None
            self.revs = revolution_windows(self.tel)
            self.cur_rev = -1                          # -1 = whole clip
            self.azw = world_azimuth(self.tel, self.x, self.t, self.fov_deg, self.W, self.az_sign)
            self.keep = np.ones(self.t.size, bool)

            # ---- panorama canvas ----
            self.glw = pg.GraphicsLayoutWidget()
            self.vb = self.glw.addViewBox()
            self.vb.invertY(True)                      # sensor row 0 (chip top, up) at the TOP
            self.img = pg.ImageItem()
            try:
                self.img.setColorMap(pg.colormap.get("inferno", source="matplotlib"))
            except Exception:
                pass
            self.vb.addItem(self.img)
            self.box = pg.RectROI([20, self.Hh * 0.35], [40, self.Hh * 0.3], pen=pg.mkPen("#39c5cf", width=2))
            self.box.addScaleHandle([1, 1], [0, 0]); self.box.addScaleHandle([0, 0], [1, 1])
            self.vb.addItem(self.box); self.box.sigRegionChanged.connect(self._on_box)
            # lasso (polygon) — outline the diagonal all-column streak that a rectangle can't fit tightly
            self.lasso_on = False
            v0 = [[16, self.Hh * 0.35], [60, self.Hh * 0.35], [60, self.Hh * 0.65], [16, self.Hh * 0.65]]
            self.poly = pg.PolyLineROI(v0, closed=True, pen=pg.mkPen("#ffaa00", width=2))
            self.poly.setVisible(False); self.poly.sigRegionChanged.connect(self._on_box)
            self.vb.addItem(self.poly)

            # ---- tactical radar ----
            self.radar = pg.PlotWidget()
            self.radar.setAspectLocked(True); self.radar.hideAxis("bottom"); self.radar.hideAxis("left")
            self.radar.setBackground("#0b0f0b"); self._radar_max = 30.0; self._draw_radar_grid()
            self.radar_pts = pg.ScatterPlotItem(size=12, pen=pg.mkPen("w"), brush=pg.mkBrush("#39c5cf"))
            self.radar.addItem(self.radar_pts)

            # ---- controls ----
            self.fov = QtWidgets.QDoubleSpinBox(); self.fov.setRange(0.1, 180); self.fov.setDecimals(2)
            self.fov.setValue(self.fov_deg); self.fov.setSuffix(" °"); self.fov.valueChanged.connect(self._refov)
            self.size_m = QtWidgets.QDoubleSpinBox(); self.size_m.setRange(0.001, 100); self.size_m.setDecimals(3)
            self.size_m.setValue(target_size_m); self.size_m.setSuffix(" m"); self.size_m.valueChanged.connect(self._solve_current)
            self.flip = QtWidgets.QCheckBox("flip azimuth sign (default = correct for the rig)")
            self.flip.toggled.connect(self._reflip)
            self.mask_n = QtWidgets.QSpinBox(); self.mask_n.setRange(0, 8); self.mask_n.setValue(0)
            self.mask_n.setToolTip("Drop events recurring in the first N revolutions (static clutter). 0 = off.")
            self.mask_btn = QtWidgets.QPushButton("Apply mask / rebuild"); self.mask_btn.clicked.connect(self._rebuild)

            self.colmode = QtWidgets.QComboBox(); self.colmode.addItems(["all columns", "few central", "single central"])
            self.colmode.currentIndexChanged.connect(self._rebuild)
            self.ncols = QtWidgets.QSpinBox(); self.ncols.setRange(2, 64); self.ncols.setValue(8)
            self.ncols.valueChanged.connect(self._rebuild)

            dur = float(self.t.max())
            self.t0 = QtWidgets.QDoubleSpinBox(); self.t0.setRange(0, dur); self.t0.setDecimals(2); self.t0.setSuffix(" s")
            self.t1 = QtWidgets.QDoubleSpinBox(); self.t1.setRange(0, dur); self.t1.setDecimals(2); self.t1.setValue(dur); self.t1.setSuffix(" s")
            self.t0.valueChanged.connect(self._rebuild); self.t1.valueChanged.connect(self._rebuild)
            self.rev_prev = QtWidgets.QPushButton("rev"); self.rev_prev.setIcon(icons.icon("arrow-left"))
            self.rev_prev.clicked.connect(lambda: self._step_rev(-1))
            self.rev_next = QtWidgets.QPushButton("rev"); self.rev_next.setIcon(icons.icon("arrow-right"))
            self.rev_next.setLayoutDirection(QtCore.Qt.RightToLeft)   # arrow on the right
            self.rev_next.clicked.connect(lambda: self._step_rev(+1))
            self.rev_lbl = QtWidgets.QLabel("rev: all"); self.rev_all = QtWidgets.QPushButton("all")
            self.rev_all.clicked.connect(self._rev_all)
            self.snap_btn = QtWidgets.QPushButton("⊕ snap box to drone"); self.snap_btn.clicked.connect(self._suggest_box)
            self.autosnap = QtWidgets.QCheckBox("auto-snap on rev step"); self.autosnap.setChecked(True)
            self.autoadv = QtWidgets.QCheckBox("auto-advance after Add"); self.autoadv.setChecked(True)
            self.exp_btn = QtWidgets.QPushButton("Converge → export figure bundle")
            self.exp_btn.setIcon(icons.icon("play", color="ACCENT_TEXT"))   # dark mark on the accent fill
            self.exp_btn.setObjectName("primary"); self.exp_btn.clicked.connect(self._experiment)
            self.lasso = QtWidgets.QCheckBox("✎ lasso (polygon) — fit the diagonal all-column streak")
            self.lasso.toggled.connect(self._toggle_lasso)
            self.mask_quant_btn = QtWidgets.QPushButton("▦ Masking quantification")
            self.mask_quant_btn.clicked.connect(self._masking)

            self.readout = QtWidgets.QLabel("draw a box around a drone instance"); self.readout.setWordWrap(True)
            self.readout.setTextFormat(QtCore.Qt.RichText)
            self.label = QtWidgets.QLineEdit(); self.label.setPlaceholderText("instance label (optional)")
            self.add_btn = QtWidgets.QPushButton("Add this box"); self.add_btn.setObjectName("primary")
            self.add_btn.setIcon(icons.icon("target", color="ACCENT_TEXT")); self.add_btn.clicked.connect(self._add_box)
            self.del_btn = QtWidgets.QPushButton("Delete selected")
            self.del_btn.setIcon(icons.icon("close")); self.del_btn.clicked.connect(self._del_box)

            self.track_btn = QtWidgets.QPushButton("Track box over all revs → timeline")
            self.track_btn.setIcon(icons.icon("sync")); self.track_btn.clicked.connect(self._track)
            self.fft_btn = QtWidgets.QPushButton("∿ FFT of boxed section"); self.fft_btn.clicked.connect(self._fft)
            self.colcmp_btn = QtWidgets.QPushButton("▥ Column comparison figure"); self.colcmp_btn.clicked.connect(self._colcmp)
            self.ctx_btn = QtWidgets.QPushButton("◳ Save context (panorama+radar)"); self.ctx_btn.clicked.connect(self._context)
            self.export_btn = QtWidgets.QPushButton("Export preliminary result…")
            self.export_btn.setIcon(icons.icon("export")); self.export_btn.clicked.connect(self._export)

            self.table = QtWidgets.QTableWidget(0, 5)
            self.table.setHorizontalHeaderLabels(["az°", "Δaz°", "range m", "passes", "label"])
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.table.cellClicked.connect(self._on_row)

            # ---- layout ----
            form = QtWidgets.QFormLayout()
            form.addRow("FOV", self.fov); form.addRow("Target size", self.size_m); form.addRow(self.flip)
            colrow = QtWidgets.QHBoxLayout(); colrow.addWidget(QtWidgets.QLabel("columns")); colrow.addWidget(self.colmode); colrow.addWidget(self.ncols)
            mrow = QtWidgets.QHBoxLayout(); mrow.addWidget(QtWidgets.QLabel("mask revs")); mrow.addWidget(self.mask_n); mrow.addWidget(self.mask_btn)
            trow = QtWidgets.QHBoxLayout(); trow.addWidget(QtWidgets.QLabel("t")); trow.addWidget(self.t0); trow.addWidget(self.t1)
            rrow = QtWidgets.QHBoxLayout(); rrow.addWidget(self.rev_prev); rrow.addWidget(self.rev_lbl, 1); rrow.addWidget(self.rev_next); rrow.addWidget(self.rev_all)
            srow = QtWidgets.QHBoxLayout(); srow.addWidget(self.snap_btn); srow.addWidget(self.autosnap); srow.addWidget(self.autoadv)
            right = QtWidgets.QVBoxLayout()
            right.addLayout(form); right.addLayout(colrow); right.addLayout(mrow); right.addLayout(trow); right.addLayout(rrow); right.addLayout(srow)
            box_grp = QtWidgets.QGroupBox("Box each drone instance per revolution"); bgl = QtWidgets.QVBoxLayout(box_grp)
            bgl.addWidget(self.lasso); bgl.addWidget(self.readout); bgl.addWidget(self.label)
            arow = QtWidgets.QHBoxLayout(); arow.addWidget(self.add_btn); arow.addWidget(self.del_btn); bgl.addLayout(arow)
            right.addWidget(box_grp)
            an = QtWidgets.QGroupBox("Analysis & converge"); anl = QtWidgets.QVBoxLayout(an)
            anl.addWidget(self.exp_btn)
            for b in (self.track_btn, self.fft_btn, self.colcmp_btn, self.mask_quant_btn, self.ctx_btn):
                anl.addWidget(b)
            right.addWidget(an)
            right.addWidget(QtWidgets.QLabel("Boxed drone instances")); right.addWidget(self.table, 1)
            right.addWidget(self.radar, 1); right.addWidget(self.export_btn)
            rw = QtWidgets.QWidget(); rw.setLayout(right); rw.setMinimumWidth(340)
            scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(rw); scroll.setMinimumWidth(360)

            split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            self.glw.setMinimumSize(380, 320); split.addWidget(self.glw); split.addWidget(scroll)
            split.setStretchFactor(0, 1); split.setStretchFactor(1, 0); split.setSizes([900, 380])
            lay = QtWidgets.QVBoxLayout(self); lay.addWidget(split)
            self._rebuild()

        # -------------------------------------------------- panorama / rebuild
        def _col_mode(self):
            return ("all", "few", "single")[self.colmode.currentIndex()]

        def _rebuild(self, *_):
            t0, t1 = self.t0.value(), self.t1.value()
            tw = (self.t >= t0) & (self.t <= t1)
            ck = column_subset_keep(self.x, self.W, self._col_mode(), self.ncols.value())
            self.keep = static_keep_mask(self.tel, self.x, self.y, self.t, self.W, self.Hh,
                                         self.mask_n.value()) & tw & ck
            az = self.azw[self.keep]; yy = self.y[self.keep]
            Hist, _, _ = np.histogram2d(az, yy, bins=[720, min(int(self.Hh), 320)], range=[[0, 360], [0, self.Hh]])
            img = np.log1p(Hist)
            vmax = np.percentile(img[img > 0], 99.5) if (img > 0).any() else 1.0
            self.img.setImage(img, levels=(0, vmax), autoLevels=False)
            self.img.setRect(QtCore.QRectF(0, 0, 360, self.Hh))
            self.vb.setRange(xRange=(0, 360), yRange=(0, self.Hh), padding=0.02)
            kept = int(self.keep.sum()); tot = self.t.size
            self.setWindowTitle(f"GottLUX Radar/Box Lab — {self.tag} [{self.fov_deg:g}°, {self._col_mode()} cols]"
                                f"  ({kept:,}/{tot:,} ev, {100*(1-kept/max(tot,1)):.0f}% dropped)")
            self._solve_current()

        # -------------------------------------------------- revolution stepping
        def _set_rev(self, k):
            self.cur_rev = k
            if k < 0 or not self.revs:
                self.rev_lbl.setText("rev: all")
                b = QtCore.QSignalBlocker(self.t0); self.t0.setValue(0); del b
                self.t1.setValue(float(self.t.max()))                # triggers _rebuild
            else:
                a, bb = self.revs[k]; self.rev_lbl.setText(f"rev: {k+1}/{len(self.revs)}  [{a:.2f}–{bb:.2f}s]")
                bl = QtCore.QSignalBlocker(self.t0); self.t0.setValue(a); del bl
                self.t1.setValue(bb)                 # triggers _rebuild
            if k >= 0 and self.autosnap.isChecked():
                self._suggest_box()

        def _step_rev(self, d):
            if not self.revs:
                return
            k = 0 if self.cur_rev < 0 else min(max(self.cur_rev + d, 0), len(self.revs) - 1)
            self._set_rev(k)

        def _rev_all(self):
            self._set_rev(-1)

        def _suggest_box(self):
            """Snap the box onto the densest mover in the current (masked, rev-windowed) view."""
            b = densest_box(self.azw[self.keep], self.y[self.keep], height=self.Hh)
            if b is None:
                return
            self.box.setPos([b[0], b[2]]); self.box.setSize([max(b[1] - b[0], 1), max(b[3] - b[2], 1)])

        # -------------------------------------------------- box solving
        def _toggle_lasso(self, on):
            self.lasso_on = bool(on)
            self.poly.setVisible(on); self.box.setVisible(not on)
            self._solve_current()

        def _poly_verts(self):
            st = self.poly.getState(); pos = self.poly.pos()
            return [(pos.x() + float(p[0]), pos.y() + float(p[1])) for p in st["points"]]

        def _box_window(self):
            if self.lasso_on:
                v = self._poly_verts()
                azs = [p[0] for p in v]; ys = [p[1] for p in v]
                return (min(azs), max(azs)), (min(ys), max(ys))     # bounding box of the lasso
            pos = self.box.pos(); size = self.box.size()
            a0 = float(pos.x()); a1 = a0 + float(size.x()); y0 = float(pos.y()); y1 = y0 + float(size.y())
            return (min(a0, a1), max(a0, a1)), (min(y0, y1), max(y0, y1))

        def _solve(self, az, ywin):
            sub = self.keep
            if self.lasso_on:
                v = self._poly_verts()
                if len(v) >= 3:
                    sel = polygon_keep(self.azw[sub], self.y[sub], v)
                    azs = [p[0] for p in v]
                    return solve_box(self.azw[sub], self.y[sub], self.t[sub], az=(min(azs), max(azs)),
                                     ywin=(0, self.Hh), sel=sel, fov_deg=self.fov.value(), width=self.W,
                                     target_size_m=self.size_m.value())
            return solve_box(self.azw[sub], self.y[sub], self.t[sub], az=az, ywin=ywin,
                             twin=(self.t0.value(), self.t1.value()), fov_deg=self.fov.value(),
                             width=self.W, target_size_m=self.size_m.value())

        def _masking(self):
            az, ywin, _ = self._cur
            drone_box = (az[0], az[1], ywin[0], ywin[1], self.az_sign, self.fov.value())
            mm = masking_metrics(self.tel, self.x, self.y, self.t, width=self.W, height=self.Hh,
                                 drone_box=drone_box)
            self._msg("Masking quantification", figure_masking_quant(mm, self.tag, self.out_root))

        def _on_box(self, *_):
            self._solve_current()

        def _solve_current(self, *_):
            az, ywin = self._box_window(); r = self._solve(az, ywin)
            if r["n_passes"]:
                self.readout.setText(
                    f"<b>bearing {r['bearing_deg']:.2f}°</b> (span {r['bearing_span_deg']:.2f}° / "
                    f"{r['n_passes']} passes)<br><b>range {r['range_m']:.1f} m</b> "
                    f"({r['range_span_m'][0]:.1f}–{r['range_span_m'][1]:.1f} m)<br>"
                    f"{r['n_events']:,} ev · az {az[0]:.0f}–{az[1]:.0f}° · Y {ywin[0]:.0f}–{ywin[1]:.0f}px")
            else:
                self.readout.setText(f"box too sparse ({r['n_events']:,} ev) — enclose a drone pass")
            self._cur = (az, ywin, r)

        # -------------------------------------------------- table + radar
        def _add_box(self):
            az, ywin, r = self._cur
            if not r["n_passes"]:
                return
            r = dict(r); r["az_window"] = [round(az[0], 1), round(az[1], 1)]
            r["y_window"] = [round(ywin[0], 1), round(ywin[1], 1)]
            r["t_window"] = [round(self.t0.value(), 2), round(self.t1.value(), 2)]
            r["rev"] = self.cur_rev; r["label"] = self.label.text().strip()
            self.boxes.append(r); self._refresh_table(); self._refresh_radar()
            if self.autoadv.isChecked() and 0 <= self.cur_rev < len(self.revs) - 1:
                self._step_rev(+1)                   # advance to the next revolution to box again

        def _del_box(self):
            row = self.table.currentRow()
            if 0 <= row < len(self.boxes):
                self.boxes.pop(row); self._refresh_table(); self._refresh_radar()

        def _refresh_table(self):
            from PySide6 import QtWidgets as _W
            self.table.setRowCount(len(self.boxes))
            for r, b in enumerate(self.boxes):
                vals = [f"{b['bearing_deg']:.2f}", f"{b['bearing_span_deg']:.2f}", f"{b['range_m']:.1f}",
                        str(b["n_passes"]), b.get("label", "")]
                for c, v in enumerate(vals):
                    self.table.setItem(r, c, _W.QTableWidgetItem(v))

        def _on_row(self, row, _c):
            if 0 <= row < len(self.boxes):
                b = self.boxes[row]; a = b["az_window"]; y = b["y_window"]
                self.box.setPos([a[0], y[0]]); self.box.setSize([max(a[1] - a[0], 1), max(y[1] - y[0], 1)])

        def _draw_radar_grid(self):
            import pyqtgraph as pg
            for rr in (0.25, 0.5, 0.75, 1.0):
                c = pg.QtWidgets.QGraphicsEllipseItem(-rr, -rr, 2 * rr, 2 * rr); c.setPen(pg.mkPen("#1d3b1d")); self.radar.addItem(c)
            for ang in range(0, 360, 30):
                a = np.deg2rad(ang); self.radar.addItem(pg.PlotDataItem([0, np.sin(a)], [0, np.cos(a)], pen=pg.mkPen("#16301a")))

        def _refresh_radar(self):
            if not self.boxes:
                self.radar_pts.setData([]); return
            self._radar_max = max(30.0, max(b["range_m"] for b in self.boxes) * 1.1)
            spots = [{"pos": (b["range_m"] / self._radar_max * np.sin(np.deg2rad(b["bearing_deg"])),
                              b["range_m"] / self._radar_max * np.cos(np.deg2rad(b["bearing_deg"]))), "data": b}
                     for b in self.boxes]
            self.radar_pts.setData(spots)

        # -------------------------------------------------- analysis buttons (save figure + open)
        def _msg(self, title, paths):
            from PySide6 import QtWidgets as _W
            p = paths[0] if isinstance(paths, (list, tuple)) else paths
            _open_path(p); _W.QMessageBox.information(self, title, f"Wrote / opened:\n{p}")

        def _track(self):
            az, ywin, _ = self._cur
            per = track_over_revs(self.azw[self.keep], self.y[self.keep], self.t[self.keep], self.tel,
                                  az=az, ywin=ywin, fov_deg=self.fov.value(), width=self.W,
                                  target_size_m=self.size_m.value())
            if len(per) < 2:
                from PySide6 import QtWidgets as _W
                _W.QMessageBox.information(self, "Track", "Too few revolutions resolved — widen the box."); return
            self._msg("Per-rev timeline", figure_timeline(per, self.tag, self.out_root))

        def _fft(self):
            az, ywin, _ = self._cur
            azc = (az[0] + az[1]) / 2.0
            azu = ((self.azw - azc + 180) % 360) - 180 + azc
            m = self.keep & (azu >= az[0]) & (azu <= az[1]) & (self.y >= ywin[0]) & (self.y <= ywin[1])
            if int(m.sum()) < 64:
                from PySide6 import QtWidgets as _W
                _W.QMessageBox.information(self, "FFT", "Too few events in the box for an FFT."); return
            spec = box_spectrum(self.t[m], spin_hz=self.spin_hz)
            self._msg("Box FFT", figure_fft(spec, self.tag, self.out_root,
                                            box=(az[0], az[1], ywin[0], ywin[1]), spin_hz=self.spin_hz))

        def _colcmp(self):
            az, ywin, _ = self._cur
            # use the un-column-filtered events (mask+time only) so 'all' really is the full sensor
            t0, t1 = self.t0.value(), self.t1.value()
            base = static_keep_mask(self.tel, self.x, self.y, self.t, self.W, self.Hh, self.mask_n.value()) \
                & (self.t >= t0) & (self.t <= t1)
            self._msg("Column comparison",
                      figure_column_comparison(self.azw[base], self.y[base], self.x[base], self.t[base],
                                               self.W, self.Hh, box=(az[0], az[1], ywin[0], ywin[1]),
                                               tag=self.tag, out_dir=self.out_root, n_few=self.ncols.value()))

        def _context(self):
            base = static_keep_mask(self.tel, self.x, self.y, self.t, self.W, self.Hh, self.mask_n.value())
            self._msg("Context (panorama + radar)",
                      figure_context(self.azw[base], self.y[base], self.Hh, self.boxes, self.tag, self.out_root))

        def _experiment(self):
            from PySide6 import QtWidgets as _W
            out = os.path.join(os.path.dirname(self.out_root), "experiment")
            _W.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                run_experiment(self.tel, self.x, self.y, self.t, fov_deg=self.fov.value(), width=self.W,
                               height=self.Hh, target_size_m=self.size_m.value(), boxes=self.boxes,
                               tag=self.tag, out_dir=out, az_sign=self.az_sign)
            finally:
                _W.QApplication.restoreOverrideCursor()
            _open_path(out)
            _W.QMessageBox.information(self, "Converge",
                                      f"Wrote the converged figure bundle to:\n{out}\n\n"
                                      f"{len(self.boxes)} boxed instances.")

        # -------------------------------------------------- knobs
        def _refov(self, v):
            self.fov_deg = float(v); self.azw = world_azimuth(self.tel, self.x, self.t, self.fov_deg, self.W, self.az_sign); self._rebuild()

        def _reflip(self, on):
            self.az_sign = -AZ_SIGN_DEFAULT if on else AZ_SIGN_DEFAULT
            self.azw = world_azimuth(self.tel, self.x, self.t, self.fov_deg, self.W, self.az_sign); self._rebuild()

        def _export(self):
            from PySide6 import QtWidgets as _W
            if not self.boxes:
                _W.QMessageBox.information(self, "Export", "Add at least one boxed instance first."); return
            try:
                base = static_keep_mask(self.tel, self.x, self.y, self.t, self.W, self.Hh, self.mask_n.value())
                out = export_result(self.tag, self.fov_deg, self.size_m.value(), self.boxes,
                                    self.azw[base], self.y[base], self.Hh, self.out_root)
                _open_path(out); _W.QMessageBox.information(self, "Export", f"Wrote preliminary result to:\n{out}")
            except Exception as e:                       # pragma: no cover - GUI path
                _W.QMessageBox.critical(self, "Export", str(e))

    return RadarBoxLab


# --------------------------------------------------------------------------- entry points
def _load_rec(path, camera, assume_spin):
    import gottlux as eb
    from gottlux.io.telemetry import Telemetry, estimate_spin_period_s
    rec = eb.load(path, camera=camera, mode="auto", progress=lambda f: None)
    if rec.telemetry is None and assume_spin is not None:
        per = (1.0 / assume_spin) if assume_spin > 0 else estimate_spin_period_s(rec.t.astype(float) / 1e6)[0]
        if per:
            rec.attach_telemetry(Telemetry.from_spin(rec.duration_s, per), refine=False)
    if rec.telemetry is None:
        raise SystemExit(f"{path} [{camera}]: no rotation telemetry (pass --assume_spin auto)")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="GottLUX Radar/Box Lab — box drones on the de-rotated "
                                             "panorama → bearing + range, per-rev tracking, FFT, columns.")
    ap.add_argument("path"); ap.add_argument("--camera", default="cam0"); ap.add_argument("--fov", type=float, default=58.0)
    ap.add_argument("--target_size", type=float, default=0.225); ap.add_argument("--tag", default=None)
    ap.add_argument("--assume_spin", default=None, metavar="auto|HZ")
    ap.add_argument("--mask_rev", type=int, default=0)
    ap.add_argument("--solve", default=None, metavar="AZ0,AZ1,Y0,Y1", help="headless: solve one box and print")
    ap.add_argument("--track", default=None, metavar="AZ0,AZ1,Y0,Y1", help="headless: per-rev timeline (print)")
    ap.add_argument("--t", default=None, metavar="T0,T1")
    a = ap.parse_args(argv)
    spin = None if a.assume_spin is None else (0.0 if str(a.assume_spin).lower() == "auto" else float(a.assume_spin))
    tag = a.tag or os.path.basename(os.path.normpath(a.path))
    rec = _load_rec(a.path, a.camera, spin)

    if a.solve or a.track:
        import json
        spec = a.solve or a.track
        az0, az1, y0, y1 = (float(v) for v in spec.split(","))
        twin = tuple(float(v) for v in a.t.split(",")) if a.t else None
        w = rec.all(); x = np.asarray(w.x, float); y = np.asarray(w.y, float); t = np.asarray(w.t_s, float)
        keep = static_keep_mask(rec.telemetry, x, y, t, rec.width, rec.height, a.mask_rev)
        azw = world_azimuth(rec.telemetry, x[keep], t[keep], a.fov, rec.width)
        if a.track:
            tmin, tmax = (twin if twin else (None, None))
            per = track_over_revs(azw, y[keep], t[keep], rec.telemetry, az=(az0, az1), ywin=(y0, y1),
                                  fov_deg=a.fov, width=rec.width, target_size_m=a.target_size, t_min=tmin, t_max=tmax)
            print(json.dumps(per, indent=2)); print(f"{len(per)} revolutions tracked")
        else:
            r = solve_box(azw, y[keep], t[keep], az=(az0, az1), ywin=(y0, y1), twin=twin,
                          fov_deg=a.fov, width=rec.width, target_size_m=a.target_size)
            print(json.dumps(r, indent=2))
        return 0

    from PySide6 import QtWidgets
    RadarBoxLab = _build_gui()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    try:
        from gottlux.app import style; style.apply_app_style(app)
    except Exception:
        pass
    win = RadarBoxLab(rec, fov_deg=a.fov, target_size_m=a.target_size, tag=tag)
    win.resize(1320, 800); win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
