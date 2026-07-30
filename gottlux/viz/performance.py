"""
performance.py — the publication figures for the operator-facing results metrics (KPIs).

One figure per metric, each readable on its own (so a weak result in one never clutters the
others), plus a staring-vs-rotating comparison:

* :func:`tracking_range_figure`   — pixels-on-target vs range, with the trackable-pixel and
  Johnson thresholds and the measured detections overlaid.
* :func:`prop_frequency_figure`   — in-band rotor SNR vs range, the SNR gate, the fitted
  power-law (``SNR ∝ D^slope``), and the resolvable region.
* :func:`range_vs_time_figure`    — measured range(t) of each track, with optional ground truth.
* :func:`time_to_contact_figure`  — warning time vs detection range for a family of approach
  speeds, with the measured closing-speed point.
* :func:`comparison_figure`       — the three headline ranges for several datasets side by side.

All inputs are the plain result dataclasses from :mod:`gottlux.core.performance` (+ the raw
measured arrays where a scatter is wanted); matplotlib is imported lazily so importing the
package stays light.
"""
from __future__ import annotations

import numpy as np

from gottlux.core import photogrammetry as pg

_TRACK_C = "#1f4e8c"
_MEAS_C = "#111111"
_GATE_C = "#c62828"
_MODEL_C = "#ef6c00"


def _fig(w=9.0, h=5.2):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h), facecolor="w")
    return fig, ax


# ------------------------------------------------------------------ 1) tracking range
def tracking_range_figure(tr, measured_ranges=None, measured_px=None, title=None):
    """Pixels-on-target vs range, with the trackable-pixel + Johnson thresholds and data."""
    fig, ax = _fig()
    L, fov, W = tr.target_size_m, tr.fov_deg, tr.width_px
    pts_d = np.asarray(measured_ranges, float).ravel() if measured_ranges is not None else np.zeros(0)
    pts_n = np.asarray(measured_px, float).ravel() if measured_px is not None else np.zeros(0)
    anchors = [tr.capability_range_m, tr.measured_max_range_m] + list(tr.johnson_ranges_m.values())
    if pts_d.size:
        anchors.append(float(np.nanmax(pts_d)))
    d_max = max([a for a in anchors if a and np.isfinite(a)] + [1.0]) * 1.15
    D = np.linspace(max(d_max / 400, 0.05), d_max, 400)
    N = pg.pixels_on_target(L, D, fov, W)
    ax.plot(D, N, "-", color=_TRACK_C, lw=2,
            label=f"pinhole  N = L·f/D  (L={L:.3f} m, f={tr.focal_px:.0f} px)")

    # trackable-pixel threshold + its capability range
    ax.axhline(tr.track_px, color=_MODEL_C, ls="--", lw=1.3,
               label=f"trackable ≥ {tr.track_px:g} px")
    if tr.capability_range_m:
        ax.plot([tr.capability_range_m, tr.capability_range_m], [0, tr.track_px],
                color=_MODEL_C, ls=":", lw=1.0)
        ax.annotate(f"model track range\n{tr.capability_range_m:.1f} m",
                    xy=(tr.capability_range_m, tr.track_px), xytext=(4, 8),
                    textcoords="offset points", fontsize=8, color=_MODEL_C)
    # Johnson ladder (faint, for context)
    for task, npx in pg.JOHNSON_PIXELS.items():
        ax.axhline(npx, color="#999", ls="-.", lw=0.7, alpha=0.5)
        ax.annotate(f"{task} {npx:g}px → {tr.johnson_ranges_m.get(task, float('nan')):.0f} m",
                    xy=(d_max, npx), xytext=(-4, 2), textcoords="offset points",
                    ha="right", va="bottom", fontsize=7, color="#666")
    # measured detections + measured reach
    if pts_d.size and pts_n.size:
        m = min(pts_d.size, pts_n.size)
        ax.scatter(pts_d[:m], pts_n[:m], s=26, c=_MEAS_C, alpha=0.7, zorder=5, label="detections")
    if tr.measured_max_range_m:
        ax.axvline(tr.measured_max_range_m, color=_MEAS_C, ls="-", lw=1.2, alpha=0.8)
        lab = f"measured reach {tr.measured_max_range_m:.1f} m"
        if tr.effective_track_px:
            lab += f"\n(≈{tr.effective_track_px:g} px)"
        ax.annotate(lab, xy=(tr.measured_max_range_m, ax.get_ylim()[1]), xytext=(-4, -12),
                    textcoords="offset points", ha="right", va="top", fontsize=8, color=_MEAS_C)
    ax.set_xlabel("range to target  D  [m]")
    ax.set_ylabel("pixels on target  N  [px across]")
    ax.set_title(title or f"Tracking range — FOV {fov:.0f}°, {W} px, L={L:.2f} m")
    ax.set_xlim(0, d_max)
    ax.set_ylim(0, max(float(np.nanmax(N[D > d_max / 8])) * 1.1, tr.track_px * 2, 15))
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ 2) prop-frequency range
def prop_frequency_figure(pf, ranges=None, snr=None, title=None):
    """In-band rotor SNR vs range: the gate, the fitted power-law, and the resolvable region."""
    fig, ax = _fig()
    rng = np.asarray(ranges, float).ravel() if ranges is not None else np.zeros(0)
    s = np.asarray(snr, float).ravel() if snr is not None else np.zeros(0)
    anchors = [pf.capability_range_m, pf.measured_max_range_m, pf.d_ref_m]
    if rng.size:
        anchors.append(float(np.nanmax(rng)))
    d_max = max([a for a in anchors if a and np.isfinite(a)] + [1.0]) * 1.2

    # measured points
    if rng.size and s.size:
        m = min(rng.size, s.size)
        ax.scatter(rng[:m], s[:m], s=30, c=_MEAS_C, alpha=0.75, zorder=5, label="measured SNR")
    # fitted power law SNR = A·D^slope through the calibration anchor
    if pf.slope is not None and pf.d_ref_m and pf.snr_ref:
        A = pf.snr_ref / (pf.d_ref_m ** pf.slope)
        D = np.linspace(max(d_max / 400, 0.05), d_max, 300)
        ax.plot(D, A * D ** pf.slope, "-", color=_MODEL_C, lw=2,
                label=f"SNR ∝ D^{pf.slope:g}" + (f"  (R²={pf.r2})" if pf.r2 is not None else ""))
    # SNR gate
    ax.axhline(pf.snr_gate, color=_GATE_C, ls="--", lw=1.3, label=f"SNR gate = {pf.snr_gate:g}")
    # capability (model) prop-frequency range
    if pf.capability_range_m:
        ax.axvline(pf.capability_range_m, color=_MODEL_C, ls=":", lw=1.3)
        ax.annotate(f"prop-freq range\n{pf.capability_range_m:.1f} m  ({pf.model})",
                    xy=(pf.capability_range_m, pf.snr_gate), xytext=(5, 8),
                    textcoords="offset points", fontsize=8, color=_MODEL_C)
    if pf.measured_max_range_m:
        ax.axvline(pf.measured_max_range_m, color=_MEAS_C, ls="-", lw=1.0, alpha=0.7)
        ax.annotate(f"measured reach {pf.measured_max_range_m:.1f} m",
                    xy=(pf.measured_max_range_m, ax.get_ylim()[1] if s.size else pf.snr_gate),
                    xytext=(-4, -10), textcoords="offset points", ha="right", va="top",
                    fontsize=8, color=_MEAS_C)
    ax.set_xlabel("range to target  D  [m]")
    ax.set_ylabel("in-band rotor SNR  [peak / noise floor]")
    ax.set_title(title or f"Prop-frequency-resolution range (gate {pf.snr_gate:g})")
    if s.size and np.nanmax(s) / max(np.nanmin(s[s > 0], initial=1.0), 1e-6) > 30:
        ax.set_yscale("log")
    ax.set_xlim(0, d_max)
    ax.grid(True, ls="--", alpha=0.35, which="both")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ 3) range vs time
def range_vs_time_figure(tracks, truth=None, title=None):
    """Measured range(t) per track, with optional ground-truth range overlaid.

    *tracks* is a list of dicts ``{"id", "t", "range_m"}``; *truth* an optional dict
    ``{"t", "range_m"}`` of logged ground-truth ranges.
    """
    fig, ax = _fig(w=9.0, h=4.6)
    plotted = False
    for tk in (tracks or []):
        t = np.asarray(tk.get("t"), float).ravel()
        r = np.asarray(tk.get("range_m"), float).ravel()
        ok = np.isfinite(t) & np.isfinite(r)
        if ok.sum() >= 1:
            ax.plot(t[ok], r[ok], "-o", ms=3, lw=1.3, alpha=0.85, label=f"track #{tk.get('id', '?')}")
            plotted = True
    if truth is not None:
        tt = np.asarray(truth.get("t"), float).ravel()
        tr = np.asarray(truth.get("range_m"), float).ravel()
        ok = np.isfinite(tt) & np.isfinite(tr)
        if ok.sum() >= 1:
            ax.plot(tt[ok], tr[ok], "k--", lw=1.6, label="ground truth")
            plotted = True
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("range to target  [m]")
    ax.set_title(title or "Measured range vs time")
    ax.grid(True, ls="--", alpha=0.35)
    if plotted:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ 4) time to contact
def time_to_contact_figure(ttc, title=None):
    """Warning time vs detection range for a family of approach speeds, + the measured point."""
    fig, ax = _fig(w=8.4, h=5.0)
    speeds = sorted(ttc.nominal_sweep_s) or [ttc.approach_speed_mps]
    d_anchor = ttc.detect_range_m or 50.0
    D = np.linspace(0.5, d_anchor * 1.2, 200)
    for v in speeds:
        ax.plot(D, D / v, lw=1.6, alpha=0.9, label=f"{v:g} m/s")
    if ttc.detect_range_m and ttc.nominal_ttc_s:
        ax.scatter([ttc.detect_range_m], [ttc.nominal_ttc_s], s=60, c=_MODEL_C, zorder=6,
                   label=f"nominal @ {ttc.approach_speed_mps:g} m/s → {ttc.nominal_ttc_s:.2f} s")
        ax.plot([ttc.detect_range_m, ttc.detect_range_m], [0, ttc.nominal_ttc_s],
                color=_MODEL_C, ls=":", lw=1.0)
    if ttc.measured_closing_speed_mps and ttc.detect_range_m and ttc.measured_ttc_at_first_s:
        ax.scatter([ttc.detect_range_m], [ttc.measured_ttc_at_first_s], s=70, marker="D",
                   c=_MEAS_C, zorder=7,
                   label=f"measured @ {ttc.measured_closing_speed_mps:g} m/s → "
                         f"{ttc.measured_ttc_at_first_s:.2f} s")
    ax.set_xlabel("detection range  D  [m]")
    ax.set_ylabel("time to contact  [s]")
    ax.set_title(title or "Time-to-contact (warning time) vs detection range")
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(fontsize=8, loc="upper left", title="approach speed")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ 5) comparison
def comparison_figure(datasets, title=None):
    """Grouped bars of the three headline ranges for several datasets (e.g. staring vs rotating).

    *datasets* maps a label → dict with optional keys ``tracking_range_m``,
    ``prop_frequency_range_m``, ``detect_range_m``.
    """
    fig, ax = _fig(w=8.6, h=4.8)
    labels = list(datasets)
    metrics = [("tracking_range_m", "tracking range", _TRACK_C),
               ("prop_frequency_range_m", "prop-freq range", _MODEL_C)]
    x = np.arange(len(labels))
    width = 0.38
    for i, (key, name, color) in enumerate(metrics):
        vals = [float(datasets[l].get(key) or np.nan) for l in labels]
        ax.bar(x + (i - 0.5) * width, vals, width, label=name, color=color, alpha=0.9)
        for xi, v in zip(x + (i - 0.5) * width, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.1f}", (xi, v), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("range  [m]")
    ax.set_title(title or "Results comparison — tracking & prop-frequency range")
    ax.grid(True, axis="y", ls="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
