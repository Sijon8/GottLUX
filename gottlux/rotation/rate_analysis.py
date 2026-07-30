"""
rate_analysis.py — rotation-rate from the event-rate autocorrelation, + high-quality event-rate
plots, for spinning-EBS data.

A spinning sensor re-images the scene once per revolution, so the global **event rate** is periodic
at the rotation frequency. :func:`find_rotation_rate` recovers that period from the rate's
autocorrelation (reusing :func:`gottlux.io.telemetry.estimate_spin_period_s`), which is the
no-telemetry way to learn the spin — and a useful sanity check even when telemetry exists.

This also answers the "FFT gravity" problem: the same rotation periodicity dominates a region's
temporal FFT (the once-per-revolution burst envelope). Knowing the spin rate sets the high-pass
cutoff that removes it (see ``derotate_hz`` in :func:`gottlux.core.frequency.region_spectrum`).
"""
from __future__ import annotations

import os

import numpy as np


def find_rotation_rate(rec, *, bin_s: float = 0.005):
    """Estimate the rotation period/rate from the event-rate autocorrelation.

    Returns a dict: ``period_s, hz, confidence`` plus the arrays for plotting
    (``t, rate_hz, lags_s, acf``). Works with or without telemetry; if telemetry is present its
    period is included as ``telemetry_period_s`` for cross-check.
    """
    from gottlux.io.telemetry import estimate_spin_period_s
    t = rec.t.astype(np.float64) / 1e6
    dur = float(t[-1] - t[0]) if t.size else 0.0
    centers, rate = rec.event_rate(bin_s)
    period, conf = estimate_spin_period_s(t, bin_s=bin_s)
    # autocorrelation of the (smoothed, mean-removed) rate for the plot
    r = rate.astype(float)
    k = max(int(round(0.01 / bin_s)), 1)
    r = np.convolve(r, np.ones(k) / k, mode="same")
    rc = r - r.mean()
    acf = np.correlate(rc, rc, "full")[rc.size - 1:]
    acf = acf / acf[0] if acf.size and acf[0] > 0 else acf
    lags = np.arange(acf.size) * bin_s
    tel_period = float(getattr(rec.telemetry, "T_rot", np.nan)) if getattr(rec, "telemetry", None) else None
    return {
        "period_s": period, "hz": (1.0 / period if period else None),
        "confidence": conf, "duration_s": round(dur, 4),
        "telemetry_period_s": tel_period,
        "t": centers, "rate_hz": rate, "lags_s": lags, "acf": acf, "bin_s": bin_s,
    }


def event_rate_figure(res, title="Event rate & rotation period"):
    """High-quality 2-panel figure: event rate vs time (top) + its autocorrelation with the
    detected rotation period and harmonics marked (bottom). *res* is :func:`find_rotation_rate`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(res["t"]); rate = np.asarray(res["rate_hz"]) / 1e6
    period = res["period_s"]; hz = res["hz"]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6.8), facecolor="w",
                                   gridspec_kw={"height_ratios": [3, 2]})
    ax0.plot(t, rate, color="#1f4e8c", lw=0.7, alpha=0.9)
    # mark the revolution boundaries from the recovered period
    if period and period > 0:
        for k in range(1, int(t[-1] / period) + 1):
            ax0.axvline(k * period, color="#ef6c00", ls=":", lw=0.7, alpha=0.5)
    ax0.set_xlabel("time [s]"); ax0.set_ylabel("event rate [Mev/s]")
    ax0.set_title(title); ax0.grid(True, ls="--", alpha=0.3)
    ax0.margins(x=0.005)
    if hz:
        ax0.text(0.99, 0.95, f"spin ≈ {hz:.3f} Hz  (T = {period:.4f} s)",
                 transform=ax0.transAxes, ha="right", va="top", fontsize=10,
                 bbox=dict(boxstyle="round", fc="#fff3e0", ec="#ef6c00"))

    lags = np.asarray(res["lags_s"]); acf = np.asarray(res["acf"])
    ax1.plot(lags, acf, color="#1f4e8c", lw=1.0)
    if period and period > 0:
        for m in range(1, 5):
            if m * period <= lags[-1]:
                ax1.axvline(m * period, color="#c62828", ls="--", lw=1.0, alpha=0.8)
        ax1.annotate(f"T = {period:.4f} s", xy=(period, acf.max() * 0.9),
                     color="#c62828", fontsize=10)
    ax1.set_xlim(0, min(lags[-1], (period or 1.0) * 5))
    ax1.set_xlabel("lag [s]"); ax1.set_ylabel("autocorrelation")
    ax1.set_title("event-rate autocorrelation → rotation period"); ax1.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    return fig


def save_rotation_rate_report(rec, out_dir, *, dpi=200) -> list:
    """Find the rotation rate and write the event-rate figure (PNG+PDF) + a JSON. Returns paths."""
    from gottlux.io import export
    os.makedirs(out_dir, exist_ok=True)
    res = find_rotation_rate(rec)
    fig = event_rate_figure(res, title=f"Event rate & rotation period — {rec.name}")
    written = export.save_figure(fig, os.path.join(out_dir, "event_rate_rotation"), dpi=dpi,
                                 formats=("png", "pdf"), close=True)
    summary = {k: res[k] for k in ("period_s", "hz", "confidence", "duration_s", "telemetry_period_s")}
    written += export.save_json(summary, os.path.join(out_dir, "rotation_rate.json"))
    return written
