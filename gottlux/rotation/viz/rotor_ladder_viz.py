"""
rotor_ladder_viz.py — figures for the 360° rotor-ladder survey (:mod:`gottlux.rotation.rotor_scan`).

Three products, each consuming a :class:`~gottlux.rotation.rotor_scan.RotorScanResult`:

* :func:`scan_map_figure`   — the **360° survey**: blade-pass frequency vs world bearing for every
  scanned cell, with the template band shaded — *where else the rotor signature appears*.
* :func:`radar_ladder_figure` — the **target-acquisition radar**: a tactical polar map of the
  matched rotor detections at their (bearing, range), coloured by blade frequency.
* :func:`recurrence_figure` — **cross-revolution recurrence**: each track's bearing vs revolution
  with the fitted per-revolution offset (the target's relative motion — the "offset from the spin").

Pure matplotlib (Agg-safe); no Qt. Tactical green-on-black for the radar to match
:mod:`gottlux.rotation.viz.radar_map`; clean light panels for the analysis plots.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------- the 360° survey
def scan_map_figure(result, title="Rotor-ladder 360° survey"):
    """Blade-pass frequency vs world bearing for every scanned cell.

    Matched cells (same rotor as the template) are filled and sized by comb strength; other
    in-band detections are open circles; the shaded band is the template ``f ± tol``. This is the
    direct answer to "where else in the 360° does this signature occur".
    """
    dets = result.detections
    fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor="w")
    if dets:
        det_on = [d for d in dets if d.detected and d.blade_hz]
        matched = [d for d in det_on if d.matches_template]
        other = [d for d in det_on if not d.matches_template]
        if other:
            ax.scatter([d.bearing_deg for d in other], [d.blade_hz for d in other],
                       s=28, facecolors="none", edgecolors="#8a8a8a", linewidths=1.0,
                       label="in-band comb (other f)")
        if matched:
            sc = ax.scatter([d.bearing_deg for d in matched], [d.blade_hz for d in matched],
                            c=[d.comb_strength for d in matched], cmap="viridis",
                            s=[40 + 320 * d.comb_strength for d in matched],
                            vmin=0, vmax=1, edgecolors="k", linewidths=0.4, zorder=3,
                            label="rotor signature (matched)")
            cb = fig.colorbar(sc, ax=ax, pad=0.01)
            cb.set_label("comb strength")
    f0 = result.f_template_hz
    if f0:
        tol = result.blade_tol
        ax.axhspan(f0 * (1 - tol), f0 * (1 + tol), color="#39c5cf", alpha=0.15, zorder=0)
        ax.axhline(f0, color="#0b7285", ls="--", lw=1.2,
                   label=f"template f = {f0:g} Hz (±{tol*100:.0f}%)")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xlabel("world bearing [deg]")
    ax.set_ylabel("blade-pass frequency [Hz]")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- the radar map
def _polar_axes(fig, rmax):
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor((0.05, 0.05, 0.05))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, rmax)
    ax.grid(color=(0.3, 0.8, 0.3), alpha=0.5)
    ax.tick_params(colors="w")
    ax.spines["polar"].set_color((0.3, 0.8, 0.3))
    return ax


def radar_ladder_figure(result, title="Rotor-ladder target acquisition"):
    """Tactical polar radar of the matched rotor detections: θ = bearing, r = range (m),
    colour = blade-pass frequency. Falls back to a unitless radial order when no range is
    available (``target_size_m == 0`` or unresolved extents)."""
    matched = result.matched or [d for d in result.detections if d.detected]
    fig = plt.figure(figsize=(7.6, 7.6), facecolor="k")
    az = np.array([d.bearing_deg for d in matched], float)
    rng = np.array([(d.range_m if d.range_m is not None else np.nan) for d in matched], float)
    hz = np.array([(d.blade_hz or np.nan) for d in matched], float)
    have_range = np.isfinite(rng).any()
    if not have_range and az.size:
        rng = np.argsort(np.argsort(az)).astype(float) + 1.0   # pseudo-range = detection order
    rmax = float(np.nanpercentile(rng, 95) * 1.15) if np.isfinite(rng).any() else 1.0
    ax = _polar_axes(fig, max(rmax, 1.0))
    if az.size:
        sc = ax.scatter(np.deg2rad(az), rng, c=hz, cmap="turbo", s=120,
                        edgecolors="w", linewidths=0.6, zorder=3)
        cb = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.8)
        cb.ax.yaxis.set_tick_params(color="w")
        plt.setp(cb.ax.get_yticklabels(), color="w")
        cb.set_label("blade-pass frequency [Hz]", color="w")
        # connect each track's passes (shows the bearing march of a moving target)
        for tr in result.tracks:
            tb = np.deg2rad(np.array(tr.bearings_deg, float))
            tr_rng = np.full_like(tb, tr.range_m if tr.range_m else (rmax * 0.6))
            ax.plot(tb, tr_rng, color=(0, 0.9, 0), lw=1.2, alpha=0.7, zorder=2)
    ax.set_title(title, color="w", fontsize=13, pad=18)
    sub = ("θ = bearing · r = range [m] · colour = blade Hz" if have_range
           else "θ = bearing · r = detection order (ranging disabled)")
    fig.text(0.5, 0.03, sub, color="c", ha="center", fontsize=9)
    return fig


# --------------------------------------------------------------------- recurrence / offset
def recurrence_figure(result, title="Cross-revolution recurrence & relative motion"):
    """Bearing vs revolution for each linked track, with the fitted per-revolution offset line.

    A horizontal line = a stationary rotor repeating each revolution; a sloped line = a moving
    target, the slope being its azimuth offset per revolution (its relative angular rate)."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="w",
                                   gridspec_kw={"width_ratios": [3, 2]})
    tracks = result.tracks
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(tracks), 1)))
    if not tracks:
        ax0.text(0.5, 0.5, "no recurring rotor track\n(need ≥ 2 revolutions)",
                 ha="center", va="center", transform=ax0.transAxes, color="#888")
    for k, tr in enumerate(tracks):
        revs = np.array(tr.revs, float)
        brg = np.array(tr.bearings_deg, float)
        c = colors[k % len(colors)]
        lab = (f"f≈{tr.median_blade_hz:g} Hz · {tr.bearing_offset_per_rev_deg:+.2f}°/rev "
               f"({tr.omega_deg_s:+.2f}°/s)")
        ax0.plot(revs, brg, "o-", color=c, lw=1.5, ms=6, label=lab)
        if revs.size >= 2:
            rr = np.linspace(revs.min(), revs.max(), 10)
            b0 = brg[0] + tr.bearing_offset_per_rev_deg * (rr - revs[0])
            ax0.plot(rr, b0, "--", color=c, lw=1.0, alpha=0.6)
    ax0.set_xlabel("revolution index")
    ax0.set_ylabel("world bearing [deg]")
    ax0.set_title("ladder recurrence (the per-rev offset = motion)")
    ax0.grid(True, ls="--", alpha=0.3)
    if tracks:
        ax0.legend(fontsize=7, loc="best")
        if revs.size:
            ax0.xaxis.get_major_locator().set_params(integer=True)

    # right panel: blade-frequency stability across revolutions
    for k, tr in enumerate(tracks):
        revs = np.array(tr.revs, float)
        f = np.array([h if h else np.nan for h in tr.blade_hz], float)
        ax1.plot(revs, f, "o-", color=colors[k % len(colors)], lw=1.4, ms=5)
    if result.f_template_hz:
        ax1.axhline(result.f_template_hz, color="#0b7285", ls="--", lw=1.0)
    ax1.set_xlabel("revolution index")
    ax1.set_ylabel("blade-pass frequency [Hz]")
    ax1.set_title("frequency stability")
    ax1.grid(True, ls="--", alpha=0.3)
    if tracks and revs.size:
        ax1.xaxis.get_major_locator().set_params(integer=True)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
