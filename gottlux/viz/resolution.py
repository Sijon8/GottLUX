"""
resolution.py — the publication figure for a pixels-on-target / perception-range study.

A single, journal-styled figure: the pinhole pixels-on-target curve ``N(D) = L·f/D`` over range,
the measured keyframe boxes overlaid as points, and the Johnson detection / recognition /
identification thresholds drawn as horizontal lines with their implied maximum ranges annotated.
"""
from __future__ import annotations

import numpy as np

from gottlux.core import photogrammetry as pg


def pixels_on_target_figure(study, size_used_m=None, title=None):
    """Build the pixels-on-target-vs-range figure for a :class:`ResolutionStudy`."""
    import matplotlib.pyplot as plt

    L = study.target_size_m if size_used_m is None else size_used_m
    fov, W = study.fov_deg, study.width_px
    # measured points
    pts = [(float(k.distance_m), max(k.w_px, k.h_px), k.label)
           for k in study.keyframes if k.distance_m is not None and k.distance_m > 0]
    ranges = pg.perception_ranges(L, fov, W)
    d_max = max([d for d, _, _ in pts] + list(ranges.values()) + [1.0]) * 1.15
    D = np.linspace(max(d_max / 400, 0.1), d_max, 400)
    N = pg.pixels_on_target(L, D, fov, W)

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor="w")
    ax.plot(D, N, "-", color="#1f4e8c", lw=2,
            label=f"pinhole model  N = L·f/D  (L={L:.3f} m, f={pg.focal_px(fov, W):.0f} px)")
    # Johnson thresholds + their max ranges
    colors = {"detection": "#2e7d32", "orientation": "#9e9d24",
              "recognition": "#ef6c00", "identification": "#c62828"}
    for task, npx in pg.JOHNSON_PIXELS.items():
        ax.axhline(npx, color=colors.get(task, "#888"), ls="--", lw=1.1, alpha=0.8)
        dr = ranges[task]
        ax.annotate(f"{task}: {npx:g} px → {dr:.0f} m",
                    xy=(d_max, npx), xytext=(-6, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8, color=colors.get(task, "#444"))
        ax.plot([dr, dr], [0, npx], color=colors.get(task, "#888"), ls=":", lw=0.9, alpha=0.7)
    # measured keyframes
    if pts:
        dd = [p[0] for p in pts]; nn = [p[1] for p in pts]
        ax.scatter(dd, nn, s=46, c="#111", zorder=5, label="measured keyframes")
        for d, n, lab in pts:
            if lab:
                ax.annotate(lab, (d, n), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("range to target  D  [m]")
    ax.set_ylabel("pixels on target  N  [px across]")
    ax.set_title(title or f"Pixels on target vs range — FOV {fov:.0f}°, {W}px")
    ax.set_xlim(0, d_max); ax.set_ylim(0, max(np.nanmax(N[D > d_max / 8]) * 1.1, 15))
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig
