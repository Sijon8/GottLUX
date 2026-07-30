"""
theme.py — a single, consistent, publication-grade look for every gottlux figure.

Journal figures live or die on small, boring details: legible fonts at column width, tight
margins, true-black axes, perceptually-uniform colormaps, and a fixed style so a paper's
figures look like a *set*. :func:`apply` installs that style globally; :func:`figure` makes a
correctly-sized figure; and the custom colormaps give event imagery and the flicker map a
distinctive, honest palette.

All figure builders in :mod:`gottlux.viz` call :func:`apply` and size to one of the standard
journal widths (single column ≈ 3.5 in, double column ≈ 7.2 in).
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Standard journal column widths (inches).
COL_SINGLE = 3.5
COL_DOUBLE = 7.2

_RC = {
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#222222",
    "axes.grid": False,
    "axes.axisbelow": True,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.3,
    "image.interpolation": "nearest",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def apply():
    """Install the gottlux publication style globally (idempotent)."""
    mpl.rcParams.update(_RC)
    _register_colormaps()


def figure(width: float = COL_SINGLE, height: float | None = None, **kw):
    """Create a correctly-styled matplotlib Figure of *width* inches (golden ratio default)."""
    import matplotlib.pyplot as plt
    apply()
    if height is None:
        height = width / 1.618
    return plt.figure(figsize=(width, height), **kw)


# ------------------------------------------------------------------ colormaps
def _register_colormaps():
    try:
        mpl.colormaps["gottlux_events"]
        return                                     # already registered
    except KeyError:
        pass
    # Event map: deep navy → magenta → amber → white. Good contrast on dark backgrounds.
    ev = LinearSegmentedColormap.from_list(
        "gottlux_events",
        ["#05060f", "#1a1145", "#6a1b9a", "#d81b60", "#ff8f00", "#fff3c4", "#ffffff"])
    # Diverging polarity map: blue (OFF) → black → red (ON).
    pol = LinearSegmentedColormap.from_list(
        "gottlux_polarity", ["#2962ff", "#0a0a14", "#ff1744"])
    for name, cmap in (("gottlux_events", ev), ("gottlux_polarity", pol)):
        try:
            mpl.colormaps.register(cmap, name=name)
        except Exception:
            pass


def flicker_rgba(freq_map, snr_map, fmin, fmax, cmap="turbo", snr_ref=None):
    """Render a flicker map to an ``(H, W, 4)`` RGBA image: **hue = frequency**, **alpha =
    confidence (log-SNR)**. Cells with no detection are fully transparent so the map can be
    laid over a dim event image. Returns the RGBA array."""
    apply()
    freq = np.asarray(freq_map, float)
    snr = np.asarray(snr_map, float)
    valid = np.isfinite(freq)
    norm = np.clip((freq - fmin) / max(fmax - fmin, 1e-9), 0, 1)
    rgba = mpl.colormaps[cmap](norm)
    if snr_ref is None:
        snr_ref = np.nanpercentile(snr[valid], 98) if valid.any() else 1.0
    alpha = np.clip(np.log1p(np.maximum(snr, 0)) / np.log1p(max(snr_ref, 1e-9)), 0, 1)
    rgba[..., 3] = np.where(valid, alpha, 0.0)
    return rgba
