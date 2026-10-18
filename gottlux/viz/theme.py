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


def flicker_rgba(freq_map, snr_map, fmin, fmax, cmap="turbo", snr_ref=None, snr_floor=None):
    """Render a flicker map to an ``(H, W, 4)`` RGBA image: **hue = frequency**, **alpha =
    confidence (log-SNR above the noise floor)**. Cells with no detection are fully
    transparent so the map can be laid over a dim event image. Returns the RGBA array.

    The SNR of a noise-only cell is not 1: the largest of a few hundred noise bins sits about
    ln(N) above their median, so pure sensor noise scores SNR ≈ 8–12. Scaling alpha from
    SNR 0 drew every noise cell at 70–80 % opacity and buried the target under random hues.
    Alpha now rises from *snr_floor* (default: the median valid cell, i.e. the typical noise
    cell when the target covers a minority of cells) to *snr_ref* (default: the 99.5th
    percentile, at least 3× the floor). Pass ``snr_floor=0`` for the old look, or a fixed
    floor when most of the frame flickers."""
    apply()
    freq = np.asarray(freq_map, float)
    snr = np.asarray(snr_map, float)
    valid = np.isfinite(freq) & np.isfinite(snr)
    norm = np.clip((np.nan_to_num(freq) - fmin) / max(fmax - fmin, 1e-9), 0, 1)
    rgba = mpl.colormaps[cmap](norm)
    if snr_floor is None:
        snr_floor = float(np.median(snr[valid])) if valid.any() else 0.0
    if snr_ref is None:
        snr_ref = float(np.percentile(snr[valid], 99.5)) if valid.any() else 1.0
        snr_ref = max(snr_ref, 3.0 * snr_floor)
    lo = np.log1p(max(snr_floor, 0.0))
    hi = max(np.log1p(max(snr_ref, 1e-9)), lo + 1e-6)
    alpha = np.clip((np.log1p(np.maximum(np.nan_to_num(snr), 0)) - lo) / (hi - lo), 0, 1)
    rgba[..., 3] = np.where(valid, alpha, 0.0)
    return rgba


def wrap_title(text, width_in, fontsize=None):
    """Break *text* onto extra lines so a title fits *width_in* inches of figure.

    Run names are long (``Flicker map — Humming_Bird_Fight_merged_2``) and a single-column
    figure is 3.5 in, so an unwrapped title ran past the axes into the colourbar or off the
    page. Breaks at spaces or underscores, inside a word only when it has to."""
    if not text:
        return text
    fs = fontsize or mpl.rcParams.get("axes.titlesize", 9)
    fs = float(fs) if isinstance(fs, (int, float)) else 9.0
    max_chars = max(12, int(width_in * 72 / (0.6 * fs)))
    out = []
    for line in str(text).split("\n"):
        while len(line) > max_chars:
            cut = max(line.rfind(" ", 0, max_chars), line.rfind("_", 0, max_chars) + 1)
            cut = cut if cut > max_chars // 3 else max_chars
            out.append(line[:cut].rstrip())
            line = line[cut:].lstrip()
        out.append(line)
    return "\n".join(out)
