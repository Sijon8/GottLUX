"""
tonemap.py — dynamic-range compression for event imagery ("map expression").

Event frames have a brutal dynamic range: a rotor disk or a specular glint can fire
hundreds of events while a faint, interesting target fires a handful, and a plain
linear map then *dilutes* the faint structure into black while the hot region clips to
white. This module is the single place that fixes that, two independent ways:

1. **Expression** — a monotone tone curve applied before the colormap, so low-intensity
   regions are lifted relative to hot ones without throwing away the ordering:
   ``linear``, ``sqrt``, ``gamma``, ``log``, ``asinh``, ``equalize`` (histogram
   equalization), ``percentile`` (hard clip at a high percentile).
2. **Scale** — where the reference white-point comes from. ``dynamic`` recomputes it
   per frame (best contrast on the current moment); ``static`` holds a fixed reference
   (frames stay comparable over time and a bright flash does not rescale everything).

Both :func:`compress` (non-negative magnitude frames) and :func:`compress_signed`
(diverging ON−OFF polarity frames) return a display array already in a colormap-ready
range plus the reference scale they used, so a caller can freeze that scale for
``static`` mode.

The tone curves themselves are pure NumPy and cheap (one pass over the frame) and nothing
here imports Qt. The one matplotlib touch-point is :func:`colormap`, which every colormap
lookup in gottlux goes through; it imports matplotlib lazily, so importing this module
stays free.
"""
from __future__ import annotations

import numpy as np

# The expressions offered in the GUI, in a sensible menu order.
EXPRESSIONS = ["linear", "sqrt", "gamma", "log", "asinh", "equalize", "percentile"]

EXPR_HELP = {
    "linear": "No compression — value maps straight to colour. Hot regions dominate.",
    "sqrt": "Square-root curve — gently lifts faint structure; a good default.",
    "gamma": "Power curve x**gamma (gamma<1 lifts shadows). Tune with the gamma control.",
    "log": "Logarithmic over ~2 decades — strongest lift of faint regions; rotor disks stop clipping.",
    "asinh": "Inverse-sinh — linear near zero, log-like for bright values (astronomy's choice).",
    "equalize": "Histogram equalization — maximizes local contrast; reveals texture, distorts magnitude.",
    "percentile": "Linear but hard-clipped at the clip percentile — kills hot outliers, keeps magnitude.",
}

SCALE_HELP = {
    "dynamic": "Recompute the white-point every frame (best instantaneous contrast).",
    "static": "Hold a fixed white-point (frames stay comparable; a flash won't rescale the view).",
}


def colormap(name, fallback: str = "viridis"):
    """The named matplotlib colormap — the single place gottlux resolves one.

    ``matplotlib.cm.get_cmap`` was removed in matplotlib 3.11, so the lookup goes through
    the ``matplotlib.colormaps`` registry (present since 3.5, and the only spelling that
    works on every version this project supports).

    A name matplotlib does not know — a stale saved preference, or a pyqtgraph-only name —
    falls back to *fallback* rather than raising. That matters because several of these
    lookups happen inside a Qt ``paintEvent``, where an escaping exception does not fail
    politely: PySide6 aborts the process.
    """
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except (KeyError, TypeError):
        return matplotlib.colormaps[fallback]


def reference_white(frame, clip_pct: float = 99.5) -> float:
    """The white-point (value mapped to 1.0): the *clip_pct* percentile of positive pixels.

    Using a high percentile rather than the raw max rejects a handful of hot/stuck pixels
    that would otherwise compress everything else into the floor.
    """
    f = np.asarray(frame, np.float64)
    pos = f[f > 0]
    if pos.size == 0:
        return 1.0
    v = float(np.percentile(pos, clip_pct))
    return max(v, 1e-9)


def _equalize(disp01: np.ndarray) -> np.ndarray:
    """Histogram-equalize an array already in [0, 1] using its non-zero pixels' CDF."""
    f = disp01.ravel()
    nz = f > 0
    if nz.sum() < 4:
        return disp01
    hist, edges = np.histogram(f[nz], bins=256, range=(0.0, 1.0))
    cdf = np.cumsum(hist).astype(np.float64)
    cdf /= cdf[-1] if cdf[-1] > 0 else 1.0
    out = np.interp(f, 0.5 * (edges[:-1] + edges[1:]), cdf, left=0.0, right=1.0)
    out[~nz] = 0.0
    return out.reshape(disp01.shape)


def _curve(fn: np.ndarray, expr: str, gamma: float) -> np.ndarray:
    """Apply a tone curve to an array already normalized to [0, 1]."""
    if expr == "linear" or expr == "percentile":
        return fn
    if expr == "sqrt":
        return np.sqrt(fn)
    if expr == "gamma":
        return np.power(fn, max(gamma, 1e-3))
    if expr == "log":
        return np.log1p(fn * 99.0) / np.log(100.0)          # ~2 decades, log(1+99x)/log(100)
    if expr == "asinh":
        return np.arcsinh(fn * 10.0) / np.arcsinh(10.0)
    if expr == "equalize":
        return _equalize(fn)
    return fn


def compress(frame, expr: str = "sqrt", vmax: float | None = None,
             clip_pct: float = 99.5, gamma: float = 0.5):
    """Tone-map a non-negative frame to a display array in ``[0, 1]``.

    Parameters
    ----------
    frame : (H, W) array       raw magnitude (event counts, rate, …)
    expr : str                 one of :data:`EXPRESSIONS`
    vmax : float | None        the white-point; ``None`` → compute it (``dynamic``). Pass a
                               frozen value (from a previous call's return) for ``static``.
    clip_pct, gamma : float    percentile for the auto white-point; gamma for the gamma curve.

    Returns ``(disp, vmax_used)`` — feed ``disp`` to a colormap with levels ``(0, 1)`` and
    keep ``vmax_used`` to freeze the scale for ``static`` mode.
    """
    f = np.asarray(frame, np.float64)
    if vmax is None:
        vmax = reference_white(f, clip_pct)
    fn = np.clip(f / vmax, 0.0, 1.0)
    return _curve(fn, expr, gamma).astype(np.float32), float(vmax)


def compress_signed(frame, expr: str = "linear", vmax: float | None = None,
                    clip_pct: float = 99.5, gamma: float = 0.5):
    """Tone-map a signed/diverging frame (e.g. ON−OFF polarity) to ``[-1, 1]``.

    The same expression is applied to the magnitude and the sign restored, so a diverging
    colormap stays centred at zero. Returns ``(disp, vmax_used)`` like :func:`compress`.
    """
    f = np.asarray(frame, np.float64)
    if vmax is None:
        a = np.abs(f)
        pos = a[a > 0]
        vmax = float(np.percentile(pos, clip_pct)) if pos.size else 1.0
        vmax = max(vmax, 1e-9)
    mag = np.clip(np.abs(f) / vmax, 0.0, 1.0)
    curved = _curve(mag, expr, gamma)
    return (np.sign(f) * curved).astype(np.float32), float(vmax)
