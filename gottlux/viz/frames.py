"""
frames.py — journal figures of the image plane: event frames and detection overlays.
"""
from __future__ import annotations

import numpy as np

from gottlux.viz import theme


def event_frame_figure(frame, mode="count", cmap=None, title=None, colorbar=True,
                       width=theme.COL_SINGLE):
    """Render an accumulated event frame (from :mod:`gottlux.core.accumulate`) as a figure.

    Picks a sensible colormap and normalization per *mode* (diverging for polarity, the
    gottlux event map otherwise) and adds a labeled colorbar.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    theme.apply()
    f = np.asarray(frame, float)
    H, W = f.shape
    fig = theme.figure(width, width * H / W + 0.4)
    ax = fig.add_subplot(111)
    if mode == "polarity":
        a = float(np.abs(f).max()) or 1.0
        im = ax.imshow(f, cmap="gottlux_polarity", norm=TwoSlopeNorm(0, -a, a),
                       origin="upper", interpolation="nearest")
        clabel = "ON − OFF events"
    else:
        cmap = cmap or ("gottlux_events" if mode in ("count", "on", "off") else "magma")
        hi = np.percentile(f[f > 0], 99.5) if np.any(f > 0) else 1.0
        im = ax.imshow(f, cmap=cmap, vmin=0, vmax=max(hi, 1e-9),
                       origin="upper", interpolation="nearest")
        clabel = {"time_surface": "recency (decayed)", "binary": "occupancy"}.get(
            mode, "events / pixel")
    ax.set_xlabel("sensor x (px)")
    ax.set_ylabel("sensor y (px)")
    ax.set_title(title or f"Event frame ({mode})")
    if colorbar:
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(clabel)
    fig.tight_layout()
    return fig


def event_rate_figure(centers_s, rate_hz, title=None, width=theme.COL_DOUBLE):
    """Plot event rate (events/s) versus time — the always-useful activity overview."""
    import matplotlib.pyplot as plt
    theme.apply()
    fig = theme.figure(width, width * 0.32)
    ax = fig.add_subplot(111)
    if len(centers_s):
        ax.fill_between(centers_s, 0, np.asarray(rate_hz) / 1e6, color="#1565c0", alpha=0.35)
        ax.plot(centers_s, np.asarray(rate_hz) / 1e6, color="#1565c0", lw=1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("event rate (Mev/s)")
    ax.set_title(title or "Event rate")
    ax.margins(x=0)
    fig.tight_layout()
    return fig


def detection_overlay_figure(frame, targets=None, detections=None, cmap="gray",
                             title=None, label_freq=True, width=theme.COL_SINGLE):
    """Overlay tracked targets (and/or raw detections) on an event frame.

    *targets* is a list of :class:`~gottlux.detectors.base.Target`; their boxes are drawn at
    the latest step with the median flutter frequency labeled. *detections* is a list of
    :class:`~gottlux.core.detect.Detection` drawn as thin boxes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    theme.apply()
    f = np.asarray(frame, float)
    H, W = f.shape
    fig = theme.figure(width, width * H / W + 0.4)
    ax = fig.add_subplot(111)
    hi = np.percentile(f[f > 0], 99.5) if np.any(f > 0) else 1.0
    ax.imshow(np.clip(f / max(hi, 1e-9), 0, 1), cmap=cmap, origin="upper",
              interpolation="nearest")

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    if detections:
        for d in detections:
            ax.add_patch(Rectangle((d.x0, d.y0), d.width, d.height, fill=False,
                                   ec="#26c6da", lw=0.8, alpha=0.7))
    if targets:
        for i, t in enumerate(targets):
            c = colors[i % 10]
            ax.plot(t.cx, t.cy, "-", color=c, lw=1.0, alpha=0.9)
            bb = t.bbox[-1]
            ax.add_patch(Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
                                   fill=False, ec=c, lw=1.4))
            lbl = f"#{t.id}"
            if label_freq and np.isfinite(t.median_freq):
                lbl += f"  {t.median_freq:.0f} Hz"
            ax.annotate(lbl, (bb[0], bb[1]), color=c, fontsize=7, ha="left", va="bottom",
                        xytext=(2, 2), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.5))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_xlabel("sensor x (px)"); ax.set_ylabel("sensor y (px)")
    ax.set_title(title or "Detections")
    fig.tight_layout()
    return fig
