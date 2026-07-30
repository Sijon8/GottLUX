"""
tracks.py — journal figures of detector output over time.

A target's story is told by three time series. Which three depends on the regime, but the
flutter frequency is always one of them — it is the measurement that says *what* the target
is, not merely where:

* **frequency vs time**            — the flutter signature (rotor/wingbeat), always shown.
* **relative distance vs time**    — the size→range proxy (calibratable to metres).
* **bearing vs time** (rotation) / **elevation vs time** (staring) — where it is.
"""
from __future__ import annotations

import numpy as np

from gottlux.viz import theme


def track_timeseries_figure(result, rotating=False, title=None, max_show=8,
                            width=theme.COL_DOUBLE):
    """Three stacked time-series panels for a :class:`~gottlux.detectors.base.DetectorResult`.

    Only the *max_show* most-confident targets are drawn (a busy scene can yield hundreds of
    flutter candidates); the title notes how many were found in total."""
    import matplotlib.pyplot as plt
    theme.apply()
    all_targets = sorted(result.targets, key=lambda t: -t.confidence)
    targets = all_targets[:max_show]
    n_total = len(all_targets)
    fig, axes = plt.subplots(3, 1, figsize=(width, width * 0.62), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    if not targets:
        for ax in axes:
            ax.text(0.5, 0.5, "no targets", ha="center", va="center", transform=ax.transAxes)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(title or f"Detector '{result.detector}' — no targets")
        return fig

    for i, t in enumerate(targets):
        c = colors[i % 10]
        lbl = f"#{t.id} ({t.median_freq:.0f} Hz, c={t.confidence:.2f})"
        axes[0].plot(t.t, t.freq_hz, "-o", color=c, ms=3, lw=1.0, label=lbl)
        rel = t.rel_distance if t.rel_distance is not None else np.full_like(t.t, np.nan)
        axes[1].plot(t.t, rel, "-o", color=c, ms=3, lw=1.0)
        if rotating and t.azimuth_deg is not None:
            axes[2].plot(t.t, t.azimuth_deg, "-o", color=c, ms=3, lw=1.0)
        elif t.elev_deg is not None:
            axes[2].plot(t.t, t.elev_deg, "-o", color=c, ms=3, lw=1.0)

    fmin, fmax = (result.params.get("freq_lo", 0), result.params.get("freq_hi", 0))
    if fmax:
        axes[0].axhspan(fmin, fmax, color="#90caf9", alpha=0.12, lw=0)
    axes[0].set_ylabel("flutter freq (Hz)")
    axes[1].set_ylabel("rel. distance\n(1/size)")
    axes[2].set_ylabel("bearing (deg)" if rotating else "elevation (deg)")
    axes[2].set_xlabel("time (s)")
    axes[0].legend(loc="upper right", ncol=1, fontsize=7)
    suffix = f" (top {len(targets)} of {n_total})" if n_total > len(targets) else ""
    axes[0].set_title((title or f"Detector '{result.detector}'") +
                      f" — {result.n_targets} target(s)" + suffix)
    fig.align_ylabels(axes)
    fig.tight_layout()
    return fig


def confidence_bar_figure(result, max_show=20, width=theme.COL_SINGLE):
    """A horizontal bar chart ranking the most-confident targets (quick triage view)."""
    import matplotlib.pyplot as plt
    theme.apply()
    targets = sorted(result.targets, key=lambda t: -t.confidence)[:max_show][::-1]
    fig = theme.figure(width, max(1.4, 0.3 * len(targets) + 0.8))
    ax = fig.add_subplot(111)
    if not targets:
        ax.text(0.5, 0.5, "no targets", ha="center", va="center", transform=ax.transAxes)
        return fig
    y = np.arange(len(targets))
    conf = [t.confidence for t in targets]
    colors = ["#43a047" if c >= 0.5 else "#fb8c00" if c >= 0.3 else "#e53935" for c in conf]
    ax.barh(y, conf, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([f"#{t.id}  {t.median_freq:.0f} Hz" for t in targets])
    ax.set_xlim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_title("Target confidence")
    fig.tight_layout()
    return fig
