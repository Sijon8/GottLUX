"""
spectral.py — journal figures for the temporal-frequency domain.

* :func:`flicker_map_figure` — the showpiece: a 2-D map of the sensor where **hue encodes the
  dominant flicker frequency** and **opacity encodes confidence (SNR)**, laid over a dim event
  image, with a frequency colorbar. You see *where* the scene flickers and *how fast* at a
  glance — a rotor disk glows in one colour while static structure stays dark.
* :func:`spectrum_figure` — a region's power spectrum with the search band shaded, the peak
  and its harmonics marked, and the SNR annotated.
* :func:`spectrogram_figure` — frequency-vs-time for a region, to show a signature evolving.
"""
from __future__ import annotations

import numpy as np

from gottlux.viz import theme


def flicker_map_figure(flicker_map, background=None, cmap="turbo", title=None,
                       width=theme.COL_SINGLE):
    """Render a :class:`~gottlux.core.frequency.FlickerMap` as a publication figure.

    Parameters
    ----------
    flicker_map : FlickerMap
    background : 2-D array | None
        Optional full-resolution event-count image shown dimmed underneath (for context).
    cmap : str
        Colormap mapping frequency → hue (``turbo`` is vivid and monotonic).
    """
    import matplotlib.pyplot as plt
    theme.apply()
    fm = flicker_map
    fmin, fmax = fm.band
    H, W = fm.shape
    fig = theme.figure(width, width * H / W + 0.6)
    ax = fig.add_axes([0.12, 0.10, 0.74, 0.82])

    if background is not None and np.size(background):
        bg = np.asarray(background, float)
        hi = np.percentile(bg[bg > 0], 99) if np.any(bg > 0) else 1.0
        ax.imshow(np.clip(bg / max(hi, 1e-9), 0, 1), cmap="gray", vmax=1.0,
                  alpha=0.55, origin="upper", extent=[0, W, H, 0], interpolation="nearest")

    rgba = theme.flicker_rgba(fm.dominant_freq, fm.snr, fmin, fmax, cmap=cmap)
    ax.imshow(rgba, origin="upper", extent=[0, W, H, 0], interpolation="nearest")
    ax.set_xlabel("sensor x (px)")
    ax.set_ylabel("sensor y (px)")
    ax.set_title(title or f"Flicker map  ({fmin:.0f}–{fmax:.0f} Hz)")

    # annotate the strongest flutter cell
    valid = np.isfinite(fm.dominant_freq)
    if valid.any():
        iy, ix = np.unravel_index(np.nanargmax(np.where(valid, fm.snr, -1)), fm.snr.shape)
        px, py = (ix + 0.5) * fm.cell, (iy + 0.5) * fm.cell
        ax.plot(px, py, "o", mfc="none", mec="white", ms=12, mew=1.5)
        ax.annotate(f"{fm.dominant_freq[iy, ix]:.0f} Hz\nSNR {fm.snr[iy, ix]:.0f}",
                    (px, py), color="white", fontsize=7, ha="left", va="bottom",
                    xytext=(6, 6), textcoords="offset points")

    cax = fig.add_axes([0.88, 0.10, 0.03, 0.82])
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=fmin, vmax=fmax))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("dominant flicker frequency (Hz)")
    return fig


def spectrum_figure(spectrum, title=None, logy=True, width=theme.COL_SINGLE):
    """Plot a :class:`~gottlux.core.frequency.Spectrum`: power vs frequency, band shaded,
    peak + harmonics marked, SNR annotated."""
    import matplotlib.pyplot as plt
    theme.apply()
    sp = spectrum
    fig = theme.figure(width, width * 0.7)
    ax = fig.add_subplot(111)
    if sp.freqs.size == 0:
        ax.text(0.5, 0.5, "no spectrum", ha="center", va="center", transform=ax.transAxes)
        return fig
    fmin, fmax = sp.band
    ax.axvspan(fmin, fmax, color="#90caf9", alpha=0.18, lw=0, label="search band")
    ax.plot(sp.freqs, sp.power, color="#1565c0", lw=1.1)
    if np.isfinite(sp.peak_freq):
        ax.plot(sp.peak_freq, sp.peak_power, "v", color="#d81b60", ms=7,
                label=f"peak {sp.peak_freq:.0f} Hz")
        for k in range(2, 5):                       # harmonic guides
            fk = k * sp.peak_freq
            if fk <= sp.freqs[-1]:
                ax.axvline(fk, color="#d81b60", ls=":", lw=0.7, alpha=0.6)
    ax.set_xlim(0, min(fmax * 1.4, sp.freqs[-1]))
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power")
    ax.set_title(title or "Region temporal spectrum")
    ax.text(0.97, 0.92, f"SNR {sp.snr:.1f}\nharm {sp.harmonic_score:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def spectrogram_figure(t_s, f_hz, S, title=None, cmap="magma", fmax=None,
                       width=theme.COL_SINGLE):
    """Plot a region spectrogram (frequency vs time) from :func:`gottlux.core.frequency.spectrogram`."""
    import matplotlib.pyplot as plt
    theme.apply()
    fig = theme.figure(width, width * 0.7)
    ax = fig.add_subplot(111)
    if np.size(S) == 0:
        ax.text(0.5, 0.5, "no spectrogram", ha="center", va="center", transform=ax.transAxes)
        return fig
    Sdb = 10 * np.log10(np.maximum(S, S[S > 0].min() if np.any(S > 0) else 1e-12))
    im = ax.pcolormesh(t_s, f_hz, Sdb, cmap=cmap, shading="auto")
    if fmax:
        ax.set_ylim(0, fmax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(title or "Region spectrogram")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("power (dB)")
    fig.tight_layout()
    return fig
