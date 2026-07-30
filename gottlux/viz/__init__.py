"""
gottlux.viz — journal-ready static figures (matplotlib).

Every builder returns a styled :class:`matplotlib.figure.Figure`; pair it with
:func:`gottlux.io.export.save_figure` to write a raster + vector master at journal DPI.

* :mod:`~gottlux.viz.theme`     publication style + custom colormaps + the flicker palette
* :mod:`~gottlux.viz.frames`    event frames, detection overlays
* :mod:`~gottlux.viz.spectral`  flicker map (showpiece), spectra, spectrograms
* :mod:`~gottlux.viz.tracks`    per-target time series, confidence ranking
* :mod:`~gottlux.viz.panorama`  de-rotated 360° panorama, polar radar

Matplotlib is imported lazily inside the builders, so importing this package is cheap and
never requires a display.
"""
from __future__ import annotations

from gottlux.viz import frames, panorama, spectral, theme, tracks  # noqa: F401

__all__ = ["theme", "frames", "spectral", "tracks", "panorama"]
