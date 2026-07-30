"""
gottlux.core — the compute engine (pure NumPy / SciPy / optional Numba).

Modules
-------
* :mod:`~gottlux.core.accumulate`  events → 2-D frames (count / polarity / time-surface / …)
* :mod:`~gottlux.core.frequency`   the temporal-frequency engine: spectra, Lomb–Scargle,
  spectrograms, and the per-cell **flicker map** (heart of flutter detection)
* :mod:`~gottlux.core.filters`     denoise pre-filters (hot-pixel, refractory, rotation-phase)
* :mod:`~gottlux.core.background`  static-clutter suppression (rotation frozen / staring)
* :mod:`~gottlux.core.geometry`    pinhole projection: pixel → bearing / elevation / range
* :mod:`~gottlux.core.detect`      spatial blob isolation → :class:`~gottlux.core.detect.Detection`
* :mod:`~gottlux.core.metrics`     coverage / localization figures of merit

These are deliberately free of any GUI/plotting import, so they are cheap and safe to use
from scripts, the headless pipeline, and background threads alike.
"""
from __future__ import annotations
