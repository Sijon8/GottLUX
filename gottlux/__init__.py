"""
GottLUX — a unified analysis & visualization instrument for event-based-sensor recordings
========================================================================================

GottLUX carries an event-camera recording all the way from raw encoded events to
calibrated, paper-ready measurements and figures, whether captured from a fixed
*staring* sensor or a *rotating* panoramic payload — with an emphasis on three
things the vendor/SDK tools do not do well:

  1. **Interact with the event stream itself** — a GPU-accelerated live viewer, a 3-D
     space-time (x, y, t) event-cloud explorer, and per-region spectral readouts.
  2. **Detect fluttering / flickering signatures** — a tunable, composable detector
     framework built around the temporal-frequency signature of rotors and wingbeats
     (drones, insects, birds), with instant visual feedback while you tune.
  3. **Reproducible, journal-ready output** — every run is archived with the exact code
     and inputs behind it, and every figure is publication-grade.

Design principles
-----------------
* **One data model.** Everything operates on a single :class:`~gottlux.io.recording.Recording`
  object: memmap-backed event arrays (``x, y, p, t``) plus metadata and optional rotation
  telemetry. Decode-once, then slice windows cheaply.
* **Bounded memory.** Multi-gigabyte files stream chunk-by-chunk straight to an on-disk
  memmap cache; a re-open is instant.
* **Vectorized + JIT.** Hot loops are NumPy-vectorized and, where it matters, Numba-JIT
  compiled — with a pure-NumPy fallback so the suite always runs.
* **Light import.** Importing :mod:`gottlux` pulls in only NumPy-level code. The GUI
  (PySide6 / pyqtgraph) and plotting (matplotlib) are imported lazily, only when used.

Quick start
-----------
>>> import gottlux as eb
>>> rec = eb.load("capture/cam0.raw")          # decode-once -> memmapped Recording
>>> rec.summary()
>>> frame = rec.accumulate(t0=1.0, dt=0.02)    # a (H, W) event-count frame
>>> from gottlux.core import frequency as fq
>>> fmap = fq.flicker_map(rec, fmin=80, fmax=800)   # per-pixel dominant flicker frequency

The interactive workbench:  ``gottlux-gui``  (or ``python -m gottlux``)
A headless run:             ``gottlux path/to/file.raw --detect drone``
"""
from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Simon Gott"
__all__ = ["__version__", "load", "Recording", "Config", "Telemetry"]

# Light-weight, eagerly-imported public surface (NumPy-level only — no Qt / matplotlib).
from gottlux.config import Config
from gottlux.io.recording import Recording, load
from gottlux.io.telemetry import Telemetry
