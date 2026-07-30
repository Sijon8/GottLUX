"""
recording.py — the one data model everything in gottlux operates on.

A :class:`Recording` wraps the decoded event arrays (memmap-backed: ``x, y, p, t``),
the sensor geometry and source metadata, and — when present — the rotation
:class:`~gottlux.io.telemetry.Telemetry`. It is deliberately thin and cheap: the heavy
arrays stay on disk as memmaps, and *windows* into the stream are returned as light
:class:`EventWindow` views (NumPy slices, no copy of the whole stream).

Construct one with the module-level :func:`load`, which accepts any of:

* a ``.raw`` file             → decode-once into the streaming memmap cache
* an HDF5 file (``.h5``/``.hdf5``) → the same decode-once cache, built by streamed
  dataset reads (see :mod:`gottlux.io.hdf5`)
* a capture **folder**        → find the camera's recording (+ sibling telemetry CSV)
* a decoded cache stem / ``.meta.json`` → open the bins directly (no ``.raw`` needed)

and from code, :meth:`Recording.from_events` for synthetic data and tests.

Everything downstream — accumulation, background masks, de-rotation, the frequency
engine, detectors, figures — takes a :class:`Recording` (or an :class:`EventWindow`),
so there is exactly one vocabulary for "a chunk of EBS data".
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from gottlux.io import cache as _cache
from gottlux.io.telemetry import Telemetry, find_telemetry_csv


# ====================================================================================
# A light-weight view into a time-window of a Recording (no full-stream copy)
# ====================================================================================
@dataclass
class EventWindow:
    """A contiguous slice of events: NumPy arrays already sliced to ``[i0:i1]``.

    Holds ``x, y, p`` (sensor coords / polarity) and ``t`` (µs, **still zero-based to the
    parent recording**, not to the window). ``width``/``height`` carry the sensor size so
    downstream code never needs the parent. Cheap to create; safe to hold many of.
    """
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    t: np.ndarray          # int64 µs, zero-based to the parent recording
    width: int
    height: int
    t0_us: int = 0         # absolute µs of the parent's t==0 (for provenance)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def t_s(self) -> np.ndarray:
        """Event times in seconds (float64), zero-based to the parent recording."""
        return self.t.astype(np.float64) / 1e6

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) / 1e6 if len(self.t) else 0.0

    def polarity_split(self):
        """Return boolean masks ``(on, off)`` for ON (p==1) and OFF (p==0) events."""
        on = self.p == 1
        return on, ~on


# ====================================================================================
# The Recording
# ====================================================================================
class Recording:
    """An immutable, memmap-backed event recording plus optional rotation telemetry.

    Attributes
    ----------
    x, y : uint16 memmap   sensor pixel coordinates (0..width-1 / 0..height-1)
    p    : uint8  memmap   polarity (1 = ON / brighter, 0 = OFF)
    t    : int64  memmap   event time in microseconds, zero-based, ascending
    width, height : int    sensor geometry
    fmt  : str             source encoding ('evt21' | 'evt2' | 'evt3' | 'synthetic')
    meta : dict            raw header key/values from the source file
    telemetry : Telemetry | None   rotation ground truth, if available
    source_path : str      where this came from
    """

    def __init__(self, data: dict, telemetry: Optional[Telemetry] = None,
                 name: str = ""):
        self.x = data["x"]
        self.y = data["y"]
        self.p = data["p"]
        self.t = data["t"]
        self.width = int(data["width"])
        self.height = int(data["height"])
        self.t0_us = int(data.get("t0_us", 0))
        self.n_on_cached = int(data.get("n_on", -1))
        self.fmt = data.get("fmt", "unknown")
        self.meta = data.get("meta", {}) or {}
        self.source_path = data.get("source_path", "")
        self.telemetry = telemetry
        self.name = name or os.path.splitext(os.path.basename(self.source_path or "recording"))[0]

    @classmethod
    def from_events(cls, x, y, p, t_us, width=None, height=None, fmt="synthetic",
                    name="synthetic", telemetry=None, meta=None) -> "Recording":
        """Build a Recording directly from in-memory event arrays (synthetic data / tests).

        Events are time-sorted and zero-based automatically. ``width``/``height`` default
        to one past the maximum observed coordinate.
        """
        x = np.asarray(x); y = np.asarray(y)
        p = np.asarray(p).astype(np.uint8); t = np.asarray(t_us).astype(np.int64)
        if len(t):
            order = np.argsort(t, kind="stable")
            x, y, p, t = x[order], y[order], p[order], t[order]
            t0 = int(t[0]); t = t - t0
        else:
            t0 = 0
        w = int(width if width is not None else (int(x.max()) + 1 if len(x) else 320))
        h = int(height if height is not None else (int(y.max()) + 1 if len(y) else 320))
        data = dict(x=x.astype(np.uint16), y=y.astype(np.uint16), p=p, t=t,
                    width=w, height=h, t0_us=t0, n_on=int((p == 1).sum()),
                    fmt=fmt, meta=meta or {}, source_path="")
        return cls(data, telemetry=telemetry, name=name)

    # --------------------------------------------------------------- basic facts
    def __len__(self) -> int:
        return int(self.x.shape[0])

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) / 1e6 if self.n else 0.0

    @property
    def t_start_s(self) -> float:
        return float(self.t[0]) / 1e6 if self.n else 0.0

    @property
    def t_stop_s(self) -> float:
        return float(self.t[-1]) / 1e6 if self.n else 0.0

    @property
    def n_on(self) -> int:
        if self.n_on_cached >= 0:
            return self.n_on_cached
        return int((np.asarray(self.p) == 1).sum())

    @property
    def n_off(self) -> int:
        return self.n - self.n_on

    @property
    def mean_event_rate(self) -> float:
        """Mean events per second over the recording span (events/s)."""
        d = self.duration_s
        return self.n / d if d > 0 else 0.0

    @property
    def is_rotating(self) -> bool:
        """True iff rotation telemetry is attached (the recording can be de-rotated)."""
        return self.telemetry is not None

    @property
    def sensor_shape(self):
        """``(height, width)`` — the shape of an accumulated frame."""
        return (self.height, self.width)

    # --------------------------------------------------------------- windowing
    def index_at(self, t_s: float, side: str = "left") -> int:
        """Index of the first event at/after *t_s* seconds (binary search on µs)."""
        return int(np.searchsorted(self.t, int(round(t_s * 1e6)), side=side))

    def window(self, t0: Optional[float] = None, t1: Optional[float] = None,
               roi=None) -> EventWindow:
        """Return an :class:`EventWindow` for events in ``[t0, t1)`` seconds (and ROI).

        ``t0``/``t1`` default to the full span. ``roi`` is ``(x0, y0, x1, y1)`` in pixels
        (inclusive-exclusive) or ``None``. The window materializes only the sliced arrays.
        """
        i0 = 0 if t0 is None else self.index_at(t0, "left")
        i1 = self.n if t1 is None else self.index_at(t1, "left")
        xs = np.asarray(self.x[i0:i1])
        ys = np.asarray(self.y[i0:i1])
        ps = np.asarray(self.p[i0:i1])
        ts = np.asarray(self.t[i0:i1])
        if roi is not None:
            x0, y0, x1, y1 = roi
            m = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            xs, ys, ps, ts = xs[m], ys[m], ps[m], ts[m]
        return EventWindow(xs, ys, ps, ts, self.width, self.height, self.t0_us)

    def all(self) -> EventWindow:
        """The whole recording as a single (memory-light) :class:`EventWindow`."""
        return EventWindow(np.asarray(self.x), np.asarray(self.y), np.asarray(self.p),
                           np.asarray(self.t), self.width, self.height, self.t0_us)

    # --------------------------------------------------------------- quick accumulation
    def accumulate(self, t0: float, dt: float, mode: str = "count",
                   roi=None) -> np.ndarray:
        """Convenience: accumulate events in ``[t0, t0+dt)`` into a ``(H, W)`` frame.

        Thin wrapper over :func:`gottlux.core.accumulate.accumulate_frame` (imported
        lazily to keep this module Qt/Numba-free at import time).
        """
        from gottlux.core.accumulate import accumulate_frame
        win = self.window(t0, t0 + dt, roi=roi)
        return accumulate_frame(win, mode=mode)

    def event_rate(self, bin_s: float = 0.01):
        """Return ``(centers_s, rate_hz)`` — the event rate vs time at *bin_s* resolution."""
        if not self.n:
            return np.zeros(0), np.zeros(0)
        edges = np.arange(0.0, self.duration_s + bin_s, bin_s)
        counts = np.histogram(self.t.astype(np.float64) / 1e6, bins=edges)[0]
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, counts / bin_s

    # --------------------------------------------------------------- telemetry
    def attach_telemetry(self, tel: Telemetry, refine: bool = True) -> "Recording":
        """Attach rotation telemetry; optionally refine its time offset to the events."""
        if refine and self.n:
            try:
                tel.refine_offset_to_events(self.t.astype(np.float64) / 1e6)
            except Exception:
                pass
        self.telemetry = tel
        return self

    # --------------------------------------------------------------- reporting
    def summary(self, printit: bool = True) -> str:
        """A compact human-readable description of the recording."""
        lines = [
            f"Recording: {self.name}",
            f"  source     : {self.source_path or '(in-memory)'}",
            f"  encoding   : {self.fmt}",
            f"  sensor     : {self.width} x {self.height} px",
            f"  events     : {self.n:,}  (ON {self.n_on:,} / OFF {self.n_off:,})",
            f"  duration   : {self.duration_s:.4f} s",
            f"  mean rate  : {self.mean_event_rate/1e6:.3f} Mev/s",
            f"  mode       : {'ROTATION (telemetry attached)' if self.is_rotating else 'STARING'}",
        ]
        if self.is_rotating:
            tel = self.telemetry
            lines += [
                f"  revolutions: {tel.n_revolutions}",
                f"  rot period : {tel.T_rot:.4f} s  ({tel.omega_deg_s:.1f} deg/s)",
            ]
        s = "\n".join(lines)
        if printit:
            print(s)
        return s

    def __repr__(self) -> str:
        return (f"<Recording {self.name!r} {self.width}x{self.height} "
                f"{self.n:,} events {self.duration_s:.3f}s "
                f"{'rotating' if self.is_rotating else 'staring'}>")


# ====================================================================================
# Loading
# ====================================================================================
def _find_raw_in_folder(folder: str, camera: str = "cam0") -> Optional[str]:
    """Pick an event recording inside *folder* — ``.raw`` first, then ``.h5``/``.hdf5`` —
    preferring one whose name matches *camera*."""
    cands = sorted(glob.glob(os.path.join(folder, "*.raw")))
    for pat in ("*.h5", "*.hdf5"):
        cands += sorted(glob.glob(os.path.join(folder, pat)))
    if not cands:
        return None
    for r in cands:
        if camera.lower() in os.path.basename(r).lower():
            return r
    return cands[0]


def _find_cache_stem_in_folder(folder: str, camera: str = "cam0") -> Optional[str]:
    """Find a decoded cache stem (``*.meta.json`` → stem) inside *folder*."""
    metas = sorted(glob.glob(os.path.join(folder, "*.meta.json")))
    if not metas:
        return None
    for m in metas:
        if camera.lower() in os.path.basename(m).lower():
            return m[: -len(".meta.json")]
    return metas[0][: -len(".meta.json")]


def load(path: str, camera: str = "cam0", mode: str = "auto",
         force_decode: bool = False, attach_telemetry: bool = True,
         progress=None) -> Recording:
    """Load a :class:`Recording` from a ``.raw``/``.h5`` file, a capture folder, or a cache stem.

    Parameters
    ----------
    path : str
        One of: a ``.raw`` file; an HDF5 event file (``.h5``/``.hdf5`` — Metavision
        ``CD/events`` or plain ``x/y/p/t`` layouts, see :mod:`gottlux.io.hdf5`); a folder
        containing either (and maybe a telemetry CSV); a decoded cache ``.meta.json`` or
        its stem; or a ``_gottlux_cache`` directory.
    camera : str
        When *path* is a folder with multiple cameras, prefer the one matching this id.
    mode : str
        ``"rotation"`` forces telemetry use, ``"staring"`` ignores telemetry, ``"auto"``
        attaches telemetry iff a CSV is found.
    force_decode : bool
        Re-decode even if a valid cache exists.
    progress : callable | None
        Optional ``progress(fraction)`` callback for the decode (drives the GUI bar).
    """
    path = os.path.abspath(path)
    data = None
    raw_for_telemetry = path

    if os.path.isdir(path):
        # A directory: prefer a real .raw (decode-once); else open bare decoded bins.
        if os.path.basename(path).startswith("_") and glob.glob(os.path.join(path, "*.meta.json")):
            stem = _find_cache_stem_in_folder(path, camera)
            data = _cache.open_cached(stem)
            raw_for_telemetry = stem
        else:
            raw = _find_raw_in_folder(path, camera)
            if raw is not None:
                data = _cache.load(raw, force=force_decode, progress=progress)
                raw_for_telemetry = raw
            else:
                stem = _find_cache_stem_in_folder(path, camera)
                if stem is not None:
                    data = _cache.open_cached(stem)
                    raw_for_telemetry = stem
    elif path.endswith(".meta.json"):
        data = _cache.open_cached(path[: -len(".meta.json")])
        raw_for_telemetry = path[: -len(".meta.json")]
    elif path.endswith(".raw") or path.lower().endswith((".h5", ".hdf5")):
        data = _cache.load(path, force=force_decode, progress=progress)
    else:
        # Bare stem? (path + ".meta.json" exists)
        if os.path.exists(path + ".meta.json"):
            data = _cache.open_cached(path)
        elif os.path.exists(path):
            data = _cache.load(path, force=force_decode, progress=progress)

    if data is None:
        raise FileNotFoundError(
            f"Could not find a decodable .raw, cache, or .meta.json at: {path!r}")

    rec = Recording(data)

    # Attach rotation telemetry if requested/available and not explicitly staring.
    if attach_telemetry and mode != "staring":
        csv = find_telemetry_csv(raw_for_telemetry)
        if csv is not None:
            try:
                rec.attach_telemetry(Telemetry(csv))
            except Exception as e:  # malformed telemetry must never block loading the events
                print(f"[gottlux] telemetry CSV present but unreadable ({e}); staying staring.")
    return rec
