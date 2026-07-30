"""
tracking.py — lightweight multi-target association for the detector framework.

Per accumulation step a detector produces a handful of *verified* candidate detections
(blobs that passed the flutter-FFT gate). This module links them across time into coherent
tracks with a greedy nearest-neighbour associator: each existing track predicts its next
position from a smoothed velocity, claims the closest unused candidate within a gate, and
*coasts* (predicts only) for a few missed steps before it is dropped. New, unclaimed
candidates spawn new tracks.

It is intentionally simple, fast, and dependency-free — enough to follow a few targets
through a scene. The accumulated per-track history (time, centroid, bbox, and the flutter
frequency/SNR/harmonic measured at each step) is exactly what
:class:`~gottlux.detectors.base.Target` needs.
"""
from __future__ import annotations

import numpy as np


class _Tracklet:
    __slots__ = ("id", "cx", "cy", "vx", "vy", "bbox", "missed", "hist")

    def __init__(self, tid, cx, cy, bbox):
        self.id = tid
        self.cx = cx
        self.cy = cy
        self.vx = 0.0
        self.vy = 0.0
        self.bbox = np.asarray(bbox, float)
        self.missed = 0
        self.hist = []          # list of dicts: t, cx, cy, bbox, freq, snr, harmonic


class MultiTracker:
    """Greedy nearest-neighbour multi-target tracker with velocity prediction + coasting."""

    def __init__(self, max_match_dist: float = 60.0, max_missed: int = 8,
                 max_tracks: int = 6, smooth: float = 0.4):
        self.max_match_dist = float(max_match_dist)
        self.max_missed = int(max_missed)
        self.max_tracks = int(max_tracks)
        self.smooth = float(smooth)
        self._active: list[_Tracklet] = []
        self._records: dict[int, list] = {}
        self._next_id = 0

    def update(self, t: float, candidates: list) -> None:
        """Associate this step's *candidates* (dicts with cx, cy, bbox, freq, snr, harmonic)
        with active tracks, spawning/coasting/retiring as needed."""
        for tr in self._active:                          # predict
            tr.cx += tr.vx
            tr.cy += tr.vy
        used = [False] * len(candidates)
        for tr in self._active:                          # greedy nearest match
            best, bd = -1, self.max_match_dist
            for k, c in enumerate(candidates):
                if used[k]:
                    continue
                d = float(np.hypot(c["cx"] - tr.cx, c["cy"] - tr.cy))
                if d < bd:
                    bd, best = d, k
            if best >= 0:
                c = candidates[best]
                used[best] = True
                s = self.smooth
                ncx = (1 - s) * c["cx"] + s * tr.cx
                ncy = (1 - s) * c["cy"] + s * tr.cy
                tr.vx = ncx - (tr.cx - tr.vx)
                tr.vy = ncy - (tr.cy - tr.vy)
                tr.cx, tr.cy = ncx, ncy
                tr.bbox = s * tr.bbox + (1 - s) * np.asarray(c["bbox"], float)
                tr.missed = 0
                self._records[tr.id].append(dict(
                    t=t, cx=ncx, cy=ncy, bbox=tr.bbox.copy(),
                    freq=c.get("freq", np.nan), snr=c.get("snr", 0.0),
                    harmonic=c.get("harmonic", 0.0)))
            else:
                tr.missed += 1
        self._active = [tr for tr in self._active if tr.missed <= self.max_missed]
        for k, c in enumerate(candidates):               # spawn from unclaimed
            if not used[k] and len(self._active) < self.max_tracks:
                tr = _Tracklet(self._next_id, c["cx"], c["cy"], c["bbox"])
                self._active.append(tr)
                self._records[self._next_id] = [dict(
                    t=t, cx=c["cx"], cy=c["cy"], bbox=np.asarray(c["bbox"], float),
                    freq=c.get("freq", np.nan), snr=c.get("snr", 0.0),
                    harmonic=c.get("harmonic", 0.0))]
                self._next_id += 1

    def tracks(self, min_len: int = 2) -> dict:
        """Return ``{id: list_of_step_records}`` for tracks with at least *min_len* steps."""
        return {tid: rec for tid, rec in self._records.items() if len(rec) >= min_len}
