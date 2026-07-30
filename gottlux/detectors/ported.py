"""
ported.py — the unified extension framework: **trackers as gottlux Detectors**.

GottLUX has two natural pluggable-algorithm shapes: the staring-side
:class:`~gottlux.detectors.base.Detector` (consumes a Recording, self-describing
:class:`Param`s, auto-built tuning UI) and the rotation-side ``Tracker`` (consumes a
per-detection *trajectory* from a shared ``isolate → build_trajectory`` front-end).

This module reconciles them into the one canonical registry by wrapping every tracker in a
:class:`TrajectoryLinkerDetector`: a Detector whose :meth:`run` executes the
shared rotation front-end (:func:`gottlux.rotation.build_context`) and hands the trajectory (and
raw events for the event-consuming trackers) to the tracker's ``track()``. Each tracker's plain
``params`` dict becomes a list of :class:`Param`s, so the ported trackers *gain* the auto-built
tuning panel for free. Track output is converted to :class:`Target`s — including the EBS
regime-split extras (``radial_velocity``, ``blade_hz``) so the staring report is expressible.
"""
from __future__ import annotations

import numpy as np

from gottlux.detectors.base import (Detector, DetectorResult, Param, Target,
                                    list_detectors, register)


# ------------------------------------------------------------------ params from a tracker
def _params_from_tracker(Tcls) -> list:
    """Turn a tracker's plain ``params`` dict into self-describing :class:`Param`s."""
    out = []
    for k, v in (getattr(Tcls, "params", {}) or {}).items():
        if isinstance(v, bool):
            out.append(Param(key=k, label=k, default=v, kind="bool", group="Tracker"))
        elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            hi = float(max(int(v) * 4, 10))
            out.append(Param(key=k, label=k, default=float(int(v)), lo=0.0, hi=hi,
                             kind="int", group="Tracker"))
        else:
            fv = float(v)
            span = max(abs(fv) * 4.0, 1.0)
            lo = 0.0 if fv >= 0 else -span
            out.append(Param(key=k, label=k, default=fv, lo=lo, hi=span,
                             kind="float", group="Tracker"))
    return out


def _window_traj(traj, t0, t1):
    """Restrict a trajectory dict to the time window [t0, t1] (keeps equal lengths)."""
    if not traj or (t0 is None and t1 is None):
        return traj
    t = np.asarray(traj.get("t", []), float)
    if not len(t):
        return traj
    m = np.ones(len(t), bool)
    if t0 is not None:
        m &= t >= t0
    if t1 is not None:
        m &= t <= t1
    return {k: (np.asarray(v)[m] if np.ndim(v) and len(np.asarray(v)) == len(t) else v)
            for k, v in traj.items()}


def _window_ev(ev, t0, t1):
    """Restrict an ev dict to [t0, t1] seconds (for event-consuming trackers)."""
    if ev is None or (t0 is None and t1 is None):
        return ev
    t = np.asarray(ev["t"]) / 1e6
    m = np.ones(len(t), bool)
    if t0 is not None:
        m &= t >= t0
    if t1 is not None:
        m &= t <= t1
    out = dict(ev)
    for k in ("x", "y", "p", "t"):
        out[k] = np.asarray(ev[k])[m]
    out["n"] = int(m.sum())
    return out


# ------------------------------------------------------------------ the adapter
class TrajectoryLinkerDetector(Detector):
    """Run an EBS ``Tracker`` as a Detector over the shared rotation front-end."""

    TRACKER = None      # the registered EBS tracker name (set on each subclass)

    def run(self, rec, cfg=None, t0=None, t1=None, progress=None) -> DetectorResult:
        from gottlux.config import Config
        from gottlux.rotation import build_context, track_analysis, trackers
        cfg = cfg if cfg is not None else Config()
        ctx = build_context(rec, cfg)
        if progress:
            try:
                progress(0.6)
            except Exception:
                pass
        Tcls = trackers.get(self.TRACKER)
        if Tcls is None:
            return DetectorResult(targets=[], detector=self.name, params=dict(self.params),
                                  regime=self.regime, diagnostics={"error": "tracker not found"})
        T = Tcls()
        T.params = {**getattr(T, "params", {}), **self.params}    # apply tuned params
        traj = _window_traj(ctx.traj or {}, t0, t1)
        ev = _window_ev(ctx.ev, t0, t1) if getattr(Tcls, "uses_events", False) else ctx.ev
        out = T.track(traj, ctx.cfg, ctx.tel, ev=ev)
        targets = [self._to_target(tr, i, ctx, track_analysis)
                   for i, tr in enumerate(out.get("tracks", []))]
        if progress:
            try:
                progress(1.0)
            except Exception:
                pass
        return DetectorResult(targets=targets, detector=self.name, params=dict(self.params),
                              regime=ctx.cfg.mode if ctx.cfg.mode in ("rotation", "staring") else self.regime,
                              signature=self.TRACKER,
                              diagnostics={"n_tracks": len(targets),
                                           "n_detections": 0 if ctx.dets is None else int(len(ctx.dets))})

    @staticmethod
    def _to_target(tr, i, ctx, ta) -> Target:
        t = np.asarray(tr.get("t", []), float)
        n = len(t)
        cx = np.asarray(tr.get("cx", np.full(n, np.nan)), float)
        cy = np.asarray(tr.get("cy", np.full(n, np.nan)), float)
        dx, dy = ta.track_bbox(tr, ctx.traj)
        if n:
            bbox = np.column_stack([cx - np.nan_to_num(dx) / 2, cy - np.nan_to_num(dy) / 2,
                                    cx + np.nan_to_num(dx) / 2, cy + np.nan_to_num(dy) / 2])
        else:
            bbox = np.zeros((0, 4))
        size = ta.apparent_size_px(dx, dy, ctx.cfg.mode)
        rel = ta.relative_distance(size)
        rv = ta.radial_velocity(t, rel) if n > 1 else None
        blade = None
        if ctx.cfg.mode != "rotation" and n:
            bf = ta.blade_fft(ctx.ev, tr)
            blade = np.full(n, bf.get("peak_hz", np.nan))
        az = tr.get("azimuth_deg"); el = tr.get("elev_deg"); rng = tr.get("range_m")
        freq = np.asarray(tr.get("freq_hz", np.full(n, np.nan)), float)
        return Target(
            id=int(tr.get("id", i)), t=t, cx=cx, cy=cy, bbox=bbox,
            freq_hz=freq, snr=np.full(n, np.nan), harmonic=np.full(n, np.nan),
            azimuth_deg=np.asarray(az, float) if az is not None else None,
            elev_deg=np.asarray(el, float) if el is not None else None,
            range_m=np.asarray(rng, float) if rng is not None else None,
            rel_distance=rel, radial_velocity=rv, blade_hz=blade)


def register_ported_trackers() -> list:
    """Register every EBS tracker as a Detector in the unified registry.

    Bare tracker names are used where free; a name already taken by a flutter detector
    (e.g. ``hummingbird``) is exposed as ``<name>_track`` so nothing is shadowed.
    Returns the list of detector names registered.
    """
    from gottlux.rotation import trackers
    existing = set(list_detectors())
    names = []
    for tname in trackers.available():
        Tcls = trackers.get(tname)
        dname = tname if tname not in existing else f"{tname}_track"
        cls = type(
            f"Linked_{tname}", (TrajectoryLinkerDetector,),
            {"name": dname,
             "description": f"EBS '{tname}' tracker, linked over the rotation front-end",
             "use_for": getattr(Tcls, "description", "") or f"{tname} tracking",
             "regime": getattr(Tcls, "regime", "both"),
             "TRACKER": tname,
             "PARAMS": _params_from_tracker(Tcls)})
        register(cls)
        names.append(dname)
        existing.add(dname)
    return names
