"""
synthetic.py — generate labeled event scenes with known flutter signatures.

Real captures rarely come with ground truth, which makes it hard to know whether a detector
*works* or merely *runs*. This module builds synthetic :class:`~gottlux.io.recording.Recording`
objects with planted, fully-known targets: a small blob that moves along a path and emits
events in **periodic bursts** at a chosen flutter frequency (a stand-in for a rotor's
blade-pass tone or a wingbeat), over a bed of unstructured background noise.

Use it to unit-test the pipeline, to sanity-check a detector's frequency readout against a
known input, and to teach the tuning workbench what a clean detection looks like.

>>> rec, truth = synthetic_scene(duration_s=2.0,
...     targets=[FlutterTarget(flutter_hz=200, x0=40, y0=160, x1=280, y1=160)])
>>> # rec is a normal Recording; truth lists each target's (t, x, y, flutter_hz)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gottlux.io.recording import Recording


@dataclass
class FlutterTarget:
    """A planted target that moves linearly and emits periodic event bursts."""
    flutter_hz: float = 200.0
    x0: float = 40.0
    y0: float = 160.0
    x1: float = 280.0
    y1: float = 160.0
    radius: float = 6.0            # blob radius (px) at the start of the clip
    radius1: float | None = None   # blob radius at the end (None → constant); set it < or >
                                   # radius to plant an approaching (growing) or receding target
    events_per_burst: int = 45     # events emitted at each flutter period
    duty: float = 0.35             # fraction of the period the burst is spread over
    harmonics: tuple = (1.0, 0.4)  # relative burst strength of fundamental + overtones
    jitter_px: float = 0.8         # per-event spatial jitter
    label: str = "target"


def _emit_target(tgt: FlutterTarget, duration_s: float, rng) -> tuple:
    """Produce (x, y, p, t_us, truth_rows) for one flutter target."""
    period = 1.0 / tgt.flutter_hz
    n_periods = int(duration_s / period)
    xs, ys, ps, ts = [], [], [], []
    truth = []
    for k in range(n_periods):
        t_center = (k + 0.5) * period
        frac = t_center / duration_s
        px = tgt.x0 + (tgt.x1 - tgt.x0) * frac
        py = tgt.y0 + (tgt.y1 - tgt.y0) * frac
        radius = tgt.radius if tgt.radius1 is None else tgt.radius + (tgt.radius1 - tgt.radius) * frac
        # number of events this burst, modulated by a small harmonic content and the disk area
        # (a closer/larger target both spans more pixels and emits more events ∝ radius²)
        area_gain = (radius / tgt.radius) ** 2 if tgt.radius > 0 else 1.0
        amp = sum(h * (0.5 + 0.5 * np.cos(2 * np.pi * (j + 1) * tgt.flutter_hz * t_center))
                  for j, h in enumerate(tgt.harmonics))
        nb = max(1, int(tgt.events_per_burst * area_gain * max(amp, 0.2) / sum(tgt.harmonics)))
        # burst spread over `duty` of the period
        bt = t_center + (rng.random(nb) - 0.5) * tgt.duty * period
        ang = rng.random(nb) * 2 * np.pi
        rad = radius * np.sqrt(rng.random(nb))
        bx = px + rad * np.cos(ang) + rng.normal(0, tgt.jitter_px, nb)
        by = py + rad * np.sin(ang) + rng.normal(0, tgt.jitter_px, nb)
        bp = rng.integers(0, 2, nb)
        xs.append(bx); ys.append(by); ps.append(bp); ts.append(bt)
        truth.append((t_center, px, py, tgt.flutter_hz))
    if not xs:
        return (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), truth)
    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(ps), np.concatenate(ts) * 1e6, truth)


def synthetic_scene(duration_s: float = 2.0, width: int = 320, height: int = 320,
                    targets: list | None = None, noise_rate_hz: float = 30_000.0,
                    static_clutter: int = 60, seed: int = 0):
    """Build a synthetic :class:`Recording` with planted flutter targets + background noise.

    Parameters
    ----------
    duration_s : float
    width, height : int
        Sensor geometry.
    targets : list[FlutterTarget] | None
        Planted targets; defaults to a single 200 Hz target crossing the centre.
    noise_rate_hz : float
        Mean rate of unstructured (Poisson, uniform-in-space) background events.
    static_clutter : int
        Number of persistent "hot" pixels that fire steadily (a background to suppress).
    seed : int

    Returns
    -------
    (Recording, truth) where ``truth`` is a list of per-target dicts with the planted path
    and flutter frequency.
    """
    rng = np.random.default_rng(seed)
    if targets is None:
        targets = [FlutterTarget()]
    Xs, Ys, Ps, Ts = [], [], [], []
    truth = []

    # background noise (uniform space, Poisson time)
    n_noise = int(noise_rate_hz * duration_s)
    if n_noise > 0:
        Xs.append(rng.integers(0, width, n_noise).astype(float))
        Ys.append(rng.integers(0, height, n_noise).astype(float))
        Ps.append(rng.integers(0, 2, n_noise))
        Ts.append(rng.random(n_noise) * duration_s * 1e6)

    # static clutter: a few pixels firing at a steady ~500 Hz (non-target periodicity to
    # exercise background suppression, deliberately outside typical target bands' interest)
    for _ in range(static_clutter):
        cx = rng.integers(0, width); cy = rng.integers(0, height)
        nt = int(500 * duration_s)
        tt = np.linspace(0, duration_s, nt, endpoint=False) + rng.random(nt) * 1e-4
        Xs.append(np.full(nt, cx, float)); Ys.append(np.full(nt, cy, float))
        Ps.append(rng.integers(0, 2, nt)); Ts.append(tt * 1e6)

    # planted flutter targets
    for tgt in targets:
        tx, ty, tp, tt, rows = _emit_target(tgt, duration_s, rng)
        Xs.append(tx); Ys.append(ty); Ps.append(tp); Ts.append(tt)
        truth.append(dict(label=tgt.label, flutter_hz=tgt.flutter_hz,
                          path=np.array(rows)))

    x = np.clip(np.concatenate(Xs), 0, width - 1).astype(np.uint16)
    y = np.clip(np.concatenate(Ys), 0, height - 1).astype(np.uint16)
    p = np.concatenate(Ps).astype(np.uint8)
    t = np.concatenate(Ts).astype(np.int64)
    rec = Recording.from_events(x, y, p, t, width=width, height=height,
                                fmt="synthetic", name="synthetic_scene")
    return rec, truth
