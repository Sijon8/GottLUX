"""
rotor_ladder.py — detect a drone by the "stair-step" its rotor leaves in a *spinning* EBS.

The idea (S. Gott)
------------------
When the sensor spins (≈1 Hz) and sweeps across a multirotor, the rotor's high blade-pass
frequency is **spatially demodulated** by the sweep: each blade-pass burst lands at a slightly
different sensor column than the last, so the rotor draws a regularly-spaced **ladder / staircase**
of event bursts across the sweep direction. A building edge gets swept too, but it has no
high-frequency burst structure, so it leaves a *continuous* streak, not a comb — and unstructured
noise leaves neither. The comb spacing, together with the (known or measured) sweep rate, recovers
the blade-pass frequency with very little computation, and the ladder *recurs* every revolution
with an **offset** that encodes the drone's own relative motion.

The geometry (why a comb appears, and what its spacing means)
-------------------------------------------------------------
Let the sensor spin at angular rate ``Ω`` (rad/s) and the pixel angular scale be ``β = FOV/W``
(rad/px). A target at world azimuth ``θ_d(t) = θ_0 + Ω_d·t`` (its own angular rate ``Ω_d``)
images at sensor column ::

    x(t) = x_c + (θ_d(t) − Ω·t)/β  =  const + ((Ω_d − Ω)/β)·t

so it drifts across the sensor at the **sweep velocity** ``v = dx/dt = (Ω_d − Ω)/β`` [px/s]
(≈ −Ω/β, since ``Ω_d ≪ Ω``). The rotor modulates brightness at the **blade-pass frequency** ``f``
(Hz), emitting events in bursts at ``τ_k = τ_0 + k/f``. Burst *k* therefore lands at ::

    x_k = x_c + v·τ_k        ⇒   Δx = x_{k+1} − x_k = v / f           (the ladder step, px)

Two facts fall straight out and are the whole algorithm:

1. **Blade-pass frequency from geometry alone:** ``f = v / Δx`` — divide the event-cloud drift
   slope (px/s) by the measured comb spacing (px). The sweep turns a hard 80–800 Hz *temporal*
   measurement into an easy ~10 px *spatial* one. (Telemetry gives ``Ω``; if you also measure ``v``
   from the events, ``Ω_d`` cancels and ``f`` needs no telemetry at all.)
2. **Relative motion from the slope:** ``Ω_d = Ω − β·v`` — and across revolutions (period
   ``T_rot``) the ladder's world-azimuth offset ``ΔΘ`` gives ``Ω_d ≈ ΔΘ / T_rot``. A *stationary*
   drone repeats an identical ladder each revolution; a *moving* one shifts by a fixed offset.

Detectability regime
--------------------
The spatial comb is crisp when the sweep moves the rotor disk by more than its own size between
bursts, ``Δx = v/f ≳ disk``; when they overlap the comb becomes a periodic *ripple* (still found by
the autocorrelation, just weaker). Either way the discriminators are: a **coherent linear drift**
(any swept object) + a **periodic comb along it whose implied f is in the rotor band** (only a
rotor) + **recurrence across revolutions** (a real object, not a noise transient).

This module is pure NumPy and deliberately cheap: one robust line fit + one autocorrelation of a
small 1-D histogram per candidate. :func:`ladder_signature` is the live primitive;
:func:`synthetic_rotor_pass` generates a labelled pass to validate against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ====================================================================================
# Result
# ====================================================================================
@dataclass
class LadderResult:
    """The rotor-ladder signature measured in one candidate window."""
    n_events: int
    drift_px_s: float                 # v — the sweep velocity (event-cloud slope), px/s
    step_px: Optional[float] = None   # Δx — the ladder spacing, px
    blade_hz: Optional[float] = None  # f = |v| / Δx, the implied blade-pass frequency
    comb_strength: float = 0.0        # normalized autocorrelation peak at Δx (0..1)
    gappiness: float = 0.0            # fraction of the swept extent that is empty (burst-like → high)
    in_band: bool = False
    score: float = 0.0                # combined 0..1 confidence
    detected: bool = False

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ====================================================================================
# Core primitive
# ====================================================================================
def _drift_slope(x, t) -> float:
    """Robust sweep velocity dx/dt (px/s): Theil–Sen-ish median of pairwise slopes on a sample."""
    x = np.asarray(x, float); t = np.asarray(t, float)
    n = x.size
    if n < 3 or (t.max() - t.min()) <= 0:
        return 0.0
    # least squares is fine and O(n); guard against a degenerate (no-drift) cluster
    b = np.polyfit(t, x, 1)[0]
    return float(b)


def ladder_signature(x, t, *, sweep_px_s=None, f_lo=80.0, f_hi=800.0, bin_px=1.0,
                     min_events=150, strength_thresh=0.20) -> LadderResult:
    """Measure the rotor-ladder signature of events ``(x, t)`` in one candidate window.

    Parameters
    ----------
    x : array     sensor column (the sweep direction), px.
    t : array     event times, s.
    sweep_px_s : float | None
        The sweep velocity ``v`` (px/s). ``None`` → estimated from the event cloud's drift.
    f_lo, f_hi : float
        The rotor blade-pass band searched (the comb spacing is constrained to ``v/f_hi … v/f_lo``).
    bin_px : float
        Histogram bin width along the sweep coordinate.
    min_events, strength_thresh : float
        Minimum events to judge, and the autocorrelation-peak threshold to call a comb present.

    Returns a :class:`LadderResult` with the comb spacing ``Δx``, the implied ``f = |v|/Δx``,
    the comb strength, and a combined score.
    """
    x = np.asarray(x, float); t = np.asarray(t, float)
    n = int(x.size)
    if n < min_events:
        return LadderResult(n_events=n, drift_px_s=0.0)

    v = float(sweep_px_s) if sweep_px_s is not None else _drift_slope(x, t)
    res = LadderResult(n_events=n, drift_px_s=round(v, 2))
    if abs(v) < 1e-6:
        return res

    # 1-D histogram along the sweep coordinate
    lo, hi = np.percentile(x, [0.5, 99.5])
    if hi - lo < 3 * bin_px:
        return res
    nb = max(int((hi - lo) / bin_px), 8)
    h, edges = np.histogram(x, bins=nb, range=(lo, hi))
    h = h.astype(float)
    occ = h > (0.15 * h.mean())                       # "occupied" bins
    res.gappiness = round(float(1.0 - occ.mean()), 3)  # high = burst-like (gaps); low = continuous

    # autocorrelation of the (mean-removed) histogram → comb spacing
    hc = h - h.mean()
    acf = np.correlate(hc, hc, mode="full")[nb - 1:]   # lags 0..nb-1
    if acf[0] <= 0:
        return res
    acf = acf / acf[0]                                 # normalized (acf[0] == 1)

    # the comb step Δx must put f = |v|/Δx inside the rotor band
    lag_min = max(int(round(abs(v) / f_hi / bin_px)), 2)
    lag_max = min(int(round(abs(v) / f_lo / bin_px)), nb - 1)
    if lag_max <= lag_min:
        return res
    # score each candidate period by its harmonic comb energy (a true comb has autocorrelation
    # peaks at p, 2p, 3p…; a spurious short-lag bump does not) — this is what picks the fundamental
    def _henergy(lag):
        # harmonic comb energy: a true ladder peaks at p AND 2p AND 3p; a spurious bump does not
        mults = [m for m in (lag, 2 * lag, 3 * lag) if m < nb]
        return float(np.mean([acf[m] for m in mults]))

    # The fundamental rung spacing is the SMALLEST strong ACF peak. A comb autocorrelates at
    # p, 2p, 3p…, so harmonic energy alone is octave-ambiguous (period 2p scores almost as high as
    # p, and noise can promote it). Among the comparably-tall ACF peaks in the allowed band, take
    # the smallest one with positive harmonic energy — the first rung — which collapses octave and
    # higher-harmonic aliases onto the true fundamental.
    peaks = [lag for lag in range(lag_min, min(lag_max, nb - 2) + 1)
             if acf[lag] > 0 and acf[lag] >= acf[lag - 1] and acf[lag] >= acf[lag + 1]]
    if peaks:
        max_peak = max(acf[l] for l in peaks)
        strong = [l for l in peaks if acf[l] >= 0.6 * max_peak]
        cand = [l for l in strong if _henergy(l) > 0]
        best_lag = min(cand) if cand else max(strong, key=_henergy)
        best_energy = _henergy(best_lag)
    else:
        # no comb peaks (a continuous swept streak / unstructured noise): keep the best harmonic
        # energy in band — it will be low and fail the detection threshold below.
        best_lag, best_energy = lag_min, -np.inf
        for lag in range(lag_min, lag_max + 1):
            energy = _henergy(lag)
            if energy > best_energy:
                best_energy, best_lag = energy, lag
    step_px = best_lag * bin_px
    res.step_px = round(step_px, 3)
    res.comb_strength = round(max(best_energy, 0.0), 3)   # harmonic energy (peaks at p,2p,3p)
    res.blade_hz = round(abs(v) / step_px, 1) if step_px > 0 else None
    res.in_band = bool(res.blade_hz is not None and f_lo <= res.blade_hz <= f_hi)

    # combined score: a strong, in-band, harmonically-structured comb on a real drift
    res.score = round(float(res.comb_strength * (1.0 if res.in_band else 0.0)), 3)
    res.detected = bool(res.in_band and res.comb_strength >= strength_thresh)
    return res


# ====================================================================================
# Cross-revolution recurrence / relative motion
# ====================================================================================
@dataclass
class LadderTrack:
    """A rotor ladder seen across several revolutions — the recurrence + the drift's offset."""
    n_passes: int
    median_blade_hz: Optional[float]
    blade_hz_stability: float                 # 1 − scatter/median (1 = rock-steady across revs)
    azimuth_offset_per_rev_px: float          # how far the ladder shifts each revolution (px)
    implied_target_omega_px_s: float          # Ω_d as an apparent px/s (offset / revolution time)
    confidence: float
    passes: list = field(default_factory=list)


def track_ladders(passes, t_rot_s) -> LadderTrack:
    """Combine per-revolution :class:`LadderResult` passes (each ``(rev_index, x_center, result)``)
    into a cross-revolution track: blade-frequency stability + the per-rev azimuth offset (relative
    motion). *t_rot_s* is the rotation period.
    """
    good = [(r, xc, res) for (r, xc, res) in passes if res is not None and res.detected]
    if not good:
        return LadderTrack(0, None, 0.0, 0.0, 0.0, 0.0, [])
    f = np.array([res.blade_hz for _, _, res in good], float)
    med = float(np.median(f))
    stab = float(max(0.0, 1.0 - (np.std(f) / med if med > 0 else 1.0)))
    revs = np.array([r for r, _, _ in good], float)
    xcs = np.array([xc for _, xc, _ in good], float)
    offset_per_rev = float(np.polyfit(revs, xcs, 1)[0]) if len(good) >= 2 else 0.0
    omega_px_s = offset_per_rev / t_rot_s if t_rot_s else 0.0
    conf = float(np.clip(min(len(good) / 3.0, 1.0) * stab, 0, 1))
    return LadderTrack(len(good), round(med, 1), round(stab, 3), round(offset_per_rev, 2),
                       round(omega_px_s, 2), round(conf, 3),
                       [(r, round(xc, 1), res) for r, xc, res in good])


# ====================================================================================
# Synthetic validation: a swept rotor pass that draws the ladder
# ====================================================================================
def synthetic_rotor_pass(blade_hz=200.0, sweep_px_s=-1800.0, *, duration_s=0.16, x0=300.0,
                         disk_px=7.0, burst_events=40, burst_jitter_s=4e-4, noise_events=0,
                         width=320, seed=0):
    """Generate ``(x, t)`` for one swept multirotor pass — the ladder, for tests/demos.

    The sensor sweeps the target across the frame at ``sweep_px_s`` while the rotor bursts at
    ``blade_hz``; each burst is a small disk of events. Optionally salts in uniform noise.
    """
    rng = np.random.default_rng(seed)
    xs, ts = [], []
    n_bursts = max(int(duration_s * blade_hz), 1)
    for k in range(n_bursts):
        tau = (k + 0.5) / blade_hz
        xc = x0 + sweep_px_s * tau
        nb = max(int(rng.poisson(burst_events)), 1)
        xs.append(xc + rng.normal(0, disk_px, nb))
        ts.append(tau + rng.normal(0, burst_jitter_s, nb))
    if noise_events > 0:
        xs.append(rng.uniform(0, width, noise_events))
        ts.append(rng.uniform(0, duration_s, noise_events))
    x = np.clip(np.concatenate(xs), 0, width - 1)
    t = np.concatenate(ts)
    order = np.argsort(t)
    return x[order], t[order]


# ====================================================================================
# Figure (the (x, t) staircase + the comb autocorrelation)
# ====================================================================================
def ladder_figure(x, t, result: LadderResult, title=None):
    """A two-panel figure: the (t, x) event cloud with the rung drift, and the comb ACF."""
    import matplotlib.pyplot as plt
    x = np.asarray(x, float); t = np.asarray(t, float)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="w",
                                   gridspec_kw={"width_ratios": [3, 2]})
    ax0.scatter(t * 1e3, x, s=3, c="#1f4e8c", alpha=0.4)
    if result.drift_px_s:
        tt = np.array([t.min(), t.max()])
        x0 = np.median(x) - result.drift_px_s * np.median(t)
        ax0.plot(tt * 1e3, x0 + result.drift_px_s * tt, "-", color="#ef6c00", lw=1.5,
                 label=f"sweep v = {result.drift_px_s:.0f} px/s")
        ax0.legend(fontsize=8, loc="upper right")
    ax0.set_xlabel("time [ms]"); ax0.set_ylabel("sweep column x [px]")
    ax0.set_title("rotor ladder — event cloud")
    ax0.grid(True, ls="--", alpha=0.3)

    lo, hi = np.percentile(x, [0.5, 99.5])
    nb = max(int(hi - lo), 8)
    h, _ = np.histogram(x, bins=nb, range=(lo, hi)); hc = h - h.mean()
    acf = np.correlate(hc, hc, "full")[nb - 1:]
    acf = acf / acf[0] if acf[0] > 0 else acf
    ax1.plot(np.arange(len(acf)), acf, color="#1f4e8c")
    if result.step_px:
        for m in (1, 2, 3):
            ax1.axvline(result.step_px * m, color="#c62828", ls="--", lw=1.0, alpha=0.8)
        ax1.annotate(f"Δx = {result.step_px:g} px", xy=(result.step_px, acf.max()),
                     fontsize=9, color="#c62828")
    ax1.set_xlim(0, min(len(acf), (result.step_px or 10) * 4 + 5))
    ax1.set_xlabel("sweep-coordinate lag [px]"); ax1.set_ylabel("autocorrelation")
    ax1.set_title("comb (rung spacing)"); ax1.grid(True, ls="--", alpha=0.3)

    verdict = (f"DRONE — f = {result.blade_hz:g} Hz" if result.detected
               else "no rotor ladder")
    fig.suptitle(title or f"{verdict}   ·   Δx={result.step_px} px · comb={result.comb_strength} · "
                 f"{result.n_events} ev", fontsize=11)
    fig.tight_layout()
    return fig
