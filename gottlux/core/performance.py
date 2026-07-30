"""
performance.py — the operator-facing results metrics ("KPIs") for a recording.

Three questions a fielded passive-vision drone-sensor must answer, and the three independent
metrics this module computes from first principles:

1. **Tracking range** — *how far away can the drone still be tracked?* A blob is trackable
   while its image is large enough to survive the detector's minimum-size gate. From the
   pinhole relation ``N(D) = L·f_px / D`` (pixels-on-target ``N`` for a target of physical
   size ``L`` at range ``D``), the maximum tracking range is the ``D`` at which ``N`` drops to
   the trackable-pixel threshold. The Johnson detection/recognition/identification ladder is
   reported alongside for the spatial-resolution context.

2. **Prop-frequency range** — *how far away can the rotor blade-pass tone still be resolved?*
   This is a *temporal* perception limit, not a spatial one: the rotor's periodic brightness
   modulation must produce a spectral peak above the FFT noise floor (SNR ≥ a gate). The number
   of coherently-modulated events scales with the rotor disk's image area ∝ ``N² ∝ 1/D²``, so
   the in-band ``SNR(D) ≈ SNR_ref · (D_ref/D)²``. Resolvable while ``SNR(D) ≥ SNR_gate`` gives
   ``D_freq = D_ref · √(SNR_ref / SNR_gate)``. With no measured tone to calibrate against, it
   falls back to a pixels-on-target threshold (you need the disk spatially resolved to read its
   flicker).

3. **Time-to-contact** — *how much warning does the operator get?* Two complementary bases:
   a **nominal** capability ``TTC = D_detect / V_approach`` (warning time if a drone closes at
   ``V_approach`` from the detection range), and a **measured** ``TTC(t) = D(t) / (−dD/dt)`` from
   the range trend of an actually-approaching track.

Each metric is a **standalone** pure-NumPy function returning its own result dataclass and is
computed in isolation — a failure or "no data" in one never affects the others (the orchestrator
in :mod:`gottlux.run.performance_report` wraps each call independently). Everything is
regime-agnostic: the same math serves a *staring* sensor and a de-rotated *rotating* one.

No Qt, no matplotlib, no detector imports — just NumPy and :mod:`gottlux.core.photogrammetry`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gottlux.core import photogrammetry as pg


# --------------------------------------------------------------------- shared status
@dataclass
class MetricStatus:
    """Whether a metric was computed, and from what — so a partial run is self-describing."""
    ok: bool = True
    status: str = "ok"          # 'ok' | 'model_only' | 'no_data' | 'failed'
    message: str = ""

    @classmethod
    def model_only(cls, why: str) -> "MetricStatus":
        return cls(ok=True, status="model_only", message=why)

    @classmethod
    def no_data(cls, why: str) -> "MetricStatus":
        return cls(ok=False, status="no_data", message=why)

    @classmethod
    def failed(cls, why: str) -> "MetricStatus":
        return cls(ok=False, status="failed", message=why)


def _finite(*arrays):
    """Stack arrays column-wise and keep only rows finite in every column."""
    cols = [np.asarray(a, float).ravel() for a in arrays]
    n = min((c.size for c in cols), default=0)
    if n == 0:
        return [np.zeros(0) for _ in cols]
    cols = [c[:n] for c in cols]
    mask = np.ones(n, bool)
    for c in cols:
        mask &= np.isfinite(c)
    return [c[mask] for c in cols]


# ==================================================================================
# 1) Tracking range
# ==================================================================================
@dataclass
class TrackingRangeResult:
    """Max range a target of size ``target_size_m`` can be tracked, model + measured."""
    target_size_m: float
    fov_deg: float
    width_px: int
    focal_px: float
    track_px: float                       # the trackable-pixel threshold used
    capability_range_m: float             # pinhole D at which N == track_px
    johnson_ranges_m: dict = field(default_factory=dict)
    measured_max_range_m: Optional[float] = None
    measured_max_px: Optional[float] = None
    effective_track_px: Optional[float] = None    # pixels-on-target at the measured reach
    n_detections: int = 0
    status: MetricStatus = field(default_factory=MetricStatus)

    @property
    def range_m(self) -> float:
        """The headline tracking range: the measured reach if available, else the capability."""
        return self.measured_max_range_m if self.measured_max_range_m is not None \
            else self.capability_range_m


def tracking_range(target_size_m: float, fov_deg: float, width_px: int,
                   track_px: Optional[float] = None, min_pixels_area: float = 60.0,
                   measured_ranges=None, measured_px=None) -> TrackingRangeResult:
    """Maximum range at which the target remains trackable.

    The capability comes from the pinhole model: trackable while pixels-on-target
    ``N ≥ track_px``. If *track_px* is not given it is derived from the detector's minimum
    connected-component **area** (``√min_pixels_area`` — the linear extent of the smallest
    accepted blob). *measured_ranges* (and/or *measured_px*) are the per-detection ranges (m)
    and apparent sizes (px) from a detector run; the farthest finite range is the measured reach.
    """
    if track_px is None:
        track_px = float(np.sqrt(max(min_pixels_area, 1.0)))
    L = float(target_size_m)
    f = pg.focal_px(fov_deg, width_px)
    cap = float(pg.range_for_pixels(L, track_px, fov_deg, width_px)) if L > 0 else float("nan")
    johnson = pg.perception_ranges(L, fov_deg, width_px) if L > 0 else {}

    res = TrackingRangeResult(target_size_m=L, fov_deg=float(fov_deg), width_px=int(width_px),
                              focal_px=round(f, 2), track_px=round(float(track_px), 3),
                              capability_range_m=round(cap, 3) if np.isfinite(cap) else None,
                              johnson_ranges_m={k: round(v, 2) for k, v in johnson.items()})
    # measured reach (kept entirely separate from the model — one can succeed without the other)
    rng = np.asarray(measured_ranges, float).ravel() if measured_ranges is not None else np.zeros(0)
    rng = rng[np.isfinite(rng) & (rng > 0)]
    if rng.size:
        res.n_detections = int(rng.size)
        res.measured_max_range_m = round(float(np.nanmax(rng)), 3)
        if L > 0 and res.measured_max_range_m > 0:   # pixels-on-target the detector actually held
            res.effective_track_px = round(
                float(pg.pixels_on_target(L, res.measured_max_range_m, fov_deg, width_px)), 2)
    if measured_px is not None:
        px = np.asarray(measured_px, float).ravel()
        px = px[np.isfinite(px)]
        if px.size:
            res.measured_max_px = round(float(np.nanmin(px)), 2)   # smallest = farthest
    if L <= 0:
        res.status = MetricStatus.failed("absolute ranging disabled (target_size_m ≤ 0)")
    elif rng.size == 0:
        res.status = MetricStatus.model_only("no detections; capability (model) range only")
    return res


# ==================================================================================
# 2) Prop-frequency-resolution range
# ==================================================================================
@dataclass
class PropFrequencyRangeResult:
    """Max range the rotor blade-pass tone is resolvable, model + measured."""
    snr_gate: float
    model: str                            # 'snr_inverse_square' | 'snr_fit' | 'pixels_on_target'
    capability_range_m: Optional[float] = None
    d_ref_m: Optional[float] = None       # calibration reference range
    snr_ref: Optional[float] = None       # SNR at d_ref_m
    slope: Optional[float] = None         # fitted log-log slope (≈ −2 expected)
    r2: Optional[float] = None
    n_freq_px: Optional[float] = None     # pixels-on-target threshold (fallback model)
    measured_max_range_m: Optional[float] = None
    n_resolved: int = 0
    status: MetricStatus = field(default_factory=MetricStatus)

    @property
    def range_m(self) -> Optional[float]:
        """Headline prop-frequency range: measured reach if available, else the model."""
        return self.measured_max_range_m if self.measured_max_range_m is not None \
            else self.capability_range_m


def prop_frequency_range(snr_gate: float, target_size_m: float, fov_deg: float, width_px: int,
                         measured_ranges=None, measured_snr=None,
                         n_freq_px: float = 8.0) -> PropFrequencyRangeResult:
    """Maximum range at which the rotor blade-pass tone is resolvable (SNR ≥ *snr_gate*).

    Preferred (data-grounded) model: fit ``SNR ∝ D^slope`` to the measured (range, in-band SNR)
    points and solve for the range at which the fit crosses *snr_gate*. With a single point the
    physically-expected slope −2 (events ∝ disk area ∝ 1/D²) is assumed. With **no** resolved
    tone to calibrate, falls back to a pixels-on-target threshold *n_freq_px* (you must spatially
    resolve the disk to read its flicker): ``D = L·f_px / n_freq_px``.
    """
    snr_gate = float(snr_gate)
    L = float(target_size_m)
    rng, snr = (np.zeros(0), np.zeros(0))
    if measured_ranges is not None and measured_snr is not None:
        rng, snr = _finite(measured_ranges, measured_snr)
        keep = (rng > 0) & (snr > 0)
        rng, snr = rng[keep], snr[keep]

    res = PropFrequencyRangeResult(snr_gate=snr_gate, model="pixels_on_target")

    # --- data-grounded model (kept independent of the spatial tracking metric) ---
    resolved = snr >= snr_gate
    res.n_resolved = int(resolved.sum())
    if rng.size >= 2 and np.unique(rng).size >= 2:
        lr, ls = np.log(rng), np.log(snr)
        slope, intercept = np.polyfit(lr, ls, 1)
        pred = intercept + slope * lr
        ss_res = float(np.sum((ls - pred) ** 2)); ss_tot = float(np.sum((ls - ls.mean()) ** 2))
        res.r2 = round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else None
        if slope < -1e-3:                          # SNR must fall with range to extrapolate a limit
            d_gate = float(np.exp((np.log(snr_gate) - intercept) / slope))
            res.model = "snr_fit"
            res.slope = round(float(slope), 3)
            res.d_ref_m = round(float(np.median(rng)), 3)
            res.snr_ref = round(float(np.median(snr)), 3)
            res.capability_range_m = round(d_gate, 3)
    elif rng.size == 1:                            # one point → assume the −2 power law
        d_ref, snr_ref = float(rng[0]), float(snr[0])
        res.model = "snr_inverse_square"; res.slope = -2.0
        res.d_ref_m = round(d_ref, 3); res.snr_ref = round(snr_ref, 3)
        res.capability_range_m = round(d_ref * np.sqrt(max(snr_ref, 0.0) / snr_gate), 3)

    # --- fallback model when no usable SNR-vs-range relation exists ---
    if res.capability_range_m is None:
        res.n_freq_px = float(n_freq_px)
        if L > 0:
            res.capability_range_m = round(float(pg.range_for_pixels(L, n_freq_px, fov_deg, width_px)), 3)
        if res.n_resolved == 0:
            res.status = MetricStatus.model_only(
                "no tone met the SNR gate in this clip; pixels-on-target model only")

    # measured reach: the farthest range where the tone actually cleared the gate
    if res.n_resolved:
        res.measured_max_range_m = round(float(np.nanmax(rng[resolved])), 3)
    return res


# ==================================================================================
# 3) Time-to-contact (operator response time)
# ==================================================================================
@dataclass
class TimeToContactResult:
    """Warning time before a closing target arrives — nominal capability + measured."""
    approach_speed_mps: float
    detect_range_m: Optional[float] = None
    nominal_ttc_s: Optional[float] = None
    nominal_sweep_s: dict = field(default_factory=dict)        # {speed_mps: ttc_s}
    measured_closing_speed_mps: Optional[float] = None
    measured_ttc_at_first_s: Optional[float] = None            # warning at first detection
    measured_min_ttc_s: Optional[float] = None
    approaching: bool = False
    n_points: int = 0
    status: MetricStatus = field(default_factory=MetricStatus)


def time_to_contact(detect_range_m: Optional[float], approach_speed_mps: float = 15.0,
                    speed_sweep=(5.0, 10.0, 15.0, 20.0),
                    range_t=None, t_s=None) -> TimeToContactResult:
    """Operator warning time.

    *detect_range_m* is the range at which the threat is first acquired (the tracking range);
    the **nominal** warning time is ``detect_range_m / approach_speed_mps`` (a sweep over
    *speed_sweep* gives the curve). When a per-detection range history (*range_t*, *t_s*) is
    supplied for an **approaching** track, the **measured** closing speed (the robust slope of
    ``−range`` vs time) yields ``TTC(t) = range(t) / closing_speed`` — reported at first
    detection (the real warning time) and at its minimum.
    """
    v = float(approach_speed_mps)
    res = TimeToContactResult(approach_speed_mps=v)
    if detect_range_m is not None and np.isfinite(detect_range_m) and detect_range_m > 0:
        D = float(detect_range_m)
        res.detect_range_m = round(D, 3)
        res.nominal_ttc_s = round(D / v, 3) if v > 0 else None
        res.nominal_sweep_s = {float(s): round(D / s, 3) for s in speed_sweep if s > 0}

    # measured closing speed from the range trend of an approaching track
    if range_t is not None and t_s is not None:
        d, t = _finite(range_t, t_s)
        order = np.argsort(t); d, t = d[order], t[order]
        if d.size >= 2 and (t[-1] - t[0]) > 0:
            res.n_points = int(d.size)
            slope = np.polyfit(t, d, 1)[0]          # m/s; negative ⇒ closing
            closing = -float(slope)
            res.approaching = closing > 0
            if closing > 1e-6:
                res.measured_closing_speed_mps = round(closing, 3)
                ttc = d / closing
                res.measured_ttc_at_first_s = round(float(d[0] / closing), 3)
                res.measured_min_ttc_s = round(float(np.nanmin(ttc)), 3)
    if res.detect_range_m is None and res.measured_closing_speed_mps is None:
        res.status = MetricStatus.no_data("no detection range and no approaching track")
    elif res.measured_closing_speed_mps is None:
        res.status = MetricStatus.model_only("nominal warning time only (no approaching track)")
    return res
