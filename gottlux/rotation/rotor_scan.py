"""
rotor_scan.py — the 360° rotor-ladder survey: classify one target, then key the whole
revolution off its propeller signature to map every bearing where the same rotor appears.

This sits one level above :mod:`gottlux.rotation.rotor_ladder` (which measures the stair-step
comb of a *single* swept pass). Here we:

1. **Classify a box** (:func:`analyze_box`) — put the analysis box on the target, recover its
   blade-pass frequency from the comb (``f = |v|/Δx``), and *quantify the propeller*: rotor rate,
   RPM, and tip speed for an assumed blade count and prop diameter, plus the target's bearing and
   (pinhole) range. This is the "what am I looking at" step and the **template** for the survey.

2. **Scan the 360°** (:func:`scan_rotation`) — use that blade frequency as a *matched key* and
   sweep the entire rotation: events are de-rotated to **world azimuth** and binned by
   ``(revolution, azimuth)``; each cell with enough events gets the same cheap comb test. Cells
   whose implied ``f`` lands within tolerance of the template are flagged as the same rotor. The
   output is a list of :class:`LadderDetection` over the whole sky — *where else the signature is*.

3. **Link across revolutions** (:func:`link_tracks`) — group a target's detections from
   successive revolutions into a :class:`RotorTrack`. A stationary drone repeats at the same
   bearing; a moving one shifts by a fixed **azimuth offset per revolution** — that offset, over
   the rotation period, is the target's relative angular rate (the "unique offset from the
   rotation"). With bearing + range per detection this projects straight onto a radar map.

Why this is cheap / live-able
-----------------------------
Every cell costs one robust line fit + one autocorrelation of a small 1-D histogram — no
per-pixel FFT, no fine temporal sampling. De-rotating to world azimuth groups a target's whole
transit (and its recurrences) into a tight bin regardless of revolution, so recurrence and the
motion offset fall out of a single binning. The expensive front-end (background suppression /
isolation) is shared with the rest of the rotation pipeline and reused, not redone here.

The detection physics, equations, and limits are documented in :mod:`gottlux.rotation.rotor_ladder`
and ``docs/ROTOR_LADDER.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gottlux.rotation.detect import estimate_range_m, focal_px
from gottlux.rotation.rotor_ladder import LadderResult, ladder_signature

#: Speed of sound (m/s, ~sea level, 20 C) for the rotor tip-speed Mach sanity number.
SPEED_OF_SOUND_MPS = 343.0


# ====================================================================================
# Result records
# ====================================================================================
@dataclass
class PropellerSignature:
    """The rotor quantities derived from a measured blade-pass frequency.

    ``blade_hz`` is the directly measured comb tone (blades passing a point per second);
    everything else follows from the assumed blade count and prop diameter and is *reporting
    only* — it never feeds back into detection.
    """
    blade_hz: float
    n_blades: int = 2
    prop_diameter_m: float = 0.127
    rotor_hz: float = 0.0          # blade_hz / n_blades  (revolutions per second of the rotor)
    rpm: float = 0.0               # rotor_hz * 60
    tip_speed_mps: float = 0.0     # pi * D * rotor_hz
    tip_mach: float = 0.0          # tip_speed / speed of sound

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def propeller_kinematics(blade_hz, n_blades=2, prop_diameter_m=0.127) -> Optional[PropellerSignature]:
    """Turn a blade-pass frequency into a :class:`PropellerSignature` (rotor Hz, RPM, tip speed).

    Returns ``None`` if *blade_hz* is missing/non-positive. ``rotor_hz = blade_hz / n_blades``
    because each rotor revolution presents *n_blades* blade-passes to a fixed observation point.
    """
    if not blade_hz or blade_hz <= 0:
        return None
    nb = max(int(n_blades), 1)
    rotor_hz = float(blade_hz) / nb
    tip = float(np.pi * prop_diameter_m * rotor_hz)
    return PropellerSignature(
        blade_hz=round(float(blade_hz), 1), n_blades=nb, prop_diameter_m=float(prop_diameter_m),
        rotor_hz=round(rotor_hz, 1), rpm=round(rotor_hz * 60.0, 0),
        tip_speed_mps=round(tip, 1), tip_mach=round(tip / SPEED_OF_SOUND_MPS, 3))


@dataclass
class LadderDetection:
    """One rotor-ladder verdict in a single ``(revolution, azimuth)`` scan cell (or a box)."""
    rev: int
    t_s: float
    bearing_deg: float
    range_m: Optional[float]
    elev_deg: Optional[float]
    blade_hz: Optional[float]
    step_px: Optional[float]
    drift_px_s: float
    comb_strength: float
    n_events: int
    size_px: float = 0.0           # apparent vertical extent used for ranging
    detected: bool = False
    matches_template: bool = False
    rotor_hz: Optional[float] = None
    rpm: Optional[float] = None

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class RotorTrack:
    """A rotor seen across several revolutions — its recurrence and per-revolution motion offset."""
    bearing_deg: float                       # representative (median) world bearing
    n_passes: int
    median_blade_hz: Optional[float]
    blade_hz_stability: float                # 1 − scatter/median across revs (1 = rock steady)
    bearing_offset_per_rev_deg: float        # how far the bearing shifts each revolution
    omega_deg_s: float                       # relative angular rate = offset / T_rot
    range_m: Optional[float]
    revs: list = field(default_factory=list)
    bearings_deg: list = field(default_factory=list)
    blade_hz: list = field(default_factory=list)

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class RotorScanResult:
    """Everything the 360° rotor-ladder survey produced for one recording."""
    detections: list                         # list[LadderDetection] over the whole rotation
    tracks: list                             # list[RotorTrack]
    template: Optional[LadderResult] = None  # the box / strongest-cell signature
    template_signature: Optional[PropellerSignature] = None
    template_bearing_deg: Optional[float] = None
    template_range_m: Optional[float] = None
    template_events: Optional[tuple] = None  # (x, t_s) of the template cell, for the figure
    f_template_hz: Optional[float] = None
    accumulated_blade_hz: Optional[float] = None     # cross-revolution coherent comb (robust f)
    accumulated_comb_strength: float = 0.0
    accumulated_acf: Optional[object] = None          # the accumulated ACF (for the figure)
    blade_tol: float = 0.25
    bin_deg: float = 3.0
    n_blades: int = 2
    prop_diameter_m: float = 0.127
    t_rot_s: float = 0.0
    sweep_px_s: Optional[float] = None
    fov_deg: Optional[float] = None

    # --------------------------------------------------------------- convenience views
    @property
    def matched(self) -> list:
        return [d for d in self.detections if d.matches_template]

    @property
    def n_matched(self) -> int:
        return len(self.matched)

    def headline(self) -> dict:
        """A compact summary for the run manifest."""
        matched = self.matched
        bands = [d.blade_hz for d in matched if d.blade_hz]
        return {
            "f_template_hz": self.f_template_hz,
            "accumulated_blade_hz": self.accumulated_blade_hz,
            "accumulated_comb_strength": self.accumulated_comb_strength,
            "rpm": (self.template_signature.rpm if self.template_signature else None),
            "n_cells": len(self.detections),
            "n_detections": int(sum(d.detected for d in self.detections)),
            "n_matched": len(matched),
            "n_tracks": len(self.tracks),
            "bearings_deg": [round(d.bearing_deg, 1) for d in matched],
            "blade_hz_range": ([round(min(bands), 1), round(max(bands), 1)] if bands else None),
            "max_omega_deg_s": (round(max((abs(t.omega_deg_s) for t in self.tracks), default=0.0), 2)),
        }

    def as_dict(self) -> dict:
        return {
            "f_template_hz": self.f_template_hz,
            "accumulated_blade_hz": self.accumulated_blade_hz,
            "accumulated_comb_strength": self.accumulated_comb_strength,
            "blade_tol": self.blade_tol,
            "bin_deg": self.bin_deg,
            "n_blades": self.n_blades,
            "prop_diameter_m": self.prop_diameter_m,
            "t_rot_s": self.t_rot_s,
            "sweep_px_s": self.sweep_px_s,
            "fov_deg": self.fov_deg,
            "template": (self.template.as_dict() if self.template else None),
            "template_signature": (self.template_signature.as_dict()
                                   if self.template_signature else None),
            "template_bearing_deg": self.template_bearing_deg,
            "template_range_m": self.template_range_m,
            "detections": [d.as_dict() for d in self.detections],
            "tracks": [t.as_dict() for t in self.tracks],
            "headline": self.headline(),
        }


# ====================================================================================
# Geometry helpers
# ====================================================================================
def sweep_velocity_px_s(cfg, tel) -> Optional[float]:
    """**Signed** sweep velocity ``v = dx/dt`` of a stationary target across the sensor, from
    telemetry.

    A fixed-world point holds its world azimuth, so ``d(boresight)/dt + a·β·dx/dt = 0`` with
    ``a = az_sign`` and ``β`` the pixel angular scale, giving ``v = −Ω / (a·β) = −(Ω·W/FOV)/a``
    px/s. The magnitude ``|Ω·W/FOV|`` turns a hard temporal measurement into the easy spatial one;
    the sign (handedness × spin direction) is what makes the figure's drift line track the cloud.
    Used directly in rotation mode (robust to the noise that biases an event-cloud slope fit) and
    as a fallback otherwise. ``None`` if there is no telemetry.
    """
    if tel is None or not getattr(tel, "omega_deg_s", 0.0) or not cfg.fov_deg or not cfg.sensor_w:
        return None
    deg_per_px = cfg.fov_deg / cfg.sensor_w
    if deg_per_px <= 0:
        return None
    return float(-(tel.omega_deg_s / deg_per_px) / (cfg.az_sign or 1.0))


def event_world_azimuth(x, t_s, cfg, tel) -> np.ndarray:
    """World azimuth (deg, 0..360) each event saw: boresight azimuth(t) + the intra-FOV term.

    In staring mode (no telemetry) this is the relative bearing within the FOV (boresight = 0).
    Mirrors :func:`gottlux.rotation.detect.build_trajectory` / ``derotate_events``.
    """
    x = np.asarray(x, float)
    deg_per_px = cfg.fov_deg / cfg.sensor_w
    intra = cfg.az_sign * (x - cfg.sensor_w / 2.0) * deg_per_px
    if tel is not None:
        bore = tel.azimuth_at(np.asarray(t_s, float))
        return np.mod(bore + intra, 360.0)
    return np.mod(intra, 360.0)


def _range_from_extent(y_cell, cfg) -> tuple:
    """(range_m, size_px) from a cell's vertical event spread via the pinhole model.

    The apparent target height is the span of a robust **density core** (events within 3·MAD of
    the median), so a few scattered background/noise events do not inflate it; the pinhole gives
    ``range = phys · focal_px / size_px``. ``range_m`` is ``None`` when ranging is disabled
    (``target_size_m == 0``) or the cell is degenerate.
    """
    y_cell = np.asarray(y_cell, float)
    if y_cell.size < 8:
        return None, 0.0
    med = np.median(y_cell)
    mad = np.median(np.abs(y_cell - med)) * 1.4826
    core = y_cell[np.abs(y_cell - med) <= max(3.0 * mad, 1.0)]
    if core.size < 8:
        core = y_cell
    lo, hi = np.percentile(core, [2.5, 97.5])
    size_px = float(hi - lo + 1.0)
    if cfg.target_size_m and cfg.target_size_m > 0 and size_px > 0:
        rng = float(estimate_range_m(size_px, cfg.fov_deg, cfg.target_size_m, cfg.sensor_w))
        return rng, size_px
    return None, size_px


def _elev_deg(y_cell, cfg) -> float:
    deg_per_px = cfg.fov_deg / cfg.sensor_w
    return float((cfg.sensor_h / 2.0 - np.median(np.asarray(y_cell, float))) * deg_per_px)


def _ang_diff(a, b) -> float:
    """Signed smallest angular difference a − b in degrees, wrapped to (−180, 180]."""
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def accumulate_comb(x_passes, sweep_px_s, *, f_lo=80.0, f_hi=1500.0, bin_px=1.0, n_lag=220):
    """Coherently accumulate the sweep-coordinate autocorrelation across several passes.

    Each pass of the *same* rotor under the *same* sweep rate leaves a comb at the *same* spacing
    ``Δx = |v|/f``, but its phase (where the rungs fall) varies. The autocorrelation is
    phase-invariant, so **averaging the per-pass ACFs reinforces the comb (~√N) while noise averages
    down** — recovering a blade frequency the overlap-regime single pass cannot. The fundamental is
    the smallest strong peak of the accumulated ACF inside the band ``|v|/f_hi … |v|/f_lo``.

    Parameters
    ----------
    x_passes : list[array]   sensor-column (sweep-coordinate) events, one array per pass/revolution.
    sweep_px_s : float       the sweep velocity (telemetry), used to map lag → frequency.

    Returns a dict ``{acf, n_passes, step_px, blade_hz, comb_strength, lag_min, lag_max}`` or
    ``None`` if no pass had enough structure.
    """
    v = abs(float(sweep_px_s)) if sweep_px_s else 0.0
    if v <= 0:
        return None
    acc = np.zeros(n_lag)
    used = 0
    for x in x_passes:
        x = np.asarray(x, float)
        if x.size < 60:
            continue
        lo, hi = np.percentile(x, [0.5, 99.5])
        if hi - lo < 3 * bin_px:
            continue
        nb = max(int((hi - lo) / bin_px), 8)
        h, _ = np.histogram(x, bins=nb, range=(lo, hi))
        hc = h - h.mean()
        a = np.correlate(hc, hc, "full")[nb - 1:]
        if a[0] <= 0:
            continue
        a = a / a[0]
        L = min(n_lag, a.size)
        acc[:L] += a[:L]
        used += 1
    if used == 0:
        return None
    acc /= used
    lag_min = max(int(round(v / f_hi / bin_px)), 2)
    lag_max = min(int(round(v / f_lo / bin_px)), n_lag - 2)
    out = {"acf": acc, "n_passes": used, "step_px": None, "blade_hz": None,
           "comb_strength": 0.0, "lag_min": lag_min, "lag_max": lag_max}
    if lag_max <= lag_min:
        return out
    peaks = [l for l in range(lag_min, lag_max + 1)
             if acc[l] > 0 and acc[l] >= acc[l - 1] and acc[l] >= acc[l + 1]]
    if not peaks:
        return out
    mx = max(acc[l] for l in peaks)
    strong = [l for l in peaks if acc[l] >= 0.6 * mx]
    best = min(strong)
    out["step_px"] = round(best * bin_px, 3)
    out["blade_hz"] = round(v / (best * bin_px), 1)
    out["comb_strength"] = round(float(acc[best]), 3)
    return out


# ====================================================================================
# 1) Classify a box  →  the template propeller signature
# ====================================================================================
def analyze_box(ev, cfg, tel=None, *, roi=None, t0=None, t1=None,
                f_lo=80.0, f_hi=800.0, min_events=120, sweep_px_s=None):
    """Measure the rotor-ladder on a spatiotemporal box and quantify the propeller + geometry.

    Parameters
    ----------
    ev : dict       the EBS event dict (``x, y, p, t`` with ``t`` in µs) — e.g. ``ctx.ev``.
    cfg : Config    resolved geometry (``fov_deg, sensor_w/h, target_size_m, rotor_blades, …``).
    tel : Telemetry | None   rotation telemetry, for the world bearing and the sweep-rate fallback.
    roi : (x0,y0,x1,y1) | None   the analysis box in pixels (``None`` = whole frame).
    t0, t1 : float | None    time window in seconds (``None`` = full span).

    Returns ``(LadderResult, LadderDetection)`` where the detection carries the bearing, range,
    propeller kinematics, and the box's ``(x, t)`` are recoverable from *ev*+*roi* for figures.
    Returns ``(LadderResult(n_events=0,…), None)`` if the box is too sparse to judge.
    """
    x = np.asarray(ev["x"], float)
    y = np.asarray(ev["y"], float)
    t = np.asarray(ev["t"], float) / 1e6
    m = np.ones(x.shape, bool)
    if t0 is not None:
        m &= t >= t0
    if t1 is not None:
        m &= t < t1
    if roi is not None:
        x0, y0, x1, y1 = roi
        m &= (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    xs, ys, ts = x[m], y[m], t[m]
    v = sweep_px_s if sweep_px_s is not None else sweep_velocity_px_s(cfg, tel)
    res = ladder_signature(xs, ts, sweep_px_s=(v if v else None),
                           f_lo=f_lo, f_hi=f_hi, min_events=min_events)
    if xs.size == 0:
        return res, None
    bearing = float(np.median(event_world_azimuth(xs, ts, cfg, tel)))
    rng, size_px = _range_from_extent(ys, cfg)
    sig = propeller_kinematics(res.blade_hz, cfg.rotor_blades, cfg.prop_diameter_m)
    det = LadderDetection(
        rev=int(tel.revolution_at(np.array([np.median(ts)]))[0]) if tel is not None else 0,
        t_s=float(np.median(ts)), bearing_deg=bearing, range_m=rng, elev_deg=_elev_deg(ys, cfg),
        blade_hz=res.blade_hz, step_px=res.step_px, drift_px_s=res.drift_px_s,
        comb_strength=res.comb_strength, n_events=res.n_events, size_px=round(size_px, 2),
        detected=res.detected, matches_template=res.detected,
        rotor_hz=(sig.rotor_hz if sig else None), rpm=(sig.rpm if sig else None))
    return res, det


# ====================================================================================
# 2) Scan the whole 360°
# ====================================================================================
def scan_rotation(ev, cfg, tel=None, *, keep=None, template_hz=None,
                  f_lo=80.0, f_hi=800.0, bin_deg=None, min_events=None,
                  blade_tol=None, sweep_px_s=None) -> RotorScanResult:
    """Sweep the whole rotation for the rotor-ladder signature and map every bearing it appears.

    Events (optionally restricted to the background-suppressed ``keep`` mask) are de-rotated to
    world azimuth and binned by ``(revolution, azimuth)``. Each populated cell is comb-tested.
    If *template_hz* is given (from :func:`analyze_box`), cells within ``blade_tol`` of it are
    flagged ``matches_template``; otherwise the strongest detected cell becomes the template.

    Parameters
    ----------
    ev : dict       the EBS event dict (``ctx.ev``).
    cfg : Config    resolved geometry + the ``ladder_*`` knobs.
    tel : Telemetry | None
    keep : ndarray[bool] | None   per-event foreground mask (``ctx.keep``); ``None`` = all events.
    template_hz : float | None    the matched-filter blade frequency; ``None`` → auto.
    bin_deg, min_events, blade_tol : override the corresponding ``cfg.ladder_*`` defaults.

    Returns a :class:`RotorScanResult`.
    """
    bin_deg = float(cfg.ladder_bin_deg if bin_deg is None else bin_deg)
    min_events = int(cfg.ladder_min_events if min_events is None else min_events)
    blade_tol = float(cfg.ladder_blade_tol if blade_tol is None else blade_tol)

    x = np.asarray(ev["x"], float)
    y = np.asarray(ev["y"], float)
    t = np.asarray(ev["t"], float) / 1e6
    if keep is not None:
        keep = np.asarray(keep, bool)
        x, y, t = x[keep], y[keep], t[keep]

    v_fallback = sweep_px_s if sweep_px_s is not None else sweep_velocity_px_s(cfg, tel)
    t_rot = float(getattr(tel, "T_rot", 0.0)) if tel is not None else 0.0

    result = RotorScanResult(
        detections=[], tracks=[], blade_tol=blade_tol, bin_deg=bin_deg,
        n_blades=cfg.rotor_blades, prop_diameter_m=cfg.prop_diameter_m,
        t_rot_s=round(t_rot, 5), sweep_px_s=(round(v_fallback, 1) if v_fallback else None),
        fov_deg=cfg.fov_deg, f_template_hz=template_hz)
    if x.size < min_events:
        return result

    world_az = event_world_azimuth(x, t, cfg, tel)
    revs = (tel.revolution_at(t).astype(int) if tel is not None else np.zeros(x.size, int))
    az_bin = np.clip((world_az / bin_deg).astype(int), 0, int(np.ceil(360.0 / bin_deg)) - 1)
    cell_id = revs.astype(np.int64) * 100000 + az_bin     # unique per (rev, bin)

    order = np.argsort(cell_id, kind="stable")
    cid = cell_id[order]
    bounds = np.flatnonzero(np.diff(cid)) + 1
    starts = np.concatenate(([0], bounds))
    stops = np.concatenate((bounds, [cid.size]))

    # In rotation mode the sweep velocity is known exactly from telemetry (signed) — use it (robust
    # to the residual noise that biases an event-cloud slope fit); fall back to estimating it only
    # when there is no telemetry.
    sweep = v_fallback if v_fallback else None
    strongest = None        # (comb, x_cell, t_cell) of the strongest detected cell → auto template
    for s, e in zip(starts, stops):
        if e - s < min_events:
            continue
        idx = order[s:e]
        xx, yy, tt, aa = x[idx], y[idx], t[idx], world_az[idx]
        res = ladder_signature(xx, tt, sweep_px_s=sweep, f_lo=f_lo, f_hi=f_hi,
                               min_events=min_events)
        rng, size_px = _range_from_extent(yy, cfg)
        sig = propeller_kinematics(res.blade_hz, cfg.rotor_blades, cfg.prop_diameter_m)
        det = LadderDetection(
            rev=int(revs[idx[0]]), t_s=float(np.median(tt)),
            bearing_deg=float(np.median(aa)), range_m=rng, elev_deg=_elev_deg(yy, cfg),
            blade_hz=res.blade_hz, step_px=res.step_px, drift_px_s=res.drift_px_s,
            comb_strength=res.comb_strength, n_events=res.n_events, size_px=round(size_px, 2),
            detected=res.detected, rotor_hz=(sig.rotor_hz if sig else None),
            rpm=(sig.rpm if sig else None))
        if res.detected:
            det._x = xx.copy()          # stash for cross-revolution accumulation (not serialized)
        result.detections.append(det)
        if res.detected and (strongest is None or res.comb_strength > strongest[0]):
            strongest = (res.comb_strength, xx.copy(), tt.copy(), det, res)

    # ---- template: explicit, else the strongest detected cell ----
    if template_hz is None and strongest is not None:
        template_hz = strongest[3].blade_hz
        result.template = strongest[4]
        result.template_events = (strongest[1], strongest[2])
        result.template_bearing_deg = strongest[3].bearing_deg
        result.template_range_m = strongest[3].range_m
    result.f_template_hz = template_hz
    if template_hz:
        result.template_signature = propeller_kinematics(
            template_hz, cfg.rotor_blades, cfg.prop_diameter_m)

    # collapse a single pass that straddled an azimuth-bin edge into one detection
    result.detections = _merge_split_cells(result.detections, bin_deg)

    # ---- flag matches to the template frequency ----
    for d in result.detections:
        d.matches_template = bool(
            d.detected and d.blade_hz is not None and template_hz
            and abs(d.blade_hz - template_hz) <= blade_tol * template_hz)

    result.tracks = link_tracks(result.detections, t_rot,
                                gate_deg=cfg.ladder_track_gate_deg)

    # cross-revolution coherent accumulation over the matched passes → a robust single blade f
    if sweep:
        xs = [d._x for d in result.detections
              if d.matches_template and hasattr(d, "_x")]
        if len(xs) >= 2:
            acc = accumulate_comb(xs, sweep, f_lo=f_lo, f_hi=f_hi)
            if acc and acc["blade_hz"]:
                result.accumulated_blade_hz = acc["blade_hz"]
                result.accumulated_comb_strength = acc["comb_strength"]
                result.accumulated_acf = acc["acf"]
    return result


def _merge_split_cells(detections, bin_deg):
    """Merge detections from the *same revolution* whose bearings fall within ~one bin of each
    other — a single target pass that straddled an azimuth-bin edge. Keeps the strongest member's
    comb measurement but reports the event-weighted mean bearing and the summed event count.
    """
    if not detections:
        return detections
    by_rev: dict = {}
    for d in detections:
        by_rev.setdefault(int(d.rev), []).append(d)
    merged = []
    gate = bin_deg * 1.5
    for rev in sorted(by_rev):
        ds = sorted(by_rev[rev], key=lambda d: d.bearing_deg)
        cluster = [ds[0]]
        for d in ds[1:]:
            if abs(_ang_diff(d.bearing_deg, cluster[-1].bearing_deg)) <= gate:
                cluster.append(d)
            else:
                merged.append(_fuse_cluster(cluster))
                cluster = [d]
        merged.append(_fuse_cluster(cluster))
    return merged


def _fuse_cluster(cluster):
    """Fuse a cluster of same-pass detections into one (strongest comb wins the measurement)."""
    if len(cluster) == 1:
        return cluster[0]
    rep = max(cluster, key=lambda d: d.comb_strength)
    w = np.array([max(d.n_events, 1) for d in cluster], float)
    rep.bearing_deg = float(np.average([d.bearing_deg for d in cluster], weights=w))
    rep.n_events = int(sum(d.n_events for d in cluster))
    rep.detected = any(d.detected for d in cluster)
    return rep


# ====================================================================================
# 3) Link detections across revolutions  →  tracks + relative motion
# ====================================================================================
def link_tracks(detections, t_rot_s, *, gate_deg=6.0, init_gate_deg=None, min_passes=2,
                only_matched=True) -> list:
    """Link per-revolution rotor-ladder detections into :class:`RotorTrack`s.

    A simple greedy nearest-neighbour across revolutions: each revolution's detections are
    assigned to the open track whose *predicted* bearing (last bearing + running per-rev offset)
    is nearest, else they seed a new track. Because a moving target's per-revolution offset is
    unknown until two passes are seen, the **first** association of a track uses a generous gate
    (*init_gate_deg*, default ``3·gate_deg``); once the offset is known, a tight predicted gate
    (*gate_deg*) keeps the link selective. A track's bearing-vs-revolution slope is the **azimuth
    offset per revolution**; divided by the rotation period it is the target's relative angular
    rate ``Ω_d`` (deg/s). Only tracks with ``≥ min_passes`` revolutions are returned (recurrence is
    what separates a real object from a one-off transient).
    """
    init_gate_deg = (3.0 * gate_deg) if init_gate_deg is None else init_gate_deg
    pool = [d for d in detections if (d.matches_template if only_matched else d.detected)]
    if not pool:
        return []
    by_rev: dict = {}
    for d in pool:
        by_rev.setdefault(int(d.rev), []).append(d)

    open_tracks: list = []      # each: dict(dets, last_bearing, last_rev, offset)
    for rev in sorted(by_rev):
        cands = sorted(by_rev[rev], key=lambda d: -d.comb_strength)
        used = set()
        for d in cands:
            best_i, best_cost = None, np.inf
            for i, tr in enumerate(open_tracks):
                if i in used:
                    continue
                pred = tr["last_bearing"] + tr["offset"] * (rev - tr["last_rev"])
                gate = gate_deg if len(tr["dets"]) >= 2 else init_gate_deg
                cost = abs(_ang_diff(d.bearing_deg, pred))
                if cost <= gate and cost < best_cost:
                    best_cost, best_i = cost, i
            if best_i is None:
                open_tracks.append(dict(dets=[d], last_bearing=d.bearing_deg,
                                        last_rev=rev, offset=0.0))
                used.add(len(open_tracks) - 1)     # a fresh track can't absorb another det this rev
            else:
                tr = open_tracks[best_i]
                tr["dets"].append(d)
                rr = np.array([x.rev for x in tr["dets"]], float)
                bb = np.rad2deg(np.unwrap(np.deg2rad([x.bearing_deg for x in tr["dets"]])))
                tr["offset"] = float(np.polyfit(rr, bb, 1)[0]) if rr.size >= 2 else 0.0
                tr["last_bearing"], tr["last_rev"] = d.bearing_deg, rev
                used.add(best_i)

    tracks = []
    for tr in open_tracks:
        ds = sorted(tr["dets"], key=lambda d: d.rev)
        if len(ds) < min_passes:
            continue
        f = np.array([d.blade_hz for d in ds if d.blade_hz], float)
        med = float(np.median(f)) if f.size else None
        stab = float(max(0.0, 1.0 - (np.std(f) / med))) if (med and med > 0 and f.size > 1) else 1.0
        rngs = [d.range_m for d in ds if d.range_m is not None and np.isfinite(d.range_m)]
        offset = tr["offset"]
        tracks.append(RotorTrack(
            bearing_deg=round(float(np.median([d.bearing_deg for d in ds])), 2),
            n_passes=len(ds), median_blade_hz=(round(med, 1) if med else None),
            blade_hz_stability=round(stab, 3),
            bearing_offset_per_rev_deg=round(float(offset), 3),
            omega_deg_s=round(float(offset / t_rot_s), 3) if t_rot_s else 0.0,
            range_m=(round(float(np.median(rngs)), 2) if rngs else None),
            revs=[int(d.rev) for d in ds],
            bearings_deg=[round(d.bearing_deg, 2) for d in ds],
            blade_hz=[d.blade_hz for d in ds]))
    tracks.sort(key=lambda t: -t.n_passes)
    return tracks


# ====================================================================================
# Convenience: run the whole survey straight off a RotationContext
# ====================================================================================
def scan_context(ctx, *, roi=None, t0=None, t1=None, template_hz=None) -> RotorScanResult:
    """Run :func:`analyze_box` (if a box/template is given) then :func:`scan_rotation` on a
    :class:`~gottlux.rotation.RotationContext`. The single entry point the CLI/GUI call."""
    cfg, ev, tel = ctx.cfg, ctx.ev, ctx.tel
    # the rotor blade-pass band is independent of the flutter band (rotors can exceed 800 Hz)
    f_lo = getattr(cfg, "ladder_f_lo", cfg.freq_lo)
    f_hi = getattr(cfg, "ladder_f_hi", cfg.freq_hi)
    template = None
    if roi is not None or t0 is not None or t1 is not None:
        template, _ = analyze_box(ev, cfg, tel, roi=roi, t0=t0, t1=t1, f_lo=f_lo, f_hi=f_hi,
                                  min_events=cfg.ladder_min_events)
        if template_hz is None and template is not None and template.detected:
            template_hz = template.blade_hz
    # an ROI also gates the survey itself (e.g. ``0,0,W,Yh`` keeps only above-horizon events so a
    # bright swept ground/clutter band does not dominate every azimuth cell)
    keep = getattr(ctx, "keep", None)
    if roi is not None:
        x = np.asarray(ev["x"]); y = np.asarray(ev["y"])
        x0, y0, x1, y1 = roi
        rmask = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        keep = rmask if keep is None else (np.asarray(keep, bool) & rmask)
    result = scan_rotation(ev, cfg, tel, keep=keep,
                           template_hz=template_hz, f_lo=f_lo, f_hi=f_hi)
    # if the box gave a usable template figure, prefer it over the auto-picked one
    if template is not None and template.detected and roi is not None:
        result.template = template
        result.template_events = _box_xt(ev, roi, t0, t1)
        sig = propeller_kinematics(template.blade_hz, cfg.rotor_blades, cfg.prop_diameter_m)
        result.template_signature = sig
    return result


def detections_table(result) -> dict:
    """Flatten a :class:`RotorScanResult`'s detections into a dict-of-columns for CSV/Parquet."""
    d = result.detections
    return {
        "rev": [x.rev for x in d],
        "t_s": [round(x.t_s, 4) for x in d],
        "bearing_deg": [round(x.bearing_deg, 2) for x in d],
        "range_m": [x.range_m for x in d],
        "elev_deg": [x.elev_deg for x in d],
        "blade_hz": [x.blade_hz for x in d],
        "rotor_hz": [x.rotor_hz for x in d],
        "rpm": [x.rpm for x in d],
        "step_px": [x.step_px for x in d],
        "drift_px_s": [round(x.drift_px_s, 1) for x in d],
        "comb_strength": [x.comb_strength for x in d],
        "n_events": [x.n_events for x in d],
        "size_px": [x.size_px for x in d],
        "detected": [int(x.detected) for x in d],
        "matches_template": [int(x.matches_template) for x in d],
    }


def tracks_table(result) -> dict:
    """Flatten a :class:`RotorScanResult`'s tracks into a dict-of-columns for CSV/Parquet."""
    t = result.tracks
    return {
        "bearing_deg": [x.bearing_deg for x in t],
        "n_passes": [x.n_passes for x in t],
        "median_blade_hz": [x.median_blade_hz for x in t],
        "blade_hz_stability": [x.blade_hz_stability for x in t],
        "bearing_offset_per_rev_deg": [x.bearing_offset_per_rev_deg for x in t],
        "omega_deg_s": [x.omega_deg_s for x in t],
        "range_m": [x.range_m for x in t],
    }


def _box_xt(ev, roi, t0, t1):
    x = np.asarray(ev["x"], float)
    y = np.asarray(ev["y"], float)
    t = np.asarray(ev["t"], float) / 1e6
    m = np.ones(x.shape, bool)
    if t0 is not None:
        m &= t >= t0
    if t1 is not None:
        m &= t < t1
    if roi is not None:
        x0, y0, x1, y1 = roi
        m &= (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    return x[m], t[m]
