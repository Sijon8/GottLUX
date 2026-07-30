"""
telemetry.py — rotation ground truth for a rotating EBS payload.

A rotating-payload capture logs, alongside the event ``.raw``, a CSV of payload azimuth
plus a hardware **Hall-sync** pulse fired once per revolution. Using this *ground truth*
(rather than estimating the rotation period by autocorrelation) gives exact per-revolution
boundaries, a true ``azimuth(t)``, and a phase that absorbs any rotation-speed variation —
the foundation for de-rotating the spinning sensor into a stabilized world frame.

CSV format (header row ``System_Time,Azimuth,Revolution,Flag``)::

    20260517_185333_220, 137.4, 12, HALL_SYNC
    └ date_HHMMSS_millis  └ deg   └ rev  └ event flag(s); HALL_SYNC marks a rev boundary
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np

from gottlux.io.paths import ext


def _parse_systime(s: str) -> float:
    """Parse a ``date_HHMMSS_millis`` system-time stamp into seconds-of-day (float)."""
    _date, hms, ms = s.strip().split("_")
    return int(hms[0:2]) * 3600 + int(hms[2:4]) * 60 + int(hms[4:6]) + int(ms) / 1000.0


def looks_like_telemetry(csv_path: str) -> bool:
    """Cheap sniff: does this CSV carry a ``System_Time`` rotation-telemetry header?"""
    try:
        with open(ext(csv_path)) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                return line.split(",")[0].strip().lower().startswith("system_time")
    except Exception:
        return False
    return False


def find_telemetry_csv(near_path: str) -> Optional[str]:
    """Locate a telemetry CSV next to a ``.raw``/cache stem (same dir, then parent)."""
    base_dir = os.path.dirname(os.path.abspath(near_path))
    for d in (base_dir, os.path.dirname(base_dir)):
        if not d or not os.path.isdir(ext(d)) and not os.path.isdir(d):
            continue
        for csv in sorted(glob.glob(os.path.join(d, "*.csv"))):
            if looks_like_telemetry(csv):
                return csv
    return None


def estimate_spin_period_s(t_events_s, *, bin_s: float = 0.005, lo_s: float = 0.2,
                           hi_s: float = 5.0):
    """Estimate a spinning payload's rotation period (s) from the **event-rate periodicity**.

    A spinning sensor re-images the same scene every revolution, so its event-rate is periodic at
    the rotation period. We bin the rate, take its autocorrelation, pick the first strong peak in
    ``[lo_s, hi_s]`` as a coarse period, then **refine** it by detecting the per-revolution rate
    peaks (the passes) and least-squares-fitting their times against integer revolution index — a
    sub-percent period that does not accumulate phase error across the clip.

    Returns ``(period_s, confidence)`` with confidence the autocorrelation peak height (0..1), or
    ``(None, 0.0)`` if no clear periodicity is found.
    """
    t = np.asarray(t_events_s, np.float64)
    if t.size < 100 or (t[-1] - t[0]) < 2 * lo_s:
        return None, 0.0
    dur = float(t[-1] - t[0])
    edges = np.arange(0.0, dur + bin_s, bin_s)
    rate = np.histogram(t - t[0], bins=edges)[0].astype(np.float64)
    k = max(int(round(0.01 / bin_s)), 1)                       # ~10 ms smoothing
    rate = np.convolve(rate, np.ones(k) / k, mode="same")
    rc = rate - rate.mean()
    acf = np.correlate(rc, rc, "full")[rate.size - 1:]
    if acf[0] <= 0:
        return None, 0.0
    acf = acf / acf[0]
    lags = np.arange(acf.size) * bin_s
    win = (lags >= lo_s) & (lags <= min(hi_s, dur / 2))
    if not win.any():
        return None, 0.0
    seed = lags[np.flatnonzero(win)[0] + int(np.argmax(acf[win]))]
    conf = float(acf[np.flatnonzero(win)[0] + int(np.argmax(acf[win]))])
    # refine via the per-revolution rate peaks (passes): find local maxima > 60th pct, ~1 per period
    thr = np.percentile(rate, 90)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peaks = []
    guard = max(int(round(0.4 * seed / bin_s)), 1)
    i = 0
    while i < rate.size:
        if rate[i] >= thr:
            j0, j1 = max(0, i - guard), min(rate.size, i + guard)
            pk = j0 + int(np.argmax(rate[j0:j1]))
            if not peaks or (pk - peaks[-1]) > guard:
                peaks.append(pk)
            i = pk + guard
        else:
            i += 1
    period = seed
    if len(peaks) >= 3:
        tp = centers[np.array(peaks)]
        idx = np.round((tp - tp[0]) / seed).astype(int)        # assign each peak a rev index
        if len(set(idx)) >= 3:
            slope = np.polyfit(idx, tp, 1)[0]
            if lo_s <= slope <= hi_s:
                period = float(slope)
    return round(period, 5), round(conf, 3)


class Telemetry:
    """Rotation telemetry parsed from a capture CSV (Hall-sync boundaries + azimuth).

    The single most important methods are :meth:`azimuth_at` (true world azimuth of the
    sensor boresight at event time ``t``) and :meth:`phase_at` (fractional position within
    the current revolution, ``0..1``) — used by de-rotation and the rotation-phase filter.
    """

    def __init__(self, csv_path: str):
        rows = []
        with open(ext(csv_path)) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if parts[0].strip().lower().startswith("system_time"):
                    continue
                rows.append(parts)
        if not rows:
            raise ValueError(f"No telemetry rows parsed from {csv_path!r}")
        st = np.array([_parse_systime(r[0]) for r in rows])
        self.csv_path = os.path.abspath(csv_path)
        self.t = st - st[0]
        self.azimuth = np.array([float(r[1]) for r in rows])
        self.rev = np.array([int(r[2]) for r in rows])
        self.flag = np.array([r[3] if len(r) > 3 else "" for r in rows])
        self.hall_t = self.t[np.char.find(self.flag.astype(str), "HALL_SYNC") >= 0]
        self.offset = 0.0           # event-vs-telemetry time alignment (refined later)
        self.synthesized = False    # True iff built from an *estimated* spin (no logged azimuth)

    @classmethod
    def from_spin(cls, duration_s: float, period_s: float, *, az0_deg: float = 0.0,
                  dt: float = 0.01) -> "Telemetry":
        """Build *estimated* telemetry for a steadily-spinning payload that logged no azimuth CSV.

        Models a constant-rate sweep: ``azimuth(t) = (t / period) · 360 + az0`` with a Hall-sync at
        every revolution boundary. The recovered **bearings are rotation-phase-relative** — the
        absolute North offset (*az0*) is unknown unless calibrated — but the rotation period, the
        per-revolution structure (recurrence) and the relative motion offset are faithful as long as
        *period_s* is accurate. Mark such results as estimated; see :func:`estimate_spin_period_s`.
        """
        self = cls.__new__(cls)
        t = np.arange(0.0, float(duration_s) + dt, dt)
        self.csv_path = ""
        self.t = t
        self.azimuth = np.mod(t / period_s * 360.0 + az0_deg, 360.0)
        self.rev = np.floor(t / period_s).astype(int)
        n_rev = int(np.floor(duration_s / period_s)) + 1
        self.hall_t = np.arange(0, n_rev + 1) * period_s
        self.flag = np.where(np.isin(t, self.hall_t), "HALL_SYNC", "")
        self.offset = 0.0
        self.synthesized = True
        return self

    # ------------------------------------------------------------------ revolution facts
    @property
    def n_revolutions(self) -> int:
        return int(self.rev.max()) if len(self.rev) else 0

    @property
    def T_rot(self) -> float:
        """Rotation period (s) — median Hall-sync interval, robust to dropped pulses."""
        if len(self.hall_t) >= 2:
            return float(np.median(np.diff(self.hall_t)))
        return float(self.t[-1] / max(self.n_revolutions, 1)) if len(self.t) else 0.0

    @property
    def omega_deg_s(self) -> float:
        T = self.T_rot
        return 360.0 / T if T > 0 else 0.0

    # ------------------------------------------------------------------ queries at event time
    def rotation_bounds(self) -> np.ndarray:
        """Revolution-boundary times (s, offset-corrected)."""
        return self.hall_t + self.offset

    def phase_at(self, te) -> np.ndarray:
        """Fractional phase within the current revolution (``0..1``) at event time(s)."""
        b = self.rotation_bounds()
        te = np.asarray(te, np.float64)
        idx = np.clip(np.searchsorted(b, te, "right") - 1, 0, len(b) - 2)
        frac = (te - b[idx]) / np.maximum(b[idx + 1] - b[idx], 1e-9)
        return np.mod(frac, 1.0)

    def revolution_at(self, te) -> np.ndarray:
        """Integer revolution index covering each event time."""
        b = self.rotation_bounds()
        return np.clip(np.searchsorted(b, np.asarray(te), "right") - 1, 0, len(b) - 1)

    def azimuth_unwrapped(self) -> np.ndarray:
        """Azimuth as a monotonic unwrapped angle (radians) for interpolation."""
        return np.unwrap(np.deg2rad(self.azimuth))

    def azimuth_at(self, te) -> np.ndarray:
        """True world azimuth of the boresight (degrees, 0..360) at event time(s)."""
        tc = np.asarray(te) - self.offset
        return np.mod(np.rad2deg(np.interp(tc, self.t, self.azimuth_unwrapped())), 360.0)

    # ------------------------------------------------------------------ alignment
    def refine_offset_to_events(self, t_events_s, bin_s: float = 0.005,
                                search_s: float = 0.4) -> float:
        """Refine the telemetry↔event time offset by aligning Hall pulses to event-rate
        structure (the rate dips/peaks as the FOV sweeps known features). Returns offset."""
        t_events_s = np.asarray(t_events_s, np.float64)
        if not len(t_events_s):
            return self.offset
        edges = np.arange(0, float(t_events_s[-1]) + bin_s, bin_s)
        rate = np.histogram(t_events_s, bins=edges)[0].astype(np.float64)
        rate -= rate.mean()
        centers = 0.5 * (edges[:-1] + edges[1:])
        best, best_lag = -np.inf, 0.0
        for lag in np.arange(-search_s, search_s, bin_s):
            grid = self.hall_t + lag
            grid = grid[(grid > centers[0]) & (grid < centers[-1])]
            if len(grid) < 2:
                continue
            score = np.interp(grid, centers, rate).sum()
            if score > best:
                best, best_lag = score, lag
        self.offset = float(best_lag)
        return self.offset
