"""
track_analysis.py  --  mode-aware analysis of tracker output (pure numpy, no plotting).

Kept deliberately simple per the measurement spec:

  ROTATION : (1) bearing to target, (2) a RELATIVE distance proxy, (3) elevation angle.
  STARING  : (1) a RELATIVE distance proxy, (2) radial velocity (closer/farther),
             (3) blade-flutter FFT inside the tracked box (rotor/prop signature).

Distance is intentionally RELATIVE — a unitless scalar that is monotonic with true
range (apparent size shrinks with distance). A separate calibration step
(scripts/calibrate_range.py) maps the proxy onto metres using the known near/far
distances of a specific flight, so no exact object size is required.
"""
from __future__ import annotations
import numpy as np

REL_K = 1000.0          # scale so the relative-distance proxy reads as friendly numbers


def track_bbox(track, traj=None, tol_s=0.05):
    """(dx, dy) bounding-box width/height for a track: from the track if present, else
    recovered from the detector trajectory by nearest-in-time association (works for any
    tracker without modifying it)."""
    n = len(track["t"])
    dx = np.asarray(track.get("dx", np.full(n, np.nan)), float)
    dy = np.asarray(track.get("dy", np.full(n, np.nan)), float)
    if np.isfinite(dx).any() and np.isfinite(dy).any():
        return dx, dy
    if not traj or "dx" not in traj or not len(traj.get("t", [])):
        return dx, dy
    tt = np.asarray(traj["t"], float); tdx = np.asarray(traj["dx"], float); tdy = np.asarray(traj["dy"], float)
    odx = np.full(n, np.nan); ody = np.full(n, np.nan)
    for i, ti in enumerate(np.asarray(track["t"], float)):
        j = int(np.argmin(np.abs(tt - ti)))
        if abs(tt[j] - ti) <= tol_s:
            odx[i], ody[i] = tdx[j], tdy[j]
    return odx, ody


def apparent_size_px(dx, dy, mode):
    """Apparent target size in pixels. Staring -> bbox diagonal. Rotation -> vertical
    extent dy (rotation-invariant: azimuth motion smears the x extent, not the y)."""
    dx = np.asarray(dx, float); dy = np.asarray(dy, float)
    if mode == "rotation":
        s = np.where(np.isfinite(dy) & (dy > 0), dy, np.hypot(dx, dy))
    else:
        s = np.hypot(dx, dy)
    return s


def relative_distance(size_px):
    """Unitless distance proxy, monotonic with true range: proxy = K / apparent_size.
    (Pinhole: size ∝ 1/range, so proxy ∝ range — a linear two-point calibration recovers
    metres.) Larger proxy = farther away."""
    size = np.asarray(size_px, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = REL_K / size
    r[~np.isfinite(r)] = np.nan
    return r


def radial_velocity(t, rel_distance):
    """d(rel_distance)/dt — positive = moving AWAY, negative = approaching. Units: proxy/s
    (becomes m/s after calibration)."""
    t = np.asarray(t, float); r = np.asarray(rel_distance, float)
    if len(t) < 2:
        return np.zeros(len(t))
    dt = np.gradient(t); dt[np.abs(dt) < 1e-9] = 1e-9
    v = np.gradient(r) / dt
    v[~np.isfinite(v)] = np.nan
    return v


def blade_fft(ev, track, fmin=20.0, fmax=600.0, bin_dt=5e-4, max_box_pad=6):
    """Blade/rotor flutter signature: FFT of the event-rate flicker INSIDE the tracked box.

    The box moves, so events are gated against the box interpolated to their own time.
    Returns dict(freqs, power, peak_hz, snr, n_events) over [fmin, fmax]. peak_hz is NaN
    if there is too little signal."""
    t = np.asarray(track["t"], float)
    if ev is None or len(t) < 3:
        return dict(freqs=np.zeros(0), power=np.zeros(0), peak_hz=np.nan, snr=0.0, n_events=0)
    cx = np.asarray(track.get("cx"), float); cy = np.asarray(track.get("cy"), float)
    dx, dy = track_bbox(track)
    if not (np.isfinite(cx).any() and np.isfinite(cy).any() and np.isfinite(dx).any()):
        return dict(freqs=np.zeros(0), power=np.zeros(0), peak_hz=np.nan, snr=0.0, n_events=0)
    et = np.asarray(ev["t"], float) / 1e6; ex = np.asarray(ev["x"], float); ey = np.asarray(ev["y"], float)
    t0, t1 = float(t[0]), float(t[-1])
    m = (et >= t0) & (et <= t1)
    if m.sum() < 16:
        return dict(freqs=np.zeros(0), power=np.zeros(0), peak_hz=np.nan, snr=0.0, n_events=int(m.sum()))
    et, ex, ey = et[m], ex[m], ey[m]
    # interpolate the box to each event time, then keep events inside it
    icx = np.interp(et, t, cx); icy = np.interp(et, t, cy)
    ihx = np.interp(et, t, np.nan_to_num(dx, nan=0.0)) / 2 + max_box_pad
    ihy = np.interp(et, t, np.nan_to_num(dy, nan=0.0)) / 2 + max_box_pad
    inside = (np.abs(ex - icx) <= ihx) & (np.abs(ey - icy) <= ihy)
    n_in = int(inside.sum())
    if n_in < 32:
        return dict(freqs=np.zeros(0), power=np.zeros(0), peak_hz=np.nan, snr=0.0, n_events=n_in)
    te = et[inside]
    nb = max(64, int((t1 - t0) / bin_dt))
    sig, edges = np.histogram(te, bins=nb, range=(t0, t1))
    sig = sig.astype(float) - sig.mean()
    win = np.hanning(len(sig))
    sp = np.abs(np.fft.rfft(sig * win)) ** 2
    fs = 1.0 / ((t1 - t0) / len(sig))
    fr = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    band = (fr >= fmin) & (fr <= fmax)
    if not band.any():
        return dict(freqs=fr, power=sp, peak_hz=np.nan, snr=0.0, n_events=n_in)
    bp = sp[band]; bf = fr[band]
    k = int(np.argmax(bp))
    med = np.median(bp) + 1e-12
    return dict(freqs=fr, power=sp, peak_hz=float(bf[k]), snr=float(bp[k] / med), n_events=n_in)


def effectiveness_summary(per_track, mode, min_track_s=0.5):
    """Compact figures of merit for the run (no behaviour scoring)."""
    if not per_track:
        return dict(n_tracks=0)
    durs = np.array([p["duration_s"] for p in per_track])
    npts = np.array([p["n_points"] for p in per_track])
    sustained = [p for p in per_track if p["duration_s"] >= min_track_s]
    out = dict(n_tracks=len(per_track), n_sustained=len(sustained),
               longest_track_s=round(float(durs.max()), 2),
               mean_track_s=round(float(durs.mean()), 2),
               total_points=int(npts.sum()))
    rel = np.concatenate([p["rel_distance"][np.isfinite(p["rel_distance"])] for p in sustained]) \
        if sustained else np.array([])
    if rel.size:
        out["rel_distance_min"] = round(float(np.percentile(rel, 5)), 1)
        out["rel_distance_max"] = round(float(np.percentile(rel, 95)), 1)
    if mode != "rotation":
        peaks = [p["blade_hz"] for p in per_track if np.isfinite(p.get("blade_hz", np.nan))]
        out["n_blade_signature"] = len(peaks)
        if peaks:
            out["median_blade_hz"] = round(float(np.median(peaks)), 1)
    return out
