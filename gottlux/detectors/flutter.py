"""
flutter.py — the composable, tunable flutter/flicker detector.

This is the workhorse the whole framework is built around. It pipelines the core stages and
exposes every knob as a tunable :class:`~gottlux.detectors.base.Param`, so you can *build and
tune* a detector for whatever flickers — instead of trusting a black box:

    foreground  →  cluster  →  FFT flutter-verify  →  track  →  localize
    (suppress    (per-step    (events inside each    (NN     (bearing/
     static       blobs)       blob → spectrum;       multi-  elevation/
     clutter)                  keep if an in-band     target  range per
                               peak beats the SNR     link)   detection)
                               gate + harmonic test)

The **verify** stage is what makes this a flutter detector rather than a motion detector: a
blob is only accepted if the events *inside it* carry a periodic temporal signature in the
target band (a rotor's blade-pass tone, a wingbeat) — strong enough over the noise floor, and
(optionally) backed by a harmonic comb. Everything that merely moves is rejected.

A detector is just this class bound to a :class:`~gottlux.detectors.signatures.Signature`.
The registered presets — ``drone``, ``insect``, ``mosquito``, ``hummingbird``, ``bird`` and a
free ``flutter`` (custom band) — differ only in their signature and a couple of defaults.
"""
from __future__ import annotations

import numpy as np

from gottlux import sensors
from gottlux.core import frequency as fq
from gottlux.core import geometry as geo
from gottlux.core.background import staring_foreground_mask
from gottlux.core.detect import cluster_frame
from gottlux.core.filters import hot_pixel_mask
from gottlux.detectors.base import Detector, DetectorResult, Param, Target, register
from gottlux.detectors.signatures import get_signature
from gottlux.detectors.tracking import MultiTracker


class FlutterDetector(Detector):
    """Tunable flutter detector, parameterized by a signature (override ``SIGNATURE``)."""

    name = "flutter"
    description = "Tunable flicker/flutter detector (cluster → FFT-verify → track), custom band."
    regime = "both"
    use_for = "anything with a periodic brightness modulation; set the band to your target."
    SIGNATURE = "custom"

    PARAMS = [
        # --- band (defaults filled from the signature in __init__) ---
        Param("freq_lo", "Freq band low", 80.0, 1.0, 2000.0, 1.0, "float", unit="Hz",
              group="Signature", help="Lower edge of the flutter pass-band to verify against."),
        Param("freq_hi", "Freq band high", 800.0, 2.0, 4000.0, 1.0, "float", unit="Hz",
              group="Signature", help="Upper edge of the flutter pass-band."),
        Param("snr_thresh", "SNR gate", 4.0, 1.0, 50.0, 0.5, "float",
              group="Signature", help="Min in-band spectral peak / noise-floor to accept a blob."),
        Param("harmonic_min", "Harmonic gate", 0.0, 0.0, 1.0, 0.05, "float",
              group="Signature",
              help="Require this fraction of the fundamental's overtones (rotors show a comb; 0 = off)."),
        # --- FFT ---
        Param("fft_fs", "FFT sample rate", 2000.0, 200.0, 8000.0, 100.0, "float", unit="Hz",
              group="FFT", help="Rate the in-blob event stream is binned to (must exceed 2×band-high)."),
        Param("fft_window_s", "FFT window", 0.30, 0.05, 2.0, 0.01, "float", unit="s",
              group="FFT", help="Trailing window of events fed to each blob's FFT."),
        Param("fft_min_events", "FFT min events", 40, 8, 2000, 1, "int",
              group="FFT", help="Minimum events inside a blob's window to attempt verification."),
        # --- clustering ---
        Param("accum_dt", "Step / cluster dt", 0.02, 0.002, 0.2, 0.001, "float", unit="s",
              group="Cluster", help="Detector step size and the per-step blob accumulation window."),
        Param("min_pixels", "Min blob area", 40, 4, 2000, 1, "int", unit="px",
              group="Cluster", help="Minimum connected-component area to consider a blob."),
        Param("dilation", "Dilate", 2, 0, 6, 1, "int", group="Cluster",
              help="Morphological dilation bridging blob gaps before labeling."),
        Param("erode", "Erode", 1, 0, 6, 1, "int", group="Cluster",
              help="Morphological erosion trimming spurs after dilation."),
        Param("pos_only", "Positive events only", True, kind="bool", group="Cluster",
              help="Cluster on ON events only (asymmetric gating, ~2× faster, often cleaner)."),
        Param("suppress_background", "Suppress static bg", True, kind="bool", group="Cluster",
              help="Remove persistent-pixel background before clustering (staring scenes)."),
        # --- tracking ---
        Param("max_tracks", "Max tracks", 6, 1, 32, 1, "int", group="Track",
              help="Maximum simultaneous targets."),
        Param("max_match_dist", "Match gate", 60.0, 5.0, 400.0, 1.0, "float", unit="px",
              group="Track", help="Max centroid jump to associate a detection to a track."),
        Param("max_missed", "Coast frames", 8, 0, 60, 1, "int", group="Track",
              help="Steps a track may coast (predict only) before being dropped."),
        Param("smooth", "Track smoothing", 0.4, 0.0, 0.95, 0.05, "float", group="Track",
              help="Position/box smoothing factor (0 = raw, →1 = heavy)."),
        Param("calibration_s", "Warm-up", 0.05, 0.0, 2.0, 0.01, "float", unit="s",
              group="Track", help="Lead-in skipped before detection starts (lets buffers fill)."),
    ]

    def __init__(self, **overrides):
        super().__init__(**overrides)
        sig = get_signature(self.SIGNATURE)
        # Seed band/SNR defaults from the signature unless the caller overrode them.
        for key, val in (("freq_lo", sig.freq_lo), ("freq_hi", sig.freq_hi),
                         ("snr_thresh", sig.default_snr)):
            if key not in overrides:
                self.params[key] = val
        if "fft_fs" not in overrides:
            self.params["fft_fs"] = max(self.params["fft_fs"], sig.nyquist_fs())
        if "harmonic_min" not in overrides and sig.expect_harmonics:
            self.params["harmonic_min"] = 0.34
        self.signature = sig

    # ------------------------------------------------------------------ run
    def run(self, rec, cfg=None, t0=None, t1=None, progress=None) -> DetectorResult:
        P = self.params
        win = rec.window(t0, t1)
        W, H = win.width, win.height
        x = np.asarray(win.x); y = np.asarray(win.y); p = np.asarray(win.p)
        ts = win.t_s
        n = len(ts)
        if n < 16:
            return DetectorResult([], self.name, dict(P), self.regime,
                                  self.signature.name, {"note": "too few events"})

        # --- foreground mask (for the clustering stream only) ---
        keep = hot_pixel_mask(win, 99.97)
        if P["suppress_background"] and not rec.is_rotating:
            keep &= staring_foreground_mask(win, bg_window_s=min(1.0, win.duration_s * 0.3))
        det_sel = keep & (p == 1) if P["pos_only"] else keep
        dxi = np.where(det_sel)[0]
        dt_det = ts[dxi]                      # detection-stream times (sorted)

        # --- stepping ---
        t_lo = float(ts[0]) + P["calibration_s"]
        t_hi = float(ts[-1])
        steps = np.arange(t_lo + P["accum_dt"], t_hi, P["accum_dt"])
        if steps.size == 0:
            return DetectorResult([], self.name, dict(P), self.regime,
                                  self.signature.name, {"note": "window too short"})
        fs = max(P["fft_fs"], 2.2 * P["freq_hi"])
        tracker = MultiTracker(P["max_match_dist"], P["max_missed"], P["max_tracks"], P["smooth"])
        edges_factory = lambda ct: None      # placeholder (region_spectrum bins internally)
        n_cand = n_verified = 0

        for si, ct in enumerate(steps):
            # cluster the foreground events in [ct-accum_dt, ct]
            lo = np.searchsorted(dt_det, ct - P["accum_dt"])
            hi = np.searchsorted(dt_det, ct)
            if hi - lo < P["min_pixels"]:
                tracker.update(ct, [])
                continue
            idx = dxi[lo:hi]
            blobs = cluster_frame(x[idx], y[idx], W, H, P["min_pixels"], P["dilation"], P["erode"])
            n_cand += len(blobs)
            # FFT-verify each blob on the full-stream trailing window inside its bbox
            verified = []
            if blobs:
                a0 = np.searchsorted(ts, ct - P["fft_window_s"])
                a1 = np.searchsorted(ts, ct)
                wx = x[a0:a1]; wy = y[a0:a1]; wt = win.t[a0:a1]
                for (cx, cy, x0, y0, x1, y1, area) in blobs:
                    inb = (wx >= x0) & (wx < x1) & (wy >= y0) & (wy < y1)
                    vt = wt[inb]
                    if vt.size <= P["fft_min_events"]:
                        continue
                    sp = fq.region_spectrum(vt, fs=fs, fmin=P["freq_lo"], fmax=P["freq_hi"])
                    if (sp.detected and sp.snr >= P["snr_thresh"]
                            and sp.harmonic_score >= P["harmonic_min"]):
                        verified.append(dict(cx=cx, cy=cy, bbox=(x0, y0, x1, y1),
                                             freq=sp.peak_freq, snr=sp.snr,
                                             harmonic=sp.harmonic_score))
            n_verified += len(verified)
            tracker.update(ct, verified)
            if progress and (si % 16 == 0):
                try:
                    progress(si / len(steps))
                except Exception:
                    pass

        targets = self._build_targets(tracker, rec, cfg)
        diag = dict(n_steps=int(steps.size), n_candidates=int(n_cand),
                    n_verified=int(n_verified), fs=float(fs),
                    foreground_frac=float(det_sel.mean()))
        return DetectorResult(targets, self.name, dict(P), self.regime,
                              self.signature.name, diag)

    # ------------------------------------------------------------------ assemble
    def _build_targets(self, tracker, rec, cfg) -> list:
        fov = (cfg.resolved_fov() if cfg is not None else None) or sensors.DEFAULT_FOV_DEG
        az_sign = cfg.az_sign if cfg is not None else -1.0
        target_size = cfg.target_size_m if cfg is not None else 0.22
        W, H = rec.width, rec.height
        out = []
        for tid, rec_list in tracker.tracks(min_len=2).items():
            arr = rec_list
            t = np.array([r["t"] for r in arr])
            cx = np.array([r["cx"] for r in arr])
            cy = np.array([r["cy"] for r in arr])
            bbox = np.array([r["bbox"] for r in arr], float)
            freq = np.array([r["freq"] for r in arr])
            snr = np.array([r["snr"] for r in arr])
            harm = np.array([r["harmonic"] for r in arr])
            diag = np.hypot(bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1])
            az = geo.world_azimuth(cx, t, rec.telemetry, fov, W, az_sign)
            elev = geo.pixel_to_elevation(cy, fov, W, H)
            rng = geo.estimate_range_m(diag, fov, target_size, W)
            rel = geo.relative_distance_proxy(diag)
            out.append(Target(id=int(tid), t=t, cx=cx, cy=cy, bbox=bbox,
                              freq_hz=freq, snr=snr, harmonic=harm,
                              azimuth_deg=az, elev_deg=elev, range_m=rng, rel_distance=rel))
        return out


# ====================================================================================
# Registered presets — each is the FlutterDetector bound to a signature
# ====================================================================================
@register
class DroneDetector(FlutterDetector):
    name = "drone"
    description = "Rotor-flutter drone detector (multirotor blade-pass band, harmonic-gated)."
    regime = "both"
    use_for = "multirotor drones, by their 80–800 Hz rotor blade-pass signature + harmonics."
    SIGNATURE = "drone"


@register
class InsectDetector(FlutterDetector):
    name = "insect"
    description = "Insect-wingbeat detector (30–250 Hz)."
    regime = "both"
    use_for = "flying insects (flies, bees, butterflies) by their wingbeat tone."
    SIGNATURE = "insect"


@register
class MosquitoDetector(FlutterDetector):
    name = "mosquito"
    description = "High-wingbeat detector for small dipterans (300–800 Hz)."
    regime = "both"
    use_for = "mosquitoes/midges by their high wingbeat tone."
    SIGNATURE = "mosquito"


@register
class HummingbirdDetector(FlutterDetector):
    name = "hummingbird"
    description = "Hovering-wingbeat detector (15–120 Hz)."
    regime = "both"
    use_for = "hummingbirds by their hovering wingbeat."
    SIGNATURE = "hummingbird"


@register
class BirdDetector(FlutterDetector):
    name = "bird"
    description = "Flapping-flight detector for larger birds (3–30 Hz)."
    regime = "both"
    use_for = "larger birds by slow flapping flight."
    SIGNATURE = "bird"


@register
class CustomFlutterDetector(FlutterDetector):
    name = "flutter"
    description = "Free-band flutter detector — dial in any periodic flicker."
    regime = "both"
    use_for = "anything periodic; set the frequency band yourself."
    SIGNATURE = "custom"


@register
class BlobTrackerDetector(FlutterDetector):
    """Greedy nearest-neighbour tracker over per-frame connected-component blobs — **no spectral
    gate**. The flutter detector drops the drone whenever its rotor tone is weak/noisy (the
    verify gate); this tracks *every* blob ≥ ``min_pixels`` and links it greedily, so the drone's
    track stays continuous. Each track is still annotated with its in-band FFT frequency/SNR
    (computed but **not** gated on), so the prop-frequency metric still has data. Mirrors the
    sandbox ``track(ev, state)`` (MultiTracker over ``ev.blobs(min_pixels=40)``) that tracks well
    in the field."""
    name = "blob_tracker"
    description = "Greedy-NN blob tracker (connected components, no FFT gate) — robust drone tracking."
    regime = "both"
    use_for = "tracking one drone by its event blob when the rotor tone is too weak/noisy to gate on."
    SIGNATURE = "drone"

    def __init__(self, **overrides):
        super().__init__(**overrides)
        # the sandbox settings that track well in the field, unless the caller overrode them
        for key, val in (("min_pixels", 40), ("max_match_dist", 40.0),
                         ("max_missed", 6), ("max_tracks", 8)):
            if key not in overrides and key in self.params:
                self.params[key] = val

    def run(self, rec, cfg=None, t0=None, t1=None, progress=None) -> DetectorResult:
        P = self.params
        win = rec.window(t0, t1)
        W, H = win.width, win.height
        x = np.asarray(win.x); y = np.asarray(win.y); p = np.asarray(win.p)
        ts = win.t_s
        if len(ts) < 16:
            return DetectorResult([], self.name, dict(P), self.regime, self.signature.name,
                                  {"note": "too few events"})
        keep = hot_pixel_mask(win, 99.97)
        if P["suppress_background"] and not rec.is_rotating:
            keep &= staring_foreground_mask(win, bg_window_s=min(1.0, win.duration_s * 0.3))
        det_sel = keep & (p == 1) if P["pos_only"] else keep
        dxi = np.where(det_sel)[0]
        dt_det = ts[dxi]
        t_lo = float(ts[0]) + P["calibration_s"]; t_hi = float(ts[-1])
        steps = np.arange(t_lo + P["accum_dt"], t_hi, P["accum_dt"])
        if steps.size == 0:
            return DetectorResult([], self.name, dict(P), self.regime, self.signature.name,
                                  {"note": "window too short"})
        fs = max(P["fft_fs"], 2.2 * P["freq_hi"])
        tracker = MultiTracker(P["max_match_dist"], P["max_missed"], P["max_tracks"], P["smooth"])
        n_cand = 0
        for si, ct in enumerate(steps):
            lo = np.searchsorted(dt_det, ct - P["accum_dt"]); hi = np.searchsorted(dt_det, ct)
            if hi - lo < P["min_pixels"]:
                tracker.update(ct, []); continue
            idx = dxi[lo:hi]
            blobs = cluster_frame(x[idx], y[idx], W, H, P["min_pixels"], P["dilation"], P["erode"])
            n_cand += len(blobs)
            cands = []
            if blobs:                                    # annotate FFT (ungated), then track EVERY blob
                a0 = np.searchsorted(ts, ct - P["fft_window_s"]); a1 = np.searchsorted(ts, ct)
                wx = x[a0:a1]; wy = y[a0:a1]; wt = win.t[a0:a1]
                for (cx, cy, x0, y0, x1, y1, area) in blobs:
                    freq = snr = harm = 0.0
                    inb = (wx >= x0) & (wx < x1) & (wy >= y0) & (wy < y1)
                    vt = wt[inb]
                    if vt.size > P["fft_min_events"]:
                        sp = fq.region_spectrum(vt, fs=fs, fmin=P["freq_lo"], fmax=P["freq_hi"])
                        freq, snr, harm = float(sp.peak_freq), float(sp.snr), float(sp.harmonic_score)
                    cands.append(dict(cx=cx, cy=cy, bbox=(x0, y0, x1, y1),
                                      freq=freq, snr=snr, harmonic=harm))
            tracker.update(ct, cands)
            if progress and (si % 16 == 0):
                try:
                    progress(si / len(steps))
                except Exception:
                    pass
        targets = self._build_targets(tracker, rec, cfg)
        diag = dict(n_steps=int(steps.size), n_candidates=int(n_cand), gated=False,
                    fs=float(fs), foreground_frac=float(det_sel.mean()))
        return DetectorResult(targets, self.name, dict(P), self.regime, self.signature.name, diag)


@register
class SingleCentroidDetector(FlutterDetector):
    """Single-target smoothed-centroid tracker — the sandbox **"Single smoothed centroid"** preset,
    1:1. Each frame, take the *strongest* connected-component blob, EMA-smooth its centroid, and
    follow it as **one** track. It drops only when there is no blob — i.e. the target left the FOV,
    or its **relative velocity went to zero** (an event sensor responds only to brightness *change*,
    so a momentarily still object — e.g. at the apex of a throttle-cut ballistic climb — produces no
    events and the track is briefly lost). Range comes from the strongest blob's bbox; the in-band
    FFT SNR is annotated (ungated) so the prop-frequency metric still has data."""
    name = "single_centroid"
    description = "Single-target smoothed-centroid tracker (strongest blob + EMA) — the sandbox preset."
    regime = "both"
    use_for = "one dominant drone on a quiet background; the stickiest, simplest single-target tracker."
    SIGNATURE = "drone"
    SMOOTH_A = 0.5            # EMA factor (sandbox preset: 0 = raw, →1 = sticky)
    FFT_EVERY_S = 0.5         # sparse FFT cadence: "we're tracking — now test for an FFT response"

    def __init__(self, **overrides):
        super().__init__(**overrides)
        for key, val in (("min_pixels", 30), ("dilation", 2), ("erode", 1)):
            if key not in overrides and key in self.params:
                self.params[key] = val

    def run(self, rec, cfg=None, t0=None, t1=None, progress=None) -> DetectorResult:
        P = self.params
        win = rec.window(t0, t1)
        W, H = win.width, win.height
        # Faithful to the sandbox preset: cluster the RAW windowed events — ALL polarity, no
        # hot-pixel / background mask — exactly what ``ev.blobs(min_pixels, dilation, erode)`` does.
        x = np.asarray(win.x); y = np.asarray(win.y)
        ts = win.t_s
        if len(ts) < 16:
            return DetectorResult([], self.name, dict(P), self.regime, self.signature.name,
                                  {"note": "too few events"})
        t_lo = float(ts[0]) + P["calibration_s"]; t_hi = float(ts[-1])
        steps = np.arange(t_lo + P["accum_dt"], t_hi, P["accum_dt"])
        if steps.size == 0:
            return DetectorResult([], self.name, dict(P), self.regime, self.signature.name,
                                  {"note": "window too short"})
        fs = max(P["fft_fs"], 2.2 * P["freq_hi"])
        rows, pos, n_lost, last_fft, n_fft = [], None, 0, -1e9, 0
        a = self.SMOOTH_A
        for ct in steps:
            lo = np.searchsorted(ts, ct - P["accum_dt"]); hi = np.searchsorted(ts, ct)
            blobs = (cluster_frame(x[lo:hi], y[lo:hi], W, H, P["min_pixels"], P["dilation"], P["erode"])
                     if hi - lo >= P["min_pixels"] else [])
            if not blobs:
                n_lost += 1; continue                    # no blob → track dropped (out-of-FOV / v→0)
            cx, cy, x0, y0, x1, y1, area = max(blobs, key=lambda b: b[6])   # strongest by area
            if pos is None:
                pos = (cx, cy)
            cx = a * pos[0] + (1 - a) * cx; cy = a * pos[1] + (1 - a) * cy   # EMA-smoothed centroid
            pos = (cx, cy)
            freq = snr = harm = 0.0                       # 0 = "not sampled this frame"
            if ct - last_fft >= self.FFT_EVERY_S:         # sparse: probe the rotor tone periodically
                a0 = np.searchsorted(ts, ct - P["fft_window_s"])
                fx = x[a0:hi]; fy = y[a0:hi]
                inb = (fx >= x0) & (fx < x1) & (fy >= y0) & (fy < y1)
                vt = win.t[a0:hi][inb]
                if vt.size > P["fft_min_events"]:
                    sp = fq.region_spectrum(vt, fs=fs, fmin=P["freq_lo"], fmax=P["freq_hi"])
                    freq, snr, harm = float(sp.peak_freq), float(sp.snr), float(sp.harmonic_score)
                    last_fft = ct; n_fft += 1
            rows.append((ct, cx, cy, (x0, y0, x1, y1), freq, snr, harm))
            if progress:
                try:
                    progress(len(rows) / max(steps.size, 1))
                except Exception:
                    pass
        targets = []
        if rows:
            fov = (cfg.resolved_fov() if cfg is not None else None) or sensors.DEFAULT_FOV_DEG
            az_sign = cfg.az_sign if cfg is not None else -1.0
            tgt_sz = cfg.target_size_m if cfg is not None else 0.22
            t = np.array([r[0] for r in rows]); cx = np.array([r[1] for r in rows])
            cy = np.array([r[2] for r in rows]); bbox = np.array([r[3] for r in rows], float)
            freq = np.array([r[4] for r in rows]); snr = np.array([r[5] for r in rows])
            harm = np.array([r[6] for r in rows])
            diagpx = np.hypot(bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1])
            targets = [Target(id=0, t=t, cx=cx, cy=cy, bbox=bbox, freq_hz=freq, snr=snr, harmonic=harm,
                              azimuth_deg=geo.world_azimuth(cx, t, rec.telemetry, fov, W, az_sign),
                              elev_deg=geo.pixel_to_elevation(cy, fov, W, H),
                              range_m=geo.estimate_range_m(diagpx, fov, tgt_sz, W),
                              rel_distance=geo.relative_distance_proxy(diagpx))]
        diag = dict(n_steps=int(steps.size), n_detected=len(rows), n_lost=int(n_lost),
                    lost_frac=round(n_lost / max(steps.size, 1), 3), accum_dt=float(P["accum_dt"]),
                    n_fft=int(n_fft), fft_every_s=self.FFT_EVERY_S, gated=False, fs=float(fs))
        return DetectorResult(targets, self.name, dict(P), self.regime, self.signature.name, diag)
