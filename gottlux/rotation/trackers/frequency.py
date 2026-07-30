"""
trackers/frequency.py  --  Frequency-verified tracker (converged port).

Unifies three MATLAB trackers into one configurable algorithm:
  * Hummingbird V37 "Edge-Compute"  (time-surface + Monte-Carlo noise floor +
    verify-first per-bbox FFT wingbeat gating + multi-track NN),
  * Gemini Quad dual-clock density+FFT classifier,
  * Gemini V3 drone tracker (ROI gating + velocity + coasting state machine).

Pipeline (per accumulation step):
  1. detection on POSITIVE events only  (asymmetric compute gating, ~2x speedup),
  2. exponential time-surface (SAE) decay -> adaptive-threshold clustering
     (morphological close + connected components, min-area gate),
  3. per-candidate "verify-first" FFT: bin the events inside the bbox over a
     trailing window at `fs`, take the spectral peak in [freq_lo, freq_hi];
     accept only if the peak exceeds an SNR threshold (this is the wingbeat /
     rotor signature that rejects everything that is merely moving),
  4. nearest-neighbour multi-track association with velocity prediction,
     bbox smoothing, and missed-frame coasting.

Profiles (registered, selectable):  `hummingbird` (10-120 Hz), `drone_fft` (80-800 Hz).
Best in STARING mode, where a target dwells in-FOV long enough to fill the FFT
window; runs on rotating data too but may not lock (targets only dwell briefly).

Optimized for compute: pure numpy + scipy.ndimage, pointer-advanced trailing
window, np.fft.rfft, vectorized decay/threshold. Reports per-frame logic timing.
"""
from __future__ import annotations
import time
import numpy as np
from scipy import ndimage
from gottlux.rotation.trackers import register
from gottlux.rotation.trackers.base import Tracker


def _mc_noise_mask(x, y, t, H, W, n_chunks=20, chunk_len=0.05, thresh=2, seed=42):
    """Monte-Carlo static-pixel floor: pixels lit in >= `thresh` of K random
    short slices are persistent background -> masked."""
    rng = np.random.default_rng(seed)
    nmap = np.zeros((H, W), np.int32)
    tmax = float(t[-1]) - chunk_len if len(t) else 0.0
    if tmax <= 0:
        return np.zeros((H, W), bool)
    for s in rng.random(n_chunks) * tmax:
        lo = np.searchsorted(t, s); hi = np.searchsorted(t, s + chunk_len)
        if hi > lo:
            m = np.zeros((H, W), bool); m[y[lo:hi], x[lo:hi]] = True
            nmap += m
    return nmap >= thresh


class FreqVerifiedTracker(Tracker):
    uses_events = True
    name = "freq_verified"
    description = "Time-surface clustering + per-target FFT wingbeat/rotor verification + NN tracking."
    # profile defaults (overridden per registered subclass)
    params = dict(freq_lo=80.0, freq_hi=800.0, accum_dt=0.01, tau=0.05,
                  calibration_s=0.03, min_cluster_size=20, fs=1000.0, ac_window=0.30,
                  ac_min_events=40, snr_thresh=0.10, max_tracks=4, max_match_dist=80.0,
                  max_missed=10, smooth=0.4, close_radius=3, pos_only=True)

    def track(self, traj, cfg, tel=None, ev=None):
        if ev is None:
            return {"tracks": []}
        P = self.params
        W, H = ev["width"], ev["height"]
        x = np.asarray(ev["x"]); y = np.asarray(ev["y"]); p = np.asarray(ev["p"])
        t = np.asarray(ev["t"]) / 1e6
        # detection stream = positive polarity (asymmetric gating)
        if P["pos_only"]:
            dm = p == 1
            dx, dy, dt = x[dm].astype(np.int64), y[dm].astype(np.int64), t[dm]
        else:
            dx, dy, dt = x.astype(np.int64), y.astype(np.int64), t
        if len(dt) < 2:
            return {"tracks": []}

        noise = _mc_noise_mask(dx, dy, dt, H, W, thresh=2)
        se = ndimage.generate_binary_structure(2, 2)
        struct_close = np.ones((2 * P["close_radius"] + 1,) * 2, bool)

        sae = np.full((H, W), -np.inf, np.float64)
        records = {}            # track id -> list of (t, cx, cy, freq)
        active = []             # dict(id, cx, cy, vx, vy, bbox, missed, freq)
        next_id = 0

        steps = np.arange(P["calibration_s"] + P["accum_dt"], float(t[-1]), P["accum_dt"])
        i_det = int(np.searchsorted(dt, P["calibration_s"]))        # detection-stream head
        i_all = int(np.searchsorted(t, P["calibration_s"]))         # all-event head
        i_tail = i_all                                              # trailing-window tail
        logic_t = 0.0

        for ct in steps:
            t0 = time.perf_counter()
            j_det = int(np.searchsorted(dt, ct))
            j_all = int(np.searchsorted(t, ct))
            # 1) update time surface with this step's detection events
            if j_det > i_det:
                sae[dy[i_det:j_det], dx[i_det:j_det]] = dt[i_det:j_det]
            # 2) decay + adaptive threshold + morphological clustering
            decay = np.exp((sae - ct) / P["tau"])
            decay[noise] = 0.0
            vp = decay[decay > 0.1]
            thr = (vp.mean() + vp.std()) if vp.size else 0.5
            binary = ndimage.binary_closing(decay > thr, structure=struct_close)
            lbl, n = ndimage.label(binary, structure=se)
            cands = []
            if n:
                areas = np.bincount(lbl.ravel())
                objs = ndimage.find_objects(lbl)
                for li in range(1, n + 1):
                    if areas[li] < P["min_cluster_size"]:
                        continue
                    sy, sx = objs[li - 1]
                    bbox = (sx.start, sy.start, sx.stop - sx.start, sy.stop - sy.start)
                    cen = (0.5 * (sx.start + sx.stop), 0.5 * (sy.start + sy.stop))
                    cands.append((areas[li], bbox, cen))
                cands.sort(key=lambda c: -c[0])
            # 3) verify-first FFT on trailing window (all-polarity events in bbox)
            while i_tail < j_all and t[i_tail] < ct - P["ac_window"]:
                i_tail += 1
            verified = []           # (bbox, centroid, freq)
            if cands and j_all > i_tail:
                wx = x[i_tail:j_all]; wy = y[i_tail:j_all]; wt = t[i_tail:j_all]
                edges = np.arange(ct - P["ac_window"], ct, 1.0 / P["fs"])
                for area, bbox, cen in cands:
                    rx, ry, rw, rh = bbox
                    inb = (wx >= rx) & (wx <= rx + rw) & (wy >= ry) & (wy <= ry + rh)
                    vt = wt[inb]
                    if vt.size <= P["ac_min_events"] or edges.size <= 10:
                        continue
                    sig = np.histogram(vt, bins=edges)[0].astype(np.float64)
                    sig -= sig.mean()
                    mag = np.abs(np.fft.rfft(sig)) / len(sig)
                    if len(mag) > 2:
                        mag[1:-1] *= 2.0
                    faxis = np.fft.rfftfreq(len(sig), d=1.0 / P["fs"])
                    band = (faxis >= P["freq_lo"]) & (faxis <= P["freq_hi"])
                    if band.any():
                        bi = np.argmax(mag[band])
                        if mag[band][bi] > P["snr_thresh"]:
                            verified.append((np.array(bbox, float), np.array(cen, float),
                                             float(faxis[band][bi])))
            # 4) NN association with velocity prediction + coasting
            for tr in active:
                tr["cx"] += tr["vx"]; tr["cy"] += tr["vy"]      # predict
            used = [False] * len(verified)
            # greedy nearest match
            for tr in active:
                best, bd = -1, P["max_match_dist"]
                for k, (bb, cen, fr) in enumerate(verified):
                    if used[k]:
                        continue
                    d = np.hypot(cen[0] - tr["cx"], cen[1] - tr["cy"])
                    if d < bd:
                        bd, best = d, k
                if best >= 0:
                    bb, cen, fr = verified[best]; used[best] = True
                    s = P["smooth"]
                    ncx, ncy = (1 - s) * cen[0] + s * tr["cx"], (1 - s) * cen[1] + s * tr["cy"]
                    tr["vx"], tr["vy"] = ncx - (tr["cx"] - tr["vx"]), ncy - (tr["cy"] - tr["vy"])
                    tr["cx"], tr["cy"] = ncx, ncy
                    tr["bbox"] = s * tr["bbox"] + (1 - s) * bb
                    tr["missed"] = 0; tr["freq"] = fr
                    records[tr["id"]].append((ct, ncx, ncy, fr))
                else:
                    tr["missed"] += 1
            active = [tr for tr in active if tr["missed"] <= P["max_missed"]]
            for k, (bb, cen, fr) in enumerate(verified):
                if not used[k] and len(active) < P["max_tracks"]:
                    active.append(dict(id=next_id, cx=cen[0], cy=cen[1], vx=0.0, vy=0.0,
                                       bbox=bb.copy(), missed=0, freq=fr))
                    records[next_id] = [(ct, cen[0], cen[1], fr)]
                    next_id += 1
            logic_t += time.perf_counter() - t0

        # assemble tracks (image -> bearing/elevation for the standard schema)
        deg_per_px = cfg.fov_deg / W
        tracks = []
        for tid, rec in records.items():
            if len(rec) < 2:
                continue
            arr = np.array(rec)            # cols: t, cx, cy, freq
            tt, cx, cy, fr = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
            if cfg.mode == "rotation" and tel is not None:
                pan = np.rad2deg(np.interp(tt - tel.offset, tel.t, tel.azimuth_unwrapped()))
                az = np.mod(pan + cfg.az_sign * (cx - W / 2) * deg_per_px, 360.0)
            else:
                az = cfg.az_sign * (cx - W / 2) * deg_per_px
            elev = (H / 2 - cy) * deg_per_px
            tracks.append(dict(id=int(tid), t=tt, azimuth_deg=az, elev_deg=elev,
                               range_m=np.full_like(tt, np.nan), cx=cx, cy=cy, freq_hz=fr))
        ms = 1e3 * logic_t / max(len(steps), 1)
        print(f"[track:{self.name}] {len(tracks)} verified track(s); "
              f"{ms:.2f} ms/frame logic ({'real-time OK' if ms < P['accum_dt']*1e3 else 'over budget'})")
        return {"tracks": tracks}


@register
class HummingbirdTracker(FreqVerifiedTracker):
    name = "hummingbird"
    description = "Wingbeat-verified hummingbird tracker (FFT 10-120 Hz; time-surface + NN swarm)."
    params = dict(FreqVerifiedTracker.params,
                  freq_lo=10.0, freq_hi=120.0, min_cluster_size=15, max_tracks=4)


@register
class DroneFFTTracker(FreqVerifiedTracker):
    name = "drone_fft"
    description = "Rotor-verified drone tracker (FFT 80-800 Hz; time-surface + velocity coasting)."
    params = dict(FreqVerifiedTracker.params,
                  freq_lo=80.0, freq_hi=800.0, min_cluster_size=20, max_tracks=4)
