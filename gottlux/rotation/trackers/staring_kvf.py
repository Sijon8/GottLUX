"""
trackers/staring_kvf.py  --  Staring-mode Kalman-Vitality tracker.

A Python port-and-build-up of a Kalman-vitality-filter (KVF) staring validator
(ported from a MATLAB reference implementation). Built for the STARING regime: a fixed sensor where the
target moves in the IMAGE PLANE. The detector's per-frame blobs (cx, cy, bbox) are
linked by a constant-velocity Kalman filter, and each track carries an evidential
"vitality" score (a continuous M-of-N shield): vitality rises when a detection is
associated and decays otherwise, so sporadic neuromorphic clutter never reaches the
confirmation threshold while a genuine, persistent signature locks on.

Differences from the reference, by design: no trained SNN is required —
the "is it a drone" decision is read from the track's own kinematics + bounding-box
behaviour downstream (core/track_analysis.drone_likeness). This keeps the tracker
fully runnable with zero heavy dependencies. Each track stores its BOUNDING-BOX SIZE
over time so range can be estimated from apparent size (lens + sensor + object size).
"""
from __future__ import annotations
import numpy as np
from gottlux.rotation.trackers import register
from gottlux.rotation.trackers.base import Tracker


class _KVTrack:
    """One constant-velocity Kalman track with an evidential vitality score."""
    def __init__(self, tid, x, y, t, p):
        self.id = tid
        self.X = np.array([x, y, 0.0, 0.0], float)        # [x, y, vx, vy]
        self.P = np.eye(4) * 50.0
        self.q = p["process_noise"]; self.r = p["meas_noise"]
        self.last_t = t
        self.vit = 0.35; self.max_vit = 0.35
        self.idx = []                                      # detection indices associated
        self.vit_hist = []

    def predict(self, t, vit_miss_rate):
        dt = max(t - self.last_t, 0.0); self.last_t = t
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.X = F @ self.X
        self.P = F @ self.P @ F.T + np.eye(4) * (self.q * (dt + 1e-3))
        self.vit = max(0.0, self.vit - vit_miss_rate * dt)
        return dt

    def update(self, x, y, vit_hit, di):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        R = np.eye(2) * self.r
        Z = np.array([x, y], float)
        Y = Z - H @ self.X
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.X = self.X + K @ Y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.vit = min(1.0, self.vit + vit_hit); self.max_vit = max(self.max_vit, self.vit)
        self.idx.append(di)


@register
class StaringKVFTracker(Tracker):
    name = "staring_kvf"
    description = ("Staring-mode Kalman-Vitality tracker: image-plane CV Kalman + an "
                   "evidential vitality shield (suppresses clutter); stores bbox size over "
                   "time for size->range estimation. For FIXED-sensor captures.")
    regime = "staring"
    params = {
        "gate_px": 45.0,         # association radius (pixels)
        "vit_hit": 0.18,         # vitality gained on association
        "vit_miss_rate": 0.6,    # vitality lost per second without association
        "confirm": 0.7,          # vitality to confirm a track
        "death": 0.08,           # vitality below which a track is dropped
        "min_points": 4,         # minimum associated detections to emit a track
        "process_noise": 8.0,
        "meas_noise": 16.0,
    }
    uses_events = False          # consumes the detector trajectory (cx, cy, bbox)

    def track(self, traj, cfg, tel=None, ev=None):
        if not traj or not len(traj.get("t", [])):
            return {"tracks": []}
        p = self.params
        t = np.asarray(traj["t"], float); cx = np.asarray(traj["cx"], float); cy = np.asarray(traj["cy"], float)
        order = np.argsort(t, kind="stable")
        live, done, next_id = [], [], 0
        for di in order:
            ti, xi, yi = t[di], cx[di], cy[di]
            for trk in live:                                   # 1. predict + decay all tracks
                trk.predict(ti, p["vit_miss_rate"])
            best, bd = None, p["gate_px"]                      # 2. nearest-neighbour association (gated)
            for trk in live:
                d = np.hypot(xi - trk.X[0], yi - trk.X[1])
                if d < bd:
                    best, bd = trk, d
            if best is not None:
                best.update(xi, yi, p["vit_hit"], di)
            else:
                nt = _KVTrack(next_id, xi, yi, ti, p); nt.idx.append(di); next_id += 1
                live.append(nt)
            for trk in live:
                trk.vit_hist.append(trk.vit)
            keep = [trk for trk in live if trk.vit >= p["death"]]   # 3. prune dead
            done += [trk for trk in live if trk not in keep]
            live = keep
        done += live

        # 4. emit confirmed / sufficiently-supported tracks, carrying bbox + range + vitality
        out = []
        W = cfg.sensor_w; H = cfg.sensor_h; dpp = cfg.fov_deg / W
        az_all = np.asarray(traj.get("azimuth_deg", np.zeros_like(t)), float)
        el_all = np.asarray(traj.get("elev_deg", np.zeros_like(t)), float)
        dx_all = np.asarray(traj.get("dx", np.full_like(t, np.nan)), float)
        dy_all = np.asarray(traj.get("dy", np.full_like(t, np.nan)), float)
        for trk in done:
            if trk.max_vit < p["confirm"] or len(trk.idx) < p["min_points"]:
                continue
            ix = np.array(sorted(trk.idx), int)
            out.append(dict(
                id=len(out), t=t[ix], cx=cx[ix], cy=cy[ix], dx=dx_all[ix], dy=dy_all[ix],
                azimuth_deg=az_all[ix], elev_deg=el_all[ix],
                vitality=np.full(len(ix), float(trk.max_vit)),
            ))
        return {"tracks": out, "n_candidates": len(done)}
