"""
trackers/builtin.py  --  reference trackers shipped with the platform.
These are simple, dependency-free baselines; the operator's ported MATLAB
algorithms can be added alongside (see TEMPLATE.py).
"""
from __future__ import annotations
import numpy as np
from gottlux.rotation.trackers import register
from gottlux.rotation.trackers.base import Tracker


def _angdiff(a, b):
    return np.abs(np.rad2deg(np.angle(np.exp(1j * np.deg2rad(a - b)))))


@register
class NearestTracker(Tracker):
    name = "nearest"
    description = "Greedy nearest-neighbour linker (gate on bearing/elevation + max time gap)."
    params = {"gate_deg": 8.0, "max_gap_s": 2.0}

    def track(self, traj, cfg, tel=None, ev=None):
        if not traj:
            return {"tracks": []}
        t = np.asarray(traj["t"]); az = np.asarray(traj["azimuth_deg"])
        el = np.asarray(traj["elev_deg"]); rng = np.asarray(traj.get("range_m", np.full_like(t, np.nan)))
        gate = self.params["gate_deg"]; max_gap = self.params["max_gap_s"]
        order = np.argsort(t)
        tracks = []          # each: dict(id, last_t, last_az, last_el, idx=[...])
        for i in order:
            best, bestd = None, 1e9
            for tr in tracks:
                if t[i] - tr["last_t"] > max_gap:
                    continue
                d = np.hypot(_angdiff(az[i], tr["last_az"]), el[i] - tr["last_el"])
                if d < bestd and d < gate:
                    best, bestd = tr, d
            if best is None:
                best = dict(id=len(tracks), last_t=t[i], last_az=az[i], last_el=el[i], idx=[])
                tracks.append(best)
            best["idx"].append(i)
            best["last_t"], best["last_az"], best["last_el"] = t[i], az[i], el[i]
        out = []
        for tr in tracks:
            ix = np.array(tr["idx"])
            out.append(dict(id=tr["id"], t=t[ix], azimuth_deg=az[ix],
                            elev_deg=el[ix], range_m=rng[ix]))
        return {"tracks": out}


@register
class SingleTargetTracker(Tracker):
    name = "single"
    description = "Single dominant target: robust (MAD) gate around the median bearing/elevation."
    params = {"mad_k": 6.0}

    def track(self, traj, cfg, tel=None, ev=None):
        if not traj:
            return {"tracks": []}
        t = np.asarray(traj["t"]); az = np.asarray(traj["azimuth_deg"])
        el = np.asarray(traj["elev_deg"]); rng = np.asarray(traj.get("range_m", np.full_like(t, np.nan)))
        k = self.params["mad_k"]
        med = np.median(el); mad = np.median(np.abs(el - med)) + 1e-6
        keep = np.abs(el - med) < k * 1.4826 * mad
        o = np.argsort(t[keep])
        return {"tracks": [dict(id=0, t=t[keep][o], azimuth_deg=az[keep][o],
                                elev_deg=el[keep][o], range_m=rng[keep][o])]}
