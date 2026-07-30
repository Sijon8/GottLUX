"""
trackers/kalman.py  --  Constant-velocity Kalman multi-target tracker.

A classical recursive-Bayesian tracker for the bearing/elevation measurement
stream produced by the detector. Where the built-in ``nearest`` tracker only
links nearest detections greedily (memoryless, gated on the *last raw* point),
this tracker carries a dynamical state per target -- angular position AND
angular velocity -- so it can:

  * PREDICT where a target should be at the next detection time (motion model),
  * GATE associations on the *predicted* position using the innovation
    covariance (statistical / Mahalanobis gate) rather than a fixed angle,
  * SMOOTH noisy bearing/elevation measurements (the Kalman gain optimally
    blends prediction and measurement by their relative uncertainties),
  * COAST through missed detections (dropouts) by propagating the state, and
  * manage a track lifecycle (tentative -> confirmed via M-of-N hits, then
    deleted after too long without an update) so spurious one-off detections
    do not spawn permanent tracks.

Physics / model
---------------
State per track:    x = [az, el, az_rate, el_rate]   (deg, deg, deg/s, deg/s)
Measurement:        z = [az, el]                      (deg, deg)
Dynamics:           nearly-constant-velocity with white-noise acceleration.
                    Over a step dt the state propagates with

                        F(dt) = | 1 0 dt 0 |
                                | 0 1 0  dt|
                                | 0 0 1  0 |
                                | 0 0 0  1 |

                    and the process-noise covariance Q(dt) is the standard
                    continuous-white-noise-acceleration (CWNA) form with
                    spectral density `q` (deg^2 / s^3) -- larger q == we expect
                    the target to maneuver harder, so the filter trusts new
                    measurements more.

Azimuth wraps at 360 deg in rotation mode. The filter therefore runs on an
*unwrapped* internal azimuth: each incoming measurement is shifted by the
multiple of 360 deg that brings it closest to the predicted azimuth, and only
the reported track azimuth is re-wrapped into [0, 360). Range is not part of
the dynamic state (it is a noisy size-from-pinhole estimate); it is carried
along each track with a light exponential smoother.
"""
from __future__ import annotations
import numpy as np
from gottlux.rotation.trackers import register
from gottlux.rotation.trackers.base import Tracker


# --------------------------------------------------------------------------- #
#  Linear-algebra helpers for the constant-velocity Kalman filter             #
# --------------------------------------------------------------------------- #
def _F(dt: float) -> np.ndarray:
    """State-transition for a 2-D constant-velocity model over step `dt`."""
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    return F


def _Q(dt: float, q: float) -> np.ndarray:
    """Continuous-white-noise-acceleration process covariance (per axis q)."""
    dt2, dt3 = dt * dt, dt * dt * dt
    qpos = q * dt3 / 3.0
    qpv = q * dt2 / 2.0
    qvel = q * dt
    return np.array([[qpos, 0.0, qpv, 0.0],
                     [0.0, qpos, 0.0, qpv],
                     [qpv, 0.0, qvel, 0.0],
                     [0.0, qpv, 0.0, qvel]])


_H = np.array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])


class _Track:
    """One target hypothesis: its Kalman state plus lifecycle bookkeeping."""

    __slots__ = ("id", "x", "P", "t", "hits", "misses", "confirmed",
                 "rng", "hist_t", "hist_az", "hist_el", "hist_rng")

    def __init__(self, tid, az, el, rng, t, P0_vel):
        self.id = tid
        self.x = np.array([az, el, 0.0, 0.0])          # zero initial velocity
        self.P = np.diag([1.0, 1.0, P0_vel, P0_vel])   # big velocity uncertainty
        self.t = t
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.rng = rng
        # filtered history (recorded at every measurement update)
        self.hist_t = [t]
        self.hist_az = [az]
        self.hist_el = [el]
        self.hist_rng = [rng]

    def predict(self, t, q):
        dt = t - self.t
        if dt <= 0:
            return
        F = _F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + _Q(dt, q)
        self.t = t

    def innovation(self, az_unwrapped, el):
        """Innovation nu and its covariance S for measurement [az, el]."""
        z = np.array([az_unwrapped, el])
        nu = z - _H @ self.x
        S = _H @ self.P @ _H.T + self._R
        return nu, S

    def update(self, az_unwrapped, el, rng, rng_alpha):
        nu, S = self.innovation(az_unwrapped, el)
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.P = (np.eye(4) - K @ _H) @ self.P
        self.hits += 1
        self.misses = 0
        if np.isfinite(rng):
            self.rng = rng if not np.isfinite(self.rng) else \
                (1 - rng_alpha) * self.rng + rng_alpha * rng
        self.hist_t.append(self.t)
        self.hist_az.append(float(self.x[0]))
        self.hist_el.append(float(self.x[1]))
        self.hist_rng.append(self.rng)


@register
class KalmanTracker(Tracker):
    name = "kalman"
    description = ("Constant-velocity Kalman multi-target tracker "
                   "(predict/gate/smooth/coast on bearing+elevation; M-of-N track mgmt).")
    params = {
        "q": 400.0,            # process-noise PSD [deg^2/s^3]; higher == more agile target
        "r_az_deg": 0.6,       # azimuth measurement-noise std [deg]
        "r_el_deg": 0.6,       # elevation measurement-noise std [deg]
        "gate_chi2": 13.8,     # 2-DOF Mahalanobis gate (~99.9%); reject worse associations
        "max_gap_s": 0.8,      # delete a track with no update for this long [s]
        "confirm_hits": 3,     # measurement hits needed to promote tentative -> confirmed
        "init_vel_var": 1.0e5, # initial velocity variance [ (deg/s)^2 ] -> trust first motions
        "range_alpha": 0.3,    # EMA weight for the (non-dynamic) range estimate
        "min_track_len": 3,    # drop output tracks shorter than this many points
    }

    def track(self, traj, cfg, tel=None, ev=None):
        if not traj:
            return {"tracks": []}
        P = self.params
        t = np.asarray(traj["t"], float)
        az = np.asarray(traj["azimuth_deg"], float)
        el = np.asarray(traj["elev_deg"], float)
        rng = np.asarray(traj.get("range_m", np.full_like(t, np.nan)), float)
        if t.size == 0:
            return {"tracks": []}

        # measurement-noise matrix shared by all tracks
        R = np.diag([P["r_az_deg"] ** 2, P["r_el_deg"] ** 2])
        _Track._R = R

        wrap = cfg.mode == "rotation"      # azimuth lives on a 360-deg circle?

        # group measurements by detection time so "miss" is well defined per step
        order = np.argsort(t, kind="stable")
        t_s, az_s, el_s, rng_s = t[order], az[order], el[order], rng[order]
        uniq_t = np.unique(t_s)

        tracks: list[_Track] = []
        finished: list[_Track] = []
        next_id = 0

        for ct in uniq_t:
            sel = np.where(t_s == ct)[0]
            meas = list(zip(az_s[sel], el_s[sel], rng_s[sel]))

            # 1) predict every live track to the current time
            for tr in tracks:
                tr.predict(ct, P["q"])

            # 2) build the gated cost matrix (Mahalanobis^2) and greedily assign
            pairs = []   # (cost, track_index, meas_index, az_unwrapped)
            for ti, tr in enumerate(tracks):
                for mi, (maz, mel, _mr) in enumerate(meas):
                    maz_u = self._unwrap(maz, tr.x[0]) if wrap else maz
                    nu, S = tr.innovation(maz_u, mel)
                    d2 = float(nu @ np.linalg.solve(S, nu))
                    if d2 <= P["gate_chi2"]:
                        pairs.append((d2, ti, mi, maz_u))
            pairs.sort(key=lambda r: r[0])
            t_used = [False] * len(tracks)
            m_used = [False] * len(meas)
            for d2, ti, mi, maz_u in pairs:
                if t_used[ti] or m_used[mi]:
                    continue
                t_used[ti] = m_used[mi] = True
                _maz, mel, mr = meas[mi]
                tracks[ti].update(maz_u, mel, mr, P["range_alpha"])
                if tracks[ti].hits >= P["confirm_hits"]:
                    tracks[ti].confirmed = True

            # 3) coast / age out tracks that were not updated this step
            for ti, tr in enumerate(tracks):
                if not t_used[ti]:
                    tr.misses += 1

            # 4) spawn tentative tracks from unassigned measurements
            for mi, (maz, mel, mr) in enumerate(meas):
                if not m_used[mi]:
                    tracks.append(_Track(next_id, maz, mel, mr, ct, P["init_vel_var"]))
                    next_id += 1

            # 5) retire tracks that have gone silent too long
            alive = []
            for tr in tracks:
                if ct - tr.t > P["max_gap_s"]:
                    finished.append(tr)
                else:
                    alive.append(tr)
            tracks = alive

        finished.extend(tracks)

        # assemble output: confirmed tracks of sufficient length, az re-wrapped
        out = []
        for tr in sorted(finished, key=lambda r: r.id):
            if not tr.confirmed or len(tr.hist_t) < P["min_track_len"]:
                continue
            az_out = np.asarray(tr.hist_az)
            if wrap:
                az_out = np.mod(az_out, 360.0)
            out.append(dict(id=int(tr.id),
                            t=np.asarray(tr.hist_t),
                            azimuth_deg=az_out,
                            elev_deg=np.asarray(tr.hist_el),
                            range_m=np.asarray(tr.hist_rng)))
        print(f"[track:{self.name}] {len(out)} confirmed track(s) "
              f"from {t.size} detections over {uniq_t.size} time steps")
        return {"tracks": out}

    @staticmethod
    def _unwrap(meas_az, pred_az):
        """Shift a 0-360 measurement by k*360 to sit closest to the prediction."""
        return meas_az + 360.0 * np.round((pred_az - meas_az) / 360.0)
