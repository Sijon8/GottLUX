"""
_synth_rotation.py — a fully-labelled synthetic *rotating* EBS scene for validating the 360°
rotor-ladder survey (:mod:`gottlux.rotation.rotor_scan`) end-to-end against ground truth.

Real rotating captures rarely come with per-target truth, so before spending effort on a real
clip we plant a scene where everything is known and check the survey recovers it:

* a **multirotor** at a known world bearing, emitting rotor bursts at a known blade-pass frequency,
  sized so the pinhole range model recovers a known range — and drifting a known number of degrees
  per revolution (a known relative angular rate, the "offset from the spin");
* a **static edge** at another bearing (a continuous swept streak — *no* comb — that the detector
  must reject); and
* unstructured background noise.

A matching telemetry CSV (azimuth ramp + Hall-sync rev boundaries) is written so the scene loads
as a genuine rotation :class:`~gottlux.io.recording.Recording`. The generator is the basis of
``tests/test_rotor_scan.py`` and ``tests/test_rotor_ladder.py``.

This module is deliberately project-agnostic: the drone size / prop / spin are all parameters
(defaulting to a 225 mm quad with 5-inch props at a ~1 Hz spin).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from gottlux.io.recording import Recording
from gottlux.io.telemetry import Telemetry
from gottlux.rotation.detect import focal_px
from gottlux.rotation.rotor_ladder import synthetic_rotor_pass


@dataclass
class RotationTruth:
    """Ground truth for a synthetic rotating scene."""
    blade_hz: float
    rotor_hz: float
    rpm: float
    n_blades: int
    range_m: float
    drone_az0_deg: float
    drift_deg_per_rev: float
    omega_d_deg_s: float
    edge_az_deg: float | None
    t_rot_s: float
    fov_deg: float
    pass_bearings_deg: list = field(default_factory=list)

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _systime(t_s: float) -> str:
    """Format seconds-from-start as the telemetry ``date_HHMMSS_millis`` stamp (base midnight)."""
    total_ms = int(round(t_s * 1000.0))
    hh = (total_ms // 3_600_000) % 24
    mm = (total_ms // 60_000) % 60
    ss = (total_ms // 1000) % 60
    ms = total_ms % 1000
    return f"20260101_{hh:02d}{mm:02d}{ss:02d}_{ms:03d}"


def write_telemetry_csv(path, *, n_rev, t_rot_s, tel_dt=0.005) -> str:
    """Write a telemetry CSV: azimuth ramps 0→360 each revolution, Hall-sync at every boundary."""
    rows = []
    duration = n_rev * t_rot_s
    # fine azimuth samples
    ts = np.arange(0.0, duration + tel_dt, tel_dt)
    for t in ts:
        rev = int(t / t_rot_s)
        az = (t / t_rot_s - rev) * 360.0
        rows.append((t, az, rev, ""))
    # explicit Hall-sync rows exactly on each revolution boundary
    for k in range(n_rev + 1):
        rows.append((k * t_rot_s, 0.0, k, "HALL_SYNC"))
    rows.sort(key=lambda r: (r[0], 0 if r[3] else 1))
    with open(path, "w", encoding="utf-8") as f:
        f.write("System_Time,Azimuth,Revolution,Flag\n")
        for t, az, rev, flag in rows:
            f.write(f"{_systime(t)},{az:.3f},{rev},{flag}\n")
    return path


def synthetic_rotation(out_dir, *, n_rev=5, t_rot_s=1.0, fov_deg=58.0, width=320, height=320,
                       blade_hz=210.0, n_blades=2, drone_az0_deg=140.0, drift_deg_per_rev=10.0,
                       range_m=14.0, target_size_m=0.225, disk_px=2.0, burst_events=26,
                       drone_elev_deg=0.0, edge_az_deg=40.0, edge_events_per_rev=420,
                       noise_rate_hz=3000.0, az_sign=-1.0, seed=0, name="synth_rotor_360"):
    """Build a labelled synthetic rotating :class:`Recording` + truth.

    Returns ``(rec, truth)`` where *rec* has telemetry attached (CSV written into *out_dir*) and
    *truth* is a :class:`RotationTruth`. The drone's events all map to its (drifting) world
    bearing; the edge is a continuous streak at ``edge_az_deg``; noise is uniform.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    duration = n_rev * t_rot_s
    deg_per_px = fov_deg / width
    omega_deg_s = 360.0 / t_rot_s
    tau = fov_deg / 360.0 * t_rot_s                 # time a fixed bearing is inside the FOV
    v_gen = width / tau                             # px/s sweep across the frame during a pass
    fpx = focal_px(fov_deg, width)
    size_px = target_size_m * fpx / max(range_m, 1e-6)

    Xs, Ys, Ps, Ts = [], [], [], []
    pass_bearings = []

    def _world_to_pass_start(theta_d):
        """Absolute time when the boresight first brings world-az theta_d into the FOV edge."""
        return ((theta_d - fov_deg / 2.0) / 360.0) * t_rot_s

    # ---- the drone: one swept rotor pass per revolution, drifting in bearing ----
    for r in range(n_rev):
        theta_d = drone_az0_deg + drift_deg_per_rev * r
        pass_bearings.append(round(theta_d, 3))
        t0 = r * t_rot_s + _world_to_pass_start(theta_d)
        xr, tr = synthetic_rotor_pass(blade_hz=blade_hz, sweep_px_s=v_gen, duration_s=tau,
                                      x0=0.0, disk_px=disk_px, burst_events=burst_events,
                                      noise_events=0, width=width, seed=seed + 1 + r)
        nb = xr.size
        yc = height / 2.0 - drone_elev_deg / deg_per_px      # place the drone at a given elevation
        yy = yc + rng.uniform(-size_px / 2.0, size_px / 2.0, nb)
        Xs.append(xr); Ys.append(yy)
        Ps.append(rng.integers(0, 2, nb)); Ts.append((t0 + tr) * 1e6)

    # ---- a static edge: continuous swept streak at a fixed bearing, no burst comb ----
    if edge_az_deg is not None:
        for r in range(n_rev):
            t0 = r * t_rot_s + _world_to_pass_start(edge_az_deg)
            ne = edge_events_per_rev
            te = rng.uniform(0, tau, ne)
            xe = np.clip(v_gen * te + rng.normal(0, 1.5, ne), 0, width - 1)
            ye = np.clip(height / 2.0 + rng.normal(0, 6.0, ne), 0, height - 1)
            Xs.append(xe); Ys.append(ye)
            Ps.append(rng.integers(0, 2, ne)); Ts.append((t0 + te) * 1e6)

    # ---- unstructured background noise ----
    n_noise = int(noise_rate_hz * duration)
    if n_noise > 0:
        Xs.append(rng.uniform(0, width, n_noise))
        Ys.append(rng.uniform(0, height, n_noise))
        Ps.append(rng.integers(0, 2, n_noise))
        Ts.append(rng.uniform(0, duration, n_noise) * 1e6)

    x = np.clip(np.concatenate(Xs), 0, width - 1).astype(np.uint16)
    y = np.clip(np.concatenate(Ys), 0, height - 1).astype(np.uint16)
    p = np.concatenate(Ps).astype(np.uint8)
    t = np.concatenate(Ts).astype(np.int64)

    csv_path = write_telemetry_csv(os.path.join(out_dir, f"{name}_telemetry.csv"),
                                   n_rev=n_rev, t_rot_s=t_rot_s)
    rec = Recording.from_events(x, y, p, t, width=width, height=height,
                                fmt="synthetic", name=name)
    rec.attach_telemetry(Telemetry(csv_path), refine=False)

    truth = RotationTruth(
        blade_hz=blade_hz, rotor_hz=blade_hz / n_blades, rpm=blade_hz / n_blades * 60.0,
        n_blades=n_blades, range_m=range_m, drone_az0_deg=drone_az0_deg,
        drift_deg_per_rev=drift_deg_per_rev, omega_d_deg_s=drift_deg_per_rev / t_rot_s,
        edge_az_deg=edge_az_deg, t_rot_s=t_rot_s, fov_deg=fov_deg, pass_bearings_deg=pass_bearings)
    return rec, truth
