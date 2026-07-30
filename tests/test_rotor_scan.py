"""
Tests for the 360° rotor-ladder survey (gottlux.rotation.rotor_scan) — classifying a target by
its propeller signature, mapping every bearing it recurs at, and recovering its range and the
per-revolution motion offset, all validated against a planted synthetic rotating scene.
"""
import numpy as np
import pytest

from gottlux.config import Config
from gottlux.rotation import _synth_rotation as synth
from gottlux.rotation import ev_dict, resolve_cfg
from gottlux.rotation import rotor_scan as rs


# --------------------------------------------------------------------- propeller kinematics
def test_propeller_kinematics():
    sig = rs.propeller_kinematics(210.0, n_blades=2, prop_diameter_m=0.127)
    assert sig.rotor_hz == pytest.approx(105.0)
    assert sig.rpm == pytest.approx(6300.0)
    assert sig.tip_speed_mps == pytest.approx(np.pi * 0.127 * 105.0, rel=0.01)
    assert 0.1 < sig.tip_mach < 0.15
    # 3-blade halves the implied rotor rate of the same blade tone
    assert rs.propeller_kinematics(210.0, n_blades=3).rotor_hz == pytest.approx(70.0)
    assert rs.propeller_kinematics(0.0) is None


def test_sweep_velocity_from_telemetry(tmp_path):
    rec, _ = synth.synthetic_rotation(str(tmp_path), n_rev=2, fov_deg=58.0)
    cfg = resolve_cfg(rec, Config(mode="rotation", fov_deg=58.0))
    v = rs.sweep_velocity_px_s(cfg, rec.telemetry)
    assert v == pytest.approx(360.0 / 1.0 * 320 / 58.0, rel=0.02)   # Ω·W/FOV


# --------------------------------------------------------------------- analyze a box
def test_analyze_box_recovers_signature(tmp_path):
    rec, truth = synth.synthetic_rotation(str(tmp_path), n_rev=2, blade_hz=200.0,
                                          drone_az0_deg=150.0, drift_deg_per_rev=0.0)
    cfg = resolve_cfg(rec, Config(mode="rotation", fov_deg=58.0, target_size_m=0.225))
    ev = ev_dict(rec.all())
    # the box isolates the target: a tight time window over its first transit (boresight sweeps
    # 150° at t = 150/360 s of a 1 s revolution) AND its narrow elevation (Y) band — the drone
    # spans all X as it sweeps but only a few rows in Y, so the Y band rejects all-azimuth noise.
    res, det = rs.analyze_box(ev, cfg, rec.telemetry, roi=(0, 148, 320, 172), t0=0.33, t1=0.50)
    assert res.detected
    assert res.blade_hz == pytest.approx(200.0, rel=0.15)
    assert det.bearing_deg == pytest.approx(150.0, abs=4.0)
    assert det.rpm == pytest.approx(6000.0, rel=0.15)


# --------------------------------------------------------------------- the 360° scan
@pytest.mark.parametrize("drift", [0.0, 8.0])
def test_scan_recovers_track_and_motion(tmp_path, drift):
    rec, truth = synth.synthetic_rotation(str(tmp_path), n_rev=6, blade_hz=210.0,
                                          drone_az0_deg=140.0, drift_deg_per_rev=drift,
                                          range_m=14.0, noise_rate_hz=3000.0)
    cfg = resolve_cfg(rec, Config(mode="rotation", fov_deg=58.0, target_size_m=0.225))
    res = rs.scan_rotation(ev_dict(rec.all()), cfg, rec.telemetry, keep=None)

    assert res.f_template_hz == pytest.approx(210.0, rel=0.15)
    assert res.n_matched >= 5                       # the signature found in most revolutions
    assert len(res.tracks) == 1                     # one coherent rotor track
    tr = res.tracks[0]
    assert tr.n_passes >= 5
    assert tr.median_blade_hz == pytest.approx(210.0, rel=0.15)
    assert tr.blade_hz_stability > 0.85
    # the per-revolution azimuth offset == the planted relative motion
    assert tr.bearing_offset_per_rev_deg == pytest.approx(drift, abs=1.5)
    assert tr.omega_deg_s == pytest.approx(truth.omega_d_deg_s, abs=1.5)
    if drift == 0.0:
        assert tr.bearing_deg == pytest.approx(140.0, abs=3.0)


def test_scan_range_in_ballpark(tmp_path):
    rec, _ = synth.synthetic_rotation(str(tmp_path), n_rev=4, range_m=14.0, drift_deg_per_rev=0.0)
    cfg = resolve_cfg(rec, Config(mode="rotation", fov_deg=58.0, target_size_m=0.225))
    res = rs.scan_rotation(ev_dict(rec.all()), cfg, rec.telemetry, keep=None)
    assert res.tracks and res.tracks[0].range_m is not None
    assert res.tracks[0].range_m == pytest.approx(14.0, rel=0.35)   # coarse kinematic range


def test_static_edge_not_matched(tmp_path):
    """The continuous swept edge has no comb → it must not be flagged as the rotor signature."""
    rec, truth = synth.synthetic_rotation(str(tmp_path), n_rev=4, drone_az0_deg=140.0,
                                          drift_deg_per_rev=0.0, edge_az_deg=40.0,
                                          noise_rate_hz=2000.0)
    cfg = resolve_cfg(rec, Config(mode="rotation", fov_deg=58.0, target_size_m=0.225))
    res = rs.scan_rotation(ev_dict(rec.all()), cfg, rec.telemetry, keep=None)
    # no matched detection should sit at the edge bearing (40°)
    for d in res.matched:
        assert abs(((d.bearing_deg - 40.0 + 180) % 360) - 180) > 8.0


def test_no_telemetry_staring_mode():
    """With no telemetry the scan still runs (relative bearing, one 'revolution')."""
    from gottlux.rotation.rotor_ladder import synthetic_rotor_pass
    from gottlux.io.recording import Recording
    x, t = synthetic_rotor_pass(blade_hz=200.0, sweep_px_s=-1800.0, disk_px=1.5,
                                burst_events=30, width=320, seed=1)
    y = np.full_like(x, 160.0)
    rec = Recording.from_events(x.astype(int), y.astype(int), np.zeros_like(x, int),
                                (t * 1e6).astype(np.int64), width=320, height=320)
    cfg = resolve_cfg(rec, Config(mode="staring", fov_deg=58.0, target_size_m=0.225))
    res = rs.scan_rotation(ev_dict(rec.all()), cfg, None, keep=None, min_events=120)
    assert isinstance(res.detections, list)         # runs without telemetry


# --------------------------------------------------------------------- cross-revolution accumulation
def test_accumulate_comb_recovers_fundamental():
    """Averaging per-pass autocorrelations reinforces the comb; the fundamental is recovered even
    when individual passes are noisy."""
    from gottlux.rotation.rotor_ladder import synthetic_rotor_pass
    v = -1800.0
    passes = []
    for k in range(8):
        x, _ = synthetic_rotor_pass(blade_hz=200.0, sweep_px_s=v, disk_px=1.5,
                                    burst_events=22, noise_events=400, seed=k)
        passes.append(x)
    acc = rs.accumulate_comb(passes, v, f_lo=80, f_hi=800)
    assert acc is not None and acc["n_passes"] == 8
    assert acc["blade_hz"] == pytest.approx(200.0, rel=0.15)       # f from the accumulated comb
    assert acc["comb_strength"] > 0.2


def test_scan_reports_accumulated_frequency(tmp_path):
    rec, _ = synth.synthetic_rotation(str(tmp_path), n_rev=6, blade_hz=210.0, drift_deg_per_rev=4.0)
    cfg = resolve_cfg(rec, Config(mode="rotation", fov_deg=58.0, target_size_m=0.225))
    res = rs.scan_rotation(ev_dict(rec.all()), cfg, rec.telemetry, keep=None)
    assert res.accumulated_blade_hz == pytest.approx(210.0, rel=0.15)


# --------------------------------------------------------------------- telemetry-less clips
def test_estimate_spin_and_synthesize_telemetry(tmp_path):
    """A rotating clip with no azimuth CSV: estimate the period from event-rate periodicity and
    synthesize telemetry, then recover the rotor across revolutions through that estimate."""
    from gottlux.io.telemetry import estimate_spin_period_s, Telemetry
    rec, truth = synth.synthetic_rotation(str(tmp_path), n_rev=6, t_rot_s=1.0, blade_hz=210.0,
                                          drift_deg_per_rev=0.0, range_m=14.0, noise_rate_hz=3000.0)
    period, conf = estimate_spin_period_s(rec.t.astype(float) / 1e6)
    assert period == pytest.approx(1.0, rel=0.05)          # recovered the planted spin period
    assert conf > 0.3
    tel = Telemetry.from_spin(rec.duration_s, period)
    assert tel.synthesized and tel.n_revolutions >= 5
    assert tel.omega_deg_s == pytest.approx(360.0, rel=0.05)
    # the survey works off the estimated telemetry
    cfg = resolve_cfg(rec, Config(mode="rotation", fov_deg=58.0, target_size_m=0.225))
    res = rs.scan_rotation(ev_dict(rec.all()), cfg, tel, keep=None)
    assert res.f_template_hz == pytest.approx(210.0, rel=0.15)
    assert res.n_matched >= 4


# --------------------------------------------------------------------- rotational background masking
def test_masking_sweep_and_movers(tmp_path):
    from gottlux.rotation import masking as mk
    rec, _ = synth.synthetic_rotation(str(tmp_path), n_rev=6, drift_deg_per_rev=4.0, range_m=14.0)
    cfg = resolve_cfg(rec, Config(mode="rotation", fov_deg=58.0, target_size_m=0.225,
                                  mask_rotations=2))
    res = mk.run_masking(rec, cfg, n_list=(0, 1, 2), min_events=120)
    red = {m.n_rotations: m.reduction_pct for m in res.sweep}
    assert red[2] >= red[1] >= red[0]                       # more reference rotations → more reduction
    assert red[2] > 10                                      # static edge removed (synth has little static)
    assert len(res.movers) >= 1                             # the moving drone survives
    assert any(130 <= m.bearing_deg <= 185 for m in res.movers)   # a mover at the planted drone arc
    tbl = mk.movers_table(res)
    assert all(len(v) == len(res.movers) for v in tbl.values())
    assert "reduction_sweep" in res.headline()


def test_multifile_masking(tmp_path):
    """The multi-file driver: each clip yields a persistent above-horizon mover (the drone)."""
    from gottlux.rotation import masking as mk
    rec1, _ = synth.synthetic_rotation(str(tmp_path / "a"), n_rev=6, drone_az0_deg=140.0,
                                       drift_deg_per_rev=2.0, drone_elev_deg=4.0)
    rec2, _ = synth.synthetic_rotation(str(tmp_path / "b"), n_rev=6, drone_az0_deg=144.0,
                                       drift_deg_per_rev=2.0, drone_elev_deg=4.0, seed=5)
    cfg = Config(mode="rotation", fov_deg=58.0, target_size_m=0.225, mask_rotations=2)
    multi = mk.run_multifile([rec1, rec2], cfg, n_list=(0, 1, 2))
    assert multi["n_files"] == 2
    assert multi["consistent"]                                  # a persistent mover in both clips
    assert all(120 <= f["primary"]["bearing_deg"] <= 185 for f in multi["per_file"])
    assert multi["bearings_agree"]                              # same target arc in both
    from gottlux.rotation.viz import masking_viz
    fig = masking_viz.multifile_figure(multi)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_cumulative_mask_and_below_horizon_track(tmp_path):
    """Cumulative masking + linked track follow a drone with NO above-horizon assumption — here the
    drone is planted BELOW the horizon (high sensor tilt / low-flying), as flagged during field validation."""
    from gottlux.rotation import background as bg, ev_dict, masking as mk
    rec, _ = synth.synthetic_rotation(str(tmp_path), n_rev=6, drone_az0_deg=140.0,
                                      drift_deg_per_rev=3.0, drone_elev_deg=-4.0)   # below horizon
    cfg = resolve_cfg(rec, Config(mode="rotation", fov_deg=58.0, target_size_m=0.225))
    ev = ev_dict(rec.all()); hot = bg.hot_pixel_mask(ev, 99.95)
    keep = mk.keep_mask_cumulative(ev, rec.telemetry, cfg, hot=hot)
    assert 0 < keep.sum() < ev["n"]                          # masked static, kept the mover
    der = mk._derotate(ev, rec.telemetry, cfg)
    movers = mk.extract_movers(ev, rec.telemetry, cfg, keep, deroted=der, min_events=120)
    assert any(m.elev_deg < 0 for m in movers)               # the drone survives below the horizon
    tracks = mk.link_mover_tracks(movers, min_revs=3)
    assert tracks and 125 <= np.median([m.bearing_deg for m in tracks[0]]) <= 175
    res = mk.MaskingResult([], movers, 2, 0.0, cfg.fov_deg, 1.0, 0.225)
    assert len(mk.densest_track(res)) >= 3                   # densest-mover track (no elevation gate)


def test_masking_registered_and_pipeline(tmp_path):
    import os
    from gottlux.run.pipeline import _ANALYSES, run_recording
    assert {"masking", "rotation_rate"} <= set(_ANALYSES)
    rec, _ = synth.synthetic_rotation(str(tmp_path), n_rev=5, drift_deg_per_rev=4.0)
    cfg = Config(mode="rotation", fov_deg=58.0, target_size_m=0.225, mask_rotations=2,
                 analyses=("masking",), output_root=str(tmp_path / "runs"), open_when_done=False)
    cfg.sensor_w, cfg.sensor_h = rec.width, rec.height
    path = run_recording(rec, cfg)
    assert os.path.isdir(os.path.join(path, "masking"))     # the analysis wrote its subfolder


# --------------------------------------------------------------------- rotation rate + FFT de-rotation
def test_find_rotation_rate(tmp_path):
    from gottlux.rotation import rate_analysis as ra
    rec, _ = synth.synthetic_rotation(str(tmp_path), n_rev=6, t_rot_s=1.0)
    res = ra.find_rotation_rate(rec)
    assert res["hz"] == pytest.approx(1.0, rel=0.06)          # spin from event-rate autocorrelation
    assert res["t"].size and res["acf"].size                  # arrays for the plot
    fig = ra.event_rate_figure(res)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_region_spectrum_derotate_suppresses_rotation():
    """The derotate high-pass removes the slow once-per-rev envelope (the 1 Hz FFT gravity)."""
    from gottlux.core import frequency as fq
    # events in bursts every 0.5 s (2 Hz envelope) — a strong slow rotation-like component
    bursts = []
    for k in range(8):
        t0 = k * 0.5
        bursts.append(t0 + np.linspace(0, 0.02, 60))
    t_us = (np.concatenate(bursts) * 1e6)
    raw = fq.region_spectrum(t_us, fs=2000, fmin=5, fmax=400, derotate_hz=0.0)
    der = fq.region_spectrum(t_us, fs=2000, fmin=5, fmax=400, derotate_hz=40.0)
    f = np.asarray(raw.freqs)
    lowmask = (f > 0) & (f < 5)
    assert np.asarray(der.power)[lowmask].sum() < np.asarray(raw.power)[lowmask].sum()


# --------------------------------------------------------------------- tables + report
def test_tables_and_report(tmp_path):
    rec, truth = synth.synthetic_rotation(str(tmp_path), n_rev=4, drift_deg_per_rev=8.0)
    cfg = resolve_cfg(rec, Config(mode="rotation", fov_deg=58.0, target_size_m=0.225))
    res = rs.scan_rotation(ev_dict(rec.all()), cfg, rec.telemetry, keep=None)

    dt = rs.detections_table(res)
    n = len(res.detections)
    assert all(len(v) == n for v in dt.values())
    assert "blade_hz" in dt and "bearing_deg" in dt and "matches_template" in dt

    from gottlux.rotation import ladder_report
    out = tmp_path / "study"
    written = ladder_report.save_scan_report(str(out), res, cfg=cfg,
                                             meta={"recording": "t", "truth": truth.as_dict()},
                                             dpi=80)
    assert any(p.endswith("rotor-ladder-360-report.tex") for p in written)
    tex = (out / "rotor-ladder-360-report.tex").read_text(encoding="utf-8")
    assert tex.count(r"\begin{") == tex.count(r"\end{")     # balanced LaTeX
    assert tex.strip().endswith(r"\end{document}")
