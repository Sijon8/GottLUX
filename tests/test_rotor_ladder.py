"""
Tests for the rotor-ladder detector (gottlux.rotation.rotor_ladder) — the spinning-sensor
stair-step signature of a drone rotor.

Validates the central physics ``f = |v| / Δx`` (recover the blade-pass frequency from the comb
spacing and the sweep rate), that a continuous edge / unstructured noise are rejected, and the
cross-revolution recurrence + motion offset.
"""
import numpy as np
import pytest

from gottlux.rotation import rotor_ladder as rl


@pytest.mark.parametrize("blade_hz,sweep", [(200.0, -1800.0), (150.0, -1500.0), (250.0, -1800.0)])
def test_recovers_blade_frequency_from_geometry(blade_hz, sweep):
    """In the resolvable regime (step ≫ disk), f = |v|/Δx is recovered from the comb alone."""
    x, t = rl.synthetic_rotor_pass(blade_hz=blade_hz, sweep_px_s=sweep, disk_px=1.5,
                                   burst_events=30, noise_events=150, seed=1)
    r = rl.ladder_signature(x, t, sweep_px_s=sweep)
    assert r.detected and r.in_band
    assert r.blade_hz == pytest.approx(blade_hz, rel=0.12)        # f from geometry
    assert r.step_px == pytest.approx(abs(sweep) / blade_hz, rel=0.15)   # Δx = v/f


def test_step_times_frequency_equals_sweep():
    sweep = -1800.0
    x, t = rl.synthetic_rotor_pass(blade_hz=200.0, sweep_px_s=sweep, disk_px=1.5, seed=2)
    r = rl.ladder_signature(x, t, sweep_px_s=sweep)
    assert r.step_px * r.blade_hz == pytest.approx(abs(sweep), rel=0.2)


def test_rejects_continuous_edge():
    """A swept building edge drifts but has no burst comb → not a rotor."""
    rng = np.random.default_rng(2)
    t = np.sort(rng.random(6000) * 0.16)
    x = np.clip(300 - 1800 * t + rng.normal(0, 3, 6000), 0, 319)   # continuous streak
    r = rl.ladder_signature(x, t, sweep_px_s=-1800)
    assert not r.detected and r.comb_strength < 0.2


def test_rejects_noise():
    for seed in range(5):
        x = np.random.default_rng(seed).uniform(0, 320, 4000)
        t = np.sort(np.random.default_rng(seed + 9).random(4000) * 0.16)
        assert not rl.ladder_signature(x, t, sweep_px_s=-1800).detected


def test_below_min_events_is_inconclusive():
    x, t = rl.synthetic_rotor_pass(blade_hz=200.0, sweep_px_s=-1800.0, burst_events=2, seed=3)
    assert rl.ladder_signature(x, t, sweep_px_s=-1800, min_events=500).detected is False


def test_ladder_figure_builds():
    import matplotlib
    matplotlib.use("Agg")
    x, t = rl.synthetic_rotor_pass(blade_hz=200.0, sweep_px_s=-1800.0, disk_px=1.5, seed=1)
    r = rl.ladder_signature(x, t, sweep_px_s=-1800)
    fig = rl.ladder_figure(x, t, r)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_cross_revolution_recurrence_and_motion():
    """A ladder that recurs across revolutions, shifting by a fixed offset = relative motion."""
    sweep, t_rot = -1800.0, 1.0
    passes = []
    for rev in range(4):
        x, t = rl.synthetic_rotor_pass(blade_hz=200.0, sweep_px_s=sweep, disk_px=1.5,
                                       x0=300 - 8 * rev, burst_events=30, seed=rev)  # drifts 8 px/rev
        res = rl.ladder_signature(x, t, sweep_px_s=sweep)
        passes.append((rev, float(np.median(x)), res))
    track = rl.track_ladders(passes, t_rot_s=t_rot)
    assert track.n_passes >= 3
    assert track.median_blade_hz == pytest.approx(200.0, rel=0.12)
    assert track.blade_hz_stability > 0.8                          # steady f across revs
    assert abs(track.azimuth_offset_per_rev_px) > 3               # detected the per-rev motion
