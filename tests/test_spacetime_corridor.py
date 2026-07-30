"""Tests for the space-time corridor + box/measure math (pure, no GL needed)."""
import numpy as np
import pytest

from gottlux.app.spacetime import (SpaceTimeView, _aabb_from_opposite, _compute_slab,
                                    _measure_stats, _nearest_screen, _project_points)


def test_time_at_zfrac_inverts_the_render_z_mapping():
    """The time-axis markers' z→time map must be the exact inverse of the render's time→z map,
    for every anchor/flip combination, so the labels land on the right plane."""
    base = {"t0": 3.0, "span": 2.0, "depth": 100.0}
    for newest_front in (True, False):
        for flip in (False, True):
            m = dict(base, newest_front=newest_front, flip=flip)
            for pos in (0.0, 0.25, 0.5, 1.0):
                frac = (1.0 - pos) if newest_front else pos      # forward map (see _render)
                if flip:
                    frac = 1.0 - frac
                t = SpaceTimeView._time_at_zfrac(None, frac, m)  # method ignores self
                assert t == pytest.approx(m["t0"] + pos * m["span"])


def test_trailing_window_is_behind_cursor_newest_at_front():
    t0, t1, newest_front = _compute_slab(0.0, 10.0, cursor=5.0, depth_s=2.0,
                                         anchor="Trailing (stream)", full=False)
    assert (t0, t1) == pytest.approx((3.0, 5.0))
    assert newest_front is True                       # cursor (newest) sits at the front plane


def test_trailing_clamps_at_recording_start():
    t0, t1, nf = _compute_slab(0.0, 10.0, cursor=1.0, depth_s=5.0,
                               anchor="Trailing (stream)", full=False)
    assert t0 == 0.0 and t1 == pytest.approx(1.0) and nf is True


def test_full_is_infinite_trailing_from_start():
    t0, t1, nf = _compute_slab(2.0, 12.0, cursor=9.0, depth_s=0.1, anchor="Trailing (stream)",
                               full=True)
    assert t0 == 2.0 and t1 == pytest.approx(9.0) and nf is True   # whole stream up to the cursor


def test_forward_matches_legacy_behaviour():
    t0, t1, nf = _compute_slab(0.0, 10.0, cursor=4.0, depth_s=2.0, anchor="Forward", full=False)
    assert (t0, t1) == pytest.approx((4.0, 6.0)) and nf is False   # cursor at front, looks ahead


def test_centered_straddles_cursor():
    t0, t1, nf = _compute_slab(0.0, 10.0, cursor=5.0, depth_s=2.0, anchor="Centered", full=False)
    assert (t0, t1) == pytest.approx((4.0, 6.0)) and nf is False


def test_degenerate_window_kept_nonzero():
    t0, t1, _ = _compute_slab(0.0, 10.0, cursor=0.0, depth_s=2.0, anchor="Trailing (stream)",
                              full=False)
    assert t1 > t0                                     # never a zero-width slab at the very start


# ----------------------------------------------------------------- projection / picking
def test_project_points_identity():
    mvp = np.eye(4)
    screen, wv = _project_points([[0, 0, 0], [1, 1, 0]], mvp, 100, 100)
    assert wv[0] == 1.0
    assert screen[0] == pytest.approx([50.0, 50.0])    # origin maps to viewport centre
    assert screen[1] == pytest.approx([100.0, 0.0])    # +x right, +y up (screen y flipped)


def test_nearest_screen_threshold():
    screen = np.array([[50.0, 50.0], [10.0, 10.0]])
    wv = np.array([1.0, 1.0])
    i, d = _nearest_screen(screen, wv, (52.0, 52.0), max_dist=5.0)
    assert i == 0 and d < 5.0
    i2, _ = _nearest_screen(screen, wv, (200.0, 200.0), max_dist=5.0)
    assert i2 == -1                                    # nothing within the click radius
    # the closest point is behind the camera (w<=0) → it is skipped, the in-front one wins
    i3, _ = _nearest_screen(screen, np.array([-1.0, 1.0]), (11.0, 11.0), max_dist=5.0)
    assert i3 == 1


def test_aabb_from_opposite_corners():
    assert _aabb_from_opposite((0, 0, 0), (10, 5, 2)) == (0, 10, 0, 5, 0, 2)
    assert _aabb_from_opposite((10, 5, 2), (0, 0, 0)) == (0, 10, 0, 5, 0, 2)   # order-independent


# ----------------------------------------------------------------- CAD measure stats
def test_measure_two_points_distance_and_frequency():
    st = _measure_stats([(0, 0, 0.0), (30, 40, 0.005)], cycles=1.0)
    assert st["n"] == 2
    assert st["dr_px"] == pytest.approx(50.0)          # 3-4-5 triangle
    assert st["dt_s"] == pytest.approx(0.005)
    assert st["freq_hz"] == pytest.approx(200.0)       # 1 cycle over 5 ms
    assert st["speed_px_s"] == pytest.approx(10000.0)


def test_measure_two_points_cycles_scale_frequency():
    st = _measure_stats([(0, 0, 0.0), (0, 0, 0.005)], cycles=2.0)
    assert st["freq_hz"] == pytest.approx(400.0)       # 2 cycles over 5 ms


def test_measure_multi_point_average_frequency():
    pts = [(0, 0, 0.0), (1, 0, 0.005), (2, 0, 0.010), (3, 0, 0.015)]
    st = _measure_stats(pts, cycles=1.0)
    assert st["n"] == 4
    assert st["avg_freq_hz"] == pytest.approx(200.0)   # (4-1)/0.015 s
    assert st["avg_period_s"] == pytest.approx(0.005)
    assert st["jitter_s"] == pytest.approx(0.0, abs=1e-9)
    assert st["path_px"] == pytest.approx(3.0)


def test_measure_points_sorted_by_time():
    # out-of-order clicks still yield a sensible average frequency
    st = _measure_stats([(0, 0, 0.010), (0, 0, 0.0), (0, 0, 0.005)], cycles=1.0)
    assert st["avg_freq_hz"] == pytest.approx(200.0)
