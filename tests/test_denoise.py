"""
Tests for the live noise-filter suite (core.denoise) and its program-wide controller, plus the
bounded polarity_ratio accumulation mode that avoids dense-region blow-out.
"""
import os

import numpy as np
import pytest

from gottlux.core.accumulate import accumulate_frame
from gottlux.core.denoise import FilterSettings, filter_window
from gottlux.io.recording import EventWindow


def _win(x, y, p, t, W=64, H=64):
    return EventWindow(np.asarray(x, np.uint16), np.asarray(y, np.uint16),
                       np.asarray(p, np.uint8), np.asarray(t, np.int64), W, H)


def test_inactive_is_noop_identity():
    w = _win([1, 2], [1, 2], [1, 0], [0, 1000])
    assert filter_window(w, FilterSettings(enabled=False)) is w
    assert filter_window(w, FilterSettings(enabled=True)) is w   # nothing toggled → no-op


def test_polarity_filter():
    w = _win([1, 2, 3], [1, 2, 3], [1, 0, 1], [0, 1, 2])
    out = filter_window(w, FilterSettings(enabled=True, polarity="on"))
    assert out.n == 2 and (np.asarray(out.p) == 1).all()


def test_hot_pixel_filter():
    xs = [5] * 100 + [10, 11, 12]
    ys = [5] * 100 + [10, 11, 12]
    ps = [1] * 103
    ts = list(range(103))
    out = filter_window(_win(xs, ys, ps, ts),
                        FilterSettings(enabled=True, hot_pixel=True, hot_pct=99.0))
    coords = set(zip(np.asarray(out.x).tolist(), np.asarray(out.y).tolist()))
    assert (5, 5) not in coords and out.n == 3


def test_refractory_filter():
    # same pixel at t = 0, 100, 100000 µs; 1000 µs dead time → keep 0 and 100000, drop 100
    w = _win([3, 3, 3], [3, 3, 3], [1, 1, 1], [0, 100, 100000])
    out = filter_window(w, FilterSettings(enabled=True, refractory=True, refractory_us=1000))
    assert out.n == 2 and set(np.asarray(out.t).tolist()) == {0, 100000}


def test_baf_removes_isolated_keeps_clustered():
    # time-sorted: a neighbour cluster + one isolated event far away
    x = [10, 40, 11, 10]
    y = [10, 40, 10, 11]
    p = [1, 1, 1, 1]
    t = [0, 5, 10, 20]
    out = filter_window(_win(x, y, p, t), FilterSettings(enabled=True, baf=True, baf_dt_us=1000))
    coords = set(zip(np.asarray(out.x).tolist(), np.asarray(out.y).tolist()))
    assert (40, 40) not in coords            # isolated shot noise removed
    assert (11, 10) in coords and (10, 11) in coords   # correlated neighbours kept


def test_polarity_ratio_is_bounded():
    # a very dense pixel (1000 ON) must not exceed the [-1, 1] range
    x = [7] * 1000 + [8]
    y = [7] * 1000 + [8]
    p = [1] * 1000 + [0]
    t = list(range(1001))
    frame = accumulate_frame(_win(x, y, p, t), mode="polarity_ratio")
    assert frame.max() <= 1.0 and frame.min() >= -1.0
    assert np.isclose(frame[7, 7], 1.0)      # all-ON pixel → +1 regardless of density
    assert np.isclose(frame[8, 8], -1.0)     # all-OFF pixel → -1


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_filter_controller_and_mainwindow(app):
    from gottlux.app.filters import FilterController
    from gottlux.app.main import MainWindow
    fc = FilterController()
    fc.update(enabled=True, baf=True, baf_dt_us=3000)
    assert fc.settings.active()
    w = _win([10, 11, 40], [10, 10, 40], [1, 1, 1], [0, 10, 5])
    assert fc.apply(w).n <= 3
    # the main window builds with the shared filter suite + toolbar
    win = MainWindow()
    assert hasattr(win, "filters")
    win.filters.update(enabled=True, hot_pixel=True)   # fires changed → panels re-render safely
