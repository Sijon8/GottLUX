"""
Tests for the shared clock's In/Out selection and the transport selection bar
(gottlux.app.transport) — the range that drives Cut → .raw, Capture, and scoped Export.
"""
import os

import pytest


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_clock_selection_in_seconds():
    from gottlux.app.transport import TimeController
    c = TimeController(); c.set_range(0.0, 2.0)
    assert c.has_selection() is False
    c.set_selection(0.25, 0.75)
    assert (c.sel_t0(), c.sel_t1()) == (0.5, 1.5)
    assert c.has_selection() is True
    c.set_selection(0.9, 0.1)                       # reversed handles normalize
    assert c.sel_t0() < c.sel_t1()
    c.set_selection(0.0, 1.0)
    assert c.has_selection() is False


def test_transport_bar_has_selection_bar(app):
    from gottlux.app.transport import TimeController, TransportBar
    c = TimeController(); c.set_range(0.0, 1.0)
    bar = TransportBar(c)
    assert bar.selbar is not None                  # the In/Out bar is present by default
    bar.selbar.primaryChanged.emit(0.2, 0.8)       # dragging it updates the shared clock
    assert c.sel_lo == pytest.approx(0.2) and c.sel_hi == pytest.approx(0.8)
    # a bar can opt out of the selection row
    assert TransportBar(c, show_selection=False).selbar is None


def test_accum_window_direction():
    """The accumulation window opens ahead of the cursor by default and behind it when reversed."""
    from gottlux.app.transport import TimeController
    c = TimeController(); c.set_range(0.0, 10.0)
    c.set_accum(2.0); c.set_cursor(5.0)
    assert c.accum_back is False
    assert c.accum_window() == pytest.approx((5.0, 7.0))      # forward: [t, t+Δ]
    c.set_accum_back(True)
    assert c.accum_back is True
    assert c.accum_window() == pytest.approx((3.0, 5.0))      # backward: [t−Δ, t]
    # explicit t / dt / back overrides, and clamping to the range
    assert c.accum_window(t=1.0, dt=2.0, back=True) == pytest.approx((0.0, 1.0))   # clamps at t0
    assert c.accum_window(t=9.5, dt=2.0, back=False) == pytest.approx((9.5, 10.0))  # clamps at t1


def test_accum_dir_signal_and_rerender():
    """Flipping the direction announces it and pokes accumChanged so bound views re-render."""
    from gottlux.app.transport import TimeController
    c = TimeController()
    dirs, accums = [], []
    c.accumDirChanged.connect(dirs.append)
    c.accumChanged.connect(accums.append)
    c.set_accum_back(True)
    assert dirs == [True] and accums == [c.accum]
    c.set_accum_back(True)                          # idempotent: no duplicate signals
    assert dirs == [True] and len(accums) == 1


def test_transport_bar_accum_dir_toggle(app):
    from gottlux.app.transport import TimeController, TransportBar
    c = TimeController()
    bar = TransportBar(c)
    assert hasattr(bar, "accum_dir_btn")            # the direction toggle is shown with accum
    bar.accum_dir_btn.setChecked(True)              # clicking it flips the shared clock
    assert c.accum_back is True
    # views with their own windowing can hide it
    assert not hasattr(TransportBar(c, show_accum_dir=False), "accum_dir_btn")
