"""
Tests for gottlux.app.glview.GLView — the 3-D views' GLViewWidget whose middle-mouse pan
keeps working at the Top/Bottom orthographic views, where stock pyqtgraph's 'view-upright'
pan collapses to a no-op (the camera vector becomes parallel to z and the basis is degenerate).
"""
import os

import pytest


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _center(v):
    c = v.opts["center"]
    return (round(c.x(), 4), round(c.y(), 4), round(c.z(), 4))


def _make_view():
    from gottlux.app.glview import GLView
    if GLView is None:
        pytest.skip("OpenGL not available")
    try:
        v = GLView()
        v.resize(800, 440)
        return v
    except Exception:                                  # GL context can't be created headless
        pytest.skip("cannot create a GL widget in this environment")


def test_pan_moves_scene_at_top_view(app):
    """At the Top view (elevation 90°) a middle-drag must still move the look-at centre."""
    v = _make_view()
    v.setCameraPosition(elevation=90.0, azimuth=-90.0, distance=500)
    before = _center(v)
    v.pan(40, 20, 0, relative="view-upright")
    assert _center(v) != before                        # was a no-op with the stock widget


def test_pan_still_works_off_pole(app):
    """An oblique view delegates to pyqtgraph's implementation and pans as before."""
    v = _make_view()
    v.setCameraPosition(elevation=30.0, azimuth=45.0, distance=500)
    before = _center(v)
    v.pan(40, 20, 0, relative="view-upright")
    assert _center(v) != before
