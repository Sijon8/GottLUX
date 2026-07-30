"""
Smoke tests for the generic screen/video capture (gottlux.app.capture): the pixel grab and the
capture dialog construct and wire up correctly (offscreen). The actual muxing/run path is
exercised by gottlux.viz.video tests.
"""
import os

import numpy as np
import pytest

from gottlux.synthetic import synthetic_scene, FlutterTarget


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_qimage_to_rgb_roundtrip(app):
    from PySide6 import QtGui
    from gottlux.app.capture import qimage_to_rgb
    img = QtGui.QImage(24, 16, QtGui.QImage.Format.Format_RGB888)
    img.fill(QtGui.QColor(10, 20, 30))
    arr = qimage_to_rgb(img)
    assert arr.shape == (16, 24, 3) and arr.dtype == np.uint8
    assert tuple(int(v) for v in arr[0, 0]) == (10, 20, 30)


def test_grab_widget(app):
    from PySide6 import QtWidgets
    from gottlux.app.capture import grab_widget
    w = QtWidgets.QLabel("x"); w.resize(40, 30)
    rgb = grab_widget(w)
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8


def test_faithful_capture_frame_is_highres_and_settings_accurate(app):
    from gottlux.app.viewer import LiveViewer
    from gottlux.app.transport import TimeController
    rec, _ = synthetic_scene(duration_s=0.5, targets=[FlutterTarget(flutter_hz=180)], seed=1)
    clk = TimeController(); clk.set_range(rec.t_start_s, rec.t_stop_s)
    v = LiveViewer(clk); v.set_recording(rec)
    assert v.sensor_size() == (320, 320)
    native = v.capture_frame(0.2)
    assert native.shape == (320, 320, 3) and native.dtype == np.uint8
    hd = v.capture_frame(0.2, size=(1920, 1080))        # true high-res, not a screen grab
    assert hd.shape == (1080, 1920, 3)
    # the tuned settings change the rendered pixels (mode + colormap)
    v.mode.setCurrentText("polarity"); v.cmap.setCurrentText("coolwarm")
    pol = v.capture_frame(0.2, size=(200, 200))
    assert pol.shape == (200, 200, 3) and pol.sum() > 0


def test_capture_dialog_resolution_targets(app):
    from PySide6 import QtWidgets
    from gottlux.app.capture import ScreenCaptureDialog
    rec, _ = synthetic_scene(duration_s=0.5, targets=[FlutterTarget()], seed=1)
    render = lambda t, dt, size: np.zeros((size[1], size[0], 3), np.uint8) if size else None
    ctx = dict(rec=rec, target=QtWidgets.QLabel("v"), set_cursor=lambda t: None,
               view="Live viewer", accum=0.02, render=render, sensor_wh=(320, 320))
    dlg = ScreenCaptureDialog(None, ctx)
    opts = [dlg.res.itemText(i) for i in range(dlg.res.count())]
    assert any("Faithful" in o for o in opts) and any("On-screen" in o for o in opts)
    dlg.res.setCurrentText("Faithful · 1080p"); assert dlg._target_size() == (1080, 1080)
    dlg.res.setCurrentText("Faithful · native"); assert dlg._target_size() == (320, 320)
    dlg.res.setCurrentText("On-screen (grab)"); assert dlg._target_size() is None
    # a view with no faithful render only offers on-screen
    ctx2 = dict(ctx); ctx2["render"] = None
    dlg2 = ScreenCaptureDialog(None, ctx2)
    assert [dlg2.res.itemText(i) for i in range(dlg2.res.count())] == ["On-screen (grab)"]


def test_capture_dialog_constructs_and_wires(app):
    from PySide6 import QtWidgets
    from gottlux.app.capture import ScreenCaptureDialog
    rec, _ = synthetic_scene(duration_s=1.0, targets=[FlutterTarget()], seed=1)
    seen = []
    ctx = dict(rec=rec, target=QtWidgets.QLabel("v"), set_cursor=lambda t: seen.append(t),
               view="Live viewer", t0=rec.t_start_s, t1=rec.t_stop_s, accum=0.02,
               fields={"Recording": rec.name, "Sensor": "GenX320"})
    dlg = ScreenCaptureDialog(None, ctx)
    assert dlg.region is None
    dlg._region_picked((5, 6, 20, 18))
    assert dlg.region == (5, 6, 20, 18)
    dlg.trim.set_range(0.25, 0.75); dlg._trim_to_spins(0.25, 0.75)
    assert dlg.out_s.value() > dlg.in_s.value()
