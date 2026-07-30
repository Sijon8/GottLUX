"""
Test the Multi-clip Overlay layout: two clips' event spaces blend into one image, each clip a
distinct colour, with a legend.
"""
import os

import pytest

from gottlux.synthetic import synthetic_scene, FlutterTarget


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_overlay_layout_blends_two_clips(app):
    from gottlux.app.multiclip import MultiClipViewer
    mc = MultiClipViewer()
    r1, _ = synthetic_scene(0.5, targets=[FlutterTarget()], seed=1); r1.name = "cam0_wide"
    r2, _ = synthetic_scene(0.5, targets=[FlutterTarget()], seed=2); r2.name = "cam1_narrow"
    mc._add_pane_with(r1); mc._add_pane_with(r2)
    assert "Overlay" in [mc.layout_cb.itemText(i) for i in range(mc.layout_cb.count())]

    mc.clock.set_cursor(0.2)
    mc.layout_cb.setCurrentText("Overlay")
    assert mc._overlay_last_rgb is not None
    assert mc._overlay_last_rgb.ndim == 3 and mc._overlay_last_rgb.shape[2] == 3
    # both clips appear in the colour legend
    assert "cam0_wide" in mc._overlay_legend.text() and "cam1_narrow" in mc._overlay_legend.text()

    # a clip's intensity frame is computed on demand regardless of visibility
    f = mc.panes[0].intensity_frame()
    assert f is not None and f.min() >= 0.0 and f.max() <= 1.0


def test_fuse_keyframes_across_range_lab_panes(app):
    """The 'add-clip' route: keyframe two Range-lab panes and converge across the sensor spaces."""
    import gottlux.core.photogrammetry as pg
    from gottlux.app.multiclip import MultiClipViewer
    from gottlux.core.dualview import ConvergedStudy

    mc = MultiClipViewer()
    r1, _ = synthetic_scene(0.6, targets=[FlutterTarget()], seed=1); r1.name = "cam0_wide"
    r2, _ = synthetic_scene(0.6, targets=[FlutterTarget()], seed=2); r2.name = "cam1_narrow"
    mc._add_pane_with(r1); mc._add_pane_with(r2)
    assert mc._range_lab_panes() == []                       # no Range-lab panes yet

    for pane, fov, ranges in ((mc.panes[0], 58.0, (10, 20)), (mc.panes[1], 40.0, (30, 50))):
        pane.set_view("Range lab")
        lab = pane._child
        lab.fov.setValue(fov); lab.size_m.setValue(0.22)
        f = pg.focal_px(fov, 320)
        for D in ranges:
            n = 0.22 * f / D
            lab._apply_box((150, 150, 150 + n, 150 + n))
            lab.ctl.set_cursor(0.2 if D == ranges[0] else 0.4)
            lab.dist.setValue(float(D)); lab._add_keyframe()

    labs = mc._range_lab_panes()
    assert len(labs) == 2 and [n for n, _ in labs] == ["cam0_wide", "cam1_narrow"]
    studies = [(n, lab._study()) for n, lab in labs]
    summary = ConvergedStudy(*studies[0], *studies[1]).summary()
    assert summary["pooled_fit"]["n"] == 4
    assert summary["fused_target_size_m"] == pytest.approx(0.22, abs=1e-2)


def test_multiclip_faithful_overlay_capture(app):
    from gottlux.app.multiclip import MultiClipViewer
    mc = MultiClipViewer()
    r1, _ = synthetic_scene(0.5, targets=[FlutterTarget()], seed=1); r1.name = "a"
    r2, _ = synthetic_scene(0.5, targets=[FlutterTarget()], seed=2); r2.name = "b"
    mc._add_pane_with(r1); mc._add_pane_with(r2)
    assert mc.capture_clock() is mc.clock                    # captures against its own clock
    assert mc.sensor_size() == (320, 320)
    frame = mc.capture_frame(0.2, size=(720, 720))           # faithful overlay at high res
    assert frame.shape == (720, 720, 3)
