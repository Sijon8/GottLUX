"""Regression tests for the button-and-action audit: every control found broken there,
exercised the way the GUI drives it."""
import os

import numpy as np
import pytest

from gottlux.io.recording import Recording

pytest.importorskip("PySide6")


@pytest.fixture
def app():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _rec(n=4000, dur=0.4, w=64, h=48, seed=0, name="clip"):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, dur, n) * 1e6).astype(np.int64)
    return Recording.from_events(rng.integers(0, w, n), rng.integers(0, h, n),
                                 rng.integers(0, 2, n), t, width=w, height=h, name=name)


@pytest.mark.parametrize("width", [320, 322, 1920])
def test_grabbed_frames_own_their_pixels(app, width):
    """BA-01: at a width divisible by 4 the frame was a view of a freed QImage."""
    from PySide6 import QtGui
    from gottlux.app.capture import qimage_to_rgb
    img = QtGui.QImage(width, 6, QtGui.QImage.Format_ARGB32)
    img.fill(QtGui.QColor(10, 200, 30))
    rgb = qimage_to_rgb(img)
    del img
    assert rgb.shape == (6, width, 3) and rgb.flags.owndata
    assert np.all(rgb == np.array([10, 200, 30], np.uint8))


def test_global_actions_have_a_clock_on_the_studio_tab(app):
    """BA-02: the Studio tab has no clock; Capture / Export / space used None."""
    from gottlux.app.main import MainWindow
    win = MainWindow()
    try:
        i = next(k for k in range(win.tabs.count()) if win.tabs.tabText(k) == "Studio")
        win.tabs.setCurrentIndex(i)
        assert win.active_clock() is win.clock
        assert win.clock in win._all_clocks()
    finally:
        win.close()
        win.deleteLater()


def test_multiclip_overlay_layout_keeps_the_panes(app):
    """BA-03: switching to Overlay deleted every pane with the old splitter."""
    from gottlux.app.multiclip import MultiClipViewer
    mc = MultiClipViewer()
    mc.set_recording(_rec(seed=1, name="a"))
    mc._add_pane_with(_rec(seed=2, name="b"), label="b")
    for layout in ("Overlay", "Grid", "Overlay", "Stacked"):
        mc.layout_cb.setCurrentText(layout)
        app.processEvents()
        assert all(p.isVisibleTo(mc) or layout == "Overlay" for p in mc.panes)
        for p in mc.panes:
            p.objectName()                      # raises if the C++ pane was deleted
    mc.close()


def test_multiclip_selection_maps_onto_each_clip(app):
    """BA-07: the transport's .raw actions said 'load a recording first' on the slate."""
    from gottlux.app.multiclip import MultiClipViewer
    mc = MultiClipViewer()
    a, b = _rec(seed=3, name="a"), _rec(seed=4, name="b")
    mc.set_recording(a)
    mc._add_pane_with(b, label="b")
    mc.panes[1].set_offset(0.1)
    assert mc.rec is a
    clips = mc.slate_clips([(0.05, 0.2)])
    assert [c[0] for c in clips] == ["a", "b"]
    (a0, a1), = clips[0][2]
    (b0, b1), = clips[1][2]
    assert a0 == pytest.approx(max(0.05, a.t_start_s)) and a1 == pytest.approx(0.2)
    assert b0 == pytest.approx(0.15) and b1 == pytest.approx(min(0.3, b.t_stop_s))
    mc.close()


def test_long_running_workers_are_waited_for_at_exit():
    """BA-04: a plain QThread is not waited for, so quitting mid-export aborted."""
    from gottlux.app.batchtrim import _TrimWorker
    from gottlux.app.loader import Worker
    from gottlux.app.studio import _ExportWorker
    from gottlux.app.videoexport import _Job
    assert issubclass(_TrimWorker, Worker) and issubclass(_ExportWorker, Worker)
    assert issubclass(_Job, Worker)


def test_range_lab_range_estimate_exists():
    """BA-05: the analysis video called a function photogrammetry does not have."""
    from gottlux.core import photogrammetry as pg
    r = float(pg.range_for_pixels(0.3, 12.0, 58.0, 320))
    assert np.isfinite(r) and r > 0


def test_composer_save_then_load_keeps_every_cell(app, tmp_path, monkeypatch):
    """BA-08: a second cell of one file got an unloadable '<path>_2' source."""
    from PySide6 import QtWidgets
    from gottlux.app.canvas import CanvasComposerWindow
    from gottlux.io import writer
    import gottlux as eb
    r = _rec(seed=5)
    raw = str(tmp_path / "clip.raw")
    writer.write_raw(raw, np.asarray(r.x), np.asarray(r.y), np.asarray(r.p),
                     np.asarray(r.t), width=r.width, height=r.height)
    rec = eb.load(raw)
    win = CanvasComposerWindow()
    win.add_clip_recording(rec)
    win.add_clip_recording(rec)
    win.add_visualization_grid(1, ["", "trails"])
    assert len(win.spec.clips) == 3
    assert len({c.source for c in win.spec.clips}) == 1
    spec_path = str(tmp_path / "comp.gottlux-canvas.json")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (spec_path, "")))
    win._save_composition()
    win2 = CanvasComposerWindow()
    win2.load_composition(spec_path)
    assert len(win2.spec.clips) == 3
    assert [c.look for c in win2.spec.clips] == ["", "", "trails"]
    win.close(); win2.close()


def test_fusion_smoothing_keeps_a_short_series_length():
    """BA-09: a kernel longer than a short clip's rate series lengthened it."""
    from gottlux.io.fusion import _detrend_norm, _smooth
    a = np.random.default_rng(6).random(30)
    assert _smooth(a, 2.0, 0.01).shape == (30,)
    assert _detrend_norm(a, 2.0, 0.01).shape == (30,)


def test_studio_reports_a_malformed_recipe(app, tmp_path, monkeypatch):
    """BA-10: the JSON error escaped the slot."""
    from PySide6 import QtWidgets
    from gottlux.app.studio import StudioPanel
    bad = tmp_path / "bad.shot.json"
    bad.write_text("{nope", encoding="utf-8")
    warned = []
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(bad), "")))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.append(a[1])))
    panel = StudioPanel()
    panel._load_recipe()
    assert warned == ["Load recipe"]
    panel.close()


def test_null_gl_snapshot_is_not_reported_saved(tmp_path):
    """BA-12: 'Saved' with nothing written when there is no framebuffer."""
    from PySide6 import QtGui
    from gottlux.app.exporting import save_gl_snapshot

    class NoGL:
        def grabFramebuffer(self):              # noqa: N802 (Qt API)
            return QtGui.QImage()
    out = str(tmp_path / "snap.png")
    assert save_gl_snapshot(NoGL(), out) == [] and not os.path.exists(out)


def test_cutting_the_whole_clip_keeps_its_last_event(tmp_path):
    """BA-13: [t0, t1) dropped the final event of a whole-clip cut."""
    from gottlux.io import writer
    import gottlux as eb
    r = _rec(seed=7)
    out = str(tmp_path / "whole.raw")
    n = writer.cut_clip(r, out, t0=r.t_start_s, t1=r.t_stop_s)
    assert n == r.n and eb.load(out).n == r.n


def test_timeline_selection_groups_start_disabled(app):
    """BA-14 / GL-30: with nothing selected the trim and cell controls stayed live."""
    from gottlux.app.timeline import TimelineEditor
    ed = TimelineEditor(recordings=[])
    assert not ed._clip_group.isEnabled() and not ed._cell_group.isEnabled()
    ed._cut_selected()                          # no selection: a message, not silence
    assert "Select a clip" in ed.status.text()
    ed.close()
