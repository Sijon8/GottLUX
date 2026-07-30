"""
Offscreen GUI smoke tests — the main window builds, every tab is present, and the headline
Multi-clip slate loads several recordings side-by-side on one clock and renders each one.

These run headless via the Qt 'offscreen' platform; skipped if PySide6/pyqtgraph are absent.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets  # noqa: E402

from gottlux.io.recording import Recording  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return a


def _synth(n, dur_s, w=96, h=96, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, dur_s, n) * 1e6).astype(np.int64)
    x = rng.integers(0, w, n)
    y = rng.integers(0, h, n)
    p = rng.integers(0, 2, n)
    return Recording.from_events(x, y, p, t, width=w, height=h, name=f"clip_{dur_s}s")


def test_main_window_has_all_tabs(app):
    from gottlux.app.main import MainWindow
    win = MainWindow()
    names = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    for expected in ("Live viewer", "Multi-clip", "Event-rate tower",
                     "Space-time 3D", "Flutter workbench", "Sandbox", "Timeline"):
        assert expected in names, f"{expected} tab missing; have {names}"
    # the full declared order, ending on the Timeline (compose/export) tab
    assert names == list(MainWindow._TAB_NAMES)
    assert len(names) == 10 and names[-1] == "Timeline"


def test_fusion_lab_tab_present(app):
    from gottlux.app.main import MainWindow
    win = MainWindow()
    names = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert "Fusion lab" in names, f"Fusion lab tab missing; have {names}"


def test_fusion_lab_loads_audio_and_aligns(app, tmp_path):
    """The Fusion lab binds a recording + an audio clip, recovers an offset, and draws both
    envelopes — headlessly (no file dialogs), exercising the plotting/alignment code paths."""
    import numpy as np

    from gottlux.app.fusionlab import FusionLab
    from gottlux.app.transport import TimeController
    from gottlux.io import fusion
    if not getattr(__import__("gottlux.app.fusionlab", fromlist=["_HAVE_PG"]), "_HAVE_PG", False):
        pytest.skip("pyqtgraph not available")

    clk = TimeController()
    fl = FusionLab(clk)
    rec = _synth(40000, 8.0, w=64, h=64, seed=9)
    clk.set_range(rec.t_start_s, rec.t_stop_s)
    fl.set_recording(rec)
    assert fl.c_ebs.xData is not None and len(fl.c_ebs.xData) > 0

    sr = 16000
    t = np.arange(int(9.0 * sr)) / sr
    x = (np.sin(2 * np.pi * 130 * t) * (0.05 + np.exp(-0.5 * ((t - 2.0) / 0.3) ** 2)))
    wav = str(tmp_path / "aud.wav")
    fusion.write_wav(wav, x, sr, subtype="int16")
    fl.audio = fusion.read_wav(wav)               # bypass the file dialog
    fl._auto_align()
    assert fl.result is not None
    assert fl.c_aud.xData is not None and len(fl.c_aud.xData) > 0
    fl.offset_sp.setValue(1.0)                     # a manual nudge recomputes the plan
    assert fl.result is not None and abs(fl.result.offset_s - 1.0) < 1e-6


def test_timeline_tab_hosts_the_editor(app):
    """The tenth tab hosts the TimelineEditor widget itself (not a launcher button)."""
    from gottlux.app.main import MainWindow
    from gottlux.app.timeline import TimelineEditor
    win = MainWindow()
    idx = [win.tabs.tabText(i) for i in range(win.tabs.count())].index("Timeline")
    assert isinstance(win.tabs.widget(idx), TimelineEditor)
    assert win.tabs.widget(idx) is win.timeline
    assert win.timeline in win.panels                     # wired like every other panel


def test_timeline_tab_is_a_video_editor_on_its_own_clock(app):
    """The Timeline tab is laid out like a video editor — preview viewport, transport,
    track lanes, stacked in one vertical splitter — and drives its own clock, so the
    spacebar and the tab-switch pause act on it rather than on the app clock."""
    from PySide6 import QtCore

    from gottlux.app.main import MainWindow
    from gottlux.app.timeline import PreviewView, TrackLanes
    from gottlux.app.transport import TransportBar
    win = MainWindow()
    ed = win.timeline
    assert ed.split.orientation() == QtCore.Qt.Vertical
    kinds = [type(ed.split.widget(i)) for i in range(ed.split.count())]
    assert kinds[1] is TransportBar and kinds[2] is TrackLanes
    assert isinstance(ed.preview, PreviewView)
    assert ed.stage.currentWidget() is ed.preview        # playback mode by default

    # its own clock: the tab reports it, so the app routes play/pause and Sync here
    assert ed.capture_clock() is ed.clock is not win.clock
    idx = [win.tabs.tabText(i) for i in range(win.tabs.count())].index("Timeline")
    win.tabs.setCurrentIndex(idx)
    assert win.active_clock() is ed.clock
    ed.clock.play()
    assert ed.clock.playing
    win._pause_all_clocks()
    assert not ed.clock.playing

    rec = _synth(6000, 0.5, seed=45)
    win._on_loaded(rec)
    win._pause_all_clocks()
    ed.clock.set_cursor(0.2)
    frame = ed.current_frame()                           # the one engine render path
    assert frame.shape == (rec.height, rec.width, 3) and frame.max() > 0


def test_timeline_tab_seeds_only_an_empty_timeline(app):
    """Loading a recording auto-seeds the Timeline tab's first clip; a later load refreshes
    an untouched auto-seed but never clobbers a timeline the user has edited."""
    from gottlux.app.main import MainWindow
    win = MainWindow()
    assert win.timeline.clips == []

    rec = _synth(8000, 0.6, seed=41)
    win._on_loaded(rec)
    win._pause_all_clocks()
    assert len(win.timeline.clips) == 1 and win.timeline.clips[0]["rec"] is rec

    # a second load refreshes the (still untouched) auto-seed to the new recording
    rec2 = _synth(6000, 0.5, seed=42)
    win._on_loaded(rec2)
    win._pause_all_clocks()
    assert len(win.timeline.clips) == 1 and win.timeline.clips[0]["rec"] is rec2

    # once the user edits the timeline (a trim here), loads leave it alone
    win.timeline.select(0)
    win.timeline._on_trim(0.25, 0.75)
    rec3 = _synth(6000, 0.4, seed=43)
    win._on_loaded(rec3)
    win._pause_all_clocks()
    assert len(win.timeline.clips) == 1
    assert win.timeline.clips[0]["rec"] is rec2           # the edited timeline survived
    assert win.timeline.clips[0]["t0"] > 0.0


def test_timeline_tab_accepts_drops_and_shows_hint(app):
    """The Timeline tab's embedded editor accepts OS drops, and while the timeline is
    empty its track lanes paint the muted drop hint (the discoverability affordance)."""
    from gottlux.app.main import MainWindow
    from gottlux.app.timeline import _DROP_HINT, TrackLanes
    win = MainWindow()
    ed = win.timeline
    assert ed.acceptDrops()
    assert isinstance(ed.lanes, TrackLanes) and ed.clips == []
    assert "Drop recordings here" in _DROP_HINT
    ed.lanes.resize(800, 140)
    assert not ed.lanes.grab().isNull()         # the empty-state paint path runs clean


def test_timeline_legacy_dialog_still_constructs(app):
    """The historical dialog entry point still works: a thin wrapper embedding the same
    editor, forwarding the editor's surface, closing itself on the editor's done signal."""
    from PySide6 import QtWidgets as qtw

    from gottlux.app.timeline import TimelineEditor, TimelineEditorDialog
    rec = _synth(5000, 0.4, seed=44)
    dlg = TimelineEditorDialog(recordings=[rec])
    assert isinstance(dlg.editor, TimelineEditor)
    assert len(dlg.clips) == 1                            # forwarded editor surface
    dlg.select(0)
    dlg._on_trim(0.1, 0.9)                                # forwarded handlers still drive it
    assert dlg.clips[0]["t0"] > 0.0
    dlg.editor.done.emit()                                # a finished stitch/hand-off …
    assert dlg.result() == qtw.QDialog.Accepted           # … accepts the dialog


def test_version_shown_in_window_titles(app):
    """The running build is identifiable: 'GottLUX <version>' appears in the main window,
    the quick viewer, and the boot splash title line."""
    import gottlux
    from gottlux.app.main import MainWindow
    from gottlux.app.quickview import QuickViewWindow
    from gottlux.app.splash import BootSplash

    tag = f"GottLUX {gottlux.__version__}"
    win = MainWindow()
    assert tag in win.windowTitle()
    qv = QuickViewWindow()
    assert tag in qv.windowTitle()
    qv.close()
    splash = BootSplash()
    labels = [lbl.text() for lbl in splash.findChildren(QtWidgets.QLabel)]
    assert tag in labels
    splash.close()


def test_split_view_record_targets_focused_pane(app):
    """In split view, the capture/record target follows the focused pane (it used to be stuck on
    the left pane, so recording 'Current view' grabbed the wrong tab)."""
    from gottlux.app.main import MainWindow
    win = MainWindow(); win.show()
    assert win._active_panel() is win.tabs.currentWidget()          # default: the left pane
    win._toggle_compare(True)                                       # split on → second pane appears
    win.tabs2.setCurrentIndex(3)
    win.tabs2.currentWidget().setFocus()
    QtWidgets.QApplication.processEvents()
    assert win._active_panel() in win._compare_panels              # now the focused right pane
    vw = win._active_view_widget()                                  # a real, grabbable view widget
    assert vw is not None and hasattr(vw, "grab")


def test_spacetime_frame_and_markers(app):
    """The space-time view draws a thin sensor-plane frame + axis markers (replacing the old filled
    plane); both toggle off, and the bezel width is adjustable."""
    from gottlux.app.spacetime import SpaceTimeView
    from gottlux.app.transport import TimeController
    rec = _synth(30000, 1.0, w=128, h=96, seed=11)
    clk = TimeController(); clk.set_range(rec.t_start_s, rec.t_stop_s)
    st = SpaceTimeView(clk)
    if st.view is None:
        pytest.skip("OpenGL not available")
    st.set_recording(rec); st.show()
    clk.set_accum(0.05); clk.set_cursor(0.5)
    st._render(force=True)
    assert not hasattr(st, "front_plane")              # the ugly filled plane is gone
    assert st.front_frame.visible()                    # frame + markers shown by default
    assert all(t.visible() for t in st._decor_labels)
    st.frame_chk.setChecked(False)
    assert not st.front_frame.visible()                # frame toggles off
    st.markers_chk.setChecked(False)
    assert not any(t.visible() for t in st._decor_labels)   # markers toggle off
    st.frame_chk.setChecked(True)
    st.bezel_sp.setValue(6)                            # bezel width is adjustable
    assert float(st.front_frame.width) == 6.0


def test_multiclip_side_by_side_renders(app):
    from gottlux.app.multiclip import MultiClipViewer
    mc = MultiClipViewer()
    a = _synth(30000, 2.0, seed=1)
    b = _synth(24000, 1.4, seed=2)
    mc.set_recording(a)               # seeds clip 0
    mc._add_pane_with(b, label="second")
    assert len(mc.panes) == 2

    # the default layout is Stacked (views one above another)
    assert mc.layout_cb.currentText() == "Stacked"
    assert mc._cols() == 1

    # the shared timeline spans the longest clip
    assert mc.clock.t1 >= 1.9

    # a per-clip slate offset shifts where the clip sits on the shared timeline
    mc.panes[1].set_offset(0.3)
    mc.clock.set_cursor(0.8)
    mc.show()
    for p in mc.panes:
        p.sync()
        assert p.current_rgb() is not None
        assert p.current_rgb().ndim == 3 and p.current_rgb().shape[2] == 3

    # all layouts work
    for which in ("Side-by-side", "Grid", "Stacked"):
        mc.layout_cb.setCurrentText(which)
        mc._relayout()

    # removing a clip recomputes the timeline
    mc._remove_pane(mc.panes[1])
    assert len(mc.panes) == 1


def test_multiclip_mixed_sensor_geometry(app):
    """A GenX320 (320×320) clip and an IMX636 (1280×720) clip stack together fine."""
    from gottlux.app.multiclip import MultiClipViewer
    mc = MultiClipViewer()
    genx320 = _synth(20000, 1.0, w=320, h=320, seed=4)
    imx636 = _synth(60000, 1.0, w=1280, h=720, seed=5)
    mc.set_recording(genx320)
    mc._add_pane_with(imx636, label="imx636")
    mc.show()
    for p in mc.panes:
        p.sync()
        assert p.current_rgb() is not None
    # each pane keeps its own resolution
    assert mc.panes[0].rec.width == 320 and mc.panes[0].rec.height == 320
    assert mc.panes[1].rec.width == 1280 and mc.panes[1].rec.height == 720
    # composite stacks them, padding to the wider frame
    comp = MultiClipViewer._compose([p.current_rgb() for p in mc.panes], vertical=True)
    assert comp.shape[1] == 1280 and comp.ndim == 3


def test_transport_play_button_geometry_constant_across_toggle(app):
    """Play ↔ pause used to swap two Unicode glyphs from *different* fallback fonts, so the
    button changed size on every toggle. With painted icons + a fixed size it must not move."""
    from gottlux.app.transport import TimeController, TransportBar
    c = TimeController(); c.set_range(0.0, 1.0)
    bar = TransportBar(c)
    assert not bar.play_btn.icon().isNull()          # a painted icon, not glyph text
    assert bar.play_btn.text() == ""
    before = (bar.play_btn.size(), bar.play_btn.sizeHint())
    bar._reflect_play(True)                          # -> pause icon
    after_pause = (bar.play_btn.size(), bar.play_btn.sizeHint())
    bar._reflect_play(False)                         # -> play icon again
    after_play = (bar.play_btn.size(), bar.play_btn.sizeHint())
    assert before == after_pause == after_play
    # the accumulation-direction toggle is width-frozen too (no row jitter)
    assert bar.accum_dir_btn.minimumWidth() == bar.accum_dir_btn.maximumWidth()
    w = bar.accum_dir_btn.width()
    c.set_accum_back(True)
    assert bar.accum_dir_btn.width() == w


def test_clock_loops_at_end(app):
    import time
    from gottlux.app.transport import TimeController
    c = TimeController()
    c.set_range(0.0, 1.0)
    c.set_fps(30.0)               # speed = 1.0 recording-second per wall-second
    c.set_cursor(0.95)
    c.set_loop(True)
    c.play()
    c._last_wall = time.perf_counter() - 0.5   # force a +0.5 s advance → overshoots the end
    c._tick()
    assert c.playing, "loop should keep playing past the end"
    assert c.cursor < 0.95, "cursor should wrap back toward the start"


def test_clock_without_loop_stops_at_end(app):
    import time
    from gottlux.app.transport import TimeController
    c = TimeController()
    c.set_range(0.0, 1.0)
    c.set_fps(30.0)
    c.set_cursor(0.95)
    c.set_loop(False)
    c.play()
    c._last_wall = time.perf_counter() - 0.5
    c._tick()
    assert not c.playing
    assert abs(c.cursor - 1.0) < 1e-6


def test_multiclip_per_function_views(app):
    """Each clip can be rendered through any function tab; the toolbar switches them all."""
    from gottlux.app.multiclip import MultiClipViewer
    mc = MultiClipViewer()
    a = _synth(20000, 1.0, w=128, h=128, seed=7)
    b = _synth(16000, 0.8, w=128, h=128, seed=8)
    mc.set_recording(a)
    mc._add_pane_with(b, label="b")
    mc.show()
    # switch every clip to a function view via the toolbar
    for view in ("Event-rate", "Space-time", "Range lab", "Event frame"):
        mc.view_cb.setCurrentText(view)
        for p in mc.panes:
            assert p.view_sel.currentText() == view
            p.sync()
            if view != "Event frame":
                assert p._child is not None    # the function panel was instantiated
    # an individual pane can be overridden independently
    mc.panes[0].set_view("Live viewer")
    assert mc.panes[0]._child is not None
    assert mc.panes[1].view_sel.currentText() == "Event frame"
    # driving the master clock updates every pane (frame + hosted views) without error
    mc.clock.set_cursor(0.4)
    for p in mc.panes:
        p.sync()


def test_multiclip_loops_by_default(app):
    from gottlux.app.multiclip import MultiClipViewer
    mc = MultiClipViewer()
    assert mc.clock.loop is True
    assert mc.loop_chk.isChecked()
    mc.loop_chk.setChecked(False)
    assert mc.clock.loop is False


def test_multiclip_out_of_range_pane(app):
    from gottlux.app.multiclip import MultiClipViewer
    mc = MultiClipViewer()
    a = _synth(10000, 0.5, seed=3)
    mc.set_recording(a)
    mc.show()
    # push the cursor far past this clip's end (with a big negative offset on the clip)
    mc.panes[0].set_offset(0.0)
    mc.clock.set_range(0.0, 5.0)
    mc.clock.set_cursor(4.0)
    mc.panes[0].sync()
    assert "outside" in mc.panes[0].busy.text().lower()


# ====================================================================== the Tools menu
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tools_menu(win):
    """The main window's Tools menu, verified present in the menu bar (None if absent).

    Deliberately avoids ``QAction.menu()``: under PySide6, letting the temporary action
    wrapper from ``menuBar().actions()`` be garbage-collected after ``.menu()`` was called
    on it invalidates the underlying C++ QMenu (shiboken ownership propagation)."""
    titles = [a.text().replace("&", "") for a in win.menuBar().actions()]
    menu = getattr(win, "tools_menu", None)
    return menu if ("Tools" in titles and menu is not None) else None


def test_tools_menu_actions_and_canvas_composer_seeding(app):
    """The Tools menu carries the three instruments, and 'Canvas composer…' opens the
    composer seeded with the current recording as its first clip."""
    from gottlux.app.canvas import CanvasComposerWindow
    from gottlux.app.main import MainWindow
    win = MainWindow()
    menu = _tools_menu(win)
    assert menu is not None, "the Tools menu is missing"
    texts = [a.text() for a in menu.actions()]
    for expected in ("Canvas composer…", "Run user script on current view…",
                     "Export tool bundle…"):
        assert expected in texts, f"{expected} missing from Tools; have {texts}"

    # with no recording, the composer opens empty
    cw = win._open_canvas_composer()
    assert isinstance(cw, CanvasComposerWindow) and len(cw.spec.clips) == 0
    cw.close()

    # with a recording loaded, it arrives as the first clip (same object, no re-decode)
    rec = _synth(20000, 1.0, seed=21)
    win._on_loaded(rec)
    win._pause_all_clocks()
    cw = win._open_canvas_composer()
    assert len(cw.spec.clips) == 1
    assert cw.recs[cw.spec.clips[0].source] is rec
    cw.close()


def test_tools_run_user_script_end_to_end(app, tmp_path, monkeypatch):
    """Triggering 'Run user script…' (file dialog monkeypatched to the bundled example)
    runs the script on exactly the In/Out selection + ROI and reports the run folder."""
    from gottlux.app.main import MainWindow
    win = MainWindow()
    rec = _synth(30000, 1.0, seed=22)
    win._on_loaded(rec)
    win._pause_all_clocks()
    win.clock.set_selection(0.25, 0.75)                 # the In/Out selection scopes the run
    win.viewer.roiChanged.emit((8, 8, 64, 64))          # the live viewer's ROI rides along

    script = os.path.join(_REPO, "examples", "user_script_example.py")
    monkeypatch.chdir(tmp_path)                         # in-memory rec → run folder in cwd
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (script, "")))
    infos, errors = [], []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: infos.append(a)))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical",
                        staticmethod(lambda *a, **k: errors.append(a)))
    act = next(a for a in _tools_menu(win).actions()
               if a.text().startswith("Run user script"))
    act.trigger()
    assert not errors, f"user script failed: {errors}"
    assert infos, "the results dialog was not shown"

    run_dirs = [d for d in os.listdir(tmp_path) if d.startswith("gottlux_script_")]
    assert len(run_dirs) == 1
    files = set(os.listdir(tmp_path / run_dirs[0]))
    assert "README.md" in files
    assert "polarity_rates.npz" in files                # written by the script itself
    msg = infos[0][2]                                   # (parent, title, text)
    assert run_dirs[0] in msg                           # the dialog shows the run folder
    # provenance records the selection window + ROI the GUI captured
    readme = (tmp_path / run_dirs[0] / "README.md").read_text(encoding="utf-8")
    assert "8,8,64,64" in readme
    t0, t1 = win.clock.sel_t0(), win.clock.sel_t1()
    assert f"[{t0:g}, {t1:g}] s" in readme


def test_tools_run_user_script_error_shown_not_raised(app, tmp_path, monkeypatch):
    """A broken script surfaces as an error dialog — the GUI never sees the exception."""
    from gottlux.app.main import MainWindow
    win = MainWindow()
    win._on_loaded(_synth(5000, 0.4, seed=23))
    win._pause_all_clocks()
    bad = tmp_path / "bad_script.py"
    bad.write_text("def process(win, ctx):\n    raise RuntimeError('boom')\n",
                   encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(bad), "")))
    errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical",
                        staticmethod(lambda *a, **k: errors.append(a)))
    win._run_user_script()                              # must not raise
    assert errors and "boom" in errors[0][2]


def test_tools_export_tool_bundle_captures_live_viz(app, tmp_path, monkeypatch):
    """'Export tool bundle…' writes a full bundle; for viz_config the live viewer's
    mode/colormap/tone-map and the clock's accumulation are baked into the scripts."""
    import json

    from gottlux.app.main import MainWindow
    win = MainWindow()
    win._on_loaded(_synth(20000, 0.8, seed=24))
    win._pause_all_clocks()
    win.viewer.mode.setCurrentText("polarity")          # auto-switches the cmap to coolwarm
    monkeypatch.chdir(tmp_path)                         # in-memory rec → bundle in cwd
    infos = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: infos.append(a)))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical",
                        staticmethod(lambda *a, **k: pytest.fail(f"export failed: {a}")))
    monkeypatch.setattr(
        QtWidgets.QInputDialog, "getItem",
        staticmethod(lambda parent, title, label, items, *a, **k:
                     (next(i for i in items if i.startswith("viz_config")), True)))
    act = next(a for a in _tools_menu(win).actions()
               if a.text().startswith("Export tool bundle"))
    act.trigger()
    assert infos, "the bundle dialog was not shown"

    bundles = [d for d in os.listdir(tmp_path) if "_tool_viz_config_" in d]
    assert len(bundles) == 1
    bundle = tmp_path / bundles[0]
    assert str(bundle) in infos[0][2] or bundles[0] in infos[0][2]
    names = set(os.listdir(bundle))
    assert {"data.h5", "run_viz_config.py", "run_viz_config.m",
            "README.md", "provenance.json"} <= names
    prov = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    assert prov["parameters"]["viz_mode"] == "polarity"
    assert prov["parameters"]["viz_cmap"] == "coolwarm"
    assert prov["parameters"]["viz_tonemap"] == win.viewer.expr.currentText()
    assert float(prov["parameters"]["viz_accum_ms"]) == \
        pytest.approx(win.clock.accum * 1e3)


# ====================================================================== quick viewer
def _wait_quickview_loaded(win, app, timeout_s=60.0):
    """Pump the event loop until the quick viewer's worker thread delivers the Recording."""
    import time
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if win.rec is not None and not getattr(win.rec, "is_preview", False):
            return True
        time.sleep(0.02)
    return False


def test_quickview_opens_h5_and_capture_folder(app, tmp_path):
    """The quick viewer opens every loadable type — here an .h5 conversion and a capture
    folder holding a .raw — and 'Open in full GottLUX' stays armed for each."""
    from gottlux.app.quickview import QuickViewWindow
    from gottlux.io import hdf5 as h5io
    from gottlux.io import writer

    rec = _synth(15000, 0.5, seed=31)
    h5 = str(tmp_path / "tiny.h5")
    h5io.write_hdf5(rec, h5)

    win = QuickViewWindow()
    # the toolbar offers both open paths
    tb_actions = [a.text() for t in win.findChildren(QtWidgets.QToolBar)
                  for a in t.actions()]
    assert "Open file…" in tb_actions and "Open folder…" in tb_actions

    win.load(h5)
    assert _wait_quickview_loaded(win, app), "the .h5 never finished loading"
    assert win.rec.n == rec.n
    assert win.full_btn.isEnabled()                     # 'Open in full GottLUX' armed

    folder = tmp_path / "capture"
    folder.mkdir()
    writer.write_raw(str(folder / "cap.raw"), rec.x, rec.y, rec.p, rec.t,
                     width=rec.width, height=rec.height)
    win.load(str(folder))                               # a directory loads too
    assert _wait_quickview_loaded(win, app), "the capture folder never finished loading"
    assert win.rec.n == rec.n
    assert win.full_btn.isEnabled()
    win.close()


def test_quickview_drag_drop_loads_dropped_file(app, tmp_path):
    """Dropping a loadable file (or folder) onto the quick viewer loads it."""
    from PySide6 import QtCore, QtGui

    from gottlux.app.quickview import QuickViewWindow
    from gottlux.io import hdf5 as h5io

    rec = _synth(12000, 0.4, seed=32)
    h5 = str(tmp_path / "drop.h5")
    h5io.write_hdf5(rec, h5)

    win = QuickViewWindow()
    assert win.acceptDrops()
    md = QtCore.QMimeData()
    md.setUrls([QtCore.QUrl.fromLocalFile(h5)])
    assert os.path.normpath(win._drop_path(md)) == os.path.normpath(h5)
    # folders resolve too (the drop handler accepts every gottlux.load path)
    md_dir = QtCore.QMimeData()
    md_dir.setUrls([QtCore.QUrl.fromLocalFile(str(tmp_path))])
    assert os.path.normpath(win._drop_path(md_dir)) == os.path.normpath(str(tmp_path))
    # a non-file payload is refused
    assert win._drop_path(QtCore.QMimeData()) is None

    ev = QtGui.QDropEvent(QtCore.QPointF(10, 10), QtCore.Qt.CopyAction, md,
                          QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    win.dropEvent(ev)                                   # the real handler kicks the load
    assert _wait_quickview_loaded(win, app), "the dropped .h5 never finished loading"
    assert win.rec.n == rec.n
    win.close()
