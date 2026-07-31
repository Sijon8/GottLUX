"""
Test the Timeline tab (gottlux.app.timeline) — the video-editor redesign.

Covers the editor's model and its wiring: add → trim → crop → reorder (buttons *and* a
drag on the track lanes) → export; the embedded preview rendering the compiled program at
the playhead through the one engine render path; the lane-block thumbnail cache; canvas
blocks (mosaics) inserted as sequence items and arranged inline on the shared
:class:`~gottlux.app.canvas.CanvasArrangeView` (a cell dragged there writes straight into
the spec, snapped to the grid); the project-canvas presets rescaling the preview and the
block arrangements; the overlay lane; the title items (a slide takes a sequence slot and
the ``.raw`` export skips it with the omission note, a running title rides the overlay
lane); OS drag-and-drop, including the 'as clips / as one mosaic' choice a multi-file drop
offers; and both exports — the video (ffmpeg-guarded) and the events-only ``.raw``, whose
plain-sequence bytes must stay identical to the pre-redesign stitcher's.

The engine-side program compilation and the layout math are unit-tested in
``test_canvas.py``; here everything runs through the real widget, offscreen, with file
dialogs and message boxes monkeypatched.
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


def _scene(dur, seed, name):
    rec, _ = synthetic_scene(duration_s=dur, targets=[FlutterTarget(flutter_hz=150)],
                             noise_rate_hz=4000, static_clutter=4, seed=seed)
    rec.name = name
    return rec


def _write_raw(rec, path):
    from gottlux.io import writer
    writer.write_raw(str(path), rec.x, rec.y, rec.p, rec.t,
                     width=rec.width, height=rec.height)
    return str(path)


def _slide_vals(text="Intro", duration_s=2.0):
    """A TitleDialog values dict for a title slide (the sequence lane)."""
    return {"text": text, "kind": "slide", "duration_s": duration_s, "whole": False,
            "anchor": "s", "font_size_px": 40, "color": (255, 255, 255),
            "bg_color": (0, 0, 0), "t0_s": 0.0}


def _running_vals(text="GottLUX", whole=True, duration_s=3.0):
    """A TitleDialog values dict for a running overlay title."""
    return {"text": text, "kind": "overlay", "duration_s": duration_s, "whole": whole,
            "anchor": "n", "font_size_px": 24, "color": (57, 197, 207),
            "bg_color": None, "t0_s": 0.0}


def _mouse(kind, x, y):
    """A synthetic left-button mouse event at widget-local ``(x, y)``."""
    from PySide6 import QtCore, QtGui
    at = QtCore.QPointF(x, y)
    return QtGui.QMouseEvent(kind, at, at, at, QtCore.Qt.LeftButton,
                             QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)


def _quiet_exports(monkeypatch, out):
    """Point every file dialog at *out* and silence the completion/reveal side effects."""
    from PySide6 import QtWidgets

    import gottlux.io.paths as paths
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (out, "")))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical",
                        staticmethod(lambda *a, **k: pytest.fail(f"export failed: {a}")))
    monkeypatch.setattr(paths, "open_in_file_browser", lambda *a, **k: None)


# ====================================================================== the model
def test_timeline_add_trim_reorder_stitch(app, tmp_path):
    from gottlux.app.timeline import TimelineEditorDialog
    from gottlux.io import writer
    from gottlux.io.recording import load

    a, b = _scene(0.5, 1, "a"), _scene(0.4, 2, "b")
    dlg = TimelineEditorDialog(recordings=[a, b])
    assert len(dlg.clips) == 2
    assert dlg.program().duration_s == pytest.approx(a.duration_s + b.duration_s, abs=1e-3)

    # trim the first clip to the middle 60% via the slider handler
    dlg.select(0)
    dlg._on_trim(0.2, 0.8)
    c0 = dlg.clips[0]
    assert c0["t0"] == pytest.approx(0.2 * c0["dur"]) and c0["t1"] == pytest.approx(0.8 * c0["dur"])
    # the program shrinks with the trim — the lanes and the clock follow it
    assert dlg.program().duration_s == pytest.approx(0.6 * c0["dur"] + b.duration_s, abs=1e-3)
    assert dlg.clock.t1 == pytest.approx(dlg.program().duration_s, abs=1e-3)

    # reorder: move the first clip later → order becomes (b, a)
    dlg._move(+1)
    assert dlg.clips[0]["rec"] is b and dlg.clips[1]["rec"] is a

    # the editor's specs stitch into one valid .raw
    specs = [(c["rec"], c["t0"], c["t1"]) for c in dlg.clips]
    out = str(tmp_path / "stitched.raw")
    res = writer.stitch_clips(out, specs, gap_s=0.05)
    assert res["n_events"] > 0 and len(res["segments"]) == 2
    assert load(out).n == res["n_events"]


def test_timeline_crop_field_roundtrips(app, tmp_path):
    """The per-clip crop (ROI) survives the select → edit → reselect loop, degenerate or
    full-frame boxes mean 'no crop', and the crop lands in the stitch specs."""
    from gottlux.app.timeline import TimelineEditorDialog
    from gottlux.io import writer
    from gottlux.io.recording import load

    a, b = _scene(0.4, 3, "a"), _scene(0.3, 4, "b")
    dlg = TimelineEditorDialog(recordings=[a, b])

    # edit the first clip's crop through the spin boxes
    dlg.select(0)
    for sp, v in zip(dlg.roi, (10, 20, 100, 110)):
        sp.setValue(v)
    assert dlg.clips[0]["roi"] == (10, 20, 100, 110)

    # switching away shows the other clip's default (full frame = no crop) …
    dlg.select(1)
    assert [sp.value() for sp in dlg.roi] == [0, 0, b.width, b.height]
    assert dlg.clips[1]["roi"] is None
    # … and switching back restores the edited values (the round-trip)
    dlg.select(0)
    assert [sp.value() for sp in dlg.roi] == [10, 20, 100, 110]
    assert dlg.clips[0]["roi"] == (10, 20, 100, 110)

    # the crop reaches the stitch engine and constrains the written events
    specs = [(c["rec"], c["t0"], c["t1"], c.get("roi")) for c in dlg.clips]
    out = str(tmp_path / "cropped.raw")
    writer.stitch_clips(out, specs[:1])                     # just the cropped clip
    r = load(out)
    assert r.n > 0
    assert int(np.asarray(r.x).min()) >= 10 and int(np.asarray(r.x).max()) < 100
    assert int(np.asarray(r.y).min()) >= 20 and int(np.asarray(r.y).max()) < 110

    # widening back to the full frame clears the crop
    dlg.select(0)
    for sp, v in zip(dlg.roi, (0, 0, a.width, a.height)):
        sp.setValue(v)
    assert dlg.clips[0]["roi"] is None


def test_timeline_per_clip_settings_are_independent(app):
    """Every clip carries its OWN visualization settings — the deck edits the selected
    clip's single full-frame cell and leaves the others alone."""
    from gottlux.app.timeline import TimelineEditor

    a, b = _scene(0.4, 19, "a"), _scene(0.3, 20, "b")
    ed = TimelineEditor(recordings=[a, b])
    ed.select(0)
    ed.cb_cmap.setCurrentText("gray")
    ed.cb_mode.setCurrentText("polarity")
    ed.sp_accum.setValue(0.005)
    assert ed.clips[0]["cell"].colormap == "gray"
    assert ed.clips[0]["cell"].mode == "polarity"
    assert ed.clips[0]["cell"].accumulation_s == pytest.approx(0.005)
    assert ed.clips[1]["cell"].colormap == "inferno"        # untouched

    # reselecting reflects each clip's own settings back into the deck
    ed.select(1)
    assert ed.cb_cmap.currentText() == "inferno"
    ed.select(0)
    assert ed.cb_cmap.currentText() == "gray"

    # a time scale slows the clip down and stretches the program accordingly
    before = ed.program().duration_s
    ed.select(1)
    ed.sp_scale.setValue(0.5)
    assert ed.program().duration_s == pytest.approx(before + b.duration_s, abs=1e-3)


def test_timeline_overlay_marking_and_lanes(app):
    """The overlay flag round-trips through the checkbox and splits the items into the
    sequence and overlay lanes."""
    from gottlux.app.timeline import TimelineEditor

    a, b = _scene(0.4, 5, "a"), _scene(0.3, 6, "b")
    ed = TimelineEditor(recordings=[a, b])
    assert ed._overlay_set() == [] and len(ed._sequential_set()) == 2

    ed.select(1)
    ed.overlay_chk.setChecked(True)
    assert ed.clips[1]["overlay"] is True
    assert [c["name"] for c in ed._overlay_set()] == ["b"]
    assert [c["name"] for c in ed._sequential_set()] == ["a"]
    # the sequence is now clip 'a' alone; 'b' rides over all of it
    assert ed.program().duration_s == pytest.approx(a.duration_s, abs=1e-3)
    assert len(ed.program().overlay_clips) == 1

    # reselecting reflects the stored flag; unchecking moves the clip back
    ed.select(0)
    assert not ed.overlay_chk.isChecked()
    ed.select(1)
    assert ed.overlay_chk.isChecked()
    ed.overlay_chk.setChecked(False)
    assert ed.clips[1]["overlay"] is False and len(ed._sequential_set()) == 2


# ====================================================================== the viewport
def test_timeline_preview_renders_the_program_at_the_playhead(app):
    """The embedded viewport shows a real engine-rendered frame of the program at the
    cursor — the same call the video export makes."""
    from gottlux.app.timeline import TimelineEditor

    a, b = _scene(0.5, 21, "a"), _scene(0.4, 22, "b")
    ed = TimelineEditor(recordings=[a, b])
    ed.clock.set_cursor(0.1)                        # inside the first clip
    frame = ed.current_frame()
    assert frame.shape == (a.height, a.width, 3) and frame.dtype == np.uint8
    assert frame.max() > 0                          # events actually rendered

    ed.sync()                                       # drives the viewport's paint path
    assert ed.preview._pix is not None
    assert (ed.preview._pix.width(), ed.preview._pix.height()) == (a.width, a.height)

    # seeking into the second clip renders that clip's segment instead
    seg = ed._segments[1]
    ed.clock.set_cursor(seg.t0_s + 0.5 * seg.duration_s)
    assert ed.current_frame().max() > 0

    # the project-canvas preset changes the preview (and export) geometry
    ed.canvas_cb.setCurrentText("640 × 640")
    assert ed.current_frame().shape == (640, 640, 3)


def test_timeline_thumbnail_cache_refreshes_on_settings(app):
    """Lane thumbnails render once per clip and are recomputed only when that clip's
    trim, crop or visualization settings change."""
    from gottlux.app.timeline import TimelineEditor, _THUMB_H

    a = _scene(0.4, 23, "a")
    ed = TimelineEditor(recordings=[a])
    item = ed.clips[0]
    pix = ed.thumbs.get(item)
    assert pix is not None and pix.height() == _THUMB_H
    assert ed.thumbs.get(item) is pix                        # cached, not re-rendered

    ed.select(0)
    ed.cb_cmap.setCurrentText("gray")
    refreshed = ed.thumbs.get(item)
    assert refreshed is not pix                              # settings moved → new thumb

    ed._on_trim(0.25, 0.75)                                  # so does the trim
    assert ed.thumbs.get(item) is not refreshed

    # a title has no source frames, so it has no thumbnail
    ed._append_title(_slide_vals("Intro"))
    assert ed.thumbs.get(ed.clips[-1]) is None


# ====================================================================== the track lanes
def test_timeline_lane_drag_reorders_the_model(app):
    """Dragging a block across the sequence lane reorders the timeline itself."""
    from PySide6 import QtCore
    from gottlux.app.timeline import TimelineEditor

    a, b = _scene(0.5, 24, "a"), _scene(0.4, 25, "b")
    ed = TimelineEditor(recordings=[a, b])
    lanes = ed.lanes
    lanes.resize(600, lanes.minimumHeight())
    y = lanes._seq_y() + lanes.SEQ_H / 2

    lanes.mousePressEvent(_mouse(QtCore.QEvent.MouseButtonPress,
                                 lanes._x(0.1 * lanes.duration), y))
    assert ed.selected_index() == 0                          # the press selected clip 'a'
    lanes.mouseMoveEvent(_mouse(QtCore.QEvent.MouseMove,
                                lanes._x(0.95 * lanes.duration), y))
    lanes.mouseReleaseEvent(_mouse(QtCore.QEvent.MouseButtonRelease,
                                   lanes._x(0.95 * lanes.duration), y))
    assert [c["name"] for c in ed.clips] == ["b", "a"]

    # clicking the ruler seeks the timeline's own clock
    lanes.mousePressEvent(_mouse(QtCore.QEvent.MouseButtonPress,
                                 lanes._x(0.5 * lanes.duration), 4))
    lanes.mouseReleaseEvent(_mouse(QtCore.QEvent.MouseButtonRelease,
                                   lanes._x(0.5 * lanes.duration), 4))
    assert ed.clock.cursor == pytest.approx(0.5 * lanes.duration, abs=0.02)

    lanes.resize(600, lanes.minimumHeight())
    assert not lanes.grab().isNull()                         # the populated paint path runs


# ====================================================================== canvas blocks
def test_timeline_canvas_block_is_a_sequence_item_arranged_inline(app):
    """'Add canvas block…' inserts a mosaic as ONE sequence item; selecting it switches
    the viewport into arrange mode, where dragging a cell writes into the block's spec
    (snapped to the 1/12 grid) with no pop-out window anywhere."""
    from gottlux.app.canvas import CanvasArrangeView
    from gottlux.app.timeline import TimelineEditor
    from gottlux.core import canvas as cv

    a, b = _scene(0.5, 26, "a"), _scene(0.4, 27, "b")
    ed = TimelineEditor(recordings=[a])
    block = ed.append_block_recordings([a, b])
    assert len(ed.clips) == 2 and block["kind"] == "block"
    assert len(block["spec"].clips) == 2                     # auto-tiled side by side
    assert block["spec"].clips[0].rect == (0, 0, a.width // 2, a.height)

    # selecting the block puts the shared arrange widget on the stage
    assert isinstance(ed.arrange, CanvasArrangeView)
    assert ed.stage.currentWidget() is ed.arrange
    assert len(ed.arrange.cells) == 2
    assert all(it._pix is not None for it in ed.arrange.cells)   # cells rendered live

    # dragging a cell round-trips into the spec, snapped to the grid
    W, H = ed._canvas_wh()
    ed.arrange.select(0)
    ed.arrange.cells[0].setPos(37, 41)
    assert block["spec"].clips[0].rect[:2] == (cv.snap(37, W), cv.snap(41, H))

    # the deck edits the SELECTED CELL's own EBS settings — look, clock, and crop
    ed.cb_cmap.setCurrentText("gray")
    ed.sp_accum.setValue(0.01)
    ed.sp_offset.setValue(0.05)
    for sp, v in zip(ed.roi, (10, 20, 100, 110)):
        sp.setValue(v)
    assert block["spec"].clips[0].colormap == "gray"
    assert block["spec"].clips[0].accumulation_s == pytest.approx(0.01)
    assert block["spec"].clips[0].t_offset_s == pytest.approx(0.05)
    assert block["spec"].clips[0].roi == (10, 20, 100, 110)
    assert block["spec"].clips[1].colormap == "inferno"      # the other cell is untouched
    assert block["spec"].clips[1].roi is None
    assert ed.clips[0].get("roi") is None                    # and so is the plain clip
    ed.arrange.select(1)                                     # the deck follows the selection
    ed._on_cell_selected(1)
    assert ed.cb_cmap.currentText() == "inferno"
    assert [sp.value() for sp in ed.roi] == [0, 0, b.width, b.height]
    ed.arrange.select(0)
    ed._on_cell_selected(0)

    # a size preset takes an exact fraction of the canvas; auto-tile re-lays them all
    ed.cell_preset_cb.setCurrentText("1/4 quadrant")
    ed._apply_cell_preset()
    assert block["spec"].clips[0].rect[2:] == (W // 2, H // 2)
    ed._autotile()
    assert [c.rect for c in block["spec"].clips] == cv.autotile(2, (W, H))

    # deselecting returns to playback mode, and the block still compiles into the program
    ed.select(-1)
    assert ed.stage.currentWidget() is ed.preview
    seg = ed._segments[1]
    assert seg.kind == "block" and len(seg.spec.clips) == 2
    assert ed.current_frame().shape == (H, W, 3)


def test_timeline_canvas_preset_rescales_blocks_and_geometry(app):
    """Changing the project canvas rescales the preview, the export geometry, and every
    canvas block's arrangement together."""
    from gottlux.app.timeline import TimelineEditor
    from gottlux.core import canvas as cv

    a, b = _scene(0.4, 28, "a"), _scene(0.3, 29, "b")
    ed = TimelineEditor(recordings=[a])
    block = ed.append_block_recordings([a, b])
    assert ed._canvas_wh() == (a.width, a.height)            # 'Native' = the first sensor

    ed.canvas_cb.setCurrentText("1280 × 720")
    assert ed._canvas_wh() == (1280, 720)
    assert (block["spec"].width, block["spec"].height) == (1280, 720)
    assert block["spec"].clips[0].rect == (0, 0, 640, 720)   # the split survived the rescale
    assert ed.current_frame().shape == (720, 1280, 3)

    ed.canvas_cb.setCurrentText(cv.CUSTOM_CANVAS)
    ed.canvas_w.setValue(256); ed.canvas_h.setValue(128)
    assert ed._canvas_wh() == (256, 128)
    assert ed.current_frame().shape == (128, 256, 3)


# ====================================================================== drag & drop
def test_timeline_editor_accepts_recording_drops(app, tmp_path, monkeypatch):
    """OS drops append clips through the add-clip path: multiple files land in name
    order, a failing path reports in the status label (no crash), a payload with no
    recordings is refused — and the legacy dialog inherits it all via its embedded
    editor."""
    from PySide6 import QtCore, QtGui

    from gottlux.app.timeline import TimelineEditor, TimelineEditorDialog

    a, b = _scene(0.3, 13, "a"), _scene(0.2, 14, "b")
    pa = _write_raw(a, tmp_path / "alpha.raw")
    pb = _write_raw(b, tmp_path / "beta.raw")
    monkeypatch.setattr(TimelineEditor, "_ask_multi_drop", lambda self, n: "clips")

    ed = TimelineEditor()
    assert ed.acceptDrops()
    md = QtCore.QMimeData()
    md.setUrls([QtCore.QUrl.fromLocalFile(pb), QtCore.QUrl.fromLocalFile(pa)])
    enter = QtGui.QDragEnterEvent(QtCore.QPoint(4, 4), QtCore.Qt.CopyAction, md,
                                  QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    ed.dragEnterEvent(enter)
    assert enter.isAccepted()
    ed.dropEvent(QtGui.QDropEvent(QtCore.QPointF(4, 4), QtCore.Qt.CopyAction, md,
                                  QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
    assert [c["name"] for c in ed.clips] == ["alpha", "beta"]   # name order, not drop order

    # payloads carrying no recordings are refused (empty, and a wrong-suffix file)
    md_none = QtCore.QMimeData()          # kept alive — the event does not own it
    empty = QtGui.QDragEnterEvent(QtCore.QPoint(4, 4), QtCore.Qt.CopyAction,
                                  md_none, QtCore.Qt.LeftButton,
                                  QtCore.Qt.NoModifier)
    ed.dragEnterEvent(empty)
    assert not empty.isAccepted()
    txt = tmp_path / "notes.txt"
    txt.write_text("not events", encoding="utf-8")
    md_txt = QtCore.QMimeData()
    md_txt.setUrls([QtCore.QUrl.fromLocalFile(str(txt))])
    wrong = QtGui.QDragEnterEvent(QtCore.QPoint(4, 4), QtCore.Qt.CopyAction, md_txt,
                                  QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    ed.dragEnterEvent(wrong)
    assert not wrong.isAccepted()

    # a failing path lands in the status label; the drop never raises
    bad = tmp_path / "empty_capture"
    bad.mkdir()
    md_bad = QtCore.QMimeData()
    md_bad.setUrls([QtCore.QUrl.fromLocalFile(str(bad))])
    ed.dropEvent(QtGui.QDropEvent(QtCore.QPointF(4, 4), QtCore.Qt.CopyAction, md_bad,
                                  QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
    assert len(ed.clips) == 2                                   # nothing was added …
    assert "failed to load" in ed.status.text()                 # … and the error shows

    # the legacy dialog embeds the same, drop-accepting editor
    dlg = TimelineEditorDialog()
    assert dlg.editor.acceptDrops()
    md2 = QtCore.QMimeData()
    md2.setUrls([QtCore.QUrl.fromLocalFile(pa)])
    dlg.editor.dropEvent(QtGui.QDropEvent(QtCore.QPointF(4, 4), QtCore.Qt.CopyAction,
                                          md2, QtCore.Qt.LeftButton,
                                          QtCore.Qt.NoModifier))
    assert len(dlg.clips) == 1 and dlg.clips[0]["name"] == "alpha"


def test_timeline_multi_drop_can_become_one_mosaic(app, tmp_path, monkeypatch):
    """Dropping several files at once offers 'as sequence clips' or 'as one mosaic
    block' — the mosaic choice tiles them into a single sequence item."""
    from PySide6 import QtCore, QtGui

    from gottlux.app.timeline import TimelineEditor

    a, b = _scene(0.3, 30, "a"), _scene(0.25, 31, "b")
    pa = _write_raw(a, tmp_path / "alpha.raw")
    pb = _write_raw(b, tmp_path / "beta.raw")
    asked = []
    monkeypatch.setattr(TimelineEditor, "_ask_multi_drop",
                        lambda self, n: asked.append(n) or "block")

    ed = TimelineEditor()
    md = QtCore.QMimeData()
    md.setUrls([QtCore.QUrl.fromLocalFile(pb), QtCore.QUrl.fromLocalFile(pa)])
    ed.dropEvent(QtGui.QDropEvent(QtCore.QPointF(4, 4), QtCore.Qt.CopyAction, md,
                                  QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
    assert asked == [2]                                     # the choice was offered
    assert len(ed.clips) == 1 and ed.clips[0]["kind"] == "block"
    assert len(ed.clips[0]["spec"].clips) == 2
    assert ed.stage.currentWidget() is ed.arrange           # ready to arrange inline


# ====================================================================== title items
def test_timeline_title_items_lanes_edit_remove(app):
    """Title items: a slide occupies its duration on the sequence lane, a running title
    rides the overlay lane; both disable the clip-only controls, edit in place, and
    remove like clips."""
    from gottlux.app.timeline import TimelineEditor

    a = _scene(0.4, 15, "a")
    ed = TimelineEditor(recordings=[a])
    ed._append_title(_slide_vals("Collect 42\nDawn pass", duration_s=2.5))
    ed._append_title(_running_vals("GottLUX"))
    assert len(ed.clips) == 3
    slide, run = ed.clips[1], ed.clips[2]
    assert slide["dur"] == pytest.approx(2.5) and slide["overlay"] is False
    assert run["overlay"] is True                       # the overlay lane
    assert slide in ed._sequential_set() and run in ed._overlay_set()
    # the slide takes a real slot on the program clock
    assert ed.program().duration_s == pytest.approx(a.duration_s + 2.5, abs=1e-3)
    assert ed._segments[1].kind == "title"

    # a title row disables the per-clip trim/crop/cut controls
    ed.select(1)
    assert not ed._clip_group.isEnabled() and not ed._look_group.isEnabled()
    ed.select(0)                                       # a real clip re-enables them
    assert ed._clip_group.isEnabled() and ed.trim.isEnabled()

    # editing rewrites name/duration/lane in place (the double-click path)
    ed._update_title(1, _slide_vals("Dusk pass", duration_s=1.0))
    assert ed.clips[1]["dur"] == pytest.approx(1.0)
    assert ed.clips[1]["name"] == "Dusk pass"

    # removal works like clips
    ed.select(2)
    ed._remove()
    assert len(ed.clips) == 2 and ed._overlay_set() == []


def test_timeline_running_title_renders_over_every_segment(app):
    """A running title is hoisted onto the program clock, so it draws over every segment
    rather than belonging to one of them."""
    from gottlux.app.timeline import TimelineEditor

    a, b = _scene(0.3, 32, "a"), _scene(0.3, 33, "b")
    ed = TimelineEditor(recordings=[a, b])
    ed._append_title(_running_vals("10× slow"))
    prog = ed.program()
    assert len(prog.overlays) == 1 and prog.overlays[0].span is None
    assert len(prog.segments) == 2
    assert all(not seg.spec.texts for seg in prog.segments)   # not baked into a segment


# ====================================================================== exports
def test_timeline_export_raw_matches_the_stitcher_byte_for_byte(app, tmp_path, monkeypatch,
                                                                exported):
    """On a plain sequence, 'Export .raw…' still IS the stitch: trim + crop + gap, one
    monotonic clock, bytes identical to calling ``stitch_clips`` directly — the artifact
    now sitting inside the export's provenance folder."""
    from gottlux.app.timeline import TimelineEditor
    from gottlux.io import writer

    a, b = _scene(0.4, 34, "a"), _scene(0.3, 35, "b")
    ed = TimelineEditor(recordings=[a, b])
    ed.select(0)
    ed._on_trim(0.2, 0.9)
    for sp, v in zip(ed.roi, (8, 8, 200, 220)):
        sp.setValue(v)
    ed.gap.setValue(0.05)

    out = str(tmp_path / "program.raw")
    ed.out_edit.setText(out)
    _quiet_exports(monkeypatch, out)
    ed._export_raw()
    folder = exported(out)

    ref = str(tmp_path / "ref.raw")
    writer.stitch_clips(ref, [(c["rec"], c["t0"], c["t1"], c.get("roi")) for c in ed.clips],
                        gap_s=0.05)
    with open(folder.artifact, "rb") as f:
        written = f.read()
    with open(ref, "rb") as f:
        expected = f.read()
    assert written == expected
    # the trim and the crop the stitch applied are the ones the usage rows record
    assert folder.usage[0]["trim_in_s"] == pytest.approx(ed.clips[0]["t0"])
    assert folder.usage[0]["trim_out_s"] == pytest.approx(ed.clips[0]["t1"])
    assert folder.usage[0]["roi"] == [8, 8, 200, 220]
    assert folder.record["settings"]["Gap between sequence items"] == "0.05 s"


def test_timeline_export_raw_skips_title_slide_with_note(app, tmp_path, monkeypatch,
                                                         exported):
    """The .raw export writes only the real clips — a title slide contributes no events —
    and the completion dialog carries the one-line omission note. A sequence lane holding
    only titles refuses to export."""
    from PySide6 import QtWidgets

    import gottlux.io.paths as paths
    from gottlux.app.timeline import TimelineEditor
    from gottlux.io import writer
    from gottlux.io.recording import load

    a = _scene(0.3, 16, "a")
    ed = TimelineEditor(recordings=[a])
    ed._append_title(_slide_vals("Intro", duration_s=2.0))
    out = str(tmp_path / "titled.raw")
    ed.out_edit.setText(out)

    infos = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *args, **k: infos.append(args)))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical",
                        staticmethod(lambda *args, **k: pytest.fail(f"export failed: {args}")))
    monkeypatch.setattr(paths, "open_in_file_browser", lambda *args, **k: None)
    ed._export_raw()

    folder = exported(out)
    ref = str(tmp_path / "ref.raw")                     # the clip alone, same trim
    writer.stitch_clips(ref, [(a, ed.clips[0]["t0"], ed.clips[0]["t1"], None)])
    assert load(folder.artifact).n == load(ref).n > 0   # the slide added nothing
    assert infos and "1 text item(s) omitted" in infos[0][2]
    # the omitted slide is still on the record — as a warning, and in the texts section
    assert any("text item(s) omitted" in w for w in folder.record["warnings"])
    assert [t["text"] for t in folder.record["texts"]] == ["Intro"]

    # only titles on the sequence lane → nothing to write, and no folder left behind
    ed2 = TimelineEditor()
    ed2._append_title(_slide_vals("Only"))
    ed2.out_edit.setText(str(tmp_path / "empty.raw"))
    ed2._export_raw()
    assert not os.path.exists(str(tmp_path / "empty.raw"))
    exported(str(tmp_path / "empty.raw"), exists=False)
    assert "Nothing to stitch" in ed2.status.text()


def test_timeline_export_raw_composites_canvas_blocks(app, tmp_path, monkeypatch, exported):
    """A timeline holding a canvas block re-encodes the events into the canvas geometry
    instead of stitching, and the one spec sidecar lands inside the export folder."""
    from gottlux.app.timeline import TimelineEditor
    from gottlux.core import canvas as cv
    from gottlux.io.recording import load

    a, b = _scene(0.3, 36, "a"), _scene(0.25, 37, "b")
    ed = TimelineEditor(recordings=[a])
    ed.append_block_recordings([a, b])
    ed.select(-1)
    out = str(tmp_path / "composited.raw")
    ed.out_edit.setText(out)
    _quiet_exports(monkeypatch, out)
    ed._export_raw()

    folder = exported(out)
    r = load(folder.artifact)
    assert (r.width, r.height) == ed._canvas_wh()
    assert r.n > 0
    sidecar = folder.spec_name()
    # exactly one spec: export_raw's own sidecar, inside the folder — no second copy
    assert sidecar and sum(n.endswith(cv.SPEC_SUFFIX) for n in folder.names) == 1
    assert not os.path.exists(os.path.splitext(out)[0] + cv.SPEC_SUFFIX)
    # the flattened program carries the plain clip's cell plus the block's two cells
    assert len(cv.load_spec(os.path.join(folder.folder, sidecar)).clips) == 3
    # and the README points at that spec as the way to re-render it
    assert sidecar in folder.readme
    assert folder.record["reproduce"]["spec"] == sidecar


def test_timeline_export_documents_every_source_of_a_multi_clip_program(
        app, tmp_path, monkeypatch, exported):
    """The motivating case: a program assembled from clips collected separately stays
    traceable to every last file.

    Six distinct ``.raw`` recordings — three on the sequence lane, one on the overlay
    lane, two more inside a canvas block — must yield six source rows (each named by its
    absolute path and identified by its digest) and six usage rows, one per placement.
    Nothing may be merged away, and a clip inside a block counts exactly as much as one
    sitting on the lane.
    """
    from gottlux.app.timeline import TimelineEditor
    from gottlux.io.recording import load

    paths = [_write_raw(_scene(0.15, 60 + i, f"clip{i}"), tmp_path / f"clip{i}.raw")
             for i in range(6)]
    recs = [load(p) for p in paths]
    ed = TimelineEditor(recordings=recs[:4])
    ed.select(3)
    ed.overlay_chk.setChecked(True)                  # the fourth clip rides the overlay
    ed.append_block_recordings(recs[4:])             # two more sources inside one block

    out = str(tmp_path / "program.raw")
    ed.out_edit.setText(out)
    _quiet_exports(monkeypatch, out)
    ed._export_raw()
    folder = exported(out)

    # one row per distinct recording — six files in, six sources out
    assert len(folder.sources) == 6
    assert {s["name"] for s in folder.sources} == {os.path.basename(p) for p in paths}
    by_name = {s["name"]: s for s in folder.sources}
    for p, rec in zip(paths, recs):
        assert os.path.abspath(p) in folder.readme   # the README names every path
        s = by_name[os.path.basename(p)]
        assert s["path"] == os.path.abspath(p) and s["directory"] == str(tmp_path)
        assert s["available"] and len(s["sha256"]) == 64 and s["format"] == "evt21"
        assert (s["width"], s["height"]) == (rec.width, rec.height)
        assert s["events"] == rec.n and s["duration_s"] == pytest.approx(rec.duration_s,
                                                                        abs=1e-5)
        assert s["sha256"][:12] in folder.readme     # the table's short digest

    # one usage row per placement, each resolving to a distinct source row
    assert len(folder.usage) == 6
    assert sorted(r["source"] for r in folder.usage) == [1, 2, 3, 4, 5, 6]
    lanes = [r["lane"] for r in folder.usage]
    assert lanes.count("overlay") == 1 and lanes.count("sequence") == 5
    # the block's two cells say which block they came from and where they land
    inblock = [r for r in folder.usage if r.get("block")]
    assert len(inblock) == 2
    assert all(len(r["dest_rect"]) == 4 and r["dest_rect"][2] > 0 for r in inblock)
    assert folder.record["settings"]["Canvas blocks"] == 1
    assert folder.record["settings"]["Overlay-lane clips"] == 1


def test_timeline_export_video_renders_the_program(app, tmp_path, monkeypatch, exported):
    """The video export writes the MP4 inside a folder holding the README, the machine-
    readable twin, and the spec that re-renders it."""
    from gottlux.app.timeline import TimelineEditor
    from gottlux.viz import video
    if not video.ffmpeg_available():
        pytest.skip("imageio-ffmpeg not available")

    a, b = _scene(0.25, 38, "a"), _scene(0.2, 39, "b")
    ed = TimelineEditor(recordings=[a, b])
    ed._append_title(_running_vals("GottLUX"))
    ed.canvas_cb.setCurrentText("640 × 640")
    out = str(tmp_path / "timeline.mp4")
    _quiet_exports(monkeypatch, out)
    ed._export_video()

    folder = exported(out)
    assert os.path.getsize(folder.artifact) > 0
    assert folder.spec_name() is not None
    assert folder.record["kind"] == "Timeline video (MP4)"
    assert folder.record["artifact"]["canvas"] == [640, 640]
    assert folder.record["artifact"]["frames"] > 0
    assert folder.record["artifact"]["fps"] == 30.0
    # the running title renders in video, so it is on the record as a text item
    assert [t["text"] for t in folder.record["texts"]] == ["GottLUX"]
    assert "Text is a **render-time** item" in folder.readme
