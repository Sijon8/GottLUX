"""Several visualizations on one canvas: Studio looks as cells, labels and time stamps,
the spec, the exports, the provenance record, and the GUI pieces that edit them."""
import os

import numpy as np
import pytest

from gottlux.core import canvas as cv
from gottlux.io.recording import Recording
from gottlux.studio import cell as lookcell
from gottlux.studio.looks import list_looks

LOOKS = sorted(list_looks())


def _blink_rec(dur=0.6, w=64, h=48, f0=40.0, seed=1, name="blink"):
    """A 10×10 square blinking ON/OFF at *f0* over light noise: every look has something."""
    rng = np.random.default_rng(seed)
    ts, xs, ys, ps = [], [], [], []
    for k in range(int(dur * f0)):
        for on, phase in ((1, 0.0), (0, 0.5)):
            t0 = (k + phase) / f0
            n = 150
            ts.append(t0 + rng.uniform(0, 0.002, n)); ps.append(np.full(n, on))
            xs.append(rng.integers(20, 30, n)); ys.append(rng.integers(15, 25, n))
    n = 3000
    ts.append(rng.uniform(0, dur, n)); ps.append(rng.integers(0, 2, n))
    xs.append(rng.integers(0, w, n)); ys.append(rng.integers(0, h, n))
    t = np.concatenate(ts); order = np.argsort(t, kind="stable")
    return Recording.from_events(np.concatenate(xs)[order], np.concatenate(ys)[order],
                                 np.concatenate(ps)[order], (t[order] * 1e6).astype(np.int64),
                                 width=w, height=h, name=name)


@pytest.fixture(scope="module")
def rec():
    return _blink_rec()


@pytest.fixture(autouse=True)
def _fresh_cells():
    lookcell.clear_cache()
    yield
    lookcell.clear_cache()


# ---------------------------------------------------------------------- the model
def test_spec_v2_round_trips_looks_and_v1_specs_load_as_frames():
    clip = cv.CanvasClip(source="a", rect=(0, 0, 100, 80), look="trails",
                         look_params={"tau_ms": 12.0}, palette="aurora", denoise=True,
                         bloom=0.5, label="wide", stamp=True)
    spec = cv.CanvasSpec(width=200, height=80, clips=[clip, cv.CanvasClip(source="b")])
    d = spec.to_dict()
    assert d["version"] == 2
    back = cv.CanvasSpec.from_dict(d)
    assert back == spec and back.clips[0].visualization == "trails"
    old = {"width": 64, "height": 48, "clips": [{"source": "a", "mode": "polarity"}]}
    v1 = cv.CanvasSpec.from_dict(old)
    assert v1.clips[0].look == "" and v1.clips[0].visualization == "frame"
    assert v1.clips[0].mode == "polarity"


@pytest.mark.parametrize("look", LOOKS)
def test_every_look_draws_a_cell(rec, look):
    clip = cv.CanvasClip(source="r", rect=(0, 0, 96, 72), look=look)
    rgb = cv.render_cell(clip, rec, 0.3)
    assert rgb.shape == (72, 96, 3) and rgb.dtype == np.uint8
    assert rgb.max() > 40, f"{look} drew nothing"


def test_looks_keep_their_aspect_inside_the_cell(rec):
    clip = cv.CanvasClip(source="r", rect=(0, 0, 200, 60), look="radar")
    rgb = cv.render_cell(clip, rec, 0.3)
    bg = rgb[0, 0]
    cols = np.where((rgb != bg).any(axis=(0, 2)))[0]
    # the round dial fits the 60 px height: it is letterboxed, not stretched to 200 px
    assert cols.size and cols.max() - cols.min() <= 64


def test_mixed_canvas_preview_equals_export_frame(rec):
    looks = ["", "trails", "surface", "spacetime"]
    spec = cv.CanvasSpec(width=192, height=144, clips=[
        cv.CanvasClip(source="r", rect=r, look=lk, label=lk or "frame", stamp=True)
        for lk, r in zip(looks, cv.autotile(4, (192, 144)), strict=True)])
    a = cv.render_frame(spec, {"r": rec}, 0.25)
    lookcell.clear_cache()                       # a fresh render (the export's view) agrees
    b = cv.render_frame(spec, {"r": rec}, 0.25)
    assert np.array_equal(a, b)
    # four different pictures, not one look four times
    quads = [a[y:y + 72, x:x + 96] for x, y, _w, _h in cv.autotile(4, (192, 144))]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(quads[i], quads[j])


def test_labels_and_stamps_are_burned_in(rec):
    plain = cv.CanvasClip(source="r", rect=(0, 0, 160, 120))
    labelled = cv.CanvasClip(source="r", rect=(0, 0, 160, 120), label="wide view",
                             stamp=True)
    a = cv.render_cell(plain, rec, 0.2)
    b = cv.render_cell(labelled, rec, 0.2)
    assert not np.array_equal(a[:30, :80], b[:30, :80])      # the corner carries text
    assert np.array_equal(a[60:, 100:], b[60:, 100:])        # the rest is untouched


def test_chunked_cells_keep_one_exposure(rec, monkeypatch):
    """A long clip is held in chunks; the locked scale of the first chunk rides every
    later one, so brightness does not jump at a boundary."""
    monkeypatch.setattr(lookcell, "EVENT_BUDGET", 2000)
    monkeypatch.setattr(lookcell, "MIN_CHUNK_S", 0.1)
    c = lookcell.LookCell(rec, "heat", size=(64, 48))
    assert c.n_chunks > 2
    c.frame(0.05)
    s0 = c._chunk[1].st["scale"]
    c.frame(rec.t_stop_s - 1e-3)
    assert c._chunk[0] == c.n_chunks - 1 and c._chunk[1].st["scale"] == s0


def test_strobe_cell_plays_its_phase_over_the_clip(rec):
    clip = cv.CanvasClip(source="r", rect=(0, 0, 64, 48), look="strobe",
                         look_params={"f0_hz": 40.0, "bins": 8, "cycles": 1}, stamp=True)
    c = lookcell.cell_for(rec, "strobe", clip.look_params, size=(64, 48))
    assert c.n_index == 8
    assert c.index_at(rec.t_start_s) == 0 and c.index_at(rec.t_stop_s) == 7
    assert "f0 = 40.0 Hz" in c.stamp(0.3)


def test_cells_are_cached_by_setting(rec):
    a = lookcell.cell_for(rec, "trails", {"tau_ms": 5.0}, size=(40, 30))
    assert lookcell.cell_for(rec, "trails", {"tau_ms": 5.0}, size=(40, 30)) is a
    assert lookcell.cell_for(rec, "trails", {"tau_ms": 9.0}, size=(40, 30)) is not a


# ---------------------------------------------------------------------- exports
@pytest.mark.parametrize("ext", [".mp4", ".gif"])
def test_mixed_canvas_exports_a_video(rec, tmp_path, ext):
    from gottlux.viz import encode
    if ext == ".mp4" and not encode.ffmpeg_ok():
        pytest.skip("imageio-ffmpeg missing")
    spec = cv.CanvasSpec(width=128, height=48, clips=[
        cv.CanvasClip(source="r", rect=(0, 0, 64, 48)),
        cv.CanvasClip(source="r", rect=(64, 0, 64, 48), look="trails", label="trails")])
    out = str(tmp_path / ("mix" + ext))
    res = cv.export_video(spec, {"r": rec}, out, fps=20.0)
    assert res["frames"] == round(rec.duration_s * 20)
    got = encode.probe(res["path"])
    assert got["frames"] == res["frames"] and got["width"] == 128


def test_export_can_be_cancelled_and_scaled(rec, tmp_path):
    spec = cv.CanvasSpec(width=128, height=96, clips=[
        cv.CanvasClip(source="r", rect=(0, 0, 128, 96), look="surface")])
    n = []
    assert cv.export_video(spec, {"r": rec}, str(tmp_path / "c.gif"), fps=20.0,
                           progress=n.append, cancel=lambda: len(n) > 2) is None
    assert not os.path.exists(tmp_path / "c.gif")
    res = cv.export_video(spec, {"r": rec}, str(tmp_path / "s.gif"), fps=10.0,
                          size=(64, 48))
    assert (res["width"], res["height"]) == (64, 48) and res["canvas"] == (128, 96)


def test_raw_export_notes_render_only_looks(rec, tmp_path):
    spec = cv.CanvasSpec(width=64, height=48, clips=[
        cv.CanvasClip(source="r", rect=(0, 0, 64, 48), look="trails")])
    res = cv.export_raw(spec, {"r": rec}, str(tmp_path / "e.raw"))
    assert res["n_events"] > 0
    assert any("Studio look" in w for w in res["warnings"])


def test_provenance_describes_the_look_not_the_unused_frame_settings():
    from gottlux.run.export_provenance import cell_usage
    look = cv.CanvasClip(source="a", look="spacetime", look_params={"orbit_deg": 90.0},
                         palette="ember", label="cloud")
    row = cell_usage(look, 1)
    assert row["visualization"].startswith("Space-time cloud")
    assert row["look_params"]["orbit_deg"] == 90.0 and row["palette"] == "ember"
    assert row["label"] == "cloud" and "colormap" not in row and "mode" not in row
    frame = cell_usage(cv.CanvasClip(source="a"), 1)
    assert frame["visualization"] == "Event frame" and frame["colormap"] == "inferno"


# ---------------------------------------------------------------------- GUI pieces
@pytest.fixture
def qapp():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_cell_style_editor_round_trips_a_look(qapp):
    from gottlux.app.cellstyle import CellStyleEditor
    ed = CellStyleEditor()
    clip = cv.CanvasClip(source="a", look="heat", look_params={"window_ms": 33.0,
                                                               "colormap": "ice"},
                         palette="aurora", denoise=False, bloom=0.4, label="x", stamp=True)
    ed.load(clip)
    assert ed.look() == "heat" and not ed.classic.isVisibleTo(ed)
    out = cv.CanvasClip(source="a")
    ed.store(out)
    assert out.look == "heat" and out.look_params["window_ms"] == 33.0
    assert out.look_params["colormap"] == "ice" and out.palette == "aurora"
    assert (out.denoise, out.bloom, out.label, out.stamp) == (False, 0.4, "x", True)
    # back to the classic frame: the look's settings are cleared
    ed.cb_look.setCurrentIndex(0)
    ed.store(out)
    assert out.look == "" and out.look_params == {}


def test_colormap_combo_follows_the_registry_and_keeps_saved_names(qapp, tmp_path,
                                                                  monkeypatch):
    from gottlux.app.colorschemes import ColormapCombo
    from gottlux.core import colors
    monkeypatch.setenv("GOTTLUX_COLORS", str(tmp_path / "c.json"))
    colors.reload()
    combo = ColormapCombo()
    combo.setCurrentText("viridis")
    colors.add_colormap("lagoon", ["#001018", "#00a0a0", "#e0fff8"])
    assert "lagoon" in combo.names() and combo.currentText() == "viridis"
    combo.setCurrentText("hot")                # valid matplotlib map, not in the list
    assert combo.currentText() == "hot"
    combo.setCurrentText("no such map")        # invalid: ignored, selection kept
    assert combo.currentText() == "hot"
    monkeypatch.undo()
    colors.reload()


def test_composer_exports_a_gif_with_mixed_looks(qapp, tmp_path, monkeypatch, exported):
    from PySide6 import QtWidgets

    import gottlux.io.paths as paths
    from gottlux.app.canvas import CanvasComposerWindow
    out = str(tmp_path / "mosaic.gif")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (out, "")))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical",
                        staticmethod(lambda *a, **k: pytest.fail(f"export failed: {a}")))
    monkeypatch.setattr(paths, "open_in_file_browser", lambda *a, **k: None)
    win = CanvasComposerWindow()
    win.canvas_w.setValue(128); win.canvas_h.setValue(48)
    r = _blink_rec(dur=0.3)
    win.add_clip_recording(r, source="a", rect=(0, 0, 64, 48))
    win.add_clip_recording(r, source="b", rect=(64, 0, 64, 48))
    win.clip_list.setCurrentRow(1)
    win.cb_look.setCurrentIndex(win.cb_look.findData("trails"))
    assert win.spec.clips[1].look == "trails" and win.spec.clips[0].look == ""
    win.video_opts.sp_fps.setValue(10.0)
    win._export_video()
    folder = exported(out)
    assert folder.record["kind"] == "Canvas video (GIF)"
    assert folder.record["artifact"]["frames"] == round(r.duration_s * 10)
    vis = [u["visualization"] for u in folder.usage]
    assert vis[0] == "Event frame" and vis[1].startswith("Polarity trails")
    win.close()


# ---------------------------------------------------------------------- fit in the cell
def test_fit_keeps_aspect_on_the_canvas_background(rec):
    spec = cv.CanvasSpec(width=200, height=48, background=(9, 99, 9), clips=[
        cv.CanvasClip(source="r", rect=(0, 0, 200, 48), fit="fit")])
    f = cv.render_frame(spec, {"r": rec}, 0.2)
    # a 64×48 sensor in a 200×48 cell: a 64-px picture centred, background either side
    assert tuple(f[24, 2]) == (9, 99, 9) and tuple(f[24, 197]) == (9, 99, 9)
    assert cv.fit_rect((64, 48), (0, 0, 200, 48)) == (68, 0, 64, 48)
    stretched = cv.render_frame(cv.CanvasSpec(width=200, height=48, background=(9, 99, 9),
                                              clips=[cv.CanvasClip(source="r",
                                                                   rect=(0, 0, 200, 48))]),
                                {"r": rec}, 0.2)
    assert tuple(stretched[24, 2]) != (9, 99, 9)       # 'auto' still fills, as before


def test_stretched_look_fills_its_cell(rec):
    clip = cv.CanvasClip(source="r", rect=(0, 0, 200, 60), look="radar", fit="stretch")
    rgb = cv.render_cell(clip, rec, 0.3)
    bg = rgb[30, 100]
    cols = np.where((np.abs(rgb.astype(int) - bg).sum(-1) > 30).any(axis=0))[0]
    assert cols.max() - cols.min() > 150               # the dial now spans the width


def test_raw_export_places_events_where_the_picture_is(rec, tmp_path):
    import gottlux as eb
    spec = cv.CanvasSpec(width=200, height=48, clips=[
        cv.CanvasClip(source="r", rect=(0, 0, 200, 48), fit="fit")])
    res = cv.export_raw(spec, {"r": rec}, str(tmp_path / "f.raw"))
    out = eb.load(res["path"])
    xs = np.asarray(out.x)
    assert xs.min() >= 68 and xs.max() < 68 + 64


def test_canvas_cli_builds_a_mosaic(tmp_path):
    from gottlux.io import writer
    from gottlux.run import canvas_cli
    r = _blink_rec(dur=0.3)
    raw = str(tmp_path / "blink.raw")
    writer.write_raw(raw, np.asarray(r.x), np.asarray(r.y), np.asarray(r.p),
                     np.asarray(r.t), width=r.width, height=r.height)
    out = str(tmp_path / "mosaic.gif")
    assert canvas_cli.main(["--mosaic", raw, "--looks", "frame,trails,surface",
                            "--size", "192x48", "--labels", "--fps", "10", "-o", out]) == 0
    from gottlux.viz import encode
    got = encode.probe(out)
    assert got["width"] == 192 and got["frames"] == round(r.duration_s * 10)
    spec = cv.load_spec(str(tmp_path / ("mosaic" + cv.SPEC_SUFFIX)))
    assert [c.visualization for c in spec.clips] == ["frame", "trails", "surface"]
    assert all(c.fit == "fit" for c in spec.clips) and spec.clips[1].label == "Polarity trails"
    # and the saved spec re-renders through the same command
    out2 = str(tmp_path / "again.gif")
    assert canvas_cli.main([str(tmp_path / ("mosaic" + cv.SPEC_SUFFIX)), "--fps", "10",
                            "-o", out2]) == 0
    assert encode.probe(out2)["frames"] == got["frames"]
