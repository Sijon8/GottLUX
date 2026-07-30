"""
Tests for the EBS canvas composer: the spec's JSON round-trip, the canvas-clock math,
frame compositing into the correct cells, the composited EVT2.1 ``.raw`` export (geometry,
counts, timestamp rescaling, sidecar), the MP4 export (skip-guarded on ffmpeg), the text
items (CanvasText round-trip, slide/overlay pixel rendering, the .raw omission note), the
standard-size / snap / auto-tile layout math, the **program** compiler behind the Timeline
tab (a clip becoming a single full-canvas cell, a mosaic passing through verbatim, a title
slide becoming a text item, overlays spanning every segment) and its render/flatten/export
paths, plus the offscreen GUI window (including OS drag-and-drop and the title rows).
"""
import os

import numpy as np
import pytest

from gottlux.core import canvas as cv
from gottlux.io.recording import Recording


# ------------------------------------------------------------------ synthetic helpers
def _point_rec(x, y, times_us, w=16, h=16, name="pt", p=1):
    """A recording whose every event sits at one pixel — makes cell placement exact."""
    n = len(times_us)
    return Recording.from_events(np.full(n, x), np.full(n, y), np.full(n, p, np.uint8),
                                 np.asarray(times_us, np.int64), width=w, height=h,
                                 name=name)


def _noise_rec(n, dur_s, w=96, h=96, seed=0, name="noise"):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, dur_s, n) * 1e6).astype(np.int64)
    return Recording.from_events(rng.integers(0, w, n), rng.integers(0, h, n),
                                 rng.integers(0, 2, n), t, width=w, height=h, name=name)


# ------------------------------------------------------------------ spec (de)serialization
def test_spec_json_roundtrip(tmp_path):
    spec = cv.CanvasSpec(width=800, height=600, background=(10, 12, 16), clips=[
        cv.CanvasClip(source="a.raw", rect=(0, 0, 400, 600), roi=(10, 20, 200, 220),
                      t_offset_s=0.5, time_scale=0.1, accumulation_s=0.005,
                      mode="polarity", colormap="coolwarm", tonemap="log", gamma=0.7,
                      loop=False),
        cv.CanvasClip(source="b.raw", rect=(400, 0, 400, 600)),
    ])
    path = str(tmp_path / ("comp" + cv.SPEC_SUFFIX))
    assert cv.save_spec(spec, path) == path
    loaded = cv.load_spec(path)
    assert loaded == spec                      # dataclass equality — every field survives
    assert loaded.clips[0].roi == (10, 20, 200, 220) and loaded.clips[1].roi is None


# ------------------------------------------------------------------ canvas-clock math
def test_clip_time_scale_offset_and_loop():
    rec = _point_rec(1, 1, [0, 1_000_000])                # exactly 1 s long
    clip = cv.CanvasClip(source="a", t_offset_s=1.0, time_scale=0.5, loop=False)
    assert cv.clip_time(clip, rec, 0.5) is None           # before the offset
    assert cv.clip_time(clip, rec, 1.0) == pytest.approx(0.0)
    assert cv.clip_time(clip, rec, 2.0) == pytest.approx(0.5)   # half clip-speed
    assert cv.clip_time(clip, rec, 3.0) == pytest.approx(1.0)   # exactly the clip end
    assert cv.clip_time(clip, rec, 3.2) is None           # past the single pass
    assert cv.clip_extent_s(clip, rec) == (1.0, pytest.approx(3.0))

    looped = cv.CanvasClip(source="a", t_offset_s=1.0, time_scale=0.5, loop=True)
    assert cv.clip_time(looped, rec, 0.5) is None         # still dark before the offset
    assert cv.clip_time(looped, rec, 3.2) == pytest.approx(0.1)   # wrapped: 1.1 % 1.0

    spec = cv.CanvasSpec(clips=[clip])
    assert cv.canvas_duration(spec, {"a": rec}) == pytest.approx(3.0)


# ------------------------------------------------------------------ frame compositing
def test_render_frame_composites_into_correct_cells():
    """Two single-pixel recordings land as bright blocks exactly inside their dest rects."""
    rec_a = _point_rec(2, 3, np.arange(0, 1_000_000, 1000), w=10, h=10, name="a")
    rec_b = _point_rec(6, 1, np.arange(0, 1_000_000, 2000), w=8, h=8, name="b")
    spec = cv.CanvasSpec(width=100, height=60, background=(0, 0, 0), clips=[
        cv.CanvasClip(source="a", rect=(0, 0, 50, 60), colormap="gray"),
        cv.CanvasClip(source="b", rect=(50, 0, 50, 60), colormap="gray"),
    ])
    recs = {"a": rec_a, "b": rec_b}
    frame = cv.render_frame(spec, recs, 0.2)
    assert frame.shape == (60, 100, 3) and frame.dtype == np.uint8
    # Nearest-resize edges are float-sensitive, so the regions carry a 1-px margin:
    # cell A: sensor (2,3) scales ×5 in x, ×6 in y → block ≈ x∈[10,15), y∈[18,24)
    assert frame[19:23, 11:14].min() == 255               # block interior is lit
    assert frame[:17, :50].max() == 0 and frame[25:, :50].max() == 0    # rest of A dark
    assert frame[:, :9].max() == 0 and frame[:, 16:49].max() == 0
    # cell B: sensor (6,1) → block ≈ x∈[87,94), y∈[8,15)
    assert frame[10:13, 89:92].min() == 255
    assert frame[:6, 50:].max() == 0 and frame[17:, 50:].max() == 0     # rest of B dark
    assert frame[:, 51:85].max() == 0 and frame[:, 96:].max() == 0


def test_render_frame_outside_extent_leaves_background():
    rec = _point_rec(2, 2, [0, 100_000], w=8, h=8)        # 0.1 s long
    spec = cv.CanvasSpec(width=40, height=40, background=(7, 8, 9), clips=[
        cv.CanvasClip(source="a", rect=(0, 0, 40, 40), t_offset_s=1.0, loop=False,
                      colormap="gray"),
    ])
    frame = cv.render_frame(spec, {"a": rec}, 0.5)        # before the clip starts
    assert np.array_equal(np.unique(frame.reshape(-1, 3), axis=0), [[7, 8, 9]])
    frame = cv.render_frame(spec, {"a": rec}, 1.0)        # clip start → the cell paints
    assert frame.max() == 255                             # the event at t_clip=0 is lit


def test_render_cell_roi_crop():
    """With an ROI, only events inside the crop appear, filling the whole cell."""
    rec = _point_rec(6, 6, np.arange(0, 500_000, 1000), w=8, h=8)
    inside = cv.CanvasClip(source="a", rect=(0, 0, 40, 40), roi=(4, 4, 8, 8),
                           colormap="gray")
    cell = cv.render_cell(inside, rec, 0.1)
    assert cell.shape == (40, 40, 3)
    # (6,6) inside the 4×4 crop maps to crop pixel (2,2) → ×10 scale → x,y ∈ [20,30)
    assert cell[20:30, 20:30].min() == 255 and cell[:20, :].max() == 0
    outside = cv.CanvasClip(source="a", rect=(0, 0, 40, 40), roi=(0, 0, 4, 4),
                            colormap="gray")
    assert cv.render_cell(outside, rec, 0.1).max() == 0   # its pixel is cropped away


# ------------------------------------------------------------------ .raw export
def test_export_raw_geometry_counts_and_time_rescale(tmp_path):
    """Two tiny clips → one EVT2.1 .raw: canvas geometry, exact counts, exact timestamps
    (scaled by 1/time_scale and offset), exact cell coordinates, and the spec sidecar."""
    import gottlux as eb
    rec_a = _point_rec(1, 1, [0, 100_000, 200_000], w=4, h=4, name="a", p=1)
    rec_b = _point_rec(2, 3, [0, 50_000], w=4, h=4, name="b", p=0)
    spec = cv.CanvasSpec(width=64, height=48, clips=[
        cv.CanvasClip(source="a", rect=(0, 0, 32, 48), loop=False),
        cv.CanvasClip(source="b", rect=(32, 0, 32, 48), t_offset_s=0.1, time_scale=0.5,
                      loop=False),
    ])
    recs = {"a": rec_a, "b": rec_b}
    fracs = []
    out = str(tmp_path / "canvas.raw")
    res = cv.export_raw(spec, recs, out, progress=fracs.append)
    assert res["n_events"] == 5 and res["width"] == 64 and res["height"] == 48
    assert fracs and fracs[-1] == pytest.approx(1.0)

    r = eb.load(out)
    assert (r.width, r.height) == (64, 48)                # geometry = the canvas size
    assert r.n == rec_a.n + rec_b.n                       # every windowed event survived
    # timestamps: A maps 1:1; B maps t → 0.1 s + 2·t (time_scale 0.5 = 2× slow)
    assert np.array_equal(np.asarray(r.t), [0, 100_000, 100_000, 200_000, 200_000])
    # spatial mapping: A(1,1) → (8,12) in its 32×48 cell; B(2,3) → (32+16, 36)
    coords = {(int(x), int(y)) for x, y in zip(r.x, r.y)}
    assert coords == {(8, 12), (48, 36)}
    # polarity preserved event-for-event (A all ON at x=8, B all OFF at x=48)
    assert all(int(p) == 1 for x, p in zip(r.x, r.p) if int(x) == 8)
    assert all(int(p) == 0 for x, p in zip(r.x, r.p) if int(x) == 48)
    # the composition spec rides along as a sidecar for reproducibility
    sidecar = os.path.splitext(out)[0] + cv.SPEC_SUFFIX
    assert res["sidecar"] == sidecar and os.path.exists(sidecar)
    assert cv.load_spec(sidecar) == spec


def test_export_raw_roi_crop_drops_outside_events(tmp_path):
    from gottlux.io import decode
    rec = _point_rec(2, 3, [0, 50_000], w=4, h=4, name="b")
    keep = cv.CanvasClip(source="b", rect=(0, 0, 16, 16), roi=(2, 2, 4, 4), loop=False)
    drop = cv.CanvasClip(source="b", rect=(16, 0, 16, 16), roi=(0, 0, 2, 2), loop=False)
    spec = cv.CanvasSpec(width=32, height=16, clips=[keep, drop])
    res = cv.export_raw(spec, {"b": rec}, str(tmp_path / "roi.raw"))
    assert res["n_events"] == rec.n                       # only the kept cell contributes
    d = decode.decode(str(tmp_path / "roi.raw"))
    assert d["x"].max() < 16                              # nothing landed in the drop cell


def test_export_raw_loop_repeats_events(tmp_path):
    """A looping clip re-emits its events each pass across the export duration."""
    from gottlux.io import decode
    rec = _point_rec(1, 1, [0, 100_000], w=4, h=4)        # 0.1 s long
    spec = cv.CanvasSpec(width=16, height=16, clips=[
        cv.CanvasClip(source="a", rect=(0, 0, 16, 16), loop=True),
    ])
    out = str(tmp_path / "loop.raw")
    res = cv.export_raw(spec, {"a": rec}, out, duration_s=0.25)
    d = decode.decode(out)
    # passes at offsets 0.0 / 0.1 / 0.2 → [0, 0.1] + [0.1, 0.2] + [0.2] (0.25 cut-off)
    assert res["n_events"] == 5
    assert np.array_equal(np.asarray(d["t"]),
                          [0, 100_000, 100_000, 200_000, 200_000])


def test_export_raw_rejects_oversize_canvas(tmp_path):
    rec = _point_rec(1, 1, [0, 1000], w=4, h=4)
    spec = cv.CanvasSpec(width=cv.MAX_RAW_DIM + 1, height=64,
                         clips=[cv.CanvasClip(source="a")])
    with pytest.raises(ValueError, match="EVT2.1"):
        cv.export_raw(spec, {"a": rec}, str(tmp_path / "big.raw"))


# ------------------------------------------------------------------ text items
def test_text_spec_json_roundtrip(tmp_path):
    """CanvasText items survive the JSON round-trip field-for-field, and a pre-text spec
    (no 'texts' key) still loads with the field defaulting empty."""
    spec = cv.CanvasSpec(width=320, height=240, clips=[cv.CanvasClip(source="a.raw")],
                         texts=[
        cv.CanvasText(text="Collect 42\nDawn pass", kind="slide", span=(0.5, 3.5),
                      font_size_px=44, color=(255, 255, 255), bg_color=(10, 12, 16)),
        cv.CanvasText(text="10× slow", kind="overlay", span=None, anchor="n",
                      margin_px=12, font_size_px=20, color=(57, 197, 207),
                      bg_color=None),
    ])
    path = str(tmp_path / ("titled" + cv.SPEC_SUFFIX))
    cv.save_spec(spec, path)
    loaded = cv.load_spec(path)
    assert loaded == spec                      # dataclass equality — every field survives
    assert loaded.texts[0].span == (0.5, 3.5) and loaded.texts[1].span is None
    assert loaded.texts[1].bg_color is None
    assert cv.CanvasSpec.from_dict({"width": 64, "height": 48, "clips": []}).texts == []


def test_render_frame_title_slide_fills_frame_with_text():
    """An active slide covers the cells: the frame is the slide's bg color with text-ink
    pixels present; outside its span the cells show again."""
    rec = _point_rec(2, 3, np.arange(0, 1_000_000, 1000), w=10, h=10)
    spec = cv.CanvasSpec(width=120, height=80, clips=[
        cv.CanvasClip(source="a", rect=(0, 0, 120, 80), colormap="gray")],
        texts=[cv.CanvasText(text="TITLE", kind="slide", span=(0.0, 1.0),
                             font_size_px=40, color=(255, 0, 0), bg_color=(0, 0, 60))])
    frame = cv.render_frame(spec, {"a": rec}, 0.5)
    for corner in (frame[0, 0], frame[0, -1], frame[-1, 0], frame[-1, -1]):
        assert np.array_equal(corner, (0, 0, 60))          # bg fills the frame
    ink = (frame[..., 0] > 200) & (frame[..., 1] < 80) & (frame[..., 2] < 80)
    assert ink.any()                                       # text ink present
    assert not (frame == 255).all(axis=2).any()            # the cell is covered
    after = cv.render_frame(spec, {"a": rec}, 1.5)         # past the span (clip loops)
    assert (after == 255).all(axis=2).any()                # the cell shows again


def test_render_frame_overlay_title_anchored_over_cells():
    """A running overlay draws its ink inside the anchored region while the cells stay
    visible below it."""
    rec = _point_rec(5, 5, np.arange(0, 1_000_000, 1000), w=10, h=10)
    spec = cv.CanvasSpec(width=100, height=90, clips=[
        cv.CanvasClip(source="a", rect=(0, 0, 100, 90), colormap="gray")],
        texts=[cv.CanvasText(text="RUN", kind="overlay", span=None, anchor="n",
                             margin_px=4, font_size_px=24, color=(0, 255, 0),
                             bg_color=None)])
    frame = cv.render_frame(spec, {"a": rec}, 0.2)
    ink = (frame[..., 1] > 200) & (frame[..., 0] < 80) & (frame[..., 2] < 80)
    assert ink.any() and ink[:40].any() and not ink[40:].any()   # ink only up top ('n')
    assert (frame[40:] == 255).all(axis=2).any()           # the cell block still shows


def test_export_raw_ignores_texts_with_note_and_identical_bytes(tmp_path):
    """export_raw ignores text items by design: the .raw is byte-identical to the
    text-free spec, the omission note rides the returned warnings, and the sidecar still
    records the texts so the composition round-trips."""
    rec = _point_rec(1, 1, [0, 100_000], w=4, h=4)
    clips = [cv.CanvasClip(source="a", rect=(0, 0, 16, 16), loop=False)]
    plain = cv.CanvasSpec(width=16, height=16, clips=clips)
    titled = cv.CanvasSpec(width=16, height=16, clips=clips,
                           texts=[cv.CanvasText(text="T", kind="slide",
                                                span=(0.0, 1.0))])
    p_plain, p_titled = str(tmp_path / "plain.raw"), str(tmp_path / "titled.raw")
    res_plain = cv.export_raw(plain, {"a": rec}, p_plain)
    res_titled = cv.export_raw(titled, {"a": rec}, p_titled)
    assert res_plain["warnings"] == []
    assert res_titled["warnings"] == [cv.text_omission_note(1)]
    assert "1 text item(s) omitted" in res_titled["warnings"][0]
    with open(p_plain, "rb") as f:
        plain_bytes = f.read()
    with open(p_titled, "rb") as f:
        titled_bytes = f.read()
    assert plain_bytes == titled_bytes                     # events only — identical
    assert cv.load_spec(res_titled["sidecar"]).texts == titled.texts


# ------------------------------------------------------------------ video export
def test_export_video_writes_mp4_when_ffmpeg_available(tmp_path):
    from gottlux.viz import video
    if not video.ffmpeg_available():
        pytest.skip("imageio-ffmpeg not available")
    rec = _noise_rec(20000, 0.4, w=32, h=32, seed=3)
    spec = cv.CanvasSpec(width=64, height=48, clips=[
        cv.CanvasClip(source="n", rect=(0, 0, 64, 48)),
    ])
    out = str(tmp_path / "canvas.mp4")
    res = cv.export_video(spec, {"n": rec}, out, fps=10.0, duration_s=0.4)
    assert res == out and os.path.exists(out) and os.path.getsize(out) > 0


def test_export_video_renders_text_frames_when_ffmpeg_available(tmp_path):
    """A titled composition exports through the render path with the timeline extended
    over the slide's span (frames past the clips still encode)."""
    from gottlux.viz import video
    if not video.ffmpeg_available():
        pytest.skip("imageio-ffmpeg not available")
    rec = _noise_rec(8000, 0.3, w=32, h=32, seed=5)
    spec = cv.CanvasSpec(width=64, height=48, clips=[
        cv.CanvasClip(source="n", rect=(0, 0, 64, 48))],
        texts=[cv.CanvasText(text="T", kind="slide", span=(0.0, 0.5),
                             font_size_px=24, color=(255, 0, 0), bg_color=(0, 0, 60))])
    out = str(tmp_path / "titled.mp4")
    fracs = []
    res = cv.export_video(spec, {"n": rec}, out, fps=10.0, progress=fracs.append)
    assert res == out and os.path.getsize(out) > 0
    assert len(fracs) == 5          # the slide's span (0.5 s at 10 fps) set the timeline


# ------------------------------------------------------------------ layout math (pure)
def test_canvas_size_presets_resolve():
    labels = [label for label, _ in cv.CANVAS_PRESETS]
    assert labels[0] == cv.NATIVE_CANVAS and labels[-1] == cv.CUSTOM_CANVAS
    for label in ("640 × 640", "1280 × 720", "1920 × 1080", "1024 × 1024"):
        assert label in labels
    assert cv.canvas_preset_size("1920 × 1080") == (1920, 1080)
    assert cv.canvas_preset_size(cv.NATIVE_CANVAS, native_wh=(320, 320)) == (320, 320)
    assert cv.canvas_preset_size(cv.CUSTOM_CANVAS, custom_wh=(777, 111)) == (777, 111)
    assert cv.canvas_preset_size(cv.NATIVE_CANVAS) == (640, 480)      # nothing to adopt


def test_cell_presets_are_exact_fractions_and_stay_on_canvas():
    assert cv.cell_preset_rect("Full", (640, 480)) == (0, 0, 640, 480)
    assert cv.cell_preset_rect("1/2 horizontal", (640, 480)) == (0, 0, 320, 480)
    assert cv.cell_preset_rect("1/2 vertical", (640, 480)) == (0, 0, 640, 240)
    assert cv.cell_preset_rect("1/4 quadrant", (640, 480)) == (0, 0, 320, 240)
    assert cv.cell_preset_rect("1/3", (600, 300)) == (0, 0, 200, 100)
    assert cv.cell_preset_rect("2/3", (600, 300)) == (0, 0, 400, 200)
    # the origin is kept, but pulled back so the cell never hangs off the canvas
    assert cv.cell_preset_rect("1/4 quadrant", (640, 480), origin=(100, 60)) == \
        (100, 60, 320, 240)
    assert cv.cell_preset_rect("1/4 quadrant", (640, 480), origin=(600, 470)) == \
        (320, 240, 320, 240)


def test_snap_to_the_twelfth_grid():
    assert cv.SNAP_DIVISIONS == 12
    step = 1200 / 12                                   # 100 px
    assert cv.snap(0, 1200) == 0
    assert cv.snap(140, 1200) == 100
    assert cv.snap(160, 1200) == 200
    assert cv.snap(1200, 1200) == 1200
    assert cv.snap(137, 1200, divisions=0) == 137      # snapping off = pass through
    # a snapped rect keeps at least one grid step of size, so a cell cannot collapse
    assert cv.snap_rect((140, 40, 3, 3), (1200, 600)) == (100, 50, 100, 50)
    assert cv.snap_rect((7, 9, 11, 13), (1200, 600), divisions=0) == (7, 9, 11, 13)
    assert step == 100


def test_autotile_covers_the_canvas_without_gaps_or_overlap():
    assert cv.autotile(0, (100, 100)) == []
    assert cv.autotile(1, (600, 400)) == [(0, 0, 600, 400)]
    assert cv.autotile(2, (600, 400)) == [(0, 0, 300, 400), (300, 0, 300, 400)]
    assert cv.autotile(4, (600, 400)) == [(0, 0, 300, 200), (300, 0, 300, 200),
                                          (0, 200, 300, 200), (300, 200, 300, 200)]
    # an awkward count still tiles exactly: every pixel covered at most once, edges met
    rects = cv.autotile(7, (601, 403))
    cover = np.zeros((403, 601), np.int32)
    for x, y, w, h in rects:
        cover[y:y + h, x:x + w] += 1
    assert cover.max() == 1                            # no cell overlaps another
    assert rects[2][0] + rects[2][2] == 601            # the last column reaches the edge
    assert rects[-1][1] + rects[-1][3] == 403          # and the last row the bottom


def test_rescale_spec_keeps_the_arrangement():
    spec = cv.CanvasSpec(width=640, height=480, clips=[
        cv.CanvasClip(source="a", rect=(0, 0, 320, 480), colormap="gray"),
        cv.CanvasClip(source="b", rect=(320, 0, 320, 480)),
    ])
    out = cv.rescale_spec(spec, (1280, 720))
    assert (out.width, out.height) == (1280, 720)
    assert out.clips[0].rect == (0, 0, 640, 720)
    assert out.clips[1].rect == (640, 0, 640, 720)
    assert out.clips[0].colormap == "gray"             # only geometry moves
    assert spec.clips[0].rect == (0, 0, 320, 480)      # the source spec is untouched


# ------------------------------------------------------------------ the program compiler
def _clip_item(rec, name="c", **kw):
    item = {"kind": "clip", "name": name, "rec": rec, "t0": 0.0, "t1": rec.duration_s,
            "roi": None, "cell": cv.CanvasClip(source="", loop=False)}
    item.update(kw)
    return item


def test_compile_program_lays_items_end_to_end():
    """Every sequential item becomes one CanvasSpec segment: a plain clip is a single cell
    covering the whole canvas, a mosaic passes through verbatim, a title slide is a text
    item — laid end to end on one program clock."""
    a = _point_rec(1, 1, [0, 500_000], w=32, h=24, name="a")          # 0.5 s
    b = _point_rec(2, 2, [0, 300_000], w=32, h=24, name="b")          # 0.3 s
    mosaic = cv.CanvasSpec(width=32, height=24, clips=[
        cv.CanvasClip(source="a", rect=(0, 0, 16, 24)),
        cv.CanvasClip(source="b", rect=(16, 0, 16, 24), loop=False)])
    items = [
        _clip_item(a, "a", key="A"),
        {"kind": "title", "name": "T", "key": "T", "dur": 2.0,
         "text": cv.CanvasText(text="T", kind="slide", span=(9.0, 9.5))},
        {"kind": "block", "name": "M", "key": "M", "spec": mosaic,
         "recs": {"a": a, "b": b}},
    ]
    prog = cv.compile_program(items)

    assert (prog.width, prog.height) == (32, 24)        # 'Native' = the first clip's sensor
    assert [s.kind for s in prog.segments] == ["clip", "title", "block"]
    assert [s.key for s in prog.segments] == ["A", "T", "M"]

    clip_seg = prog.segments[0]                         # a clip = ONE full-canvas cell
    assert len(clip_seg.spec.clips) == 1
    assert clip_seg.spec.clips[0].rect == (0, 0, 32, 24)
    assert clip_seg.recs[clip_seg.spec.clips[0].source] is a
    assert clip_seg.duration_s == pytest.approx(0.5)

    title_seg = prog.segments[1]                        # a slide = a text item, no cells
    assert not title_seg.spec.clips and len(title_seg.spec.texts) == 1
    assert title_seg.spec.texts[0].span == (0.0, 2.0)   # re-spanned onto the slide's slot
    assert title_seg.t0_s == pytest.approx(0.5)

    block_seg = prog.segments[2]                        # a mosaic = its own spec, verbatim
    assert [c.rect for c in block_seg.spec.clips] == [(0, 0, 16, 24), (16, 0, 16, 24)]
    assert block_seg.t0_s == pytest.approx(2.5)
    assert block_seg.duration_s == pytest.approx(0.5)   # the longest cell's extent
    assert prog.duration_s == pytest.approx(3.0)
    assert prog.segment_at(0.25) is clip_seg and prog.segment_at(2.6) is block_seg
    assert prog.segment_at(99.0) is block_seg           # past the end clamps to the last


def test_compile_program_honours_trim_scale_and_gap():
    rec = _point_rec(1, 1, np.arange(0, 1_000_001, 100_000), w=16, h=16)   # 1 s
    items = [_clip_item(rec, "a", t0=0.25, t1=0.75, key="A"),
             _clip_item(rec, "b", key="B")]
    items[1]["cell"].time_scale = 0.5                   # 2× slow → twice the program time
    prog = cv.compile_program(items, canvas_wh=(64, 48), gap_s=0.1)

    assert [s.kind for s in prog.segments] == ["clip", "gap", "clip"]
    assert prog.segments[0].duration_s == pytest.approx(0.5)
    assert prog.segments[2].duration_s == pytest.approx(2.0)
    assert prog.duration_s == pytest.approx(2.6)
    # the In point becomes a clock offset, so local time 0 lands on the trimmed start
    cell = prog.segments[0].spec.clips[0]
    assert cell.t_offset_s == pytest.approx(-0.25)
    assert cv.clip_time(cell, rec, 0.0) == pytest.approx(0.25)
    assert cv.clip_time(cell, rec, 0.5) == pytest.approx(0.75)


def test_compile_program_overlays_span_every_segment():
    """Overlay-lane items ride the program clock: a running title lands on the program's
    overlays and an overlay clip is copied into every segment with its offset re-based."""
    a = _point_rec(1, 1, [0, 400_000], w=16, h=16, name="a")
    b = _point_rec(2, 2, [0, 400_000], w=16, h=16, name="b")
    ov = _point_rec(3, 3, np.arange(0, 1_000_001, 50_000), w=16, h=16, name="ov")
    items = [
        _clip_item(a, "a", key="A"),
        _clip_item(b, "b", key="B"),
        _clip_item(ov, "ov", overlay=True, key="OV"),
        {"kind": "title", "name": "run", "overlay": True, "dur": 0.0,
         "text": cv.CanvasText(text="run", kind="overlay", span=None)},
    ]
    prog = cv.compile_program(items)

    assert [s.key for s in prog.segments] == ["A", "B"]        # the overlays take no slot
    assert len(prog.overlays) == 1 and prog.overlays[0].span is None
    assert len(prog.overlay_clips) == 1
    for seg in prog.segments:
        assert len(seg.spec.clips) == seg.n_own + 1            # the overlay cell rides along
        assert seg.spec.clips[-1].t_offset_s == pytest.approx(-seg.t0_s)
    # the flattened event export keeps the sequential lane only — the overlay is not
    # emitted once per segment
    spec, recs = cv.program_spec(prog)
    assert len(spec.clips) == 2 and set(recs.values()) == {a, b}
    assert spec.clips[1].t_offset_s == pytest.approx(prog.segments[1].t0_s)


def test_render_program_frame_follows_the_playhead():
    """The program renders through the one render_frame path: each segment at its own
    local time, with running titles drawn over all of them on the program clock."""
    a = _point_rec(2, 3, np.arange(0, 500_000, 1000), w=10, h=10, name="a")
    b = _point_rec(7, 7, np.arange(0, 500_000, 1000), w=10, h=10, name="b")
    items = [_clip_item(a, "a", key="A"), _clip_item(b, "b", key="B")]
    for it in items:
        it["cell"].colormap = "gray"
    prog = cv.compile_program(items, canvas_wh=(40, 40))

    first = cv.render_program_frame(prog, 0.1)
    assert first.shape == (40, 40, 3) and first.dtype == np.uint8
    assert first[12:15, 8:11].max() == 255                     # a's pixel (2,3) → ×4 scale
    assert first[28:, 28:].max() == 0                          # b's pixel is not showing
    second = cv.render_program_frame(prog, 0.6)                # inside the second segment
    assert second[28:31, 28:31].max() == 255
    assert second[12:15, 8:11].max() == 0

    # a running title draws over whichever segment is live
    titled = cv.compile_program(items + [
        {"kind": "title", "name": "R", "overlay": True, "dur": 0.0,
         "text": cv.CanvasText(text="RUN", kind="overlay", span=None, anchor="n",
                               margin_px=2, font_size_px=12, color=(0, 255, 0),
                               bg_color=None)}], canvas_wh=(40, 40))
    frame = cv.render_program_frame(titled, 0.6)
    ink = (frame[..., 1] > 200) & (frame[..., 0] < 80) & (frame[..., 2] < 80)
    assert ink.any() and ink[:20].any()
    assert cv.render_program_frame(cv.Program(), 0.0).shape == (480, 640, 3)   # empty


def test_export_program_video_when_ffmpeg_available(tmp_path):
    from gottlux.viz import video
    if not video.ffmpeg_available():
        pytest.skip("imageio-ffmpeg not available")
    a = _noise_rec(6000, 0.3, w=32, h=32, seed=11, name="a")
    b = _noise_rec(6000, 0.2, w=32, h=32, seed=12, name="b")
    prog = cv.compile_program([_clip_item(a, "a"), _clip_item(b, "b")], canvas_wh=(48, 48))
    out = str(tmp_path / "program.mp4")
    fracs = []
    res = cv.export_program_video(prog, out, fps=10.0, progress=fracs.append)
    assert res == out and os.path.getsize(out) > 0
    assert len(fracs) == 5                       # 0.5 s of program at 10 fps


# ------------------------------------------------------------------ GUI (offscreen)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_canvas_composer_window_offscreen():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    from gottlux.app.canvas import CanvasComposerWindow

    win = CanvasComposerWindow()
    a = _noise_rec(30000, 1.0, seed=1, name="clip_a")
    b = _noise_rec(24000, 0.6, seed=2, name="clip_b")
    win.add_clip_recording(a, source="a")
    win.add_clip_recording(b, source="b")
    assert len(win.spec.clips) == 2 and len(win._items) == 2
    assert win.clock.t1 >= 0.9                            # timeline spans the longest clip

    win.show()
    win.clock.set_cursor(0.2)
    frame = win.current_frame()                           # one composited engine frame
    assert frame.shape == (win.spec.height, win.spec.width, 3) and frame.max() > 0
    assert all(it._pix is not None for it in win._items)  # both cells rendered live

    # dragging a cell writes its position back into the spec
    win._items[0].setPos(30, 40)
    assert win.spec.clips[0].rect[:2] == (30, 40)

    # the settings panel applies live to the selected clip
    win.clip_list.setCurrentRow(1)
    win.cb_cmap.setCurrentText("gray")
    win.sp_scale.setValue(0.5)
    assert win.spec.clips[1].colormap == "gray"
    assert win.spec.clips[1].time_scale == pytest.approx(0.5)
    assert win.clock.t1 == pytest.approx(1.2, abs=0.1)    # extent grew with the slow-down

    # removing a clip shrinks the composition
    win.clip_list.setCurrentRow(0)
    win._remove_selected()
    assert len(win.spec.clips) == 1 and len(win._items) == 1
    win.close()


def test_canvas_composer_drag_drop_and_text_items(tmp_path):
    """OS drops become cells through the add-clip path (name order; non-recording
    payloads refused), and text items list after the cells with the 'T' prefix, disable
    the per-clip panel, and remove like clips."""
    pytest.importorskip("PySide6")
    from PySide6 import QtCore, QtGui, QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    from gottlux.app.canvas import CanvasComposerWindow
    from gottlux.io import writer

    ra = _noise_rec(4000, 0.2, w=32, h=32, seed=6, name="a")
    rb = _noise_rec(4000, 0.2, w=32, h=32, seed=7, name="b")
    pa, pb = str(tmp_path / "alpha.raw"), str(tmp_path / "beta.raw")
    writer.write_raw(pa, ra.x, ra.y, ra.p, ra.t, width=32, height=32)
    writer.write_raw(pb, rb.x, rb.y, rb.p, rb.t, width=32, height=32)

    win = CanvasComposerWindow()
    assert win.acceptDrops() and not win.view.acceptDrops()   # the window handles drops

    md = QtCore.QMimeData()
    md.setUrls([QtCore.QUrl.fromLocalFile(pb), QtCore.QUrl.fromLocalFile(pa)])
    enter = QtGui.QDragEnterEvent(QtCore.QPoint(4, 4), QtCore.Qt.CopyAction, md,
                                  QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)
    win.dragEnterEvent(enter)
    assert enter.isAccepted()
    win.dropEvent(QtGui.QDropEvent(QtCore.QPointF(4, 4), QtCore.Qt.CopyAction, md,
                                   QtCore.Qt.LeftButton, QtCore.Qt.NoModifier))
    assert [os.path.basename(c.source) for c in win.spec.clips] == \
        ["alpha.raw", "beta.raw"]                             # name order, not drop order

    md_none = QtCore.QMimeData()          # kept alive — the event does not own it
    reject = QtGui.QDragEnterEvent(QtCore.QPoint(4, 4), QtCore.Qt.CopyAction,
                                   md_none, QtCore.Qt.LeftButton,
                                   QtCore.Qt.NoModifier)
    win.dragEnterEvent(reject)
    assert not reject.isAccepted()                            # no recordings → refused

    txt = cv.CanvasText(text="10× slow crop", kind="overlay", span=None, anchor="n")
    win.add_text_item(txt)
    assert win.spec.texts == [txt]
    assert win.clip_list.count() == 3
    assert win.clip_list.item(2).text().startswith("T ")      # titled row, 'T' prefix
    win.clip_list.setCurrentRow(2)                            # a text row: panel off
    assert not win._sel_group.isEnabled()
    win._remove_selected()                                    # removes the text, no cell
    assert win.spec.texts == [] and len(win.spec.clips) == 2
    win.close()


def test_canvas_composer_export_raw_surfaces_text_note(tmp_path, monkeypatch):
    """The composer's .raw export completion dialog carries the text-omission note when
    the composition holds text items (the sidecar still records them)."""
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    import gottlux.io.paths as paths
    from gottlux.app.canvas import CanvasComposerWindow

    win = CanvasComposerWindow()
    win.add_clip_recording(_point_rec(1, 1, [0, 50_000], w=4, h=4), source="pt")
    win.add_text_item(cv.CanvasText(text="T", kind="slide", span=(0.0, 1.0)))
    out = str(tmp_path / "composer.raw")
    infos = []
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *args, **k: (out, "EVT raw (*.raw)")))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *args, **k: infos.append(args)))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical",
                        staticmethod(lambda *args, **k: pytest.fail(f"export failed: {args}")))
    monkeypatch.setattr(paths, "open_in_file_browser", lambda *args, **k: None)
    win._export_raw()
    assert os.path.exists(out)
    assert infos and "1 text item(s) omitted" in infos[0][2]
    assert len(cv.load_spec(os.path.splitext(out)[0] + cv.SPEC_SUFFIX).texts) == 1
    win.close()
