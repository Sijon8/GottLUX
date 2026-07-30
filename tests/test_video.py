"""
Tests for the analysis-video renderer (gottlux.viz.video). The frame rendering
(colourize / draw_box) is verified directly; the MP4 muxing is tolerated whether or not
imageio-ffmpeg is installed (it fails soft to None).
"""
import os

import numpy as np

from gottlux.viz import video
from gottlux.synthetic import synthetic_scene, FlutterTarget


def test_colorize_returns_rgb_uint8():
    frame = np.zeros((40, 50), float); frame[10:20, 15:25] = 7.0
    rgb = video.colorize(frame, cmap="inferno")
    assert rgb.shape == (40, 50, 3) and rgb.dtype == np.uint8


def test_draw_box_outlines_in_color():
    rgb = np.zeros((50, 50, 3), np.uint8)
    video.draw_box(rgb, (10, 12, 30, 34), color=(57, 197, 207), width=2)
    # the box edges carry the colour; the interior stays black
    assert tuple(rgb[12, 20]) == (57, 197, 207)        # top edge
    assert tuple(rgb[34, 20]) == (57, 197, 207)        # bottom edge
    assert tuple(rgb[22, 22]) == (0, 0, 0)             # interior untouched
    # None box is a no-op
    assert video.draw_box(rgb.copy(), None) is not None


def test_infographic_frame_adds_banner():
    f = np.zeros((60, 80, 3), np.uint8)
    out = video.infographic_frame(f, title="cam0 — Live viewer", subtitle="GenX320",
                                  footer_lines=["t = 1.20 s", "events = 12.2M"])
    assert out.dtype == np.uint8 and out.shape[1] == 80
    assert out.shape[0] > 60                          # grew by the title + footer bars


def test_context_poster_builds_infographic():
    f = (np.random.default_rng(0).random((90, 120, 3)) * 255).astype(np.uint8)
    poster = video.context_poster(f, "Export — cam0", {"Sensor": "GenX320", "Events": "12,203,963",
                                                       "Window": "1.0–1.5 s"})
    assert poster.ndim == 3 and poster.shape[2] == 3 and poster.dtype == np.uint8
    assert poster.shape[1] >= 900                     # poster canvas width


def test_export_bundle_writes_context_infographic(tmp_path):
    from gottlux.app.exporting import export_bundle
    rec, _ = synthetic_scene(duration_s=0.5, targets=[FlutterTarget(flutter_hz=150)], seed=1)
    written, manifest = export_bundle(str(tmp_path / "out"), rec, 0.0, 0.4,
                                      want={"infographic"}, purpose="demo", note="context test")
    assert "infographic" in manifest["produced"]
    assert any(p.endswith("_context.png") and os.path.exists(p) for p in written)


def test_overlay_frames_distinct_colors():
    a = np.zeros((20, 20)); a[5:10, 5:10] = 1.0
    b = np.zeros((20, 20)); b[12:16, 12:16] = 1.0
    o = video.overlay_frames([a, b])
    assert o.shape == (20, 20, 3) and o.dtype == np.uint8
    assert tuple(o[7, 7]) == video.OVERLAY_COLORS[0]      # clip 0 region → first colour
    assert tuple(o[13, 13]) == video.OVERLAY_COLORS[1]    # clip 1 region → second colour
    assert tuple(o[0, 0]) == (0, 0, 0)                    # empty stays black
    # mismatched geometries are resized to the largest
    assert video.overlay_frames([a, np.ones((10, 10))]).shape == (20, 20, 3)


def test_apply_colormap_and_resize():
    disp = np.linspace(0, 1, 64).reshape(8, 8)
    rgb = video.apply_colormap(disp, "viridis")
    assert rgb.shape == (8, 8, 3) and rgb.dtype == np.uint8
    big = video.resize_rgb(rgb, (32, 24))
    assert big.shape == (24, 32, 3)


def test_write_video_handles_odd_and_mismatched_frames(tmp_path):
    """Odd dimensions, a stray differently-shaped frame, and 2-D frames must not abort the export
    (they used to crash the H.264 encoder / raise a size-mismatch)."""
    if not video.ffmpeg_available():
        import pytest
        pytest.skip("imageio-ffmpeg not available")
    # odd width AND height
    odd = video.write_video(str(tmp_path / "odd.mp4"),
                            (np.zeros((437, 801, 3), np.uint8) for _ in range(4)), fps=24)
    assert odd and os.path.getsize(odd) > 0
    # one frame in the stream is a different shape (e.g. a render that fell back to a placeholder)
    frames = iter([np.zeros((1080, 1920, 3), np.uint8), np.zeros((8, 8, 3), np.uint8),
                   np.zeros((1080, 1920, 3), np.uint8)])
    mix = video.write_video(str(tmp_path / "mix.mp4"), frames, fps=24)
    assert mix and os.path.getsize(mix) > 0
    # 2-D grayscale frames are promoted to RGB
    gray = video.write_video(str(tmp_path / "gray.mp4"),
                             (np.zeros((100, 120), np.uint8) for _ in range(4)), fps=24)
    assert gray and os.path.getsize(gray) > 0


def test_render_box_track_video_runs(tmp_path):
    """End-to-end render: returns the path (ffmpeg present) or None (absent) — never raises."""
    rec, _ = synthetic_scene(duration_s=0.4, targets=[FlutterTarget(flutter_hz=150)],
                             noise_rate_hz=4000, static_clutter=4, seed=1)
    out = str(tmp_path / "clip_analysis.mp4")
    box_at = lambda t: (40 + 100 * t, 150, 70 + 100 * t, 180)   # a moving box
    res = video.render_box_track_video(rec, out, box_at, 0.0, 0.4, fps=20, accum_dt=0.05,
                                       label_fn=lambda t: f"t={t:.2f}s")
    assert res in (out, None)
    if res is not None:
        assert os.path.exists(out) and os.path.getsize(out) > 0
