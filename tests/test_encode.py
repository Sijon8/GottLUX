"""The one encoder: every format writes what its facts claim, odd sizes, scaling, cancel."""
import os

import numpy as np
import pytest

from gottlux.viz import encode


def _frames(w=81, h=61):
    def frame_at(i):
        f = np.zeros((h, w, 3), np.uint8)
        f[:, (3 * i) % w] = 255
        f[..., 1] = (5 * i) % 256
        return f
    return frame_at


needs_ffmpeg = pytest.mark.skipif(not encode.ffmpeg_ok(), reason="imageio-ffmpeg missing")


@pytest.mark.parametrize("fmt", [pytest.param("mp4", marks=needs_ffmpeg),
                                 pytest.param("webm", marks=needs_ffmpeg), "gif", "png"])
def test_each_format_writes_what_its_facts_say(tmp_path, fmt):
    path = str(tmp_path / ("clip" + encode.FORMATS[fmt][0]))
    seen = []
    facts = encode.encode_frames(_frames(), 20, path, fps=20, progress=seen.append)
    assert facts["format"] == fmt and facts["frames"] == 20
    assert seen[-1] == pytest.approx(1.0) and seen == sorted(seen)
    got = encode.probe(facts["path"])
    assert got["frames"] == 20
    assert (got["width"], got["height"]) == (facts["width"], facts["height"])
    if fmt in ("mp4", "webm"):          # H.264 / VP9 need even sizes: rounded down
        assert (facts["width"], facts["height"]) == (80, 60)
        assert got["fps"] == pytest.approx(20, rel=1e-3)
    else:
        assert (facts["width"], facts["height"]) == (81, 61)
    assert facts["duration_s"] == pytest.approx(20 / facts["fps_effective"])


def test_gif_reports_the_rate_that_really_plays(tmp_path):
    facts = encode.encode_frames(_frames(), 6, str(tmp_path / "a.gif"), fps=30)
    assert facts["frame_ms"] == 30 and facts["fps_effective"] == pytest.approx(1000 / 30)
    assert encode.probe(facts["path"])["fps"] == pytest.approx(1000 / 30)


def test_size_rescales_every_frame(tmp_path):
    facts = encode.encode_frames(_frames(), 4, str(tmp_path / "s.gif"), size=(40, 30))
    assert (facts["width"], facts["height"]) == (40, 30)
    assert encode.probe(facts["path"])["width"] == 40


def test_a_frame_of_another_size_is_coerced_not_fatal(tmp_path):
    base = _frames()

    def frame_at(i):
        return base(i) if i != 2 else np.zeros((10, 12, 3), np.uint8)
    facts = encode.encode_frames(frame_at, 5, str(tmp_path / "c.gif"))
    assert facts["frames"] == 5 and (facts["width"], facts["height"]) == (81, 61)


@pytest.mark.parametrize("fmt", [pytest.param("mp4", marks=needs_ffmpeg), "gif", "png"])
def test_cancel_leaves_nothing_behind(tmp_path, fmt):
    path = str(tmp_path / ("x" + encode.FORMATS[fmt][0]))
    calls = []
    facts = encode.encode_frames(_frames(), 40, path, progress=calls.append,
                                 cancel=lambda: len(calls) >= 5)
    assert facts is None and not os.path.exists(path)


def test_threaded_rendering_keeps_frame_order(tmp_path):
    a = encode.encode_frames(_frames(), 12, str(tmp_path / "a_png"), fmt="png", workers=1)
    b = encode.encode_frames(_frames(), 12, str(tmp_path / "b_png"), fmt="png", workers=4)
    for i in range(12):
        name = f"frame_{i:05d}.png"
        with open(os.path.join(a["path"], name), "rb") as fa, \
                open(os.path.join(b["path"], name), "rb") as fb:
            assert fa.read() == fb.read()


def test_quality_scale_and_path_helpers():
    assert encode.crf_for("mp4", 10) == 16 and encode.crf_for("mp4", 1) == 34
    assert encode.crf_for("webm", 10) == 18 and encode.crf_for("webm", 1) == 50
    assert encode.format_for("a/b.WEBM") == "webm" and encode.format_for("frames") == "png"
    assert encode.with_extension("out.mp4", "gif") == "out.gif"
    assert encode.with_extension("out", "mp4") == "out.mp4"
    assert encode.with_extension("out.gif", "png") == "out"
    with pytest.raises(encode.EncodeError):
        encode.encode_frames(_frames(), 0, "x.gif")
