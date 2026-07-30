"""
Tests for gottlux.app.screenrec — the live screen / view recorder. The Qt chrome (region
overlay, HUD, setup dialog) must construct, the geometry helpers must be right, and the
recorder loop must stream synthetic frames to a valid MP4 (driven manually, no real timer).
"""
import os
import tempfile

import numpy as np
import pytest


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_video_writer_is_robust(tmp_path):
    """The incremental writer survives an odd first frame and a mismatched mid-stream frame."""
    from gottlux.viz.video import VideoWriter, ffmpeg_available
    if not ffmpeg_available():
        pytest.skip("imageio-ffmpeg not available")
    w = VideoWriter(str(tmp_path / "inc.mp4"), fps=20)
    assert w.ok
    for i in range(6):
        shape = (200, 300, 3) if i == 3 else (437, 801, 3)     # one off-size frame mid-stream
        assert w.append(np.full(shape, i * 30, np.uint8))
    path = w.close()
    assert path and os.path.getsize(path) > 0 and w.frames == 6


def test_widget_global_rect(app):
    from PySide6 import QtWidgets
    from gottlux.app.screenrec import widget_global_rect
    w = QtWidgets.QLabel("x"); w.resize(321, 241); w.show()
    r = widget_global_rect(w)
    assert (r.width(), r.height()) == (321, 241)


def test_dialog_and_chrome_construct(app):
    from PySide6 import QtWidgets
    from gottlux.app import screenrec as sr
    win = QtWidgets.QMainWindow(); view = QtWidgets.QLabel("view")
    win.setCentralWidget(view); win.resize(320, 240); win.show()
    d = sr.ScreenRecordDialog(win, view_widget=view, window=win, default_dir=tempfile.gettempdir())
    assert [d.target.itemText(i) for i in range(d.target.count())] == \
        ["Entire app window", "Full screen (desktop)", "Screen region (drag to select)",
         "Current view"]
    assert d.target.currentText() == "Entire app window"      # whole environment is the default
    hud = sr.RecorderHUD(); hud.set_status(75, 91)
    hud.place_clear_of(sr.widget_global_rect(view))            # positions without raising
    ov = sr.RegionOverlay(); assert ov.geometry().width() > 0


def _fake_clock():
    """A controllable monotonic clock: returns (now_fn, advance_fn)."""
    t = [0.0]
    return (lambda: t[0]), (lambda dt: t.__setitem__(0, t[0] + dt))


def test_recorder_streams_frames(app, tmp_path):
    """Paced to the wall clock: advancing time by 1/fps per tick writes exactly one frame per tick,
    so the file's frame count equals elapsed × fps (real-time playback)."""
    from gottlux.viz.video import ffmpeg_available
    if not ffmpeg_available():
        pytest.skip("imageio-ffmpeg not available")
    from gottlux.app.screenrec import ScreenRecorder
    now, advance = _fake_clock()
    frame = np.zeros((240, 320, 3), np.uint8)
    rec = ScreenRecorder(lambda: frame, str(tmp_path / "rec.mp4"), fps=20, clock=now)
    assert rec.ok
    rec.start()
    for _ in range(8):
        advance(1 / 20)                                        # one frame-interval of wall time
        rec._tick()
    path = rec.stop()
    assert path and os.path.getsize(path) > 0 and rec.frames == 8


def test_recorder_stop_is_idempotent(app, tmp_path):
    """The encoder thread owns the close, so stop() finalises exactly once and a second stop() (the
    HUD close + Stop button + app-quit net can all fire) returns the same path without re-closing."""
    from gottlux.viz.video import ffmpeg_available
    if not ffmpeg_available():
        pytest.skip("imageio-ffmpeg not available")
    from gottlux.app.screenrec import ScreenRecorder
    now, advance = _fake_clock()
    frame = np.zeros((120, 160, 3), np.uint8)
    rec = ScreenRecorder(lambda: frame, str(tmp_path / "idem.mp4"), fps=20, clock=now)
    rec.prime(frame)                                          # frame 1 (covers t=0)
    rec.start()
    for _ in range(6):                                        # advance to t=0.30 → target 6 frames
        advance(1 / 20)
        rec._tick()
    p1 = rec.stop()
    p2 = rec.stop()                                            # must not crash or corrupt the file
    assert p1 and os.path.getsize(p1) > 0 and p2 == p1 and rec.frames == 6


def test_recorder_paces_missed_intervals_to_wallclock(app, tmp_path):
    """If a full second of wall-clock passes before a tick (a busy GUI), the recorder back-fills the
    missed frames (duplicating the latest grab) so the frame count tracks elapsed × fps — the fix
    for videos that played fast / looked dropped."""
    from gottlux.viz.video import ffmpeg_available
    if not ffmpeg_available():
        pytest.skip("imageio-ffmpeg not available")
    from gottlux.app.screenrec import ScreenRecorder
    now, advance = _fake_clock()
    frame = np.zeros((120, 160, 3), np.uint8)
    rec = ScreenRecorder(lambda: frame, str(tmp_path / "pace.mp4"), fps=30, clock=now)
    rec.start()
    advance(1.0)                                              # one second elapsed, only now a tick
    rec._tick()
    rec.stop()
    assert rec.frames == 30                                   # ≈ 1 s × 30 fps, not 1


def test_make_scaler_upscales_only():
    """The resolution presets upscale toward a target height but never downscale below the source;
    '2×' always doubles; 'On-screen' is a no-op."""
    from gottlux.app.screenrec import make_scaler
    assert make_scaler("On-screen") is None
    small = np.zeros((240, 320, 3), np.uint8)
    up = make_scaler("1080p")(small)
    assert up.shape[:2] == (1080, 1440)                        # 4:3 preserved, scaled to 1080 tall
    big = np.zeros((1440, 1920, 3), np.uint8)
    assert make_scaler("1080p")(big).shape[:2] == (1440, 1920)  # already taller → left untouched
    assert make_scaler("2×")(small).shape[:2] == (480, 640)


def test_recorder_upscales_and_primes(app, tmp_path):
    """Priming writes the first frame (so FFMPEG warm-up is paid before the clock), and the
    resolution preset is honoured in the saved file."""
    import imageio
    from gottlux.viz.video import ffmpeg_available
    if not ffmpeg_available():
        pytest.skip("imageio-ffmpeg not available")
    from gottlux.app.screenrec import ScreenRecorder, make_scaler
    now, advance = _fake_clock()
    frame = np.zeros((240, 320, 3), np.uint8); frame[:, :, 1] = 120
    out = str(tmp_path / "hi.mp4")
    rec = ScreenRecorder(lambda: frame, out, fps=20, scale=make_scaler("1080p"), clock=now)
    rec.prime(frame)                                           # absorbs FFMPEG startup; frame 1
    rec.start()
    for _ in range(7):                                         # advance to t=0.35 → target 7 frames
        advance(1 / 20)
        rec._tick()
    path = rec.stop()
    assert path and rec.frames == 7                            # primed frame + paced writes
    r = imageio.get_reader(path); h, w = r.get_data(0).shape[:2]; r.close()
    assert (h, w) == (1080, 1440)                              # saved at the chosen resolution
