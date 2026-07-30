"""
Tests for the deduped primitives introduced in the architectural pass: the one shared
event-frame render pipeline (core/render), the single detections-table flattener
(detectors/base), unique export folders (io/paths), and the settings-aware video export.
"""
import os

import numpy as np
import pytest

from gottlux.synthetic import synthetic_scene, FlutterTarget


# ------------------------------------------------------------------ shared render pipeline
def test_render_frame_modes():
    from gottlux.core.render import render_frame
    rec, _ = synthetic_scene(0.5, targets=[FlutterTarget()], seed=1)
    for mode, lo_hi in (("count", (0.0, 1.0)), ("polarity", (-1.0, 1.0)),
                        ("polarity_ratio", (-1.0, 1.0)), ("time_surface", (0.0, 1.0))):
        disp, levels, vmax, win = render_frame(rec, 0.2, 0.02, mode=mode, expr="sqrt")
        assert levels == lo_hi
        assert disp.shape == (rec.height, rec.width)
        assert win.n >= 0
    # a static white-point reference is echoed back so a "frozen scale" view can hold it
    _, _, vmax, _ = render_frame(rec, 0.2, 0.02, mode="count", vmax_ref=None)
    assert vmax is not None
    d2, _, _, _ = render_frame(rec, 0.2, 0.02, mode="count", vmax_ref=vmax)
    assert d2.max() <= 1.0 + 1e-6


def test_render_frame_direction_windows():
    """back=False integrates ahead of the cursor, back=True behind it — over the same Δ they read
    different events (the windows touch only at the cursor)."""
    from gottlux.core.render import render_frame
    rec, _ = synthetic_scene(1.0, targets=[FlutterTarget()], noise_rate_hz=6000, seed=3)
    _, _, _, fwd = render_frame(rec, 0.5, 0.05, mode="count", back=False)   # [0.50, 0.55]
    _, _, _, bwd = render_frame(rec, 0.5, 0.05, mode="count", back=True)    # [0.45, 0.50]
    assert fwd.n > 0 and bwd.n > 0
    # the two windows abut at the cursor, so their event sets are (essentially) disjoint
    assert fwd.t.min() >= 0.5 * 1e6 - 1 and bwd.t.max() <= 0.5 * 1e6 + 1


# ------------------------------------------------------------------ shared detections table
def test_detections_table_single_source():
    from gottlux.detectors import get_detector
    from gottlux.detectors.base import detections_table
    from gottlux.config import Config
    rec, _ = synthetic_scene(1.2, targets=[FlutterTarget(flutter_hz=200, x0=60, x1=240,
                                                         radius=8)], seed=7, static_clutter=20)
    res = get_detector("drone").run(rec, Config(mode="staring", target_size_m=0.22))
    tbl = detections_table(res)
    for k in ("target_id", "t_s", "range_m", "apparent_px", "snr", "freq_hz", "confidence"):
        assert k in tbl and isinstance(tbl[k], np.ndarray)
    n = sum(t.n for t in res.targets)
    assert len(tbl["t_s"]) == n
    assert detections_table(None)["t_s"].size == 0       # tolerates no result


# ------------------------------------------------------------------ unique export folders
def test_unique_export_dir(tmp_path):
    from gottlux.io.paths import unique_export_dir, safe_name
    assert safe_name("cam0/wide!*") == "cam0_wide"
    d1 = unique_export_dir(str(tmp_path), "cam0/wide", "research", stamp="S")
    d2 = unique_export_dir(str(tmp_path), "cam0/wide", "research", stamp="S")
    assert os.path.isdir(d1) and os.path.isdir(d2) and d1 != d2     # collision-safe
    assert os.path.basename(d1) == "cam0_wide_research_S"
    assert os.path.dirname(d1) == str(tmp_path)


# ------------------------------------------------------------------ settings-aware video export
@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_export_bundle_video_uses_view_render(app, tmp_path):
    from gottlux.app.viewer import LiveViewer
    from gottlux.app.transport import TimeController
    from gottlux.app.exporting import export_bundle
    rec, _ = synthetic_scene(0.6, targets=[FlutterTarget()], seed=1)
    clk = TimeController(); clk.set_range(rec.t_start_s, rec.t_stop_s)
    v = LiveViewer(clk); v.set_recording(rec); v.mode.setCurrentText("count")
    written, manifest = export_bundle(str(tmp_path), rec, 0.0, 0.4, want={"video"},
                                      render=v.capture_frame, sensor_wh=v.sensor_size(),
                                      fps=12, video_res="720p", accum=0.02, purpose="demo")
    mp4 = [p for p in written if p.endswith(".mp4")]
    # the muxer may be unavailable in CI; if present the video must exist and be non-empty
    if "video" in manifest["produced"]:
        assert len(mp4) == 1 and os.path.getsize(mp4[0]) > 0
