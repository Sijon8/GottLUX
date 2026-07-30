"""
Tests for the pixels-on-target / perception-range solver (core.photogrammetry), the paper-ready
export bundle (run.resolution_report), and the Range-lab GUI (app.rangelab).
"""
import os

import numpy as np
import pytest

from gottlux.core import photogrammetry as pg


# ------------------------------------------------------------------ core math
def test_pinhole_round_trip():
    fov, W, L = 50.0, 1280.0, 0.30
    f = pg.focal_px(fov, W)
    assert f > 0
    D = 100.0
    n = pg.pixels_on_target(L, D, fov, W)
    # N = L f / D, and range_for_pixels inverts it exactly
    assert np.isclose(n, L * f / D)
    assert np.isclose(pg.range_for_pixels(L, n, fov, W), D)
    # implied size recovers L from a measured box at known range
    assert np.isclose(pg.size_from_pixels(n, D, fov, W), L)


def test_perception_ranges_ordering():
    r = pg.perception_ranges(0.30, 50.0, 1280.0)
    # fewer pixels needed → can be farther: detection > orientation > recognition > identification
    assert r["detection"] > r["orientation"] > r["recognition"] > r["identification"] > 0


def test_fit_target_size_recovers_size():
    fov, W, L = 50.0, 1280.0, 0.30
    f = pg.focal_px(fov, W)
    kfs = []
    for i, D in enumerate([40.0, 80.0, 160.0]):
        n = L * f / D
        kfs.append(pg.Keyframe(t_s=float(i), bbox=(0.0, 0.0, n, n), distance_m=D))
    study = pg.ResolutionStudy(target_size_m=0.25, fov_deg=fov, width_px=int(W),
                               height_px=720, keyframes=kfs)
    fit = study.fit_target_size()
    assert fit["n"] == 3
    assert abs(fit["fitted_target_size_m"] - L) < 1e-6
    assert fit["r2"] is None or fit["r2"] > 0.999


def test_summary_structure():
    study = pg.ResolutionStudy(0.3, 50.0, 1280, 720,
                               [pg.Keyframe(1.0, (10, 10, 24, 24), distance_m=50.0, label="a")])
    s = study.summary()
    assert "perception_ranges_m" in s and "keyframes" in s
    assert s["keyframes"][0]["measured_px_on_target"] == 14.0


# ------------------------------------------------------------------ export bundle
def test_save_resolution_study_writes_bundle(tmp_path):
    from gottlux.run.resolution_report import save_resolution_study
    fov, W, L = 50.0, 1280.0, 0.30
    f = pg.focal_px(fov, W)
    kfs = [pg.Keyframe(float(i), (0.0, 0.0, L * f / D, L * f / D), distance_m=D, label=f"k{i}")
           for i, D in enumerate([40.0, 90.0, 200.0])]
    study = pg.ResolutionStudy(L, fov, int(W), 720, kfs)
    written = save_resolution_study(str(tmp_path / "study"), study, title="t")
    names = [os.path.basename(p) for p in written]
    assert any(n.endswith("_pixels_on_target.pdf") for n in names)
    assert any(n.endswith("_pixels_on_target.png") for n in names)
    assert any(n.endswith("_keyframes.csv") for n in names)
    assert any(n.endswith("_resolution_solved.csv") for n in names)
    assert any(n.endswith("_resolution.json") for n in names)
    assert any(n.endswith("_resolution_report.md") for n in names)
    for p in written:
        assert os.path.exists(p) and os.path.getsize(p) > 0


# ------------------------------------------------------------------ GUI
@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _synth_rec(w=1280, h=720, n=40000, dur=1.0, seed=0):
    from gottlux.io.recording import Recording
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, dur, n) * 1e6).astype(np.int64)
    x = rng.integers(0, w, n); y = rng.integers(0, h, n); p = rng.integers(0, 2, n)
    return Recording.from_events(x, y, p, t, width=w, height=h, name="cam1_imx636")


def test_rangelab_keyframe_and_solve(app, tmp_path):
    from gottlux.app.rangelab import RangeLab
    from gottlux.app.transport import TimeController
    from gottlux.run.resolution_report import save_resolution_study

    clk = TimeController()
    lab = RangeLab(clk)
    rec = _synth_rec()
    lab.set_recording(rec)
    clk.set_range(rec.t_start_s, rec.t_stop_s)
    lab.show()

    # box around a "drone" and a keyframe at t=0.2 with a known range
    lab._apply_box((600, 350, 614, 364))     # 14 px box
    clk.set_cursor(0.2)
    lab.dist.setValue(50.0)
    lab._add_keyframe()
    lab._apply_box((600, 350, 607, 357))     # 7 px box, farther
    clk.set_cursor(0.6)
    lab.dist.setValue(100.0)
    lab._add_keyframe()
    assert len(lab.keyframes) == 2
    assert lab.table.rowCount() == 2

    # interpolation produces a box between the two keyframes
    b = lab.box_at(0.4)
    assert b is not None and 7 <= (b[2] - b[0]) <= 14

    # solve runs and the study exports a bundle
    lab.fov.setValue(50.0); lab.size_m.setValue(0.30)
    lab._solve()
    written = save_resolution_study(str(tmp_path / rec.name), lab._study())
    assert any(p.endswith("_resolution_report.md") for p in written)


def test_rangelab_is_single_clip(app):
    """The Range lab is single-clip — dual-clip fusion moved to the Multi-clip tab."""
    from gottlux.app.rangelab import RangeLab
    from gottlux.app.transport import TimeController
    lab = RangeLab(TimeController())
    lab.set_recording(_synth_rec(w=320, h=320))
    assert not hasattr(lab, "view_sel") and not hasattr(lab, "load_b_btn")
    # keyframing + a single study still work
    lab._apply_box((150, 150, 170, 170)); lab.ctl.set_cursor(0.2)
    lab.dist.setValue(10.0); lab._add_keyframe()
    assert lab._study().keyframes
    # faithful high-res capture (event frame + box) works at any resolution
    assert lab.sensor_size() == (320, 320)
    assert lab.capture_frame(0.2).shape == (320, 320, 3)
    assert lab.capture_frame(0.2, size=(1080, 1080)).shape == (1080, 1080, 3)
