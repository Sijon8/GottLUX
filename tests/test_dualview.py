"""
Tests for the dual-clip co-registration + converged study (gottlux.core.dualview) and the
raw clip stitcher (gottlux.io.writer.stitch_clips).
"""
import numpy as np
import pytest

from gottlux.core import dualview as dv
from gottlux.core import photogrammetry as pg
from gottlux.core.photogrammetry import ResolutionStudy, Keyframe


# ------------------------------------------------------------------ co-registration
def test_view_geom_vertical_fov_and_scales():
    g = dv.ViewGeom(58.0, 320, 320)
    assert g.fov_v_deg == pytest.approx(58.0, abs=1e-6)        # square sensor → equal FOVs
    assert g.dpp_x == pytest.approx(58.0 / 320)
    g2 = dv.ViewGeom(58.0, 320, 240)
    assert g2.fov_v_deg < g2.fov_h_deg                          # shorter axis → narrower FOV


def test_fov_scale_maps_by_fov_ratio():
    wide, narrow = dv.ViewGeom(58.0, 320, 320), dv.ViewGeom(40.0, 320, 320)
    cr = dv.CoRegistration(mode="fov_scale")
    # a point on the centre line stays centred
    xs, ys = dv.map_point(160.0, 160.0, wide, narrow, cr)
    assert xs == pytest.approx(160.0) and ys == pytest.approx(160.0)
    # an off-centre point scales by the FOV ratio (58/40 = 1.45) about the centre
    xs, _ = dv.map_point(60.0, 160.0, wide, narrow, cr)
    assert xs == pytest.approx(160.0 + (60.0 - 160.0) * (58.0 / 40.0), abs=1e-6)


def test_none_mode_does_not_superimpose():
    cr = dv.CoRegistration(mode="none")
    assert cr.superimpose is False
    assert dv.map_point(60, 60, dv.ViewGeom(58, 320, 320), dv.ViewGeom(40, 320, 320), cr) is None
    assert dv.map_box((10, 10, 30, 30), dv.ViewGeom(58, 320, 320),
                      dv.ViewGeom(40, 320, 320), cr) is None


def test_parallax_adds_range_dependent_disparity():
    wide, narrow = dv.ViewGeom(58.0, 320, 320), dv.ViewGeom(40.0, 320, 320)
    base = dv.CoRegistration(mode="fov_scale")
    par = dv.CoRegistration(mode="parallax", baseline_m=0.025)
    x_base, _ = dv.map_point(160.0, 160.0, wide, narrow, base)
    x_near, _ = dv.map_point(160.0, 160.0, wide, narrow, par, range_m=5.0)
    x_far, _ = dv.map_point(160.0, 160.0, wide, narrow, par, range_m=100.0)
    # disparity ∝ 1/range: larger near, vanishing far, and = baseline·f/range
    assert (x_near - x_base) > (x_far - x_base) > 0
    assert (x_near - x_base) == pytest.approx(0.025 * narrow.focal_px() / 5.0, rel=1e-6)


def test_manual_affine_nudge():
    cr = dv.CoRegistration(mode="manual", manual_dx=5.0, manual_dy=-3.0, manual_scale=2.0)
    src = dst = dv.ViewGeom(58.0, 320, 320)
    xs, ys = dv.map_point(160.0, 160.0, src, dst, cr)
    assert xs == pytest.approx(165.0) and ys == pytest.approx(157.0)   # centre + (dx, dy)


def test_map_box_returns_bounds():
    wide, narrow = dv.ViewGeom(58.0, 320, 320), dv.ViewGeom(40.0, 320, 320)
    b = dv.map_box((150, 150, 170, 170), wide, narrow, dv.CoRegistration(mode="fov_scale"))
    assert b is not None and b[2] > b[0] and b[3] > b[1]


# ------------------------------------------------------------------ converged study
def test_pooled_fit_recovers_size_across_two_optics():
    L = 0.22
    fa, fb = pg.focal_px(58, 320), pg.focal_px(40, 320)
    pts = [(L * fa / 10, 10, fa), (L * fa / 20, 20, fa),     # wide clip
           (L * fb / 30, 30, fb), (L * fb / 50, 50, fb)]     # narrow clip
    fit = dv.pooled_target_size_fit(pts)
    assert fit["n"] == 4
    assert fit["fitted_target_size_m"] == pytest.approx(L, abs=1e-3)
    assert fit["r2"] > 0.999


def test_converged_study_summary():
    fa, fb = pg.focal_px(58, 320), pg.focal_px(40, 320)
    A = ResolutionStudy(0.25, 58.0, 320, 320,
                        [Keyframe(1.0, (0, 0, 0.22 * fa / 10, 0.22 * fa / 10), distance_m=10.0)])
    B = ResolutionStudy(0.25, 40.0, 320, 320,
                        [Keyframe(1.0, (0, 0, 0.22 * fb / 30, 0.22 * fb / 30), distance_m=30.0)])
    s = dv.ConvergedStudy("wide", A, "narrow", B).summary()
    assert s["converged"] and set(s["clips"]) == {"wide", "narrow"}
    assert s["fused_target_size_m"] == pytest.approx(0.22, abs=1e-2)
    # narrower FOV resolves a given task at longer range
    assert (s["clips"]["narrow"]["perception_ranges_m"]["recognition"]
            > s["clips"]["wide"]["perception_ranges_m"]["recognition"])


# ------------------------------------------------------------------ stitch
def _scene(dur, seed, name):
    from gottlux.synthetic import synthetic_scene, FlutterTarget
    rec, _ = synthetic_scene(duration_s=dur, targets=[FlutterTarget(flutter_hz=150)],
                             noise_rate_hz=5000, static_clutter=5, seed=seed)
    rec.name = name
    return rec


def test_stitch_clips_concatenates_with_rebased_time(tmp_path):
    from gottlux.io import writer
    from gottlux.io.recording import load
    a, b = _scene(0.5, 1, "a"), _scene(0.4, 2, "b")
    out = str(tmp_path / "stitched.raw")
    res = writer.stitch_clips(out, [a, (b, None, None)], gap_s=0.1)
    assert res["n_events"] == a.n + b.n
    assert len(res["segments"]) == 2
    # second segment starts after the first + the gap
    assert res["segments"][1]["placed_t0_s"] >= res["segments"][0]["placed_t1_s"] + 0.1 - 1e-6
    # the output is a valid, readable .raw whose duration spans both clips
    rec = load(out)
    assert rec.n == a.n + b.n
    assert rec.duration_s >= 0.8


def test_stitch_rejects_mismatched_geometry(tmp_path):
    from gottlux.io import writer
    from gottlux.synthetic import synthetic_scene, FlutterTarget
    a = _scene(0.3, 1, "a")
    b, _ = synthetic_scene(duration_s=0.3, width=640, height=480,
                           targets=[FlutterTarget(flutter_hz=150)], seed=4)
    with pytest.raises(ValueError):
        writer.stitch_clips(str(tmp_path / "x.raw"), [a, b])


# ------------------------------------------------------------------ export class/note
def test_export_bundle_records_purpose_and_note(tmp_path):
    import json
    import os
    from gottlux.app.exporting import export_bundle, PURPOSES
    rec = _scene(0.5, 1, "clip")
    written, manifest = export_bundle(str(tmp_path / "out"), rec, 0.0, 0.4,
                                      want={"config", "event_rate"},
                                      purpose="demo", note="for the SOF Week demo")
    assert manifest["purpose"] == "demo" and "demo" in PURPOSES
    assert manifest["note"] == "for the SOF Week demo"
    # files land in a unique, helpfully-named subfolder inside the chosen folder
    out_dir = manifest["out_dir"]
    assert os.path.dirname(out_dir) == str(tmp_path / "out")
    assert "demo" in os.path.basename(out_dir)
    saved = json.load(open(os.path.join(out_dir, "manifest.json"), encoding="utf-8"))
    assert saved["purpose"] == "demo" and saved["note"] == "for the SOF Week demo"
    # an unknown purpose falls back to the default
    _, m2 = export_bundle(str(tmp_path / "o2"), rec, 0.0, 0.4, want={"config"}, purpose="bogus")
    assert m2["purpose"] == "research"


# ------------------------------------------------------------------ CLI cut / stitch
def test_cli_cut_and_stitch(tmp_path):
    from gottlux.io import writer
    from gottlux.io.recording import load
    from gottlux.cli import main
    rng = np.random.default_rng(0)

    def mkraw(path, n, dur):
        t = np.sort(rng.uniform(0, dur, n) * 1e6).astype(np.int64)
        writer.write_raw(str(path), rng.integers(0, 320, n), rng.integers(0, 320, n),
                         rng.integers(0, 2, n), t, width=320, height=320)

    a, b = tmp_path / "a.raw", tmp_path / "b.raw"
    mkraw(a, 4000, 0.5); mkraw(b, 3000, 0.4)

    out_cut = tmp_path / "a_cut.raw"
    assert main(["gottlux", str(a), "--cut", "0.1,0.3", "--out_raw", str(out_cut), "--no_open"]) == 0
    assert out_cut.exists() and load(str(out_cut)).n > 0

    out_stitch = tmp_path / "stitched.raw"
    assert main(["gottlux", str(a), "--stitch", str(b), "--out_raw", str(out_stitch),
                 "--no_open"]) == 0
    assert load(str(out_stitch)).n > 0
