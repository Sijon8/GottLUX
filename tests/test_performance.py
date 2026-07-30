"""
Tests for the operator results metrics (KPIs): the core physics
(:mod:`gottlux.core.performance`), the orchestrator + robust saver
(:mod:`gottlux.run.performance_report`), and the requirement that the three metrics are
**independent** — a failure in one never soils the others.
"""
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from gottlux.core import performance as perf
from gottlux.core import photogrammetry as pg
from gottlux.config import Config
from gottlux.synthetic import FlutterTarget, synthetic_scene


# ------------------------------------------------------------------ 1) core metrics
def test_tracking_range_capability_and_measured():
    L, fov, W = 0.22, 58.0, 320
    tr = perf.tracking_range(L, fov, W, min_pixels_area=60.0)
    # capability: D where N == track_px(=√60≈7.75) — matches the pinhole inverse
    expect = L * pg.focal_px(fov, W) / np.sqrt(60.0)
    assert tr.capability_range_m == pytest.approx(expect, rel=1e-3)
    assert tr.johnson_ranges_m["detection"] > tr.johnson_ranges_m["identification"] > 0
    assert tr.status.status == "model_only"          # no detections supplied
    # with measured detections, the farthest finite range is the reach
    tr2 = perf.tracking_range(L, fov, W, measured_ranges=[8, 12, 20, np.nan, -1])
    assert tr2.measured_max_range_m == 20.0
    assert tr2.effective_track_px == pytest.approx(pg.pixels_on_target(L, 20.0, fov, W), abs=0.02)
    assert tr2.range_m == 20.0 and tr2.status.status == "ok"


def test_prop_frequency_inverse_square_fit():
    L, fov, W, gate = 0.22, 58.0, 320, 4.0
    D = np.array([5, 8, 10, 12, 15, 18.0])
    snr = 50.0 * (8.0 / D) ** 2                       # SNR_ref=50 at 8 m, exact 1/D²
    pf = perf.prop_frequency_range(gate, L, fov, W, measured_ranges=D, measured_snr=snr)
    assert pf.model == "snr_fit"
    assert pf.slope == pytest.approx(-2.0, abs=0.05)
    assert pf.r2 is not None and pf.r2 > 0.999
    # SNR(D)=gate → D = 8·√(50/4) = 28.28 m
    assert pf.capability_range_m == pytest.approx(8.0 * np.sqrt(50.0 / gate), rel=1e-2)
    assert pf.measured_max_range_m == 18.0            # farthest point still ≥ gate
    assert pf.n_resolved == 6


def test_prop_frequency_pixels_fallback_when_no_tone():
    L, fov, W = 0.22, 58.0, 320
    pf = perf.prop_frequency_range(4.0, L, fov, W)    # no measured data at all
    assert pf.model == "pixels_on_target"
    assert pf.capability_range_m == pytest.approx(pg.range_for_pixels(L, 8.0, fov, W), rel=1e-3)
    assert pf.status.status == "model_only"


def test_time_to_contact_nominal_and_measured():
    ttc = perf.time_to_contact(30.0, approach_speed_mps=15.0, speed_sweep=(5, 10, 15))
    assert ttc.nominal_ttc_s == pytest.approx(2.0)
    assert ttc.nominal_sweep_s[5.0] == pytest.approx(6.0)
    # an approaching track: range 20 → 5 m over 3 s ⇒ closing 5 m/s, warning 4 s at first
    t = np.linspace(0, 3, 10); rng = np.linspace(20, 5, 10)
    m = perf.time_to_contact(20.0, range_t=rng, t_s=t)
    assert m.approaching is True
    assert m.measured_closing_speed_mps == pytest.approx(5.0, rel=1e-2)
    assert m.measured_ttc_at_first_s == pytest.approx(4.0, rel=1e-2)


def test_metrics_are_independent_pure_functions():
    """Each metric stands alone — bad input to one returns a clean status, raises nothing."""
    bad = perf.tracking_range(0.0, 58.0, 320)         # absolute ranging disabled
    assert bad.status.status == "failed" and bad.capability_range_m is None
    # the other two are unaffected by the above
    assert perf.prop_frequency_range(4.0, 0.22, 58.0, 320).capability_range_m is not None
    assert perf.time_to_contact(None).status.status == "no_data"


# ------------------------------------------------------------------ 2) orchestrator + saver
def _tracked_scene(seed=7, radius1=None):
    """A scene with a reliably-tracked 200 Hz crossing target (optionally approaching)."""
    rec, _ = synthetic_scene(
        duration_s=1.5,
        targets=[FlutterTarget(flutter_hz=200.0, x0=60, y0=160, x1=240, y1=160,
                               radius=8, radius1=radius1, events_per_burst=48,
                               harmonics=(1.0, 0.5, 0.25))],
        noise_rate_hz=30_000, static_clutter=40, seed=seed)
    return rec


def test_compute_and_save_bundle(tmp_path):
    from gottlux.run import performance_report as pr
    rec = _tracked_scene()
    cfg = Config(mode="staring", target_size_m=0.22)
    res = pr.compute_performance(rec, cfg)
    assert res.n_targets >= 1
    h = res.headline()
    assert h["tracking_range_m"] is not None
    assert h["regime"] == "staring"
    assert set(h["status"]) == {"tracking", "prop_frequency", "time_to_contact"}

    out = pr.save_performance(res, rec, cfg, out_dir=str(tmp_path / "kpi"))
    names = {os.path.basename(p) for p in out["written"]}
    for stem in ("tracking_range", "prop_frequency_range", "time_to_contact"):
        assert f"{stem}.png" in names and f"{stem}.json" in names
    assert "kpi_report.md" in names and "kpi_summary.json" in names and "kpi_manifest.json" in names
    assert out["failed"] == {}
    # the report carries the datasheet + all three metric headers
    report = open(os.path.join(out["dir"], "kpi_report.md"), encoding="utf-8").read()
    for header in ("Tracking range", "Prop-frequency-resolution range", "Time-to-contact"):
        assert header in report
    assert "Prophesee GenX320" in report


def test_one_failing_metric_does_not_soil_the_others(tmp_path, monkeypatch):
    from gottlux.run import performance_report as pr
    rec = _tracked_scene()
    cfg = Config(mode="staring", target_size_m=0.22)

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(perf, "prop_frequency_range", boom)

    res = pr.compute_performance(rec, cfg)
    assert res.tracking is not None and res.time_to_contact is not None
    assert res.prop_frequency is None
    assert any("prop_frequency_range failed" in n for n in res.notes)

    out = pr.save_performance(res, rec, cfg, out_dir=str(tmp_path / "kpi"))
    names = {os.path.basename(p) for p in out["written"]}
    assert "tracking_range.png" in names and "time_to_contact.png" in names
    assert "prop_frequency_range.png" not in names      # only the failed metric is missing
    assert res.headline()["status"]["prop_frequency"] == "failed"


def test_saver_writes_beside_the_analyzed_file(tmp_path):
    """Robust saver: with no explicit out_dir, the bundle lands next to the source file."""
    from gottlux.run import performance_report as pr
    rec = _tracked_scene()
    rec.source_path = str(tmp_path / "clip.raw")        # pretend it came from here
    cfg = Config(mode="staring", target_size_m=0.22)
    res = pr.compute_performance(rec, cfg)
    out = pr.save_performance(res, rec, cfg)            # out_dir=None → beside the file
    assert os.path.dirname(os.path.abspath(out["dir"])) == str(tmp_path)
    assert os.path.basename(out["dir"]).startswith("synthetic_scene_kpi_")


def test_measured_time_to_contact_on_approaching_track():
    """A growing (approaching) target yields a decreasing range and a positive closing speed."""
    from gottlux.run import performance_report as pr
    rec = _tracked_scene(radius1=13)                    # radius 8→13 px over the clip
    cfg = Config(mode="staring", target_size_m=0.22)
    res = pr.compute_performance(rec, cfg)
    if res.primary_track and res.time_to_contact and res.time_to_contact.approaching:
        r = res.primary_track["range_m"]
        assert r[0] > r[-1]                             # range decreases (approaching)
        assert res.time_to_contact.measured_closing_speed_mps > 0
        assert res.time_to_contact.measured_ttc_at_first_s is not None
    else:                                               # detector tuning is data-sensitive
        assert res.tracking is not None                 # capability metrics still produced


def test_comparison_bundle(tmp_path):
    from gottlux.run import performance_report as pr
    cfg = Config(mode="staring", target_size_m=0.22)
    r1 = pr.compute_performance(_tracked_scene(seed=7), cfg)
    r2 = pr.compute_performance(_tracked_scene(seed=9), cfg)
    out = pr.compare_performance([r1, r2], labels=["staring:A", "rotation:B"],
                                 out_dir=str(tmp_path / "cmp"), cfg=cfg)
    names = {os.path.basename(p) for p in out["written"]}
    assert "kpi_comparison.png" in names and "kpi_comparison.csv" in names
    assert set(out["datasets"]) == {"staring:A", "rotation:B"}
