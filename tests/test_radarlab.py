"""Tests for the Radar/Box Lab (gottlux.app.radarlab): the pure de-rotation/solve math and an
offscreen smoke build of the Qt widget."""
import os
import types

import numpy as np
import pytest

from gottlux.app import radarlab
from gottlux.io.telemetry import Telemetry
from gottlux.rotation.detect import focal_px

W = H = 320
T_ROT = 1.0
FOV = 58.0


def _tel(duration=10.0):
    return Telemetry.from_spin(duration, T_ROT)


def _drone_events(target_az=45.0, y0=150, y1=190, n_rev=8, per=400, x_col=W // 2):
    """Synthetic drone: a burst at the centre column each revolution, when the boresight points at
    target_az, spread over y0..y1 — so its de-rotated world azimuth is ~target_az every pass."""
    tel = _tel(n_rev + 1.0)
    t_phase = (target_az / 360.0) * T_ROT                  # boresight = target_az at this phase
    xs, ys, ts = [], [], []
    rng = np.random.default_rng(0)
    for k in range(n_rev):
        t0 = t_phase + k * T_ROT
        ts.append(t0 + rng.normal(0, 0.004, per))
        xs.append(np.full(per, x_col, float))
        ys.append(rng.uniform(y0, y1, per))
    return tel, np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)


def test_world_azimuth_centre_column_is_boresight():
    tel = _tel()
    t = np.array([0.25, 0.5, 0.75])                        # boresight 90,180,270 deg with T_rot=1
    az = radarlab.world_azimuth(tel, np.full(3, W / 2), t, FOV, W)
    np.testing.assert_allclose(az, [90, 180, 270], atol=1.0)


def test_solve_box_recovers_bearing_and_range():
    tel, x, y, t = _drone_events(target_az=45.0, y0=150, y1=190)
    azw = radarlab.world_azimuth(tel, x, t, FOV, W)
    r = radarlab.solve_box(azw, y, t, az=(40, 50), ywin=(140, 200), fov_deg=FOV, width=W,
                           target_size_m=0.225, min_pass=50)
    assert r["n_passes"] >= 6
    assert abs(r["bearing_deg"] - 45.0) < 1.0              # de-rotated centroid ≈ target azimuth
    # range = size * focal_px / y_extent ; y_extent ≈ 40 px (the 150..190 spread)
    expected = 0.225 * focal_px(FOV, W) / 40.0
    assert 0.5 * expected < r["range_m"] < 2.0 * expected
    assert all(d["bearing_SE_deg"] < 0.5 for d in r["track"])


def test_solve_box_sparse_returns_empty():
    tel, x, y, t = _drone_events(per=5)
    azw = radarlab.world_azimuth(tel, x, t, FOV, W)
    r = radarlab.solve_box(azw, y, t, az=(40, 50), ywin=(140, 200), fov_deg=FOV, width=W, min_pass=120)
    assert r["n_passes"] == 0 and r["bearing_deg"] is None


def test_static_keep_mask_drops_recurring_voxels():
    tel = _tel(8.0)
    rng = np.random.default_rng(1)
    # static feature: same (phase,x,y) every revolution → should be dropped
    n_rev = 7
    st_t = np.array([0.3 + k for k in range(n_rev)], float)
    st_x = np.full(n_rev, 100.0); st_y = np.full(n_rev, 80.0)
    # transient: a single late burst at a new voxel → should be kept
    tr_t = np.array([6.31, 6.32, 6.33]); tr_x = np.full(3, 200.0); tr_y = np.full(3, 250.0)
    x = np.concatenate([st_x, tr_x]); y = np.concatenate([st_y, tr_y]); t = np.concatenate([st_t, tr_t])
    keep = radarlab.static_keep_mask(tel, x, y, t, W, H, n_rev=3)
    assert not keep[:n_rev].any()                          # static recurring → dropped
    assert keep[n_rev:].all()                              # transient → kept


def test_static_keep_mask_off_keeps_all():
    tel = _tel(4.0)
    t = np.linspace(0, 3, 100); x = np.full(100, 50.0); y = np.full(100, 50.0)
    assert radarlab.static_keep_mask(tel, x, y, t, W, H, n_rev=0).all()


def test_gui_constructs_offscreen():
    """Build the Qt widget headlessly (offscreen) on a synthetic recording — catches import/wiring
    regressions without a display."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6 import QtWidgets

    tel, x, y, t = _drone_events(target_az=45.0)
    rec = types.SimpleNamespace(
        width=W, height=H, name="synthetic", telemetry=tel,
        all=lambda: types.SimpleNamespace(x=x, y=y, t_s=t))

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    RadarBoxLab = radarlab._build_gui()
    win = RadarBoxLab(rec, fov_deg=FOV, target_size_m=0.225, tag="synthetic")
    # current box solves; force a box over the drone and add it
    win.box.setPos([40, 140]); win.box.setSize([10, 60])
    win._solve_current()
    win._add_box()
    assert len(win.boxes) == 1
    assert abs(win.boxes[0]["bearing_deg"] - 45.0) < 2.0
    win._suggest_box()                                     # snap-to-drone wiring works
    win.deleteLater()


def test_revolution_windows_are_ordered_and_cover():
    tel = _tel(8.0)
    wins = radarlab.revolution_windows(tel)
    assert len(wins) >= 5
    assert all(b > a for a, b in wins)                     # each window non-empty
    assert all(wins[i][1] <= wins[i + 1][0] + 1e-6 for i in range(len(wins) - 1))  # ordered
    # clipping to a sub-range keeps only overlapping windows
    sub = radarlab.revolution_windows(tel, t_min=2.0, t_max=4.0)
    assert all(b > 2.0 and a < 4.0 for a, b in sub)


def test_column_subset_keep_modes():
    x = np.arange(320, dtype=float)
    assert radarlab.column_subset_keep(x, 320, "all").all()
    assert radarlab.column_subset_keep(x, 320, "single").sum() == 1          # exactly the centre col
    few = radarlab.column_subset_keep(x, 320, "few", n_cols=8)
    assert 7 <= few.sum() <= 10 and few[160]                                  # ~8 central columns


def test_track_over_revs_returns_per_rev_bearings():
    tel, x, y, t = _drone_events(target_az=120.0, n_rev=8, per=300)
    azw = radarlab.world_azimuth(tel, x, t, FOV, W)
    per = radarlab.track_over_revs(azw, y, t, tel, az=(110, 130), ywin=(140, 200),
                                   fov_deg=FOV, width=W, min_pass=40)
    assert len(per) >= 6
    assert all(abs(d["bearing_deg"] - 120.0) < 2.0 for d in per)
    assert [d["rev"] for d in per] == sorted(d["rev"] for d in per)           # chronological


def test_box_spectrum_returns_spectrum():
    tel, x, y, t = _drone_events(per=500)
    azw = radarlab.world_azimuth(tel, x, t, FOV, W)
    m = (azw >= 40) & (azw <= 50)
    spec = radarlab.box_spectrum(t[m], spin_hz=1.0)
    assert spec.freqs.size > 0 and spec.power.size == spec.freqs.size


def test_figures_write(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    tel, x, y, t = _drone_events(target_az=45.0, per=400)
    azw = radarlab.world_azimuth(tel, x, t, FOV, W)
    out = str(tmp_path)
    box = {"az_window": [40, 50], "y_window": [150, 190], "bearing_deg": 45.0, "range_m": 5.0}
    assert os.path.exists(radarlab.figure_context(azw, y, H, [box], "synthetic", out)[0])
    per = radarlab.track_over_revs(azw, y, t, tel, az=(40, 50), ywin=(140, 200), fov_deg=FOV, width=W, min_pass=40)
    assert os.path.exists(radarlab.figure_timeline(per, "synthetic", out)[0])
    spec = radarlab.box_spectrum(t[(azw >= 40) & (azw <= 50)], spin_hz=1.0)
    assert os.path.exists(radarlab.figure_fft(spec, "synthetic", out, box=(40, 50, 150, 190))[0])
    assert os.path.exists(radarlab.figure_column_comparison(azw, y, x, t, W, H, box=(40, 50, 150, 190),
                                                            tag="synthetic", out_dir=out)[0])


def test_densest_box_finds_the_drone():
    tel, x, y, t = _drone_events(target_az=200.0, y0=150, y1=190, per=400)
    azw = radarlab.world_azimuth(tel, x, t, FOV, W)
    box = radarlab.densest_box(azw, y, height=H)
    assert box is not None
    az0, az1, y0, y1 = box
    assert az0 <= 200.0 <= az1                              # box brackets the drone azimuth
    assert y0 <= 170 <= y1                                  # and its Y band
    assert radarlab.densest_box(azw[:10], y[:10], height=H) is None   # too few events → None


def test_polygon_keep_lasso():
    # a quad covering az 40-50, y 140-200
    verts = [(40, 140), (50, 140), (50, 200), (40, 200)]
    azw = np.array([45, 60, 45, 45.0]); y = np.array([170, 170, 100, 250.0])
    keep = radarlab.polygon_keep(azw, y, verts)
    assert keep.tolist() == [True, False, False, False]
    assert not radarlab.polygon_keep(azw, y, [(0, 0), (1, 1)]).any()      # <3 verts → empty


def test_solve_box_with_sel_matches_gating():
    tel, x, y, t = _drone_events(target_az=45.0, y0=150, y1=190)
    azw = radarlab.world_azimuth(tel, x, t, FOV, W)
    box = dict(az=(40, 50), ywin=(140, 200), fov_deg=FOV, width=W, min_pass=50)
    r_box = radarlab.solve_box(azw, y, t, **box)
    sel = (azw >= 40) & (azw <= 50) & (y >= 140) & (y <= 200)
    r_sel = radarlab.solve_box(azw, y, t, sel=sel, **box)
    assert r_sel["n_passes"] == r_box["n_passes"]
    assert abs(r_sel["bearing_deg"] - r_box["bearing_deg"]) < 1e-6


def test_masking_metrics_quantifies():
    tel, x, y, t = _drone_events(target_az=45.0, per=300, n_rev=6)
    mm = radarlab.masking_metrics(tel, x, y, t, width=W, height=H,
                                  drone_box=(40, 50, 140, 200, 1.0, FOV), n_list=(0, 1, 2, 3))
    red = [r["reduction_pct"] for r in mm["rows"]]
    assert red[0] == 0.0 and all(red[i] <= red[i + 1] + 1e-6 for i in range(len(red) - 1))  # monotone↑
    assert all("event_rate_Mev_s" in r and "drone_retained_pct" in r for r in mm["rows"])


def test_figure_masking_quant_writes(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    tel, x, y, t = _drone_events(per=300, n_rev=6)
    mm = radarlab.masking_metrics(tel, x, y, t, width=W, height=H,
                                  drone_box=(40, 50, 140, 200, 1.0, FOV), n_list=(0, 1, 2, 3))
    assert os.path.exists(radarlab.figure_masking_quant(mm, "synthetic", str(tmp_path))[0])


def test_default_manifest_structure():
    m = radarlab.default_manifest("t", fov_deg=20, target_size_m=0.225, az_sign=1.0)
    assert set(m["plots"]) == {"experiment_summary", "context", "panorama", "radar", "ebs_radar_map",
                               "timeline", "masking_quant", "event_rate", "mask_reduction", "mask_panoramas",
                               "fft_vs_rev", "box_fft_dashboard", "egomotion_correction", "drone_detector",
                               "column_comparison", "fft"}
    assert all("enabled" in p and "style" in p for p in m["plots"].values())


def test_manifest_only_renders_selected(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    tel, x, y, t = _drone_events(target_az=45.0, y0=150, y1=190, per=400, n_rev=8)
    boxes = [{"az_window": [40, 50], "y_window": [145, 195], "bearing_deg": 45.0, "range_m": 5.0,
              "t_window": [0.0, 8.0], "n_passes": 8}]
    out = radarlab.run_experiment(tel, x, y, t, fov_deg=FOV, width=W, height=H, target_size_m=0.225,
                                  boxes=boxes, tag="synthetic", out_dir=str(tmp_path), only=["timeline"])
    assert os.path.exists(os.path.join(out, "box_timeline.png"))               # selected → rendered
    assert not os.path.exists(os.path.join(out, "experiment_summary.png"))     # not selected → untouched
    assert os.path.exists(os.path.join(out, "report_manifest.json"))           # ontology emitted


def test_manifest_disable_skips_plot(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    tel, x, y, t = _drone_events(target_az=45.0, y0=150, y1=190, per=400, n_rev=8)
    boxes = [{"az_window": [40, 50], "y_window": [145, 195], "bearing_deg": 45.0, "range_m": 5.0,
              "t_window": [0.0, 8.0], "n_passes": 8}]
    man = radarlab.default_manifest("synthetic", fov_deg=FOV, target_size_m=0.225, az_sign=1.0)
    man["plots"]["context"]["enabled"] = False
    man["plots"]["timeline"]["style"]["bearing_color"] = "darkgreen"           # trivial style edit
    radarlab.run_experiment(tel, x, y, t, fov_deg=FOV, width=W, height=H, target_size_m=0.225,
                            boxes=boxes, tag="synthetic", out_dir=str(tmp_path), manifest=man)
    assert not os.path.exists(os.path.join(tmp_path, "context_panorama_radar.png"))  # disabled
    assert os.path.exists(os.path.join(tmp_path, "experiment_summary.png"))          # still on


def test_run_experiment_converges(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    tel, x, y, t = _drone_events(target_az=45.0, y0=150, y1=190, per=400, n_rev=8)
    boxes = [{"az_window": [40, 50], "y_window": [145, 195], "bearing_deg": 45.0, "range_m": 5.0,
              "t_window": [0.0, 8.0], "n_passes": 8}]
    out = radarlab.run_experiment(tel, x, y, t, fov_deg=FOV, width=W, height=H, target_size_m=0.225,
                                  boxes=boxes, tag="synthetic", out_dir=str(tmp_path))
    assert os.path.exists(os.path.join(out, "experiment_summary.png"))
    assert os.path.exists(os.path.join(out, "experiment.json"))
    assert os.path.exists(os.path.join(out, "README.md"))
    import json
    s = json.load(open(os.path.join(out, "experiment.json")))
    assert s["n_instances"] == 1 and s["n_track_revs"] >= 5
    assert set(map(int, s["data_rate_reduction_pct"].keys())) >= {0, 1, 2}
    assert abs(s["bearing_deg_median"] - 45.0) < 2.0
