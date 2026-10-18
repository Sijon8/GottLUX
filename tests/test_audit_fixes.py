"""Regression tests for the 2026-10-18 audit (docs/ISSUES.md): run-folder location, preset
detector bands, plot fixes, and the minimum window size."""
import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gottlux.config import Config  # noqa: E402
from gottlux.io.recording import Recording  # noqa: E402

# ------------------------------------------------------------------ pipeline (GL-001/002)


def test_run_folder_honours_output_root(flutter_rec, tmp_path):
    from gottlux.run.pipeline import run_recording
    rec, _ = flutter_rec
    cfg = Config(mode="staring", analyses=("overview",), output_root=str(tmp_path),
                 open_when_done=False)
    path = run_recording(rec, cfg)
    assert os.path.commonpath([path, str(tmp_path)]) == str(tmp_path)


@pytest.mark.parametrize("name, band", [("hummingbird", (15.0, 120.0)),
                                        ("insect", (30.0, 250.0)),
                                        ("bird", (3.0, 30.0))])
def test_preset_detector_keeps_its_band(name, band):
    from gottlux.run.pipeline import adopt_signature_band
    cfg = Config(detector=name)
    adopt_signature_band(cfg)
    assert (cfg.freq_lo, cfg.freq_hi) == band


def test_explicit_band_wins_over_preset():
    from gottlux.run.pipeline import adopt_signature_band
    cfg = Config(detector="hummingbird")
    cfg.band_explicit = True                     # what the CLI sets for --freq_lo/--freq_hi
    adopt_signature_band(cfg)
    assert (cfg.freq_lo, cfg.freq_hi) == (Config.freq_lo, Config.freq_hi)
    cfg = Config(detector="hummingbird", freq_lo=40.0, freq_hi=90.0)
    adopt_signature_band(cfg)
    assert (cfg.freq_lo, cfg.freq_hi) == (40.0, 90.0)


def test_cli_marks_explicit_band():
    from gottlux.cli import _config_from_args, build_parser
    a = build_parser().parse_args(["x.raw", "--detector", "hummingbird"])
    assert not _config_from_args(a).band_explicit
    a = build_parser().parse_args(["x.raw", "--detector", "hummingbird", "--freq_lo", "80"])
    assert _config_from_args(a).band_explicit


# ------------------------------------------------------------------ plots (GL-003…006)


def test_event_rate_has_no_end_dip():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 2.987e6, 200_000)).astype(np.int64)    # flat 66.9 kev/s
    rec = Recording.from_events(rng.integers(0, 32, t.size), rng.integers(0, 32, t.size),
                                rng.integers(0, 2, t.size), t, width=32, height=32)
    c, r = rec.event_rate(0.01)
    assert abs(r[-1] / np.median(r) - 1) < 0.25          # the last bin was ~0.7 before
    assert c[-1] <= rec.t_stop_s


def test_event_rate_follows_recording_clock():
    t = np.arange(0, 1_000_000, 100, dtype=np.int64) + 5_000_000      # starts at 5 s
    rec = Recording({"x": np.zeros(t.size, np.uint16), "y": np.zeros(t.size, np.uint16),
                     "p": np.ones(t.size, np.uint8), "t": t, "width": 4, "height": 4})
    c, r = rec.event_rate(0.1)
    assert c[0] > 5.0 and abs(np.median(r) - 10_000) < 1


def test_flicker_rgba_hides_noise_cells():
    from gottlux.viz.theme import flicker_rgba
    rng = np.random.default_rng(0)
    snr = rng.uniform(8, 12, (40, 40))                   # noise-only cells score ~8-12
    snr[18:22, 18:22] = 200.0                            # one real flutter patch
    freq = rng.uniform(80, 800, snr.shape)
    alpha = flicker_rgba(freq, snr, 80, 800)[..., 3]
    assert np.median(alpha) < 0.05
    assert alpha[20, 20] > 0.9


def test_wrap_title_fits_width():
    from gottlux.viz.theme import wrap_title
    lines = wrap_title("Flicker map — Humming_Bird_Fight_merged_2_with_a_long_name", 2.6)
    assert "\n" in lines
    assert all(len(line) <= 2.6 * 72 / (0.6 * 9) for line in lines.split("\n"))


def test_panorama_bins_one_row_each():
    """Per-row elevation bins: no empty alternating rows (the striping artefact)."""
    from gottlux.io.telemetry import Telemetry
    from gottlux.viz.panorama import panorama_figure
    rng = np.random.default_rng(2)
    n = 400_000
    t = np.sort(rng.uniform(0, 4e6, n)).astype(np.int64)
    rec = Recording.from_events(rng.integers(0, 64, n), rng.integers(0, 64, n),
                                rng.integers(0, 2, n), t, width=64, height=64)
    rec.telemetry = Telemetry.from_spin(rec.duration_s, 1.0)
    fig = panorama_figure(rec, fov_deg=40)
    img = fig.axes[0].images[0].get_array()
    rows = np.asarray(img).mean(axis=1)
    assert rows.min() > 0.3 * np.median(rows)            # stripes left every other row ~empty
    import matplotlib.pyplot as plt
    plt.close(fig)


# ------------------------------------------------------------------ GUI layout (GL-010…014)

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    from gottlux.app import style
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    style.apply_app_style(a)
    return a


def test_main_window_fits_a_laptop_screen(app):
    from gottlux.app.main import MainWindow
    win = MainWindow()
    hint = win.minimumSizeHint()
    assert hint.height() <= 720, hint            # was 963-1038 px: taller than the screen
    assert hint.width() <= 1280, hint            # was 1549 px (an unwrapped hint label)


def test_density_strip_never_sets_a_minimum_width(app):
    from gottlux.app.ebsviewer import DensityStrip
    strip = DensityStrip()
    strip.resize(1500, strip.height())
    strip.build(np.arange(0, 3_000_000, 10, dtype=np.int64), 300_000, 3.0)
    assert strip.minimumSizeHint().width() < 200
    strip.resize(400, strip.height())            # repaint on resize must not recurse
    app.processEvents()


def test_label_box_at_360_does_not_wrap(app):
    from gottlux.app.ebsviewer import EBSViewer
    v = EBSViewer()
    v.boxes = [{"kind": "box", "t": 0.0, "t0": 0.0, "t1": 0.0, "label": "drone",
                "az0": 350.0, "az1": 360.0, "y0": 10, "y1": 20}]
    rgb = np.zeros((40, 720, 3), np.uint8)
    v._draw_label_boxes(rgb, 0.0)
    assert not rgb[:, 0].any()                   # no stray edge at 0°
    assert rgb[10, 700:].any() and rgb[15, 719].any()


def test_unreadable_sidecar_is_kept(app, tmp_path):
    from gottlux.app.ebsviewer import EBSViewer
    raw = tmp_path / "clip.raw"
    side = tmp_path / "clip_annotations.json"
    side.write_text("{ not json", encoding="utf-8")
    v = EBSViewer()
    v._load_annotations(str(raw))
    assert list(tmp_path.glob("clip_annotations.json.unreadable-*"))
