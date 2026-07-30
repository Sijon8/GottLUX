"""Tests for the v1.3 additions: tone-mapping, NUFFT, ISI, measurement, report, export."""
import numpy as np

from gottlux.core import frequency as fq, tonemap
from gottlux.io.recording import Recording


def _periodic_pixel(freq_hz, duration_s=1.0, n_per_burst=18, pixel=(10, 10), seed=0):
    rng = np.random.default_rng(seed)
    period = 1.0 / freq_hz
    ts = []
    for k in range(int(duration_s / period)):
        c = (k + 0.5) * period
        ts.append(c + (rng.random(n_per_burst) - 0.5) * 0.15 * period)
    t = np.sort(np.concatenate(ts)) * 1e6
    x = np.full(t.size, pixel[0], np.uint16); y = np.full(t.size, pixel[1], np.uint16)
    p = np.ones(t.size, np.uint8)
    return Recording.from_events(x, y, p, t.astype(np.int64), width=64, height=64)


# ----------------------------------------------------------------- tone-mapping
def test_tonemap_expressions_bounded_and_monotone():
    rng = np.random.default_rng(0)
    frame = rng.poisson(3.0, (40, 40)).astype(np.float32)
    frame[5, 5] = 5000                                  # a hot outlier
    for expr in tonemap.EXPRESSIONS:
        disp, vmax = tonemap.compress(frame, expr=expr)
        assert disp.shape == frame.shape
        assert disp.min() >= 0.0 and disp.max() <= 1.0 + 1e-6, expr
        assert np.isfinite(vmax) and vmax > 0
    # a compressive curve must lift a mid value above the linear mapping
    mid = np.array([[0.25 * 8.0]], np.float32)          # 0.25 of an 8-count white-point
    lin, _ = tonemap.compress(mid, expr="linear", vmax=8.0)
    lg, _ = tonemap.compress(mid, expr="log", vmax=8.0)
    assert lg[0, 0] > lin[0, 0]


def test_tonemap_static_scale_freezes_reference():
    a = np.full((8, 8), 10.0, np.float32)
    b = np.full((8, 8), 1000.0, np.float32)
    _, vmax = tonemap.compress(a, expr="linear")        # capture white-point from frame a
    disp_b, _ = tonemap.compress(b, expr="linear", vmax=vmax)
    assert disp_b.max() <= 1.0 + 1e-6                    # frozen scale clips, doesn't rescale


def test_compress_signed_symmetric():
    f = np.array([[-50.0, 0.0, 50.0]], np.float32)
    disp, vmax = tonemap.compress_signed(f, expr="sqrt")
    assert disp.min() >= -1 - 1e-6 and disp.max() <= 1 + 1e-6
    assert abs(disp[0, 0] + disp[0, 2]) < 1e-6          # symmetric about zero


# ----------------------------------------------------------------- NUFFT / whitening
def test_nufft_recovers_frequency():
    rec = _periodic_pixel(173.0, duration_s=1.0, n_per_burst=16)
    sp = fq.nufft_spectrum(rec.window().t, fmin=20, fmax=600, n_freq=800)
    assert sp.detected and abs(sp.peak_freq - 173.0) < 6.0
    assert sp.snr > 5.0


def test_whiten_emphasizes_peak_over_sloped_noise():
    freqs = np.linspace(1, 500, 400)
    power = 1.0 / freqs                                  # red (1/f) noise floor
    power[200] += 0.05                                   # a modest line mid-band
    raw_ratio = power[200] / np.median(power)
    wh = fq.whiten_power(power, "median")
    white_ratio = wh[200] / np.median(wh)
    assert white_ratio > raw_ratio                       # whitening makes the line stand out more


def test_region_spectrum_normalize_runs():
    rec = _periodic_pixel(200.0)
    for nrm in ("none", "median", "zscore"):
        sp = fq.region_spectrum(rec.window().t, fs=2000, fmin=20, fmax=600, normalize=nrm)
        assert sp.freqs.size and np.isfinite(sp.peak_freq)


# ----------------------------------------------------------------- ISI / measure
def test_isi_frequency_on_single_pixel():
    """A single pixel firing once per cycle has ISI ≈ period → recovers the rate."""
    rng = np.random.default_rng(1)
    f0 = 120.0
    t = (np.arange(0, 1.0, 1.0 / f0) + rng.normal(0, 0.0005, int(f0))) * 1e6
    f, strength = fq.isi_frequency(np.sort(t), fmin=20, fmax=600)
    # the modal-bin concentration is far above a uniform 1/n_bins (~0.017) baseline
    assert abs(f - f0) < 20.0 and strength > 0.15


def test_measure_between_frequency_and_speed():
    m = fq.measure_between((100, 100, 0.0), (140, 100, 0.005), cycles=1.0)
    assert abs(m["freq_hz"] - 200.0) < 1e-6
    assert abs(m["dr_px"] - 40.0) < 1e-6
    assert abs(m["speed_px_s"] - 8000.0) < 1e-6


# ----------------------------------------------------------------- report / bundle
def test_detection_report_and_bundle(tmp_path):
    from gottlux.config import Config
    from gottlux.detectors import get_detector
    from gottlux.run.report import build_detection_report, save_detection_report
    from gottlux.app.exporting import export_bundle
    from gottlux.synthetic import synthetic_scene

    rec, _ = synthetic_scene(duration_s=1.0, seed=2, noise_rate_hz=8000)
    res = get_detector("drone", snr_thresh=3.0).run(rec, Config(mode="staring"), 0.0, 1.0)
    md, data = build_detection_report(res, rec, cfg=Config(mode="staring"), window=(0.0, 1.0))
    assert "Detection report" in md and md.count("## ") >= 6
    assert {"parameters", "recording", "targets", "diagnostics"} <= set(data)
    files = save_detection_report(str(tmp_path / "r"), res, rec, window=(0.0, 1.0))
    assert any(p.endswith("_report.md") for p in files)

    fm = fq.flicker_map(rec, fmin=80, fmax=800, cell=8, t0=0.0, t1=1.0)
    written, manifest = export_bundle(str(tmp_path / "bundle"), rec, 0.0, 1.0,
                                      want=["event_cube", "event_rate", "flicker_cube", "config"],
                                      flicker_map=fm)
    assert set(manifest["produced"]) == {"event_cube", "event_rate", "flicker_cube", "config"}
    assert any(p.endswith("manifest.json") for p in written)
