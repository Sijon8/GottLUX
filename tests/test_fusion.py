"""Tests for the EBS+acoustic fusion I/O and study pipeline (gottlux.io.fusion / run.fusion_study).

Covers: WAV round-trips (16/24-bit), the audio RMS envelope, the cross-correlation offset
estimator (sign + magnitude), the full align→export of a synthetic pair (decodable .raw + .wav),
an uncompressed Prophesee-HDF5 → .raw conversion, and a tiny end-to-end fusion study.
"""
import os

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

import gottlux as eb  # noqa: E402
from gottlux.io import fusion  # noqa: E402
from gottlux.io.recording import Recording  # noqa: E402


# --------------------------------------------------------------------- WAV I/O
@pytest.mark.parametrize("subtype", ["int16", "int24"])
def test_wav_roundtrip_pcm(tmp_path, subtype):
    rng = np.random.default_rng(0)
    sr = 48000
    lim = {"int16": 32000, "int24": (1 << 23) - 5}[subtype]
    x = rng.integers(-lim, lim, 5000).astype(np.float64)
    p = str(tmp_path / "rt.wav")
    fusion.write_wav(p, x, sr, subtype=subtype)
    a = fusion.read_wav(p)
    assert a.sample_rate == sr
    assert a.n == x.size
    assert np.array_equal(a.samples, x)        # integer PCM round-trips exactly
    assert a.subtype == subtype


def test_wav_stereo_downmix(tmp_path):
    import wave
    sr = 16000
    n = 2000
    left = np.full(n, 100, np.int16)
    right = np.full(n, 300, np.int16)
    inter = np.empty(n * 2, np.int16)
    inter[0::2] = left
    inter[1::2] = right
    p = str(tmp_path / "stereo.wav")
    with wave.open(p, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(inter.tobytes())
    a = fusion.read_wav(p)
    assert a.n == n
    assert np.allclose(a.samples, 200.0)       # mean of the two channels


def test_rms_envelope_tracks_loudness():
    sr = 48000
    t = np.arange(2 * sr) / sr
    # loud in the second half only
    x = np.sin(2 * np.pi * 200 * t) * np.where(t > 1.0, 1.0, 0.05)
    a = fusion.AudioClip(x, sr)
    c, rms = a.rms_envelope(0.05)
    assert c.size == rms.size > 10
    assert rms[-1] > 5 * rms[0]                # later (loud) bins dwarf early (quiet) bins


# --------------------------------------------------------------------- offset estimator
def _bump(n, center_bin, width_bins, bin_s):
    i = np.arange(n)
    return np.exp(-0.5 * ((i - center_bin) / width_bins) ** 2)


def test_estimate_offset_sign_and_magnitude():
    """A shared event at EBS-time 5.0 s and audio-time 2.0 s ⇒ offset = +3.0 s (add to audio)."""
    bin_s = 0.01
    ebs = _bump(1000, 500, 20, bin_s)          # peak at 5.0 s
    aud = _bump(1200, 200, 20, bin_s)          # peak at 2.0 s
    res = fusion.estimate_offset(ebs, aud, bin_s=bin_s, max_lag_s=10.0)
    assert abs(res.offset_s - 3.0) < 0.05
    assert res.peak_corr > 0.5


def test_estimate_offset_negative_lag():
    bin_s = 0.01
    ebs = _bump(1000, 200, 20, bin_s)          # peak at 2.0 s
    aud = _bump(1200, 700, 20, bin_s)          # peak at 7.0 s ⇒ offset = -5.0 s
    res = fusion.estimate_offset(ebs, aud, bin_s=bin_s, max_lag_s=10.0)
    assert abs(res.offset_s + 5.0) < 0.05


# --------------------------------------------------------------------- align + export a pair
def _synth_recording(dur_s=10.0, peak_s=5.0, w=64, h=64, seed=1):
    """A recording whose event RATE bumps around *peak_s* (dense cluster) over a sparse background."""
    rng = np.random.default_rng(seed)
    n_bg = 20000
    t_bg = rng.uniform(0, dur_s, n_bg)
    t_pk = np.clip(rng.normal(peak_s, 0.25, 30000), 0, dur_s)   # the activity swell
    t = np.sort(np.concatenate([t_bg, t_pk]))
    tus = (t * 1e6).astype(np.int64)
    x = rng.integers(0, w, t.size).astype(np.uint16)
    y = rng.integers(0, h, t.size).astype(np.uint16)
    p = rng.integers(0, 2, t.size).astype(np.uint8)
    return Recording.from_events(x, y, p, tus, width=w, height=h, name="synth")


def _synth_audio(dur_s=12.0, peak_s=2.0, sr=16000, seed=2, f0=520.0):
    """A rotor-like harmonic tone (f0 + 2f0 + 3f0) with a loudness swell at *peak_s*."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur_s * sr)) / sr
    env = 0.05 + np.exp(-0.5 * ((t - peak_s) / 0.3) ** 2)       # loudness swell at peak_s
    tone = (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
            + 0.3 * np.sin(2 * np.pi * 3 * f0 * t))
    x = (tone + 0.1 * rng.standard_normal(t.size)) * env
    return fusion.AudioClip(x, sr, source_path="synth.wav", subtype="int16")


def test_plan_alignment_and_export(tmp_path):
    rec = _synth_recording(dur_s=10.0, peak_s=5.0)
    aud = _synth_audio(dur_s=12.0, peak_s=2.0)
    res = fusion.plan_alignment(rec, aud, bin_s=0.01, max_lag_s=10.0)
    assert abs(res.offset_s - 3.0) < 0.1       # 5.0 (EBS) - 2.0 (audio)
    # overlap: audio on EBS clock spans [3, 15]; EBS spans [0, 10] ⇒ overlap [3, 10] = 7 s
    assert abs(res.overlap_s - 7.0) < 0.2

    out = str(tmp_path / "fuse")
    man = fusion.export_aligned(rec, aud, res, out, base_name="pair")
    raw_path = os.path.join(out, man["ebs_raw"])
    wav_path = os.path.join(out, man["audio_wav"])
    assert os.path.exists(raw_path) and os.path.exists(wav_path)
    arec = eb.load(raw_path, progress=lambda f: None)
    awav = fusion.read_wav(wav_path)
    assert abs(arec.duration_s - awav.duration_s) < 0.05      # same shared-clock duration
    assert abs(arec.duration_s - res.overlap_s) < 0.2
    assert man["aligned_duration_s"] > 0


def test_export_no_overlap_raises(tmp_path):
    rec = _synth_recording(dur_s=5.0)
    aud = _synth_audio(dur_s=5.0)
    res = fusion.plan_alignment(rec, aud, offset_s=99.0)       # pushed far past any overlap
    with pytest.raises(fusion.FusionError):
        fusion.export_aligned(rec, aud, res, str(tmp_path / "x"), base_name="x")


# --------------------------------------------------------------------- HDF5 → raw
def test_hdf5_to_raw_roundtrip(tmp_path):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(3)
    n = 4000
    dt = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")])
    ev = np.zeros(n, dt)
    ev["x"] = rng.integers(0, 320, n)
    ev["y"] = rng.integers(0, 320, n)
    ev["p"] = rng.integers(0, 2, n)
    ev["t"] = np.sort(rng.integers(0, 2_000_000, n))
    h5 = str(tmp_path / "ev.hdf5")
    with h5py.File(h5, "w") as f:
        g = f.create_group("CD")
        g.create_dataset("events", data=ev)
        f.attrs["geometry"] = "320x320"
        f.attrs["format"] = "EVT21;height=320;width=320"
        f.attrs["time_shift"] = 4018013

    d = fusion.read_hdf5_events(h5)
    assert d["width"] == 320 and d["height"] == 320
    assert d["x"].size == n and d["time_shift"] == 4018013

    raw = str(tmp_path / "ev.raw")
    n_w = fusion.hdf5_to_raw(h5, raw)
    assert n_w == n
    rec = eb.load(raw, progress=lambda f: None)
    assert rec.n == n
    assert rec.width == 320 and rec.height == 320


def test_read_hdf5_missing_group_raises(tmp_path):
    h5py = pytest.importorskip("h5py")
    h5 = str(tmp_path / "bad.hdf5")
    with h5py.File(h5, "w") as f:
        f.create_dataset("not_events", data=np.zeros(4))
    with pytest.raises(fusion.FusionError):
        fusion.read_hdf5_events(h5)


# --------------------------------------------------------------------- detection engine (fast, pure)
def test_harmonic_f0_finds_fundamental():
    from gottlux.run import fusion_detect as fd
    f = np.arange(0, 1600, 2.0)
    p = 1.0 + np.zeros_like(f)
    for h, amp in [(1, 40), (2, 20), (3, 10)]:          # comb at 500, 1000, 1500
        p += amp * np.exp(-0.5 * ((f - h * 500.0) / 6.0) ** 2)
    f0, snr, harm = fd.harmonic_f0(f, p, f0_range=(380, 780))
    assert abs(f0 - 500.0) < 8.0 and snr > 5 and harm >= 2


def test_acoustic_track_locks_rotor_tone():
    from gottlux.run import fusion_detect as fd
    aud = _synth_audio(dur_s=6.0, peak_s=3.0, f0=560.0)
    tr = fd.acoustic_track(aud)
    assert abs(tr.median_f0() - 560.0) < 25.0
    assert tr.detection_confidence() > 0.8


def test_convergence_and_fuse_synthetic():
    from gottlux.run import fusion_detect as fd
    t = np.linspace(0, 5, 60)
    ac = fd.DomainTrack(t, np.full(t.size, 520.0), np.full(t.size, 50.0),
                        np.full(t.size, 3.0), np.full(t.size, 0.95), "acoustic")
    eb_t = fd.DomainTrack(t, np.full(t.size, 530.0), np.full(t.size, 10.0),
                          np.full(t.size, 0.6), np.full(t.size, 0.85), "ebs")
    conv = fd.convergence(ac, eb_t)
    assert conv["temporal_overlap"] > 0.9          # both present everywhere
    assert conv["band_agreement"] > 0.8            # 520 vs 530 → tight
    fz = fd.fuse(ac, eb_t, conv)
    assert fz["p_dual_bayes"] > 0.9 and fz["p_fused"] > 0.6


def test_prune_boxes():
    from gottlux.run import fusion_detect as fd
    boxes = {"t": np.arange(10.0), "bbox": np.zeros((10, 4)), "cx": np.arange(10.0),
             "cy": np.arange(10.0), "det": object()}
    out = fd.prune_boxes(boxes, drop_time_ranges=[(3, 5)])
    assert out["t"].tolist() == [0, 1, 2, 6, 7, 8, 9]
    out2 = fd.prune_boxes(boxes, keep_window=(2, 4))
    assert out2["t"].tolist() == [2, 3, 4]


# --------------------------------------------------------------------- end-to-end study
def test_run_fusion_study_end_to_end(tmp_path):
    from gottlux.run.fusion_study import FusionConfig, run_fusion_study
    rec = _synth_recording(dur_s=8.0, peak_s=4.0)
    aud = _synth_audio(dur_s=9.0, peak_s=2.0, f0=520.0)
    raw = str(tmp_path / "scene.raw")
    from gottlux.io import writer
    writer.write_raw(raw, np.asarray(rec.x), np.asarray(rec.y), np.asarray(rec.p),
                     np.asarray(rec.t), width=rec.width, height=rec.height)
    wav = str(tmp_path / "scene.wav")
    fusion.write_wav(wav, aud.samples, aud.sample_rate, subtype="int16")

    out = str(tmp_path / "study")
    s = run_fusion_study(raw, wav, out, cfg=FusionConfig(), label="scene")
    assert os.path.exists(os.path.join(out, "fusion_summary.json"))
    assert os.path.exists(os.path.join(out, "fusion_report.md"))
    assert os.path.exists(os.path.join(out, "alignment_overlay.png"))
    assert os.path.exists(os.path.join(out, "acoustic_ridge.png"))
    assert os.path.exists(os.path.join(out, "clips", "scene.raw"))
    assert os.path.exists(os.path.join(out, "clips", "scene.wav"))
    assert s["alignment"]["overlap_s"] > 0
    # the synthetic audio fundamental is 520 Hz — the harmonic acoustic tracker should land near it
    assert abs(s["acoustic"]["f0_hz"] - 520.0) < 30.0
    assert s["fusion"]["C_acoustic"] > 0.8
