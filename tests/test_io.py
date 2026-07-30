"""I/O tests: the EVT2.1 encode↔decode roundtrip and the Recording data model."""
import numpy as np
import pytest

import gottlux as eb
from gottlux.io import decode, writer
from gottlux.io.recording import Recording


def test_write_decode_roundtrip(tmp_path):
    """Events encoded to EVT2.1 and decoded back must match bit-for-bit (after sort)."""
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.integers(0, 320, n).astype(np.uint16)
    y = rng.integers(0, 240, n).astype(np.uint16)
    p = rng.integers(0, 2, n).astype(np.uint8)
    t = np.sort(rng.integers(0, 2_000_000, n)).astype(np.int64)
    path = tmp_path / "rt.raw"
    writer.write_raw(str(path), x, y, p, t, width=320, height=240)

    d = decode.decode(str(path))
    assert d["width"] == 320 and d["height"] == 240
    assert d["fmt"] == "evt21"
    # decode zero-bases time; compare against zero-based input
    order = np.argsort(t, kind="stable")
    assert np.array_equal(d["x"], x[order])
    assert np.array_equal(d["y"], y[order])
    assert np.array_equal(d["p"], p[order])
    assert np.array_equal(d["t"], (t - t[0])[order])


def test_streaming_write_is_block_invariant(tmp_path):
    """The streamed encoder must produce a byte-identical file no matter the block size — so the
    chunk-boundary TIME_HIGH state is carried correctly (the heart of the bounded-RAM rewrite)."""
    rng = np.random.default_rng(4)
    n = 60_000
    x = rng.integers(0, 320, n).astype(np.uint16); y = rng.integers(0, 240, n).astype(np.uint16)
    p = rng.integers(0, 2, n).astype(np.uint8); t = np.sort(rng.integers(0, 4_000_000, n)).astype(np.int64)
    big = tmp_path / "big.raw"; small = tmp_path / "small.raw"
    writer.write_raw(str(big), x, y, p, t, width=320, height=240, block=10_000_000)
    writer.write_raw(str(small), x, y, p, t, width=320, height=240, block=137)   # many tiny blocks
    assert big.read_bytes() == small.read_bytes()
    d = decode.decode(str(small))                       # and it still round-trips
    order = np.argsort(t, kind="stable")
    assert np.array_equal(d["t"], (t - t[0])[order])


def test_cut_clip_streams_window_and_roi(tmp_path):
    """A streamed cut writes exactly the windowed (and ROI'd) events, re-zeroed in time, regardless
    of block size — the RAM-safe replacement for materializing the whole crop."""
    rng = np.random.default_rng(5)
    n = 50_000
    x = rng.integers(0, 320, n).astype(np.uint16); y = rng.integers(0, 240, n).astype(np.uint16)
    p = rng.integers(0, 2, n).astype(np.uint8); t = np.sort(rng.integers(0, 5_000_000, n)).astype(np.int64)
    rec = Recording.from_events(x, y, p, t, width=320, height=240)
    out = tmp_path / "cut.raw"
    n_cut = writer.cut_clip(rec, str(out), t0=1.0, t1=3.0, block=911)   # forces multi-block streaming
    win = rec.window(1.0, 3.0)
    assert n_cut == win.n
    d = decode.decode(str(out))
    assert d["t"][0] == 0                                # re-zeroed to the clip start
    assert int(d["t"][-1]) == int(win.t[-1] - win.t[0])
    assert np.array_equal(np.sort(d["x"]), np.sort(np.asarray(win.x)))
    # ROI cut keeps only events inside the box
    roi = (50, 40, 150, 120)
    n_roi = writer.cut_clip(rec, str(tmp_path / "roi.raw"), t0=1.0, t1=3.0, roi=roi, block=911)
    wx, wy = np.asarray(win.x), np.asarray(win.y)
    expect = int(((wx >= 50) & (wx < 150) & (wy >= 40) & (wy < 120)).sum())
    assert n_roi == expect


def test_stitch_streams_blocks(tmp_path):
    """Stitching is streamed too: the merged file's event count and placed spans are independent of
    the block size, and the result is a valid, loadable .raw."""
    rng = np.random.default_rng(6)

    def _rec(seed, dur_us):
        m = 20_000
        r = np.random.default_rng(seed)
        return Recording.from_events(r.integers(0, 128, m), r.integers(0, 96, m),
                                     r.integers(0, 2, m), np.sort(r.integers(0, dur_us, m)),
                                     width=128, height=96)
    a, b = _rec(1, 1_000_000), _rec(2, 800_000)
    specs = [(a, 0.1, 0.6), (b, 0.0, 0.5)]
    out = tmp_path / "stitch.raw"
    res = writer.stitch_clips(str(out), specs, gap_s=0.05, block=503)
    assert len(res["segments"]) == 2 and res["n_events"] > 0
    expect = a.window(0.1, 0.6).n + b.window(0.0, 0.5).n
    assert res["n_events"] == expect
    assert eb.load(str(out)).n == res["n_events"]      # round-trips through the decoder
    # second segment is placed after the first + the gap
    assert res["segments"][1]["placed_t0_s"] >= res["segments"][0]["placed_t1_s"] + 0.049


def _grid_raw(tmp_path, name="src.raw", n=1000, step_us=1000, seed=0):
    """A synthetic EVT2.1 .raw with events on a clean time grid (so 64-µs cut boundaries don't
    straddle an event) — returns ``(path, x, y, p, t)``."""
    rng = np.random.default_rng(seed)
    t = (np.arange(n) * step_us).astype(np.int64)
    x = rng.integers(0, 320, n).astype(np.uint16); y = rng.integers(0, 240, n).astype(np.uint16)
    p = rng.integers(0, 2, n).astype(np.uint8)
    path = str(tmp_path / name)
    writer.write_raw(path, x, y, p, t, width=320, height=240)
    return path, x, y, p, t


def _multiset(d):
    o = np.lexsort((d["p"], d["y"], d["x"], d["t"]))
    return np.stack([d["t"][o], d["x"][o], d["y"][o], d["p"][o]])


def test_rawcut_matches_decode_based(tmp_path):
    """The decode-free byte cut yields the same events as the decode-based cut over the same window
    (clean-grid times so the 64-µs TIME_HIGH boundaries don't add/drop edge events)."""
    from gottlux.io import rawcut
    src, x, y, p, t = _grid_raw(tmp_path)
    # scan reports the duration/geometry with no decode
    ix = rawcut.scan(src)
    assert ix["width"] == 320 and ix["height"] == 240
    assert ix["duration_s"] == pytest.approx((t[-1] - t[0]) / 1e6, abs=1e-3)
    # direct cut on the bytes
    direct = tmp_path / "direct.raw"
    res = rawcut.cut_evt21(src, str(direct), t0=0.2, t1=0.5)
    # decode-based cut of the same window for reference
    rec = Recording.from_events(x, y, p, t, width=320, height=240)
    ref = tmp_path / "ref.raw"
    n_ref = writer.cut_clip(rec, str(ref), t0=0.2, t1=0.5)
    assert res["n_events"] == n_ref
    da, db = decode.decode(str(direct)), decode.decode(str(ref))
    assert np.array_equal(_multiset(da), _multiset(db))     # identical event multiset
    assert int(da["t"][0]) == 0                              # re-zeroed to the clip start


def test_cut_file_dispatch_direct_and_roi_fallback(tmp_path):
    """cut_file uses the no-decode byte cut for an EVT2.1 file with no ROI, and falls back to the
    decode-based cut when an ROI crop is requested — both write a valid, loadable .raw."""
    src, x, y, p, t = _grid_raw(tmp_path, n=2000, step_us=500, seed=1)
    rec = Recording.from_events(x, y, p, t, width=320, height=240)
    out = tmp_path / "direct.raw"
    n = writer.cut_file(src, str(out), t0=0.1, t1=0.4)       # → direct byte path
    assert n == rec.window(0.1, 0.4).n
    assert eb.load(str(out)).n == n
    # ROI → decode-based fallback, still correct
    roi = (50, 40, 150, 120)
    n_roi = writer.cut_file(src, str(tmp_path / "roi.raw"), t0=0.1, t1=0.4, roi=roi)
    w = rec.window(0.1, 0.4)
    wx, wy = np.asarray(w.x), np.asarray(w.y)
    assert n_roi == int(((wx >= 50) & (wx < 150) & (wy >= 40) & (wy < 120)).sum())


def test_recording_from_events_and_window():
    rng = np.random.default_rng(1)
    n = 10_000
    x = rng.integers(0, 64, n); y = rng.integers(0, 64, n)
    p = rng.integers(0, 2, n); t = np.sort(rng.integers(0, 1_000_000, n))
    rec = Recording.from_events(x, y, p, t)
    assert rec.n == n
    assert 0.9 < rec.duration_s < 1.0
    assert rec.n_on + rec.n_off == n
    # window slicing is consistent with a manual mask
    win = rec.window(0.2, 0.5)
    ts = rec.t.astype(float) / 1e6
    expect = int(((ts >= 0.2) & (ts < 0.5)).sum())
    assert win.n == expect
    on, off = win.polarity_split()
    assert on.sum() + off.sum() == win.n


def test_accumulate_via_recording():
    rng = np.random.default_rng(2)
    n = 20_000
    rec = Recording.from_events(rng.integers(0, 100, n), rng.integers(0, 100, n),
                                rng.integers(0, 2, n), np.sort(rng.integers(0, 1_000_000, n)),
                                width=100, height=100)
    frame = rec.accumulate(0.0, 0.5, mode="count")
    assert frame.shape == (100, 100)
    win = rec.window(0.0, 0.5)
    assert frame.sum() == win.n        # count frame totals to the number of events
