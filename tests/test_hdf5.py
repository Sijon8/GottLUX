"""
Tests for full HDF5 support (:mod:`gottlux.io.hdf5`) — write, read, open everywhere.

The promise under test: HDF5 is a first-class recording container next to ``.raw``.
Covered here:

* the streamed ``.raw`` → ``.h5`` conversion (Metavision-compatible compound ``CD/events``
  layout, gzip-chunked, block-appended) round-trips **byte-identically** through
  ``gottlux.load``, and provenance (geometry attrs + the ``gottlux`` attrs group) is stored;
* a windowed/ROI export honours its bounds with ``cut_clip`` semantics (re-zeroed times);
* the plain parallel ``x/y/p/t`` layouts (root-level and inside ``events/``) read too;
* an ``.h5`` builds the same decode-once bin cache a ``.raw`` does (``fmt='hdf5'``, cache
  beside the file, a distinct stem from a sibling ``.raw``) and re-opens instantly;
* the CLI ``--to-hdf5`` action via ``main()`` (default OUT, explicit OUT + window flags),
  and an ``.h5`` alone in a capture folder is discovered by the loader;
* the ``fusion.read_hdf5_events`` wrapper keeps its original contract;
* an ECF-compressed file (Prophesee filter 36559, built for real with the low-level API)
  raises the clear :class:`FusionError` — including the opt-in ``gottlux[hdf5]`` install
  hint when ``hdf5plugin`` is absent;
* the bundled example clip converts and opens identically (artifacts cleaned up).
"""
import gc
import json
import os
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import gottlux as eb  # noqa: E402
from gottlux.io import cache, writer  # noqa: E402
from gottlux.io import hdf5 as h5io  # noqa: E402
from gottlux.io.fusion import FusionError  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "examples", "data")
SHORT = os.path.join(DATA, "Humming_Bird_Fight_merged_shortest.raw")
needs_short = pytest.mark.skipif(not os.path.exists(SHORT),
                                 reason="bundled example clip missing: "
                                        + os.path.basename(SHORT))


def _synth_raw(tmp_path, n=30_000, dur_s=2.0, w=320, h=240, seed=0, name="src.raw"):
    """A small EVT2.1 ``.raw`` of uniform random events (via the round-trip-tested writer)."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, w, n).astype(np.uint16)
    y = rng.integers(0, h, n).astype(np.uint16)
    p = rng.integers(0, 2, n).astype(np.uint8)
    t = np.sort(rng.integers(0, int(dur_s * 1e6), n)).astype(np.int64)
    path = str(tmp_path / name)
    writer.write_raw(path, x, y, p, t, width=w, height=h)
    return path


def _assert_same_events(a, b):
    """Two Recordings hold byte-identical event arrays and geometry."""
    assert a.n == b.n
    assert (a.width, a.height) == (b.width, b.height)
    assert np.array_equal(np.asarray(a.x), np.asarray(b.x))
    assert np.array_equal(np.asarray(a.y), np.asarray(b.y))
    assert np.array_equal(np.asarray(a.p), np.asarray(b.p))
    assert np.array_equal(np.asarray(a.t), np.asarray(b.t))


# --------------------------------------------------------------------- raw → h5 round-trip
def test_raw_to_h5_roundtrip_byte_identical(tmp_path):
    """Convert a .raw with multi-block streaming; loading both through gottlux.load must
    yield byte-identical arrays, and the file must be Metavision-compatible with the
    gottlux provenance attrs."""
    raw = _synth_raw(tmp_path)
    h5 = str(tmp_path / "src.h5")
    fractions = []
    n = h5io.write_hdf5(raw, h5, block=7_001, progress=fractions.append)
    assert fractions and fractions[-1] == 1.0

    ra = eb.load(raw, progress=lambda f: None)
    rb = eb.load(h5, progress=lambda f: None)
    assert n == ra.n
    _assert_same_events(ra, rb)
    assert rb.fmt == "hdf5"

    with h5py.File(h5, "r") as f:
        ds = f["CD"]["events"]
        assert set(ds.dtype.names) >= {"x", "y", "p", "t"}
        assert ds.compression == "gzip" and ds.chunks       # chunked + compressed
        assert str(f.attrs["geometry"]) == "320x240"
        assert int(f.attrs["width"]) == 320 and int(f.attrs["height"]) == 240
        g = f["gottlux"]
        assert str(g.attrs["source"]) == "src.raw"
        assert str(g.attrs["version"]) == eb.__version__
        assert int(g.attrs["t0_us"]) == ra.t0_us            # full export: origin = 1st event


def test_windowed_roi_export_honours_bounds(tmp_path):
    """t0/t1 + ROI export exactly the cut_clip event set, re-zeroed to the window start."""
    raw = _synth_raw(tmp_path, seed=1)
    rec = eb.load(raw, progress=lambda f: None)
    h5 = str(tmp_path / "clip.h5")
    roi = (40, 30, 200, 150)
    n = h5io.write_hdf5(rec, h5, t0=0.5, t1=1.5, roi=roi, block=911)
    win = rec.window(0.5, 1.5, roi=roi)
    assert n == win.n > 0

    clip = eb.load(h5, progress=lambda f: None)
    assert clip.n == win.n
    assert np.array_equal(np.asarray(clip.x), np.asarray(win.x))
    assert np.array_equal(np.asarray(clip.y), np.asarray(win.y))
    assert np.array_equal(np.asarray(clip.p), np.asarray(win.p))
    wt = np.asarray(win.t)
    assert np.array_equal(np.asarray(clip.t), wt - wt[0])   # re-zeroed (cache re-bases)
    # geometry is the sensor's, not the ROI's; provenance records the absolute origin
    assert (clip.width, clip.height) == (rec.width, rec.height)
    with h5py.File(h5, "r") as f:
        origin = int(rec.t[rec.index_at(0.5)])              # first event of the window
        assert int(f["gottlux"].attrs["t0_us"]) == rec.t0_us + origin


# --------------------------------------------------------------------- plain x/y/p/t layouts
@pytest.mark.parametrize("where", ["root", "events_group"])
def test_plain_xypt_layout_reads(tmp_path, where):
    rng = np.random.default_rng(3)
    n = 5_000
    x = rng.integers(0, 320, n).astype(np.uint16)
    y = rng.integers(0, 320, n).astype(np.uint16)
    p = rng.integers(0, 2, n).astype(np.int16)
    t = np.sort(rng.integers(10_000, 1_000_000, n)).astype(np.int64)   # non-zero start
    h5 = str(tmp_path / "plain.h5")
    with h5py.File(h5, "w") as f:
        tgt = f if where == "root" else f.create_group("events")
        for k, v in (("x", x), ("y", y), ("p", p), ("t", t)):
            tgt.create_dataset(k, data=v)
        f.attrs["geometry"] = "320x320"

    d = h5io.read_events(h5)
    assert d["n"] == n and d["fmt"] == "hdf5"
    assert d["width"] == 320 and d["height"] == 320
    assert np.array_equal(d["x"], x) and np.array_equal(d["y"], y)
    assert np.array_equal(d["p"], p.astype(np.uint8))
    assert np.array_equal(d["t"], t - t[0])                 # zero-based
    assert d["t0_us"] == int(t[0])

    # and the loader opens it through the same decode-once cache
    rec = eb.load(h5, progress=lambda f_: None)
    assert rec.n == n and rec.fmt == "hdf5"
    assert np.array_equal(np.asarray(rec.t), t - t[0])


# --------------------------------------------------------------------- decode-once cache
def test_h5_cache_built_once_then_instant(tmp_path, monkeypatch):
    """First open streams the .h5 into the bin cache (fmt='hdf5', beside the file, its own
    stem next to the sibling .raw's); a re-open must be a pure memmap hit."""
    raw = _synth_raw(tmp_path, seed=2)
    h5 = str(tmp_path / "src.h5")
    h5io.write_hdf5(raw, h5)

    assert not cache.has_valid_cache(h5)
    r1 = eb.load(h5, progress=lambda f: None)
    assert cache.has_valid_cache(h5)

    cache_dir, stem = cache.cache_location(h5)
    assert os.path.basename(cache_dir) == "_gottlux_cache"
    assert os.path.dirname(cache_dir) == str(tmp_path)      # beside the file, as usual
    assert os.path.basename(stem) == "src.h5"               # full-name stem: no clash with src.raw
    with open(stem + ".meta.json") as f:
        meta = json.load(f)
    assert meta["fmt"] == "hdf5" and meta["layout"] == "bin"

    # the sibling .raw keeps its own bare-stem cache — the two never fight
    eb.load(raw, progress=lambda f: None)
    assert cache.has_valid_cache(raw) and cache.has_valid_cache(h5)

    # a second open must NOT re-read the HDF5 — pure cache hit
    def boom(*a, **k):
        raise AssertionError("re-decoded the .h5 despite a fresh cache")
    monkeypatch.setattr(cache, "_hdf5_to_bin", boom)
    r2 = eb.load(h5, progress=lambda f: None)
    _assert_same_events(r1, r2)


def test_h5_discovered_in_capture_folder(tmp_path):
    """A capture folder holding only an .h5 loads through the normal folder route."""
    raw = _synth_raw(tmp_path, seed=5, name="cam0_rec.raw")
    folder = tmp_path / "capture"
    folder.mkdir()
    h5 = str(folder / "cam0_rec.h5")
    h5io.write_hdf5(raw, h5)
    rec = eb.load(str(folder), progress=lambda f: None)
    assert rec.source_path.endswith("cam0_rec.h5")
    assert rec.n == eb.load(raw, progress=lambda f: None).n


# --------------------------------------------------------------------- CLI
def test_cli_to_hdf5_via_main(tmp_path, monkeypatch, capsys):
    from gottlux.cli import main
    raw = _synth_raw(tmp_path, seed=4, name="clip.raw")

    # default OUT = input stem + '.h5' (argv route, as the console script sees it)
    monkeypatch.setattr(sys, "argv", ["gottlux", raw, "--to-hdf5"])
    assert main() == 0
    out_h5 = str(tmp_path / "clip.h5")
    assert os.path.exists(out_h5)
    msg = capsys.readouterr().out
    rec = eb.load(out_h5, progress=lambda f: None)
    assert "clip.h5" in msg and f"{rec.n:,}" in msg         # prints the path + event count

    # explicit OUT honours the existing window flags
    out2 = str(tmp_path / "sub.h5")
    assert main(["gottlux", raw, "--to-hdf5", out2,
                 "--t_start", "0.2", "--t_stop", "0.8"]) == 0
    full = eb.load(raw, progress=lambda f: None)
    assert eb.load(out2, progress=lambda f: None).n == full.window(0.2, 0.8).n
    assert "sub.h5" in capsys.readouterr().out


# --------------------------------------------------------------------- fusion wrapper compat
def test_fusion_wrapper_delegates(tmp_path):
    """fusion.read_hdf5_events keeps its documented dict shape over the new reader."""
    from gottlux.io import fusion
    raw = _synth_raw(tmp_path, seed=6)
    h5 = str(tmp_path / "src.h5")
    h5io.write_hdf5(raw, h5)
    d = fusion.read_hdf5_events(h5)
    assert set(d) == {"x", "y", "p", "t_us", "width", "height", "meta", "time_shift"}
    assert (d["width"], d["height"]) == (320, 240)
    assert d["t_us"].size == 30_000 and int(d["t_us"].min()) == 0   # re-zeroed
    assert d["time_shift"] is None
    assert d["meta"].get("gottlux.source") == "src.raw"


# --------------------------------------------------------------------- ECF (filter 36559)
def _make_ecf_like(tmp_path):
    """An HDF5 whose ``CD/events`` pipeline demands the (unregistered) ECF filter 36559 —
    built with the low-level API + ``write_direct_chunk``, so reading it fails exactly the
    way a real Metavision "compress on save" file does without the codec."""
    path = str(tmp_path / "ecf.hdf5")
    n = 100
    ev = np.zeros(n, h5io.EVENT_DTYPE)
    ev["t"] = np.arange(n)
    with h5py.File(path, "w") as f:
        g = f.create_group("CD")
        space = h5py.h5s.create_simple((n,), (n,))
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((n,))
        dcpl.set_filter(36559, h5py.h5z.FLAG_MANDATORY)
        tid = h5py.h5t.py_create(h5io.EVENT_DTYPE, logical=True)
        dsid = h5py.h5d.create(g.id, b"events", tid, space, dcpl=dcpl)
        h5py.Dataset(dsid).id.write_direct_chunk((0,), ev.tobytes(), filter_mask=0)
        f.attrs["geometry"] = "320x320"
    return path


def test_ecf_error_mentions_optin_install(tmp_path, monkeypatch):
    path = _make_ecf_like(tmp_path)
    # simulate 'hdf5plugin not installed' (it may be, locally): None in sys.modules
    # makes `import hdf5plugin` raise ImportError.
    monkeypatch.setitem(sys.modules, "hdf5plugin", None)
    with pytest.raises(FusionError) as ei:
        h5io.read_events(path)
    msg = str(ei.value)
    assert "36559" in msg and "ECF" in msg
    assert "gottlux[hdf5]" in msg and "hdf5plugin" in msg   # the opt-in install hint
    # the loader path surfaces the same actionable error (no half-built cache kept)
    with pytest.raises(FusionError):
        eb.load(path, progress=lambda f: None)
    assert not cache.has_valid_cache(path)


def test_ecf_error_still_clear_when_plugin_present(tmp_path):
    """With hdf5plugin importable (but no ECF codec in its build), the read still raises
    the clear FusionError — only the redundant install hint may be dropped."""
    pytest.importorskip("hdf5plugin")
    path = _make_ecf_like(tmp_path)
    with pytest.raises(FusionError, match="36559"):
        h5io.read_events(path)


def test_unknown_layout_raises(tmp_path):
    h5 = str(tmp_path / "bad.h5")
    with h5py.File(h5, "w") as f:
        f.create_dataset("not_events", data=np.zeros(4))
    with pytest.raises(FusionError):
        h5io.read_events(h5)


# --------------------------------------------------------------------- the bundled example
@needs_short
def test_bundled_example_converts_and_opens(tmp_path):
    """The real clip: examples/data .raw → .h5 → gottlux.load, byte-identical; every
    artifact (the .h5 in tmp, the decode cache created beside the example) is removed."""
    pre_existing = os.path.isdir(os.path.join(DATA, "_gottlux_cache"))
    h5 = str(tmp_path / "bird.h5")
    try:
        n = h5io.write_hdf5(SHORT, h5)
        ra = eb.load(SHORT, progress=lambda f: None)
        rb = eb.load(h5, progress=lambda f: None)
        assert n == ra.n == rb.n > 0
        _assert_same_events(ra, rb)
        del ra, rb                                          # release the memmaps (Windows)
        gc.collect()
    finally:
        if not pre_existing:                                # leave examples/ pristine
            res = cache.clear_cache(SHORT)
            assert not res["skipped"], f"could not clean example cache: {res['skipped']}"
