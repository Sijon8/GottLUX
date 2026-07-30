"""
Tests for the massive-file fast preview (:mod:`gottlux.io.preview`) — the sampled open.

The promise under test: a large ``.raw`` can be *shown* in roughly a second by indexing the
whole file (one expansion-free I/O pass) and decoding only sampled windows, and everything
the samples contain is **byte-identical** to what the normal full decode produces for the
same spans. Covered here:

* the index alone recovers the full-file duration (tight where the stream is dense,
  bracketing the event span where the file has leading/trailing idle time);
* beginning/middle/end slices byte-match the same spans of a full :func:`decode.decode`;
* coverage spans are ascending, disjoint, and grow — bounded — when :meth:`window` seeks
  into an un-decoded gap (and do NOT grow for over-long or whole-file requests);
* the activation policy: size threshold, ``GOTTLUX_PREVIEW_THRESHOLD_MB`` override,
  ``0`` disables, cache hits and non-EVT2.x files never preview;
* the EVT2.0 index/slice twin, and EVT3 raising :class:`UnsupportedPreview`;
* the GUI loader's two-phase emit (``preview`` then ``loaded``) and the headless
  ``gottlux.load`` staying preview-free no matter the env.
"""
import os

import numpy as np
import pytest

import gottlux as eb
from gottlux.io import cache, decode, preview, writer

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "examples", "data")
QUAD = os.path.join(DATA, "5inch_quadcopter.raw")                    # 45 MB, idle head+tail
BIRD = os.path.join(DATA, "Humming_Bird_Fight_merged_2.raw")         # 11 MB, dense
SHORT = os.path.join(DATA, "Humming_Bird_Fight_merged_shortest.raw")  # 0.9 MB, dense

needs = {p: pytest.mark.skipif(not os.path.exists(p),
                               reason=f"bundled example clip missing: {os.path.basename(p)}")
         for p in (QUAD, BIRD, SHORT)}

_FULL: dict = {}


def _full(path):
    """Full in-memory reference decode, cached across tests (single-chunk => canonical order)."""
    if path not in _FULL:
        _FULL[path] = decode.decode(path)
    return _FULL[path]


def _synth_raw(tmp_path, n=60_000, dur_s=6.0, seed=0, name="synth.raw"):
    """A small EVT2.1 file with uniform random events (via the tested round-trip writer)."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 320, n).astype(np.uint16)
    y = rng.integers(0, 240, n).astype(np.uint16)
    p = rng.integers(0, 2, n).astype(np.uint8)
    t = np.sort(rng.integers(0, int(dur_s * 1e6), n)).astype(np.int64)
    path = str(tmp_path / name)
    writer.write_raw(path, x, y, p, t, width=320, height=240)
    return path


def _assert_spans_byte_match(rec, d):
    """Every covered span of the preview must equal the full decode's events exactly.

    ``rec.t0_us == d['t0_us']`` (asserted) makes both time axes share one origin, so the
    expected slice is found with the same ``searchsorted`` the Recording itself uses."""
    assert rec.t0_us == d["t0_us"]
    pt = np.asarray(rec.t)
    for a_s, b_s in rec.coverage:
        a_us, b_us = int(round(a_s * 1e6)), int(round(b_s * 1e6))
        i0, i1 = np.searchsorted(d["t"], [a_us, b_us], side="left")
        j0, j1 = np.searchsorted(pt, [a_us, b_us], side="left")
        assert j1 - j0 == i1 - i0, f"span ({a_s:.3f},{b_s:.3f}): {j1-j0} vs {i1-i0} events"
        assert np.array_equal(np.asarray(rec.x)[j0:j1], d["x"][i0:i1])
        assert np.array_equal(np.asarray(rec.y)[j0:j1], d["y"][i0:i1])
        assert np.array_equal(np.asarray(rec.p)[j0:j1], d["p"][i0:i1])
        assert np.array_equal(pt[j0:j1], d["t"][i0:i1])


# ====================================================================================
# Index: duration from one I/O pass
# ====================================================================================
@pytest.mark.parametrize("path", [pytest.param(BIRD, marks=needs[BIRD]),
                                  pytest.param(SHORT, marks=needs[SHORT])])
def test_index_duration_matches_full_decode(path):
    """On a dense stream the index's duration equals the decoded event span to ~a TIME_HIGH."""
    ix = preview.scan_index(path)
    d = _full(path)
    assert ix["fmt"] == "evt21"
    full_dur = float(d["t"][-1]) / 1e6
    assert ix["duration_s"] == pytest.approx(full_dur, abs=0.001)   # 64 µs grid + low bits


@needs[QUAD]
def test_index_brackets_event_span_with_idle_time(path=QUAD):
    """A file with idle lead-in/tail: the index still *brackets* the events — the timeline
    the preview shows covers every event (that is what the transport bar needs)."""
    ix = preview.scan_index(path)
    d = _full(path)
    origin_us = ix["origin_high"] * ix["unit_us"]
    end_us = (int(ix["th_vals"][-1]) + 1) * ix["unit_us"]
    assert origin_us <= d["t0_us"]                                  # starts at/before 1st event
    assert end_us >= d["t0_us"] + int(d["t"][-1])                   # ends at/after last event
    assert ix["duration_s"] >= float(d["t"][-1]) / 1e6 - 1e-4


# ====================================================================================
# Sampled slices: byte-identical to the full decode
# ====================================================================================
@pytest.mark.parametrize("path,slice_s", [pytest.param(QUAD, 0.2, marks=needs[QUAD]),
                                          pytest.param(SHORT, 0.02, marks=needs[SHORT])])
def test_sampled_slices_byte_match_full_decode(path, slice_s):
    d = _full(path)
    rec = preview.preview_recording(path, slice_s=slice_s)
    assert rec.is_preview
    _assert_spans_byte_match(rec, d)


@needs[SHORT]
def test_short_file_single_slice_covers_everything():
    """duration <= n_slices*slice_s -> one window over the whole file == the full decode."""
    d = _full(SHORT)
    rec = preview.preview_recording(SHORT)          # default 3 x 2 s >> 0.13 s clip
    assert len(rec.coverage) == 1
    assert rec.n == len(d["t"])
    assert np.array_equal(np.asarray(rec.t), d["t"])
    assert np.array_equal(np.asarray(rec.x), d["x"])


@needs[QUAD]
def test_coverage_spans_ascending_disjoint_and_sampled():
    d = _full(QUAD)
    rec = preview.preview_recording(QUAD, slice_s=0.2)
    cov = rec.coverage
    assert len(cov) >= 2                            # beginning + middle at least
    for (a, b) in cov:
        assert 0.0 <= a < b <= rec.duration_s + 1e-6
    for (_, b0), (a1, _) in zip(cov, cov[1:]):
        assert b0 <= a1                             # disjoint, ascending
    assert rec.n < len(d["t"])                      # a *sample*, not the full decode
    assert rec.duration_s >= float(d["t"][-1]) / 1e6 - 1e-4   # but the WHOLE timeline
    assert rec.t_stop_s == rec.duration_s


# ====================================================================================
# On-demand slice decode on seek
# ====================================================================================
@needs[QUAD]
def test_window_into_gap_decodes_on_demand_and_byte_matches():
    d = _full(QUAD)
    rec = preview.preview_recording(QUAD, slice_s=0.2)
    cov = rec.coverage
    g = (cov[0][1] + cov[1][0]) / 2                 # the middle of the first un-decoded gap
    assert not rec.covers(g, g + 0.05)
    w = rec.window(g, g + 0.05)
    assert rec.covers(g, g + 0.05)                  # coverage grew to include the request
    i0, i1 = np.searchsorted(d["t"], [int(round(g * 1e6)),
                                      int(round((g + 0.05) * 1e6))], side="left")
    assert np.array_equal(w.x, d["x"][i0:i1])
    assert np.array_equal(w.y, d["y"][i0:i1])
    assert np.array_equal(w.p, d["p"][i0:i1])
    assert np.array_equal(w.t, d["t"][i0:i1])
    # ...and the merged coverage list stayed ascending + disjoint
    for (_, b0), (a1, _) in zip(rec.coverage, rec.coverage[1:]):
        assert b0 <= a1


@needs[QUAD]
def test_oversized_and_whole_file_windows_do_not_decode():
    """window() only auto-decodes bounded requests — an accumulation window, not the file."""
    rec = preview.preview_recording(QUAD, slice_s=0.2)
    before = rec.coverage
    rec.window()                                                    # whole file: no decode
    rec.window(1.0, 1.0 + preview.ON_DEMAND_MAX_SPAN_S + 1.0)       # over-long: no decode
    assert rec.coverage == before


# ====================================================================================
# Activation policy (threshold + env override + cache hits)
# ====================================================================================
def test_threshold_default_and_env_override(monkeypatch):
    monkeypatch.delenv("GOTTLUX_PREVIEW_THRESHOLD_MB", raising=False)
    assert preview.preview_threshold_bytes() == int(preview.PREVIEW_THRESHOLD_MB * 2**20)
    monkeypatch.setenv("GOTTLUX_PREVIEW_THRESHOLD_MB", "50")
    assert preview.preview_threshold_bytes() == 50 * 2**20
    monkeypatch.setenv("GOTTLUX_PREVIEW_THRESHOLD_MB", "0")          # 0 disables
    assert preview.preview_threshold_bytes() is None
    monkeypatch.setenv("GOTTLUX_PREVIEW_THRESHOLD_MB", "-5")         # negative disables
    assert preview.preview_threshold_bytes() is None
    monkeypatch.setenv("GOTTLUX_PREVIEW_THRESHOLD_MB", "not-a-number")
    assert preview.preview_threshold_bytes() == int(preview.PREVIEW_THRESHOLD_MB * 2**20)


def test_should_preview_policy(tmp_path, monkeypatch):
    path = _synth_raw(tmp_path)                     # ~1 MB
    monkeypatch.delenv("GOTTLUX_PREVIEW_THRESHOLD_MB", raising=False)
    assert preview.should_preview(path) is False    # small file, default 200 MB threshold
    monkeypatch.setenv("GOTTLUX_PREVIEW_THRESHOLD_MB", "0.5")
    assert preview.should_preview(path) is True     # large (vs threshold), uncached EVT2.1
    monkeypatch.setenv("GOTTLUX_PREVIEW_THRESHOLD_MB", "0")
    assert preview.should_preview(path) is False    # previews disabled
    monkeypatch.setenv("GOTTLUX_PREVIEW_THRESHOLD_MB", "0.5")
    assert preview.should_preview(str(tmp_path / "missing.raw")) is False
    other = tmp_path / "notes.txt"
    other.write_bytes(b"\x00" * 2**20)
    assert preview.should_preview(str(other)) is False              # not a .raw
    cache.load(path)                                # decode-once -> a valid cache now exists
    assert cache.has_valid_cache(path) is True
    assert preview.should_preview(path) is False    # cache hits open instantly, no preview


# ====================================================================================
# The EVT2.0 twin, and EVT3 falling back
# ====================================================================================
def _write_evt2(path, x, y, p, t, width=320, height=240):
    """Hand-assemble an EVT2.0 file: 32-bit words, type nibble at bit 28 — 0x8 TIME_HIGH
    (28-bit high, unit 64 µs), 0/1 CD with t%64 at bits 22–27, x at 11–21, y at 0–10."""
    order = np.argsort(t, kind="stable")
    x, y, p, t = x[order], y[order], p[order], t[order]
    words, hi_prev = [], -1
    for xi, yi, pi, ti in zip(x.tolist(), y.tolist(), p.tolist(), t.tolist()):
        hi = ti // 64
        if hi != hi_prev:
            words.append((0x8 << 28) | (hi & 0x0FFFFFFF))
            hi_prev = hi
        words.append((int(pi) << 28) | ((ti % 64) << 22) | (int(xi) << 11) | int(yi))
    header = (f"% evt 2.0\n% format EVT2;height={height};width={width}\n"
              f"% geometry {width}x{height}\n% end\n").encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        f.write(np.array(words, dtype="<u4").tobytes())


def test_evt2_index_and_slices_byte_match(tmp_path):
    rng = np.random.default_rng(2)
    n, dur_us = 40_000, 5_000_000
    x = rng.integers(0, 320, n).astype(np.int64)
    y = rng.integers(0, 240, n).astype(np.int64)
    p = rng.integers(0, 2, n).astype(np.int64)
    t = np.sort(rng.integers(0, dur_us, n)).astype(np.int64)
    path = str(tmp_path / "evt2.raw")
    _write_evt2(path, x, y, p, t)
    d = decode.decode(path)
    assert d["fmt"] == "evt2"
    ix = preview.scan_index(path)
    assert ix["fmt"] == "evt2"
    assert ix["duration_s"] == pytest.approx(float(d["t"][-1]) / 1e6, abs=0.001)
    rec = preview.preview_recording(path, slice_s=0.3)
    assert len(rec.coverage) >= 2
    _assert_spans_byte_match(rec, d)


def test_evt3_is_unsupported_for_preview(tmp_path, monkeypatch):
    """EVT3's stateful stream can't be seeded from an index — no preview, normal load."""
    path = str(tmp_path / "evt3.raw")
    words = np.array([0x8005, 0x6000, 0x0010, 0x2020], dtype="<u2")  # TH, TL, y, single CD
    with open(path, "wb") as f:
        f.write(b"% evt 3.0\n% format EVT3;height=240;width=320\n% end\n")
        f.write(words.tobytes())
    monkeypatch.setenv("GOTTLUX_PREVIEW_THRESHOLD_MB", "0.000001")
    assert preview.should_preview(path) is False
    with pytest.raises(preview.UnsupportedPreview):
        preview.preview_recording(path)


# ====================================================================================
# Loader two-phase emit + headless semantics unchanged
# ====================================================================================
def test_headless_load_never_returns_preview(tmp_path, monkeypatch):
    """gottlux.load ignores the preview machinery entirely — library behaviour unchanged."""
    path = _synth_raw(tmp_path, name="headless.raw")
    monkeypatch.setenv("GOTTLUX_PREVIEW_THRESHOLD_MB", "0.001")      # would preview in the GUI
    rec = eb.load(path)
    assert getattr(rec, "is_preview", False) is False
    assert rec.n == 60_000


def test_loader_emits_preview_then_full(tmp_path, monkeypatch):
    """The GUI loader is two-phase for a large uncached .raw: a playable sampled preview
    lands first (whole-file span), then the real memmap-backed Recording supersedes it."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtWidgets
    from gottlux.app.loader import RecordingLoader
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path = _synth_raw(tmp_path, name="two_phase.raw")
    monkeypatch.setenv("GOTTLUX_PREVIEW_THRESHOLD_MB", "0.001")
    got = []
    loader = RecordingLoader(path)
    # DirectConnection: slots run on the worker thread, so no event loop is needed
    loader.preview.connect(lambda r: got.append(("preview", r)), QtCore.Qt.DirectConnection)
    loader.loaded.connect(lambda r: got.append(("loaded", r)), QtCore.Qt.DirectConnection)
    loader.failed.connect(lambda m: got.append(("failed", m)), QtCore.Qt.DirectConnection)
    loader.start()
    assert loader.wait(120_000)
    assert [k for k, _ in got] == ["preview", "loaded"]
    prev, full = got[0][1], got[1][1]
    assert prev.is_preview is True
    assert getattr(full, "is_preview", False) is False
    assert full.n == 60_000
    assert prev.n <= full.n
    assert prev.duration_s == pytest.approx(full.duration_s, abs=0.001)  # whole-file span
    # a second load is a cache hit -> no preview phase at all
    got2 = []
    loader2 = RecordingLoader(path)
    loader2.preview.connect(lambda r: got2.append(("preview", r)), QtCore.Qt.DirectConnection)
    loader2.loaded.connect(lambda r: got2.append(("loaded", r)), QtCore.Qt.DirectConnection)
    loader2.start()
    assert loader2.wait(120_000)
    assert [k for k, _ in got2] == ["loaded"]
