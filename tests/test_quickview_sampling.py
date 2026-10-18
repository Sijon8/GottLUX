"""Quick-view byte sampling of large .raw files (gottlux.io.preview.sample_recording): the
sampled spans hold exactly the events a full decode has there, seeks decode exactly, the
pre-roll and time zero match, EVT3 samples its start only, provenance says "partial", and
the quick viewer opens a large file as samples and can decode the rest on request."""
import os
import time

import numpy as np
import pytest

from gottlux.io import decode as dec
from gottlux.io import preview
from gottlux.io.writer import write_raw


def _write(path, t_us, seed=0, W=320, H=240):
    rng = np.random.default_rng(seed)
    n = t_us.size
    write_raw(str(path), rng.integers(0, W, n), rng.integers(0, H, n),
              rng.integers(0, 2, n), t_us, width=W, height=H)
    return str(path)


def _varied(seconds=8, seed=0, t_off=0.0):
    rng = np.random.default_rng(seed)
    parts = []
    for s in range(seconds):
        n = int(20_000 + 60_000 * (0.5 + 0.5 * np.sin(s)))       # density changes per second
        parts.append(np.sort(rng.uniform(s, s + 1, n)))
    return ((np.concatenate(parts) + t_off) * 1e6).astype(np.int64)


def _same_in(full, rec, a_s, b_s):
    a, b = int(a_s * 1e6), int(b_s * 1e6)
    ft, st = np.asarray(full["t"]), np.asarray(rec.t)
    fs, ss = (ft >= a) & (ft < b), (st >= a) & (st < b)
    return (fs.sum() == ss.sum() and fs.sum() > 0
            and np.array_equal(ft[fs], st[ss])
            and np.array_equal(full["x"][fs], np.asarray(rec.x)[ss])
            and np.array_equal(full["y"][fs], np.asarray(rec.y)[ss])
            and np.array_equal(full["p"][fs], np.asarray(rec.p)[ss]))


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    return _write(tmp_path_factory.mktemp("s") / "big.raw", _varied())


def test_sampled_spans_match_full_decode(raw):
    full = dec.decode(raw)
    stages = []
    rec = preview.sample_recording(raw, chunk_mb=0.4, on_stage=stages.append)
    assert len(stages) == 2 and rec.is_sampled and rec.seekable
    assert [c[0] for c in rec.chunks] == ["start", "middle", "end"]
    assert rec.t0_us == full["t0_us"]
    assert abs(rec.duration_s - full["t"][-1] / 1e6) < 1e-3     # the whole file's span
    assert len(rec.coverage) == 3
    for a, b in rec.coverage:
        assert _same_in(full, rec, a, b), (a, b)
    assert 0 < rec.sampled_bytes < rec.file_bytes
    assert "start/middle/end" in rec.describe()


def test_seek_into_a_gap_decodes_exactly(raw):
    full = dec.decode(raw)
    rec = preview.sample_recording(raw, chunk_mb=0.4)
    (_, end_first), (start_second, _) = rec.coverage[0], rec.coverage[1]
    gap = end_first + 0.3 * (start_second - end_first)            # inside the first gap
    assert not rec.covers(gap, gap + 0.2)
    rec.ensure_window(gap, gap + 0.2)
    assert rec.covers(gap, gap + 0.2)
    assert _same_in(full, rec, gap, gap + 0.2)
    assert rec.chunks[-1][0] == "seek"


def test_preroll_and_time_zero_match_full_decode(tmp_path):
    rng = np.random.default_rng(3)
    pre = np.sort(rng.uniform(0, 0.001, 300))                     # a corrupt leading cluster
    t = np.concatenate([pre, _varied(seconds=6, seed=4, t_off=10.0) / 1e6])
    path = _write(tmp_path / "preroll.raw", (t * 1e6).astype(np.int64), seed=5)
    full = dec.decode(path)
    rec = preview.sample_recording(path, chunk_mb=0.3)
    assert rec.t0_us == full["t0_us"]                              # the pre-roll is dropped
    for a, b in rec.coverage:
        assert _same_in(full, rec, a, b)


def test_small_file_decodes_whole(tmp_path):
    path = _write(tmp_path / "small.raw", _varied(seconds=2, seed=6))
    rec = preview.sample_recording(path, chunk_mb=25)
    full = dec.decode(path)
    assert rec.chunks[0][0] == "whole" and rec.n == len(full["t"])


def test_evt3_samples_start_only(tmp_path):
    from tests.test_decode_vectorized import _synth_evt3_words
    words = _synth_evt3_words(60_000)
    path = tmp_path / "e3.raw"
    with open(path, "wb") as f:
        f.write(b"% evt 3.0\n% format EVT3;height=320;width=320\n% geometry 320x320\n% end\n")
        f.write(np.asarray(words, "<u2").tobytes())
    assert dec.detect_format(dec.parse_header(str(path))[0]) == "evt3"
    stages = []
    rec = preview.sample_recording(str(path), chunk_mb=0.05, on_stage=stages.append)
    assert len(stages) == 1 and not rec.seekable and not rec.duration_known
    assert [c[0] for c in rec.chunks] == ["start"]
    assert "EVT3" in rec.describe()


def test_should_sample_threshold_env_and_cache(raw, monkeypatch, tmp_path):
    monkeypatch.setenv("GOTTLUX_QUICKVIEW_SAMPLE_MB", "1")
    assert preview.should_sample(raw)
    monkeypatch.setenv("GOTTLUX_QUICKVIEW_SAMPLE_MB", "0")
    assert not preview.should_sample(raw)
    monkeypatch.delenv("GOTTLUX_QUICKVIEW_SAMPLE_MB")
    assert not preview.should_sample(raw)                          # under 100 MB by default
    small = _write(tmp_path / "c.raw", _varied(seconds=3, seed=7))
    monkeypatch.setenv("GOTTLUX_QUICKVIEW_SAMPLE_MB", "0.5")
    assert preview.should_sample(small)
    from gottlux.io import cache
    cache.load(small)                                              # a cache hit is instant
    assert not preview.should_sample(small)


def test_provenance_marks_partial_decode(raw):
    from gottlux.run.export_provenance import partial_note, source_facts
    rec = preview.sample_recording(raw, chunk_mb=0.4)
    f = source_facts(rec, sha256=False)
    assert f["decoded"] == "partial" and f["events_decoded"] == rec.n
    assert f["events"] is None or f["events"] > rec.n               # never the sampled count
    assert len(f["coverage_s"]) == 3 and "start/middle/end" in f["sampled"]
    assert "partly decoded" in partial_note(rec)
    from gottlux.run.export_provenance import _readme_sources
    lines = []
    _readme_sources(lines, [f])
    assert any("Partly decoded when exported" in ln for ln in lines)
    import gottlux
    full = gottlux.load(raw)
    assert source_facts(full, sha256=False)["decoded"] == "full"
    assert partial_note(full) is None


def test_quickview_opens_large_file_as_samples(raw, monkeypatch, tmp_path):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("pyqtgraph")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import shutil
    path = str(tmp_path / "qv.raw")
    shutil.copy(raw, path)                                         # no cache beside it
    monkeypatch.setenv("GOTTLUX_QUICKVIEW_SAMPLE_MB", "1")
    monkeypatch.setattr(preview, "SAMPLE_CHUNK_MB", 0.4)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from gottlux.app.quickview import QuickViewWindow
    win = QuickViewWindow()

    def wait(cond, secs=60):
        end = time.time() + secs
        while time.time() < end and not cond():
            app.processEvents(); time.sleep(0.02)
        return cond()

    win.load(path)
    assert wait(lambda: getattr(win.rec, "is_sampled", False) and len(win.rec.chunks) >= 3)
    assert not win.decode_btn.isHidden() and "sampled" in win.info_lbl.text()
    assert abs(win.clock.t1 - win.rec.t_stop_s) < 1e-6             # timeline spans the file
    win._decode_whole()
    assert wait(lambda: win.rec is not None and not getattr(win.rec, "is_preview", False))
    assert win.decode_btn.isHidden()
    win.close()
