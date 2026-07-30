"""Cache lifecycle management: discovery/report (``cache_info``), safe clearing
(``clear_cache`` — never the source data), legacy-dirname support, the CLI flags, and the
locked-file fallback (a re-decode whose bins are memmapped elsewhere lands in a temp dir)."""
import gc
import json
import os

import numpy as np
import pytest

from gottlux.io import cache, writer


def _mkraw(path, n=3000, dur_s=0.5, seed=0):
    """A tiny synthetic EVT2.1 .raw with *n* events."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, dur_s, n) * 1e6).astype(np.int64)
    writer.write_raw(str(path), rng.integers(0, 320, n).astype(np.uint16),
                     rng.integers(0, 320, n).astype(np.uint16),
                     rng.integers(0, 2, n).astype(np.uint8), t, width=320, height=320)
    return str(path)


def _build_cache(raw):
    """Decode *raw* into its cache and release every memmap (so Windows can delete/rename)."""
    d = cache.load(str(raw))
    n = d["n"]
    del d
    gc.collect()
    return n


def test_cache_info_file_and_folder(tmp_path):
    """cache_info discovers a built cache via the file OR the capture folder (one level
    down included), with the right source / size / count / staleness."""
    raw = _mkraw(tmp_path / "cap0.raw")
    n = _build_cache(raw)

    for target in (raw, str(tmp_path)):
        entries = cache.cache_info(target)
        assert len(entries) == 1
        e = entries[0]
        assert e["stem"] == "cap0"
        assert os.path.normcase(e["source"]) == os.path.normcase(raw)
        assert e["n"] == n and n > 0
        assert e["bytes"] >= 13 * n              # x2+y2+p1+t8 bytes/event + meta
        assert e["decoder_version"] is not None and not e["stale"]
        assert len(e["files"]) == 5              # meta + 4 bins

    # one level of subfolders is scanned too
    sub = tmp_path / "flight2"; sub.mkdir()
    raw2 = _mkraw(sub / "cap1.raw", seed=1)
    _build_cache(raw2)
    entries = cache.cache_info(str(tmp_path))
    assert sorted(e["stem"] for e in entries) == ["cap0", "cap1"]

    # a newer source file marks the cache stale (a re-decode is due)
    later = os.path.getmtime(raw) + 100
    os.utime(raw, (later, later))
    e = cache.cache_info(raw)[0]
    assert e["stale"]

    # the report is printable and names the stems + the reclaim command
    text = cache.format_cache_report(entries)
    assert "cap0" in text and "cap1" in text and "--clear-cache" in text
    assert cache.format_cache_report([]) == "No decode caches found."


def test_clear_cache_removes_bins_never_the_raw(tmp_path):
    raw = _mkraw(tmp_path / "cap0.raw")
    _build_cache(raw)
    raw_bytes = open(raw, "rb").read()

    res = cache.clear_cache(str(tmp_path))
    assert res["n_stems"] == 1 and res["freed_bytes"] > 0 and not res["skipped"]
    assert len(res["removed"]) == 5
    assert cache.cache_info(str(tmp_path)) == []
    assert not (tmp_path / "_gottlux_cache").exists()   # empty cache dir pruned
    # the source data is untouched, byte for byte
    assert os.path.exists(raw) and open(raw, "rb").read() == raw_bytes


def test_stale_only_keeps_fresh_caches(tmp_path):
    a = _mkraw(tmp_path / "a.raw", seed=1)
    b = _mkraw(tmp_path / "b.raw", seed=2)
    _build_cache(a); _build_cache(b)
    # make a's cache stale (decoder bumped since it was written)
    meta = tmp_path / "_gottlux_cache" / "a.meta.json"
    m = json.loads(meta.read_text()); m["decoder_version"] = -1
    meta.write_text(json.dumps(m))
    assert [e["stem"] for e in cache.cache_info(str(tmp_path)) if e["stale"]] == ["a"]

    res = cache.clear_cache(str(tmp_path), stale_only=True)
    assert res["n_stems"] == 1
    left = cache.cache_info(str(tmp_path))
    assert [e["stem"] for e in left] == ["b"] and not left[0]["stale"]
    assert cache.has_valid_cache(b)              # b still opens instantly
    assert os.path.exists(a) and os.path.exists(b)


def test_foreign_dirname_not_discovered(tmp_path):
    """Only ``_gottlux_cache`` dirs are treated as caches — a renamed copy is ignored
    (and therefore never deleted) by the management API."""
    raw = _mkraw(tmp_path / "cap0.raw")
    _build_cache(raw)
    (tmp_path / "_gottlux_cache").rename(tmp_path / "_other_tool_cache")

    assert cache.cache_info(str(tmp_path)) == []
    res = cache.clear_cache(str(tmp_path))
    assert res["n_stems"] == 0 and not res["removed"]
    assert (tmp_path / "_other_tool_cache").exists()   # untouched
    assert os.path.exists(raw)


def test_cli_cache_info_and_clear(tmp_path, capsys):
    from gottlux.cli import main
    raw = _mkraw(tmp_path / "cap0.raw")
    n = _build_cache(raw)

    assert main(["gottlux", "--cache-info", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "cap0" in out and f"{n:,} ev" in out and "fresh" in out

    assert main(["gottlux", "--clear-cache", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Cleared 1" in out and "freed" in out
    assert cache.cache_info(str(tmp_path)) == []
    assert os.path.exists(raw)

    # nothing left → both report that, and exit 0
    assert main(["gottlux", "--cache-info", str(tmp_path)]) == 0
    assert "No decode caches found" in capsys.readouterr().out
    assert main(["gottlux", "--clear-cache", str(tmp_path), "--stale-only"]) == 0
    assert "No stale decode cache(s) found" in capsys.readouterr().out


def test_locked_bins_fall_back_to_temp_cache(tmp_path, monkeypatch):
    """A needed re-decode whose .bin files are memmapped (as by another process) must not
    crash: it decodes into a per-process temp dir under the platform relocation root,
    which --clear-cache on that root then reclaims."""
    monkeypatch.setattr(cache, "platform_root", lambda: str(tmp_path / "relocroot"))
    monkeypatch.setattr(cache, "_FALLBACK_STEMS", {})
    raw = _mkraw(tmp_path / "cap0.raw")
    d = cache.load(raw)
    n, mm = d["n"], d["x"]                       # hold a memmap on one bin, like the GUI
    cache_dir, stem = cache.cache_location(raw)
    try:
        open(stem + ".x.bin", "wb").close()      # does this platform lock mapped files?
        pytest.skip("platform does not lock memory-mapped files (POSIX semantics)")
    except OSError:
        pass                                     # Windows: EINVAL/EACCES — the real gotcha

    # mark the on-disk cache stale, so a plain load() *needs* the blocked re-decode
    meta = json.load(open(stem + ".meta.json"))
    meta["decoder_version"] = -1
    json.dump(meta, open(stem + ".meta.json", "w"))

    d2 = cache.load(raw)                         # must not raise
    assert d2["n"] == n
    fb_dir = os.path.dirname(d2["x"].filename)
    assert os.path.normcase(fb_dir) != os.path.normcase(cache_dir)
    reloc = os.path.join(str(tmp_path / "relocroot"), "_gottlux_cache")
    assert os.path.normcase(fb_dir).startswith(os.path.normcase(reloc))
    assert np.array_equal(np.asarray(d2["x"]), np.asarray(mm))

    # a second load in this process reuses the same temp cache (no second decode dir)
    d3 = cache.load(raw)
    assert os.path.normcase(os.path.dirname(d3["x"].filename)) == os.path.normcase(fb_dir)
    assert len([e for e in os.listdir(reloc) if e.startswith("tmp_")]) == 1

    # release every memmap, then --clear-cache on the relocation root reclaims the temp dir
    del d, d2, d3, mm
    gc.collect()
    res = cache.clear_cache(reloc)
    assert res["n_stems"] == 1 and not res["skipped"]
    assert not os.path.exists(os.path.join(fb_dir, "cap0.meta.json"))
