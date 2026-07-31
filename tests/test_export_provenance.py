"""
Tests for gottlux.run.export_provenance — the self-documenting export folder.

Covers the three pieces every export path leans on: the **folder** (its name, its
uniqueness, and the empty-only discard that cleans up after an encoder that never ran);
the **source facts** (a streamed SHA-256 that an independent ``hashlib`` pass must agree
with, geometry and format read from the container header, event count and duration read
from a decode cache when one already exists — and never a decode forced to get them, which
is what keeps a fifteen-clip export from re-decoding fifteen files); and the two
**documents**, whose README must state the same facts in the prescribed order that
``provenance.json`` carries machine-readably.

Degradation is a first-class case here: a missing source, an unreadable one, and an
in-memory recording with no file at all each have to be *recorded as such* rather than
raised on, because an export must never be lost to a broken source path.
"""
import hashlib
import json
import os
import shutil

import numpy as np
import pytest

from gottlux.io.recording import Recording
from gottlux.run import export_provenance as prov

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "examples", "data")
SHORT = os.path.join(DATA, "Humming_Bird_Fight_merged_shortest.raw")
needs_short = pytest.mark.skipif(not os.path.exists(SHORT),
                                 reason="bundled example clip missing: "
                                        + os.path.basename(SHORT))


def _clip_copy(tmp_path, name="clip.raw"):
    """The bundled example clip, copied where it provably has no decode cache yet."""
    dest = str(tmp_path / name)
    shutil.copyfile(SHORT, dest)
    return dest


def _rec(n=100, w=32, h=24, name="mem"):
    """An in-memory recording — no file, no path, nothing to hash."""
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 0.5, n) * 1e6).astype(np.int64)
    return Recording.from_events(rng.integers(0, w, n), rng.integers(0, h, n),
                                 rng.integers(0, 2, n), t, width=w, height=h, name=name)


# ==================================================================== the folder
def test_export_folder_is_created_beside_the_chosen_output(tmp_path):
    """``<parent>/<stem>_export_<UTC-stamp>/``, with the artifact addressed inside it."""
    out = str(tmp_path / "movie.mp4")
    folder = prov.export_folder(out, stamp="20260731_120000Z")

    assert os.path.isdir(folder)
    assert os.path.dirname(folder) == str(tmp_path)          # beside the chosen location
    assert os.path.basename(folder) == "movie_export_20260731_120000Z"
    # the artifact keeps its own name, one level in — not a file plus loose sidecars
    assert prov.artifact_path(folder, out) == os.path.join(folder, "movie.mp4")


def test_export_folder_never_overwrites_a_same_second_sibling(tmp_path):
    out = str(tmp_path / "movie.mp4")
    first = prov.export_folder(out, stamp="20260731_120000Z")
    second = prov.export_folder(out, stamp="20260731_120000Z")
    assert first != second and os.path.isdir(second)


def test_discard_folder_removes_only_an_empty_folder(tmp_path):
    """The cleanup an export uses when the encoder was never there — it must never take a
    partial artifact with it."""
    empty = prov.export_folder(str(tmp_path / "gone.mp4"))
    assert prov.discard_folder(empty) is True and not os.path.exists(empty)

    kept = prov.export_folder(str(tmp_path / "kept.mp4"))
    with open(os.path.join(kept, "kept.mp4"), "wb") as f:
        f.write(b"partial")
    assert prov.discard_folder(kept) is False and os.path.isdir(kept)


# ==================================================================== source facts
@needs_short
def test_source_facts_hashes_a_real_clip_without_forcing_a_decode(tmp_path):
    """The bundled clip, with no cache anywhere near it: the digest must match an
    independent ``hashlib`` pass, the geometry and format come off the container header,
    and no decode cache may appear as a side effect of asking."""
    clip = _clip_copy(tmp_path)
    assert os.listdir(tmp_path) == ["clip.raw"]              # nothing cached yet

    facts = prov.source_facts(clip)

    expected = hashlib.sha256()
    with open(clip, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            expected.update(chunk)
    assert facts["sha256"] == expected.hexdigest() and len(facts["sha256"]) == 64

    assert facts["path"] == os.path.abspath(clip) and facts["directory"] == str(tmp_path)
    assert facts["name"] == "clip.raw" and facts["available"] is True
    assert facts["bytes"] == os.path.getsize(clip) and facts["size"] == "948.7 kB"
    assert facts["format"] == "evt21" and (facts["width"], facts["height"]) == (320, 320)
    # the count would have cost a full decode, so it is left unread — and said to be
    assert facts["events"] is None and facts["duration_s"] is None
    assert facts["facts_from"] == "container header"
    assert "no decode cache" in facts["note"]
    # asking must not have built one
    assert os.listdir(tmp_path) == ["clip.raw"]


@needs_short
def test_source_facts_reads_count_and_duration_from_an_existing_cache(tmp_path):
    """Once a cache exists — the one ``load()`` itself would build — the event count and
    the duration come out of it, exactly matching a real load, and asking again builds
    nothing further."""
    import gottlux as eb
    from gottlux.io import cache

    clip = _clip_copy(tmp_path)
    rec = eb.load(clip)                                      # the decode a load performs
    assert cache.has_valid_cache(clip)
    before = sorted(os.listdir(tmp_path))

    facts = prov.source_facts(clip)
    assert facts["facts_from"] == "decode cache"
    assert facts["events"] == rec.n > 0
    assert facts["duration_s"] == pytest.approx(rec.duration_s, abs=1e-5)
    assert (facts["width"], facts["height"]) == (rec.width, rec.height)
    assert facts["note"] is None
    assert sorted(os.listdir(tmp_path)) == before            # no second cache


def test_source_facts_prefers_a_loaded_recording_over_the_disk(tmp_path):
    """A recording already in hand answers for its own geometry, count and duration —
    the file is still hashed, but nothing is re-read to describe it."""
    from gottlux.io import writer

    rec = _rec(n=500, w=48, h=36, name="loaded")
    path = str(tmp_path / "src.raw")
    writer.write_raw(path, rec.x, rec.y, rec.p, rec.t, width=rec.width, height=rec.height)

    facts = prov.source_facts(path, rec=rec)
    assert facts["facts_from"] == "loaded recording"
    assert facts["events"] == rec.n and (facts["width"], facts["height"]) == (48, 36)
    assert facts["duration_s"] == pytest.approx(rec.duration_s, abs=1e-6)
    assert len(facts["sha256"]) == 64 and facts["available"] is True
    assert facts["label"] == "loaded"                        # the recording's own name


# ------------------------------------------------------------------ degradation
def test_source_facts_records_a_missing_source_instead_of_raising(tmp_path):
    """A source that moved since it was placed: the path is still recorded, the fact that
    it was gone is stated, and the export goes on."""
    facts = prov.source_facts(str(tmp_path / "vanished.raw"))
    assert facts["available"] is False
    assert facts["path"] == os.path.abspath(str(tmp_path / "vanished.raw"))
    assert facts["directory"] == str(tmp_path) and facts["name"] == "vanished.raw"
    assert facts["sha256"] is None and facts["events"] is None
    assert "not present at this path" in facts["note"]


def test_source_facts_records_an_in_memory_recording_as_having_no_file():
    facts = prov.source_facts("", rec=_rec(name="live"))
    assert facts["path"] is None and facts["sha256"] is None
    assert facts["label"] == "live" and facts["name"] == "live"
    assert facts["events"] == 100 and (facts["width"], facts["height"]) == (32, 24)
    assert "in-memory recording" in facts["note"]


def test_source_facts_degrades_on_an_unreadable_container(tmp_path):
    """A file that is present but not a recording is hashed and sized like any other —
    only the geometry it cannot supply is left blank."""
    junk = str(tmp_path / "notreally.raw")
    with open(junk, "wb") as f:
        f.write(b"this is not an event stream")

    facts = prov.source_facts(junk)
    assert facts["available"] is True and len(facts["sha256"]) == 64
    assert facts["bytes"] == 27
    assert facts["events"] is None
    assert facts["note"]                                     # it says why, and does not raise


def test_source_facts_accepts_a_recording_positionally(tmp_path):
    """``source_facts(rec)`` — a recording passed where a path is expected answers for
    itself, which is how the export paths hand over what they already hold."""
    from gottlux.io import writer

    rec = _rec(n=200, name="positional")
    path = str(tmp_path / "pos.raw")
    writer.write_raw(path, rec.x, rec.y, rec.p, rec.t, width=rec.width, height=rec.height)
    rec.source_path = path

    facts = prov.source_facts(rec)
    assert facts["path"] == os.path.abspath(path) and facts["events"] == rec.n


# ==================================================================== the documents
def _write(tmp_path, **kw):
    """One finished export folder: an artifact, two sources, two usage rows."""
    from gottlux.core.canvas import CanvasClip

    out = str(tmp_path / "out.raw")
    folder = prov.export_folder(out, stamp="20260731_120000Z")
    artifact = prov.artifact_path(folder, out)
    with open(artifact, "wb") as f:
        f.write(b"\x00" * 2048)

    a = prov.source_facts(str(tmp_path / "one.raw"))         # missing on purpose
    b = prov.source_facts("", rec=_rec(name="two"))
    usage = [prov.cell_usage(CanvasClip(source="two", rect=(0, 0, 64, 48),
                                        roi=(1, 2, 30, 20), t_offset_s=0.5,
                                        time_scale=0.25, accumulation_s=0.01,
                                        mode="polarity", colormap="viridis",
                                        tonemap="log", gamma=0.8, loop=True),
                            2, name="cell two", lane="sequence", trim_in_s=0.1,
                            trim_out_s=0.9)]
    kw.setdefault("extra", {})["usage"] = usage
    readme, record = prov.write_provenance(
        folder, kw.pop("kind", "Canvas .raw (composited events)"),
        prov.artifact_facts(artifact, events=1234, duration_s=2.5, width=64, height=48),
        [a, b], kw.pop("settings", {"Canvas": "64 × 48 px", "Cells": 2}), **kw)
    return folder, readme, record


def test_write_provenance_writes_both_documents(tmp_path):
    folder, readme, record = _write(tmp_path)
    assert os.path.basename(readme) == "README.md"
    assert os.path.basename(record) == "provenance.json"
    assert os.path.dirname(readme) == os.path.dirname(record) == folder


def test_provenance_json_parses_and_carries_the_required_keys(tmp_path):
    from gottlux import __version__

    folder, _readme, record = _write(tmp_path)
    with open(record, encoding="utf-8") as f:
        data = json.load(f)

    assert {"schema_version", "gottlux_version", "created_utc", "kind", "artifact",
            "sources", "usage", "settings", "files"} <= set(data)
    assert data["schema_version"] == prov.SCHEMA_VERSION
    assert data["gottlux_version"] == __version__
    assert data["created_utc"].endswith("Z")
    assert data["kind"] == "Canvas .raw (composited events)"

    assert data["artifact"]["name"] == "out.raw" and data["artifact"]["bytes"] == 2048
    assert data["artifact"]["events"] == 1234 and len(data["artifact"]["sha256"]) == 64

    assert len(data["sources"]) == 2 and len(data["usage"]) == 1
    row = data["usage"][0]
    assert row["source"] == 2                                # 1-based into sources[]
    assert row["dest_rect"] == [0, 0, 64, 48] and row["roi"] == [1, 2, 30, 20]
    assert row["time_scale"] == 0.25 and row["loop"] is True
    assert data["environment"]["gottlux_version"] == __version__
    # the files table names every file actually in the folder, each with a role
    assert {f["name"] for f in data["files"]} == set(os.listdir(folder))
    assert all(f["role"] for f in data["files"])


def test_readme_states_the_facts_in_the_prescribed_order(tmp_path):
    """The document is a report, and its order is part of the contract: what was produced,
    when, the sources, how each was used, the settings, the text, how to reproduce it, and
    what is in the folder."""
    _folder, readme, _record = _write(
        tmp_path, extra={"texts": [{"text": "Hello", "kind": "slide"}],
                         "reproduce": {"spec": "out.gottlux-canvas.json",
                                       "steps": ["Reload the spec."]}})
    with open(readme, encoding="utf-8") as f:
        text = f.read()

    headings = ["# GottLUX export — Canvas .raw (composited events)",
                "## What was produced", "## When, and with what", "## Source recordings",
                "## How each source was used", "## Export settings", "## Titles and text",
                "## Reproducing it", "## Files in this folder"]
    positions = [text.index(h) for h in headings]            # every one present…
    assert positions == sorted(positions)                    # …and in this order

    assert "`out.raw`" in text and "1,234" in text           # the artifact and its events
    assert "2 distinct recording(s)" in text
    assert "not present at this path" in text                # the missing source, stated
    assert "in-memory recording" in text
    assert "x 0, y 0, 64 × 48 px" in text                    # the cell's rect, in words
    assert "x 1–30, y 2–20 px" in text                       # and its ROI crop
    assert "0.25 (clip seconds per program second) — 4× slow motion" in text
    assert "10 ms" in text                                   # accumulation, in milliseconds
    assert "Text is a **render-time** item" in text          # the video-only caveat
    assert "out.gottlux-canvas.json" in text
    assert "| `README.md` |" in text and "| `provenance.json` |" in text


def test_readme_omits_the_text_section_when_there_is_none(tmp_path):
    _folder, readme, _record = _write(tmp_path)
    with open(readme, encoding="utf-8") as f:
        text = f.read()
    assert "## Titles and text" not in text
    assert "## Reproducing it" in text
    assert "saves no composition spec" in text               # and says so plainly


def test_artifact_facts_hashes_and_sizes_the_produced_file(tmp_path):
    path = str(tmp_path / "made.mp4")
    with open(path, "wb") as f:
        f.write(b"x" * 1500)
    facts = prov.artifact_facts(path, frames=42, fps=30.0, codec="H.264")
    assert facts["name"] == "made.mp4" and facts["bytes"] == 1500
    assert facts["size"] == "1.5 kB" and len(facts["sha256"]) == 64
    assert facts["frames"] == 42 and facts["codec"] == "H.264"
    # a field a caller has no answer for is left out rather than recorded as null
    assert "duration_s" not in prov.artifact_facts(path, duration_s=None)


def test_artifact_facts_survives_a_file_that_is_not_there(tmp_path):
    facts = prov.artifact_facts(str(tmp_path / "never.mp4"), frames=0)
    assert facts["bytes"] is None and facts["sha256"] is None and facts["frames"] == 0


def test_seconds_never_print_as_negative_zero():
    """A compiled program routinely yields an offset a hair below zero; the document must
    not report it as "-0.000 s"."""
    assert prov._seconds(-1e-9) == "0.000 s"
    assert prov._seconds(-0.0) == "0.000 s"
    assert prov._seconds(0.1499) == "0.150 s"
    assert prov._seconds(-1.25) == "-1.250 s"          # a real negative still reads so


def test_time_scale_states_what_the_clock_does():
    assert prov._scale(1.0) == "1 (clip seconds per program second) — real time"
    assert prov._scale(0.5) == "0.5 (clip seconds per program second) — 2× slow motion"
    assert prov._scale(4.0) == "4 (clip seconds per program second) — 4× speed-up"
    assert prov._scale(0.0).endswith("stopped")


def test_human_bytes_reads_in_decimal_units():
    assert prov.human_bytes(0) == "0 B"
    assert prov.human_bytes(999) == "999 B"
    assert prov.human_bytes(1234) == "1.2 kB"
    assert prov.human_bytes(1_234_567) == "1.2 MB"
    assert prov.human_bytes(9_876_543_210) == "9.9 GB"
    assert prov.human_bytes(None) == "—"
