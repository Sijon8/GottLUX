"""
export_provenance.py — every exported video and event file lands in a self-documenting folder.

A composition assembled over weeks is only as trustworthy as its paper trail. A timeline
built from fifteen ``.raw`` clips collected on different days is unreadable six months
later unless the export itself records which files it drew on, where those files live, and
what was done to each one. This module is that record: every export path in the suite
writes a folder rather than a loose file,

    <parent>/<stem>_export_<UTC-stamp>/
        <artifact>                    the .mp4 / .raw exactly as it would have been written
        README.md                     the human-readable provenance document
        provenance.json               the same facts, machine-readable
        <stem>.gottlux-canvas.json    the composition spec, where one exists (re-renderable)
        …                             anything else that export path produces (e.g. a poster)

and the completion dialog reports the folder.

The conventions are :mod:`gottlux.run.tool_export`'s — the same README + ``provenance.json``
pair, the same file-roles table, the same streamed SHA-256
(:func:`gottlux.io.paths.file_sha256`) — applied to the export side rather than to a
standalone-tool bundle. The README states, in this order: what was produced · when, with
which GottLUX version, on which platform · the **source recordings**, one table row per
clip · **how each source was used** · the **export settings** · the **titles/text** when
any exist · **how to reproduce it** · and the **files in the folder** with their roles.

Source facts are gathered as cheaply as the situation allows: an already-loaded
:class:`~gottlux.io.recording.Recording` answers for itself, an on-disk source falls back
to its decode cache's meta (:mod:`gottlux.io.cache`) and then to the container header. A
provenance write never forces a decode, and a missing or unreadable source is *recorded as
missing* rather than raised on — an export is never lost to a broken source path.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone

from gottlux import __version__
from gottlux.io.paths import ext, file_sha256, unique_export_dir

#: The machine-readable record's schema id — bumped when the JSON shape changes.
SCHEMA_VERSION = "gottlux.export.provenance/1"

#: The two documents every export folder carries.
README_NAME = "README.md"
PROVENANCE_NAME = "provenance.json"

#: How many hex characters of a SHA-256 the tables show before the full digest.
SHORT_SHA = 12


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def human_bytes(n) -> str:
    """``1234567`` → ``'1.2 MB'`` — decimal units, matching the cache report's style."""
    if n is None:
        return "—"
    x = float(n)
    for unit in ("B", "kB", "MB", "GB"):
        if x < 1000 or unit == "GB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1000.0
    return f"{x:.1f} GB"


# ==================================================================== the folder
def export_folder(out_path, stamp=None) -> str:
    """Create and return the export folder for an artifact headed for *out_path*.

    ``<parent of out_path>/<stem>_export_<UTC-stamp>/`` — built through
    :func:`gottlux.io.paths.unique_export_dir`, so a same-second collision gets a numeric
    suffix instead of overwriting. The artifact is then written INSIDE the folder under
    its original file name (:func:`artifact_path`): the chosen save location gains one
    clearly-labelled folder rather than a file plus loose sidecars.
    """
    out_path = os.path.abspath(str(out_path))
    stem = os.path.splitext(os.path.basename(out_path))[0] or "export"
    return unique_export_dir(os.path.dirname(out_path) or os.getcwd(), stem,
                             purpose="export", stamp=stamp)


def artifact_path(folder, out_path) -> str:
    """Where the artifact lands inside *folder* — under its own file name, unchanged."""
    return os.path.join(folder, os.path.basename(str(out_path)))


def discard_folder(folder) -> bool:
    """Remove an export folder that never received an artifact (an encoder that was not
    installed, a write that raised). Empty-only by construction — :func:`os.rmdir` refuses
    a folder holding anything — so a partial artifact is always left where it can be seen.
    """
    try:
        os.rmdir(ext(str(folder)))
        return True
    except OSError:
        return False


# ==================================================================== source facts
def _cache_facts(path) -> dict:
    """Geometry / count / duration from an **existing** decode cache; never builds one.

    :func:`gottlux.io.cache.has_valid_cache` gates the lookup, so a file that has never
    been decoded simply yields nothing; :func:`~gottlux.io.cache.open_cached` then memmaps
    the finished bins, from which the count and the span are two element reads. The
    mapping is dropped immediately so the cache stays free to be re-decoded.
    """
    from gottlux.io import cache as _cache
    if not _cache.has_valid_cache(path):
        return {}
    _, stem = _cache.cache_location(path)
    data = _cache.open_cached(stem)
    if not data:
        return {}
    try:
        n = int(data["n"])
        t = data["t"]
        dur = (float(t[-1]) - float(t[0])) / 1e6 if n else 0.0
        return {"format": data.get("fmt"), "width": int(data["width"]),
                "height": int(data["height"]), "events": n, "duration_s": round(dur, 6),
                "facts_from": "decode cache"}
    finally:
        data.clear()


def _header_facts(path) -> dict:
    """What a header read alone can answer — container format and sensor geometry.

    Cheap by construction: :func:`gottlux.io.decode.parse_header` reads the ``.raw``
    preamble only, and an HDF5 source is opened for its attributes and dataset length
    (:class:`gottlux.io.hdf5.H5EventSource`). Neither decodes events.
    """
    from gottlux.io.hdf5 import H5EventSource, is_hdf5_path
    if is_hdf5_path(path):
        with H5EventSource(path) as src:
            return {"format": "hdf5", "width": src.width, "height": src.height,
                    "events": int(src.n), "facts_from": "HDF5 header"}
    from gottlux.io import decode as _dec
    meta, _off = _dec.parse_header(path)
    width, height = _dec.geometry(meta)
    return {"format": _dec.detect_format(meta), "width": int(width), "height": int(height),
            "facts_from": "container header"}


def source_facts(path, rec=None, sha256=True) -> dict:
    """Everything known about one source recording, gathered as cheaply as possible.

    *path* is the source file (a :class:`~gottlux.io.recording.Recording` may be passed
    instead, or alongside as *rec* — a loaded recording answers for its own geometry,
    count and duration without touching the disk). Returns a dict carrying the absolute
    ``path``, its ``directory`` and file ``name``, ``bytes`` + a human-readable ``size``,
    the streamed ``sha256``, the container ``format`` (``evt21``/``evt2``/``evt3``/
    ``hdf5``), sensor ``width``/``height``, ``events``, ``duration_s``, an ``available``
    flag, ``facts_from`` (which of the three sources answered), and a ``note`` whenever
    something could not be read.

    Degrades rather than raises: an in-memory recording records the fact that it has no
    file, a missing path records the path and says so, and a file with no decode cache
    records its header facts and notes that the event count would have cost a full decode.
    """
    if rec is None and hasattr(path, "source_path"):      # a Recording answers for itself
        rec, path = path, getattr(path, "source_path", "")
    facts = {"path": None, "directory": None, "name": None, "label": None,
             "bytes": None, "size": None, "sha256": None, "format": None,
             "width": None, "height": None, "events": None, "duration_s": None,
             "available": False, "facts_from": None, "note": None}
    if rec is not None:
        facts.update(label=str(getattr(rec, "name", "") or "") or None,
                     format=getattr(rec, "fmt", None),
                     width=int(rec.width), height=int(rec.height), events=int(rec.n),
                     duration_s=round(float(rec.duration_s), 6),
                     facts_from="loaded recording")
    path = str(path or "")
    if not path:
        facts["label"] = facts["label"] or "(in-memory recording)"
        facts["name"] = facts["label"]
        facts["note"] = ("in-memory recording — it has no file on disk, so no path, size "
                         "or hash can be recorded")
        return facts

    facts["path"] = os.path.abspath(path)
    facts["directory"] = os.path.dirname(facts["path"])
    facts["name"] = os.path.basename(facts["path"])
    facts["label"] = facts["label"] or os.path.splitext(facts["name"])[0]
    if not os.path.exists(ext(facts["path"])):
        facts["note"] = "source file was not present at this path when the export ran"
        return facts
    facts["available"] = True
    try:
        facts["bytes"] = int(os.path.getsize(ext(facts["path"])))
        facts["size"] = human_bytes(facts["bytes"])
    except OSError as e:
        facts["note"] = f"file size unreadable ({e})"
    if sha256:
        try:
            facts["sha256"] = file_sha256(facts["path"])
        except OSError as e:
            facts["note"] = f"SHA-256 not computed ({e})"

    if facts["events"] is None:                    # no loaded recording — probe cheaply
        for probe in (_cache_facts, _header_facts):
            try:
                found = probe(facts["path"])
            except Exception:                      # unreadable / unsupported container
                found = {}
            for key, value in found.items():
                if facts.get(key) is None:
                    facts[key] = value
            if facts["events"] is not None:
                break
    if facts["events"] is None and facts["note"] is None:
        facts["note"] = ("event count and duration not read — this file has no decode "
                         "cache, and reading them would have forced a full decode")
    return facts


def artifact_facts(path, **fields) -> dict:
    """The produced file's identity and size, merged with what the export path reports.

    *fields* carries whatever that path knows — ``frames``, ``fps``, ``codec``,
    ``duration_s``, ``width``, ``height``, ``events``, ``warnings``; values left ``None``
    are simply omitted from the README and the JSON.
    """
    facts = {"name": os.path.basename(str(path)), "path": os.path.abspath(str(path)),
             "bytes": None, "size": None, "sha256": None}
    try:
        facts["bytes"] = int(os.path.getsize(ext(facts["path"])))
        facts["size"] = human_bytes(facts["bytes"])
        facts["sha256"] = file_sha256(facts["path"])
    except OSError:
        pass
    facts.update({k: v for k, v in fields.items() if v is not None})
    return facts


def cell_usage(clip, source, **fields) -> dict:
    """One "how this source was used" row from a placed
    :class:`~gottlux.core.canvas.CanvasClip`.

    *source* is the 1-based index of that clip's recording in the ``sources`` list;
    *fields* adds whatever the calling export path also knows (``name``, ``lane``,
    ``trim_in_s``, ``trim_out_s``, ``block``, ``program_span_s``, ``gap_after_s``, …).
    """
    row = {"source": int(source), "dest_rect": [int(v) for v in clip.rect],
           "roi": None if clip.roi is None else [int(v) for v in clip.roi],
           "t_offset_s": round(float(clip.t_offset_s), 6),
           "time_scale": float(clip.time_scale),
           "accumulation_s": float(clip.accumulation_s),
           "mode": str(clip.mode), "colormap": str(clip.colormap),
           "tonemap": str(clip.tonemap), "gamma": float(clip.gamma),
           "loop": bool(clip.loop)}
    row.update({k: v for k, v in fields.items() if v is not None})
    return row


# ==================================================================== formatting
def _seconds(v) -> str:
    # ``+ 0.0`` normalizes a rounded-down negative zero, so an offset of -1e-9 prints as
    # "0.000 s" rather than the "-0.000 s" a compiled program routinely produces.
    return f"{round(float(v), 3) + 0.0:.3f} s"


def _millis(v) -> str:
    return f"{float(v) * 1e3:g} ms"


def _yes_no(v) -> str:
    return "yes" if v else "no"


def _rect(v) -> str:
    x, y, w, h = (int(c) for c in v)
    return f"x {x}, y {y}, {w} × {h} px"


def _roi(v) -> str:
    if not v:
        return "none — the full sensor"
    x0, y0, x1, y1 = (int(c) for c in v)
    return f"x {x0}–{x1}, y {y0}–{y1} px"


def _span(v) -> str:
    return f"{float(v[0]):.3f}–{float(v[1]):.3f} s"


def _scale(v) -> str:
    """The clock mapping, with what it actually does to the playback spelled out."""
    x = float(v)
    if x == 1:
        how = "real time"
    elif x <= 0:
        how = "stopped"
    elif x < 1:
        how = f"{1 / x:g}× slow motion"
    else:
        how = f"{x:g}× speed-up"
    return f"{x:g} (clip seconds per program second) — {how}"


def _plain(v) -> str:
    return str(v)


#: The usage rows, in README order, as ``(key, label, formatter)``. A row whose key the
#: caller did not supply is left out, so each export path prints exactly what applies to it.
_USAGE_ROWS = (
    ("lane", "Lane", _plain),
    ("block", "Inside canvas block", _plain),
    ("program_span_s", "Program span", _span),
    ("trim_in_s", "Trim in point", _seconds),
    ("trim_out_s", "Trim out point", _seconds),
    ("roi", "Source ROI crop", _roi),
    ("dest_rect", "Destination rect on the canvas", _rect),
    ("t_offset_s", "Time offset", _seconds),
    ("time_scale", "Time scale", _scale),
    ("accumulation_s", "Accumulation (exposure)", _millis),
    ("mode", "Accumulation mode", _plain),
    ("colormap", "Colormap", _plain),
    ("tonemap", "Tone map", _plain),
    ("gamma", "Gamma", _plain),
    ("loop", "Loop", _yes_no),
    ("gap_after_s", "Gap after this item", _seconds),
    ("note", "Note", _plain),
)

#: The artifact rows, in README order — same contract as :data:`_USAGE_ROWS`.
_ARTIFACT_ROWS = (
    ("size", "Size", _plain),
    ("duration_s", "Duration", _seconds),
    ("frames", "Frames written", lambda v: f"{int(v):,}"),
    ("fps", "Frame rate", lambda v: f"{float(v):g} fps"),
    ("codec", "Codec", _plain),
    ("events", "Events", lambda v: f"{int(v):,}"),
    ("canvas", "Canvas geometry", lambda v: f"{int(v[0])} × {int(v[1])} px"),
    ("width", "Encoded geometry", None),              # paired with height, handled below
    ("sha256", "SHA-256", lambda v: f"`{v}`"),
)


def _short(sha) -> str:
    return f"{sha[:SHORT_SHA]}…" if sha else "—"


def _default_role(name) -> str:
    """The files-table role for a file the export wrote but did not describe."""
    lower = name.lower()
    if lower.endswith(".gottlux-canvas.json"):
        return ("canvas composition spec — reload it in the Canvas composer to re-render "
                "this export")
    if lower.endswith(".png"):
        return "still frame written by this export"
    if lower.endswith((".mp4", ".raw", ".h5", ".hdf5")):
        return "written by this export"
    return "written by this export"


def _file_rows(folder, artifact_name, kind, declared) -> list:
    """Every file in *folder* with its role — the artifact, the caller's declarations,
    whatever else the export left there, then the two documents this call writes."""
    rows = [(artifact_name, f"the exported artifact — {kind}")]
    seen = {artifact_name}
    for name, role in declared:
        if name not in seen:
            rows.append((name, role))
            seen.add(name)
    try:
        present = sorted(os.listdir(ext(folder)))
    except OSError:
        present = []
    for name in present:
        if name not in seen and name not in (README_NAME, PROVENANCE_NAME):
            rows.append((name, _default_role(name)))
            seen.add(name)
    rows.append((PROVENANCE_NAME, "machine-readable twin of this document"))
    rows.append((README_NAME, "this document — what was produced, from which sources, "
                              "with which settings"))
    return rows


# ==================================================================== the README
def _readme_artifact(lines, kind, artifact, notes, warnings):
    lines += [f"# GottLUX export — {kind}", "",
              "## What was produced", "",
              f"* **File:** `{artifact.get('name', '')}`",
              f"* **Kind:** {kind}"]
    for key, label, fmt in _ARTIFACT_ROWS:
        value = artifact.get(key)
        if value is None:
            continue
        if key == "width":
            height = artifact.get("height")
            if height is None:
                continue
            lines.append(f"* **{label}:** {int(value)} × {int(height)} px")
            continue
        lines.append(f"* **{label}:** {fmt(value)}")
    for note in notes:
        lines.append(f"* {note}")
    for warning in warnings:
        lines.append(f"* **Note:** {warning}")
    lines.append("")


def _readme_when(lines, created, env):
    lines += ["## When, and with what", "",
              f"* **Exported:** {created} (UTC)",
              f"* **GottLUX version:** {env['gottlux_version']}",
              f"* **Python:** {env['python']}",
              f"* **Platform:** {env['platform']} ({env['machine']})",
              "",
              f"`{PROVENANCE_NAME}` beside this file carries the same facts machine-readably.",
              ""]


def _readme_sources(lines, sources):
    lines += ["## Source recordings", "",
              f"{len(sources)} distinct recording(s) went into this export. Every clip "
              "placed on the timeline or the canvas resolves to one of these rows; the "
              "'How each source was used' section below refers to them by number.", "",
              "| # | File | Directory | Size | SHA-256 | Format | Resolution | Events | "
              "Duration |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i, s in enumerate(sources, 1):
        res = (f"{s['width']} × {s['height']}"
               if s.get("width") and s.get("height") else "—")
        events = f"{s['events']:,}" if s.get("events") is not None else "—"
        dur = f"{s['duration_s']:g} s" if s.get("duration_s") is not None else "—"
        lines.append(
            f"| {i} | `{s.get('name') or '—'}` | `{s.get('directory') or '—'}` | "
            f"{s.get('size') or '—'} | `{_short(s.get('sha256'))}` | "
            f"{s.get('format') or '—'} | {res} | {events} | {dur} |")
    lines += ["", "Full digests and absolute paths:", ""]
    for i, s in enumerate(sources, 1):
        lines.append(f"{i}. `{s.get('path') or s.get('label') or '—'}`")
        lines.append(f"   * SHA-256: `{s.get('sha256') or 'not computed'}`")
        if s.get("facts_from"):
            lines.append(f"   * Geometry/count read from: {s['facts_from']}")
        if s.get("note"):
            lines.append(f"   * Note: {s['note']}")
    lines.append("")


def _readme_usage(lines, usage, sources):
    lines += ["## How each source was used", ""]
    if not usage:
        lines += ["No per-clip placement applies to this export path.", ""]
        return
    for i, row in enumerate(usage, 1):
        index = row.get("source")
        src = sources[index - 1] if isinstance(index, int) and 1 <= index <= len(sources) \
            else {}
        name = row.get("name") or src.get("label") or src.get("name") or "clip"
        lines += [f"### {i}. {name} — source {index}: "
                  f"`{src.get('name') or src.get('label') or '—'}`", ""]
        for key, label, fmt in _USAGE_ROWS:
            if key not in row:
                continue
            value = row[key]
            # ``roi`` renders its own "no crop"; any other missing value prints as a dash
            shown = fmt(value) if (value is not None or key == "roi") else "—"
            lines.append(f"* **{label}:** {shown}")
        lines.append("")


def _readme_settings(lines, settings):
    lines += ["## Export settings", ""]
    if not settings:
        lines += ["No further parameters apply to this export path.", ""]
        return
    lines += ["| Setting | Value |", "| --- | --- |"]
    for key, value in settings.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")


def _readme_texts(lines, texts):
    if not texts:
        return
    lines += ["## Titles and text", "",
              "Text is a **render-time** item: it is drawn into a video export and cannot "
              "be carried by an event (`.raw`) export, which holds events only.", ""]
    for i, txt in enumerate(texts, 1):
        if isinstance(txt, dict):
            body = txt.get("text", "")
            detail = " · ".join(f"{k}: {v}" for k, v in txt.items() if k != "text")
            lines.append(f"{i}. “{body}” — {detail}" if detail else f"{i}. “{body}”")
        else:
            lines.append(f"{i}. {txt}")
    lines.append("")


def _readme_reproduce(lines, reproduce, folder_name):
    lines += ["## Reproducing it", ""]
    spec = (reproduce or {}).get("spec")
    if spec:
        lines += [f"The composition spec is saved in this folder as `{spec}` — it names "
                  "every source, every cell rect, and every per-clip setting, so the "
                  "export is re-renderable without rebuilding it by hand.", ""]
    else:
        lines += ["This export path saves no composition spec; the settings and usage "
                  "sections above are the full recipe.", ""]
    steps = (reproduce or {}).get("steps") or []
    for i, step in enumerate(steps, 1):
        lines.append(f"{i}. {step}")
    if steps:
        lines.append("")
    command = (reproduce or {}).get("command")
    if command:
        lines += ["Equivalent command:", "", "```", command, "```", ""]
    lines += [f"The sources listed above are identified by SHA-256, so a re-render can be "
              f"verified against the exact files this folder (`{folder_name}`) was built "
              "from.", ""]


def _readme_files(lines, rows):
    lines += ["## Files in this folder", "", "| File | Role |", "| --- | --- |"]
    for name, role in rows:
        lines.append(f"| `{name}` | {role} |")
    lines.append("")


def _render_readme(kind, artifact, sources, settings, extra, created, env, files, folder):
    lines: list = []
    _readme_artifact(lines, kind, artifact, extra.get("notes") or [],
                     extra.get("warnings") or [])
    _readme_when(lines, created, env)
    _readme_sources(lines, sources)
    _readme_usage(lines, extra.get("usage") or [], sources)
    _readme_settings(lines, settings)
    _readme_texts(lines, extra.get("texts") or [])
    _readme_reproduce(lines, extra.get("reproduce") or {}, os.path.basename(folder))
    _readme_files(lines, files)
    return "\n".join(lines)


# ==================================================================== the writer
def _environment() -> dict:
    return {"gottlux_version": __version__, "python": sys.version.split()[0],
            "platform": platform.platform(), "machine": platform.machine()}


def write_provenance(folder, kind, artifact, sources, settings=None, extra=None) -> tuple:
    """Write ``README.md`` + ``provenance.json`` into *folder*; returns ``(readme, json)``.

    Parameters
    ----------
    folder : str
        The export folder (:func:`export_folder`), already holding the artifact.
    kind : str
        What was produced, in words — ``'Timeline video (MP4)'``, ``'Canvas .raw'``, ….
    artifact : dict
        :func:`artifact_facts` for the produced file.
    sources : list[dict]
        One :func:`source_facts` dict per **distinct** source recording, in the order the
        README numbers them; each usage row's ``source`` is a 1-based index into this list.
    settings : dict | None
        The full export parameter list as ``{label: value}``, printed verbatim.
    extra : dict | None
        Optional sections — ``usage`` (rows, see :func:`cell_usage`), ``texts`` (title/text
        items, video-only), ``reproduce`` (``{'spec': file name, 'steps': [...],
        'command': str}``), ``files`` (``[(name, role), …]`` for files this export wrote
        beside the artifact), plus ``warnings`` and ``notes`` appended to the artifact
        section.
    """
    extra = dict(extra or {})
    settings = dict(settings or {})
    sources = list(sources or [])
    created, env = _utc_stamp(), _environment()
    files = _file_rows(folder, artifact.get("name", ""), kind, extra.get("files") or [])

    record = {
        "schema_version": SCHEMA_VERSION,
        "gottlux_version": __version__,
        "created_utc": created,
        "kind": kind,
        "artifact": artifact,
        "sources": sources,
        "usage": list(extra.get("usage") or []),
        "settings": settings,
        "texts": list(extra.get("texts") or []),
        "reproduce": dict(extra.get("reproduce") or {}),
        "warnings": list(extra.get("warnings") or []),
        "notes": list(extra.get("notes") or []),
        "environment": env,
        "files": [{"name": name, "role": role} for name, role in files],
    }
    prov_path = os.path.join(folder, PROVENANCE_NAME)
    with open(ext(prov_path), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    readme_path = os.path.join(folder, README_NAME)
    with open(ext(readme_path), "w", encoding="utf-8") as f:
        f.write(_render_readme(kind, artifact, sources, settings, extra, created, env,
                               files, folder))
    return readme_path, prov_path
