"""
tool_export.py — export a gottlux analysis as a standalone Python + MATLAB tool bundle.

The inverse of "use it as a library": take an algorithm **out** of gottlux. For a chosen
tool (see :data:`gottlux.export_tools.TOOLS`) this writes a self-documenting bundle folder

    <input-stem>_tool_<NAME>_<stamp>/
        data.h5           the recording (honoring any --t_start/--t_stop/--roi window),
                          written via :func:`gottlux.io.hdf5.write_hdf5` — the same
                          Metavision-compatible layout --to-hdf5 produces
        run_<NAME>.py     a self-contained Python script (numpy + h5py, scipy where
                          noted; matplotlib optional) — NO gottlux import
        run_<NAME>.m      the MATLAB twin (base MATLAB, native h5read — no toolboxes)
        README.md         full provenance (source path, size, SHA-256, geometry, window),
                          a table of every file accessed/produced, the baked parameters,
                          the generating command line, and run instructions for both scripts
        provenance.json   the same provenance facts, machine-readable

The exporter bakes the *current* CLI values (band, sample rate, accumulation window, ROI,
sensor geometry — and for ``viz_config``, the visualization mode/colormap/tonemap) into
plain variables at the top of each script, so the recipient tunes them with a text editor —
and both scripts run against any GottLUX-exported ``.h5``, not just the bundled ``data.h5``.

CLI:  ``gottlux INPUT --export-tool NAME [--tool-format python|matlab|both] [--tool-out DIR]``
      ``gottlux INPUT --export-tool viz_config [--viz_mode count|polarity] [--viz_cmap NAME]
      [--viz_tonemap CURVE] [--viz_accum_ms MS]``
      ``gottlux --export-tool list``  prints the available tools.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from gottlux import __version__
from gottlux.export_tools import M_LOADER, PY_LOADER, TOOLS, render
from gottlux.export_tools.viz_config import matlab_cmap

#: Visualization-flag domains (shared by the CLI choices and the API validation).
VIZ_MODES = ("count", "polarity")
VIZ_TONEMAPS = ("linear", "sqrt", "gamma", "log", "asinh", "percentile")

#: Baked-parameter keys that correspond 1:1 to a CLI flag — used to reconstruct the
#: generating command line recorded in the README and provenance.json.
_PARAM_FLAGS = {
    "accum_dt": "--accum_dt",
    "fs": "--fft_fs",
    "freq_lo": "--freq_lo",
    "freq_hi": "--freq_hi",
    "viz_mode": "--viz_mode",
    "viz_cmap": "--viz_cmap",
    "viz_tonemap": "--viz_tonemap",
    "viz_accum_ms": "--viz_accum_ms",
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def list_tools_text() -> str:
    """The ``--export-tool list`` listing: every tool + its one-line description."""
    lines = ["Exportable standalone tools (gottlux INPUT --export-tool NAME):"]
    for name in sorted(TOOLS):
        lines.append(f"  {name:17s} {TOOLS[name].description}")
    lines.append("\nEach bundle holds data.h5 + run_<NAME>.py (numpy/scipy/h5py only) + "
                 "run_<NAME>.m (base MATLAB) + a README with full provenance + "
                 "provenance.json (the same facts, machine-readable).")
    return "\n".join(lines)


def _window_bounds(rec, t0, t1):
    """The effective exported time window ``(t_lo, t_hi)`` in seconds."""
    t_lo = rec.t_start_s if t0 is None else max(float(t0), rec.t_start_s)
    t_hi = rec.t_stop_s if t1 is None else min(float(t1), rec.t_stop_s)
    return t_lo, max(t_hi, t_lo)


def _viz_settings(cfg, viz) -> dict:
    """Resolve the visualization flags to concrete baked values.

    Defaults follow the suite's own display conventions: ``count`` mode with the
    ``inferno`` colormap and the ``sqrt`` tone curve (the GUI's faint-structure default);
    ``polarity`` mode switches to a diverging ``coolwarm`` map with a ``linear`` curve so
    zero stays visually centred. The accumulation window defaults to
    :attr:`~gottlux.config.Config.viz_accum_dt` (in ms).
    """
    viz = dict(viz or {})
    mode = str(viz.get("mode") or "count")
    if mode not in VIZ_MODES:
        raise ValueError(f"viz mode must be one of {VIZ_MODES}, not {mode!r}")
    tone = viz.get("tonemap") or ("linear" if mode == "polarity" else "sqrt")
    if tone not in VIZ_TONEMAPS:
        raise ValueError(f"viz tonemap must be one of {VIZ_TONEMAPS}, not {tone!r}")
    cmap = str(viz.get("cmap") or ("coolwarm" if mode == "polarity" else "inferno"))
    accum_ms = viz.get("accum_ms")
    accum_ms = float(accum_ms) if accum_ms is not None else float(cfg.viz_accum_dt) * 1e3
    if accum_ms <= 0:
        raise ValueError(f"viz accum_ms must be positive, not {accum_ms!r}")
    return {"viz_mode": mode, "viz_cmap": cmap, "viz_cmap_m": matlab_cmap(cmap),
            "viz_tonemap": str(tone), "viz_accum_ms": f"{accum_ms:g}", "snap_count": 6}


def _baked_params(tool, rec, cfg, t0, t1, roi, viz=None) -> dict:
    """Resolve every placeholder the tool's templates need from the current run context.

    Values follow the same defaults the suite itself uses (:class:`gottlux.config.Config`);
    the ROI defaults to the full sensor, the rate bin follows the ``overview`` analysis's
    rule (span/600, floored at 5 ms), and the visualization values come from
    :func:`_viz_settings`.
    """
    t_lo, t_hi = _window_bounds(rec, t0, t1)
    span = max(t_hi - t_lo, 1e-3)
    r = roi if roi is not None else (0, 0, rec.width, rec.height)
    vals = {
        "version": __version__,
        "stamp": _utc_stamp(),
        "source": os.path.basename(rec.source_path or "") or rec.name,
        "accum_dt": f"{cfg.accum_dt:g}",
        "max_frames": 200,
        "rate_bin": f"{max(span / 600.0, 0.005):g}",
        "fs": f"{max(cfg.fft_fs, 2.2 * cfg.freq_hi):g}",
        "freq_lo": f"{cfg.freq_lo:g}",
        "freq_hi": f"{cfg.freq_hi:g}",
        "cell": 8,
        "roi_x0": int(r[0]), "roi_y0": int(r[1]), "roi_x1": int(r[2]), "roi_y1": int(r[3]),
        **_viz_settings(cfg, viz),
    }
    # only the keys this tool's manifest declares (+ the universal header keys) are baked
    keys = {k for k, _ in tool.params} | {"version", "stamp", "source"}
    return {k: v for k, v in vals.items() if k in keys}


# ====================================================================== provenance facts
def _source_facts(rec) -> dict:
    """Where the information came from: the source recording's identity and geometry.

    The SHA-256 reuses the run-provenance hashing (:func:`gottlux.io.paths.file_sha256`,
    streamed). An in-memory recording has no file to hash; path/bytes/sha256 are ``None``.
    """
    from gottlux.io.paths import ext, file_sha256
    facts = {"path": None, "bytes": None, "sha256": None,
             "format": rec.fmt, "width": int(rec.width), "height": int(rec.height),
             "name": rec.name,
             "total_events": int(rec.n), "total_duration_s": round(rec.duration_s, 6)}
    if rec.source_path and os.path.exists(ext(rec.source_path)):
        facts["path"] = os.path.abspath(rec.source_path)
        try:
            facts["bytes"] = int(os.path.getsize(ext(rec.source_path)))
            facts["sha256"] = file_sha256(rec.source_path)
        except OSError:
            pass
    return facts


def _window_facts(rec, t0, t1, roi, n_events) -> dict:
    """What slice of the source landed in ``data.h5``: window, ROI, count, duration."""
    t_lo, t_hi = _window_bounds(rec, t0, t1)
    return {"t_start_s": round(t_lo, 6), "t_stop_s": round(t_hi, 6),
            "duration_s": round(t_hi - t_lo, 6),
            "roi": list(int(v) for v in roi) if roi is not None else None,
            "n_events": int(n_events),
            "windowed": not (t0 is None and t1 is None and roi is None)}


def _quote(s: str) -> str:
    """Shell-quote a path for the reproduction command (only when it needs it)."""
    return f'"{s}"' if (" " in s or "\t" in s) else s


def _command_line(tool, source_facts, params, fmt, out_dir, t0, t1, roi) -> str:
    """An equivalent ``gottlux`` command line that regenerates this bundle.

    Reconstructed from the resolved values (the exporter does not see the original argv):
    the input path, the tool/format/output flags, the window flags, and every baked
    parameter that maps 1:1 to a CLI flag (:data:`_PARAM_FLAGS`).
    """
    src = source_facts["path"] or source_facts["name"]
    parts = ["gottlux", _quote(src), "--export-tool", tool.name, "--tool-format", fmt]
    if out_dir:
        parts += ["--tool-out", _quote(os.path.abspath(out_dir))]
    if t0 is not None:
        parts += ["--t_start", f"{float(t0):g}"]
    if t1 is not None:
        parts += ["--t_stop", f"{float(t1):g}"]
    if roi is not None:
        parts += ["--roi", ",".join(str(int(v)) for v in roi)]
    baked = {k for k, _ in tool.params}
    for key, flag in _PARAM_FLAGS.items():
        if key in baked and key in params:
            parts += [flag, str(params[key])]
    return " ".join(parts)


def _file_rows(tool, source_facts, wrote_py, wrote_m) -> list:
    """The files-accessed/produced table: every file the exporter read or wrote, with a
    one-line role. Ordered read-then-written, in bundle-creation order."""
    rows = []
    if source_facts["path"]:
        rows.append((source_facts["path"], "read",
                     "source recording — decoded, windowed, and re-encoded into data.h5"))
    else:
        rows.append((f"(in-memory recording '{source_facts['name']}')", "read",
                     "source events — supplied in memory, windowed into data.h5"))
    rows.append(("data.h5", "written",
                 "the exported events (Metavision-compatible compound CD/events, "
                 "gzip-chunked) — the scripts' default input"))
    if wrote_py:
        rows.append((f"run_{tool.name}.py", "written",
                     "self-contained Python analysis script (numpy/h5py, scipy where "
                     "noted; no gottlux import)"))
    if wrote_m:
        rows.append((f"run_{tool.name}.m", "written",
                     "the MATLAB twin (base MATLAB, native h5read, no toolboxes)"))
    rows.append(("provenance.json", "written",
                 "machine-readable twin of the README's provenance facts"))
    rows.append(("README.md", "written",
                 "this document — provenance, file roles, parameters, run instructions"))
    return rows


def _provenance(tool, source_facts, window, params, files, command, data_h5_facts) -> dict:
    """The machine-readable provenance record written as ``provenance.json``."""
    return {
        "schema": "gottlux.tool_bundle.provenance/1",
        "gottlux_version": __version__,
        "created_utc": _utc_stamp(),
        "tool": {"name": tool.name, "description": tool.description,
                 "ported_from": tool.module},
        "source": source_facts,
        "window": window,
        "bundle": {"data_h5": data_h5_facts},
        "files": [{"name": name, "access": access, "role": role}
                  for name, access, role in files],
        "parameters": {k: params[k] for k, _ in tool.params},
        "command": command,
        "how_to_run": {
            "python": ["install the dependencies: pip install numpy scipy h5py "
                       "(matplotlib optional — adds the plot outputs)",
                       f"run from the bundle folder: python run_{tool.name}.py",
                       f"analyze a different GottLUX-exported .h5: "
                       f"python run_{tool.name}.py path/to/other.h5"],
            "matlab": ["MATLAB R2019a or later, base MATLAB only (native h5read)",
                       f"open the bundle folder in MATLAB, open run_{tool.name}.m, "
                       f"press Run (or: >> run_{tool.name})",
                       f"analyze a different GottLUX-exported .h5: set the DATA_FILE "
                       f"variable at the top of run_{tool.name}.m"],
        },
    }


# ============================================================================ the README
def _readme(tool, params, wrote_py, wrote_m, source_facts, window, files, command,
            data_h5_facts) -> str:
    src = source_facts
    lines = [
        f"# Standalone tool: {tool.name}",
        "",
        f"Exported by GottLUX {__version__} on {_utc_stamp()} (UTC).",
        "",
        f"**What it computes:** {tool.description}.",
        f"The math is ported (simplified, and honestly labeled where so) from `{tool.module}`.",
        "",
        "## Data provenance — where the information came from",
        "",
    ]
    if src["path"]:
        lines += [
            f"* **Source recording:** `{src['path']}`",
            f"* **File size:** {src['bytes']:,} bytes",
            f"* **SHA-256:** `{src['sha256']}`",
        ]
    else:
        lines += [f"* **Source recording:** in-memory recording `{src['name']}` "
                  "(no source file on disk — size/hash not applicable)"]
    roi_note = (f"; ROI {tuple(window['roi'])} px" if window["roi"] else
                "; full sensor (no ROI)")
    lines += [
        f"* **Container format:** {src['format']}",
        f"* **Sensor geometry:** {src['width']} × {src['height']} px",
        f"* **Source stream:** {src['total_events']:,} events over "
        f"{src['total_duration_s']:g} s",
        f"* **Exported window:** t = {window['t_start_s']:g}–{window['t_stop_s']:g} s "
        f"({window['duration_s']:g} s){roi_note}",
        f"* **Events in `data.h5`:** {window['n_events']:,} "
        f"({data_h5_facts['bytes']:,} bytes, SHA-256 `{data_h5_facts['sha256']}`)",
        "",
        "`provenance.json` beside this file carries the same facts machine-readably.",
        "",
        "## Files accessed and produced",
        "",
        "| File | Access | Role |",
        "| --- | --- | --- |",
    ]
    for name, access, role in files:
        lines.append(f"| `{name}` | {access} | {role} |")
    lines += [
        "",
        "## Baked parameters",
        "",
        "The exporter wrote the export-time settings as plain variables at the top of "
        "each script — edit them with a text editor and re-run:",
        "",
    ]
    for key, desc in tool.params:
        lines.append(f"* `{key}` = `{params[key]}` — {desc}")
    lines += [
        "",
        "Generating command line (an equivalent invocation that reproduces this bundle):",
        "",
        "```",
        command,
        "```",
        "",
        "## How to run",
        "",
    ]
    if wrote_py:
        lines += [
            f"### Python script (`run_{tool.name}.py`)",
            "",
            "1. Requirements: Python 3.9+ with **numpy**, **scipy**, and **h5py** "
            "installed (`pip install numpy scipy h5py`). `matplotlib` is optional — "
            "without it the plot outputs are skipped and the data outputs still appear.",
            f"2. From this folder, run: `python run_{tool.name}.py` — it analyzes the "
            "bundled `data.h5`.",
            "3. The outputs listed below appear beside the script.",
            f"4. To analyze a different recording: `python run_{tool.name}.py "
            "path/to/other.h5` — any GottLUX-exported `.h5` works "
            "(`gottlux INPUT --to-hdf5`).",
            "",
        ]
    if wrote_m:
        lines += [
            f"### MATLAB script (`run_{tool.name}.m`) — R2019a or later",
            "",
            "1. Requirements: base MATLAB only (native `h5read`, no toolboxes).",
            f"2. Open this folder in MATLAB, open `run_{tool.name}.m`, and press "
            f"**Run** (or type `run_{tool.name}` at the prompt).",
            "3. Figure window(s) open and the outputs listed below appear in the folder.",
            "4. To analyze a different recording: set the `DATA_FILE` variable at the "
            "top of the script to the path of any GottLUX-exported `.h5`.",
            "",
        ]
    lines += ["## Outputs", ""]
    for fname, what in tool.outputs:
        lines.append(f"* `{fname}` — {what}")
    lines += [
        "",
        "Both scripts read any GottLUX-exported HDF5 event file — the compound "
        "`CD/events` layout and the plain parallel `x/y/p/t` fallback — so they keep "
        "working on future exports (`gottlux INPUT --to-hdf5`).",
        "",
    ]
    return "\n".join(lines)


def export_tool(source, name, out_dir=None, fmt="both", cfg=None,
                t0=None, t1=None, roi=None, viz=None, progress=None) -> dict:
    """Write the standalone bundle for tool *name* from *source* (a path or a Recording).

    Parameters
    ----------
    source : str | Recording
        The recording to bundle (any path :func:`gottlux.load` accepts).
    name : str
        A key of :data:`gottlux.export_tools.TOOLS`.
    out_dir : str | None
        Parent directory for the bundle folder (default: beside the input file).
    fmt : str
        ``"python"`` | ``"matlab"`` | ``"both"`` — which scripts to write (data.h5,
        the README, and provenance.json are always written).
    cfg : Config | None
        Supplies the baked band/fs/accum values (defaults to a fresh Config).
    t0, t1, roi
        Window (s) / ROI (px) applied to ``data.h5`` with
        :func:`~gottlux.io.hdf5.write_hdf5` semantics, and baked into the scripts' ROI.
    viz : dict | None
        Visualization settings for the ``viz_config`` tool: keys ``mode``
        (``count``/``polarity``), ``cmap`` (a matplotlib colormap name), ``tonemap``
        (one of :data:`VIZ_TONEMAPS`), ``accum_ms`` (window per rendered frame, ms).
        Missing keys take the documented defaults (:func:`_viz_settings`).

    Returns a dict: ``path`` (the bundle folder), ``written`` (all file paths), ``n_events``.
    """
    if name not in TOOLS:
        raise KeyError(f"unknown tool {name!r}; available: {sorted(TOOLS)} "
                       "(gottlux --export-tool list)")
    if fmt not in ("python", "matlab", "both"):
        raise ValueError(f"tool format must be python|matlab|both, not {fmt!r}")
    tool = TOOLS[name]

    if isinstance(source, (str, os.PathLike)):
        import gottlux as eb
        rec = eb.load(str(source), progress=progress or (lambda f: None))
    else:
        rec = source
    from gottlux.config import Config
    cfg = cfg or Config()

    stem = os.path.splitext(os.path.basename(rec.source_path or rec.name))[0] or rec.name
    parent = out_dir or (os.path.dirname(os.path.abspath(rec.source_path))
                         if rec.source_path else os.getcwd())
    bundle = os.path.join(parent, f"{stem}_tool_{name}_{_utc_stamp()}")
    os.makedirs(bundle, exist_ok=True)
    written = []

    # 1) the data carrier — same writer, same layout, same window semantics as --to-hdf5
    from gottlux.io import hdf5 as h5io
    data_path = os.path.join(bundle, "data.h5")
    n_events = h5io.write_hdf5(rec, data_path, t0=t0, t1=t1, roi=roi)
    written.append(data_path)

    # 2) the scripts, with the current CLI values baked in as editable variables
    params = _baked_params(tool, rec, cfg, t0, t1, roi, viz)
    wrote_py = fmt in ("python", "both")
    wrote_m = fmt in ("matlab", "both")
    if wrote_py:
        py = render(tool.py_template, {**params, "loader": PY_LOADER})
        py_path = os.path.join(bundle, f"run_{name}.py")
        with open(py_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(py)
        written.append(py_path)
    if wrote_m:
        m = render(tool.m_template, {**params, "loader": M_LOADER})
        m_path = os.path.join(bundle, f"run_{name}.m")
        with open(m_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(m)
        written.append(m_path)

    # 3) the self-documentation: provenance facts (source hash, window, file roles, the
    #    generating command), machine-readable first, then the human README
    from gottlux.io.paths import file_sha256
    source_facts = _source_facts(rec)
    window = _window_facts(rec, t0, t1, roi, n_events)
    data_h5_facts = {"bytes": int(os.path.getsize(data_path)),
                     "sha256": file_sha256(data_path)}
    files = _file_rows(tool, source_facts, wrote_py, wrote_m)
    command = _command_line(tool, source_facts, params, fmt, out_dir, t0, t1, roi)
    prov = _provenance(tool, source_facts, window, params, files, command, data_h5_facts)
    prov_path = os.path.join(bundle, "provenance.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    written.append(prov_path)

    readme = _readme(tool, params, wrote_py, wrote_m, source_facts, window, files,
                     command, data_h5_facts)
    readme_path = os.path.join(bundle, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    written.append(readme_path)
    return {"path": bundle, "written": written, "n_events": int(n_events)}


def run_tool_export_cli(args) -> int:
    """The ``--export-tool`` CLI action (also handles ``--export-tool list``)."""
    if str(args.export_tool).lower() == "list":
        print(list_tools_text())
        return 0
    if args.export_tool not in TOOLS:
        print(f"unknown tool {args.export_tool!r} — available: "
              + ", ".join(sorted(TOOLS)) + "  (gottlux --export-tool list)")
        return 2
    if not args.path:
        print("--export-tool needs an INPUT recording (or use '--export-tool list')")
        return 2
    from gottlux.cli import _config_from_args, _parse_roi
    cfg = _config_from_args(args)
    viz = {"mode": getattr(args, "viz_mode", None),
           "cmap": getattr(args, "viz_cmap", None),
           "tonemap": getattr(args, "viz_tonemap", None),
           "accum_ms": getattr(args, "viz_accum_ms", None)}
    res = export_tool(args.path, args.export_tool, out_dir=args.tool_out,
                      fmt=args.tool_format, cfg=cfg,
                      t0=args.t_start, t1=args.t_stop, roi=_parse_roi(args.roi), viz=viz)
    print(f"tool bundle → {res['path']}")
    for p in res["written"]:
        print(f"  {os.path.basename(p)}")
    print(f"  ({res['n_events']:,} events in data.h5; see the bundle README for how to run)")
    if not args.no_open:
        from gottlux.io.paths import open_in_file_browser
        open_in_file_browser(res["path"])
    return 0
