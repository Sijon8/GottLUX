"""
userscripts.py — run a custom Python file on the recording being analyzed (or on exactly
the portion in view), with the results landing in a small provenance-stamped run folder.

Where :mod:`gottlux.plugins` extends the *registries* (a plugin becomes a detector or an
analysis inside the suite), a **user script** goes the other way: gottlux hands the events
to arbitrary user code and takes care of everything around the call — loading, windowing,
saving, and a README that records exactly what ran on exactly which data.

The contract
------------
A user script is any ``.py`` file defining::

    def process(win, ctx): ...

* ``win`` is a :class:`gottlux.io.recording.EventWindow` — the events already sliced to
  the requested time window and ROI, with the usual fields ``x, y, p, t`` (``t`` in µs,
  zero-based to the parent recording), ``width``/``height``, plus two extras attached for
  scripts: ``win.rec`` (the parent :class:`~gottlux.io.recording.Recording`) and
  ``win.roi`` (the applied ``(x0, y0, x1, y1)`` tuple, or ``None`` for the full frame).
* ``ctx`` is a plain dict: ``{"rec": Recording, "t0": float, "t1": float,
  "roi": tuple | None, "source_path": str, "output_dir": str, "args": list[str]}``.
  ``t0``/``t1`` are the *resolved* window bounds in seconds (the full span when no window
  was requested); ``output_dir`` is the run folder, already created, where the script may
  write files of its own; ``args`` carries the CLI's ``--script-args`` tokens.

The return value decides what gets saved (all handling is in :func:`run_script`):

* ``None``                       → nothing (the script wrote its own outputs, or only printed);
* a dict of name → array/scalar  → ``results.npz`` + a printed per-entry summary;
* a matplotlib ``Figure``        → ``figure.png`` + ``figure.pdf`` (300 DPI);
* a dict ``{"events": (x, y, p, t)}`` with ``t`` in µs → a derived ``derived.raw``
  (EVT2.1, via :mod:`gottlux.io.writer`); any *other* keys in the same dict are still
  saved to ``results.npz``.

Scripts may import anything installed — gottlux included. Each :func:`load_script` call
re-imports the file fresh (under a collision-free module name), so an edit-and-rerun loop
always executes the current code.

Every run folder gets a ``README.md`` recording the script path + SHA-256, the source
recording path + SHA-256, the window/ROI, the gottlux version, the timestamp, the wall
time, and what each output file is — the same traceability promise the analysis run
folders make (:mod:`gottlux.run.provenance`), at user-script scale.

CLI:  ``gottlux INPUT --run-script my_script.py [--t_start/--t_stop/--roi] [--script-args "..."]``
See ``examples/user_script_example.py`` for a worked, commented script.
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

from gottlux import __version__
from gottlux.io.paths import file_sha256


class UserScriptError(RuntimeError):
    """A user script could not be loaded, ran into an error, or returned an unsupported
    value. Raised instead of letting the script's failure crash the caller; the original
    exception (when there is one) rides along as ``__cause__``."""


#: Monotonic counter making every loaded script module name unique within the process —
#: re-running an edited script must execute the *edited* code, never a cached module.
_SEQ = itertools.count()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


# ====================================================================================
# Loading
# ====================================================================================
def load_script(path: str):
    """Import the user script at *path* and return the module (fresh on every call).

    The file is imported under a collision-free generated name (never shadowing an
    installed package, never colliding with another script of the same filename), and a
    broken script leaves no half-imported module behind. The returned module is verified
    to define a callable ``process``.

    Raises :class:`UserScriptError` — with the underlying cause attached — when the file
    is missing, is not a ``.py`` file, fails to import, or defines no ``process(win, ctx)``.
    """
    p = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(p):
        raise UserScriptError(f"user script not found: {path!r}")
    if not p.lower().endswith(".py"):
        raise UserScriptError(f"a user script must be a .py file, got: {path!r}")
    stem = os.path.splitext(os.path.basename(p))[0]
    name = f"gottlux_userscript_{stem}_{next(_SEQ)}"
    # Compile the CURRENT source bytes directly — never through the __pycache__ .pyc
    # machinery, whose mtime+size validation can serve a stale module after a same-size
    # edit within one mtime tick (exactly the edit-and-rerun loop scripts live in).
    spec = importlib.util.spec_from_loader(name, loader=None, origin=p)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = p
    sys.modules[name] = mod                          # registered first, as importlib expects
    try:
        with open(p, "rb") as f:
            code = compile(f.read(), p, "exec")
        exec(code, mod.__dict__)
    except BaseException as e:
        sys.modules.pop(name, None)                  # a broken script leaves no half-module
        raise UserScriptError(
            f"{os.path.basename(p)} failed to import: {e.__class__.__name__}: {e}") from e
    fn = getattr(mod, "process", None)
    if not callable(fn):
        raise UserScriptError(
            f"{os.path.basename(p)} defines no callable process(win, ctx) — "
            "see the gottlux.userscripts module docstring for the contract")
    return mod


# ====================================================================================
# Result handling (one small saver per contract branch)
# ====================================================================================
def _summarize_value(v) -> str:
    """One human-readable token for an NPZ entry: shape+dtype for arrays, ``= v`` for scalars."""
    a = np.asarray(v)
    return f"= {a.item()}" if a.ndim == 0 else f"{a.shape} {a.dtype}"


def _save_arrays(result: dict, folder: str) -> tuple[str, str]:
    """Save a name → array/scalar dict as ``results.npz``; return ``(filename, summary)``."""
    arrays = {}
    for k, v in result.items():
        try:
            a = np.asarray(v)
        except Exception as e:
            raise UserScriptError(
                f"result entry {k!r} is not convertible to a NumPy array: {e}") from e
        if a.dtype == object:
            raise UserScriptError(
                f"result entry {k!r} is not a numeric array/scalar (dtype=object); "
                "the dict-of-arrays contract covers NumPy arrays and scalars")
        arrays[str(k)] = a
    fname = "results.npz"
    np.savez_compressed(os.path.join(folder, fname), **arrays)
    summary = " · ".join(f"{k} {_summarize_value(v)}" for k, v in arrays.items())
    return fname, summary


def _save_events(events, rec, folder: str) -> tuple[str, str]:
    """Save an ``(x, y, p, t)`` tuple (``t`` in µs) as a derived EVT2.1 ``derived.raw``."""
    try:
        x, y, p, t = events
    except (TypeError, ValueError) as e:
        raise UserScriptError(
            "the 'events' entry must be a 4-tuple (x, y, p, t) of equal-length arrays") from e
    x = np.asarray(x); y = np.asarray(y)
    p = np.asarray(p); t = np.asarray(t)
    if not (len(x) == len(y) == len(p) == len(t)):
        raise UserScriptError(
            f"the 'events' arrays disagree in length: x={len(x)} y={len(y)} "
            f"p={len(p)} t={len(t)}")
    from gottlux.io import writer
    fname = "derived.raw"
    n = writer.write_raw(os.path.join(folder, fname), x, y, p, t.astype(np.int64),
                         width=rec.width, height=rec.height)
    return fname, f"derived event stream ({n:,} events, EVT2.1, {rec.width}x{rec.height} px)"


def _save_figure(fig, folder: str) -> list[tuple[str, str]]:
    """Save a matplotlib figure as PNG + PDF; return ``[(filename, description), ...]``."""
    from gottlux.io.export import save_figure
    written = save_figure(fig, os.path.join(folder, "figure"), close=True)
    return [(os.path.basename(pth), "figure returned by process() (300 DPI)")
            for pth in written]


def _handle_result(result, rec, folder: str) -> tuple[str, list[tuple[str, str]]]:
    """Dispatch the script's return value per the contract.

    Returns ``(kind, outputs)`` where *kind* names the branch taken (``"none"`` /
    ``"arrays"`` / ``"figure"`` / ``"events"`` / ``"events+arrays"``) and *outputs* lists
    ``(filename, description)`` pairs for the README. An unsupported type raises
    :class:`UserScriptError` — a typo'd return should be loud, not silently dropped.
    """
    if result is None:
        return "none", []
    try:
        from matplotlib.figure import Figure
    except Exception:                                # matplotlib absent: no figure branch
        Figure = ()
    if Figure and isinstance(result, Figure):
        return "figure", _save_figure(result, folder)
    if isinstance(result, dict):
        outputs, kinds = [], []
        rest = dict(result)
        if "events" in rest:
            outputs.append(_save_events(rest.pop("events"), rec, folder))
            kinds.append("events")
        if rest:
            outputs.append(_save_arrays(rest, folder))
            kinds.append("arrays")
        return "+".join(kinds) or "arrays", outputs
    raise UserScriptError(
        f"process() returned an unsupported {type(result).__name__}; supported returns: "
        "None, a dict of name -> array/scalar, a matplotlib Figure, or a dict "
        "{'events': (x, y, p, t)}")


# ====================================================================================
# Provenance
# ====================================================================================
def _sha256_or_note(path: str) -> str:
    """SHA-256 of *path*, or a plain-language note when there is no file to hash."""
    if path and os.path.exists(path):
        try:
            return file_sha256(path)
        except Exception as e:
            return f"(unhashable: {e.__class__.__name__})"
    return "(no source file — in-memory recording)"


def _write_readme(folder, script_path, rec, t_lo, t_hi, roi, windowed, args,
                  outputs, kind, wall_s, n_view) -> str:
    """Write the run folder's ``README.md`` — the full provenance record — and return its path."""
    roi_txt = ",".join(str(int(v)) for v in roi) if roi is not None else "full frame"
    win_txt = f"[{t_lo:g}, {t_hi:g}] s" + ("" if windowed else " (full span)")
    src = os.path.abspath(rec.source_path) if rec.source_path else "(in-memory recording)"
    lines = [
        "# GottLUX user-script run",
        "",
        f"`{os.path.basename(script_path)}` executed by GottLUX {__version__} "
        f"on {_utc_stamp()}.",
        "",
        "## Provenance",
        "",
        f"* script            : `{script_path}`",
        f"* script SHA-256    : `{_sha256_or_note(script_path)}`",
        f"* recording         : `{src}` — {rec.name}, {rec.n:,} events, "
        f"{rec.width}x{rec.height} px, {rec.duration_s:.3f} s",
        f"* recording SHA-256 : `{_sha256_or_note(rec.source_path)}`",
        f"* window            : {win_txt}",
        f"* roi               : {roi_txt}",
        f"* events in view    : {n_view:,}",
        f"* script args       : {args if args else '(none)'}",
        f"* gottlux version   : {__version__}",
        f"* wall time         : {wall_s:.2f} s",
        f"* result kind       : {kind}",
        "",
        "## Outputs",
        "",
    ]
    if outputs:
        lines += [f"* `{fname}` — {desc}" for fname, desc in outputs]
    else:
        lines.append("* (no return value was saved)")
    lines += [
        "",
        "Files present in this folder but not listed above were written by the script "
        "itself via `ctx[\"output_dir\"]`.",
        "",
    ]
    readme = os.path.join(folder, "README.md")
    with open(readme, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return readme


# ====================================================================================
# The runner
# ====================================================================================
def run_script(path, rec, t0=None, t1=None, roi=None, output_dir=None,
               script_args=None) -> dict:
    """Run the user script at *path* on *rec* (windowed/ROI'd) and save what it returns.

    Parameters
    ----------
    path : str
        The user script — a ``.py`` file defining ``process(win, ctx)``.
    rec : Recording | str
        The recording to analyze (a path is loaded via :func:`gottlux.load`).
    t0, t1 : float | None
        Time window in seconds (``None`` = the full span) — the same semantics as
        :meth:`Recording.window`, so ``process`` sees only ``[t0, t1)``.
    roi : tuple | None
        ``(x0, y0, x1, y1)`` pixel rectangle (inclusive-exclusive), or ``None``.
    output_dir : str | None
        Parent directory for the stamped run folder (default: beside the recording's
        source file, or the current directory for an in-memory recording).
    script_args : list[str] | None
        Extra tokens handed to the script as ``ctx["args"]`` (the CLI's ``--script-args``).

    Returns a dict: ``folder`` (the run folder), ``outputs`` (``(filename, description)``
    pairs), ``result_kind`` (which contract branch the return value took), ``wall_s``
    (the script's wall time), and ``n_events`` (events the script saw).

    Failures anywhere in the script — import, ``process()``, an unsupported return —
    surface as one :class:`UserScriptError` with the cause attached; the caller's own
    state is never corrupted by a broken script.
    """
    if isinstance(rec, (str, os.PathLike)):
        import gottlux as eb
        rec = eb.load(str(rec), progress=lambda f: None)
    mod = load_script(path)
    script_path = os.path.abspath(os.path.expanduser(str(path)))

    # The window the script sees: exactly the requested slice, with the back-references
    # the contract promises (win.rec, win.roi) attached.
    roi_t = tuple(int(v) for v in roi) if roi is not None else None
    win = rec.window(t0, t1, roi=roi_t)
    win.rec = rec
    win.roi = roi_t
    t_lo = rec.t_start_s if t0 is None else float(t0)
    t_hi = rec.t_stop_s if t1 is None else float(t1)

    # The run folder exists before the script runs — ctx["output_dir"] must be writable.
    stem = os.path.splitext(os.path.basename(script_path))[0]
    parent = output_dir or (os.path.dirname(os.path.abspath(rec.source_path))
                            if rec.source_path else os.getcwd())
    base_name = f"gottlux_script_{stem}_{_utc_stamp()}"
    folder = os.path.join(parent, base_name)
    k = 2
    while os.path.exists(folder):                    # same script, same second: still unique
        folder = os.path.join(parent, f"{base_name}_{k}")
        k += 1
    os.makedirs(folder)

    args = list(script_args or [])
    ctx = {"rec": rec, "t0": t_lo, "t1": t_hi, "roi": roi_t,
           "source_path": rec.source_path or "", "output_dir": folder, "args": args}

    start = time.perf_counter()
    try:
        result = mod.process(win, ctx)
    except Exception as e:
        raise UserScriptError(
            f"{os.path.basename(script_path)} raised {e.__class__.__name__}: {e}") from e
    wall_s = time.perf_counter() - start

    kind, outputs = _handle_result(result, rec, folder)
    windowed = not (t0 is None and t1 is None)
    _write_readme(folder, script_path, rec, t_lo, t_hi, roi_t, windowed, args,
                  outputs, kind, wall_s, win.n)

    # The wall-time report (mirrors the other headless actions' printed digests).
    roi_note = "" if roi_t is None else f", ROI {','.join(str(v) for v in roi_t)}"
    print(f"user script {os.path.basename(script_path)} -> {folder}")
    print(f"  {win.n:,} events in view ([{t_lo:g}, {t_hi:g}] s{roi_note}) "
          f"- {wall_s:.2f} s wall")
    for fname, desc in outputs:
        print(f"  {fname}: {desc}")
    if not outputs:
        print("  (no return value; any outputs were written by the script itself)")

    return {"folder": folder, "outputs": outputs, "result_kind": kind,
            "wall_s": wall_s, "n_events": win.n}
