"""
plugins.py — load user extension modules at startup, without touching the package.

GottLUX's registries (detectors via :func:`gottlux.detectors.register`, analyses via the
:data:`gottlux.run.pipeline._ANALYSES` mapping) are populated by *importing* the modules
that call ``@register``. That means extending gottlux never requires forking it: a user
module that subclasses :class:`~gottlux.detectors.base.Detector` and decorates it with
``@register`` is a fully-fledged detector the moment the module is imported — it shows in
``--list_detectors``, runs via ``--detector``, and gets an auto-built tuning panel in the
Flutter workbench.

This module supplies the missing piece: **who imports the user's module?** The environment
variable ``GOTTLUX_PLUGINS`` holds an :data:`os.pathsep`-separated list of plugin sources —
``.py`` files, or directories whose top-level ``.py`` files are all loaded — and
:func:`load_plugins` imports each one at CLI/GUI startup, *after* the registries exist.
A broken plugin is reported (one line per failure) and skipped; it never takes the program
down, and it never prevents the remaining plugins from loading.

    # Windows                                          # Linux/macOS
    set GOTTLUX_PLUGINS=C:\\lab\\my_detector.py          export GOTTLUX_PLUGINS=~/lab/my_detector.py
    gottlux clip.raw --detector blink                  gottlux clip.raw --detector blink

See ``docs/EXTENDING.md`` for the full guide and ``examples/custom_detector.py`` for a
worked, runnable plugin.
"""
from __future__ import annotations

import importlib.util
import os
import sys

#: The environment variable naming the plugin sources (os.pathsep-separated).
PLUGINS_ENV = "GOTTLUX_PLUGINS"

#: Absolute paths already imported this process — loading is idempotent, so the CLI
#: handing off to the GUI (which calls load_plugins again) does not double-import.
_LOADED: set[str] = set()


def _plugin_files(spec: str) -> list[str]:
    """Expand one GOTTLUX_PLUGINS entry into concrete ``.py`` file paths.

    A ``.py`` file names itself; a directory contributes its top-level ``.py`` files
    (sorted, skipping ``_``-prefixed names — the same convention as a package's private
    modules). Anything else yields nothing (the caller reports it as not found).
    """
    path = os.path.abspath(os.path.expanduser(spec))
    if os.path.isfile(path) and path.lower().endswith(".py"):
        return [path]
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path)
                      if f.lower().endswith(".py") and not f.startswith("_"))
    return []


def _import_file(path: str):
    """Import one plugin file under a stable, collision-free module name."""
    stem = os.path.splitext(os.path.basename(path))[0]
    name = f"gottlux_plugin_{stem}_{abs(hash(path)) & 0xFFFFFF:06x}"
    if name in sys.modules:                          # same file, same process: already in
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {path!r}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                          # registered first, as importlib expects
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)                  # a broken plugin leaves no half-module
        raise
    return mod


def load_plugins(env: str | None = None, report=None) -> list[tuple[str, Exception | None]]:
    """Import every plugin named by ``GOTTLUX_PLUGINS`` (errors reported, never fatal).

    Parameters
    ----------
    env : str | None
        The raw pathsep-separated plugin list; defaults to ``os.environ[GOTTLUX_PLUGINS]``.
    report : callable | None
        Called with one message string per failure (default: print to stderr).

    Returns
    -------
    list of ``(path, error)`` tuples — one per plugin file attempted this call, with
    ``error is None`` on success. Files already loaded earlier in the process are skipped
    (loading is idempotent), so the CLI → GUI handoff imports each plugin exactly once.
    """
    raw = os.environ.get(PLUGINS_ENV, "") if env is None else env
    say = report or (lambda msg: print(f"[gottlux] {msg}", file=sys.stderr))
    results: list[tuple[str, Exception | None]] = []
    for entry in (e.strip() for e in raw.split(os.pathsep)):
        if not entry:
            continue
        files = _plugin_files(entry)
        if not files:
            err = FileNotFoundError(f"no plugin .py file(s) at {entry!r}")
            say(f"plugin skipped: {err}")
            results.append((entry, err))
            continue
        for path in files:
            if path in _LOADED:
                continue
            try:
                _import_file(path)
                _LOADED.add(path)
                results.append((path, None))
            except Exception as e:
                say(f"plugin failed: {path}: {e.__class__.__name__}: {e}")
                results.append((path, e))
    return results
