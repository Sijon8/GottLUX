r"""
paths.py — filesystem helpers, with Windows long-path safety.

Deeply-nested cloud-synced data paths routinely exceed Windows' legacy 260-character
``MAX_PATH`` limit. Every file access in :mod:`gottlux.io` therefore goes through
:func:`ext`, which adds the extended-length ``\\?\`` prefix so ``open()`` / ``stat()``
keep working past 260 chars. Caches and outputs auto-relocate to a short platform path
when the preferred location would overflow.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

#: Usable Windows MAX_PATH (260 minus the NUL terminator), kept with headroom for files
#: created *inside* a chosen directory.
MAX_PATH = 259


def ext(path: str) -> str:
    r"""Return a Windows extended-length (``\\?\``) path so I/O works past ``MAX_PATH``.

    No-op on non-Windows, on empty input, and on already-prefixed paths. UNC shares
    (``\\server\share``) are rewritten to ``\\?\UNC\server\share``.
    """
    if os.name != "nt" or not path:
        return path
    p = os.path.abspath(path)
    if p.startswith("\\\\?\\"):
        return p
    if p.startswith("\\\\"):                      # UNC \\server\share
        return "\\\\?\\UNC\\" + p[2:]
    return "\\\\?\\" + p


def open_in_file_browser(path: str) -> bool:
    """Open *path* in the OS file browser (best-effort; never raises)."""
    try:
        path = os.path.abspath(path)
        if os.name == "nt":
            os.startfile(path)                    # noqa: S606  (intended)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return True
    except Exception:
        return False


def safe_name(name, limit=48) -> str:
    """A filesystem-safe slug from *name* (alphanumerics / ``-`` / ``_`` kept)."""
    s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name)).strip("_")
    return s[:limit] or "export"


def unique_export_dir(parent, name, purpose=None, stamp=None) -> str:
    """Create and return a uniquely + helpfully named subfolder inside *parent*.

    ``<parent>/<name>[_<purpose>]_<UTC-stamp>/`` — so every export lands in its own clearly
    labelled folder in the chosen save location, never overwriting a previous one. A numeric
    suffix is appended in the (rare) event of a same-second collision.
    """
    from datetime import datetime, timezone
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    parts = [safe_name(name)] + ([safe_name(purpose, 16)] if purpose else []) + [stamp]
    base = os.path.join(parent, "_".join(parts))
    d, i = base, 1
    while os.path.exists(d):
        d = f"{base}_{i}"; i += 1
    os.makedirs(ext(d), exist_ok=True)
    return d


def analysis_subdir(near_path, name) -> str:
    """A stable, clearly-named results subfolder for ONE analysis, grouped under a single
    ``gottlux_results/`` root beside the data.

    The convention for ad-hoc / interactive analyses: ``<data dir>/gottlux_results/<name>/`` — so
    every distinct analysis lands in its own labelled folder, all under one root that is easy to
    find, archive, or ``.gitignore`` (re-running an analysis overwrites its folder in place rather
    than littering timestamped copies). Use :func:`unique_export_dir` instead when you want each
    run preserved separately.
    """
    p = ext(near_path)
    base = near_path if os.path.isdir(p) else os.path.dirname(os.path.abspath(near_path))
    d = os.path.join(base or ".", "gottlux_results", safe_name(name))
    os.makedirs(ext(d), exist_ok=True)
    return d


def platform_root() -> str:
    r"""A short, stable, user-writable fallback root for relocated caches / outputs.

    Windows: ``%LOCALAPPDATA%\gottlux`` (falling back to ``~\.gottlux`` if
    ``LOCALAPPDATA`` is unset). POSIX: the XDG cache home (``$XDG_CACHE_HOME`` or
    ``~/.cache``) + ``/gottlux``. Never inside the package tree, which may be a
    read-only site-packages install.
    """
    if os.name == "nt":
        return _windows_cache_root()
    return _posix_cache_root()


def _windows_cache_root() -> str:
    r"""``%LOCALAPPDATA%\gottlux`` (default ``~\.gottlux`` when ``LOCALAPPDATA`` is unset)."""
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return os.path.join(base, "gottlux")
    return os.path.join(os.path.expanduser("~"), ".gottlux")


def _posix_cache_root() -> str:
    """``$XDG_CACHE_HOME``/gottlux (default ``~/.cache/gottlux``)."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "gottlux")


def short_id(path: str) -> str:
    """A short, stable, collision-resistant id for a capture directory."""
    base = os.path.basename(os.path.normpath(path))[:24]
    h = hashlib.sha1(os.path.abspath(path).encode("utf-8", "replace")).hexdigest()[:6]
    return f"{base}_{h}"


def fits(path: str, headroom: int = 0, windows: "bool | None" = None) -> bool:
    """True if the absolute *path* plus *headroom* characters stays within ``MAX_PATH``.

    The 259-char limit is a Windows-only concern, so on other platforms this is always
    True. *windows* (default: ``os.name == 'nt'``) exists so the check itself is testable
    everywhere.
    """
    if windows is None:
        windows = os.name == "nt"
    if not windows:
        return True
    return len(os.path.abspath(path)) + headroom <= MAX_PATH


def file_sha256(path: str, block: int = 1 << 20) -> str:
    """Streamed SHA-256 of a file's bytes (for provenance / cache validation)."""
    h = hashlib.sha256()
    with open(ext(path), "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()
