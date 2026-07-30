"""
examples.py — discover bundled demo/example recordings to surface on first launch.

The GUI offers one-click demo clips when it opens with nothing loaded. This module finds them
generically: it looks for an examples directory — by default a ``RawExamples/`` (or ``examples/``)
folder beside the project, overridable with the ``GOTTLUX_EXAMPLES`` environment variable — and
lists the event recordings in it. Nothing is hard-coded: any ``.raw``, ``.h5``/``.hdf5``, or
``*.meta.json`` capture dropped into that directory shows up automatically, so the same build
works on any machine.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

#: Directory names searched (in order) under each candidate base. ``examples/data`` is
#: where the repo ships its sample clips; the others cover local data drops.
EXAMPLE_DIRNAMES = (os.path.join("examples", "data"), "RawExamples", "examples", "Examples")

#: Recording file extensions offered as examples (the loader also accepts capture folders).
_EXAMPLE_EXTS = (".raw", ".h5", ".hdf5", ".meta.json")


def _candidate_dirs() -> List[str]:
    """Ordered, de-duplicated list of directories that might hold the example recordings.

    A set ``GOTTLUX_EXAMPLES`` is **authoritative** — it is the sole candidate, so a deliberate
    override never silently falls back to a bundled folder. Unset → search the default locations
    (``RawExamples``/``examples`` beside the project, the package, then the working directory).
    """
    env = os.environ.get("GOTTLUX_EXAMPLES")
    if env:
        return [env]
    pkg = os.path.dirname(os.path.abspath(__file__))      # .../gottlux
    root = os.path.dirname(pkg)                            # project root (parent of the package)
    cands: List[str] = []
    for base in (root, pkg, os.getcwd()):
        for name in EXAMPLE_DIRNAMES:
            cands.append(os.path.join(base, name))
    seen, out = set(), []
    for c in cands:
        a = os.path.abspath(c)
        if a not in seen:
            seen.add(a)
            out.append(c)
    return out


def examples_dir() -> Optional[str]:
    """The first existing candidate examples directory, or ``None`` if none is present."""
    for d in _candidate_dirs():
        if os.path.isdir(d):
            return d
    return None


@dataclass
class Example:
    """One discoverable demo recording (a path plus display metadata)."""
    path: str
    title: str
    detail: str
    is_rotating: bool


def _human_size(nbytes: float) -> str:
    """A compact human-readable file size (e.g. ``74 MB``, ``1.2 GB``)."""
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB", "MB") else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


#: Trailing redundant tokens to drop from a derived title (a size baked into the file name).
_TITLE_SIZE_RE = re.compile(r"[\s_]*\d+(?:\.\d+)?\s*[KMGT]B\s*$", re.IGNORECASE)


def _pretty_title(name: str) -> str:
    """A readable title from a file name: drop the extension, underscores → spaces, and strip a
    trailing size token (it's shown separately) — generic, no clip is special-cased."""
    stem = name[:-len(".meta.json")] if name.endswith(".meta.json") else os.path.splitext(name)[0]
    stem = _TITLE_SIZE_RE.sub("", stem)                   # e.g. "..._249.5MB" → "..."
    title = re.sub(r"\s+", " ", stem.replace("_", " ")).strip()
    return title or name


def list_examples(directory: Optional[str] = None) -> List[Example]:
    """Every example recording in *directory* (default: the discovered :func:`examples_dir`).

    Returns them sorted by name. The rotation flag is a cheap *display* hint from the file name
    (``"rotat"`` present) — the authoritative geometry is determined when the clip is decoded.
    Returns an empty list if no examples directory is found.
    """
    directory = directory or examples_dir()
    if not directory or not os.path.isdir(directory):
        return []
    out: List[Example] = []
    for entry in sorted(os.listdir(directory)):
        low = entry.lower()
        if not low.endswith(_EXAMPLE_EXTS):
            continue
        path = os.path.join(directory, entry)
        try:
            size = os.path.getsize(path)
            detail = _human_size(size)
        except OSError:
            detail = ""
        is_rot = "rotat" in low or "spin" in low
        detail = f"{detail} · {'rotating sensor' if is_rot else 'staring sensor'}" if detail \
            else ("rotating sensor" if is_rot else "staring sensor")
        out.append(Example(path=path, title=_pretty_title(entry), detail=detail, is_rotating=is_rot))
    return out


def has_examples() -> bool:
    """True if at least one bundled example recording can be found."""
    return bool(list_examples())
