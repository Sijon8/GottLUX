"""
export.py — write a rendered shot as GIF, MP4, WebM or a PNG sequence, with its recipe.

Encoding goes through :mod:`gottlux.viz.encode`, shared with the Canvas composer and the
Timeline, so a format means the same everywhere. GIF: one adaptive palette for the whole
clip (measured on sample frames, so colours do not shimmer between frames), every frame
mapped to it, ``loop=0``. GIF stores frame delays in centiseconds, so the delay is rounded
to 10 ms and the effective rate reported (25 fps is exact; 30 fps plays at 33.3 fps).

MP4: H.264 ``yuv420p`` with ``+faststart`` (plays inline on the web and in chat apps);
WebM: VP9. Both need even frame sizes, which the renderer guarantees. ``quality`` 1 … 10
maps to H.264 crf 34 … 16 and VP9 crf 50 … 18.

Every export writes ``<file>.shot.json`` beside it: the full :class:`Shot` plus the frame
count, size and frame rate actually written, so the clip can be re-rendered or tweaked.
"""
from __future__ import annotations

import datetime as _dt
import json
import os

from gottlux.studio.render import ShotRenderer
from gottlux.studio.shot import Shot

FORMATS = ("gif", "mp4", "webm", "png")


def default_name(rec, shot: Shot, renderer: ShotRenderer | None = None) -> str:
    stem = getattr(rec, "name", "") or "recording"
    t0, t1 = (renderer.t0, renderer.t1) if renderer else (shot.t0 or 0.0, shot.t1 or 0.0)
    fmt = shot.format if shot.format in FORMATS else "gif"
    ext = "" if fmt == "png" else "." + fmt
    return f"{stem}_{shot.look}_{t0:.2f}-{t1:.2f}s{ext}"


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{root}_{i}{ext}"):
        i += 1
    return f"{root}_{i}{ext}"


def default_folder() -> str:
    """``~/Videos/GottLux Studio`` (created on first use)."""
    home = os.path.expanduser("~")
    base = os.path.join(home, "Videos") if os.path.isdir(os.path.join(home, "Videos")) else home
    return os.path.join(base, "GottLux Studio")


def _encode(renderer, path, fmt, progress, cancel):
    """Write the shot through the one encoder every GottLUX export uses
    (:func:`gottlux.viz.encode.encode_frames`): the same codecs, quality scale and GIF
    palette rule as the Canvas composer and the Timeline. ``None`` if cancelled."""
    from gottlux.viz import encode
    shot = renderer.shot
    facts = encode.encode_frames(renderer.frame, len(renderer), path, fmt=fmt, fps=shot.fps,
                                 quality=shot.quality, gif_colors=shot.gif_colors,
                                 progress=progress, cancel=cancel)
    if facts is None:
        return None
    extra = {"fps_effective": float(facts["fps_effective"])}
    if fmt == "gif":
        extra["frame_ms"] = int(facts["frame_ms"])
    return extra


def export_shot(rec, shot: Shot, path: str | None = None, *, renderer=None, progress=None,
                cancel=None, folder: str | None = None) -> dict | None:
    """Render *shot* and write it. Returns a facts dict (``path``, ``format``, ``frames``,
    ``size``, ``fps``, ``duration_s``, ``recipe``) or ``None`` if *cancel* fired.

    The format comes from *path*'s extension, else ``shot.format``. Without *path* the file
    is named after the recording, look and span inside *folder* (default
    :func:`default_folder`) and never overwrites an existing file.
    """
    r = renderer or ShotRenderer(rec, shot)
    if path:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        fmt = ext if ext in FORMATS else ("png" if not ext else shot.format)
    else:
        fmt = shot.format if shot.format in FORMATS else "gif"
        folder = folder or default_folder()
        os.makedirs(folder, exist_ok=True)
        path = unique_path(os.path.join(folder, default_name(rec, shot, r)))
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    extra = _encode(r, path, fmt, progress, cancel)
    if extra is None:
        if os.path.isfile(path):
            os.remove(path)
        return None
    w, h = r.size
    facts = {"path": os.path.abspath(path), "format": fmt, "frames": len(r), "size": [w, h],
             "fps": float(shot.fps), "duration_s": len(r) / float(extra["fps_effective"]),
             "span_s": [r.t0, r.t1], **extra}
    if "f0" in r.st:
        facts["f0_hz"] = float(r.st["f0"])
    from gottlux.run.export_provenance import partial_note
    note = partial_note(rec)                  # a sampled / preview recording: say so
    facts["source_decode"] = "partial" if note else "full"
    if note:
        facts["source_note"] = note
    recipe = os.path.abspath(path) + ".shot.json"
    from gottlux import __version__
    # the recipe sits beside its file, so it names the file, not where this machine keeps it
    written = {k: v for k, v in facts.items() if k != "path"}
    written["file"] = os.path.basename(facts["path"])
    with open(recipe, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"gottlux_version": __version__,
                   "rendered_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                   "source": shot.source or os.path.basename(getattr(rec, "source_path", "") or ""),
                   "shot": shot.to_dict(), "output": written}, f, indent=2)
    facts["recipe"] = recipe
    return facts
