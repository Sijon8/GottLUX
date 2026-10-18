"""
export.py — write a rendered shot as GIF, MP4, WebM or a PNG sequence, with its recipe.

GIF: one adaptive palette for the whole clip (measured on sample frames, so colours do not
shimmer between frames), every frame mapped to it, ``loop=0``. GIF stores frame delays in
centiseconds, so the delay is rounded to 10 ms and the effective rate reported (25 fps is
exact; 30 fps plays at 33.3 fps).

MP4: H.264 ``yuv420p`` with ``+faststart`` (plays inline on the web and in chat apps);
WebM: VP9. Both need even frame sizes, which the renderer guarantees.

Every export writes ``<file>.shot.json`` beside it: the full :class:`Shot` plus the frame
count, size and frame rate actually written, so the clip can be re-rendered or tweaked.
"""
from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np

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


def _gif(renderer, path, progress, cancel):
    from PIL import Image
    n = len(renderer)
    # palette from up to 8 frames spread over the clip
    picks = np.unique(np.linspace(0, n - 1, min(8, n)).astype(int))
    sample = [renderer.frame(int(i)) for i in picks]
    h, w = sample[0].shape[:2]
    mosaic = Image.fromarray(np.concatenate(sample, axis=0))
    colors = int(np.clip(renderer.shot.gif_colors, 8, 256))
    pal = mosaic.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    frames = []
    for i in range(n):
        if cancel and cancel():
            return None
        im = Image.fromarray(renderer.frame(i)).quantize(palette=pal, dither=Image.Dither.NONE)
        frames.append(im)
        if progress:
            progress((i + 1) / n)
    delay = max(20, int(round(100.0 / renderer.shot.fps)) * 10)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=delay, loop=0,
                   disposal=1, optimize=False)
    return {"fps_effective": 1000.0 / delay, "frame_ms": delay}


def _video(renderer, path, fmt, progress, cancel):
    import imageio.v2 as imageio
    q = int(np.clip(renderer.shot.quality, 1, 10))
    if fmt == "mp4":
        kw = dict(codec="libx264", quality=q, pixelformat="yuv420p",
                  ffmpeg_params=["-movflags", "+faststart"])
    else:
        crf = int(np.interp(q, [1, 10], [50, 18]))
        kw = dict(codec="libvpx-vp9", pixelformat="yuv420p",
                  ffmpeg_params=["-b:v", "0", "-crf", str(crf)])
    writer = imageio.get_writer(path, fps=renderer.shot.fps, macro_block_size=2, **kw)
    n = len(renderer)
    try:
        for i in range(n):
            if cancel and cancel():
                return None
            writer.append_data(renderer.frame(i))
            if progress:
                progress((i + 1) / n)
    finally:
        writer.close()
    return {"fps_effective": float(renderer.shot.fps)}


def _pngs(renderer, folder, progress, cancel):
    from PIL import Image
    os.makedirs(folder, exist_ok=True)
    n = len(renderer)
    for i in range(n):
        if cancel and cancel():
            return None
        Image.fromarray(renderer.frame(i)).save(os.path.join(folder, f"frame_{i:05d}.png"))
        if progress:
            progress((i + 1) / n)
    return {"fps_effective": float(renderer.shot.fps)}


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
    if fmt == "gif":
        extra = _gif(r, path, progress, cancel)
    elif fmt in ("mp4", "webm"):
        extra = _video(r, path, fmt, progress, cancel)
    else:
        extra = _pngs(r, path, progress, cancel)
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
