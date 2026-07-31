"""
video.py — render an event recording (with optional box overlays) to an analysis video.

A small, reusable layer between the accumulators/tone-mapping and an MP4: colourize a frame,
draw a tracked box and a text readout on it, and mux a sequence to disk. Used by the Range
lab's "analysis video" (the tracked box + range-vs-time over the clip) and available to any
view that wants to save a clip of what it shows.

The frame *rendering* (colourize / draw_box / draw_label) is pure NumPy (+ Pillow for text)
and is unit-tested; only :func:`write_video` needs the optional ``imageio``/``imageio-ffmpeg``
muxer, and it fails soft (returns ``None``) so a missing codec never crashes an export.
"""
from __future__ import annotations

import numpy as np

from gottlux.core import tonemap
from gottlux.core.accumulate import accumulate_frame


def colorize(frame2d, cmap="inferno", expr="sqrt") -> np.ndarray:
    """Tone-map a 2-D event frame and apply a colormap → an ``(H, W, 3)`` uint8 RGB image."""
    disp, _ = tonemap.compress(np.asarray(frame2d, float), expr=expr)
    try:
        import matplotlib
        rgb = matplotlib.colormaps[cmap](np.clip(disp, 0, 1))[..., :3]
    except Exception:                              # grayscale fallback if matplotlib/cmap missing
        rgb = np.repeat(np.clip(disp, 0, 1)[..., None], 3, axis=2)
    return (rgb * 255).astype(np.uint8)


def apply_colormap(disp01, cmap="inferno") -> np.ndarray:
    """Map an already-normalized ``[0, 1]`` array through *cmap* → ``(H, W, 3)`` uint8 RGB.

    Unlike :func:`colorize` (which tone-maps raw counts), this takes a value a view has *already*
    mapped (so a faithful capture reuses the view's exact dynamic-range settings).
    """
    disp01 = np.clip(np.asarray(disp01, float), 0.0, 1.0)
    try:
        import matplotlib
        rgb = matplotlib.colormaps[cmap](disp01)[..., :3]
    except Exception:
        rgb = np.repeat(disp01[..., None], 3, axis=2)
    return (rgb * 255).astype(np.uint8)


def disp_to_rgb(disp, levels=(0.0, 1.0), cmap="inferno") -> np.ndarray:
    """Colourize a tone-mapped array given its ``(lo, hi)`` display levels (pairs with
    :func:`gottlux.core.render.render_frame`)."""
    lo, hi = levels
    norm = (np.asarray(disp, float) - lo) / (hi - lo + 1e-12)
    return apply_colormap(norm, cmap)


def resize_rgb(rgb, size, smooth=False) -> np.ndarray:
    """Resize an RGB image to ``size`` = ``(width, height)``. Nearest by default (crisp events)."""
    if size is None:
        return np.asarray(rgb, np.uint8)
    try:
        from PIL import Image
        resample = Image.BILINEAR if smooth else Image.NEAREST
        return np.asarray(Image.fromarray(np.asarray(rgb, np.uint8)).resize(
            (int(size[0]), int(size[1])), resample))
    except Exception:
        return np.asarray(rgb, np.uint8)


#: Distinct event colors for overlaying multiple clips (clip 0, 1, 2, …).
OVERLAY_COLORS = [(0, 229, 255), (255, 70, 70), (90, 255, 140), (255, 90, 255), (255, 200, 0)]


def _resize2d(f, hw):
    """Nearest-resize a 2-D array to ``(h, w)``."""
    h, w = hw
    if f.shape[:2] == (h, w):
        return f
    try:
        from PIL import Image
        u8 = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        return np.asarray(Image.fromarray(u8).resize((w, h), Image.NEAREST), float) / 255.0
    except Exception:
        return f


def overlay_frames(frames, colors=None, size=None) -> np.ndarray:
    """Blend several normalized ``[0, 1]`` intensity frames — each tinted a **distinct color** —
    into one RGB image (additive). This is how the Multi-clip *Overlay* layout superimposes two
    co-observing event spaces (e.g. wide + narrow) so motion in each is told apart by colour.

    Differing geometries are resized to the largest. Returns an ``(H, W, 3)`` uint8 image.
    """
    frames = [np.clip(np.asarray(f, float), 0.0, 1.0) for f in frames if f is not None]
    if not frames:
        return np.zeros((1, 1, 3), np.uint8)
    palette = colors or OVERLAY_COLORS
    H = max(f.shape[0] for f in frames); W = max(f.shape[1] for f in frames)
    out = np.zeros((H, W, 3), float)
    for i, f in enumerate(frames):
        f = _resize2d(f, (H, W))
        out += f[..., None] * np.asarray(palette[i % len(palette)], float)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return resize_rgb(out, size) if size else out


def draw_box(rgb, bbox, color=(57, 197, 207), width=2) -> np.ndarray:
    """Draw a rectangle outline ``(x0, y0, x1, y1)`` (sensor px) on an RGB image, in place."""
    if bbox is None:
        return rgb
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    x0, x1 = sorted((max(0, min(w - 1, x0)), max(0, min(w - 1, x1))))
    y0, y1 = sorted((max(0, min(h - 1, y0)), max(0, min(h - 1, y1))))
    c = np.asarray(color, np.uint8)
    for d in range(max(int(width), 1)):
        if y0 + d <= y1:
            rgb[y0 + d, x0:x1 + 1] = c; rgb[y1 - d, x0:x1 + 1] = c
        if x0 + d <= x1:
            rgb[y0:y1 + 1, x0 + d] = c; rgb[y0:y1 + 1, x1 - d] = c
    return rgb


def draw_label(rgb, text, xy=(4, 4), color=(255, 255, 255)) -> np.ndarray:
    """Draw a small text label onto an RGB image (Pillow); a no-op if Pillow is unavailable."""
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(rgb)
        ImageDraw.Draw(im).text(xy, text, fill=tuple(int(c) for c in color))
        return np.asarray(im)
    except Exception:
        return rgb


def _font(size=13):
    """A truetype font if one is findable, else PIL's default bitmap font."""
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "arial.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def infographic_frame(rgb, title=None, subtitle=None, footer_lines=None,
                      bar=(12, 16, 22)) -> np.ndarray:
    """Wrap an RGB frame with a context **banner**: a title/subtitle strip on top and a footer
    of context lines below — so a captured frame or video self-documents.

    Pure Pillow/NumPy. Returns a new ``(H', W', 3)`` uint8 image (taller than the input).
    """
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return np.asarray(rgb, np.uint8)
    rgb = np.asarray(rgb, np.uint8)
    h, w = rgb.shape[:2]
    top = 34 if title else 0
    foot = (len(footer_lines) * 16 + 8) if footer_lines else 0
    canvas = np.zeros((h + top + foot, w, 3), np.uint8)
    canvas[:] = bar
    canvas[top:top + h] = rgb
    im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
    if title:
        d.text((8, 5), title, fill=(255, 255, 255), font=_font(15))
        if subtitle:
            d.text((8, 21), subtitle, fill=(150, 200, 215), font=_font(11))
    for i, line in enumerate(footer_lines or []):
        d.text((8, top + h + 4 + i * 16), line, fill=(190, 205, 215), font=_font(12))
    return np.asarray(im)


def context_poster(frame_rgb, title, fields, width=920, accent=(57, 197, 207)) -> np.ndarray:
    """A self-documenting **infographic poster** for an export: a representative frame beside a
    panel of context key/values (sensor, window, settings, results).

    *fields* is an ordered ``dict`` (or list of ``(label, value)``) of context rows. Returns an
    ``(H, W, 3)`` uint8 image. Pure Pillow/NumPy.
    """
    from PIL import Image, ImageDraw
    rows = list(fields.items()) if hasattr(fields, "items") else list(fields)
    frame = Image.fromarray(np.asarray(frame_rgb, np.uint8)).convert("RGB")
    pad, title_h = 16, 40
    img_w = int(width * 0.52)
    scale = img_w / max(frame.width, 1)
    img_h = int(frame.height * scale)
    frame = frame.resize((img_w, max(img_h, 1)))
    panel_x = img_w + pad * 2
    body_h = max(img_h, len(rows) * 22 + 16)
    H = title_h + body_h + pad * 2
    im = Image.new("RGB", (width, H), (12, 16, 22))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, width, title_h], fill=(20, 28, 36))
    d.line([0, title_h, width, title_h], fill=accent, width=2)
    d.text((pad, 11), title, fill=(255, 255, 255), font=_font(18))
    im.paste(frame, (pad, title_h + pad))
    y = title_h + pad
    for label, value in rows:
        d.text((panel_x, y), f"{label}", fill=(140, 160, 175), font=_font(12))
        d.text((panel_x + 165, y), f"{value}", fill=(235, 240, 245), font=_font(12))
        y += 22
    d.text((pad, H - 16), "GottLUX", fill=(90, 110, 125), font=_font(10))
    return np.asarray(im)


def _open_writer(path, fps):
    """Open an imageio MP4 writer with a **fast** H.264 preset so live capture can keep up (the
    default ``medium`` preset encodes 1080p at well under real time and was starving the screen
    recorder's queue → dropped frames). Falls back to the plain writer if the params are rejected."""
    import imageio
    try:
        return imageio.get_writer(path, fps=fps, macro_block_size=2, codec="libx264",
                                  output_params=["-preset", "veryfast"])
    except Exception:                              # pragma: no cover - older imageio / odd backend
        return imageio.get_writer(path, fps=fps, macro_block_size=2)


def ffmpeg_available() -> bool:
    """True if the FFMPEG muxer is usable (``imageio`` + ``imageio-ffmpeg`` with a real binary).

    Lets callers tell "the codec is missing — install imageio-ffmpeg" apart from "encoding
    failed for some other reason", instead of blaming a missing package for every failure.
    """
    try:
        import imageio  # noqa: F401
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
        return True
    except Exception:
        return False


def _coerce_frame(f, target_hw):
    """Coerce *f* to a contiguous ``(H, W, 3)`` uint8 array sized to *target_hw* ``(h, w)``.

    Enforces the two invariants libx264/yuv420p needs: every frame shares one size, and that
    size is even. A frame that came through a different shape (e.g. one whose render failed and
    fell back to a placeholder) is resized to the target rather than aborting the whole clip.
    """
    f = np.asarray(f, np.uint8)
    if f.ndim == 2:
        f = np.repeat(f[..., None], 3, axis=2)
    if f.ndim == 3 and f.shape[2] > 3:
        f = f[..., :3]
    if f.shape[:2] != target_hw:
        f = resize_rgb(f, (target_hw[1], target_hw[0]))          # resize_rgb takes (width, height)
    return np.ascontiguousarray(f, np.uint8)


def write_video(path, frames, fps=25):
    """Mux an iterable of ``(H, W, 3)`` uint8 frames to *path* (MP4). Returns *path* or ``None``.

    Robust by construction: the first frame fixes the clip's size (rounded **down to even**,
    which libx264 requires), and every later frame is coerced to it — so odd dimensions or a
    single odd-shaped frame can no longer corrupt or abort the export. Still fails soft (returns
    ``None``) if the muxer is missing or encoding errors, so a bad codec never loses the rest of
    a bundle; pair with :func:`ffmpeg_available` to report *why*.
    """
    try:
        import os
        import imageio
    except Exception as e:                         # pragma: no cover - imageio not installed
        print(f"[gottlux] video export unavailable (imageio import failed: {e})")
        return None
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        target_hw = None
        wrote = 0
        # macro_block_size=2 keeps imageio from rounding dimensions up to a multiple of 16 while
        # still guaranteeing the even width/height the H.264 encoder needs; the fast preset keeps
        # encoding ahead of real time.
        with _open_writer(path, fps) as w:
            for f in frames:
                f = np.asarray(f, np.uint8)
                if f.ndim == 2:
                    f = np.repeat(f[..., None], 3, axis=2)
                if f.ndim == 3 and f.shape[2] > 3:
                    f = f[..., :3]
                if target_hw is None:
                    h, wd = f.shape[:2]
                    target_hw = (max(2, h - h % 2), max(2, wd - wd % 2))
                w.append_data(_coerce_frame(f, target_hw))
                wrote += 1
        return path if wrote else None
    except Exception as e:                         # pragma: no cover - codec/env dependent
        print(f"[gottlux] video export failed while encoding ({e})")
        return None


class VideoWriter:
    """Incremental MP4 writer for **live** capture, where frames arrive over time (a screen
    recorder, a live grab loop). Open it, ``append(rgb)`` each frame, then ``close()``.

    Applies the same robustness as :func:`write_video`: the first frame fixes an **even** clip
    size and every later frame is coerced to it, so odd dimensions or a window that resizes
    mid-recording can't corrupt or abort the file. ``ok`` is False if the muxer couldn't open.
    """

    def __init__(self, path, fps=30):
        self.path = path
        self.fps = fps
        self.frames = 0
        self.error = None
        self._target_hw = None
        self._w = None
        try:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            self._w = _open_writer(path, fps)
        except Exception as e:                     # pragma: no cover - codec/env dependent
            self.error = e
            print(f"[gottlux] live video writer unavailable ({e})")

    @property
    def ok(self) -> bool:
        return self._w is not None

    @property
    def size(self):
        """The clip's actual ``(width, height)`` — the even-rounded size the first frame
        fixed — or ``None`` before any frame was written. Export provenance records it,
        since it can differ by a pixel from the canvas geometry the frames were rendered
        at (H.264 requires even dimensions)."""
        if self._target_hw is None:
            return None
        h, w = self._target_hw
        return (int(w), int(h))

    def append(self, rgb) -> bool:
        """Coerce *rgb* to the clip's even size and write it. Returns False on any failure."""
        if self._w is None or rgb is None:
            return False
        try:
            f = np.asarray(rgb, np.uint8)
            if f.ndim == 2:
                f = np.repeat(f[..., None], 3, axis=2)
            if f.ndim == 3 and f.shape[2] > 3:
                f = f[..., :3]
            if self._target_hw is None:
                h, w = f.shape[:2]
                self._target_hw = (max(2, h - h % 2), max(2, w - w % 2))
            self._w.append_data(_coerce_frame(f, self._target_hw))
            self.frames += 1
            return True
        except Exception as e:                     # pragma: no cover
            self.error = e
            return False

    def close(self):
        """Finalise the file. Returns the path if any frame was written, else ``None``."""
        if self._w is not None:
            try:
                self._w.close()
            except Exception as e:                 # pragma: no cover
                self.error = e
            self._w = None
        return self.path if self.frames else None


def render_box_track_video(rec, out_path, box_at, t0, t1, *, fps=25, accum_dt=0.02,
                           cmap="inferno", color=(57, 197, 207), label_fn=None,
                           filters=None, max_frames=1200):
    """Render an analysis video: accumulated frames over ``[t0, t1]`` with a tracked box overlay.

    *box_at(t)* returns the ``(x0, y0, x1, y1)`` box at time *t* (e.g. the Range lab's keyframe
    interpolation), or ``None``. *label_fn(t)* optionally returns an overlay string (e.g. the
    time and estimated range). *filters* is an optional live denoise suite. Returns the written
    path (or ``None`` if the muxer is unavailable).
    """
    t0, t1 = float(t0), float(t1)
    n = int(np.clip(round((t1 - t0) / max(accum_dt, 1e-4)), 1, max_frames))
    times = t0 + (np.arange(n) + 0.5) * (t1 - t0) / n

    def _frames():
        for t in times:
            win = rec.window(t, min(t + accum_dt, t1))
            if filters is not None:
                win = filters.apply(win)
            rgb = colorize(accumulate_frame(win, mode="count"), cmap=cmap)
            draw_box(rgb, box_at(t) if box_at else None, color=color)
            if label_fn is not None:
                draw_label(rgb, label_fn(t))
            yield rgb

    return write_video(out_path, _frames(), fps=fps)
