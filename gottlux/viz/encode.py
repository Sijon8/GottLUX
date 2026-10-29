"""
encode.py — the one encoder behind every rendered export: MP4, WebM, GIF or PNG frames.

Every exporter in GottLUX that turns rendered frames into a file — the Studio, the Canvas
composer, the Timeline, the headless canvas renderer — goes through :func:`encode_frames`, so
a format means the same thing everywhere: the same codec settings, the same quality scale,
the same frame-rate rules, the same cancel and progress behaviour, the same facts returned.

Frames come from a **random-access** source, ``frame_at(i) → (H, W, 3) uint8`` for
``i = 0 … n − 1``: the GIF palette is measured on frames spread over the whole clip before
any frame is written, and frames can be rendered a few ahead on worker threads while the
encoder writes (``workers``), which keeps the encoder busy without changing a pixel.

Formats
-------
``mp4``   H.264 (libx264), ``yuv420p``, ``+faststart`` (plays inline on the web and in chat
          apps), constant-quality ``crf`` from *quality*, preset ``fast``.
``webm``  VP9 (libvpx-vp9), constant quality (``-b:v 0 -crf``), multi-threaded rows.
``gif``   one adaptive palette for the whole clip (median cut over up to 8 sampled frames, so
          colours do not shimmer), no dithering, ``loop=0``. GIF stores delays in
          centiseconds, so the delay rounds to 10 ms and the facts report the rate that
          really plays (25 fps is exact; 30 fps plays at 33.3 fps).
``png``   a folder of ``frame_00000.png`` …, lossless, for editing elsewhere.

*quality* is 1 … 10 for every lossy format (10 = best). H.264 maps it to ``crf`` 34 … 16,
VP9 to ``crf`` 50 … 18; GIF ignores it (its knob is *gif_colors*).

Frame sizes: the first frame fixes the clip size; H.264/VP9 need even dimensions, so an odd
size rounds **down** to even, and any later frame of another size is resized to it rather
than aborting the export (the same invariants :class:`gottlux.viz.video.VideoWriter` keeps).
*size* rescales every frame to ``(width, height)`` first.

A cancelled export (*cancel()* returned True) removes the partial file and returns ``None``;
a missing muxer raises :class:`EncodeError` with an install hint.
"""
from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np

#: Format → (file extension, codec description). The extension of *path* picks the format.
FORMATS = {
    "mp4": (".mp4", "H.264 (libx264) in MP4"),
    "webm": (".webm", "VP9 (libvpx-vp9) in WebM"),
    "gif": (".gif", "GIF (one adaptive palette)"),
    "png": ("", "PNG sequence (lossless)"),
}

#: Human-readable labels for a format picker, in menu order.
FORMAT_LABELS = (
    ("mp4", "MP4 — H.264 (plays everywhere)"),
    ("webm", "WebM — VP9 (web, smaller)"),
    ("gif", "GIF — animated image (loops, chat apps)"),
    ("png", "PNG frames — lossless sequence"),
)


class EncodeError(RuntimeError):
    """The export could not be encoded (missing muxer, codec failure)."""


def format_for(path: str, default: str = "mp4") -> str:
    """The format implied by *path*'s extension (no extension → ``png`` frames folder)."""
    ext = os.path.splitext(str(path))[1].lower().lstrip(".")
    if ext in FORMATS:
        return ext
    if not ext:
        return "png"
    return default


def with_extension(path: str, fmt: str) -> str:
    """*path* with the extension *fmt* needs (a PNG export is a folder: no extension)."""
    root, ext = os.path.splitext(str(path))
    want = FORMATS[fmt][0]
    if fmt == "png":
        return root if ext.lower() in {e for e, _ in FORMATS.values() if e} else str(path)
    return str(path) if ext.lower() == want else root + want


def crf_for(fmt: str, quality: float) -> int:
    """The constant-rate factor a 1 … 10 *quality* means for *fmt* (H.264 34 … 16, VP9
    50 … 18)."""
    q = float(np.clip(quality, 1, 10))
    lo, hi = (34, 16) if fmt == "mp4" else (50, 18)
    return int(round(np.interp(q, [1, 10], [lo, hi])))


def gif_delay_ms(fps: float) -> int:
    """GIF frame delay in ms: centisecond steps, at least 20 ms (browsers clamp faster)."""
    return max(20, int(round(100.0 / max(float(fps), 1e-6))) * 10)


def even_size(w: int, h: int) -> tuple:
    return max(2, int(w) - int(w) % 2), max(2, int(h) - int(h) % 2)


def ffmpeg_ok() -> bool:
    try:
        import imageio  # noqa: F401
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
        return True
    except Exception:
        return False


def _coerce(f, size=None) -> np.ndarray:
    """Contiguous ``(H, W, 3)`` uint8, resized to *size* ``(w, h)`` when it differs."""
    f = np.asarray(f)
    if f.dtype != np.uint8:
        f = np.clip(f, 0, 255).astype(np.uint8)
    if f.ndim == 2:
        f = np.repeat(f[..., None], 3, axis=2)
    elif f.ndim == 3 and f.shape[2] > 3:
        f = f[..., :3]
    if size is not None and (f.shape[1], f.shape[0]) != tuple(size):
        from PIL import Image
        f = np.asarray(Image.fromarray(np.ascontiguousarray(f)).resize(
            (int(size[0]), int(size[1])), Image.BILINEAR))
    return np.ascontiguousarray(f, np.uint8)


class _Sizer:
    """Coerces every frame to the size the first one fixes (even for H.264/VP9)."""

    def __init__(self, size=None, even=False):
        self.request, self.even, self.size = size, even, None

    def __call__(self, f):
        if self.size is None:
            f0 = _coerce(f, self.request)
            h, w = f0.shape[:2]
            self.size = even_size(w, h) if self.even else (w, h)
        return _coerce(f, self.size)


def _frames(frame_at, n, workers, size_fn):
    """Yield frames 0 … n−1 in order, rendering up to *workers* ahead on threads."""
    if workers <= 1 or n <= 2:
        for i in range(n):
            yield size_fn(frame_at(i))
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gottlux-render") as ex:
        ahead = {}
        nxt = 0
        for i in range(min(n, workers * 2)):
            ahead[i] = ex.submit(frame_at, i)
            nxt = i + 1
        for i in range(n):
            f = ahead.pop(i).result()
            if nxt < n:
                ahead[nxt] = ex.submit(frame_at, nxt)
                nxt += 1
            yield size_fn(f)
        for fut in ahead.values():               # only on an early exit (cancel)
            fut.cancel()


def encode_frames(frame_at, n, path, fmt: str | None = None, fps: float = 30.0,
                  quality: float = 8, gif_colors: int = 128, size=None, progress=None,
                  cancel=None, workers: int = 2, preset: str = "fast") -> dict | None:
    """Encode frames ``frame_at(0 … n−1)`` to *path*. Returns the facts dict or ``None``
    when cancelled.

    Facts: ``path``, ``format``, ``codec``, ``frames`` (written), ``fps`` (requested),
    ``fps_effective`` (what plays: differs for GIF), ``duration_s`` (frames ÷ effective
    rate), ``width``/``height`` (as encoded) and, for GIF, ``frame_ms``.
    """
    n = max(int(n), 0)
    fmt = fmt or format_for(path)
    if fmt not in FORMATS:
        raise ValueError(f"unknown export format {fmt!r}; one of {', '.join(FORMATS)}")
    if n == 0:
        raise EncodeError("nothing to export: the clip has no frames")
    fps = float(fps)
    if not fps > 0:
        raise ValueError("the frame rate must be positive")
    path = os.path.abspath(path)
    parent = path if fmt == "png" else os.path.dirname(path)
    os.makedirs(parent or ".", exist_ok=True)

    sized = _Sizer(size, even=fmt in ("mp4", "webm"))

    def tick(i):
        if progress:
            try:
                progress((i + 1) / n)
            except Exception:
                pass

    def cancelled():
        try:
            return bool(cancel and cancel())
        except Exception:
            return False

    if fmt == "gif":
        return _gif(frame_at, n, path, fps, gif_colors, sized, tick, cancelled, workers)
    if fmt == "png":
        return _pngs(frame_at, n, path, fps, sized, tick, cancelled, workers)
    return _video(frame_at, n, path, fmt, fps, quality, preset, sized, tick, cancelled,
                  workers)


def _facts(path, fmt, frames, fps, fps_eff, size, **extra):
    w, h = size or (0, 0)
    return {"path": path, "format": fmt, "codec": FORMATS[fmt][1], "frames": int(frames),
            "fps": float(fps), "fps_effective": float(fps_eff),
            "duration_s": frames / float(fps_eff) if fps_eff else 0.0,
            "width": int(w), "height": int(h), **extra}


def _video(frame_at, n, path, fmt, fps, quality, preset, sized, tick, cancelled, workers):
    if not ffmpeg_ok():
        raise EncodeError("video encoding needs imageio-ffmpeg "
                          "(pip install imageio imageio-ffmpeg)")
    import imageio.v2 as imageio
    crf = str(crf_for(fmt, quality))
    if fmt == "mp4":
        kw = dict(codec="libx264", quality=None, pixelformat="yuv420p",
                  ffmpeg_params=["-preset", preset, "-crf", crf, "-movflags", "+faststart"])
    else:
        kw = dict(codec="libvpx-vp9", quality=None, pixelformat="yuv420p",
                  ffmpeg_params=["-b:v", "0", "-crf", crf, "-deadline", "good",
                                 "-cpu-used", "4", "-row-mt", "1"])
    writer = None
    wrote = 0
    ok = False
    try:
        for i, f in enumerate(_frames(frame_at, n, workers, sized)):
            if cancelled():
                return None
            if writer is None:
                writer = imageio.get_writer(path, fps=fps, macro_block_size=2, **kw)
            writer.append_data(f)
            wrote += 1
            tick(i)
        ok = True
    except EncodeError:
        raise
    except Exception as e:
        raise EncodeError(f"encoding failed: {e}") from e
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception as e:
                if ok:
                    raise EncodeError(f"encoding failed while finishing the file: {e}") from e
        if not ok and os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
    return _facts(path, fmt, wrote, fps, fps, sized.size)


def _gif(frame_at, n, path, fps, gif_colors, sized, tick, cancelled, workers):
    from PIL import Image
    picks = np.unique(np.linspace(0, n - 1, min(8, n)).astype(int))
    sample = [sized(frame_at(int(i))) for i in picks]
    mosaic = Image.fromarray(np.concatenate(sample, axis=0))
    colors = int(np.clip(gif_colors, 8, 256))
    pal = mosaic.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    frames = []
    for i, f in enumerate(_frames(frame_at, n, workers, sized)):
        if cancelled():
            return None
        frames.append(Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE))
        tick(i)
    delay = gif_delay_ms(fps)
    try:
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=delay, loop=0,
                       disposal=1, optimize=False)
    except Exception as e:
        if os.path.isfile(path):
            os.remove(path)
        raise EncodeError(f"GIF encoding failed: {e}") from e
    return _facts(path, "gif", len(frames), fps, 1000.0 / delay, sized.size,
                  frame_ms=delay)


def _pngs(frame_at, n, folder, fps, sized, tick, cancelled, workers):
    from PIL import Image
    os.makedirs(folder, exist_ok=True)
    wrote = 0
    for i, f in enumerate(_frames(frame_at, n, workers, sized)):
        if cancelled():
            shutil.rmtree(folder, ignore_errors=True)
            return None
        Image.fromarray(f).save(os.path.join(folder, f"frame_{i:05d}.png"))
        wrote += 1
        tick(i)
    return _facts(folder, "png", wrote, fps, fps, sized.size)


def probe(path) -> dict:
    """``{"frames", "width", "height", "fps"}`` of a written video/GIF, read back from the
    file (ffmpeg for MP4/WebM, Pillow for GIF) — what the tests and the export checks use
    to confirm a file holds what its facts claim."""
    fmt = format_for(path)
    if fmt == "gif":
        from PIL import Image
        with Image.open(path) as im:
            frames = getattr(im, "n_frames", 1)
            w, h = im.size
            ms = im.info.get("duration", 0) or 0
        return {"frames": int(frames), "width": int(w), "height": int(h),
                "fps": 1000.0 / ms if ms else 0.0}
    if fmt == "png":
        files = sorted(f for f in os.listdir(path) if f.endswith(".png"))
        from PIL import Image
        w = h = 0
        if files:
            with Image.open(os.path.join(path, files[0])) as im:
                w, h = im.size
        return {"frames": len(files), "width": w, "height": h, "fps": 0.0}
    import imageio.v2 as imageio
    rd = imageio.get_reader(path, "ffmpeg")
    try:
        meta = rd.get_meta_data()
        frames = 0
        for _ in rd:
            frames += 1
        w, h = meta.get("size", (0, 0))
        return {"frames": frames, "width": int(w), "height": int(h),
                "fps": float(meta.get("fps", 0.0))}
    finally:
        rd.close()
