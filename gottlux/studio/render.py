"""
render.py — turn a :class:`~gottlux.studio.shot.Shot` into frames.

:class:`ShotRenderer` slices the recording once (the shot's span plus the look's look-back,
cropped, optionally denoised), asks the look to measure its exposure, then renders any
frame on request: ``renderer.frame(i)`` → ``uint8 (H, W, 3)``. Frames are independent, so a
GUI preview can scrub, and an exporter can stream them in order.

Per frame: the look returns ``(alpha, colour)`` layers → composited over the palette
background (added on a dark background, alpha-blended on a light one) → display gamma →
resampled to the output width → optional bloom → overlays (time stamp, speed, title,
ON/OFF legend, event-rate strip with a playhead).
"""
from __future__ import annotations

import numpy as np

from gottlux.studio.looks import Events, get_look, hex_rgb
from gottlux.studio.shot import Shot

_RESAMPLE = {"nearest": 0, "bilinear": 2, "lanczos": 1}       # PIL Image.Resampling values


def _even(v: float) -> int:
    v = max(2, int(np.floor(v + 0.5)))                 # half-up, as in the web port
    return v + (v % 2)


def clamp_crop(crop, W, H):
    """``(x0, y0, x1, y1)`` clamped to the sensor, at least 4 px each way; ``None`` = full."""
    if not crop:
        return None
    x0, y0, x1, y1 = (int(round(v)) for v in crop)
    x0, x1 = sorted((max(0, min(x0, W)), max(0, min(x1, W))))
    y0, y1 = sorted((max(0, min(y0, H)), max(0, min(y1, H))))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    if (x0, y0, x1, y1) == (0, 0, W, H):
        return None
    return (x0, y0, x1, y1)


def prepare_events(rec, shot: Shot, t0: float, t1: float, lookback: float) -> Events:
    """The shot's events: ``[t0 - lookback, t1]``, cropped, denoised, crop-relative."""
    from gottlux.core.denoise import FilterSettings, filter_window
    crop = clamp_crop(shot.crop, rec.width, rec.height)
    a = max(rec.t_start_s, t0 - lookback)
    win = rec.window(a, t1 + 1e-9, roi=crop)
    if shot.denoise and len(win.t):
        win = filter_window(win, FilterSettings(enabled=True, hot_pixel=True, hot_pct=99.9,
                                                baf=True, baf_dt_us=float(shot.baf_us)))
    x0, y0, x1, y1 = crop or (0, 0, rec.width, rec.height)
    return Events(np.asarray(win.x, np.int32) - x0, np.asarray(win.y, np.int32) - y0,
                  np.asarray(win.p) == 1, np.asarray(win.t, np.int64), x1 - x0, y1 - y0,
                  t0, t1)


def composite(layers, bg, shape) -> np.ndarray:
    h, w = shape
    out = np.empty((h, w, 3), np.float32)
    out[:] = bg
    additive = float(np.mean(bg)) < 0.5
    for a, col in layers:
        a = np.asarray(a, np.float32)[..., None]
        col = np.asarray(col, np.float32)
        if col.ndim == 1:
            col = col[None, None, :]
        if additive:
            out += a * col
        else:
            out = out * (1.0 - a) + a * col
    return np.clip(out, 0.0, 1.0)


def speed_text(speed: float) -> str:
    """'10× slower', '2.5× slower', 'real time', '4× faster'."""
    if abs(speed - 1.0) < 1e-6:
        return "real time"
    x = 1.0 / speed if speed < 1 else speed
    num = f"{x:.0f}" if x >= 10 or abs(x - round(x)) < 0.05 else f"{x:.1f}"
    return f"{num}× {'slower' if speed < 1 else 'faster'}"


_FONT_CACHE: dict = {}


def font(size: int):
    from PIL import ImageFont
    if size not in _FONT_CACHE:
        try:
            from matplotlib import font_manager
            _FONT_CACHE[size] = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
        except Exception:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


class ShotRenderer:
    """Render a shot from a loaded :class:`~gottlux.io.recording.Recording`.

    ``events`` lets a caller reuse a prepared :class:`Events` (the GUI does, when only a
    look parameter changed); it must cover the same span, crop and denoise settings.
    """

    def __init__(self, rec, shot: Shot, events: Events | None = None):
        self.rec = rec
        self.shot = shot
        self.look = get_look(shot.look)
        self.P = self.look.resolve(shot.params)
        self.t0, self.t1 = shot.span(rec.t_start_s, rec.t_stop_s)
        self.lookback = self.look.memory_s(self.P)
        self.ev = events if events is not None else prepare_events(
            rec, shot, self.t0, self.t1, self.lookback)
        n = shot.n_frames(self.t0, self.t1)
        self.n_frames = int(self.look.n_frames(self, self.P) or n)
        self.times = self.t0 + (np.arange(self.n_frames) + 1) * (self.t1 - self.t0) / self.n_frames
        cw, ch = self.look.canvas_size(self.ev.W, self.ev.H, self.P)
        self.out_w = _even(shot.width)
        self.out_h = _even(self.out_w * ch / cw)
        self.draw_size = (cw, ch) if self.look.native else (self.out_w, self.out_h)
        pal = shot.colors()
        self.colors = {k: hex_rgb(v) for k, v in pal.items()}
        self.st = self.look.prepare(self.ev, self, self.P)
        self.strip_h = _even(max(20, self.out_h // 9)) if shot.rate_strip else 0
        self._rate = None
        if shot.rate_strip:
            sl = self.ev.span(self.t0, self.t1)
            r = np.histogram(self.ev.t[sl], bins=self.out_w, range=(self.t0, self.t1))[0]
            self._rate = r / max(r.max(), 1)

    # ------------------------------------------------------------------ frame order
    def __len__(self) -> int:
        n = self.n_frames
        return 2 * n - 2 if self.shot.pingpong and n > 2 else n

    def source_index(self, i: int) -> int:
        n = self.n_frames
        return i if i < n else 2 * n - 2 - i

    def __iter__(self):
        for i in range(len(self)):
            yield self.frame(i)

    @property
    def size(self) -> tuple:
        """Output ``(width, height)`` in px, including the rate strip."""
        return self.out_w, self.out_h + self.strip_h

    @property
    def duration_s(self) -> float:
        return len(self) / float(self.shot.fps)

    # ------------------------------------------------------------------ one frame
    def frame(self, i: int) -> np.ndarray:
        from PIL import Image, ImageFilter
        k = self.source_index(int(i) % max(len(self), 1))
        tau = float(self.times[k])
        w, h = self.draw_size
        layers = self.look.render(self.ev, k, tau, self.P, self.st, self.colors, (w, h))
        img = composite(layers, self.colors["bg"], (h, w))
        if self.shot.gamma and abs(self.shot.gamma - 1.0) > 1e-3:
            img = img ** (1.0 / float(self.shot.gamma))
        pil = Image.fromarray((img * 255 + 0.5).astype(np.uint8))
        if pil.size != (self.out_w, self.out_h):
            pil = pil.resize((self.out_w, self.out_h), _RESAMPLE.get(self.shot.resample, 2))
        if self.shot.bloom > 0:
            arr = np.asarray(pil, np.float32) / 255.0
            radius = max(1.5, self.out_w / 140.0)
            blur = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius)), np.float32) / 255.0
            arr = 1.0 - (1.0 - arr) * (1.0 - np.clip(self.shot.bloom * blur, 0, 1))
            pil = Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))
        if self.strip_h:
            canvas = Image.new("RGB", (self.out_w, self.out_h + self.strip_h),
                               tuple(int(c * 255 * 0.6) for c in self.colors["bg"]))
            canvas.paste(pil, (0, 0))
            pil = canvas
        self._overlay(pil, k, tau)
        return np.asarray(pil)

    def _overlay(self, pil, k, tau):
        from PIL import ImageDraw
        s = self.shot
        d = ImageDraw.Draw(pil)
        dark = float(np.mean(self.colors["bg"])) < 0.5
        ink = (240, 239, 236) if dark else (20, 20, 19)
        dim = (180, 178, 170) if dark else (82, 81, 78)
        shadow = (0, 0, 0) if dark else (255, 255, 255)
        fs = max(11, self.out_w // 34)
        f, fsm = font(fs), font(max(9, int(fs * 0.78)))
        pad = max(6, fs // 2)

        def text(xy, txt, fnt, fill, anchor="la"):
            x, y = xy
            d.text((x + 1, y + 1), txt, font=fnt, fill=shadow, anchor=anchor)
            d.text((x, y), txt, font=fnt, fill=fill, anchor=anchor)

        y = pad
        if s.stamp:
            text((pad, y), self.look.stamp(k, tau, self.P, self.st), f, ink)
            y += int(fs * 1.25)
        if s.speed_label and self.look.name != "strobe":
            lab = speed_text(float(s.speed))
            text((pad, y), lab, fsm, dim)
        if s.title:
            text((self.out_w - pad, pad), s.title, f, ink, anchor="ra")
        if s.legend and self.look.uses_polarity(self.P):
            sw = max(8, fs // 2)
            x = self.out_w - pad
            yb = self.out_h - pad - sw
            for lab, key in (("OFF", "off"), ("ON", "on")):
                tw = d.textlength(lab, font=fsm)
                x -= tw
                text((x, yb - 2), lab, fsm, dim)
                x -= sw + 4
                col = tuple(int(c * 255) for c in self.colors[key])
                d.rectangle([x, yb, x + sw, yb + sw], fill=col)
                x -= pad
        if self.strip_h and self._rate is not None:
            y0 = self.out_h + 2
            hh = self.strip_h - 4
            col = tuple(int(c * 255) for c in self.colors["on"])
            for xi, rv in enumerate(self._rate):
                if rv > 0:
                    d.line([(xi, y0 + hh), (xi, y0 + hh - max(1, int(rv * hh)))], fill=col)
            px = int((tau - self.t0) / max(self.t1 - self.t0, 1e-9) * (self.out_w - 1))
            d.line([(px, self.out_h), (px, self.out_h + self.strip_h)], fill=ink, width=2)


def render_preview(rec, shot: Shot, i: int = 0, width: int = 360) -> np.ndarray:
    """One frame of *shot* at a small width (for thumbnails and quick checks)."""
    from dataclasses import replace
    return ShotRenderer(rec, replace(shot, width=width)).frame(i)
