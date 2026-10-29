"""
shot.py — the *shot*: everything that defines one short animation, as plain data.

A shot is a recipe, not pixels: which recording, which span of it, which crop, which
**look** (see :mod:`gottlux.studio.looks`) with which parameters, how fast it plays, and how
it is framed and exported. Being plain data it round-trips through JSON, so every exported
GIF/MP4 carries a ``.shot.json`` beside it that re-renders it exactly, and the web Studio
puts the same recipe in its URL.

Timing (the one formula every front end shares)
-----------------------------------------------
``speed`` is recording seconds shown per wall-clock second (0.1 = ten times slower than
life). A shot spanning ``[t0, t1]`` played at ``fps`` therefore has::

    n_frames = max(1, round((t1 - t0) * fps / speed))

and frame *k* (0-based) ends at recording time ``tau_k = t0 + (k + 1) (t1 - t0) / n_frames``.
A look renders what the sensor had reported *by* ``tau_k`` (each look decides how far back
it looks). A look may override the count (the stroboscope plays phase bins, not time).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

from gottlux.core import colors as _colors

# ON/OFF colour pairs + background — the built-in palettes of the colour registry
# (gottlux.core.colors), where user palettes are added too. "fire-ice" is the default: the
# orange/blue pair that passes the colour-vision checks on its background (OKLab ΔE 26.7
# under protanopia).
PALETTES = _colors.BUILTIN_PALETTES

# output presets: (format, width px, fps)
PRESETS = {
    "gif-small":  ("gif", 320, 20),
    "gif":        ("gif", 480, 25),
    "gif-large":  ("gif", 640, 25),
    "mp4":        ("mp4", 720, 30),
    "mp4-hd":     ("mp4", 1080, 30),
    "mp4-smooth": ("mp4", 1080, 60),
    "webm":       ("webm", 720, 30),
}


@dataclass
class Shot:
    """One animation recipe. Every field has a usable default."""

    source: str = ""                          # recording path (informational for in-memory use)
    look: str = "trails"
    params: dict = field(default_factory=dict)  # look parameters (missing keys → defaults)
    t0: float | None = None                # recording seconds; None → recording start
    t1: float | None = None                # None → t0 + 1 s (or the recording end)
    crop: tuple | None = None              # (x0, y0, x1, y1) sensor px, end exclusive
    speed: float = 0.1                        # recording s per wall s (0.1 = 10x slow motion)
    fps: float = 25.0
    width: int = 480                          # output width px (height follows the aspect)
    palette: str = "fire-ice"
    denoise: bool = True                      # hot-pixel cut + background-activity filter
    baf_us: float = 2000.0                    # BAF neighbour window (µs)
    bloom: float = 0.0                        # 0 = off; ~0.5–1.5 = glow
    gamma: float = 1.0
    resample: str = "bilinear"                # nearest | bilinear | lanczos
    stamp: bool = True                        # time label
    speed_label: bool = True                  # "10x slower"
    title: str = ""
    legend: bool = False                      # ON/OFF swatches
    rate_strip: bool = False                  # event-rate strip with a playhead
    pingpong: bool = False                    # play forward then backward (seamless GIF loop)
    format: str = "gif"                       # gif | mp4 | webm | png
    gif_colors: int = 128
    quality: int = 8                          # mp4/webm quality 1..10

    # ------------------------------------------------------------------ timing
    def span(self, rec_t0: float, rec_t1: float) -> tuple:
        """The shot's ``(t0, t1)`` clamped to the recording ``[rec_t0, rec_t1]``."""
        t0 = rec_t0 if self.t0 is None else max(rec_t0, min(float(self.t0), rec_t1))
        t1 = min(t0 + 1.0, rec_t1) if self.t1 is None else max(t0, min(float(self.t1), rec_t1))
        if t1 <= t0:
            t1 = min(rec_t1, t0 + 1e-3)
        return t0, t1

    def n_frames(self, t0: float, t1: float) -> int:
        # half-up rounding (Python's round() is half-to-even; the web port uses Math.round)
        return max(1, int(math.floor((t1 - t0) * self.fps / max(self.speed, 1e-6) + 0.5)))

    def colors(self) -> dict:
        return _colors.palette(self.palette)

    # ------------------------------------------------------------------ (de)serialise
    def to_dict(self) -> dict:
        d = asdict(self)
        d["crop"] = list(self.crop) if self.crop else None
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> Shot:
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        if known.get("crop"):
            known["crop"] = tuple(int(v) for v in known["crop"])
        return cls(**known)

    @classmethod
    def from_json(cls, text: str) -> Shot:
        return cls.from_dict(json.loads(text))

    def apply_preset(self, name: str) -> Shot:
        fmt, width, fps = PRESETS[name]
        self.format, self.width, self.fps = fmt, width, fps
        return self
