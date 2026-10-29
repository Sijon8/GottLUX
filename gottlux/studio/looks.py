"""
looks.py — the Studio's visualization methods ("looks"), each from first principles.

An event camera reports, per pixel, the instant its log-brightness changed by more than a
contrast threshold: an event ``e_i = (x_i, y_i, p_i, t_i)`` with polarity ``p = ON`` (brighter)
or ``OFF`` (darker). There are no frames. Every look below is a different answer to "what
picture do the events up to time τ make?", evaluated independently for each output frame
``τ_k`` (so a preview can jump to any frame, and frames can render in any order).

==================  ==========================================================================
look                picture at time τ
==================  ==========================================================================
``trails``          Σ exp(−(τ − t_i)/T) per pixel and polarity: a decaying light trail
``heat``            event count in the window [τ − W, τ], through a colormap
``surface``         time surface: exp(−(τ − T_last)/T) of each pixel's most recent event
``paint``           light painting: everything since the shot start, hue = time of last event
``spacetime``       the (x, y, t) cloud of [τ − D, τ] as a 3-D box, optionally orbiting
``strobe``          stroboscope: events folded by phase t·f0 mod 1, one phase bin per frame
``waterfall``       kymograph: the live frame plus a (time × x) history strip below it
``flicker``         per-cell dominant flicker frequency (hue) over the live trails
==================  ==========================================================================

Each look returns *layers* — ``(alpha, colour)`` pairs, alpha in [0, 1] per pixel — and the
renderer composites them over the palette background (additively on a dark background, so
ON and OFF overlapping read as white; with alpha on a light one).

Exposure is **locked per shot**: a look measures its scale once on sample frames
(:meth:`Look.prepare`) instead of per frame, so an animation does not pump in brightness as
activity rises and falls.
"""
from __future__ import annotations

import numpy as np

from gottlux.detectors.base import Param

# ------------------------------------------------------------------ colour helpers
#: The colormaps the web Studio implements too (``web/js/looks.js``). In the desktop app a
#: colormap parameter accepts any scheme of the colour registry (:mod:`gottlux.core.colors`),
#: user schemes included.
COLORMAPS = ("ice", "ember", "inferno", "magma", "viridis", "cividis", "gray", "turbo")


def hex_rgb(c: str) -> np.ndarray:
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def lut(name: str) -> np.ndarray:
    """A 256×3 float32 colour table for *name* (GottLUX and user ramps by their stops,
    then matplotlib; an unknown name gives inferno)."""
    from gottlux.core import colors
    return colors.lut(name, fallback="inferno")


def apply_lut(v: np.ndarray, name: str) -> np.ndarray:
    idx = np.clip((np.nan_to_num(v) * 255).astype(np.int32), 0, 255)
    return lut(name)[idx]


def tone(v: np.ndarray, mode: str) -> np.ndarray:
    v = np.clip(v, 0, 1)
    if mode == "sqrt":
        return np.sqrt(v)
    if mode == "log":
        return np.log1p(30 * v) / np.log1p(30)
    return v


# ------------------------------------------------------------------ event container
class Events:
    """Shot-local events: crop-relative ``x, y`` (int32), ``on`` (bool), ``t`` (float64 s on
    the recording clock), ``t_us`` (int64 µs) and the crop size ``W × H``; plus the shot's
    span ``t0, t1``."""

    def __init__(self, x, y, on, t_us, W, H, t0, t1):
        self.x = np.asarray(x, np.int32)
        self.y = np.asarray(y, np.int32)
        self.on = np.asarray(on, bool)
        self.t_us = np.asarray(t_us, np.int64)
        self.t = self.t_us.astype(np.float64) * 1e-6
        self.W, self.H = int(W), int(H)
        self.t0, self.t1 = float(t0), float(t1)
        self.flat = self.y.astype(np.int64) * self.W + self.x

    @property
    def n(self) -> int:
        return int(self.t.size)

    def span(self, a: float, b: float) -> slice:
        """Index slice of the events with ``a <= t <= b``."""
        return slice(int(np.searchsorted(self.t, a, "left")),
                     int(np.searchsorted(self.t, b, "right")))

    def splat(self, sl, weights=None, mask=None) -> np.ndarray:
        """Per-pixel sum (``H × W`` float32) of *weights* (default 1) over ``sl`` [∩ mask]."""
        f = self.flat[sl]
        w = None if weights is None else weights
        if mask is not None:
            f = f[mask]
            w = None if w is None else w[mask]
        img = np.bincount(f, weights=w, minlength=self.W * self.H)
        return img.astype(np.float32).reshape(self.H, self.W)

    def last_event(self, sl):
        """Per pixel: time and polarity of the most recent event in ``sl`` (NaN / False where
        none). Events are time-sorted, so the last occurrence of a pixel is the latest."""
        T = np.full(self.W * self.H, np.nan)
        P = np.zeros(self.W * self.H, bool)
        f = self.flat[sl]
        if f.size:
            rev = f[::-1]
            u, first = np.unique(rev, return_index=True)
            idx = sl.start + (f.size - 1 - first)
            T[u] = self.t[idx]
            P[u] = self.on[idx]
        return T.reshape(self.H, self.W), P.reshape(self.H, self.W)


def _locked_scale(samples, pct=99.5, floor=1e-6):
    """One exposure for the whole shot: the *pct* percentile of the non-zero sample values."""
    vals = np.concatenate([np.ravel(s)[np.ravel(s) > 0] for s in samples] or [np.zeros(0)])
    return max(float(np.percentile(vals, pct)) if vals.size else 1.0, floor)


def _sample_times(r, n=12):
    k = np.unique(np.floor(np.linspace(0, r.n_frames - 1, min(n, r.n_frames)) + 1e-9).astype(int))
    return [r.times[i] for i in k]


# ------------------------------------------------------------------ base class
class Look:
    """Base class: a named, self-describing visualization method.

    Subclasses set ``name``, ``title``, ``description``, ``PARAMS`` (a list of
    :class:`~gottlux.detectors.base.Param`, so the GUI builds their panel automatically) and
    implement :meth:`render`. ``native = True`` looks draw at sensor resolution and the
    renderer resamples; ``False`` looks draw straight at the output size (3-D points stay
    crisp).
    """
    name = ""
    title = ""
    description = ""
    PARAMS: list = []
    native = True
    heavy = False

    def defaults(self) -> dict:
        return {p.key: p.default for p in self.PARAMS}

    def resolve(self, params: dict | None) -> dict:
        P = self.defaults()
        specs = {p.key: p for p in self.PARAMS}
        for k, v in (params or {}).items():
            if k in specs:
                P[k] = specs[k].coerce(v)
        return P

    def memory_s(self, P) -> float:
        """How much recording time before a frame the look reads."""
        return 0.0

    def n_frames(self, r, P):
        """Override the shot's frame count (``None`` = the shot's timing)."""
        return None

    def canvas_size(self, W, H, P) -> tuple:
        """The look's drawing size before resampling (native looks) or its aspect."""
        return W, H

    def prepare(self, ev: Events, r, P) -> dict:
        return {}

    def render(self, ev: Events, k: int, tau: float, P, st, colors, size) -> list:
        raise NotImplementedError

    def stamp(self, k: int, tau: float, P, st) -> str:
        return f"t = {tau:.3f} s"

    def uses_polarity(self, P) -> bool:
        """True when ON/OFF colours carry meaning (the renderer shows the key only then)."""
        return True


_REGISTRY: dict = {}


def register(cls):
    _REGISTRY[cls.name] = cls()
    return cls


def get_look(name: str) -> Look:
    if name not in _REGISTRY:
        raise KeyError(f"unknown look {name!r}; available: {', '.join(_REGISTRY)}")
    return _REGISTRY[name]


def list_looks() -> dict:
    return dict(_REGISTRY)


# ------------------------------------------------------------------ the looks
@register
class Trails(Look):
    name = "trails"
    title = "Polarity trails"
    description = ("Every event leaves a light that fades as exp(-age/T): ON events in one "
                   "colour, OFF in the other. Motion draws comet tails; a beating wing draws "
                   "alternating ON/OFF bands.")
    PARAMS = [
        Param("tau_ms", "Trail T", 6.0, 0.2, 500.0, 0.1, unit="ms", group="Look",
              help="Fade time constant: an event's light drops to 37 % after T."),
        Param("saturate", "Saturate at", 2.5, 0.2, 50.0, 0.1, group="Look",
              help="Summed (faded) events per pixel that reach full brightness."),
        Param("polarity", "Polarity", "both", kind="choice", choices=("both", "on", "off"),
              group="Look", help="Show both polarities or only one."),
    ]

    def memory_s(self, P):
        return 5e-3 * P["tau_ms"]

    def render(self, ev, k, tau, P, st, colors, size):
        T = 1e-3 * P["tau_ms"]
        sl = ev.span(tau - 5 * T, tau)
        w = np.exp(-(tau - ev.t[sl]) / T)
        on = ev.on[sl]
        layers = []
        if P["polarity"] in ("both", "on"):
            layers.append((np.clip(ev.splat(sl, w, on) / P["saturate"], 0, 1), colors["on"]))
        if P["polarity"] in ("both", "off"):
            layers.append((np.clip(ev.splat(sl, w, ~on) / P["saturate"], 0, 1), colors["off"]))
        return layers


@register
class Heat(Look):
    name = "heat"
    title = "Accumulated frame"
    description = ("The classic event frame: events per pixel in a sliding window [τ − W, τ], "
                   "tone-mapped through a colormap, with one exposure for the whole shot.")
    PARAMS = [
        Param("window_ms", "Window W", 10.0, 0.5, 1000.0, 0.5, unit="ms", group="Look",
              help="Accumulation window behind each frame."),
        Param("colormap", "Colormap", "inferno", kind="colormap", choices=COLORMAPS,
              group="Look"),
        Param("tone", "Tone map", "sqrt", kind="choice", choices=("linear", "sqrt", "log"),
              group="Look", help="Compresses bright regions so faint motion stays visible."),
        Param("signed", "ON − OFF", False, kind="bool", group="Look",
              help="Show the polarity balance in the palette's ON/OFF colours instead."),
    ]

    def memory_s(self, P):
        return 1e-3 * P["window_ms"]

    def uses_polarity(self, P):
        return bool(P["signed"])

    def _raw(self, ev, tau, P):
        sl = ev.span(tau - 1e-3 * P["window_ms"], tau)
        on = ev.on[sl]
        return ev.splat(sl, mask=on), ev.splat(sl, mask=~on)

    def prepare(self, ev, r, P):
        samples = []
        for tau in _sample_times(r):
            a, b = self._raw(ev, tau, P)
            samples.append(a + b if not P["signed"] else np.abs(a - b))
        return {"scale": _locked_scale(samples)}

    def render(self, ev, k, tau, P, st, colors, size):
        on, off = self._raw(ev, tau, P)
        if P["signed"]:
            d = (on - off) / st["scale"]
            return [(tone(np.clip(d, 0, 1), P["tone"]), colors["on"]),
                    (tone(np.clip(-d, 0, 1), P["tone"]), colors["off"])]
        v = tone((on + off) / st["scale"], P["tone"])
        return [(v, apply_lut(v, P["colormap"]))]


@register
class Surface(Look):
    name = "surface"
    title = "Time surface"
    description = ("Each pixel shows how recently it fired: exp(−(τ − T_last)/T), where "
                   "T_last is the time of the pixel's latest event. Edges glow at the front of "
                   "a moving object and fade behind it (the 'surface of active events').")
    PARAMS = [
        Param("tau_ms", "Decay T", 25.0, 1.0, 1000.0, 0.5, unit="ms", group="Look"),
        Param("color", "Colour by", "polarity", kind="choice", choices=("polarity", "colormap"),
              group="Look", help="Polarity of the last event, or a colormap of recency."),
        Param("colormap", "Colormap", "ice", kind="colormap", choices=COLORMAPS, group="Look"),
    ]

    def memory_s(self, P):
        return 5e-3 * P["tau_ms"]

    def uses_polarity(self, P):
        return P["color"] == "polarity"

    def render(self, ev, k, tau, P, st, colors, size):
        T = 1e-3 * P["tau_ms"]
        Tl, Pl = ev.last_event(ev.span(tau - 5 * T, tau))
        v = np.where(np.isfinite(Tl), np.exp(-(tau - np.nan_to_num(Tl, nan=-1e9)) / T), 0.0)
        v = v.astype(np.float32)
        if P["color"] == "colormap":
            return [(v, apply_lut(v, P["colormap"]))]
        return [(v * Pl, colors["on"]), (v * ~Pl, colors["off"])]


@register
class Paint(Look):
    name = "paint"
    title = "Light painting"
    description = ("A long exposure that builds up over the shot: every event since the start, "
                   "brightness = log(events per pixel), hue = when the pixel last fired "
                   "(early → late along the colormap). Flight paths draw themselves.")
    PARAMS = [
        Param("colormap", "Time colormap", "ice", kind="colormap", choices=COLORMAPS,
              group="Look"),
        Param("min_count", "Min events", 2, 1, 20, 1, "int", group="Look",
              help="Pixels with fewer events stay dark (drops residual shot noise)."),
        Param("fade_s", "Fade after", 0.0, 0.0, 30.0, 0.05, unit="s", group="Look",
              help="0 = keep everything; otherwise only the last N seconds are painted."),
    ]

    def uses_polarity(self, P):
        return False

    def _sl(self, ev, tau, P):
        a = ev.t0 if P["fade_s"] <= 0 else max(ev.t0, tau - P["fade_s"])
        return ev.span(a, tau)

    def prepare(self, ev, r, P):
        cnt = ev.splat(self._sl(ev, ev.t1, P))
        return {"scale": _locked_scale([cnt], 99.0)}

    def render(self, ev, k, tau, P, st, colors, size):
        sl = self._sl(ev, tau, P)
        cnt = ev.splat(sl)
        Tl, _ = ev.last_event(sl)
        frac = (np.nan_to_num(Tl, nan=ev.t0) - ev.t0) / max(ev.t1 - ev.t0, 1e-9)
        a = np.clip(np.log1p(cnt) / np.log1p(st["scale"]), 0, 1) * (cnt >= P["min_count"])
        return [(a.astype(np.float32), apply_lut(0.15 + 0.85 * frac, P["colormap"]))]


@register
class SpaceTime(Look):
    name = "spacetime"
    title = "Space-time cloud"
    description = ("The events of the last D milliseconds as points in a 3-D box (x, y, time); "
                   "the front face is 'now'. A beating wing becomes a row of fins along the "
                   "time axis, a moving target a tube. Set Orbit to swing the camera around.")
    native = False
    heavy = True
    PARAMS = [
        Param("depth_ms", "Depth D", 300.0, 5.0, 3000.0, 5.0, unit="ms", group="Look",
              help="Length of the time axis."),
        Param("time_scale", "Time stretch", 1.6, 0.3, 6.0, 0.1, group="Look",
              help="Length of the time axis relative to the sensor's larger side."),
        Param("azimuth", "Azimuth", -55.0, -180.0, 180.0, 1.0, unit="°", group="Camera"),
        Param("elevation", "Elevation", 22.0, -80.0, 80.0, 1.0, unit="°", group="Camera"),
        Param("orbit_deg", "Orbit", 0.0, -720.0, 720.0, 5.0, unit="°", group="Camera",
              help="Total camera rotation over the shot (360 = one full orbit)."),
        Param("freeze", "Freeze time", False, kind="bool", group="Camera",
              help="Hold the cloud at the shot's end [t1 − D, t1] and let only the camera move "
                   "(with Orbit: a turntable of one moment)."),
        Param("perspective", "Perspective", 0.35, 0.0, 0.9, 0.05, group="Camera"),
        Param("color", "Colour by", "polarity", kind="choice", choices=("polarity", "age"),
              group="Look"),
        Param("colormap", "Age colormap", "ice", kind="colormap", choices=COLORMAPS,
              group="Look"),
        Param("box", "Box", True, kind="bool", group="Look", help="Draw the space-time box."),
        Param("point_px", "Point size", 2, 1, 4, 1, "int", unit="px", group="Look"),
        Param("brightness", "Brightness", 1.0, 0.1, 5.0, 0.1, group="Look"),
        Param("max_points", "Max points", 120000, 5000, 2000000, 5000, "int", group="Look",
              help="Subsample above this many events per frame (keeps rendering quick)."),
    ]

    def memory_s(self, P):
        return 1e-3 * P["depth_ms"]

    def uses_polarity(self, P):
        return P["color"] == "polarity"

    def stamp(self, k, tau, P, st):
        if P["freeze"]:
            t1 = st["t1"]
            return f"t = {t1 - 1e-3 * P['depth_ms']:.3f} – {t1:.3f} s"
        return f"t = {tau:.3f} s"

    def canvas_size(self, W, H, P):
        return 4, 3

    def _camera(self, ev, k, r_frames, P):
        az = np.deg2rad(P["azimuth"] + P["orbit_deg"] * k / max(r_frames, 1))
        el = np.deg2rad(P["elevation"])
        D = P["time_scale"] * max(ev.W, ev.H)
        R = 0.5 * float(np.sqrt(ev.W ** 2 + ev.H ** 2 + D ** 2))
        return az, el, D, R

    def _project(self, X, Y, Z, az, el, R, P, size):
        x1 = X * np.cos(az) + Z * np.sin(az)
        z1 = -X * np.sin(az) + Z * np.cos(az)
        y2 = Y * np.cos(el) - z1 * np.sin(el)
        z2 = Y * np.sin(el) + z1 * np.cos(el)
        f = 1.0 / (1.0 + P["perspective"] * z2 / R)
        w, h = size
        s = 0.94 * min(w, h) / (2 * R)
        return w / 2 + s * x1 * f, h / 2 - s * y2 * f, f

    def prepare(self, ev, r, P):
        st = {"frames": r.n_frames, "t1": ev.t1}
        ks = np.linspace(0, r.n_frames - 1, min(10, r.n_frames))
        ks = np.unique(np.floor(ks + 1e-9).astype(int))
        samples = [self._accum(ev, int(i), r.times[i], P, st, r.draw_size)[0] for i in ks]
        st["scale"] = _locked_scale(samples, 98.0)      # measured at the drawing size
        return st

    def _accum(self, ev, k, tau, P, st, size, colors=None):
        w, h = size
        if P["freeze"]:
            tau = ev.t1
        sl = ev.span(tau - 1e-3 * P["depth_ms"], tau)
        idx = np.arange(sl.start, sl.stop)
        if idx.size > P["max_points"]:
            idx = idx[:: int(np.ceil(idx.size / P["max_points"]))]
        az, el, D, R = self._camera(ev, k, st["frames"], P)
        age = (tau - ev.t[idx]) / (1e-3 * P["depth_ms"])
        X = ev.x[idx] - ev.W / 2
        Y = -(ev.y[idx] - ev.H / 2)
        Z = age * D - D / 2
        u, v, f = self._project(X, Y, Z, az, el, R, P, size)
        ps = int(P["point_px"])
        ok = (u >= 0) & (u < w - ps) & (v >= 0) & (v < h - ps)
        base = (v[ok].astype(np.int64)) * w + u[ok].astype(np.int64)
        offs = [dy * w + dx for dy in range(ps) for dx in range(ps)]
        flat = np.concatenate([base + o for o in offs])
        wt = np.tile((1.0 - 0.6 * age[ok]) * f[ok], len(offs))
        acc = np.bincount(flat, weights=wt, minlength=w * h).reshape(h, w)
        if colors is None:
            return acc, None
        if P["color"] == "age":
            col = apply_lut(1.0 - age[ok], P["colormap"])
        else:
            col = np.where(ev.on[idx][ok][:, None], colors["on"], colors["off"])
        col = np.tile(col, (len(offs), 1))
        rgb = np.stack([np.bincount(flat, weights=wt * col[:, c], minlength=w * h)
                        for c in range(3)], 1).reshape(h, w, 3)
        return acc, rgb

    def _box(self, ev, k, P, st, size):
        az, el, D, R = self._camera(ev, k, st["frames"], P)
        w, h = size
        hx, hy, hz = ev.W / 2, ev.H / 2, D / 2
        c = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1)
                      for sz in (-1, 1)])
        edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
                 if np.sum(np.abs(c[i] - c[j]) > 1e-9) == 1]
        img = np.zeros((h, w), np.float32)
        for i, j in edges:
            front = c[i][2] < 0 and c[j][2] < 0          # the "now" face
            s = np.linspace(0, 1, 400)[:, None]
            pts = c[i] * (1 - s) + c[j] * s
            u, v, _ = self._project(pts[:, 0], pts[:, 1], pts[:, 2], az, el, R, P, size)
            ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            img[v[ok].astype(int), u[ok].astype(int)] = 0.55 if front else 0.22
        return img

    def render(self, ev, k, tau, P, st, colors, size):
        acc, rgb = self._accum(ev, k, tau, P, st, size, colors)
        a = np.clip(P["brightness"] * acc / st["scale"], 0, 1).astype(np.float32)
        col = rgb / np.maximum(acc, 1e-9)[..., None]
        layers = [(a, col.astype(np.float32))]
        if P["box"]:
            guide = np.full(3, 0.75 if colors["bg"].mean() < 0.5 else 0.25, np.float32)
            layers.insert(0, (self._box(ev, k, P, st, size), guide))
        return layers


@register
class Strobe(Look):
    name = "strobe"
    title = "Event stroboscope"
    description = ("For a periodic target (wingbeat, rotor): measure its frequency f0, fold "
                   "every event by its phase t·f0 mod 1, and play the phase bins as frames. "
                   "Dozens of cycles stack onto one, like a strobe light freezing a fan. "
                   "Crop to the target first so f0 is measured on it.")
    PARAMS = [
        Param("f0_hz", "f0 (0 = auto)", 0.0, 0.0, 2000.0, 0.1, unit="Hz", group="Look",
              help="Fold frequency; 0 measures it from the event-rate spectrum in the band."),
        Param("band_lo", "Search from", 15.0, 1.0, 2000.0, 1.0, unit="Hz", group="Look"),
        Param("band_hi", "Search to", 150.0, 2.0, 4000.0, 1.0, unit="Hz", group="Look"),
        Param("bins", "Frames per cycle", 24, 4, 120, 1, "int", group="Look"),
        Param("width", "Phase window", 0.08, 0.01, 0.5, 0.01, group="Look",
              help="Fraction of a cycle each frame collects (> 1/bins overlaps → smoother)."),
        Param("cycles", "Loops", 2, 1, 20, 1, "int", group="Look",
              help="How many times the cycle repeats in the output."),
    ]

    def n_frames(self, r, P):
        return int(P["bins"] * P["cycles"])

    @staticmethod
    def estimate_f0(t, t0, t1, lo, hi, on=None):
        """The fold frequency: the strongest line in [lo, hi] Hz of the event rate.

        A target that turns ON at one phase and OFF half a cycle later (a blinking light,
        often a wing) puts *two* bursts per cycle into the total event rate, whose strongest
        line is then 2·f0. With polarities given (*on*), the ON − OFF rate is added — its
        fundamental is f0 — and a peak whose half frequency still carries at least 30 % of
        its power is taken to be that harmonic.
        """
        fs = max(2000.0, 4.0 * hi)
        n = max(int((t1 - t0) * fs), 8)
        nfft = 1 << int(np.ceil(np.log2(max(n, 2) * 16)))
        win = np.hanning(n)

        def power(weights=None):
            sig = np.histogram(t, bins=n, range=(t0, t1), weights=weights)[0].astype(float)
            P = np.abs(np.fft.rfft((sig - sig.mean()) * win, nfft)) ** 2
            return P / max(P[1:].max(), 1e-30)

        S = power()
        if on is not None:
            S = S + power(np.where(on, 1.0, -1.0))
        f = np.fft.rfftfreq(nfft, 1 / fs)
        band = np.where((f >= lo) & (f <= hi))[0]
        if band.size < 3:
            return float("nan")
        i = int(band[np.argmax(S[band])])
        half = int(np.floor(i / 2 + 0.5))           # half-up, as in the web port
        if f[half] >= lo and half > 1:
            j = half - 2 + int(np.argmax(S[half - 2:half + 3]))
            if S[j] >= 0.3 * S[i]:
                i = j
        if 0 < i < S.size - 1:
            den = S[i - 1] - 2 * S[i] + S[i + 1]
            d = 0.5 * (S[i - 1] - S[i + 1]) / den if den else 0.0
        else:
            d = 0.0
        return float(f[i] + d * (f[1] - f[0]))

    def prepare(self, ev, r, P):
        sl = ev.span(ev.t0, ev.t1)
        f0 = P["f0_hz"] or self.estimate_f0(ev.t[sl], ev.t0, ev.t1, P["band_lo"], P["band_hi"],
                                            on=ev.on[sl])
        if not np.isfinite(f0) or f0 <= 0:
            f0 = 0.5 * (P["band_lo"] + P["band_hi"])
        st = {"f0": f0, "sl": sl, "phase": (ev.t[sl] * f0) % 1.0}
        samples = [self._raw(ev, (b + 0.5) / P["bins"], P, st)[0] for b in range(P["bins"])]
        st["scale"] = _locked_scale(samples, 99.5)
        return st

    def _raw(self, ev, phi, P, st):
        d = np.abs(((st["phase"] - phi + 0.5) % 1.0) - 0.5)
        half = 0.5 * P["width"]
        m = d < half
        w = (1.0 - d / half) * m
        on = ev.on[st["sl"]]
        return ev.splat(st["sl"], w, on & m), ev.splat(st["sl"], w, ~on & m)

    def render(self, ev, k, tau, P, st, colors, size):
        phi = ((k % P["bins"]) + 0.5) / P["bins"]
        on, off = self._raw(ev, phi, P, st)
        s = st["scale"]
        return [(np.clip(on / s, 0, 1), colors["on"]), (np.clip(off / s, 0, 1), colors["off"])]

    def stamp(self, k, tau, P, st):
        deg = int(round(360.0 * ((k % P["bins"]) + 0.5) / P["bins"]))
        return f"f0 = {st['f0']:.1f} Hz · phase {deg}°"


@register
class Waterfall(Look):
    name = "waterfall"
    title = "Waterfall (kymograph)"
    description = ("The live frame on top and, below it, the last H milliseconds collapsed "
                   "along one axis and stacked by time (newest row at the top). A moving target "
                   "draws a slanted track, a beating wing a ladder of stripes.")
    PARAMS = [
        Param("axis", "Collapse", "y", kind="choice", choices=("y", "x"), group="Look",
              help="'y': history of each column (strip below the frame); "
                   "'x': history of each row (strip to the right)."),
        Param("history_ms", "History H", 400.0, 10.0, 10000.0, 10.0, unit="ms", group="Look"),
        Param("rows", "History px", 160, 32, 1024, 8, "int", group="Look"),
        Param("tau_ms", "Frame trail", 6.0, 0.2, 200.0, 0.1, unit="ms", group="Look"),
    ]

    def memory_s(self, P):
        return 1e-3 * max(P["history_ms"], 5 * P["tau_ms"])

    def canvas_size(self, W, H, P):
        return (W, H + P["rows"]) if P["axis"] == "y" else (W + P["rows"], H)

    def _hist(self, ev, tau, P):
        Hs = 1e-3 * P["history_ms"]
        sl = ev.span(tau - Hs, tau)
        age = np.clip(((tau - ev.t[sl]) / Hs * P["rows"]).astype(np.int64), 0, P["rows"] - 1)
        pos = ev.x[sl] if P["axis"] == "y" else ev.y[sl]
        L = ev.W if P["axis"] == "y" else ev.H
        flat = age * L + pos
        on = ev.on[sl]
        a = np.bincount(flat[on], minlength=P["rows"] * L).reshape(P["rows"], L)
        b = np.bincount(flat[~on], minlength=P["rows"] * L).reshape(P["rows"], L)
        return a.astype(np.float32), b.astype(np.float32)

    def prepare(self, ev, r, P):
        st = {}
        hs = [sum(self._hist(ev, tau, P)) for tau in _sample_times(r)]
        st["hscale"] = _locked_scale(hs, 99.5)
        return st

    def render(self, ev, k, tau, P, st, colors, size):
        trails = Trails().render(ev, k, tau, {"tau_ms": P["tau_ms"], "saturate": 2.5,
                                               "polarity": "both"}, {}, colors, size)
        a, b = self._hist(ev, tau, P)
        a, b = np.clip(a / st["hscale"], 0, 1), np.clip(b / st["hscale"], 0, 1)
        cw, ch = self.canvas_size(ev.W, ev.H, P)
        out = []
        for (img, col), hist in zip(trails, (a, b), strict=True):
            full = np.zeros((ch, cw), np.float32)
            full[:ev.H, :ev.W] = img
            if P["axis"] == "y":
                full[ev.H:, :] = hist
            else:
                full[:, ev.W:] = hist.T
            out.append((full, col))
        guide = np.zeros((ch, cw), np.float32)
        if P["axis"] == "y":
            guide[ev.H, :] = 0.5
        else:
            guide[:, ev.W] = 0.5
        out.append((guide, np.full(3, 0.6, np.float32)))
        return out


@register
class Flicker(Look):
    name = "flicker"
    title = "Flicker hue"
    description = ("Where the scene flickers and how fast: each cell's dominant frequency in "
                   "a sliding window (hue, turbo colormap over the band), opacity = spectral "
                   "SNR above the scene's noise floor, over dimmed live trails. Rotors and "
                   "wings light up in their own colour.")
    heavy = True
    PARAMS = [
        Param("window_ms", "Window", 250.0, 50.0, 2000.0, 10.0, unit="ms", group="Look",
              help="Spectrum window; longer = finer frequency resolution (1/window)."),
        Param("cell", "Cell", 8, 2, 32, 1, "int", unit="px", group="Look"),
        Param("band_lo", "Band from", 15.0, 1.0, 4000.0, 1.0, unit="Hz", group="Look"),
        Param("band_hi", "Band to", 150.0, 2.0, 5000.0, 1.0, unit="Hz", group="Look"),
        Param("underlay", "Trails", 0.5, 0.0, 1.0, 0.05, group="Look",
              help="Brightness of the live trails underneath."),
    ]

    def memory_s(self, P):
        return 1e-3 * P["window_ms"]

    def render(self, ev, k, tau, P, st, colors, size):
        from gottlux.core import frequency as fq
        from gottlux.io.recording import EventWindow
        from gottlux.viz.theme import flicker_rgba
        sl = ev.span(tau - 1e-3 * P["window_ms"], tau)
        layers = [(a * P["underlay"], c) for a, c in Trails().render(
            ev, k, tau, {"tau_ms": 6.0, "saturate": 2.5, "polarity": "both"}, {}, colors, size)]
        if sl.stop - sl.start < 50:
            return layers
        win = EventWindow(ev.x[sl], ev.y[sl], ev.on[sl].astype(np.uint8), ev.t_us[sl],
                          ev.W, ev.H)
        lo, hi = P["band_lo"], max(P["band_hi"], P["band_lo"] + 1)
        fm = fq.flicker_map(win, fmin=lo, fmax=hi, fs=max(2000.0, 4 * hi), cell=P["cell"],
                            min_events_per_cell=12)
        rgba = flicker_rgba(fm.dominant_freq, fm.snr, lo, hi, cmap="turbo")
        c = P["cell"]
        up = np.kron(rgba, np.ones((c, c, 1)))[:ev.H, :ev.W].astype(np.float32)
        pad_h, pad_w = ev.H - up.shape[0], ev.W - up.shape[1]
        if pad_h or pad_w:
            up = np.pad(up, ((0, pad_h), (0, pad_w), (0, 0)))
        layers.append((0.85 * up[..., 3], up[..., :3]))
        return layers

    def stamp(self, k, tau, P, st):
        return f"t = {tau:.3f} s · hue {P['band_lo']:.0f}–{P['band_hi']:.0f} Hz"

    def uses_polarity(self, P):
        return False


@register
class Radar(Look):
    name = "radar"
    title = "Radar sweep (rotating sensor)"
    description = ("For a spinning sensor: every event is placed at its world bearing on a "
                   "360° dial — the rotation phase t/P plus the pixel's offset across the field "
                   "of view — with the sensor row as radius (sky inside, ground at the rim, like "
                   "a fisheye looking up). Events fade like radar phosphor; the wedge is where "
                   "the sensor is looking now.")
    native = False
    PARAMS = [
        Param("period_s", "Spin period (0 = auto)", 0.0, 0.0, 30.0, 0.0001, unit="s",
              group="Look", help="Rotation period; 0 estimates it from the event-rate "
                                 "periodicity (needs two or more revolutions)."),
        Param("fov_deg", "Field of view", 58.0, 5.0, 180.0, 0.5, unit="°", group="Look",
              help="Horizontal field of view of the sensor (GenX320 default 58°)."),
        Param("direction", "Rotation", "cw", kind="choice", choices=("cw", "ccw"),
              group="Look", help="Direction the boresight turns on the dial."),
        Param("mirror", "Column sign", "auto", kind="choice", choices=("auto", "+1", "-1"),
              group="Look", help="Sign of the pixel-column → bearing offset; 'auto' picks the "
                                 "sharper panorama (the wrong sign smears static objects over "
                                 "twice the field of view)."),
        Param("persistence_ms", "Persistence", 250.0, 5.0, 20000.0, 5.0, unit="ms",
              group="Look", help="Phosphor fade time constant."),
        Param("radius", "Radius", "ground-out", kind="choice",
              choices=("ground-out", "sky-out"), group="Look",
              help="Which end of the sensor's rows sits at the rim."),
        Param("max_points", "Max points", 250000, 10000, 5000000, 10000, "int", group="Look",
              help="Subsample above this many events per frame (keeps rendering quick)."),
        Param("saturate", "Saturate at", 3.0, 0.2, 100.0, 0.1, group="Look"),
        Param("dial", "Dial", True, kind="bool", group="Look",
              help="Range rings, 30° bearing ticks and the field-of-view wedge."),
    ]

    def memory_s(self, P):
        return 3e-3 * P["persistence_ms"]                  # e^-3 = 5 %

    def canvas_size(self, W, H, P):
        return 1, 1

    @staticmethod
    def _bearing(t, x, W, t_ref, period, direction, mirror, fov):
        return (direction * 360.0 * (t - t_ref) / period
                + mirror * (x / W - 0.5) * fov) % 360.0

    def prepare(self, ev, r, P):
        period = P["period_s"]
        if period <= 0:
            # the whole recording has the most revolutions (a 3 % period error drifts the
            # bearings by ~11° per revolution); fall back to the shot's own events
            from gottlux.io.telemetry import estimate_spin_period_s
            t_all = np.asarray(r.rec.t[::max(1, r.rec.n // 2_000_000)], np.float64) * 1e-6
            period, _ = estimate_spin_period_s(t_all)
            if not period:
                period, _ = estimate_spin_period_s(ev.t)
            period = float(period or 1.0)
        direction = 1.0 if P["direction"] == "cw" else -1.0
        t_ref = float(r.rec.t_start_s)        # bearings from the recording start: same dial
        if P["mirror"] == "auto":             # orientation whichever span is shot
            sl = ev.span(ev.t0, ev.t1)
            step = max(1, (sl.stop - sl.start) // 400_000)
            ts, xs, ys = ev.t[sl][::step], ev.x[sl][::step], ev.y[sl][::step]
            score = {}
            for m in (1.0, -1.0):
                b = self._bearing(ts, xs, ev.W, t_ref, period, direction, m, P["fov_deg"])
                h = np.histogram2d(b, ys, bins=[720, max(ev.H // 2, 1)],
                                   range=[[0, 360], [0, ev.H]])[0]
                score[m] = float((h ** 2).sum())               # sharper → more concentrated
            mirror = max(score, key=score.get)
        else:
            mirror = float(P["mirror"])
        st = {"period": period, "direction": direction, "mirror": mirror, "t_ref": t_ref}
        w, h = r.draw_size
        yy, xx = np.mgrid[0:h, 0:w]
        c = (w - 1) / 2.0
        st["R"] = 0.47 * min(w, h)
        st["r0"] = 0.18 * st["R"]
        st["rad"] = np.hypot(xx - c, yy - c)
        st["ang"] = np.degrees(np.arctan2(xx - c, -(yy - c))) % 360.0   # clockwise from up
        st["inside"] = (st["rad"] <= st["R"]) & (st["rad"] >= st["r0"])
        rings = np.zeros((h, w), np.float32)                  # static: drawn once per shot
        for frac in (0.0, 0.5, 1.0):
            rings[np.abs(st["rad"] - (st["r0"] + frac * (st["R"] - st["r0"]))) < 0.7] = 0.25
        ticks = ((np.abs(((st["ang"] + 15.0) % 30.0) - 15.0) < 0.5)
                 & (st["rad"] > st["R"] * 0.94) & (st["rad"] <= st["R"]))
        rings[ticks] = 0.5
        st["rings"] = rings
        samples = [sum(self._accum(ev, tau, P, st, (w, h))) for tau in _sample_times(r, 8)]
        st["scale"] = _locked_scale(samples, 97.0)
        return st

    def _accum(self, ev, tau, P, st, size):
        w, h = size
        T = 1e-3 * P["persistence_ms"]
        sp = ev.span(tau - 3 * T, tau)
        sl = slice(sp.start, sp.stop, max(1, -(-(sp.stop - sp.start) // P["max_points"])))
        b = np.radians(self._bearing(ev.t[sl], ev.x[sl], ev.W, st["t_ref"], st["period"],
                                     st["direction"], st["mirror"], P["fov_deg"]))
        row = (ev.y[sl] + 0.5) / ev.H
        rr = st["r0"] + (st["R"] - st["r0"]) * (row if P["radius"] == "ground-out" else 1.0 - row)
        c = (w - 1) / 2.0
        u = (c + rr * np.sin(b)).astype(np.int64)
        v = (c - rr * np.cos(b)).astype(np.int64)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        wt = np.exp(-(tau - ev.t[sl]) / T)
        flat = v * w + u
        on = ev.on[sl]
        a = np.bincount(flat[ok & on], weights=wt[ok & on], minlength=w * h).reshape(h, w)
        o = np.bincount(flat[ok & ~on], weights=wt[ok & ~on], minlength=w * h).reshape(h, w)
        return a.astype(np.float32), o.astype(np.float32)

    def render(self, ev, k, tau, P, st, colors, size):
        a, o = self._accum(ev, tau, P, st, size)
        s = st["scale"] / P["saturate"] * 3.0
        layers = []
        if P["dial"]:
            boresight = (st["direction"] * 360.0 * (tau - st["t_ref"]) / st["period"]) % 360.0
            d = np.abs(((st["ang"] - boresight + 180.0) % 360.0) - 180.0)
            inside = st["inside"]
            wedge = (inside & (d <= P["fov_deg"] / 2)).astype(np.float32) * 0.10
            edge = (inside & (np.abs(d - P["fov_deg"] / 2) < 0.6)).astype(np.float32) * 0.35
            guide = np.full(3, 0.75 if colors["bg"].mean() < 0.5 else 0.25, np.float32)
            layers += [(wedge + edge, colors["on"]), (st["rings"], guide)]
        layers += [(np.clip(a / s, 0, 1), colors["on"]), (np.clip(o / s, 0, 1), colors["off"])]
        return layers

    def stamp(self, k, tau, P, st):
        return f"t = {tau:.3f} s · {st['period']:.3f} s/rev"


def look_specs() -> dict:
    """Every look's title, description and parameter specs as plain JSON-able data — the
    single source the web Studio's ``web/js/looks.js`` spec block is generated from
    (``python scripts/sync_web_studio.py``; a test keeps the two in step)."""
    out = {}
    for name, look in _REGISTRY.items():
        out[name] = {
            "title": look.title, "description": look.description, "native": look.native,
            "heavy": look.heavy,
            # a desktop "colormap" parameter is a plain choice of the web's own maps
            "params": [{"key": p.key, "label": p.label, "default": p.default, "lo": p.lo,
                        "hi": p.hi, "step": p.step,
                        "kind": "choice" if p.kind == "colormap" else p.kind,
                        "choices": list(p.choices), "unit": p.unit, "group": p.group,
                        "help": p.help} for p in look.PARAMS],
        }
    return out
