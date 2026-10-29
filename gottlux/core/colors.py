"""
colors.py — one registry of colour schemes for every picker, render path and export.

Two kinds of scheme, because event imagery uses colour for two different jobs:

* a **colormap** maps a magnitude in ``[0, 1]`` (an event count, a recency, a frequency) to
  a colour: a ramp of colour *stops*, interpolated linearly into a 256-entry table;
* a **palette** names the colours of the two polarities and the background — ``on``,
  ``off`` and ``bg`` — for the views that draw ON and OFF events in their own colours (the
  Studio's trails, time surface, stroboscope, space-time cloud, radar).

The built-ins are matplotlib's perceptual maps plus GottLUX's own ramps (``ice``,
``ember``, ``gottlux_events``, ``gottlux_polarity``) and the Studio palettes. **User
schemes** are added from the GUI (*View → Colour schemes…*), from code
(:func:`add_colormap`, :func:`add_palette`) or by importing a shared ``.json`` file
(:func:`import_schemes`); they are saved to :func:`schemes_path` and appear in every
colormap and palette picker at once. Every colormap — built-in or user — is also registered
with matplotlib under its name, so figures, the live viewers and video exports resolve a
user scheme exactly as they resolve ``inferno``.

The file format (``version`` 1)::

    {"version": 1,
     "colormaps": {"sunset": {"stops": ["#1a0a2e", "#c2185b", "#ffb347"]}},
     "palettes":  {"neon":   {"on": "#39ff14", "off": "#ff2bd6", "bg": "#05050a"}}}

``stops`` may carry ``positions`` (ascending, 0 … 1) for uneven spacing.

Mapping a value through a table here is **bit-exact** with matplotlib's own
``Colormap.__call__`` followed by ``(rgb * 255).astype(uint8)`` — the index is
``min(floor(v · 256), 255)`` — so :func:`apply_u8` can replace the matplotlib call on the
per-frame paths without changing a single pixel.

A broken user file never breaks the program: it is copied aside as
``colors.json.unreadable-<epoch>`` and the built-ins stay available.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import warnings
import weakref

import numpy as np

#: GottLUX's own ramps, as colour stops (dark = low/early → light = high/late).
_GOTTLUX_RAMPS = {
    "ice": ["#0a1a33", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb",
            "#ffffff"],
    "ember": ["#120404", "#4a0c06", "#8f2306", "#d24a0c", "#f58a1f", "#ffc65a", "#fff3c4"],
    # event map: deep navy → magenta → amber → white (good contrast on dark backgrounds)
    "gottlux_events": ["#05060f", "#1a1145", "#6a1b9a", "#d81b60", "#ff8f00", "#fff3c4",
                       "#ffffff"],
    # diverging polarity map: blue (OFF) → black → red (ON)
    "gottlux_polarity": ["#2962ff", "#0a0a14", "#ff1744"],
}

#: The colormaps every picker offers, in menu order (user schemes follow these).
BUILTIN_COLORMAPS = (
    "inferno", "viridis", "magma", "plasma", "cividis", "gray", "turbo",
    "ice", "ember", "gottlux_events",
    "coolwarm", "RdBu", "bwr", "seismic", "PiYG", "Spectral", "gottlux_polarity",
)

#: Diverging maps (a neutral midpoint): pickers can list them apart, and polarity views
#: default to one.
DIVERGING = frozenset({"coolwarm", "RdBu", "bwr", "seismic", "PiYG", "Spectral",
                       "gottlux_polarity"})

#: ON/OFF colour pairs + background. "fire-ice" is the default: the orange/blue pair that
#: passes the colour-vision checks on its background (OKLab ΔE 26.7 under protanopia).
BUILTIN_PALETTES = {
    "fire-ice": {"on": "#ff7a3d", "off": "#3f97ff", "bg": "#05070a"},
    "ember":    {"on": "#ffb347", "off": "#c2185b", "bg": "#07040a"},
    "aurora":   {"on": "#7cf5c4", "off": "#8a6cff", "bg": "#03060a"},
    "gottlux":  {"on": "#39c5cf", "off": "#39c5cf", "bg": "#05080b"},
    "mono":     {"on": "#ffffff", "off": "#ffffff", "bg": "#000000"},
    "paper":    {"on": "#1a1a19", "off": "#1a1a19", "bg": "#f6f5f1"},
    "print":    {"on": "#c0392b", "off": "#1f5fbf", "bg": "#fcfcfb"},
}

_FALLBACK = "inferno"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.+()-]{0,39}$")
_HEX_RE = re.compile(r"^#?[0-9A-Fa-f]{6}$")

_lock = threading.RLock()
_user_maps: dict = {}          # name -> {"stops": [...], "positions": [...] | None}
_user_pals: dict = {}          # name -> {"on", "off", "bg"}
_loaded_from: str | None = None
_luts: dict = {}               # name -> (256, 3) float32
_luts_u8: dict = {}            # name -> (256, 3) uint8
_luts_packed: dict = {}        # name -> (256,) uint32 (RGB + pad)
_listeners: list = []
_warned_paths: set = set()
_mpl_registered: set = set()   # user names this module registered with matplotlib


def _drop_tables(name):
    for cache in (_luts, _luts_u8, _luts_packed):
        cache.pop(name, None)


# ====================================================================================
# Where user schemes live
# ====================================================================================
def schemes_path() -> str:
    """The user-schemes file: ``$GOTTLUX_COLORS`` if set, else ``colors.json`` in the
    per-user GottLUX folder (``%LOCALAPPDATA%\\gottlux`` on Windows, the XDG config home
    + ``/gottlux`` elsewhere)."""
    env = os.environ.get("GOTTLUX_COLORS", "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(env))
    if os.name == "nt":
        from gottlux.io.paths import platform_root
        root = platform_root()
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"),
                                                                  ".config")
        root = os.path.join(base, "gottlux")
    return os.path.join(root, "colors.json")


# ====================================================================================
# Validation helpers
# ====================================================================================
def hex_to_rgb01(c) -> np.ndarray:
    """``'#rrggbb'`` (or ``(r, g, b)`` 0–255) → float32 RGB in [0, 1]."""
    if isinstance(c, str):
        s = c.strip().lstrip("#")
        if not _HEX_RE.match(s):
            raise ValueError(f"not a #rrggbb colour: {c!r}")
        return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0
    v = np.asarray(c, np.float32).reshape(-1)[:3]
    if v.size != 3:
        raise ValueError(f"not an RGB colour: {c!r}")
    return np.clip(v / (255.0 if v.max() > 1.0 else 1.0), 0, 1).astype(np.float32)


def rgb01_to_hex(c) -> str:
    v = np.clip(np.asarray(c, float).reshape(-1)[:3], 0, 1)
    return "#" + "".join(f"{int(round(x * 255)):02x}" for x in v)


def _norm_hex(c) -> str:
    return rgb01_to_hex(hex_to_rgb01(c))


def valid_name(name) -> bool:
    return isinstance(name, str) and bool(_NAME_RE.match(name.strip()))


def _check_new_name(name: str, kind: str):
    name = str(name).strip()
    if not valid_name(name):
        raise ValueError("a scheme name is 1–40 letters, digits, spaces or _ . + - ( ) "
                         "(starting with a letter or digit)")
    if kind == "colormap" and _reserved_colormap(name):
        raise ValueError(f"{name!r} is a built-in colormap; pick another name")
    if kind == "palette" and name in BUILTIN_PALETTES:
        raise ValueError(f"{name!r} is a built-in palette; pick another name")
    return name


def _clean_map(entry) -> dict:
    stops = [_norm_hex(c) for c in (entry.get("stops") or [])]
    if len(stops) < 2:
        raise ValueError("a colormap needs at least two colour stops")
    if len(stops) > 64:
        raise ValueError("a colormap has at most 64 stops")
    pos = entry.get("positions")
    if pos is not None:
        pos = [float(p) for p in pos]
        if len(pos) != len(stops):
            raise ValueError("positions must match the stops one for one")
        if any(b < a for a, b in zip(pos, pos[1:], strict=False)) or pos[0] < 0 or pos[-1] > 1:
            raise ValueError("positions must ascend within 0 … 1")
        if pos[0] != 0.0 or pos[-1] != 1.0:
            raise ValueError("positions must start at 0 and end at 1")
    return {"stops": stops, "positions": pos}


def _clean_pal(entry) -> dict:
    return {k: _norm_hex(entry[k]) for k in ("on", "off", "bg")}


# ====================================================================================
# Loading / saving the user file
# ====================================================================================
def _ensure_loaded():
    if _loaded_from != schemes_path():
        reload()


def reload() -> None:
    """(Re)read the user-schemes file. A missing file means no user schemes; an unreadable
    one is copied aside (``.unreadable-<epoch>``) once and ignored."""
    global _loaded_from
    path = schemes_path()
    maps, pals = {}, {}
    with _lock:
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                for name, entry in (data.get("colormaps") or {}).items():
                    try:
                        if valid_name(name) and not _reserved_colormap(name.strip()):
                            maps[name.strip()] = _clean_map(entry)
                    except Exception as e:
                        warnings.warn(f"colour scheme {name!r} in {path} skipped: {e}",
                                      RuntimeWarning, stacklevel=2)
                for name, entry in (data.get("palettes") or {}).items():
                    try:
                        if valid_name(name) and name.strip() not in BUILTIN_PALETTES:
                            pals[name.strip()] = _clean_pal(entry)
                    except Exception as e:
                        warnings.warn(f"palette {name!r} in {path} skipped: {e}",
                                      RuntimeWarning, stacklevel=2)
            except Exception as e:
                if path not in _warned_paths:
                    _warned_paths.add(path)
                    try:
                        import shutil
                        shutil.copy2(path, f"{path}.unreadable-{int(time.time())}")
                    except OSError:
                        pass
                    warnings.warn(f"could not read the colour schemes in {path} ({e}); "
                                  "using the built-ins", RuntimeWarning, stacklevel=2)
        for name in list(_user_maps) + list(maps):
            _drop_tables(name)
        _user_maps.clear(); _user_maps.update(maps)
        _user_pals.clear(); _user_pals.update(pals)
        _loaded_from = path
        _register_with_matplotlib()
    _notify()


def save() -> str:
    """Write the user schemes atomically (``.tmp`` then ``os.replace``). Returns the path."""
    path = schemes_path()
    with _lock:
        data = {"version": 1,
                "colormaps": {k: {kk: vv for kk, vv in v.items() if vv is not None}
                              for k, v in _user_maps.items()},
                "palettes": dict(_user_pals)}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    return path


# ====================================================================================
# Change notification (pickers repopulate themselves)
# ====================================================================================
def subscribe(callback) -> None:
    """Call *callback()* whenever the set of schemes changes (added, removed, reloaded).

    Bound methods are held weakly, so a picker that subscribes never outlives its widget."""
    ref = weakref.WeakMethod(callback) if hasattr(callback, "__self__") else (lambda: callback)
    with _lock:
        if all(r() != callback for r in _listeners):
            _listeners.append(ref)


def unsubscribe(callback) -> None:
    with _lock:
        _listeners[:] = [r for r in _listeners if r() is not None and r() != callback]


def _notify():
    with _lock:
        refs = list(_listeners)
    for r in refs:
        cb = r()
        if cb is None:
            with _lock:
                if r in _listeners:
                    _listeners.remove(r)
            continue
        try:
            cb()
        except Exception:
            unsubscribe(cb)                 # a deleted widget's slot: drop it


# ====================================================================================
# Colormaps
# ====================================================================================
def _reserved_colormap(name) -> bool:
    """A name a user scheme may not take: a built-in, or a map matplotlib already knows
    that this module did not register."""
    return (name in BUILTIN_COLORMAPS or name in _GOTTLUX_RAMPS
            or (_matplotlib_has(name) and name not in _mpl_registered))


def _matplotlib_has(name) -> bool:
    try:
        import matplotlib
        matplotlib.colormaps[name]
        return True
    except Exception:
        return False


def colormap_names(builtin: bool = True, user: bool = True) -> list:
    """Every colormap a picker should offer: the built-ins in menu order, then the user's
    in the order they were added."""
    _ensure_loaded()
    out = list(BUILTIN_COLORMAPS) if builtin else []
    if user:
        out += [n for n in _user_maps if n not in out]
    return out


def is_user_colormap(name) -> bool:
    _ensure_loaded()
    return name in _user_maps


def has_colormap(name) -> bool:
    """True for a built-in, a user scheme, or any other name matplotlib resolves."""
    _ensure_loaded()
    return isinstance(name, str) and (name in _user_maps or name in _GOTTLUX_RAMPS
                                      or _matplotlib_has(name))


def colormap_stops(name) -> list | None:
    """The hex stops of a stop-defined colormap (GottLUX or user), else ``None``."""
    _ensure_loaded()
    if name in _user_maps:
        return list(_user_maps[name]["stops"])
    if name in _GOTTLUX_RAMPS:
        return list(_GOTTLUX_RAMPS[name])
    return None


def _stops_table(stops, positions=None, n=256) -> np.ndarray:
    rgb = np.stack([hex_to_rgb01(c) for c in stops])
    x = np.linspace(0, 1, len(stops)) if positions is None else np.asarray(positions, float)
    g = np.linspace(0, 1, n)
    return np.stack([np.interp(g, x, rgb[:, i]) for i in range(3)], 1).astype(np.float32)


def lut(name, fallback: str = _FALLBACK) -> np.ndarray:
    """The ``(256, 3)`` float32 colour table of *name* (an unknown name → *fallback*)."""
    _ensure_loaded()
    t = _luts.get(name)
    if t is not None:
        return t
    with _lock:
        if name in _user_maps:
            e = _user_maps[name]
            t = _stops_table(e["stops"], e.get("positions"))
        elif name in _GOTTLUX_RAMPS:
            t = _stops_table(_GOTTLUX_RAMPS[name])
        else:
            try:
                import matplotlib
                t = np.asarray(matplotlib.colormaps[name](np.linspace(0, 1, 256))[:, :3],
                               np.float32)
            except Exception:
                if name == fallback:
                    raise
                return lut(fallback, fallback)
        _luts[name] = t
    return t


def lut_u8(name, fallback: str = _FALLBACK) -> np.ndarray:
    """The ``(256, 3)`` uint8 table matplotlib's ``(cmap(v)[..., :3] * 255).astype(uint8)``
    would produce for each of its 256 entries."""
    t = _luts_u8.get(name)
    if t is not None:
        return t
    with _lock:
        mpl_t = _mpl_table(name, fallback)
        t = (mpl_t * 255).astype(np.uint8)
        _luts_u8[name] = t
    return t


def _mpl_table(name, fallback):
    """The float64 RGB table matplotlib itself holds for *name* (registered user schemes
    included) — what bit-exactness with ``Colormap.__call__`` has to start from."""
    _ensure_loaded()
    try:
        import matplotlib
        cm = matplotlib.colormaps[name]
    except Exception:
        if name == fallback:
            return lut(name).astype(np.float64)
        return _mpl_table(fallback, fallback)
    if cm.N == 256:
        return np.asarray(cm(np.arange(256))[:, :3], np.float64)
    return np.asarray(cm(np.linspace(0, 1, 256))[:, :3], np.float64)


def apply_u8(v01, name, fallback: str = _FALLBACK) -> np.ndarray:
    """Map ``[0, 1]`` values through *name* → ``(..., 3)`` uint8 RGB.

    Bit-exact with ``(matplotlib.colormaps[name](np.clip(v, 0, 1))[..., :3] * 255)
    .astype(np.uint8)`` for every finite input — the scaling happens in the input's own
    dtype, as matplotlib's does — at a quarter of the cost: one gather of packed 4-byte
    table entries instead of a float64 RGBA image. NaN maps to the lowest entry.
    """
    v = np.asarray(v01)
    if v.dtype.kind != "f":
        v = v.astype(np.float64)
    with np.errstate(invalid="ignore"):
        idx = (np.clip(v, 0, 1) * 256).astype(np.int32)
    np.clip(idx, 0, 255, out=idx)                      # 1.0 → 255; NaN's INT_MIN → 0
    packed = _lut_packed(name, fallback)
    out = packed[idx].view(np.uint8).reshape(v.shape + (4,))
    return np.ascontiguousarray(out[..., :3])


def _lut_packed(name, fallback):
    """The uint8 table packed as one uint32 per entry (RGB + a pad byte), so a lookup
    gathers whole words."""
    t = _luts_packed.get(name)
    if t is None:
        t4 = np.zeros((256, 4), np.uint8)
        t4[:, :3] = lut_u8(name, fallback)
        t = t4.view(np.uint32).reshape(256)
        _luts_packed[name] = t
    return t


def mpl_colormap(name, fallback: str = _FALLBACK):
    """The matplotlib colormap for *name* (user schemes included); unknown → *fallback*."""
    _ensure_loaded()
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except (KeyError, TypeError):
        return matplotlib.colormaps[fallback]


def _register_with_matplotlib():
    """Make GottLUX ramps and user schemes resolvable through ``matplotlib.colormaps``
    (and drop user schemes that no longer exist)."""
    try:
        import matplotlib
        from matplotlib.colors import ListedColormap
    except Exception:                                  # pragma: no cover - core dep
        return
    reg = matplotlib.colormaps
    for name, stops in _GOTTLUX_RAMPS.items():
        if name not in reg:
            try:
                reg.register(ListedColormap(_stops_table(stops).astype(np.float64),
                                            name=name), name=name)
            except Exception:
                pass
    for name in list(_mpl_registered):
        if name not in _user_maps:
            try:
                reg.unregister(name)
            except Exception:
                pass
            _mpl_registered.discard(name)
    for name, e in _user_maps.items():
        cmap = ListedColormap(_stops_table(e["stops"], e.get("positions")).astype(np.float64),
                              name=name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)   # "Overwriting the cmap"
                reg.register(cmap, name=name, force=True)
            _mpl_registered.add(name)
        except Exception:
            pass


def add_colormap(name, stops, positions=None, persist: bool = True) -> str:
    """Add (or replace) the user colormap *name* from ≥ 2 colour stops. Returns the name."""
    _ensure_loaded()
    with _lock:
        if str(name).strip() not in _user_maps:
            name = _check_new_name(name, "colormap")
        name = str(name).strip()
        _user_maps[name] = _clean_map({"stops": list(stops), "positions": positions})
        _drop_tables(name)
        _register_with_matplotlib()
    if persist:
        save()
    _notify()
    return name


def remove_colormap(name, persist: bool = True) -> bool:
    """Delete a user colormap (built-ins cannot be removed). Returns True if removed."""
    _ensure_loaded()
    with _lock:
        if name not in _user_maps:
            return False
        del _user_maps[name]
        _drop_tables(name)
        _register_with_matplotlib()
    if persist:
        save()
    _notify()
    return True


# ====================================================================================
# Palettes (ON / OFF / background)
# ====================================================================================
def palette_names(builtin: bool = True, user: bool = True) -> list:
    _ensure_loaded()
    out = list(BUILTIN_PALETTES) if builtin else []
    if user:
        out += [n for n in _user_pals if n not in out]
    return out


def has_palette(name) -> bool:
    _ensure_loaded()
    return name in BUILTIN_PALETTES or name in _user_pals


def is_user_palette(name) -> bool:
    _ensure_loaded()
    return name in _user_pals


def palette(name, fallback: str = "fire-ice") -> dict:
    """``{"on", "off", "bg"}`` hex colours of *name* (unknown → *fallback*)."""
    _ensure_loaded()
    if name in _user_pals:
        return dict(_user_pals[name])
    return dict(BUILTIN_PALETTES.get(name, BUILTIN_PALETTES[fallback]))


def add_palette(name, on, off, bg, persist: bool = True) -> str:
    """Add (or replace) the user palette *name*. Returns the name."""
    _ensure_loaded()
    with _lock:
        if str(name).strip() not in _user_pals:
            name = _check_new_name(name, "palette")
        name = str(name).strip()
        _user_pals[name] = _clean_pal({"on": on, "off": off, "bg": bg})
    if persist:
        save()
    _notify()
    return name


def remove_palette(name, persist: bool = True) -> bool:
    _ensure_loaded()
    with _lock:
        if name not in _user_pals:
            return False
        del _user_pals[name]
    if persist:
        save()
    _notify()
    return True


# ====================================================================================
# Sharing schemes as files
# ====================================================================================
def export_schemes(path, colormaps=(), palettes=()) -> str:
    """Write the named user colormaps/palettes (built-in colormaps by their stops where
    they have them) to *path* in the user-file format, for sharing."""
    _ensure_loaded()
    data = {"version": 1, "colormaps": {}, "palettes": {}}
    for n in colormaps:
        if n in _user_maps:
            e = _user_maps[n]
            data["colormaps"][n] = {k: v for k, v in e.items() if v is not None}
        else:
            stops = colormap_stops(n)
            if stops is None:                          # a matplotlib map: sample it
                stops = [rgb01_to_hex(c) for c in lut(n)[::17]]
            data["colormaps"][n] = {"stops": stops}
    for n in palettes:
        data["palettes"][n] = palette(n)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
    return path


def import_schemes(path, persist: bool = True) -> list:
    """Add every scheme in a shared file. A name that clashes with a built-in gets a
    ``" (imported)"`` suffix. Returns the names added (colormaps then palettes)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    added = []
    for name, entry in (data.get("colormaps") or {}).items():
        n = str(name).strip()
        if _reserved_colormap(n):
            n = f"{n} (imported)"[:40]
        added.append(add_colormap(n, entry.get("stops") or [], entry.get("positions"),
                                  persist=False))
    for name, entry in (data.get("palettes") or {}).items():
        n = str(name).strip()
        if n in BUILTIN_PALETTES:
            n = f"{n} (imported)"[:40]
        added.append(add_palette(n, entry["on"], entry["off"], entry["bg"], persist=False))
    if persist and added:
        save()
    return added


# GottLUX ramps resolve through matplotlib from the first import on.
_register_with_matplotlib()
