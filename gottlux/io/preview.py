"""
preview.py — open a multi-GB ``.raw`` in seconds: index the whole file, decode only samples.

A cold open used to run the full sequential decode into the 4-bin cache before anything
rendered — minutes for a large recording. But almost none of that work is needed to *show*
the file: the encoded stream itself is a time-ordered word list punctuated by TIME_HIGH
markers, so with one cheap I/O pass we know the exact duration and a time→byte mapping for
every moment, and can then decode just a few sampled windows to have something playable.

Three pieces:

1. **Index** (:func:`scan_index`) — one expansion-free I/O pass recording every TIME_HIGH's
   (running-max value, payload word position), generalizing :func:`gottlux.io.rawcut.scan`
   beyond EVT2.1: EVT2.0 is the same scheme on 32-bit words, EVT3 needs sequential 12-bit
   wrap-epoch counting. Gives the full-file duration instantly.
2. **Slice decode** (:func:`decode_slice`) — map a time window to a word range through the
   index, seek there, and run the existing per-chunk decoder with a *seeded* state
   (``init_state(fmt, {'hi': …, 'seen': True})``), so a mid-file window decodes without
   replaying the file. Exact for EVT2.x; EVT3's y-row/vector state can't be seeded from an
   index, so EVT3 falls back to the normal full decode (no preview).
3. **:class:`PreviewRecording`** — beginning/middle/end slices concatenated into a normal
   :class:`~gottlux.io.recording.Recording`. ``duration_s``/``t_stop_s`` span the WHOLE
   file (from the index) so the transport bar covers it; ``coverage`` lists the decoded
   spans; seeking into an undecoded region triggers a bounded on-demand slice decode
   (~sub-second) inside :meth:`~PreviewRecording.window`.

Policy: the GUI loader shows a preview only for a **large, uncached** EVT2.x ``.raw``
(:func:`should_preview` — default threshold :data:`PREVIEW_THRESHOLD_MB`, env override
``GOTTLUX_PREVIEW_THRESHOLD_MB``, ``0``/negative disables), then continues the untouched
full ``cache.load`` in the background and swaps the finished Recording in. Library
behaviour (``gottlux.load``) is unchanged unless a preview is explicitly requested.
HDF5 recordings (``.h5``/``.hdf5``) deliberately take the **full-decode path** instead:
they have no raw byte stream to index (the events sit in gzip'd chunked datasets), the
first open streams them once into the same decode cache, and every later open is an
instant memmap hit — so the preview machinery would only ever matter for the very first
open of a very large ``.h5``.
"""
from __future__ import annotations

import os
import threading

import numpy as np

from gottlux.io import cache as _cache
from gottlux.io import decode as _dec
from gottlux.io import rawcut as _rawcut
from gottlux.io.paths import ext
from gottlux.io.recording import Recording

#: File size (MB) above which the GUI opens with a sampled preview first.
PREVIEW_THRESHOLD_MB = 200.0
_ENV_THRESHOLD = "GOTTLUX_PREVIEW_THRESHOLD_MB"

#: Sampled-open shape: N_SLICES windows of ~SLICE_S seconds (beginning / middle / end).
N_SLICES = 3
SLICE_S = 2.0
#: Hard cap on payload words per slice — bounds the read+decode for very dense files.
MAX_SLICE_WORDS = 8_000_000

#: window() auto-decodes an uncovered request only up to this span (an accumulation window
#: is tens of ms; a multi-second request would defeat the point of sampling).
ON_DEMAND_MAX_SPAN_S = 5.0
_ON_DEMAND_QUANTUM_US = 250_000        # round on-demand decodes out to ¼-s blocks

#: TIME_HIGH unit in µs per encoding (EVT3's 12-bit high counts 4096-µs epochs).
_HIGH_UNIT_US = {"evt21": 64, "evt2": 64, "evt3": 4096}
_PREROLL_US = 3_000_000                # >=3 s leading gap = corrupt pre-roll (cf. strip_preroll)


class UnsupportedPreview(Exception):
    """Raised when a file can't be preview-sampled — caller should run the normal full load."""


# ====================================================================================
# Policy
# ====================================================================================
def preview_threshold_bytes():
    """The preview activation threshold in bytes, or ``None`` when previews are disabled.

    Default :data:`PREVIEW_THRESHOLD_MB`; the ``GOTTLUX_PREVIEW_THRESHOLD_MB`` env var
    overrides it (a float, MB; ``0`` or negative disables previews entirely)."""
    mb = PREVIEW_THRESHOLD_MB
    v = os.environ.get(_ENV_THRESHOLD)
    if v is not None:
        try:
            mb = float(v)
        except ValueError:
            pass
    if mb <= 0:
        return None
    return int(mb * 2**20)


def should_preview(path) -> bool:
    """True iff opening *path* in the GUI should show a sampled preview first: a large
    (over-threshold), **uncached** EVT2.x ``.raw``. Cache hits are already instant, EVT3
    can't be slice-decoded exactly, HDF5 recordings have no raw byte index (they take the
    normal full decode into the cache — see the module docstring), and any error here
    just means 'no preview'."""
    thr = preview_threshold_bytes()
    if thr is None:
        return False
    try:
        p = str(path)
        if not p.lower().endswith(".raw") or not os.path.exists(ext(p)):
            return False
        if os.path.getsize(ext(p)) < thr:
            return False
        if _cache.has_valid_cache(p):
            return False
        meta, _ = _dec.parse_header(p)
        return _dec.detect_format(meta) in ("evt21", "evt2")
    except Exception:
        return False


# ====================================================================================
# Index: one expansion-free I/O pass -> every TIME_HIGH (running-max value, word position)
# ====================================================================================
def _scan_time_high_evt2(in_path, off, chunk_words=16_000_000):
    """EVT2.0 twin of :func:`rawcut.scan_time_high`: 32-bit words, type nibble at bit 28."""
    vals, poss, total = [], [], 0
    with open(ext(in_path), "rb") as f:
        f.seek(off)
        while True:
            buf = f.read(chunk_words * 4)
            if not buf:
                break
            got = len(buf) // 4
            w = np.frombuffer(buf[: got * 4], dtype="<u4")
            idx = np.nonzero((w >> np.uint32(28)) == np.uint32(0x8))[0]
            if idx.size:
                vals.append((w[idx] & np.uint32(0x0FFFFFFF)).astype(np.int64))
                poss.append(idx.astype(np.int64) + total)
            total += got
    if vals:
        th_vals = np.maximum.accumulate(np.concatenate(vals))
        th_pos = np.concatenate(poss)
    else:
        th_vals = np.zeros(0, np.int64)
        th_pos = np.zeros(0, np.int64)
    return th_vals, th_pos, total


def _scan_time_high_evt3(in_path, off, chunk_words=32_000_000):
    """EVT3 index pass: 16-bit words, 12-bit TIME_HIGH with sequential wrap-epoch counting
    (mirrors the epoch state in :func:`decode.chunk_evt3`) — still expansion-free."""
    vals, poss, total = [], [], 0
    epoch, last_raw = 0, 0
    with open(ext(in_path), "rb") as f:
        f.seek(off)
        while True:
            buf = f.read(chunk_words * 2)
            if not buf:
                break
            got = len(buf) // 2
            w = np.frombuffer(buf[: got * 2], dtype="<u2")
            idx = np.nonzero((w >> np.uint16(12)) == np.uint16(0x8))[0]
            if idx.size:
                raw = (w[idx] & np.uint16(0x0FFF)).astype(np.int64)
                prev = np.empty(raw.size, np.int64)
                prev[0] = last_raw
                prev[1:] = raw[:-1]
                ep = epoch + np.cumsum((raw < prev).astype(np.int64))   # +1 per 12-bit wrap
                vals.append(ep * 4096 + raw)
                poss.append(idx.astype(np.int64) + total)
                epoch = int(ep[-1])
                last_raw = int(raw[-1])
            total += got
    if vals:
        th_vals = np.maximum.accumulate(np.concatenate(vals))
        th_pos = np.concatenate(poss)
    else:
        th_vals = np.zeros(0, np.int64)
        th_pos = np.zeros(0, np.int64)
    return th_vals, th_pos, total


def scan_index(in_path, progress=None) -> dict:
    """Index any supported ``.raw`` in one I/O-bound pass (no event expansion).

    Returns ``{fmt, unit_us, th_vals, th_pos, total_words, off, meta, width, height,
    origin_idx, origin_high, duration_s, n_time_high}`` — the instant full-file duration
    plus the time→word mapping every slice decode needs. ``th_vals`` are running-max
    TIME_HIGH values (in ``unit_us`` units), ``th_pos`` their payload word positions.
    """
    meta, off = _dec.parse_header(in_path)
    fmt = _dec.detect_format(meta)
    width, height = _dec.geometry(meta)
    if fmt == "evt21":
        th_vals, th_pos, total = _rawcut.scan_time_high(in_path, off, progress=progress)
    elif fmt == "evt2":
        th_vals, th_pos, total = _scan_time_high_evt2(in_path, off)
    else:
        th_vals, th_pos, total = _scan_time_high_evt3(in_path, off)
    unit = _HIGH_UNIT_US[fmt]
    oi = _rawcut._origin_index(th_vals, gap_hi=_PREROLL_US // unit) if th_vals.size else 0
    origin_high = int(th_vals[oi]) if th_vals.size else 0
    dur = (int(th_vals[-1]) - origin_high) * unit / 1e6 if th_vals.size else 0.0
    return dict(fmt=fmt, unit_us=unit, th_vals=th_vals, th_pos=th_pos, total_words=total,
                off=off, meta=meta, width=(width or 320), height=(height or 320),
                origin_idx=oi, origin_high=origin_high, duration_s=max(dur, 0.0),
                n_time_high=int(th_vals.size))


# ====================================================================================
# Slice decode: time window -> word range -> seeded chunk decode (no replay from t=0)
# ====================================================================================
def _slice_bounds(index, t0_us, t1_us, max_words=None):
    """Map an absolute-µs window to ``(start_word, end_word, seed_hi, cov0_us, cov1_us)``.

    ``[cov0_us, cov1_us)`` is the span the word range covers *completely* — it can start a
    little before ``t0_us`` (the slice begins at the active TIME_HIGH) and, when *max_words*
    truncates a dense region, end before ``t1_us``. ``t0_us``/``t1_us`` of ``None`` mean the
    file start / end."""
    th_vals, th_pos = index["th_vals"], index["th_pos"]
    unit, total = index["unit_us"], index["total_words"]
    oi = index["origin_idx"]
    a0 = int(th_vals[oi]) if t0_us is None else max(int(t0_us) // unit, 0)
    s = max(int(np.searchsorted(th_vals, a0, side="right")) - 1, 0)
    s = int(np.searchsorted(th_vals, th_vals[s], side="left"))   # first of an equal run
    s = max(s, oi)
    if t1_us is None:
        e = th_vals.size
    else:
        a1 = -(-int(t1_us) // unit)                              # ceil
        e = int(np.searchsorted(th_vals, a1, side="left"))
    start_word = int(th_pos[s])
    end_word = int(th_pos[e]) if e < th_pos.size else int(total)
    if max_words is not None and end_word - start_word > int(max_words):
        end_word = start_word + int(max_words)
    cov0 = int(th_vals[s]) * unit
    if end_word >= total:
        cov1 = (int(th_vals[-1]) + 1) * unit                     # ran to EOF: covered past the last event
    else:
        j = int(np.searchsorted(th_pos, end_word, side="left"))
        if j >= th_pos.size:                                     # truncated inside the trailing run
            cov1 = int(th_vals[-1]) * unit
        elif int(th_pos[j]) == end_word:
            cov1 = int(th_vals[j]) * unit
        else:                                                    # truncated between two TIME_HIGHs
            cov1 = int(th_vals[max(j - 1, s)]) * unit
    return start_word, end_word, int(th_vals[s]), cov0, cov1


def decode_slice(in_path, index, t0_us=None, t1_us=None, max_words=MAX_SLICE_WORDS) -> dict:
    """Decode one time window straight out of the middle of the file (EVT2.x only).

    Seeks to the indexed word range and runs the normal chunk decoder with a seeded
    TIME_HIGH state, so cost is proportional to the window, not the file. Returns
    ``{x, y, p, t, cov0_us, cov1_us}`` — events time-sorted, ``t`` in **absolute** µs
    (the recording's global clock), coords clipped to the header geometry, and
    ``[cov0_us, cov1_us)`` the span that is guaranteed complete."""
    fmt = index["fmt"]
    if fmt not in ("evt21", "evt2"):
        raise UnsupportedPreview(f"slice decode needs EVT2.x, got {fmt!r}")
    if not index["n_time_high"]:
        raise UnsupportedPreview("no TIME_HIGH words to index")
    wb = np.dtype(_dec.word_dtype(fmt)).itemsize
    sw, ew, seed, cov0, cov1 = _slice_bounds(index, t0_us, t1_us, max_words=max_words)
    empty = dict(x=np.zeros(0, np.uint16), y=np.zeros(0, np.uint16),
                 p=np.zeros(0, np.uint8), t=np.zeros(0, np.int64),
                 cov0_us=cov0, cov1_us=max(cov1, cov0))
    if ew <= sw or cov1 <= cov0:
        return empty
    with open(ext(in_path), "rb") as f:
        f.seek(index["off"] + sw * wb)
        buf = f.read((ew - sw) * wb)
    w = np.frombuffer(buf[: (len(buf) // wb) * wb], dtype=_dec.word_dtype(fmt))
    st = _dec.init_state(fmt, dict(hi=int(seed), seen=True))
    x, y, p, t = _dec.CHUNK[fmt](w, st)
    if not len(t):
        return empty
    o = np.argsort(t, kind="stable")
    x, y, p, t = x[o], y[o], p[o], t[o]
    k = int(np.searchsorted(t, cov1, side="left"))   # drop the ragged tail past full coverage
    x, y, p, t = x[:k], y[:k], p[:k], t[:k]
    x = np.clip(x, 0, index["width"] - 1).astype(np.uint16)
    y = np.clip(y, 0, index["height"] - 1).astype(np.uint16)
    return dict(x=x, y=y, p=p.astype(np.uint8), t=t.astype(np.int64),
                cov0_us=cov0, cov1_us=cov1)


# ====================================================================================
# The sampled Recording
# ====================================================================================
class PreviewRecording(Recording):
    """A partially-decoded :class:`Recording` sampled out of a large ``.raw``.

    Looks and behaves like a normal Recording over the decoded spans (``coverage``), but
    ``duration_s``/``t_stop_s`` report the WHOLE file's span (from the index) so the
    transport bar covers the full recording, and :meth:`window` decodes an uncovered
    request on demand (bounded; see :data:`ON_DEMAND_MAX_SPAN_S`) before slicing. The GUI
    replaces it with the real memmap-backed Recording when the background decode lands.
    """

    is_preview = True

    def __init__(self, data, index, coverage_us, full_duration_s, name=""):
        super().__init__(data, name=name)
        self._index = index
        self._cov_us = [(int(a), int(b)) for a, b in coverage_us]
        self._full_duration_s = float(full_duration_s)
        self._lock = threading.RLock()

    # ------------------------------------------------------------- whole-file facts
    @property
    def coverage(self):
        """Decoded spans as ``(t0_s, t1_s)`` tuples, ascending and disjoint."""
        return [(a / 1e6, b / 1e6) for a, b in self._cov_us]

    @property
    def duration_s(self) -> float:
        return self._full_duration_s

    @property
    def t_stop_s(self) -> float:
        return self._full_duration_s

    def covers(self, t0_s: float, t1_s: float) -> bool:
        """True iff ``[t0_s, t1_s)`` lies inside one decoded span."""
        a, b = int(t0_s * 1e6), int(np.ceil(t1_s * 1e6))
        return any(c0 <= a and b <= c1 for c0, c1 in self._cov_us)

    # ------------------------------------------------------------- windowing (on demand)
    def index_at(self, t_s, side="left") -> int:
        with self._lock:
            return super().index_at(t_s, side)

    def window(self, t0=None, t1=None, roi=None):
        with self._lock:
            if t0 is not None and t1 is not None and 0 < (t1 - t0) <= ON_DEMAND_MAX_SPAN_S:
                self.ensure_window(t0, t1)
            return super().window(t0, t1, roi=roi)

    def ensure_window(self, t0_s, t1_s):
        """Decode whatever part of ``[t0_s, t1_s)`` is not covered yet (bounded, typically
        sub-second) and extend ``coverage``. Never raises — on any failure the preview just
        keeps rendering the spans it already has."""
        with self._lock:
            try:
                self._fill_gaps(t0_s, t1_s)
            except Exception:
                pass

    def _fill_gaps(self, t0_s, t1_s):
        q = _ON_DEMAND_QUANTUM_US
        lo = max(int(t0_s * 1e6) // q * q, 0)
        hi = -(-int(np.ceil(t1_s * 1e6)) // q) * q
        for _ in range(16):                       # hard stop; each pass must extend coverage
            gap = self._first_gap(lo, hi)
            if gap is None or not self._decode_gap(*gap):
                break

    def _first_gap(self, lo, hi):
        """First uncovered ``(ga, gb, cap)`` inside ``[lo, hi)``; *cap* is the start of the
        next covered span (a decode must not run past it) or ``None``."""
        cur = lo
        for c0, c1 in self._cov_us:
            if c1 <= cur:
                continue
            if c0 >= hi:
                return (cur, hi, c0) if cur < hi else None
            if c0 > cur:
                return cur, c0, c0
            cur = max(cur, c1)
            if cur >= hi:
                return None
        return (cur, hi, None) if cur < hi else None

    def _decode_gap(self, ga, gb, cap) -> bool:
        sl = decode_slice(self.source_path, self._index,
                          self.t0_us + ga, self.t0_us + gb, max_words=MAX_SLICE_WORDS)
        c0 = max(sl["cov0_us"] - self.t0_us, ga)          # rel µs; head below ga already decoded
        c1 = sl["cov1_us"] - self.t0_us
        if cap is not None:
            c1 = min(c1, cap)                             # never run into the next covered span
        if c1 <= c0:
            return False
        t_new = sl["t"] - self.t0_us
        k0 = int(np.searchsorted(t_new, c0, side="left"))
        k1 = int(np.searchsorted(t_new, c1, side="left"))
        i = int(np.searchsorted(np.asarray(self.t), c0, side="left"))
        self.x = np.concatenate([self.x[:i], sl["x"][k0:k1], self.x[i:]])
        self.y = np.concatenate([self.y[:i], sl["y"][k0:k1], self.y[i:]])
        self.p = np.concatenate([self.p[:i], sl["p"][k0:k1], self.p[i:]])
        self.t = np.concatenate([self.t[:i], t_new[k0:k1], self.t[i:]])
        if self.n_on_cached >= 0:
            self.n_on_cached += int((sl["p"][k0:k1] == 1).sum())
        self._cov_us = _merge_spans(self._cov_us + [(int(c0), int(c1))])
        return True


def _merge_spans(spans):
    """Merge touching/overlapping ``(a, b)`` spans into a sorted disjoint list."""
    out = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def preview_recording(path, slice_s=SLICE_S, n_slices=N_SLICES,
                      max_slice_words=MAX_SLICE_WORDS, index=None, progress=None
                      ) -> PreviewRecording:
    """Build a playable sampled view of *path* in ~1–2 s: index the file, decode
    beginning/middle/end windows of ~*slice_s* seconds (each capped at *max_slice_words*),
    and wrap them as a :class:`PreviewRecording` spanning the full file duration.

    Pass a precomputed *index* (from :func:`scan_index`) to skip the index pass. Raises
    :class:`UnsupportedPreview` for EVT3 / empty files — callers fall back to the normal
    full load."""
    path = os.path.abspath(path)
    ix = index if index is not None else scan_index(path, progress=progress)
    if ix["fmt"] not in ("evt21", "evt2"):
        raise UnsupportedPreview(f"sampled preview needs EVT2.x, got {ix['fmt']!r}")
    if not ix["n_time_high"]:
        raise UnsupportedPreview("no TIME_HIGH words — nothing to index")
    unit = ix["unit_us"]
    origin_us = ix["origin_high"] * unit
    d = ix["duration_s"]
    if d <= n_slices * float(slice_s):
        spans = [(0.0, None)]                       # short (or dense-short) file: one window
    else:
        s = float(slice_s)
        spans = [(0.0, s), (0.5 * (d - s), 0.5 * (d + s)), (d - s, None)]
    parts = []
    for r0, r1 in spans:
        sl = decode_slice(path, ix, origin_us + int(r0 * 1e6),
                          None if r1 is None else origin_us + int(r1 * 1e6),
                          max_words=max_slice_words)
        if not len(sl["t"]):
            continue
        if parts and sl["cov0_us"] < parts[-1]["cov1_us"]:
            # sparse TIME_HIGHs can make a slice reach back into the previous one — trim
            k = int(np.searchsorted(sl["t"], parts[-1]["cov1_us"], side="left"))
            for key in ("x", "y", "p", "t"):
                sl[key] = sl[key][k:]
            sl["cov0_us"] = parts[-1]["cov1_us"]
            if not len(sl["t"]) or sl["cov1_us"] <= sl["cov0_us"]:
                continue
        parts.append(sl)
    if not parts:
        raise UnsupportedPreview("sampled windows decoded no events")
    x = np.concatenate([q["x"] for q in parts])
    y = np.concatenate([q["y"] for q in parts])
    p = np.concatenate([q["p"] for q in parts])
    t = np.concatenate([q["t"] for q in parts])
    t0_us = int(t[0])
    end_us = (int(ix["th_vals"][-1]) + 1) * unit    # just past the last possible event
    full_dur = max((end_us - t0_us) / 1e6, float(t[-1] - t0_us) / 1e6)
    cov_us = _merge_spans([(max(q["cov0_us"] - t0_us, 0), q["cov1_us"] - t0_us)
                           for q in parts])
    data = dict(x=x, y=y, p=p, t=(t - t0_us).astype(np.int64), t0_us=t0_us,
                width=ix["width"], height=ix["height"], n=len(t),
                n_on=int((p == 1).sum()), fmt=ix["fmt"], meta=ix["meta"],
                source_path=path)
    name = os.path.splitext(os.path.basename(path))[0]
    return PreviewRecording(data, ix, cov_us, full_dur, name=name)
