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

A second, lighter mode serves the **quick viewer** (:func:`sample_recording`, policy
:func:`should_sample` — default :data:`SAMPLE_THRESHOLD_MB` = 100 MB, env
``GOTTLUX_QUICKVIEW_SAMPLE_MB``): no index pass at all. It reads the first, middle and last
:data:`SAMPLE_CHUNK_MB` (25 MB) of the encoded stream by seeking; the last chunk's final
TIME_HIGH gives the end time, the first chunk's TIME_HIGHs the pre-roll origin, and the middle
and end chunks are entered at their first TIME_HIGH word with a seeded decoder (exact for
EVT2.x). Seeks elsewhere are placed by interpolating byte↔time anchors and probing. EVT3
samples only its start. Nothing is decoded in the background.

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


# ====================================================================================
# Byte-sampled open (quick viewer): first / middle / last N MB, no whole-file pass
# ====================================================================================
#: Files at least this large (MB) open in the quick viewer as byte samples.
SAMPLE_THRESHOLD_MB = 100.0
_ENV_SAMPLE = "GOTTLUX_QUICKVIEW_SAMPLE_MB"
#: Size of each sampled chunk (MB of the encoded stream).
SAMPLE_CHUNK_MB = 25.0
#: Words decoded per seek into an un-sampled region (~16 MB of EVT2.1, ~8 MB of EVT2.0).
SEEK_WORDS = 2_000_000
_MAX_ANCHORS_PER_CHUNK = 4000


def sample_threshold_bytes():
    """The quick viewer's sampling threshold in bytes, or ``None`` when sampling is off.

    Default :data:`SAMPLE_THRESHOLD_MB` (100 MB); the ``GOTTLUX_QUICKVIEW_SAMPLE_MB`` env var
    overrides it (``0`` or negative disables)."""
    mb = SAMPLE_THRESHOLD_MB
    v = os.environ.get(_ENV_SAMPLE)
    if v is not None:
        try:
            mb = float(v)
        except ValueError:
            pass
    return None if mb <= 0 else int(mb * 2**20)


def should_sample(path) -> bool:
    """True iff the quick viewer should open *path* as byte samples: an over-threshold,
    **uncached** ``.raw`` (EVT2.1 / EVT2.0 sample start, middle and end; EVT3 samples only
    its start). A cache hit is already instant; HDF5 takes the normal decode."""
    thr = sample_threshold_bytes()
    if thr is None:
        return False
    try:
        p = str(path)
        if not p.lower().endswith(".raw") or not os.path.exists(ext(p)):
            return False
        if os.path.getsize(ext(p)) < thr or _cache.has_valid_cache(p):
            return False
        meta, _ = _dec.parse_header(p)
        return _dec.detect_format(meta) in ("evt21", "evt2", "evt3")
    except Exception:
        return False


def _th_words(fmt, w):
    """Word indices and values of the TIME_HIGH words in an EVT2.x word array."""
    if fmt == "evt21":
        idx = np.nonzero((w >> np.uint64(60)).astype(np.uint8) == 0x8)[0]
        vals = ((w[idx] >> np.uint64(32)) & np.uint64(0x0FFFFFFF)).astype(np.int64)
    else:
        idx = np.nonzero((w >> np.uint32(28)) == np.uint32(0x8))[0]
        vals = (w[idx] & np.uint32(0x0FFFFFFF)).astype(np.int64)
    return idx.astype(np.int64), vals


def _read_words(path, off, fmt, w0, w1):
    wb = np.dtype(_dec.word_dtype(fmt)).itemsize
    with open(ext(path), "rb") as f:
        f.seek(off + int(w0) * wb)
        buf = f.read(max(int(w1) - int(w0), 0) * wb)
    return np.frombuffer(buf[: (len(buf) // wb) * wb], dtype=_dec.word_dtype(fmt))


def _decode_entered(fmt, w, w0, total, unit):
    """Decode an EVT2.x word range entered mid-file at its first TIME_HIGH.

    Returns ``(x, y, p, t, cov0_us, cov1_us, anchors)``: events time-sorted in absolute µs,
    the span they cover completely (from the first TIME_HIGH; to the last TIME_HIGH, or to
    just past it when the range runs to the end of the file), and ``(word, µs)`` anchors
    for later seeks."""
    idx, vals = _th_words(fmt, w)
    empty = (np.zeros(0, np.uint16), np.zeros(0, np.uint16), np.zeros(0, np.uint8),
             np.zeros(0, np.int64))
    if not idx.size:
        return (*empty, 0, 0, np.zeros((0, 2), np.int64))
    vals = np.maximum.accumulate(vals)
    i0 = int(idx[0])
    st = _dec.init_state(fmt, dict(hi=int(vals[0]), seen=True))
    x, y, p, t = _dec.CHUNK[fmt](w[i0:], st)
    cov0 = int(vals[0]) * unit
    at_eof = w0 + len(w) >= total
    cov1 = (int(vals[-1]) + 1) * unit if at_eof else int(vals[-1]) * unit
    if len(t):
        o = np.argsort(t, kind="stable")
        x, y, p, t = x[o], y[o], p[o], t[o]
        keep = (t >= cov0) & (t < cov1)
        x, y, p, t = x[keep], y[keep], p[keep], t[keep]
    step = max(1, idx.size // _MAX_ANCHORS_PER_CHUNK)
    anchors = np.stack([idx[::step] + int(w0), vals[::step] * unit], axis=1)
    return x, y, p, t, cov0, max(cov1, cov0), anchors


class SampledRecording(PreviewRecording):
    """A :class:`PreviewRecording` built from **byte samples** of a large ``.raw`` with no
    whole-file pass: the first, middle and last :data:`SAMPLE_CHUNK_MB` of the encoded
    stream (EVT3: the first only). The timeline spans the whole file (its end time comes
    from the last chunk); seeking into an un-sampled region decodes :data:`SEEK_WORDS`
    words there, placed by interpolating the byte↔time anchors gathered so far.

    ``chunks`` lists what was decoded (``(label, byte0, byte1)`` in the encoded payload),
    ``file_bytes`` the payload size, ``sampled_bytes`` the bytes decoded so far, and
    ``duration_known`` is False when the end time could not be read (EVT3)."""

    is_sampled = True

    def __init__(self, data, coverage_us, full_duration_s, name, *, fmt, off, total_words,
                 anchors, chunks, duration_known):
        super().__init__(data, None, coverage_us, full_duration_s, name=name)
        self._off = int(off)
        self._total_words = int(total_words)
        self._wb = np.dtype(_dec.word_dtype(fmt)).itemsize
        self._unit = _HIGH_UNIT_US[fmt]
        self._anchors = np.asarray(anchors, np.int64).reshape(-1, 2)
        self.chunks = list(chunks)
        self.duration_known = bool(duration_known)
        self.seekable = fmt in ("evt21", "evt2")

    @property
    def file_bytes(self) -> int:
        return self._total_words * self._wb

    @property
    def sampled_bytes(self) -> int:
        spans = _merge_spans([(b0, b1) for _, b0, b1 in self.chunks])
        return int(sum(b - a for a, b in spans))

    def describe(self) -> str:
        """One line for status bars and provenance: what has been decoded. Sizes are in
        MB of 2^20 bytes, the unit the thresholds are set in."""
        def mb(b):
            return f"{b / 2**20:,.1f} MB" if b < 10 * 2**20 else f"{b / 2**20:,.0f} MB"
        labels = [c[0] for c in self.chunks if c[0] in ("start", "middle", "end", "whole")]
        each = mb(int(self.chunks[0][2] - self.chunks[0][1])) if self.chunks else "0 MB"
        what = "whole file" if labels == ["whole"] else f"{'/'.join(labels)} {each} each"
        tail = "" if self.duration_known else " · end time unknown (EVT3: start only)"
        return (f"sampled {what} of {mb(self.file_bytes)} "
                f"({100.0 * self.sampled_bytes / max(self.file_bytes, 1):.1f} % decoded){tail}")

    # ---------------------------------------------------------------- seek-decode
    def _decode_gap(self, ga, gb, cap) -> bool:
        if not self.seekable or self._anchors.shape[0] < 2:
            return False
        target = self.t0_us + int(ga)
        order = np.argsort(self._anchors[:, 1], kind="stable")
        times, words = self._anchors[order, 1], self._anchors[order, 0]
        est = float(np.interp(target, times, words))
        start = int(max(0, min(est - 0.05 * SEEK_WORDS, self._total_words - 1)))
        for _ in range(6):                        # a few probes; each adds anchors
            end = min(start + SEEK_WORDS, self._total_words)
            w = _read_words(self.source_path, self._off, self.fmt, start, end)
            x, y, p, t, cov0, cov1, anc = _decode_entered(self.fmt, w, start,
                                                          self._total_words, self._unit)
            if anc.size:
                self._anchors = np.concatenate([self._anchors, anc])
            if cov1 <= cov0:
                return False
            if target < cov0 and start > 0:       # landed too late: step back
                rate = (end - start) / max(cov1 - cov0, 1)
                start = int(max(0, start - (cov0 - target) * rate - 0.1 * SEEK_WORDS))
                continue
            if target >= cov1 and end < self._total_words:   # too early: step forward
                rate = (end - start) / max(cov1 - cov0, 1)
                start = int(min(self._total_words - 1, end + (target - cov1) * rate
                                - 0.05 * SEEK_WORDS))
                continue
            self.chunks.append(("seek", start * self._wb, end * self._wb))
            return self._insert(x, y, p, t, max(cov0 - self.t0_us, int(ga)),
                                cov1 - self.t0_us, cap)
        return False

    def _insert(self, x, y, p, t_abs, c0, c1, cap) -> bool:
        """Merge decoded events covering relative ``[c0, c1)`` (clipped to *cap* and to the
        spans already covered) into the sorted arrays and the coverage list."""
        if cap is not None:
            c1 = min(c1, cap)
        for a, b in self._cov_us:                 # never duplicate a covered span
            if a <= c0 < b:
                c0 = b
            if a < c1 <= b:
                c1 = a
        if c1 <= c0:
            return False
        t_new = t_abs - self.t0_us
        k0 = int(np.searchsorted(t_new, c0, side="left"))
        k1 = int(np.searchsorted(t_new, c1, side="left"))
        i = int(np.searchsorted(np.asarray(self.t), c0, side="left"))
        self.x = np.concatenate([self.x[:i], x[k0:k1].astype(np.uint16), self.x[i:]])
        self.y = np.concatenate([self.y[:i], y[k0:k1].astype(np.uint16), self.y[i:]])
        self.p = np.concatenate([self.p[:i], p[k0:k1].astype(np.uint8), self.p[i:]])
        self.t = np.concatenate([self.t[:i], t_new[k0:k1].astype(np.int64), self.t[i:]])
        if self.n_on_cached >= 0:
            self.n_on_cached += int((p[k0:k1] == 1).sum())
        self._cov_us = _merge_spans(self._cov_us + [(int(c0), int(c1))])
        return True


def sample_recording(path, chunk_mb=SAMPLE_CHUNK_MB, on_stage=None) -> SampledRecording:
    """Open *path* by decoding byte samples only — the quick viewer's large-file path.

    1. Read the first and the last *chunk_mb* MB of the encoded stream. The last chunk's
       final TIME_HIGH gives the file's end time without reading anything in between.
    2. Decode the first chunk from the file start (the decoder's own state) and drop a
       pre-roll exactly as a full decode would, then call ``on_stage(rec)`` — the window can
       play this within about a second.
    3. Enter a middle chunk and the last chunk at their first TIME_HIGH word (EVT2.x words
       are self-describing, so a seeded decoder is exact from there), merge them, and call
       ``on_stage(rec)`` again. Returns the final :class:`SampledRecording`.

    EVT3 keeps a time-wrap and row state that cannot be recovered mid-file, so an EVT3
    file is sampled at its start only (and its timeline covers just that span). A file no
    bigger than three chunks is decoded whole."""
    path = os.path.abspath(path)
    meta, off = _dec.parse_header(path)
    fmt = _dec.detect_format(meta)
    width, height = _dec.geometry(meta)
    wb = np.dtype(_dec.word_dtype(fmt)).itemsize
    total = max((os.path.getsize(ext(path)) - off) // wb, 0)
    n = max(1, int(float(chunk_mb) * 2**20) // wb)
    unit = _HIGH_UNIT_US[fmt]
    whole = total <= 3 * n
    seekable = fmt in ("evt21", "evt2") and not whole
    head_end = total if whole else n
    head = _read_words(path, off, fmt, 0, head_end)
    name = os.path.splitext(os.path.basename(path))[0]

    # --- whole-file time span: first TIME_HIGHs of the head + last TIME_HIGH of the tail
    end_us, origin_us, tail = None, None, None
    if fmt in ("evt21", "evt2"):
        hidx, hvals = _th_words(fmt, head)
        hvals = np.maximum.accumulate(hvals) if hvals.size else hvals
        if seekable:
            tail = _read_words(path, off, fmt, total - n, total)
            _, tvals = _th_words(fmt, tail)
            last_hi = int(tvals.max()) if tvals.size else (int(hvals[-1]) if hvals.size else 0)
        else:
            last_hi = int(hvals[-1]) if hvals.size else 0
        end_us = (last_hi + 1) * unit
        if hvals.size:
            # the full decode's pre-roll rule, with the span taken over the whole file
            look = max(5, hvals.size // 200)
            d = np.diff(hvals[: look + 1])
            oi = 0
            if d.size and hvals.size >= 50:       # as _rawcut._origin_index
                j = int(np.argmax(d))
                if int(d[j]) * unit >= _PREROLL_US and int(d[j]) >= (last_hi - int(hvals[0])) // 2:
                    oi = j + 1
            origin_us = int(hvals[oi]) * unit

    # --- stage 1: the head, decoded from the file start
    x, y, p, t = _dec.CHUNK[fmt](head, _dec.init_state(fmt))
    if len(t):
        o = np.argsort(t, kind="stable")
        x, y, p, t = x[o], y[o], p[o], t[o]
        if origin_us is not None:
            k = int(np.searchsorted(t, origin_us, side="left"))
            x, y, p, t = x[k:], y[k:], p[k:], t[k:]
        else:
            x, y, p, t = _dec.strip_preroll(x, y, p, t)
    if not len(t):
        raise UnsupportedPreview("the first sampled chunk decoded no events")
    if fmt in ("evt21", "evt2") and not whole:
        head_cov1 = int(hvals[-1]) * unit             # the trailing period continues past it
        keep = t < head_cov1
        x, y, p, t = x[keep], y[keep], p[keep], t[keep]
    else:
        head_cov1 = int(t[-1]) + 1
    width = width or int(x.max()) + 1
    height = height or int(y.max()) + 1
    t0_us = int(t[0])
    if fmt in ("evt21", "evt2"):
        anchors = np.stack([hidx[:: max(1, hidx.size // _MAX_ANCHORS_PER_CHUNK)],
                            hvals[:: max(1, hvals.size // _MAX_ANCHORS_PER_CHUNK)] * unit], 1) \
            if hidx.size else np.zeros((0, 2), np.int64)
    else:
        anchors = np.zeros((0, 2), np.int64)
    full_dur = ((end_us - t0_us) / 1e6) if end_us is not None and not whole else \
        (head_cov1 - t0_us) / 1e6
    data = dict(x=np.clip(x, 0, width - 1).astype(np.uint16),
                y=np.clip(y, 0, height - 1).astype(np.uint16), p=p.astype(np.uint8),
                t=(t - t0_us).astype(np.int64), t0_us=t0_us, width=width, height=height,
                n=len(t), n_on=int((p == 1).sum()), fmt=fmt, meta=meta, source_path=path)
    rec = SampledRecording(data, [(0, head_cov1 - t0_us)], max(full_dur, 1e-6), name, fmt=fmt,
                           off=off, total_words=total, anchors=anchors,
                           chunks=[("whole" if whole else "start", 0, head_end * wb)],
                           duration_known=(end_us is not None) or whole)
    rec.seekable = seekable
    if on_stage is not None:
        on_stage(rec)
    if not seekable:
        return rec

    # --- stage 2: the middle and the last chunk, each entered at its first TIME_HIGH
    m0 = max(n, total // 2 - n // 2)
    for label, w0, words in (("middle", m0, _read_words(path, off, fmt, m0, m0 + n)),
                             ("end", total - n, tail)):
        xs, ys, ps, ts, c0, c1, anc = _decode_entered(fmt, words, w0, total, unit)
        if anc.size:
            rec._anchors = np.concatenate([rec._anchors, anc])
        rec.chunks.append((label, w0 * wb, (w0 + len(words)) * wb))
        if c1 > c0:
            rec._insert(xs, ys, ps, ts, c0 - t0_us, c1 - t0_us, None)
    if on_stage is not None:
        on_stage(rec)
    return rec
