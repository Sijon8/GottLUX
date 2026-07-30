"""
rawcut.py — cut a time window out of an EVT2.1 ``.raw`` **without decoding it**.

Decoding a multi-GB recording just to crop a few seconds out is the wasteful part: it reads the
whole file, expands every 32-bit column mask into events, sorts each chunk, and writes four cache
bins — minutes of work and a large disk footprint, all to throw most of it away.

This module cuts directly on the encoded byte stream. EVT2.1 is a flat list of 64-bit words where
a **TIME_HIGH** word (type ``0x8``) carries the 28-bit high timestamp (unit 64 µs) and every CD
word that follows inherits it. So a time-window cut is a *byte-range copy*:

1. **Index pass** (:func:`scan`) — read the file once recording only each TIME_HIGH's value and
   word position (no event expansion, no sort, no disk writes). It is I/O-bound — several times
   faster than a full decode — and the index is a few MB even for a long recording.
2. **Copy pass** (:func:`cut_evt21`) — copy the words from the active TIME_HIGH at ``t0`` up to the
   one at ``t1``, rewriting each TIME_HIGH so the clip is re-zeroed; CD (and trigger/other) words
   pass through untouched. Bounded RAM (one streamed chunk), and the output is a valid EVT2.1 file.

Cuts land on the 64-µs TIME_HIGH grid (≪ any crop a user cares about). ROI crops and the EVT2.0 /
EVT3 encodings still go through the decode-based :func:`gottlux.io.writer.cut_clip` (a smart
dispatcher, :func:`gottlux.io.writer.cut_file`, picks the right path automatically).
"""
from __future__ import annotations

import os

import numpy as np

from gottlux.io.decode import detect_format, geometry, parse_header
from gottlux.io.paths import ext

_TYPE_SHIFT = np.uint64(60)
_TH_HIGH_SHIFT = np.uint64(32)
_TS_LOW_SHIFT = np.uint64(54)
_M28 = np.uint64(0x0FFFFFFF)
_M32 = np.uint64(0xFFFFFFFF)
_M6 = np.uint64(0x3F)
_TH_CLEAR = ~(_M28 << _TH_HIGH_SHIFT)          # keep the type nibble + low 32 bits; clear high field
_TH_TYPE = np.uint8(0x8)
_HIGH_UNIT_US = 64
_PREROLL_GAP_HI = 3_000_000 // _HIGH_UNIT_US    # a >=3 s gap = corrupt pre-roll (cf. strip_preroll)


class UnsupportedRaw(Exception):
    """Raised when a file can't be cut on the bytes (not EVT2.1) — caller should fall back."""


def _header_info(in_path):
    meta, off = parse_header(in_path)
    fmt = detect_format(meta)
    w, h = geometry(meta)
    return meta, off, fmt, (w or 320), (h or 320)


def _header_bytes(meta, width, height) -> bytes:
    """Reconstruct a valid EVT2.1 ASCII header, preserving the source's keys where present."""
    m = dict(meta) if meta else {}
    m.setdefault("format", f"EVT21;height={height};width={width}")
    m.setdefault("geometry", f"{width}x{height}")
    return ("".join(f"% {k} {v}\n" for k, v in m.items()) + "% end\n").encode("ascii", "replace")


def _origin_index(th_vals, gap_hi=_PREROLL_GAP_HI) -> int:
    """First 'real' TIME_HIGH index, skipping a corrupt leading cluster (cf. ``strip_preroll``).

    *gap_hi* is the pre-roll gap threshold in TIME_HIGH units — pass the equivalent of 3 s
    for encodings whose high-timestamp unit differs from EVT2.x (see ``gottlux.io.preview``).
    """
    n = th_vals.size
    if n < 50:
        return 0
    look = max(5, n // 200)
    d = np.diff(th_vals[: look + 1])
    if d.size:
        j = int(np.argmax(d))
        span = int(th_vals[-1] - th_vals[0])
        if int(d[j]) >= gap_hi and int(d[j]) >= span // 2:
            return j + 1
    return 0


def scan_time_high(in_path, off, chunk_words=8_000_000, progress=None):
    """One I/O pass: return ``(th_vals, th_pos, total_words)`` — the (running-max) high value and
    payload word index of every TIME_HIGH word. No event expansion."""
    path = ext(in_path)
    payload = max(os.path.getsize(path) - off, 1)
    vals, poss, total, done = [], [], 0, 0
    with open(path, "rb") as f:
        f.seek(off)
        while True:
            buf = f.read(chunk_words * 8)
            if not buf:
                break
            got = len(buf) // 8
            w = np.frombuffer(buf[: got * 8], dtype="<u8")
            idx = np.nonzero((w >> _TYPE_SHIFT).astype(np.uint8) == _TH_TYPE)[0]
            if idx.size:
                vals.append(((w[idx] >> _TH_HIGH_SHIFT) & _M28).astype(np.int64))
                poss.append(idx.astype(np.int64) + total)
            total += got
            done += len(buf)
            if progress:
                try:
                    progress(min(done / payload, 1.0))
                except Exception:
                    pass
    if vals:
        th_vals = np.maximum.accumulate(np.concatenate(vals))   # mirror the decoder's monotonic clamp
        th_pos = np.concatenate(poss)
    else:
        th_vals = np.zeros(0, np.int64)
        th_pos = np.zeros(0, np.int64)
    return th_vals, th_pos, total


def scan(in_path, progress=None) -> dict:
    """Index an EVT2.1 ``.raw`` (the reusable, expensive half of a direct cut).

    Returns a dict with ``th_vals, th_pos, total_words, off, meta, width, height, origin_idx,
    origin_high, duration_s, n_time_high``. Raises :class:`UnsupportedRaw` for non-EVT2.1 files.
    """
    meta, off, fmt, width, height = _header_info(in_path)
    if fmt != "evt21":
        raise UnsupportedRaw(f"direct cut needs EVT2.1, got {fmt!r}")
    th_vals, th_pos, total = scan_time_high(in_path, off, progress=progress)
    oi = _origin_index(th_vals) if th_vals.size else 0
    origin_high = int(th_vals[oi]) if th_vals.size else 0
    dur = (int(th_vals[-1]) - origin_high) * _HIGH_UNIT_US / 1e6 if th_vals.size else 0.0
    return dict(th_vals=th_vals, th_pos=th_pos, total_words=total, off=off, meta=meta,
                width=width, height=height, origin_idx=oi, origin_high=origin_high,
                duration_s=max(dur, 0.0), n_time_high=int(th_vals.size))


def raw_bounds(in_path, progress=None) -> dict:
    """A light summary of a ``.raw`` (duration / geometry) from the index pass alone — lets the GUI
    offer a cut before any decode. Returns ``{duration_s, width, height, n_time_high}``."""
    ix = scan(in_path, progress=progress)
    return {"duration_s": ix["duration_s"], "width": ix["width"], "height": ix["height"],
            "n_time_high": ix["n_time_high"]}


def _copy_rebase(in_path, out_path, off, start_word, end_word, base, header, progress=None,
                 chunk_words=4_000_000):
    """Stream-copy payload words ``[start_word, end_word)`` to *out_path*, rewriting each TIME_HIGH
    to ``high − base`` (re-zeroing the clip). Returns the number of expanded events written."""
    base_u = np.uint64(int(base))
    n_events = 0
    remaining = end_word - start_word
    with open(ext(in_path), "rb") as fi, open(ext(out_path), "wb") as fo:
        fo.write(header)
        fi.seek(off + start_word * 8)
        done = 0
        while remaining > 0:
            buf = fi.read(min(remaining, chunk_words) * 8)
            if not buf:
                break
            got = len(buf) // 8
            w = np.frombuffer(buf[: got * 8], dtype="<u8").copy()
            typ = (w >> _TYPE_SHIFT).astype(np.uint8)
            th = typ == _TH_TYPE
            if th.any():
                hi = (w[th] >> _TH_HIGH_SHIFT) & _M28
                new_hi = np.where(hi >= base_u, hi - base_u, np.uint64(0)) & _M28
                w[th] = (w[th] & _TH_CLEAR) | (new_hi << _TH_HIGH_SHIFT)
            cd = (typ == 0) | (typ == 1)
            if cd.any():
                n_events += int(np.bitwise_count((w[cd] & _M32).astype(np.uint32)).sum())
            fo.write(w.tobytes())
            remaining -= got
            done += got
            if progress:
                try:
                    progress(min(done / max(end_word - start_word, 1), 1.0))
                except Exception:
                    pass
    return n_events


def cut_evt21(in_path, out_path, t0=None, t1=None, index=None, progress=None) -> dict:
    """Cut ``[t0, t1)`` seconds out of an EVT2.1 ``.raw`` directly on the bytes — no decode.

    *t0*/*t1* are seconds from the recording start (``None`` = open ends). Pass a precomputed
    *index* (from :func:`scan`) to skip the index pass. Returns ``{path, n_events, duration_s,
    width, height}``. Raises :class:`UnsupportedRaw` for non-EVT2.1 input.
    """
    have_index = index is not None
    ix = index or scan(in_path, progress=(lambda f: progress(0.5 * f)) if progress else None)
    th_vals, th_pos = ix["th_vals"], ix["th_pos"]
    width, height, off = ix["width"], ix["height"], ix["off"]
    header = _header_bytes(ix["meta"], width, height)
    copy_prog = progress if have_index else ((lambda f: progress(0.5 + 0.5 * f)) if progress else None)

    if th_vals.size == 0:
        with open(ext(out_path), "wb") as f:
            f.write(header)
        return {"path": out_path, "n_events": 0, "duration_s": 0.0, "width": width, "height": height}

    origin_high, oi = ix["origin_high"], ix["origin_idx"]
    last_high = int(th_vals[-1])
    a0 = origin_high if t0 is None else origin_high + int(round(float(t0) * 1e6 / _HIGH_UNIT_US))
    a1 = (last_high + 1) if t1 is None else origin_high + int(round(float(t1) * 1e6 / _HIGH_UNIT_US))
    if a1 <= a0:
        with open(ext(out_path), "wb") as f:
            f.write(header)
        return {"path": out_path, "n_events": 0, "duration_s": 0.0, "width": width, "height": height}

    s_idx = max(oi, int(np.searchsorted(th_vals, a0, side="right")) - 1)
    base = int(th_vals[s_idx])
    start_word = int(th_pos[s_idx])
    e_idx = int(np.searchsorted(th_vals, a1, side="left"))
    end_word = int(th_pos[e_idx]) if e_idx < th_pos.size else int(ix["total_words"])
    last_kept = max(s_idx, min(e_idx, th_vals.size) - 1)
    duration_s = max(int(th_vals[last_kept]) - base, 0) * _HIGH_UNIT_US / 1e6

    n = _copy_rebase(in_path, out_path, off, start_word, end_word, base, header, progress=copy_prog)
    return {"path": out_path, "n_events": int(n), "duration_s": duration_s,
            "width": width, "height": height}
