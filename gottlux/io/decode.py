"""
decode.py — pure-NumPy decoder for Prophesee ``.raw`` event streams.

A single ``.raw`` file can be encoded in one of three incompatible bit layouts. gottlux
reads all three with no dependency on the Metavision SDK (the converted ``.hdf5`` files
use Prophesee's ECF codec / HDF5 filter 36559, which *does* need the SDK — so we decode
the plain ``.raw`` instead):

================  ==========  =====================================================
encoding          word size   notes
================  ==========  =====================================================
**EVT2.1**        64-bit      X320 / GenX320 payload; vectorized CD (32-px column mask)
**EVT2.0**        32-bit      single CD events
**EVT3**          16-bit      IMX636 / Gen4; *stateful* (y-row, vector base-x, time epoch)
================  ==========  =====================================================

The bit layouts below were reverse-engineered and validated **bit-exact** against the
ECF-HDF5 oracle (expanded pixel-event counts and time spans match to the event).

This module holds the *stateless math*: header parsing, format detection, and per-chunk
decoders that carry an explicit state dict across chunk boundaries (so :mod:`gottlux.io.cache`
can stream a multi-GB file in bounded RAM). :func:`decode` is the convenience full-in-memory
path for small files and tests; large files go through ``cache.load``.

Decoded event arrays
--------------------
``x`` (uint16), ``y`` (uint16), ``p`` (uint8: 1=ON/brighter, 0=OFF), ``t`` (int64 µs,
zero-based and sorted ascending).
"""
from __future__ import annotations

import numpy as np

from gottlux.io.paths import ext

# --- EVT2.1 bit field shifts/masks (64-bit little-endian words) ----------------------
_TYPE_SHIFT = np.uint64(60)
_TS_LOW_SHIFT = np.uint64(54)
_X_SHIFT = np.uint64(43)
_Y_SHIFT = np.uint64(32)
_TS_HIGH_SHIFT = np.uint64(32)
_M6 = np.uint64(0x3F)
_M11 = np.uint64(0x7FF)
_M28 = np.uint64(0x0FFFFFFF)
_M32 = np.uint64(0xFFFFFFFF)
_TIME_HIGH_UNIT_US = 64

#: Bumped whenever decode logic changes, to invalidate stale on-disk caches.
DECODER_VERSION = 5

_WORD_DTYPE = {"evt3": "<u2", "evt2": "<u4", "evt21": "<u8"}
_EMPTY4 = (np.zeros(0, np.int32), np.zeros(0, np.int32),
           np.zeros(0, np.uint8), np.zeros(0, np.int64))


# ====================================================================================
# Header / format / geometry
# ====================================================================================
def parse_header(path: str):
    """Parse the ASCII ``% key value`` header. Returns ``(meta_dict, payload_offset)``."""
    with open(ext(path), "rb") as f:
        head = f.read(8192)
    end = head.find(b"% end\n")
    if end < 0:
        raise RuntimeError(f"No '% end' header terminator found in {path!r}")
    meta = {}
    for line in head[:end].decode("ascii", "replace").splitlines():
        line = line.strip()
        if line.startswith("%"):
            kv = line[1:].strip().split(" ", 1)
            if len(kv) == 2:
                meta[kv[0]] = kv[1]
    return meta, end + len(b"% end\n")


def detect_format(meta: dict) -> str:
    """Return the declared encoding: ``'evt3'``, ``'evt21'`` or ``'evt2'`` (EVT2.0).

    IMX636 / Gen4 sensors usually record EVT3; the X320 payload records EVT2.1. Each uses
    a completely different bit layout, so dispatch on this is mandatory.
    """
    raw = (str(meta.get("format", "")) + " " + str(meta.get("evt", ""))) \
        .lower().replace(" ", "").replace(".", "")
    if "evt3" in raw:
        return "evt3"
    if "evt21" in raw:
        return "evt21"
    if "evt2" in raw:                              # EVT2.0 (also matches 'evt20')
        return "evt2"
    return "evt21"                                 # legacy default when the header is silent


def geometry(meta: dict):
    """Return ``(width, height)`` from the header, or ``(None, None)`` if absent
    (the decoder then infers the geometry from the maximum observed coordinates)."""
    w = h = None
    for tok in meta.get("format", "").split(";"):
        if tok.startswith("width="):
            w = int(tok.split("=")[1])
        elif tok.startswith("height="):
            h = int(tok.split("=")[1])
    if (w is None or h is None) and "geometry" in meta:
        gw, gh = meta["geometry"].lower().split("x")
        w, h = int(gw), int(gh)
    return w, h


# ====================================================================================
# Cross-chunk decode state + helpers
# ====================================================================================
def init_state(fmt: str, initial: dict | None = None) -> dict:
    """Fresh decoder state for *fmt*, carried across chunks so GB files stream in
    bounded RAM (running TIME_HIGH for evt2/evt21; full stateful set for evt3).

    *initial* overlays fields onto the fresh state — e.g. ``{'hi': ..., 'seen': True}``
    lets a slice decode start mid-file at an indexed TIME_HIGH word (the sampled-preview
    path, :mod:`gottlux.io.preview`) instead of replaying the file from the start."""
    if fmt == "evt3":
        st = dict(epoch=0, last_raw=0, th_cum=0, tl=0, y=0, base_x_next=0, base_pol=0)
    else:
        st = dict(hi=0, seen=False)
    if initial:
        st.update(initial)
    return st


def _ffill(pos, vals, n, init=0, dtype=np.int64):
    """Forward-fill: a length-*n* array where index *i* holds the value at the most recent
    position ``<= i`` in *pos*; indices before the first take *init* (carried chunk state)."""
    out = np.full(n, init, dtype)
    if len(pos):
        last = np.full(n, -1, np.int64)
        last[pos] = np.arange(len(pos))
        np.maximum.accumulate(last, out=last)
        ok = last >= 0
        out[ok] = np.asarray(vals)[last[ok]]
    return out


def _bit_positions(mask, nbits):
    """Positions of the set bits of *mask*: ``(bit, word_index)`` arrays, ordered bit-major
    (all bit-0 hits, then all bit-1 hits, …) — exactly the order the old per-bit boolean-
    select loop produced, so the stable time-sorts downstream stay byte-identical.

    Vectorized: each little-endian byte plane of the mask is expanded with
    ``np.unpackbits`` and scanned with a single ``nonzero`` — a handful of C passes
    instead of *nbits* Python-level selects (the decoder's old hot loop)."""
    dt = np.uint8 if nbits <= 8 else (np.uint16 if nbits <= 16 else np.uint32)
    m = np.ascontiguousarray(mask, dtype=dt)
    planes = m.view(np.uint8).reshape(-1, m.dtype.itemsize)
    bits, idxs = [], []
    for j in range((nbits + 7) // 8):
        u = np.unpackbits(np.ascontiguousarray(planes[:, j])[None, :], axis=0,
                          bitorder="little")
        b, i = np.nonzero(u)
        if i.size:
            bits.append((b + 8 * j).astype(np.int32))
            idxs.append(i)
    if not bits:
        return np.zeros(0, np.int32), np.zeros(0, np.intp)
    return np.concatenate(bits), np.concatenate(idxs)


def strip_preroll(x, y, p, t):
    """Drop a tiny leading cluster separated from the main stream by a huge time gap.

    Such clusters come from corrupt / uninitialized high-timestamps and otherwise appear
    as hundreds of seconds of 'empty' lead-in before the real data begins.
    """
    n = len(t)
    if n < 200:
        return x, y, p, t
    span = int(t[-1] - t[0])
    if span <= 0:
        return x, y, p, t
    look = max(10, n // 500)                       # inspect only the first ~0.2% of events
    gaps = np.diff(t[:look + 1])
    j = int(np.argmax(gaps))
    gap = int(gaps[j])
    if gap >= 3_000_000 and gap >= span // 2:      # >=3 s AND >= half the span -> pre-roll junk
        drop = j + 1
        return x[drop:], y[drop:], p[drop:], t[drop:]
    return x, y, p, t


# ====================================================================================
# Per-chunk decoders (one per encoding). Each takes a NumPy word array + state dict,
# mutates state in place, and returns (x, y, p, t) for events fully resolved in-chunk.
# ====================================================================================
def chunk_evt21(w, st):
    """EVT2.1: 64-bit vectorized CD (32-px column mask) + TIME_HIGH (unit 64 µs)."""
    n = len(w)
    typ = (w >> _TYPE_SHIFT).astype(np.uint8)
    is_th = typ == 0x8
    is_cd = (typ == 0x0) | (typ == 0x1)
    thp = np.where(is_th)[0]
    seen = st["seen"]
    if thp.size:
        raw = np.maximum.accumulate(
            np.maximum(((w[thp] >> _TS_HIGH_SHIFT) & _M28).astype(np.int64), st["hi"]))
        th_ff = _ffill(thp, raw, n, init=st["hi"])
        st["hi"] = int(raw[-1])
        st["seen"] = True
    else:
        th_ff = np.full(n, st["hi"], np.int64)
    has_th = np.ones(n, bool)
    if not seen:
        has_th[: (thp[0] if thp.size else n)] = False   # CD before the first-ever TIME_HIGH is junk
    cd_sel = is_cd & has_th
    if not cd_sel.any():
        return _EMPTY4
    cd = w[cd_sel]
    cd_typ = typ[cd_sel]
    t_us = th_ff[cd_sel] * _TIME_HIGH_UNIT_US + ((cd >> _TS_LOW_SHIFT) & _M6).astype(np.int64)
    x_base = ((cd >> _X_SHIFT) & _M11).astype(np.int32)
    y_base = ((cd >> _Y_SHIFT) & _M11).astype(np.int32)
    mask = (cd & _M32).astype(np.uint32)
    bit, idx = _bit_positions(mask, 32)
    if not idx.size:
        return _EMPTY4
    return x_base[idx] + bit, y_base[idx], cd_typ[idx].astype(np.uint8), t_us[idx]


def chunk_evt2(w, st):
    """EVT2.0: 32-bit single CD events + TIME_HIGH (unit 64 µs)."""
    n = len(w)
    typ = (w >> np.uint32(28)) & np.uint32(0xF)
    is_th = typ == np.uint32(0x8)
    is_cd = (typ == np.uint32(0)) | (typ == np.uint32(1))
    thp = np.where(is_th)[0]
    seen = st["seen"]
    if thp.size:
        raw = np.maximum.accumulate(
            np.maximum((w[thp] & np.uint32(0x0FFFFFFF)).astype(np.int64), st["hi"]))
        th_ff = _ffill(thp, raw, n, init=st["hi"])
        st["hi"] = int(raw[-1])
        st["seen"] = True
    else:
        th_ff = np.full(n, st["hi"], np.int64)
    has_th = np.ones(n, bool)
    if not seen:
        has_th[: (thp[0] if thp.size else n)] = False
    cd_sel = is_cd & has_th
    if not cd_sel.any():
        return _EMPTY4
    cd = w[cd_sel]
    t_us = th_ff[cd_sel] * _TIME_HIGH_UNIT_US + ((cd >> np.uint32(22)) & np.uint32(0x3F)).astype(np.int64)
    x = ((cd >> np.uint32(11)) & np.uint32(0x7FF)).astype(np.int32)
    y = (cd & np.uint32(0x7FF)).astype(np.int32)
    p = (typ[cd_sel] & np.uint32(1)).astype(np.uint8)
    return x, y, p, t_us


def chunk_evt3(w, st):
    """EVT3.0: 16-bit, stateful. Event types: 0x0 ADDR_Y, 0x2 ADDR_X, 0x3 VECT_BASE_X,
    0x4 VECT_12, 0x5 VECT_8, 0x6 TIME_LOW, 0x8 TIME_HIGH. The time epoch/low, current y,
    vector base-x and base polarity all carry across chunk boundaries via *st*."""
    n = len(w)
    if not n:
        return _EMPTY4
    typ = (w >> np.uint16(12)).astype(np.uint8)
    data = (w & np.uint16(0x0FFF)).astype(np.int32)
    # --- timestamp: 12-bit high (epoch-wrapping) + 12-bit low ---
    thp = np.where(typ == 0x8)[0]
    if thp.size:
        raw = data[thp].astype(np.int64)
        prev = np.empty(raw.size, np.int64)
        prev[0] = st["last_raw"]
        prev[1:] = raw[:-1]
        epoch = st["epoch"] + np.cumsum((raw < prev).astype(np.int64))   # +1 each 12-bit-high wrap
        true_hi = epoch * 4096 + raw
        th_ff = _ffill(thp, true_hi, n, init=st["th_cum"])
        st["epoch"] = int(epoch[-1])
        st["last_raw"] = int(raw[-1])
        st["th_cum"] = int(true_hi[-1])
    else:
        th_ff = np.full(n, st["th_cum"], np.int64)
    tlp = np.where(typ == 0x6)[0]
    tl_ff = _ffill(tlp, data[tlp], n, init=st["tl"], dtype=np.int64)
    if tlp.size:
        st["tl"] = int(data[tlp][-1])
    t_full = th_ff * 4096 + tl_ff
    # --- current y row ---
    yp = np.where(typ == 0x0)[0]
    y_ff = _ffill(yp, data[yp] & 0x7FF, n, init=st["y"], dtype=np.int32)
    if yp.size:
        st["y"] = int(data[yp][-1] & 0x7FF)
    parts = [(data[typ == 0x2] & 0x7FF, y_ff[typ == 0x2],
              (data[typ == 0x2] >> 11) & 1, t_full[typ == 0x2])]
    # --- vectors (base x carried across chunks via base_x_next) ---
    is_base = typ == 0x3
    is_v12 = typ == 0x4
    is_v8 = typ == 0x5
    bpp = np.where(is_base)[0]
    base_pol_ff = _ffill(bpp, (data[bpp] >> 11) & 1, n, init=st["base_pol"], dtype=np.int32)
    if bpp.size:
        st["base_pol"] = int((data[bpp][-1] >> 11) & 1)
    width = np.where(is_v12, 12, np.where(is_v8, 8, 0)).astype(np.int64)
    cumw = np.cumsum(width)
    excl = cumw - width
    segidx = np.cumsum(is_base.astype(np.int64))
    seg_val = np.concatenate([[st["base_x_next"]], (data[bpp] & 0x7FF).astype(np.int64)]) \
        if bpp.size else np.array([st["base_x_next"]], np.int64)
    seg_excl = np.concatenate([[0], excl[bpp]]) if bpp.size else np.array([0], np.int64)
    base_x_word = seg_val[segidx] + (excl - seg_excl[segidx])
    cumw_total = int(cumw[-1]) if n else 0
    last_seg = int(segidx[-1]) if n else 0
    st["base_x_next"] = int(seg_val[last_seg] + (cumw_total - seg_excl[last_seg]))  # carry base x
    for is_v, wbits, mbits in ((is_v12, 12, 0xFFF), (is_v8, 8, 0xFF)):
        if is_v.any():
            m = data[is_v] & mbits
            bit, idx = _bit_positions(m, wbits)
            if idx.size:
                bx = base_x_word[is_v].astype(np.int32)
                parts.append((bx[idx] + bit, y_ff[is_v][idx],
                              base_pol_ff[is_v][idx], t_full[is_v][idx]))
    parts = [q for q in parts if len(q[3])]
    if not parts:
        return _EMPTY4
    return (np.concatenate([q[0] for q in parts]).astype(np.int32),
            np.concatenate([q[1] for q in parts]).astype(np.int32),
            np.concatenate([q[2] for q in parts]).astype(np.uint8),
            np.concatenate([q[3] for q in parts]).astype(np.int64))


#: format -> (chunk decoder, word dtype)
CHUNK = {"evt3": chunk_evt3, "evt2": chunk_evt2, "evt21": chunk_evt21}


def word_dtype(fmt: str) -> str:
    """NumPy dtype string for one encoded word of *fmt*."""
    return _WORD_DTYPE[fmt]


# ====================================================================================
# Full in-memory decode (small files / tests). Large files: gottlux.io.cache.load()
# ====================================================================================
def decode(path: str) -> dict:
    """Decode an entire ``.raw`` file into memory: time-sorted, zero-based µs.

    Returns a dict with ``x, y, p, t`` arrays plus ``t0_us, width, height, meta, fmt``.
    For multi-GB files prefer :func:`gottlux.io.cache.load`, which streams to a memmap
    cache and uses bounded RAM.
    """
    meta, off = parse_header(path)
    fmt = detect_format(meta)
    width, height = geometry(meta)
    wb = np.dtype(_WORD_DTYPE[fmt]).itemsize
    with open(ext(path), "rb") as f:
        f.seek(off)
        raw = f.read()
    w = np.frombuffer(raw[: (len(raw) // wb) * wb], dtype=_WORD_DTYPE[fmt])
    x, y, p, t = CHUNK[fmt](w, init_state(fmt))
    if not len(t):
        z16 = np.zeros(0, np.uint16)
        z8 = np.zeros(0, np.uint8)
        z64 = np.zeros(0, np.int64)
        return dict(x=z16, y=z16, p=z8, t=z64, t0_us=0,
                    width=width or 320, height=height or 320, meta=meta, fmt=fmt)
    order = np.argsort(t, kind="stable")
    x, y, p, t = x[order], y[order], p[order], t[order]
    x, y, p, t = strip_preroll(x, y, p, t)
    if width is None:
        width = int(x.max()) + 1 if len(x) else 320
    if height is None:
        height = int(y.max()) + 1 if len(y) else 320
    x = np.clip(x, 0, width - 1).astype(np.uint16)
    y = np.clip(y, 0, height - 1).astype(np.uint16)
    t0 = int(t[0]) if len(t) else 0
    return dict(x=x, y=y, p=p.astype(np.uint8), t=(t - t0).astype(np.int64),
                t0_us=t0, width=width, height=height, meta=meta, fmt=fmt)
