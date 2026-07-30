"""
writer.py — encode events back to a valid Prophesee EVT2.1 ``.raw``, and cut/stitch clips.

The inverse of the decoder: given event arrays, emit a byte-valid EVT2.1 stream (CD words
+ TIME_HIGH markers) that Metavision Studio and gottlux both read back identically. Used to
**cut clips** (a time and/or ROI subrange written as a new self-contained ``.raw``), which
is how the GUI's video-editor-style trimming and the headless ``--cut`` work.

Why this is **streamed**
------------------------
A multi-GB recording holds hundreds of millions of events, so building the whole encoded word
array in RAM — plus the ``2n`` sort keys and the ``argsort`` the old encoder used to interleave
TIME_HIGH markers — needed ~5–6× the clip in memory and blew up on large crops. Two facts make
that unnecessary:

* the decoded cache is **already time-sorted**, so cutting/stitching needs **no global sort**; and
* the EVT2.1 stream is local — a TIME_HIGH word is only needed when the 64-µs high field changes.

So the encoder works in **bounded blocks** (:class:`RawWriter`): each block of already-sorted
events is encoded to words and appended straight to disk, carrying just the last TIME_HIGH value
across the boundary. Peak memory is one block (~hundreds of MB) regardless of how large the clip
is, and :func:`cut_clip` / :func:`stitch_clips` slice the memmap block-by-block rather than
materializing the whole window. Output is byte-identical to the old one-shot encoder.
"""
from __future__ import annotations

import os

import numpy as np

from gottlux.io.paths import ext

#: Events per streamed block — caps the encoder's peak RAM (a few hundred MB) regardless of how
#: long the clip is. ~4M keeps the transient working set well under ~½ GB on a 16 GB laptop.
BLOCK = 4_000_000

_TYPE_SHIFT = np.uint64(60)
_TS_LOW_SHIFT = np.uint64(54)
_X_SHIFT = np.uint64(43)
_Y_SHIFT = np.uint64(32)
_TH_TYPE = np.uint64(0x8) << np.uint64(60)
_M6 = np.uint64(0x3F)
_M28 = np.uint64(0x0FFFFFFF)


def _default_meta(width, height):
    return {"format": f"EVT21;height={height};width={width}",
            "geometry": f"{width}x{height}", "generation": "320.0"}


def _is_sorted(t) -> bool:
    return t.size < 2 or bool(np.all(t[1:] >= t[:-1]))


def _encode_block(x, y, p, t_us, last_th):
    """Encode one block of **time-sorted** events to EVT2.1 ``<u8`` words.

    *last_th* is the 64-µs high field of the previous block's final event (or ``None`` for the
    first block); a TIME_HIGH word is emitted wherever the high field changes — including at the
    block's first event if it differs from *last_th*. No sort and no ``2n`` key array: CD and
    TIME_HIGH words are placed by direct indexing. Returns ``(words, new_last_th)``.
    """
    x = np.asarray(x).astype(np.uint64); y = np.asarray(y).astype(np.uint64)
    p = np.asarray(p).astype(np.uint64); ts = np.asarray(t_us).astype(np.uint64)
    n = ts.shape[0]
    if n == 0:
        return np.zeros(0, "<u8"), last_th
    ts_high = ts >> np.uint64(6)
    ts_low = ts & _M6
    cd = (p << _TYPE_SHIFT) | (ts_low << _TS_LOW_SHIFT) | (x << _X_SHIFT) \
        | (y << _Y_SHIFT) | np.uint64(1)
    change = np.empty(n, bool)
    change[0] = (last_th is None) or (int(ts_high[0]) != int(last_th))
    if n > 1:
        change[1:] = ts_high[1:] != ts_high[:-1]
    th_pos = np.nonzero(change)[0]
    k = th_pos.shape[0]
    th = _TH_TYPE | ((ts_high[th_pos] & _M28) << _Y_SHIFT)
    out = np.empty(n + k, np.uint64)
    ar = np.arange(n)
    # CD word i lands at i + (number of TIME_HIGH words at positions <= i); each TIME_HIGH lands
    # immediately before the CD word it precedes — the exact interleave the one-shot encoder made.
    out[ar + np.searchsorted(th_pos, ar, side="right")] = cd
    out[th_pos + np.arange(k)] = th
    return out.astype("<u8"), int(ts_high[-1])


class RawWriter:
    """Append-only EVT2.1 ``.raw`` writer that encodes **time-sorted** event blocks to disk.

    Open it, call :meth:`write_block` with successive blocks whose times are globally
    non-decreasing (and already re-based to start at ≥ 0), then :meth:`close`. Peak memory is one
    block. ``n`` is the running event count. Usable as a context manager.
    """

    def __init__(self, path, width=320, height=320, meta=None):
        self.path = ext(path)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        header = "".join(f"% {k} {v}\n" for k, v in (meta or _default_meta(width, height)).items())
        header += "% end\n"
        self._f = open(self.path, "wb")
        self._f.write(header.encode("ascii", "replace"))
        self._last_th = None
        self.n = 0

    def write_block(self, x, y, p, t_us) -> int:
        m = int(np.asarray(t_us).shape[0])
        if m == 0:
            return self.n
        words, self._last_th = _encode_block(x, y, p, t_us, self._last_th)
        self._f.write(words.tobytes())
        self.n += m
        return self.n

    def close(self) -> int:
        if self._f is not None:
            self._f.close()
            self._f = None
        return self.n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def write_raw(path, x, y, p, t_us, width=320, height=320, meta=None, block=BLOCK) -> int:
    """Encode events to a valid EVT2.1 ``.raw`` file. Returns the number of events written.

    Events are time-sorted if needed (a no-op when they already are, e.g. straight off the cache),
    then encoded **block-by-block** so even a huge in-memory array never materializes its full
    encoded form at once. ``meta`` overrides the header key/values (geometry filled in if omitted).
    """
    x = np.asarray(x); y = np.asarray(y); p = np.asarray(p)
    ts = np.asarray(t_us)
    n = len(ts)
    if n:
        ts = ts.astype(np.int64)
        if not _is_sorted(ts):
            order = np.argsort(ts, kind="stable")
            x, y, p, ts = x[order], y[order], p[order], ts[order]
    with RawWriter(path, width, height, meta) as w:
        for s in range(0, n, block):
            e = min(s + block, n)
            w.write_block(x[s:e], y[s:e], p[s:e], ts[s:e])
    return w.n


def cut_clip(rec, out_path, t0=None, t1=None, roi=None, t_origin_us=None,
             block=BLOCK, progress=None) -> int:
    """Write events of *rec* in ``[t0, t1)`` seconds (and optional ROI) to a new ``.raw``.

    Streams the slice **block-by-block** straight off the memmapped cache, so cropping a clip out
    of a multi-GB recording uses a bounded amount of RAM no matter how long the kept span is.

    Times in the new file are re-zeroed. By default the origin is the clip's first kept event;
    pass *t_origin_us* to re-base every output to a **common** origin instead — what a batch trim
    of several synced cameras needs so they stay frame-aligned. *progress*, if given, is called
    with a fraction in ``[0, 1]``. Returns events written.
    """
    i0 = 0 if t0 is None else rec.index_at(t0, "left")
    i1 = rec.n if t1 is None else rec.index_at(t1, "left")
    with RawWriter(out_path, rec.width, rec.height) as w:
        if i1 > i0:
            origin = np.int64(t_origin_us) if t_origin_us is not None else np.int64(rec.t[i0])
            total = i1 - i0
            for s in range(i0, i1, block):
                e = min(s + block, i1)
                xs = np.asarray(rec.x[s:e]); ys = np.asarray(rec.y[s:e])
                ps = np.asarray(rec.p[s:e]); ts = np.asarray(rec.t[s:e]).astype(np.int64)
                if roi is not None:
                    x0, y0, x1, y1 = roi
                    m = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
                    xs, ys, ps, ts = xs[m], ys[m], ps[m], ts[m]
                w.write_block(xs, ys, ps, np.clip(ts - origin, 0, None))
                if progress:
                    try:
                        progress(min((e - i0) / total, 1.0))
                    except Exception:
                        pass
    return w.n


def cut_file(in_path, out_path, t0=None, t1=None, roi=None, progress=None) -> int:
    """Cut ``[t0, t1)`` seconds (and optional ROI) of a ``.raw`` **file** to a new ``.raw``.

    The smart entry point for cropping a file you may not have open: for an **EVT2.1** file with no
    ROI it cuts **directly on the bytes — no decode** (fast, bounded RAM; see
    :mod:`gottlux.io.rawcut`); otherwise (an ROI crop, or an EVT2.0 / EVT3 file) it falls back to
    decoding into the cache and streaming with :func:`cut_clip`. Returns events written.
    """
    if roi is None:
        from gottlux.io import rawcut
        try:
            return int(rawcut.cut_evt21(in_path, out_path, t0=t0, t1=t1, progress=progress)["n_events"])
        except rawcut.UnsupportedRaw:
            pass                                       # not EVT2.1 → decode-based fallback below
    import gottlux as eb
    rec = eb.load(in_path, progress=(lambda f: progress(0.5 * f)) if progress else None)
    return cut_clip(rec, out_path, t0=t0, t1=t1, roi=roi,
                    progress=(lambda f: progress(0.5 + 0.5 * f)) if progress else None)


def trim_folder(folder, t0=0.0, t1=None, out_subdir="trimmed", pattern="*.raw",
                copy_suffixes=(".bias",), progress=None) -> dict:
    """Trim **every** ``.raw`` in *folder* by the SAME ``[t0, t1)`` window into
    ``folder/out_subdir/``, re-based to one **common origin** (``t0``) so synchronized clips
    (e.g. a bifocal cam0/cam1 pair) keep their slate alignment. Output files keep their original
    names; matching sidecars (``.bias`` …) are copied alongside. ``t1=None`` trims to the end.

    Returns a manifest dict (also written as ``trim_manifest.json`` in the output folder).
    """
    import glob
    import json
    import shutil

    import gottlux as eb
    folder = os.path.abspath(folder)
    out_dir = os.path.join(folder, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    raws = sorted(glob.glob(os.path.join(folder, pattern)))
    origin_us = int(round(float(t0) * 1e6))
    clips = []
    for raw in raws:
        rec = eb.load(raw, progress=lambda f: None)
        lo = max(float(t0), rec.t_start_s)
        hi = rec.t_stop_s if t1 is None else float(t1)
        out = os.path.join(out_dir, os.path.basename(raw))
        n = cut_clip(rec, out, t0=t0, t1=t1, t_origin_us=origin_us)
        stem = os.path.splitext(raw)[0]
        copied = []
        for suf in copy_suffixes:                      # carry the bias (etc.) sidecar across
            sc = stem + suf
            if os.path.exists(sc):
                dst = os.path.join(out_dir, os.path.basename(sc))
                shutil.copy2(sc, dst); copied.append(os.path.basename(dst))
        clips.append({"src": os.path.basename(raw), "out": os.path.basename(out),
                      "n_events": int(n), "kept_s": round(max(hi - lo, 0.0), 4),
                      "src_duration_s": round(rec.duration_s, 4), "sidecars": copied})
        if progress:
            progress(raw, n)
    manifest = {"source_folder": folder, "out_dir": out_dir,
                "window_s": [float(t0), (None if t1 is None else float(t1))],
                "common_origin_us": origin_us, "n_clips": len(clips), "clips": clips}
    try:
        with open(os.path.join(out_dir, "trim_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        pass
    return manifest


def _spec_parts(spec):
    """Normalize a stitch spec → ``(rec, t0, t1, roi)`` (a bare Recording means its whole span)."""
    if hasattr(spec, "window"):
        return spec, None, None, None
    rec, t0, t1 = spec[0], spec[1], spec[2]
    return rec, t0, t1, (spec[3] if len(spec) > 3 else None)


def stitch_clips(out_path, clips, gap_s=0.0, block=BLOCK, progress=None) -> dict:
    """Concatenate several (trimmed) clips end-to-end into one valid EVT2.1 ``.raw``.

    *clips* is an ordered list of segments — each a :class:`~gottlux.io.recording.Recording`
    or a ``(rec, t0, t1)`` / ``(rec, t0, t1, roi)`` tuple selecting a sub-window (and optional
    ROI). Every segment is **streamed** in blocks, re-based so its events follow the previous
    segment's on a single monotonic clock (with an optional ``gap_s`` blank between them), so the
    merge never holds more than one block in RAM. This is the engine behind the timeline editor's
    "stitch into one file" — append, trim, reorder.

    All segments must share the sensor geometry. Returns ``{"path", "n_events", "duration_s",
    "segments"}`` (segments carry their placed time spans).
    """
    specs = [_spec_parts(s) for s in clips]
    W = H = None
    for rec, _t0, _t1, _roi in specs:
        W, H = rec.width, rec.height
        break
    total = 0
    for rec, t0, t1, _roi in specs:
        i0 = 0 if t0 is None else rec.index_at(t0, "left")
        i1 = rec.n if t1 is None else rec.index_at(t1, "left")
        total += max(i1 - i0, 0)
    total = max(total, 1)

    segments = []
    cursor_us = np.int64(0)
    gap_us = np.int64(max(int(round(gap_s * 1e6)), 0))
    done = 0
    w = RawWriter(out_path, W or 320, H or 320)
    try:
        for i, (rec, t0, t1, roi) in enumerate(specs):
            if (rec.width, rec.height) != (W, H):
                raise ValueError(f"stitch_clips: clip {i} geometry {rec.width}×{rec.height} "
                                 f"≠ {W}×{H}; all clips must share the sensor geometry")
            i0 = 0 if t0 is None else rec.index_at(t0, "left")
            i1 = rec.n if t1 is None else rec.index_at(t1, "left")
            placed_t0 = float(cursor_us) / 1e6
            n_before = w.n
            if i1 <= i0:
                segments.append({"index": i, "name": rec.name, "n": 0,
                                 "placed_t0_s": placed_t0, "placed_t1_s": placed_t0})
                continue
            seg_origin = np.int64(rec.t[i0])
            seg_cursor = cursor_us
            last_t = int(cursor_us)
            for s in range(i0, i1, block):
                e = min(s + block, i1)
                xs = np.asarray(rec.x[s:e]); ys = np.asarray(rec.y[s:e])
                ps = np.asarray(rec.p[s:e]); ts = np.asarray(rec.t[s:e]).astype(np.int64)
                if roi is not None:
                    x0, y0, x1, y1 = roi
                    m = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
                    xs, ys, ps, ts = xs[m], ys[m], ps[m], ts[m]
                t_new = (ts - seg_origin) + seg_cursor
                w.write_block(xs, ys, ps, t_new)
                if t_new.size:
                    last_t = int(t_new[-1])
                done += (e - s)
                if progress:
                    try:
                        progress(min(done / total, 1.0))
                    except Exception:
                        pass
            segments.append({"index": i, "name": rec.name, "n": int(w.n - n_before),
                             "placed_t0_s": placed_t0, "placed_t1_s": float(last_t) / 1e6})
            cursor_us = np.int64(last_t) + gap_us
    finally:
        w.close()
    return {"path": out_path, "n_events": w.n, "duration_s": float(cursor_us) / 1e6,
            "segments": segments}
