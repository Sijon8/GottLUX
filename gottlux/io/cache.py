"""
cache.py — decode-once, streaming, memmapped event cache.

Decoding the same file repeatedly is the single biggest waste of iteration time, and
materializing a billion-event stream in RAM is fatal on a 16 GB laptop. This module solves
both: a ``.raw`` is decoded exactly once, **chunk-by-chunk straight to disk** as four
memmappable ``.bin`` files (``x, y`` uint16, ``p`` uint8, ``t`` int64 µs), with the EVT3
decode state carried across chunk boundaries. Peak memory stays ~1–2 GB regardless of file
size; a subsequent open is an instant memmap (~ms).

HDF5 recordings (``.h5``/``.hdf5`` — see :mod:`gottlux.io.hdf5`) go through the very same
pipeline: the first open streams the file's event datasets in bounded blocks into the
identical four-bin layout (``fmt`` recorded as ``'hdf5'``), and every later open is the
same instant memmap hit a ``.raw`` gets. The cache stem keeps the full file name for
non-``.raw`` sources (``clip.h5`` → ``clip.h5.x.bin``), so a converted pair like
``clip.raw`` / ``clip.h5`` never fights over one cache.

The cache auto-invalidates when the source file changes, when the decoder version bumps,
or when the on-disk layout changes. It auto-relocates to a short platform path when the
data lives at a Windows-long path that would overflow ``MAX_PATH``.

Lifecycle management
--------------------
Caches are cheap to rebuild but not free to keep (~19 bytes/event ≈ 1.5–2× the ``.raw``),
so this module also provides the user-visible management API behind
``gottlux --cache-info`` / ``--clear-cache``: :func:`cache_info` discovers every cache for
a file, a capture folder (one level deep), or the platform relocation root;
:func:`clear_cache` deletes cache files safely (never the source data, and files that are
memmapped by another process are *skipped and reported*, not crashed on); and
:func:`format_cache_report` renders the report the CLI prints.

Cross-process resilience: when a needed re-decode finds its ``.bin`` files open in another
process (on Windows a memmapped file cannot be reopened for write — OSError EINVAL /
"user-mapped section open"), :func:`load` falls back automatically to decoding into a
fresh per-process temp cache dir under the platform relocation root instead of crashing.
Those temp dirs are reclaimed by ``gottlux --clear-cache`` on the relocation root.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading

import numpy as np

from gottlux.io import decode as _dec
from gottlux.io.hdf5 import is_hdf5_path
from gottlux.io.paths import ext, fits, platform_root, short_id

_CACHE_DIRNAME = "_gottlux_cache"

# One decode per cache stem at a time (within this process). Two threads asking for the
# same uncached file — e.g. the quick viewer's background decode plus the full suite it
# hands off to mid-preview — must not both stream into the same ``.bin`` files; the
# second waits and then opens the finished cache instantly.
_DECODE_LOCKS: dict = {}
_DECODE_LOCKS_GUARD = threading.Lock()


def _decode_lock(stem):
    with _DECODE_LOCKS_GUARD:
        return _DECODE_LOCKS.setdefault(os.path.normcase(os.path.abspath(stem)),
                                        threading.Lock())

#: Every dirname that can hold decoded caches — the discovery set for
#: :func:`cache_info` / :func:`clear_cache`.
_ALL_CACHE_DIRNAMES = (_CACHE_DIRNAME,)

#: Per-process map of original cache stem → temp fallback stem, so a stem whose bins are
#: held by another process is decoded into %TEMP%-style storage only once per process.
_FALLBACK_STEMS: dict = {}


class CacheBusyError(OSError):
    """The cache ``.bin`` files could not be (re)opened for writing — typically because
    another process has them memory-mapped (Windows: EINVAL / EACCES, "the requested
    operation cannot be performed on a file with a user-mapped section open")."""


def _open_bins(stem):
    """Open ``{stem}.{x,y,p,t}.bin`` for writing → an ``{key: file}`` dict.

    Raises :class:`CacheBusyError` when a bin cannot be (re)opened — typically memmapped
    by another process — so the caller can fall back to a temp cache dir.
    """
    fh = {}
    try:
        for k in ("x", "y", "p", "t"):
            fh[k] = open(f"{stem}.{k}.bin", "wb")
    except OSError as e:
        for h in fh.values():
            h.close()
        raise CacheBusyError(str(e)) from e
    return fh


def _decode_to_bin(path, stem, progress=None, chunk_words=20_000_000):
    """Stream-decode *path* to ``{stem}.{x,y,p,t}.bin`` one bounded chunk at a time.

    Returns a meta dict (``t0_us, width, height, n, n_on, fmt, meta``). *progress*, if
    given, is called with a fraction in [0, 1] (the GUI drives a progress bar from it).
    """
    meta, off = _dec.parse_header(path)
    fmt = _dec.detect_format(meta)
    width, height = _dec.geometry(meta)
    wb = np.dtype(_dec.word_dtype(fmt)).itemsize
    total_words = max((os.path.getsize(ext(path)) - off) // wb, 1)
    chunk = _dec.CHUNK[fmt]
    st = _dec.init_state(fmt)
    fh = _open_bins(stem)
    n_total = n_on = 0
    t0 = None
    first = True
    maxx = maxy = 0
    done = 0
    try:
        with open(ext(path), "rb") as f:
            f.seek(off)
            while True:
                buf = f.read(chunk_words * wb)
                if not buf:
                    break
                w = np.frombuffer(buf[: (len(buf) // wb) * wb], dtype=_dec.word_dtype(fmt))
                done += len(w)
                x, y, p, t = chunk(w, st)
                if progress:
                    try:
                        progress(min(done / total_words, 0.999))
                    except Exception:
                        pass
                if not len(t):
                    continue
                o = np.argsort(t, kind="stable")
                x, y, p, t = x[o], y[o], p[o], t[o]
                if first:
                    x, y, p, t = _dec.strip_preroll(x, y, p, t)
                    if not len(t):
                        continue
                    t0 = int(t[0])
                    first = False
                t = t - t0
                if width is not None:
                    np.clip(x, 0, width - 1, out=x)
                else:
                    maxx = max(maxx, int(x.max()))
                if height is not None:
                    np.clip(y, 0, height - 1, out=y)
                else:
                    maxy = max(maxy, int(y.max()))
                fh["x"].write(x.astype(np.uint16).tobytes())
                fh["y"].write(y.astype(np.uint16).tobytes())
                fh["p"].write(p.astype(np.uint8).tobytes())
                fh["t"].write(t.astype(np.int64).tobytes())
                n_total += len(t)
                n_on += int((p == 1).sum())
    finally:
        for k in fh:
            fh[k].close()
    if width is None:
        width = (maxx + 1) if n_total else 320
    if height is None:
        height = (maxy + 1) if n_total else 320
    if progress:
        try:
            progress(1.0)
        except Exception:
            pass
    return dict(t0_us=t0 or 0, width=width, height=height,
                n=n_total, n_on=n_on, fmt=fmt, meta=meta)


def _hdf5_to_bin(path, stem, progress=None, block_events=8_000_000):
    """Stream an HDF5 event file to ``{stem}.{x,y,p,t}.bin`` one bounded block at a time.

    The HDF5 twin of :func:`_decode_to_bin`: identical four-bin layout, identical
    normalization (per-block time sort, zero-basing to the first event, geometry clip /
    max-tracking, ON-count bookkeeping) — only the source differs (chunked dataset reads
    via :class:`gottlux.io.hdf5.H5EventSource` instead of a word decode, so HDF5's
    random access replaces the sequential decoder state). ``fmt`` is ``'hdf5'``.
    """
    from gottlux.io import hdf5 as _h5
    fh = _open_bins(stem)
    n_total = n_on = 0
    t0 = None
    maxx = maxy = 0
    done = 0
    try:
        with _h5.H5EventSource(path) as src:
            width, height = src.width, src.height
            meta = src.meta
            for x, y, p, t in src.blocks(block_events):
                done += len(t)
                if progress:
                    try:
                        progress(min(done / max(src.n, 1), 0.999))
                    except Exception:
                        pass
                if not len(t):
                    continue
                t = np.asarray(t).astype(np.int64)
                o = np.argsort(t, kind="stable")
                x = np.asarray(x)[o]; y = np.asarray(y)[o]
                p = np.asarray(p)[o]; t = t[o]
                if t0 is None:
                    t0 = int(t[0])
                t = t - t0
                p = (p > 0).astype(np.uint8)       # some tools store polarity as ±1
                if width is not None:
                    x = np.clip(x, 0, width - 1)
                else:
                    maxx = max(maxx, int(x.max()))
                if height is not None:
                    y = np.clip(y, 0, height - 1)
                else:
                    maxy = max(maxy, int(y.max()))
                fh["x"].write(x.astype(np.uint16).tobytes())
                fh["y"].write(y.astype(np.uint16).tobytes())
                fh["p"].write(p.astype(np.uint8).tobytes())
                fh["t"].write(t.astype(np.int64).tobytes())
                n_total += len(t)
                n_on += int((p == 1).sum())
    finally:
        for k in fh:
            fh[k].close()
    if width is None:
        width = (maxx + 1) if n_total else 320
    if height is None:
        height = (maxy + 1) if n_total else 320
    if progress:
        try:
            progress(1.0)
        except Exception:
            pass
    return dict(t0_us=t0 or 0, width=int(width), height=int(height),
                n=n_total, n_on=n_on, fmt="hdf5", meta=meta)


def _build_bins(path, stem, progress=None):
    """Stream *path* — a ``.raw`` or an HDF5 recording — into the four-bin cache layout."""
    if is_hdf5_path(path):
        return _hdf5_to_bin(path, stem, progress=progress)
    return _decode_to_bin(path, stem, progress=progress)


def _cache_base(path) -> str:
    """The cache stem basename for a data file.

    A ``.raw`` keeps the bare stem (the layout every existing cache already uses); any
    other source keeps its full file name (``clip.h5`` → ``clip.h5.x.bin`` …) so two
    sources sharing a stem — e.g. ``clip.raw`` and its converted ``clip.h5`` — get two
    independent caches instead of silently reusing each other's.
    """
    name = os.path.basename(path)
    stem, suffix = os.path.splitext(name)
    return stem if suffix.lower() == ".raw" else name


def cache_location(path, cache_dir=None):
    """Resolve the cache directory + file stem for *path*, relocating off long paths."""
    cap_dir = os.path.dirname(os.path.abspath(path))
    base = _cache_base(path)
    if cache_dir is None:
        cache_dir = os.path.join(cap_dir, _CACHE_DIRNAME)
        if not fits(os.path.join(cache_dir, base + ".meta.json"), headroom=16):
            cache_dir = os.path.join(platform_root(), _CACHE_DIRNAME, short_id(cap_dir))
    return cache_dir, os.path.join(cache_dir, base)


def _meta_fresh(path, meta_path) -> bool:
    """True iff *meta_path* describes a fresh, layout-compatible decode of *path*."""
    try:
        if not os.path.exists(meta_path) or \
                os.path.getmtime(ext(path)) > os.path.getmtime(meta_path):
            return False
        with open(meta_path) as f:
            m = json.load(f)
        return m.get("decoder_version") == _dec.DECODER_VERSION and m.get("layout") == "bin"
    except Exception:
        return False


def has_valid_cache(path, cache_dir=None) -> bool:
    """True iff a fresh, layout-compatible decode of *path* is already on disk — i.e.
    :func:`load` would open it instantly. The sampled-preview policy uses this: a cache
    hit needs no preview."""
    _, stem = cache_location(path, cache_dir)
    return _meta_fresh(path, stem + ".meta.json")


def _fallback_stem(stem):
    """A fresh per-process temp cache stem for *stem* (whose bins another process holds).

    Lives in ``tempfile.mkdtemp`` under the platform relocation root
    (``<platform_root>/_gottlux_cache/tmp*``) so a later ``gottlux --clear-cache`` on the
    relocation root reclaims it. Remembered in :data:`_FALLBACK_STEMS` so one process
    decodes at most once per stem.
    """
    root = os.path.join(platform_root(), _CACHE_DIRNAME)
    os.makedirs(root, exist_ok=True)
    d = tempfile.mkdtemp(prefix="tmp_", dir=root)
    fb = os.path.join(d, os.path.basename(stem))
    _FALLBACK_STEMS[os.path.normcase(os.path.abspath(stem))] = fb
    return fb


def load(path, cache_dir=None, force=False, progress=None) -> dict:
    """Decode-once into a streamed memmap cache and return memmap-backed arrays.

    *path* is a ``.raw`` or an HDF5 recording (``.h5``/``.hdf5``) — both stream into the
    same bin layout. Returns a dict: ``x, y, p, t`` memmaps plus ``t0_us, width, height,
    n, n_on, fmt, meta, source_path``. Cheap to call repeatedly — only the first call
    decodes.
    """
    cache_dir, stem = cache_location(path, cache_dir)
    meta_path = stem + ".meta.json"
    with _decode_lock(stem):                       # the need-check re-runs under the lock, so a
        need = force or not _meta_fresh(path, meta_path)
        if need and not force:                     # thread that waited sees the fresh cache
            fb = _FALLBACK_STEMS.get(os.path.normcase(os.path.abspath(stem)))
            if fb and _meta_fresh(path, fb + ".meta.json"):
                stem, meta_path = fb, fb + ".meta.json"   # this process already fell back
                need = False
        if need:
            os.makedirs(cache_dir, exist_ok=True)
            try:
                info = _build_bins(path, stem, progress=progress)
            except CacheBusyError:
                # The existing bins are memmapped by another process (Windows: a
                # user-mapped file cannot be reopened for write — EINVAL/EACCES). Decode
                # into a per-process temp cache dir under the platform relocation root
                # instead of crashing; --clear-cache on that root reclaims these.
                stem = _fallback_stem(stem)
                meta_path = stem + ".meta.json"
                info = _build_bins(path, stem, progress=progress)
            with open(meta_path, "w") as f:
                json.dump(dict(t0_us=info["t0_us"], width=info["width"], height=info["height"],
                               n=info["n"], n_on=info.get("n_on", 0),
                               decoder_version=_dec.DECODER_VERSION, layout="bin",
                               fmt=info["fmt"], meta=info["meta"]), f, indent=2)
        elif progress:
            try:
                progress(1.0)                      # cache hit -> report complete immediately
            except Exception:
                pass
    with open(meta_path) as f:
        info = json.load(f)
    n = int(info["n"])

    def mm(k, dt):
        fp = f"{stem}.{k}.bin"
        if n == 0 or not os.path.exists(fp):
            return np.zeros(0, dt)
        return np.memmap(fp, dtype=dt, mode="r", shape=(n,))

    return dict(x=mm("x", np.uint16), y=mm("y", np.uint16),
                p=mm("p", np.uint8), t=mm("t", np.int64),
                t0_us=info["t0_us"], width=info["width"], height=info["height"], n=n,
                n_on=int(info.get("n_on", 0)), fmt=info.get("fmt", "evt21"),
                meta=info["meta"], source_path=os.path.abspath(path))


def open_cached(stem) -> dict | None:
    """Open an already-decoded cache by its file *stem* (no source ``.raw`` needed).

    This is what lets gottlux load bare ``_gottlux_cache`` bins directly (e.g. when only
    the decoded data survives, not the original capture).
    Returns ``None`` if the meta or bins are missing.
    """
    meta_path = stem + ".meta.json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        info = json.load(f)
    n = int(info["n"])

    def mm(k, dt):
        fp = f"{stem}.{k}.bin"
        if n == 0 or not os.path.exists(fp):
            return np.zeros(0, dt)
        return np.memmap(fp, dtype=dt, mode="r", shape=(n,))

    return dict(x=mm("x", np.uint16), y=mm("y", np.uint16),
                p=mm("p", np.uint8), t=mm("t", np.int64),
                t0_us=info["t0_us"], width=info["width"], height=info["height"], n=n,
                n_on=int(info.get("n_on", 0)), fmt=info.get("fmt", "evt21"),
                meta=info.get("meta", {}), source_path=os.path.abspath(stem))


# --------------------------------------------------------------------- lifecycle management
def _stems_in(cache_dir):
    """Yield ``(cache_dir, base)`` for every cache stem directly inside *cache_dir*, then
    one level down (the relocation root nests stems in ``<short_id>/`` and ``tmp*/``)."""
    try:
        entries = sorted(os.listdir(ext(cache_dir)))
    except OSError:
        return
    for e in entries:
        full = os.path.join(cache_dir, e)
        if e.endswith(".meta.json") and os.path.isfile(ext(full)):
            yield cache_dir, e[: -len(".meta.json")]
        elif os.path.isdir(ext(full)):
            try:
                subs = sorted(os.listdir(ext(full)))
            except OSError:
                continue
            for s in subs:
                if s.endswith(".meta.json") and os.path.isfile(ext(os.path.join(full, s))):
                    yield full, s[: -len(".meta.json")]


def _discover_stems(path_or_dir):
    """Yield unique ``(cache_dir, base)`` pairs for *path_or_dir* — a data file, a cache
    dir, a capture folder (one level of subfolders included), or the relocation root."""
    p = os.path.abspath(path_or_dir)
    seen = set()

    def emit(pairs):
        for cd, base in pairs:
            key = os.path.normcase(os.path.join(cd, base))
            if key not in seen:
                seen.add(key)
                yield cd, base

    if os.path.isfile(ext(p)):                     # one file → every dir holding its stem
        cap_dir = os.path.dirname(p)
        base = _cache_base(p)
        candidates = [os.path.join(cap_dir, name) for name in _ALL_CACHE_DIRNAMES]
        candidates.append(cache_location(p)[0])    # incl. the (possibly relocated) choice
        yield from emit((cd, base) for cd in candidates
                        if os.path.exists(ext(os.path.join(cd, base + ".meta.json"))))
        return
    if not os.path.isdir(ext(p)):
        return
    if os.path.basename(p) in _ALL_CACHE_DIRNAMES:   # a cache dir itself
        yield from emit(_stems_in(p))
        return
    roots = [p]                                    # a capture folder: recurse one level
    try:
        roots += [os.path.join(p, e) for e in sorted(os.listdir(ext(p)))
                  if os.path.isdir(ext(os.path.join(p, e)))
                  and e not in _ALL_CACHE_DIRNAMES]
    except OSError:
        pass
    for root in roots:
        for name in _ALL_CACHE_DIRNAMES:
            yield from emit(_stems_in(os.path.join(root, name)))


def _find_source(cache_dir, base):
    """The source data file a beside-the-data cache was decoded from (or ``None``).

    Matches the bare-stem convention of a ``.raw`` cache and the full-file-name stems
    non-``.raw`` sources use (``clip.h5.meta.json`` ← ``clip.h5``)."""
    cap_dir = os.path.dirname(cache_dir)
    try:
        names = os.listdir(ext(cap_dir))
    except OSError:
        return None
    hits = [f for f in names
            if (f == base or os.path.splitext(f)[0] == base)
            and not f.endswith((".bin", ".meta.json"))
            and os.path.isfile(ext(os.path.join(cap_dir, f)))]
    hits.sort(key=lambda f: (not f.lower().endswith(".raw"), f))   # prefer the .raw
    return os.path.join(cap_dir, hits[0]) if hits else None


def cache_info(path_or_dir=None) -> list:
    """Discover decode caches and return one entry dict per cache stem.

    *path_or_dir* may be a data file, a capture folder (subfolders one level down are
    included), a cache directory itself, or the platform relocation root; ``None`` means
    the current directory. Each entry has: ``source`` (the data file, or ``None`` for a
    bare/relocated cache), ``cache_dir``, ``stem`` (basename), ``files`` (the cache files
    on disk), ``bytes``, ``n`` (event count, ``None`` if the meta is unreadable),
    ``decoder_version``, and ``stale`` (a re-decode would be needed: source newer,
    decoder or layout changed, bins missing, or meta unreadable).
    """
    entries = []
    for cache_dir, base in _discover_stems(path_or_dir or os.getcwd()):
        stem = os.path.join(cache_dir, base)
        meta_path = stem + ".meta.json"
        n = dv = None
        stale = True
        try:
            with open(ext(meta_path)) as f:
                m = json.load(f)
            n = int(m.get("n", 0))
            dv = m.get("decoder_version")
            stale = dv != _dec.DECODER_VERSION or m.get("layout") != "bin"
        except Exception:
            pass
        files = [meta_path] + [f"{stem}.{k}.bin" for k in ("x", "y", "p", "t")
                               if os.path.exists(ext(f"{stem}.{k}.bin"))]
        if not stale and n:                        # incomplete bins → a re-decode is due
            stale = len(files) < 5
        source = _find_source(cache_dir, base)
        if not stale and source is not None:
            stale = not _meta_fresh(source, meta_path)
        size = 0
        for fp in files:
            try:
                size += os.path.getsize(ext(fp))
            except OSError:
                pass
        entries.append(dict(source=source, cache_dir=cache_dir, stem=base, files=files,
                            bytes=size, n=n, decoder_version=dv, stale=stale))
    return entries


def clear_cache(path_or_dir=None, stale_only=False) -> dict:
    """Delete the decode caches under *path_or_dir* (see :func:`cache_info` for what is
    discovered — incl. the relocation root's per-process temp dirs).

    **Never touches the source data** — only ``*.bin`` + ``*.meta.json`` inside cache
    dirs. Files that cannot be deleted (memmapped by a running session) are skipped and
    reported, never raised on. With *stale_only*, fresh caches are kept. Returns
    ``{removed, skipped, freed_bytes, n_stems}``; ``skipped`` is ``[(path, error), …]``.
    """
    removed, skipped, freed, n_stems, dirs = [], [], 0, 0, []
    for e in cache_info(path_or_dir):
        if stale_only and not e["stale"]:
            continue
        n_stems += 1
        dirs.append(e["cache_dir"])
        for fp in e["files"]:
            try:
                size = os.path.getsize(ext(fp))
                os.remove(ext(fp))
                freed += size
                removed.append(fp)
            except OSError as err:                 # in use (memmapped) elsewhere → report
                skipped.append((fp, getattr(err, "strerror", None) or str(err)))
    for d in dict.fromkeys(dirs):                  # prune now-empty cache dirs (best effort)
        for prune in (d, os.path.dirname(d)):
            if os.path.basename(prune) in _ALL_CACHE_DIRNAMES or \
                    os.path.basename(os.path.dirname(prune)) in _ALL_CACHE_DIRNAMES:
                try:
                    os.rmdir(ext(prune))
                except OSError:
                    break                          # not empty (or gone) — stop pruning up
    return dict(removed=removed, skipped=skipped, freed_bytes=freed, n_stems=n_stems)


def _fmt_bytes(n) -> str:
    """``1234567`` → ``'1.2 MB'`` (decimal units, one decimal)."""
    x = float(n)
    for unit in ("B", "kB", "MB", "GB"):
        if x < 1000 or unit == "GB":
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1000.0
    return f"{x:.1f} GB"


def format_cache_report(entries) -> str:
    """Render :func:`cache_info` entries as the human-readable ``--cache-info`` report."""
    if not entries:
        return "No decode caches found."
    total = sum(e["bytes"] for e in entries)
    stale = [e for e in entries if e["stale"]]
    lines = [f"Decode caches — {len(entries)} stem(s), {_fmt_bytes(total)} total"
             + (f" ({len(stale)} stale, {_fmt_bytes(sum(e['bytes'] for e in stale))})"
                if stale else "")]
    for e in entries:
        tag = "STALE" if e["stale"] else "fresh"
        ev = f"{e['n']:,} ev" if e["n"] is not None else "?, meta unreadable"
        src = os.path.basename(e["source"]) if e["source"] else "(no source file)"
        lines.append(f"  [{tag}] {e['stem']:<24s} {ev:>14s}  {_fmt_bytes(e['bytes']):>9s}"
                     f"  v{e['decoder_version']}  ·  {src}")
        lines.append(f"          in {e['cache_dir']}")
    lines.append("Reclaim with:  gottlux --clear-cache [PATH]   (--stale-only keeps fresh caches)")
    return "\n".join(lines)
