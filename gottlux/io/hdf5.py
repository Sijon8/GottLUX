"""
hdf5.py — HDF5 event files: convert to, read from, open everywhere.

HDF5 is the interchange format the wider event-vision world speaks (Metavision Studio
exports it; the Python data stacks read it natively), so gottlux treats it as a
first-class recording container next to ``.raw``:

* :func:`write_hdf5` — stream a :class:`~gottlux.io.recording.Recording` (or a ``.raw``
  path) to a Metavision-compatible file: one compound ``CD/events`` dataset (fields
  ``x, y, p, t``), chunked + gzip-compressed, **appended in bounded blocks** so a multi-GB
  recording converts inside the same ~1–2 GB envelope the decoder keeps. Sensor geometry
  is stored as root attributes, and a small ``gottlux`` attrs group records provenance
  (source file name, the absolute ``t0_us`` origin, the gottlux version). An optional
  window/ROI exports a sub-clip with :func:`~gottlux.io.writer.cut_clip` semantics.
* :func:`read_events` / :class:`H5EventSource` — read them back, accepting the Metavision
  compound layout **and** plain parallel ``x/y/p/t`` datasets (at the root or under an
  ``events/`` group). :func:`read_events` materializes a whole file into the same dict
  shape the decode cache produces; :class:`H5EventSource` streams bounded blocks, which is
  how :mod:`gottlux.io.cache` builds its ``.bin`` cache from an ``.h5`` — after which the
  file opens instantly everywhere a ``.raw`` does (GUI, quick viewer, CLI, library).
* **ECF**: Metavision's "compress on save" output stores the events with Prophesee's ECF
  codec (HDF5 filter 36559), which stock h5py cannot decompress. The reader imports
  :mod:`hdf5plugin` when available (registering the extra codecs its build ships); when a
  file still refuses to decompress, the read raises a clear
  :class:`~gottlux.io.fusion.FusionError` naming the opt-in fix
  (``pip install gottlux[hdf5]``). The plugin wheel is large, so it is deliberately NOT
  part of the ``data``/``all`` extras.

Pure NumPy + (lazily imported) h5py; no Qt, no matplotlib.
"""
from __future__ import annotations

import os

import numpy as np

#: Recording file extensions this module owns.
HDF5_EXTS = (".h5", ".hdf5")

#: The compound row dtype of a Metavision ``CD/events`` dataset (14 bytes/event).
EVENT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")])

#: Events per HDF5 chunk in files we write (~3.7 MB of rows — a good gzip/IO unit).
H5_CHUNK_EVENTS = 262_144

#: Events per streamed block on both the write and read paths — bounds peak RAM the same
#: way the ``.raw`` decoder's word chunking does (~112 MB of compound rows per block).
BLOCK_EVENTS = 8_000_000

#: Whether the last :func:`_h5py` call could import ``hdf5plugin`` (decides if the ECF
#: error message should suggest installing it — no point when it is already present).
_HDF5PLUGIN_OK = False


def is_hdf5_path(path) -> bool:
    """True iff *path* names an HDF5 recording by extension (``.h5`` / ``.hdf5``)."""
    return str(path).lower().endswith(HDF5_EXTS)


def _err(msg):
    """A :class:`~gottlux.io.fusion.FusionError` (imported lazily — no import cycle)."""
    from gottlux.io.fusion import FusionError
    return FusionError(msg)


def _h5py():
    """Import h5py (+ best-effort ``hdf5plugin``) or raise the actionable error.

    ``hdf5plugin``, when installed, registers its extra HDF5 decompression filters at
    import — including Prophesee's ECF codec (filter 36559) in builds that ship it — which
    is exactly what reading a Metavision "compress on save" file needs. Its absence is not
    an error here; it only sharpens the message when an ECF read actually fails.
    """
    global _HDF5PLUGIN_OK
    try:
        import h5py
    except Exception as e:                          # pragma: no cover
        raise _err(f"reading/writing HDF5 needs h5py ({e}); pip install h5py")
    try:
        import hdf5plugin  # noqa: F401  — importing it registers the filter plugins
        _HDF5PLUGIN_OK = True
    except Exception:
        _HDF5PLUGIN_OK = False
    return h5py


def _codec_error(path, e):
    """The clear, actionable error for an event dataset that refuses to decompress.

    In practice this is Prophesee's ECF codec (HDF5 filter 36559 — the Metavision
    "compress on save" output) read without the codec registered. The opt-in install hint
    is appended only when ``hdf5plugin`` is actually absent.
    """
    msg = (f"{path!r}: the event stream is stored with a compression codec that is not "
           f"registered — in practice Prophesee's ECF codec (HDF5 filter 36559) — and "
           f"cannot be decompressed here ({e}). Convert from the original .raw recording "
           f"instead of this HDF5.")
    if not _HDF5PLUGIN_OK:
        msg += (" pip install gottlux[hdf5] (hdf5plugin) enables reading "
                "ECF-compressed files.")
    return _err(msg)


def _geometry_from_attrs(attrs):
    """Sensor ``(width, height)`` as declared by the file's attributes, or ``(None, None)``.

    Tries, in order: integer ``width``/``height`` attrs (what :func:`write_hdf5` stores), a
    ``geometry`` ``"WxH"`` string, and a Metavision ``format`` string with ``width=`` /
    ``height=`` tokens. Callers fall back to the maximum observed coordinate when the file
    does not say.
    """
    try:
        return int(attrs["width"]), int(attrs["height"])
    except (KeyError, TypeError, ValueError):
        pass
    geom = str(attrs.get("geometry", "")).lower()
    if "x" in geom:
        try:
            w, h = (int(v) for v in geom.split("x")[:2])
            return w, h
        except ValueError:
            pass
    fmt = str(attrs.get("format", ""))
    w = h = None
    for part in fmt.replace(";", " ").split():
        if part.startswith("width="):
            w = int(part.split("=")[1])
        elif part.startswith("height="):
            h = int(part.split("=")[1])
    if w and h:
        return w, h
    return None, None


def _locate(f, path):
    """Find the event data in an open HDF5 file → ``(kind, node)``.

    *kind* ``"compound"``: a Metavision-style compound dataset with ``x/y/p/t`` fields
    (``CD/events``, or a root-level ``events`` dataset). *kind* ``"plain"``: four parallel
    1-D datasets ``x, y, p, t`` — inside an ``events/`` group or at the root — returned as
    a name→dataset dict. Anything else raises the actionable :class:`FusionError`.
    """
    import h5py
    fields = ("x", "y", "p", "t")
    for key in ("CD/events", "events"):
        if key not in f:
            continue
        node = f[key]
        if isinstance(node, h5py.Dataset):
            if node.dtype.names and all(k in node.dtype.names for k in fields):
                return "compound", node
        elif all(k in node for k in fields):
            return "plain", {k: node[k] for k in fields}
    if all(k in f for k in fields):
        return "plain", {k: f[k] for k in fields}
    raise _err(f"{path!r}: no 'CD/events' (or 'events') compound dataset and no plain "
               "x/y/p/t datasets — not a recognized HDF5 event file")


# ====================================================================================
# Reading
# ====================================================================================
class H5EventSource:
    """An open HDF5 event file, streamable in bounded blocks (a context manager).

    Locates the event data in any supported layout (see :func:`_locate`), exposes the
    stream's cheap facts — ``n``; attr-declared ``width``/``height`` (``None`` when the
    file does not say); the stringified root + ``gottlux`` attrs as ``meta``;
    ``time_shift`` — and yields raw ``(x, y, p, t)`` blocks in file order via
    :meth:`blocks`. The decode cache streams through this to build its ``.bin`` files;
    :func:`read_events` materializes it.
    """

    def __init__(self, path):
        h5py = _h5py()
        self.path = os.path.abspath(path)
        self._f = h5py.File(self.path, "r")
        try:
            self._kind, self._node = _locate(self._f, self.path)
            if self._kind == "compound":
                self.n = int(self._node.shape[0])
            else:
                lens = {k: int(d.shape[0]) for k, d in self._node.items()}
                if len(set(lens.values())) > 1:
                    raise _err(f"{self.path!r}: the x/y/p/t datasets disagree in "
                               f"length ({lens})")
                self.n = lens["t"]
            attrs = dict(self._f.attrs)
        except Exception:
            self._f.close()
            raise
        self.width, self.height = _geometry_from_attrs(attrs)
        ts = attrs.get("time_shift")
        try:
            self.time_shift = int(ts) if ts is not None else None
        except (TypeError, ValueError):
            self.time_shift = None
        self.meta = {k: str(v) for k, v in attrs.items()}
        if "gottlux" in self._f:                    # carry our provenance attrs along
            for k, v in self._f["gottlux"].attrs.items():
                self.meta[f"gottlux.{k}"] = str(v)

    def blocks(self, block=BLOCK_EVENTS):
        """Yield raw ``(x, y, p, t)`` array blocks of ≤ *block* events, in file order.

        A read failure inside a structurally-valid events dataset is diagnosed as the
        unregistered-codec (ECF) case — see :func:`_codec_error`."""
        for s in range(0, self.n, int(block)):
            e = min(s + int(block), self.n)
            try:
                if self._kind == "compound":
                    rows = self._node[s:e]
                    out = (rows["x"], rows["y"], rows["p"], rows["t"])
                else:
                    out = tuple(self._node[k][s:e] for k in ("x", "y", "p", "t"))
            except OSError as err:
                raise _codec_error(self.path, err) from err
            yield out

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def read_events(path, block=BLOCK_EVENTS) -> dict:
    """Read a whole HDF5 event file into the dict shape the decode cache produces.

    Accepts every layout :class:`H5EventSource` recognizes. Returns ``x, y`` (uint16,
    clipped to the declared geometry when there is one), ``p`` (uint8, 1 = ON), ``t``
    (int64 µs, time-sorted and re-zeroed to the first event) plus ``t0_us, width, height,
    n, n_on, fmt='hdf5', meta, source_path`` — and ``time_shift`` (the Metavision save
    attribute, ``None`` when absent) for provenance.

    Raises :class:`~gottlux.io.fusion.FusionError` if h5py is missing, the layout is
    unknown, or the events are stored with an unregistered codec (Prophesee's ECF —
    ``pip install gottlux[hdf5]`` for builds of ``hdf5plugin`` that ship it).
    """
    path = os.path.abspath(path)
    xs, ys, ps, ts = [], [], [], []
    with H5EventSource(path) as src:
        for x, y, p, t in src.blocks(block):
            xs.append(x); ys.append(y); ps.append(p); ts.append(t)
        w_attr, h_attr, meta, tshift = src.width, src.height, src.meta, src.time_shift
    x = np.concatenate(xs) if xs else np.zeros(0, np.uint16)
    y = np.concatenate(ys) if ys else np.zeros(0, np.uint16)
    p = np.concatenate(ps) if ps else np.zeros(0, np.uint8)
    t = (np.concatenate(ts) if ts else np.zeros(0)).astype(np.int64)
    if t.size and np.any(t[1:] < t[:-1]):           # rare: an unsorted plain dump
        o = np.argsort(t, kind="stable")
        x, y, p, t = x[o], y[o], p[o], t[o]
    t0 = int(t[0]) if t.size else 0
    t = t - t0
    p = (np.asarray(p) > 0).astype(np.uint8)        # some tools store polarity as ±1
    w = w_attr if w_attr is not None else (int(x.max()) + 1 if x.size else 320)
    h = h_attr if h_attr is not None else (int(y.max()) + 1 if y.size else 320)
    x = np.clip(x, 0, w - 1).astype(np.uint16)
    y = np.clip(y, 0, h - 1).astype(np.uint16)
    return dict(x=x, y=y, p=p, t=t, t0_us=t0, width=int(w), height=int(h),
                n=int(t.size), n_on=int((p == 1).sum()), fmt="hdf5", meta=meta,
                source_path=path, time_shift=tshift)


# ====================================================================================
# Writing
# ====================================================================================
def write_hdf5(source, out_path, t0=None, t1=None, roi=None, progress=None,
               block=BLOCK_EVENTS, gzip_level=4) -> int:
    """Write *source* (a Recording, or a path :func:`gottlux.load` accepts) as HDF5.

    The output is Metavision-compatible: one compound ``CD/events`` dataset (fields
    ``x, y, p, t``), chunked and gzip-compressed, **appended in bounded blocks** straight
    off the memmap-backed arrays — the full stream is never materialized, so a multi-GB
    recording converts inside the same ~1–2 GB envelope the decoder keeps. Root attributes
    carry the sensor geometry (``geometry``, ``width``, ``height``); a small ``gottlux``
    attrs group records the provenance (source file name, the absolute ``t0_us`` of the
    first written event, the gottlux version).

    ``t0``/``t1`` (seconds) and ``roi`` (``(x0, y0, x1, y1)`` pixels) export a sub-clip
    with :func:`~gottlux.io.writer.cut_clip` semantics — times re-zeroed to the first kept
    event. *progress*, if given, is called with a fraction in [0, 1] (a path source spends
    the first half on its decode). Returns the number of events written.
    """
    h5py = _h5py()
    if isinstance(source, (str, os.PathLike)):
        import gottlux as eb
        rec = eb.load(str(source),
                      progress=(lambda f: progress(0.5 * f)) if progress else None)
        wp = (lambda f: progress(0.5 + 0.5 * f)) if progress else None
    else:
        rec, wp = source, progress
    from gottlux import __version__

    if not is_hdf5_path(out_path):
        out_path += ".h5"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    i0 = 0 if t0 is None else rec.index_at(t0, "left")
    i1 = rec.n if t1 is None else rec.index_at(t1, "left")
    origin = np.int64(rec.t[i0]) if i1 > i0 else np.int64(0)
    total = max(i1 - i0, 1)

    with h5py.File(out_path, "w") as f:
        f.attrs["geometry"] = f"{rec.width}x{rec.height}"
        f.attrs["width"] = int(rec.width)
        f.attrs["height"] = int(rec.height)
        gl = f.create_group("gottlux")
        gl.attrs["source"] = os.path.basename(rec.source_path or "") or rec.name
        gl.attrs["t0_us"] = int(rec.t0_us) + int(origin)
        gl.attrs["version"] = __version__
        ds = f.create_group("CD").create_dataset(
            "events", shape=(0,), maxshape=(None,), dtype=EVENT_DTYPE,
            chunks=(int(min(H5_CHUNK_EVENTS, total)),),
            compression="gzip", compression_opts=int(gzip_level), shuffle=True)
        n_out = 0
        for s in range(i0, i1, int(block)):
            e = min(s + int(block), i1)
            xs = np.asarray(rec.x[s:e]); ys = np.asarray(rec.y[s:e])
            ps = np.asarray(rec.p[s:e]); tsb = np.asarray(rec.t[s:e]).astype(np.int64)
            if roi is not None:
                x0, y0, x1, y1 = roi
                m = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
                xs, ys, ps, tsb = xs[m], ys[m], ps[m], tsb[m]
            if xs.shape[0]:
                rows = np.empty(xs.shape[0], EVENT_DTYPE)
                rows["x"] = xs; rows["y"] = ys; rows["p"] = ps
                rows["t"] = np.clip(tsb - origin, 0, None)
                ds.resize((n_out + rows.shape[0],))
                ds[n_out:] = rows
                n_out += rows.shape[0]
            if wp:
                try:
                    wp(min((e - i0) / total, 1.0))
                except Exception:
                    pass
    if progress:
        try:
            progress(1.0)
        except Exception:
            pass
    return n_out
