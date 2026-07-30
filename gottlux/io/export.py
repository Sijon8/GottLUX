"""
export.py — reproducible saving of data and journal-ready figures.

One place, one convention, for getting results *out* of gottlux in forms that survive
peer review and re-analysis:

* :func:`save_table`   — tabular results (detections, tracks, spectra, metric sweeps) to
  **Parquet** (fast, typed, columnar) *and* **CSV** (universal), from a dict-of-arrays or a
  pandas DataFrame. Parquet/pandas are optional — CSV is always written.
* :func:`save_arrays`  — raw NumPy arrays / images to compressed ``.npz``.
* :func:`save_hdf5`    — a structured group of arrays + attributes to a single ``.h5`` (the
  archival container for an event subset or a flicker-map cube). Optional (needs h5py).
* :func:`save_figure`  — a matplotlib figure at journal DPI in **both** a raster
  (``png``/``tiff``) and a vector (``pdf``) format, with metadata embedded.
* :func:`save_json`    — config / manifest / metric dicts, pretty-printed and NumPy-safe.

Every saver returns the list of paths it wrote, so a pipeline can record exactly what
landed on disk.
"""
from __future__ import annotations

import json
import os

import numpy as np


def _ensure_dir(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)


class _NumpyJSON(json.JSONEncoder):
    """JSON encoder that understands NumPy scalars/arrays and dataclasses."""
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, "__dataclass_fields__"):
            from dataclasses import asdict
            return asdict(o)
        return super().default(o)


def save_json(obj, path) -> list[str]:
    """Write *obj* as pretty, NumPy-safe JSON. Returns ``[path]``."""
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NumpyJSON)
    return [path]


def save_table(data, path_base, formats=("parquet", "csv")) -> list[str]:
    """Save a table (dict-of-arrays or DataFrame) to *path_base* + extensions.

    Parquet is written when pandas+pyarrow are available; CSV is always written so results
    are never locked behind an optional dependency.
    """
    written = []
    base = os.path.splitext(path_base)[0]
    _ensure_dir(base)
    df = None
    try:
        import pandas as pd
        df = data if hasattr(data, "to_parquet") else pd.DataFrame(
            {k: np.asarray(v).ravel() for k, v in data.items()})
    except Exception:
        df = None

    if df is not None and "parquet" in formats:
        try:
            df.to_parquet(base + ".parquet", index=False)
            written.append(base + ".parquet")
        except Exception:
            pass
    if "csv" in formats:
        if df is not None:
            df.to_csv(base + ".csv", index=False)
        else:                                       # pure-NumPy CSV fallback
            keys = list(data.keys())
            cols = [np.asarray(data[k]).ravel() for k in keys]
            n = max((len(c) for c in cols), default=0)
            with open(base + ".csv", "w", encoding="utf-8") as f:
                f.write(",".join(keys) + "\n")
                for i in range(n):
                    f.write(",".join(str(c[i]) if i < len(c) else "" for c in cols) + "\n")
        written.append(base + ".csv")
    return written


def save_arrays(path, **arrays) -> list[str]:
    """Save named NumPy arrays to a compressed ``.npz``. Returns ``[path]``."""
    if not path.endswith(".npz"):
        path += ".npz"
    _ensure_dir(path)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return [path]


def save_hdf5(path, datasets: dict, attrs: dict | None = None) -> list[str]:
    """Save a group of arrays (+ scalar attributes) to a single HDF5 file.

    Returns the path, or ``[]`` if h5py is unavailable (callers should treat HDF5 as a
    nice-to-have archival format and rely on Parquet/NPZ otherwise).
    """
    try:
        import h5py
    except Exception:
        return []
    if not path.endswith((".h5", ".hdf5")):
        path += ".h5"
    _ensure_dir(path)
    with h5py.File(path, "w") as f:
        for k, v in datasets.items():
            v = np.asarray(v)
            f.create_dataset(k, data=v, compression="gzip" if v.size > 1024 else None)
        for k, v in (attrs or {}).items():
            try:
                f.attrs[k] = v
            except Exception:
                f.attrs[k] = str(v)
    return [path]


def save_figure(fig, path_base, dpi: int = 300, formats=("png", "pdf"),
                metadata: dict | None = None, close: bool = False) -> list[str]:
    """Save a matplotlib *fig* at *dpi* in each of *formats* (raster + vector).

    A ``png``/``tiff`` raster gives a drop-in figure; the ``pdf`` (or ``svg``) is the
    scalable, print-quality master. Returns the paths written.
    """
    base = os.path.splitext(path_base)[0]
    _ensure_dir(base)
    written = []
    for fmt in formats:
        p = f"{base}.{fmt}"
        try:
            fig.savefig(p, dpi=dpi, bbox_inches="tight",
                        metadata=_fig_metadata(fmt, metadata))
            written.append(p)
        except Exception:
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            written.append(p)
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return written


def _fig_metadata(fmt, metadata):
    """Best-effort embedded metadata appropriate to the format (silently skipped if not)."""
    if not metadata:
        return None
    md = {k: str(v) for k, v in metadata.items()}
    if fmt in ("png",):
        return md
    if fmt in ("pdf",):
        return {"Title": md.get("title", "gottlux figure"),
                "Creator": "gottlux", "Subject": md.get("subject", "")}
    return None
