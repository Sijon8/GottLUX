"""
provenance.py — make every result traceable to the exact code and inputs that produced it.

Each headless run gets its own timestamped folder containing:

* ``run_manifest.json`` — the full :class:`~gottlux.config.Config`, input file SHA-256(s),
  the environment (Python, platform, package versions), and a results summary.
* ``RUN_SUMMARY.txt``   — a human-readable digest.
* ``_source_snapshot/`` — a copy of the gottlux source that ran, so a figure can always be
  regenerated even after the code moves on.
* per-analysis subfolders (``overview/``, ``spectral/``, ``panorama/``, ``detect/``, …).

Output auto-relocates to a short platform path if the preferred location would overflow the
Windows ``MAX_PATH`` limit.
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from datetime import datetime, timezone

from gottlux import __version__
from gottlux.io import export
from gottlux.io.paths import ext, file_sha256, fits, platform_root, short_id


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _package_versions() -> dict:
    out = {}
    for pkg in ("numpy", "scipy", "matplotlib", "numba", "pandas", "pyarrow",
                "scikit-image", "PySide6", "pyqtgraph", "imageio", "h5py"):
        try:
            import importlib.metadata as md
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = None
    return out


class RunFolder:
    """A unique, self-describing output folder for one analysis run."""

    def __init__(self, cfg, rec, root=None):
        self.cfg = cfg
        self.rec = rec
        mode = "rotation" if rec.is_rotating else "staring"
        name = cfg.label or rec.name
        stamp = _utc_stamp()
        folder = f"gottlux_run_{mode}_{name}_{stamp}"
        if root is None:
            base = (os.path.dirname(rec.source_path) if rec.source_path
                    else os.getcwd())
            root = os.path.join(base, "gottlux_runs")
        preferred = os.path.join(root, folder)
        if not fits(os.path.join(preferred, "_source_snapshot", "gottlux", "x.py"),
                    headroom=24):
            preferred = os.path.join(platform_root(), "gottlux_runs",
                                     short_id(folder) + "_" + stamp)
        self.path = preferred
        os.makedirs(self.path, exist_ok=True)
        self.results = {}
        self.artifacts = []

    # ------------------------------------------------------------------ structure
    def subdir(self, name) -> str:
        p = os.path.join(self.path, name)
        os.makedirs(p, exist_ok=True)
        return p

    def record(self, key, value):
        """Record a result (anything JSON-serializable) under *key* for the manifest."""
        self.results[key] = value

    def add_artifacts(self, paths):
        self.artifacts.extend(paths)

    # ------------------------------------------------------------------ snapshot
    def snapshot_source(self):
        """Copy the gottlux ``.py`` source into ``_source_snapshot/`` (best-effort)."""
        import gottlux
        src_root = os.path.dirname(os.path.abspath(gottlux.__file__))
        dst_root = os.path.join(self.path, "_source_snapshot", "gottlux")
        for dirpath, _dirs, files in os.walk(src_root):
            rel = os.path.relpath(dirpath, src_root)
            if "__pycache__" in rel:
                continue
            for f in files:
                if not f.endswith(".py"):
                    continue
                dst_dir = os.path.join(dst_root, rel)
                try:
                    os.makedirs(ext(dst_dir), exist_ok=True)
                    shutil.copy2(ext(os.path.join(dirpath, f)), ext(os.path.join(dst_dir, f)))
                except Exception:
                    pass

    # ------------------------------------------------------------------ manifest
    def write_manifest(self):
        inputs = {}
        if self.rec.source_path and os.path.exists(ext(self.rec.source_path)):
            try:
                inputs[os.path.basename(self.rec.source_path)] = file_sha256(self.rec.source_path)
            except Exception:
                pass
        manifest = {
            "gottlux_version": __version__,
            "created_utc": _utc_stamp(),
            "config": self.cfg.to_dict(),
            "optics": {**self.cfg.optics(), "profile": self.cfg.active_profile().to_dict()},
            "recording": {
                "name": self.rec.name, "source": self.rec.source_path,
                "format": self.rec.fmt, "width": self.rec.width, "height": self.rec.height,
                "n_events": self.rec.n, "duration_s": round(self.rec.duration_s, 4),
                "rotating": self.rec.is_rotating,
            },
            "inputs_sha256": inputs,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "packages": _package_versions(),
            },
            "results": self.results,
            "artifacts": [os.path.relpath(a, self.path) for a in self.artifacts],
        }
        export.save_json(manifest, os.path.join(self.path, "run_manifest.json"))
        return manifest

    def write_summary(self, extra_lines=None):
        lines = [
            "gottlux run summary",
            "=" * 40,
            f"recording : {self.rec.name}",
            f"events    : {self.rec.n:,}  ({self.rec.duration_s:.3f} s, {self.rec.fmt})",
            f"mode      : {'ROTATION' if self.rec.is_rotating else 'STARING'}",
            f"folder    : {self.path}",
            "",
            "results:",
        ]
        for k, v in self.results.items():
            lines.append(f"  {k}: {v}")
        if extra_lines:
            lines += [""] + list(extra_lines)
        text = "\n".join(lines) + "\n"
        with open(os.path.join(self.path, "RUN_SUMMARY.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        return text
