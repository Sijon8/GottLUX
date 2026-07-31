"""
fetch_wheels.py — build the offline dependency bundle for GottLUX.

Run this on a machine that *has* internet access. It downloads every wheel the
suite needs into ``vendor/wheels/`` and writes ``BUNDLE_INFO.txt`` describing
exactly what the bundle targets. Copying the whole project folder (including
``vendor/``) to an air-gapped machine is then enough to install there with
``install_offline.bat`` / ``install_offline.sh``.

Wheels are specific to an operating system and a Python version. By default the
bundle targets the interpreter running this script; ``--python-version`` and
``--platform`` build a bundle for a different target, which is the right choice
when the offline machine differs from this one.

    python scripts/fetch_wheels.py                       # target: this machine
    python scripts/fetch_wheels.py --extras gui          # a slimmer bundle
    python scripts/fetch_wheels.py --python-version 3.11 --platform win_amd64
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "vendor", "wheels")

#: Dependency sets, mirroring the extras declared in pyproject.toml.
CORE = ["numpy>=2.0", "scipy>=1.11", "matplotlib>=3.8", "imageio>=2.34",
        "imageio-ffmpeg>=0.5", "pillow>=10.0"]
GUI = ["PySide6>=6.6", "pyqtgraph>=0.13", "PyOpenGL>=3.1"]
FAST = ["numba>=0.60"]
DATA = ["pandas>=2.0", "pyarrow>=14", "scikit-image>=0.22", "h5py>=3.10"]
DEV = ["pytest>=7.0", "pytest-cov", "pytest-timeout", "ruff"]
#: Build backend, so an editable install resolves its build requirements offline.
BUILD = ["pip", "setuptools", "wheel"]

SETS = {"core": CORE, "gui": GUI, "fast": FAST, "data": DATA, "dev": DEV}


def specs_for(extras) -> list:
    """The requirement list for the named extras (``core`` is always included)."""
    out = list(CORE)
    for name in extras:
        if name == "core":
            continue
        if name not in SETS:
            raise SystemExit(f"unknown extra {name!r}; choose from {sorted(SETS)}")
        out += SETS[name]
    return out + BUILD


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download the offline dependency bundle.")
    ap.add_argument("--extras", default="gui,fast,data",
                    help="comma list of extras to include (default: gui,fast,data — "
                         "everything the '.[all]' install provides)")
    ap.add_argument("--dest", default=DEST, help=f"output folder (default: {DEST})")
    ap.add_argument("--python-version", default=None,
                    help="target Python (e.g. 3.11); default is the running interpreter")
    ap.add_argument("--platform", dest="plat", default=None,
                    help="target platform tag (e.g. win_amd64, manylinux2014_x86_64)")
    args = ap.parse_args(argv)

    extras = [e.strip() for e in args.extras.split(",") if e.strip()]
    specs = specs_for(extras)
    os.makedirs(args.dest, exist_ok=True)

    cmd = [sys.executable, "-m", "pip", "download", "--dest", args.dest,
           "--only-binary=:all:"]
    if args.python_version:
        cmd += ["--python-version", args.python_version]
    if args.plat:
        cmd += ["--platform", args.plat]
    cmd += specs

    print("Downloading:", " ".join(specs), "\n")
    rc = subprocess.call(cmd)
    if rc != 0:
        print("\npip download failed. A target platform/version combination that has no "
              "matching wheel for some package is the usual cause.", file=sys.stderr)
        return rc

    files = sorted(f for f in os.listdir(args.dest) if f.endswith(".whl"))
    total = sum(os.path.getsize(os.path.join(args.dest, f)) for f in files)
    py = args.python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
    plat = args.plat or platform.platform()

    info = os.path.join(args.dest, "BUNDLE_INFO.txt")
    with open(info, "w", encoding="utf-8", newline="") as fh:
        fh.write("GottLUX offline dependency bundle\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"created (UTC)   : {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}\n")
        fh.write(f"target Python   : {py}\n")
        fh.write(f"target platform : {plat}\n")
        fh.write(f"extras included : core,{','.join(extras)}\n")
        fh.write(f"wheels          : {len(files)}\n")
        fh.write(f"total size      : {total / (1 << 20):.1f} MB\n\n")
        fh.write("Install on the offline machine, from the project root:\n")
        fh.write('    pip install --no-index --find-links vendor\\wheels -e ".[all]"\n')
        fh.write("or simply run install_offline.bat (Windows) / install_offline.sh (POSIX).\n\n")
        fh.write("A Python version on the target that differs from 'target Python' above\n")
        fh.write("will fail to resolve these wheels. Rebuild the bundle with\n")
        fh.write("    python scripts/fetch_wheels.py --python-version <X.Y>\n\n")
        fh.write("Contents\n--------\n")
        for f in files:
            size = os.path.getsize(os.path.join(args.dest, f)) / (1 << 20)
            fh.write(f"  {size:8.1f} MB  {f}\n")

    print(f"\n{len(files)} wheels, {total / (1 << 20):.1f} MB -> {args.dest}")
    print(f"wrote {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
