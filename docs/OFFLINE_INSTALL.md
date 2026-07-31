# Offline (air-gapped) installation

GottLUX installs without any internet connection when the project folder carries a
**wheel bundle** — every dependency, pre-downloaded, in `vendor/wheels/`. The offline
machine never contacts a package index: `pip` runs with `--no-index`, resolving
everything from that folder.

This is the supported path for lab machines, secure enclaves, and field systems that
cannot reach PyPI.

---

## What the offline machine still needs

Two things the bundle cannot supply:

1. **Python 3.10 or newer**, already installed. Wheels are Python-version specific — a
   bundle built for 3.13 will not install on 3.11. `vendor/wheels/BUNDLE_INFO.txt`
   records exactly which version the bundle targets.
2. **Linux only:** the Qt system libraries (`libegl1`, `libgl1`, `libxkbcommon0`,
   `libxcb-cursor0`), which come from the distribution's own package media, not pip. The
   headless core (CLI analysis, figures, video export) runs without them.

---

## Installing from a bundle

If the project folder already contains `vendor/wheels/`, installation is one step:

**Windows**

```powershell
install_offline.bat
```

**Linux / macOS**

```bash
sh install_offline.sh
```

Either script verifies the bundle, installs the full `.[all]` stack from it, confirms the
package imports, and (on Windows) offers to register GottLUX as the `.raw` / `.h5` /
`.hdf5` double-click handler.

The equivalent manual command, from the project root:

```powershell
pip install --no-index --find-links vendor\wheels -e ".[all]"
```

---

## Building a bundle on a networked machine

```bash
python scripts/fetch_wheels.py
```

This downloads the full stack into `vendor/wheels/` (roughly 435 MB for Windows /
Python 3.13) and writes `BUNDLE_INFO.txt` recording the target platform, Python version,
extras, and a per-wheel listing.

Then copy **the whole project folder, including `vendor/`,** to the offline machine —
USB drive, network share, or approved transfer — and run the installer there.

### Targeting a different machine

The bundle must match the offline machine's Python version and operating system, not the
one building it:

```bash
python scripts/fetch_wheels.py --python-version 3.11                      # different Python
python scripts/fetch_wheels.py --platform manylinux2014_x86_64            # Linux target
python scripts/fetch_wheels.py --python-version 3.11 --platform win_amd64 # both
```

### Smaller bundles

`--extras` selects which optional stacks are included (`core` is always present):

| command | contents | approximate size |
|---|---|---|
| `--extras gui,fast,data` *(default)* | everything `.[all]` provides | ~435 MB |
| `--extras gui` | core + the interactive instrument | ~250 MB |
| `--extras gui,fast,data,dev` | the above plus pytest and ruff | ~450 MB |
| `--extras ""` | headless core only — CLI, figures, video | ~150 MB |

---

## Why `vendor/` is not in version control

The bundle is several hundred megabytes of third-party binaries whose contents depend on
the target platform, so it is listed in `.gitignore` and never committed. A clone of this
repository does not carry one; `scripts/fetch_wheels.py` regenerates it on demand.

---

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `No matching distribution found for <package>` | The bundle targets a different Python version than the one running the install. Compare `python --version` against `vendor/wheels/BUNDLE_INFO.txt`, then rebuild with `--python-version`. |
| `Could not install packages due to an OSError … No such file or directory` with a very long path | Windows 260-character path limit, hit by Qt's deeply nested files. Install into a short path (for example `D:\GottLUX` rather than a deeply nested folder), or enable Long Path support: `Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1` (administrator, then reboot). |
| Install succeeds, `gottlux-gui` reports a Qt platform-plugin error (Linux) | The Qt system libraries listed above are missing; they must come from the distribution's package media. |
| `pip` tries to reach the network | `--no-index` was omitted. Use the installer scripts, which always pass it. |
| 3-D tabs blank | `PyOpenGL` is present in the default bundle; a `--extras` selection that omitted `gui` explains its absence. |

---

## Verifying an offline install

```powershell
gottlux --list_sensors                       # the CLI resolves
gottlux-view examples\data\Humming_Bird_Fight_merged_shortest.raw
```

A bundle built with `--extras …,dev` can additionally run the full suite on the target
machine:

```powershell
pytest
```
