# Installing GottLUX

GottLUX runs on **Windows 10/11** and **Linux** (x86-64; exercised on Ubuntu in CI).
macOS is expected to work for the headless core but is not yet routinely tested.

It needs **Python ≥ 3.10** (3.10–3.13 are exercised in CI). Everything else is a `pip`
dependency — there is no vendor SDK to install: GottLUX ships its own pure-NumPy Prophesee
`.raw` decoder (EVT2.1 / EVT2.0 / EVT3), and video export uses the ffmpeg binary bundled by
`imageio-ffmpeg`.

> The browser demo at <https://sijon8.github.io/GottLUX/> plays real sample clips
> with zero setup — a way to preview the suite before installing anything.

> **No internet on the target machine?** GottLUX installs fully offline from a
> pre-downloaded wheel bundle — see [`docs/OFFLINE_INSTALL.md`](docs/OFFLINE_INSTALL.md).

---

## 1. Quick install (both platforms)

From a clone of this repository:

```bash
git clone https://github.com/Sijon8/GottLUX.git
cd GottLUX
pip install -e ".[all]"
```

`.[all]` installs the full experience: the PySide6 GUI, the 3-D views (PyOpenGL), Numba JIT
acceleration, and the fast data-export stack (pandas/pyarrow/scikit-image/h5py). See §4 for
slimmer installs.

That registers four commands on the PATH:

| command | what it opens |
|---|---|
| `gottlux-gui` | the full 10-tab interactive instrument |
| `gottlux-view file.raw` | the **lightweight quick viewer** (fast single-clip player) |
| `gottlux file.raw` | headless analysis → a reproducible run folder |
| `gottlux-calibrate` | range / dual-camera timing calibration utilities |

Smoke-test the install with the bundled sample data:

```bash
gottlux-view examples/data/Humming_Bird_Fight_merged_shortest.raw
```

> **Virtual environments** are recommended but optional:
> `python -m venv .venv` then `.venv\Scripts\Activate.ps1` (Windows) or
> `source .venv/bin/activate` (Linux) before the `pip install`.

---

## 2. Windows specifics

- **Python**: install from [python.org](https://www.python.org/downloads/) (3.10+), ticking
  *Add python.exe to PATH*.
- **No-install launchers**: from a source checkout, the `.bat` launchers in the repo root
  (`gottlux_gui.bat`, `gottlux.bat`, `gottlux_view.bat`, `gottlux-calibrate.bat`) run
  without any `pip install` — each sets `PYTHONPATH` and runs the module directly.
  Dependencies must still be installed (`pip install -r requirements.txt`).
- **Double-click `.raw` / `.h5` / `.hdf5` files**: register the quick viewer as the
  per-user handler for all three:

  ```powershell
  gottlux-view --register
  ```

  Windows protects the current default app, so the first open may still require
  *right-click → Open with → GottLUX quick viewer → Always*. `gottlux-view --unregister`
  removes the associations and restores each extension's previous default.

## 3. Linux specifics

- **System libraries for Qt** (GUI only — the headless core needs none). Debian/Ubuntu:

  ```bash
  sudo apt-get install -y libegl1 libgl1 libxkbcommon0 libxcb-cursor0
  ```

  (These are the same packages CI installs; most desktop installs already have them.)
- **Shell launchers**: source-checkout equivalents of the Windows `.bat` files ship in the
  repo root (`gottlux_gui.sh`, `gottlux.sh`, `gottlux_view.sh`, `gottlux-calibrate.sh`).
- **Double-click `.raw` / `.h5` / `.hdf5` files**: the same command works via XDG
  desktop/MIME registration:

  ```bash
  gottlux-view --register
  ```

  This installs a per-user MIME type + `.desktop` entry (see `packaging/linux/` for the
  files, and its README for manual installation).
- **Wayland/X11**: both work; if the session mis-picks, force one with
  `QT_QPA_PLATFORM=xcb` (or `wayland`).

---

## 4. Install profiles (extras)

| profile | command | installs |
|---|---|---|
| Everything (recommended) | `pip install -e ".[all]"` | GUI + 3-D views + JIT + fast export |
| Headless / server | `pip install -e .` | full analysis pipeline, figures, video — no Qt |
| GUI only | `pip install -e ".[gui]"` | the interactive instrument (PySide6, pyqtgraph, PyOpenGL) |
| Speed | add `fast` | Numba JIT for the hot loops (pure-NumPy fallback otherwise) |
| Data export | add `data` | Parquet/HDF5/scikit-image export paths |
| Extra HDF5 codecs | add `hdf5` | `hdf5plugin`'s generic codec set for compressed vendor `.h5` files (Prophesee's ECF codec specifically needs their [`hdf5_ecf`](https://github.com/prophesee-ai/hdf5_ecf) plugin) |
| Development | `pip install -e ".[all,dev]"` | + pytest, coverage, ruff |

Everything outside the core degrades gracefully: a minimal install still decodes, analyzes,
and writes figures headlessly.

---

## 5. First run

```bash
# open the GUI and pick a bundled example from the welcome dialog
gottlux-gui

# or go straight at a sample clip
gottlux-gui examples/data/5inch_quadcopter.raw

# quick look at any file (fast viewer; "Open in full GottLUX" hands off to the suite)
gottlux-view examples/data/Humming_Bird_Fight_merged_2.raw

# headless: standard analyses -> a timestamped, self-documenting run folder
gottlux examples/data/5inch_quadcopter.raw
```

**First open of a file** builds a decode-once cache (`_gottlux_cache/` beside the data) so
every later open is instant. **Files larger than ~200 MB open in preview mode**: sampled
sections from across the recording appear in seconds while the full decode completes in the
background (see the README's *Massive files* section). Set `GOTTLUX_PREVIEW_THRESHOLD_MB`
to tune or disable (`0`) this behavior.

---

## 6. Troubleshooting

| symptom | fix |
|---|---|
| `gottlux: command not found` | the Python *Scripts*/*bin* dir isn't on PATH — re-open the terminal, or run `python -m gottlux` |
| GUI won't start on a server / over SSH | `export QT_QPA_PLATFORM=offscreen` for headless rendering, or install the Qt system libs (§3) |
| 3-D tabs (Space-time, Tower) blank or erroring | `pip install PyOpenGL` — included in `.[all]`/`.[gui]`, but not in a bare install |
| Video export unavailable | `pip install imageio-ffmpeg` (bundled ffmpeg); no system ffmpeg needed |
| Slow first open of a big file | expected once per file (cache build); preview mode shows data immediately, and re-opens are instant |
| First-ever GUI launch is slow | expected once per install (~a minute building JIT/font caches); subsequent launches take a few seconds |
| Windows: double-click opens the wrong app | `gottlux-view --register`, then right-click → *Open with* → *Always* once |

If a problem persists, [open an issue](https://github.com/Sijon8/GottLUX/issues)
including the OS, Python version, and the console output.
