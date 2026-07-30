# GottLUX

[![CI](https://github.com/Sijon8/GottLUX/actions/workflows/ci.yml/badge.svg)](https://github.com/Sijon8/GottLUX/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](INSTALL.md)

**An open-source event-based sensor toolkit that removes the barrier to working with
event cameras and makes them easy and useful: sub-second viewing of multi-gigabyte
recordings, frequency/flutter analysis and tracking, 3-D space-time visualization, a
clip-and-canvas video editor, and export to standalone MATLAB and Python scripts.**

An event camera doesn't take pictures; it reports per-pixel brightness *changes* with
microsecond timing. That makes it a remarkable instrument for anything that moves or
flickers — rotor blades, wingbeats, vibration — and a frustrating one to work with using
frame-based tools. Historically that friction (proprietary SDKs, opaque formats, one-off
scripts, no way to simply *watch* a recording) has been a moat around event-based sensing.
GottLUX removes it.

1. **Open the files, including the huge ones.** Prophesee `.raw` recordings open natively,
   as does **HDF5** — see [Working with `.raw` files](#working-with-raw-files) below.
   Multi-gigabyte captures open in seconds through sampled **preview mode** while the full
   decode finishes in the background, so no one waits minutes to find out whether a
   recording is worth keeping.
2. **See the data.** A sub-second **quick viewer** for triage (also the double-click
   handler), a live viewer with independent speed and accumulation controls, side-by-side
   **multi-clip** playback on one shared clock, and a per-pixel **event-rate tower**.
3. **Explore space-time.** The **3-D space-time explorer** plots the event cloud as
   `(x, y, t)` with time as depth — a fluttering target becomes a visibly *striped column*.
   A trailing corridor from 5 ms to the whole recording, a movable **interactive FFT box**,
   and a **two-point measure** read frequency, period, and apparent speed directly off the
   cloud.
4. **Edit and compose like a video editor.** The **Timeline** tab is a real editor for
   recordings: a preview viewport over a transport over track lanes, with clips as
   duration-proportional blocks — drag files in, trim, crop, reorder, add title slides and
   running titles. A **canvas block** arranges several recordings as a mosaic on one frame,
   each cell with its own colormap, tone-map, accumulation, and playback speed. Export the
   result as a **video** or as a single composited **`.raw`**.
5. **Track and detect.** A self-describing, composable detector framework over a deep
   temporal-frequency engine (spectra, non-uniform FFT, whitening, flicker maps) with a
   signature library for drones, insects, birds, and industrial flicker; a suite of
   **trackers** (nearest-neighbour, Kalman, centroid, frequency-locked) that tune live
   against playback; de-rotation, masking, and dual-camera fusion for **rotating payloads**;
   and EBS↔acoustic fusion.
6. **Build on it — open source, end to end.** MIT-licensed with no vendor SDK anywhere in
   the stack. `import gottlux` is a light NumPy-level library; custom detectors and
   analyses register through `GOTTLUX_PLUGINS` without forking; `--run-script` runs any
   `process(win, ctx)` function against the exact window in view; and `--export-tool` emits
   **standalone MATLAB and Python scripts** with full provenance, so an analysis can leave
   GottLUX entirely. See [`docs/EXTENDING.md`](docs/EXTENDING.md).

Every run is archived with its exact code, config, and input checksums; figures are
journal-grade (raster **and** vector). Runs on **Windows and Linux**.

## Working with `.raw` files

Prophesee `.raw` is the native format, read by a **vendor-free, vectorized decoder** built
into the package — **EVT2.1 / EVT2.0 / EVT3**, covering sensors such as the **GenX320**
(320×320) and **IMX636** (1280×720). No Metavision SDK, no licence server, no conversion
step before work can start.

- **Decode once.** The first open streams a file into a memmapped cache beside it
  (bounded memory, any file size); every later open is instant. `--cache-info` and
  `--clear-cache` report and reclaim that space, and never touch source data.
- **Edit without decoding.** A time-window cut of an EVT2.1 file operates directly on the
  byte stream, so trimming a multi-gigabyte capture to the interesting seconds is nearly
  instant.
- **Write it back.** Cuts, stitches, timeline sequences, and canvas mosaics all export to
  **byte-valid EVT2.1 `.raw`** that any other tool can read.
- **Convert both ways.** `gottlux file.raw --to-hdf5` streams to HDF5 in the Metavision
  `CD/events` layout; `.h5` and `.hdf5` files (that layout or plain `x/y/p/t` datasets)
  open everywhere `.raw` does.

---

## Install

```bash
git clone https://github.com/Sijon8/GottLUX.git
cd GottLUX
pip install -e ".[all]"
```

Full instructions (Linux system packages, install profiles, launchers, file associations,
troubleshooting): **[INSTALL.md](INSTALL.md)**.

## Try it in the browser

No install needed: **<https://sijon8.github.io/GottLUX/>** runs the GottLUX web app on
real sample clips — play and scrub hummingbird fights and quadcopter captures, drag a box
to get a region spectrum, view the flicker map, or drop in any local Prophesee `.raw` file.
Everything is decoded and processed entirely in the browser; nothing is uploaded. The app
is served straight from [`web/`](web/) in this repo via GitHub Pages (the URL goes live
after the first push to `main` once Settings → Pages → Source is set to "GitHub Actions").

## Quick start

Bundled sample recordings live in [`examples/data/`](examples/README.md) — real GenX320
captures of hummingbirds and a 5-inch quadcopter, staring and rotating:

```bash
# instant look — the lightweight viewer (also the double-click .raw handler)
gottlux-view examples/data/Humming_Bird_Fight_merged_shortest.raw

# the full 10-tab instrument
gottlux-gui examples/data/5inch_quadcopter.raw

# headless analysis -> a timestamped, self-documenting run folder
gottlux examples/data/5inch_quadcopter.raw --detector drone --freq_lo 90 --freq_hi 700
```

The quick viewer is the fast path: it opens in well under a second, plays the clip on a
loop, and its **Open in full GottLUX** button hands the already-decoded recording to the
full suite with no re-load. Register it as the `.raw` / `.h5` / `.hdf5` double-click
handler with `gottlux-view --register` (Windows registry or Linux XDG — same command).

## The interactive instrument

```bash
gottlux-gui                  # welcome dialog offers the bundled examples
gottlux-gui path/to/file.raw
```

The tabs — one shared recording, all seekable on one clock:

* **Live viewer** — scrub/play the stream as accumulated frames. Independent **speed**
  (down to glass-smooth slow motion) and **accumulation window**, switchable mode and
  colormap, a draggable ROI, and **dynamic-range control** — a static/dynamic white point
  and a map expression (log / √ / γ / asinh / equalize / percentile) so bright regions stop
  diluting faint targets.
* **Multi-clip** — several clips side-by-side on one shared clock with per-clip slate
  offsets, or superimposed in distinct colors (e.g. wide vs. narrow sensor).
* **Range lab** — keyframed target boxing → pixels-on-target → perception-range solve;
  dual-clip convergence for two co-observing sensors.
* **Event-rate tower** — the same instant as a 3-D relief: x/y on the ground, events/s as
  height; a buzzing region rises as a tower.
* **Space-time 3D** — the event cloud with time as depth (a fluttering target is a
  *striped column*). Temporal corridor from 5 ms to 5 s, a movable **interactive FFT box**
  (FFT / non-uniform FFT / ISI), and a **two-point measure** that reads frequency and speed
  straight off the cloud.
* **Flutter workbench** — the tuning lab: a **flicker map** (hue = frequency, opacity =
  SNR), a draggable region spectrum (peak / SNR / harmonic comb, optional whitening/NUFFT),
  the chosen detector's auto-built tuning panel, off-thread runs, **live tracking** while
  tuning, and a first-principles detection report export.
* **Sandbox** — a ground-up bench: select → compose a primitive op-chain → inspect raw
  `(x, y, p, t)` → analyze → export the manipulated sub-stream.
* **EBS viewer** — a second, deliberately-different player with 11 live view modes and
  band/stack column expressions.
* **Fusion lab** — co-register the recording with a time-synchronized audio `.wav`
  (auto cross-correlation alignment), compare the rotor tone both sensors see, and export
  the aligned pair.
* **Timeline** — a video editor for recordings: an embedded preview viewport over a
  transport over two track lanes, with clips drawn as duration-proportional blocks
  (thumbnail, name, span) that select on click, reorder by drag, and edit on double-click.
  Trim, crop, gap and per-clip EBS settings (colormap, tone-map, accumulation, mode, time
  scale) live on the deck; a **canvas block** drops a whole mosaic into a single slot and
  is arranged *inline* in the preview; a project-canvas preset (Native / 640×640 /
  1280×720 / 1920×1080 / 1024×1024 / Custom) plus fraction cell sizes, 1/12-grid snapping
  and auto-tile keep the geometry exact. Export the whole program as a video, or the
  events as one byte-valid `.raw`.

A **Capture…** toolbar action records any view over a time range to MP4 with a context
banner and a self-documenting poster/manifest; the program-wide **Export** writes every
artifact with an infographic context sheet.

A **system-wide light/dark theme** rides on a toolbar toggle (sun / moon): the whole
instrument — every tab, dialog, painted icon, plot canvas and 3-D background — switches
live, and the choice is remembered and shared with the quick viewer.

## Massive files

GottLUX never requires waiting on a multi-gigabyte decode to find out whether a recording
is even interesting:

- **Decode-once cache.** The first open of a file streams it (bounded memory) into a
  memmapped cache beside the data; every later open is instant.
- **Preview mode.** Files over a threshold (default 200 MB, `GOTTLUX_PREVIEW_THRESHOLD_MB`)
  open with **sampled sections from the beginning, middle, and end** of the recording,
  playable in seconds, with the timeline spanning the *whole* file. Seeking into an
  un-decoded region decodes just that slice on demand, and the full cache completes in the
  background, then swaps in seamlessly.
- **Cut without decoding.** A time-window cut of an EVT2.1 file operates directly on the
  bytes (`File → Quick-cut`), so trimming a huge capture down to the interesting seconds
  never pays the full decode.
- **Cache management.** `gottlux --cache-info` / `--clear-cache` report and reclaim cache
  space — cache files only, never source data.

## Compose

The **Canvas composer** (GUI: *Tools → Canvas composer…*) places several recordings —
possibly from different collects, sensors, and time bases — as independently positioned,
sized, and styled cells on one fixed canvas. Each cell carries its own visualization
settings (accumulation mode and window, tone-map expression, colormap, an optional source
ROI crop) **and its own clock mapping** (canvas-time offset, a time scale for
slow-motion/speed-up, looping), so a real-time wide view can play beside a 10×
slow-motion crop of the same moment. A composition is a small JSON spec
(`.gottlux-canvas.json`) — save it, reload it, re-render it — and exports two ways:
rendered frame-by-frame to **MP4**, or re-encoded as **events** into one composited
EVT2.1 **`.raw`** (geometry and clock mapping applied to the merged stream; the spec
rides along as a sidecar so the styled view stays reproducible).

The **Timeline** tab is the sequential side of the same engine, and shares its cell
machinery outright. It compiles into a *program*: every sequential item is a canvas — a
plain clip is one cell covering the whole canvas carrying that clip's own visualization
settings, a canvas block is its multi-cell composition verbatim, a title slide is a text
item — with running titles and overlay clips applied across every segment. One render
path (`core.canvas.render_frame`) drives the embedded preview and the video export alike,
so the scrubbed preview matches the export exactly. Mosaics are arranged **inline** using
the same draggable/resizable cells as the composer window, so no pop-out is needed;
`Export .raw…` stays events-only (a plain sequence stitches exactly as before — trim,
crop, gap, one monotonic clock — while a timeline holding mosaics composites into the
canvas geometry, and titles are noted rather than written).

## Headless analysis & editing

```bash
gottlux file.raw                                  # standard analyses -> run folder
gottlux file.raw --detector flutter --freq_lo 40 --freq_hi 120
gottlux file.raw --plots flicker_map,spectrum,tracks --detector drone
gottlux --list_detectors ; gottlux --list_plots ; gottlux --list_sensors

# results metrics (KPI bundle saved next to the file; regime auto-detected)
gottlux staring.raw  --performance --target_size 0.22
gottlux staring.raw  --performance --compare-with rotating.raw

# edit: cut & stitch .raw losslessly (bounded memory; EVT2.1 cuts need no decode)
gottlux clip.raw --cut 1.0,1.5 --roi 120,90,200,170
gottlux a.raw --stitch b.raw,c.raw --stitch_gap 0.1

# EBS + audio fusion
gottlux file.raw --fusion --audio capture.wav

# run a custom Python file on exactly the requested slice (see docs/EXTENDING.md)
gottlux file.raw --run-script my_script.py --t_start 1 --t_stop 2 --roi 120,90,200,170

# convert to/from HDF5 (Metavision-compatible layout; .h5 opens everywhere .raw does)
gottlux file.raw --to-hdf5
gottlux file.raw --to-hdf5 out.h5 --t_start 2 --t_stop 4 --roi 100,80,220,200
```

Each run writes a timestamped `gottlux_run_…/` folder: every figure as PNG + PDF, data as
Parquet + CSV (+ NPZ/HDF5), a `run_manifest.json` (full config, input SHA-256, environment,
resolved optics), a `RUN_SUMMARY.txt`, and a `_source_snapshot/` of the exact code that ran.

The three results metrics answer: *how far can the target be tracked*, *how far is its
rotor/wing frequency resolvable*, and *how much approach warning does a closing target
give* — each computed as both a sensor-model **capability** value and a **measured** value
from the detector's tracks, kept separate so model and data validate each other.

## Sensor & optics profiles

Pixel-to-bearing/range geometry comes from a **sensor/camera profile** (resolution, pixel
pitch, lens focal length, FOV…). Profiles live in `gottlux/sensors.py`; the default is a
Prophesee GenX320 behind a 1.8 mm S-mount lens.

```bash
gottlux --list_sensors
gottlux file.raw --sensor imx636          # a different rig
gottlux file.raw --fov_deg 40             # override just this clip's FOV
```

```python
from gottlux import sensors
tele = sensors.register(sensors.GENX320.with_lens(6.0, key="genx320_6mm"))
```

## Use it as a library

```python
import gottlux as eb

rec = eb.load("capture/cam0.raw")          # decode-once -> memmapped Recording
rec.summary()

frame = rec.accumulate(t0=1.0, dt=0.02, mode="count")
t, rate = rec.event_rate(0.01)

from gottlux.core import frequency as fq
spec = fq.region_spectrum(rec.window(2, 4, roi=(140, 150, 180, 175)).t, fmin=80, fmax=800)
print(spec.peak_freq, spec.snr, spec.harmonic_score)

from gottlux.detectors import get_detector
det = get_detector("drone", snr_thresh=5, harmonic_min=0.34)
result = det.run(rec, eb.Config(mode="staring", sensor="genx320"))
print(result.summary())
```

`import gottlux` is deliberately light (NumPy-level only); Qt and matplotlib load lazily.

## Extend it

Every major surface is a registry extensible **without forking**: subclass
`Detector`, declare its `PARAMS`, `@register` it, and it appears in `--list_detectors`,
runs via `--detector`, and gets an auto-built tuning panel in the Flutter workbench.
Point the `GOTTLUX_PLUGINS` environment variable at a `.py` file (or a folder of
them) and both the CLI and the GUI load it at startup — see the runnable
[`examples/custom_detector.py`](examples/custom_detector.py). For one-off custom
processing there are **user scripts**: `gottlux INPUT --run-script my_script.py` (or GUI
*Tools → Run user script on current view…*) hands the windowed/ROI'd events to a
`process(win, ctx)` function in an arbitrary `.py` file and saves whatever it returns
into a provenance-stamped run folder. The reverse direction is
`--export-tool`: export an analysis as a **standalone Python + MATLAB script pair**
(numpy/scipy/h5py only — no gottlux) bundled with the recording as `data.h5`, a README,
and a machine-readable `provenance.json` (source hash, window, file roles, the generating
command line). Six tools ship, including `viz_config`, which reproduces the current
visualization settings (mode / colormap / tone-map / accumulation) as rendered frames.

```bash
export GOTTLUX_PLUGINS=~/lab/my_detector.py     # set GOTTLUX_PLUGINS=... on Windows
gottlux clip.raw --detector blink               # the plugin's detector, like a built-in
gottlux clip.raw --run-script my_script.py      # custom code on the recording in view
gottlux clip.raw --export-tool region_spectrum  # a no-gottlux-needed tool bundle
gottlux --export-tool list
```

The full guide — custom detectors, custom headless analyses, custom GUI tabs, plugins,
user scripts, and the standalone-export path: **[docs/EXTENDING.md](docs/EXTENDING.md)**.

## Documentation

* [`INSTALL.md`](INSTALL.md) — Windows & Linux installation, launchers, troubleshooting.
* [`docs/GUI_GUIDE.md`](docs/GUI_GUIDE.md) — a walkthrough of the interactive dashboard.
* [`docs/DETECTOR_GUIDE.md`](docs/DETECTOR_GUIDE.md) — building and tuning flutter
  detectors, with worked settings for drones, insects, and birds.
* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the module map and design principles.
* [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) — the math: decoding, tone-mapping, the
  frequency engine (NUFFT, whitening, ISI), the flicker map, geometry, and detection.
* [`docs/ROTOR_LADDER.md`](docs/ROTOR_LADDER.md) — detecting a rotor by the stair-step it
  leaves in a *spinning* event sensor (`f = |v|/Δx`).
* [`docs/EXTENDING.md`](docs/EXTENDING.md) — building custom tools: detectors,
  analyses, GUI tabs, `GOTTLUX_PLUGINS`, user scripts (`--run-script`), and the
  `--export-tool` standalone-script path.
* [`docs/DESIGN_HISTORY.md`](docs/DESIGN_HISTORY.md) — what GottLUX is, why it matters, and
  the design decisions behind it.
* [`FUTURE_WORK.md`](FUTURE_WORK.md) — the open roadmap; contributions welcome.
* [`CHANGELOG.md`](CHANGELOG.md) — version history.

## Contributing & development

```bash
pip install -e ".[all,dev]"
pytest                      # synthetic-data suite; no private captures needed
ruff check . && ruff format .
```

CI runs the test suite on Python 3.10–3.13 on Linux and on Python 3.13 on Windows, plus lint. See
[CONTRIBUTING.md](CONTRIBUTING.md) for structure, standards, and the release process, and
[FUTURE_WORK.md](FUTURE_WORK.md) for ideas looking for an owner.

## Citation & license

MIT — see [LICENSE](LICENSE). If GottLUX is useful in published research, please cite it
([CITATION.cff](CITATION.cff)).
