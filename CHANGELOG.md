# Changelog

All notable changes to GottLUX are documented here. The project follows
[Keep a Changelog](https://keepachangelog.com) and [Semantic Versioning](https://semver.org).

## [Unreleased]

## [1.0.0] — 2026-07-29

The first public release of **GottLUX** — an open-source suite for processing
event-based-sensor (neuromorphic camera) data: easy file processing, exporting,
visualization, and analysis, built to remove the friction around EBS data and workflows.

### Added

- **Drag-and-drop clips.** The Timeline tab/editor and the Canvas composer accept OS
  drops of `.raw` / `.h5` / `.hdf5` files and capture folders — each dropped recording
  appends through the normal add-clip path (multiple files land in name order; a
  per-file load failure is reported without aborting the rest), and a multi-file drop on
  the Timeline offers 'as sequence clips' or 'as one mosaic block'. An empty timeline
  shows a muted "Drop recordings here, or Add clips…" hint on its track lanes.
- **Titles (video export only).** The canvas engine gains `CanvasText` items — full-frame
  title *slides* and anchored *running* overlay lines (TrueType-rendered, spans on the
  canvas clock, JSON round-trip in the composition spec). 'Add title…' creates them in
  the Timeline editor (a slide occupies its duration on the sequential lane; a running
  title rides the overlay lane) and in the Canvas composer (listed after the cells,
  double-click to edit). Video export renders them through the engine; event exports
  (`.raw` stitch / composited `.raw`) carry events only and surface a one-line
  "text item(s) omitted" note, while the spec sidecar still records the texts.

- **Vendor-free decoding.** A pure-NumPy Prophesee `.raw` decoder (EVT2.1 / EVT2.0 / EVT3,
  vectorized hot loops) — no SDK required — plus a decode-once streaming memmap cache so
  every re-open of a file is instant, with `--cache-info` / `--clear-cache` management and
  automatic cross-process fallback when caches are held open elsewhere.
- **HDF5 in and out.** `.h5`/`.hdf5` event files (Metavision `CD/events` layout or plain
  `x/y/p/t` datasets) open everywhere `.raw` does, and `gottlux file.raw --to-hdf5`
  converts the other way — streaming, bounded memory, multi-GB safe. Compressed vendor
  files decode when a codec plugin is on the HDF5 plugin path (optional `gottlux[hdf5]`
  extra for the common codecs).
- **Massive-file preview.** Files over a threshold (default 200 MB,
  `GOTTLUX_PREVIEW_THRESHOLD_MB`) open in seconds with sampled sections from across the
  whole recording, a full-length timeline, on-demand slice decode while seeking, and a
  background full decode that swaps in seamlessly.
- **The interactive instrument.** A 10-tab PySide6 dashboard on one shared clock: live
  viewer (speed, accumulation, colormaps, dynamic-range control), multi-clip side-by-side
  playback, range lab (pixels-on-target range solving, dual-view convergence), event-rate
  tower, measurable space-time 3-D explorer with an interactive FFT box, flutter workbench
  (flicker map, region spectrum, auto-built detector tuning, live tracking), a low-level
  sandbox with an in-loop tracking lab, a classic alternate viewer, an EBS↔audio
  fusion lab, and a video-editor timeline (preview viewport, transport, track lanes,
  inline mosaics — see below). Painted vector icons render crisply at any DPI on Windows
  and Linux. A **system-wide light/dark theme** switches the whole instrument — tabs,
  dialogs, icons, plot canvases, 3-D backgrounds and timeline lanes — from one toolbar
  toggle, live, persisted across launches and shared with the quick viewer.
- **The quick viewer.** `gottlux-view` opens any recording in well under a second, plays
  it on a loop, and hands off to the full suite without re-decoding. `--register` makes it
  the double-click handler for `.raw` / `.h5` / `.hdf5` (Windows per-user registry or
  Linux XDG — same command), with each extension's previous default backed up and
  restored on `--unregister`.
- **Detection & analysis.** A deep temporal-frequency engine (binned FFT, non-uniform
  FFT, Lomb–Scargle, inter-event-interval, whitening, flicker maps), a self-describing
  composable detector framework with a signature library (drones, insects, birds,
  industrial flicker), rotating-payload processing (de-rotation, panorama, background
  masking, dual-camera fusion & calibration, regime-split trackers, radar/MTI figure and
  video suite), acoustic fusion, results metrics, and lossless `.raw` editing
  (cut/stitch; EVT2.1 time-cuts need no decode).
- **Reproducible output.** Every headless run writes a timestamped folder with figures
  (PNG + PDF), data (Parquet/CSV/NPZ/HDF5), a manifest (full config, input SHA-256,
  environment, resolved optics), and a snapshot of the exact code that ran.
- **Export to MATLAB and Python.** `gottlux INPUT --export-tool NAME` writes a standalone
  bundle — `data.h5` plus a runnable Python script (numpy/scipy/h5py only, no GottLUX
  install) and its MATLAB twin (base `h5read`, toolbox-free) — with the current
  window/ROI/band parameters baked in as editable variables. Six tools ship: event
  frames, event rate, region spectrum, flicker map, centroid tracker, viz config.
- **Provenance-stamped tool bundles.** Every `--export-tool` bundle carries a
  full-provenance README (source path, size, SHA-256, geometry, window, the role of every
  file, the generating command line) plus a machine-readable `provenance.json`, and the
  `viz_config` tool exports the current visualization settings (mode / colormap /
  tone-map / accumulation) as a standalone frame renderer — from the CLI via
  `--viz_mode`/`--viz_cmap`/`--viz_tonemap`/`--viz_accum_ms`, or from the GUI Tools menu
  with the live viewer's settings baked in.
- **Canvas composer.** A composition window (GUI: *Tools → Canvas composer…*) placing
  several recordings — possibly different collects, sensors, and time bases — as
  positioned, sized, styled cells on one canvas, each cell with its own visualization
  settings (accumulation mode + window, tone-map, colormap, source ROI crop) and clock
  mapping (offset, time scale, loop). Compositions save/load as a JSON spec
  (`.gottlux-canvas.json`) and export to MP4 or to one composited EVT2.1 `.raw` with the
  spec as a sidecar. The cell stage is a reusable widget (`CanvasArrangeView`), shared
  with the Timeline tab so a mosaic can be arranged without opening this window.
- **Timeline — a video editor for recordings.** The **Timeline** tab (also reachable as a
  dialog from the toolbar and the Live viewer) is laid out like a simple video editor: an
  **embedded preview viewport** rendering the whole timeline at the playhead, a
  **transport** on the timeline's own clock (spacebar plays/pauses when the tab has
  focus), and **track lanes** — a sequence lane and an overlay lane drawn as
  duration-proportional blocks with cached midpoint-frame thumbnails, click to select,
  drag to reorder, double-click to edit, ruler to seek. It compiles into a *program* where
  every sequential item is a `CanvasSpec` segment (a clip is one full-canvas cell carrying
  its own visualization settings, a mosaic passes through verbatim, a title slide is a text
  item, running titles and overlay clips span every segment), and **one** render path
  (`core.canvas.render_frame`) drives both the preview and the video export.
  **Add canvas block…** inserts a whole mosaic as a single sequence item, arranged
  **inline** in the preview on the composer's shared cell stage — no pop-out window
  required; dropping several files at once offers 'as sequence clips' or 'as one mosaic
  block'. A **project canvas** preset (Native / 640×640 / 1280×720 / 1920×1080 / 1024×1024
  / Custom) sets the preview and export geometry, cells take exact fractions of it, drags
  snap to a 1/12 grid, and Auto-tile lays N cells into the best-fit grid — all of that
  geometry math pure and tested in `core.canvas`. **Export video…** renders the whole
  program; **Export .raw…** stays events-only — a plain sequence stitches byte-for-byte as
  before (trim + crop + gap, one monotonic clock), a timeline holding canvas blocks
  composites the events into the canvas geometry with a spec sidecar, and titles plus
  render-only settings are noted rather than written.
- **User scripts.** `gottlux INPUT --run-script FILE.py` (GUI: *Tools → Run user script
  on current view…*) hands the windowed/ROI'd events to a `process(win, ctx)` function in
  an arbitrary `.py` file and saves what it returns (dict of arrays → NPZ; matplotlib
  figure → PNG + PDF; `{"events": (x, y, p, t)}` → a derived `.raw`) into a stamped run
  folder whose README records the script + input SHA-256, window/ROI, version, and wall
  time.
- **Easy Python development & algorithmic interface.** `import gottlux` is a light,
  NumPy-level library (`load` → memmapped `Recording` → windows, accumulation, spectra,
  detectors); custom detectors/analyses register via `GOTTLUX_PLUGINS` without forking,
  and [`docs/EXTENDING.md`](docs/EXTENDING.md) is the complete extension guide with a
  runnable example.
- **GottLUX Web.** A fully-static in-browser companion (`web/`, deployed to GitHub
  Pages): a JavaScript port of the decoder (checksum-parity with the Python decoder on a
  public self-test page), canvas playback with logarithmic 0.005×–2× speed, an event-rate
  timeline, drag-a-box region spectrum, flicker-map overlay, and drag-drop for local
  files — processed entirely client-side. Five web-sized sample clips included.
- **Cross-platform.** Windows 10/11 and Linux: pip install with graceful extras,
  `.bat`/`.sh` no-install launchers, XDG desktop/MIME packaging, CI on Python 3.10–3.13
  (Linux) and Windows.
- **Sample data.** Four real GenX320 captures in `examples/data/` (hummingbirds and a
  5-inch quadcopter — staring and rotating) that power the welcome dialog, the docs
  quick-starts, and the test suite's integration checks.
