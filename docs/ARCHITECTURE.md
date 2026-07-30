# gottlux — Architecture

gottlux is built in clean layers, each importing only the layers beneath it. The guiding
rule is **one data model, light imports, vectorized compute, reproducible output**.

```
gottlux/
  __init__.py          public API: load(), Recording, Config, Telemetry, __version__
  __main__.py          `python -m gottlux` -> cli.main (single dispatch point)
  cli.py               argument parsing; no-path -> GUI, a path -> headless pipeline
  config.py            Config dataclass — every run parameter, documented + serializable
  synthetic.py         labeled synthetic scenes with known flutter targets (tests + demos)

  io/                  ── data in and out ───────────────────────────────────────────────
    paths.py           Windows long-path (\\?\) safety, platform fallbacks, SHA-256
    decode.py          pure-NumPy EVT2.1 / EVT2.0 / EVT3 decoders (stateless, chunkable)
    cache.py           decode-once, streaming, memmapped cache (bounded RAM for GB files)
    preview.py         sampled fast preview for massive files (sections now, full decode behind)
    recording.py       Recording + EventWindow — THE data model everything operates on
    telemetry.py       rotation ground truth (Hall-sync boundaries + azimuth)
    writer.py          EVT2.1 encoder + clip cutter/stitcher (streamed, bounded-RAM; inverse of decode)
    rawcut.py          decode-free EVT2.1 time-window cut — index TIME_HIGH, byte-copy + re-zero
    export.py          reproducible saving: Parquet/CSV/NPZ/HDF5 + journal figures + JSON
    fusion.py          EBS↔audio fusion: .wav + Prophesee HDF5 I/O, envelope x-corr alignment, aligned export

  core/                ── the compute engine (NumPy / SciPy / optional Numba) ────────────
    accel.py           Numba JIT wrapper with a transparent pure-NumPy fallback
    accumulate.py      events -> 2-D frames (count / polarity / time-surface / ON / OFF)
    tonemap.py         dynamic-range compression: white-point + log/√/γ/asinh/equalize curves
    render.py          THE canonical frame pipeline (window→filter→accumulate→tonemap)
    frequency.py       *** spectra, Lomb–Scargle, spectrograms, the flicker map ***
    filters.py         denoise: hot-pixel, refractory, rotation-phase anomaly
    background.py      static-clutter suppression (rotation frozen / staring persistent)
    geometry.py        pinhole projection: pixel -> bearing / elevation / world-azimuth / range
    detect.py          spatial blob isolation -> Detection; single-frame clusterer
    metrics.py         coverage + localization figures of merit
    performance.py     the operator KPIs: tracking range, prop-frequency range, TTC

  rotation/            ── the spinning-sensor stack ─────────────────────────────────────
    masking.py …       de-rotation, frozen phase-space background masking, panorama, radar
    rotor_ladder.py    the rotor-ladder primitive (f = |v|/Δx) — see ROTOR_LADDER.md
    rotor_scan.py      the 360° rotor-ladder survey, tracks, and report

  detectors/           ── the tunable flutter/flicker detection framework ───────────────
    base.py            Detector / Param / Target / DetectorResult + registry
    signatures.py      drone / insect / mosquito / hummingbird / bird / custom presets
    tracking.py        greedy NN multi-target association with coasting
    flutter.py         FlutterDetector (the composable workhorse) + registered presets

  viz/                 ── journal-ready static figures (matplotlib, lazy import) ─────────
    theme.py           publication style + custom colormaps + the flicker palette
    frames.py          event frames, event-rate, detection overlays
    spectral.py        flicker map (showpiece), spectra, spectrograms
    tracks.py          per-target time series, confidence ranking
    panorama.py        de-rotated 360° panorama, polar radar

  app/                 ── the interactive instrument (PySide6 + pyqtgraph, lazy) ─────────
    style.py           dark instrument theme
    loader.py          background threads: RecordingLoader, DetectorWorker (UI never freezes)
    widgets.py         ParamPanel — auto-builds a tuning UI from a detector's Param list
    viewer.py          LiveViewer — scrub/play event frames (smooth slow-motion)
    spacetime.py       SpaceTimeView — the 3-D (x, y, t) event cloud
    workbench.py       FlutterWorkbench — flicker map + region FFT + tunable detector
    fusionlab.py       FusionLab — align an EBS recording to a .wav (event-rate ↔ RMS x-corr)
    main.py            MainWindow + main() entry

  run/                 ── headless orchestration + provenance ───────────────────────────
    provenance.py      RunFolder: unique folder + manifest + source snapshot + summary
    pipeline.py        run_path / run_recording: overview / spectral / panorama / detect / metrics
    performance_report.py  the KPI bundle writer (guarded saves — one bad artifact never loses the rest)
    track_study.py     single-clip tracker + KPIs + figures/videos (`--track-study`)
    fusion_detect.py   per-domain rotor detection (harmonic acoustic + in-box EBS), convergence, fusion
    fusion_study.py    align a .raw to a .wav → detect + fuse + figures/report (the fusion tier)
```

(The tree is abridged to the load-bearing modules; the GUI package `app/` in particular has one
module per tab plus shared transport/capture/export machinery.)

## The one data model

Everything operates on a single [`Recording`](../gottlux/io/recording.py): memmap-backed
event arrays (`x, y` uint16, `p` uint8, `t` int64 µs) plus geometry, source metadata, and
(when present) rotation `Telemetry`. Windows into the stream are returned as light
`EventWindow` views (NumPy slices — no copy of the whole stream), so there is exactly one
vocabulary for "a chunk of EBS data". Build one with `eb.load(...)`, or
`Recording.from_events(...)` for synthetic data and tests.

## Why the layering

* **Light import.** `import gottlux` pulls in only NumPy-level code. Qt (`app/`), plotting
  (`viz/`), and even Numba are imported lazily, only when used — so scripts, the headless
  pipeline, and background threads stay cheap and never need a display.
* **Vectorized + JIT.** Hot loops (time-surface scatter, refractory, per-cell FFT) are
  NumPy-vectorized and, where it matters, Numba-JIT compiled — with a pure-NumPy fallback so
  the suite always runs.
* **Bounded memory.** The decoder streams chunk-by-chunk straight to an on-disk memmap cache,
  carrying EVT3 state across chunk boundaries, so a multi-GB file decodes in ~1–2 GB of RAM
  and a re-open is an instant memmap.
* **Reproducible.** Every headless run snapshots its own config, input hashes, environment,
  and source code, so any figure can be regenerated later.

## Extending it

* **A new detector**: subclass `Detector`, declare `PARAMS` (each a self-describing `Param`)
  and a `regime`, implement `run()`, and `@register` it. It appears automatically in
  `--list_detectors`, the CLI `--detector`, and the GUI workbench (with a full tuning panel
  built from its `PARAMS`). See `gottlux/detectors/flutter.py` and `signatures.py`.
* **A new figure**: add a builder to `gottlux/viz/` that returns a styled Figure, and save it
  with `gottlux.io.export.save_figure`.
* **A new analysis**: add a function to `gottlux/run/pipeline.py` and register it in
  `_ANALYSES` (or wire it into the detect/metrics flow).
