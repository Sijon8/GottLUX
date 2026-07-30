# Future work

The open, prioritized backlog for GottLUX. Contributions welcome — each item links the
code it touches; anything here is fair game for a PR (see [CONTRIBUTING.md](CONTRIBUTING.md)).
Items marked 🧭 are good first issues.

## Data formats & I/O

- **Zero-setup ECF decode** — Metavision "compress on save" HDF5 uses Prophesee's ECF
  codec (HDF5 filter 36559). GottLUX reads these when a codec plugin is on the HDF5
  plugin path (Prophesee's open-source [`hdf5_ecf`](https://github.com/prophesee-ai/hdf5_ecf));
  `hdf5plugin` (the `gottlux[hdf5]` extra) covers the common generic codecs but not ECF.
  Packaging a prebuilt ECF codec (or a pure-Python ECF decoder) would make these files
  open with zero setup (`gottlux/io/hdf5.py`).
- **More event formats** — AEDAT4 (iniVation/DV), `.es` (event-stream), and a plain
  NPZ/Parquet event importer, all normalizing into the same `Recording` model
  (`gottlux/io/`). The decoder registry (`gottlux/io/decode.py`) is the extension point.
- **Live camera acquisition** — GottLUX is file-based today. An OpenEB/Prophesee HAL source
  that streams into the existing `Recording`/ring-buffer model would enable live tuning of
  detectors on hardware (`gottlux/io/`, new `source_live.py`; the detector live-tracking
  loop in the Flutter workbench is already off-thread).

## Performance

- **Content-hash cache invalidation** 🧭 — the decode cache invalidates on file mtime, which
  cloud-sync tools churn (spurious re-decodes) and same-mtime edits evade. Hash the first +
  last few MB instead (`gottlux/io/cache.py`).
- **Wider Numba coverage** — the EVT3 decode path and the flicker-map inner loops still have
  pure-NumPy hot spots; extend the JIT-with-fallback pattern in `gottlux/core/accel.py`.
- **GPU accumulation** — an optional CuPy/OpenGL-compute path for accumulate/tonemap at
  very high event rates (`gottlux/core/accumulate.py`).

## Viewer & GUI

- **Preview-mode polish** — shade undecoded spans on the transport bar while a massive file
  is still back-filling; per-span decode progress (`gottlux/app/transport.py`,
  `gottlux/io/preview.py`).
- **Timeline editor, phase 2** — the editor now trims, crops, reorders, stitches, and
  composites an overlay lane; still open: embedded preview/scrub in the clip timeline,
  razor/ripple edits, alignment-aware stitch, filtered/denoised → `.raw` render, undo/redo
  (`gottlux/app/timeline.py`).
- **macOS support** — expected to largely work (Qt + pure-Python stack) but untested; needs
  a CI job, `.command` launchers, and a duti-based file-association helper.
- **Packaged binaries** — PyInstaller (Windows) / AppImage (Linux) one-file builds so
  non-Python users can run the viewer.

## Composition & user scripts

- **Canvas audio track alignment** — the canvas composer has no audio lane. Aligning a
  time-synchronized `.wav` to the canvas clock (reusing the fusion lab's envelope
  cross-correlation) and muxing it into the MP4 export would make composited videos
  self-contained (`gottlux/core/canvas.py`, `gottlux/app/canvas.py`).
- **Per-cell labels & timecode burn-in** 🧭 — optional rendered cell titles (source name,
  time scale) and a canvas-clock timecode overlay in `render_frame`/`export_video`, so an
  exported composition explains itself without consulting the sidecar spec
  (`gottlux/core/canvas.py`).
- **User-script sandboxing** — user scripts (`--run-script`) execute with the full
  privileges of the Python interpreter, by design (they may import anything, gottlux
  included). There is **no sandboxing**: running a script from an untrusted source is
  running untrusted code. An opt-in restricted mode (subprocess isolation, resource
  limits, no-network) would make sharing scripts between groups safer
  (`gottlux/userscripts.py`).

## Web app

- **WebGL space-time view** — a GPU point-cloud rendering of the (x, y, t) event volume in
  the browser demo, matching the desktop 3-D view (`web/`).
- **EVT3 browser fixture** 🧭 — the web decoder implements all three encodings
  (EVT2.1/EVT2.0/EVT3) and EVT3 passed a Node parity check against the Python decoder,
  but no EVT3 fixture ships on the self-test page yet; add one (plus a real IMX636
  sample clip) so the path is exercised in CI-of-the-browser (`web/selftest/`).
- **h5wasm HDF5 loading** — read `.h5` event files client-side via
  [h5wasm](https://github.com/usnistgov/h5wasm), mirroring the desktop HDF5 path.
- **Range-request sampled preview** — sample large remote `.raw` files with HTTP Range
  requests so full-size captures preview instantly without a full download, mirroring the
  desktop massive-file preview.
- **Shareable permalink state** — encode clip, time cursor, ROI, and band settings in the
  URL fragment so an interesting finding can be shared as a link.

## Analysis

- **Real-data golden gates in CI** — byte-exact decoder cross-checks and tracker
  golden-file parity against the bundled sample clips, recorded as checksums so CI catches
  regressions the synthetic suite can't (`tests/`).
- **Prose run report** — a generated `manifest.md` companion narrating each run folder
  (what was analyzed, with what settings, what was found) (`gottlux/run/provenance.py`).
- **Detector zoo growth** — more built-in signatures (fixed-wing props, flapping-wing
  regimes, industrial flicker sources) with worked tuning examples
  (`gottlux/detectors/signatures.py`).
- **ML export bridges** — one-command export of labeled event windows to SNN/DVS training
  formats (tonic/expelliarmus-compatible tensors) (`gottlux/io/export.py`).

## Extensibility

- **More exported standalone tools** 🧭 — the `--export-tool` registry
  (`gottlux/export_tools/`) has six tools; good candidates for new Python+MATLAB
  template pairs: a spectrogram (frequency-vs-time), a Lomb–Scargle periodogram for
  sparse regions, and a de-rotated panorama accumulator. Adding one is a single module
  (templates + manifest + a `TOOLS` entry) — the exporter, `--export-tool list`, and the
  tests pick it up automatically.
- **pip-installable plugin entry points** — `GOTTLUX_PLUGINS` loads loose `.py` files;
  the natural next step is a `gottlux.plugins` [entry-point
  group](https://packaging.python.org/en/latest/specifications/entry-points/) so a
  detector pack installed with `pip install gottlux-birds` registers itself with no
  environment variable (`gottlux/plugins.py`).
- **Analyses registry hardening** — plugins currently extend the pipeline via the
  module-level `_ANALYSES` dict; promote it to a small public `register_analysis()` API
  with name-collision warnings (`gottlux/run/pipeline.py`).

## Distribution & community

- **PyPI release** — `pip install gottlux`; add a tag-triggered trusted-publishing workflow
  (`.github/workflows/`).
- **Zenodo data release** — a larger companion capture library (longer sessions,
  dual-sensor rigs, more target classes) published with a DOI and linked from the README;
  the in-repo `examples/data/` stays a small curated sample.
- **Visual tour** 🧭 — screenshots/GIFs of each tab in the README, and a 10-minute tutorial
  notebook walking a sample clip end-to-end.
- **Docs site** — the `docs/*.md` set is mkdocs-ready; publish to GitHub Pages.

---

*Recently completed (v1.0.0):* massive-file sampled preview with background decode ·
vector icon system (DPI-crisp on Windows/Linux) · first-class Linux support (XDG file
association, launchers, packaging) · decoder hot-loop vectorization · cache management
(`--clear-cache`, size reporting) · first-class HDF5 (`--to-hdf5`, `.h5` opens
everywhere; ECF-compressed vendor files decode when an ECF codec plugin is on the HDF5
plugin path — see the item above) ·
MATLAB bridge (`--export-tool` writes native-`h5read` MATLAB scripts beside their Python
twins, with a `viz_config` tool and per-bundle `provenance.json`) · plugin loading
(`GOTTLUX_PLUGINS`) and the [EXTENDING](docs/EXTENDING.md) guide · the canvas composer
(multi-collect composition; MP4 and composited-`.raw` export) · user scripts
(`--run-script`, `process(win, ctx)`) · the timeline editor's per-clip crop and overlay
lane.
