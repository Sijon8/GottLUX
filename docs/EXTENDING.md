# Extending GottLUX — building custom tools

GottLUX is designed to be *extended*, not just run: every major surface — detectors,
headless analyses, GUI tabs — is an extensible registry, and there is a supported path
for taking an algorithm **out** of GottLUX as a standalone script. This guide covers all
six extension points, grounded in the real APIs:

1. [Custom detectors](#1-custom-detectors) — subclass `Detector`, declare `PARAMS`, `@register`.
2. [Custom analyses](#2-custom-analyses) — a function in the pipeline's analyses registry.
3. [A custom GUI tab](#3-a-custom-gui-tab) — the minimal widget contract.
4. [Plugins without forking](#4-plugins-without-forking-gottlux_plugins) — `GOTTLUX_PLUGINS`.
5. [User scripts](#5-user-scripts---run-script) — `process(win, ctx)` via `--run-script`.
6. [Standalone tool export](#6-standalone-tool-export---export-tool) — `--export-tool`.

Related reading: [`DETECTOR_GUIDE.md`](DETECTOR_GUIDE.md) (tuning the built-in detectors),
[`ARCHITECTURE.md`](ARCHITECTURE.md) (the module map), [`ALGORITHMS.md`](ALGORITHMS.md)
(the math the built-ins implement).

---

## 1. Custom detectors

A detector answers *where, when, and at what frequency is something fluttering?* The
framework lives in [`gottlux/detectors/base.py`](../gottlux/detectors/base.py) and has
four pieces a custom detector touches:

* **`Param`** — a self-describing tunable knob:

  ```python
  @dataclass
  class Param:
      key: str
      label: str
      default: float
      lo: float = 0.0
      hi: float = 1.0
      step: float = 0.0            # 0 → a sensible (range/100) default in the GUI
      kind: str = "float"          # 'float' | 'int' | 'bool' | 'choice'
      choices: tuple = ()
      unit: str = ""
      help: str = ""
      group: str = "General"
  ```

* **`Detector`** — the base class. A subclass sets the identity attributes
  (`name`, `description`, `regime` — `'staring' | 'rotation' | 'both'` — and `use_for`),
  declares a `PARAMS` list, and implements the one required method:

  ```python
  def run(self, rec, cfg=None, t0=None, t1=None, progress=None) -> DetectorResult
  ```

  `rec` is a `Recording` (slice it cheaply with `rec.window(t0, t1, roi=...)`), `cfg` an
  optional `gottlux.Config`, `t0`/`t1` bound the analysis window in seconds, and
  `progress`, if given, accepts a fraction in `[0, 1]`. The coerced parameter values are
  in `self.params` (the base `__init__` fills them from `PARAMS` and clamps overrides
  with `Param.coerce`).

* **`Target` / `DetectorResult`** — what `run` returns: a `DetectorResult(targets,
  detector, params, regime, signature, diagnostics)` whose `targets` are `Target`
  objects carrying parallel per-detection arrays (`t, cx, cy, bbox, freq_hz, snr,
  harmonic`, optional geometry columns). `Target.confidence` and
  `DetectorResult.summary()` come free.

* **`@register`** — a class decorator that adds the subclass to the global registry
  under its `name`.

Registration is the whole integration. Once registered, the detector:

* is constructible via `get_detector("yourname", **param_overrides)`;
* appears in `gottlux --list_detectors` and runs headless via
  `gottlux clip.raw --detector yourname`;
* shows up in the **Flutter workbench**'s detector picker, which **auto-builds its
  tuning panel from the declared `PARAMS`** (`workbench._rebuild_params` does
  `ParamPanel(det.PARAMS)`) — sliders, ranges, units, tooltips and grouping all come
  from the `Param` declarations, with live re-runs as a slider drags.

### Worked example

The complete, runnable version is
[`examples/custom_detector.py`](../examples/custom_detector.py) (a naive global
blink-rate detector, heavily commented, usable both as a plugin and as a script).
The skeleton:

```python
import numpy as np
from gottlux.detectors.base import Detector, DetectorResult, Param, Target, register

@register
class BlinkRateDetector(Detector):
    name = "blink"
    description = "Global blink/flicker-rate detector (whole-frame FFT)."
    regime = "both"
    use_for = "scenes dominated by one blinking source."
    PARAMS = [
        Param("freq_lo", "Band low", 5.0, 1.0, 500.0, 1.0, "float", unit="Hz",
              group="Band", help="Lower edge of the blink band."),
        Param("freq_hi", "Band high", 400.0, 2.0, 2000.0, 1.0, "float", unit="Hz",
              group="Band", help="Upper edge of the blink band."),
        Param("snr_thresh", "SNR gate", 3.0, 1.0, 30.0, 0.5, "float", group="Band",
              help="Min spectral peak / noise floor to accept a step."),
        Param("step_s", "Step", 0.25, 0.05, 2.0, 0.05, "float", unit="s", group="Timing",
              help="Analysis window length per detection attempt."),
    ]

    def run(self, rec, cfg=None, t0=None, t1=None, progress=None):
        from gottlux.core.frequency import region_spectrum
        P = self.params
        win = rec.window(t0, t1)
        rows = []
        for s in np.arange(float(win.t_s[0]), float(win.t_s[-1]), P["step_s"]):
            lo, hi = np.searchsorted(win.t_s, [s, s + P["step_s"]])
            sp = region_spectrum(win.t[lo:hi], fs=2.5 * P["freq_hi"],
                                 fmin=P["freq_lo"], fmax=P["freq_hi"])
            if sp.detected and sp.snr >= P["snr_thresh"]:
                rows.append((s, float(np.mean(win.x[lo:hi])), float(np.mean(win.y[lo:hi])),
                             sp.peak_freq, sp.snr, sp.harmonic_score))
        targets = []
        if rows:
            a = np.array(rows)
            half = 0.05 * max(win.width, win.height)
            bbox = np.stack([a[:, 1] - half, a[:, 2] - half,
                             a[:, 1] + half, a[:, 2] + half], axis=1)
            targets = [Target(id=0, t=a[:, 0], cx=a[:, 1], cy=a[:, 2], bbox=bbox,
                              freq_hz=a[:, 3], snr=a[:, 4], harmonic=a[:, 5])]
        return DetectorResult(targets, self.name, dict(P), self.regime)
```

Reuse the suite's building blocks instead of reinventing them:
`gottlux.core.frequency` (spectra, NUFFT, ISI, flicker maps),
`gottlux.core.detect.cluster_frame` (connected-component blobs),
`gottlux.core.background` / `gottlux.core.filters` (foreground masks),
`gottlux.detectors.tracking.MultiTracker` (greedy NN association with coasting).
`gottlux/detectors/flutter.py` shows them composed into the full
foreground → cluster → FFT-verify → track pipeline.

---

## 2. Custom analyses

Headless runs (`gottlux clip.raw --analyses …`) are orchestrated by
[`gottlux/run/pipeline.py`](../gottlux/run/pipeline.py). An **analysis** is a plain
function

```python
def analysis_myname(rec, cfg, run):    # Recording, Config, RunFolder
    ...
```

registered in the module-level analyses registry, `pipeline._ANALYSES` (a
`dict[str, callable]`). The core entries (`overview`, `spectral`, `panorama`,
`performance`) are inserted there directly, and the ported rotation suite merges in the
same way (`_ANALYSES.update(ROTATION_ANALYSES)`) — a plugin does likewise:

```python
from gottlux.run import pipeline
pipeline._ANALYSES["burst_stats"] = analysis_burst_stats
```

After that, `gottlux clip.raw --analyses overview,burst_stats` runs it; a failing
analysis is caught, logged, and never breaks the rest of the run.

**What an analysis receives and returns.** It returns nothing; it *writes* — into the
reproducible run folder through the provenance API
([`gottlux/run/provenance.py`](../gottlux/run/provenance.py)):

* `run.subdir("myname")` → the analysis's own output directory (created automatically);
* `run.record("myname", {...})` → key numbers for `run_manifest.json` / `RUN_SUMMARY.txt`;
* `run.add_artifacts(paths)` → the files the analysis wrote, listed in the manifest.

Use the savers in `gottlux.io.export` (`save_figure`, `save_table`, `save_arrays`,
`save_hdf5`, `save_json`) — each returns the list of paths it wrote, which is exactly
what `run.add_artifacts` wants. A minimal, complete analysis:

```python
import os
from gottlux.io import export

def analysis_burst_stats(rec, cfg, run):
    """Events per 100 ms bin — a toy 'burstiness' table + a headline number."""
    out = run.subdir("burst_stats")                        # its own run-folder subdir
    centers, rate = rec.event_rate(0.1)
    arts = export.save_table({"t_s": centers, "rate_hz": rate},
                             os.path.join(out, "burst_stats"))
    run.record("burst_stats", {"peak_rate_hz": float(rate.max()) if len(rate) else 0.0})
    run.add_artifacts(arts)
```

Everything else — the manifest with config + input SHA-256 + environment, the source
snapshot, the summary — is handled by the pipeline around the analysis.

---

## 3. A custom GUI tab

Every tab in the dashboard is an ordinary `QWidget` bound to the app's **shared
`TimeController`** (the clock: cursor, playback, accumulation window, In/Out selection —
`gottlux/app/transport.py`). The minimal contract, as `MainWindow` uses it
([`gottlux/app/main.py`](../gottlux/app/main.py)):

* **required** — `__init__(self, controller, filters=None)` stores the clock, and
  `set_recording(self, rec)` is called whenever a recording (or its preview) loads;
* **react to the clock** — connect `controller.cursorChanged` / `controller.accumChanged`
  to the tab's render, and read the current window with `controller.accum_window()`;
* **optional niceties** — `sync()` (the "Sync views" toolbar action), `sensor_size()` +
  `capture_frame(t, dt=None, size=None)` (the faithful **Capture…** video path), and
  `capture_clock()` if the tab runs its own clock (the Multi-clip slate does).

The smallest real tab to crib from is the event-rate tower,
[`gottlux/app/tower.py`](../gottlux/app/tower.py) (~380 lines including its controls):
its `__init__` takes `(controller, filters)`, `set_recording` stores the recording and
re-renders, `_render` pulls `t0, t1 = self.ctl.accum_window()`, slices
`rec.window(t0, t1)`, applies the shared filter suite, and draws.

To mount a tab, add an instance to `MainWindow.__init__`'s `self.panels` tuple and a
matching title to `MainWindow._TAB_NAMES` — every panel in that tuple automatically gets
`set_recording` on load and participates in error-isolated panel updates. (Tabs are the
one extension that currently needs those two lines in the package; detectors and
analyses need zero.)

---

## 4. Plugins without forking: `GOTTLUX_PLUGINS`

All of the above can live **outside** the gottlux package. The environment variable
`GOTTLUX_PLUGINS` holds an `os.pathsep`-separated list (`;` on Windows, `:` on Linux) of
plugin sources — `.py` files, or directories whose top-level `.py` files are all loaded —
and both entry points (`gottlux` CLI and the GUI) import them at startup, *after* the
registries exist ([`gottlux/plugins.py`](../gottlux/plugins.py)):

```bash
# Windows                                        # Linux/macOS
set GOTTLUX_PLUGINS=C:\lab\my_detector.py        export GOTTLUX_PLUGINS=~/lab/my_detector.py
gottlux --list_detectors                         gottlux clip.raw --detector blink
```

Import-time side effects are the registration mechanism: the plugin module's `@register`
classes and `pipeline._ANALYSES[...] = ...` assignments run when the plugin loads, and
the CLI/GUI then see them exactly like built-ins (including the workbench's auto-built
tuning panel). A broken plugin is reported as a one-line error and skipped — it never
takes the program down. Loading is idempotent, so a GUI launched through the CLI does
not import the module twice.

[`examples/custom_detector.py`](../examples/custom_detector.py) is written to work both
ways: point `GOTTLUX_PLUGINS` at it, or run it directly
(`python examples/custom_detector.py [clip]`).

---

## 5. User scripts — `--run-script`

Where a plugin extends the *registries* (a plugin becomes a detector or an analysis
inside the suite), a **user script** goes the other way: gottlux hands the events to
arbitrary user code and takes care of everything around the call — loading, windowing,
saving, and a README that records exactly what ran on exactly which data
([`gottlux/userscripts.py`](../gottlux/userscripts.py)).

**The contract.** A user script is any `.py` file defining

```python
def process(win, ctx): ...
```

* `win` is a `gottlux.io.recording.EventWindow` — the events already sliced to the
  requested time window and ROI, with the usual fields `x, y, p, t` (`t` in µs,
  zero-based to the parent recording) and `width`/`height`, plus two extras attached for
  scripts: `win.rec` (the parent `Recording`) and `win.roi` (the applied
  `(x0, y0, x1, y1)` tuple, or `None` for the full frame).
* `ctx` is a plain dict: `{"rec": Recording, "t0": float, "t1": float,
  "roi": tuple | None, "source_path": str, "output_dir": str, "args": list[str]}`.
  `t0`/`t1` are the *resolved* window bounds in seconds (the full span when no window
  was requested); `output_dir` is the run folder, already created, where the script may
  write files of its own; `args` carries the CLI's `--script-args` tokens.

**Return-type handling.** The return value decides what gets saved:

| `process` returns | saved as |
|---|---|
| `None` | nothing (the script wrote its own outputs, or only printed) |
| a dict of name → array/scalar | `results.npz` + a printed per-entry summary |
| a matplotlib `Figure` | `figure.png` + `figure.pdf` (300 DPI) |
| a dict `{"events": (x, y, p, t)}`, `t` in µs | a derived EVT2.1 `derived.raw` (any *other* keys in the same dict still go to `results.npz`) |

Any other return type raises a clear `UserScriptError` — a typo'd return is loud, never
silently dropped. Scripts may import anything installed — gottlux included — and each run
re-imports the file fresh, so an edit-and-rerun loop always executes the current code.
A failing script surfaces as one `UserScriptError` with the cause attached; it never
corrupts the caller's state.

**Invocation.** From the CLI:

```bash
gottlux INPUT --run-script my_script.py [--t_start T0 --t_stop T1] [--roi x0,y0,x1,y1] \
        [--script-args "tokens..."]
```

From the GUI: **Tools → Run user script on current view…** runs the script on exactly
the portion in view — the In/Out selection (or the cursor's accumulation window) and the
live viewer's current ROI.

Every run writes a stamped `gottlux_script_<name>_<UTC>/` folder whose `README.md`
records the script path + SHA-256, the source recording path + SHA-256, the window/ROI,
the gottlux version, the wall time, and what each output file is — the same traceability
promise the analysis run folders make, at user-script scale. The worked, commented
example is [`examples/user_script_example.py`](../examples/user_script_example.py).

---

## 6. Standalone tool export — `--export-tool`

The reverse direction: hand a collaborator one gottlux analysis **without gottlux**.

```bash
gottlux --export-tool list                 # the available tools + one-line descriptions
gottlux clip.raw --export-tool region_spectrum --roi 120,90,200,170 --freq_lo 90 --freq_hi 700
gottlux clip.raw --export-tool centroid_tracker --tool-format python --tool-out D:\handoff
```

This writes a self-documenting bundle folder `<input-stem>_tool_<NAME>_<stamp>/`:

* `data.h5` — the recording (honoring `--t_start/--t_stop/--roi`), in the same
  Metavision-compatible HDF5 layout `--to-hdf5` writes;
* `run_<NAME>.py` — a self-contained Python script depending only on
  **numpy + h5py** (+ scipy where noted; matplotlib strictly optional) — no gottlux import;
* `run_<NAME>.m` — the MATLAB twin, base MATLAB only (native `h5read`, no toolboxes);
* `README.md` — full provenance (source path, size, SHA-256, geometry, window), a table
  of every file accessed/produced, the baked parameters, the generating command line, and
  how to run each script;
* `provenance.json` — the same provenance facts, machine-readable.

The current CLI values (band, sample rate, accumulation window, ROI, sensor geometry)
are baked in as plain variables at the top of each script, so the recipient tunes them in
a text editor — and both scripts run against *any* GottLUX-exported `.h5`
(`python run_x.py other.h5`, or set `DATA_FILE` in MATLAB), not just the bundled one.
The GUI equivalent is **Tools → Export tool bundle…**, which exports the current
window/ROI.

The tools (each an honest, labeled simplification of the named gottlux module — see
[`gottlux/export_tools/`](../gottlux/export_tools/__init__.py)):

| tool | computes | ported from |
|---|---|---|
| `event_frames` | per-pixel event-count frames over a window | `core/accumulate.py` |
| `event_rate` | the event-rate-vs-time curve + plot | `io/recording.py` |
| `region_spectrum` | windowed ROI event-time FFT: peak frequency + SNR | `core/frequency.py` |
| `flicker_map` | per-cell dominant-frequency map | `core/frequency.py` |
| `centroid_tracker` | frame-differenced centroid track | `detectors/flutter.py` (sandbox preset) |
| `viz_config` | the exported visualization config (mode / tone-map / colormap / accumulation) rendered as frames | `core/render.py` / `core/tonemap.py` |

For `viz_config` the exporter bakes in the visualization settings as well — from the CLI
via `--viz_mode` / `--viz_cmap` / `--viz_tonemap` / `--viz_accum_ms`, or, when exported
from the GUI Tools menu, the live viewer's current mode, colormap, tone curve, and
accumulation window.

Adding a tool is adding one module under `gottlux/export_tools/` (a Python + MATLAB
template pair, a parameter manifest, an `ExportTool` entry in `TOOLS`) — the exporter,
`--export-tool list`, and the tests pick it up from the registry.
