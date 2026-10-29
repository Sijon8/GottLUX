# GottLUX issue tracker — audit of 2026-10-18

**State.** The audit fixes (GL-01…25, 35, 36) and the Studio were published to GitHub `main`
on 2026-10-18 (`95ebdec..3e34bfc`, 14 commits); the website now carries `studio.html`. Since
then the quick viewer opens large files as byte samples and provenance records partial
decodes (see `CHANGELOG.md`, *quick viewer opens large files as samples*). The CI abort on
Python 3.10/3.11 is traced to PySide6 6.12.0 and that release is excluded (GL-39), and two
crash sources are fixed: garbage collection on worker threads (GL-26) and loader threads
destroyed while running (GL-40). The open items below (GL-27…34, 37, 38) are unchanged; GL-41 and GL-42 are new.

Built on `95ebdec` plus the pending panorama box labels, MATLAB HDF5 matrix reader and file
associations (`62ae51b`), so the fixes sit on top of them.

**How this was checked.** Python 3.12, PySide6 6.11, pyqtgraph 0.14, matplotlib 3.11, numpy 2.5.
The full test suite was run offscreen. Every tab was rendered offscreen with the application's
own theme (`style.load_theme` + `apply_app_style`, as `main()` does), dark and light, at
1280×820, 1366×700 and 1920×1040, with `Humming_Bird_Fight_merged_2.raw` loaded. The headless
pipeline was run on the hummingbird clip (`--detector hummingbird`, every named plot) and on
`5inch_quadcopter_rotating.raw` (`--assume_spin auto`). Before/after screenshots came from a
worktree of `62ae51b` against this branch. The two OpenGL tabs (Space-time 3D, Event-rate tower)
could not be rendered: offscreen Qt has no OpenGL, and a hidden native window has no GL context.

Severity: **High** = wrong results or an unusable window; **Medium** = misleading output or a
broken control; **Low** = cosmetic or minor.

Each fixed item has a regression test in `tests/test_audit_fixes.py` unless noted. All 16 fail on `62ae51b` (the GL-35 test hangs there) and pass on this branch.

## Fixed on this branch

| ID | Sev | Area | Issue | Evidence | Fix |
|---|---|---|---|---|---|
| GL-01 | High | Pipeline | **Preset detectors ran at the drone band in headless runs.** `hummingbird` (15–120 Hz), `insect`, `mosquito` and `bird` all ran at 80–800 Hz with the drone SNR gate. | `gottlux Humming_Bird_Fight_merged_2.raw --detector hummingbird`: report lists `freq_lo 80`, `freq_hi 800`; the flicker map peaks at 80 Hz, the band floor; track frequencies span 80–790 Hz. | Every detector call passed `cfg.freq_lo/freq_hi/snr_thresh` as overrides, so the preset never seeded its signature band. New `run.pipeline.adopt_signature_band()` gives a preset its band unless the caller set one (`--freq_lo/--freq_hi/--snr` mark it explicit). The flicker map and spectra follow, and the manifest records the band used. After the fix the flicker map shows the flight path at about 72 Hz, and the hover wingbeat measures 71.6 Hz. |
| GL-02 | High | Pipeline | **`--out` / `Config.output_root` was ignored.** Run folders went beside the recording, or into the working directory for in-memory recordings. | `--out X` wrote to `examples/data/gottlux_runs/`. The six synthetic runs from 22 Aug in the repo root's `gottlux_runs/` came from `tests/test_pipeline.py`, which sets `output_root=tmp_path`. | `RunFolder(cfg, rec, root=cfg.output_root)`. |
| GL-03 | High | GUI layout | **The main window could not be smaller than 1549 × 1038 px**: taller than a 1080p screen minus the taskbar, and wider than 1280/1366 laptops or 1080p at 125 % scaling (1536 px). The bottom rows of every tab were off-screen. Unthemed it was 1118 × 963. | `MainWindow().minimumSizeHint()`. Asked for 1280 × 820, the window came up 1549 × 1038. | Three causes. (a) The EBS viewer's stack of control rows (820 px minimum) now sits in a vertical scroll area that asks for its full height but can shrink. (b) The Multi-clip "No clips yet…" hint was one unwrapped line (1527 px minimum); it now wraps. (c) The EBS viewer density strip set its waveform as a QLabel pixmap, which becomes the label's minimum size; see GL-04. Now **1156 × 665**. |
| GL-04 | Medium | GUI | EBS viewer **event-density strip covered only part of its width**: it was drawn at the size the widget had while its tab was hidden, and never rescaled. | Screenshot: the waveform stops about halfway across. | It now paints the cached image scaled into the widget in `paintEvent`, with no pixmap and no minimum size. It is vectorised (the per-column Python loop is gone). Rescaling in `resizeEvent` instead recursed until the stack overflowed, and the test guards against that. |
| GL-05 | Medium | GUI | **Spin-box arrows and menu-button indicators drew as plain squares.** | Zoomed screenshot of any spin box: two identical squares. | Qt stylesheets don't draw the CSS "zero-size box with one coloured border" triangle. `style._arrow()` writes a 10 px SVG triangle per colour to the temp folder, and the sheet uses `image: url(...)`. No test: checked visually. |
| GL-06 | Medium | GUI | **After a dark → light switch, plot axes and titles stayed light grey on the light canvas**, nearly invisible in the Fusion lab and the workbench spectrum. | Light-theme screenshots after toggling. | `restyle_plots` only repainted backgrounds. `_reink_axes` now re-colours axis pens, tick text, axis labels and plot titles that still carry the outgoing theme's foreground. Items with a colour of their own are left alone. |
| GL-07 | Medium | Plots | **Flicker maps were confetti**: noise cells were drawn at 70–80 % opacity, burying the target, in both the workbench and the headless figure. | SNR of noise-only cells on the hummingbird clip: median 11, 90th percentile 15–24; the old alpha scale started at SNR 0. | `theme.flicker_rgba` scales alpha from a noise floor (the median valid cell) to the 99.5th percentile, at least 3× the floor. `snr_floor=0` restores the old look. |
| GL-08 | Medium | Plots | **The `spectrogram` figure covered only the clip from 30 % onward, and its time axis started at 0**: window-relative, mislabelled. | 3 s clip: the axis ran 0–2.05 s. | It now covers the whole clip (or the middle 30 s) on the recording's clock. |
| GL-09 | Low | Plots | **Every event-rate trace dipped at the end**: the last partial bin was divided by the full bin width. The bins also assumed the stream starts at t = 0. | Fusion lab and `event_rate` figure: the last point at about 0.65× the level. | `Recording.event_rate` bins on the recording's own clock and divides the edge bin by the span it covers. It drops a sliver bin under a quarter of a bin wide. |
| GL-10 | Medium | Plots | **Detection overlay**: all 37 targets were labelled on top of each other, and labels ran off the image. | `overlay.png` from the hummingbird run. | It draws the 8 most confident targets ("top 8 of N" in the title). Labels clip to the image, and near the right edge they sit left of the box. No test: checked visually. |
| GL-11 | Low | Plots | Tracks figure: the legend sat inside the top panel, frameless, over the traces. | `tracks.png`. | One legend beside the three panels. The title reads "top 8 of N targets". No test: checked visually. |
| GL-12 | Low | Plots | Flicker-map figure: a long run name ran into the colour bar (the axes are at fixed fractions). | `flicker_map.png`: the title cut at "…merged_". | New `theme.wrap_title()` is used here, in the tracks figure and in the overlay. The colour-bar label now says what the opacity means. |
| GL-13 | Low | Plots | Spectrum figure: the legend overlapped the peak marker. | `spectrum.png`. | `loc="best"`, framed. No test: checked visually. |
| GL-14 | Medium | Plots | **The 360° panorama was striped**: 240 elevation bins over about 300 integer sensor rows put one row in some bins and two in others. | `panorama.png` from the rotating quadcopter. | One elevation bin per sensor row by default (`el_bins=None`). A number keeps the old binning. |
| GL-15 | Medium | Plots | **Target radar**: each target got its own colour scale (vmax = 1.5 × its frequency), so every target had the same hue. There was no colour bar and no radius label. | `radar.png`. | One shared frequency scale, a colour bar, and a "radius = range (m)" note under the dial (a polar y-label lands on the 270° tick label). No test: checked visually. |
| GL-16 | Medium | Pipeline | **`--assume_spin` only reached the rotation analyses**: the default analyses and every `--plots` figure stayed in staring mode. | `--assume_spin auto --plots panorama` gave a "Relative-bearing strip (no telemetry)" and a `staring` run folder. | `run_recording` synthesises the spin telemetry first. The quadcopter run now reports a period of 0.9388 s (autocorrelation 0.889) and draws a de-rotated panorama. No test: checked by running it. |
| GL-17 | Medium | GUI | **EBS viewer played faster than the requested frame rate**: the adaptive timer used render time + 1 ms and ignored the target interval (the variable was computed but unused). | Reading the code (ruff F841 `target_ms`). | `max(target_ms, render_ms + 1)`. No test: timing-dependent. |
| GL-18 | Low | GUI | The EBS viewer opened on a **black frame** ("window: 0 ev"): t = 0 with a trailing accumulation window. | Screenshot. | It opens one accumulation window in. |
| GL-19 | Low | GUI | The EBS viewer kept a **blank 44 px band** under the view-mode row in the Events, Panorama and Radar modes (empty parameter pages). | Screenshot. | The page stack is hidden when the page is empty. |
| GL-20 | Low | GUI | The EBS viewer annotation strip was hard-coded dark: a black bar in the light theme. | Light screenshot. | It uses the theme's `BG2`. |
| GL-21 | Low | GUI | Workbench colour key: "opacity = SNR" was clipped to "opacity". | Screenshot. | The bar width is computed from the label widths. |
| GL-22 | Low | GUI | Fusion lab rate axis mixed "1.2e+06" with "800000". | Screenshot. | The axis has units (`ev/s`), so pyqtgraph scales the ticks (Mev/s). |
| GL-23 | Medium | GUI (box labels, uncommitted work) | **Annotation sidecar**: a failed save was swallowed (`except: pass`), so labels made beside a read-only recording were lost silently. An unreadable sidecar was overwritten on the next save. The write was not atomic. | Reading the code. Shared data folders are often read-only. | The save writes to `.tmp` then `os.replace`, and a failure warns once. An unreadable sidecar is copied to `*.unreadable-<epoch>` before anything can overwrite it. |
| GL-24 | Low | GUI (box labels, uncommitted work) | A box dragged to the panorama's right edge (az1 = 360°) folded to column 0 and drew a stray edge at 0°. | `_draw_label_boxes` with `az1 = 360`. | It is clamped to the last column. |
| GL-25 | Low | GUI | The compare pane paused only the shared clock on a tab switch, so its Multi-clip clock kept running. | Reading the code. | It uses `_pause_all_clocks()`, like the main tabs. No test. |
| GL-35 | High | Rotation | **`estimate_spin_period_s` could loop forever**, hanging `--assume_spin auto` and anything else that estimates a spin. The pass scan jumps to `pk + guard`; when the strongest sample in the window is the window's left edge (a pass wider than 0.4 periods, at or above the 90th-percentile threshold, whose maximum is its first bin), `pk + guard` is the current index and the scan never advances. | Found while building the Studio's radar look: a synthetic spinning scene hung in `telemetry.py:99`. `test_spin_period_estimate_terminates` hangs on `62ae51b` (killed by a 20 s timeout) and passes here. | It always advances (`max(pk + guard, i + 1)`). |
| GL-36 | Low | Rotation | The same estimator timed each pass by its argmax. On a flat-topped pass (a source crossing the field of view) the argmax lands anywhere on the top, which put the period off by about 2 %; for de-rotation that is about 7° of drift per revolution. | Synthetic 0.5 s spin: 0.5096–0.5109 s before; 0.4989–0.5053 s after (three noise seeds). On `5inch_quadcopter_rotating.raw` it moves from 0.93880 to 0.93996 s. | Passes are timed by their rate-weighted centroid above the threshold. |
| GL-26 | Medium | Tests / GUI | **Intermittent `Fatal Python error: Aborted` / access violation** in the full suite, always inside a test that runs a Python worker thread (the screen recorder's encoder). Root cause: windows and panels earlier tests dropped sit in reference cycles; CPython frees them at its next cyclic collection, which runs in *whichever* thread allocates — here the encoder thread — so Qt destructors ran off the GUI thread. | `test_quickview_sampling.py` + `test_screenrec.py` under Python 3.11 + coverage: 2 aborts in 3 runs before, 6 passes in 6 after. | Tests: an autouse fixture (`tests/conftest.py`) runs pending deletions and `gc.collect()` on the GUI thread after every test. App: `gottlux.app.gcguard` turns automatic cyclic collection off and collects on a GUI-thread timer (pyqtgraph's `GarbageCollector`), installed by `gottlux-gui` and `gottlux-view`, since the app also runs Python work on worker threads (recorder, Studio export, decode and sampling loaders). |
| GL-39 | High | CI | **CI aborted (exit 134) on Python 3.10 and 3.11 after every test had passed**, from the 18 Oct publish (`3e34bfc`) on; 3.12, 3.13 and Windows passed, and `95ebdec` had passed. Cause: **PySide6 6.12.0** (released 8 Oct 2026, so CI picked it up between the two runs), not GottLUX code. Its Linux wheels drop a reference to `True` on every signal emit and to `None` on every call that returns nothing (PYSIDE-3477), so at interpreter shutdown `True` reaches a refcount of zero: `Fatal Python error: bool_dealloc`. Python 3.12+ is immune (`True`/`None` are immortal there). | The CI annotations: exit 134, with the end-of-run warnings already reported. Reproduced in a `python:3.11` Linux container: the full suite aborts at shutdown with `bool_dealloc`. Minimal check on x86-64 Linux, Python 3.11.17: 100,000 `Signal(bool).emit(True)` calls abort with `bool_dealloc` and 100,000 `setObjectName()` calls with `none_dealloc` on 6.12.0; on 6.11.2 both leave the refcounts unchanged. PySide6 6.12.0 on Windows is unaffected. | `PySide6>=6.6,!=6.12.0` in `pyproject.toml` (`gui`, `all`) and `requirements.txt`; lift it once a fixed release ships. No test (a dependency pin). |
| GL-40 | Medium | GUI | **Closing a window or quitting while a decode was running aborted the process** (`QThread: Destroyed while thread '' is still running`). The loader threads (`RecordingLoader`, `SampledLoader`, `PreviewPrefetcher`, `DetectorWorker`) were held only by the window that started them, so when it closed and was collected, or Python exited, Qt destroyed a running thread. | Running `tests/test_quickview_sampling.py`, or `test_quickview_opens_h5_and_capture_folder`, on its own on Linux: exit 134 with that message. Now 3 of 3 runs exit 0 for each. | Every loader is a `gottlux.app.loader.Worker`, referenced until it finishes. `gottlux-gui`, `gottlux-view` and the test session wait for running workers at exit (`wait_for_workers`). A decode cannot be cancelled midway, so it completes its cache first. Test: `test_worker_outlives_its_owner_and_is_waited_for`. |

## Button and action audit — 2026-10-29

Every button, toolbar and menu action, checkable control and combo entry of the main window
(all tabs and the split pane), the Canvas composer, the Timeline, the Studio and the quick
viewer was triggered with a recording loaded (1,228 actions, offscreen, dialogs stubbed and
recorded, each tab in its own process with a hang watchdog), plus 63 end-to-end exports
whose files were read back (videos decoded frame by frame, `.raw` re-opened). Tests:
`tests/test_audit_buttons.py`.

| ID | Sev | Area | Issue | Fix |
|---|---|---|---|---|
| BA-01 | High | Capture / screen rec | `qimage_to_rgb` returned a view of a freed QImage when a row had no padding (width divisible by 4 — every common screen): garbled frames, then an access violation in the encoder. | The frame is copied. |
| BA-02 | High | Main window | On the Studio tab Capture…, Export…, tool bundle, user script and the space bar raised: the Studio's `capture_clock()` is `None`. | Fall back to the app clock. |
| BA-03 | High | Multi-clip | Layout › Overlay deleted every pane with the dropped splitter; switching back raised. | Panes and the overlay view move to a hidden holder first. |
| BA-04 | High | Studio, Batch-trim | Their workers were plain QThreads, not waited for at exit: quitting mid-export aborted. | Both are `loader.Worker`s. |
| BA-05 | High | Range lab | *Analysis video* always failed: it called a range function that does not exist. | `photogrammetry.range_for_pixels`. |
| BA-06 | Medium | Capture | *Abort* kept the partial MP4 and reported the planned frame count. | An aborted capture is discarded and says so; counts are the frames written. |
| BA-07 | Medium | Multi-clip | Selection → .raw, Merge highlights and MP4 on the slate's transport always said "Load a recording first". | The slate exposes its clips; each clip is cut on its own clock, one file per clip. |
| BA-08 | Medium | Canvas composer | A second cell of the same file got the source `<path>_2`, which does not reload. | Cells of one recording share its source; a save notes cells with no file; relative sources resolve beside the spec. |
| BA-09 | Medium | Fusion lab | Recordings shorter than ~2 s failed: `np.convolve(mode="same")` returns the kernel length when it is longer. | The kernel is clamped to the series. |
| BA-10 | Medium | Studio | *Load recipe…* with a malformed file raised out of the slot. | A warning dialog. |
| BA-11 | Medium | Batch-trim | A window past a clip's end wrote empty 80-byte `.raw` files and reported success (default start 3 s). | Empty outputs are listed in the result; the start defaults to 0. |
| BA-12 | Low | Export | *Save 3-D snapshot* said Saved with no framebuffer. | A null grab or a failed save writes nothing and reports nothing. |
| BA-13 | Low | `.raw` cut | `cut_clip` used `[t0, t1)`, so a whole-clip cut dropped the final event. | A window reaching the recording's end is closed. |
| BA-14 | Low | Timeline | With nothing selected the trim and cell controls stayed live (GL-30) and *Cut selected* did nothing. | Disabled until a selection; *Cut selected* says what to select. |
| BA-15 | Low | Workbench | *Find rotation rate* opened a file browser unasked and hid write errors. | The report path (or the error) is in the label's tooltip; a read-only data folder falls back to the user folder. |
| BA-16 | Low | Loading | Load failures showed a raw traceback. | The message, with the traceback under *Details*. |
| BA-17 | Low | Screen rec | An exception with no message read "could not be saved ()". | Its type is named. |
| BA-18 | Info | Exports | Formats, controls and encoder settings differed between the Studio and the composer / Timeline (MP4 only, fixed 30 fps, no cancel). | One encoder and one set of options everywhere (see the changelog). Capture stays MP4 (screen grabs are sequential). |
| BA-19 | Low | Space-time | Exporting the rotor-ladder study without an OpenGL context crashed in `renderToArray`. | GL renders are skipped without a context (`capture.gl_ready`). |

## Open — found, not changed

| ID | Sev | Area | Issue | Evidence / next step |
|---|---|---|---|---|
| GL-27 | Medium | Detectors | **The hummingbird detector breaks one bird into 35 short tracks**, and per-step frequencies jump between about 17 Hz and 120 Hz. | `tracks.png` after GL-01: 35 targets in a 3 s clip with two birds. The match gate (60 px, 8 coasted steps of 20 ms) is smaller than the bird's movement during the dart at 0.78 s, and each step's peak is the strongest bin in a 0.3 s window (4 Hz resolution, harmonics compete). A tracked-box spectrum (`examples/hummingbird_gallery.py`) gives a stable 64–76 Hz line. Next: a velocity-aware gate, and harmonic-sum peak picking. |
| GL-28 | Low | GUI | The panorama's degree-axis strip overwrites the bottom 14 sensor rows instead of sitting below the image, so those rows can't be seen or labelled. | `_pano_axis_overlay` writes `out[-14:]`. Moving it below the image changes the panorama pixel geometry the box labels (GL-23/24) rely on, so it's left for the owner. |
| GL-29 | Low | GUI | EBS viewer scrubbing is quantised to 1/1000 of the recording (3.6 s steps on a 1-hour file). | `QSlider(0..1000)`. |
| GL-31 | Low | Plots | `viz.theme.apply()` rewrites matplotlib's global `rcParams` for the whole process, so a user who imports `gottlux.viz` gets their own figures restyled. | Use `mpl.rc_context(_RC)` per figure builder. |
| GL-32 | Low | Metrics | Staring runs report rotation-coverage metrics (`coverage_azimuth_deg 58`, `coverage_gain_vs_static`) from the default 58° FOV. | `metrics.json` of every staring run. |
| GL-33 | Low | Lint | 3,336 ruff findings; CI lint is `continue-on-error`. 2,934 are the house style (E702 1,970, E501 866, E701 98). The rest are worth a pass: 16 F841 unused locals (two of them were GL-15 and GL-17), 7 B007, 2 B904, 52 B905. | `ruff check . --statistics`. |
| GL-34 | Info | GUI | Space-time 3D and the Event-rate tower were not checked (they need OpenGL). | Check on a desktop session. |
| GL-37 | Medium | Detectors | **A target that turns ON at one phase and OFF half a cycle later puts two bursts per cycle into the total event rate, so its spectrum peaks at 2·f0.** The Studio's stroboscope now adds the ON − OFF rate and a subharmonic rule. `core.frequency.region_spectrum` (used by every flutter detector and the workbench) still measures the total rate only. | Synthetic 50 Hz ON/OFF blink in `tests/test_studio.py`: the total-rate peak was 100.08 Hz; with the signed rate it is 50.0 Hz. On the hummingbird's hover both give ≈ 72 Hz. Whether the detectors' harmonic comb hides this on real rotors was not checked. |
| GL-38 | Info | Tests | `test_main_window_fits_a_laptop_screen` measures text widths, so it skips when offscreen Qt has no fonts (Windows needs `QT_QPA_FONTDIR`; Linux CI uses fontconfig). Without fonts the placeholder glyphs made the window 1951 px wide. | Seen in a local run without `QT_QPA_FONTDIR`. |
| GL-41 | Medium | Platform | **PySide6's Linux ARM (aarch64) wheels lose references to `True` and `None` on Python 3.10/3.11 in every version tried**: 6.9.3, 6.10.3, 6.11.0, 6.11.1, 6.11.2 and 6.12.0. The GL-39 pin does not help on ARM. | The GL-39 minimal check in `python:3.10` and `python:3.11` aarch64 containers aborts (`bool_dealloc`, `none_dealloc`) on every one. | Use Python 3.12+ on ARM Linux (immortal singletons); consider requiring it there with an environment marker, or report upstream. |
| GL-42 | Medium | Tests / GUI | **On Windows the full suite sometimes crashes at interpreter exit after every test has passed** (access violation, no Python frame): 2 of 5 full Windows runs on 18 Oct, Python 3.11 and 3.12. Windows CI passed. Windows the tests create (e.g. `MainWindow()` in `test_main_window_has_all_tabs`) are never closed and stay alive to the end, so Qt deletes them at shutdown, in no fixed order, together with pyqtgraph's parentless `ViewBoxMenu` popups. | Deleting the leftovers explicitly at session end crashes every time once a `ViewBoxMenu` goes before its `MainWindow`; a fresh `MainWindow` deleted on its own is fine. | Give the GUI tests a fixture that closes and deletes each window they create (the window first, its menus with it), and find what keeps dropped windows alive. |

## How to reproduce the checks

```bash
# environment (uv; any venv with the [all,dev,hdf5] extras works)
python -m pytest -q                                    # QT_QPA_PLATFORM=offscreen
python examples/hummingbird_gallery.py                 # renderings, ~4 min with videos
gottlux examples/data/Humming_Bird_Fight_merged_2.raw --detector hummingbird \
        --plots event_rate,flicker_map,spectrum,spectrogram,tracks,overlay --out runs --no_open
gottlux examples/data/5inch_quadcopter_rotating.raw --assume_spin auto --detector drone \
        --plots panorama,radar --out runs --no_open
```

The screenshot harness (`shots.py`, offscreen, both themes, several window sizes) is kept
outside the repository. It is not part of the package.
