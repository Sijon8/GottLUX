# GottLUX issue tracker — audit of 2026-10-18

Built on `95ebdec` plus the pending panorama box labels, MATLAB HDF5 matrix reader and file
associations, so the fixes sit on top of them.

**How this was checked.** Python 3.12, PySide6 6.11, pyqtgraph 0.14, matplotlib 3.11, numpy 2.5.
The full test suite was run offscreen. Every tab was rendered offscreen with the application's
own theme (`style.load_theme` + `apply_app_style`, as `main()` does), dark and light, at
1280×820, 1366×700 and 1920×1040, with `Humming_Bird_Fight_merged_2.raw` loaded. The headless
pipeline was run on the hummingbird clip (`--detector hummingbird`, every named plot) and on
`5inch_quadcopter_rotating.raw` (`--assume_spin auto`). Before/after screenshots came from a
worktree of `05f8cb0` against this branch. The two OpenGL tabs (Space-time 3D, Event-rate tower)
could not be rendered: offscreen Qt has no OpenGL, and a hidden native window has no GL context.

Severity: **High** = wrong results or an unusable window; **Medium** = misleading output or a
broken control; **Low** = cosmetic or minor.

Each fixed item has a regression test in `tests/test_audit_fixes.py` unless noted. All 15 tests
fail on `05f8cb0` and pass on this branch.

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

## Open — found, not changed

| ID | Sev | Area | Issue | Evidence / next step |
|---|---|---|---|---|
| GL-26 | Medium | Tests | **Intermittent `Fatal Python error: Aborted`** in the full suite: one run of three died about 65 % through, while pytest-timeout was starting its timer thread. No Python frame names the test. | Seen once, with `QT_QPA_PLATFORM=offscreen` on Windows. Suspects: a `QThread` destroyed while running (`RecordingLoader` / `PreviewPrefetcher`), or an exception escaping a paint handler (PySide6 aborts on those; see the 3.11 colormap fix in the changelog). Next: `pytest -v -p no:randomly -X faulthandler` until it repeats, then bisect. |
| GL-27 | Medium | Detectors | **The hummingbird detector breaks one bird into 35 short tracks**, and per-step frequencies jump between about 17 Hz and 120 Hz. | `tracks.png` after GL-01: 35 targets in a 3 s clip with two birds. The match gate (60 px, 8 coasted steps of 20 ms) is smaller than the bird's movement during the dart at 0.78 s, and each step's peak is the strongest bin in a 0.3 s window (4 Hz resolution, harmonics compete). A tracked-box spectrum (`examples/hummingbird_gallery.py`) gives a stable 64–76 Hz line. Next: a velocity-aware gate, and harmonic-sum peak picking. |
| GL-28 | Low | GUI | The panorama's degree-axis strip overwrites the bottom 14 sensor rows instead of sitting below the image, so those rows can't be seen or labelled. | `_pano_axis_overlay` writes `out[-14:]`. Moving it below the image changes the panorama pixel geometry the box labels (GL-23/24) rely on, so it's left for the owner. |
| GL-29 | Low | GUI | EBS viewer scrubbing is quantised to 1/1000 of the recording (3.6 s steps on a 1-hour file). | `QSlider(0..1000)`. |
| GL-30 | Low | GUI | Timeline "Selected clip" panel: with no clip selected, the trim slider and In/Out boxes stay enabled and show 0.000 s. | Screenshot. |
| GL-31 | Low | Plots | `viz.theme.apply()` rewrites matplotlib's global `rcParams` for the whole process, so a user who imports `gottlux.viz` gets their own figures restyled. | Use `mpl.rc_context(_RC)` per figure builder. |
| GL-32 | Low | Metrics | Staring runs report rotation-coverage metrics (`coverage_azimuth_deg 58`, `coverage_gain_vs_static`) from the default 58° FOV. | `metrics.json` of every staring run. |
| GL-33 | Low | Lint | 3,336 ruff findings; CI lint is `continue-on-error`. 2,934 are the house style (E702 1,970, E501 866, E701 98). The rest are worth a pass: 16 F841 unused locals (two of them were GL-15 and GL-17), 7 B007, 2 B904, 52 B905. | `ruff check . --statistics`. |
| GL-34 | Info | GUI | Space-time 3D and the Event-rate tower were not checked (they need OpenGL). | Check on a desktop session. |

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
