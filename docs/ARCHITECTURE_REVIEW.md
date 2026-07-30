# GottLUX — architecture review (first-principles pass, 2026-06-16)

A ground-up review aimed at removing duplication and the class of bugs that showed up when
**two clips are loaded and the user moves between views**. This records what was found, what was
fixed, and what remains.

---

## 1. The one duplication that mattered most: the render pipeline

The exact sequence *window → live filters → accumulate(mode) → tone-map(expr/gamma/scale) →
colourize* was **copied in six places** — the on-screen `_render` **and** the faithful
`capture_frame`/`intensity_frame` of the Live viewer, the Multi-clip panes, and the Range lab.
Six copies means six chances to drift (a fixed colormap here, a missing filter there), and it is
exactly where the on-screen image and the captured video could disagree.

**Fixed:** one canonical pipeline in **`gottlux/core/render.py::render_frame`** (pure NumPy,
returns `(disp, levels, vmax, win)`) + **`viz/video.disp_to_rgb`** for colourising. All six call
sites now go through it, so *what is shown is what is captured*, by construction.

## 2. Clock ownership — the "moving between views" conflict

GottLUX has **one shared clock** for the main tabs, but the **Multi-clip slate runs on its own
clock** (each pane has a child clock = master cursor + slate offset). Two real conflicts fell out:

- **Spacebar** toggled only the app clock, so on the Multi-clip tab it did nothing useful.
- A tab with its own clock (Multi-clip) **kept animating in the background** after a switch
  away, because tab-change only paused the app clock.

**Fixed (first-principles model):** `MainWindow.active_clock()` returns *the clock of the visible
tab*; spacebar, Capture and the Export window now route through it, and switching tabs
**pauses every clock**. A panel advertises its own clock via `capture_clock()` (only Multi-clip
does); everything else shares the app clock. This is the single rule that removes the cross-view
surprises.

## 3. Detections table — flattened three ways

The per-detection table (tracks → rows) was flattened independently in the headless pipeline, the
GUI export bundle, and the KPI extractor, with slightly different columns each time.

**Fixed:** one **`detectors/base.detections_table(result)`** (a superset of columns) is the single
source for all three.

## 4. Exports now land in unique, self-describing folders

Every multi-artifact export (the program-wide Export, the Range-lab study, the Multi-clip fusion,
the KPI bundle) now creates a **uniquely + helpfully named subfolder**
`<name>[_<purpose>]_<UTC-stamp>/` inside the chosen folder
(`io/paths.unique_export_dir`), so exports never overwrite each other and are easy to find. Single
saves (a cut/stitched `.raw`, one figure, a capture `.mp4`) stay as named files.

---

## Capture & video — faithful, settings-accurate, high-res

- **Faithful render** for the frame views (Live viewer, Range lab) and the **Multi-clip Overlay**;
  **offscreen GL render** for the 3-D views (Space-time, Event-rate tower) via
  `GLViewWidget.renderToArray`. Every view captures at native / 720p / 1080p / 2× / 4×.
- The program-wide **Export** has a **video output** that reproduces the *active view's tuned
  settings* over the window (fps / resolution / infographic banner tunable in the dialog), into the
  unique export folder.

## The clip-editing suite (now discoverable)

A single **✄ Clip editor** toolbar button opens the comprehensive editor: add clips, **trim**
(In/Out), **crop** (per-clip ROI), **reorder**, set a gap, **cut a single clip**, or **stitch** all
into one valid `.raw` (`io/writer.cut_clip` / `stitch_clips`). The In/Out **selection bar above the
timeline** (every tab) feeds Cut / Capture / Export.

---

## Remaining (honest list)

- **3-D GL capture orientation** is a transpose of `renderToArray`; if a clip comes out mirrored,
  the fix is a one-line axis flip in `app/capture.gl_to_rgb` (needs a real display to confirm).
- **`_render` display glue** (colorbar labels, readouts) still lives per-view — only the *pipeline*
  was centralized, which is the part that mattered; the thin display code is intentionally local.
- **Split-view** shares the app clock across both panes by design; per-pane independent clocks in
  split view are not implemented.
- The Multi-clip **per-clip child panels** are full panel instances; a lighter render-only path
  could reduce memory for many clips. Not a correctness issue.
- A full **NLE** (preview-while-editing, razor/ripple, undo) remains future work
  (see [`../FUTURE_WORK.md`](../FUTURE_WORK.md), *Timeline editor, phase 2*).

_Verified by the synthetic test suite (139 tests). GUI surfaces are smoke-tested; the on-screen
look of the new GL capture and editor wiring is worth a quick interactive pass._
