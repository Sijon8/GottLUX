# gottlux — Interactive Dashboard Guide

Launch it with **`gottlux-gui`** (or `python -m gottlux`). Open a recording from the toolbar
(**Open file…** for a `.raw`/`.meta.json`, **Open folder…** for a capture folder). Decoding runs
on a background thread behind a progress bar; the same recording then feeds all ten tabs.

This guide walks the core analysis tabs (Live viewer, Event-rate tower, Space-time 3D, Flutter
workbench, Sandbox, Fusion lab, Timeline) plus the **Canvas composer** and the **Tools menu**;
the **Multi-clip**, **Range lab**, and **EBS viewer** tabs are summarized in the
[README](../README.md).

The window title bar shows the loaded recording's geometry, event count, duration, encoding,
and mode (ROTATION if telemetry was found, else STARING).

---

## The shared transport (on every tab)

Every visualization carries the same **transport bar** — **▶ play / ❚❚ pause**, a **seek
slider**, a **time readout**, an **FPS** control, and an **Accumulation** control (a **log
slider + numeric box**, spanning 0.5 ms … 2 s) — and they all drive **one shared clock**. So:

- **Seek once, seek everywhere.** Drag the seek bar (or hit play) on any tab and every other
  tab is looking at the *same moment*. Switching tabs resumes at exactly the same instant.
- **Accumulation is program-wide.** Drag the **Accum** slider (or type a value) on any tab and
  it updates everywhere — one integration time for the whole program (the workbench
  additionally has its own longer *analysis window* for frequency resolution).
- **Playback is in *equivalent* FPS.** **FPS** is a slow-motion-camera capture rate, not a
  screen refresh: **30 fps plays at real time**, *higher* FPS plays *slower* (100000 fps ≈ 3333×
  slow-motion), lower FPS plays faster than real time — the recording advances `30 ÷ FPS`
  recording-seconds per real second, and a small readout next to the box shows the resulting
  speed. Pick a preset (0.5 … 100000) or type any value. **Accumulation (exposure) is a separate,
  independent control** spanning **10 µs … 2 s** — set it to 1/FPS to match a high-speed camera's
  time resolution.
- **Live everywhere.** Press play and the live viewer animates, the tower and 3-D cloud slide forward
  in time, and the workbench's flicker map steps forward (throttled, since it is heavier).
- Only the **visible** tab renders, so playback stays fast.

**`⟳ Sync views`** (toolbar) recomputes every tab at the current cursor on demand — so a
moment found by scrubbing the live viewer is reflected in the 3-D cloud and the workbench
without touching each one.

**`⊞ Split view`** (toolbar) opens a second tabbed pane next to the first, so two
visualizations sit **side by side** (e.g. the live viewer and the space-time cloud) analysing
the same moment — both panes follow the shared clock.

Every 3-D view (tower, space-time) carries a CAD-style **orientation cube** in its top-right
corner: click a face to snap to the Top / Front / Side view, or the small buttons for the hidden
faces and an isometric home — no mouse-orbiting to find a known angle.

Every colour map has a **legend** (a colorbar on the live viewer, a frequency gradient on the
flicker map, a colour key on the 3-D view), so a colour always has a readable meaning.

**`☀ Light theme` / `☾ Dark theme`** (toolbar) switches the whole instrument between the dark
instrument panel and a daylight counterpart — every tab, dialog, painted icon, plot canvas and
timeline lane at once, live, with no restart. The action names the theme it switches *to*. The
choice is remembered and shared with the quick viewer (`gottlux-view`), which carries the same
toggle in its top-right corner. The 3-D views' **Background** preset defaults to *App theme*, so
their canvas follows the switch too; pick any other preset there to pin a fixed colour. Result
figures and exported video stay exactly as they were — the theme dresses the live UI, not the
data.

**`Export ▾`** (on every tab) saves the current view as a **journal figure** (PNG + PDF) or
writes the data as a **block/cube** re-openable in NumPy / MATLAB / any HDF5 tool:
the **space-time event cube** `V[y, x, t]` (live viewer & 3-D), the **flicker-map cube**
(per-cell frequency / SNR arrays), a **detections table** and a **first-principles detection
report** (workbench), or a **3-D snapshot**.

**`⤓ Export…`** (toolbar) is the *overall* export: one dialog for ticking **exactly** which
artifacts to write — frame figure, event cube, event-rate series, flicker-map figure/cube,
region spectrum, detections, detection report, run config — into one folder with a
`manifest.json`. Items whose inputs aren’t ready yet (no flicker map computed, no detector run)
are shown disabled with the reason.

**`⏺ Screen rec…`** (toolbar) is a live screen recorder (Snipping-Tool style): record the **entire
app window** (menus, controls and the view, across tab switches) — the default — or a dragged
screen region / the current view, straight to MP4. **`● Capture…`** instead re-renders the active
tab's exact tuned settings to a high-res video + infographic poster.

**Editing `.raw` clips.** **`✂ Cut selection → .raw`** (toolbar) writes the In/Out selection to a
new clip; the **Timeline tab** (below — also opened as a dialog by the toolbar's **`✄ Clip
editor`**) is a video editor over several clips: trim, per-clip spatial **crop** (an ROI applied
on cut/stitch), reorder by dragging blocks on a track lane, mosaics (**canvas blocks**) arranged
inline in its preview, and an **overlay lane** whose items render over the whole program. It
exports the program as a video, or the sequential lane as one `.raw`. Two File-menu
tools work on huge files efficiently: **File → Quick-cut a .raw (no decode)…** crops a multi-GB
recording to a time window **without opening/decoding it** (it indexes the EVT2.1 byte stream, shows
the duration, and writes the clip directly), and **File → Batch-trim folder…** shortens every `.raw`
in a folder by one shared window, re-based to a common origin so synced clips stay aligned. All
`.raw` writers stream in bounded blocks, so even a long crop/merge stays within a few hundred MB.

---

## Live viewer

A fast, scrubbable, playable window into the stream, with a **colorbar** showing the current
value range and units. Uses the shared transport (play / seek / FPS / accumulation) above,
plus:

| control | what it does |
|---|---|
| **Mode** | `count` / `time_surface` (sharp motion) / `polarity` (ON−OFF) / `on` / `off` / `binary` |
| **Color** | colormap (inferno, viridis, turbo, …) — the colorbar updates to match |
| **Scale** | **dynamic** (re-pick the white-point every frame) or **static** (hold it fixed so frames stay comparable and a flash doesn’t rescale the view) |
| **Expr** | **map expression** — the tone curve that fixes *dilution* (hot regions washing out faint ones): `linear / sqrt / gamma / log / asinh / equalize / percentile`. Start with `sqrt` or `log`. |
| **γ** | exponent for the `gamma` expression (γ<1 lifts faint regions) |
| **Freeze scale** | snap the static white-point to the current frame |
| **ROI** | drag a box on the image; it’s mirrored into the workbench’s region for spectral analysis |

The readout shows the window’s event count, ON/OFF split, and rate. *Tip:* if a bright rotor
disk is drowning out a faint target, switch **Expr** to `log` (or `equalize`) — the faint
structure lifts out of the floor without clipping the bright region.

---

## Event-rate tower

The same instant of data shown as a **3-D relief** instead of a flat image: the sensor’s x/y
pixels are the ground plane (**labelled with the sensor dimensions**, e.g. 320×320) and each
cell rises to its **event rate (events/second)**. A buzzing rotor or wingbeat region stands up
as a sharp **tower** out of the noise floor — often easier to localise than a colour hot-spot.
Live and seekable on the shared clock, with the corner **orientation cube**.

| control | what it does |
|---|---|
| **Cell** | grid resolution — one tower/bar per cell; smaller = finer relief, slower (use a larger cell for the bars style) |
| **Style** | **surface** (smooth relief) or **bars** (a discrete cubic bin per cell, height = events) |
| **Color** | colormap applied by height |
| **Scale** / **Expr** | the same static/dynamic white-point and dynamic-range expression as the live viewer (so one hot cell doesn’t flatten everything) |
| **Height** | vertical exaggeration of the towers |
| **Background** | scene canvas (**App theme**, the default, follows the light/dark toggle; or a fixed Charcoal / Black / Slate / Navy / Steel / White) |
| **Brightness** | scale the rendered colours up or down |
| **Export ▾** | 3-D snapshot, or the event cube |

---

## Space-time 3D

The event cloud with **time as depth**. Orbit with the left mouse, zoom with the wheel, pan
with the right mouse (or use the corner **orientation cube**). A fluttering target shows up as a
**striped column** — periodic bursts stacked along the time axis — standing out from diffuse
noise. A translucent **front plane** marks the corridor's *"now"* end: in the default trailing
stream the newest events sit on it and history recedes and fades into the distance, flowing
forward as playback advances. It is **seekable and live**, and now also **measurable**.

**Left — render & corridor**

| control | what it does |
|---|---|
| **Max pts** | render budget (events are subsampled to this) |
| **Color** | `polarity` / `time` / `density` — the key updates |
| **Size** | point size |
| **Z-stretch** | stretch the time axis taller so flutter stripes separate |
| **Corridor** | tick to set the shown time-slab depth **here** (a log slider, **5 ms … 60 s**) independently of the program accumulation — squeeze to a few wingbeats or open to a whole flight |
| **Full ∞** | infinite corridor — show the entire stream from the start up to the cursor, deepening continuously during playback (points stay subsampled to the render budget) |
| **Anchor** | where the slab sits around the cursor — **Trailing (stream)** (the default: a stream flowing toward "now", newest at the front, history receding), **Forward** (look ahead), or **Centered** |
| **Time axis** | which axis carries time (**Z / X / Y**) — articulate the corridor |
| **Flip time** | reverse the time direction |
| **Export ▾** | 3-D snapshot, or the space-time event cube |

**Right — measurement deck (tabbed: Box · Spectrum · Measure)**

- **Box tab — Interaction + Analysis box.** Three mode buttons — **Orbit / Edit box / Measure** —
  set what the mouse does, also reachable by **holding Shift** (edit the box from any mode) or
  pressing **M** (toggle Measure); a status line shows the active mode. Position the box by its
  **X / Y / Time** centre with **independent X size, Y size and Time depth** sliders. In **Edit
  box** mode (or holding Shift): **left-drag moves** the box, **drag a yellow corner handle to
  resize** it (the opposite corner stays put), and **Shift+wheel scales** it uniformly.
- **Spectrum tab — frequency in box.** **Analyze box** pulls the events inside and reports the
  dominant frequency by **FFT (binned)**, **NUFFT (non-uniform, no Nyquist ceiling)** or **ISI
  (near-zero-compute inter-event interval)**, with SNR, harmonic comb and a live spectrum; a
  **Band** and **Norm** (whitening) refine the read. **Pop out ⧉** detaches the spectrum into its
  own resizable window (close it or click the banner to re-dock); **Save plot…** writes the
  current spectrum as a figure (PNG + PDF).
- **Measure tab — CAD.** In **Measure** mode, **click points on events** in the cloud (each snaps
  to the nearest event); they are drawn as markers joined by a path. **Two points** read out Δx,
  Δy, distance (px), Δt, the implied **frequency**, **period** and apparent **image-plane speed**.
  **Three or more** points dropped on successive flutter bursts give the **average frequency** with
  interval jitter and total path length. **Cycles (2-pt)** scales the two-point frequency if the
  points span more than one period; **Esc** clears, **⌫** removes the last point.

*Tips:* raise the **Corridor** to ~0.05–0.2 s to see a thicker slab, then **Z-stretch** to make
it tall; drop the box onto a striped column and Analyze to read its exact tone, or hop to Measure
mode and click two adjacent stripes to read the frequency straight off the cloud.

---

## Flutter workbench (the tuning lab)

This is where a detector is **built and tuned** with instant feedback. Three things work
together:

**1. The flicker map** (left image), with a **frequency legend** beneath it. Each cell is
colored by its **dominant flutter frequency** (hue, per the legend) and opacity by **SNR** —
so hot, colored regions are flickering, and the colour encodes how fast. Static structure
stays dark; a dim event image shows underneath for context. The map covers
`[cursor, cursor + Window]`, so **seek** (or play) the shared transport to move through time;
it recomputes on settle, or hit **Recompute flicker map**.

**2. The region spectrum** (lower-left plot). Drag the **orange box** over any region. Its
**live temporal spectrum** updates instantly — the curve, the shaded search band, the marked
peak — and the **readout line** beneath spells out **peak Hz · SNR · harmonic score · event
count** (hover it for what each means). Point the box at a suspected target to read its exact
flutter tone, then set the band around it.

**3. The control deck** (right) is now organised into **tabs** — no long scrolling, every
option visible — each control carrying a hover description:

- **Detect** — the **Detector** (`drone`, `insect`, `mosquito`, `hummingbird`, `bird`, or free
  `flutter`) with its description; the **Analysis window** (length anchored at the cursor, and
  cell size) with **Recompute** and **Export ▾**; **Run** (off-thread); and **Live track while
  seeking/playing** — runs the detector continuously on a short trailing window while scrubbing,
  so a tracker can be **tuned in real time** with the overlay following.
- **Tune** — every detector parameter as a labeled slider (auto-built — see
  `docs/DETECTOR_GUIDE.md`), **Reset to defaults**, and an auto-generated **Suggested settings**
  guide (range, default and meaning of each knob, plus the recommended workflow).
- **Spectrum** — the region-spectrum **Transform** (FFT or **NUFFT**, the non-uniform FFT) and
  **Normalize** (`none` / `median` / `zscore` whitening) to **emphasize peaking** of a faint tone.
- **Targets** — a sortable **table** (ID · Freq · Confidence · SNR · #detections · duration,
  strongest first; hover a header for its definition). Click a row to emphasise that target’s box.

**Export ▾** also writes a **detection report** — a first-principles Markdown + JSON record of
the method, every parameter and its meaning, the assumptions, the results and an interpretation
of *why* — so a detection is fully documented and reproducible.

**The tuning loop:** compute the flicker map → box a candidate and read its spectrum (try
NUFFT + a normalization) → set the band/SNR/harmonic gates → Run (or turn on **Live track**) →
read the targets table and overlay → adjust (or Reset) → re-run. That loop is the whole point of
the workbench.

---

## Sandbox (the algorithm bench)

Where the other tabs are opinionated instruments, the Sandbox is the opposite: a place to work
**directly with the raw event arrays**, **write tracking/filter algorithms that run live**, and
**measure how well they work** — all on the same seekable stream. Left is the image (with live
overlays); right is a control deck of tabs:

- **Build** — assemble the input. **Select** a time **Window** at the cursor, a **Polarity**
  (all / ON / OFF), optionally restrict to the cyan **ROI box**; compose an **Op-chain** of
  primitives applied **in order** (**Hot-pixel remove**, **Refractory**, **Suppress static bg**,
  **Subsample**) watching how many events each keeps; and read the first `(x, y, p, t)` in **Raw
  events**. Whatever survives the op-chain is what the algorithm sees.
- **Track** — the live algorithm lab. Write a `track(ev, state)` function and it runs **on every
  frame while seeking/playing**, overlaying its output (boxes + centroids + fading trails). `ev`
  gives the frame's events plus helpers (`blobs()`, `frame`, `np`, …); `state` is a dict that
  **persists across frames** (the tracker's memory); `MultiTracker` and `cluster_frame` are
  injected. Return
  a detection list (`dict(cx,cy,bbox,id,label,score)`) to track, a bool mask to filter, a
  `dict(x,y,p,t)` to replace, or `None`; append a dict — `return out, {"name": value}` — to emit
  custom live metrics. **Presets** (NN tracker, classifying tracker, blobs, centroid, filter)
  give a working start; **Compile** (Ctrl+Enter), **Run once**, **Reset state**.
- **Efficacy** — *how well it works*, with **the operator's eyes as the ground truth** and these
  proxies to quantify what is seen: a headline **Lock score** (0–1) from **on-target SNR** (is the
  box on a real flutter tone?), **jitter** (box steadiness), **coverage** (frames held), plus the
  **class label + confidence** the tracker emits and any custom metrics. **A/B vs baseline**: hit
  **Set baseline = current** to freeze the current tracker, keep tuning, and watch candidate
  (coloured) vs baseline (grey) overlaid with a baseline/candidate/Δ table — so every edit shows
  whether it helped. A rolling plot traces any metric over time.
- **Performance** — *how fast it runs*: per-frame compute time, max sustainable FPS, throughput,
  and a **real-time** verdict against the live-data budget (the accumulation window's own
  duration), with a history plot.
- **Analyze** — the temporal **spectrum** (FFT / NUFFT / ISI) of exactly what survived, and
  **Export this selection…** to write the manipulated sub-stream (NPZ + CSV) for reuse.

---

## Fusion lab (EBS + acoustic co-registration)

Bring the loaded EBS recording and a **time-synchronized audio `.wav`** onto one clock, then study
them together. Left: the EBS **event-rate** envelope above the audio **RMS** envelope (X-linked,
with a playhead on the shared cursor). Right deck:

- **Load audio (.wav)…** — pick the audio recording of the same scene (PCM 8/16/24/32-bit or
  float; multi-channel is down-mixed). Its sample rate / subtype / duration are shown.
- **Temporal alignment** — **Auto-align (cross-correlate)** recovers the audio→EBS **offset** from
  the two envelopes (de-trended, so the shared closest-point-of-approach swell drives the match);
  the **Offset** spinbox **nudges** it by hand (like a video editor's audio-sync slider)
  and re-reads the overlap and correlation peak live.
- **Export & analyze** — **Export aligned pair** writes the EBS `.raw` + the `.wav` cropped to
  their overlap and re-zeroed to a shared `t = 0` (+ a `.bias` sidecar and a `fusion_manifest.json`).
  **Run fusion study** additionally **detects the rotor in each domain** (harmonic-tracked acoustic
  f₀; in-box EBS rotor flicker on the `single_centroid`-tracked drone), measures **convergence** and
  a **fused P(drone)** (convergence-gated Bayesian + cross-spectral coherence), and writes the
  figures, `ebs_boxes.csv` + `track_dashboard.png` (to **review/prune** spurious track frames),
  summary JSON and report into a folder.

See [FUSION_EBS_ACOUSTIC.md](FUSION_EBS_ACOUSTIC.md) for the method, and the CLI parity:
`gottlux <ebs.raw> --fusion --audio <audio.wav>`.

---

## Timeline (the video editor tab)

A permanent tab laid out like a simple video editor (the toolbar's **`✄ Clip editor`** action and
the Live viewer's Export menu open the same editor as a dialog), top to bottom:

1. **The preview viewport** — the whole timeline rendered at the playhead, live, inside the tab.
2. **The transport** — play / pause / scrub on the timeline's *own* clock, spanning the whole
   program. Spacebar plays and pauses while the tab has focus.
3. **The track lanes** — a **sequence** lane and an **overlay** lane, drawn as blocks whose width
   is their duration and which carry a name, a span, and a small cached midpoint-frame
   **thumbnail**. Click a block to select it (the deck follows), drag it to reorder, double-click
   to edit; click or drag the **ruler** to seek. A playhead line runs across both lanes.

**Everything renders through one path.** The timeline compiles into a *program* in which every
sequential item is a canvas: a plain clip becomes one cell covering the whole canvas carrying that
clip's own visualization settings, a canvas block contributes its multi-cell composition verbatim,
and a title slide becomes a text item — with running titles and overlay clips applied across every
segment. The preview and **Export video…** walk that same program, so scrubbing and exporting can
never disagree.

**Per-clip settings.** With a clip selected, the deck edits its **trim** (In/Out handles or numeric
boxes), its **crop** (a per-clip ROI applied on cut/stitch), its lane, and its own EBS settings —
accumulation mode and window, tone-map + gamma, colormap, time scale, loop. Different clips on one
timeline can therefore carry completely different settings, exactly as mosaic cells do.
**Cut selected → .raw** writes just that clip's trim + crop; a **gap** inserts blank time between
consecutive sequence items (in the preview, the video, and the `.raw` alike).

**Canvas blocks (mosaics), inline.** **Add canvas block…** inserts a whole composition as a *single
sequence item*; dropping more than one file at once offers **as sequence clips** or **as one mosaic
block**. Selecting a block switches the preview viewport into **arrange mode** — the same
draggable/resizable cell stage the Canvas composer uses — where the deck edits the selected *cell's*
EBS settings and geometry. Deselecting, or **Done arranging**, returns to playback. No pop-out
window is involved (the standalone composer under *Tools* still works, on the shared widget).

**Standard sizes.** A **project canvas** preset — Native (the first clip's sensor), 640×640,
1280×720, 1920×1080, 1024×1024, or a custom W×H — sets the preview and export geometry and
rescales every block's arrangement with it. A selected cell takes exact **fractions** of that
canvas (Full, 1/2 horizontal, 1/2 vertical, 1/3, 1/4 quadrant, 2/3), drags snap to a 1/12 grid
(toggleable), and **Auto-tile cells** lays every cell of the block into the best-fit grid.

**Exports.** **Export video…** renders the whole program — segments, overlay lane, titles — frame
by frame to an MP4. **Export .raw…** stays events-only: a plain sequence stitches end-to-end into
one combined, byte-valid EVT2.1 `.raw` whose segments share a single monotonic clock (trim + crop
+ gap, exactly as before), while a timeline containing canvas blocks composites the events into the
canvas geometry instead and writes the composition JSON as its sidecar. Titles and per-clip
visualization settings are render-only, so the event export notes them rather than writing them;
the result converts to HDF5 with `gottlux <out>.raw --to-hdf5` like any other `.raw`.

Loading a recording seeds an empty timeline with that recording as the first clip; a timeline
already being edited is left untouched.

### Export provenance folders

Every export writes a **folder**, not a loose file. Choosing `program.mp4` produces
`program_export_<UTC-stamp>/` beside the chosen location, containing:

| File | Role |
| --- | --- |
| `program.mp4` / `program.raw` | the artifact, under the name that was chosen for it |
| `README.md` | the human-readable provenance document |
| `provenance.json` | the same facts, machine-readable (`schema_version`, `artifact`, `sources[]`, `usage[]`, `settings`, `files[]`) |
| `program.gottlux-canvas.json` | the composition spec, where the export path has one — reload it to re-render |

The README states, in order: **what was produced** (file, kind, size, duration, frame count,
canvas geometry, fps, codec) · **when**, with which GottLUX version and platform · the **source
recordings**, one table row per clip giving its file name, absolute directory, size, SHA-256
(short and full), format, resolution, event count and duration · **how each source was used**
— per clip, whichever of trim in/out, source ROI crop, destination rect on the canvas, time
offset, time scale, accumulation, mode, colormap, tone-map and loop applies to that export path
· the full **export settings** · any **titles/text**, noted as rendering in video only ·
**how to reproduce it** · and **every file in the folder** with its role.

Sources are counted per *distinct recording*: a clip placed twice is one source row and two
usage rows, and clips inside a canvas block count individually — a fifteen-clip timeline lists
all fifteen files. Facts are read as cheaply as possible: a loaded recording answers for itself,
an on-disk source falls back to its decode cache and then to the container header, and no
provenance write ever forces a decode. A source that has moved or become unreadable is recorded
as missing rather than aborting the export.

The completion dialog reports the folder path and opens it in the file browser. View captures
(**Capture view…**) use the same convention: the MP4, the optional poster PNG, and the two
documents — the capture's former `*_manifest.json` is merged into `provenance.json`.

---

## Canvas composer (Tools → Canvas composer…)

A separate composition window: several recordings — possibly from different collects,
sensors, and time bases — placed as draggable, resizable cells on one fixed-size canvas.
A settings deck edits the *selected* cell live: placement, source ROI crop, clock mapping
(canvas-time offset, a time scale for slow-motion/speed-up, looping), and look
(accumulation mode + window, tone-map expression, colormap) — so a real-time wide view
can play beside a 10× slow-motion crop of the same moment. A shared transport plays and
scrubs the canvas timeline. Compositions save/load as a small JSON spec
(`.gottlux-canvas.json`) and export as an **MP4** (rendered frames) or as one composited
EVT2.1 **`.raw`** — the events geometrically remapped into their cells and rescaled onto
the canvas clock, with the spec written into the export folder so the styled view stays
reproducible. (An event file carries events, not rendering: colormap/tone-map/
accumulation apply only at render time, which is exactly what the spec preserves.) Both
exports land in a provenance folder documenting every cell's source recording and
settings — see [Export provenance folders](#export-provenance-folders).
The cell stage is a reusable widget, so the Timeline tab's inline arrange mode is
*literally* this composer's cells — this window remains the place to build a standalone
composition and save/load its spec.

## The Tools menu

Each Tools action acts on the **current view state** — the In/Out selection (or the
cursor's accumulation window) and the live viewer's ROI — so what runs is exactly what is
on screen:

- **Canvas composer…** — the composition window above.
- **Run user script on current view…** — pick a `.py` file defining `process(win, ctx)`
  and it runs on exactly the portion in view; whatever it returns (arrays, a matplotlib
  figure, a derived event stream) is saved into a provenance-stamped run folder. The
  contract and the CLI twin (`--run-script`) are documented in
  [EXTENDING.md](EXTENDING.md) §5.
- **Export tool bundle…** — write a standalone Python + MATLAB tool bundle (`data.h5`,
  scripts, README, `provenance.json`) for the current window/ROI, runnable with no
  gottlux installed on the receiving end. For the `viz_config` tool the live viewer's
  current mode / colormap / tone-map / accumulation are baked in. See
  [EXTENDING.md](EXTENDING.md) §6.

---

## Notes

- **First render after launch** may take ~½ s while the Numba kernels compile (a background
  warm-up hides most of this); everything is fast thereafter.
- The GUI never blocks: decoding and detector runs happen on worker threads.
- For a fixed, **journal-ready** version of any of these views, run the headless pipeline
  (`gottlux <path> --detector …`) — it writes publication PNG+PDF figures and data tables into
  a reproducible run folder. See the README.
