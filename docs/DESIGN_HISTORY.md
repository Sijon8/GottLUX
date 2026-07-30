# gottlux — Design & Development History

This document is the *why* behind gottlux: what it is, why it matters, the principles that
shaped it, the major design decisions (and the alternatives rejected), and how it evolved
release to release. For *how to use it* see the [README](../README.md) and
[`GUI_GUIDE`](GUI_GUIDE.md); for *how it works internally* see
[`ARCHITECTURE`](ARCHITECTURE.md) and [`ALGORITHMS`](ALGORITHMS.md); for *building detectors*
see [`DETECTOR_GUIDE`](DETECTOR_GUIDE.md).

---

## 1 · What gottlux is (high level)

**gottlux is a single, contiguous Python instrument for getting science out of event-based
(neuromorphic) sensor recordings** — from raw encoded events all the way to calibrated,
journal-ready measurements and figures, with a particular strength in **detecting fluttering /
flickering targets** (drones, insects, birds) by their temporal-frequency signature.

An event-based sensor (EBS) does not produce frames. Each pixel independently fires an
asynchronous *event* `(x, y, polarity, t)` the instant its log-intensity changes by a
threshold, at microsecond time resolution and very high dynamic range. That makes EBS superb
for high-speed, low-light, sparse-motion sensing — and badly served by frame-based tools,
which immediately throw away the very thing that makes the data special: **precise event
timing**.

gottlux is built around that timing. Its organising insight is that **a rotor or a wingbeat
modulates brightness periodically**, so its events arrive in periodic bursts at a
characteristic frequency. Everything that distinguishes a drone from a bird, or an insect from
wind-blown clutter, lives in that temporal signature — and gottlux is a laboratory for
measuring, visualizing, and detecting it.

## 2 · Why it matters

* **The vendor/SDK tools render; they do not analyze.** They show a flickering picture.
  They offer no way to *measure* the flicker, *tune* a detector against it, or produce a
  reproducible, publication-grade result. gottlux is the missing analysis layer.
* **Temporal frequency is an underused discriminant.** Spatial motion detectors fire on
  anything that moves. A *flutter* detector fires only on things that move *periodically in the
  right band* — a fundamentally stronger discriminant for small aerial targets, and one that
  EBS is uniquely able to feed (its µs timing resolves blade-pass tones a camera cannot).
* **Reproducibility.** Detection claims need provenance: the exact method, data, parameters,
  and assumptions behind a number. gottlux treats that as a first-class output, not an
  afterthought (run manifests, source snapshots, and first-principles detection reports).

## 3 · Design principles (the rules everything obeys)

1. **One data model.** Everything — accumulation, background suppression, the frequency engine,
   detectors, figures, the GUI — operates on a single
   [`Recording`](../gottlux/io/recording.py) (memmap-backed `x, y, p, t` + metadata + optional
   rotation telemetry) and its light `EventWindow` slices. One vocabulary for "a chunk of EBS
   data" means parts compose without glue code.
2. **Decode once; bounded memory.** Multi-gigabyte `.raw` files stream chunk-by-chunk straight
   to an on-disk memmap cache; re-opening is instant and RAM stays flat regardless of file
   size. Windows are zero-copy NumPy slices.
3. **Vectorized first, JIT where it pays, always a fallback.** Hot loops are NumPy-vectorized
   (`bincount`-based accumulation, one batched FFT for the whole flicker map); the few genuine
   per-event loops are Numba-JIT compiled — but every kernel has a pure-NumPy fallback so the
   suite *always* runs, even with no Numba and no GPU. (Target hardware has no CUDA.)
4. **Light import, lazy heavy deps.** `import gottlux` pulls in only NumPy-level code; Qt,
   pyqtgraph, and matplotlib are imported lazily, only when a GUI or figure is actually used.
5. **Self-describing, composable detectors.** A detector publishes its tunable parameters as
   `Param` objects, and the GUI *auto-builds* its tuning panel — no per-detector UI. Writing a
   new detector gets a full interactive tuner, CLI flag, and report for free.
6. **Reproducible, journal-ready output by default.** Figures are saved as raster **and** vector
   at 300 DPI; data as Parquet/CSV/NPZ/HDF5; runs as folders with a manifest, input SHA-256,
   environment capture, and a source snapshot.
7. **Python-only.** One language, one toolchain, one install — MATLAB users are served by
   the standalone `--export-tool` bridge instead of an in-process bridge.

## 4 · Architecture at a glance

```
io/        decode (EVT2.1/2.0/3) · streaming memmap cache · Recording/EventWindow · telemetry · export
core/      accumulate · tonemap · filters · background · frequency engine · detect (blobs) · geometry · metrics
detectors/ Param/Detector/Target framework · composable FlutterDetector · signature library · tracking
viz/       publication theme · figure builders (frames, spectral, panorama, tracks)
run/       headless pipeline · provenance · detection report
app/       PySide6 GUI: shared clock/transport · live viewer · event-rate tower · space-time 3D ·
           flutter workbench · sandbox · legends · export dialog
```

The decisive seam is **`core/frequency.py`** — the temporal-frequency engine. Four lenses on the
same question ("where, and at what frequency, is the scene flickering?"): the binned
`region_spectrum` (with harmonic-comb scoring), the non-uniform `nudft`/`nufft_spectrum`, the
`lomb_scargle` periodogram, the near-zero-compute `isi_frequency`, and the 2-D `flicker_map`.
Detectors are thin orchestration over this engine plus spatial clustering and tracking.

## 5 · Key design decisions (and the roads not taken)

**Decode-once memmap cache, not in-RAM decode.** A 29 M-event `.raw` is multiple GB decoded.
Holding it in RAM caps file size to memory; re-decoding every open is slow. A streaming decode
to an on-disk memmap caps *RAM* instead of *file size* and makes re-opens instant. Cost: a
one-time decode and disk for the cache — accepted.

**Flutter-verify as a gate inside the detector, not a post-filter.** A blob is accepted only if
the events *inside it* carry an in-band periodic signature above an SNR gate (optionally with a
harmonic comb). This is what turns a motion detector into a *flutter* detector. Doing it inline
(rather than detecting motion then filtering tracks) means the frequency evidence is available
at association time and the diagnostics honestly report candidates-vs-verified.

**One batched FFT for the flicker map.** The map bins the sensor into cells, builds every cell's
time series in one `bincount`, and FFTs all cells at once, auto-coarsening the sample rate to
stay within a memory budget. The naive per-cell loop was ~30× slower. A whole-recording variant
(`flicker_map_max`) tiles short windows and keeps each cell's best, avoiding the smear a single
long FFT causes for a moving target.

**Self-describing `Param`s drive the GUI.** The alternative — hand-written tuning UI per
detector — does not scale and rots. Making parameters self-describing (range, step, unit, help,
group) means the tuner, the CLI, the suggestions panel, and the report all read from one source
of truth.

**A shared `TimeController` clock for every view.** Each visualization could own its own time
state, but then "seek a moment" and "set the accumulation time" would be per-widget and
inconsistent. One clock that every transport bar binds to makes seeking and accumulation
*program-wide* — the property the instrument is built around.

**Tone-mapping separated from colormap (v1.3).** EBS frames have brutal dynamic range; a linear
map dilutes faint targets to black while a hot rotor disk clips to white. Rather than bake a fix
into one view, dynamic-range compression (`core/tonemap.py`: log/√/γ/asinh/equalize/percentile)
and the static-vs-dynamic white-point are a reusable core module the viewer, the tower, and the
sandbox all share.

**Non-uniform FFT alongside the binned FFT (v1.3).** Binning imposes a sample rate and a Nyquist
ceiling and smears very sparse streams. A direct non-uniform DFT evaluates the transform exactly
at a chosen frequency grid straight from event times — no binning, no Nyquist tie — at
`O(n_freq·n_events)` cost, with dense streams subsampled (random subsampling preserves a
periodic point-process's period). It is offered as an option, not a default, because the binned
FFT is far cheaper when the data is dense and regular.

## 6 · How it evolved

GottLUX grew feature-by-feature against real captures: the decode-once cache and the
frequency engine came first; the shared program clock made every view seekable; the
dynamic-range/tone-mapping core, the event-rate tower, and the measurable space-time
corridor made faint flutter visible; the self-describing detector framework turned tuning
into an interactive loop (live tracking, the sandbox algorithm lab); and the provenance
layer made every result reproducible by construction. The granular release-by-release
record lives in the [`CHANGELOG`](../CHANGELOG.md).

## 7 · Deeper definitions (a glossary of the load-bearing terms)

* **Event / CD event** — a single `(x, y, polarity, t)` change report. *Polarity* 1 = brighter
  (ON), 0 = darker (OFF). *t* is microseconds, monotonic, zero-based to the recording.
* **EVT2.1 / EVT2.0 / EVT3** — Prophesee binary encodings of the event stream (GenX320, IMX636
  sensors). The decoder turns any of them into the common data model.
* **Recording / EventWindow** — the one data model and a zero-copy time/ROI slice of it.
* **Accumulation** — summing a time window of events onto the sensor grid to form a frame:
  `count`, `polarity` (ON−OFF), `on`/`off`, `time_surface` (exponentially-decayed last-event
  time — sharp motion), `binary`.
* **Tone-map expression / scale** — the monotone curve applied before the colormap to compress
  dynamic range, and where the white-point comes from (per-frame *dynamic* vs frozen *static*).
* **Flicker map** — a 2-D image of the sensor where each cell's colour is its dominant in-band
  flutter frequency and its opacity is its SNR: *where the scene flickers, and how fast*.
* **Region spectrum** — the temporal power spectrum of a region's binned event stream, with the
  in-band peak, its **SNR** (peak ÷ robust noise floor), and a **harmonic-comb score** (do the
  fundamental's overtones light up — a rotor discriminant).
* **NUFFT / non-uniform FFT** — the transform evaluated directly at chosen frequencies from raw
  event times, with no binning or Nyquist ceiling.
* **ISI (inter-event interval) frequency** — a near-zero-compute periodicity estimate from the
  distribution of successive event-time gaps; best per-pixel/tiny-region, a first look to confirm
  with a real spectrum.
* **Detector / Target / DetectorResult** — a tunable, registered method; a tracked target
  carrying its kinematics *and* flutter signature; and the run output with full provenance.
* **Confidence** — a 0–1 score blending track persistence, mean SNR, frequency stability, and
  harmonic support, so no single lucky window scores high.
* **Signature** — a named frequency band + expectations (e.g. harmonics) that seeds a detector's
  defaults for a target class (drone, insect, …).
* **Staring vs rotation regime** — a fixed sensor vs a panning/rotating payload; the latter uses
  rotation **telemetry** to de-rotate to a world frame and to phase-filter the swept static
  background.
* **Corridor** — the depth (in time) of the slab shown in the 3-D view; in v1.3 it can be set
  independently of the program accumulation, 5 ms to 5 s.
* **Provenance / run folder / detection report** — the reproducibility outputs: a manifest +
  input hash + environment + source snapshot per run, and a first-principles record of a
  detection's method, settings, assumptions, and interpretation.
