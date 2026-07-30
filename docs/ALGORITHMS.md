# gottlux — Algorithms

The math behind each stage, from raw bytes to calibrated measurements. Cross-references point
at the implementing module.

---

## 1. Decoding (`io/decode.py`)

A Prophesee `.raw` is a stream of fixed-width words in one of three incompatible layouts.
gottlux decodes all three with pure NumPy (no Metavision SDK):

* **EVT2.1** (64-bit, GenX320/X320). `type = bits[63:60]`. A CD word carries a base
  `(x, y)`, a 6-bit low timestamp, and a **32-bit column mask** — one word expands to up to
  32 pixel events at `x_base+k`. `TIME_HIGH` (type `0x8`) carries a 28-bit high timestamp in
  units of 64 µs, so `t_µs = ts_high·64 + ts_low`.
* **EVT2.0** (32-bit). Single CD events, same TIME_HIGH scheme.
* **EVT3** (16-bit, IMX636/Gen4). **Stateful**: events reference an implicit current `y`-row
  (`ADDR_Y`), a vector base-x (`VECT_BASE_X` + `VECT_12`/`VECT_8` bitmasks), and a 12-bit-high
  + 12-bit-low timestamp whose high field **wraps** (an epoch counter tracks the wraps).

The decoders are written as **stateless chunk functions** carrying an explicit state dict
across chunk boundaries, which is what lets `io/cache.py` stream a multi-GB file straight to a
memmapped cache in bounded RAM. The bit layouts were validated **bit-exact** against the
ECF-HDF5 oracle (expanded event counts and time spans match to the event), and the
encode↔decode roundtrip is unit-tested.

Robustness: events before the first `TIME_HIGH` are dropped (uninitialized timestamps); a
tiny pre-roll cluster isolated by a multi-second gap is trimmed (corrupt lead-in); timestamps
are forced non-decreasing against glitch jumps.

---

## 2. Accumulation (`core/accumulate.py`)

An event window becomes a `(H, W)` frame. `count`/`on`/`off`/`polarity`/`binary` are single
`numpy.bincount` calls over the flattened pixel index. The **time surface** (SAE) is the one
loop worth JIT-compiling: the most-recent event time per pixel, then an exponential decay
`exp((t_pixel − t_now)/τ)` — a sharp, motion-emphasizing image used by the live viewer and as
a clustering substrate.

### 2.1 Tone-mapping (`core/tonemap.py`)
A frame's value range is brutal: a hot rotor disk can fire 100× a faint target, so a linear
colour map dilutes the faint target to black. Two orthogonal controls fix it. A **white-point**
(value mapped to full colour) is the `clip_pct` percentile of positive pixels — *dynamic*
(recomputed per frame) or *static* (frozen so frames stay comparable). A **tone curve** then
compresses the normalized `[0,1]` frame: `sqrt`, `gamma` (`x^γ`), `log` (`log(1+99x)/log(100)`,
~2 decades), `asinh` (linear near zero, log-like bright), `equalize` (CDF of non-zero pixels),
or `percentile` (linear, hard-clipped). A signed variant keeps a diverging polarity map centred
at zero. All are one cheap NumPy pass and are shared by the live viewer, the event-rate tower,
and the sandbox.

---

## 3. The frequency engine (`core/frequency.py`)

A spinning rotor or beating wing modulates scene brightness **periodically**, so its events
arrive in periodic bursts at a characteristic frequency (plus harmonics). Everything that
tells a drone or insect apart from something merely moving lives in that temporal signature.

### 3.1 Region spectrum
Bin a region's event **times** to a regular series at `fs` (an event-count signal), detrend,
Hann-window, take `rfft`, and find the strongest line in the pass-band `[fmin, fmax]`.

* **SNR** = in-band peak power / median noise floor (dimensionless, comparable across regions).
* **Harmonic-comb score** = the fraction of the fundamental's overtones (`2f₀, 3f₀, …`) that
  show a local power bump. A rotor's periodic-but-non-sinusoidal modulation lights up this
  comb; broadband motion does not — a strong, cheap discriminator.

`fs` must exceed `2·fmax` (Nyquist); gottlux enforces a comfortable margin automatically.

**Spectral normalization (`normalize=` / `whiten_power`).** EBS noise is not white — the
spectrum slopes — so a real line can hide under low-frequency haze. Optional whitening divides
each bin by a sliding-median noise floor (`"median"`) or standardizes it against a local
mean/std (`"zscore"`, "sigmas above noise") **before** the peak pick, so a faint tone or comb
stands up sharply. Default `"none"` preserves raw power.

### 3.1b Non-uniform FFT (`nudft` / `nufft_spectrum`)
A direct evaluation of `|Σₖ exp(-2πi f tₖ)|² / N²` on a chosen frequency grid, straight from the
event **times** — **no binning and no sample-rate Nyquist ceiling**: the caller picks the exact
frequencies to probe (a fine grid zoomed onto a suspected tone, or frequencies above what a
modest `fs` allows). The events are treated as unit impulses (a point process); the mean is *not* subtracted
from uniform weights (doing so zeroes the transform — only genuine non-uniform weights are
centred). Cost is `O(n_freq · n_events)`, evaluated in event-chunks to bound memory; dense
streams are subsampled. Returns the same `Spectrum` type as `region_spectrum`, so the harmonic
comb, SNR and plotting all carry over.

### 3.1c Inter-event-interval frequency (`isi_frequency`)
A near-zero-compute periodicity lens: sort the times, take successive gaps, map each to a rate
`1/Δt`, and find the dominant rate from a log-spaced histogram, returning `(freq, concentration)`.
A source that fires once per cycle (a single pixel under a rotor/wingbeat) concentrates its
intervals into a sharp mode. Best per-pixel / tiny-region — a fast first look to confirm with a
real spectrum (for a busy multi-pixel region the within-burst intervals can dominate).

### 3.1d Two-point measurement (`measure_between`)
Given two picked space-time points `(x, y, t)` — e.g. one flutter stripe and the next in the 3-D
cloud — return the time gap, the implied frequency `cycles/Δt`, the period, the in-image
separation and the apparent image-plane speed. The arithmetic is trivial; the value is reading a
wingbeat straight off the cloud.

### 3.2 Lomb–Scargle
For sparse / unevenly-sampled streams, a Lomb–Scargle periodogram works directly on the event
**times** without binning. Cost is `O(n_freq · n_events)`, so dense streams are uniformly
subsampled first (random subsampling of a periodic point process preserves its period).

### 3.3 Spectrogram
An STFT of the binned event signal — frequency vs time — to watch a signature evolve (a drone
spinning up, a maneuver, a wing-loading change).

### 3.4 The flicker map (the showpiece)
A 2-D map over the sensor of *where it flickers and how fast*, in one vectorized pass:

1. Bin the sensor into `cell × cell` spatial cells and time into `1/fs` bins.
2. With a single `bincount` on the combined `(cell, time-bin)` index, build the
   `(n_cells, n_time-bins)` event-count matrix.
3. FFT **every cell at once** (`rfft` along the time axis), find each cell's in-band peak
   frequency and SNR.

The result is an image: hue = dominant frequency, opacity = SNR. A memory budget auto-coarsens
the time sampling for very long windows. `flicker_map_max` tiles short windows over a whole
recording and keeps each cell's strongest hit — fast, and correct for a target that moves
between cells (a single long FFT would smear it).

---

## 4. Denoising (`core/filters.py`) and background (`core/background.py`)

* **Hot-pixel**: drop pixels in the top firing-rate percentile (stuck/defective).
* **Refractory**: drop per-pixel events arriving sooner than a refractory period (chatter).
* **Rotation-phase anomaly** (rotation only): on a spinning sensor a static scene point is
  swept past a pixel once per revolution, always at the **same** rotation phase. Per pixel the
  filter accumulates the circular concentration `R` of event phases and the revolution recurrence
  (both vectorized, O(N)). A pixel is *locked background* if it recurs across many revolutions
  with high `R`; an event is kept (anomalous/moving) if its pixel is not locked, or its phase
  deviates from the pixel mean beyond a tolerance.
* **Frozen rotation reference**: learn the static rotating scene once from the first N
  revolutions in `(phase, y, x)` voxels and **never update it**, so a target entering later is
  never eroded (the failure mode of a cumulative running model).
* **Staring persistent-pixel**: on a fixed sensor, pixels with a high baseline rate over a
  learning window are flicker/clutter and are suppressed.

---

## 5. Geometry & ranging (`core/geometry.py`)

The pinhole model with focal length `f_px = (sensor_px/2) / tan(FOV/2)`:

* **Bearing** (staring): `az_sign · (x − W/2) · (FOV/W)`.
* **Elevation**: about the **height** centre, `(H/2 − y) · (FOV/W)` — correct on non-square
  sensors (a target at the vertical centre reads exactly 0°).
* **World azimuth** (rotation): `azimuth(t) + intra-FOV bearing`, where `azimuth(t)` comes
  from telemetry. A fixed target's world bearing is **invariant across a sweep** — as the FOV
  pans, `azimuth(t)` rises while `(x−W/2)` falls and they cancel.
* **Range**: `D = L · f_px / s_px` from a known physical size `L` and apparent size `s_px`.
  When `L` is unknown, use the unitless **relative-distance proxy** `1/s_px` (monotonic with
  range) and calibrate it to metres per flight from a known near/far.

`FOV` here is the **horizontal** field of view (it spans the sensor *width* `W`). It comes from
the active **sensor profile** (`gottlux/sensors.py`, `Config.resolved_fov()`): the default
GenX320 + 1.8 mm rig is **58°** horizontal (its 76° figure is the *diagonal*). Override it per
clip with `--fov_deg` / `Config(fov_deg=…)`, or select another rig with `--sensor`.

---

## 5b. Results metrics / KPIs (`core/performance.py`)

The three operator-facing results, each an **independent** function of the optics + a measured
series, so a weak one never invalidates the others:

* **Tracking range.** A blob is trackable while pixels-on-target `N = L·f_px/D` stays above the
  detector's minimum-size gate. With min connected-component area `A_min`, the trackable linear
  threshold is `√A_min`, so the capability range is `D_track = L·f_px / √A_min`. The Johnson
  detection/recognition/identification ladder (§4 of `photogrammetry`) is reported for context,
  and the **measured reach** is the farthest range a detection actually held (which also yields an
  *effective* trackable-pixel count, calibrating the gate to observed performance).
* **Prop-frequency-resolution range.** The rotor blade-pass tone is resolvable while the in-band
  FFT SNR clears the gate. The number of coherently-modulated events scales with the rotor disk's
  image **area** `∝ N² ∝ 1/D²`, so `SNR(D) ≈ SNR_ref·(D_ref/D)²`. Fitting `log SNR` vs `log D`
  (slope ≈ −2 expected, reported with R²) and solving for `SNR = gate` gives `D_freq`. With no
  measured tone to calibrate, it falls back to a pixels-on-target threshold (the disk must be
  spatially resolved for its flicker to be readable). This is a *temporal* perception limit, distinct from the
  spatial tracking range.
* **Time-to-contact.** Nominal warning `TTC = D_detect / V_approach` (a sweep over approach speeds
  gives the curve); measured `TTC(t) = range(t) / closing-speed`, the closing speed being the
  robust slope of `−range` vs time for an approaching track.

Each metric carries a status (`ok` / `model_only` / `no_data` / `failed`); the bundle is written
by `run/performance_report.py` into a folder beside the analyzed file, one artifact per guarded
write. Regime is auto-detected — a rotating capture adds the swept-coverage figures of merit (§7),
a staring capture the radial velocity / blade FFT.

---

## 5c. Dual-view co-registration (`core/dualview.py`)

To superimpose two co-observing views (a wide acquisition FOV and a narrow precision FOV), a point
is mapped through the shared bearing: pixel → angle in the source view, angle → pixel in the
target view. With per-pixel scales `dpp = FOV/W`:

```
α  = (x_src − W_src/2)·dpp_src + offset      x_dst = W_dst/2 + α / dpp_dst
```

so the mapping is a pure angular ratio (the narrow view is the central crop of the wide view),
exact regardless of which is wider. **Parallax** mode adds the stereo disparity from the rig
baseline, `Δx = b·f_px / D` — only ~0.7 px at a 25 mm baseline and 10 m, vanishing with range, so
`fov_scale` is usually enough. A **converged** study pools ranged keyframes from both clips into
one target-size fit `L = Σ N·(f/D) / Σ (f/D)²` (each keyframe weighted by its own clip's focal
length), then reports each clip's Johnson perception ranges for the fused `L`.

---

## 5d. Rotor-ladder — drone detection on a *spinning* sensor (`rotation/rotor_ladder.py`)

When the sensor spins and sweeps across a multirotor, the spin **spatially demodulates** the rotor:
each blade-pass burst lands at a different sensor column, drawing a regularly-spaced stair-step. For
spin rate `Ω`, pixel scale `β = FOV/W`, target rate `Ω_d`, the target drifts at `v = (Ω_d − Ω)/β`
px/s and the rungs are spaced `Δx = v/f` for blade-pass frequency `f` — so

```
f = |v| / Δx            (blade-pass from geometry)
Ω_d = Ω − β·v           (relative motion; also the per-revolution ladder offset / T_rot)
```

Detection is one robust line fit (the drift `v`) + one autocorrelation of the sweep-coordinate
histogram, with the rung spacing chosen by **harmonic comb energy** (peaks at `Δx, 2Δx, 3Δx`). A
static edge has the drift but no comb; noise has neither. The full derivation, regime
(`Δx ≳ disk`), and validation are in [`ROTOR_LADDER.md`](ROTOR_LADDER.md).

---

## 6. Detection pipeline (`detectors/flutter.py`)

```
foreground  →  cluster  →  FFT flutter-verify  →  track  →  localize
```

1. **Foreground**: hot-pixel + (staring) persistent-background suppression; optionally ON-only.
2. **Cluster** (per `accum_dt` step): rasterize → morphological close → connected components →
   area gate → candidate blobs.
3. **Verify** (the decisive stage): for each blob, take the **all-polarity** events inside its
   box over a trailing `fft_window_s`, compute the region spectrum, and accept only if an
   in-band peak beats the SNR gate **and** (optionally) the harmonic gate. This is what makes
   it a *flutter* detector, not a motion detector — everything that merely moves is rejected.
4. **Track**: greedy nearest-neighbour association with velocity prediction and missed-frame
   coasting links verified detections into `Target`s.
5. **Localize**: bearing / elevation / range / relative-distance per detection via §5.

A `Target` carries its kinematics **and** its flutter signature (per-detection frequency, SNR,
harmonic score), and a 0–1 **confidence** blending persistence, mean SNR, frequency stability,
and harmonic support — one number to rank/threshold by.

---

## 7. Figures of merit (`core/metrics.py`)

* **Coverage** — swept solid angle `Ω = Δaz · (sin e_hi − sin e_lo)` (steradians), sphere
  fraction, gain over a static sensor, revisit interval / update rate (rotation).
* **Localization** — bearing standard error, elevation spread, range statistics from the
  detector's track table.
