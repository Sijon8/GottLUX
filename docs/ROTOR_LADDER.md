# The rotor-ladder: detecting a drone with a *spinning* event sensor

*Idea: Simon Gott. Formalized and implemented in `gottlux/rotation/rotor_ladder.py`.*

## The observation

In the rotating-EBS datasets — sensor spinning at ≈ 1 Hz — when the sweep passes across a
multirotor, the rotor leaves a **regularly-spaced stair-step / ladder of event bursts** across the
sweep direction. A building or other static structure gets swept too, but it leaves a *continuous*
streak, not a comb; unstructured ego-motion/background leaves neither. The ladder is therefore a
**telltale of a drone**, the spacing of its rungs is a clean, low-compute measurement, and the
ladder **recurs every revolution with an offset that encodes the drone's own motion**.

## Why it happens (and what the rung spacing means)

The key insight is that **the spin spatially demodulates the rotor's high temporal frequency.** A
rotor blade-pass at 80–800 Hz is hard to sample in time; but while the sensor sweeps across the
drone, *each blade-pass burst lands at a different sensor column than the last*, so the high
temporal frequency is laid out as an easy-to-read spatial comb.

Let the sensor spin at angular rate `Ω` (rad/s) with pixel angular scale `β = FOV/W` (rad/px), and
let the target have its own angular rate `Ω_d`. The target images at sensor column

```
x(t) = const + ((Ω_d − Ω)/β)·t        ⇒    sweep velocity   v = dx/dt = (Ω_d − Ω)/β   [px/s]
```

The rotor emits event bursts at the blade-pass frequency `f`, at times `τ_k = τ_0 + k/f`. Burst *k*
therefore lands at `x_k = const + v·τ_k`, so consecutive rungs are spaced

```
Δx = v / f          (the ladder step, px)
```

Two results fall straight out — they are the entire algorithm:

1. **Blade-pass frequency from geometry alone:**
   ```
   f = |v| / Δx
   ```
   Divide the event-cloud's drift slope (px/s) by the comb spacing (px). The spin has converted a
   200 Hz temporal problem into a ~9 px spatial one. `v` comes from telemetry (`Ω`) or is measured
   from the events directly — and if measured, `Ω_d` cancels and `f` needs no telemetry at all.

2. **Relative motion from the slope / the per-revolution offset:**
   ```
   Ω_d = Ω − β·v          and across revolutions:   Ω_d ≈ ΔΘ / T_rot
   ```
   A *stationary* drone repeats an identical ladder each revolution; a *moving* one shifts the
   whole ladder by a fixed offset per revolution. So the offset between successive rotations is the
   drone's relative motion — exactly the original intuition.

## Why it discriminates

| object | swept drift? | high-f burst comb? | verdict |
|---|---|---|---|
| **multirotor** | yes | **yes** (rungs at `Δx = v/f`, `f` in 80–800 Hz) | detected |
| building / static edge | yes | no (continuous streak) | rejected |
| ego-motion / noise | no coherent drift | no | rejected |

A static high-contrast edge has *no* high-RPM signature, so it can't fake the comb; and the comb's
implied `f` must land in the rotor band, which background structure does not.

## The algorithm (cheap by construction)

Per candidate window of events `(x, t)`:

1. **Drift** `v` — from telemetry (`Ω/β`) or a robust line fit of `x` vs `t`.
2. **Comb** — histogram the events along the sweep coordinate, take **one autocorrelation**, and
   pick the rung spacing `Δx` by **harmonic comb energy** (a true ladder has autocorrelation peaks
   at `Δx, 2Δx, 3Δx`; a spurious bump does not). Constrain `Δx` to the band `|v|/f_hi … |v|/f_lo`.
3. **Report** `f = |v|/Δx`, the comb strength, and `detected = (f in band) and (comb strong)`.
4. **Across revolutions** (`track_ladders`): a steady `f` over ≥ 2–3 revolutions confirms a real
   object; the per-revolution azimuth offset gives the relative motion `Ω_d`.

Cost is one line fit + one autocorrelation of a small 1-D histogram per candidate — no per-pixel
FFT, no fine temporal sampling. It is a natural fit for the low-SWaP edge target.

## Validated behaviour (synthetic)

`synthetic_rotor_pass` plants a swept rotor pass; `test_rotor_ladder.py` confirms:

- **`f` recovered exactly** from the geometry in the resolvable regime — e.g. a planted 200 Hz at
  `v = −1800 px/s` gives `Δx = 9.0 px → f = 200.0 Hz`; 150 Hz → 150 Hz.
- **Edges and noise rejected** (comb strength < 0.1 vs > 0.26 for a clean drone).
- **Recurrence + motion**: a ladder drifting 8 px/revolution is tracked with a stable `f` and the
  per-revolution offset recovered.

## Regime and limits (honest)

The spatial comb is **crisp when the rung spacing exceeds the rotor-disk image size** (`Δx ≳ disk`)
— i.e. farther / smaller drones, or faster spin. For a near, large disk the rungs overlap and the
comb softens into a ripple (still found by the autocorrelation, but weaker); there the
**cross-revolution accumulation** (combs add coherently for a stationary target) and a temporal
cross-check restore it. The current detector flags this honestly via the comb-strength score.

## Use it

```python
from gottlux.rotation import rotor_ladder as rl
res = rl.ladder_signature(x_px, t_s, sweep_px_s=v_from_telemetry)   # one candidate pass
# res.blade_hz, res.step_px, res.comb_strength, res.detected
```

In the GUI it is a live readout on a drawn region (Space-time box).

## The 360° survey — classify once, then map the whole sky

`gottlux/rotation/rotor_scan.py` lifts the single-pass primitive into a full-rotation product, and
it is wired as a first-class headless analysis:

```
gottlux rotating.raw --analyses rotor_ladder --blades 2 --target_size 0.225 \
        [--roi x0,y0,x1,y1 --t_start T0 --t_stop T1]
```

1. **Classify the box** (`analyze_box`) — measure the ladder on the analysis box (the `--roi` +
   time window, or the GUI box) and quantify the **propeller**: blade-pass `f` → rotor rate
   `f_rot = f / N_blades` → RPM, plus tip speed `π·D·f_rot` and its Mach number, and the target's
   bearing and pinhole range. This is the *template*.
2. **Scan the 360°** (`scan_rotation`) — de-rotate every (background-suppressed) event to world
   azimuth, bin by `(revolution, azimuth)`, and run the same cheap comb test per cell. Cells whose
   implied `f` lands within tolerance of the template are flagged as the same rotor — a map of
   *where else* the signature occurs. If no box is given, the strongest detected cell auto-seeds
   the template. **The sweep rate is taken from telemetry** (`|v| = Ω·W/FOV`), so `f = |v|/Δx` is
   robust to the residual noise that would bias an event-cloud slope fit.
3. **Link across revolutions** (`link_tracks`) — group a target's detections into a track; the
   bearing-vs-revolution slope is the **per-revolution azimuth offset = relative motion**
   (`Ω_d = offset / T_rot`). A stationary rotor repeats at one bearing; a moving one marches.

Outputs (run folder `rotor_ladder/`): the **360° survey** map (blade-Hz vs bearing), a
**target-acquisition radar** (θ = bearing, r = range, colour = blade-Hz), a **recurrence/offset**
plot, the template ladder figure, a JSON + detections/tracks CSV, and a compilable **LaTeX report**.

## Verify before spending effort

The test suite plants a synthetic *rotating* scene with a fully-known drone (blade frequency,
bearing, range, and a known per-revolution drift) plus a static edge and noise, and runs the whole
survey against that ground truth. On the default plant it recovers the blade frequency and RPM
to ≈5 %, the motion offset essentially exactly (e.g. planted 8.0°/rev → 8.0°/rev recovered), and
links all passes into one track; range is a coarser ±20–30 % kinematic estimate. Tests live in
`tests/test_rotor_scan.py` (the survey) and `tests/test_rotor_ladder.py` (the primitive).

Next steps: cross-revolution coherent accumulation for the overlap regime, and fusing the ladder
`f` with the staring drone-FFT for a single confident rotor estimate.
