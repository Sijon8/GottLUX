# gottlux — Building & Tuning a Flutter Detector

This is the practical guide to the part gottlux is built to be best at: **detecting things
that flutter or flicker** (drones, insects, birds) and **tuning the algorithm** until it locks
on cleanly. The core idea is simple — a target that beats wings or spins rotors modulates
brightness *periodically*, so it produces events in periodic bursts at a characteristic
frequency. gottlux finds that frequency and rejects everything that merely moves.

---

## The fastest path (the workbench)

1. `gottlux-gui` → load a recording → **Flutter workbench** tab.
2. Pick a detector (`drone`, `insect`, …). Its tuning sliders appear automatically.
3. **Compute flicker map** for a short window. Hot, colored regions = something flickering;
   the hue is its frequency. Static background stays dark.
4. Drag the **region box** onto a candidate. Read its **live spectrum** below — peak
   frequency, SNR, and the harmonic comb. This identifies the band to target.
5. Adjust the band/gates (sliders) and **Run detector**. Tracked targets overlay on the image,
   ranked by confidence. Tweak → re-run → compare.

The same on the command line:

```powershell
gottlux capture.raw --detector drone --freq_lo 90 --freq_hi 700 --snr 5
gottlux capture.raw --detector flutter --freq_lo 40 --freq_hi 120   # custom band
```

---

## What each knob does

The parameters are grouped (the GUI mirrors this). Start from a preset and move one at a time.

**Signature** — *what to look for*
| param | effect | raise it / lower it |
|---|---|---|
| `freq_lo` / `freq_hi` | the pass-band searched for the flutter line | tighten around the expected tone to reject neighbours; widen if unsure |
| `snr_thresh` | min in-band peak / noise to accept a blob | **raise** to cut false positives in busy scenes; **lower** for faint/distant targets |
| `harmonic_min` | fraction of overtones required (rotors show a comb) | raise toward 0.3–0.5 to demand rotor-like structure; 0 disables (good for smooth wingbeats) |

**FFT** — *how the frequency is measured*
| param | effect |
|---|---|
| `fft_fs` | sample rate the in-blob stream is binned to; must exceed `2·freq_hi` (auto-enforced) |
| `fft_window_s` | trailing window fed to each blob's FFT; **longer = finer frequency resolution + higher SNR**, but blurs fast maneuvers and needs the target to dwell |
| `fft_min_events` | events required inside a blob to attempt verification; raise to avoid noisy spectra on tiny blobs |

**Cluster** — *finding the blob*
| param | effect |
|---|---|
| `accum_dt` | step size & per-step accumulation window; smaller = sharper motion, noisier |
| `min_pixels` | minimum blob area; raise to ignore speckle, lower for small/distant targets |
| `dilation` / `erode` | bridge gaps / trim spurs before labeling |
| `pos_only` | cluster on ON events only (faster, often cleaner) |
| `suppress_background` | remove persistent-pixel background first (staring scenes) |

**Track** — *linking over time*
| param | effect |
|---|---|
| `max_match_dist` | max centroid jump to associate; raise for fast targets |
| `max_missed` | steps a track may coast before being dropped; raise to bridge intermittent verification (reduces fragmentation) |
| `smooth` | position/box smoothing (0 = raw, →1 = heavy) |
| `max_tracks` | simultaneous targets |

---

## Recipes

**Multirotor drone (sky background).** `drone` preset is a good start: band 80–800 Hz,
`harmonic_min ≈ 0.34`. Against clean sky, *lower* `snr_thresh` (3–4) to reach distant
targets; over clutter, *raise* it (6–10) and keep the harmonic gate on. If the track
fragments as the drone crosses, raise `max_missed` and `max_match_dist`.

**Insect / bee (30–250 Hz).** `insect` preset, `harmonic_min = 0` (wingbeats are closer to
sinusoidal). Small targets → lower `min_pixels` (10–25) and `fft_min_events`. A longer
`fft_window_s` (0.4–0.6 s) sharpens the wingbeat line if the insect hovers.

**Mosquito / midge (300–800 Hz).** `mosquito` preset. Ensure `fft_fs ≥ 1800`. These are tiny
and faint — lower `min_pixels` and `snr_thresh`, and expect to rely on the spectrum more than
the blob.

**Bird (3–30 Hz).** `bird` preset. Slow flapping needs a **long** `fft_window_s` (0.5–1.0 s)
to resolve a few-Hz line, and a larger `min_pixels`.

**Unknown flicker.** Use `flutter` (custom band). First run the **flicker map** wide
(e.g. 10–1000 Hz) to see where energy is, point the region box at it to read the exact tone,
then narrow `freq_lo`/`freq_hi` around it.

---

## Reading the result

`DetectorResult.summary()` lists each target's detection count, duration, median frequency,
and **confidence** (0–1, blending persistence, mean SNR, frequency stability, and harmonic
support). Threshold with `result.confident(0.5)`. The headless run also writes a per-detection
table (Parquet + CSV) with frequency, SNR, harmonic, bearing/elevation/range, plus track and
radar figures.

**Sanity-check on a known signal.** `gottlux.synthetic.synthetic_scene(...)` plants a target
at a chosen frequency; run the tuned detector on it and confirm the recovered
`median_freq` matches. This is exactly what the test suite does (recovery to within a few Hz).

---

## Writing a custom detector

Subclass `Detector`, declare `PARAMS` (each a self-describing `Param`, which becomes a GUI
slider for free), set `regime`, implement `run(rec, cfg, t0, t1) -> DetectorResult`, and
`@register`. The simplest custom detector is just `FlutterDetector` bound to a new
`Signature` — see the presets at the bottom of `gottlux/detectors/flutter.py`. Compose the
core stages (`background`, `detect.cluster_frame`, `frequency.region_spectrum`,
`tracking.MultiTracker`, `geometry`) however the target demands.
