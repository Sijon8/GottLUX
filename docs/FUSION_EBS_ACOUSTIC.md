# Fused EBS + Acoustic — methods

How GottLUX co-registers an event-based-sensor recording with a time-synchronized audio capture of
the same flight, **detects the drone in each domain**, and **fuses** the two. The worked numbers
below come from a real field capture (a small multirotor observed by a GenX320 and a microphone).

Reusable, project-agnostic code:

- `gottlux/io/fusion.py` — audio + Prophesee-HDF5 I/O, envelope cross-correlation, aligned export.
- `gottlux/run/fusion_detect.py` — per-domain detection, confidence, convergence, fusion, coherence.
- `gottlux/run/fusion_study.py` — orchestration + figures + report.
- `gottlux/app/fusionlab.py` — the **Fusion lab** GUI tab. CLI: `gottlux <ebs.raw> --fusion --audio <wav>`.

---

## 1. Temporal alignment

The two sensors run on independent clocks. Each produces an **activity envelope** — the EBS event
rate `r(t)` and the audio RMS `a(τ)` — that swells/fades with range. Both are binned at 10 ms,
smoothed (0.3 s) and **de-trended** (a 2 s moving mean removed), then cross-correlated; the lag of
the peak (constrained to ±`max_lag_s`) is the offset. **Convention:** `offset_s` is added to audio
timestamps to reach the EBS clock. The aligned pair (`.raw` + `.wav`, both cropped to the overlap
and re-zeroed to `t = 0`) is written to `clips/`.

The cross-correlation peak is often modest (the two envelopes are *different* physical signals), so
the offset is best **cross-checked** against an independent value — here the auto estimate
(−4.21 s) reproduces the collaborator's manual ≈ 4.0 s (the Prophesee `time_shift`). A manual nudge
(GUI / CLI `--offset`) overrides it.

> The collaborator's *aligned* EBS was saved as a Prophesee **ECF-codec HDF5 (filter 36559)**,
> unreadable without the Metavision SDK — so GottLUX re-aligns from the original `.raw`.

## 2. Per-domain rotor detection (in the operational band)

A small multirotor's blade-pass fundamental lives in **~380–780 Hz** with harmonics above (the band
searched is 300–1400 Hz). Each domain is reduced to a time-resolved fundamental f₀(t) + a detection
confidence.

### Acoustic — harmonic-aware (`acoustic_track`)
Band-pass (SOS, stable at the high sample rate) → STFT → for each frame, score every candidate
fundamental by the **harmonic sum** Σ_{h=1..H} S(h·f₀) and take the best; continuity-track the
ridge. This is supported by the whole rotor comb, so it does **not** latch onto low-frequency wind/
airframe rumble (the failure mode of a naive single-bin peak, which had wrongly converged ~100 Hz).
Per-frame features: harmonic SNR, supported-harmonic count, spectral flatness (tonality).

### EBS — in-box rotor flicker (`ebs_box_track` → `ebs_rotor_track`)
The blades modulate luminance on the prop disk → events fire periodically at the blade-pass rate **in
the drone's image region**. The *global* event rate buries this in clutter, so the method
**spatially gates**:
track the drone with `single_centroid` (the staring-tier tracker, 85 ms frames) and take the
**in-box** event-rate spectrum (`region_spectrum`) per frame. The EBS observable is the dominant
in-band modulation **peak** — the optical flicker is one strong line, not the rich comb acoustics has
(a harmonic-sum estimator just splits it to a sub-harmonic). Per-frame: peak f₀, in-box SNR,
harmonic score.

### Confidence (each 0–1)
A logistic on the per-frame evidence: `C = σ(a·SNR_dB + b·harmonics + c·tonality − d)`. The
clip-level detection confidence is the 90th percentile of the per-frame confidence (a drone need
only be present for part of the pass).

### Operator pruning
`single_centroid` returns one continuous track; frames where the box jumps to clutter or the target
leaves the FOV are spurious. Review `track_dashboard.png` / `ebs_boxes.csv`, then pass
`drop_time_ranges=[(t0,t1),…]` / `keep_window=(t0,t1)` to drop those frames before the in-box FFT and
the fusion — the **clean-ROI** workflow.

## 3. Convergence — do the two corroborate?

Reported transparently as separate quantities (not one tuned number):

- **temporal overlap** — Jaccard of the "present" frames (do both fire at the same times?);
- **band agreement** — `exp(−(Δlog f / σ)²)` on the *median* fundamentals, harmonic-aware (same band/
  harmonic family?);
- **f₀(t) co-variation** — Pearson r of the two f₀ tracks over co-present frames (do they rise/fall
  together with RPM? — invariant to a constant comb-line offset);

with a per-frame instantaneous κ(t). Headline `κ = temporal_overlap · band_agreement`.

## 4. Fusion / synthesis

- **Convergence-gated Bayesian (#2):** `P_dual = Bayes(Cₐ, Cₑ)`; fused
  `P = P_dual · (floor + (1−floor)·κ)` — a **soft** gate (`floor = 0.4`): a clear independent dual
  detection is not zeroed by an imperfect frequency lock, while genuine convergence lifts it to the
  full Bayesian value. A confident single-sensor hit that the other sensor contradicts is suppressed.
- **Cross-spectral coherence (#3):** magnitude-squared coherence γ²(f) between the in-box EBS
  event-rate and the audio (both resampled to a common rate over the track span). A coherence peak in
  the rotor band is phase-stable shared periodicity — the strongest "both sensors, one rotor"
  evidence, and robust to a *constant* acoustic-propagation delay (only linear phase). It is weak
  when the two domains emphasize different comb lines or the delay drifts with range.

Other options the framework leaves open: a joint harmonic-template matched filter (one fused f₀), a
learned late-fusion classifier (needs labels), and Kalman/vitality temporal integration of the
per-frame fused score.

## 5. Honest characterization

On the field capture, with the EBS track **pruned to the operator-verified drone-in-frame window**,
fusion delivers **strong dual-sensor detection with good frequency convergence**: the acoustic
(≈530 Hz) and EBS-in-box (≈576 Hz) rotor fundamentals agree to ~6% (band agreement ≈0.91), are
temporally coincident (overlap ≈0.92), and fuse to **P(drone) ≈ 0.91**. (On the *raw* automated
track the EBS line was a noisy ≈700 Hz — clean spatial gating is what lets the rotor signature
emerge.) Cross-spectral **coherence** stays the soft metric (γ²≈0.23): a single-frequency phase lock
is smeared by the RPM-drifting tone and the range-varying acoustic-propagation delay, so band +
temporal agreement carry the result. Routes to tighten coherence: harmonic-locked +
range-compensated coherence, and per-motor RPM resolution.

*The field capture behind these numbers is not included in the repository; the method reproduces on
any synchronized EBS + audio pair.*

## 6. Outputs

`clips/` (aligned pair + manifest) · `acoustic_ridge` (spectrogram + tracked f₀) · `f0_timeline`
(both domains) · `fusion_timeline` (confidences + κ + P_gated dashboard) · `coherence` (log-y) ·
`track_dashboard.png` (in-frame EBS box review) + `ebs_boxes.csv`/`.parquet` (review/prune) ·
`fusion_summary.json` · `fusion_report.md` (+ `.tex`/`.docx`). The optional envelope-overlay
alignment figure (`alignment_overlay`, on by default) is omitted when the EBS and audio envelopes
don't visually mirror each other — alignment is still performed and validated against an independent
offset, just not shown as a figure.

## 7. Reproduce

```bash
gottlux <ebs.raw> --fusion --audio <audio.wav> [--offset S] [--freq_lo Hz --freq_hi Hz]
```

Also: `fusion.hdf5_to_raw(h5, out.raw)` converts a *readable* Prophesee HDF5 to `.raw`;
ECF-codec files raise a clear `FusionError` unless an ECF decoding plugin (e.g.
Prophesee's open-source `hdf5_ecf`) is on the HDF5 plugin path.
