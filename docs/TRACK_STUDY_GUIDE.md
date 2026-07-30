# Track-study pipeline — recreate & update any plot

The authoritative how-to for the drone tracking results: what the pipeline is, where each plot
comes from, and how to re-run or change any of it.

## Where it lives (now a first-class part of GottLUX)
- **`gottlux/run/track_study.py`** — the reusable, single-clip pipeline. `run_track_study(...)`
  does: load → run the tracker (detector) → the three KPIs → write the KPI bundle + every tracking
  figure/video. Each figure is its own function (so any single figure can be regenerated).
- **`gottlux/detectors/flutter.py`** — the trackers, incl. **`single_centroid`** (the sandbox
  "Single smoothed centroid" preset: strongest connected-component blob per frame, EMA-smoothed
  centroid, all polarity, no mask) and `blob_tracker` (greedy-NN).
- **`gottlux/run/performance_report.py`** — the KPI engine (range / prop-frequency / TTC) and its
  figures, `detections.csv`, `kpi_report.md`.
- **`gottlux/io/writer.py`** — `cut_clip` / `trim_folder` (the batch-trim with common-origin align).

## Two ways to run it
1. **One clip, from the CLI:**
   `gottlux clips/narrow_20deg.raw --track-study --fov_deg 20 --target_size 0.225 --accum_dt 0.085 --fft-video`
   (writes a `*_track_study/` folder beside the file). Flags: `--detector` (default
   `single_centroid`), `--fov_deg`/`--sensor`, `--target_size`, `--accum_dt`, `--freq_lo`/`--freq_hi`,
   `--t_start`/`--t_stop`, `--out`, `--fft-video`, `--no_open`.
2. **From Python** (regenerate / tweak one figure):
   ```python
   from gottlux.run import track_study as ts
   r = ts.run_track_study("clips/narrow_20deg.raw", fov_deg=20, target_size_m=0.225,
                          out_dir="out", accum_dt=0.085, band=(80, 800), make_fft_video=True)
   ts.lock_score(r["det"], 0.085, "out", "narrow_20deg")          # rebuild just the lock diagram
   ts.track_fft_dashboard(r["rec"], r["det"], 0.085, (80, 800), "out", "narrow_20deg")
   ```

## Which function builds which plot (to update one)
| Output | Function (`gottlux/run/track_study.py`) | Change it by |
|---|---|---|
| `range_vs_time_full` | `range_vs_time_full(det, accum_dt, out, label)` | dropout gap = `accum_dt·1.8` |
| `lock_score` | `lock_score(det, accum_dt, out, label)` | weights in the function (0.5/0.3/0.2) |
| `track_dashboard.png` | `track_dashboard(rec, det, accum_dt, out, label, n=12)` | `n`, `thumb`, `pad` |
| `track_fft_dashboard` | `track_fft_dashboard(rec, det, accum_dt, band, out, label, n=6)` | `n`, `band` |
| `*_overlay.mp4` | `tracked_overlay_video(...)` | box colour, trail length (1.2 s), `scale` |
| `*_track_fft_live.mp4` | `track_fft_video(..., max_frames=200)` | `max_frames`, `band` |
| `tracking_range` / `prop_frequency_range` / `range_vs_time` / `time_to_contact` / `detections.csv` | `gottlux/run/performance_report.py` | `target_size`, `fov`, `snr_thresh`, FFT band |
| `track_timeseries`, `radar` | `extra_figs(...)` → `gottlux/viz/{tracks,panorama}.py` | — |

## Key parameters (what they control)
- **`detector`** — which tracker. `single_centroid` (recommended; one continuous track),
  `blob_tracker` (greedy-NN multi), `drone` (FFT-gated — drops the drone, avoid for tracking).
- **`accum_dt`** — accumulation window (s); the tracker's per-frame integration. 0.085 = the
  sandbox setting. Also the overlay's frame window and playback cadence.
- **`fov_deg` / `target_size_m`** — the pinhole geometry; range `D = L·f_px/N`,
  `f_px = (W/2)/tan(FOV/2)`. Wide 58°, narrow 20°; **L = 0.225 m** (the drone diagonal, because the
  measured apparent size N is the bbox diagonal).
- **`band`** `(freq_lo, freq_hi)` Hz — the rotor search band for the FFT (80–800).
- **FFT cadence** — `SingleCentroidDetector.FFT_EVERY_S` (0.5 s): the FFT is probed sparsely while
  tracking, so spectral plots aren't over-sampled.
- **Spectrum rendering** — both the `track_fft_dashboard` and the live `track_fft_video` draw each
  in-box spectrum through the shared helper `track_study._draw_rotor_spectrum`: linear y
  **normalized to the median noise floor** (floor at 1×, axis in "× noise"), with the near-DC (~0 Hz)
  envelope spike excluded from the display and the y-scale so the rotor band (~70–800 Hz) sets the
  scale. The `prop_frequency_range` SNR-vs-range plot is deliberately kept **log** (SNR spans ~3
  decades). To change this behaviour, edit `_draw_rotor_spectrum` once — both outputs follow.

## The data flow (one clip)
```
.raw → decode-once memmap cache → window/accumulate per 85 ms step
     → single_centroid: strongest blob + EMA centroid → ONE track (t, cx, cy, bbox, range)
     → KPIs (range/prop-freq/TTC, from the per-detection table)
     → figures + videos (this module)   → out_dir/
```
Decode caches go to `%TEMP%` (via `cache_local=True`) so a re-run never collides with a GUI session
that has the same clip open (a real lock gotcha when a clip lives on a cloud-synced folder).

## How this came to be
The pipeline was prototyped as a one-off experiment script with the plot builders inline. Those
builders have since been promoted into `gottlux/run/track_study.py` (this module) and exposed as
`gottlux … --track-study`, so the whole study is reusable and CLI-invocable rather than a one-off.
