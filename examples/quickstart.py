"""
quickstart.py — a self-contained tour of the gottlux API on synthetic data.

Run it (`python examples/quickstart.py`) with no real captures needed: it plants a 200 Hz
"drone" target in a synthetic scene, then walks through loading, accumulation, the frequency
engine, the flicker map, the tunable detector, and journal-figure export — printing what each
step found and writing a few figures next to this script.

This doubles as living documentation of how the pieces fit together.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")          # headless: write figures, don't pop a window
import numpy as np

import gottlux as eb
from gottlux.core import accumulate as acc, frequency as fq
from gottlux.detectors import get_detector
from gottlux.io import export
from gottlux.synthetic import FlutterTarget, synthetic_scene
from gottlux.viz import frames, spectral, tracks

OUT = os.path.join(os.path.dirname(__file__), "quickstart_out")


def main():
    # 1) a synthetic scene with a known 200 Hz target crossing the frame
    rec, truth = synthetic_scene(
        duration_s=2.0,
        targets=[FlutterTarget(flutter_hz=200.0, x0=40, y0=160, x1=280, y1=160,
                               harmonics=(1.0, 0.5, 0.25))],
        noise_rate_hz=40_000, static_clutter=60, seed=1)
    print("Loaded recording:")
    rec.summary()
    print(f"  (planted target at {truth[0]['flutter_hz']:.0f} Hz)\n")

    # 2) accumulate a frame and look at the event rate
    frame = acc.accumulate_frame(rec.window(0.9, 0.95), mode="count")
    print(f"Frame @0.9-0.95s: {frame.shape}, {int(frame.sum()):,} events\n")

    # 3) the frequency engine: spectrum of the target region + a flicker map
    roi = rec.window(0.5, 1.5, roi=(120, 150, 180, 175))
    sp = fq.region_spectrum(roi.t, fs=2000, fmin=50, fmax=800)
    print(f"Region spectrum: peak {sp.peak_freq:.0f} Hz, SNR {sp.snr:.0f}, "
          f"harmonic {sp.harmonic_score:.2f}")
    fm = fq.flicker_map(rec, fmin=80, fmax=800, fs=2000, cell=6, t0=0.5, t1=1.2)
    valid = np.isfinite(fm.dominant_freq)
    iy, ix = np.unravel_index(np.nanargmax(np.where(valid, fm.snr, -1)), fm.snr.shape)
    print(f"Flicker map: strongest cell at {fm.dominant_freq[iy, ix]:.0f} Hz "
          f"(SNR {fm.snr[iy, ix]:.0f})\n")

    # 4) the tunable detector
    det = get_detector("drone", snr_thresh=4, harmonic_min=0.2)
    res = det.run(rec, eb.Config(mode="staring", sensor="genx320"))   # GenX320 rig: 58° horizontal FOV
    print(res.summary())
    best = max(res.targets, key=lambda t: t.confidence)
    print(f"\n-> best target: {best.median_freq:.1f} Hz "
          f"(planted 200 Hz), confidence {best.confidence:.2f}\n")

    # 5) journal figures + data export
    os.makedirs(OUT, exist_ok=True)
    bg = acc.accumulate_frame(rec.window(0.5, 1.2), mode="count")
    export.save_figure(spectral.flicker_map_figure(fm, background=bg),
                       os.path.join(OUT, "flicker_map"), close=True)
    export.save_figure(spectral.spectrum_figure(sp),
                       os.path.join(OUT, "spectrum"), close=True)
    export.save_figure(frames.detection_overlay_figure(bg, targets=res.targets),
                       os.path.join(OUT, "detections"), close=True)
    export.save_figure(tracks.track_timeseries_figure(res),
                       os.path.join(OUT, "tracks"), close=True)
    print(f"Wrote figures (PNG + PDF) to {OUT}")


if __name__ == "__main__":
    main()
