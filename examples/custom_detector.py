"""
custom_detector.py — a complete, runnable example of extending gottlux with your own detector.

This file is BOTH:

1. **A plugin.** Point ``GOTTLUX_PLUGINS`` at it and the ``blink`` detector below is
   registered at startup — it appears in ``gottlux --list_detectors``, runs headless with
   ``gottlux clip.raw --detector blink``, and the Flutter workbench auto-builds a tuning
   panel for it from its ``PARAMS`` list. No forking, no editing the package.

       # Windows                                        # Linux/macOS
       set GOTTLUX_PLUGINS=%CD%\\examples\\custom_detector.py
       gottlux --list_detectors                         export GOTTLUX_PLUGINS=$PWD/examples/...
                                                        gottlux --list_detectors

2. **A standalone script.** ``python examples/custom_detector.py <clip>`` loads the clip
   (any ``.raw`` / ``.h5`` gottlux opens, or omit the argument for a synthetic scene) and
   runs the detector directly, printing what it found.

The detector itself is deliberately naive — a global "blink-rate" estimator — so every line
stays readable. It answers one question per time step: *is the whole scene's event stream
pulsing at some rate in my band, and where is that activity centred?* A real detector would
cluster spatially first (see gottlux/detectors/flutter.py for the full pipeline); this one
shows the minimal skeleton every detector shares: declare Params, implement run(), return a
DetectorResult of Targets. See docs/EXTENDING.md for the guided tour.
"""
from __future__ import annotations

import numpy as np

# Everything a custom detector needs comes from one module. `register` adds the class to
# the global registry the CLI / GUI / get_detector() all read.
from gottlux.detectors.base import Detector, DetectorResult, Param, Target, register


@register                        # <- this single line is what makes it a gottlux detector
class BlinkRateDetector(Detector):
    """Naive global blink-rate detector: FFT the whole-frame event-count series per step."""

    # -- identity (shown by --list_detectors and in the workbench's picker) --------------
    name = "blink"               # the registry key: --detector blink / get_detector("blink")
    description = "Example plugin: global blink/flicker-rate detector (whole-frame FFT)."
    regime = "both"              # 'staring' | 'rotation' | 'both'
    use_for = "learning the detector API; scenes dominated by one blinking source."

    # -- tunable knobs -------------------------------------------------------------------
    # Each Param is self-describing (range, step, unit, help, group). The GUI builds its
    # tuning panel from this list; the CLI accepts overrides via get_detector(**kw).
    PARAMS = [
        Param("freq_lo", "Band low", 5.0, 1.0, 500.0, 1.0, "float", unit="Hz",
              group="Band", help="Lower edge of the blink band searched for a peak."),
        Param("freq_hi", "Band high", 400.0, 2.0, 2000.0, 1.0, "float", unit="Hz",
              group="Band", help="Upper edge of the blink band."),
        Param("snr_thresh", "SNR gate", 3.0, 1.0, 30.0, 0.5, "float",
              group="Band", help="Min spectral peak / noise floor to call a step a blink."),
        Param("step_s", "Step", 0.25, 0.05, 2.0, 0.05, "float", unit="s",
              group="Timing", help="Analysis window length; one detection attempt per step."),
    ]

    def run(self, rec, cfg=None, t0=None, t1=None, progress=None) -> DetectorResult:
        """The one required method: events in, a DetectorResult of Targets out.

        *rec* is a Recording (or anything with .window()); *cfg* is the optional gottlux
        Config; *t0*/*t1* bound the analysis window in seconds; *progress* (if given) takes
        a fraction in [0, 1]. self.params holds the coerced Param values.
        """
        # region_spectrum is the same FFT workhorse the built-in detectors use.
        from gottlux.core.frequency import region_spectrum

        P = self.params
        win = rec.window(t0, t1)                       # cheap slice of the memmapped arrays
        ts = win.t_s                                   # event times in seconds (sorted)
        if ts.size < 64:
            return DetectorResult([], self.name, dict(P), self.regime,
                                  diagnostics={"note": "too few events"})

        # Sample rate for the binned FFT: comfortably above Nyquist for the band top.
        fs = max(2.5 * P["freq_hi"], 1000.0)

        rows = []                                      # one row per step that passed the gate
        steps = np.arange(float(ts[0]), float(ts[-1]), P["step_s"])
        for si, s in enumerate(steps):
            lo = np.searchsorted(ts, s)
            hi = np.searchsorted(ts, s + P["step_s"])
            if hi - lo < 64:
                continue
            # 1) the temporal question: does this step's stream carry an in-band peak?
            sp = region_spectrum(win.t[lo:hi], fs=fs, fmin=P["freq_lo"], fmax=P["freq_hi"])
            if not (sp.detected and sp.snr >= P["snr_thresh"]):
                continue
            # 2) the spatial question (naively): where is the activity centred?
            cx = float(np.mean(win.x[lo:hi]))
            cy = float(np.mean(win.y[lo:hi]))
            half = 0.05 * max(win.width, win.height)   # a nominal box around the centroid
            rows.append((s + 0.5 * P["step_s"], cx, cy,
                         (cx - half, cy - half, cx + half, cy + half),
                         float(sp.peak_freq), float(sp.snr), float(sp.harmonic_score)))
            if progress:
                progress((si + 1) / len(steps))

        # Package the accepted steps as ONE Target (naive: a single global source). A
        # Target carries parallel arrays — one entry per detection.
        targets = []
        if rows:
            t = np.array([r[0] for r in rows])
            targets = [Target(id=0, t=t,
                              cx=np.array([r[1] for r in rows]),
                              cy=np.array([r[2] for r in rows]),
                              bbox=np.array([r[3] for r in rows], float),
                              freq_hz=np.array([r[4] for r in rows]),
                              snr=np.array([r[5] for r in rows]),
                              harmonic=np.array([r[6] for r in rows]))]
        diag = {"n_steps": int(steps.size), "n_accepted": len(rows), "fs": float(fs)}
        return DetectorResult(targets, self.name, dict(P), self.regime, diagnostics=diag)


# ======================================================================================
# Standalone entry point: `python examples/custom_detector.py [clip.raw|clip.h5]`
# (Never runs when the file is imported as a GOTTLUX_PLUGINS plugin.)
# ======================================================================================
def main(argv=None):
    import sys
    argv = list(sys.argv if argv is None else argv)
    if len(argv) > 1:
        import gottlux as eb
        rec = eb.load(argv[1], progress=lambda f: None)
        print(f"Loaded {rec.name}: {rec.n:,} events, {rec.duration_s:.2f} s")
    else:
        # no clip given: plant a known 200 Hz blinker so the demo always has data
        from gottlux.synthetic import FlutterTarget, synthetic_scene
        rec, _ = synthetic_scene(duration_s=1.5,
                                 targets=[FlutterTarget(flutter_hz=200.0)],
                                 noise_rate_hz=20_000, seed=11)
        print("No clip given - using a synthetic scene with a planted 200 Hz target.")

    # get_detector() proves the @register above worked; overrides tune any Param.
    from gottlux.detectors import get_detector
    det = get_detector("blink", snr_thresh=3.0)
    res = det.run(rec)
    print(res.summary())
    if res.targets:
        best = max(res.targets, key=lambda tg: tg.confidence)
        print(f"-> dominant blink rate: {best.median_freq:.1f} Hz "
              f"(confidence {best.confidence:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
