"""
user_script_example.py — a complete, worked example of the gottlux user-script contract.

A user script is any ``.py`` file defining ``process(win, ctx)`` (the full contract lives
in the :mod:`gottlux.userscripts` docstring). This file is BOTH:

1. **A user script.** Run it on any recording, on exactly the window/ROI of interest:

       gottlux clip.raw --run-script examples/user_script_example.py
       gottlux clip.raw --run-script examples/user_script_example.py --t_start 0.2 --t_stop 1.0
       gottlux clip.raw --run-script examples/user_script_example.py --script-args 0.02

   The GUI's user-script action runs the same file on exactly the portion in view.

2. **A standalone demo.** ``python examples/user_script_example.py [clip.raw]`` runs the
   script through :func:`gottlux.userscripts.run_script` directly — on a synthetic scene
   when no clip is given — so the whole loop is inspectable without any hardware data.

What it computes
----------------
The **per-polarity event rate**: ON (brighter) and OFF (darker) events binned on a common
time grid, plus their ratio. A flickering or fluttering source pumps both polarities almost
symmetrically, so the ON/OFF ratio hovering near 1.0 is a quick health check that a scene's
activity is a real oscillator rather than a one-sided artifact (hot pixels, illumination
drift). Deliberately small — every contract feature appears once:

* reads the sliced events off ``win`` (only the requested window/ROI is visible);
* reads the resolved window bounds and extra tokens off ``ctx``;
* writes one file of its own into ``ctx["output_dir"]`` (scripts may do this freely);
* returns a matplotlib ``Figure`` — auto-saved as ``figure.png`` + ``figure.pdf`` — or,
  when matplotlib is unavailable, the dict of arrays — auto-saved as ``results.npz``.
"""
from __future__ import annotations

import os

import numpy as np


def process(win, ctx):
    """Per-polarity event-rate series + ON/OFF ratio for the events in view.

    *win* holds the already-sliced events (``x, y, p, t`` — ``t`` in µs, zero-based to the
    parent recording — plus ``win.rec`` and ``win.roi``); *ctx* carries the resolved
    window (``t0``/``t1`` in seconds), the run folder, and any ``--script-args`` tokens.
    """
    # -- optional tuning via --script-args: the first token overrides the rate bin (s) ---
    bin_s = 0.01
    if ctx["args"]:
        try:
            bin_s = float(ctx["args"][0])
        except ValueError:
            pass                                     # non-numeric token: keep the default

    if len(win) == 0:
        print("no events in the selected window/ROI - nothing to compute")
        return None                                  # None -> gottlux saves nothing

    # -- the computation: ON and OFF counts on one shared time grid ----------------------
    ts = win.t_s                                     # event times in seconds (sorted)
    edges = np.arange(ctx["t0"], ctx["t1"] + bin_s, bin_s)
    if edges.size < 2:
        edges = np.array([ctx["t0"], ctx["t0"] + bin_s])
    on, off = win.polarity_split()                   # boolean masks over the window
    on_rate = np.histogram(ts[on], bins=edges)[0] / bin_s
    off_rate = np.histogram(ts[off], bins=edges)[0] / bin_s
    centers = 0.5 * (edges[:-1] + edges[1:])
    # per-bin ratio (NaN where OFF is silent), plus one whole-window scalar
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(off_rate > 0, on_rate / off_rate, np.nan)
    global_ratio = float(on.sum()) / max(float((~on).sum()), 1.0)
    print(f"ON/OFF ratio over the window: {global_ratio:.3f} "
          f"({int(on.sum()):,} ON / {int((~on).sum()):,} OFF, bin {bin_s:g} s)")

    # -- a script may write its own files straight into the run folder -------------------
    np.savez_compressed(os.path.join(ctx["output_dir"], "polarity_rates.npz"),
                        centers_s=centers, on_rate_hz=on_rate, off_rate_hz=off_rate,
                        ratio=ratio, global_ratio=global_ratio)

    # -- the return value: a Figure (auto-saved PNG+PDF), or the dict (auto-saved NPZ) ---
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {"centers_s": centers, "on_rate_hz": on_rate, "off_rate_hz": off_rate,
                "ratio": ratio, "global_ratio": global_ratio}

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    ax1.plot(centers, on_rate / 1e3, label="ON (brighter)", lw=1.2)
    ax1.plot(centers, off_rate / 1e3, label="OFF (darker)", lw=1.2)
    ax1.set_ylabel("event rate (kev/s)")
    ax1.legend(loc="upper right")
    ax1.set_title(f"Per-polarity event rate - {win.rec.name} "
                  f"[{ctx['t0']:g}, {ctx['t1']:g}] s, bin {bin_s:g} s")
    ax2.plot(centers, ratio, color="tab:purple", lw=1.0)
    ax2.axhline(1.0, color="0.5", ls="--", lw=0.8)
    ax2.axhline(global_ratio, color="tab:red", ls=":", lw=0.8,
                label=f"window mean {global_ratio:.2f}")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("ON/OFF ratio")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    return fig                                       # -> figure.png + figure.pdf


# ======================================================================================
# Standalone entry point: `python examples/user_script_example.py [clip.raw|clip.h5]`
# (Never runs when the file is loaded via --run-script or the GUI action.)
# ======================================================================================
def main(argv=None):
    import sys
    argv = list(sys.argv if argv is None else argv)
    if len(argv) > 1:
        import gottlux as eb
        rec = eb.load(argv[1], progress=lambda f: None)
        print(f"Loaded {rec.name}: {rec.n:,} events, {rec.duration_s:.2f} s")
    else:
        # no clip given: plant a known 200 Hz flutterer so the demo always has data
        from gottlux.synthetic import FlutterTarget, synthetic_scene
        rec, _ = synthetic_scene(duration_s=1.2,
                                 targets=[FlutterTarget(flutter_hz=200.0)],
                                 noise_rate_hz=20_000, seed=5)
        print("No clip given - using a synthetic scene with a planted 200 Hz target.")

    # run_script() is exactly what --run-script and the GUI action call: it loads THIS
    # file fresh, slices the window, runs process(), and saves the return + a README.
    from gottlux.userscripts import run_script
    run_script(__file__, rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
