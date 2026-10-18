"""
studio_showcase.py — the showcase GIFs in docs/STUDIO.md, as reproducible Studio recipes.

    python examples/studio_showcase.py [out_dir] [--only NAME ...] [--mp4]

Each entry is a plain :class:`gottlux.studio.Shot`; the same recipes can be pasted into the
desktop Studio (Load recipe…) or run with ``gottlux-studio REC --recipe <name>.gif.shot.json``.
Spans and crops were picked from the clips: in the hummingbird fight the first bird darts
across the frame at 0.55–0.95 s and hovers at sensor (26, 145) from 1.00 to 1.45 s; the
5-inch quadcopter crosses the staring sensor at 3.2–4.3 s; the rotating clip spins at
0.9388 s per revolution (estimated from its event-rate periodicity).
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
HB = os.path.join(DATA, "Humming_Bird_Fight_merged_2.raw")
QUAD = os.path.join(DATA, "5inch_quadcopter.raw")
QUAD_ROT = os.path.join(DATA, "5inch_quadcopter_rotating.raw")
SPIN = 0.9388                                # s/rev of the rotating clip (estimated from it)
WIDE = (0, 40, 320, 280)                     # the hummingbirds' flight band, 4:3

SHOTS = {
    # name: (recording, Shot kwargs)
    "hb_trails": (HB, dict(look="trails", t0=0.55, t1=1.10, crop=WIDE, speed=0.12, fps=25,
                           width=480, bloom=0.7, params={"tau_ms": 5.0},
                           title="hummingbird dart", legend=True)),
    "hb_strobe": (HB, dict(look="strobe", t0=1.00, t1=1.45, crop=(0, 110, 68, 178), fps=16,
                           width=360, bloom=0.8, params={"bins": 24, "cycles": 1,
                                                         "width": 0.09})),
    "hb_orbit": (HB, dict(look="spacetime", t0=0.55, t1=1.45, speed=0.9 / 3.6, fps=25,
                          width=480, stamp=True, speed_label=False,
                          params={"depth_ms": 900.0, "freeze": True, "orbit_deg": 360.0,
                                  "time_scale": 2.2, "point_px": 1, "brightness": 1.6,
                                  "elevation": 18.0})),
    "hb_waterfall": (HB, dict(look="waterfall", t0=0.40, t1=1.60, crop=WIDE, speed=0.25,
                              fps=25, width=400, params={"history_ms": 600.0, "rows": 180})),
    "hb_paint": (HB, dict(look="paint", t0=0.30, t1=2.95, crop=WIDE, speed=0.6, fps=20,
                          width=420, bloom=0.5, speed_label=False, gif_colors=96)),
    "hb_flicker": (HB, dict(look="flicker", t0=0.60, t1=1.60, crop=WIDE, speed=0.25, fps=20,
                            width=480, params={"band_lo": 20.0, "band_hi": 120.0})),
    "hb_surface": (HB, dict(look="surface", t0=0.55, t1=0.95, crop=WIDE, speed=0.08, fps=25,
                            width=480, params={"tau_ms": 15.0})),
    "quad_flicker": (QUAD, dict(look="flicker", t0=3.20, t1=4.30, speed=0.35, fps=20,
                                width=420, params={"band_lo": 90.0, "band_hi": 700.0,
                                                   "window_ms": 200.0})),
    # exactly four revolutions in real time: the beam ends where it began (seamless loop)
    "quad_radar": (QUAD_ROT, dict(look="radar", t0=1.0, t1=1.0 + 4 * SPIN, speed=1.0, fps=25,
                                  width=420, bloom=0.4, gif_colors=96,
                                  params={"persistence_ms": 300.0})),
}


def main(argv=None):
    import gottlux
    from gottlux.studio import Shot, export_shot
    argv = list(sys.argv[1:] if argv is None else argv)
    mp4 = "--mp4" in argv
    only = []
    if "--only" in argv:
        i = argv.index("--only")
        only = [a for a in argv[i + 1:] if not a.startswith("--")]
        argv = argv[:i]
    argv = [a for a in argv if not a.startswith("--")]
    out = argv[0] if argv else os.path.join(HERE, "..", "docs", "media", "studio")
    os.makedirs(out, exist_ok=True)
    recs = {}
    for name, (path, kw) in SHOTS.items():
        if only and name not in only:
            continue
        rec = recs.setdefault(path, gottlux.load(path))
        shot = Shot(source=os.path.basename(path), **kw)
        if mp4:
            shot.format = "mp4"
        t = time.time()
        f = export_shot(rec, shot, os.path.join(out, f"{name}.{shot.format}"))
        mb = os.path.getsize(f["path"]) / 1e6
        print(f"{name:13s} {f['frames']:4d} fr {f['size'][0]}x{f['size'][1]} "
              f"{f['duration_s']:4.1f}s {mb:5.2f} MB  ({time.time() - t:4.1f}s)"
              + (f"  f0 {f['f0_hz']:.1f} Hz" if "f0_hz" in f else ""))


if __name__ == "__main__":
    main()
