"""
canvas_cli.py — ``gottlux-canvas``: render a canvas composition from the command line.

    gottlux-canvas SPEC.gottlux-canvas.json -o OUT.mp4|OUT.webm|OUT.gif
                   [--fps 30] [--quality 8] [--scale 0.5 | --width 1280] [--duration S]
    gottlux-canvas --mosaic REC [REC ...] --looks frame,trails,surface,spacetime
                   [--size 1280x720] [--time-scale 0.25] [--labels] [--stamps]
                   [--save-spec SPEC] -o OUT.mp4
    gottlux-canvas SPEC --raw OUT.raw                    # the composited events instead

The first form re-renders a composition saved by the Canvas composer (or written beside any
canvas / timeline export) through exactly the path the GUI's *Export video…* uses. The
second builds one on the fly: every recording × every look, tiled over the canvas, labelled
— the quickest way to see one moment as several visualizations side by side. The spec used
is written next to the output (``OUT.gottlux-canvas.json``) so the video can be re-made.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gottlux-canvas",
        description="Render a canvas composition (several recordings and visualizations on "
                    "one canvas) to MP4 / WebM / GIF, or its composited events to .raw.")
    p.add_argument("spec", nargs="?", help="a .gottlux-canvas.json composition")
    p.add_argument("-o", "--out", help="output video (.mp4 / .webm / .gif)")
    p.add_argument("--raw", help="write the composited events to this .raw instead")
    p.add_argument("--mosaic", nargs="+", metavar="REC",
                   help="build a composition from these recordings instead of a spec")
    p.add_argument("--looks", default="frame",
                   help="comma-separated visualizations per recording for --mosaic: "
                        "'frame' (the event frame) or Studio looks (trails, heat, surface, "
                        "paint, spacetime, strobe, waterfall, flicker, radar)")
    p.add_argument("--size", default="1280x720", help="canvas W×H for --mosaic")
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="clip seconds per canvas second for --mosaic (0.25 = 4× slower)")
    p.add_argument("--colormap", default="inferno", help="colormap of 'frame' cells")
    p.add_argument("--palette", default="fire-ice", help="ON/OFF palette of look cells")
    p.add_argument("--labels", action="store_true", help="label every cell with its look")
    p.add_argument("--stamps", action="store_true", help="time stamp in every cell")
    p.add_argument("--save-spec", help="also save the composition used to this path")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--quality", type=int, default=8, help="1 … 10 (MP4 / WebM)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--scale", type=float, default=None, help="encode at this scale")
    g.add_argument("--width", type=int, default=None, help="encode at this width (px)")
    p.add_argument("--duration", type=float, default=None, help="seconds of canvas time")
    return p


def _size(text):
    w, h = str(text).lower().replace("×", "x").split("x")
    return max(16, int(w)), max(16, int(h))


def mosaic_spec(paths, looks, size=(1280, 720), time_scale=1.0, colormap="inferno",
                palette="fire-ice", labels=False, stamps=False):
    """Every recording × every look as tiled, labelled cells → ``(spec, recs)``."""
    import gottlux
    from gottlux.core import canvas as cv
    from gottlux.studio.looks import get_look
    recs, cells = {}, []
    for path in paths:
        key = os.path.basename(path)
        recs[key] = gottlux.load(path)
        for lk in looks:
            look = "" if lk in ("", "frame") else lk
            name = "Event frame" if not look else get_look(look).title
            cells.append(cv.CanvasClip(source=key, look=look, colormap=colormap,
                                       palette=palette, denoise=bool(look), fit="fit",
                                       time_scale=time_scale, stamp=stamps,
                                       label=(f"{name} · {key}" if len(paths) > 1 else name)
                                       if labels else ""))
    for c, r in zip(cells, cv.autotile(len(cells), size), strict=True):
        c.rect = r
    return cv.CanvasSpec(width=size[0], height=size[1], clips=cells), recs


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    from gottlux.core import canvas as cv
    if a.mosaic:
        looks = [s.strip() for s in a.looks.split(",") if s.strip()] or ["frame"]
        spec, recs = mosaic_spec(a.mosaic, looks, _size(a.size), a.time_scale, a.colormap,
                                 a.palette, a.labels, a.stamps)
    elif a.spec:
        spec = cv.load_spec(a.spec)
        base = os.path.dirname(os.path.abspath(a.spec))
        for c in spec.clips:                     # sources may be relative to the spec
            if not os.path.isabs(c.source) and not os.path.exists(c.source):
                cand = os.path.join(base, c.source)
                if os.path.exists(cand):
                    c.source = cand
        recs = cv.load_recordings(spec)
    else:
        build_parser().print_usage()
        return 2
    if a.save_spec:
        cv.save_spec(spec, a.save_spec)

    def bar(frac):
        n = int(frac * 30)
        sys.stdout.write(f"\r  [{'#' * n}{'.' * (30 - n)}] {frac * 100:5.1f}%")
        sys.stdout.flush()

    if a.raw:
        res = cv.export_raw(spec, recs, a.raw, duration_s=a.duration, progress=bar)
        print(f"\r{res['n_events']:,} events → {res['path']}")
        for w in res.get("warnings") or []:
            print("note:", w)
        return 0
    if not a.out:
        print("give -o OUT.mp4|.webm|.gif (or --raw OUT.raw)", file=sys.stderr)
        return 2
    size = None
    if a.scale:
        size = (max(2, round(spec.width * a.scale)), max(2, round(spec.height * a.scale)))
    elif a.width:
        size = (a.width, max(2, round(spec.height * a.width / spec.width)))
    res = cv.export_video(spec, recs, a.out, fps=a.fps, duration_s=a.duration,
                          quality=a.quality, size=size, progress=bar)
    if not res:
        print("\nvideo encoding needs imageio-ffmpeg (pip install imageio imageio-ffmpeg)",
              file=sys.stderr)
        return 1
    sidecar = os.path.splitext(res["path"])[0] + cv.SPEC_SUFFIX
    cv.save_spec(spec, sidecar)
    print(f"\r{res['format'].upper()} {res['width']}x{res['height']}  {res['frames']} frames  "
          f"{res['duration_s']:.2f} s  ({len(spec.clips)} cells)")
    print(res["path"])
    print(json.dumps({"spec": sidecar}, indent=None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
