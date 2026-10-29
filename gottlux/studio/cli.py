"""
cli.py — ``gottlux-studio``: render a shot from the command line.

    gottlux-studio REC [--look trails] [--t0 S] [--t1 S] [--crop x0,y0,x1,y1]
                       [--speed 0.1] [--fps 25] [--width 480] [--preset gif|mp4|...]
                       [--set key=value ...] [--palette fire-ice] [--bloom 0.6]
                       [--title TEXT] [--legend] [--rate-strip] [--pingpong]
                       [-o OUT.gif|OUT.mp4|OUT.webm|DIR]
    gottlux-studio REC --recipe shot.json [-o OUT]      # re-render a saved recipe
    gottlux-studio REC --reel DIR [--t0 ..]             # every look, one short clip each
    gottlux-studio --list                               # looks and their parameters
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def _parse_value(v: str):
    low = v.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    try:
        return int(v) if low.lstrip("-").isdigit() else float(v)
    except ValueError:
        return v


def build_parser() -> argparse.ArgumentParser:
    from gottlux.studio import PALETTES, PRESETS, list_looks
    p = argparse.ArgumentParser(
        prog="gottlux-studio",
        description="Render a short GIF / MP4 / WebM 'shot' of an event-camera recording.")
    p.add_argument("path", nargs="?", help="a .raw / .h5 recording (or a decoded cache stem)")
    p.add_argument("--list", action="store_true", help="list the looks and their parameters")
    p.add_argument("--look", default=None, choices=sorted(list_looks()))
    p.add_argument("--t0", type=float, default=None, help="shot start (s)")
    p.add_argument("--t1", type=float, default=None, help="shot end (s)")
    p.add_argument("--crop", default=None, help="x0,y0,x1,y1 sensor pixels")
    p.add_argument("--speed", type=float, default=None,
                   help="recording seconds per wall second (0.1 = 10x slow motion)")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--width", type=int, default=None, help="output width (px)")
    p.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="format + width + fps in one go")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="a look parameter (repeatable), e.g. --set tau_ms=8")
    p.add_argument("--palette", default=None,
                   help=f"ON/OFF palette: {', '.join(sorted(PALETTES))} or one of your own "
                        "colour schemes")
    p.add_argument("--bloom", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--legend", action="store_true", help="ON/OFF colour key")
    p.add_argument("--rate-strip", action="store_true", help="event-rate strip with a playhead")
    p.add_argument("--no-stamp", action="store_true", help="no time label")
    p.add_argument("--no-speed-label", action="store_true")
    p.add_argument("--pingpong", action="store_true", help="forward then backward (seamless loop)")
    p.add_argument("--no-denoise", action="store_true", help="keep hot pixels and shot noise")
    p.add_argument("--colors", type=int, default=None, help="GIF palette size (8-256)")
    p.add_argument("--recipe", default=None, help="a .shot.json to start from")
    p.add_argument("--reel", default=None, metavar="DIR",
                   help="render every look (short clips) into DIR")
    p.add_argument("-o", "--out", default=None,
                   help="output file (.gif/.mp4/.webm) or folder; default ~/Videos/GottLux Studio")
    return p


def _list_looks():
    from gottlux.studio import list_looks
    for name, look in list_looks().items():
        print(f"{name:10s} {look.title}{'  (slow)' if look.heavy else ''}")
        print(f"           {look.description}")
        for prm in look.PARAMS:
            unit = f" {prm.unit}" if prm.unit else ""
            extra = f" one of {list(prm.choices)}" if prm.kind == "choice" else (
                f" [{prm.lo:g}..{prm.hi:g}]" if prm.kind in ("int", "float") else "")
            print(f"             --set {prm.key}={prm.default}{unit}{extra}")
        print()


def shot_from_args(a):
    import json

    from gottlux.studio import Shot
    if a.recipe:
        with open(a.recipe, encoding="utf-8") as f:
            data = json.load(f)
        shot = Shot.from_dict(data.get("shot", data))
    else:
        shot = Shot()
    if a.preset:
        shot.apply_preset(a.preset)
    if a.look:
        if a.look != shot.look:
            shot.params = {}
        shot.look = a.look
    for k in ("t0", "t1", "speed", "fps", "width", "palette", "bloom", "gamma", "title"):
        v = getattr(a, k)
        if v is not None:
            setattr(shot, k, v)
    if a.crop:
        shot.crop = tuple(int(float(v)) for v in a.crop.split(","))
    for kv in a.set:
        k, _, v = kv.partition("=")
        shot.params[k.strip()] = _parse_value(v)
    shot.legend = shot.legend or a.legend
    shot.rate_strip = shot.rate_strip or a.rate_strip
    shot.pingpong = shot.pingpong or a.pingpong
    if a.no_stamp:
        shot.stamp = False
    if a.no_speed_label:
        shot.speed_label = False
    if a.no_denoise:
        shot.denoise = False
    if a.colors:
        shot.gif_colors = a.colors
    return shot


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.list:
        _list_looks()
        return 0
    if not a.path:
        build_parser().print_usage()
        return 2
    import gottlux
    from gottlux.studio import export_shot, list_looks
    rec = gottlux.load(a.path)
    shot = shot_from_args(a)
    shot.source = os.path.basename(a.path)

    def bar(frac):
        n = int(frac * 30)
        sys.stdout.write(f"\r  [{'#' * n}{'.' * (30 - n)}] {frac * 100:5.1f}%")
        sys.stdout.flush()

    if a.reel:
        os.makedirs(a.reel, exist_ok=True)
        for name in list_looks():
            from dataclasses import replace
            s = replace(shot, look=name, params={})
            t = time.time()
            facts = export_shot(rec, s, folder=a.reel, progress=bar)
            print(f"\r{name:10s} {facts['frames']:4d} frames  {time.time() - t:5.1f}s  "
                  f"{os.path.basename(facts['path'])}")
        return 0
    out = a.out
    if out and os.path.isdir(out):
        facts = export_shot(rec, shot, folder=out, progress=bar)
    else:
        facts = export_shot(rec, shot, out, progress=bar)
    size_mb = os.path.getsize(facts["path"]) / 1e6 if os.path.isfile(facts["path"]) else 0.0
    print(f"\r{facts['format'].upper()} {facts['size'][0]}x{facts['size'][1]}  "
          f"{facts['frames']} frames  {facts['duration_s']:.1f}s  {size_mb:.1f} MB")
    if "f0_hz" in facts:
        print(f"f0 = {facts['f0_hz']:.2f} Hz")
    print(facts["path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
