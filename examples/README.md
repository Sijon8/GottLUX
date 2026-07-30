# Examples

## `quickstart.py` — the API in one file

```bash
python examples/quickstart.py
```

A self-contained tour on synthetic data (no captures needed): it plants a 200 Hz flutter
target in a synthetic scene, then walks through loading, accumulation, the frequency
engine, the flicker map, the tunable detector, and journal-figure export.

## `data/` — bundled sample recordings

Real Prophesee **GenX320** (320×320, EVT2.1) captures, small enough to live in the repo.
They show up automatically in the GUI's welcome dialog and *Examples ▾* menu.

| clip | size | regime | what to look for |
|---|---|---|---|
| `Humming_Bird_Fight_merged_shortest.raw` | 0.9 MB | staring | the smallest, fastest demo — two hummingbirds contesting a feeder; wingbeat flutter pops out in the Flutter workbench |
| `Humming_Bird_Fight_merged_2.raw` | 11 MB | staring | a longer hummingbird sequence for the space-time 3-D view (wingbeats as striped columns) |
| `5inch_quadcopter.raw` | 45 MB | staring | a 5-inch quadcopter; try the `drone` detector and the flicker map (blade-pass tone) |
| `5inch_quadcopter_rotating.raw` | 74 MB | **rotating** | the same class of target seen from a *spinning* sensor — the rotation/de-rotation and panorama pipeline's home turf |

Try them:

```bash
gottlux-view examples/data/Humming_Bird_Fight_merged_shortest.raw   # instant quick look
gottlux-gui  examples/data/5inch_quadcopter.raw                     # the full instrument
gottlux      examples/data/5inch_quadcopter.raw                     # headless analysis run
gottlux      examples/data/5inch_quadcopter.raw --detector drone --freq_lo 90 --freq_hi 700
```

The first open of each clip builds its decode-once cache (`_gottlux_cache/` beside the
file); after that, opens are instant. Larger clips of these same campaigns (plus dual-EBS
and acoustic-fusion sessions) are planned as an external data release — see
[FUTURE_WORK.md](../FUTURE_WORK.md).
