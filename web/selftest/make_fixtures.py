"""
make_fixtures.py — regenerate the GottLUX Web self-test fixtures from the repo's
own Python decoder (the parity oracle).

Run from the repo root (or anywhere, with the repo on PYTHONPATH):

    python web/selftest/make_fixtures.py

Outputs (next to this script):
    evt21_fixture.raw / evt21_expected.json
        A synthetic scene (gottlux.synthetic) with a stationary planted 200 Hz
        flutter target, written as a real EVT2.1 stream by gottlux.io.writer,
        then decoded back with gottlux.io.decode — the expected.json holds the
        oracle's counts/spans/checksums plus the planted frequency + ROI.
    evt2_fixture.raw / evt2_expected.json
        A hand-packed EVT2.0 stream (struct layout mirroring decode.chunk_evt2)
        that exercises: CD words before the first TIME_HIGH (junk, dropped),
        a non-monotonic TIME_HIGH word (running-max clamp), ignored word types
        (0xA/0xE), and a pre-roll cluster removed by strip_preroll. Its
        expected.json is likewise produced by the Python decoder.

Policy: whenever gottlux/io/decode.py changes behaviour (DECODER_VERSION bump),
re-run this script and re-run web/selftest.html in a browser.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:                       # allow running without PYTHONPATH
    sys.path.insert(0, REPO)

from gottlux.core.frequency import region_spectrum          # noqa: E402
from gottlux.io import decode as D                          # noqa: E402
from gottlux.io.writer import write_raw                     # noqa: E402
from gottlux.synthetic import FlutterTarget, synthetic_scene  # noqa: E402


def expected_from_decode(dec: dict) -> dict:
    """Order-independent ground truth for a decoded file (what the JS must match)."""
    x = dec["x"].astype(np.int64)
    y = dec["y"].astype(np.int64)
    p = dec["p"]
    t = dec["t"].astype(np.int64)              # zero-based microseconds
    return {
        "decoder_version": int(D.DECODER_VERSION),
        "fmt": dec["fmt"],
        "n": int(len(t)),
        "width": int(dec["width"]),
        "height": int(dec["height"]),
        "n_on": int((p == 1).sum()),
        "t_first_us": int(t[0]) if len(t) else 0,
        "t_last_us": int(t[-1]) if len(t) else 0,
        "t0_us": int(dec["t0_us"]),
        "sum_x_mod_2_32": int(x.sum()) % 2**32,
        "sum_y_mod_2_32": int(y.sum()) % 2**32,
        "sum_t_us_mod_2_53": int(t.sum()) % 2**53,
    }


def make_evt21() -> None:
    """Synthetic scene -> writer -> decoder oracle, with a planted 200 Hz target."""
    planted_hz = 200.0
    rec, _truth = synthetic_scene(
        duration_s=2.0, width=320, height=320,
        targets=[FlutterTarget(flutter_hz=planted_hz, x0=160, y0=160, x1=160, y1=160,
                               radius=6.0, events_per_burst=60)],
        noise_rate_hz=30_000.0, static_clutter=40, seed=7)
    path = os.path.join(HERE, "evt21_fixture.raw")
    n_written = write_raw(path, rec.x, rec.y, rec.p, rec.t, width=320, height=320)
    dec = D.decode(path)
    assert dec["fmt"] == "evt21", dec["fmt"]
    assert int(len(dec["t"])) == int(n_written), (len(dec["t"]), n_written)

    # ROI around the stationary target; oracle-check the planted tone with the
    # repo's own region_spectrum so the JS 2-Hz assertion is meaningful.
    roi = [140, 140, 180, 180]                 # x0, y0, x1, y1 (incl-excl)
    band = [50.0, 450.0]
    m = ((dec["x"] >= roi[0]) & (dec["x"] < roi[2])
         & (dec["y"] >= roi[1]) & (dec["y"] < roi[3]))
    spec = region_spectrum(dec["t"][m], fs=2000.0, fmin=band[0], fmax=band[1])
    assert abs(spec.peak_freq - planted_hz) <= 2.0, spec.peak_freq
    assert spec.snr > 10, spec.snr

    exp = expected_from_decode(dec)
    exp.update({"planted_hz": planted_hz, "roi": roi, "band_hz": band,
                "oracle_peak_hz": round(float(spec.peak_freq), 3),
                "oracle_snr": round(float(spec.snr), 2)})
    with open(os.path.join(HERE, "evt21_expected.json"), "w", encoding="utf-8") as f:
        json.dump(exp, f, indent=2)
    print(f"evt21_fixture.raw  n={exp['n']}  span={exp['t_last_us']/1e6:.3f}s  "
          f"oracle peak {spec.peak_freq:.2f} Hz (snr {spec.snr:.1f})")


def make_evt2() -> None:
    """Hand-pack an EVT2.0 stream bit-consistent with decode.chunk_evt2."""
    width, height = 304, 240
    rng = np.random.default_rng(123)

    words: list[int] = []
    # (a) junk CD words BEFORE the first TIME_HIGH -> must be dropped ('seen' gate)
    for i in range(3):
        words.append((0 << 28) | (5 << 22) | ((10 + i) << 11) | 20)

    # (b) pre-roll cluster near t=0, then a >=3 s gap: strip_preroll removes it
    pre_t = np.array([100, 150, 200, 250, 300], np.int64)
    main_t = np.sort(rng.integers(4_000_000, 5_000_000, 5000)).astype(np.int64)
    times = np.concatenate([pre_t, main_t])
    xs = rng.integers(0, width, times.size)
    ys = rng.integers(0, height, times.size)
    ps = rng.integers(0, 2, times.size)

    last_th = None
    for i, (x, y, p, t) in enumerate(zip(xs, ys, ps, times)):
        th = int(t) >> 6
        if last_th is None or th != last_th:
            words.append((0x8 << 28) | (th & 0x0FFFFFFF))
            last_th = th
        words.append((int(p) << 28) | ((int(t) & 0x3F) << 22) | (int(x) << 11) | int(y))
        if i == 2500:
            # (c) a NON-MONOTONIC TIME_HIGH (value 100 units back): the decoder's
            #     running max must clamp it, leaving timestamps unchanged
            words.append((0x8 << 28) | ((th - 100) & 0x0FFFFFFF))
            # (d) ignored word types sprinkled mid-stream
            words.append((0xA << 28) | 0x123)      # EXT_TRIGGER-ish
            words.append((0xE << 28) | 0x456)      # OTHERS
    header = ("% evt 2.0\n"
              f"% format EVT2;height={height};width={width}\n"
              f"% geometry {width}x{height}\n"
              "% end\n")
    path = os.path.join(HERE, "evt2_fixture.raw")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.array(words, dtype="<u4").tobytes())

    dec = D.decode(path)                       # the oracle
    assert dec["fmt"] == "evt2", dec["fmt"]
    assert len(dec["t"]) == 5000, len(dec["t"])   # 3 junk + 5 pre-roll dropped
    exp = expected_from_decode(dec)
    with open(os.path.join(HERE, "evt2_expected.json"), "w", encoding="utf-8") as f:
        json.dump(exp, f, indent=2)
    print(f"evt2_fixture.raw   n={exp['n']}  span={exp['t_last_us']/1e6:.3f}s  "
          f"({width}x{height})")


if __name__ == "__main__":
    make_evt21()
    make_evt2()
    print("fixtures written to", HERE)
