# GottLUX Web

A fully static, dependency-free in-browser viewer + analysis app for Prophesee
event-camera `.raw` recordings — the lightweight companion to the desktop
GottLUX suite, hosted from this `web/` folder on GitHub Pages.

Press play on real event-camera data, drag a box on the image, and read the
rotor / wingbeat spectrum — all client-side, vanilla JS + canvas, no CDNs, no
build step, no uploads.

## What's in here

| Path | Role |
| --- | --- |
| `index.html` + `style.css` | App shell, dark instrument theme |
| `js/decoder.js` | Prophesee decoder, ported 1:1 from `gottlux/io/decode.py` (EVT2.1, EVT2.0, EVT3). Loadable from the main thread, a Worker, or Node. |
| `js/decode_worker.js` | Runs the decoder off-thread with progress + transferable buffers |
| `js/analysis.js` | Radix-2 FFT, region spectrum (peak/SNR/harmonic comb) and flicker map — ports of the essentials of `gottlux/core/frequency.py` |
| `js/analysis_worker.js` | Off-thread spectrum / flicker-map jobs |
| `js/viewer.js` | UI: clip picker, accumulation renderer, transport (logarithmic 0.005×–2× playback speed), event-rate timeline, ROI spectrum, flicker overlay |
| `data/` | Sample clips + `manifest.json` (see contract below) |
| `selftest.html` + `selftest/` | Decoder-parity self-test (see below) |

## Local development

No build step. From the repo root:

```sh
python -m http.server -d web
```

then open <http://localhost:8000/>. (A server is needed because the app uses
`fetch` and Web Workers; `file://` won't work.)

Without `web/data/manifest.json` the clip picker shows a note and the app runs
in drop-a-file mode — drag any Prophesee `.raw` onto the page. Files opened
this way are decoded entirely in the browser tab and never leave the machine.

## Data manifest contract

`data/manifest.json` lists the hosted sample clips:

```json
{ "clips": [ { "file": "<name>.raw", "title": "...",
               "regime": "staring" | "rotating",
               "duration_s": 2.0, "size_mb": 8.5,
               "description": "one engaging sentence",
               "suggested": { "accum_ms": 20, "band_hz": [80, 800] } } ] }
```

The app fetches `data/manifest.json` relative to `index.html` and loads each
clip from `data/<file>`. `suggested` pre-fills the accumulation-window slider
and the analysis band.

## Deploy

The repository's Pages workflow ships the `web/` folder as-is — everything is
static and self-contained (strict same-origin: no external scripts, fonts or
styles). Any static host works.

## Self-test & decoder-parity policy

`selftest.html` proves the JS decoder matches the desktop suite's Python
decoder on real bytes:

- `selftest/make_fixtures.py` (run with the repo on `PYTHONPATH`:
  `python web/selftest/make_fixtures.py`) generates
  - an **EVT2.1** fixture via `gottlux.synthetic` + `gottlux.io.writer` with a
    planted 200 Hz flutter target, and
  - a hand-packed **EVT2.0** fixture that exercises the nasty paths (CD words
    before the first TIME_HIGH, a non-monotonic TIME_HIGH, ignored word types,
    a pre-roll cluster removed by `strip_preroll`),

  then decodes both with `gottlux.io.decode` (the oracle) and records counts,
  spans and checksums into `*_expected.json`.
- `selftest.html` decodes the same bytes with `js/decoder.js`, compares every
  field, and additionally runs the JS region spectrum on the planted target,
  asserting the peak is within 2 Hz of the planted frequency.

Results render into `#selftest-results` with a `data-status="pass|fail"`
attribute (machine-checkable) and a `SELFTEST PASS m/n` console line.

**Policy:** whenever `gottlux/io/decode.py` changes behaviour (i.e. its
`DECODER_VERSION` is bumped), regenerate the fixtures with `make_fixtures.py`,
update `js/decoder.js` to match, and re-run `selftest.html` before shipping.

## Honest limitations

- The web spectrum engine implements binned-FFT `region_spectrum` and
  `flicker_map` only; Lomb–Scargle, NUDFT, spectral whitening, spectrograms,
  de-rotation, detectors and exports live in the desktop suite.
- Whole-file in-memory decode: comfortable to a few hundred MB of `.raw`;
  multi-GB recordings are desktop territory (`gottlux.io.cache` streams them).

MIT license, same as the rest of the repository.
