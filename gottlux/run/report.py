"""
report.py — a rigorous, first-principles record of a detection run.

When a detector finds (or fails to find) a target, the *number* is not enough for science:
you need to be able to say exactly **what method produced it, on what data, under what
assumptions, with what settings, and why the result is or isn't believable**. This module
turns a :class:`~gottlux.detectors.base.DetectorResult` (plus the recording and config it
came from) into that record — a human-readable Markdown report and a machine-readable JSON
sidecar that together fully document the detection.

The report is generated from first principles for the composable flutter detector: it walks
the actual pipeline stages, prints every parameter *with its physical meaning and the value
used*, states the modelling assumptions explicitly, lays out the results, and then gives a
plain-language interpretation of *why* each target was accepted and how confident to be.

Used by the workbench's "Export detection report" and by the headless pipeline.
"""
from __future__ import annotations

import numpy as np

from gottlux.io import export


# ====================================================================================
# Public API
# ====================================================================================
def build_detection_report(result, rec, cfg=None, window=None) -> tuple[str, dict]:
    """Return ``(markdown, data)`` fully documenting a detector run.

    Parameters
    ----------
    result : DetectorResult     the run output (targets + params + diagnostics).
    rec : Recording             the source recording (for provenance/metadata).
    cfg : Config | None         the run configuration (geometry/ranging assumptions).
    window : (t0, t1) | None    the analysis window in seconds, if a sub-window was used.
    """
    md = _Markdown()
    data = {}

    _section_header(md, data, result, rec, window)
    _section_recording(md, data, rec)
    _section_method(md, result)
    _section_parameters(md, data, result)
    _section_assumptions(md, result, cfg)
    _section_results(md, data, result)
    _section_diagnostics(md, data, result)
    _section_interpretation(md, result)

    if cfg is not None:                       # archive the resolved capture rig for provenance
        data["optics"] = cfg.optics()
        data["sensor_profile"] = cfg.active_profile().to_dict()

    return md.text(), data


def save_detection_report(path_base, result, rec, cfg=None, window=None) -> list[str]:
    """Write the report as ``<base>_report.md`` + ``<base>_report.json``. Returns paths."""
    md, data = build_detection_report(result, rec, cfg=cfg, window=window)
    written = []
    path_md = path_base + "_report.md"
    export._ensure_dir(path_md)
    with open(path_md, "w", encoding="utf-8") as f:
        f.write(md)
    written.append(path_md)
    written += export.save_json(data, path_base + "_report.json")
    return written


# ====================================================================================
# Sections
# ====================================================================================
def _section_header(md, data, result, rec, window):
    md.h1(f"Detection report — `{result.detector}` on {rec.name}")
    win = f"[{window[0]:.3f}, {window[1]:.3f}] s" if window else "full recording"
    md.p(f"Detector **{result.detector}** ({result.regime} regime, signature "
         f"`{result.signature}`) run over **{win}**. "
         f"{result.n_targets} target(s); {len(result.confident(0.5))} confident (≥0.5).")
    data["detector"] = result.detector
    data["regime"] = result.regime
    data["signature"] = result.signature
    data["window_s"] = list(window) if window else None
    data["n_targets"] = result.n_targets


def _section_recording(md, data, rec):
    md.h2("1 · Data")
    meta = dict(name=rec.name, source=rec.source_path or "(in-memory)", encoding=rec.fmt,
                width=rec.width, height=rec.height, n_events=rec.n,
                duration_s=round(rec.duration_s, 6),
                mean_event_rate_evs=round(rec.mean_event_rate, 1),
                mode=("rotation" if rec.is_rotating else "staring"))
    md.table(["field", "value"], [[k, v] for k, v in meta.items()])
    md.p("The detector operates on the decoded event stream `(x, y, p, t)`; no frames are "
         "formed except internally for clustering. Event times are microsecond-resolution and "
         "monotonic, so all temporal-frequency statements below are exact to the sensor clock.")
    data["recording"] = meta


def _section_method(md, result):
    md.h2("2 · Method (from first principles)")
    md.p("A spinning rotor or a beating wing modulates scene brightness **periodically**. An "
         "event camera reports per-pixel log-intensity changes, so that periodic modulation "
         "produces **events in periodic bursts** at the blade-pass / wingbeat frequency (plus "
         "harmonics from the non-sinusoidal profile). The detector's whole job is to find image "
         "regions whose *event timing* carries that periodic signature — not merely regions that "
         "are moving. It does so as a pipeline:")
    md.olist([
        "**Foreground.** Remove hot/stuck pixels (high event-count percentile) and, for a "
        "staring sensor, subtract the persistent-pixel background, leaving only changing scene "
        "content. This is a *spatial* gate; it never uses frequency, so it cannot fabricate a tone.",
        "**Cluster.** Within each short step window, accumulate the foreground events to a "
        "binary occupancy image, bridge gaps (dilate) and trim spurs (erode), then take "
        "connected components above a minimum area. Each component is a *candidate blob*.",
        "**FFT-verify (the decisive stage).** For each blob, take the events inside its box over "
        "a trailing window, bin them to a regular series, and FFT. Accept the blob **only** if an "
        "in-band spectral peak exceeds the SNR gate (and, if required, shows a harmonic comb). "
        "This is what turns a motion detector into a *flutter* detector: anything that merely "
        "translates across the sensor has a broadband, non-periodic temporal signature and is "
        "rejected here.",
        "**Track.** Associate accepted detections across steps by nearest-neighbour gating into "
        "persistent tracks, allowing a few coasted (predict-only) steps through occlusion.",
        "**Localize.** Convert each track's image-plane centroid/size to bearing, elevation and "
        "(given a known target size) range via a pinhole model; de-rotate to world azimuth if "
        "telemetry is present.",
    ])
    md.p("The confidence score reported per target blends four independent pieces of evidence — "
         "track persistence, mean in-band SNR, frequency stability, and harmonic support — so no "
         "single lucky window can produce a high score.")


def _section_parameters(md, data, result):
    md.h2("3 · Parameters used")
    md.p("Every knob below was held at the listed value for this run. Meanings are the physical "
         "role of each parameter, not just its name.")
    specs = _param_specs(result.detector)
    rows = []
    pdata = {}
    for key, val in result.params.items():
        spec = specs.get(key)
        label = spec.label if spec else key
        unit = f" {spec.unit}" if (spec and spec.unit) else ""
        meaning = spec.help if spec else ""
        rows.append([f"`{key}`", label, f"{_fmt(val)}{unit}", meaning])
        pdata[key] = val
    md.table(["key", "name", "value", "meaning"], rows)
    data["parameters"] = pdata


def _section_assumptions(md, result, cfg):
    md.h2("4 · Assumptions")
    P = result.params
    items = [
        f"The target's flutter tone lies within the search band "
        f"**{_fmt(P.get('freq_lo'))}–{_fmt(P.get('freq_hi'))} Hz**; a tone outside this band is "
        f"invisible to the verify stage by construction.",
        f"The binned event series is sampled at **{_fmt(P.get('fft_fs'))} Hz**, which must exceed "
        f"twice the band-high to satisfy Nyquist; the effective rate is raised automatically if not.",
        "The modulation is approximately stationary over each FFT window "
        f"(**{_fmt(P.get('fft_window_s'))} s**): long enough for frequency resolution, short "
        "enough that the target does not leave its box.",
        "Background structure is static relative to the target on the timescale of the analysis "
        "window (so persistent-pixel suppression is valid for a staring sensor).",
    ]
    if cfg is not None:
        fov = cfg.resolved_fov()
        prof = cfg.active_profile()
        w, h = cfg.resolved_sensor_wh()
        items.append(f"Geometry/ranging uses a pinhole model for the **{prof.name}** rig "
                     f"({w}×{h} px, {prof.focal_length_mm:g} mm lens, horizontal "
                     f"FOV ≈ **{_fmt(fov)}°**) and an assumed target size of "
                     f"**{_fmt(cfg.target_size_m)} m**; range scales inversely with apparent size "
                     "and is only as good as that size estimate.")
    md.ulist(items)


def _section_results(md, data, result):
    md.h2("5 · Results")
    if not result.targets:
        md.p("**No targets** survived verification and tracking. See diagnostics for where the "
             "pipeline shed candidates (commonly: no in-band peak above the SNR gate).")
        data["targets"] = []
        return
    rows = []
    tdata = []
    for t in sorted(result.targets, key=lambda t: -t.confidence):
        az = f"{np.nanmean(t.azimuth_deg):.1f}" if t.azimuth_deg is not None else "—"
        rng = f"{np.nanmedian(t.range_m):.1f}" if t.range_m is not None else "—"
        rows.append([t.id, f"{t.median_freq:.0f}", f"{t.confidence:.2f}",
                     f"{np.nanmean(t.snr):.1f}", f"{np.nanmean(t.harmonic):.2f}",
                     f"{t.freq_stability:.2f}", t.n, f"{t.duration_s:.2f}", az, rng])
        tdata.append(dict(id=int(t.id), median_freq_hz=float(t.median_freq),
                          confidence=float(t.confidence), mean_snr=float(np.nanmean(t.snr)),
                          mean_harmonic=float(np.nanmean(t.harmonic)),
                          freq_stability=float(t.freq_stability), n_detections=int(t.n),
                          duration_s=float(t.duration_s)))
    md.table(["ID", "Freq Hz", "Conf", "SNR", "Harm", "FreqStab", "#det", "Dur s", "Az°", "Rng m"],
             rows)
    data["targets"] = tdata


def _section_diagnostics(md, data, result):
    md.h2("6 · Diagnostics")
    diag = result.diagnostics or {}
    if diag:
        md.table(["quantity", "value"], [[k, _fmt(v)] for k, v in diag.items()])
        nc = diag.get("n_candidates", 0)
        nv = diag.get("n_verified", 0)
        if nc:
            md.p(f"Verification kept **{nv}/{nc}** candidate blobs "
                 f"({100.0 * nv / nc:.1f}%). A very low pass-rate with no targets usually means "
                 "the SNR/harmonic gates are too strict or the band is off the target's tone; a "
                 "very high pass-rate with many weak tracks means the gates are too loose.")
    else:
        md.p("No diagnostics recorded.")
    data["diagnostics"] = {k: _jsonable(v) for k, v in diag.items()}


def _section_interpretation(md, result):
    md.h2("7 · Why this result (and how much to trust it)")
    if not result.targets:
        md.ulist([
            "Loosen the **SNR gate** first — it is the single most decisive knob.",
            "Widen or re-centre the **frequency band** on the tone you read from the region "
            "spectrum in the workbench.",
            "Lengthen the **FFT window** for finer frequency resolution if the target is faint "
            "but steady.",
            "Confirm the foreground stage is not erasing the target (disable background "
            "suppression briefly).",
        ])
        return
    best = max(result.targets, key=lambda t: t.confidence)
    md.p(f"The strongest target (#{best.id}) sits at **{best.median_freq:.0f} Hz** with mean SNR "
         f"**{np.nanmean(best.snr):.1f}**, frequency stability **{best.freq_stability:.2f}** and "
         f"harmonic support **{np.nanmean(best.harmonic):.2f}**, followed for "
         f"**{best.duration_s:.2f} s** across **{best.n}** detections. High SNR with a steady "
         "frequency and present harmonics is the signature of a genuine rotor/wingbeat rather "
         "than a chance fluctuation; a high score driven only by persistence (with low SNR and "
         "no harmonics) deserves a second look at the region spectrum.")
    md.p("Reproducibility: this report plus the JSON sidecar fully specify the method and "
         "settings; rerunning the same detector with the same parameters on the same window "
         "reproduces these targets deterministically.")


# ====================================================================================
# Helpers
# ====================================================================================
def _param_specs(detector_name):
    try:
        from gottlux.detectors import list_detectors
        cls = list_detectors().get(detector_name)
        return {p.key: p for p in getattr(cls, "PARAMS", [])} if cls else {}
    except Exception:
        return {}


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    return v


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


class _Markdown:
    """A tiny Markdown builder (headers, paragraphs, lists, tables)."""

    def __init__(self):
        self._lines = []

    def h1(self, s): self._lines += [f"# {s}", ""]
    def h2(self, s): self._lines += [f"## {s}", ""]
    def p(self, s): self._lines += [s, ""]

    def ulist(self, items):
        self._lines += [f"- {it}" for it in items] + [""]

    def olist(self, items):
        self._lines += [f"{i+1}. {it}" for i, it in enumerate(items)] + [""]

    def table(self, header, rows):
        self._lines.append("| " + " | ".join(str(h) for h in header) + " |")
        self._lines.append("| " + " | ".join("---" for _ in header) + " |")
        for r in rows:
            self._lines.append("| " + " | ".join(str(c) for c in r) + " |")
        self._lines.append("")

    def text(self):
        return "\n".join(self._lines).rstrip() + "\n"
