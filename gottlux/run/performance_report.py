"""
performance_report.py — compute the three operator KPIs for a recording and save a
paper-ready bundle, robustly, into a results folder beside the analyzed file.

Pipeline
--------
1. :func:`compute_performance` runs the rotor detector **once**, extracts the measured
   per-detection series from its tracks, then computes each KPI
   (:mod:`gottlux.core.performance`) **independently** — every metric is wrapped in its own
   guard so a failure or a "no data" in one (e.g. the prop-frequency tone never cleared the
   gate) never soils the other two. Even with no detection at all, the capability (model)
   ranges are still produced.
2. :func:`save_performance` writes the bundle with a **robust, fault-isolated saver**: every
   figure / table / JSON is written in its own try/except, so one bad artifact can never lose
   the rest, and a ``kpi_manifest.json`` records exactly what landed and what failed. The
   output folder is created next to the analyzed file by default.
3. :func:`run_performance` is the one-call "process": load → compute → save beside the file.
4. :func:`compare_performance` overlays several datasets (e.g. *staring* vs *rotating*).

The report is regime-aware: a *rotating* capture adds the swept-coverage figures of merit, a
*staring* capture notes radial velocity / blade-flutter. Nothing here imports Qt.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from gottlux import sensors
from gottlux.core import performance as perf
from gottlux.io import export


def _log(msg):
    print(f"[gottlux] {msg}")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


# ==================================================================================
# Result container
# ==================================================================================
@dataclass
class PerformanceResult:
    """Everything the report and saver need: the three KPIs + provenance + measured series."""
    name: str
    regime: str
    window_s: tuple
    optics: dict
    target_size_m: float
    detector: str
    n_targets: int
    tracking: Optional[perf.TrackingRangeResult] = None
    prop_frequency: Optional[perf.PropFrequencyRangeResult] = None
    time_to_contact: Optional[perf.TimeToContactResult] = None
    detections: dict = field(default_factory=dict)        # arrays for the scatter plots/tables
    primary_track: dict = field(default_factory=dict)     # {id, t, range_m} for range(t)/TTC
    ground_truth: Optional[dict] = None
    coverage: dict = field(default_factory=dict)          # rotation-only figures of merit
    notes: list = field(default_factory=list)

    def headline(self) -> dict:
        """The KPI summary numbers (what goes in the paper's results table)."""
        tr, pf, tc = self.tracking, self.prop_frequency, self.time_to_contact
        return {
            "dataset": self.name,
            "regime": self.regime,
            "target_size_m": self.target_size_m,
            "tracking_range_m": (tr.range_m if tr else None),
            "tracking_range_model_m": (tr.capability_range_m if tr else None),
            "tracking_range_measured_m": (tr.measured_max_range_m if tr else None),
            "prop_frequency_range_m": (pf.range_m if pf else None),
            "prop_frequency_model": (pf.model if pf else None),
            "prop_frequency_measured_m": (pf.measured_max_range_m if pf else None),
            "detect_range_m": (tc.detect_range_m if tc else None),
            "nominal_ttc_s": (tc.nominal_ttc_s if tc else None),
            "approach_speed_mps": (tc.approach_speed_mps if tc else None),
            "measured_ttc_at_first_s": (tc.measured_ttc_at_first_s if tc else None),
            "measured_closing_speed_mps": (tc.measured_closing_speed_mps if tc else None),
            "status": {
                "tracking": (tr.status.status if tr else "failed"),
                "prop_frequency": (pf.status.status if pf else "failed"),
                "time_to_contact": (tc.status.status if tc else "failed"),
            },
        }


# ==================================================================================
# Compute
# ==================================================================================
def _detections_table(det_result, target_size_m, fov_deg, width_px) -> dict:
    """Per-detection arrays for the metrics/plots (the shared detector table)."""
    from gottlux.detectors.base import detections_table
    return detections_table(det_result)


def _primary_track(det_result) -> dict:
    """The most-confident, longest track — used for range(t) and the measured TTC."""
    if not det_result or not det_result.targets:
        return {}
    t = max(det_result.targets, key=lambda tk: (tk.confidence, tk.n))
    return {"id": int(t.id), "t": np.asarray(t.t, float),
            "range_m": (np.asarray(t.range_m, float) if t.range_m is not None
                        else np.full(t.n, np.nan))}


def _guard(label, fn, notes):
    """Run a metric computation in isolation; a failure is recorded, never raised."""
    try:
        return fn()
    except Exception as e:                       # pragma: no cover - defensive
        notes.append(f"{label} failed: {e}")
        _log(f"KPI '{label}' failed: {e}")
        return None


def compute_performance(rec, cfg, det_result=None, ground_truth=None,
                        approach_speed=None) -> PerformanceResult:
    """Compute the three KPIs for *rec*. Each metric is isolated; partial results are fine."""
    from gottlux.core import metrics as met

    fov = cfg.resolved_fov() or sensors.DEFAULT_FOV_DEG
    W = (cfg.resolved_sensor_wh()[0] or rec.width)
    L = float(cfg.target_size_m)
    v_app = float(approach_speed if approach_speed is not None
                  else getattr(cfg, "approach_speed_mps", 15.0))
    t0 = rec.t_start_s if cfg.t_start is None else cfg.t_start
    t1 = rec.t_stop_s if cfg.t_stop is None else cfg.t_stop
    regime = "rotation" if rec.is_rotating else "staring"
    notes = []

    # --- run the detector once (its failure must not stop the capability metrics) ---
    if det_result is None:
        det_result = _guard("detector", lambda: _run_detector(rec, cfg), notes)
    det_name = (det_result.detector if det_result else (cfg.detector or "drone"))
    table = _detections_table(det_result, L, fov, W)
    n_targets = (det_result.n_targets if det_result else 0)
    if n_targets == 0:
        notes.append("no targets detected — capability (model) ranges only")

    rng_all = table.get("range_m", np.zeros(0))
    px_all = table.get("apparent_px", np.zeros(0))
    snr_all = table.get("snr", np.zeros(0))
    primary = _primary_track(det_result)

    # --- three INDEPENDENT metrics ---
    tracking = _guard("tracking_range", lambda: perf.tracking_range(
        L, fov, W, min_pixels_area=float(cfg.min_pixels),
        measured_ranges=rng_all, measured_px=px_all), notes)

    prop = _guard("prop_frequency_range", lambda: perf.prop_frequency_range(
        float(cfg.snr_thresh), L, fov, W,
        measured_ranges=rng_all, measured_snr=snr_all), notes)

    detect_range = None
    if tracking is not None:
        detect_range = tracking.range_m
    ttc = _guard("time_to_contact", lambda: perf.time_to_contact(
        detect_range, approach_speed_mps=v_app,
        range_t=primary.get("range_m"), t_s=primary.get("t")), notes)

    # --- regime context (rotation coverage figures of merit) ---
    coverage = {}
    if rec.is_rotating:
        coverage = _guard("coverage", lambda: met.coverage_metrics(rec, fov), notes) or {}

    return PerformanceResult(
        name=rec.name, regime=regime, window_s=(round(float(t0), 4), round(float(t1), 4)),
        optics=cfg.optics(), target_size_m=L, detector=det_name, n_targets=n_targets,
        tracking=tracking, prop_frequency=prop, time_to_contact=ttc,
        detections=table, primary_track=primary, ground_truth=ground_truth,
        coverage=coverage, notes=notes)


def _run_detector(rec, cfg):
    from gottlux.detectors import get_detector
    name = cfg.detector or "drone"
    det = get_detector(name, freq_lo=cfg.freq_lo, freq_hi=cfg.freq_hi, fft_fs=cfg.fft_fs,
                       fft_window_s=cfg.fft_window_s, snr_thresh=cfg.snr_thresh,
                       accum_dt=cfg.accum_dt)
    t0 = None if cfg.t_start is None else cfg.t_start
    t1 = None if cfg.t_stop is None else cfg.t_stop
    return det.run(rec, cfg, t0=t0, t1=t1)


# ==================================================================================
# Robust saving
# ==================================================================================
def _result_dir(rec, cfg, out_dir=None) -> str:
    """Pick (and create) the output folder — beside the analyzed file by default."""
    if out_dir is None:
        base = cfg.output_root or (os.path.dirname(rec.source_path) if rec.source_path
                                   else os.getcwd())
        out_dir = os.path.join(base, f"{rec.name}_kpi_{_stamp()}")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:                            # fall back to cwd if the path is unwritable
        out_dir = os.path.join(os.getcwd(), f"{rec.name}_kpi_{_stamp()}")
        os.makedirs(out_dir, exist_ok=True)
    return out_dir


class _SaveLedger:
    """Records every save attempt so one failure never aborts the bundle."""
    def __init__(self):
        self.written, self.failed = [], {}

    def do(self, label, fn):
        try:
            paths = fn() or []
            self.written.extend(paths)
            return paths
        except Exception as e:
            self.failed[label] = str(e)
            _log(f"save '{label}' failed: {e}")
            return []


def save_performance(result: PerformanceResult, rec, cfg, out_dir=None, dpi=None) -> dict:
    """Write the KPI bundle robustly. Returns ``{"dir", "written", "failed", "manifest"}``."""
    out = _result_dir(rec, cfg, out_dir)
    dpi = dpi or getattr(cfg, "fig_dpi", 300)
    fmts = (getattr(cfg, "fig_format", "png"), "pdf")
    led = _SaveLedger()
    det = result.detections

    # ---- per-metric figures (each isolated) ----
    if result.tracking is not None:
        from gottlux.viz import performance as vperf
        led.do("fig:tracking_range", lambda: export.save_figure(
            vperf.tracking_range_figure(result.tracking, det.get("range_m"), det.get("apparent_px"),
                                        title=f"Tracking range — {result.name} ({result.regime})"),
            os.path.join(out, "tracking_range"), dpi=dpi, formats=fmts, close=True))
    if result.prop_frequency is not None:
        from gottlux.viz import performance as vperf
        led.do("fig:prop_frequency", lambda: export.save_figure(
            vperf.prop_frequency_figure(result.prop_frequency, det.get("range_m"), det.get("snr"),
                                        title=f"Prop-frequency range — {result.name} ({result.regime})"),
            os.path.join(out, "prop_frequency_range"), dpi=dpi, formats=fmts, close=True))
    if result.primary_track or result.ground_truth:
        from gottlux.viz import performance as vperf
        tracks = [result.primary_track] if result.primary_track else []
        led.do("fig:range_vs_time", lambda: export.save_figure(
            vperf.range_vs_time_figure(tracks, truth=result.ground_truth,
                                       title=f"Range vs time — {result.name}"),
            os.path.join(out, "range_vs_time"), dpi=dpi, formats=fmts, close=True))
    if result.time_to_contact is not None:
        from gottlux.viz import performance as vperf
        led.do("fig:time_to_contact", lambda: export.save_figure(
            vperf.time_to_contact_figure(result.time_to_contact,
                                         title=f"Time-to-contact — {result.name}"),
            os.path.join(out, "time_to_contact"), dpi=dpi, formats=fmts, close=True))

    # ---- tables + per-metric JSON (each isolated) ----
    if det.get("t_s") is not None and len(det.get("t_s")):
        led.do("table:detections", lambda: export.save_table(det, os.path.join(out, "detections")))
    led.do("json:tracking", lambda: export.save_json(
        _asdict(result.tracking), os.path.join(out, "tracking_range.json")))
    led.do("json:prop_frequency", lambda: export.save_json(
        _asdict(result.prop_frequency), os.path.join(out, "prop_frequency_range.json")))
    led.do("json:time_to_contact", lambda: export.save_json(
        _asdict(result.time_to_contact), os.path.join(out, "time_to_contact.json")))

    # ---- combined summary + report ----
    headline = result.headline()
    led.do("json:summary", lambda: export.save_json(headline, os.path.join(out, "kpi_summary.json")))
    led.do("report:md", lambda: _write(os.path.join(out, "kpi_report.md"),
                                        _report_md(result, rec, cfg)))

    manifest = {
        "created_utc": _stamp(),
        "dataset": result.name,
        "regime": result.regime,
        "window_s": list(result.window_s),
        "optics": result.optics,
        "headline": headline,
        "notes": result.notes,
        "saved": [os.path.basename(p) for p in led.written],
        "failed": led.failed,
    }
    led.do("json:manifest", lambda: export.save_json(manifest, os.path.join(out, "kpi_manifest.json")))
    _log(f"KPI bundle → {out}  ({len(led.written)} files"
         + (f", {len(led.failed)} failed" if led.failed else "") + ")")
    return {"dir": out, "written": led.written, "failed": led.failed, "manifest": manifest}


def _asdict(obj):
    return asdict(obj) if obj is not None else None


def _write(path, text) -> list:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return [path]


# ==================================================================================
# Markdown report (regime-aware, with the sensor datasheet)
# ==================================================================================
def _fmt(v, unit=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:g}{unit}" if isinstance(v, (int, float)) else str(v)


def _report_md(result: PerformanceResult, rec, cfg) -> str:
    tr, pf, tc = result.tracking, result.prop_frequency, result.time_to_contact
    o = result.optics
    L = result.target_size_m
    lines = [
        f"# Results metrics (KPIs) — {result.name}",
        "",
        f"**Capture regime:** {result.regime}  ·  **detector:** `{result.detector}`  ·  "
        f"**targets:** {result.n_targets}  ·  **window:** {result.window_s[0]}–{result.window_s[1]} s",
        "",
        "## Sensor & optics",
        f"- {o.get('sensor_name')} — {o.get('width_px')}×{o.get('height_px')} px @ "
        f"{_fmt(o.get('pixel_pitch_um'))} µm, {_fmt(o.get('focal_length_mm'))} mm lens, "
        f"horizontal FOV **{_fmt(o.get('fov_horizontal_deg'),'°')}**",
        f"- Target critical size **L = {_fmt(L,' m')}** (used by the pinhole range model)",
        "",
    ]

    # ---- 1) tracking range ----
    lines += ["## 1 · Tracking range", ""]
    if tr is not None:
        lines += [
            "*How far the target stays large enough to track.* Pinhole model "
            "`N(D) = L·f_px/D`; trackable while pixels-on-target `N ≥ "
            f"{_fmt(tr.track_px)}` px (from the detector's min blob area).",
            "",
            f"- **Model tracking range:** {_fmt(tr.capability_range_m,' m')}",
            f"- **Measured reach (detections):** {_fmt(tr.measured_max_range_m,' m')}"
            + (f"  — at ≈{_fmt(tr.effective_track_px)} px on target" if tr.effective_track_px else ""),
            f"- **Johnson ladder:** "
            + ", ".join(f"{k} {_fmt(v,' m')}" for k, v in (tr.johnson_ranges_m or {}).items()),
            f"- status: `{tr.status.status}`" + (f" — {tr.status.message}" if tr.status.message else ""),
            "",
        ]
        if (tr.measured_max_range_m and tr.capability_range_m
                and tr.measured_max_range_m > 1.3 * tr.capability_range_m):
            lines += ["> The measured reach exceeds the conservative model threshold — the rotor's "
                      "flicker is detected below the nominal min-blob size. Calibrate `track_px` to "
                      f"the effective {_fmt(tr.effective_track_px)} px for an empirical bound.", ""]
    else:
        lines += ["_metric unavailable._", ""]

    # ---- 2) prop-frequency range ----
    lines += ["## 2 · Prop-frequency-resolution range", ""]
    if pf is not None:
        lines += [
            "*How far the rotor blade-pass tone stays resolvable.* In-band SNR falls with range "
            f"as the disk shrinks (`SNR ∝ D^−2`); resolvable while `SNR ≥ {_fmt(pf.snr_gate)}`.",
            "",
            f"- **Prop-frequency range:** {_fmt(pf.range_m,' m')}  (model: `{pf.model}`"
            + (f", slope {_fmt(pf.slope)}" if pf.slope is not None else "")
            + (f", R²={_fmt(pf.r2)}" if pf.r2 is not None else "") + ")",
            f"- **Measured reach (tone cleared gate):** {_fmt(pf.measured_max_range_m,' m')} "
            f"from {pf.n_resolved} detection(s)",
            f"- status: `{pf.status.status}`" + (f" — {pf.status.message}" if pf.status.message else ""),
            "",
        ]
    else:
        lines += ["_metric unavailable._", ""]

    # ---- 3) time to contact ----
    lines += ["## 3 · Time-to-contact (operator warning time)", ""]
    if tc is not None:
        sweep = ", ".join(f"{_fmt(s)} m/s → {_fmt(v,' s')}" for s, v in (tc.nominal_sweep_s or {}).items())
        lines += [
            "*How much warning the operator gets.* Nominal `TTC = D_detect / V_approach`; measured "
            "`TTC = range / closing-speed` from an approaching track.",
            "",
            f"- **Detection range used:** {_fmt(tc.detect_range_m,' m')}",
            f"- **Nominal warning @ {_fmt(tc.approach_speed_mps)} m/s:** {_fmt(tc.nominal_ttc_s,' s')}",
            f"- **Nominal sweep:** {sweep or '—'}",
            f"- **Measured closing speed:** {_fmt(tc.measured_closing_speed_mps,' m/s')}"
            + (f"  (approaching)" if tc.approaching else ""),
            f"- **Measured warning at first detection:** {_fmt(tc.measured_ttc_at_first_s,' s')}",
            f"- status: `{tc.status.status}`" + (f" — {tc.status.message}" if tc.status.message else ""),
            "",
        ]
    else:
        lines += ["_metric unavailable._", ""]

    # ---- regime context ----
    if result.regime == "rotation" and result.coverage:
        c = result.coverage
        lines += [
            "## Coverage (rotating sensor)", "",
            f"- swept azimuth **{_fmt(c.get('coverage_azimuth_deg'),'°')}**, elevation band "
            f"**{_fmt(c.get('coverage_elev_band_deg'),'°')}**, "
            f"swept solid angle **{_fmt(c.get('swept_solid_angle_sr'),' sr')}** "
            f"({_fmt(c.get('sphere_fraction_pct'),'%')} of sphere)",
            f"- gain vs a static EBS **×{_fmt(c.get('coverage_gain_vs_static'))}**; revisit "
            f"**{_fmt(c.get('revisit_interval_s'),' s')}** "
            f"({_fmt(c.get('update_rate_hz'),' Hz')}, {_fmt(c.get('n_revolutions'))} revs)",
            "",
        ]
    elif result.regime == "staring":
        lines += ["## Staring context", "",
                  "A fixed sensor also yields per-track radial velocity and the blade-flutter FFT "
                  "from the events inside the tracked box (see the detection report).", ""]

    if result.notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in result.notes] + [""]
    lines += ["_Generated by GottLUX (gottlux.run.performance_report)._", ""]
    return "\n".join(lines)


# ==================================================================================
# One-call process + comparison
# ==================================================================================
def _expand_ground_truth(gt, rec, cfg):
    """Normalize *gt* to ``{"t", "range_m"}``. A scalar known range becomes a flat line across
    the analysis window; an already-shaped dict passes through; anything else → ``None``."""
    if gt is None:
        return None
    if isinstance(gt, (int, float)):
        t0 = rec.t_start_s if cfg.t_start is None else cfg.t_start
        t1 = rec.t_stop_s if cfg.t_stop is None else cfg.t_stop
        return {"t": [float(t0), float(t1)], "range_m": [float(gt), float(gt)]}
    if isinstance(gt, dict) and "range_m" in gt:
        return gt
    return None


def run_performance(path, cfg, ground_truth=None, out_dir=None) -> dict:
    """Load *path*, compute the KPIs, and save the bundle beside the file. Returns the save dict."""
    import gottlux as eb
    rec = eb.load(path, camera=cfg.camera, mode=cfg.mode, progress=lambda f: None)
    if cfg.sensor_w is None:
        cfg.sensor_w = rec.width
    if cfg.sensor_h is None:
        cfg.sensor_h = rec.height
    gt = _expand_ground_truth(ground_truth, rec, cfg)
    result = compute_performance(rec, cfg, ground_truth=gt,
                                 approach_speed=cfg.approach_speed_mps)
    saved = save_performance(result, rec, cfg, out_dir=out_dir)
    if getattr(cfg, "open_when_done", True):          # reveal the KPI bundle when it lands
        from gottlux.io.paths import open_in_file_browser
        open_in_file_browser(saved["dir"])
    return saved


def compare_performance(results, labels=None, out_dir=".", cfg=None, dpi=300) -> dict:
    """Overlay several :class:`PerformanceResult` (e.g. staring vs rotating) into one comparison."""
    labels = labels or [r.name for r in results]
    datasets = {lab: r.headline() for lab, r in zip(labels, results)}
    led = _SaveLedger()
    os.makedirs(out_dir, exist_ok=True)
    from gottlux.viz import performance as vperf
    fmts = (getattr(cfg, "fig_format", "png") if cfg else "png", "pdf")
    led.do("fig:comparison", lambda: export.save_figure(
        vperf.comparison_figure(datasets, title="Results comparison — staring vs rotating"),
        os.path.join(out_dir, "kpi_comparison"), dpi=dpi, formats=fmts, close=True))
    led.do("table:comparison", lambda: export.save_table(
        {"dataset": list(datasets),
         **{k: [datasets[d].get(k) for d in datasets]
            for k in ("regime", "tracking_range_m", "prop_frequency_range_m",
                      "detect_range_m", "nominal_ttc_s")}},
        os.path.join(out_dir, "kpi_comparison")))
    led.do("json:comparison", lambda: export.save_json(
        datasets, os.path.join(out_dir, "kpi_comparison.json")))
    return {"dir": out_dir, "written": led.written, "failed": led.failed, "datasets": datasets}
