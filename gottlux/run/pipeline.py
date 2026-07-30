"""
pipeline.py — headless orchestration: a Recording in, a reproducible run folder out.

Each requested analysis writes its figures (raster + vector) and data (Parquet + CSV / NPZ)
into its own subfolder, and contributes a few numbers to the manifest. The available
analyses:

================  =========================================================================
``overview``      event-rate plot, a representative accumulated frame, basic statistics
``spectral``      whole-recording flicker map (tiled) + a global region spectrum
``panorama``      de-rotated 360° panorama (rotation) / relative-bearing strip (staring)
``detect``        run the configured detector → track time-series, overlay, radar, tables
``metrics``       coverage + localization figures of merit (Markdown + JSON)
================  =========================================================================

Call :func:`run_path` (a file/folder) or :func:`run_recording` (an already-loaded Recording).
"""
from __future__ import annotations

import os

import numpy as np

from gottlux import sensors
from gottlux.config import Config
from gottlux.core import frequency as fq
from gottlux.core import metrics as met
from gottlux.core.accumulate import accumulate_frame
from gottlux.io import export
from gottlux.io.paths import open_in_file_browser
from gottlux.run.provenance import RunFolder


def _log(msg):
    print(f"[gottlux] {msg}")


def _window(rec, cfg):
    """Resolve the analysis window from ``cfg.t_start`` / ``cfg.t_stop`` (default = full)."""
    t0 = rec.t_start_s if cfg.t_start is None else max(cfg.t_start, rec.t_start_s)
    t1 = rec.t_stop_s if cfg.t_stop is None else min(cfg.t_stop, rec.t_stop_s)
    return t0, t1


def _region_spectrum(t_us, cfg, fmin, fmax):
    """A region spectrum honoring ``cfg.spectrum_method`` (fft/nufft) and ``spectrum_normalize``."""
    if cfg.spectrum_method == "nufft":
        return fq.nufft_spectrum(t_us, fmin=fmin, fmax=fmax, normalize=cfg.spectrum_normalize)
    return fq.region_spectrum(t_us, fs=cfg.fft_fs, fmin=fmin, fmax=fmax,
                              normalize=cfg.spectrum_normalize)


# ====================================================================================
# Individual analyses
# ====================================================================================
def analysis_overview(rec, cfg, run):
    from gottlux.viz import frames
    out = run.subdir("overview")
    arts = []
    # event-rate
    centers, rate = rec.event_rate(max(rec.duration_s / 600, 0.005))
    fig = frames.event_rate_figure(centers, rate, title=f"Event rate — {rec.name}")
    arts += export.save_figure(fig, os.path.join(out, "event_rate"), dpi=cfg.fig_dpi,
                               formats=(cfg.fig_format, "pdf"), close=True)
    # representative frame near the densest moment
    t_peak = centers[int(np.argmax(rate))] if len(centers) else rec.t_start_s
    frame = accumulate_frame(rec.window(t_peak, t_peak + cfg.accum_dt), mode="count")
    fig = frames.event_frame_figure(frame, mode="count",
                                    title=f"Event frame @ {t_peak:.2f}s")
    arts += export.save_figure(fig, os.path.join(out, "event_frame"), dpi=cfg.fig_dpi,
                               formats=(cfg.fig_format, "pdf"), close=True)
    arts += export.save_table({"t_s": centers, "rate_hz": rate},
                              os.path.join(out, "event_rate"))
    run.record("overview", {"mean_rate_Mev_s": round(rec.mean_event_rate / 1e6, 3),
                            "peak_rate_Mev_s": round(float(rate.max()) / 1e6, 3) if len(rate) else 0,
                            "n_on": rec.n_on, "n_off": rec.n_off})
    run.add_artifacts(arts)


def analysis_spectral(rec, cfg, run):
    from gottlux.viz import spectral
    out = run.subdir("spectral")
    arts = []
    fmin, fmax = cfg.freq_lo, cfg.freq_hi
    _log(f"flicker map (tiled) {fmin:.0f}-{fmax:.0f} Hz ...")
    fm = fq.flicker_map_max(rec, fmin=fmin, fmax=fmax, fs=cfg.fft_fs, cell=8,
                            window_s=min(1.0, max(0.2, rec.duration_s / 8)),
                            progress=None)
    bg = accumulate_frame(rec.window(None, None) if rec.duration_s < 3
                          else rec.window(rec.t_start_s, rec.t_start_s + 1.0), mode="count")
    fig = spectral.flicker_map_figure(fm, background=bg,
                                      title=f"Flicker map — {rec.name}")
    arts += export.save_figure(fig, os.path.join(out, "flicker_map"), dpi=cfg.fig_dpi,
                               formats=(cfg.fig_format, "pdf"), close=True)
    arts += export.save_arrays(os.path.join(out, "flicker_map"),
                               dominant_freq=fm.dominant_freq, snr=fm.snr,
                               band_power=fm.band_power, event_count=fm.event_count)
    arts += export.save_hdf5(os.path.join(out, "flicker_map"),
                             {"dominant_freq": fm.dominant_freq, "snr": fm.snr},
                             {"fmin": fmin, "fmax": fmax, "cell": fm.cell, "fs": fm.fs})
    # global spectrum (whole frame, mid window)
    mid = rec.t_start_s + rec.duration_s * 0.4
    sp = _region_spectrum(rec.window(mid, min(mid + 2.0, rec.t_stop_s)).t, cfg, fmin, fmax)
    fig = spectral.spectrum_figure(sp, title="Global temporal spectrum")
    arts += export.save_figure(fig, os.path.join(out, "global_spectrum"), dpi=cfg.fig_dpi,
                               formats=(cfg.fig_format, "pdf"), close=True)
    valid = np.isfinite(fm.dominant_freq)
    run.record("spectral", {
        "active_cells": int(valid.sum()),
        "peak_flutter_hz": round(float(fm.dominant_freq[valid][np.argmax(fm.snr[valid])]), 1)
        if valid.any() else None,
        "max_cell_snr": round(float(fm.snr.max()), 1),
        "global_peak_hz": round(sp.peak_freq, 1) if sp.detected else None})
    run.add_artifacts(arts)


def analysis_panorama(rec, cfg, run):
    from gottlux.viz import panorama
    out = run.subdir("panorama")
    fov = cfg.resolved_fov() or sensors.DEFAULT_FOV_DEG
    fig = panorama.panorama_figure(rec, fov_deg=fov)
    arts = export.save_figure(fig, os.path.join(out, "panorama"), dpi=cfg.fig_dpi,
                              formats=(cfg.fig_format, "pdf"), close=True)
    run.record("panorama", {"rotating": rec.is_rotating, "fov_deg": fov})
    run.add_artifacts(arts)


def analysis_detect(rec, cfg, run):
    from gottlux.detectors import get_detector
    from gottlux.viz import frames, panorama, tracks
    if not cfg.detector:
        return
    out = run.subdir(os.path.join("detect", cfg.detector))
    _log(f"running detector '{cfg.detector}' ...")
    det = get_detector(cfg.detector, freq_lo=cfg.freq_lo, freq_hi=cfg.freq_hi,
                       fft_fs=cfg.fft_fs, fft_window_s=cfg.fft_window_s,
                       snr_thresh=cfg.snr_thresh, accum_dt=cfg.accum_dt)
    res = det.run(rec, cfg, progress=None)
    arts = []
    rotating = rec.is_rotating
    fig = tracks.track_timeseries_figure(res, rotating=rotating,
                                         title=f"'{cfg.detector}' targets — {rec.name}")
    arts += export.save_figure(fig, os.path.join(out, "tracks"), dpi=cfg.fig_dpi,
                               formats=(cfg.fig_format, "pdf"), close=True)
    fig = tracks.confidence_bar_figure(res)
    arts += export.save_figure(fig, os.path.join(out, "confidence"), dpi=cfg.fig_dpi,
                               formats=(cfg.fig_format, "pdf"), close=True)
    # overlay on a representative frame
    if res.targets:
        t_mid = float(np.median([t.t[len(t.t) // 2] for t in res.targets]))
        frame = accumulate_frame(rec.window(t_mid, t_mid + cfg.accum_dt * 4), mode="count")
        fig = frames.detection_overlay_figure(frame, targets=res.targets,
                                              title=f"'{cfg.detector}' detections")
        arts += export.save_figure(fig, os.path.join(out, "overlay"), dpi=cfg.fig_dpi,
                                   formats=(cfg.fig_format, "pdf"), close=True)
        fig = panorama.radar_figure(res.targets)
        arts += export.save_figure(fig, os.path.join(out, "radar"), dpi=cfg.fig_dpi,
                                   formats=(cfg.fig_format, "pdf"), close=True)
    # per-detection table (the shared flattener)
    from gottlux.detectors.base import detections_table
    arts += export.save_table(detections_table(res), os.path.join(out, f"{cfg.detector}_detections"))
    export.save_json({"detector": res.detector, "params": res.params,
                      "diagnostics": res.diagnostics, "summary": res.summary()},
                     os.path.join(out, "detector_result.json"))
    run.record(f"detect:{cfg.detector}", {
        "n_targets": res.n_targets,
        "confident_targets": len(res.confident(0.5)),
        "top_confidence": round(max((t.confidence for t in res.targets), default=0.0), 2),
        "median_freq_hz": round(float(np.nanmedian([t.median_freq for t in res.targets])), 1)
        if res.targets else None})
    run.add_artifacts(arts)
    return res


def analysis_metrics(rec, cfg, run, det_result=None):
    out = run.subdir("metrics")
    fov = cfg.resolved_fov() or sensors.DEFAULT_FOV_DEG
    res = met.coverage_metrics(rec, fov)
    if det_result is not None and det_result.targets:
        def _gather(attr):
            parts = [getattr(t, attr) for t in det_result.targets
                     if getattr(t, attr) is not None]
            return np.concatenate(parts) if parts else np.zeros(0)
        tbl = {"azimuth_deg": _gather("azimuth_deg"),
               "elev_deg": _gather("elev_deg"),
               "range_m": _gather("range_m")}
        res.update(met.localization_metrics(tbl))
    md = met.to_markdown(res, title=f"gottlux metrics — {rec.name}")
    with open(os.path.join(out, "metrics.md"), "w", encoding="utf-8") as f:
        f.write(md)
    export.save_json(res, os.path.join(out, "metrics.json"))
    run.record("metrics", res)


def analysis_performance(rec, cfg, run, det_result=None):
    """Compute the operator results metrics (KPIs) and save the bundle into the run folder.

    Each of the three metrics is computed and saved independently (one failing never soils the
    others); see :mod:`gottlux.run.performance_report`."""
    from gottlux.run import performance_report as pr
    out = run.subdir("performance")
    _log("results metrics (tracking range · prop-frequency range · time-to-contact) ...")
    result = pr.compute_performance(rec, cfg, det_result=det_result,
                                    approach_speed=cfg.approach_speed_mps)
    saved = pr.save_performance(result, rec, cfg, out_dir=out)
    run.record("performance", result.headline())
    run.add_artifacts(saved["written"])
    return result


def analysis_export_cube(rec, cfg, run):
    """Voxelize the (windowed) stream into a space-time event cube and save it (CLI parity
    with the GUI's Export ▾ → event cube)."""
    from gottlux.app.exporting import save_event_cube
    out = run.subdir("cube")
    t0, t1 = _window(rec, cfg)
    mode = cfg.accum_mode if cfg.accum_mode in ("count", "polarity") else "count"
    _log(f"event cube: {cfg.cube_nt} slices over [{t0:.3f}, {t1:.3f}] s "
         f"(spatial bin {cfg.cube_bin}, mode {mode}) ...")
    arts = save_event_cube(os.path.join(out, rec.name), rec, t0, t1,
                           nt=cfg.cube_nt, spatial_bin=cfg.cube_bin, mode=mode)
    run.record("cube", {"window_s": [round(t0, 4), round(t1, 4)], "nt": cfg.cube_nt,
                        "spatial_bin": cfg.cube_bin, "mode": mode})
    run.add_artifacts(arts)


def _run_detector(rec, cfg):
    """Instantiate + run the configured detector (default ``drone``) over the window."""
    from gottlux.detectors import get_detector
    name = cfg.detector or "drone"
    det = get_detector(name, freq_lo=cfg.freq_lo, freq_hi=cfg.freq_hi, fft_fs=cfg.fft_fs,
                       fft_window_s=cfg.fft_window_s, snr_thresh=cfg.snr_thresh,
                       accum_dt=cfg.accum_dt)
    t0, t1 = _window(rec, cfg)
    return det.run(rec, cfg, t0=t0, t1=t1)


def analysis_report(rec, cfg, run, det_result):
    """Write the first-principles detection report (running a detector if one hasn't been)."""
    if det_result is None:
        if not cfg.detector:
            _log("report requested without --detector; defaulting to 'drone'.")
        det_result = _run_detector(rec, cfg)
    from gottlux.run.report import save_detection_report
    out = run.subdir("report")
    t0, t1 = _window(rec, cfg)
    _log(f"detection report for '{det_result.detector}' ...")
    arts = save_detection_report(os.path.join(out, det_result.detector), det_result, rec,
                                 cfg=cfg, window=(t0, t1))
    run.add_artifacts(arts)
    return det_result


_ANALYSES = {
    "overview": analysis_overview,
    "spectral": analysis_spectral,
    "panorama": analysis_panorama,
    "performance": analysis_performance,
}

# Merge in the ported EBS rotation analyses (radar, mti, rate_surface, validation,
# panorama_video, masksweep, tracking, fusion, rotation_metrics) — one analyses registry
# spanning the staring and rotation suites. A broken analysis must not break the import.
try:
    from gottlux.run.rotation_pipeline import ROTATION_ANALYSES as _ROT
    _ANALYSES.update(_ROT)
except Exception as _e:   # pragma: no cover
    import warnings
    warnings.warn(f"rotation analyses unavailable: {_e}")


# ====================================================================================
# Individual named plots — `--plots a,b,c` outputs exactly these figures.
# Each builder: (rec, cfg, run, det_result) -> list[saved paths]. `needs_detector`
# marks the ones that require a detector run first.
# ====================================================================================
def _save(fig, run, name, cfg):
    out = run.subdir("plots")
    return export.save_figure(fig, os.path.join(out, name), dpi=cfg.fig_dpi,
                              formats=(cfg.fig_format, "pdf"), close=True)


def _plot_event_rate(rec, cfg, run, det):
    from gottlux.viz import frames
    c, r = rec.event_rate(max(rec.duration_s / 600, 0.005))
    return _save(frames.event_rate_figure(c, r, title=f"Event rate — {rec.name}"),
                 run, "event_rate", cfg)


def _plot_event_frame(rec, cfg, run, det):
    from gottlux.viz import frames
    c, r = rec.event_rate(max(rec.duration_s / 600, 0.005))
    t = c[int(np.argmax(r))] if len(c) else rec.t_start_s
    frame = accumulate_frame(rec.window(t, t + cfg.accum_dt), mode="count")
    return _save(frames.event_frame_figure(frame, mode="count",
                 title=f"Event frame @ {t:.2f}s"), run, "event_frame", cfg)


def _plot_flicker_map(rec, cfg, run, det):
    from gottlux.viz import spectral
    fm = fq.flicker_map_max(rec, fmin=cfg.freq_lo, fmax=cfg.freq_hi, fs=cfg.fft_fs, cell=8,
                            window_s=min(1.0, max(0.2, rec.duration_s / 8)))
    bg = accumulate_frame(rec.window(rec.t_start_s, rec.t_start_s + min(1.0, rec.duration_s)),
                          mode="count")
    return _save(spectral.flicker_map_figure(fm, background=bg,
                 title=f"Flicker map — {rec.name}"), run, "flicker_map", cfg)


def _plot_spectrum(rec, cfg, run, det):
    from gottlux.viz import spectral
    mid = rec.t_start_s + rec.duration_s * 0.4
    sp = _region_spectrum(rec.window(mid, min(mid + 2.0, rec.t_stop_s)).t, cfg,
                          cfg.freq_lo, cfg.freq_hi)
    return _save(spectral.spectrum_figure(sp, title="Global temporal spectrum"),
                 run, "spectrum", cfg)


def _plot_spectrogram(rec, cfg, run, det):
    from gottlux.viz import spectral
    mid = rec.t_start_s + rec.duration_s * 0.3
    tt, ff, S = fq.spectrogram(rec.window(mid, min(mid + 3.0, rec.t_stop_s)).t,
                               fs=cfg.fft_fs, fmax=cfg.freq_hi)
    return _save(spectral.spectrogram_figure(tt, ff, S, title="Global spectrogram",
                 fmax=cfg.freq_hi), run, "spectrogram", cfg)


def _plot_panorama(rec, cfg, run, det):
    from gottlux.viz import panorama
    return _save(panorama.panorama_figure(rec, fov_deg=cfg.resolved_fov() or sensors.DEFAULT_FOV_DEG),
                 run, "panorama", cfg)


def _plot_radar(rec, cfg, run, det):
    from gottlux.viz import panorama
    return _save(panorama.radar_figure(det.targets if det else []),
                 run, "radar", cfg)


def _plot_tracks(rec, cfg, run, det):
    from gottlux.viz import tracks
    return _save(tracks.track_timeseries_figure(det, rotating=rec.is_rotating,
                 title=f"'{det.detector}' targets — {rec.name}"), run, "tracks", cfg)


def _plot_confidence(rec, cfg, run, det):
    from gottlux.viz import tracks
    return _save(tracks.confidence_bar_figure(det), run, "confidence", cfg)


def _plot_overlay(rec, cfg, run, det):
    from gottlux.viz import frames
    if not det or not det.targets:
        return []
    t_mid = float(np.median([t.t[len(t.t) // 2] for t in det.targets]))
    frame = accumulate_frame(rec.window(t_mid, t_mid + cfg.accum_dt * 4), mode="count")
    return _save(frames.detection_overlay_figure(frame, targets=det.targets,
                 title=f"'{det.detector}' detections"), run, "overlay", cfg)


# name -> (builder, needs_detector)
PLOTS = {
    "event_rate":  (_plot_event_rate, False),
    "event_frame": (_plot_event_frame, False),
    "flicker_map": (_plot_flicker_map, False),
    "spectrum":    (_plot_spectrum, False),
    "spectrogram": (_plot_spectrogram, False),
    "panorama":    (_plot_panorama, False),
    "radar":       (_plot_radar, True),
    "tracks":      (_plot_tracks, True),
    "confidence":  (_plot_confidence, True),
    "overlay":     (_plot_overlay, True),
}


def list_plots() -> dict:
    """Mapping of plot name -> needs-a-detector? (for ``--list_plots``)."""
    return {k: needs for k, (_, needs) in PLOTS.items()}


def analysis_plots(rec, cfg, run, names):
    """Output exactly the requested named figures (running the detector once if needed)."""
    names = [n for n in names if n in PLOTS]
    unknown = [n for n in names if n not in PLOTS]
    for n in unknown:
        _log(f"unknown plot '{n}', skipping (see --list_plots)")
    det_result = None
    if any(PLOTS[n][1] for n in names):
        det_name = cfg.detector or "drone"
        if not cfg.detector:
            _log(f"a requested plot needs a detector; defaulting to '{det_name}' "
                 f"(set --detector to choose)")
        from gottlux.detectors import get_detector
        det = get_detector(det_name, freq_lo=cfg.freq_lo, freq_hi=cfg.freq_hi,
                           fft_fs=cfg.fft_fs, snr_thresh=cfg.snr_thresh, accum_dt=cfg.accum_dt)
        _log(f"running detector '{det_name}' for plot(s) ...")
        det_result = det.run(rec, cfg)
    for n in names:
        try:
            _log(f"plot: {n}")
            arts = PLOTS[n][0](rec, cfg, run, det_result)
            run.add_artifacts(arts)
        except Exception as e:
            import traceback
            _log(f"plot '{n}' failed: {e}")
            traceback.print_exc()
    run.record("plots", list(names))
    return det_result


# ====================================================================================
# Orchestration
# ====================================================================================
def run_recording(rec, cfg: Config, analyses=None) -> str:
    """Run analyses (or, if ``cfg.plots`` is set, just those specific figures) on an
    already-loaded *rec*; returns the run-folder path."""
    run = RunFolder(cfg, rec)
    _log(f"run folder: {run.path}")
    run.snapshot_source()
    det_result = None

    if cfg.plots:
        # specific-figure mode: output exactly the requested plots
        det_result = analysis_plots(rec, cfg, run, list(cfg.plots))
    else:
        analyses = list(analyses if analyses is not None else cfg.analyses)
        if cfg.detector and "detect" not in analyses:
            analyses.append("detect")
        for name in analyses:
            try:
                if name == "detect":
                    det_result = analysis_detect(rec, cfg, run)
                elif name == "metrics":
                    pass                          # run last, with detector result
                elif name == "performance":
                    analysis_performance(rec, cfg, run, det_result)   # reuse the detector run
                elif name in _ANALYSES:
                    _log(f"analysis: {name}")
                    _ANALYSES[name](rec, cfg, run)
                else:
                    _log(f"unknown analysis '{name}', skipping")
            except Exception as e:
                import traceback
                _log(f"analysis '{name}' failed: {e}")
                traceback.print_exc()

    # --- lab exports (CLI parity with the GUI) ---
    if cfg.export_cube:
        try:
            analysis_export_cube(rec, cfg, run)
        except Exception as e:
            _log(f"event-cube export failed: {e}")
    if cfg.make_report:
        try:
            det_result = analysis_report(rec, cfg, run, det_result)
        except Exception as e:
            _log(f"detection report failed: {e}")
    if cfg.make_performance and "performance" not in (cfg.analyses or ()):
        try:
            analysis_performance(rec, cfg, run, det_result)
        except Exception as e:
            _log(f"results metrics (performance) failed: {e}")

    try:
        analysis_metrics(rec, cfg, run, det_result)
    except Exception as e:
        _log(f"metrics failed: {e}")

    run.write_manifest()
    summary = run.write_summary()
    _log("done.\n" + summary)
    if cfg.open_when_done:
        open_in_file_browser(run.path)
    return run.path


def run_path(path, cfg: Config, analyses=None) -> str:
    """Load *path* and run the pipeline. Returns the run-folder path."""
    import gottlux as eb
    rec = eb.load(path, camera=cfg.camera, mode=cfg.mode,
                  progress=lambda f: None)
    if cfg.sensor_w is None:
        cfg.sensor_w = rec.width
    if cfg.sensor_h is None:
        cfg.sensor_h = rec.height
    rec.summary()
    return run_recording(rec, cfg, analyses)
