"""
rotation_pipeline.py — headless rotation analyses for spinning-sensor recordings.

These plug the ported rotation viz suite into the gottlux run folder so a single ``gottlux``
invocation reproduces the EBS rotation outputs alongside the staring-substrate ones:

================  =========================================================================
``radar``         360° polar radar map (PNG) + animated sweep (MP4, with ``--video``)
``mti``           moving-target-indication az/elev-vs-time panels (PNG)
``rate_surface``  event-rate relief video Z = ev/s (MP4, with ``--video``)
``validation``    3-panel RAW | TARGET | RADAR validation video (MP4, with ``--video``)
``masksweep``     successive-mask drop-off panoramas + montage + counts CSV
``tracking``      run the configured tracker(s) → regime-split report + overlay + enriched CSV
``fusion``        dual-EBS co-registration + fusion + mixed-FOV range cross-check (PNG)
``rotation_metrics`` EBS coverage / revisit / localization figures of merit (Markdown + JSON)
``rotor_ladder``  360° rotor-ladder survey: propeller-signature classification + where-else map +
                  range/bearing radar + cross-revolution motion offset + LaTeX report
``rotation_rate`` spin rate from the event-rate autocorrelation + event-rate/autocorrelation figure
``masking``       rotational background masking (§4.1): data-rate-reduction sweep vs N + de-rotated
                  moving-object map + bearing/range radar + volumetric (az,elev,range) point cloud
================  =========================================================================

All consume the shared :class:`~gottlux.rotation.RotationContext` (built once per recording).
"""
from __future__ import annotations

import os

from gottlux.rotation import build_context


def _log(msg):
    print(f"[gottlux] {msg}")


def _maybe_synthesize_telemetry(rec, cfg):
    """For a rotating clip with no azimuth CSV, synthesize telemetry from a steady spin
    (``cfg.assume_spin_hz``) so the rotation analyses can de-rotate. No-op otherwise."""
    if rec.telemetry is not None or getattr(cfg, "assume_spin_hz", None) is None:
        return
    from gottlux.io.telemetry import Telemetry, estimate_spin_period_s
    hz = cfg.assume_spin_hz
    if hz and hz > 0:
        period, conf = 1.0 / hz, None
    else:
        period, conf = estimate_spin_period_s(rec.t.astype(float) / 1e6)
    if not period:
        _log("assume_spin: could not estimate a rotation period; leaving as staring.")
        return
    rec.telemetry = Telemetry.from_spin(rec.duration_s, period)
    if cfg.mode != "rotation":
        cfg.mode = "rotation"
    conf_s = "" if conf is None else f", autocorr {conf}"
    _log(f"assume_spin: synthesized telemetry from period {period:.4f} s "
         f"({1.0 / period:.3f} Hz{conf_s}) — bearings are rotation-phase-relative.")


def _ctx(rec, cfg):
    """Build (and cache on the recording) the rotation front-end context."""
    c = getattr(rec, "_gottlux_rot_ctx", None)
    if c is None:
        _maybe_synthesize_telemetry(rec, cfg)
        c = build_context(rec, cfg)
        try:
            rec._gottlux_rot_ctx = c
        except Exception:
            pass
    return c


def analysis_radar(rec, cfg, run):
    from gottlux.rotation.viz import radar_map
    ctx = _ctx(rec, cfg)
    out = run.subdir("radar")
    arts = []
    if ctx.traj:
        p = radar_map.render_radar_map(ctx.traj, ctx.cfg, os.path.join(out, "radar_map.png"))
        if p:
            arts.append(p)
        if cfg.make_video:
            try:
                v = radar_map.render_radar_sweep(ctx.traj, ctx.cfg,
                                                 os.path.join(out, "radar_sweep.mp4"), tel=ctx.tel)
                if v:
                    arts.append(v)
            except Exception as e:
                _log(f"radar sweep video failed: {e}")
    run.record("radar", {"n_detections": 0 if ctx.dets is None else int(len(ctx.dets))})
    run.add_artifacts(arts)


def analysis_mti(rec, cfg, run):
    from gottlux.rotation.viz import mti
    ctx = _ctx(rec, cfg)
    out = run.subdir("mti")
    arts = []
    p = mti.render_mti(ctx.ev, ctx.keep, ctx.cfg, os.path.join(out, "mti.png"), tel=ctx.tel)
    if p:
        arts.append(p)
    run.add_artifacts(arts)


def analysis_rate_surface(rec, cfg, run):
    if not cfg.make_video:
        _log("rate_surface is a video; pass --video to render it. Skipping.")
        return
    from gottlux.rotation.viz import rate_surface
    ctx = _ctx(rec, cfg)
    out = run.subdir("rate_surface")
    p = rate_surface.run_rate_surface(ctx.ev, ctx.cfg, os.path.join(out, "rate_surface.mp4"))
    if p:
        run.add_artifacts([p])


def analysis_validation(rec, cfg, run):
    if not cfg.make_video:
        _log("validation is a video; pass --video to render it. Skipping.")
        return
    from gottlux.rotation.viz import validation_render
    ctx = _ctx(rec, cfg)
    out = run.subdir("validation")
    p = validation_render.run_validation(ctx.ev, ctx.keep, ctx.traj, ctx.cfg,
                                         os.path.join(out, "validation.mp4"), tel=ctx.tel)
    if p:
        run.add_artifacts([p])


def analysis_panorama_video(rec, cfg, run):
    if not cfg.make_video:
        _log("panorama video requested without --video; skipping.")
        return
    from gottlux.rotation.viz import panorama_video
    ctx = _ctx(rec, cfg)
    out = run.subdir("panorama_video")
    p = panorama_video.render_panorama_video(ctx.ev, ctx.cfg,
                                             os.path.join(out, "panorama_sweep.mp4"), tel=ctx.tel)
    if p:
        run.add_artifacts([p])


def analysis_masksweep(rec, cfg, run):
    from gottlux.rotation.viz import mask_sweep
    ctx = _ctx(rec, cfg)
    if ctx.tel is None:
        _log("masksweep needs rotation telemetry; skipping.")
        return
    out = run.subdir("masksweep")
    summary, arts = mask_sweep.render_mask_sweep(ctx.ev, ctx.tel, ctx.cfg, ctx.hot, out, rec.name)
    run.record("masksweep", summary)
    run.add_artifacts([p for p, _ in arts])


def analysis_tracking(rec, cfg, run):
    from gottlux.rotation import track_analysis, trackers
    from gottlux.rotation.viz import tracking_report
    ctx = _ctx(rec, cfg)
    names = [s.strip() for s in (cfg.tracker or "nearest").split(",") if s.strip()]
    for name in names:
        Tcls = trackers.get(name)
        if Tcls is None:
            _log(f"unknown tracker '{name}', skipping")
            continue
        _log(f"tracker: {name}")
        T = Tcls()
        out_res = T.track(ctx.traj or {}, ctx.cfg, ctx.tel, ev=ctx.ev)
        tracks = out_res.get("tracks", [])
        out = run.subdir(os.path.join("tracking", name))
        summary, arts = tracking_report.render_tracking_report(
            tracks, ctx.traj, ctx.ev, ctx.keep, ctx.cfg, ctx.tel,
            out, rec.name, tracker=name, video=cfg.make_video)
        run.record(f"tracking:{name}", summary)
        run.add_artifacts([p for p, _ in arts])


def analysis_fusion(rec, cfg, run):
    """Dual-EBS fusion: load the sibling camera from the same capture folder and fuse."""
    import gottlux as eb
    from gottlux.rotation import detect, fuse
    src = getattr(rec, "source_path", "") or ""
    folder = os.path.dirname(src) if os.path.isfile(src) else src
    if not folder or not os.path.isdir(folder):
        _log("fusion needs a capture folder with two cameras; skipping.")
        return
    # rec is camera A (cfg.camera); load the other camera
    cam_a = cfg.camera
    cam_b = "cam0" if cam_a != "cam0" else "cam1"
    try:
        rec_b = eb.load(folder, camera=cam_b, mode=cfg.mode)
    except Exception as e:
        _log(f"fusion: could not load second camera ({cam_b}): {e}; skipping.")
        return
    ctx_a = _ctx(rec, cfg)
    from gottlux.config import Config
    cfg_b = Config.from_dict(cfg.to_dict())
    cfg_b.camera = cam_b
    cfg_b.fov_deg = None                                   # let it resolve cam_b's FOV
    from gottlux.rotation import resolve_cfg, ev_dict, background
    cfg_b = resolve_cfg(rec_b, cfg_b)
    ev_b = ev_dict(rec_b.all())
    tel_b = rec_b.telemetry
    hot_b = background.hot_pixel_mask(ev_b, cfg_b.hot_pixel_pct)
    if cfg_b.mode == "rotation" and tel_b is not None and cfg_b.use_ref_mask:
        ref_end = background.reference_end_time(tel_b, cfg_b.mask_rotations)
        ref = background.build_reference(ev_b, tel_b, ref_end, n_phase=cfg_b.n_phase)
        drop_b = background.rotation_drop_mask(ev_b, tel_b, ref, cfg_b.n_phase, hot_b)
    else:
        drop_b = background.staring_drop_mask(ev_b, hot_b, cfg_b.bg_window_s)
    keep_b, dets_b = detect.isolate_target(ev_b, drop_b, cfg_b.accum_dt, cfg_b.min_pixels,
                                           cfg_b.cluster_dilation, cfg_b.cluster_erode)
    traj_b = detect.build_trajectory(dets_b, cfg_b, tel_b)
    if not ctx_a.traj or not traj_b:
        _log("fusion: one camera has no trajectory; skipping.")
        return
    out = run.subdir("fusion")
    summary = fuse.fuse_trajectories(ctx_a.traj, traj_b, os.path.join(out, "fusion.png"),
                                     name_a=cam_a, name_b=cam_b,
                                     gate_deg=cfg.fuse_gate_deg,
                                     bearing_offset_deg=cfg.az_offset_deg)
    run.record("fusion", summary)
    run.add_artifacts([os.path.join(out, "fusion.png")])


def analysis_rotor_ladder(rec, cfg, run):
    """The 360° rotor-ladder survey: classify the target via its propeller signature, then map
    every bearing the same rotor recurs at, with range, and the cross-revolution motion offset.

    The analysis box is the ``--roi`` (+ ``--t_start/--t_stop`` window) when given; otherwise the
    strongest detected cell becomes the template. Writes the figure suite + JSON + CSV + a
    compilable LaTeX report into the ``rotor_ladder`` run subfolder.
    """
    from gottlux.rotation import ladder_report, rotor_scan
    ctx = _ctx(rec, cfg)
    roi = cfg.roi
    res = rotor_scan.scan_context(ctx, roi=roi, t0=cfg.t_start, t1=cfg.t_stop)
    out = run.subdir("rotor_ladder")
    roi_s = (" --roi " + ",".join(str(v) for v in roi)) if roi else ""
    meta = {"recording": rec.name, "sensor_px": f"{rec.width}x{rec.height}",
            "reproduce": f"gottlux {rec.name}.raw --analyses rotor_ladder "
                         f"--blades {cfg.rotor_blades} --target_size {cfg.target_size_m:g}{roi_s}"}
    written = ladder_report.save_scan_report(out, res, cfg=cfg, meta=meta, dpi=cfg.fig_dpi,
                                             title=f"Rotor-ladder survey — {rec.name}")
    run.record("rotor_ladder", res.headline())
    run.add_artifacts(written)


def analysis_masking(rec, cfg, run):
    """First-class rotational background masking (the original MATLAB rotational-subtraction recipe): the data-rate-reduction sweep vs the
    reference depth N, the de-rotated moving-object map, a bearing×range radar, and the volumetric
    (azimuth, elevation, range) point cloud of the surviving movers. ``--mask_rotations`` sets N."""
    import numpy as np
    from gottlux.io import export
    from gottlux.rotation import background as bg, ev_dict, masking as mk, resolve_cfg
    from gottlux.rotation.viz import masking_viz, radar_map
    _maybe_synthesize_telemetry(rec, cfg)
    cfg = resolve_cfg(rec, cfg)
    if rec.telemetry is None:
        _log("masking needs rotation telemetry (real, or --assume_spin); skipping.")
        return
    ev, tel = ev_dict(rec.all()), rec.telemetry
    hot = bg.hot_pixel_mask(ev, cfg.hot_pixel_pct)
    deroted = mk._derotate(ev, tel, cfg)
    waz, elev = deroted[0], deroted[1]
    metrics = mk.sweep(ev, tel, cfg, (0, 1, 2, 3, 4), hot=hot, deroted=deroted)
    N = int(getattr(cfg, "mask_rotations", 2) or 2)
    keep = mk.keep_mask(ev, tel, cfg, N, hot=hot)
    movers = mk.extract_movers(ev, tel, cfg, keep, deroted=deroted)
    red = next((m.reduction_pct for m in metrics if m.n_rotations == N), float("nan"))
    result = mk.MaskingResult(metrics, movers, N, red, cfg.fov_deg,
                              round(float(getattr(tel, "T_rot", 0.0)), 5), cfg.target_size_m)
    out = run.subdir("masking")
    arts = []
    try:
        fig = masking_viz.masking_figure(result, waz[keep], elev[keep],
                                         title=f"Rotational masking — {rec.name}")
        arts += export.save_figure(fig, os.path.join(out, "masking"), dpi=cfg.fig_dpi,
                                   formats=(cfg.fig_format, "pdf"), close=True)
    except Exception as e:
        _log(f"masking figure failed: {e}")
    try:
        ah = [m for m in movers if m.above_horizon and m.range_m]
        if ah:
            traj = {"azimuth_deg": np.array([m.bearing_deg for m in ah]),
                    "range_m": np.array([m.range_m for m in ah]),
                    "altitude_z_m": np.array([m.range_m * np.tan(np.deg2rad(m.elev_deg)) for m in ah]),
                    "t": np.array([m.t_s for m in ah])}
            p = radar_map.render_radar_map(traj, cfg, os.path.join(out, "masking_radar.png"),
                                           title=f"Moving-object radar — {rec.name}")
            if p:
                arts.append(p)
    except Exception as e:
        _log(f"masking radar failed: {e}")
    export.save_json(result.as_dict(), os.path.join(out, "masking.json"))
    arts += export.save_table(mk.movers_table(result), os.path.join(out, "masking_movers"))
    run.record("masking", result.headline())
    run.add_artifacts(arts)


def analysis_rotation_rate(rec, cfg, run):
    """Recover the rotation rate from the event-rate autocorrelation and write a high-quality
    event-rate-vs-time + autocorrelation figure (the spin-frequency the FFT-derotation keys on)."""
    from gottlux.rotation import rate_analysis
    out = run.subdir("rotation_rate")
    res = rate_analysis.find_rotation_rate(rec)
    written = rate_analysis.save_rotation_rate_report(rec, out, dpi=cfg.fig_dpi)
    run.record("rotation_rate", {"hz": res["hz"], "period_s": res["period_s"],
                                 "confidence": res["confidence"],
                                 "telemetry_period_s": res["telemetry_period_s"]})
    run.add_artifacts(written)


def analysis_rotation_metrics(rec, cfg, run):
    from gottlux.rotation import metrics as rmet
    ctx = _ctx(rec, cfg)
    out = run.subdir("rotation_metrics")
    res = rmet.compute(ctx.ev, ctx.keep, ctx.traj, ctx.centroids, ctx.cfg, ctx.tel)
    md = rmet.to_markdown(res, rec.name)
    with open(os.path.join(out, "rotation_metrics.md"), "w", encoding="utf-8") as f:
        f.write(md)
    from gottlux.io import export
    export.save_json(res, os.path.join(out, "rotation_metrics.json"))
    run.record("rotation_metrics", res)


#: name -> analysis function, merged into pipeline._ANALYSES
ROTATION_ANALYSES = {
    "radar": analysis_radar,
    "mti": analysis_mti,
    "rate_surface": analysis_rate_surface,
    "validation": analysis_validation,
    "panorama_video": analysis_panorama_video,
    "masksweep": analysis_masksweep,
    "tracking": analysis_tracking,
    "fusion": analysis_fusion,
    "rotation_metrics": analysis_rotation_metrics,
    "rotor_ladder": analysis_rotor_ladder,
    "rotation_rate": analysis_rotation_rate,
    "masking": analysis_masking,
}
