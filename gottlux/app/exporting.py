"""
exporting.py — turn the current view into a saved figure or a data block/cube.

Two kinds of export, used by the per-tab **Export** buttons:

* **Figures** — render the current view as a journal-grade figure (PNG + vector PDF) via the
  :mod:`gottlux.viz` builders, or grab a 3-D snapshot straight from the OpenGL view.
* **Data cubes** — write the underlying numbers as a reusable block: the **space-time event
  cube** ``V[y, x, t]`` (a voxelized ``(x, y, t)`` volume over the current window), the
  **flicker-map cube** (per-cell dominant-frequency / SNR / power arrays), or a detections
  table. Cubes are saved as compressed ``.npz`` **and** ``.h5`` with axis vectors and metadata,
  so they load cleanly in NumPy, MATLAB, or any HDF5 tool.
"""
from __future__ import annotations

import os

import numpy as np

from gottlux.io import export


# ====================================================================================
# Data cubes
# ====================================================================================
def event_cube(rec, t0, t1, nt=64, spatial_bin=1, mode="count"):
    """Voxelize events in ``[t0, t1]`` into a space-time cube ``V[y, x, t]``.

    Parameters
    ----------
    nt : int            number of time slices along the cube's depth axis
    spatial_bin : int   pixel binning (1 = full sensor resolution; 2 = half, …)
    mode : str          ``"count"`` (events/voxel) or ``"polarity"`` (ON − OFF per voxel)

    Returns ``(cube, meta)`` where meta has the axis vectors (``x_px, y_px, t_s``) and shape.
    """
    win = rec.window(t0, t1)
    W, H = rec.width, rec.height
    nx = (W + spatial_bin - 1) // spatial_bin
    ny = (H + spatial_bin - 1) // spatial_bin
    nt = max(int(nt), 1)
    cube = np.zeros((ny, nx, nt), np.float32)
    if win.n:
        x = np.asarray(win.x) // spatial_bin
        y = np.asarray(win.y) // spatial_bin
        ts = win.t_s
        span = max(ts[-1] - ts[0], 1e-9)
        ti = np.clip(((ts - ts[0]) / span * nt).astype(np.int64), 0, nt - 1)
        flat = (y.astype(np.int64) * nx + x.astype(np.int64)) * nt + ti
        if mode == "polarity":
            w = np.where(np.asarray(win.p) == 1, 1.0, -1.0)
            cube = np.bincount(flat, weights=w, minlength=ny * nx * nt).reshape(ny, nx, nt).astype(np.float32)
        else:
            cube = np.bincount(flat, minlength=ny * nx * nt).reshape(ny, nx, nt).astype(np.float32)
    meta = dict(t0_s=float(t0), t1_s=float(t1), nt=nt, spatial_bin=spatial_bin,
                mode=mode, width=W, height=H, shape=list(cube.shape),
                x_px=(np.arange(nx) * spatial_bin).tolist(),
                y_px=(np.arange(ny) * spatial_bin).tolist(),
                t_s=np.linspace(t0, t1, nt).tolist())
    return cube, meta


def save_event_cube(path_base, rec, t0, t1, nt=64, spatial_bin=1, mode="count"):
    """Build and save a space-time event cube as ``.npz`` + ``.h5``. Returns paths written."""
    cube, meta = event_cube(rec, t0, t1, nt=nt, spatial_bin=spatial_bin, mode=mode)
    written = export.save_arrays(path_base + "_eventcube",
                                 cube=cube, x_px=np.array(meta["x_px"]),
                                 y_px=np.array(meta["y_px"]), t_s=np.array(meta["t_s"]))
    written += export.save_hdf5(path_base + "_eventcube",
                                {"cube": cube, "x_px": np.array(meta["x_px"]),
                                 "y_px": np.array(meta["y_px"]), "t_s": np.array(meta["t_s"])},
                                {k: v for k, v in meta.items()
                                 if k not in ("x_px", "y_px", "t_s")})
    written += export.save_json(meta, path_base + "_eventcube_meta.json")
    return written


def save_flicker_cube(path_base, flicker_map):
    """Save a flicker-map's per-cell arrays (dominant freq / SNR / power / counts)."""
    fm = flicker_map
    written = export.save_arrays(path_base + "_flickermap",
                                 dominant_freq=fm.dominant_freq, snr=fm.snr,
                                 band_power=fm.band_power, event_count=fm.event_count)
    written += export.save_hdf5(path_base + "_flickermap",
                                {"dominant_freq": fm.dominant_freq, "snr": fm.snr,
                                 "band_power": fm.band_power, "event_count": fm.event_count},
                                {"fmin": fm.band[0], "fmax": fm.band[1],
                                 "cell": fm.cell, "fs": fm.fs})
    return written


# ====================================================================================
# Figures / snapshots
# ====================================================================================
def save_gl_snapshot(gl_view, path):
    """Grab the OpenGL 3-D view's framebuffer to a PNG. Returns ``[path]`` or ``[]``."""
    try:
        img = gl_view.grabFramebuffer()
        if not path.lower().endswith(".png"):
            path += ".png"
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        img.save(path)
        return [path]
    except Exception:
        return []


# ====================================================================================
# Overall, selectable bundle export (the "choose exactly what to export" dialog backend)
# ====================================================================================
#: Every artifact the bundle exporter can produce, with a human label (drives the dialog).
BUNDLE_ITEMS = {
    "frame_fig": "Event-frame figure (PNG + PDF)",
    "event_cube": "Space-time event cube V[y,x,t] (NPZ + HDF5 + JSON)",
    "event_rate": "Event-rate-vs-time series (CSV/Parquet)",
    "flicker_fig": "Flicker-map figure (PNG + PDF)",
    "flicker_cube": "Flicker-map cube (NPZ + HDF5)",
    "spectrum": "Region spectrum series (CSV/Parquet)",
    "detections": "Detections table (CSV/Parquet)",
    "report": "Detection report (Markdown + JSON, first-principles)",
    "config": "Run configuration + provenance manifest (JSON)",
    "infographic": "Context infographic poster (PNG) — a representative frame + the capture context",
    "video": "Analysis video (MP4) — the current view rendered over the window with its settings",
}


def _video_target_size(sensor_wh, res):
    """``(w, h)`` for a video resolution choice given the source ``sensor_wh``."""
    if not sensor_wh:
        return None
    w, h = int(sensor_wh[0]), int(sensor_wh[1])
    if res == "native":
        return (w, h)
    if res in ("720p", "1080p"):
        H = 720 if res == "720p" else 1080
        return (round(w * H / h), H)
    if res in ("2x", "4x"):
        m = 2 if res == "2x" else 4
        return (w * m, h * m)
    return (w, h)


#: Output purpose tags (the export "class"). A label recorded in the manifest for your own
#: file tracking — it organizes/annotates an export without changing what is produced.
PURPOSES = ("research", "demo", "graphic")


def export_bundle(out_dir, rec, t0, t1, *, want, mode="count", cmap="inferno", expr="sqrt",
                  flicker_map=None, spectrum=None, result=None, cfg=None,
                  nt=64, spatial_bin=1, window_label="window", purpose="research", note="",
                  render=None, sensor_wh=None, fps=25, accum=0.02,
                  video_res="1080p", video_banner=True):
    """Write a *selected* set of artifacts to ``out_dir``; return ``(written, manifest)``.

    ``want`` is an iterable of keys from :data:`BUNDLE_ITEMS`. Only the requested artifacts are
    produced (and only those for which the needed inputs — a flicker map, a detector result —
    were supplied), so the caller (the global Export dialog) controls exactly what lands on
    disk. ``purpose`` (one of :data:`PURPOSES`) and a free-text ``note`` are recorded in the
    manifest for your own tracking. A ``manifest.json`` recording the selection, window, purpose,
    note and file list is always written.
    """
    from gottlux.core.accumulate import accumulate_frame
    from gottlux.io.paths import unique_export_dir
    want = set(want)
    # land in a uniquely + helpfully named subfolder inside the chosen folder
    out_dir = unique_export_dir(out_dir, rec.name, purpose)
    base = os.path.join(out_dir, rec.name)
    written = []
    produced = []

    def _did(key, paths):
        if paths:
            written.extend(paths)
            produced.append(key)

    if "frame_fig" in want:
        try:
            from gottlux.viz import frames
            frame = accumulate_frame(rec.window(t0, t1), mode=mode)
            fig = frames.event_frame_figure(frame, mode=mode, cmap=cmap,
                                            title=f"{rec.name} @ {t0:.3f}s ({mode})")
            _did("frame_fig", export.save_figure(fig, base + "_frame", dpi=300,
                                                 formats=("png", "pdf"), close=True))
        except Exception as e:
            print(f"[export_bundle] frame_fig failed: {e}")
    if "event_cube" in want:
        _did("event_cube", save_event_cube(base, rec, t0, t1, nt=nt,
                                            spatial_bin=spatial_bin, mode=mode))
    if "event_rate" in want:
        c, r = rec.event_rate(bin_s=max((t1 - t0) / 800.0, 1e-3))
        _did("event_rate", export.save_table({"t_s": c, "rate_evs": r}, base + "_event_rate"))
    if "flicker_fig" in want and flicker_map is not None:
        try:
            from gottlux.viz import spectral
            bg = accumulate_frame(rec.window(t0, t1), mode="count")
            fig = spectral.flicker_map_figure(flicker_map, background=bg,
                                              title=f"Flicker map — {rec.name}")
            _did("flicker_fig", export.save_figure(fig, base + "_flicker", close=True))
        except Exception as e:
            print(f"[export_bundle] flicker_fig failed: {e}")
    if "flicker_cube" in want and flicker_map is not None:
        _did("flicker_cube", save_flicker_cube(base, flicker_map))
    if "spectrum" in want and spectrum is not None and spectrum.freqs.size:
        _did("spectrum", export.save_table(
            {"freq_hz": spectrum.freqs, "power": spectrum.power}, base + "_spectrum"))
    if "detections" in want and result is not None and result.targets:
        _did("detections", _save_detections(base, result))
    if "report" in want and result is not None:
        from gottlux.run import report as _report
        _did("report", _report.save_detection_report(base, result, rec, cfg=cfg,
                                                      window=(t0, t1)))
    if "infographic" in want:
        try:
            from gottlux.viz import video as _vid
            rgb = _vid.colorize(accumulate_frame(rec.window(t0, t1), mode=mode), cmap=cmap, expr=expr)
            fields = {"Recording": rec.name, "Sensor": f"{rec.width}×{rec.height} px · {rec.fmt}",
                      "Geometry": "rotation" if rec.is_rotating else "staring",
                      "Events": f"{rec.n:,}", "Window": f"{t0:.3f}–{t1:.3f} s",
                      "Accum mode": mode, "Purpose": purpose}
            if note:
                fields["Note"] = note
            poster = _vid.context_poster(rgb, f"{rec.name} — context", fields)
            from PIL import Image
            Image.fromarray(poster).save(base + "_context.png")
            _did("infographic", [base + "_context.png"])
        except Exception as e:
            print(f"[export_bundle] infographic failed: {e}")
    if "video" in want and render is not None:
        try:
            from gottlux.viz import video as _vid
            from gottlux.app.transport import REALTIME_FPS
            size = _video_target_size(sensor_wh, video_res)
            # fps is an equivalent (slow-mo) capture rate: sample at it, write at real-time
            # cadence so the clip plays slowed by fps/30 (matches the viewer).
            out_fps = REALTIME_FPS
            nfr = int(np.clip(round((t1 - t0) * fps), 1, 18000))
            times = t0 + (np.arange(nfr) + 0.5) * (t1 - t0) / max(nfr, 1)

            def _frames():
                for tt in times:
                    rgb = render(float(tt), accum, size)
                    if rgb is None:
                        continue
                    if video_banner:
                        rgb = _vid.infographic_frame(
                            rgb, title=rec.name, subtitle=f"{mode} · {cmap} · {expr}",
                            footer_lines=[f"t = {tt:.3f} s   ·   window {t0:.2f}–{t1:.2f} s"])
                    yield rgb
            p = _vid.write_video(base + "_video.mp4", _frames(), fps=out_fps)
            _did("video", [p] if p else [])
        except Exception as e:
            print(f"[export_bundle] video failed: {e}")
    if "config" in want:
        meta = dict(recording=rec.name, source=rec.source_path, encoding=rec.fmt,
                    width=rec.width, height=rec.height, n_events=rec.n,
                    duration_s=rec.duration_s, window_s=[t0, t1], accum_mode=mode,
                    expression=expr, colormap=cmap)
        if cfg is not None:
            try:
                meta["config"] = cfg.to_dict()
            except Exception:
                pass
        _did("config", export.save_json(meta, base + "_config.json"))

    manifest = dict(recording=rec.name, out_dir=out_dir, window_s=[float(t0), float(t1)],
                    purpose=(purpose if purpose in PURPOSES else "research"),
                    note=str(note or ""),
                    requested=sorted(want), produced=sorted(produced),
                    files=[os.path.basename(p) for p in written])
    written += export.save_json(manifest, os.path.join(out_dir, "manifest.json"))
    return written, manifest


def _save_detections(base, result):
    """Flatten a DetectorResult's tracks to a per-detection table and save it."""
    from gottlux.detectors.base import detections_table
    return export.save_table(detections_table(result), base + "_detections")
