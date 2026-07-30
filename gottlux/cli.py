"""
cli.py — the ``gottlux`` command-line entry point.

Behaviour mirrors the philosophy of the whole tool: *do the obvious thing*.

* **No path** → open the GUI dashboard.
* **A path** → run the headless pipeline on it (decode-once, analyses, run folder).
* ``--gui`` with a path → open the GUI pre-pointed at that path.
* ``--list_detectors`` / ``--list_signatures`` / ``--list_plots`` → print a registry and exit.

The CLI is at **parity with the GUI's laboratory features**: it can voxelize a space-time
**event cube** (``--export-cube``), write a first-principles **detection report**
(``--report``), and switch the spectral transform to a **non-uniform FFT** (``--nufft``) with
optional **whitening** (``--whiten``). ``gottlux -h`` prints every option with examples.

Every :class:`~gottlux.config.Config` field that matters from the command line has a flag;
the rest keep their documented defaults.
"""
from __future__ import annotations

import argparse
import sys

from gottlux import __version__
from gottlux.config import Config

_EPILOG = r"""
examples
--------
  gottlux                                   open the GUI dashboard
  gottlux capture\cam0.raw                  decode + standard analyses -> a run folder
  gottlux capture\cam0.raw --gui            open the GUI pre-pointed at a file
  gottlux capture\cam0.raw --view           open the lightweight quick viewer for one clip
  gottlux --register-raw                     make double-clicking a .raw open the quick viewer
  gottlux file.raw --detector drone --report
                                            run the drone detector + write a first-principles report
  gottlux file.raw --export-cube --cube_nt 96 --t_start 1.0 --t_stop 1.5
                                            voxelize a 0.5 s window into a 96-slice event cube
  gottlux file.raw --plots flicker_map,spectrum --nufft --whiten median
                                            only those figures, using a non-uniform FFT + whitening
  gottlux file.raw --detector insect --freq_lo 30 --freq_hi 250 --snr 5
                                            tune the band/gate for an insect wingbeat
  gottlux staring.raw --performance --target_size 0.22 --ttc_speed 15
                                            results metrics (tracking/prop-freq range + TTC) beside the file
  gottlux staring.raw --performance --compare-with rotating.raw
                                            also emit a staring-vs-rotating comparison
  gottlux capture\cam0.raw --to-hdf5        convert to a Metavision-compatible HDF5 (CD/events);
                                            .h5 files open everywhere a .raw does
  gottlux file.raw --export-tool region_spectrum --roi 120,90,200,170 --freq_lo 90 --freq_hi 700
                                            export a standalone Python+MATLAB spectrum tool
                                            (data.h5 + scripts, no gottlux needed to run them)
  gottlux --export-tool list                print the exportable standalone tools
  gottlux file.raw --run-script my_script.py --t_start 1 --t_stop 2
                                            run a custom process(win, ctx) on that window;
                                            returns land in a provenance-stamped folder
  gottlux --list_detectors                  print the detector registry and exit
  gottlux --list_plots                      print the individually-exportable figures
  gottlux --cache-info captures\            report decode-cache disk usage under a folder
  gottlux --clear-cache --stale-only        reclaim only the out-of-date caches under cwd

outputs land in a reproducible run folder (manifest + input SHA-256 + environment + source
snapshot); figures are saved as raster AND vector at 300 DPI, data as Parquet/CSV/NPZ/HDF5.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gottlux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Event-based-sensor (neuromorphic) analysis & visualization instrument. "
                    "No path → GUI dashboard; a path → headless analysis into a run folder.",
        epilog=_EPILOG)
    p.add_argument("path", nargs="?",
                   help="a .raw or .h5/.hdf5 file, a capture folder, or a decoded cache stem")
    p.add_argument("--version", action="version", version=f"gottlux {__version__}")
    p.add_argument("--gui", action="store_true", help="open the GUI (pre-pointed at PATH if given)")
    p.add_argument("--view", action="store_true",
                   help="open PATH in the lightweight quick viewer (a single fast live view, "
                        "with a button to hand it on to the full suite) instead of analyzing it")
    p.add_argument("--headless", action="store_true", help="force headless even with no analyses")
    p.add_argument("--no-preview", action="store_true",
                   help="GUI/viewer: always decode fully before showing — disable the sampled "
                        "fast preview a large uncached .raw normally opens with (equivalent to "
                        "GOTTLUX_PREVIEW_THRESHOLD_MB=0)")
    p.add_argument("--register-raw", action="store_true",
                   help="register the quick viewer as the .raw file handler (per-user; Windows "
                        "registry or Linux XDG) so double-clicking a .raw opens it; then exit")
    p.add_argument("--unregister-raw", action="store_true", help="undo --register-raw and exit")

    g = p.add_argument_group("capture")
    g.add_argument("--mode", choices=["auto", "rotation", "staring"], default="auto",
                   help="capture geometry (auto = rotation iff telemetry present)")
    g.add_argument("--camera", default="cam0", help="which camera when a folder has several")
    g.add_argument("--sensor", default=None, metavar="PROFILE",
                   help="hardware profile (sensor + optics) for clips on different gear; "
                        "e.g. genx320 (default), imx636. See --list_sensors")
    g.add_argument("--fov_deg", type=float, default=None,
                   help="horizontal field of view (degrees); overrides the sensor profile")
    g.add_argument("--target_size", type=float, default=None,
                   help="target physical size (m) for absolute ranging")
    g.add_argument("--blades", type=int, default=None, metavar="N",
                   help="blades per rotor (rotor_ladder: blade-pass Hz → rotor RPM; default 2)")
    g.add_argument("--prop_diameter", type=float, default=None, metavar="M",
                   help="propeller diameter (m) for the rotor tip-speed readout (default 0.127 = 5\")")
    g.add_argument("--list_sensors", action="store_true",
                   help="list the built-in sensor/camera profiles and exit")
    g.add_argument("--force_decode", action="store_true", help="ignore any existing decode cache")
    g.add_argument("--assume_spin", default=None, metavar="auto|HZ",
                   help="rotating clip with NO telemetry CSV: synthesize telemetry from a steady "
                        "spin so the rotation analyses can de-rotate. 'auto' estimates the period "
                        "from the event-rate periodicity; a number sets the spin rate (Hz). "
                        "Bearings become rotation-phase-relative (absolute North uncalibrated).")
    g.add_argument("--mask_rotations", type=int, default=None, metavar="N",
                   help="ROTATION background suppression: build the frozen phase-space reference "
                        "from the first N revolutions (default 1). N=2-3 suppresses more static "
                        "clutter and amplifies the moving target (>90%% data-rate reduction at "
                        "N=3 on the field data), at the cost of mild self-masking of a slow target.")

    g = p.add_argument_group("window of interest")
    g.add_argument("--t_start", type=float, default=None, help="restrict to events at/after this time (s)")
    g.add_argument("--t_stop", type=float, default=None, help="restrict to events before this time (s)")
    g.add_argument("--roi", default=None, metavar="x0,y0,x1,y1",
                   help="restrict to a sensor sub-rectangle in pixels")

    g = p.add_argument_group("analyses")
    g.add_argument("--analyses", default="overview,spectral,panorama",
                   help="comma list of analyses. core: overview,spectral,panorama,detect,metrics "
                        "· rotation (ported from EBS): radar,mti,rate_surface,validation,"
                        "panorama_video,masksweep,tracking,fusion,rotation_metrics,rotor_ladder,"
                        "rotation_rate,masking "
                        "(use '' for none, e.g. with --export-cube)")
    g.add_argument("--plots", default=None,
                   help="output ONLY these specific figures (comma list); see --list_plots")
    g.add_argument("--list_plots", action="store_true", help="list available --plots names and exit")
    g.add_argument("--list_analyses", action="store_true", help="list available --analyses and exit")
    g.add_argument("--accum_dt", type=float, default=None, help="accumulation window (s)")
    g.add_argument("--tracker", default=None,
                   help="comma list of EBS trackers for the 'tracking' analysis "
                        "(nearest,single,kalman,cmax,staring_kvf,hummingbird,drone_fft)")
    g.add_argument("--video", action="store_true",
                   help="also render the video analyses (rate_surface, validation, radar sweep, "
                        "panorama_video, tracker overlay) — slower")

    g = p.add_argument_group("spectral transform")
    g.add_argument("--nufft", action="store_true",
                   help="use a non-uniform FFT (straight from event times; no Nyquist ceiling) "
                        "for the region/global spectrum instead of the binned FFT")
    g.add_argument("--whiten", choices=["none", "median", "zscore"], default="none",
                   help="spectral whitening to emphasize peaking over colored noise (default none)")

    g = p.add_argument_group("detection")
    g.add_argument("--detector", default=None, help="run a registered detector (e.g. drone, insect)")
    g.add_argument("--freq_lo", type=float, default=None, help="flutter band low (Hz)")
    g.add_argument("--freq_hi", type=float, default=None, help="flutter band high (Hz)")
    g.add_argument("--fft_fs", type=float, default=None, help="FFT sample rate (Hz)")
    g.add_argument("--snr", type=float, default=None, help="flutter SNR acceptance gate")
    g.add_argument("--list_detectors", action="store_true", help="list registered detectors and exit")
    g.add_argument("--list_signatures", action="store_true", help="list flutter signatures and exit")

    g = p.add_argument_group("lab exports (GUI parity)")
    g.add_argument("--export-cube", "--export_cube", dest="export_cube", action="store_true",
                   help="also voxelize the (windowed) stream into a space-time event cube "
                        "V[y,x,t] (NPZ + HDF5 + JSON)")
    g.add_argument("--cube_nt", type=int, default=64, help="event-cube time slices (default 64)")
    g.add_argument("--cube_bin", type=int, default=1, help="event-cube spatial pixel binning (default 1)")
    g.add_argument("--report", action="store_true",
                   help="also write a first-principles detection report (Markdown + JSON); "
                        "runs the detector (default 'drone') if --detector is not set")

    g = p.add_argument_group("results metrics (KPIs)")
    g.add_argument("--performance", "--kpi", dest="performance", action="store_true",
                   help="compute the operator results metrics — tracking range, prop-frequency "
                        "range, time-to-contact — and save a paper-ready bundle beside the file")
    g.add_argument("--ttc_speed", "--approach_speed", dest="ttc_speed", type=float, default=None,
                   metavar="MPS", help="nominal closing speed (m/s) for time-to-contact "
                                       "(default 15); the measured TTC is separate")
    g.add_argument("--known_range", type=float, default=None, metavar="M",
                   help="a known constant ground-truth range (m) to overlay/validate against")
    g.add_argument("--truth_csv", default=None, metavar="CSV",
                   help="CSV with 't_s,range_m' columns of logged ground-truth ranges to overlay")
    g.add_argument("--compare-with", "--compare_with", dest="compare_with", default=None,
                   metavar="PATH", help="also run on this second recording and emit a "
                                        "staring-vs-rotating comparison")

    g = p.add_argument_group("track study (single-clip tracker + KPI + figures/videos)")
    g.add_argument("--track-study", "--track_study", dest="track_study", action="store_true",
                   help="run the single-clip track study on PATH: a tracker (default "
                        "single_centroid) + the three KPIs, writing the tracking figures "
                        "(range_vs_time_full, lock_score, track_dashboard, track_fft_dashboard) and "
                        "the tracked-overlay MP4 into a folder beside the file. Honours --fov_deg / "
                        "--sensor, --target_size, --accum_dt (default 0.085), --detector, "
                        "--freq_lo/--freq_hi, --t_start/--t_stop, --out.")
    g.add_argument("--fft-video", "--fft_video", dest="fft_video", action="store_true",
                   help="with --track-study, also render the live track + rotor-FFT demo video")

    g = p.add_argument_group("EBS + acoustic fusion (align a .raw to a .wav)")
    g.add_argument("--fusion", action="store_true",
                   help="co-register PATH (an EBS .raw) with a time-synced audio --audio .wav: "
                        "recover the temporal offset (event-rate vs RMS envelope), write the "
                        "aligned pair (.raw + .wav, shared t=0) into clips/, and emit the "
                        "cross-domain spectra + spectrogram + report into a folder beside PATH. "
                        "Honours --freq_lo/--freq_hi (tonal band), --offset, --out.")
    g.add_argument("--audio", default=None, metavar="WAV",
                   help="the time-synchronized audio .wav to fuse with PATH (for --fusion)")
    g.add_argument("--offset", type=float, default=None, metavar="S",
                   help="force the audio→EBS offset (s) instead of auto cross-correlation")

    g = p.add_argument_group("editing (cut / stitch .raw)")
    g.add_argument("--trim", default=None, metavar="T0[,T1]",
                   help="batch-trim: cut EVERY .raw in the PATH folder (or just the PATH file) to "
                        "[T0,T1] s (T1 optional → to the end), re-based to a COMMON origin so "
                        "synchronized clips stay slate-aligned, into a 'trimmed/' subfolder. "
                        "e.g. `gottlux capture_dir --trim 3` drops each clip's first 3 s")
    g.add_argument("--cut", default=None, metavar="T0,T1",
                   help="cut [t0,t1] seconds of PATH to a new .raw. An EVT2.1 file (no --roi) is cut "
                        "directly on the bytes — no decode, bounded RAM; --roi / EVT2 / EVT3 decode")
    g.add_argument("--stitch", default=None, metavar="P2,P3,...",
                   help="stitch PATH + these clips (comma list) end-to-end into one .raw")
    g.add_argument("--stitch_gap", type=float, default=0.0, metavar="S",
                   help="blank gap (s) inserted between stitched clips (default 0)")
    g.add_argument("--out_raw", default=None, help="output .raw path for --cut / --stitch")

    g = p.add_argument_group("conversion (.raw → HDF5)")
    g.add_argument("--to-hdf5", "--to_hdf5", dest="to_hdf5", nargs="?", const="", default=None,
                   metavar="OUT",
                   help="convert PATH to a Metavision-compatible HDF5 event file (compound "
                        "CD/events dataset, gzip-chunked, streamed in bounded blocks) and exit. "
                        "OUT defaults to the input stem + '.h5'. Honours --t_start/--t_stop/--roi "
                        "to export a sub-clip. gottlux opens .h5 files everywhere it opens .raw")

    g = p.add_argument_group("standalone tool export (take an algorithm OUT of gottlux)")
    g.add_argument("--export-tool", "--export_tool", dest="export_tool", default=None,
                   metavar="NAME",
                   help="export NAME as a self-contained tool bundle beside the file: data.h5 "
                        "(the recording, honoring --t_start/--t_stop/--roi) + run_NAME.py "
                        "(numpy/scipy/h5py only — no gottlux import) + run_NAME.m (base "
                        "MATLAB, native h5read) + a README, with your current band/window "
                        "values baked in as editable variables. 'list' prints the tools")
    g.add_argument("--tool-format", "--tool_format", dest="tool_format",
                   choices=["python", "matlab", "both"], default="both",
                   help="which script(s) to write into the bundle (default both)")
    g.add_argument("--tool-out", "--tool_out", dest="tool_out", default=None, metavar="DIR",
                   help="parent directory for the bundle folder (default: beside INPUT)")
    g.add_argument("--viz_mode", "--viz-mode", dest="viz_mode",
                   choices=["count", "polarity"], default="count",
                   help="viz_config bundles: accumulation mode baked into the exported "
                        "scripts — per-pixel event count, or signed ON−OFF polarity "
                        "(default count)")
    g.add_argument("--viz_cmap", "--viz-cmap", dest="viz_cmap", default=None, metavar="NAME",
                   help="viz_config bundles: matplotlib colormap baked into the Python "
                        "script (the nearest base-MATLAB equivalent goes into the .m); "
                        "default inferno, or coolwarm for --viz_mode polarity")
    g.add_argument("--viz_tonemap", "--viz-tonemap", dest="viz_tonemap",
                   choices=["linear", "sqrt", "gamma", "log", "asinh", "percentile"],
                   default=None,
                   help="viz_config bundles: tone curve applied before the colormap; "
                        "default sqrt, or linear for --viz_mode polarity")
    g.add_argument("--viz_accum_ms", "--viz-accum-ms", dest="viz_accum_ms", type=float,
                   default=None, metavar="MS",
                   help="viz_config bundles: accumulation window per rendered frame in "
                        "milliseconds (default 20)")

    g = p.add_argument_group("user script (run custom Python on the recording)")
    g.add_argument("--run-script", "--run_script", dest="run_script", default=None,
                   metavar="FILE.py",
                   help="run FILE.py's process(win, ctx) on INPUT — honouring --t_start/"
                        "--t_stop/--roi, so the script sees exactly the requested slice — "
                        "and save what it returns (dict of arrays → NPZ; matplotlib Figure "
                        "→ PNG+PDF; {'events': (x,y,p,t)} → a derived .raw) into a stamped "
                        "folder with a provenance README (script + input SHA-256, window, "
                        "version). See examples/user_script_example.py")
    g.add_argument("--script-args", "--script_args", dest="script_args", default=None,
                   metavar="STR",
                   help="extra whitespace-separated tokens forwarded to the script as "
                        "ctx['args'] (e.g. --script-args \"0.02 fast\")")

    g = p.add_argument_group("decode-cache management")
    g.add_argument("--cache-info", "--cache_info", dest="cache_info", action="store_true",
                   help="report the decode caches under PATH (a file, a capture folder, or "
                        "the relocation root; default: the current directory) — size, event "
                        "count, decoder version, staleness — and exit")
    g.add_argument("--clear-cache", "--clear_cache", dest="clear_cache", action="store_true",
                   help="delete the decode caches under PATH (default: the current directory) "
                        "and exit; never touches the source data, and files memmapped by a "
                        "running session are skipped and reported")
    g.add_argument("--stale-only", "--stale_only", dest="stale_only", action="store_true",
                   help="with --clear-cache: only remove stale caches (source changed, "
                        "decoder/layout changed, or incomplete) — keep fresh ones")

    g = p.add_argument_group("output")
    g.add_argument("--out", default=None, help="output root for the run folder")
    g.add_argument("--no_open", action="store_true", help="don't open the run folder when done")
    return p


def _ground_truth_from_args(a):
    """Build a ground-truth range input from ``--known_range`` (constant) or ``--truth_csv``."""
    if a.known_range is not None:
        return float(a.known_range)
    if a.truth_csv:
        import csv
        t, r = [], []
        with open(a.truth_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                keys = {k.lower().strip(): k for k in row}
                tk = keys.get("t_s") or keys.get("t") or keys.get("time_s")
                rk = keys.get("range_m") or keys.get("range") or keys.get("distance_m")
                if tk and rk:
                    try:
                        t.append(float(row[tk])); r.append(float(row[rk]))
                    except (TypeError, ValueError):
                        pass
        if r:
            return {"t": t, "range_m": r}
    return None


def _run_performance_cli(args, cfg):
    """The dedicated results-metrics process: compute the KPIs and save beside the file(s)."""
    import os
    import gottlux as eb
    from gottlux.run import performance_report as pr
    gt = _ground_truth_from_args(args)
    if not args.compare_with:
        pr.run_performance(args.path, cfg, ground_truth=gt)
        return 0
    # comparison: run on both recordings, then overlay (e.g. staring vs rotating)
    results, labels = [], []
    for path in (args.path, args.compare_with):
        c = _config_from_args(args)
        rec = eb.load(path, camera=c.camera, mode=c.mode, progress=lambda f: None)
        if c.sensor_w is None:
            c.sensor_w = rec.width
        if c.sensor_h is None:
            c.sensor_h = rec.height
        res = pr.compute_performance(rec, c, ground_truth=pr._expand_ground_truth(gt, rec, c),
                                     approach_speed=c.approach_speed_mps)
        pr.save_performance(res, rec, c)
        results.append(res)
        labels.append(f"{res.regime}:{res.name}")
    base = os.path.dirname(os.path.abspath(args.path)) or "."
    pr.compare_performance(results, labels=labels,
                           out_dir=os.path.join(base, "kpi_comparison"), cfg=cfg)
    return 0


def _run_trim_cli(args):
    """Headless batch-trim (``--trim``): shorten every clip in a folder to a common window."""
    import os
    from gottlux.io import writer
    from gottlux.io.paths import open_in_file_browser
    parts = [float(v) for v in str(args.trim).replace(" ", "").split(",") if v != ""]
    t0 = parts[0] if parts else 0.0
    t1 = parts[1] if len(parts) > 1 else None
    is_dir = os.path.isdir(args.path)
    folder = args.path if is_dir else (os.path.dirname(os.path.abspath(args.path)) or ".")
    pattern = "*.raw" if is_dir else os.path.basename(args.path)
    m = writer.trim_folder(folder, t0=t0, t1=t1, pattern=pattern)
    print(f"trimmed {m['n_clips']} clip(s) to [{t0:g}, {'end' if t1 is None else f'{t1:g}'}] s "
          f"→ {m['out_dir']}")
    for c in m["clips"]:
        print(f"  {c['out']}  ({c['n_events']:,} ev, kept {c['kept_s']:g} s"
              + (f", +{','.join(c['sidecars'])}" if c['sidecars'] else "") + ")")
    if not args.no_open:
        open_in_file_browser(m["out_dir"])
    return 0


def _run_track_study_cli(args):
    """Headless single-clip track study (``--track-study``)."""
    import os
    from gottlux import sensors
    from gottlux.io.paths import open_in_file_browser
    from gottlux.run.track_study import run_track_study
    prof = sensors.get(args.sensor) if args.sensor else sensors.get(sensors.DEFAULT_PROFILE)
    fov = float(args.fov_deg if args.fov_deg is not None else prof.fov_horizontal_deg)
    L = float(args.target_size if args.target_size is not None else 0.18)
    accum = float(args.accum_dt if args.accum_dt is not None else 0.085)
    det = args.detector or "single_centroid"
    band = (float(args.freq_lo or 80.0), float(args.freq_hi or 800.0))
    stem = os.path.splitext(args.path)[0]
    out_dir = args.out or (stem + "_track_study")
    res = run_track_study(args.path, fov_deg=fov, target_size_m=L, out_dir=out_dir, detector=det,
                          accum_dt=accum, band=band, t_start=args.t_start, t_stop=args.t_stop,
                          make_fft_video=bool(args.fft_video), cache_local=True)
    h, c = res["headline"], res["diagnostics"]
    print(f"track study → {out_dir}")
    print(f"  detector={det} · accum {accum*1e3:.0f} ms · FOV {fov:g}° · L {L:g} m · band {band[0]:g}-{band[1]:g} Hz")
    print(f"  tracking range {h.get('tracking_range_m')} m · prop-freq {h.get('prop_frequency_range_m')} m "
          f"· nominal TTC {h.get('nominal_ttc_s')} s · detected {c.get('n_detected')}/{c.get('n_steps')}")
    if not args.no_open:
        open_in_file_browser(out_dir)
    return 0


def _run_fusion_cli(args):
    """Headless EBS+acoustic fusion study (``--fusion``): align PATH (.raw) to ``--audio`` (.wav)."""
    import os
    from gottlux.io.paths import open_in_file_browser
    from gottlux.run.fusion_study import FusionConfig, run_fusion_study
    if not args.audio:
        print("--fusion needs --audio <file.wav>")
        return 2
    cfg = FusionConfig()
    if args.freq_lo is not None or args.freq_hi is not None:
        cfg.band_hz = (float(args.freq_lo or cfg.band_hz[0]), float(args.freq_hi or cfg.band_hz[1]))
    stem = os.path.splitext(args.path)[0]
    out_dir = args.out or (stem + "_fusion")
    s = run_fusion_study(args.path, args.audio, out_dir, offset_s=args.offset, cfg=cfg)
    a, ac, ebs, fz, co = s["alignment"], s["acoustic"], s["ebs"], s["fusion"], s["coherence"]
    print(f"fusion study → {out_dir}")
    print(f"  offset {a['offset_s']:+.3f} s · peak corr {a['peak_corr']:.2f} · "
          f"overlap {a['overlap_s']:.2f} s")
    print(f"  acoustic f0 {ac['f0_hz']:.0f} Hz (C={ac['confidence']:.2f}) · "
          f"EBS in-box f0 {ebs['f0_hz']:.0f} Hz (C={ebs['confidence']:.2f})")
    print(f"  fused P(drone)={fz['p_fused']:.2f} · coherence γ²={co['peak_coh']:.2f} @ {co['peak_hz']:.0f} Hz")
    if not args.no_open:
        open_in_file_browser(out_dir)
    return 0


def _run_user_script_cli(args):
    """Headless user-script run (``--run-script``): ``process(win, ctx)`` on the recording,
    with the window/ROI flags shaping exactly what the script sees."""
    import gottlux as eb
    from gottlux.io.paths import open_in_file_browser
    from gottlux.userscripts import UserScriptError, run_script
    if not args.path:
        print("--run-script needs an INPUT recording (gottlux INPUT --run-script FILE.py)")
        return 2
    rec = eb.load(args.path, camera=args.camera, mode=args.mode,
                  force_decode=args.force_decode, progress=lambda f: None)
    extra = args.script_args.split() if args.script_args else []
    try:
        res = run_script(args.run_script, rec, t0=args.t_start, t1=args.t_stop,
                         roi=_parse_roi(args.roi), output_dir=args.out, script_args=extra)
    except UserScriptError as e:
        print(f"user script failed: {e}")
        return 2
    if not args.no_open:
        open_in_file_browser(res["folder"])
    return 0


def _run_to_hdf5_cli(args):
    """Headless ``.raw`` → HDF5 conversion (``--to-hdf5``): stream PATH (windowed/ROI'd if
    asked) into a Metavision-compatible ``.h5`` and print where it landed."""
    import os
    from gottlux.io import hdf5 as h5io
    out = args.to_hdf5 or (os.path.splitext(args.path)[0] + ".h5")
    roi = _parse_roi(args.roi)
    n = h5io.write_hdf5(args.path, out, t0=args.t_start, t1=args.t_stop, roi=roi)
    if not h5io.is_hdf5_path(out):                 # write_hdf5 appended '.h5' — report truth
        out += ".h5"
    span = "" if (args.t_start is None and args.t_stop is None and roi is None) else \
        " (windowed/ROI sub-clip)"
    print(f"wrote {os.path.abspath(out)}  ({n:,} events{span})")
    return 0


def _run_editing_cli(args, cfg):
    """Headless cut / stitch of ``.raw`` clips (``--cut`` / ``--stitch``)."""
    import os
    from gottlux.io import writer
    from gottlux.io.paths import open_in_file_browser
    stem = os.path.splitext(args.path)[0]
    if args.cut:
        t0, t1 = (float(v) for v in args.cut.replace(" ", "").split(","))
        out = args.out_raw or f"{stem}_cut_{t0:g}-{t1:g}.raw"
        # cut_file cuts an EVT2.1 file directly on the bytes (no decode); ROI / EVT2 / EVT3 decode.
        n = writer.cut_file(args.path, out, t0=t0, t1=t1, roi=_parse_roi(args.roi))
        print(f"cut [{t0:g}, {t1:g}] s{' + ROI' if args.roi else ''} → {out}  ({n:,} events)")
    elif args.stitch:
        import gottlux as eb
        rec = eb.load(args.path, camera=cfg.camera, mode=cfg.mode, progress=lambda f: None)
        others = [eb.load(pth, progress=lambda f: None)
                  for pth in args.stitch.split(",") if pth.strip()]
        out = args.out_raw or f"{stem}_stitched.raw"
        res = writer.stitch_clips(out, [rec, *others], gap_s=args.stitch_gap)
        print(f"stitched {1 + len(others)} clips → {out}  "
              f"({res['n_events']:,} events, {res['duration_s']:.3f} s)")
    if not args.no_open:
        open_in_file_browser(os.path.dirname(os.path.abspath(args.out_raw or stem)))
    return 0


def _run_cache_cli(args):
    """Headless decode-cache management (``--cache-info`` / ``--clear-cache``)."""
    import os
    from gottlux.io import cache
    target = args.path or os.getcwd()
    if args.clear_cache:
        res = cache.clear_cache(target, stale_only=args.stale_only)
        what = "stale decode cache(s)" if args.stale_only else "decode cache(s)"
        if not res["n_stems"]:
            print(f"No {what} found under {os.path.abspath(target)}")
            return 0
        print(f"Cleared {res['n_stems']} {what} — freed {cache._fmt_bytes(res['freed_bytes'])} "
              f"({len(res['removed'])} file(s))")
        for fp, err in res["skipped"]:
            print(f"  skipped (in use): {fp}  [{err}]")
        return 0
    print(cache.format_cache_report(cache.cache_info(target)))
    return 0


def _parse_roi(s):
    """Parse a ``"x0,y0,x1,y1"`` ROI string into a 4-int tuple (or ``None``)."""
    if not s:
        return None
    parts = [int(round(float(v))) for v in s.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must be 'x0,y0,x1,y1'")
    return tuple(parts)


def _config_from_args(a) -> Config:
    cfg = Config(mode=a.mode, camera=a.camera)
    if a.sensor:
        cfg.sensor = a.sensor
    if a.fov_deg is not None:
        cfg.fov_deg = a.fov_deg
    if a.target_size is not None:
        cfg.target_size_m = a.target_size
    if a.blades is not None:
        cfg.rotor_blades = a.blades
    if a.prop_diameter is not None:
        cfg.prop_diameter_m = a.prop_diameter
    if a.assume_spin is not None:
        cfg.assume_spin_hz = 0.0 if str(a.assume_spin).strip().lower() == "auto" \
            else float(a.assume_spin)
    if a.mask_rotations is not None:
        cfg.mask_rotations = a.mask_rotations
    if a.accum_dt is not None:
        cfg.accum_dt = a.accum_dt
    if a.detector:
        cfg.detector = a.detector
    if a.freq_lo is not None:
        cfg.freq_lo = a.freq_lo
    if a.freq_hi is not None:
        cfg.freq_hi = a.freq_hi
    if a.fft_fs is not None:
        cfg.fft_fs = a.fft_fs
    if a.snr is not None:
        cfg.snr_thresh = a.snr
    # window of interest
    cfg.t_start = a.t_start
    cfg.t_stop = a.t_stop
    cfg.roi = _parse_roi(a.roi)
    # spectral transform
    cfg.spectrum_method = "nufft" if a.nufft else "fft"
    cfg.spectrum_normalize = a.whiten
    # lab exports
    cfg.export_cube = a.export_cube
    cfg.cube_nt = a.cube_nt
    cfg.cube_bin = a.cube_bin
    cfg.make_report = a.report
    cfg.make_performance = a.performance
    if a.ttc_speed is not None:
        cfg.approach_speed_mps = a.ttc_speed
    if a.tracker:
        cfg.tracker = a.tracker
    cfg.make_video = a.video
    cfg.analyses = tuple(s.strip() for s in a.analyses.split(",") if s.strip())
    if a.plots:
        cfg.plots = tuple(s.strip() for s in a.plots.split(",") if s.strip())
    cfg.output_root = a.out
    cfg.open_when_done = not a.no_open
    return cfg


def _utf8_console():
    """Make the console UTF-8 so Hz dashes / unit symbols print cleanly on Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def main(argv=None):
    _utf8_console()
    argv = list(sys.argv if argv is None else argv)
    args = build_parser().parse_args(argv[1:])

    # User plugins (GOTTLUX_PLUGINS): import custom detector/analysis modules before any
    # registry is consulted. Errors are reported per-file and never fatal (plugins.py).
    from gottlux.plugins import load_plugins
    load_plugins()

    if args.register_raw or args.unregister_raw:
        from gottlux.app import file_assoc
        ok, msg = (file_assoc.unregister() if args.unregister_raw else file_assoc.register())
        print(msg)
        return 0 if ok else 1

    if args.list_detectors:
        from gottlux.detectors import get_detector, list_detectors
        print("Registered detectors:")
        for name in sorted(list_detectors()):
            d = get_detector(name)
            print(f"  {name:13s} [{d.regime:8s}] {d.description}")
            if d.use_for:
                print(f"               use for: {d.use_for}")
        return 0
    if args.list_signatures:
        from gottlux.detectors import list_signatures
        print("Flutter signatures (band Hz):")
        for name, s in list_signatures().items():
            print(f"  {name:13s} {s.freq_lo:6.0f}–{s.freq_hi:<6.0f}  {s.description}")
        return 0
    if args.list_sensors:
        from gottlux import sensors
        print("Sensor / camera profiles (select with --sensor, or Config(sensor=…)):")
        for key, prof in sensors.list_profiles().items():
            default = "  [default]" if key == sensors.DEFAULT_PROFILE else ""
            print(f"  {key:10s} {prof.name}{default}")
            print(f"             {prof.width_px}×{prof.height_px} px @ {prof.pixel_pitch_um:g} µm  ·  "
                  f"{prof.focal_length_mm:g} mm lens  ·  "
                  f"FOV {prof.fov_horizontal_deg:g}° H / {prof.fov_vertical_deg:g}° V / "
                  f"{prof.fov_diagonal_deg:g}° diag")
        print("\nOverride per clip: --fov_deg sets horizontal FOV; --target_size sets target size.")
        return 0
    if args.list_plots:
        from gottlux.run.pipeline import list_plots
        print("Available --plots (figures you can output individually):")
        for name, needs in list_plots().items():
            tag = "  (needs a detector)" if needs else ""
            print(f"  {name}{tag}")
        print("\nExample:  gottlux file.raw --plots flicker_map,spectrum,tracks --detector drone")
        return 0
    if args.list_analyses:
        from gottlux.run.pipeline import _ANALYSES
        core = ("overview", "spectral", "panorama", "detect", "metrics", "performance")
        print("Available --analyses:")
        print("  core:", ", ".join(core))
        rot = sorted(k for k in _ANALYSES if k not in core)
        print("  rotation:", ", ".join(rot))
        print("\nExample:  gottlux capture/ --analyses radar,mti,tracking --tracker kalman --video")
        return 0

    # Standalone tool export: write the self-contained bundle (or list the tools) and exit.
    if args.export_tool is not None:
        from gottlux.run.tool_export import run_tool_export_cli
        return run_tool_export_cli(args)

    # Decode-cache management: report / reclaim and exit (PATH defaults to the cwd).
    if args.cache_info or args.clear_cache:
        return _run_cache_cli(args)

    # Batch-trim: headless, works on a folder PATH (must run before the GUI/load dispatch).
    if args.trim is not None:
        return _run_trim_cli(args)

    # Single-clip track study (headless): a tracker + KPIs + tracking figures/videos.
    if args.track_study and args.path:
        return _run_track_study_cli(args)

    # EBS + acoustic fusion (headless): align a .raw to a .wav, cross-domain spectra + report.
    if args.fusion and args.path:
        return _run_fusion_cli(args)

    # .raw → HDF5 conversion (headless): write the Metavision-compatible .h5 and exit.
    if args.to_hdf5 is not None and args.path:
        return _run_to_hdf5_cli(args)

    # User script (headless): run a custom process(win, ctx) on the recording and exit.
    if args.run_script:
        return _run_user_script_cli(args)

    # Disable the sampled fast-preview open (both GUI flavours read the env at load time).
    if args.no_preview:
        import os
        os.environ["GOTTLUX_PREVIEW_THRESHOLD_MB"] = "0"

    # Quick viewer: the lightweight single-view player for one recording.
    if args.view:
        from gottlux.app.quickview import main as view_main
        return view_main([argv[0]] + ([args.path] if args.path else []))

    # GUI: no path, or --gui without a forced headless run.
    want_gui = (args.path is None) or (args.gui and not args.headless)
    if want_gui:
        from gottlux.app.main import main as gui_main
        gui_argv = [argv[0]] + ([args.path] if args.path else [])
        return gui_main(gui_argv)

    cfg = _config_from_args(args)
    if args.cut or args.stitch:
        return _run_editing_cli(args, cfg)
    if args.performance:
        return _run_performance_cli(args, cfg)
    from gottlux.run.pipeline import run_path
    run_path(args.path, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
