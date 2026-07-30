"""
Functional tests for the ported EBS rotation/fusion/tracker/viz suite (gottlux.rotation).

These build synthetic recordings and run the *real* ported code end-to-end:
  * STARING : build_context → isolate → trajectory → a tracker → track_analysis → tracking_report
  * ROTATION: synthetic telemetry → build_context → per-pass centroids → metrics → radar/mti figures

They prove the merge ports actually compute on the Recording data model, not just import.
"""
import os

import numpy as np
import pytest

from gottlux.config import Config
from gottlux.io.recording import Recording
from gottlux.io.telemetry import Telemetry


# ------------------------------------------------------------------ synthetic data
def _bg(n, dur_s, w, h, seed):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, dur_s, n)
    x = rng.integers(0, w, n)
    y = rng.integers(0, h, n)
    p = rng.integers(0, 2, n)
    return t, x, y, p


def _blob(cx, cy, half, density, t_us, rng):
    """A solid (2*half)² block of events at (cx, cy) all sharing time t_us."""
    xs, ys = np.meshgrid(np.arange(cx - half, cx + half), np.arange(cy - half, cy + half))
    xs = xs.ravel(); ys = ys.ravel()
    reps = max(1, density)
    xs = np.tile(xs, reps); ys = np.tile(ys, reps)
    t = np.full(xs.shape, t_us, np.int64)
    p = rng.integers(0, 2, xs.shape)
    return xs, ys, p, t


def synth_staring(w=320, h=320, dur_s=1.5, seed=0):
    """A moving solid block across a fixed sensor, on a sparse random background."""
    rng = np.random.default_rng(seed)
    tb, xb, yb, pb = _bg(60000, dur_s, w, h, seed)
    XS = [xb.astype(np.int64)]; YS = [yb.astype(np.int64)]
    PS = [pb.astype(np.uint8)]; TS = [(tb * 1e6).astype(np.int64)]
    n_frames = 120
    for k in range(n_frames):
        tt = k / n_frames * dur_s
        cx = int(40 + (w - 80) * tt / dur_s)
        cy = int(h / 2 + 30 * np.sin(2 * np.pi * tt))
        x, y, p, t = _blob(cx, cy, 8, 3, int(tt * 1e6), rng)
        XS.append(x); YS.append(y); PS.append(p); TS.append(t)
    x = np.concatenate(XS); y = np.concatenate(YS)
    p = np.concatenate(PS); t = np.concatenate(TS)
    m = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return Recording.from_events(x[m], y[m], p[m], t[m], width=w, height=h, name="synth_staring")


def write_telemetry_csv(path, dur_s, omega_deg_s, dt_s=0.005):
    """Write a synthetic rotation telemetry CSV (System_Time,Azimuth,Revolution,Flag)."""
    ts = np.arange(0, dur_s, dt_s)
    az = (omega_deg_s * ts) % 360.0
    rev = np.floor(omega_deg_s * ts / 360.0).astype(int)
    lines = ["System_Time,Azimuth,Revolution,Flag"]
    base_s = 18 * 3600  # 18:00:00
    last_rev = -1
    for tt, a, r in zip(ts, az, rev):
        sod = base_s + tt
        hh = int(sod // 3600) % 24
        mm = int((sod % 3600) // 60)
        ss = int(sod % 60)
        ms = int(round((sod - int(sod)) * 1000))
        stamp = f"20260101_{hh:02d}{mm:02d}{ss:02d}_{ms:03d}"
        flag = "HALL_SYNC" if r != last_rev else ""
        last_rev = r
        lines.append(f"{stamp},{a:.3f},{r},{flag}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def synth_rotation(tmp_dir, w=320, h=320, dur_s=5.0, omega=360.0, fov=50.0,
                   bearing=200.0, elev=0.0, seed=1):
    """A target at a fixed WORLD bearing, seen once per revolution as the FOV pans past it."""
    rng = np.random.default_rng(seed)
    csv = write_telemetry_csv(os.path.join(tmp_dir, "data_synth.csv"), dur_s, omega)
    tel = Telemetry(csv)
    deg_per_px = fov / w
    az_sign = -1.0
    XS = []; YS = []; PS = []; TS = []
    # sparse background
    tb, xb, yb, pb = _bg(40000, dur_s, w, h, seed + 9)
    XS.append(xb.astype(np.int64)); YS.append(yb.astype(np.int64))
    PS.append(pb.astype(np.uint8)); TS.append((tb * 1e6).astype(np.int64))
    # target events densely sampled in time; place where in-FOV
    tt = np.arange(0, dur_s, 0.001)
    pan = np.rad2deg(np.interp(tt, tel.t, tel.azimuth_unwrapped()))
    # x = W/2 + (bearing - pan)/(az_sign*deg_per_px)
    xc = w / 2 + (bearing - pan) / (az_sign * deg_per_px)
    yc = h / 2 - elev / deg_per_px
    invis = (xc < 10) | (xc > w - 10)
    for ti, xi in zip(tt[~invis], xc[~invis]):
        x, y, p, t = _blob(int(xi), int(yc), 7, 2, int(ti * 1e6), rng)
        XS.append(x); YS.append(y); PS.append(p); TS.append(t)
    x = np.concatenate(XS); y = np.concatenate(YS)
    p = np.concatenate(PS); t = np.concatenate(TS)
    m = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    rec = Recording.from_events(x[m], y[m], p[m], t[m], width=w, height=h, name="synth_rotation")
    # attach without re-refining the offset: the synthetic events were placed at offset 0,
    # so the recovered world bearing should match the ground-truth bearing exactly.
    rec.attach_telemetry(tel, refine=False)
    return rec


# ------------------------------------------------------------------ tests
def test_staring_pipeline_end_to_end(tmp_path):
    from gottlux.rotation import build_context, track_analysis
    from gottlux.rotation import trackers
    from gottlux.rotation.viz import tracking_report

    rec = synth_staring()
    cfg = Config(mode="staring")
    ctx = build_context(rec, cfg)
    assert ctx.cfg.mode == "staring"
    assert ctx.keep is not None and ctx.keep.sum() > 0, "no target events isolated"
    assert ctx.dets.shape[0] > 0, "no detections"
    assert ctx.traj and len(ctx.traj["t"]) > 0

    # a tracker links the trajectory
    T = trackers.get("nearest")()
    out = T.track(ctx.traj, ctx.cfg, ctx.tel, ev=ctx.ev)
    tracks = out.get("tracks", [])
    assert len(tracks) >= 1, "nearest tracker produced no tracks"

    # the regime-split tracking report writes its artifacts
    summary, arts = tracking_report.render_tracking_report(
        tracks, ctx.traj, ctx.ev, ctx.keep, ctx.cfg, ctx.tel,
        str(tmp_path), tag="staring", tracker="nearest", video=False)
    assert summary.get("n_tracks", 0) >= 1
    written = [p for p, _ in arts if os.path.exists(p)]
    assert any(p.endswith(".png") for p in written)
    assert any(p.endswith(".csv") for p in written)


def test_rotation_pipeline_centroids_metrics_viz(tmp_path):
    from gottlux.rotation import build_context, metrics
    from gottlux.rotation.viz import radar_map, mti

    rec = synth_rotation(str(tmp_path))
    assert rec.is_rotating
    cfg = Config(mode="rotation", fov_deg=50.0)
    ctx = build_context(rec, cfg)
    assert ctx.tel is not None
    assert ctx.keep.sum() > 0, "no rotation target events isolated"

    # accumulation-independent per-pass centroids: at least one pass recovered,
    # at the right world bearing (~200°)
    P = ctx.centroids
    assert P.shape[0] >= 1, "no per-pass centroids"
    med_bearing = float(np.median(P[:, 1]))
    assert abs(((med_bearing - 200.0 + 180) % 360) - 180) < 8.0, f"bearing off: {med_bearing}"

    # quantitative metrics
    res = metrics.compute(ctx.ev, ctx.keep, ctx.traj, P, ctx.cfg, ctx.tel)
    assert res["sphere_fraction_pct"] > 0
    assert "revisit_interval_s" in res

    # rotation viz figures render to disk
    rp = radar_map.render_radar_map(ctx.traj, ctx.cfg, str(tmp_path / "radar.png"))
    assert rp and os.path.exists(rp)
    mp = mti.render_mti(ctx.ev, ctx.keep, ctx.cfg, str(tmp_path / "mti.png"), tel=ctx.tel)
    assert mp and os.path.exists(mp)


@pytest.mark.parametrize("name", ["nearest", "single", "kalman", "staring_kvf"])
def test_trackers_run_on_trajectory(name, tmp_path):
    """Trajectory-consuming trackers run without error on a staring trajectory."""
    from gottlux.rotation import build_context, trackers
    rec = synth_staring()
    ctx = build_context(rec, Config(mode="staring"))
    T = trackers.get(name)()
    out = T.track(ctx.traj, ctx.cfg, ctx.tel, ev=ctx.ev)
    assert isinstance(out, dict) and "tracks" in out


def test_unified_registry_has_flutter_and_trackers():
    """The one detector registry holds both flutter detectors and the ported EBS trackers."""
    from gottlux.detectors import list_detectors
    names = set(list_detectors())
    # flutter presets from the substrate
    assert {"drone", "insect", "bird"} <= names
    # ported EBS trackers (hummingbird clashes with the flutter preset -> hummingbird_track)
    for t in ("nearest", "single", "kalman", "cmax", "staring_kvf"):
        assert t in names, f"ported tracker {t} not in unified registry"
    assert "hummingbird_track" in names or "hummingbird" in names


def test_headless_rotation_pipeline_run_folder(tmp_path):
    """A single headless run produces the EBS rotation artifacts in one run folder."""
    from gottlux.run.pipeline import run_recording
    cap = tmp_path / "cap"
    cap.mkdir(parents=True, exist_ok=True)
    rec = synth_rotation(str(cap))
    cfg = Config(mode="rotation", fov_deg=50.0)
    cfg.analyses = ("radar", "mti", "tracking", "rotation_metrics")
    cfg.tracker = "nearest"
    cfg.output_root = str(tmp_path / "runs")
    cfg.open_when_done = False
    run_path = run_recording(rec, cfg)
    assert os.path.isdir(run_path)
    # the rotation analyses each wrote their subfolder
    for sub in ("radar", "mti", "rotation_metrics"):
        assert os.path.isdir(os.path.join(run_path, sub)), f"missing {sub} subdir"
    # a manifest was written
    assert any(f.lower().startswith("manifest") or f.endswith(".json")
               for f in os.listdir(run_path))


def test_linked_tracker_detector_produces_targets():
    """An EBS tracker, run via the unified Detector interface, yields Targets with kinematics."""
    from gottlux.detectors import get_detector
    rec = synth_staring()
    det = get_detector("nearest")
    res = det.run(rec, Config(mode="staring"))
    assert res.detector == "nearest"
    assert res.regime == "staring"
    assert res.n_targets >= 1
    tgt = res.targets[0]
    assert tgt.n >= 1
    # the staring extras are populated by the linker
    assert tgt.rel_distance is not None
