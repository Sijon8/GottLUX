"""Tests for the sandbox tracking lab + FPS playback control.

These exercise the *logic* of the live-algorithm sandbox — the ``track(ev, state)`` API,
the bundled presets, and the return-value classifier — by driving them directly against a
synthetic recording, with no Qt widgets created. Plus the FPS controller's clamping.
"""
import math

import numpy as np
import pytest

from gottlux.app.sandbox import (_AlgoEnv, _classify_return, _norm_det, _PRESETS,
                                 _split_metrics)
from gottlux.app.transport import (FPS_PRESETS, REALTIME_FPS, TimeController,
                                   _speed_factor_text)


# ----------------------------------------------------------------- FPS control
@pytest.fixture(scope="module")
def qapp():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_fps_presets_sorted_up_to_100000():
    assert FPS_PRESETS == sorted(FPS_PRESETS)
    assert max(FPS_PRESETS) == 100000                 # high-speed-camera equivalents (20k/50k/100k)
    assert FPS_PRESETS[0] <= 1.0                      # offers genuine slow-motion too


def test_equivalent_fps_inverts_speed(qapp):
    """FPS is an equivalent capture rate: 30 fps = real time, higher = slower (slow-motion)."""
    ctl = TimeController()
    assert ctl.fps == 30.0
    ctl.set_fps(30); assert ctl.speed == pytest.approx(1.0)        # real time
    ctl.set_fps(REALTIME_FPS); assert ctl.slowmo == pytest.approx(1.0)
    ctl.set_fps(10000)
    assert ctl.speed == pytest.approx(REALTIME_FPS / 10000)        # 333x slow-motion
    assert ctl.slowmo == pytest.approx(10000 / REALTIME_FPS)
    ctl.set_fps(0.0); assert ctl.fps == pytest.approx(0.1)         # floored, never zero/negative
    # accumulation no longer changes the playback speed (independent control)
    spd = ctl.speed
    ctl.set_accum(0.5); assert ctl.speed == pytest.approx(spd)


def test_speed_factor_text():
    assert _speed_factor_text(REALTIME_FPS) == "real-time"
    assert "slow" in _speed_factor_text(10000)
    assert "fast" in _speed_factor_text(1.0)


# ----------------------------------------------------------------- algorithm environment
def _env(rec, t0, dt=0.05):
    win = rec.window(t0, t0 + dt)
    return _AlgoEnv(win, t0, t0 + dt)


def test_algoenv_exposes_events_frame_and_blobs(flutter_rec):
    rec, _ = flutter_rec
    env = _env(rec, rec.t_start_s + 0.2)
    assert env.W == rec.width and env.H == rec.height
    assert env.n == len(env.x) == len(env.t)
    assert env.frame.shape == (rec.height, rec.width)
    # blobs() returns the canonical detection dicts
    found = sum(len(_env(rec, t0).blobs(min_pixels=30))
                for t0 in np.linspace(rec.t_start_s, rec.t_stop_s - 0.05, 12))
    assert found > 0                                  # the planted target raster-clusters


# ----------------------------------------------------------------- presets
def _exec_preset(src):
    from gottlux.core.detect import cluster_frame
    from gottlux.detectors.tracking import MultiTracker
    g = {"np": np, "math": math, "MultiTracker": MultiTracker, "cluster_frame": cluster_frame}
    try:
        from scipy import ndimage
        g["ndimage"] = ndimage
    except Exception:
        pass
    ns = {}
    exec(compile(src, "<preset>", "exec"), g, ns)
    return ns["track"]


def test_all_presets_compile_and_run(flutter_rec):
    rec, _ = flutter_rec
    for name, src in _PRESETS.items():
        fn = _exec_preset(src)
        state, produced, last_kind = {}, 0, "none"
        for t0 in np.linspace(rec.t_start_s, rec.t_stop_s - 0.05, 15):
            win = rec.window(t0, t0 + 0.05)
            ret = fn(_AlgoEnv(win, t0, t0 + 0.05), state)
            last_kind, payload = _classify_return(ret, win.n)
            assert last_kind in ("none", "mask", "replace", "dets")
            if last_kind == "dets":
                produced += len(payload)
        if "filter" in name.lower():
            assert last_kind == "mask"                # the filter demo returns a keep-mask
        else:
            assert produced > 0, f"{name!r} tracked nothing on a planted target"


def test_nn_tracker_keeps_a_persistent_id():
    """On a clean scene the MultiTracker preset follows the target under one dominant id.

    (The shared ``flutter_rec`` is deliberately noisy, which spawns throwaway tracks; here we
    use a low-noise scene so the persistence of the *real* track is unambiguous.)
    """
    from collections import Counter

    from gottlux.synthetic import FlutterTarget, synthetic_scene
    rec, _ = synthetic_scene(
        duration_s=1.0, noise_rate_hz=1500, static_clutter=0, seed=5,
        targets=[FlutterTarget(flutter_hz=200.0, x0=40, y0=160, x1=280, y1=160)])

    fn = _exec_preset(_PRESETS["Greedy NN tracker (MultiTracker)"])
    state, counts, n_frames = {}, Counter(), 25
    for t0 in np.linspace(rec.t_start_s, rec.t_stop_s - 0.05, n_frames):
        win = rec.window(t0, t0 + 0.05)
        for d in fn(_AlgoEnv(win, t0, t0 + 0.05), state) or []:
            counts[d["id"]] += 1
    assert counts, "tracker followed nothing on a clean planted target"
    # one id (the real target) should persist across a large fraction of the frames,
    # and the tracker must never exceed its active-track cap
    assert max(counts.values()) >= n_frames // 2
    assert len(state["trk"]._active) <= 8


# ----------------------------------------------------------------- return classifier
def test_classifier_handles_every_return_type():
    assert _classify_return(None, 4)[0] == "none"
    assert _classify_return(np.ones(4, bool), 4)[0] == "mask"
    assert _classify_return(np.ones(3, bool), 4)[0] == "none"          # wrong length ignored
    assert _classify_return({"x": [1], "y": [2], "t": [3]}, 4)[0] == "replace"
    kind, payload = _classify_return([{"cx": 1, "cy": 2}, (3, 4), {"bbox": (0, 0, 4, 4)}], 4)
    assert kind == "dets" and len(payload) == 3
    assert payload[1]["id"] == 1                                       # index id for tuple
    assert payload[2]["cx"] == 2.0 and payload[2]["cy"] == 2.0         # centroid from bbox


def test_norm_det_builds_bbox_from_width_height():
    d = _norm_det({"x": 5, "y": 6, "w": 10, "h": 4, "id": 9, "label": "a"}, 0)
    assert d["id"] == 9 and d["label"] == "a"
    assert d["bbox"] == (0.0, 4.0, 10.0, 8.0)
    # a bare centroid gets a default box centred on it
    d2 = _norm_det({"cx": 100, "cy": 100}, 2)
    x0, y0, x1, y1 = d2["bbox"]
    assert x0 < 100 < x1 and y0 < 100 < y1


def test_norm_det_captures_score():
    d = _norm_det({"cx": 1, "cy": 2, "conf": 0.7, "label": "drone"}, 0)
    assert d["score"] == pytest.approx(0.7) and d["label"] == "drone"


# ----------------------------------------------------------------- (output, metrics) split
def test_split_metrics():
    out, m = _split_metrics(([{"cx": 1, "cy": 2}], {"foo": 1}))
    assert isinstance(out, list) and m == {"foo": 1}
    assert _split_metrics([{"cx": 1, "cy": 2}])[1] is None          # not a 2-tuple
    # (det_dict, det_dict) is two detections, NOT (output, metrics)
    assert _split_metrics(({"cx": 1, "cy": 2}, {"cx": 3, "cy": 4}))[1] is None
    out4, m4 = _split_metrics((np.ones(3, bool), {"k": 2}))         # mask + metrics
    assert m4 == {"k": 2} and out4.dtype == bool
    # classifier reduces a (output, metrics) pair to its output
    kind, payload = _classify_return(([{"cx": 5, "cy": 6}], {"m": 1}), 8)
    assert kind == "dets" and len(payload) == 1


def test_classifying_preset_emits_metrics_and_labels(flutter_rec):
    rec, _ = flutter_rec
    fn = _exec_preset(_PRESETS["Classifying tracker (+ metrics)"])
    state, got_metrics, got_label = {}, False, False
    for t0 in np.linspace(rec.t_start_s, rec.t_stop_s - 0.05, 15):
        win = rec.window(t0, t0 + 0.05)
        out, metrics = _split_metrics(fn(_AlgoEnv(win, t0, t0 + 0.05), state))
        if metrics is not None and "n_tracks" in metrics:
            got_metrics = True
        for d in (out or []):
            if d.get("label") and d.get("score") is not None:
                got_label = True
    assert got_metrics and got_label
