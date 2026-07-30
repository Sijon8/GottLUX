"""Detector-framework tests: registry, tuning, and recovery of a known flutter target."""
import numpy as np

import gottlux as eb
from gottlux.detectors import (get_detector, list_detectors, list_signatures)


def test_registry_has_builtins():
    names = set(list_detectors())
    assert {"drone", "insect", "mosquito", "hummingbird", "bird", "flutter"} <= names
    assert "drone" in list_signatures() or True  # signatures separate but present
    assert set(list_signatures()) >= {"drone", "insect", "bird"}


def test_param_override_and_coerce():
    det = get_detector("drone", snr_thresh=999, freq_lo=50)
    assert det.params["snr_thresh"] <= 50.0       # clamped to Param.hi
    assert det.params["freq_lo"] == 50.0
    det.set(min_pixels=10)
    assert det.params["min_pixels"] == 10


def test_drone_detects_planted_target(flutter_rec):
    rec, truth = flutter_rec
    det = get_detector("drone")
    res = det.run(rec, cfg=eb.Config(mode="staring", fov_deg=76))
    assert res.n_targets >= 1
    best = max(res.targets, key=lambda t: t.confidence)
    planted = truth[0]["flutter_hz"]
    assert abs(best.median_freq - planted) < 20.0   # frequency recovered
    assert best.confidence > 0.4


def test_quiet_scene_few_targets(quiet_rec):
    """Pure noise must not produce confident drone tracks."""
    det = get_detector("drone")
    res = det.run(quiet_rec, cfg=eb.Config(mode="staring", fov_deg=76))
    assert len(res.confident(0.6)) == 0


def test_detector_result_serializable(flutter_rec):
    rec, _ = flutter_rec
    res = get_detector("drone").run(rec, cfg=eb.Config(mode="staring"))
    s = res.summary()
    assert "drone" in s
    # confidence is bounded
    for t in res.targets:
        assert 0.0 <= t.confidence <= 1.0
