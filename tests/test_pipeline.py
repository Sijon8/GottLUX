"""Pipeline + provenance smoke test: a run produces a complete, reproducible folder."""
import json
import os

import matplotlib
matplotlib.use("Agg")

from gottlux.config import Config
from gottlux.run.pipeline import run_recording


def test_pipeline_writes_run_folder(flutter_rec, tmp_path):
    rec, _ = flutter_rec
    cfg = Config(mode="staring", fov_deg=76, detector="drone",
                 freq_lo=80, freq_hi=800, fft_fs=2000,
                 analyses=("overview", "spectral"),
                 output_root=str(tmp_path), open_when_done=False)
    path = run_recording(rec, cfg)
    assert os.path.isdir(path)

    # manifest exists and is well-formed
    with open(os.path.join(path, "run_manifest.json")) as f:
        man = json.load(f)
    assert man["gottlux_version"]
    assert man["recording"]["n_events"] == rec.n
    assert "config" in man and man["config"]["detector"] == "drone"
    assert "environment" in man and man["environment"]["python"]

    # key artifacts present
    assert os.path.exists(os.path.join(path, "RUN_SUMMARY.txt"))
    assert os.path.exists(os.path.join(path, "overview", "event_rate.csv"))
    assert os.path.exists(os.path.join(path, "spectral", "flicker_map.npz"))
    assert os.path.isdir(os.path.join(path, "_source_snapshot", "gottlux"))
    # the detector ran (auto-added because cfg.detector is set)
    assert os.path.isdir(os.path.join(path, "detect", "drone"))
