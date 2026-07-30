"""
Tests for the rotor-ladder study exporter (gottlux.rotation.ladder_report).

The writer is pure (no Qt/OpenGL): scenes are handed in as RGB arrays, so we can validate the
whole bundle — the per-angle scene PNGs, the rotor-ladder + spectrum figures, the measurements
table, and a compilable LaTeX report that references the figures and explains the algorithm.
"""
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np

from gottlux.core import frequency as fq
from gottlux.rotation import ladder_report
from gottlux.rotation import rotor_ladder as rl


def _scene(seed, h=48, w=64):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3)).astype(np.uint8)


def test_save_ladder_study_writes_full_bundle(tmp_path):
    x, t = rl.synthetic_rotor_pass(blade_hz=200.0, sweep_px_s=-1800.0, disk_px=1.5,
                                   burst_events=30, noise_events=120, seed=1)
    result = rl.ladder_signature(x, t, sweep_px_s=-1800.0)
    spectrum = fq.region_spectrum(t * 1e6, fs=2000.0, fmin=80.0, fmax=800.0)   # t in µs
    scenes = {"iso": _scene(0), "top": _scene(1), "side": _scene(2)}
    out = str(tmp_path / "study")

    written = ladder_report.save_ladder_study(
        out, scenes=scenes, x=x, t=t, result=result, spectrum=spectrum,
        band=(80, 800), meta={"recording": "clip_a", "sensor_px": "320x320"},
        title="rotor ladder")

    names = {os.path.basename(p) for p in written}
    for f in ("scene-iso.png", "scene-top.png", "scene-side.png",
              "rotor-ladder.png", "rotor-ladder.pdf", "spectrum.png",
              "measurements.json", "rotor-ladder-report.tex"):
        assert f in names, f"missing {f} in {sorted(names)}"
        assert os.path.exists(os.path.join(out, f))

    tex = open(os.path.join(out, "rotor-ladder-report.tex"), encoding="utf-8").read()
    # a self-contained, compilable document
    assert r"\documentclass" in tex and r"\begin{document}" in tex and r"\end{document}" in tex
    # the figures are referenced (labelled) and the algorithm equation is present
    assert "scene-iso.png" in tex and "rotor-ladder.png" in tex and "spectrum.png" in tex
    assert r"f = \frac{|v|}{\Delta x}" in tex
    assert "How the rotor-ladder detector works" in tex
    assert "Measured quantities" in tex
    # the measured blade frequency made it into the report
    assert f"{result.blade_hz:g}" in tex
    # underscores in the recording name were escaped for LaTeX
    assert "clip_a" not in tex and r"clip\_a" in tex


def test_save_ladder_study_is_robust_when_empty(tmp_path):
    """With no scenes and no events it still emits a valid (figure-free) LaTeX report."""
    out = str(tmp_path / "empty")
    written = ladder_report.save_ladder_study(out, meta={"recording": "x"})
    assert any(p.endswith("rotor-ladder-report.tex") for p in written)
    tex = open(os.path.join(out, "rotor-ladder-report.tex"), encoding="utf-8").read()
    assert r"\end{document}" in tex
    assert "How the rotor-ladder detector works" in tex      # the explanation is always included
