"""Shared pytest fixtures: synthetic recordings with known flutter signatures."""
import numpy as np
import pytest

from gottlux.synthetic import FlutterTarget, synthetic_scene


@pytest.fixture(scope="session")
def flutter_rec():
    """A 1.5 s scene with one planted 200 Hz target crossing the centre + noise + clutter."""
    rec, truth = synthetic_scene(
        duration_s=1.5,
        targets=[FlutterTarget(flutter_hz=200.0, x0=50, y0=160, x1=260, y1=160,
                               harmonics=(1.0, 0.5, 0.25))],
        noise_rate_hz=30_000, static_clutter=40, seed=7)
    return rec, truth


@pytest.fixture(scope="session")
def quiet_rec():
    """A scene with only background noise (no planted target)."""
    rec, _ = synthetic_scene(duration_s=1.0, targets=[], noise_rate_hz=20_000,
                             static_clutter=0, seed=3)
    return rec
