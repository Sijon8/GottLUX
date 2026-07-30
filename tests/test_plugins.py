"""
Tests for plugin loading (:mod:`gottlux.plugins`, the ``GOTTLUX_PLUGINS`` hook) and the
worked example plugin (``examples/custom_detector.py``).

The promise under test: a user module that ``@register``-s a detector is loaded at CLI/GUI
startup from ``GOTTLUX_PLUGINS`` (an ``os.pathsep``-separated list of ``.py`` files or
directories) — errors reported per file, never fatal; loading idempotent across the
CLI → GUI handoff — so gottlux is extensible without forking.
"""
import os
import subprocess
import sys
import textwrap

from gottlux.detectors import list_detectors
from gottlux.plugins import PLUGINS_ENV, load_plugins

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "examples", "custom_detector.py")


def _write_plugin(path, det_name):
    """A minimal valid plugin: registers one Detector subclass under *det_name*."""
    path.write_text(textwrap.dedent(f'''
        from gottlux.detectors.base import Detector, DetectorResult, register

        @register
        class P(Detector):
            name = "{det_name}"
            description = "test plugin detector"
            def run(self, rec, cfg=None, t0=None, t1=None, progress=None):
                return DetectorResult([], self.name, dict(self.params), self.regime)
        '''), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------- loading mechanics
def test_single_file_registers_detector_and_is_idempotent(tmp_path):
    p = _write_plugin(tmp_path / "plug_one.py", "plugtest_one")
    results = load_plugins(env=p)
    assert results == [(os.path.abspath(p), None)]
    assert "plugtest_one" in list_detectors()               # @register reached the registry
    # second call: already loaded -> skipped entirely (the CLI -> GUI handoff)
    assert load_plugins(env=p) == []


def test_directory_loads_all_top_level_py(tmp_path):
    d = tmp_path / "plugdir"
    d.mkdir()
    _write_plugin(d / "a_plug.py", "plugtest_dir_a")
    _write_plugin(d / "b_plug.py", "plugtest_dir_b")
    (d / "_private.py").write_text("raise RuntimeError('must not be imported')",
                                   encoding="utf-8")
    results = load_plugins(env=str(d))
    assert [os.path.basename(p) for p, e in results] == ["a_plug.py", "b_plug.py"]
    assert all(e is None for _, e in results)
    assert {"plugtest_dir_a", "plugtest_dir_b"} <= set(list_detectors())


def test_broken_plugin_reported_not_fatal(tmp_path):
    bad = tmp_path / "bad_plug.py"
    bad.write_text("raise ValueError('boom at import')", encoding="utf-8")
    good = _write_plugin(tmp_path / "good_plug.py", "plugtest_survivor")
    msgs = []
    results = load_plugins(env=str(bad) + os.pathsep + good, report=msgs.append)
    errs = {os.path.basename(p): e for p, e in results}
    assert isinstance(errs["bad_plug.py"], ValueError)
    assert errs["good_plug.py"] is None                     # the good one still loaded
    assert "plugtest_survivor" in list_detectors()
    assert len(msgs) == 1 and "boom at import" in msgs[0]   # one line per failure
    # a broken plugin leaves no half-imported module behind
    assert not any("bad_plug" in m for m in sys.modules)


def test_missing_entry_reported(tmp_path):
    missing = str(tmp_path / "nope.py")
    msgs = []
    results = load_plugins(env=missing, report=msgs.append)
    assert len(results) == 1 and isinstance(results[0][1], FileNotFoundError)
    assert msgs and "nope.py" in msgs[0]
    assert load_plugins(env="") == []                       # empty spec: a clean no-op


# --------------------------------------------------------------------- the CLI hook
def test_cli_startup_loads_env_plugins(tmp_path, monkeypatch, capsys):
    """gottlux --list_detectors shows a GOTTLUX_PLUGINS detector like any built-in."""
    p = _write_plugin(tmp_path / "cli_plug.py", "plugtest_cli")
    monkeypatch.setenv(PLUGINS_ENV, p)
    from gottlux.cli import main
    assert main(["gottlux", "--list_detectors"]) == 0
    assert "plugtest_cli" in capsys.readouterr().out


# --------------------------------------------------------------------- the worked example
def test_example_plugin_registers_blink_detector():
    """examples/custom_detector.py works as a GOTTLUX_PLUGINS file: importing it
    registers the 'blink' detector with an auto-panel-ready PARAMS list."""
    results = load_plugins(env=EXAMPLE)
    assert all(e is None for _, e in results)
    dets = list_detectors()
    assert "blink" in dets
    specs = dets["blink"].param_specs()
    assert {p.key for p in specs} >= {"freq_lo", "freq_hi", "snr_thresh", "step_s"}


def test_example_runs_standalone_and_finds_planted_tone():
    """`python examples/custom_detector.py` (no clip -> synthetic 200 Hz scene) exits 0
    and reports the planted tone."""
    r = subprocess.run([sys.executable, EXAMPLE], capture_output=True, text=True,
                       timeout=300, cwd=REPO,
                       env={**os.environ, "PYTHONPATH": REPO, "MPLBACKEND": "Agg"})
    assert r.returncode == 0, r.stderr
    assert "blink" in r.stdout
    assert "200" in r.stdout                                # ~200 Hz reported
