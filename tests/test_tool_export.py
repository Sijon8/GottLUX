"""
Tests for the standalone tool export (:mod:`gottlux.export_tools` + ``--export-tool``).

The promise under test: an exported bundle's scripts run **without gottlux** — the Python
one on numpy/scipy/h5py alone, the MATLAB one on base ``h5read`` — against the bundled
``data.h5`` or any GottLUX-exported ``.h5``. Covered here:

* every tool's Python *and* MATLAB template renders with no unresolved ``{placeholder}``
  tokens, and ``render()`` refuses to emit a script when a key is missing;
* each generated **Python script actually runs** as a subprocess (exit 0) and writes its
  documented output file(s) — the critical guarantee — with no ``import gottlux`` in it;
* the standalone ``region_spectrum``'s peak frequency matches gottlux's own
  :func:`~gottlux.core.frequency.region_spectrum` on a planted 200 Hz target;
* MATLAB scripts get a syntax-sanity check only (no MATLAB on CI): non-empty, placeholder
  free, block keywords balanced against ``end``;
* the CLI: ``--export-tool list``, the bundle round-trip honouring the existing
  window/ROI flags, and graceful handling of an unknown tool name;
* the provenance upgrade: the README states where the information came from (source
  path/size + a real SHA-256, verified by recomputing), tabulates every file accessed
  or produced, records the baked parameters + generating command line, and walks through
  running both scripts; ``provenance.json`` carries the same facts machine-readably;
* the ``viz_config`` tool: the CLI ``--viz_mode/--viz_cmap/--viz_tonemap/--viz_accum_ms``
  flags land in the generated scripts, and the Python script renders the configured
  frames (count and polarity modes) as a bare subprocess.
"""
import gc
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

import gottlux as eb  # noqa: E402
from gottlux.config import Config  # noqa: E402
from gottlux.core import frequency as fq  # noqa: E402
from gottlux.export_tools import M_LOADER, PY_LOADER, TOOLS, render  # noqa: E402
from gottlux.io import hdf5 as h5io  # noqa: E402
from gottlux.run.tool_export import export_tool, list_tools_text  # noqa: E402
from gottlux.synthetic import FlutterTarget, synthetic_scene  # noqa: E402

#: The documented output files each generated Python script must produce (README promise).
EXPECTED_OUTPUTS = {
    "event_frames": ("event_frames.npz",),
    "event_rate": ("event_rate.csv",),
    "region_spectrum": ("region_spectrum.csv", "region_spectrum.json"),
    "flicker_map": ("flicker_map.npz",),
    "centroid_tracker": ("centroid_track.csv",),
    "viz_config": ("viz_frame_01.png", "viz_frames.npz"),
}


@pytest.fixture(scope="module")
def clip_h5(tmp_path_factory):
    """A tiny synthetic clip as a GottLUX-exported .h5: one strong 200 Hz target crossing
    the frame (plus light noise), dense enough for every tool to find something."""
    rec, _ = synthetic_scene(
        duration_s=1.2,
        targets=[FlutterTarget(flutter_hz=200.0, x0=60, y0=160, x1=260, y1=160,
                               events_per_burst=80, harmonics=(1.0, 0.4))],
        noise_rate_hz=10_000, static_clutter=0, seed=9)
    path = str(tmp_path_factory.mktemp("tool_clip") / "clip.h5")
    h5io.write_hdf5(rec, path)
    return path


def _run_script(py_path, *argv):
    """Run a generated script exactly as a recipient would: a bare subprocess."""
    return subprocess.run([sys.executable, py_path, *argv], capture_output=True,
                          text=True, cwd=os.path.dirname(py_path), timeout=300,
                          env={**os.environ, "MPLBACKEND": "Agg"})


# --------------------------------------------------------------------- template rendering
_PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


def _dummy_values(tool):
    vals = {k: 1 for k, _ in tool.params}
    vals.update(version="x", stamp="s", source="f.raw")
    return vals


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_templates_render_without_leftover_placeholders(name):
    tool = TOOLS[name]
    py = render(tool.py_template, {**_dummy_values(tool), "loader": PY_LOADER})
    m = render(tool.m_template, {**_dummy_values(tool), "loader": M_LOADER})
    for text in (py, m):
        assert text.strip()
        assert not _PLACEHOLDER.search(text), _PLACEHOLDER.findall(text)
    # the standalone promise: the generated Python never imports gottlux
    assert "import gottlux" not in py
    assert f"run_{name}" in py and f"run_{name}" in m       # self-naming headers


def test_render_refuses_missing_keys():
    tool = TOOLS["region_spectrum"]
    with pytest.raises(KeyError, match="unresolved"):
        render(tool.py_template, {"loader": PY_LOADER})     # every baked param missing


# --------------------------------------------------------------------- the scripts RUN
@pytest.mark.parametrize("name", sorted(TOOLS))
def test_generated_python_script_runs_and_writes_outputs(clip_h5, tmp_path, name):
    """The critical guarantee: python run_<tool>.py exits 0 against the bundled data.h5
    and produces its documented output file(s)."""
    res = export_tool(clip_h5, name, out_dir=str(tmp_path), fmt="python")
    bundle = res["path"]
    assert os.path.basename(bundle).startswith(f"clip_tool_{name}_")
    py = os.path.join(bundle, f"run_{name}.py")
    assert os.path.exists(py) and os.path.exists(os.path.join(bundle, "data.h5"))
    assert not os.path.exists(os.path.join(bundle, f"run_{name}.m"))  # python-only bundle

    r = _run_script(py)
    assert r.returncode == 0, r.stderr
    for fname in EXPECTED_OUTPUTS[name]:
        assert os.path.exists(os.path.join(bundle, fname)), (fname, r.stdout)
    # matplotlib exists in this environment, so the optional plots must appear too
    assert any(f.endswith(".png") for f in os.listdir(bundle)), r.stdout


def test_scripts_accept_another_h5_argument(clip_h5, tmp_path):
    """`python run_x.py other.h5` — the scripts run against ANY GottLUX-exported .h5."""
    res = export_tool(clip_h5, "event_rate", out_dir=str(tmp_path), fmt="python")
    py = os.path.join(res["path"], "run_event_rate.py")
    r = _run_script(py, clip_h5)                            # the original, not data.h5
    assert r.returncode == 0, r.stderr
    assert os.path.exists(os.path.join(res["path"], "event_rate.csv"))


def test_centroid_tracker_follows_planted_path(clip_h5, tmp_path):
    """The exported tracker follows the planted left-to-right crossing (x: 60 → 260)."""
    res = export_tool(clip_h5, "centroid_tracker", out_dir=str(tmp_path), fmt="python")
    py = os.path.join(res["path"], "run_centroid_tracker.py")
    r = _run_script(py)
    assert r.returncode == 0, r.stderr
    rows = np.genfromtxt(os.path.join(res["path"], "centroid_track.csv"),
                         delimiter=",", names=True)
    assert rows.size >= 5                                   # tracked a real fraction of steps
    cx = np.atleast_1d(rows["cx_px"])
    assert cx[-1] > cx[0] + 50                              # moved rightward with the target


# --------------------------------------------------------------------- numeric parity
def test_region_spectrum_matches_gottlux_on_planted_200hz(tmp_path):
    """The standalone spectrum's peak frequency == gottlux's region_spectrum (same events,
    same fs/band) on a planted 200 Hz stationary target, within a couple of FFT bins."""
    rec, _ = synthetic_scene(
        duration_s=1.5,
        targets=[FlutterTarget(flutter_hz=200.0, x0=160, y0=160, x1=160, y1=160,
                               events_per_burst=90, harmonics=(1.0,))],
        noise_rate_hz=5_000, static_clutter=0, seed=4)
    src = str(tmp_path / "planted.h5")
    h5io.write_hdf5(rec, src)
    roi = (120, 120, 200, 200)
    cfg = Config(freq_lo=100.0, freq_hi=400.0, fft_fs=2000.0)
    res = export_tool(src, "region_spectrum", out_dir=str(tmp_path), fmt="python",
                      cfg=cfg, roi=roi)
    r = _run_script(os.path.join(res["path"], "run_region_spectrum.py"))
    assert r.returncode == 0, r.stderr
    with open(os.path.join(res["path"], "region_spectrum.json")) as f:
        standalone = json.load(f)

    # gottlux's own answer over the exact same exported events + parameters
    bundled = eb.load(os.path.join(res["path"], "data.h5"), progress=lambda f_: None)
    ref = fq.region_spectrum(bundled.window(roi=roi).t, fs=2000.0, fmin=100.0, fmax=400.0)
    del bundled                                             # release the memmap (Windows)
    gc.collect()
    assert ref.detected
    assert abs(standalone["peak_hz"] - ref.peak_freq) <= 2.0    # same bin (df ≈ 0.7 Hz)
    assert abs(standalone["peak_hz"] - 200.0) <= 10.0           # and it IS the planted tone
    assert standalone["snr"] > 5.0


# --------------------------------------------------------------------- MATLAB sanity
_M_BLOCK = re.compile(r"^\s*(function|for|while|if|switch|try)\b", re.M)
_M_END = re.compile(r"^\s*end\b", re.M)


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_matlab_script_syntax_sanity(clip_h5, tmp_path, name):
    """No MATLAB on CI: assert the generated .m is non-empty, placeholder-free, names its
    tool, and every block keyword (function/for/if/…) has a matching line-start ``end``."""
    res = export_tool(clip_h5, name, out_dir=str(tmp_path), fmt="matlab")
    m_path = os.path.join(res["path"], f"run_{name}.m")
    assert os.path.exists(m_path)
    assert not os.path.exists(os.path.join(res["path"], f"run_{name}.py"))
    with open(m_path, encoding="utf-8") as f:
        text = f.read()
    assert text.strip() and not _PLACEHOLDER.search(text)
    assert text.lstrip().startswith("%")                    # a commented header
    assert "h5read" in text and "gl_load_events" in text    # native HDF5 + the loader
    starts = len(_M_BLOCK.findall(text))
    ends = len(_M_END.findall(text))
    assert starts == ends, f"{name}: {starts} block starts vs {ends} 'end's"


# --------------------------------------------------------------------- CLI
def test_cli_export_tool_list(capsys):
    from gottlux.cli import main
    assert main(["gottlux", "--export-tool", "list"]) == 0
    out = capsys.readouterr().out
    for name in TOOLS:
        assert name in out
        assert TOOLS[name].description.split("—")[0].strip()[:20] in out
    assert "MATLAB" in out
    assert list_tools_text() in out


def test_cli_bundle_roundtrip_honours_window_flags(clip_h5, tmp_path, capsys):
    """The full CLI route: window flags shape data.h5 exactly as --to-hdf5 would, and the
    bundle is complete (data + both scripts + README naming the baked parameters)."""
    from gottlux.cli import main
    assert main(["gottlux", clip_h5, "--export-tool", "event_rate",
                 "--tool-out", str(tmp_path), "--t_start", "0.2", "--t_stop", "0.9",
                 "--no_open"]) == 0
    out = capsys.readouterr().out
    bundles = [d for d in os.listdir(tmp_path) if d.startswith("clip_tool_event_rate_")]
    assert len(bundles) == 1 and bundles[0] in out
    bundle = os.path.join(str(tmp_path), bundles[0])
    for fname in ("data.h5", "run_event_rate.py", "run_event_rate.m", "README.md"):
        assert os.path.exists(os.path.join(bundle, fname))

    full = eb.load(clip_h5, progress=lambda f: None)
    sub = eb.load(os.path.join(bundle, "data.h5"), progress=lambda f: None)
    assert sub.n == full.window(0.2, 0.9).n                 # the window flags were honoured
    del full, sub
    gc.collect()

    with open(os.path.join(bundle, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    assert "rate_bin" in readme                             # the baked-parameter manifest
    assert "python run_event_rate.py" in readme             # how to run: Python...
    assert "MATLAB" in readme                               # ...and MATLAB
    assert "--to-hdf5" in readme                            # works on any exported .h5


def test_cli_unknown_tool_and_missing_path(capsys):
    from gottlux.cli import main
    assert main(["gottlux", "--export-tool", "not_a_tool"]) == 2
    assert "unknown tool" in capsys.readouterr().out
    assert main(["gottlux", "--export-tool", "event_rate"]) == 2   # no INPUT path
    assert "INPUT" in capsys.readouterr().out


def test_export_tool_rejects_bad_args(clip_h5, tmp_path):
    with pytest.raises(KeyError, match="unknown tool"):
        export_tool(clip_h5, "nope", out_dir=str(tmp_path))
    with pytest.raises(ValueError, match="python|matlab|both"):
        export_tool(clip_h5, "event_rate", out_dir=str(tmp_path), fmt="fortran")
    with pytest.raises(ValueError, match="viz mode"):
        export_tool(clip_h5, "viz_config", out_dir=str(tmp_path),
                    viz={"mode": "sparkle"})
    with pytest.raises(ValueError, match="tonemap"):
        export_tool(clip_h5, "viz_config", out_dir=str(tmp_path),
                    viz={"tonemap": "vibes"})


# --------------------------------------------------------------------- provenance README
def _sha256_of(path):
    """An independent SHA-256 (hashlib directly — not the library helper under test)."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_readme_carries_full_provenance(clip_h5, tmp_path):
    """The README states where the information came from — source absolute path, size,
    a REAL SHA-256 (recomputed here), geometry, format, window facts — plus the files
    table, the generating command line, and run instructions for both scripts."""
    res = export_tool(clip_h5, "event_rate", out_dir=str(tmp_path), fmt="both")
    with open(os.path.join(res["path"], "README.md"), encoding="utf-8") as f:
        readme = f.read()

    # where the information came from: path, size, hash (verified by recomputing)
    assert os.path.abspath(clip_h5) in readme
    assert f"{os.path.getsize(clip_h5):,} bytes" in readme
    assert _sha256_of(clip_h5) in readme
    assert _sha256_of(os.path.join(res["path"], "data.h5")) in readme  # bundle integrity
    rec = eb.load(clip_h5, progress=lambda f_: None)
    assert f"{rec.width} × {rec.height} px" in readme                  # sensor geometry
    assert rec.fmt in readme                                           # container format
    assert f"{res['n_events']:,}" in readme                            # exported count
    del rec
    gc.collect()

    # the section skeleton + the files table names every bundle file
    for section in ("## Data provenance", "## Files accessed and produced",
                    "## Baked parameters", "## How to run", "## Outputs"):
        assert section in readme, section
    for p in res["written"]:
        assert os.path.basename(p) in readme

    # the generating command line and the how-to-run essentials
    assert "gottlux" in readme and "--export-tool event_rate" in readme
    assert "python run_event_rate.py" in readme
    assert "pip install numpy scipy h5py" in readme
    assert "R2019a" in readme and "DATA_FILE" in readme
    assert "MATLAB" in readme
    # export identity
    from gottlux import __version__
    assert __version__ in readme


def test_provenance_json_machine_readable(clip_h5, tmp_path):
    """provenance.json parses, carries the required keys, and its facts are TRUE —
    hash/size/count recomputed or cross-checked here, window/ROI echoed exactly."""
    roi = (40, 40, 300, 300)
    cfg = Config(freq_lo=100.0, freq_hi=400.0, fft_fs=2000.0)
    res = export_tool(clip_h5, "region_spectrum", out_dir=str(tmp_path), fmt="both",
                      cfg=cfg, t0=0.1, t1=1.0, roi=roi)
    prov_path = os.path.join(res["path"], "provenance.json")
    assert prov_path in res["written"]
    with open(prov_path, encoding="utf-8") as f:
        prov = json.load(f)

    for key in ("schema", "gottlux_version", "created_utc", "tool", "source", "window",
                "bundle", "files", "parameters", "command", "how_to_run"):
        assert key in prov, key
    assert prov["tool"]["name"] == "region_spectrum"
    assert prov["source"]["path"] == os.path.abspath(clip_h5)
    assert prov["source"]["bytes"] == os.path.getsize(clip_h5)
    assert prov["source"]["sha256"] == _sha256_of(clip_h5)
    data_h5 = os.path.join(res["path"], "data.h5")
    assert prov["bundle"]["data_h5"]["sha256"] == _sha256_of(data_h5)
    assert prov["bundle"]["data_h5"]["bytes"] == os.path.getsize(data_h5)

    # the exported-window facts match what actually landed in data.h5
    assert prov["window"]["n_events"] == res["n_events"]
    assert prov["window"]["roi"] == list(roi)
    assert prov["window"]["t_start_s"] == pytest.approx(0.1, abs=1e-6)
    assert prov["window"]["t_stop_s"] == pytest.approx(1.0, abs=1e-6)
    assert prov["window"]["duration_s"] == pytest.approx(0.9, abs=1e-6)
    assert prov["window"]["windowed"] is True

    # every bundle file appears in the files table, with the source marked as read
    names = {row["name"] for row in prov["files"]}
    for p in res["written"]:
        assert os.path.basename(p) in names
    reads = [row for row in prov["files"] if row["access"] == "read"]
    assert any(row["name"] == os.path.abspath(clip_h5) for row in reads)

    # parameters echo the manifest keys; the command regenerates the same shape
    for key, _desc in TOOLS["region_spectrum"].params:
        assert key in prov["parameters"]
    assert "--export-tool region_spectrum" in prov["command"]
    assert "--roi 40,40,300,300" in prov["command"]
    assert "--t_start 0.1" in prov["command"] and "--t_stop 1" in prov["command"]
    assert prov["how_to_run"]["python"] and prov["how_to_run"]["matlab"]


# --------------------------------------------------------------------- viz_config
def test_viz_config_defaults_baked(clip_h5, tmp_path):
    """With no viz flags the exported scripts carry the documented defaults: count mode,
    inferno (Python) / its MATLAB equivalent, sqrt tonemap, 20 ms accumulation."""
    res = export_tool(clip_h5, "viz_config", out_dir=str(tmp_path), fmt="both")
    with open(os.path.join(res["path"], "run_viz_config.py"), encoding="utf-8") as f:
        py = f.read()
    with open(os.path.join(res["path"], "run_viz_config.m"), encoding="utf-8") as f:
        m = f.read()
    assert '"count"' in py and '"inferno"' in py and '"sqrt"' in py
    assert re.search(r"ACCUM_MS\s*=\s*20\b", py)
    assert "'count'" in m and "'sqrt'" in m
    assert "'hot'" in m                                     # inferno's base-MATLAB stand-in
    assert "import gottlux" not in py


def test_viz_config_cli_flags_reach_scripts_and_run(clip_h5, tmp_path, capsys):
    """The full CLI route: the four viz flags land in both scripts and provenance.json,
    and the generated Python script renders the polarity view as a bare subprocess."""
    from gottlux.cli import main
    assert main(["gottlux", clip_h5, "--export-tool", "viz_config",
                 "--tool-out", str(tmp_path), "--viz_mode", "polarity",
                 "--viz_cmap", "coolwarm", "--viz_tonemap", "log",
                 "--viz_accum_ms", "10", "--no_open"]) == 0
    capsys.readouterr()
    bundles = [d for d in os.listdir(tmp_path) if d.startswith("clip_tool_viz_config_")]
    assert len(bundles) == 1
    bundle = os.path.join(str(tmp_path), bundles[0])

    with open(os.path.join(bundle, "run_viz_config.py"), encoding="utf-8") as f:
        py = f.read()
    assert '"polarity"' in py and '"coolwarm"' in py and '"log"' in py
    assert re.search(r"ACCUM_MS\s*=\s*10\b", py)
    with open(os.path.join(bundle, "run_viz_config.m"), encoding="utf-8") as f:
        m = f.read()
    assert "'polarity'" in m and "'log'" in m
    assert "'jet'" in m                                     # coolwarm's base-MATLAB stand-in

    with open(os.path.join(bundle, "provenance.json"), encoding="utf-8") as f:
        prov = json.load(f)
    assert prov["parameters"]["viz_mode"] == "polarity"
    assert prov["parameters"]["viz_cmap"] == "coolwarm"
    assert prov["parameters"]["viz_tonemap"] == "log"
    assert prov["parameters"]["viz_accum_ms"] == "10"
    assert "--viz_mode polarity" in prov["command"]

    r = _run_script(os.path.join(bundle, "run_viz_config.py"))
    assert r.returncode == 0, r.stderr
    assert os.path.exists(os.path.join(bundle, "viz_frame_01.png"))
    with np.load(os.path.join(bundle, "viz_frames.npz")) as z:
        frames = z["frames"]
        assert frames.ndim == 3 and frames.shape[0] >= 1
        assert float(frames.min()) >= -1.0 and float(frames.max()) <= 1.0  # signed display
        assert float(frames.min()) < 0.0 < float(frames.max())  # both polarities rendered


def test_viz_config_renders_all_snapshots_and_honours_roi(clip_h5, tmp_path):
    """Count mode: every documented snapshot PNG appears and the baked ROI crops the
    rendered frames to the requested view rectangle."""
    roi = (40, 100, 280, 220)
    res = export_tool(clip_h5, "viz_config", out_dir=str(tmp_path), fmt="python", roi=roi)
    r = _run_script(os.path.join(res["path"], "run_viz_config.py"))
    assert r.returncode == 0, r.stderr
    pngs = sorted(f for f in os.listdir(res["path"])
                  if re.fullmatch(r"viz_frame_\d\d\.png", f))
    assert len(pngs) == 6                                   # the baked snap_count
    with np.load(os.path.join(res["path"], "viz_frames.npz")) as z:
        frames = z["frames"]
        assert frames.shape[1:] == (roi[3] - roi[1], roi[2] - roi[0])  # (H, W) of the crop
        assert float(frames.min()) >= 0.0 and float(frames.max()) <= 1.0
        assert float(frames.max()) > 0.0                    # rendered actual structure
