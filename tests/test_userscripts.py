"""
Tests for the user-script runner (:mod:`gottlux.userscripts` + ``--run-script``).

The promise under test: a plain ``.py`` file defining ``process(win, ctx)`` runs on the
recording — or on exactly the requested window/ROI — and whatever it returns is saved per
the contract, with a provenance README beside it. Covered here:

* **dispatch**, one branch per contract return type: ``None`` (nothing saved), a dict of
  name → array/scalar (``results.npz`` + printed summary), a matplotlib ``Figure``
  (``figure.png`` + ``figure.pdf``), and ``{"events": (x, y, p, t)}`` (a derived
  ``derived.raw`` gottlux itself reads back);
* **window/ROI fidelity**: the script sees only the requested slice, and ``win.rec`` /
  ``win.roi`` / the resolved ``ctx`` bounds are exactly as documented;
* **error isolation**: a missing file, a broken import, a raising ``process()``, a
  missing ``process``, and an unsupported return each surface as one clean
  :class:`~gottlux.userscripts.UserScriptError` — never a crash;
* **the CLI**: ``--run-script`` end-to-end off a real encoded ``.raw`` with a
  monkeypatched ``sys.argv``, ``--script-args`` forwarding, the no-INPUT and
  script-failure exits;
* **README provenance**: script + recording SHA-256, window/ROI, version, wall time;
* the bundled ``examples/user_script_example.py`` runs under the same contract.
"""
import gc
import os
import sys
import textwrap

import numpy as np
import pytest

import gottlux as eb
from gottlux import __version__
from gottlux.io.paths import file_sha256
from gottlux.io.recording import Recording
from gottlux.userscripts import UserScriptError, load_script, run_script

#: The window/ROI used by the fidelity + CLI tests (x in [10, 50), full-height, mid-half s).
T0, T1 = 0.25, 0.5
ROI = (10, 0, 50, 240)


@pytest.fixture(scope="module")
def rec():
    """A fully deterministic in-memory recording: 2000 events at 2 kHz over ~1 s,
    x sweeping 0..319, alternating polarity — every slice is predictable by hand."""
    n = 2000
    i = np.arange(n)
    return Recording.from_events(x=i % 320, y=(i // 320) % 240, p=(i % 2),
                                 t_us=i * 500, width=320, height=240, name="synthclip")


def _script(tmp_path, body, name="script.py") -> str:
    """Write a user script to *tmp_path* and return its path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(p)


def _run_folder_files(res) -> set:
    return set(os.listdir(res["folder"]))


# --------------------------------------------------------------------- dispatch: None
def test_none_return_saves_nothing_but_readme(rec, tmp_path):
    s = _script(tmp_path, """
        def process(win, ctx):
            return None
    """)
    res = run_script(s, rec, output_dir=str(tmp_path))
    assert res["result_kind"] == "none"
    assert res["outputs"] == []
    assert _run_folder_files(res) == {"README.md"}         # provenance only, no payloads
    assert res["n_events"] == rec.n                        # no window -> the full stream


# --------------------------------------------------------------- dispatch: dict of arrays
def test_dict_of_arrays_saved_as_npz_with_printed_summary(rec, tmp_path, capsys):
    s = _script(tmp_path, """
        import numpy as np
        def process(win, ctx):
            return {"a": np.arange(5, dtype=np.float64), "b": 3.5}
    """)
    res = run_script(s, rec, output_dir=str(tmp_path))
    assert res["result_kind"] == "arrays"
    with np.load(os.path.join(res["folder"], "results.npz")) as z:
        assert set(z.files) == {"a", "b"}
        assert np.array_equal(z["a"], np.arange(5, dtype=np.float64))
        assert float(z["b"]) == 3.5
    out = capsys.readouterr().out                          # the contract's printed summary
    assert "results.npz" in out
    assert "a (5,)" in out and "b = 3.5" in out
    assert "s wall" in out                                 # the wall-time report


# --------------------------------------------------------------------- dispatch: Figure
def test_figure_saved_as_png_and_pdf(rec, tmp_path):
    pytest.importorskip("matplotlib")
    s = _script(tmp_path, """
        def process(win, ctx):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(win.t_s, win.x, ".")
            return fig
    """)
    res = run_script(s, rec, output_dir=str(tmp_path))
    assert res["result_kind"] == "figure"
    files = _run_folder_files(res)
    assert {"figure.png", "figure.pdf", "README.md"} <= files


# --------------------------------------------------------------- dispatch: derived events
def test_events_dict_writes_a_raw_gottlux_reads_back(rec, tmp_path):
    """{'events': ...} becomes derived.raw (EVT2.1) — and extra keys still land in the
    NPZ, so a script can emit a filtered stream AND its bookkeeping in one return."""
    s = _script(tmp_path, """
        import numpy as np
        def process(win, ctx):
            keep = slice(None, None, 2)                     # every other event
            return {"events": (win.x[keep], win.y[keep], win.p[keep], win.t[keep]),
                    "n_kept": np.int64(len(win.x[keep]))}
    """)
    res = run_script(s, rec, output_dir=str(tmp_path))
    assert res["result_kind"] == "events+arrays"
    raw = os.path.join(res["folder"], "derived.raw")
    assert os.path.exists(raw)
    derived = eb.load(raw, progress=lambda f: None)        # gottlux reads its own output
    try:
        assert derived.n == rec.n // 2
        assert derived.width == rec.width and derived.height == rec.height
    finally:
        del derived                                        # release the memmap (Windows)
        gc.collect()
    with np.load(os.path.join(res["folder"], "results.npz")) as z:
        assert int(z["n_kept"]) == rec.n // 2


# --------------------------------------------------------------- window / ROI fidelity
def test_script_sees_only_the_requested_slice(rec, tmp_path):
    """The script's entire view — win arrays, win.rec/win.roi, ctx bounds — matches the
    requested [t0, t1) x ROI exactly, verified against a hand-built mask."""
    s = _script(tmp_path, """
        import numpy as np
        def process(win, ctx):
            return {"n": np.int64(win.n),
                    "t_min_us": win.t.min(), "t_max_us": win.t.max(),
                    "x_min": win.x.min(), "x_max": win.x.max(),
                    "roi_ok": np.uint8(win.roi == (10, 0, 50, 240)),
                    "rec_ok": np.uint8(win.rec is ctx["rec"]),
                    "t0": ctx["t0"], "t1": ctx["t1"]}
    """)
    res = run_script(s, rec, t0=T0, t1=T1, roi=ROI, output_dir=str(tmp_path))
    t_s = np.asarray(rec.t, np.float64) / 1e6
    mask = (t_s >= T0) & (t_s < T1) & (np.asarray(rec.x) >= ROI[0]) \
        & (np.asarray(rec.x) < ROI[2])
    expected = int(mask.sum())
    assert expected > 0                                    # the fixture really has events there
    assert res["n_events"] == expected
    with np.load(os.path.join(res["folder"], "results.npz")) as z:
        assert int(z["n"]) == expected
        assert int(z["t_min_us"]) >= int(T0 * 1e6)
        assert int(z["t_max_us"]) < int(T1 * 1e6)
        assert int(z["x_min"]) >= ROI[0] and int(z["x_max"]) < ROI[2]
        assert int(z["roi_ok"]) == 1 and int(z["rec_ok"]) == 1
        assert float(z["t0"]) == T0 and float(z["t1"]) == T1


def test_full_span_resolved_into_ctx(rec, tmp_path):
    """With no window requested, ctx carries the recording's real span (not None)."""
    s = _script(tmp_path, """
        def process(win, ctx):
            return {"t0": ctx["t0"], "t1": ctx["t1"], "n": win.n}
    """)
    res = run_script(s, rec, output_dir=str(tmp_path))
    with np.load(os.path.join(res["folder"], "results.npz")) as z:
        assert float(z["t0"]) == rec.t_start_s
        assert float(z["t1"]) == rec.t_stop_s
        assert int(z["n"]) == rec.n                        # [None, None] loses no events


# --------------------------------------------------------------------- error isolation
def test_missing_and_non_py_files_fail_cleanly(rec, tmp_path):
    with pytest.raises(UserScriptError, match="not found"):
        run_script(str(tmp_path / "nope.py"), rec, output_dir=str(tmp_path))
    txt = tmp_path / "notes.txt"
    txt.write_text("not python", encoding="utf-8")
    with pytest.raises(UserScriptError, match=r"\.py file"):
        load_script(str(txt))


def test_broken_import_and_missing_process_fail_cleanly(rec, tmp_path):
    s = _script(tmp_path, "import definitely_not_a_module_xyz\n", name="broken.py")
    with pytest.raises(UserScriptError, match="failed to import"):
        run_script(s, rec, output_dir=str(tmp_path))
    s2 = _script(tmp_path, "VALUE = 42\n", name="no_process.py")
    with pytest.raises(UserScriptError, match=r"process\(win, ctx\)"):
        run_script(s2, rec, output_dir=str(tmp_path))


def test_raising_script_surfaces_one_clean_error_with_cause(rec, tmp_path):
    s = _script(tmp_path, """
        def process(win, ctx):
            raise ValueError("boom")
    """, name="raiser.py")
    with pytest.raises(UserScriptError, match="ValueError: boom") as ei:
        run_script(s, rec, output_dir=str(tmp_path))
    assert isinstance(ei.value.__cause__, ValueError)      # the original rides along


def test_unsupported_return_type_is_loud(rec, tmp_path):
    s = _script(tmp_path, """
        def process(win, ctx):
            return 42
    """, name="bad_return.py")
    with pytest.raises(UserScriptError, match="unsupported int"):
        run_script(s, rec, output_dir=str(tmp_path))


def test_edited_script_reruns_fresh(rec, tmp_path):
    """load_script never serves a stale cached module: an edit takes effect immediately."""
    s = _script(tmp_path, "def process(win, ctx):\n    return {'v': 1}\n", name="edit.py")
    res1 = run_script(s, rec, output_dir=str(tmp_path))
    _script(tmp_path, "def process(win, ctx):\n    return {'v': 2}\n", name="edit.py")
    res2 = run_script(s, rec, output_dir=str(tmp_path))
    with np.load(os.path.join(res1["folder"], "results.npz")) as z:
        assert int(z["v"]) == 1
    with np.load(os.path.join(res2["folder"], "results.npz")) as z:
        assert int(z["v"]) == 2


# --------------------------------------------------------------------- README provenance
def test_readme_records_full_provenance(rec, tmp_path):
    s = _script(tmp_path, """
        def process(win, ctx):
            return None
    """, name="prov.py")
    res = run_script(s, rec, t0=T0, t1=T1, roi=ROI, output_dir=str(tmp_path),
                     script_args=["alpha", "7"])
    with open(os.path.join(res["folder"], "README.md"), encoding="utf-8") as f:
        readme = f.read()
    assert "SHA-256" in readme
    assert file_sha256(s) in readme                        # the exact script hash
    assert "in-memory" in readme                           # no source file to hash
    assert "[0.25, 0.5] s" in readme                       # the window, resolved
    assert "10,0,50,240" in readme                         # the ROI
    assert __version__ in readme                           # gottlux version
    assert "wall time" in readme
    assert "alpha" in readme                               # the script args


# ------------------------------------------------------------------------------- CLI
@pytest.fixture()
def clip_raw(rec, tmp_path):
    """The fixture recording encoded to a real EVT2.1 .raw (the CLI route decodes it)."""
    from gottlux.io import writer
    path = str(tmp_path / "clip.raw")
    writer.write_raw(path, rec.x, rec.y, rec.p, rec.t, width=rec.width, height=rec.height)
    return path


def test_cli_end_to_end_with_monkeypatched_argv(rec, clip_raw, tmp_path, monkeypatch, capsys):
    """The full route a user takes: gottlux clip.raw --run-script ... with the window/ROI
    flags — via a monkeypatched sys.argv, exactly as the console entry point sees it."""
    from gottlux.cli import main
    s = _script(tmp_path, """
        import numpy as np
        def process(win, ctx):
            return {"n": np.int64(win.n)}
    """, name="count.py")
    out_dir = tmp_path / "runs"
    out_dir.mkdir()
    monkeypatch.setattr(sys, "argv",
                        ["gottlux", clip_raw, "--run-script", s,
                         "--t_start", str(T0), "--t_stop", str(T1),
                         "--roi", ",".join(str(v) for v in ROI),
                         "--out", str(out_dir), "--no_open"])
    assert main() == 0
    printed = capsys.readouterr().out
    folders = [d for d in os.listdir(out_dir) if d.startswith("gottlux_script_count_")]
    assert len(folders) == 1 and folders[0] in printed
    folder = os.path.join(str(out_dir), folders[0])

    # the decoded, windowed count matches the hand-built mask on the original arrays
    t_s = np.asarray(rec.t, np.float64) / 1e6
    mask = (t_s >= T0) & (t_s < T1) & (np.asarray(rec.x) >= ROI[0]) \
        & (np.asarray(rec.x) < ROI[2])
    with np.load(os.path.join(folder, "results.npz")) as z:
        assert int(z["n"]) == int(mask.sum())

    with open(os.path.join(folder, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    assert file_sha256(s) in readme                        # script hash
    assert file_sha256(clip_raw) in readme                 # source recording hash
    assert "[0.25, 0.5] s" in readme and "10,0,50,240" in readme


def test_cli_forwards_script_args(clip_raw, tmp_path):
    from gottlux.cli import main
    s = _script(tmp_path, """
        import numpy as np
        def process(win, ctx):
            return {"n_args": np.int64(len(ctx["args"])),
                    "first": float(ctx["args"][0])}
    """, name="argsy.py")
    out_dir = tmp_path / "runs_args"
    out_dir.mkdir()
    assert main(["gottlux", clip_raw, "--run-script", s,
                 "--script-args", "0.02 fast", "--out", str(out_dir), "--no_open"]) == 0
    folder = os.path.join(str(out_dir), os.listdir(out_dir)[0])
    with np.load(os.path.join(folder, "results.npz")) as z:
        assert int(z["n_args"]) == 2
        assert float(z["first"]) == 0.02


def test_cli_script_failure_exits_cleanly(clip_raw, tmp_path, capsys):
    from gottlux.cli import main
    s = _script(tmp_path, """
        def process(win, ctx):
            raise RuntimeError("kaput")
    """, name="kaput.py")
    code = main(["gottlux", clip_raw, "--run-script", s,
                 "--out", str(tmp_path), "--no_open"])
    assert code == 2
    out = capsys.readouterr().out
    assert "user script failed" in out and "kaput" in out


def test_cli_run_script_needs_input(tmp_path, capsys):
    from gottlux.cli import main
    s = _script(tmp_path, "def process(win, ctx):\n    return None\n")
    assert main(["gottlux", "--run-script", s]) == 2
    assert "INPUT" in capsys.readouterr().out


# --------------------------------------------------------------------- bundled example
def test_bundled_example_runs_under_the_contract(rec, tmp_path):
    """examples/user_script_example.py — the documented worked example — runs on the
    fixture recording, returns a Figure, and writes its own NPZ into the run folder."""
    pytest.importorskip("matplotlib")
    example = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "examples", "user_script_example.py")
    assert os.path.exists(example)
    res = run_script(example, rec, t0=0.1, t1=0.6, output_dir=str(tmp_path))
    assert res["result_kind"] == "figure"
    files = _run_folder_files(res)
    assert {"figure.png", "figure.pdf", "polarity_rates.npz", "README.md"} <= files
    with np.load(os.path.join(res["folder"], "polarity_rates.npz")) as z:
        # alternating polarity in the fixture -> the whole-window ON/OFF ratio is ~1
        assert abs(float(z["global_ratio"]) - 1.0) < 0.05
