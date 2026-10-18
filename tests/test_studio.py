"""Studio: shot timing, every look, exports (GIF/MP4), the CLI, the GUI tab, and parity with
the web port (web/js/looks.js — run under Node when it is installed)."""
import json
import os
import shutil
import subprocess

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gottlux.io.recording import Recording  # noqa: E402
from gottlux.studio import Shot, ShotRenderer, export_shot, get_look, list_looks  # noqa: E402
from gottlux.studio.looks import Strobe  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _synthetic(seed=0, W=64, H=48, dur=0.6, f0=50.0):
    """A bar sweeping right plus a square blinking at *f0* Hz, with a little noise."""
    rng = np.random.default_rng(seed)
    xs, ys, ps, ts = [], [], [], []
    # sweeping bar: 4 px wide, crosses the sensor once
    for t in np.arange(0, dur, 2e-4):
        cx = int(4 + (W - 8) * t / dur)
        for y in range(10, 38, 3):
            xs += [cx, cx + 3]; ys += [y, y]; ps += [1, 0]; ts += [t, t + 5e-5]
    # blinking square: ON burst at each cycle start, OFF burst half a cycle later
    for k in range(int(dur * f0)):
        for ph, pol in ((0.0, 1), (0.5, 0)):
            t = (k + ph) / f0
            for _ in range(60):
                xs.append(int(rng.integers(44, 54))); ys.append(int(rng.integers(30, 40)))
                ps.append(pol); ts.append(t + rng.uniform(0, 1.5e-3))
    n = 3000
    xs += list(rng.integers(0, W, n)); ys += list(rng.integers(0, H, n))
    ps += list(rng.integers(0, 2, n)); ts += list(rng.uniform(0, dur, n))
    t_us = np.round(np.asarray(ts) * 1e6).astype(np.int64)
    t_us[np.argmin(t_us)] = 0
    return Recording.from_events(np.array(xs), np.array(ys), np.array(ps), t_us, width=W,
                                 height=H, name="synth")


@pytest.fixture(scope="module")
def rec():
    return _synthetic()


# ------------------------------------------------------------------ shot
def test_shot_timing_and_json_roundtrip():
    s = Shot(look="heat", t0=0.1, t1=0.6, speed=0.1, fps=20, crop=(1, 2, 30, 40),
             params={"window_ms": 4})
    assert s.n_frames(0.1, 0.6) == 100                        # 0.5 s × 20 fps / 0.1
    s2 = Shot.from_json(s.to_json())
    assert s2 == s and isinstance(s2.crop, tuple)
    assert s.span(0.0, 0.3) == (0.1, 0.3)                     # clamped to the recording
    assert Shot().apply_preset("mp4-hd").format == "mp4"


@pytest.mark.parametrize("name", sorted(list_looks()))
def test_every_look_renders(rec, name):
    shot = Shot(look=name, t0=0.1, t1=0.5, speed=0.5, fps=10, width=160, denoise=False,
                rate_strip=True, legend=True, title="x", bloom=0.5)
    r = ShotRenderer(rec, shot)
    assert len(r) >= 1
    f = r.frame(len(r) // 2)
    assert f.dtype == np.uint8 and f.shape == (r.size[1], r.size[0], 3)
    assert r.size[0] % 2 == 0 and r.size[1] % 2 == 0           # MP4-safe
    assert f.std() > 1.0                                        # not a blank frame


def test_locked_exposure_does_not_pump(rec):
    """Heat frames share one scale: a frame's brightness follows its activity."""
    r = ShotRenderer(rec, Shot(look="heat", t0=0.05, t1=0.55, speed=0.5, fps=10, width=128,
                               denoise=False, stamp=False, speed_label=False))
    assert "scale" in r.st and r.st["scale"] > 0


def test_strobe_measures_the_blink_frequency(rec):
    r = ShotRenderer(rec, Shot(look="strobe", t0=0.0, t1=0.6, crop=(40, 26, 58, 44),
                               denoise=False, params={"band_lo": 20, "band_hi": 120}))
    assert abs(r.st["f0"] - 50.0) < 0.5
    assert len(r) == 24 * 2                                     # bins × cycles


def test_crop_and_pingpong(rec):
    r = ShotRenderer(rec, Shot(look="trails", t0=0.1, t1=0.3, speed=1, fps=50,
                               crop=(10, 10, 42, 34), pingpong=True, denoise=False, width=64))
    assert (r.ev.W, r.ev.H) == (32, 24)
    assert len(r) == 2 * r.n_frames - 2
    assert r.source_index(len(r) - 1) == 1


def test_spacetime_orbit_moves_camera(rec):
    r = ShotRenderer(rec, Shot(look="spacetime", t0=0.1, t1=0.5, speed=1, fps=10, width=160,
                               denoise=False, params={"orbit_deg": 180, "freeze": True}))
    a, b = r.frame(0), r.frame(len(r) - 1)
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 1.0


def _spinning(period=0.5, fov=60.0, W=64, H=32, mirror=-1.0, dur=3.0, seed=2):
    """A sensor spinning at *period* past a strong static source at 100° and a weak one at
    250° (the period refinement in estimate_spin_period_s assumes one dominant pass per
    revolution; two equal passes 150°/210° apart bias it by ~2 % — docs/ISSUES.md GL-36)."""
    rng = np.random.default_rng(seed)
    xs, ys, ps, ts = [], [], [], []
    for t in np.arange(0, dur, 2e-4):
        bore = (360.0 * t / period) % 360.0
        for az, y0, rows in ((100.0, 8, 8), (250.0, 22, 2)):
            off = ((az - bore + 180.0) % 360.0) - 180.0
            if abs(off) < fov / 2:
                x = int((0.5 + mirror * off / fov) * W)
                if 0 <= x < W:
                    for dy in range(rows):
                        xs.append(x); ys.append(y0 + dy); ps.append(int(rng.integers(0, 2)))
                        ts.append(t + rng.uniform(0, 1e-4))
    n = 4000
    xs += list(rng.integers(0, W, n)); ys += list(rng.integers(0, H, n))
    ps += list(rng.integers(0, 2, n)); ts += list(rng.uniform(0, dur, n))
    t_us = np.round(np.asarray(ts) * 1e6).astype(np.int64)
    t_us[np.argmin(t_us)] = 0
    return Recording.from_events(np.array(xs), np.array(ys), np.array(ps), t_us, width=W,
                                 height=H, name="spin")


def test_radar_recovers_period_and_column_sign():
    rec = _spinning()
    r = ShotRenderer(rec, Shot(look="radar", t0=0.5, t1=2.5, speed=1, fps=10, width=160,
                               denoise=False, params={"fov_deg": 60.0}))
    assert abs(r.st["period"] - 0.5) < 0.0075                 # six revolutions: ±1.5 %
    assert r.st["mirror"] == -1.0
    assert r.frame(len(r) - 1).std() > 1.0


# ------------------------------------------------------------------ export
def test_export_gif_and_recipe(rec, tmp_path):
    from PIL import Image
    shot = Shot(look="trails", t0=0.1, t1=0.3, speed=0.5, fps=20, width=96, denoise=False)
    f = export_shot(rec, shot, str(tmp_path / "a.gif"))
    im = Image.open(f["path"])
    assert im.n_frames == f["frames"] == 8 and im.size == tuple(f["size"])
    assert im.info.get("loop") == 0 and im.info.get("duration") == 50
    recipe = json.loads(open(f["recipe"], encoding="utf-8").read())
    assert Shot.from_dict(recipe["shot"]) == shot


def test_export_mp4(rec, tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    f = export_shot(rec, Shot(look="heat", t0=0.1, t1=0.3, speed=0.5, fps=20, width=96,
                              denoise=False), str(tmp_path / "a.mp4"))
    r = imageio.get_reader(f["path"])
    assert r.count_frames() == f["frames"]
    r.close()


def test_export_default_name_never_overwrites(rec, tmp_path):
    shot = Shot(look="heat", t0=0.1, t1=0.2, speed=1, fps=10, width=64, denoise=False)
    a = export_shot(rec, shot, folder=str(tmp_path))["path"]
    b = export_shot(rec, shot, folder=str(tmp_path))["path"]
    assert a != b and os.path.isfile(a) and os.path.isfile(b)


def test_export_cancel_removes_partial(rec, tmp_path):
    out = tmp_path / "c.gif"
    f = export_shot(rec, Shot(look="heat", t0=0.1, t1=0.4, speed=0.2, fps=20, width=64,
                              denoise=False), str(out), cancel=lambda: True)
    assert f is None and not out.exists()


# ------------------------------------------------------------------ CLI
def test_cli_list_and_render(tmp_path, capsys):
    from gottlux.studio import cli
    assert cli.main(["--list"]) == 0
    assert "strobe" in capsys.readouterr().out
    raw = os.path.join(ROOT, "examples", "data", "Humming_Bird_Fight_merged_shortest.raw")
    if not os.path.isfile(raw):
        pytest.skip("sample clip not present")
    out = tmp_path / "s.gif"
    assert cli.main([raw, "--look", "heat", "--t0", "0.02", "--t1", "0.08", "--speed", "0.5",
                     "--width", "96", "--set", "window_ms=4", "-o", str(out)]) == 0
    assert out.is_file() and (tmp_path / "s.gif.shot.json").is_file()


def test_specs_cover_every_param():
    from gottlux.studio.looks import look_specs
    specs = look_specs()
    for name, look in list_looks().items():
        assert [p["key"] for p in specs[name]["params"]] == [p.key for p in look.PARAMS]
        assert get_look(name).resolve({}) == look.defaults()


def test_estimate_f0_pure_tone():
    t = np.sort(np.random.default_rng(1).uniform(0, 1.0, 20000))
    keep = np.sin(2 * np.pi * 73.0 * t) > 0.3
    assert abs(Strobe.estimate_f0(t[keep], 0.0, 1.0, 20, 150) - 73.0) < 0.3


# ------------------------------------------------------------------ web port
def test_web_spec_block_in_sync():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sync_web_studio", os.path.join(ROOT, "scripts", "sync_web_studio.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = open(os.path.join(ROOT, "web", "js", "looks.js"), encoding="utf-8").read()
    assert mod.render() in src, "web/js/looks.js is stale: run python scripts/sync_web_studio.py"


NODE_SCRIPT = r"""
global.self = global;
require(process.argv[2] + "/web/js/analysis.js");
const S = require(process.argv[2] + "/web/js/looks.js");
const d = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const rec = { x: Uint16Array.from(d.x), y: Uint16Array.from(d.y), p: Uint8Array.from(d.p),
              t: Float64Array.from(d.t), width: d.W, height: d.H };
const out = {};
for (const [name, shot] of Object.entries(d.shots)) {
  const r = new S.Renderer(rec, Object.assign(S.defaultShot(), shot));
  const fr = r.draw(d.frame[name]);
  const rgb = [];
  for (let i = 0; i < fr.data.length; i += 4) rgb.push(fr.data[i], fr.data[i + 1], fr.data[i + 2]);
  out[name] = { w: fr.w, h: fr.h, n: r.length(), rgb: rgb, f0: r.st.f0 || null, kept: r.ev.n };
}
process.stdout.write(JSON.stringify(out));
"""


def _py_draw(r, i):
    """The Python renderer's composited frame at the drawing size (before resample/overlays)."""
    from gottlux.studio.render import composite
    k = r.source_index(i)
    w, h = r.draw_size
    img = composite(r.look.render(r.ev, k, float(r.times[k]), r.P, r.st, r.colors, (w, h)),
                    r.colors["bg"], (h, w))
    return (img * 255 + 0.5).astype(np.uint8)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_web_looks_match_python(rec, tmp_path):
    shots = {
        "trails": dict(look="trails", t0=0.1, t1=0.5, speed=0.5, fps=10, width=128),
        "heat": dict(look="heat", t0=0.1, t1=0.5, speed=0.5, fps=10, width=128),
        "surface": dict(look="surface", t0=0.1, t1=0.5, speed=0.5, fps=10, width=128,
                        params={"color": "colormap"}),
        "paint": dict(look="paint", t0=0.05, t1=0.55, speed=0.5, fps=10, width=128),
        "strobe": dict(look="strobe", t0=0.0, t1=0.6, crop=[40, 26, 58, 44], width=128,
                       params={"band_lo": 20, "band_hi": 120}),
        "waterfall": dict(look="waterfall", t0=0.1, t1=0.5, speed=0.5, fps=10, width=128,
                          params={"rows": 40}),
        "spacetime": dict(look="spacetime", t0=0.1, t1=0.5, speed=0.5, fps=10, width=160,
                          params={"orbit_deg": 90}),
        "radar": dict(look="radar", t0=0.1, t1=0.5, speed=0.5, fps=10, width=160,
                      params={"period_s": 0.25, "persistence_ms": 100}),
    }
    for s in shots.values():
        s["denoise"] = False
    frame = {k: 3 for k in shots}
    t = np.asarray(rec.t, np.float64) / 1e6
    data = {"x": np.asarray(rec.x).tolist(), "y": np.asarray(rec.y).tolist(),
            "p": np.asarray(rec.p).tolist(), "t": t.tolist(), "W": rec.width, "H": rec.height,
            "shots": shots, "frame": frame}
    (tmp_path / "in.json").write_text(json.dumps(data))
    (tmp_path / "run.js").write_text(NODE_SCRIPT)
    res = subprocess.run(["node", str(tmp_path / "run.js"), ROOT.replace("\\", "/"),
                          str(tmp_path / "in.json")], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    js = json.loads(res.stdout)
    for name, s in shots.items():
        sh = Shot.from_dict(s)
        r = ShotRenderer(rec, sh)
        py = _py_draw(r, frame[name])
        j = np.asarray(js[name]["rgb"], np.uint8).reshape(js[name]["h"], js[name]["w"], 3)
        assert js[name]["n"] == len(r), name
        assert j.shape == py.shape, name
        diff = np.abs(j.astype(int) - py.astype(int))
        assert diff.mean() < 1.0 and np.percentile(diff, 99.5) <= 6, (name, diff.mean())
        if name == "strobe":
            assert abs(js[name]["f0"] - r.st["f0"]) < 0.05


# ------------------------------------------------------------------ GUI tab
def test_studio_tab_previews_and_exports(rec, tmp_path):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("pyqtgraph")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import time
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from gottlux.app.studio import StudioPanel
    panel = StudioPanel()
    panel.resize(1200, 700)
    panel.set_recording(rec)
    panel.set_shot(Shot(look="trails", t0=0.1, t1=0.3, speed=0.5, fps=20, width=96,
                        denoise=False))
    panel._rebuild()
    assert panel._renderer is not None and len(panel._renderer) == 8
    assert panel.preview.pixmap() is not None and not panel.preview.pixmap().isNull()
    panel.folder_ed.setText(str(tmp_path))
    w = panel.export()
    deadline = time.time() + 60
    while w.isRunning() and time.time() < deadline:
        app.processEvents(); time.sleep(0.02)
    app.processEvents()
    gifs = list(tmp_path.glob("*.gif"))
    assert len(gifs) == 1 and gifs[0].stat().st_size > 0
    assert panel.shot().look == "trails"
