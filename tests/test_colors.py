"""The colour registry: built-ins, user colormaps and palettes, persistence, bit-exact
lookups, and every consumer resolving a user scheme by name."""
import json
import os
import warnings

import numpy as np
import pytest

from gottlux.core import colors


@pytest.fixture
def schemes(tmp_path, monkeypatch):
    """A fresh, empty user-schemes file for one test."""
    path = tmp_path / "colors.json"
    monkeypatch.setenv("GOTTLUX_COLORS", str(path))
    colors.reload()
    yield path
    monkeypatch.undo()
    colors.reload()


def test_builtins_come_first_and_resolve_everywhere(schemes):
    import matplotlib
    names = colors.colormap_names()
    assert names[:3] == ["inferno", "viridis", "magma"]
    for name in ("ice", "ember", "gottlux_events", "gottlux_polarity"):
        assert name in names and name in matplotlib.colormaps
    assert colors.palette_names()[0] == "fire-ice"
    assert colors.palette("nope") == colors.palette("fire-ice")


def test_user_colormap_persists_reloads_and_reaches_matplotlib(schemes):
    import matplotlib
    from gottlux.core import tonemap
    from gottlux.viz import video
    name = colors.add_colormap("sunset", ["#1a0a2e", "#c2185b", "#ffb347"])
    assert name == "sunset" and colors.is_user_colormap("sunset")
    assert colors.colormap_names()[-1] == "sunset"
    saved = json.loads(schemes.read_text())
    assert saved["colormaps"]["sunset"]["stops"] == ["#1a0a2e", "#c2185b", "#ffb347"]
    colors.reload()
    assert colors.is_user_colormap("sunset")
    # matplotlib, the tone-map lookup and the video colourizer all resolve it by name
    assert "sunset" in matplotlib.colormaps
    assert tonemap.colormap("sunset").name == "sunset"
    rgb = video.apply_colormap(np.array([[0.0, 1.0]]), "sunset")
    assert tuple(rgb[0, 0]) == (0x1a, 0x0a, 0x2e) and tuple(rgb[0, 1]) == (0xff, 0xb3, 0x47)
    assert colors.remove_colormap("sunset")
    assert "sunset" not in matplotlib.colormaps and "sunset" not in colors.colormap_names()


def test_names_that_shadow_builtins_are_refused(schemes):
    for bad in ("viridis", "ice", "RdBu"):
        with pytest.raises(ValueError):
            colors.add_colormap(bad, ["#000000", "#ffffff"])
    with pytest.raises(ValueError):
        colors.add_palette("fire-ice", "#ffffff", "#000000", "#000000")
    with pytest.raises(ValueError):
        colors.add_colormap("one stop", ["#000000"])
    with pytest.raises(ValueError):
        colors.add_colormap("bad hex", ["#000000", "not-a-colour"])
    with pytest.raises(ValueError):
        colors.add_colormap("", ["#000000", "#ffffff"])


@pytest.mark.parametrize("name", ["inferno", "viridis", "gray", "turbo", "coolwarm", "ice",
                                  "ember", "gottlux_events", "user"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_apply_u8_is_bit_exact_with_matplotlib(schemes, name, dtype):
    import matplotlib
    if name == "user":
        colors.add_colormap("user", ["#000000", "#ff0000", "#ffff00", "#ffffff"],
                            positions=[0.0, 0.2, 0.7, 1.0])
    v = np.random.default_rng(4).random((64, 80)).astype(dtype)
    v[0, :6] = [0.0, 1.0, 1 / 256, 255 / 256, 0.5, 0.999999]
    ref = (matplotlib.colormaps[name](np.clip(v, 0, 1))[..., :3] * 255).astype(np.uint8)
    got = colors.apply_u8(v, name)
    assert got.dtype == np.uint8 and got.flags.c_contiguous
    assert np.array_equal(got, ref)


def test_palettes_and_studio_shots_use_user_palettes(schemes):
    from gottlux.studio.shot import Shot
    colors.add_palette("neon", "#39ff14", "#ff2bd6", "#05050a")
    assert "neon" in colors.palette_names()
    assert Shot(palette="neon").colors() == {"on": "#39ff14", "off": "#ff2bd6", "bg": "#05050a"}
    assert colors.remove_palette("neon") and not colors.has_palette("neon")


def test_unreadable_file_is_kept_aside_and_builtins_survive(schemes):
    schemes.write_text("{not json")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        colors.reload()
    assert any("could not read the colour schemes" in str(x.message) for x in w)
    assert colors.colormap_names()[0] == "inferno"
    aside = [f for f in os.listdir(schemes.parent) if ".unreadable-" in f]
    assert len(aside) == 1


def test_import_export_round_trip_renames_clashes(schemes, tmp_path):
    colors.add_colormap("mine", ["#102030", "#a0b0c0"])
    colors.add_palette("my pal", "#ff0000", "#00ff00", "#000000")
    out = tmp_path / "share.json"
    colors.export_schemes(str(out), ["mine", "viridis"], ["my pal"])
    data = json.loads(out.read_text())
    assert set(data["colormaps"]) == {"mine", "viridis"} and "my pal" in data["palettes"]
    colors.remove_colormap("mine"); colors.remove_palette("my pal")
    added = colors.import_schemes(str(out))
    # a built-in name in a shared file comes in renamed, never replacing the built-in
    assert "mine" in added and "viridis (imported)" in added and "my pal" in added
    assert colors.is_user_colormap("viridis (imported)")


def test_listeners_are_told_and_dropped_with_their_owner(schemes):
    calls = []

    class Owner:
        def ping(self):
            calls.append(1)

    o = Owner()
    colors.subscribe(o.ping)
    colors.add_colormap("ping", ["#000000", "#ffffff"])
    assert calls
    del o                                      # held weakly: no call, no error after this
    n = len(calls)
    colors.remove_colormap("ping")
    assert len(calls) == n


def test_colormap_params_accept_registry_names_and_web_spec_stays_a_choice(schemes):
    from gottlux.studio.looks import get_look, look_specs
    colors.add_colormap("deep sea", ["#000814", "#003566", "#ffc300"])
    heat = get_look("heat")
    assert heat.resolve({"colormap": "deep sea"})["colormap"] == "deep sea"
    assert heat.resolve({"colormap": "no such map"})["colormap"] == "inferno"
    spec = look_specs()["heat"]["params"]
    cm = next(p for p in spec if p["key"] == "colormap")
    assert cm["kind"] == "choice" and "deep sea" not in cm["choices"]


def test_scheme_editor_saves_a_colormap_every_picker_offers(schemes):
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from gottlux.app.colorschemes import ColormapCombo, ColorSchemesDialog, PaletteCombo
    combo, pals = ColormapCombo(), PaletteCombo()
    dlg = ColorSchemesDialog()
    assert len(dlg._stops) >= 2                       # a built-in loads as editable stops
    dlg.map_name.setText("my ramp")
    dlg._set_stops(["#000000", "#336699", "#ffffff"])
    dlg._reverse()
    dlg._save_map()
    assert colors.colormap_stops("my ramp") == ["#ffffff", "#336699", "#000000"]
    assert dlg.last_created("colormap") == "my ramp" and "my ramp" in combo.names()
    dlg.pal_name.setText("my pal")
    dlg._save_pal()
    assert "my pal" in pals.names()
    row = colors.colormap_names().index("my ramp")
    dlg.map_list.setCurrentRow(row)
    dlg._delete_map()
    assert "my ramp" not in combo.names()
    dlg.close()
