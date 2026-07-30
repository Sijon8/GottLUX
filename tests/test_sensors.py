"""
Tests for the sensor/camera profile registry (gottlux.sensors) and its wiring into Config,
the CLI, the photogrammetry presets and the resolution report.

The GenX320 datasheet is the single source of truth for the lab's default rig; these tests
pin its exact values, confirm the optics are internally consistent (focal length + pixel pitch
reproduce the quoted fields of view), and exercise the override paths a clip on different
hardware would use.
"""
import math

import pytest

from gottlux import sensors as S
from gottlux.config import Config, CAMERA_FOV_DEG


# ------------------------------------------------------------------ datasheet (verbatim)
def test_genx320_datasheet_values():
    g = S.GENX320
    assert g.key == "genx320" and g.name == "Prophesee GenX320" and g.vendor == "Prophesee"
    # sensor
    assert (g.width_px, g.height_px) == (320, 320)
    assert g.pixel_pitch_um == 6.3
    assert g.event_rate == "~10,000 fps equivalent"
    assert g.latency == "<150 µs at 1,000 lux, <1,000 µs at 5 lux"
    assert g.dynamic_range == ">140 dB" and g.dynamic_range_db == 140.0
    assert g.power == "<50 mW (sensor only)" and g.power_mw == 50.0
    assert g.interface == "MIPI CSI-2 (D-PHY)"
    # camera / optics
    assert g.lens_mount == "Interchangeable M12 S-Mount lens holder"
    assert g.focal_length_mm == 1.8
    assert (g.fov_diagonal_deg, g.fov_horizontal_deg, g.fov_vertical_deg) == (76.0, 58.0, 58.0)
    assert g.f_number == 2.8 and g.iris == "fixed"
    assert g.ir_cut_filter is False


def test_datasheet_rows_match_input_strings():
    """The report/table view reproduces the exact datasheet wording the user supplied."""
    rows = {label: value for section in S.GENX320.datasheet().values() for label, value in section}
    assert rows["Resolution"] == "320 × 320 px"
    assert rows["Event rate"] == "~10,000 fps equivalent"
    assert rows["Latency"] == "<150 µs at 1,000 lux, <1,000 µs at 5 lux"
    assert rows["Dynamic range"] == ">140 dB"
    assert rows["Power consumption"] == "<50 mW (sensor only)"
    assert rows["Interface"] == "MIPI CSI-2 (D-PHY)"
    assert rows["Lens mount"] == "Interchangeable M12 S-Mount lens holder"
    assert rows["Focal length"] == "1.8 mm"
    assert rows["Diagonal field of view"] == "76°"
    assert rows["Horizontal field of view"] == "58°"
    assert rows["Vertical field of view"] == "58°"
    assert rows["Focal ratio (f-stop)"] == "f/2.8 Fixed Iris"
    assert rows["IR-cut filter"] == "No"


# ------------------------------------------------------------------ optics consistency
def test_genx320_fov_is_internally_consistent():
    """320×320 @ 6.3 µm behind a 1.8 mm lens really does subtend the quoted FOVs (<1°)."""
    g = S.GENX320
    assert g.computed_fov_deg("horizontal") == pytest.approx(58.0, abs=1.0)
    assert g.computed_fov_deg("vertical") == pytest.approx(58.0, abs=1.0)
    assert g.computed_fov_deg("diagonal") == pytest.approx(76.0, abs=1.0)
    # 320 px * 6.3 µm = 2.016 mm array; diagonal = √2 * that ≈ 2.85 mm
    assert g.diagonal_px == pytest.approx(math.hypot(320, 320))
    assert g.focal_px() == pytest.approx(1.8 / (6.3e-3))


# ------------------------------------------------------------------ registry helpers
def test_registry_default_and_forgiving_lookup():
    assert S.DEFAULT_PROFILE == "genx320"
    assert S.get(None).key == "genx320"
    assert S.get("GenX320").key == "genx320"          # case-insensitive
    assert S.get("  imx636 ").key == "imx636"          # trimmed
    assert S.get("does-not-exist").key == "genx320"    # unknown → default (old manifests load)


def test_with_lens_recomputes_all_fovs():
    """Same sensor, different optic: every FOV follows the new focal length."""
    g6 = S.GENX320.with_lens(6.0)
    assert g6.focal_length_mm == 6.0
    assert g6.fov_horizontal_deg < S.GENX320.fov_horizontal_deg   # longer lens ⇒ narrower
    assert g6.fov_horizontal_deg == pytest.approx(
        S.fov_deg_from_optics(6.0, 6.3, 320), abs=1e-3)
    # pixel pitch unchanged, so it stays self-consistent
    assert g6.computed_fov_deg("horizontal") == pytest.approx(g6.fov_horizontal_deg, abs=1e-2)


def test_with_resolution_for_a_different_sensor():
    custom = S.GENX320.with_resolution(640, 480, pixel_pitch_um=9.0, key="vga", name="VGA rig")
    assert (custom.width_px, custom.height_px, custom.pixel_pitch_um) == (640, 480, 9.0)
    # FOVs recomputed for the new geometry + the same lens
    assert custom.fov_horizontal_deg == pytest.approx(S.fov_deg_from_optics(1.8, 9.0, 640), abs=1e-3)


def test_register_and_select_custom_profile():
    rig = S.GENX320.with_lens(12.0, key="genx320_tele", name="GenX320 + 12 mm")
    try:
        S.register(rig)
        assert "genx320_tele" in S.list_profiles()
        cfg = Config(sensor="genx320_tele")
        assert cfg.active_profile().name == "GenX320 + 12 mm"
        assert cfg.resolved_fov() == pytest.approx(rig.fov_horizontal_deg)
    finally:
        S.PROFILES.pop("genx320_tele", None)


# ------------------------------------------------------------------ Config resolution
def test_config_defaults_to_genx320():
    c = Config()
    assert c.sensor == "genx320"
    assert c.resolved_fov() == 58.0                    # horizontal, not the 76° diagonal
    assert c.resolved_sensor_wh() == (320, 320)
    assert c.resolved_pixel_pitch_um() == 6.3
    o = c.optics()
    assert o["sensor"] == "genx320" and o["fov_horizontal_deg"] == 58.0
    assert o["width_px"] == 320 and o["focal_length_mm"] == 1.8


def test_explicit_overrides_win_over_profile():
    c = Config(fov_deg=40.0, sensor_w=640, sensor_h=480)
    assert c.resolved_fov() == 40.0
    assert c.resolved_sensor_wh() == (640, 480)


def test_camera_override_only_for_non_default_optic():
    # cam0 is the default rig → resolves from the profile, NOT a hardcoded value
    assert "cam0" not in CAMERA_FOV_DEG
    assert Config(camera="cam0").resolved_fov() == 58.0
    # cam1 is the narrower second optic → an explicit override
    assert Config(camera="cam1").resolved_fov() == 50.0


def test_selecting_a_different_sensor_uses_its_own_fov():
    """Regression: a non-default sensor on the default camera must not inherit cam0's FOV."""
    c = Config(sensor="imx636")              # camera defaults to "cam0"
    assert c.resolved_sensor_wh() == (1280, 720)
    assert c.resolved_fov() == pytest.approx(S.IMX636.fov_horizontal_deg)
    assert c.resolved_fov() != 58.0


# ------------------------------------------------------------------ serialization
def test_profile_round_trip():
    d = S.GENX320.to_dict()
    assert S.SensorProfile.from_dict(d) == S.GENX320
    # unknown keys are ignored (forward/backward compatible)
    d2 = dict(d, some_future_field=123)
    assert S.SensorProfile.from_dict(d2) == S.GENX320


def test_config_serialization_carries_sensor():
    d = Config(sensor="imx636").to_dict()
    assert d["sensor"] == "imx636"
    assert Config.from_dict(d).sensor == "imx636"
    # a legacy manifest with no sensor key still loads (defaults applied)
    assert Config.from_dict({"camera": "cam0"}).sensor == "genx320"


# ------------------------------------------------------------------ downstream wiring
def test_photogrammetry_presets_derive_from_registry():
    from gottlux.core import photogrammetry as pg
    assert "Prophesee GenX320" in pg.SENSORS
    assert pg.SENSORS["Prophesee GenX320"]["pixel_pitch_um"] == 6.3
    assert pg.LENSES[next(iter(pg.LENSES))] == S.GENX320.focal_length_mm   # 1.8 mm preset


def test_resolution_report_includes_datasheet(tmp_path):
    from gottlux.core import photogrammetry as pg
    from gottlux.run.resolution_report import save_resolution_study
    study = pg.ResolutionStudy(0.22, 58.0, 320, 320,
                               [pg.Keyframe(1.0, (40, 40, 60, 60), distance_m=10.0)])
    written = save_resolution_study(str(tmp_path / "rig"), study, profile=S.GENX320)
    md = next(p for p in written if p.endswith("_resolution_report.md"))
    text = open(md, encoding="utf-8").read()
    assert "Capture rig (sensor + camera)" in text
    assert "Prophesee GenX320" in text
    assert ">140 dB" in text and "MIPI CSI-2 (D-PHY)" in text


def test_cli_sensor_flag_and_listing(capsys):
    from gottlux.cli import build_parser, _config_from_args, main
    a = build_parser().parse_args(["file.raw", "--sensor", "imx636", "--fov_deg", "40"])
    cfg = _config_from_args(a)
    assert cfg.sensor == "imx636" and cfg.resolved_fov() == 40.0
    assert main(["gottlux", "--list_sensors"]) == 0
    out = capsys.readouterr().out
    assert "genx320" in out and "[default]" in out and "imx636" in out
