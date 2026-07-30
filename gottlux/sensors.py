"""
sensors.py — the hardware datasheet registry (sensor + camera/optics).

Where :mod:`gottlux.config` is the single source of truth for a *run's parameters*, this
module is the single source of truth for the *hardware* a clip was captured with: the
event sensor's datasheet (resolution, event rate, latency, dynamic range, power, interface)
and the camera/optics in front of it (lens mount, focal length, fields of view, f-number,
IR-cut). The numeric optical fields (resolution, pixel pitch, focal length, FOV) feed the
ranging/photogrammetry math; the descriptive fields are carried verbatim into run manifests
and reports for provenance.

One source, many clips
----------------------
Most captures use the lab's default rig — the **Prophesee GenX320** behind a **1.8 mm M12
S-mount** lens (:data:`GENX320`, the :data:`DEFAULT_PROFILE`). But some clips use a different
sensor or different optics, so a profile is a *variable you can change*:

* pick a built-in by name — ``Config(sensor="imx636")``, ``gottlux file.raw --sensor imx636``;
* swap only the lens on the same sensor — ``GENX320.with_lens(6.0)`` (recomputes every FOV
  from the pixel pitch), or per run ``Config(sensor="genx320", fov_deg=...)``;
* define a wholly custom rig — :func:`register` a :class:`SensorProfile` (or derive one with
  :meth:`SensorProfile.replace`) and refer to it by its ``key`` everywhere.

The optics here are internally consistent: a GenX320 (320×320 @ 6.3 µm pitch ⇒ 2.0 mm ×
2.0 mm array, 2.85 mm diagonal) behind a 1.8 mm lens subtends ≈58° horizontally, ≈58°
vertically and ≈76° diagonally — exactly the datasheet numbers below. The **horizontal** FOV
is the one the bearing/range math uses (it spans the sensor *width*); the 76° often quoted for
this rig is the *diagonal*.

This module is a dependency-light leaf (stdlib only) so :mod:`gottlux.config` can import it
without pulling in NumPy/Qt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace, asdict, fields
from typing import Dict, Optional


# --------------------------------------------------------------------- optics primitives
# (inlined here — a few lines of trig — so this module stays a stdlib-only leaf. The same
#  formulas live in gottlux.core.photogrammetry for the NumPy/array-valued solver paths.)
def fov_deg_from_optics(focal_length_mm: float, pixel_pitch_um: float, n_px: float) -> float:
    """Field of view (deg) along an axis of *n_px* pixels for a lens + pixel pitch."""
    half_mm = float(n_px) * float(pixel_pitch_um) * 1e-3 / 2.0
    return float(math.degrees(2.0 * math.atan2(half_mm, float(focal_length_mm))))


def focal_px_from_pitch(focal_length_mm: float, pixel_pitch_um: float) -> float:
    """Focal length in **pixels** = focal length / pixel pitch — the most fundamental scale."""
    return float(focal_length_mm) / (float(pixel_pitch_um) * 1e-3)


# --------------------------------------------------------------------- the profile
@dataclass(frozen=True)
class SensorProfile:
    """One capture rig: an event sensor + the camera/optics in front of it.

    Frozen (immutable) so a profile in the registry can't be mutated out from under a run;
    derive a variant with :meth:`replace` / :meth:`with_lens` / :meth:`with_resolution`.
    Every field has a default so a custom rig can be built by overriding just what differs.
    """

    # -------------------------------- identity --------------------------------
    key: str = "custom"
    """Short lookup id used everywhere a profile is selected (``Config.sensor``, ``--sensor``)."""
    name: str = "Custom sensor"
    """Human-readable name woven into reports/manifests."""
    vendor: str = ""

    # ------------------------- sensor (the event array) ------------------------
    width_px: int = 320
    """Active-array width in pixels (spans the **horizontal** FOV)."""
    height_px: int = 320
    """Active-array height in pixels (spans the vertical FOV)."""
    pixel_pitch_um: float = 6.3
    """Physical pixel pitch (µm) — what ties pixels to a metric angle through the lens."""

    event_rate: str = ""
    """Datasheet event throughput, verbatim (e.g. ``"~10,000 fps equivalent"``)."""
    latency: str = ""
    """Datasheet latency, verbatim (e.g. ``"<150 µs at 1,000 lux, <1,000 µs at 5 lux"``)."""
    dynamic_range: str = ""
    """Datasheet dynamic range, verbatim (e.g. ``">140 dB"``)."""
    dynamic_range_db: Optional[float] = None
    """Parsed dynamic-range magnitude (dB) for plots/tables; ``None`` if not given."""
    power: str = ""
    """Datasheet power consumption, verbatim (e.g. ``"<50 mW (sensor only)"``)."""
    power_mw: Optional[float] = None
    """Parsed power magnitude (mW); ``None`` if not given."""
    interface: str = ""
    """Sensor data interface (e.g. ``"MIPI CSI-2 (D-PHY)"``)."""

    # --------------------------- camera (the optics) ---------------------------
    lens_mount: str = ""
    """Lens holder / mount (e.g. ``"Interchangeable M12 S-Mount lens holder"``)."""
    focal_length_mm: float = 1.8
    """Lens focal length (mm). With :attr:`pixel_pitch_um` this sets the fields of view."""
    fov_diagonal_deg: float = 0.0
    """Diagonal field of view (deg) — across the array diagonal."""
    fov_horizontal_deg: float = 0.0
    """Horizontal field of view (deg) — across the sensor *width*; drives bearing/range math."""
    fov_vertical_deg: float = 0.0
    """Vertical field of view (deg) — across the sensor height."""
    f_number: Optional[float] = None
    """Aperture f-number (e.g. ``2.8`` for f/2.8); ``None`` if unknown."""
    iris: str = ""
    """Iris type (e.g. ``"fixed"``)."""
    ir_cut_filter: bool = False
    """Whether an IR-cut filter is fitted (event sensors usually run without one)."""

    note: str = ""
    """Free-text provenance note."""

    # --------------------------------------------------------------- derived
    @property
    def diagonal_px(self) -> float:
        """The array diagonal in pixels."""
        return float(math.hypot(self.width_px, self.height_px))

    def focal_px(self) -> float:
        """Focal length in pixels (focal length ÷ pixel pitch)."""
        return focal_px_from_pitch(self.focal_length_mm, self.pixel_pitch_um)

    def computed_fov_deg(self, axis: str = "horizontal") -> float:
        """The FOV (deg) implied by the optics (focal length + pitch) along *axis*.

        Use this to cross-check the stored datasheet FOV, or as the value after a lens swap.
        ``axis`` ∈ {``"horizontal"``, ``"vertical"``, ``"diagonal"``}.
        """
        n = {"vertical": self.height_px, "diagonal": self.diagonal_px}.get(axis, self.width_px)
        return fov_deg_from_optics(self.focal_length_mm, self.pixel_pitch_um, n)

    # --------------------------------------------------------------- derivation
    def replace(self, **overrides) -> "SensorProfile":
        """A copy of this profile with fields overridden (the 'change it as a variable' path)."""
        return replace(self, **overrides)

    def with_lens(self, focal_length_mm: float, key: Optional[str] = None,
                  name: Optional[str] = None, f_number: Optional[float] = None) -> "SensorProfile":
        """Same sensor, different lens: recompute all three FOVs from the new focal length."""
        f = float(focal_length_mm)
        return replace(
            self,
            key=key or f"{self.key}+{f:g}mm",
            name=name or f"{self.name.split(' + ')[0]} + {f:g} mm lens",
            focal_length_mm=f,
            f_number=(self.f_number if f_number is None else f_number),
            fov_horizontal_deg=round(fov_deg_from_optics(f, self.pixel_pitch_um, self.width_px), 3),
            fov_vertical_deg=round(fov_deg_from_optics(f, self.pixel_pitch_um, self.height_px), 3),
            fov_diagonal_deg=round(fov_deg_from_optics(f, self.pixel_pitch_um, self.diagonal_px), 3),
        )

    def with_resolution(self, width_px: int, height_px: int,
                        pixel_pitch_um: Optional[float] = None,
                        key: Optional[str] = None, name: Optional[str] = None) -> "SensorProfile":
        """Different sensor geometry: set resolution (and optionally pitch), recompute FOVs."""
        pitch = self.pixel_pitch_um if pixel_pitch_um is None else float(pixel_pitch_um)
        base = replace(self, width_px=int(width_px), height_px=int(height_px), pixel_pitch_um=pitch,
                       key=key or self.key, name=name or self.name)
        return base.with_lens(base.focal_length_mm, key=base.key, name=base.name)

    # --------------------------------------------------------------- (de)serialization
    def to_dict(self) -> dict:
        """JSON-serializable snapshot (archived in the run manifest for provenance)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SensorProfile":
        """Reconstruct from a dict, ignoring unknown keys (forward/backward compatible)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def datasheet(self) -> "Dict[str, list]":
        """Ordered ``section -> [(label, value)]`` rows for a report/table (datasheet view)."""
        return {
            "Sensor": [
                ("Sensor", self.name),
                ("Resolution", f"{self.width_px} × {self.height_px} px"),
                ("Pixel pitch", f"{self.pixel_pitch_um:g} µm"),
                ("Event rate", self.event_rate),
                ("Latency", self.latency),
                ("Dynamic range", self.dynamic_range),
                ("Power consumption", self.power),
                ("Interface", self.interface),
            ],
            "Camera": [
                ("Lens mount", self.lens_mount),
                ("Focal length", f"{self.focal_length_mm:g} mm"),
                ("Diagonal field of view", f"{self.fov_diagonal_deg:g}°"),
                ("Horizontal field of view", f"{self.fov_horizontal_deg:g}°"),
                ("Vertical field of view", f"{self.fov_vertical_deg:g}°"),
                ("Focal ratio (f-stop)",
                 (f"f/{self.f_number:g}{(' ' + self.iris.title() + ' Iris') if self.iris else ''}"
                  if self.f_number else "")),
                ("IR-cut filter", "Yes" if self.ir_cut_filter else "No"),
            ],
        }


# --------------------------------------------------------------------- built-in profiles
#: The lab's default rig — Prophesee GenX320 behind a 1.8 mm M12 S-mount lens.
#: FOVs are the datasheet values (58° H / 58° V / 76° diagonal); they match the optics
#: implied by 320×320 @ 6.3 µm and f = 1.8 mm to <1° (cross-checked in tests).
GENX320 = SensorProfile(
    key="genx320",
    name="Prophesee GenX320",
    vendor="Prophesee",
    # sensor
    width_px=320,
    height_px=320,
    pixel_pitch_um=6.3,
    event_rate="~10,000 fps equivalent",
    latency="<150 µs at 1,000 lux, <1,000 µs at 5 lux",
    dynamic_range=">140 dB",
    dynamic_range_db=140.0,
    power="<50 mW (sensor only)",
    power_mw=50.0,
    interface="MIPI CSI-2 (D-PHY)",
    # camera / optics
    lens_mount="Interchangeable M12 S-Mount lens holder",
    focal_length_mm=1.8,
    fov_diagonal_deg=76.0,
    fov_horizontal_deg=58.0,
    fov_vertical_deg=58.0,
    f_number=2.8,
    iris="fixed",
    ir_cut_filter=False,
    note=("Prophesee GenX320 — 1/5\" optical format, 320×320 active pixels @ 6.3 µm pitch "
          "(2.0×2.0 mm array, 2.85 mm diagonal), 1.8 mm M12 S-mount lens. 76° is the "
          "diagonal FOV; the horizontal FOV (used for bearing/range) is 58°."),
)

#: Prophesee IMX636 / Gen4 — a larger HD event sensor (different geometry).
IMX636 = SensorProfile(
    key="imx636",
    name="Prophesee IMX636 (Gen4 HD)",
    vendor="Prophesee / Sony",
    width_px=1280,
    height_px=720,
    pixel_pitch_um=4.86,
    dynamic_range=">120 dB",
    dynamic_range_db=120.0,
    interface="MIPI CSI-2 (D-PHY)",
    lens_mount="Interchangeable M12 S-Mount lens holder",
    focal_length_mm=8.0,
    f_number=2.8,
    iris="fixed",
    ir_cut_filter=False,
    note="Prophesee IMX636 / Gen4 — 1280×720 @ 4.86 µm pitch.",
).with_lens(8.0, key="imx636", name="Prophesee IMX636 (Gen4 HD)")


#: The default profile id — what a run uses when nothing else is specified.
DEFAULT_PROFILE = "genx320"

#: Last-resort horizontal FOV (deg) when a run's FOV is otherwise completely unknown — the
#: default rig's horizontal FOV. Sourced from the profile so the two never drift apart.
DEFAULT_FOV_DEG = GENX320.fov_horizontal_deg

#: The mutable registry of named profiles. Add your own rigs with :func:`register`.
PROFILES: "Dict[str, SensorProfile]" = {
    GENX320.key: GENX320,
    IMX636.key: IMX636,
}


# --------------------------------------------------------------------- registry helpers
def get(key: Optional[str]) -> SensorProfile:
    """Look up a profile by key (case-insensitive). ``None``/unknown → the default rig.

    Lookups are forgiving so a stray ``"GenX320"``/``"genx320"`` both resolve, and an old
    manifest naming a profile we no longer ship still loads (falling back to the default).
    """
    if not key:
        return PROFILES[DEFAULT_PROFILE]
    return PROFILES.get(str(key).strip().lower(), PROFILES[DEFAULT_PROFILE])


def register(profile: SensorProfile) -> SensorProfile:
    """Add (or replace) a profile in the registry, keyed by its ``key``. Returns it."""
    PROFILES[profile.key.strip().lower()] = profile
    return profile


def list_profiles() -> "Dict[str, SensorProfile]":
    """A copy of the registry (id → profile), for ``--list_sensors`` and the GUI picker."""
    return dict(PROFILES)
