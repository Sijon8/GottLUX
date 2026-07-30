"""
base.py — the detector framework: self-describing, tunable, composable, registered.

A *detector* answers one question: **where, when, and at what frequency is something
fluttering/flickering like a target?** Everything here is built so you can *tune and build*
detectors, not just run a fixed one:

* :class:`Param` — a self-describing tunable knob (range, step, unit, help, group). A
  detector publishes a list of these, and the GUI auto-builds a live slider panel from
  them; changing one and re-running gives instant visual feedback.
* :class:`Detector` — the base class. Subclass it, declare ``PARAMS`` and a ``regime``,
  implement :meth:`run`, and ``@register`` it — it then appears automatically in the CLI
  (``--detector``), the ``--list_detectors`` output, and the GUI's detector picker.
* :class:`Target` — a tracked target over time, carrying its kinematics *and* its measured
  flutter signature (per-detection frequency, SNR, harmonic-comb score).
* :class:`DetectorResult` — the targets plus the exact params and diagnostics behind them.

The built-in :class:`~gottlux.detectors.flutter.FlutterDetector` (registered as ``drone``,
``insect``, ``bird``) composes the core stages — foreground → cluster → FFT flutter-verify →
track — and is fully driven by these Params + a :class:`~gottlux.detectors.signatures.Signature`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ====================================================================================
# Tunable parameter descriptor (drives both code defaults and the GUI tuning panel)
# ====================================================================================
@dataclass
class Param:
    """A single self-describing, tunable detector parameter."""
    key: str
    label: str
    default: float
    lo: float = 0.0
    hi: float = 1.0
    step: float = 0.0                 # 0 → a sensible (range/100) default in the GUI
    kind: str = "float"               # 'float' | 'int' | 'bool' | 'choice'
    choices: tuple = ()
    unit: str = ""
    help: str = ""
    group: str = "General"

    def coerce(self, v):
        """Coerce a raw value to this parameter's type and clamp to range."""
        if self.kind == "bool":
            return bool(v)
        if self.kind == "choice":
            return v if v in self.choices else self.default
        if self.kind == "int":
            return int(np.clip(int(round(float(v))), self.lo, self.hi))
        return float(np.clip(float(v), self.lo, self.hi))


# ====================================================================================
# Result model
# ====================================================================================
@dataclass
class Target:
    """A tracked target: kinematics over time plus its measured flutter signature."""
    id: int
    t: np.ndarray                      # detection times (s)
    cx: np.ndarray                     # image-plane centroid x (px)
    cy: np.ndarray                     # image-plane centroid y (px)
    bbox: np.ndarray                   # (N, 4) [x0, y0, x1, y1]
    freq_hz: np.ndarray                # verified flutter frequency per detection (Hz)
    snr: np.ndarray                    # spectral peak/noise per detection
    harmonic: np.ndarray               # harmonic-comb score per detection (0..1)
    azimuth_deg: Optional[np.ndarray] = None
    elev_deg: Optional[np.ndarray] = None
    range_m: Optional[np.ndarray] = None
    rel_distance: Optional[np.ndarray] = None
    # EBS regime-split extras: the staring report carries radial velocity +
    # blade-flutter frequency, the rotation report carries bearing/elev/range — both expressible.
    radial_velocity: Optional[np.ndarray] = None     # d(rel_distance)/dt proxy (proxy/s)
    blade_hz: Optional[np.ndarray] = None            # blade/rotor flutter frequency (Hz)

    def __len__(self):
        return int(np.size(self.t))

    @property
    def n(self) -> int:
        return len(self)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.n > 1 else 0.0

    @property
    def median_freq(self) -> float:
        f = self.freq_hz[np.isfinite(self.freq_hz)]
        return float(np.median(f)) if f.size else float("nan")

    @property
    def freq_stability(self) -> float:
        """1 − normalized frequency scatter (1 = rock-steady tone, 0 = all over)."""
        f = self.freq_hz[np.isfinite(self.freq_hz)]
        if f.size < 2 or np.median(f) <= 0:
            return 0.0
        return float(max(0.0, 1.0 - np.std(f) / np.median(f)))

    @property
    def confidence(self) -> float:
        """A 0..1 confidence: blends track persistence, mean SNR, frequency stability and
        harmonic support — a single number to rank/threshold targets by."""
        if self.n == 0:
            return 0.0
        persistence = min(self.n / 8.0, 1.0)
        snr_term = float(np.tanh(np.nanmean(self.snr) / 8.0))
        harm = float(np.nanmean(self.harmonic)) if np.isfinite(self.harmonic).any() else 0.0
        return float(np.clip(0.35 * persistence + 0.35 * snr_term +
                             0.15 * self.freq_stability + 0.15 * harm, 0, 1))


@dataclass
class DetectorResult:
    """The output of a detector run: targets + full provenance of how they were found."""
    targets: list                      # list[Target]
    detector: str
    params: dict
    regime: str
    signature: str = ""
    diagnostics: dict = field(default_factory=dict)

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    def confident(self, thresh: float = 0.5) -> list:
        """Targets whose confidence ≥ *thresh*, sorted strongest first."""
        return sorted([t for t in self.targets if t.confidence >= thresh],
                      key=lambda t: -t.confidence)

    def summary(self) -> str:
        lines = [f"Detector '{self.detector}' ({self.regime}) - {self.n_targets} target(s)"]
        for t in sorted(self.targets, key=lambda t: -t.confidence):
            lines.append(f"  #{t.id}: {t.n} dets, {t.duration_s:.2f}s, "
                         f"{t.median_freq:.0f} Hz, conf {t.confidence:.2f}")
        return "\n".join(lines)


# ====================================================================================
# Detector base + registry
# ====================================================================================
class Detector:
    """Base class for all detectors. Subclass, declare ``PARAMS``/``regime``, implement
    :meth:`run`, and decorate with :func:`register`."""

    name: str = "base"
    description: str = ""
    regime: str = "both"               # 'staring' | 'rotation' | 'both'
    use_for: str = ""                  # one-line "use it for…" shown in the GUI
    PARAMS: list[Param] = []

    def __init__(self, **overrides):
        self.params = {p.key: p.default for p in self.PARAMS}
        for k, v in overrides.items():
            if k in self.params:
                spec = next(p for p in self.PARAMS if p.key == k)
                self.params[k] = spec.coerce(v)
            else:
                self.params[k] = v

    def set(self, **kw) -> "Detector":
        """Update parameters (coerced/clamped to their declared ranges). Chainable."""
        for k, v in kw.items():
            if k in self.params:
                spec = next((p for p in self.PARAMS if p.key == k), None)
                self.params[k] = spec.coerce(v) if spec else v
        return self

    def run(self, rec, cfg=None, t0=None, t1=None, progress=None) -> DetectorResult:
        raise NotImplementedError

    @classmethod
    def param_specs(cls) -> list:
        return list(cls.PARAMS)


_REGISTRY: dict[str, type] = {}


def register(cls):
    """Class decorator: add a Detector subclass to the global registry by its ``name``."""
    _REGISTRY[cls.name] = cls
    return cls


def get_detector(name: str, **overrides) -> Detector:
    """Instantiate a registered detector by name (with optional param overrides)."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**overrides)


def list_detectors() -> dict:
    """Mapping of registered detector name → class."""
    return dict(_REGISTRY)


def detections_table(result) -> dict:
    """Flatten a :class:`DetectorResult`'s tracks to one per-detection table (dict of arrays).

    The single source of the detections table — used by the headless pipeline, the GUI export
    bundle and the KPI extractor (previously duplicated in all three).
    """
    keys = ("target_id", "t_s", "cx", "cy", "freq_hz", "snr", "harmonic", "azimuth_deg",
            "elev_deg", "range_m", "rel_distance", "apparent_px", "confidence")
    cols = {k: [] for k in keys}
    for t in (result.targets if result else []):
        bbox = getattr(t, "bbox", None)
        diag = (np.hypot(t.bbox[:, 2] - t.bbox[:, 0], t.bbox[:, 3] - t.bbox[:, 1])
                if bbox is not None and len(bbox) else np.full(t.n, np.nan))

        def _col(arr, i):
            return float(arr[i]) if arr is not None else np.nan
        for i in range(t.n):
            cols["target_id"].append(int(t.id)); cols["t_s"].append(float(t.t[i]))
            cols["cx"].append(float(t.cx[i])); cols["cy"].append(float(t.cy[i]))
            cols["freq_hz"].append(float(t.freq_hz[i])); cols["snr"].append(float(t.snr[i]))
            cols["harmonic"].append(float(t.harmonic[i]))
            cols["azimuth_deg"].append(_col(t.azimuth_deg, i))
            cols["elev_deg"].append(_col(t.elev_deg, i))
            cols["range_m"].append(_col(t.range_m, i))
            cols["rel_distance"].append(_col(t.rel_distance, i))
            cols["apparent_px"].append(float(diag[i]) if i < len(diag) else np.nan)
            cols["confidence"].append(float(t.confidence))
    return {k: (np.asarray(v, int) if k == "target_id" else np.asarray(v, float))
            for k, v in cols.items()}
