"""
gottlux.detectors — the tunable flutter/flicker detection framework.

A detector finds **where, when, and at what frequency** something is fluttering. The design
goal is that you can *tune and build* detectors, not just run a fixed one:

* :class:`~gottlux.detectors.base.Detector`   — base class; subclass + ``@register``
* :class:`~gottlux.detectors.base.Param`      — self-describing tunable knob (drives the GUI)
* :class:`~gottlux.detectors.base.Target` / :class:`~gottlux.detectors.base.DetectorResult`
* :class:`~gottlux.detectors.flutter.FlutterDetector` — the composable workhorse
* :mod:`~gottlux.detectors.signatures`        — drone/insect/bird/… frequency presets
* :func:`~gottlux.detectors.base.get_detector` / :func:`~gottlux.detectors.base.list_detectors`

Registered presets: ``drone``, ``insect``, ``mosquito``, ``hummingbird``, ``bird``, ``flutter``.

>>> from gottlux.detectors import get_detector
>>> det = get_detector("drone", snr_thresh=5)      # tune any Param at construction
>>> result = det.run(rec, cfg)
>>> print(result.summary())
"""
from __future__ import annotations

from gottlux.detectors.base import (Detector, DetectorResult, Param, Target,
                                    get_detector, list_detectors, register)
from gottlux.detectors.signatures import Signature, get_signature, list_signatures

# Import the built-ins so their @register decorators populate the registry on package import.
from gottlux.detectors import flutter as _flutter   # noqa: F401,E402

# Reconcile the two extension frameworks: register the ported EBS trackers
# as Detectors in this one registry, so the unified --list_detectors / GUI picker shows both
# the flutter detectors and the EBS tracker suite. A broken plugin must not break import.
try:
    from gottlux.detectors.ported import register_ported_trackers
    register_ported_trackers()
except Exception as _e:   # pragma: no cover
    import warnings
    warnings.warn(f"ported EBS trackers unavailable: {_e}")

__all__ = ["Detector", "DetectorResult", "Param", "Target", "register",
           "get_detector", "list_detectors", "Signature", "get_signature",
           "list_signatures"]
