"""
gottlux.rotation.trackers  --  pluggable tracking-algorithm registry.

Usage
-----
    from gottlux.rotation import trackers
    trackers.available()         # -> ['nearest', 'single', ...]
    T = trackers.get('nearest')()
    result = T.track(traj, cfg, tel)

Add your own (e.g. ported MATLAB algorithms): create a module here, subclass
Tracker, and decorate the class with @trackers.register. Drop it next to
builtin.py and import it below (or it auto-imports if added to _MODULES).
"""
from __future__ import annotations
import importlib
from gottlux.rotation.trackers.base import Tracker

_REGISTRY = {}


def register(cls):
    """Class decorator: register a Tracker subclass by its `name`."""
    _REGISTRY[cls.name] = cls
    return cls


def get(name):
    return _REGISTRY.get(name)


def available():
    return sorted(_REGISTRY)


# import modules that define + register trackers
_MODULES = ["builtin", "frequency", "kalman", "cmax", "staring_kvf"]   # add your module names here (e.g. "matlab_ported")
for _m in _MODULES:
    try:
        importlib.import_module(f"gottlux.rotation.trackers.{_m}")
    except Exception as _e:       # a broken plugin shouldn't crash the platform
        import warnings
        warnings.warn(f"tracker module '{_m}' failed to import: {_e}")
