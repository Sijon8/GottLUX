"""
trackers/base.py  --  Tracker plugin interface.

A Tracker consumes the per-detection trajectory produced by the detector and
returns one or more linked TRACKS. This is the extension point for the operator's
ported MATLAB tracking algorithms: subclass Tracker, implement track(), and
decorate with @register (see builtin.py / TEMPLATE.py).
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class Tracker(ABC):
    name = "base"
    description = ""

    #: which capture regime this tracker is meant for: "rotation", "staring", or "both".
    #: Used to filter the tracker list by the Rotating/Staring toggle in the GUI/CLI.
    regime = "both"

    #: tunable parameters with defaults (surfaced in the GUI later); override per tracker
    params: dict = {}

    #: True if this tracker consumes the RAW event stream (its own detection),
    #: rather than the detector's per-detection trajectory.
    uses_events = False

    @abstractmethod
    def track(self, traj: dict, cfg, tel=None, ev=None) -> dict:
        """Link/track targets.

        Parameters
        ----------
        traj : dict of equal-length arrays from detect.build_trajectory:
               t, azimuth_deg, elev_deg, range_m, altitude_z_m, cx, cy, dx, dy, area, n_events
        cfg  : the run Config
        tel  : Telemetry (rotation mode) or None
        ev   : raw event dict (x, y, p, t[us], width, height) when uses_events; else may be None

        Returns
        -------
        dict with key 'tracks' -> list of per-track dicts, each with arrays:
            id, t, azimuth_deg, elev_deg, range_m   (extra keys allowed, e.g. freq_hz, cx, cy)
        """
        raise NotImplementedError
