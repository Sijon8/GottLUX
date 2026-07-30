"""
gottlux.io — data in and out.

* :func:`~gottlux.io.recording.load`        load a Recording (decode-once, memmapped)
* :class:`~gottlux.io.recording.Recording`  the unified event data model
* :class:`~gottlux.io.recording.EventWindow` a cheap time-window view
* :class:`~gottlux.io.telemetry.Telemetry`  rotation ground truth
* :mod:`~gottlux.io.decode`                 the multi-format ``.raw`` decoder
* :mod:`~gottlux.io.cache`                  the streaming memmap cache
* :mod:`~gottlux.io.hdf5`                   HDF5 event files (write + read, all layouts)
* :mod:`~gottlux.io.writer`                 encode events back to ``.raw`` / cut clips
* :mod:`~gottlux.io.export`                 reproducible data + figure saving
"""
from __future__ import annotations

from gottlux.io.recording import EventWindow, Recording, load
from gottlux.io.telemetry import Telemetry

__all__ = ["load", "Recording", "EventWindow", "Telemetry"]
