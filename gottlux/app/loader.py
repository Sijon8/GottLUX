"""
loader.py — background decoding so the UI never freezes.

Decoding a multi-GB capture can take seconds; doing it on the GUI thread would lock the
window. :class:`RecordingLoader` runs :func:`gottlux.io.load` on a worker thread, forwarding
the decoder's progress fraction to a signal that drives the progress bar, and hands back the
finished :class:`~gottlux.io.recording.Recording` (or an error string) when done.

For a large, uncached ``.raw`` the load is **two-phase**: a sampled
:class:`~gottlux.io.preview.PreviewRecording` (index + beginning/middle/end slices, ~1–2 s)
is emitted on the ``preview`` signal so the window is playable almost immediately, then the
untouched full cache decode continues and ``loaded`` delivers the real memmap-backed
Recording that the GUI swaps in seamlessly. Small files and cache hits skip the preview.

A second worker, :class:`DetectorWorker`, runs a detector off-thread the same way, so tuning
sweeps stay responsive.
"""
from __future__ import annotations

from PySide6 import QtCore

import gottlux as eb


class RecordingLoader(QtCore.QThread):
    progress = QtCore.Signal(float)          # 0..1
    preview = QtCore.Signal(object)          # PreviewRecording (large uncached .raw only)
    loaded = QtCore.Signal(object)           # Recording
    failed = QtCore.Signal(str)

    def __init__(self, path, camera="cam0", mode="auto", parent=None, allow_preview=True):
        super().__init__(parent)
        self.path = path
        self.camera = camera
        self.mode = mode
        self.allow_preview = allow_preview

    def run(self):
        if self.allow_preview:
            try:                              # best-effort: the full load below is authoritative
                from gottlux.io import preview as _preview
                if _preview.should_preview(self.path):
                    self.preview.emit(_preview.preview_recording(self.path))
            except Exception:
                pass
        try:
            rec = eb.load(self.path, camera=self.camera, mode=self.mode,
                          progress=lambda f: self.progress.emit(float(f)))
            self.loaded.emit(rec)
        except Exception as e:                # surface decode errors to the UI, never crash
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class PreviewPrefetcher(QtCore.QThread):
    """Decode an un-covered span of a :class:`~gottlux.io.preview.PreviewRecording` off the
    GUI thread — the seek-into-a-gap path of the sampled preview.

    While the preview is on screen, seeking (or playing) into a region that was never
    decoded triggers a bounded slice decode; running it here keeps the window responsive
    (the views keep rendering the spans they have). ``done`` fires once coverage was
    extended so the caller can re-render at the now-decoded cursor."""

    done = QtCore.Signal()

    def __init__(self, rec, t0, t1, parent=None):
        super().__init__(parent)
        self.rec = rec
        self.t0 = float(t0)
        self.t1 = float(t1)

    def run(self):
        try:
            self.rec.ensure_window(self.t0, self.t1)   # never raises; best-effort by contract
        finally:
            self.done.emit()


class DetectorWorker(QtCore.QThread):
    progress = QtCore.Signal(float)
    done = QtCore.Signal(object)             # DetectorResult
    failed = QtCore.Signal(str)

    def __init__(self, detector, rec, cfg, t0=None, t1=None, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.rec = rec
        self.cfg = cfg
        self.t0 = t0
        self.t1 = t1

    def run(self):
        try:
            res = self.detector.run(self.rec, self.cfg, t0=self.t0, t1=self.t1,
                                    progress=lambda f: self.progress.emit(float(f)))
            self.done.emit(res)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")
