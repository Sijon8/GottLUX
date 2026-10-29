"""
videoexport.py — the export controls and the background job every video export runs through.

:class:`VideoOptions` is the small block of controls the Canvas composer and the Timeline
show under *Export*: **format** (MP4 · WebM · GIF), **frame rate**, **size** (the canvas
at 100 / 75 / 50 / 25 %, or a fixed width) and **quality**. The last choice is remembered.
:func:`ask_path` asks where to save with a filter per format (the chosen filter or the
typed extension decides the format), and :func:`run_job` renders and encodes on a worker
thread behind a progress dialog with **Cancel**, so the window keeps painting and a long
export can be stopped; it returns when the job is done, so callers stay simple.

All of them encode through :func:`gottlux.viz.encode.encode_frames`, which is also what the
Studio uses: one codec setup, one quality scale, one set of facts for the provenance record.
"""
from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from gottlux.app.loader import Worker
from gottlux.viz import encode

#: Formats a rendered-composition export offers (PNG frames stay a Studio/CLI option).
VIDEO_FORMATS = ("mp4", "webm", "gif")

_FILTERS = {
    "mp4": "MP4 video (*.mp4)",
    "webm": "WebM video (*.webm)",
    "gif": "Animated GIF (*.gif)",
}

#: Size presets: ``(label, kind, value)`` — a scale of the canvas or a fixed width.
SIZES = (
    ("Canvas size (100 %)", "scale", 1.0),
    ("75 %", "scale", 0.75),
    ("50 %", "scale", 0.5),
    ("25 %", "scale", 0.25),
    ("1920 px wide", "width", 1920),
    ("1280 px wide", "width", 1280),
    ("720 px wide", "width", 720),
    ("480 px wide", "width", 480),
)


def output_size(canvas_wh, size_kind, size_value):
    """The ``(w, h)`` frames are encoded at, or ``None`` for the canvas size itself."""
    W, H = (max(int(v), 1) for v in canvas_wh)
    if size_kind == "scale":
        s = float(size_value)
        if abs(s - 1.0) < 1e-9:
            return None
        return max(2, int(round(W * s))), max(2, int(round(H * s)))
    w = int(size_value)
    return max(2, w), max(2, int(round(H * w / W)))


class VideoOptions(QtWidgets.QWidget):
    """Format / frame rate / size / quality for a rendered export, remembered per *key*."""

    changed = QtCore.Signal()

    def __init__(self, key="canvas", parent=None, fps=30.0):
        super().__init__(parent)
        self._key = f"export/{key}"
        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.cb_format = QtWidgets.QComboBox()
        for fmt, label in encode.FORMAT_LABELS:
            if fmt in VIDEO_FORMATS:
                self.cb_format.addItem(label, fmt)
        self.cb_format.setToolTip("MP4 plays everywhere; WebM is smaller for the web; a GIF "
                                  "loops in chat apps and slides (best kept short and small).")
        form.addRow("Format", self.cb_format)
        self.sp_fps = QtWidgets.QDoubleSpinBox()
        self.sp_fps.setRange(1.0, 240.0); self.sp_fps.setDecimals(2)
        self.sp_fps.setValue(float(fps)); self.sp_fps.setSuffix(" fps")
        self.sp_fps.setToolTip("Frames per second of the video. Each frame samples the "
                               "timeline at i / fps, so a higher rate is smoother, not "
                               "slower. (A GIF rounds its delay to 10 ms: 30 fps plays at "
                               "33.3.)")
        form.addRow("Frame rate", self.sp_fps)
        self.cb_size = QtWidgets.QComboBox()
        for label, kind, value in SIZES:
            self.cb_size.addItem(label, (kind, value))
        self.cb_size.setToolTip("Encode at the canvas size, a fraction of it, or a fixed "
                                "width (the height keeps the canvas aspect).")
        form.addRow("Size", self.cb_size)
        self.sp_quality = QtWidgets.QSpinBox()
        self.sp_quality.setRange(1, 10); self.sp_quality.setValue(8)
        self.sp_quality.setToolTip("1 … 10 (10 = best, largest file). MP4: crf 34 … 16; "
                                   "WebM: crf 50 … 18. A GIF uses its colour count instead.")
        form.addRow("Quality", self.sp_quality)
        self.sp_colors = QtWidgets.QSpinBox()
        self.sp_colors.setRange(8, 256); self.sp_colors.setValue(128)
        self.sp_colors.setToolTip("Colours in the GIF's one palette (measured over the "
                                  "whole clip, so colours do not shimmer).")
        form.addRow("GIF colours", self.sp_colors)
        self._form = form
        self._restore()
        for w in (self.cb_format, self.cb_size):
            w.currentIndexChanged.connect(self._on_change)
        for w in (self.sp_fps, self.sp_quality, self.sp_colors):
            w.valueChanged.connect(self._on_change)
        self._sync()

    # --- persistence ------------------------------------------------------------------
    def _settings(self):
        try:
            from gottlux.app import style
            return style.settings()
        except Exception:
            return None

    def _restore(self):
        s = self._settings()
        if s is None:
            return
        try:
            fmt = s.value(f"{self._key}/format", "mp4")
            i = self.cb_format.findData(fmt)
            if i >= 0:
                self.cb_format.setCurrentIndex(i)
            self.sp_fps.setValue(float(s.value(f"{self._key}/fps", self.sp_fps.value())))
            self.cb_size.setCurrentIndex(int(s.value(f"{self._key}/size", 0)))
            self.sp_quality.setValue(int(s.value(f"{self._key}/quality", 8)))
            self.sp_colors.setValue(int(s.value(f"{self._key}/gif_colors", 128)))
        except (TypeError, ValueError):
            pass

    def _save(self):
        s = self._settings()
        if s is None:
            return
        s.setValue(f"{self._key}/format", self.format())
        s.setValue(f"{self._key}/fps", float(self.sp_fps.value()))
        s.setValue(f"{self._key}/size", int(self.cb_size.currentIndex()))
        s.setValue(f"{self._key}/quality", int(self.sp_quality.value()))
        s.setValue(f"{self._key}/gif_colors", int(self.sp_colors.value()))

    def _on_change(self, *_):
        self._sync()
        self._save()
        self.changed.emit()

    def _sync(self):
        gif = self.format() == "gif"
        self._form.setRowVisible(self.sp_quality, not gif)
        self._form.setRowVisible(self.sp_colors, gif)

    # --- values -----------------------------------------------------------------------
    def format(self) -> str:
        return str(self.cb_format.currentData() or "mp4")

    def set_format(self, fmt):
        i = self.cb_format.findData(fmt)
        if i >= 0:
            self.cb_format.setCurrentIndex(i)

    def fps(self) -> float:
        return float(self.sp_fps.value())

    def size_for(self, canvas_wh):
        kind, value = self.cb_size.currentData()
        return output_size(canvas_wh, kind, value)

    def options(self, canvas_wh) -> dict:
        return {"fmt": self.format(), "fps": self.fps(), "size": self.size_for(canvas_wh),
                "quality": int(self.sp_quality.value()),
                "gif_colors": int(self.sp_colors.value())}

    def describe(self, canvas_wh, duration_s) -> str:
        """'MP4 · 1280 × 720 · 30 fps · 151 frames' — what an export will write."""
        size = self.size_for(canvas_wh) or tuple(int(v) for v in canvas_wh)
        w, h = size
        if self.format() in ("mp4", "webm"):
            w, h = encode.even_size(w, h)
        n = max(1, int(round(float(duration_s) * self.fps())))
        return (f"{self.format().upper()} · {w} × {h} · {self.fps():g} fps · {n:,} frames "
                f"({float(duration_s):.2f} s)")


def ask_path(parent, title, default_stem, fmt="mp4"):
    """A save dialog with one filter per format → ``(path, fmt)``, or ``(None, None)``.

    The format comes from the typed extension if it names one, else from the chosen filter,
    else *fmt*; the path is given that format's extension."""
    filters = ";;".join(_FILTERS[f] for f in VIDEO_FORMATS)
    start = encode.with_extension(default_stem, fmt)
    out, chosen = QtWidgets.QFileDialog.getSaveFileName(parent, title, start, filters,
                                                        _FILTERS.get(fmt, ""))
    if not out:
        return None, None
    ext = os.path.splitext(out)[1].lower().lstrip(".")
    if ext in VIDEO_FORMATS:
        use = ext
    else:
        use = next((f for f, flt in _FILTERS.items() if flt == chosen), fmt)
    return encode.with_extension(out, use), use


class _Job(Worker):
    """Runs ``work(progress, cancel)`` on a worker thread; keeps its result or error."""

    progressed = QtCore.Signal(float)

    def __init__(self, work, parent=None):
        super().__init__(parent)
        self._work = work
        self._cancel = False
        self.result = None
        self.error = None

    def cancel(self):
        self._cancel = True

    @property
    def cancelled(self) -> bool:
        return self._cancel

    def run(self):
        try:
            self.result = self._work(self.progressed.emit, lambda: self._cancel)
        except BaseException as e:            # surfaced to the GUI thread by run_job
            self.error = e


def run_job(parent, title, label, work, cancellable=True):
    """Run ``work(progress, cancel)`` off the GUI thread behind a progress dialog.

    The dialog is window-modal (the window keeps painting; only *Cancel* takes input) and
    the call returns once the job has finished → ``(result, cancelled)``. An exception
    raised by *work* is re-raised here, on the GUI thread.
    """
    dlg = QtWidgets.QProgressDialog(label, "Cancel" if cancellable else None, 0, 1000, parent)
    dlg.setWindowTitle(title)
    dlg.setWindowModality(QtCore.Qt.WindowModal)
    dlg.setMinimumWidth(380)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False); dlg.setAutoReset(False)
    if not cancellable:
        dlg.setCancelButton(None)
    job = _Job(work)
    loop = QtCore.QEventLoop()
    job.progressed.connect(lambda f: dlg.setValue(int(max(0.0, min(float(f), 1.0)) * 1000)))
    job.finished.connect(loop.quit)
    state = {"done": False}

    def on_cancel():                             # closing the dialog emits canceled too
        if not state["done"]:
            job.cancel()
            dlg.setLabelText("Cancelling…")
    dlg.canceled.connect(on_cancel)
    guard = QtCore.QTimer(interval=100)          # never miss a finish that beat exec()
    guard.timeout.connect(lambda: loop.quit() if job.isFinished() else None)
    dlg.show()
    job.start()
    guard.start()
    if not job.isFinished():
        loop.exec()
    guard.stop()
    job.wait()
    state["done"] = True
    cancelled = job.cancelled
    dlg.close()
    dlg.deleteLater()
    if job.error is not None:
        raise job.error
    return job.result, cancelled
