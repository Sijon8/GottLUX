"""
capture.py — record what any view shows to a video / poster, with a context infographic.

A generic screen capture: it drives the shared clock across a time range, **grabs the active
view's pixels** at each step (works for the pyqtgraph 2-D views and the OpenGL 3-D view),
optionally crops to a **drag-selected region**, wraps each frame in a **context banner**
(recording, view, time, settings — the "infographic"), and muxes to MP4.

The clip lands in the suite's standard **provenance folder**
(:mod:`gottlux.run.export_provenance`): the MP4, the optional **poster** PNG, a README
naming the source recording with its directory and SHA-256, and ``provenance.json``. The
capture's former ``*_manifest.json`` is *merged into* that ``provenance.json`` — its
title/view/window/fps/region/context fields are the record's settings — so one convention
covers every export in the suite instead of two.

The pixel grab and the dialog live here (they need Qt); the frame compositing/encoding is in
:mod:`gottlux.viz.video`.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from gottlux.app import icons
from gottlux.app.timeline import RangeSlider
from gottlux.app.transport import REALTIME_FPS
from gottlux.viz import video as _video

# Cap on rendered frames per clip — guards extreme slow-motion (high equivalent FPS × long
# window) from producing an unbounded render. ~10 min of 30 fps output.
_MAX_FRAMES = 18000


# --------------------------------------------------------------------- pixel grab
def qimage_to_rgb(qimg) -> np.ndarray:
    """Convert a QImage to a contiguous ``(H, W, 3)`` uint8 RGB array."""
    qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    w, h, bpl = qimg.width(), qimg.height(), qimg.bytesPerLine()
    buf = np.frombuffer(memoryview(qimg.constBits()), np.uint8)
    return np.ascontiguousarray(buf[:bpl * h].reshape(h, bpl)[:, :w * 3].reshape(h, w, 3))


def gl_to_rgb(arr) -> np.ndarray:
    """Normalize a ``GLViewWidget.renderToArray`` result (``(w, h, 4)``) to ``(h, w, 3)`` RGB."""
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return np.ascontiguousarray(arr.transpose(1, 0, 2)[..., :3]).astype(np.uint8)
    return np.zeros((8, 8, 3), np.uint8)


def grab_widget(widget) -> np.ndarray:
    """Grab a widget's pixels to RGB — uses the GL framebuffer for an OpenGL view, else ``grab()``."""
    try:
        fb = getattr(widget, "grabFramebuffer", None)
        if callable(fb):
            return qimage_to_rgb(widget.grabFramebuffer())
        return qimage_to_rgb(widget.grab().toImage())
    except Exception:
        return np.zeros((8, 8, 3), np.uint8)


# --------------------------------------------------------------------- region selector
class RegionSelector(QtCore.QObject):
    """Drag a rubber-band over *target* to pick a crop rectangle (target-local px)."""
    selected = QtCore.Signal(object)         # (x, y, w, h) or None

    def __init__(self, target):
        super().__init__(target)
        self.target = target
        self._band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, target)
        self._origin = None
        target.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.MouseButtonPress:
            self._origin = ev.position().toPoint()
            self._band.setGeometry(QtCore.QRect(self._origin, QtCore.QSize())); self._band.show()
            return True
        if ev.type() == QtCore.QEvent.MouseMove and self._origin is not None:
            self._band.setGeometry(QtCore.QRect(self._origin, ev.position().toPoint()).normalized())
            return True
        if ev.type() == QtCore.QEvent.MouseButtonRelease and self._origin is not None:
            r = self._band.geometry(); self._band.hide(); self._origin = None
            self.target.removeEventFilter(self)
            rect = (r.x(), r.y(), r.width(), r.height()) if r.width() > 4 and r.height() > 4 else None
            self.selected.emit(rect)
            return True
        return False


# --------------------------------------------------------------------- capture dialog
class ScreenCaptureDialog(QtWidgets.QDialog):
    """Record the active view to a video + poster + manifest, with a context infographic.

    *ctx* keys: ``rec`` (Recording), ``target`` (the widget to grab), ``set_cursor`` (callable
    ``t→None`` on the shared clock), ``view`` (name), ``t0``/``t1`` (default range, s), ``accum``,
    and ``fields`` (an ordered dict of context key/values).
    """

    def __init__(self, parent, ctx):
        super().__init__(parent)
        self.ctx = ctx
        self.region = None
        self.setWindowTitle("Capture view — video + infographic")
        self.setMinimumWidth(540)
        rec = ctx.get("rec")
        dur = float(getattr(rec, "duration_s", 1.0) or 1.0)
        t0 = float(ctx.get("t0", getattr(rec, "t_start_s", 0.0)))
        t1 = float(ctx.get("t1", getattr(rec, "t_stop_s", dur)))

        self.trim = RangeSlider(); self.trim.setEnabled(True)
        self.trim.set_range((t0 - (rec.t_start_s if rec else 0)) / dur if dur else 0.0,
                            (t1 - (rec.t_start_s if rec else 0)) / dur if dur else 1.0)
        self.in_s = QtWidgets.QDoubleSpinBox(); self.in_s.setRange(0, 1e6); self.in_s.setDecimals(3)
        self.out_s = QtWidgets.QDoubleSpinBox(); self.out_s.setRange(0, 1e6); self.out_s.setDecimals(3)
        self.in_s.setValue(t0); self.out_s.setValue(t1)
        self._base_t = rec.t_start_s if rec else 0.0; self._dur = dur
        self.trim.rangeChanged.connect(self._trim_to_spins)
        self.fps = QtWidgets.QSpinBox(); self.fps.setRange(1, 100000)
        _fps0 = ctx.get("fps")
        self.fps.setValue(int(min(max(round(float(_fps0)), 1), 100000)) if _fps0 else 30)
        self.fps.setToolTip(
            "Equivalent (slow-motion) capture FPS — the same meaning as the viewer's FPS, "
            "pre-filled from it. The file is written at real-time cadence (30 fps), so it plays "
            "slowed by FPS/30 (e.g. 1000 → ~33× slow-motion). 30 = real time.")
        # accumulation (exposure) per rendered frame — pre-filled from the view, adjustable here
        self.accum = QtWidgets.QDoubleSpinBox(); self.accum.setRange(1e-5, 2.0)
        self.accum.setDecimals(5); self.accum.setSingleStep(0.001); self.accum.setSuffix(" s")
        self.accum.setValue(float(ctx.get("accum") or 0.02))
        self.accum.setToolTip("Accumulation (exposure) integrated into each rendered frame — "
                              "pre-filled from the viewer.")
        # live readout of what FPS + In/Out will produce (frames · duration · slow factor)
        self.calc_lbl = QtWidgets.QLabel(""); self.calc_lbl.setObjectName("muted")
        for _w in (self.fps, self.in_s, self.out_s):
            _w.valueChanged.connect(self._update_calc)
        self.trim.rangeChanged.connect(lambda *_: self._update_calc())
        # faithful render path: reproduce the tuned view at a chosen resolution (not a screen grab)
        self.render = ctx.get("render")
        self.sensor_wh = ctx.get("sensor_wh")
        self.res = QtWidgets.QComboBox()
        if self.render is not None and self.sensor_wh:
            self.res.addItems(["Faithful · 1080p", "Faithful · 720p", "Faithful · native",
                               "Faithful · 2×", "Faithful · 4×", "On-screen (grab)"])
            self.res.setToolTip("High-res faithful render reproduces this view's exact tuned "
                                "settings at the chosen resolution. 'On-screen' grabs the widget.")
        else:
            self.res.addItems(["On-screen (grab)"])
            self.res.setToolTip("This view captures the on-screen widget pixels.")
        self.banner = QtWidgets.QCheckBox("Context banner (infographic overlay)"); self.banner.setChecked(True)
        self.poster = QtWidgets.QCheckBox("Also save a poster PNG"); self.poster.setChecked(True)
        self.poster.setToolTip("Save a context poster (the middle frame plus the settings "
                               "panel) beside the clip. The README and provenance.json are "
                               "written either way.")
        self.title = QtWidgets.QLineEdit(f"{getattr(rec, 'name', 'capture')} — {ctx.get('view','view')}")
        self.note = QtWidgets.QLineEdit(); self.note.setPlaceholderText("note (optional) — into the manifest")
        self.region_btn = QtWidgets.QPushButton("Select region on view…")
        self.region_btn.clicked.connect(self._select_region)
        self.region_lbl = QtWidgets.QLabel("region: full view"); self.region_lbl.setObjectName("muted")
        self.out_edit = QtWidgets.QLineEdit(self._default_out(rec))
        browse = QtWidgets.QPushButton("Browse…"); browse.clicked.connect(self._browse)

        form = QtWidgets.QFormLayout()
        form.addRow("Title", self.title)
        form.addRow("Note", self.note)
        trow = QtWidgets.QHBoxLayout(); trow.addWidget(QtWidgets.QLabel("In")); trow.addWidget(self.in_s)
        trow.addWidget(QtWidgets.QLabel("Out")); trow.addWidget(self.out_s)
        trow.addWidget(QtWidgets.QLabel("FPS (equiv)")); trow.addWidget(self.fps)
        trow.addWidget(QtWidgets.QLabel("Accum")); trow.addWidget(self.accum)
        trow.addWidget(QtWidgets.QLabel("Res")); trow.addWidget(self.res, 1)
        rrow = QtWidgets.QHBoxLayout(); rrow.addWidget(self.region_btn); rrow.addWidget(self.region_lbl, 1)
        orow = QtWidgets.QHBoxLayout(); orow.addWidget(self.out_edit, 1); orow.addWidget(browse)

        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(form)
        v.addWidget(QtWidgets.QLabel("Time range (drag handles):")); v.addWidget(self.trim)
        v.addLayout(trow); v.addWidget(self.calc_lbl)
        v.addWidget(self.banner); v.addWidget(self.poster)
        self._update_calc()
        v.addLayout(rrow)
        v.addWidget(QtWidgets.QLabel("Output (.mp4):")); v.addLayout(orow)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        cap = bb.addButton("Capture", QtWidgets.QDialogButtonBox.AcceptRole)
        cap.setIcon(icons.icon("capture"))
        cap.clicked.connect(self._run)
        rec = bb.addButton("Screen record…", QtWidgets.QDialogButtonBox.ActionRole)
        rec.setIcon(icons.icon("record"))
        rec.setToolTip("Record the on-screen pixels of this view live (like a screen recorder), "
                       "instead of re-rendering frames — the most faithful 'capture what I see'.")
        rec.clicked.connect(self._screen_record)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    # ----- helpers -----
    def _default_out(self, rec):
        base = (os.path.splitext(rec.source_path)[0] if rec and rec.source_path
                else os.path.join(os.getcwd(), getattr(rec, "name", "capture")))
        view = str(self.ctx.get("view", "view")).lower().replace(" ", "_")
        return f"{base}_{view}_capture.mp4"

    def _browse(self):
        out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Capture output", self.out_edit.text(),
                                                       "MP4 video (*.mp4)")
        if out:
            self.out_edit.setText(out)

    def _trim_to_spins(self, lo, hi):
        b1 = QtCore.QSignalBlocker(self.in_s); self.in_s.setValue(self._base_t + lo * self._dur); del b1
        b2 = QtCore.QSignalBlocker(self.out_s); self.out_s.setValue(self._base_t + hi * self._dur); del b2

    def _frame_plan(self):
        """The export's timing for the current FPS + In/Out.

        The FPS field is an *equivalent* (slow-motion) capture rate, exactly like the viewer's:
        we sample one frame per 1/FPS of recording time, then write the file at real-time cadence
        (``REALTIME_FPS`` = 30 fps), so the clip plays slowed by ``FPS / 30``. Returns
        ``(n_frames, out_fps, slow_factor, requested_frames)`` — ``n`` is clamped to
        :data:`_MAX_FRAMES`.
        """
        t0, t1 = sorted((self.in_s.value(), self.out_s.value()))
        equiv = float(self.fps.value())
        requested = max(int(round((t1 - t0) * equiv)), 1)
        return min(requested, _MAX_FRAMES), REALTIME_FPS, equiv / REALTIME_FPS, requested

    def _speed_text(self, slow):
        if abs(slow - 1.0) < 1e-3:
            return "real-time"
        return f"{slow:.0f}× slow-mo" if slow > 1 else f"{1.0 / slow:.0f}× fast"

    def _update_calc(self):
        n, out_fps, slow, requested = self._frame_plan()
        capped = "  (capped — lower FPS or shorten In/Out for the true rate)" if requested > n else ""
        self.calc_lbl.setText(
            f"→ {n} frames · {n / out_fps:.1f}s video @ {out_fps:g} fps · {self._speed_text(slow)}{capped}")

    def _target_size(self):
        """``(width, height)`` for the chosen faithful resolution, or ``None`` for on-screen grab."""
        choice = self.res.currentText()
        if "On-screen" in choice or not self.sensor_wh or self.render is None:
            return None
        w, h = int(self.sensor_wh[0]), int(self.sensor_wh[1])
        if "1080p" in choice:
            return (round(w * 1080 / h), 1080)
        if "720p" in choice:
            return (round(w * 720 / h), 720)
        if "2×" in choice:
            return (w * 2, h * 2)
        if "4×" in choice:
            return (w * 4, h * 4)
        return (w, h)                                    # native

    def _select_region(self):
        target = self.ctx.get("target")
        if target is None:
            return
        self.region_lbl.setText("drag a rectangle on the view…")
        sel = RegionSelector(target)
        sel.selected.connect(self._region_picked)
        self._sel = sel                                  # keep a ref alive

    def _region_picked(self, rect):
        self.region = rect
        self.region_lbl.setText(f"region: {rect}" if rect else "region: full view")

    def _screen_record(self):
        """Hand off to the live screen recorder, pre-aimed at this view (records on-screen pixels)."""
        from gottlux.app.screenrec import ScreenRecordDialog
        target = self.ctx.get("target")
        out = os.path.dirname(os.path.abspath(self.out_edit.text().strip() or os.getcwd())) or None
        self.reject()                                    # close this dialog; the recorder takes over
        ScreenRecordDialog(self.parent() or self, view_widget=target, default_dir=out).exec()

    # ----- run -----
    def _run(self):
        target = self.ctx.get("target"); set_cursor = self.ctx.get("set_cursor")
        if target is None or set_cursor is None:
            QtWidgets.QMessageBox.warning(self, "Capture", "No view to capture."); return
        out = self.out_edit.text().strip()
        if not out:
            QtWidgets.QMessageBox.warning(self, "Capture", "Choose an output path."); return
        if not out.lower().endswith(".mp4"):
            out += ".mp4"
        t0, t1 = sorted((self.in_s.value(), self.out_s.value()))
        if t1 - t0 < 0.01:
            QtWidgets.QMessageBox.warning(self, "Capture", "Time range is too short."); return
        # FPS is an equivalent (slow-mo) rate: sample at that rate, write at real-time cadence
        # so the file plays slowed by FPS/30 (matches the viewer).
        n, out_fps, slow, _requested = self._frame_plan()
        times = t0 + (np.arange(n) + 0.5) * (t1 - t0) / n
        fields = dict(self.ctx.get("fields") or {})
        if self.note.text().strip():
            fields["Note"] = self.note.text().strip()
        fields["Playback"] = self._speed_text(slow)

        size = self._target_size()
        use_render = self.render is not None and size is not None     # faithful high-res path
        dt = float(self.accum.value())
        fields["Accum (exposure)"] = f"{dt * 1e3:.1f} ms"            # banner reflects the chosen exposure
        prog = QtWidgets.QProgressDialog(
            f"Capturing view ({'faithful ' + ('%d×%d' % size) if use_render else 'on-screen'})…",
            "Abort", 0, n, self)
        prog.setWindowModality(QtCore.Qt.WindowModal); prog.setMinimumWidth(360); prog.show()
        x = y = w = h = None
        if self.region and not use_render:               # region crop applies to the on-screen grab
            x, y, w, h = self.region
        mid_frame = [None]

        def frames():
            for i, t in enumerate(times):
                if prog.wasCanceled():
                    break
                if use_render:                           # reproduce the tuned view at target res
                    rgb = self.render(float(t), dt, size)
                    if rgb is None:
                        rgb = np.zeros((8, 8, 3), np.uint8)
                else:                                    # grab exactly what's on screen
                    set_cursor(float(t)); QtWidgets.QApplication.processEvents()
                    rgb = grab_widget(target)
                    if w:                                # crop to the selected region
                        rgb = rgb[max(y, 0):y + h, max(x, 0):x + w]
                if self.banner.isChecked():
                    rgb = _video.infographic_frame(
                        rgb, title=self.title.text(),
                        subtitle=" · ".join(f"{k}: {v}" for k, v in list(fields.items())[:3]),
                        footer_lines=[f"t = {t:.3f} s   ·   {self.ctx.get('view','')}"])
                if i == n // 2:
                    mid_frame[0] = rgb
                prog.setValue(i + 1)
                yield rgb

        from gottlux.run import export_provenance as eprov
        folder = eprov.export_folder(out)
        target = eprov.artifact_path(folder, out)
        path = _video.write_video(target, frames(), fps=out_fps)
        prog.close()
        if not path:
            eprov.discard_folder(folder)
            if not _video.ffmpeg_available():
                msg = ("Video export needs the FFMPEG muxer, which isn't available.\n\n"
                       "Install it with:\n    pip install imageio-ffmpeg")
            else:
                msg = ("Video export failed while encoding (see the console log for the FFMPEG "
                       "error). The view may have produced no frames at this resolution — try the "
                       "'On-screen (grab)' resolution, or a different range.")
            QtWidgets.QMessageBox.warning(self, "Capture", msg)
            return
        written = [path]
        if self.poster.isChecked():
            written += self._save_poster(folder, target, mid_frame[0], fields)
        self._write_provenance(folder, target, fields, (t0, t1), n, out_fps, slow,
                               use_render, size)
        from gottlux.io.paths import open_in_file_browser
        open_in_file_browser(folder)
        QtWidgets.QMessageBox.information(
            self, "Capture",
            f"Saved ({n} frames · {n / out_fps:.1f}s @ {out_fps:g} fps · {self._speed_text(slow)}) "
            f"→\n{folder}\n\n"
            + "\n".join(os.path.basename(p) for p in written)
            + "\nREADME.md\nprovenance.json")
        self.accept()

    def _save_poster(self, folder, target, frame, fields):
        """Write the context poster (middle frame + settings panel) into the folder."""
        out_png = os.path.join(folder, os.path.splitext(os.path.basename(target))[0]
                               + "_poster.png")
        try:
            if frame is None:
                frame = grab_widget(self.ctx.get("target"))
            from PIL import Image
            Image.fromarray(_video.context_poster(frame, self.title.text(), fields)).save(out_png)
            return [out_png]
        except Exception as e:
            print(f"[capture] poster failed: {e}")
            return []

    def _write_provenance(self, folder, target, fields, window, n, out_fps, slow,
                          use_render, size):
        """Write the folder's README.md + provenance.json.

        This is where the capture's old ``*_manifest.json`` went: its title, view, window,
        fps, region and context fields are the provenance record's settings, so a capture
        self-documents through the same one convention every other export uses.
        """
        from gottlux.run import export_provenance as eprov
        rec = self.ctx.get("rec")
        source = eprov.source_facts(getattr(rec, "source_path", "") or "", rec=rec)
        settings = {
            "Title": self.title.text(),
            "View": self.ctx.get("view", ""),
            "Capture path": (f"faithful re-render at {size[0]} × {size[1]} px" if use_render
                             else "on-screen pixel grab"),
            "Region crop (view px)": self.region or "none — the whole view",
            "Equivalent (slow-motion) FPS": f"{self.fps.value():g} fps",
            "File cadence": f"{out_fps:g} fps ({self._speed_text(slow)})",
            "Accumulation (exposure)": f"{self.accum.value() * 1e3:g} ms",
            "Context banner": "on" if self.banner.isChecked() else "off",
        }
        settings.update({f"Context — {k}": v for k, v in fields.items()})
        usage = [{"source": 1, "name": str(self.ctx.get("view", "view")),
                  "trim_in_s": float(window[0]), "trim_out_s": float(window[1]),
                  "accumulation_s": float(self.accum.value()),
                  "note": "the in/out points are absolute recording times, and each frame "
                          "integrates the accumulation window above"}]
        return eprov.write_provenance(
            folder, "View capture (MP4)",
            eprov.artifact_facts(target, frames=int(n), fps=float(out_fps),
                                 duration_s=n / float(out_fps),
                                 codec="H.264 (libx264) in MP4"),
            [source], settings,
            extra={"usage": usage,
                   "reproduce": {"steps": [
                       "Load the source recording listed above and open the view named in "
                       "the settings table.",
                       "Apply the context settings, then use 'Capture view…' with the same "
                       "In/Out, FPS, accumulation and resolution."]}})
