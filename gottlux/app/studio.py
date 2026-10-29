"""
studio.py — the **Studio** tab: try visualization looks on a span of the recording and push
out short GIF / MP4 / WebM shots.

Left: a live, looping preview of the shot (drag on it in *Crop* mode to frame the target).
Right: the look and its parameters (auto-built from the look's ``Param`` list, like the
detector tuner), the shot span / speed, the style, and export. Everything the panel holds is
a :class:`gottlux.studio.Shot`; *Copy recipe* puts it on the clipboard as JSON and every
export writes it beside the file, so a shot can be re-made exactly or handed to
``gottlux-studio`` / the web Studio.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from gottlux.app import icons, style
from gottlux.app.colorschemes import PaletteCombo
from gottlux.app.loader import Worker
from gottlux.studio import (
    FORMATS,
    PRESETS,
    Shot,
    ShotRenderer,
    default_folder,
    export_shot,
    list_looks,
)
from gottlux.studio.render import clamp_crop, prepare_events, speed_text

SPEEDS = (("real time", 1.0), ("0.5×", 0.5), ("0.25×", 0.25), ("0.1× (10× slower)", 0.1),
          ("0.05×", 0.05), ("0.02×", 0.02), ("0.01× (100× slower)", 0.01))
PREVIEW_MAX_W = 640


class _PreviewView(QtWidgets.QLabel):
    """Aspect-fit image view; in crop mode a drag emits a rectangle in image pixels."""
    cropDrawn = QtCore.Signal(object)

    def __init__(self):
        super().__init__(alignment=QtCore.Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.crop_mode = False
        self._img_size = (1, 1)
        self._origin = None
        self._band = None
        self._qimg = None

    def show_frame(self, arr: np.ndarray):
        h, w = arr.shape[:2]
        arr = np.ascontiguousarray(arr)
        self._qimg = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self._img_size = (w, h)
        self._refit()

    def _refit(self):
        if self._qimg is None:
            return
        pm = QtGui.QPixmap.fromImage(self._qimg).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(pm)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refit()

    def _to_image(self, pt):
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None
        ox = (self.width() - pm.width()) / 2
        oy = (self.height() - pm.height()) / 2
        s = pm.width() / max(self._img_size[0], 1)
        return ((pt.x() - ox) / s, (pt.y() - oy) / s)

    def mousePressEvent(self, e):
        if not self.crop_mode:
            return
        self._origin = e.position().toPoint()
        if self._band is None:
            self._band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._band.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
        self._band.show()

    def mouseMoveEvent(self, e):
        if self._origin is not None:
            self._band.setGeometry(QtCore.QRect(self._origin, e.position().toPoint()).normalized())

    def mouseReleaseEvent(self, e):
        if self._origin is None:
            return
        r = self._band.geometry()
        self._origin = None
        self._band.hide()
        a, b = self._to_image(r.topLeft()), self._to_image(r.bottomRight())
        if a and b:
            self.cropDrawn.emit((a[0], a[1], b[0], b[1]))


class _ExportWorker(Worker):
    """The export, off the GUI thread. A loader ``Worker``: referenced until it finishes and
    waited for at exit, so closing the window mid-export cannot destroy a running thread."""
    progress = QtCore.Signal(float)
    finished_ok = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, rec, shot, path, folder):
        super().__init__()
        self.rec, self.shot, self.path, self.folder = rec, shot, path, folder
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            facts = export_shot(self.rec, self.shot, self.path, folder=self.folder,
                                progress=self.progress.emit, cancel=lambda: self._cancel)
            self.finished_ok.emit(facts)
        except Exception as e:                           # reported in the panel, never raised
            self.failed.emit(f"{type(e).__name__}: {e}")


class StudioPanel(QtWidgets.QWidget):
    """The Studio tab (GottLUX panel protocol: ``set_recording`` / ``sync``)."""

    def __init__(self, clock=None, filters=None, parent=None):
        super().__init__(parent)
        from gottlux.app.uikit import plot_with_deck
        from gottlux.app.widgets import ParamPanel
        self._ParamPanel = ParamPanel
        self.clock = clock
        self.rec = None
        self._renderer = None
        self._events_key = None
        self._events = None
        self._frame = 0
        self._worker = None
        self._looks = list_looks()

        # ---------------- view: preview + transport
        view = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(view); vl.setContentsMargins(0, 0, 0, 0)
        self.preview = _PreviewView()
        self.preview.setText("Load a recording, pick a look, press Export.")
        self.preview.cropDrawn.connect(self._crop_drawn)
        vl.addWidget(self.preview, 1)
        tr = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QToolButton(); self.play_btn.setIcon(icons.icon("pause"))
        self.play_btn.setCheckable(True); self.play_btn.setChecked(True)
        self.play_btn.setToolTip("Play / pause the preview loop")
        self.play_btn.toggled.connect(self._toggle_play)
        tr.addWidget(self.play_btn)
        self.frame_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_sl.valueChanged.connect(self._scrub)
        tr.addWidget(self.frame_sl, 1)
        self.crop_btn = QtWidgets.QPushButton("Crop"); self.crop_btn.setCheckable(True)
        self.crop_btn.setToolTip("Show the whole sensor; drag a box around the subject.")
        self.crop_btn.toggled.connect(self._toggle_crop)
        tr.addWidget(self.crop_btn)
        clear = QtWidgets.QPushButton("Full frame")
        clear.setToolTip("Remove the crop")
        clear.clicked.connect(lambda: self._set_crop(None))
        tr.addWidget(clear)
        vl.addLayout(tr)
        self.readout = QtWidgets.QLabel("—"); self.readout.setObjectName("muted")
        vl.addWidget(self.readout)

        # ---------------- deck
        deck = QtWidgets.QWidget()
        dl = QtWidgets.QVBoxLayout(deck)
        lk = QtWidgets.QGroupBox("Visualization")
        ll = QtWidgets.QVBoxLayout(lk)
        self.look_cb = QtWidgets.QComboBox()
        for name, look in self._looks.items():
            self.look_cb.addItem(look.title + ("  (slower)" if look.heavy else ""), name)
        self.look_cb.currentIndexChanged.connect(self._look_changed)
        ll.addWidget(self.look_cb)
        self.look_desc = QtWidgets.QLabel(); self.look_desc.setWordWrap(True)
        self.look_desc.setObjectName("muted")
        ll.addWidget(self.look_desc)
        self._param_host = QtWidgets.QVBoxLayout(); ll.addLayout(self._param_host)
        self.params = None
        dl.addWidget(lk)

        sh = QtWidgets.QGroupBox("Shot")
        sf = QtWidgets.QFormLayout(sh)
        self.t0_sp = self._dspin(0, 1e6, 3, " s"); self.t1_sp = self._dspin(0, 1e6, 3, " s")
        span = QtWidgets.QHBoxLayout()
        span.addWidget(self.t0_sp); span.addWidget(QtWidgets.QLabel("→"))
        span.addWidget(self.t1_sp)
        sf.addRow("Span", span)
        sb = QtWidgets.QHBoxLayout()
        inout = QtWidgets.QPushButton("Use In/Out"); inout.clicked.connect(self._use_inout)
        inout.setToolTip("Take the span from the shared transport's In/Out selection")
        whole = QtWidgets.QPushButton("Whole clip"); whole.clicked.connect(self._whole)
        sb.addWidget(inout); sb.addWidget(whole)
        sf.addRow("", sb)
        self.speed_cb = QtWidgets.QComboBox()
        for lab, v in SPEEDS:
            self.speed_cb.addItem(lab, v)
        self.speed_cb.setCurrentIndex(3)
        self.speed_cb.setToolTip("Recording seconds shown per second of output")
        sf.addRow("Speed", self.speed_cb)
        self.crop_lbl = QtWidgets.QLabel("full frame"); self.crop_lbl.setObjectName("muted")
        sf.addRow("Crop", self.crop_lbl)
        self.denoise_chk = QtWidgets.QCheckBox("Denoise (hot pixels + shot noise)")
        self.denoise_chk.setChecked(True)
        sf.addRow("", self.denoise_chk)
        dl.addWidget(sh)

        sty = QtWidgets.QGroupBox("Style")
        yf = QtWidgets.QFormLayout(sty)
        self.palette_cb = PaletteCombo()
        self.palette_cb.setToolTip("ON / OFF / background colours. 'Add a colour scheme…' at "
                                   "the end of the list makes your own palette.")
        yf.addRow("Palette", self.palette_cb)
        self.bloom_sp = self._dspin(0, 3, 2, ""); self.bloom_sp.setSingleStep(0.1)
        yf.addRow("Bloom", self.bloom_sp)
        self.gamma_sp = self._dspin(0.3, 3, 2, ""); self.gamma_sp.setValue(1.0)
        self.gamma_sp.setSingleStep(0.1)
        yf.addRow("Gamma", self.gamma_sp)
        self.title_ed = QtWidgets.QLineEdit(); self.title_ed.setPlaceholderText("optional title")
        yf.addRow("Title", self.title_ed)
        chk = QtWidgets.QGridLayout()
        self.stamp_chk = QtWidgets.QCheckBox("Time"); self.stamp_chk.setChecked(True)
        self.speedlab_chk = QtWidgets.QCheckBox("Speed"); self.speedlab_chk.setChecked(True)
        self.legend_chk = QtWidgets.QCheckBox("ON/OFF key")
        self.strip_chk = QtWidgets.QCheckBox("Rate strip")
        self.ping_chk = QtWidgets.QCheckBox("Ping-pong loop")
        for i, c in enumerate((self.stamp_chk, self.speedlab_chk, self.legend_chk,
                               self.strip_chk, self.ping_chk)):
            chk.addWidget(c, i // 2, i % 2)
        yf.addRow("Show", chk)
        dl.addWidget(sty)

        ex = QtWidgets.QGroupBox("Export")
        ef = QtWidgets.QFormLayout(ex)
        self.preset_cb = QtWidgets.QComboBox()
        for name, (fmt, w, fps) in PRESETS.items():
            self.preset_cb.addItem(f"{name}  ·  {fmt.upper()} {w}px {fps}fps", name)
        self.preset_cb.setCurrentIndex(list(PRESETS).index("gif"))
        self.preset_cb.currentIndexChanged.connect(self._apply_preset)
        ef.addRow("Preset", self.preset_cb)
        self.fmt_cb = QtWidgets.QComboBox(); self.fmt_cb.addItems(list(FORMATS))
        ef.addRow("Format", self.fmt_cb)
        self.width_sp = QtWidgets.QSpinBox(); self.width_sp.setRange(64, 3840)
        self.width_sp.setSingleStep(16); self.width_sp.setSuffix(" px")
        ef.addRow("Width", self.width_sp)
        self.fps_sp = self._dspin(1, 120, 1, " fps")
        ef.addRow("Frame rate", self.fps_sp)
        fold = QtWidgets.QHBoxLayout()
        self.folder_ed = QtWidgets.QLineEdit(default_folder())
        browse = QtWidgets.QPushButton("…"); browse.setFixedWidth(30)
        browse.clicked.connect(self._browse)
        fold.addWidget(self.folder_ed, 1); fold.addWidget(browse)
        ef.addRow("Folder", fold)
        bl = QtWidgets.QHBoxLayout()
        self.export_btn = QtWidgets.QPushButton("Export"); self.export_btn.setObjectName("primary")
        self.export_btn.setIcon(icons.icon("export"))
        self.export_btn.clicked.connect(self.export)
        self.cancel_btn = QtWidgets.QPushButton("Cancel"); self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(lambda: self._worker and self._worker.cancel())
        openf = QtWidgets.QPushButton("Open folder"); openf.clicked.connect(self._open_folder)
        bl.addWidget(self.export_btn, 1); bl.addWidget(self.cancel_btn); bl.addWidget(openf)
        ef.addRow(bl)
        rl = QtWidgets.QHBoxLayout()
        cp = QtWidgets.QPushButton("Copy recipe"); cp.clicked.connect(self._copy_recipe)
        ld = QtWidgets.QPushButton("Load recipe…"); ld.clicked.connect(self._load_recipe)
        rl.addWidget(cp); rl.addWidget(ld)
        ef.addRow(rl)
        self.progress = QtWidgets.QProgressBar(); self.progress.setVisible(False)
        ef.addRow(self.progress)
        self.export_lbl = QtWidgets.QLabel(""); self.export_lbl.setWordWrap(True)
        self.export_lbl.setObjectName("muted")
        self.export_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        ef.addRow(self.export_lbl)
        dl.addWidget(ex)
        dl.addStretch(1)

        outer = QtWidgets.QVBoxLayout(self); outer.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(plot_with_deck(view, deck, min_view=320, min_deck=330, init_deck=380))

        # ---------------- timers + change wiring
        self._timer = QtCore.QTimer(self); self._timer.timeout.connect(self._tick)
        self._rebuild_timer = QtCore.QTimer(self, singleShot=True, interval=220)
        self._rebuild_timer.timeout.connect(self._rebuild)
        for w in (self.t0_sp, self.t1_sp, self.bloom_sp, self.gamma_sp, self.fps_sp,
                  self.width_sp):
            w.valueChanged.connect(self._changed)
        for w in (self.speed_cb, self.palette_cb, self.fmt_cb):
            w.currentIndexChanged.connect(self._changed)
        for w in (self.denoise_chk, self.stamp_chk, self.speedlab_chk, self.legend_chk,
                  self.strip_chk, self.ping_chk):
            w.toggled.connect(self._changed)
        self.title_ed.textChanged.connect(self._changed)
        self._crop = None
        self._apply_preset()
        self._look_changed()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _dspin(lo, hi, dec, suffix):
        s = QtWidgets.QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(dec)
        s.setSuffix(suffix); s.setKeyboardTracking(False)
        return s

    def shot(self) -> Shot:
        """The panel's current settings as a :class:`Shot`."""
        name = self.look_cb.currentData()
        return Shot(source=os.path.basename(getattr(self.rec, "source_path", "") or "")
                    if self.rec else "",
                    look=name, params=self.params.values() if self.params else {},
                    t0=self.t0_sp.value(), t1=self.t1_sp.value(), crop=self._crop,
                    speed=float(self.speed_cb.currentData()), fps=self.fps_sp.value(),
                    width=self.width_sp.value(), palette=self.palette_cb.currentText(),
                    denoise=self.denoise_chk.isChecked(), bloom=self.bloom_sp.value(),
                    gamma=self.gamma_sp.value(), stamp=self.stamp_chk.isChecked(),
                    speed_label=self.speedlab_chk.isChecked(), title=self.title_ed.text(),
                    legend=self.legend_chk.isChecked(), rate_strip=self.strip_chk.isChecked(),
                    pingpong=self.ping_chk.isChecked(), format=self.fmt_cb.currentText())

    def set_shot(self, shot: Shot):
        """Load a recipe into the controls."""
        idx = self.look_cb.findData(shot.look)
        if idx >= 0:
            self.look_cb.setCurrentIndex(idx)
        if self.params:
            self.params.set_values(shot.params)
        blockers = [QtCore.QSignalBlocker(w) for w in (
            self.t0_sp, self.t1_sp, self.speed_cb, self.fps_sp, self.width_sp, self.palette_cb,
            self.denoise_chk, self.bloom_sp, self.gamma_sp, self.stamp_chk, self.speedlab_chk,
            self.title_ed, self.legend_chk, self.strip_chk, self.ping_chk, self.fmt_cb)]
        if shot.t0 is not None:
            self.t0_sp.setValue(shot.t0)
        if shot.t1 is not None:
            self.t1_sp.setValue(shot.t1)
        best = min(range(len(SPEEDS)), key=lambda i: abs(np.log(SPEEDS[i][1] / shot.speed)))
        self.speed_cb.setCurrentIndex(best)
        self.fps_sp.setValue(shot.fps); self.width_sp.setValue(shot.width)
        self.palette_cb.setCurrentText(shot.palette); self.denoise_chk.setChecked(shot.denoise)
        self.bloom_sp.setValue(shot.bloom); self.gamma_sp.setValue(shot.gamma)
        self.stamp_chk.setChecked(shot.stamp); self.speedlab_chk.setChecked(shot.speed_label)
        self.title_ed.setText(shot.title); self.legend_chk.setChecked(shot.legend)
        self.strip_chk.setChecked(shot.rate_strip); self.ping_chk.setChecked(shot.pingpong)
        self.fmt_cb.setCurrentText(shot.format)
        del blockers
        self._set_crop(shot.crop)

    # ------------------------------------------------------------------ panel protocol
    def set_recording(self, rec):
        self.rec = rec
        self._events_key = None
        for sp in (self.t0_sp, self.t1_sp):
            sp.blockSignals(True); sp.setRange(rec.t_start_s, rec.t_stop_s); sp.blockSignals(False)
        if self.clock is not None and self.clock.has_selection():
            self._use_inout()
        else:
            self.t0_sp.setValue(rec.t_start_s)
            self.t1_sp.setValue(min(rec.t_stop_s, rec.t_start_s + 1.0))
        self._set_crop(None)
        self._changed()

    def sync(self):
        if self.clock is not None:
            c = self.clock.cursor
            self.t0_sp.setValue(max(self.rec.t_start_s, c - 0.5) if self.rec else c)
            self.t1_sp.setValue(min(self.rec.t_stop_s, c + 0.5) if self.rec else c)

    def capture_clock(self):
        return None

    # ------------------------------------------------------------------ change handling
    def _look_changed(self, *_):
        name = self.look_cb.currentData()
        look = self._looks[name]
        self.look_desc.setText(look.description)
        if self.params is not None:
            self.params.setParent(None); self.params.deleteLater()
        self.params = self._ParamPanel(look.PARAMS)
        self.params.changed.connect(lambda *_: self._changed())
        self._param_host.addWidget(self.params)
        self._changed()

    def _apply_preset(self, *_):
        fmt, w, fps = PRESETS[self.preset_cb.currentData()]
        for wdg, v in ((self.width_sp, w), (self.fps_sp, fps)):
            wdg.blockSignals(True); wdg.setValue(v); wdg.blockSignals(False)
        self.fmt_cb.setCurrentText(fmt)
        self._changed()

    def _changed(self, *_):
        self._rebuild_timer.start()

    def _rebuild(self):
        """Rebuild the preview renderer (events re-sliced only when span/crop/denoise change)."""
        if self.rec is None:
            return
        shot = self.shot()
        prev_w = min(shot.width, PREVIEW_MAX_W, max(240, self.preview.width()))
        pshot = replace(shot, width=prev_w)
        look = self._looks[shot.look]
        P = look.resolve(shot.params)
        t0, t1 = shot.span(self.rec.t_start_s, self.rec.t_stop_s)
        key = (t0, t1, shot.crop, shot.denoise, shot.baf_us, look.memory_s(P))
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            if key != self._events_key:
                self._events = prepare_events(self.rec, shot, t0, t1, look.memory_s(P))
                self._events_key = key
            self._renderer = ShotRenderer(self.rec, pshot, events=self._events)
        except Exception as e:
            self._renderer = None
            self.readout.setText(f"preview failed: {type(e).__name__}: {e}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        n = len(self._renderer)
        self.frame_sl.blockSignals(True); self.frame_sl.setRange(0, max(n - 1, 0))
        self.frame_sl.blockSignals(False)
        self._frame = min(self._frame, n - 1)
        self._timer.setInterval(int(1000 / min(shot.fps, 30)))
        if self._should_play():
            self._timer.start()
        self._show()

    def _show(self):
        r = self._renderer
        if r is None:
            return
        if self.crop_btn.isChecked():
            self.preview.show_frame(self._overview())
            self.readout.setText("Drag a box around the subject.")
            return
        try:
            self.preview.show_frame(r.frame(self._frame))
        except Exception as e:
            self.readout.setText(f"render failed: {type(e).__name__}: {e}")
            return
        full = self.shot()
        k = r.source_index(self._frame)
        w = full.width
        h = int(round(w * r.size[1] / max(r.size[0], 1)))
        self.readout.setText(
            f"frame {self._frame + 1}/{len(r)} · t = {r.times[k]:.3f} s · "
            f"{len(r) / full.fps:.1f} s at {full.fps:g} fps · "
            f"{'phase loop' if full.look == 'strobe' else speed_text(full.speed)} · "
            f"export {w}×{h} {full.format.upper()}")

    def _overview(self) -> np.ndarray:
        """Whole-sensor event density over the span (crop picker background)."""
        t0, t1 = self.t0_sp.value(), self.t1_sp.value()
        win = self.rec.window(t0, t1)
        W, H = self.rec.width, self.rec.height
        img = np.bincount(np.asarray(win.y, np.int64) * W + np.asarray(win.x),
                          minlength=W * H).reshape(H, W).astype(np.float32)
        hi = np.percentile(img[img > 0], 99.0) if np.any(img > 0) else 1.0
        g = np.sqrt(np.clip(img / hi, 0, 1))
        rgb = (np.stack([g * 0.9, g * 0.95, g], -1) * 255).astype(np.uint8)
        if self._crop:
            x0, y0, x1, y1 = self._crop
            rgb[y0:y1, [x0, x1 - 1]] = (57, 197, 207)
            rgb[[y0, y1 - 1], x0:x1] = (57, 197, 207)
        return rgb

    def _should_play(self) -> bool:
        """The preview loops only while it can be seen: playing, not picking a crop, and
        on screen (a hidden Studio tab looping cost a quarter of a core)."""
        return (self.play_btn.isChecked() and not self.crop_btn.isChecked()
                and self.isVisible())

    def showEvent(self, ev):                    # noqa: N802 (Qt API)
        super().showEvent(ev)
        if self._should_play() and self._renderer is not None:
            self._timer.start()

    def hideEvent(self, ev):                    # noqa: N802 (Qt API)
        super().hideEvent(ev)
        self._timer.stop()

    def _tick(self):
        if not self.isVisible():
            self._timer.stop()
            return
        if self._renderer is None or not len(self._renderer):
            return
        self._frame = (self._frame + 1) % len(self._renderer)
        self.frame_sl.blockSignals(True); self.frame_sl.setValue(self._frame)
        self.frame_sl.blockSignals(False)
        self._show()

    def _scrub(self, v):
        self._frame = int(v)
        self._show()

    def _toggle_play(self, on):
        self.play_btn.setIcon(icons.icon("pause" if on else "play"))
        if self._should_play():
            self._timer.start()
        else:
            self._timer.stop()

    def _toggle_crop(self, on):
        self.preview.crop_mode = on
        if on:
            self._timer.stop()
        elif self._should_play():
            self._timer.start()
        self._show()

    def _crop_drawn(self, r):
        x0, y0, x1, y1 = r
        self._set_crop((int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))))
        self.crop_btn.setChecked(False)

    def _set_crop(self, crop):
        W = self.rec.width if self.rec else 320
        H = self.rec.height if self.rec else 320
        self._crop = clamp_crop(crop, W, H)
        self.crop_lbl.setText("full frame" if not self._crop else
                              "x {0}–{2}, y {1}–{3}".format(*self._crop))
        self._changed()

    def _use_inout(self):
        if self.clock is None:
            return
        self.t0_sp.setValue(self.clock.sel_t0()); self.t1_sp.setValue(self.clock.sel_t1())

    def _whole(self):
        if self.rec is not None:
            self.t0_sp.setValue(self.rec.t_start_s); self.t1_sp.setValue(self.rec.t_stop_s)

    # ------------------------------------------------------------------ export
    def export(self, path: str | None = None):
        """Render the current shot at full size on a worker thread."""
        if self.rec is None or (self._worker is not None and self._worker.isRunning()):
            return None
        shot = self.shot()
        self._worker = _ExportWorker(self.rec, shot, path, self.folder_ed.text().strip() or None)
        self._worker.progress.connect(lambda f: self.progress.setValue(int(f * 100)))
        self._worker.finished_ok.connect(self._exported)
        self._worker.failed.connect(lambda m: self._done(f"Export failed: {m}"))
        self.progress.setValue(0); self.progress.setVisible(True)
        self.export_btn.setEnabled(False); self.cancel_btn.setEnabled(True)
        self._worker.start()
        return self._worker

    def _exported(self, facts):
        if facts is None:
            self._done("Export cancelled.")
            return
        mb = os.path.getsize(facts["path"]) / 1e6 if os.path.isfile(facts["path"]) else 0.0
        self.last_export = facts
        self._done(f"{facts['format'].upper()} {facts['size'][0]}×{facts['size'][1]}, "
                   f"{facts['frames']} frames, {facts['duration_s']:.1f} s, {mb:.1f} MB\n"
                   f"{facts['path']}")

    def _done(self, msg):
        self.progress.setVisible(False)
        self.export_btn.setEnabled(True); self.cancel_btn.setEnabled(False)
        self.export_lbl.setText(msg)

    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Export folder", self.folder_ed.text())
        if d:
            self.folder_ed.setText(d)

    def _open_folder(self):
        from gottlux.io.paths import open_in_file_browser
        folder = self.folder_ed.text().strip() or default_folder()
        os.makedirs(folder, exist_ok=True)
        open_in_file_browser(folder)

    def _copy_recipe(self):
        QtWidgets.QApplication.clipboard().setText(self.shot().to_json())
        self.export_lbl.setText("Recipe copied (JSON): paste it into a file, or use it with "
                                "gottlux-studio --recipe.")

    def _load_recipe(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load shot recipe", self.folder_ed.text(), "Shot recipes (*.json)")
        if not path:
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.set_shot(Shot.from_dict(data.get("shot", data)))

    def apply_theme(self, *_):
        self.preview.setStyleSheet(f"background:{style.BG2};")
