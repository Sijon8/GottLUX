"""
rangelab.py — the Range lab: draw a bounding box, keyframe it, solve pixels-on-target.

Workflow
--------
1. Scrub to a moment where the drone is visible and **drag the box** around it.
2. Enter the **known range** to the drone at that instant and press **Add keyframe** — the box,
   the time, and the range are stored. Repeat at a few moments (near and far).
3. Set the **target size** (m) and **FOV** (deg). The lab solves, live:
   * measured pixels-on-target (from the box) vs the pinhole-expected pixels at that range,
   * the implied physical size (calibration cross-check),
   * the IFOV and the ground-sample distance, and
   * the **maximum perception range** for detection / recognition / identification
     (Johnson's criteria) for the target size — i.e. *how far away the drone can be resolved*.
4. **Export study…** writes the paper-ready bundle (figure PDF+PNG, tables Parquet+CSV, a
   JSON summary, and a Markdown methods/results report).

With ``Follow keyframes`` on, the box is linearly interpolated between keyframes as you scrub,
so it tracks the drone for visual confirmation. Bound to the shared program clock.
"""
from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from gottlux.app import icons
from gottlux.app import style
from gottlux.core import photogrammetry as pg_solve
from gottlux.core import tonemap
from gottlux.core.accumulate import accumulate_frame
from gottlux import sensors
from gottlux.config import CAMERA_FOV_DEG          # per-camera FOV overrides (single source)

#: Default horizontal FOV when the recording's camera isn't a recognized override — the
#: active sensor profile's horizontal FOV (GenX320 + 1.8 mm ⇒ 58°).
_DEFAULT_FOV_DEG = sensors.get(sensors.DEFAULT_PROFILE).fov_horizontal_deg


class RangeLab(QtWidgets.QWidget):
    """Bounding-box keyframing + pixels-on-target / perception-range solver."""

    roiChanged = QtCore.Signal(object)        # panel-protocol parity (unused)

    def __init__(self, controller, filters=None, parent=None, compact=False):
        super().__init__(parent)
        self.ctl = controller
        self.filters = filters                 # shared live noise-filter suite (FilterController | None)
        self.rec = None
        self.keyframes = []                    # list[photogrammetry.Keyframe]
        self._follow = False
        # Side-by-side/multi-clip mode: this lab sits next to its twin and the two will be
        # *fused* (keyframes converged) from the slate's "Fuse keyframes" button, so the
        # per-pane converge hint and per-pane Export/Crop/Video controls are redundant noise.
        # Drop them and keep only what each sensor genuinely needs on its own: its FOV/size,
        # the box, the keyframe table, and the live solve.
        self._compact = bool(compact)

        # --- image + box ---
        self.glw = pg.GraphicsLayoutWidget()
        self.vb = self.glw.addViewBox(lockAspect=True, invertY=True)
        self.img = pg.ImageItem(axisOrder="row-major")
        self.vb.addItem(self.img)
        try:
            self.img.setColorMap(pg.colormap.get("inferno", source="matplotlib"))
        except Exception:
            pass
        self.box = pg.RectROI([40, 40], [40, 40], pen=pg.mkPen(style.ACCENT, width=2))
        self.box.addScaleHandle([1, 1], [0, 0]); self.box.addScaleHandle([0, 0], [1, 1])
        self.vb.addItem(self.box)
        self.box.sigRegionChanged.connect(self._on_box)

        # --- inputs ---
        self.fov = QtWidgets.QDoubleSpinBox(); self.fov.setRange(0.1, 180.0)
        self.fov.setDecimals(2); self.fov.setValue(_DEFAULT_FOV_DEG); self.fov.setSuffix(" °")
        self.fov.setToolTip("Horizontal field of view across the sensor width "
                            "(GenX320 + 1.8 mm rig ≈ 58°; 76° is its diagonal).")
        self.fov.valueChanged.connect(self._solve)
        self.size_m = QtWidgets.QDoubleSpinBox(); self.size_m.setRange(0.001, 100.0)
        self.size_m.setDecimals(3); self.size_m.setValue(0.30); self.size_m.setSuffix(" m")
        self.size_m.setToolTip("Target's critical (largest) physical dimension.")
        self.size_m.valueChanged.connect(self._solve)
        self.geom_lbl = QtWidgets.QLabel("sensor: —")
        self.geom_lbl.setObjectName("muted")
        self.converge_hint = QtWidgets.QLabel(
            "Two sensors? Load both in the <b>Multi-clip</b> tab, set each pane's View to "
            "<b>Range lab</b>, keyframe both, then <b>Fuse keyframes</b> there.")
        self.converge_hint.setObjectName("muted"); self.converge_hint.setWordWrap(True)

        # --- keyframe controls ---
        self.dist = QtWidgets.QDoubleSpinBox(); self.dist.setRange(0.0, 100000.0)
        self.dist.setDecimals(2); self.dist.setValue(0.0); self.dist.setSuffix(" m")
        self.dist.setToolTip("Known range to the drone at this moment (0 = unknown).")
        self.label = QtWidgets.QLineEdit(); self.label.setPlaceholderText("label (optional)")
        self.add_btn = QtWidgets.QPushButton("Add keyframe @ cursor")
        self.add_btn.setIcon(icons.icon("target"))
        self.add_btn.setToolTip("Store the current box + time + range as a keyframe.")
        self.add_btn.clicked.connect(self._add_keyframe)
        self.del_btn = QtWidgets.QPushButton("Delete selected")
        self.del_btn.setIcon(icons.icon("close"))
        self.del_btn.clicked.connect(self._del_keyframe)
        self.follow_chk = QtWidgets.QCheckBox("Follow keyframes (interpolate box)")
        self.follow_chk.toggled.connect(self._set_follow)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["t (s)", "w×h px", "range m", "px", "label"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.cellClicked.connect(self._on_row)
        self.table.setMinimumWidth(300)

        self.solve_lbl = QtWidgets.QLabel("—"); self.solve_lbl.setWordWrap(True)
        self.solve_lbl.setTextFormat(QtCore.Qt.RichText)
        self.export_btn = QtWidgets.QPushButton("Export study…")
        self.export_btn.setObjectName("primary")
        self.export_btn.setIcon(icons.icon("export", color="ACCENT_TEXT"))
        self.export_btn.setToolTip("Write the paper-ready bundle (figure PDF+PNG, tables, report).")
        self.export_btn.clicked.connect(self._export)
        self.crop_btn = QtWidgets.QPushButton("Crop box → .raw…")
        self.crop_btn.setIcon(icons.icon("cut"))
        self.crop_btn.setToolTip("Cut the current box region (spatial crop) of this view to a "
                                 "new, valid .raw clip.")
        self.crop_btn.clicked.connect(self._crop_raw)
        self.video_btn = QtWidgets.QPushButton("Analysis video…")
        self.video_btn.setIcon(icons.icon("film"))
        self.video_btn.setToolTip("Render an MP4 of the tracked box (interpolated across "
                                  "keyframes) with a time/range readout.")
        self.video_btn.clicked.connect(self._analysis_video)

        # --- layout: image on the left, controls on the right ---
        right = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()
        form.addRow("FOV", self.fov)
        form.addRow("Target size", self.size_m)
        right.addLayout(form)
        right.addWidget(self.geom_lbl)
        if not self._compact:
            right.addWidget(self.converge_hint)
        kf = QtWidgets.QGroupBox("Keyframe")
        kfl = QtWidgets.QFormLayout(kf)
        kfl.addRow("Range", self.dist)
        kfl.addRow("Label", self.label)
        kfl.addRow(self.add_btn)
        kfl.addRow(self.del_btn)
        kfl.addRow(self.follow_chk)
        right.addWidget(kf)
        right.addWidget(self.table, 1)
        solve_box = QtWidgets.QGroupBox("Solved")
        sbl = QtWidgets.QVBoxLayout(solve_box)
        sbl.addWidget(self.solve_lbl)
        right.addWidget(solve_box)
        if not self._compact:                  # per-pane export is redundant once panes are fused
            right.addWidget(self.export_btn)
            exrow = QtWidgets.QHBoxLayout()
            exrow.addWidget(self.crop_btn); exrow.addWidget(self.video_btn)
            right.addLayout(exrow)
        rw = QtWidgets.QWidget(); rw.setLayout(right)
        self.glw.setMinimumSize(220, 180)

        from gottlux.app.transport import TransportBar
        from gottlux.app.uikit import plot_with_deck
        lay = QtWidgets.QVBoxLayout(self)
        # A draggable, scroll-wrapped control deck (not a fixed 330 px column) so the lab is
        # usable on a laptop and in the split view, and roomy on a large display.
        lay.addWidget(plot_with_deck(self.glw, rw, min_deck=288, init_deck=320), 1)
        self.transport = TransportBar(self.ctl, host=self)
        lay.addWidget(self.transport)

        self.ctl.cursorChanged.connect(self._render)
        self.ctl.accumChanged.connect(self._render)
        if self.filters is not None:
            self.filters.changed.connect(self._render)

    # ------------------------------------------------------------------ data
    def set_recording(self, rec):
        self.rec = rec
        self.vb.setRange(xRange=(0, rec.width), yRange=(0, rec.height), padding=0)
        self.geom_lbl.setText(f"sensor: {rec.width}×{rec.height} px  ·  {rec.fmt}")
        # a sensible default FOV: a recognized per-camera override from the file name, else
        # the active sensor profile's horizontal FOV.
        fov = next((f for cam, f in CAMERA_FOV_DEG.items() if cam in (rec.name or "").lower()),
                   _DEFAULT_FOV_DEG)
        b = QtCore.QSignalBlocker(self.fov); self.fov.setValue(fov); del b
        self._render(); self._solve()

    def sync(self):
        self._render(force=True); self._solve()

    def _set_follow(self, on):
        self._follow = bool(on)
        self._render()

    def showEvent(self, ev):
        super().showEvent(ev); self._render()

    # ------------------------------------------------------------------ render
    def _render(self, *_, force=False):
        if self.rec is None or (not force and not self.isVisible()):
            return
        if self._follow and len(self.keyframes) >= 1:
            self._apply_box(self.box_at(self.ctl.cursor))
        from gottlux.core.render import render_frame
        disp, levels, _v, _w = render_frame(self.rec, self.ctl.cursor, self.ctl.accum,
                                            mode="count", expr="sqrt", filters=self.filters,
                                            back=self.ctl.accum_back)
        self.img.setImage(disp, levels=levels, autoLevels=False)

    # ------------------------------------------------------------------ box
    def _cur_bbox(self):
        pos = self.box.pos(); size = self.box.size()
        x0, y0 = float(pos.x()), float(pos.y())
        return (x0, y0, x0 + float(size.x()), y0 + float(size.y()))

    def _apply_box(self, bbox):
        if bbox is None:
            return
        x0, y0, x1, y1 = bbox
        b = QtCore.QSignalBlocker(self.box)
        self.box.setPos([x0, y0]); self.box.setSize([max(x1 - x0, 1), max(y1 - y0, 1)])
        del b

    def _on_box(self, *_):
        self._solve()

    # ------------------------------------------------------------------ faithful capture
    def sensor_size(self):
        return (self.rec.width, self.rec.height) if self.rec is not None else None

    def capture_frame(self, t, dt=None, size=None):
        """Faithful RGB of the Range lab at time *t* — the event frame + the tracked box — at any
        resolution (for the high-res Capture)."""
        if self.rec is None:
            return None
        from gottlux.core.render import render_frame
        from gottlux.viz import video as _vid
        dt = self.ctl.accum if dt is None else float(dt)
        disp, levels, _v, _w = render_frame(self.rec, t, dt, mode="count", expr="sqrt",
                                            filters=self.filters, back=self.ctl.accum_back)
        rgb = _vid.disp_to_rgb(disp, levels, "inferno")
        box = self.box_at(t) if (self._follow and self.keyframes) else self._cur_bbox()
        _vid.draw_box(rgb, box, color=(57, 197, 207), width=2)
        return _vid.resize_rgb(rgb, size) if size else rgb

    def box_at(self, t):
        """Linearly-interpolated bounding box at time *t* from the stored keyframes."""
        if not self.keyframes:
            return None
        ks = sorted(self.keyframes, key=lambda k: k.t_s)
        if len(ks) == 1:
            return ks[0].bbox
        ts = np.array([k.t_s for k in ks])
        t = min(max(t, ts[0]), ts[-1])
        return tuple(float(np.interp(t, ts, [k.bbox[i] for k in ks])) for i in range(4))

    # ------------------------------------------------------------------ keyframes
    def _add_keyframe(self):
        if self.rec is None:
            return
        d = self.dist.value()
        kf = pg_solve.Keyframe(t_s=float(self.ctl.cursor), bbox=self._cur_bbox(),
                               distance_m=(d if d > 0 else None), label=self.label.text().strip())
        self.keyframes.append(kf)
        self._refresh_table(); self._solve()

    def _del_keyframe(self):
        row = self.table.currentRow()
        if 0 <= row < len(self._sorted()):
            kf = self._sorted()[row]
            self.keyframes.remove(kf)
            self._refresh_table(); self._solve()

    def _sorted(self):
        return sorted(self.keyframes, key=lambda k: k.t_s)

    def _on_row(self, row, _col):
        ks = self._sorted()
        if 0 <= row < len(ks):
            kf = ks[row]
            self._apply_box(kf.bbox)
            self.ctl.set_cursor(kf.t_s)
            if kf.distance_m:
                b = QtCore.QSignalBlocker(self.dist); self.dist.setValue(kf.distance_m); del b

    def _refresh_table(self):
        ks = self._sorted()
        self.table.setRowCount(len(ks))
        for r, k in enumerate(ks):
            meas = max(k.w_px, k.h_px)
            vals = [f"{k.t_s:.3f}", f"{k.w_px:.0f}×{k.h_px:.0f}",
                    "" if k.distance_m is None else f"{k.distance_m:.1f}",
                    f"{meas:.0f}", k.label]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(v))

    # ------------------------------------------------------------------ solve
    def _study(self):
        w = self.rec.width if self.rec else 320
        h = self.rec.height if self.rec else 320
        return pg_solve.ResolutionStudy(target_size_m=self.size_m.value(), fov_deg=self.fov.value(),
                                        width_px=int(w), height_px=int(h),
                                        keyframes=list(self.keyframes))

    def _solve(self, *_):
        if self.rec is None:
            return
        fov, W, H = self.fov.value(), self.rec.width, self.rec.height
        L = self.size_m.value()
        bbox = self._cur_bbox()
        meas = max(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]))
        ifov = pg_solve.ifov_mrad(fov, W)
        ang = pg_solve.angular_size_deg(meas, fov, W)
        d = self.dist.value()
        parts = [f"<b>current box</b>: {meas:.0f} px across · {ang:.3f}° · IFOV {ifov:.3f} mrad/px"]
        if d > 0:
            exp = float(pg_solve.pixels_on_target(L, d, fov, W))
            implied = float(pg_solve.size_from_pixels(meas, d, fov, W))
            gsd = float(pg_solve.gsd_m(d, fov, W))
            parts.append(f"at {d:.1f} m: expected <b>{exp:.1f}</b> px (measured {meas:.0f}); "
                         f"implied size <b>{implied:.3f} m</b>; GSD {gsd:.3f} m/px")
        study = self._study()
        fit = study.fit_target_size()
        L_used = fit.get("fitted_target_size_m") if fit.get("n", 0) else L
        if fit.get("n", 0):
            r2 = f", R²={fit['r2']}" if fit.get("r2") is not None else ""
            parts.append(f"<b>fit</b> (n={fit['n']}): L_fit={fit.get('fitted_target_size_m')} m{r2}")
        rng = pg_solve.perception_ranges(L_used, fov, W)
        parts.append("<b>max perception range</b> (L={:.3f} m): ".format(L_used)
                     + " · ".join(f"{k} {v:.0f} m" for k, v in rng.items()))
        self.solve_lbl.setText("<br>".join(parts))

    # ------------------------------------------------------------------ export
    def _export(self):
        if not self.keyframes:
            QtWidgets.QMessageBox.information(self, "Export study",
                                             "Add at least one keyframe first.")
            return
        parent = QtWidgets.QFileDialog.getExistingDirectory(self, "Export resolution study to folder")
        if not parent:
            return
        from gottlux.run.resolution_report import save_resolution_study
        from gottlux.io.paths import open_in_file_browser, unique_export_dir
        d = unique_export_dir(parent, (self.rec.name if self.rec else "study"), "rangelab")
        base = os.path.join(d, (self.rec.name if self.rec else "study"))
        try:
            written = save_resolution_study(
                base, self._study(), title=f"Pixels on target — {self.rec.name if self.rec else ''}",
                profile=sensors.get(sensors.DEFAULT_PROFILE))
            open_in_file_browser(d)
            QtWidgets.QMessageBox.information(
                self, "Export study", "Wrote:\n" + "\n".join(os.path.basename(p) for p in written))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export study", str(e))

    # ------------------------------------------------------------------ crop / video
    def _crop_raw(self):
        """Cut the current box region (a spatial crop) of this view to a new valid .raw."""
        if self.rec is None:
            return
        x0, y0, x1, y1 = self._cur_bbox()
        roi = (int(min(x0, x1)), int(min(y0, y1)), int(max(x0, x1)), int(max(y0, y1)))
        base = (os.path.splitext(self.rec.source_path)[0] if self.rec.source_path
                else os.path.join(os.getcwd(), self.rec.name))
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Crop box → .raw", f"{base}_crop.raw", "EVT raw (*.raw)")
        if not out:
            return
        if not out.lower().endswith(".raw"):
            out += ".raw"
        from gottlux.io import writer
        from gottlux.io.paths import open_in_file_browser
        try:
            n = writer.cut_clip(self.rec, out, roi=roi)
            open_in_file_browser(os.path.dirname(os.path.abspath(out)))
            QtWidgets.QMessageBox.information(self, "Crop → .raw",
                                             f"Wrote {n:,} events (ROI {roi}) →\n{out}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Crop → .raw", str(e))

    def _analysis_video(self):
        """Render an MP4 of the tracked box (interpolated across keyframes) + a time/range readout."""
        if self.rec is None:
            return
        if self.keyframes:
            ts = [k.t_s for k in self.keyframes]
            t0, t1 = min(ts), max(ts)
            if t1 - t0 < 0.05:
                t0, t1 = self.rec.t_start_s, self.rec.t_stop_s
            box_at = self.box_at
        else:                                       # no keyframes → the static current box
            t0, t1 = self.rec.t_start_s, self.rec.t_stop_s
            cur = self._cur_bbox()
            box_at = lambda _t: cur
        base = (os.path.splitext(self.rec.source_path)[0] if self.rec.source_path
                else os.path.join(os.getcwd(), self.rec.name))
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Analysis video", f"{base}_analysis.mp4", "MP4 video (*.mp4)")
        if not out:
            return
        fov, W, L = self.fov.value(), self.rec.width, self.size_m.value()

        def label(t):
            bb = box_at(t)
            if bb is None:
                return f"t={t:.2f}s"
            meas = max(abs(bb[2] - bb[0]), abs(bb[3] - bb[1]))
            rng = float(pg_solve.estimate_range_m(meas, fov, L, W))
            return f"t={t:.2f}s  {meas:.0f}px" + (f"  ~{rng:.1f}m" if rng == rng else "")

        from gottlux.viz.video import render_box_track_video
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            path = render_box_track_video(self.rec, out, box_at, t0, t1,
                                          accum_dt=max(self.ctl.accum, 0.01), fps=25,
                                          filters=self.filters, label_fn=label)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if path:
            from gottlux.io.paths import open_in_file_browser
            open_in_file_browser(os.path.dirname(os.path.abspath(path)))
            QtWidgets.QMessageBox.information(self, "Analysis video", f"Saved:\n{path}")
        else:
            from gottlux.viz.video import ffmpeg_available
            msg = ("Video export needs the FFMPEG muxer, which isn't available.\n\n"
                   "Install it with:\n    pip install imageio-ffmpeg") if not ffmpeg_available() \
                else "Video export failed while encoding (see the console log for the FFMPEG error)."
            QtWidgets.QMessageBox.warning(self, "Analysis video", msg)
