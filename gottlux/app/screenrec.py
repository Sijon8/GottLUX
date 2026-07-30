"""
screenrec.py — a live screen / view recorder, like the Windows Snipping Tool's screen record.

Drag a rectangle anywhere on the display (snipping-tool style) and record it straight to an MP4,
or record **the current view** / **the whole window** with one click. Recording the *screen
region* a view occupies captures exactly what you see — the GL scene, the colour key, the nav
cube, every tuned setting — pixel-for-pixel, with none of the framebuffer caveats a widget grab
has. So this doubles as the most faithful "save what's on my screen" capture.

The pieces:

* :class:`RegionOverlay`   — a translucent full-desktop overlay to drag-select a region.
* :class:`RecorderHUD`     — a small always-on-top "● 00:03 · 91f / Stop" bar shown while recording.
* :class:`ScreenRecorder`  — a :class:`~PySide6.QtCore.QTimer`-driven loop that grabs frames and
                             streams them to a :class:`gottlux.viz.video.VideoWriter`.
* :class:`ScreenRecordDialog` — the small setup dialog (target · fps · cursor · output).

The frame encoding lives in :mod:`gottlux.viz.video`; only the pixel grab + the Qt chrome live here.
"""
from __future__ import annotations

import os
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from gottlux.app import icons
from gottlux.app import style
from gottlux.app.capture import qimage_to_rgb
from gottlux.viz.video import VideoWriter

_ACCENT = QtGui.QColor(0, 229, 255)


# --------------------------------------------------------------------- pixel grab
def grab_region_rgb(rect: QtCore.QRect, cursor: bool = False) -> "np.ndarray | None":
    """Grab a global-screen *rect* to an ``(H, W, 3)`` uint8 RGB array (device pixels).

    Grabs the **whole** screen the rectangle's centre lands on and crops in device pixels, rather
    than asking ``grabWindow`` for a sub-rectangle (whose offset is interpreted inconsistently
    across platforms/monitors and was a source of black/torn frames). DPI-correct. Optionally
    composites a marker at the live cursor position."""
    if rect is None or rect.width() < 2 or rect.height() < 2:
        return None
    screen = QtGui.QGuiApplication.screenAt(rect.center()) or QtWidgets.QApplication.primaryScreen()
    if screen is None:
        return None
    pm = screen.grabWindow(0)                                       # the whole screen (robust)
    if pm.isNull():
        return None
    img = pm.toImage()
    dpr = float(screen.devicePixelRatio() or 1.0)
    loc = rect.translated(-screen.geometry().topLeft())            # screen-local, logical px
    x, y = int(round(loc.x() * dpr)), int(round(loc.y() * dpr))
    w, h = int(round(loc.width() * dpr)), int(round(loc.height() * dpr))
    x = max(0, min(x, img.width() - 2)); y = max(0, min(y, img.height() - 2))
    w = max(2, min(w, img.width() - x)); h = max(2, min(h, img.height() - y))
    rgb = qimage_to_rgb(img.copy(x, y, w, h))
    if cursor:
        _draw_cursor(rgb, rect, screen)
    return rgb


def screen_of(widget) -> QtCore.QRect:
    """The full global geometry of the screen a widget currently sits on (for full-screen capture)."""
    handle = widget.screen() if widget is not None and hasattr(widget, "screen") else None
    handle = handle or QtWidgets.QApplication.primaryScreen()
    return handle.geometry()


def _draw_cursor(rgb, rect, screen):
    """Composite a small cursor dot into *rgb* (device pixels) at the live pointer position."""
    try:
        dpr = float(screen.devicePixelRatio())
        gp = QtGui.QCursor.pos()
        cx = int(round((gp.x() - rect.x()) * dpr))
        cy = int(round((gp.y() - rect.y()) * dpr))
        h, w = rgb.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return
        r = max(4, int(round(5 * dpr)))
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = xx * xx + yy * yy <= r * r
        ring = (xx * xx + yy * yy <= r * r) & (xx * xx + yy * yy >= (r - max(1, r // 3)) ** 2)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        sub = rgb[y0:y1, x0:x1]
        dm = disk[y0 - (cy - r):y1 - (cy - r), x0 - (cx - r):x1 - (cx - r)]
        rm = ring[y0 - (cy - r):y1 - (cy - r), x0 - (cx - r):x1 - (cx - r)]
        sub[dm] = (255, 255, 255)
        sub[rm] = (10, 10, 10)
    except Exception:
        pass


def widget_global_rect(widget) -> QtCore.QRect:
    """The on-screen rectangle a widget currently occupies, in global coordinates."""
    return QtCore.QRect(widget.mapToGlobal(QtCore.QPoint(0, 0)), widget.size())


# Output-resolution presets for the recorder. A screen recorder can only show on-screen pixels, so
# the taller presets *upscale* (smoothly) — a larger, presentation-ready file — and never downscale
# below what was captured. For pixel-exact output choose "On-screen".
RES_PRESETS = ["On-screen", "720p", "1080p", "1440p", "2160p (4K)", "2×", "3×"]
_RES_DEFAULT = "1080p"


def make_scaler(choice):
    """Build a ``frame → frame`` upscaler for a :data:`RES_PRESETS` label, or ``None`` (on-screen)."""
    if not choice or choice.startswith("On-screen"):
        return None
    from gottlux.viz.video import resize_rgb

    def scaler(rgb):
        h, w = rgb.shape[:2]
        if h < 1 or w < 1:
            return rgb
        if choice.endswith("×"):                       # integer super-sampling (always applies)
            f = int(choice[0])
            return resize_rgb(rgb, (w * f, h * f), smooth=True)
        th = int(choice.split("p")[0])                 # "1080p" / "2160p (4K)" → target height
        if th <= h:
            return rgb                                  # never downscale below the captured pixels
        return resize_rgb(rgb, (max(2, round(w * th / h)), th), smooth=True)

    return scaler


# --------------------------------------------------------------------- region selection
class RegionOverlay(QtWidgets.QWidget):
    """A translucent overlay across the whole virtual desktop; drag to pick a region.

    Emits :attr:`selected` with the chosen **global** ``QRect`` (or ``None`` if cancelled/Esc)."""

    selected = QtCore.Signal(object)

    def __init__(self):
        super().__init__(None, QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
                         | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setCursor(QtCore.Qt.CrossCursor)
        scr = QtWidgets.QApplication.primaryScreen()
        self.setGeometry(scr.virtualGeometry() if scr else QtCore.QRect(0, 0, 1920, 1080))
        self._origin = None
        self._cur = None

    def _rect(self):
        if self._origin is None or self._cur is None:
            return None
        return QtCore.QRect(self._origin, self._cur).normalized()

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 96))
        r = self._rect()
        if r is not None and r.width() > 0:
            p.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
            p.fillRect(r, QtCore.Qt.transparent)                   # punch a live hole to aim through
            p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
            p.setPen(QtGui.QPen(_ACCENT, 2)); p.setBrush(QtCore.Qt.NoBrush); p.drawRect(r)
            p.setPen(QtGui.QColor(225, 245, 255))
            p.drawText(r.topLeft() + QtCore.QPoint(2, -6), f"{r.width()}×{r.height()}")
        else:
            p.setPen(QtGui.QColor(225, 245, 255))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter,
                       "Drag a rectangle to record  ·  Esc to cancel")

    def mousePressEvent(self, e):
        self._origin = e.position().toPoint(); self._cur = self._origin; self.update()

    def mouseMoveEvent(self, e):
        if self._origin is not None:
            self._cur = e.position().toPoint(); self.update()

    def mouseReleaseEvent(self, e):
        r = self._rect()
        g = r.translated(self.geometry().topLeft()) if (r and r.width() > 8 and r.height() > 8) else None
        self.selected.emit(g)
        self.close()

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.selected.emit(None); self.close()


# --------------------------------------------------------------------- recording HUD
class RecorderHUD(QtWidgets.QWidget):
    """A small always-on-top ``● mm:ss · Nf  [■ Stop]`` bar shown while recording."""

    stopped = QtCore.Signal()

    def __init__(self):
        super().__init__(None, QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
                         | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        wrap = QtWidgets.QHBoxLayout(self); wrap.setContentsMargins(0, 0, 0, 0)
        box = QtWidgets.QFrame(); box.setObjectName("hud")
        box.setStyleSheet(f"#hud{{background:{style.BG2};border:1px solid {style.ACCENT};"
                          f"border-radius:8px;}}"
                          f"QLabel{{color:{style.FG};}} QPushButton{{padding:3px 10px;}}")
        row = QtWidgets.QHBoxLayout(box); row.setContentsMargins(10, 6, 8, 6); row.setSpacing(8)
        # the blink dot is a *painted* circle rendered at the device pixel ratio (not a font
        # glyph, which lands in a fallback emoji font and blurs on fractional DPI)
        self._dot = QtWidgets.QLabel()
        dot_sz = QtCore.QSize(12, 12)
        # on = the palette's alarm red, off = the same hue dimmed into the HUD's plate
        self._dot_on_pm = icons.icon("record", color=style.BAD).pixmap(dot_sz)
        self._dot_off_pm = icons.icon("record",
                                      color=QtGui.QColor(style.BAD).darker(300)).pixmap(dot_sz)
        self._dot.setPixmap(self._dot_on_pm)
        self._lbl = QtWidgets.QLabel("00:00 · 0f"); self._lbl.setMinimumWidth(96)
        self._lbl.setStyleSheet("font-family:Consolas,'DejaVu Sans Mono',monospace;")
        stop = QtWidgets.QPushButton("Stop"); stop.setIcon(icons.icon("stop"))
        stop.clicked.connect(self.stopped)
        row.addWidget(self._dot); row.addWidget(self._lbl); row.addWidget(stop)
        wrap.addWidget(box)
        self._blink = QtCore.QTimer(self); self._blink.setInterval(550)
        self._blink.timeout.connect(self._toggle_dot); self._blink.start()
        self._on = True

    def _toggle_dot(self):
        self._on = not self._on
        self._dot.setPixmap(self._dot_on_pm if self._on else self._dot_off_pm)

    def set_status(self, secs, frames):
        self._lbl.setText(f"{int(secs) // 60:02d}:{int(secs) % 60:02d} · {frames}f")

    def closeEvent(self, ev):
        # Closing the HUD by any route (window close, app shutdown) finalises the recording too.
        self.stopped.emit()
        super().closeEvent(ev)

    def place_clear_of(self, rect: QtCore.QRect):
        """Position the HUD just outside *rect* (so it isn't recorded), clamped to the desktop."""
        self.adjustSize()
        scr = QtGui.QGuiApplication.screenAt(rect.center()) or QtWidgets.QApplication.primaryScreen()
        avail = scr.availableGeometry() if scr else QtCore.QRect(0, 0, 1920, 1080)
        x = rect.left()
        y = rect.bottom() + 8                              # prefer just below the recorded area
        if y + self.height() > avail.bottom():
            y = rect.top() - self.height() - 8             # else just above
        if y < avail.top():
            y = avail.top() + 8                            # else pinned to the top edge
        x = max(avail.left(), min(x, avail.right() - self.width()))
        self.move(x, y)


# --------------------------------------------------------------------- recorder loop
class ScreenRecorder(QtCore.QObject):
    """Grab ``grab_fn()`` → RGB every ``1/fps`` s and stream it to an MP4 until :meth:`stop`.

    The grab runs on the GUI thread (Qt screen grabs must), but **encoding runs on a worker
    thread** so a slow frame can't stutter the UI being recorded. Call :meth:`prime` with the
    first frame before :meth:`start` so the (slow) one-time FFMPEG startup is paid up front.

    **Wall-clock pacing.** The GUI thread is busy painting the app being recorded, so the grab
    timer jitters and rarely hits the requested fps exactly. Rather than write one frame per tick
    (which makes a "30 fps" file that actually holds ~22 frames/s play ~35 % fast — the "it
    dropped the video" symptom), each tick writes however many frames are needed to keep the
    file's frame count equal to ``elapsed × fps`` — duplicating the latest grab to fill a missed
    interval. The result plays back at true real-time regardless of grab jitter.

    *scale* is an optional ``frame → frame`` callable (e.g. an upscaler) applied before encoding.
    """

    def __init__(self, grab_fn, out_path, fps=30, scale=None, clock=None, parent=None):
        super().__init__(parent)
        import queue
        import threading
        self._grab = grab_fn
        self._scale = scale
        self._clock = clock or time.perf_counter      # injectable for deterministic tests
        self._fps = float(max(fps, 1))
        self._writer = VideoWriter(out_path, fps=fps)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(max(1, int(round(1000.0 / self._fps))))
        self._timer.timeout.connect(self._tick)
        self._q = queue.Queue(maxsize=120)         # bounded → drop, never balloon memory
        self._queue_empty = queue.Empty
        self._stop_evt = threading.Event()
        self._thread_cls = threading.Thread
        self._worker = None
        self._t0 = None
        self._enqueued = 0                         # frames scheduled to the writer (for pacing)
        self._result = None                        # path the worker stored when it closed the file
        self.dropped = 0

    @property
    def ok(self) -> bool:
        return self._writer.ok

    @property
    def frames(self) -> int:
        return self._writer.frames

    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else (self._clock() - self._t0)

    def _apply_scale(self, rgb):
        if self._scale is None or rgb is None:
            return rgb
        try:
            return self._scale(rgb)
        except Exception:
            return rgb

    def prime(self, frame):
        """Encode the first frame synchronously so FFMPEG's startup cost is paid before the clock
        starts — this is what removes the start-of-recording hiccup and the time-jump."""
        if self._writer.append(self._apply_scale(frame)):
            self._enqueued = 1                     # the primed frame covers t=0 in the pacing

    def start(self):
        self._t0 = self._clock()
        self._worker = self._thread_cls(target=self._drain, daemon=True)
        self._worker.start()
        self._timer.start()

    def _tick(self):
        if self._t0 is None:
            return
        target = int(round(self.elapsed() * self._fps))   # frames we *should* have written by now
        repeat = target - self._enqueued
        if repeat <= 0:                            # on/ahead of schedule — nothing to write yet
            return
        repeat = min(repeat, int(2 * self._fps))   # cap a catch-up burst (e.g. after a stall)
        try:
            rgb = self._grab()
        except Exception:
            rgb = None
        if rgb is None:
            return
        try:
            self._q.put_nowait((rgb, repeat))      # encoder writes the frame `repeat` times
            self._enqueued += repeat
        except Exception:
            self.dropped += repeat                 # encoder badly behind → drop (rare, fast preset)

    def _drain(self):
        """Worker thread: pull (frame, repeat) items and encode them (off the GUI thread), then
        close the file **here** so the writer is only ever touched by one thread — no cross-thread
        close race (the bug that left recordings unsaved/corrupt under load)."""
        while not (self._stop_evt.is_set() and self._q.empty()):
            try:
                rgb, repeat = self._q.get(timeout=0.1)
            except self._queue_empty:
                continue
            try:
                frame = self._apply_scale(rgb)
                for _ in range(max(1, int(repeat))):
                    self._writer.append(frame)
            finally:
                self._q.task_done()
        self._result = self._writer.close()

    def stop(self):
        """Stop and finalise. Waits for the encoder to flush its backlog and close the file (the
        flush must finish for the MP4 to be valid), so this returns the saved path or ``None``."""
        self._timer.stop()
        self._stop_evt.set()
        if self._worker is not None:
            self._worker.join()                    # full join — the backlog must be encoded to save
        else:                                      # never started (e.g. only the primed frame)
            self._result = self._writer.close()
        return self._result


# --------------------------------------------------------------------- setup dialog
# "Entire app window" is the default: it captures the menus, toolbars, control decks (every
# variable) and the active view together, and follows you across tab switches.
_TARGETS = ["Entire app window", "Full screen (desktop)", "Screen region (drag to select)",
            "Current view"]


def _unique_path(path):
    """A path that doesn't clobber an existing file (append _1, _2, … ) — so a second recording
    never silently overwrites the first."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{stem}_{i}{ext}"):
        i += 1
    return f"{stem}_{i}{ext}"


def _xdg_videos_dir(config_path=None):
    """The user's XDG videos directory from ``~/.config/user-dirs.dirs`` (Linux), or None.

    Parses the ``XDG_VIDEOS_DIR="$HOME/Videos"`` line, expanding ``$HOME``. Relative or
    missing entries return None (the freedesktop spec says relative means "disabled")."""
    if config_path is None:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"),
                                                                 ".config")
        config_path = os.path.join(base, "user-dirs.dirs")
    try:
        with open(config_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("XDG_VIDEOS_DIR"):
                    val = line.partition("=")[2].strip().strip('"')
                    val = val.replace("$HOME", os.path.expanduser("~"))
                    # user-dirs.dirs is a POSIX format — accept POSIX-absolute too, so the
                    # parser stays testable on Windows.
                    import posixpath
                    return val if (posixpath.isabs(val) or os.path.isabs(val)) else None
    except OSError:
        pass
    return None


def _safe_default_output(default_dir=None):
    """A writable default output path. Prefers the user's Videos folder over the project dir, which
    often lives in a cloud-synced tree where files can be transiently locked."""
    base = default_dir if (default_dir and os.path.isdir(default_dir)) else None
    if base is None:
        import tempfile
        cands = [] if os.name == "nt" else [_xdg_videos_dir()]
        cands += [os.path.join(os.path.expanduser("~"), "Videos"),
                  os.path.join(os.path.expanduser("~"), "Movies")]
        for cand in cands:
            if cand and os.path.isdir(cand):
                base = cand
                break
        else:
            base = tempfile.gettempdir()
    return _unique_path(os.path.join(base, "gottlux_screenrec.mp4"))


class ScreenRecordDialog(QtWidgets.QDialog):
    """Set up a screen recording: pick a target, fps, cursor, and output, then record live.

    *view_widget* is the active tab's view area (so "Current view" records exactly it). *window*
    defaults to the dialog's parent window. *default_dir* seeds the output path."""

    def __init__(self, parent, view_widget=None, window=None, default_dir=None):
        super().__init__(parent)
        self.setWindowTitle("Screen record — live capture to MP4")
        self.setMinimumWidth(520)
        self._view = view_widget
        self._win = window or (parent.window() if parent is not None else None)
        self._rec = None
        self._hud = None
        self._region = None
        self._status = QtCore.QTimer(self); self._status.setInterval(200)
        self._status.timeout.connect(self._refresh_status)

        self.target = QtWidgets.QComboBox(); self.target.addItems(_TARGETS)
        if self._view is None:
            self.target.model().item(3).setEnabled(False)        # no single view to record
        self.target.setToolTip(
            "What to record:\n• Entire app window — the whole GottLUX window: menus, toolbars, "
            "control decks and the active view, across tab switches (the full environment).\n"
            "• Full screen — the entire monitor.\n• Screen region — drag any rectangle "
            "(snipping-tool style).\n• Current view — only the active tab's view area.")
        self.target.currentIndexChanged.connect(self._on_target_changed)
        self.res = QtWidgets.QComboBox(); self.res.addItems(RES_PRESETS)
        self.res.setCurrentText("On-screen")
        self.res.setToolTip(
            "Output resolution. 'On-screen' records the pixels 1:1 (best for the whole window / "
            "screen, which are already large — upscaling them only slows encoding). The taller "
            "presets upscale a *small* view to a presentation-ready file; '2×/3×' super-sample.")
        self.fps = QtWidgets.QSpinBox(); self.fps.setRange(5, 60); self.fps.setValue(30)
        self.fps.setSuffix(" fps")
        self.fps.setToolTip("Capture frame rate. The clip plays in real time (the recorder paces to "
                            "the wall clock, so jitter won't speed it up). Lower it if a large, "
                            "high-resolution capture can't keep up.")
        self.cursor = QtWidgets.QCheckBox("Capture cursor"); self.cursor.setChecked(True)
        self.cursor.setToolTip("Composite a marker at the pointer (region/view/window all show it).")
        self.out_edit = QtWidgets.QLineEdit(_safe_default_output(default_dir))
        browse = QtWidgets.QPushButton("Browse…"); browse.clicked.connect(self._browse)

        form = QtWidgets.QFormLayout()
        form.addRow("Record", self.target)
        opts = QtWidgets.QHBoxLayout()
        opts.addWidget(QtWidgets.QLabel("Resolution")); opts.addWidget(self.res)
        opts.addSpacing(10); opts.addWidget(self.fps); opts.addWidget(self.cursor)
        opts.addStretch(1)
        form.addRow("Options", opts)
        orow = QtWidgets.QHBoxLayout(); orow.addWidget(self.out_edit, 1); orow.addWidget(browse)

        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(form)
        v.addWidget(QtWidgets.QLabel("Output (.mp4):")); v.addLayout(orow)
        hint = QtWidgets.QLabel("Tip: 'Entire app window' captures the whole environment — menus, "
                                "controls and the view — and keeps recording as you switch tabs.")
        hint.setObjectName("muted"); hint.setWordWrap(True); v.addWidget(hint)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        self._rec_btn = bb.addButton("Record", QtWidgets.QDialogButtonBox.AcceptRole)
        self._rec_btn.setIcon(icons.icon("record"))
        self._rec_btn.clicked.connect(self._begin); bb.rejected.connect(self.reject)
        v.addWidget(bb)

    # ----- helpers -----
    def _browse(self):
        out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Screen recording output",
                                                       self.out_edit.text(), "MP4 video (*.mp4)")
        if out:
            self.out_edit.setText(out)

    def _out_path(self):
        out = self.out_edit.text().strip()
        if out and not out.lower().endswith(".mp4"):
            out += ".mp4"
        return out

    # ----- start -----
    def _begin(self):
        out = self._out_path()
        if not out:
            QtWidgets.QMessageBox.warning(self, "Screen record", "Choose an output path."); return
        from gottlux.viz.video import ffmpeg_available
        if not ffmpeg_available():
            QtWidgets.QMessageBox.warning(
                self, "Screen record",
                "Video export needs the FFMPEG muxer, which isn't available.\n\n"
                "Install it with:\n    pip install imageio-ffmpeg")
            return
        self._out = out
        choice = self.target.currentText()
        cur = self.cursor.isChecked()
        if choice.startswith("Screen region"):
            # End this dialog's modality *before* showing the overlay — an application-modal dialog
            # would otherwise swallow the overlay's mouse events. The dialog stays alive (parented)
            # to host the recorder; it's re-shown only if the user cancels the region pick.
            self.accept()
            self._overlay = RegionOverlay()
            self._overlay.selected.connect(self._region_picked)
            self._overlay.showFullScreen()
        elif choice.startswith("Current view") and self._view is not None:
            view = self._view
            self._start(lambda: grab_region_rgb(widget_global_rect(view), cur),
                        widget_global_rect(view))
        elif choice.startswith("Full screen"):
            rect = screen_of(self._win)
            self._start(lambda: grab_region_rgb(rect, cur), rect)
        else:                                                    # Entire app window (default)
            win = self._win
            self._start(lambda: grab_region_rgb(win.frameGeometry(), cur), win.frameGeometry())

    def _on_target_changed(self, *_):
        # whole-window / full-screen / region are already large → native; a single small view → 1080p
        view_only = self.target.currentText().startswith("Current view")
        self.res.setCurrentText("1080p" if view_only else "On-screen")

    def _region_picked(self, rect):
        if rect is None:                                         # cancelled — re-show the setup
            self.show(); return
        self._region = QtCore.QRect(rect)
        self._start(lambda: grab_region_rgb(self._region, self.cursor.isChecked()), self._region)

    def _start(self, grab_fn, rect):
        # sanity: one grab before we commit, so a bad target fails loudly instead of writing nothing
        probe = grab_fn()
        if probe is None:
            QtWidgets.QMessageBox.warning(self, "Screen record",
                                          "Couldn't grab that target. Make sure it's on-screen.")
            self.show(); return
        scale = make_scaler(self.res.currentText())
        self._rec = ScreenRecorder(grab_fn, self._out, fps=self.fps.value(), scale=scale, parent=self)
        if not self._rec.ok:
            QtWidgets.QMessageBox.warning(self, "Screen record",
                                          "Couldn't open the video writer (see console).")
            self.show(); return
        # Pay FFMPEG's one-time startup now (encode the first frame) so the live loop and the
        # elapsed clock start clean — no hiccup, no jump-to-six-seconds.
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self._rec.prime(probe)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.accept()                                            # close the setup dialog
        self._finished = False
        self._hud = RecorderHUD()
        self._hud.stopped.connect(self._finish)
        self._hud.place_clear_of(rect)
        self._hud.show()
        self._rec.start()
        self._status.start()
        # Safety net: if the app quits while recording, still flush + finalise the file.
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._finish)

    def _refresh_status(self):
        if self._rec is not None and self._hud is not None:
            self._hud.set_status(self._rec.elapsed(), self._rec.frames)

    def _finish(self):
        # Idempotent: the HUD close, the Stop button, and the app-quit safety net can all fire this.
        if getattr(self, "_finished", False):
            return
        self._finished = True
        self._status.stop()
        rec, self._rec = self._rec, None
        frames = rec.frames if rec is not None else 0
        elapsed = rec.elapsed() if rec is not None else 0.0
        path = rec.stop() if rec is not None else None
        if self._hud is not None:
            hud, self._hud = self._hud, None
            try:
                hud.stopped.disconnect(self._finish)     # don't re-enter via the close below
            except (TypeError, RuntimeError):
                pass
            hud.close()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            try:
                app.aboutToQuit.disconnect(self._finish)
            except (TypeError, RuntimeError):
                pass
        try:
            shutting = QtCore.QCoreApplication.closingDown()
        except Exception:
            shutting = False
        if shutting:
            self.deleteLater()                           # app is quitting: file is flushed, stay quiet
            return
        if not path:
            err = getattr(rec, "_writer", None)
            err = getattr(err, "error", None) if err is not None else None
            msg = (f"The recording could not be saved ({err})." if err
                   else "Nothing was recorded (no frames were captured).")
            QtWidgets.QMessageBox.warning(self.parent() or self, "Screen record", msg)
        else:
            from gottlux.io.paths import open_in_file_browser
            open_in_file_browser(os.path.dirname(os.path.abspath(path)))
            QtWidgets.QMessageBox.information(
                self.parent() or self, "Screen record",
                f"Saved {frames} frames ({elapsed:.1f}s) →\n{os.path.basename(path)}")
        self.deleteLater()                               # this hidden setup dialog has done its job
