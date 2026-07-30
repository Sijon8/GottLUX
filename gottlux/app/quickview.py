"""
quickview.py — the lightweight, fast live viewer for a single recording.

The viewer opens every loadable type: a ``.raw`` file, an HDF5 event file
(``.h5``/``.hdf5``), a decoded-cache ``*.meta.json`` stem, or a whole capture folder —
via the open dialogs ('Open file…' / 'Open folder…') or by dropping any of them onto the
window. 'Open in full GottLUX' hands whichever of these is loaded to the full suite.

This is the window that opens when you double-click a ``.raw`` recording: a *single*
:class:`~gottlux.app.viewer.LiveViewer` (scrub · play · mode · color · scale · FPS · accum ·
In/Out cutting + inline Save/Merge/MP4) wrapped in a thin frame — none of the ten analysis
tabs of the full instrument, so it pops up and renders fast. One prominent button hands the
recording to the full GottLUX suite when you want the 3-D cloud, workbench, trackers, etc.

**Single-instance, for fast clip-hopping.** The first double-click starts the process and the
window stays warm; every later double-click is handed to that running window over a local
socket and the *new* process exits immediately — before importing the heavy viewer / pyqtgraph
/ numba stack — so the next clip appears in a fraction of the cold-start time. To make that
work, the heavy imports below are deferred into the methods that need them; importing this
module is only PySide6.

Run it directly with ``gottlux-view path\\to\\clip.raw`` (or ``python -m gottlux.app.quickview
clip.raw``), or register it as the ``.raw`` file handler with ``gottlux-view --register`` so a
double-click in your file manager (Windows Explorer, GNOME Files, …) lands here.
"""
from __future__ import annotations

import os
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from gottlux import __version__  # the parent package is imported anyway; adds no weight

# Keep top-level windows alive past the function that creates them (Qt won't, Python GC will).
_WINDOWS: list = []
# Per-user named pipe for the single-instance hand-off (Explorer click -> running window).
_SERVER_NAME = "GottLUX.QuickView"


class QuickViewWindow(QtWidgets.QMainWindow):
    """A fast, single-view player for one recording, with inline editing + open-in-suite."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # deferred (heavy) imports — kept out of module import so the forwarder process is light
        from gottlux.app.transport import TimeController
        from gottlux.app.viewer import LiveViewer

        self.setWindowTitle(f"GottLUX {__version__} — quick view")
        self.resize(1100, 760)
        self.setAcceptDrops(True)       # any loadable file/folder can be dropped on the window
        self.rec = None
        self._path = None
        self._load_token = 0            # rising id so a rapid click supersedes an in-flight decode
        self._loaders = []              # keep loader threads alive until they finish
        self._prefetcher = None         # in-flight on-demand slice decode (preview only)
        self._ipc = None               # QLocalServer when this is the single instance

        # one private clock for this window (the full suite, when opened, uses its own)
        self.clock = TimeController(self)
        # no live-filter suite here — keep it lightweight; render handles filters=None
        self.viewer = LiveViewer(self.clock, filters=None)
        self.setCentralWidget(self.viewer)

        self._build_toolbar()
        self._build_status()
        self._set_actions_enabled(False)
        self._warm_up()
        # while a sampled preview is showing, seeking into an un-decoded span decodes it
        # on a worker thread (bounded, ~sub-second) so the timeline is seekable everywhere
        self.clock.cursorChanged.connect(self._prefetch_preview)

        # Spacebar = play/pause for this window (unless typing in a field). Mirrors the suite.
        QtWidgets.QApplication.instance().installEventFilter(self)

    def _warm_up(self):
        """Compile the Numba kernels off-thread so a later time-surface render isn't laggy."""
        import threading
        try:
            from gottlux.core.accel import warmup
            threading.Thread(target=warmup, daemon=True).start()
        except Exception:
            pass

    # ------------------------------------------------------------------ chrome
    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)

        act_open = QtGui.QAction("Open file…", self)
        act_open.setToolTip("Open another recording in this quick viewer — .raw, .h5/.hdf5, "
                            "or a decoded-cache .meta.json.")
        act_open.triggered.connect(self.open_file_dialog)
        tb.addAction(act_open)
        act_folder = QtGui.QAction("Open folder…", self)
        act_folder.setToolTip("Open a capture folder — a directory holding a .raw / .h5 "
                              "(and maybe a telemetry CSV) or a decoded cache.")
        act_folder.triggered.connect(self.open_folder_dialog)
        tb.addAction(act_folder)
        tb.addSeparator()

        # Cut → .raw, Merge highlights → .raw, and Export MP4 live inline on the transport's
        # In/Out strip (below) — right where you're working, no trip up to the toolbar.
        hint = QtWidgets.QLabel("  Save/Merge .raw · MP4 · highlights → on the In/Out bar below  ")
        hint.setObjectName("muted")
        tb.addWidget(hint)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)

        self.info_lbl = QtWidgets.QLabel("  No recording loaded.  ")
        self.info_lbl.setObjectName("muted")
        tb.addWidget(self.info_lbl)
        tb.addSeparator()

        self.full_btn = QtWidgets.QPushButton("Open in full GottLUX")
        from gottlux.app import icons
        self.full_btn.setIcon(icons.icon("chevron-right", color="ACCENT_TEXT"))
        self.full_btn.setLayoutDirection(QtCore.Qt.RightToLeft)   # chevron on the right
        self.full_btn.setObjectName("primary")
        self.full_btn.setToolTip("Hand this recording to the full GottLUX instrument — Space-time "
                                 "3-D, workbench, trackers, range lab and the rest. The decode is "
                                 "already cached, so it opens instantly.")
        self.full_btn.clicked.connect(self._open_full_suite)
        tb.addWidget(self.full_btn)

        # the light/dark toggle, in the window's top-right corner (the suite carries the
        # same action on its main toolbar; either one persists the choice for both)
        self.act_theme = QtGui.QAction(icons.theme_icon(), "", self)
        self.act_theme.setToolTip("Switch between the dark and the light theme; the choice "
                                  "is remembered and shared with the full GottLUX suite.")
        self.act_theme.triggered.connect(self._toggle_theme)
        tb.addAction(self.act_theme)

    def _toggle_theme(self):
        """Flip dark ↔ light for this window — and, through the persisted setting, for the
        next launch of either window.

        The palette, stylesheet and icon colours come from the switch itself; what is left
        here is the viewer's own plot canvas (already built, so pyqtgraph's new default
        would not reach it) and the toggle's mark, which changes shape sun ↔ moon.
        """
        from gottlux.app import icons, style
        name = style.toggle_theme(QtWidgets.QApplication.instance())
        style.refresh_window(self)
        self.act_theme.setIcon(icons.theme_icon())
        self.status.showMessage(f"{name.capitalize()} theme.", 4000)
        return name

    def _build_status(self):
        self.status = self.statusBar()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        self.status.addPermanentWidget(self.progress)
        self.status.showMessage("Open a recording to begin — .raw, .h5/.hdf5, a decoded "
                                "cache, or a capture folder (drag-and-drop works too).")

    def _set_actions_enabled(self, on):
        self.full_btn.setEnabled(on)

    # ------------------------------------------------------------------ loading
    def open_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open event recording", os.path.dirname(self._path or ""),
            "Event recordings (*.raw *.h5 *.hdf5 *.meta.json);;All files (*)")
        if path:
            self.load(path)

    def open_folder_dialog(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open capture folder", os.path.dirname(self._path or ""))
        if path:
            self.load(path)

    # ------------------------------------------------------------------ drag & drop
    @staticmethod
    def _drop_path(md):
        """The first local file or folder among dragged URLs, or ``None`` when none is."""
        if not md.hasUrls():
            return None
        for url in md.urls():
            p = url.toLocalFile()
            if p and os.path.exists(p):
                return p
        return None

    def dragEnterEvent(self, ev):
        """Accept a dragged file/folder — every type :func:`gottlux.load` accepts."""
        if self._drop_path(ev.mimeData()):
            ev.acceptProposedAction()
        else:
            super().dragEnterEvent(ev)

    def dropEvent(self, ev):
        """Load the dropped file or capture folder (the same paths the open dialogs take)."""
        p = self._drop_path(ev.mimeData())
        if p:
            ev.acceptProposedAction()
            self.load(p)
        else:
            super().dropEvent(ev)

    def load(self, path):
        """Decode *path* on a worker thread. Rapid calls supersede each other (latest wins)."""
        from gottlux.app.loader import RecordingLoader
        path = os.path.abspath(path)
        self._path = path
        self._load_token += 1
        token = self._load_token
        self.setWindowTitle(f"GottLUX {__version__} — quick view — {os.path.basename(path)}")
        self.status.showMessage(f"Decoding {os.path.basename(path)} …")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        loader = RecordingLoader(path)
        loader.progress.connect(lambda f, t=token: self._on_progress(f, t))
        loader.preview.connect(lambda rec, t=token: self._on_preview(rec, t))
        loader.loaded.connect(lambda rec, t=token: self._on_loaded(rec, t))
        loader.failed.connect(lambda msg, t=token: self._on_failed(msg, t))
        loader.finished.connect(
            lambda lo=loader: self._loaders.remove(lo) if lo in self._loaders else None)
        self._loaders.append(loader)             # hold a ref so the QThread isn't GC'd mid-run
        loader.start()

    def _on_progress(self, f, token):
        if token == self._load_token:
            self.progress.setValue(int(f * 100))

    def _on_preview(self, rec, token=None):
        """Phase 1 of a large-file open: show the sampled preview, keep the bar running."""
        if token is not None and token != self._load_token:
            return
        self.rec = rec
        self.info_lbl.setText(
            f"  {rec.name}  ·  {rec.width}×{rec.height}  ·  preview ({rec.n:,} ev sampled)  ·  "
            f"{rec.duration_s:.2f}s  ·  {rec.fmt}  ")
        self.status.showMessage("Preview — playing sampled view while the full decode finishes…")
        self.clock.pause()
        self.clock.set_range(rec.t_start_s, rec.t_stop_s)   # the WHOLE file span (from the index)
        self.clock.set_cursor(rec.t_start_s)
        self.clock.set_selection(0.0, 1.0)
        self.clock.clear_highlights()
        self.viewer.set_recording(rec)
        self._set_actions_enabled(True)
        self.clock.set_loop(True)
        self.clock.play()

    def _on_loaded(self, rec, token=None):
        if token is not None and token != self._load_token:
            return                               # a newer click superseded this decode
        upgrading = self.rec is not None and getattr(self.rec, "is_preview", False)
        self.rec = rec
        self.progress.setVisible(False)
        self.info_lbl.setText(
            f"  {rec.name}  ·  {rec.width}×{rec.height}  ·  {rec.n:,} ev  ·  "
            f"{rec.duration_s:.2f}s  ·  {rec.fmt}  ·  "
            f"{'ROTATION' if rec.is_rotating else 'STARING'}  ")
        self.status.showMessage(f"Loaded {rec.name} ({rec.n:,} events).")
        if upgrading:
            # swap the full memmap-backed Recording under the running preview without
            # resetting the cursor / play state / In-Out the user may already have set
            cursor, playing = self.clock.cursor, self.clock.playing
            self.clock.pause()
            self.clock.set_range(rec.t_start_s, rec.t_stop_s)
            self.clock.set_cursor(min(cursor, rec.t_stop_s))
            self.viewer.set_recording(rec)
            self._set_actions_enabled(True)
            if playing:
                self.clock.play()
            return
        # prime the clock before the view renders
        self.clock.pause()
        self.clock.set_range(rec.t_start_s, rec.t_stop_s)
        self.clock.set_cursor(rec.t_start_s)
        self.clock.set_selection(0.0, 1.0)         # reset In/Out to the full clip
        self.clock.clear_highlights()              # drop any highlight bands from the prior clip
        self.viewer.set_recording(rec)
        self._set_actions_enabled(True)
        # auto-start, looping — the clip plays the moment it loads and repeats end-to-end
        self.clock.set_loop(True)
        self.clock.play()

    def _on_failed(self, msg, token=None):
        if token is not None and token != self._load_token:
            return
        self.progress.setVisible(False)
        self.status.showMessage("Load failed.")
        QtWidgets.QMessageBox.critical(self, "Load failed", msg)

    def _prefetch_preview(self, t):
        """While a sampled preview is showing, a cursor moving into a span that was never
        decoded kicks a bounded on-demand slice decode on a worker thread (one in flight;
        the next cursor tick retries). Mirrors ``MainWindow._prefetch_preview``."""
        rec = self.rec
        if rec is None or not getattr(rec, "is_preview", False):
            return
        t0, t1 = self.clock.accum_window(t)
        t1 = min(max(t1, t0 + 0.25) + 0.25, rec.t_stop_s)   # decode a little ahead of the cursor
        if t1 <= t0 or rec.covers(t0, t1):
            return
        if self._prefetcher is not None and self._prefetcher.isRunning():
            return
        from gottlux.app.loader import PreviewPrefetcher
        self._prefetcher = PreviewPrefetcher(rec, t0, t1)
        self._prefetcher.done.connect(
            lambda: self.clock.cursorChanged.emit(self.clock.cursor))
        self._prefetcher.start()

    # ------------------------------------------------------------------ single instance
    def _start_server(self):
        """Listen for later launches so this one window serves every double-click."""
        from PySide6 import QtNetwork
        self._ipc = QtNetwork.QLocalServer(self)
        if not self._ipc.listen(_SERVER_NAME):
            QtNetwork.QLocalServer.removeServer(_SERVER_NAME)    # clear a stale socket, retry once
            if not self._ipc.listen(_SERVER_NAME):
                self._ipc = None
                return
        self._ipc.newConnection.connect(self._on_ipc_connection)

    def _on_ipc_connection(self):
        sock = self._ipc.nextPendingConnection()
        if sock is None:
            return
        if sock.waitForReadyRead(2000):
            path = sock.readAll().data().decode("utf-8", "replace").strip()
            if path and os.path.exists(path):
                self.load(path)
        sock.disconnectFromServer()
        self.showNormal(); self.raise_(); self.activateWindow()   # bring forward for the new clip

    # ------------------------------------------------------------------ hand-off
    def _open_full_suite(self):
        """Open this recording in the full tabbed instrument, reusing the warm decode.

        If the background decode already finished, the full Recording is handed over —
        instant. If we are still showing a sampled preview, the *path* is handed over
        instead: the full window re-runs the two-phase load (its preview shows just as
        fast, and :func:`gottlux.io.cache.load` serializes on the in-flight decode so the
        work is not repeated)."""
        if self._path is None:
            return
        from gottlux.app.main import MainWindow
        full = MainWindow()
        _WINDOWS.append(full)
        full.show()
        if self.rec is not None and not getattr(self.rec, "is_preview", False):
            full._on_loaded(self.rec)              # reuse the already-decoded Recording — instant
        else:
            full.load(self._path)
        QtCore.QTimer.singleShot(0, self.close)    # close once the full window is up

    # ------------------------------------------------------------------ spacebar
    def eventFilter(self, obj, event):
        if (event.type() == QtCore.QEvent.KeyPress and event.key() == QtCore.Qt.Key_Space
                and not event.isAutoRepeat() and self.isActiveWindow() and self.rec is not None):
            fw = QtWidgets.QApplication.focusWidget()
            editing = isinstance(fw, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox,
                                      QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)) or \
                (isinstance(fw, QtWidgets.QComboBox) and fw.isEditable())
            if not editing:
                self.clock.toggle()
                return True
        return super().eventFilter(obj, event)

    def closeEvent(self, ev):
        try:
            if self._ipc is not None:
                self._ipc.close(); self._ipc = None    # free the single-instance name
        except Exception:
            pass
        try:
            QtWidgets.QApplication.instance().removeEventFilter(self)
        except Exception:
            pass
        if self in _WINDOWS:
            _WINDOWS.remove(self)
        super().closeEvent(ev)


def _first_path(argv):
    return next((a for a in argv[1:] if not a.startswith("-")), None)


def _forward_if_running(path):
    """If a quick viewer is already open, hand it PATH (single-instance) and return True.

    Runs in the freshly-launched process; on success it returns *before* the heavy viewer /
    pyqtgraph / numba imports happen, so hopping between clips reuses the warm window fast.
    """
    from PySide6 import QtNetwork
    sock = QtNetwork.QLocalSocket()
    sock.connectToServer(_SERVER_NAME)
    if not sock.waitForConnected(500):
        return False
    sock.write(os.path.abspath(path).encode("utf-8"))
    sock.waitForBytesWritten(2000)
    sock.flush()
    sock.disconnectFromServer()
    if sock.state() != QtNetwork.QLocalSocket.UnconnectedState:
        sock.waitForDisconnected(500)
    return True


def main(argv=None):
    """Entry point for ``gottlux-view`` / ``python -m gottlux.app.quickview``.

    Flags: ``--register`` / ``--unregister`` manage the ``.raw`` file association (Windows
    registry or Linux XDG, per platform — see :mod:`gottlux.app.file_assoc`) and exit;
    otherwise a window opens (pre-loaded with PATH if one is given). If a quick viewer is
    already running, PATH is handed to it and this process exits immediately.
    """
    argv = list(sys.argv if argv is None else argv)

    if "--register" in argv or "--unregister" in argv:
        # The messages carry unit/arrow glyphs (e.g. →); make the console UTF-8 so they
        # print cleanly on Windows (the default cp1252 console otherwise raises).
        for _stream in (sys.stdout, sys.stderr):
            try:
                _stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
        from gottlux.app import file_assoc
        ok, msg = (file_assoc.unregister() if "--unregister" in argv else file_assoc.register())
        print(msg)
        return 0 if ok else 1

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    path = _first_path(argv)
    # Single-instance fast path: an open viewer takes the file; we exit before the heavy imports.
    if path and os.path.exists(path) and _forward_if_running(path):
        return 0

    from gottlux.app import icons, style
    app.setOrganizationName("GottLUX")
    app.setApplicationName("gottlux")
    app.setWindowIcon(icons.app_icon())       # the painted 'event burst' mark, all sizes
    style.apply_app_style(app)                # opens in the persisted light/dark theme
    win = QuickViewWindow()
    _WINDOWS.append(win)
    win._start_server()                          # become the instance future launches hand off to
    win.show()
    if path and os.path.exists(path):
        QtCore.QTimer.singleShot(80, lambda: win.load(path))
    else:
        QtCore.QTimer.singleShot(0, win.open_file_dialog)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
