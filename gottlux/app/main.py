"""
main.py — the gottlux desktop application: a tabbed instrument around one Recording.

Open a ``.raw`` (or a capture folder, or a decoded cache) and it loads on a background
thread behind a progress bar; for a large, uncached ``.raw`` a sampled preview (see
:mod:`gottlux.io.preview`) appears in a second or two — spanning the whole file's
timeline, seekable anywhere — and the finished full decode swaps in underneath without
resetting the cursor or play state. The same :class:`~gottlux.io.recording.Recording`
is then shared by every panel:

* **Live viewer**     — scrub/play the stream as accumulated frames (smooth slow-motion).
* **Space-time**      — the 3-D (x, y, t) event cloud you orbit and fly through.
* **Workbench**       — the flicker-map + region-spectrum + tunable-detector tuning lab.

Run it with ``gottlux-gui`` (or ``python -m gottlux``), optionally passing a path to open.
"""
from __future__ import annotations

import os
import sys

from PySide6 import QtCore, QtGui, QtWidgets

# Only light modules are imported at module level so that importing this module (and
# constructing the boot splash) is cheap — the heavy panel widgets, which pull in
# pyqtgraph/scipy/OpenGL, are imported lazily inside MainWindow.__init__ so the splash
# is already on screen, with real progress, before that cost is paid. See splash.py.
from gottlux import __version__
from gottlux.app import icons
from gottlux.app import style
from gottlux.app.splash import BootSplash


class MainWindow(QtWidgets.QMainWindow):
    _TAB_NAMES = ("Live viewer", "Multi-clip", "Range lab", "Event-rate tower",
                  "Space-time 3D", "Flutter workbench", "Sandbox", "EBS viewer", "Fusion lab",
                  "Timeline")

    def __init__(self, on_progress=None):
        super().__init__()
        # ``on_progress(frac, msg)`` (e.g. the boot splash) is pinged as each panel is built,
        # so a cold start shows genuine progress instead of a frozen screen. No-op if absent.
        progress = on_progress or (lambda frac, msg=None: None)
        self.setWindowTitle(f"GottLUX {__version__} — unified event-based-sensor "
                            "analysis instrument")
        self.resize(1280, 820)
        self.rec = None
        self._loader = None
        self._prefetcher = None            # in-flight on-demand slice decode (preview only)
        self._current_roi = None           # the live viewer's ROI (or None) — reused by Tools
        self._tool_windows = []            # Canvas composer windows (kept alive past this scope)

        # Heavy widgets are imported here (not at module top) so the splash paints first.
        progress(0.10, "Loading modules…")
        from gottlux.app.ebsviewer import EBSViewer
        from gottlux.app.filters import FilterController
        from gottlux.app.fusionlab import FusionLab
        from gottlux.app.multiclip import MultiClipViewer
        from gottlux.app.rangelab import RangeLab
        from gottlux.app.sandbox import Sandbox
        from gottlux.app.spacetime import SpaceTimeView
        from gottlux.app.timeline import TimelineEditor
        from gottlux.app.tower import EventRateTower
        from gottlux.app.transport import TimeController
        from gottlux.app.viewer import LiveViewer
        from gottlux.app.workbench import FlutterWorkbench

        progress(0.18, "Preparing shared clock…")
        # one shared clock for every view: seek + accumulation stay in lock-step
        self.clock = TimeController(self)
        # one shared live noise-filter suite applied across the viewing tabs
        self.filters = FilterController(self)

        # panels (all bound to the shared clock; the Multi-clip slate keeps its own clock).
        # Built one at a time, each behind a progress tick — the Live viewer bears the
        # one-time pyqtgraph init and is the slowest, hence its wide progress band.
        progress(0.25, "Building Live viewer…")
        self.viewer = LiveViewer(self.clock, self.filters)
        progress(0.46, "Building Multi-clip slate…")
        self.multiclip = MultiClipViewer(self.clock, self.filters)
        progress(0.52, "Building Range lab…")
        self.rangelab = RangeLab(self.clock, self.filters)
        progress(0.58, "Building Event-rate tower…")
        self.tower = EventRateTower(self.clock, self.filters)
        progress(0.64, "Building Space-time 3-D…")
        self.spacetime = SpaceTimeView(self.clock, self.filters)
        progress(0.70, "Building Flutter workbench…")
        self.workbench = FlutterWorkbench(self.clock)
        progress(0.80, "Building Sandbox…")
        self.sandbox = Sandbox(self.clock)
        progress(0.84, "Building EBS viewer…")
        self.ebsviewer = EBSViewer()          # the classic viewer (its own playhead)
        progress(0.85, "Building Fusion lab…")
        self.fusionlab = FusionLab(self.clock, self.filters)
        progress(0.86, "Building Timeline editor…")
        self.timeline = TimelineEditor()      # its own clock; painted lanes, no pyqtgraph/OpenGL
        self.panels = (self.viewer, self.multiclip, self.rangelab, self.tower,
                       self.spacetime, self.workbench, self.sandbox, self.ebsviewer,
                       self.fusionlab, self.timeline)
        progress(0.87, "Assembling workspace…")
        self.tabs = self._make_tabs()
        self._compare_panels = None        # built lazily when the user splits the view
        self.tabs2 = None
        # a horizontal splitter so a second pane can sit side-by-side for comparison
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.addWidget(self.tabs)
        self.setCentralWidget(self.splitter)

        # pause playback when switching tabs (every clock, so a tab with its own clock — the
        # Multi-clip slate — doesn't keep animating in the background after you leave it)
        self.tabs.currentChanged.connect(lambda *_: self._pause_all_clocks())
        # link viewer ROI -> workbench region (point the workbench at what you see)
        self.viewer.roiChanged.connect(self._roi_to_workbench)
        # while a sampled preview is showing, seeking into an un-decoded span decodes it
        # on a worker thread (bounded, ~sub-second) so the timeline is seekable everywhere
        self.clock.cursorChanged.connect(self._prefetch_preview)

        self._build_menu()
        self._build_filter_toolbar()
        self._build_status()
        progress(0.94, "Warming up compute kernels…")
        self._warm_up()
        progress(1.0, "Ready.")

        # Spacebar = play/pause anywhere in the app (the shared clock). An application-wide event
        # filter is used so focus doesn't matter — except while typing in a text/number/combo
        # field, where space must still insert a space.
        QtWidgets.QApplication.instance().installEventFilter(self)

    # ------------------------------------------------------------------ clock ownership
    def active_clock(self):
        """The clock of the currently-shown tab. Most tabs share the app clock; the Multi-clip
        slate runs on its own. Controls (spacebar, capture, export window) route here so they
        always act on the view you're looking at."""
        panel = self.tabs.currentWidget()
        return panel.capture_clock() if hasattr(panel, "capture_clock") else self.clock

    def _all_clocks(self):
        clocks = [self.clock]
        for panel in self._all_panels():
            c = panel.capture_clock() if hasattr(panel, "capture_clock") else None
            if c is not None and c not in clocks:
                clocks.append(c)
        return clocks

    def _pause_all_clocks(self):
        for c in self._all_clocks():
            c.pause()

    def eventFilter(self, obj, event):
        if (event.type() == QtCore.QEvent.KeyPress and event.key() == QtCore.Qt.Key_Space
                and not event.isAutoRepeat()):
            fw = QtWidgets.QApplication.focusWidget()
            editing = isinstance(fw, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox,
                                      QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)) or \
                (isinstance(fw, QtWidgets.QComboBox) and fw.isEditable())
            if not editing and self.rec is not None:
                self.active_clock().toggle()          # play/pause the tab you're looking at
                return True
        return super().eventFilter(obj, event)

    def _build_filter_toolbar(self):
        """A program-wide live noise-reduction strip, applied to every viewing tab."""
        from gottlux.app.filters import FilterBar
        self.addToolBarBreak()
        tb = self.addToolBar("Filters")
        tb.setMovable(False)
        tb.addWidget(QtWidgets.QLabel("  Live denoise  "))
        tb.addWidget(FilterBar(self.filters))

    # ------------------------------------------------------------------ menu / status
    def _build_menu(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        # painted vector icons + text labels (never icon-only: the labels carry the meaning)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        act_file = QtGui.QAction("Open file…", self)
        act_file.triggered.connect(self.open_file_dialog)
        act_folder = QtGui.QAction("Open folder…", self)
        act_folder.triggered.connect(self.open_folder_dialog)
        tb.addAction(act_file)
        tb.addAction(act_folder)
        # Examples — bundled demo clips, one click to load (populated live so dropped-in
        # files appear without a restart; the dropdown arrow comes from the stylesheet)
        self.examples_btn = QtWidgets.QToolButton()
        self.examples_btn.setText("Examples")
        self.examples_btn.setToolTip("Open one of the bundled demo recordings.")
        self.examples_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.examples_menu = QtWidgets.QMenu(self.examples_btn)
        self.examples_menu.aboutToShow.connect(self._populate_examples_menu)
        self.examples_btn.setMenu(self.examples_menu)
        tb.addWidget(self.examples_btn)
        tb.addSeparator()
        act_sync = QtGui.QAction(icons.icon("sync"), "Sync views", self)
        act_sync.setToolTip("Recompute every tab at the current cursor — so the moment you "
                            "found in the live viewer shows in the 3-D and workbench views too.")
        act_sync.triggered.connect(self._sync_views)
        tb.addAction(act_sync)
        act_export = QtGui.QAction(icons.icon("export"), "Export…", self)
        act_export.setToolTip("Overall export — choose exactly which figures, cubes, tables and "
                              "reports to write into one folder with a manifest.")
        act_export.triggered.connect(self._export_all)
        tb.addAction(act_export)
        act_capture = QtGui.QAction(icons.icon("capture"), "Capture…", self)
        act_capture.setToolTip("Save the current view to a video + infographic poster — a faithful, "
                               "high-res re-render of this tab's exact tuned settings over the "
                               "In/Out range (with a context banner and a manifest).")
        act_capture.triggered.connect(self._capture_view)
        tb.addAction(act_capture)
        act_record = QtGui.QAction(icons.icon("record"), "Screen rec…", self)
        act_record.setToolTip("Live screen recorder (like the Snipping Tool): drag any region of "
                              "the display, or record the current view / whole window, straight to "
                              "MP4 — captures exactly what's on screen.")
        act_record.triggered.connect(self._screen_record)
        tb.addAction(act_record)
        act_cut = QtGui.QAction(icons.icon("cut"), "Cut selection → .raw", self)
        act_cut.setToolTip("Cut the In/Out selection (the bar above the timeline) to a new "
                           "valid .raw clip. Select the whole recording = the whole clip.")
        act_cut.triggered.connect(self._cut_selection)
        tb.addAction(act_cut)
        act_editor = QtGui.QAction(icons.icon("cut"), "Clip editor", self)
        act_editor.setToolTip("The comprehensive .raw clip editor — add clips, trim (In/Out), "
                              "crop (ROI), reorder, and stitch into one valid .raw.")
        act_editor.triggered.connect(self._open_editor)
        tb.addAction(act_editor)
        act_compare = QtGui.QAction(icons.icon("split"), "Split view", self)
        act_compare.setCheckable(True)
        act_compare.setToolTip("Show a second pane side-by-side so two visualizations can be "
                               "compared at the same moment (both follow the shared clock).")
        act_compare.toggled.connect(self._toggle_compare)
        tb.addAction(act_compare)
        tb.addSeparator()
        self.act_theme = QtGui.QAction(icons.theme_icon(), self._theme_label(), self)
        self.act_theme.setToolTip("Switch the whole instrument between the dark and the light "
                                  "theme. The choice is remembered for the next launch (the "
                                  "quick viewer opens in it too).")
        self.act_theme.triggered.connect(self._toggle_theme)
        tb.addAction(self.act_theme)

        m = self.menuBar().addMenu("&File")
        m.addAction(act_file)
        m.addAction(act_folder)
        m.addSeparator()
        act_quickcut = QtGui.QAction(icons.icon("cut"), "Quick-cut a .raw (no decode)…", self)
        act_quickcut.setToolTip("Crop a .raw to a time window WITHOUT opening/decoding it — index "
                                "the EVT2.1 byte stream and write the clip directly. Lets you "
                                "shorten a multi-GB recording without the slow open.")
        act_quickcut.triggered.connect(self._open_quick_cut)
        m.addAction(act_quickcut)
        act_trim = QtGui.QAction(icons.icon("cut"), "Batch-trim folder…", self)
        act_trim.setToolTip("Shorten every .raw in a folder by one shared time window (e.g. drop "
                            "the first few seconds of ego-motion) into a 'trimmed/' subfolder, "
                            "re-based to a common origin so synced clips stay aligned.")
        act_trim.triggered.connect(self._open_batch_trim)
        m.addAction(act_trim)
        m.addSeparator()
        quit_act = QtGui.QAction("Quit", self)
        quit_act.triggered.connect(self.close)
        m.addAction(quit_act)

        # Tools — the composition / scripting / hand-off instruments. Each acts on the
        # CURRENT view state: the In/Out selection (or the cursor's accumulation window)
        # and the live viewer's ROI, so what runs is exactly what is on screen.
        tm = self.menuBar().addMenu("&Tools")
        act_canvas = QtGui.QAction(icons.icon("split"), "Canvas composer…", self)
        act_canvas.setToolTip("Compose several recordings — possibly different sensors and "
                              "time bases — as positioned, styled cells on one canvas; play "
                              "it, save the spec, export as MP4 or a composited .raw.")
        act_canvas.triggered.connect(self._open_canvas_composer)
        tm.addAction(act_canvas)
        act_script = QtGui.QAction(icons.icon("target"),
                                   "Run user script on current view…", self)
        act_script.setToolTip("Run a .py file defining process(win, ctx) on exactly the "
                              "portion in view — the In/Out selection (or the cursor's "
                              "accumulation window) and the current ROI. Results land in a "
                              "provenance-stamped run folder.")
        act_script.triggered.connect(self._run_user_script)
        tm.addAction(act_script)
        act_bundle = QtGui.QAction(icons.icon("export"), "Export tool bundle…", self)
        act_bundle.setToolTip("Write a standalone Python + MATLAB tool bundle (data.h5, "
                              "scripts, README, provenance) for the current window/ROI — "
                              "runnable with no gottlux installed on the receiving end.")
        act_bundle.triggered.connect(self._export_tool_bundle)
        tm.addAction(act_bundle)
        self.tools_menu = tm

    # ------------------------------------------------------------------ theme
    @staticmethod
    def _theme_label():
        """The toolbar action's label — what pressing it switches *to*."""
        return "Light theme" if style.THEME == "dark" else "Dark theme"

    def _toggle_theme(self):
        """Flip dark ↔ light live and persist the choice.

        :func:`gottlux.app.style.toggle_theme` rewrites the palette, re-applies the
        application stylesheet, re-points pyqtgraph, drops the icon cache and announces
        the change; what is left here is this window's own chrome: the plot canvases and
        3-D backgrounds already built, and a repolish so every styled widget re-reads the
        sheet it was constructed under.
        """
        app = QtWidgets.QApplication.instance()
        name = style.toggle_theme(app)
        for panel in self._all_panels():                 # 3-D backgrounds, inline sheets
            hook = getattr(panel, "apply_theme", None)
            if callable(hook):
                hook()
        style.refresh_window(self)
        self.act_theme.setIcon(icons.theme_icon())
        self.act_theme.setText(self._theme_label())
        self.status.showMessage(f"{name.capitalize()} theme.", 4000)
        return name

    def _build_status(self):
        self.status = self.statusBar()
        # The recording summary lives here (a permanent status-bar widget) rather than crammed
        # into the top toolbar, where it pushed the action buttons into an overflow menu on a
        # narrow window / split view.
        self.info_lbl = QtWidgets.QLabel("No recording loaded.")
        self.status.addPermanentWidget(self.info_lbl)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        self.status.addPermanentWidget(self.progress)
        self.status.showMessage("Ready.")

    def _warm_up(self):
        """Compile the Numba kernels in the background so first render is instant."""
        import threading
        from gottlux.core.accel import warmup
        threading.Thread(target=warmup, daemon=True).start()

    # ------------------------------------------------------------------ loading
    def open_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open event recording", "",
            "Event recordings (*.raw *.h5 *.hdf5 *.meta.json);;All files (*)")
        if path:
            self.load(path)

    def open_folder_dialog(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open capture folder")
        if path:
            self.load(path)

    # ------------------------------------------------------------------ bundled examples
    def _populate_examples_menu(self):
        """(Re)build the Examples dropdown from whatever is in the examples folder right now."""
        from gottlux import examples as ex
        self.examples_menu.clear()
        items = ex.list_examples()
        if not items:
            a = self.examples_menu.addAction("No examples found")
            a.setEnabled(False)
            return
        for e in items:
            act = self.examples_menu.addAction(f"{e.title}    ({e.detail})")
            act.triggered.connect(lambda _=False, p=e.path: self.load(p))
        folder = ex.examples_dir()
        if folder:
            self.examples_menu.addSeparator()
            openf = self.examples_menu.addAction("Open examples folder…")
            openf.triggered.connect(lambda: self._reveal(folder))

    @staticmethod
    def _reveal(folder):
        from gottlux.io.paths import open_in_file_browser
        open_in_file_browser(folder)

    def maybe_show_welcome(self):
        """First launch with nothing loaded: offer the bundled example clips (opt-out persisted)."""
        from gottlux import examples as ex
        from gottlux.app.welcome import WelcomeDialog, show_examples_on_start
        if self.rec is not None or not show_examples_on_start() or not ex.has_examples():
            return
        dlg = WelcomeDialog(self)
        if dlg.exec() and dlg.chosen_path:
            self.load(dlg.chosen_path)

    def load(self, path):
        from gottlux.app.loader import RecordingLoader
        if self._loader and self._loader.isRunning():
            return
        self.status.showMessage(f"Decoding {os.path.basename(path)} …")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._loader = RecordingLoader(path)
        self._loader.progress.connect(lambda f: self.progress.setValue(int(f * 100)))
        self._loader.preview.connect(self._on_preview)
        self._loader.loaded.connect(self._on_loaded)
        self._loader.failed.connect(self._on_failed)
        self._loader.start()

    def _on_preview(self, rec):
        """Phase 1 of a large-file open: show the sampled preview at once, spanning the
        WHOLE file's timeline (from the index), while the full cache decode keeps running
        behind the progress bar. ``_on_loaded`` swaps the real Recording in when it lands."""
        self.rec = rec
        self.info_lbl.setText(
            f"  {rec.name}  ·  {rec.width}×{rec.height}  ·  preview ({rec.n:,} ev sampled)  ·  "
            f"{rec.duration_s:.2f}s  ·  {rec.fmt}  ")
        self.status.showMessage("Preview — playing sampled view while the full decode finishes…")
        self.clock.pause()
        self.clock.set_range(rec.t_start_s, rec.t_stop_s)   # the full-file span, not the samples
        self.clock.set_cursor(rec.t_start_s)
        self.clock.set_selection(0.0, 1.0)
        self.clock.clear_highlights()
        for panel in self._all_panels():
            try:
                panel.set_recording(rec)
            except Exception as e:
                self.status.showMessage(f"panel error: {e}")
        self.clock.set_loop(True)
        self.clock.play()

    def _on_loaded(self, rec):
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
            for panel in self._all_panels():
                try:
                    panel.set_recording(rec)
                except Exception as e:
                    self.status.showMessage(f"panel error: {e}")
            if playing:
                self.clock.play()
            return
        # set the shared clock's range to the recording before the panels render
        self.clock.pause()
        self.clock.set_range(rec.t_start_s, rec.t_stop_s)
        self.clock.set_cursor(rec.t_start_s)
        self.clock.set_selection(0.0, 1.0)            # reset the In/Out selection to the full clip
        self.clock.clear_highlights()                 # drop any highlight bands from the prior clip
        for panel in self._all_panels():
            try:
                panel.set_recording(rec)
            except Exception as e:
                self.status.showMessage(f"panel error: {e}")
        # auto-start, looping — the clip plays the moment it loads and repeats end-to-end
        self.clock.set_loop(True)
        self.clock.play()

    def _prefetch_preview(self, t):
        """While a sampled preview is showing, a cursor moving into a span that was never
        decoded kicks a bounded on-demand slice decode on a worker thread (latest wins; one
        in flight). The views keep rendering the covered spans and pick the new events up
        on the next cursor tick."""
        rec = self.rec
        if rec is None or not getattr(rec, "is_preview", False):
            return
        t0, t1 = self.clock.accum_window(t)
        t1 = min(max(t1, t0 + 0.25) + 0.25, rec.t_stop_s)   # decode a little ahead of the cursor
        if t1 <= t0 or rec.covers(t0, t1):
            return
        if self._prefetcher is not None and self._prefetcher.isRunning():
            return                                   # one in flight; the next tick retries
        from gottlux.app.loader import PreviewPrefetcher
        self._prefetcher = PreviewPrefetcher(rec, t0, t1)
        # re-render every view at the (now decoded) cursor once coverage grew
        self._prefetcher.done.connect(
            lambda: self.clock.cursorChanged.emit(self.clock.cursor))
        self._prefetcher.start()

    def _on_failed(self, msg):
        self.progress.setVisible(False)
        self.status.showMessage("Load failed.")
        QtWidgets.QMessageBox.critical(self, "Load failed", msg)

    def _sync_views(self):
        """Force every tab to render at the current cursor (the 'one moment, all views' button)."""
        if self.rec is None:
            return
        self.clock.pause()
        for panel in self._all_panels():
            try:
                panel.sync()
            except Exception as e:
                self.status.showMessage(f"sync error: {e}")
        self.status.showMessage(f"Synced all views to t = {self.clock.cursor:.3f} s.")

    def _roi_to_workbench(self, roi):
        self._current_roi = roi           # remembered for the Tools actions (script / bundle)
        self.workbench.set_region(roi)

    # ------------------------------------------------------------------ tools menu
    def _current_window(self):
        """The time span currently in play: the In/Out selection when one is set, else the
        accumulation window at the cursor (honoring the accumulation direction). Read off
        the active tab's clock so a tab with its own clock (Multi-clip) is respected."""
        clock = self.active_clock()
        if clock.has_selection():
            return clock.sel_t0(), clock.sel_t1()
        return clock.accum_window()

    def _open_canvas_composer(self):
        """Open the Canvas composer, seeded with the current recording as its first clip."""
        from gottlux.app.canvas import CanvasComposerWindow
        win = CanvasComposerWindow(self)
        self._tool_windows.append(win)    # keep a Python ref so the window is not GC'd
        if self.rec is not None:
            try:
                win.add_clip_recording(self.rec)
            except Exception as e:
                self.status.showMessage(f"canvas seed error: {e}")
        win.show()
        return win

    def _run_user_script(self):
        """Run a user script (``process(win, ctx)``) on exactly the portion in view.

        The window is the In/Out selection when set (else the cursor's accumulation
        window), the ROI is the live viewer's box when shown. The results dialog reports
        the run folder and a summary; script failures surface as an error dialog, never
        as an exception out of the GUI."""
        if self.rec is None:
            QtWidgets.QMessageBox.information(self, "User script", "Load a recording first.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Run user script on current view", "",
            "Python scripts (*.py);;All files (*)")
        if not path:
            return
        self.active_clock().pause()
        t0, t1 = self._current_window()
        roi = self._current_roi
        from gottlux.userscripts import run_script
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            res = run_script(path, self.rec, t0=t0, t1=t1, roi=roi)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "User script", str(e))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        roi_note = "" if roi is None else f", ROI {','.join(str(int(v)) for v in roi)}"
        lines = [f"Run folder:\n{res['folder']}", "",
                 f"{res['n_events']:,} events in view ([{t0:.3f}, {t1:.3f}] s{roi_note}) · "
                 f"{res['wall_s']:.2f} s wall · result: {res['result_kind']}"]
        if res["outputs"]:
            lines += [""] + [f"{fname} — {desc}" for fname, desc in res["outputs"]]
        else:
            lines += ["", "(no return value was saved; any outputs were written by the "
                          "script itself)"]
        QtWidgets.QMessageBox.information(self, "User script", "\n".join(lines))

    def _export_tool_bundle(self):
        """Pick a standalone tool (see :data:`gottlux.export_tools.TOOLS`) and export it."""
        if self.rec is None:
            QtWidgets.QMessageBox.information(self, "Export tool bundle",
                                              "Load a recording first.")
            return
        from gottlux.export_tools import TOOLS
        items = [f"{name} — {TOOLS[name].description}" for name in sorted(TOOLS)]
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "Export tool bundle",
            "Standalone tool to export — a Python + MATLAB pair with data.h5,\n"
            "README and provenance, runnable with no gottlux installed:",
            items, 0, False)
        if not ok or not choice:
            return
        self._run_tool_export(choice.split(" — ")[0])

    def _run_tool_export(self, name):
        """Export tool bundle *name* for the current window/ROI; for ``viz_config`` the
        live viewer's current mode / colormap / tone-map / accumulation are baked in."""
        self.active_clock().pause()
        t0, t1 = self._current_window()
        roi = self._current_roi
        viz = None
        if name == "viz_config":
            from gottlux.run.tool_export import VIZ_TONEMAPS
            mode = self.viewer.mode.currentText()
            expr = self.viewer.expr.currentText()
            viz = {"mode": ("polarity" if mode in ("polarity", "polarity_ratio")
                            else "count"),
                   "cmap": self.viewer.cmap.currentText(),
                   "tonemap": expr if expr in VIZ_TONEMAPS else None,
                   "accum_ms": self.clock.accum * 1e3}
        from gottlux.app.uikit import with_progress
        from gottlux.run.tool_export import export_tool
        try:
            res = with_progress(self, "Exporting tool bundle",
                                lambda cb: export_tool(self.rec, name, t0=t0, t1=t1,
                                                       roi=roi, viz=viz),
                                label="Writing the bundle…")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export tool bundle", str(e))
            return
        QtWidgets.QMessageBox.information(
            self, "Export tool bundle",
            f"Bundle → {res['path']}\n\n{res['n_events']:,} events in data.h5 · "
            f"{len(res['written'])} files written · the bundle README describes how to "
            "run the scripts.")
        return res

    # ------------------------------------------------------------------ side-by-side compare
    def _make_tabs(self, panels=None):
        panels = panels if panels is not None else self.panels
        tw = QtWidgets.QTabWidget()
        for w, name in zip(panels, self._TAB_NAMES):
            tw.addTab(w, name)
        return tw

    def _all_panels(self):
        """Every live panel — the main set plus the compare pane's set if it exists."""
        return self.panels + (self._compare_panels or ())

    def _toggle_compare(self, on):
        """Show/hide a second tabbed pane next to the first for side-by-side analysis."""
        if on:
            if self.tabs2 is None:
                self._build_compare_pane()
            self.tabs2.show()
            self.splitter.setSizes([self.width() // 2, self.width() // 2])
        elif self.tabs2 is not None:
            self.tabs2.hide()

    def _build_compare_pane(self):
        from gottlux.app.ebsviewer import EBSViewer
        from gottlux.app.fusionlab import FusionLab
        from gottlux.app.multiclip import MultiClipViewer
        from gottlux.app.rangelab import RangeLab
        from gottlux.app.sandbox import Sandbox
        from gottlux.app.spacetime import SpaceTimeView
        from gottlux.app.timeline import TimelineEditor
        from gottlux.app.tower import EventRateTower
        from gottlux.app.viewer import LiveViewer
        from gottlux.app.workbench import FlutterWorkbench
        self.viewer2 = LiveViewer(self.clock, self.filters)
        self.multiclip2 = MultiClipViewer(self.clock, self.filters)
        self.rangelab2 = RangeLab(self.clock, self.filters)
        self.tower2 = EventRateTower(self.clock, self.filters)
        self.spacetime2 = SpaceTimeView(self.clock, self.filters)
        self.workbench2 = FlutterWorkbench(self.clock)
        self.sandbox2 = Sandbox(self.clock)
        self.ebsviewer2 = EBSViewer()
        self.fusionlab2 = FusionLab(self.clock, self.filters)
        self.timeline2 = TimelineEditor()
        self._compare_panels = (self.viewer2, self.multiclip2, self.rangelab2, self.tower2,
                                self.spacetime2, self.workbench2, self.sandbox2, self.ebsviewer2,
                                self.fusionlab2, self.timeline2)
        self.tabs2 = self._make_tabs(self._compare_panels)
        self.tabs2.setCurrentIndex(3)        # default the right pane to a different view
        self.tabs2.currentChanged.connect(lambda *_: self.clock.pause())
        self.splitter.addWidget(self.tabs2)
        if self.rec is not None:
            for p in self._compare_panels:
                try:
                    p.set_recording(self.rec)
                except Exception as e:
                    self.status.showMessage(f"panel error: {e}")

    def _cut_selection(self):
        """Cut the In/Out selection (the bar above the timeline) to a new valid .raw clip."""
        if self.rec is None:
            QtWidgets.QMessageBox.information(self, "Cut", "Load a recording first.")
            return
        t0, t1 = self.clock.sel_t0(), self.clock.sel_t1()
        if t1 - t0 < 1e-4:
            QtWidgets.QMessageBox.information(self, "Cut", "Select a non-empty In/Out range first.")
            return
        base = (os.path.splitext(self.rec.source_path)[0] if self.rec.source_path
                else os.path.join(os.getcwd(), self.rec.name))
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Cut selection → .raw", f"{base}_clip_{t0:.2f}-{t1:.2f}.raw", "EVT raw (*.raw)")
        if not out:
            return
        if not out.lower().endswith(".raw"):
            out += ".raw"
        from gottlux.io import writer
        from gottlux.io.paths import open_in_file_browser
        from gottlux.app.uikit import with_progress
        try:
            n = with_progress(self, "Cutting selection → .raw",
                              lambda cb: writer.cut_clip(self.rec, out, t0=t0, t1=t1, progress=cb),
                              label="Writing the clip…")
            open_in_file_browser(os.path.dirname(os.path.abspath(out)))
            QtWidgets.QMessageBox.information(
                self, "Cut", f"Wrote {n:,} events ([{t0:.3f}, {t1:.3f}] s) →\n{out}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Cut", str(e))

    def _export_all(self):
        """Open the program-wide export dialog with the current session state."""
        if self.rec is None:
            QtWidgets.QMessageBox.information(self, "Export", "Load a recording first.")
            return
        from gottlux.app.export_dialog import GlobalExportDialog
        panel = self.tabs.currentWidget()
        clock = self.active_clock()                   # window from the view you're looking at
        if clock.has_selection():                     # the In/Out selection scopes the export
            t0, t1 = clock.sel_t0(), clock.sel_t1()
        else:
            t0, t1 = clock.accum_window()             # honours the accumulation direction
        # the active view's faithful render path (for the settings-accurate video output)
        render = getattr(panel, "capture_frame", None)
        sensor_wh = (panel.sensor_size() if hasattr(panel, "sensor_size") else None) \
            or (self.rec.width, self.rec.height)
        ctx = dict(rec=self.rec, t0=t0, t1=t1, accum=clock.accum, fps=clock.fps,
                   mode=self.viewer.mode.currentText(), cmap=self.viewer.cmap.currentText(),
                   expr=self.viewer.expr.currentText(),
                   flicker_map=getattr(self.workbench, "_fm", None),
                   spectrum=getattr(self.workbench, "_sp", None),
                   result=self.workbench.result(), cfg=self.workbench.cfg,
                   render=render, sensor_wh=sensor_wh, view=self._TAB_NAMES[self.tabs.currentIndex()])
        GlobalExportDialog(self, ctx).exec()

    def _open_editor(self):
        """Open the comprehensive clip editor, pre-loaded with the current recording."""
        from gottlux.app.timeline import TimelineEditorDialog
        TimelineEditorDialog(self, recordings=[self.rec] if self.rec is not None else None).exec()

    def _open_quick_cut(self):
        """Open the quick-cut tool — crop a .raw to a window without opening/decoding it."""
        from gottlux.app.quickcut import QuickCutDialog
        path = self.rec.source_path if (self.rec is not None
                                        and str(getattr(self.rec, "source_path", "")).lower().endswith(".raw")) else None
        QuickCutDialog(self, path=path).exec()

    def _open_batch_trim(self):
        """Open the batch-trim tool — shorten every .raw in a folder by one shared window."""
        from gottlux.app.batchtrim import BatchTrimDialog
        folder = (os.path.dirname(self.rec.source_path) if self.rec is not None
                  and getattr(self.rec, "source_path", None) else None)
        BatchTrimDialog(self, folder=folder).exec()

    def _capture_view(self):
        """Record the currently-shown view to a video + infographic poster + manifest."""
        if self.rec is None:
            QtWidgets.QMessageBox.information(self, "Capture", "Load a recording first.")
            return
        self.clock.pause()
        tabs = self._active_tabs()                        # the focused pane when the view is split
        panel = tabs.currentWidget()
        idx = tabs.currentIndex()
        name = self._TAB_NAMES[idx] if idx < len(self._TAB_NAMES) else "view"
        # the panel's viz area (2-D image deck or 3-D GL view), not the whole tab incl. controls
        target = getattr(panel, "glw", None) or getattr(panel, "view", None) or panel
        rec = self.rec
        fields = {
            "Recording": rec.name,
            "Sensor": f"{rec.width}×{rec.height} px · {rec.fmt}",
            "Geometry": "rotation" if rec.is_rotating else "staring",
            "Events": f"{rec.n:,}",
            "Duration": f"{rec.duration_s:.2f} s",
            "Accum (exposure)": f"{self.clock.accum * 1e3:.1f} ms",
        }
        # some tabs (Multi-clip) run on their own clock; capture against that one
        clock = panel.capture_clock() if hasattr(panel, "capture_clock") else self.clock
        cap_t0, cap_t1 = ((clock.sel_t0(), clock.sel_t1())             # honor the In/Out selection
                          if clock.has_selection() else (clock.t0, clock.t1))
        # a faithful high-res render path if the view supports it (else on-screen grab)
        render = getattr(panel, "capture_frame", None)
        sensor_wh = (panel.sensor_size() if hasattr(panel, "sensor_size") else None) \
            or (rec.width, rec.height)
        ctx = dict(rec=rec, target=target, set_cursor=clock.set_cursor, view=name,
                   t0=cap_t0, t1=cap_t1, accum=clock.accum, fps=clock.fps, fields=fields,
                   render=render, sensor_wh=sensor_wh)
        from gottlux.app.capture import ScreenCaptureDialog
        ScreenCaptureDialog(self, ctx).exec()

    def _active_tabs(self):
        """The tab widget the user is currently working in — the focused pane when split."""
        if self.tabs2 is not None and self.tabs2.isVisible():
            fw = QtWidgets.QApplication.focusWidget()
            if fw is not None and self.tabs2.isAncestorOf(fw):
                return self.tabs2
        return self.tabs

    def _active_panel(self):
        """The panel the user is currently working in (handles the split / side-by-side view)."""
        return self._active_tabs().currentWidget()

    def _active_view_widget(self):
        """The active tab's on-screen view area (its image deck / GL view), else the tab itself."""
        panel = self._active_panel()
        return getattr(panel, "glw", None) or getattr(panel, "view", None) or panel

    def _screen_record(self):
        """Open the live screen / view recorder — records the actual on-screen pixels to MP4,
        like the Windows Snipping Tool's screen record. Defaults to the whole app window (menus,
        controls and the view), so it captures the entire environment across tab switches. No
        recording needs to be loaded; the output defaults to a safe, writable location."""
        from gottlux.app.screenrec import ScreenRecordDialog
        ScreenRecordDialog(self, view_widget=self._active_view_widget(),
                           window=self, default_dir=None).exec()


def main(argv=None):
    """Entry point for ``gottlux-gui`` / ``python -m gottlux`` (with no path)."""
    argv = list(sys.argv if argv is None else argv)

    # User plugins (GOTTLUX_PLUGINS): import custom detector/analysis modules before the
    # workbench builds its detector picker. Idempotent — a launch routed through the CLI
    # (which already loaded them) does not import twice. Errors reported, never fatal.
    from gottlux.plugins import load_plugins
    load_plugins()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setOrganizationName("GottLUX")        # so QSettings (welcome opt-out) has a home
    app.setApplicationName("gottlux")
    # Adopt the persisted light/dark palette before anything paints — this touches only the
    # module constants (no QApplication, no pyqtgraph), so the splash itself comes up in the
    # user's theme rather than flipping to it a second later.
    style.load_theme()
    app.setWindowIcon(icons.app_icon())       # the painted 'event burst' mark, all sizes

    # Put a progress splash on screen *first* (it needs only PySide6), then do the heavy
    # imports + panel construction behind it. On a cold start this is the difference between
    # a minute of blank screen and a window that says it is booting, and how far along.
    splash = BootSplash()
    splash.show()
    splash.step(0.03, "Starting up…")
    style.apply_app_style(app)                # pulls in pyqtgraph; report it as a real step
    splash.step(0.06, "Loading…")
    win = MainWindow(on_progress=splash.step)
    win.show()
    splash.finish(win)
    # optional path argument; otherwise offer the bundled examples on first opening
    path = next((a for a in argv[1:] if not a.startswith("-")), None)
    if path and os.path.exists(path):
        QtCore.QTimer.singleShot(150, lambda: win.load(path))
    else:
        QtCore.QTimer.singleShot(200, win.maybe_show_welcome)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
