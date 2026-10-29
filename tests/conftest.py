"""Shared pytest fixtures: synthetic recordings with known flutter signatures, and the
locator every export test uses to find the provenance folder an export wrote."""
import glob
import json
import os

import numpy as np
import pytest

from gottlux.synthetic import FlutterTarget, synthetic_scene


@pytest.fixture(autouse=True, scope="session")
def _private_colour_schemes(tmp_path_factory):
    """Point the colour registry at a file of the session's own, so no test reads or
    writes the user's real colour schemes (gottlux.core.colors.schemes_path)."""
    path = tmp_path_factory.mktemp("colors") / "colors.json"
    old = os.environ.get("GOTTLUX_COLORS")
    os.environ["GOTTLUX_COLORS"] = str(path)
    from gottlux.core import colors
    colors.reload()
    try:                                       # and remembered GUI choices (export options…)
        from PySide6 import QtCore             # go to an INI file of the session's own
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
        QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope,
                                 str(tmp_path_factory.mktemp("qsettings")))
    except ImportError:
        pass
    yield path
    if old is None:
        os.environ.pop("GOTTLUX_COLORS", None)
    else:
        os.environ["GOTTLUX_COLORS"] = old


@pytest.fixture(autouse=True)
def _free_qt_objects_on_the_gui_thread():
    """After each test, delete the Qt objects it left behind — on the GUI thread.

    Windows and panels a test drops often sit in reference cycles (signal lambdas holding
    ``self``), so CPython frees them only at its next cyclic garbage collection — which can run
    on *any* thread that happens to allocate. When that is a later test's worker thread (the
    screen recorder's encoder, an export worker), Qt destructors run off the GUI thread and the
    process dies with an access violation / abort: the intermittent abort of the full suite
    (docs/ISSUES.md GL-26)."""
    yield
    import gc
    import sys
    widgets = sys.modules.get("PySide6.QtWidgets")
    app = widgets.QApplication.instance() if widgets else None
    if app is None:
        return
    app.processEvents()                       # run pending deleteLater()s
    gc.collect()                              # free cyclic Qt wrappers here, on the GUI thread
    app.processEvents()


def pytest_sessionfinish(session, exitstatus):
    """Let a decode a GUI test left running finish before the interpreter exits: a running
    QThread torn down at exit aborts the process (gottlux.app.loader.Worker)."""
    import sys
    loader = sys.modules.get("gottlux.app.loader")
    if loader is not None:
        loader.wait_for_workers(timeout_s=120)


@pytest.fixture(scope="session")
def flutter_rec():
    """A 1.5 s scene with one planted 200 Hz target crossing the centre + noise + clutter."""
    rec, truth = synthetic_scene(
        duration_s=1.5,
        targets=[FlutterTarget(flutter_hz=200.0, x0=50, y0=160, x1=260, y1=160,
                               harmonics=(1.0, 0.5, 0.25))],
        noise_rate_hz=30_000, static_clutter=40, seed=7)
    return rec, truth


@pytest.fixture(scope="session")
def quiet_rec():
    """A scene with only background noise (no planted target)."""
    rec, _ = synthetic_scene(duration_s=1.0, targets=[], noise_rate_hz=20_000,
                             static_clutter=0, seed=3)
    return rec


class ExportFolder:
    """One export's provenance folder, with its documents already parsed.

    Every GUI export path writes ``<parent>/<stem>_export_<UTC-stamp>/`` holding the
    artifact under its original file name beside ``README.md`` and ``provenance.json``
    (:mod:`gottlux.run.export_provenance`), so a test is handed the folder rather than the
    loose path the save dialog returned: ``.artifact`` is the produced file inside it,
    ``.readme`` its README text, ``.record`` the parsed ``provenance.json``, and
    ``.names`` every file in the folder.
    """

    def __init__(self, folder, out):
        self.folder = folder
        self.artifact = os.path.join(folder, os.path.basename(out))
        self.names = sorted(os.listdir(folder))
        with open(os.path.join(folder, "README.md"), encoding="utf-8") as f:
            self.readme = f.read()
        with open(os.path.join(folder, "provenance.json"), encoding="utf-8") as f:
            self.record = json.load(f)

    @property
    def sources(self):
        return self.record["sources"]

    @property
    def usage(self):
        return self.record["usage"]

    def spec_name(self):
        """The composition spec file in the folder, if this export path saved one."""
        return next((n for n in self.names if n.endswith(".gottlux-canvas.json")), None)


@pytest.fixture
def exported():
    """Locate the provenance folder an export wrote for a chosen output path.

    Returns a callable: ``exported(out)`` finds ``<stem>_export_*`` beside *out*, asserts
    the artifact and both documents are there, and hands back an :class:`ExportFolder`.
    ``exported(out, exists=False)`` asserts instead that no export folder was written —
    what a refused export must leave behind.
    """
    def locate(out, exists=True):
        stem = os.path.splitext(os.path.basename(str(out)))[0]
        parent = os.path.dirname(os.path.abspath(str(out)))
        found = sorted(glob.glob(os.path.join(parent, f"{stem}_export_*")))
        if not exists:
            assert not found, f"an export folder was written for {out}: {found}"
            return None
        assert len(found) == 1, f"expected one export folder for {out}, found {found}"
        folder = ExportFolder(found[0], out)
        assert os.path.exists(folder.artifact), \
            f"the artifact is not inside the export folder: {folder.names}"
        assert {"README.md", "provenance.json"} <= set(folder.names)
        assert not os.path.exists(str(out)), \
            "the artifact was written loose beside the folder as well"
        return folder
    return locate
