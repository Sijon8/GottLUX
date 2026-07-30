"""
gottlux.app — the interactive PySide6 + pyqtgraph desktop instrument.

Launch with the ``gottlux-gui`` console script or ``python -m gottlux`` (no path). Importing
this subpackage pulls in Qt, so the headless core never imports it implicitly.

* :func:`~gottlux.app.main.main`         the application entry point
* :class:`~gottlux.app.main.MainWindow`  the tabbed main window
* :class:`~gottlux.app.viewer.LiveViewer`        scrub/play event frames
* :class:`~gottlux.app.spacetime.SpaceTimeView`  the 3-D event cloud
* :class:`~gottlux.app.workbench.FlutterWorkbench` the detection tuning lab
"""
from __future__ import annotations

__all__ = ["main"]


def main(argv=None):
    """Lazy proxy to :func:`gottlux.app.main.main` (keeps Qt import out of package import)."""
    from gottlux.app.main import main as _main
    return _main(argv)
