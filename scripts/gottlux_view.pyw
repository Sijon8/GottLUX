"""
gottlux_view.pyw — windowless launcher for the GottLUX quick viewer (the .raw file handler).

The Windows ``.raw`` association points ``pythonw.exe`` at this file with the clicked path as
its argument. Running as a plain script (not ``-m``), it adds the repository root to
``sys.path`` so ``import gottlux`` works from a source checkout (harmless if the package is
already installed), then hands off to :func:`gottlux.app.quickview.main`.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gottlux.app.quickview import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv))
