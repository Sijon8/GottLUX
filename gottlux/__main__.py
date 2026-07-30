"""``python -m gottlux`` — delegate to the CLI, which decides GUI vs headless.

The single source of dispatch truth is :func:`gottlux.cli.main`: it handles the
``--list_*`` flags, opens the GUI when given no path (or ``--gui``), and otherwise runs the
headless pipeline. Keeping all of that in one place avoids subtle argument-routing bugs.
"""
from __future__ import annotations

import sys

from gottlux.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv))
