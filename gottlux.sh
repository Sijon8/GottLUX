#!/usr/bin/env sh
# GottLUX launcher (no pip install needed). No path -> GUI; a path -> headless analysis.
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" exec python3 -m gottlux "$@"
