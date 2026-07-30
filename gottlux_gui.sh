#!/usr/bin/env sh
# GottLUX GUI launcher - opens the tabbed dashboard (Live viewer / Multi-clip / tower / 3-D / workbench / sandbox).
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" exec python3 -m gottlux.app.main "$@"
