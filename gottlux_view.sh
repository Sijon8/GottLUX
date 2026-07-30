#!/usr/bin/env sh
# GottLUX quick viewer - the lightweight, fast single-view player for one .raw recording.
#   ./gottlux_view.sh clip.raw        open a clip in the quick viewer
#   ./gottlux_view.sh --register      make double-clicking a .raw open this viewer (per-user, XDG)
#   ./gottlux_view.sh --unregister    undo the association
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" exec python3 -m gottlux.app.quickview "$@"
