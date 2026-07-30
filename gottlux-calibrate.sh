#!/usr/bin/env sh
# GottLUX calibration: map a tracker's relative-distance proxy to metres, or measure the
# dual-EBS inter-camera time offset.  Examples:
#   ./gottlux-calibrate.sh <tag>_tracks.csv --near 30 --far 300
#   ./gottlux-calibrate.sh --intercam path/to/capture_folder
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" exec python3 -m gottlux.run.calibrate "$@"
