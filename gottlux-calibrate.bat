@echo off
REM GottLUX calibration: map a tracker's relative-distance proxy to metres, or measure the
REM dual-EBS inter-camera time offset.  Examples:
REM   gottlux-calibrate <tag>_tracks.csv --near 30 --far 300
REM   gottlux-calibrate --intercam path\to\capture_folder
setlocal
set "HERE=%~dp0"
set "PYTHONPATH=%HERE%;%PYTHONPATH%"
python -m gottlux.run.calibrate %*
endlocal
