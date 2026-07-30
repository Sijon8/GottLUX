@echo off
REM GottLUX quick viewer — the lightweight, fast single-view player for one .raw recording.
REM   gottlux_view.bat clip.raw        open a clip in the quick viewer
REM   gottlux_view.bat --register      make double-clicking a .raw open this viewer (per-user)
REM   gottlux_view.bat --unregister    undo the association
setlocal
set "HERE=%~dp0"
set "PYTHONPATH=%HERE%;%PYTHONPATH%"
python -m gottlux.app.quickview %*
endlocal
