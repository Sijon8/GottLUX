@echo off
REM GottLUX launcher (no pip install needed). No path -> GUI; a path -> headless analysis.
setlocal
set "HERE=%~dp0"
set "PYTHONPATH=%HERE%;%PYTHONPATH%"
python -m gottlux %*
endlocal
