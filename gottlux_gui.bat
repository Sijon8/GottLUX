@echo off
REM GottLUX GUI launcher — opens the tabbed dashboard (Live viewer · Multi-clip · tower · 3-D · workbench · sandbox).
setlocal
set "HERE=%~dp0"
set "PYTHONPATH=%HERE%;%PYTHONPATH%"
python -m gottlux.app.main %*
endlocal
