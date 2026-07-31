@echo off
rem ============================================================================
rem  install_offline.bat - install GottLUX with no internet connection.
rem
rem  Requires a wheel bundle in vendor\wheels\ (see docs\OFFLINE_INSTALL.md).
rem  Every dependency is installed from that folder; pip never reaches the
rem  network because --no-index is passed.
rem ============================================================================
setlocal
cd /d "%~dp0"

set "WHEELS=%~dp0vendor\wheels"

if not exist "%WHEELS%" (
    echo [!] No wheel bundle found at:
    echo     %WHEELS%
    echo.
    echo     Build one on a networked machine with:
    echo         python scripts\fetch_wheels.py
    echo     then copy the whole vendor\ folder onto this machine.
    echo     See docs\OFFLINE_INSTALL.md for details.
    pause
    exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if errorlevel 1 (
        echo [!] Python was not found on PATH.
        echo     GottLUX needs Python 3.10 or newer. The installer for Python
        echo     itself is not part of this bundle and must be present already.
        pause
        exit /b 1
    )
    set "PY=py -3"
) else (
    set "PY=python"
)

echo.
echo Installing GottLUX from "%WHEELS%" (offline)...
echo.
%PY% -m pip install --no-index --find-links "%WHEELS%" -e ".[all]"
if errorlevel 1 (
    echo.
    echo [!] Offline install failed - see the message above.
    echo     A common cause is a Python version that does not match the
    echo     wheels in the bundle. Check with:  %PY% --version
    echo     and compare against vendor\wheels\BUNDLE_INFO.txt
    pause
    exit /b 1
)

echo.
echo Verifying...
%PY% -c "import gottlux; print('GottLUX', gottlux.__version__, 'installed')"
if errorlevel 1 (
    echo [!] The package did not import cleanly.
    pause
    exit /b 1
)

echo.
set /p REG="Make GottLUX the double-click opener for .raw / .h5 / .hdf5? [y/N] "
if /i "%REG%"=="y" %PY% -m gottlux.app.quickview --register

echo.
echo Done. Start the instrument with:  gottlux-gui
echo Quick look at one recording:      gottlux-view path\to\file.raw
pause
