#!/usr/bin/env sh
# =============================================================================
#  install_offline.sh - install GottLUX with no internet connection.
#
#  Requires a wheel bundle in vendor/wheels/ (see docs/OFFLINE_INSTALL.md).
#  Every dependency is installed from that folder; pip never reaches the
#  network because --no-index is passed.
# =============================================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
WHEELS="$HERE/vendor/wheels"

if [ ! -d "$WHEELS" ]; then
    echo "[!] No wheel bundle found at:"
    echo "    $WHEELS"
    echo
    echo "    Build one on a networked machine with:"
    echo "        python3 scripts/fetch_wheels.py"
    echo "    then copy the whole vendor/ folder onto this machine."
    echo "    See docs/OFFLINE_INSTALL.md for details."
    exit 1
fi

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    PY=python
fi
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "[!] No python interpreter was found. GottLUX needs Python 3.10 or newer,"
    echo "    which is not part of this bundle and must already be installed."
    exit 1
fi

# On Windows the .bat installer is the supported path: a POSIX shell here is Git Bash
# or WSL, and each brings its own interpreter and path translation.
case "$("$PY" -c 'import sys; print(sys.platform)' 2>/dev/null)" in
    win32|cygwin)
        echo "[i] This is a Windows interpreter. install_offline.bat is the supported"
        echo "    installer on Windows; continuing, but use the .bat if this fails."
        echo
        ;;
esac

# Preflight: report an incompatible bundle clearly rather than leaving pip to say
# "no matching distribution found ... (from versions: none)".
if ! "$PY" "$HERE/scripts/check_bundle.py" "$WHEELS"; then
    exit 1
fi

echo
echo "Installing GottLUX from $WHEELS (offline)..."
echo
"$PY" -m pip install --no-index --find-links "$WHEELS" -e "$HERE[all]"

echo
echo "Verifying..."
"$PY" -c "import gottlux; print('GottLUX', gottlux.__version__, 'installed')"

echo
echo "Done. Start the instrument with:  gottlux-gui"
echo "Quick look at one recording:      gottlux-view path/to/file.raw"
echo
echo "The Qt GUI additionally needs system libraries that pip cannot supply:"
echo "    sudo apt-get install -y libegl1 libgl1 libxkbcommon0 libxcb-cursor0"
echo "On an offline machine these must come from the distribution's own media."
