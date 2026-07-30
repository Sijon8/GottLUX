"""
make_icons.py — render the painted GottLUX app mark to the shipped icon files.

The mark itself lives in :mod:`gottlux.app.icons` (the 'event burst': a bright dot with two
arc rings on the dark instrument plate) and is painted fresh at every size, so nothing here
is hand-pixelled. This script renders it to:

* ``packaging/windows/gottlux.ico`` — a multi-size Windows icon (16/24/32/48/64/128/256),
  each entry stored as PNG (valid on Windows Vista+). Used as the window icon by installers
  and as the ``.raw`` file-type icon via ``file_assoc.py``'s ProgID ``DefaultIcon``.
* ``packaging/gottlux_icon.png`` — a single 256 px PNG (Linux desktop entries, docs).

The .ico container is written directly (ICONDIR + PNG-compressed entries) so no extra
image library is needed. Runs headless::

    python scripts/make_icons.py
"""
from __future__ import annotations

import os
import struct
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
PNG_SIZE = 256


def _mark_png_bytes(size: int) -> bytes:
    """The app mark rendered at *size*×*size*, as PNG bytes."""
    from PySide6 import QtCore, QtGui
    from gottlux.app import icons
    img = QtGui.QImage(size, size, QtGui.QImage.Format_ARGB32_Premultiplied)
    img.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(img)
    icons.VectorIconEngine("gottlux").paint(p, QtCore.QRect(0, 0, size, size),
                                            QtGui.QIcon.Normal, QtGui.QIcon.Off)
    p.end()
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


def write_ico(path: str, sizes=ICO_SIZES):
    """Pack one PNG per size into a multi-size .ico (ICONDIR + PNG entries)."""
    pngs = [(s, _mark_png_bytes(s)) for s in sizes]
    header = struct.pack("<HHH", 0, 1, len(pngs))              # reserved, type=icon, count
    entries, blobs = b"", b""
    offset = len(header) + 16 * len(pngs)                      # first image offset
    for s, data in pngs:
        entries += struct.pack("<BBBBHHII",
                               s % 256, s % 256,               # width/height (0 means 256)
                               0, 0,                           # palette, reserved
                               1, 32,                          # planes, bit depth
                               len(data), offset)
        blobs += data
        offset += len(data)
    with open(path, "wb") as f:
        f.write(header + entries + blobs)


def main():
    from PySide6 import QtGui
    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([])
    assert app is not None

    ico_path = os.path.join(_REPO, "packaging", "windows", "gottlux.ico")
    png_path = os.path.join(_REPO, "packaging", "gottlux_icon.png")
    os.makedirs(os.path.dirname(ico_path), exist_ok=True)

    write_ico(ico_path)
    with open(png_path, "wb") as f:
        f.write(_mark_png_bytes(PNG_SIZE))
    print(f"wrote {ico_path} ({os.path.getsize(ico_path):,} bytes, sizes {ICO_SIZES})")
    print(f"wrote {png_path} ({os.path.getsize(png_path):,} bytes, {PNG_SIZE} px)")


if __name__ == "__main__":
    main()
