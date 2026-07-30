"""
Tests for the painted vector icon system (gottlux.app.icons): every named icon renders
non-empty pixels at 16 px and at devicePixelRatio 2 (the fractional-DPI crispness fix),
the icon() cache returns the same QIcon, the palette rules hold (muted when disabled),
and no emoji/dingbat glyph "icons" remain anywhere in the GUI sources.
"""
import glob
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _opaque_pixels(pm):
    """How many pixels of the pixmap were actually painted (alpha > 0)."""
    img = pm.toImage()
    return sum(1 for y in range(img.height()) for x in range(img.width())
               if img.pixelColor(x, y).alpha() > 0)


def test_required_icon_set_present():
    from gottlux.app import icons
    required = {"play", "pause", "stop", "record", "capture", "film", "save", "export",
                "sync", "cut", "split", "add", "close", "arrow-left", "arrow-right",
                "chevron-right", "target", "sun", "moon"}
    assert required <= set(icons.ICON_NAMES)


def test_every_icon_renders_at_16px(app):
    from gottlux.app import icons
    for name in icons.ICON_NAMES:
        pm = icons.icon(name).pixmap(QtCore.QSize(16, 16))
        assert not pm.isNull(), name
        assert pm.width() == 16 and pm.height() == 16, name
        assert _opaque_pixels(pm) > 8, f"{name!r} painted (almost) nothing at 16 px"


def test_every_icon_renders_at_devicepixelratio_2(app):
    """At 200 % scaling the pixmap must hold 2× the device pixels and carry the ratio —
    that (not font fallback) is what keeps the marks crisp on fractional-DPI displays."""
    from gottlux.app import icons
    for name in icons.ICON_NAMES:
        pm = icons.icon(name).pixmap(QtCore.QSize(16, 16), 2.0)
        assert pm.width() == 32 and pm.height() == 32, name          # device pixels
        assert pm.devicePixelRatio() == pytest.approx(2.0), name     # logical 16 px
        assert _opaque_pixels(pm) > 8, name


def test_icon_cache_returns_same_qicon(app):
    from gottlux.app import icons
    assert icons.icon("play") is icons.icon("play")
    assert icons.icon("play") is not icons.icon("pause")
    assert icons.icon("close", color="#ffffff") is icons.icon("close", color="#ffffff")
    assert icons.icon("close", color="#ffffff") is not icons.icon("close")


def test_unknown_icon_name_raises(app):
    from gottlux.app import icons
    with pytest.raises(KeyError):
        icons.icon("definitely-not-an-icon")


def test_disabled_icons_use_muted_palette(app):
    from gottlux.app import icons, style
    pm = icons.icon("play").pixmap(QtCore.QSize(16, 16), QtGui.QIcon.Disabled, QtGui.QIcon.Off)
    img = pm.toImage()
    cols = {img.pixelColor(x, y).name() for y in range(img.height()) for x in range(img.width())
            if img.pixelColor(x, y).alpha() > 200}
    assert cols == {style.MUTED.lower()}


def test_record_icon_is_alarm_red(app):
    from gottlux.app import icons, style
    pm = icons.icon("record").pixmap(QtCore.QSize(16, 16))
    img = pm.toImage()
    assert img.pixelColor(8, 8).name() == style.BAD.lower()


def test_color_override_is_honoured(app):
    from gottlux.app import icons
    pm = icons.icon("record", color="#ff3b3b").pixmap(QtCore.QSize(16, 16))
    assert pm.toImage().pixelColor(8, 8).name() == "#ff3b3b"


def test_app_icon_renders(app):
    from gottlux.app import icons
    pm = icons.app_icon().pixmap(QtCore.QSize(32, 32))
    assert not pm.isNull()
    assert _opaque_pixels(pm) > 400          # the mark sits on a filled dark plate


# ------------------------------------------------------------------ no glyph "icons" remain
# The codepoints that used to be set as button TEXT and rendered through the platform's
# font-fallback lottery (emoji bitmaps on Windows, tofu on Linux). None may reappear in
# the GUI sources — every one now has a painted replacement in gottlux.app.icons.
_BANNED = {
    0x1F3AC: "clapper board emoji",
    0x1F4BE: "floppy disk emoji",
    0x23FA: "record symbol",
    0x2702: "black scissors",
    0x2704: "white scissors",
    0x25B6: "play triangle",
    0x275A: "heavy vertical bar (pause)",
    0x2715: "multiplication x (close)",
    0x271A: "heavy plus",
    0x25C9: "fisheye (target)",
    0x27E6: "white square bracket",
    0x27F3: "clockwise gapped arrow (sync)",
    0x2913: "down arrow to bar (export)",
}


def test_no_glyph_icons_left_in_gui_sources():
    import gottlux.app
    root = os.path.dirname(os.path.abspath(gottlux.app.__file__))
    offenders = []
    for path in sorted(glob.glob(os.path.join(root, "*.py"))):
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                for ch in line:
                    if ord(ch) in _BANNED:
                        offenders.append(f"{os.path.basename(path)}:{lineno} "
                                         f"U+{ord(ch):04X} ({_BANNED[ord(ch)]})")
    assert not offenders, "glyph 'icons' remain in GUI sources:\n" + "\n".join(offenders)


def test_generated_icon_files_exist_and_are_valid():
    """scripts/make_icons.py output is committed: a multi-size .ico + a 256 px PNG."""
    import gottlux
    repo = os.path.dirname(os.path.dirname(os.path.abspath(gottlux.__file__)))
    ico = os.path.join(repo, "packaging", "windows", "gottlux.ico")
    png = os.path.join(repo, "packaging", "gottlux_icon.png")
    assert os.path.exists(ico) and os.path.exists(png)
    with open(ico, "rb") as f:
        head = f.read(6)
    import struct
    reserved, kind, count = struct.unpack("<HHH", head)
    assert (reserved, kind) == (0, 1) and count >= 7     # icon container, 16..256 entries
    with open(png, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"
