"""
Cross-platform (Linux-support) unit tests: the Windows-only MAX_PATH check must be a no-op
on POSIX, the XDG desktop/mime content generators must produce spec-shaped files identical
to the static copies shipped in packaging/linux/, the screenrec XDG videos-dir parser must
handle real user-dirs.dirs content, and the POSIX launchers must stay LF + point at the
right modules. Everything here is pure — no registry writes, no real xdg tools.
"""
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------- paths.fits / platform_root
def test_fits_is_windows_only():
    """A 400-char path 'fits' anywhere except Windows — the 259-char limit is NT-specific."""
    from gottlux.io.paths import MAX_PATH, fits
    long_path = os.path.join(os.path.abspath(os.sep), "x" * 400)
    assert len(long_path) > MAX_PATH
    assert fits(long_path, windows=False) is True
    assert fits(long_path, windows=True) is False
    assert fits(long_path, headroom=16, windows=False) is True
    # the default mirrors the running platform
    assert fits(long_path) is (os.name != "nt")


def test_fits_short_path_fits_everywhere():
    from gottlux.io.paths import fits
    assert fits(os.path.join(os.path.abspath(os.sep), "short.raw"), windows=True) is True


def test_posix_cache_root_honours_xdg_cache_home(monkeypatch):
    """The POSIX relocation root is $XDG_CACHE_HOME/gottlux, defaulting to ~/.cache/gottlux —
    never inside the package tree."""
    from gottlux.io import paths
    monkeypatch.setenv("XDG_CACHE_HOME", os.path.join("/tmp", "xdgcache"))
    assert paths._posix_cache_root() == os.path.join("/tmp", "xdgcache", "gottlux")
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert paths._posix_cache_root() == os.path.join(os.path.expanduser("~"), ".cache",
                                                     "gottlux")
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(paths.__file__)))
    assert not paths._posix_cache_root().startswith(pkg_root)


@pytest.mark.skipif(os.name == "nt", reason="POSIX branch of platform_root")
def test_platform_root_is_xdg_cache_on_posix():
    from gottlux.io.paths import _posix_cache_root, platform_root
    assert platform_root() == _posix_cache_root()


def test_windows_cache_root_honours_localappdata(monkeypatch):
    r"""The Windows relocation root is %LOCALAPPDATA%\gottlux, defaulting to ~\.gottlux —
    never inside the package tree (a site-packages install may be read-only)."""
    from gottlux.io import paths
    lad = os.path.join(os.path.abspath(os.sep), "lad")
    monkeypatch.setenv("LOCALAPPDATA", lad)
    assert paths._windows_cache_root() == os.path.join(lad, "gottlux")
    monkeypatch.delenv("LOCALAPPDATA")
    assert paths._windows_cache_root() == os.path.join(os.path.expanduser("~"), ".gottlux")
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(paths.__file__)))
    assert not os.path.normcase(paths._windows_cache_root()).startswith(os.path.normcase(pkg_root))


@pytest.mark.skipif(os.name != "nt", reason="Windows branch of platform_root")
def test_platform_root_windows_under_localappdata_not_package():
    r"""On Windows platform_root() lives under %LOCALAPPDATA% (user-writable), NOT under the
    gottlux package tree — a site-packages install must never receive relocated caches."""
    from gottlux.io import paths
    root = os.path.normcase(paths.platform_root())
    lad = os.environ.get("LOCALAPPDATA")
    assert lad, "LOCALAPPDATA expected on Windows"
    assert root == os.path.normcase(os.path.join(lad, "gottlux"))
    assert root.startswith(os.path.normcase(lad) + os.sep)
    pkg_root = os.path.normcase(os.path.dirname(os.path.dirname(os.path.abspath(paths.__file__))))
    assert not root.startswith(pkg_root)


# --------------------------------------------------------------------- XDG content generators
def test_xdg_mime_xml_shape():
    from gottlux.app.file_assoc import MIME_TYPE, MIME_TYPE_H5, xdg_mime_xml
    xml = xdg_mime_xml()
    assert xml.startswith('<?xml version="1.0"')
    assert 'xmlns="http://www.freedesktop.org/standards/shared-mime-info"' in xml
    assert f'<mime-type type="{MIME_TYPE}">' in xml
    assert f'<mime-type type="{MIME_TYPE_H5}">' in xml
    # low weight on purpose: .raw collides with camera photo formats, and generic
    # .h5/.hdf5 files belong to many other tools
    assert '<glob pattern="*.raw" weight="40"/>' in xml
    assert '<glob pattern="*.h5" weight="40"/>' in xml
    assert '<glob pattern="*.hdf5" weight="40"/>' in xml
    import xml.etree.ElementTree as ET
    ET.fromstring(xml)                                   # well-formed


def test_xdg_desktop_entry_shape():
    from gottlux.app.file_assoc import MIME_TYPE, MIME_TYPE_H5, xdg_desktop_entry
    txt = xdg_desktop_entry()
    lines = txt.splitlines()
    assert lines[0] == "[Desktop Entry]"
    entries = dict(l.split("=", 1) for l in lines[1:] if l)
    assert entries["Type"] == "Application"
    assert entries["Exec"] == "gottlux-view %f"
    assert entries["Terminal"] == "false"
    assert entries["MimeType"] == f"{MIME_TYPE};{MIME_TYPE_H5};"
    assert entries["Categories"] == "Science;Graphics;"
    # a custom launch command (source checkout) lands verbatim in Exec
    assert "Exec=/opt/py/bin/python -m gottlux.app.quickview %f" in \
        xdg_desktop_entry("/opt/py/bin/python -m gottlux.app.quickview %f")


def test_packaged_linux_files_match_generators():
    """packaging/linux/ ships exactly what file_assoc generates — one source of truth."""
    from gottlux.app.file_assoc import xdg_desktop_entry, xdg_mime_xml
    with open(os.path.join(REPO, "packaging", "linux", "gottlux-raw.xml"),
              encoding="utf-8", newline="") as f:
        assert f.read() == xdg_mime_xml()
    with open(os.path.join(REPO, "packaging", "linux", "gottlux-view.desktop"),
              encoding="utf-8", newline="") as f:
        assert f.read() == xdg_desktop_entry()


def test_xdg_register_paths_land_in_data_home(monkeypatch, tmp_path):
    """The register/unregister targets follow $XDG_DATA_HOME (default ~/.local/share)."""
    from gottlux.app import file_assoc
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    mime_path, desktop_path = file_assoc._xdg_paths()
    assert mime_path == str(tmp_path / "mime" / "packages" / "gottlux-raw.xml")
    assert desktop_path == str(tmp_path / "applications" / "gottlux-view.desktop")
    monkeypatch.delenv("XDG_DATA_HOME")
    mime_path, _ = file_assoc._xdg_paths()
    assert mime_path == os.path.join(os.path.expanduser("~"), ".local", "share",
                                     "mime", "packages", "gottlux-raw.xml")


# --------------------------------------------------------------------- screenrec XDG videos dir
def test_xdg_videos_dir_parses_and_expands_home(tmp_path):
    from gottlux.app.screenrec import _xdg_videos_dir
    cfg = tmp_path / "user-dirs.dirs"
    cfg.write_text(
        '# This file is written by xdg-user-dirs-update\n'
        'XDG_DESKTOP_DIR="$HOME/Desktop"\n'
        'XDG_VIDEOS_DIR="$HOME/My Videos"\n',
        encoding="utf-8")
    # $HOME expands literally (the file's separators are POSIX '/')
    assert _xdg_videos_dir(str(cfg)) == os.path.expanduser("~") + "/My Videos"


def test_xdg_videos_dir_absolute_and_disabled(tmp_path):
    """Absolute entries pass through; a relative entry means 'disabled' per the spec."""
    from gottlux.app.screenrec import _xdg_videos_dir
    cfg = tmp_path / "user-dirs.dirs"
    cfg.write_text('XDG_VIDEOS_DIR="/data/vids"\n', encoding="utf-8")
    assert _xdg_videos_dir(str(cfg)) == "/data/vids"
    cfg.write_text('XDG_VIDEOS_DIR="Videos"\n', encoding="utf-8")
    assert _xdg_videos_dir(str(cfg)) is None
    assert _xdg_videos_dir(str(tmp_path / "missing")) is None    # no config file → None


# --------------------------------------------------------------------- launchers + packaging
@pytest.mark.parametrize("script, module", [
    ("gottlux.sh", "gottlux"),
    ("gottlux_gui.sh", "gottlux.app.main"),
    ("gottlux_view.sh", "gottlux.app.quickview"),
    ("gottlux-calibrate.sh", "gottlux.run.calibrate"),
])
def test_posix_launchers_are_lf_and_target_the_right_module(script, module):
    with open(os.path.join(REPO, script), "rb") as f:
        data = f.read()
    assert b"\r" not in data, f"{script} must use LF line endings"
    text = data.decode("utf-8")
    assert text.startswith("#!/usr/bin/env sh\n")
    assert f"python3 -m {module} " in text


def test_gui_desktop_entry_shape():
    with open(os.path.join(REPO, "packaging", "linux", "gottlux-gui.desktop"),
              encoding="utf-8") as f:
        txt = f.read()
    assert txt.startswith("[Desktop Entry]\n")
    assert "Exec=gottlux-gui\n" in txt and "Terminal=false\n" in txt
