"""
Tests for the bundled-example discovery (gottlux.examples) and the first-launch picker
(gottlux.app.welcome) — the demo clips surfaced when the app opens with nothing loaded.

Discovery is generic: any .raw dropped into a RawExamples/ (or GOTTLUX_EXAMPLES) folder shows up,
no clip is hard-coded. The picker only *chooses* a path; the loading is the caller's job.
"""
import os

import pytest


# ----------------------------------------------------------------- discovery (pure, no Qt)
def _make_examples(tmp_path):
    files = {
        "5inch_quadcopter.raw": b"\x00" * 2048,
        "5inch_quadcopter_rotating.raw": b"\x00" * 4096,
        "Bees_50mm_249.5MB.raw": b"\x00" * 1024,
        "notes.txt": b"ignore me",                      # non-recording, must be skipped
    }
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    return tmp_path


def test_discovery_via_env_override(tmp_path, monkeypatch):
    from gottlux import examples as ex
    d = _make_examples(tmp_path)
    monkeypatch.setenv("GOTTLUX_EXAMPLES", str(d))
    assert os.path.abspath(ex.examples_dir()) == os.path.abspath(str(d))
    items = ex.list_examples()
    names = [os.path.basename(e.path) for e in items]
    assert names == ["5inch_quadcopter.raw", "5inch_quadcopter_rotating.raw", "Bees_50mm_249.5MB.raw"]
    assert "notes.txt" not in names                     # only recordings are listed
    assert ex.has_examples()


def test_rotation_hint_and_titles(tmp_path, monkeypatch):
    from gottlux import examples as ex
    _make_examples(tmp_path)
    monkeypatch.setenv("GOTTLUX_EXAMPLES", str(tmp_path))
    by_name = {os.path.basename(e.path): e for e in ex.list_examples()}
    assert by_name["5inch_quadcopter_rotating.raw"].is_rotating is True
    assert by_name["5inch_quadcopter.raw"].is_rotating is False
    # underscores → spaces; a trailing size token baked into the name is stripped from the title
    assert by_name["5inch_quadcopter.raw"].title == "5inch quadcopter"
    assert by_name["Bees_50mm_249.5MB.raw"].title == "Bees 50mm"
    # the human size still appears in the detail line
    assert "KB" in by_name["5inch_quadcopter.raw"].detail or "MB" in by_name["5inch_quadcopter.raw"].detail


def test_no_examples_when_dir_absent(tmp_path, monkeypatch):
    from gottlux import examples as ex
    empty = tmp_path / "nope"
    monkeypatch.setenv("GOTTLUX_EXAMPLES", str(empty))   # points at a non-existent dir
    assert ex.examples_dir() is None
    assert ex.list_examples() == []
    assert ex.has_examples() is False


def test_human_size_units():
    from gottlux.examples import _human_size
    assert _human_size(512) == "512 B"
    assert _human_size(2048) == "2 KB"
    assert _human_size(47_000_000).endswith("MB")
    assert _human_size(3_000_000_000).endswith("GB")


# ----------------------------------------------------------------- the picker (offscreen Qt)
os_environ_set = os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from gottlux.examples import Example  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_welcome_dialog_lists_and_selects(app, tmp_path):
    from gottlux.app.welcome import WelcomeDialog
    items = [
        Example(path=str(tmp_path / "a.raw"), title="Quad", detail="45 MB · staring sensor", is_rotating=False),
        Example(path=str(tmp_path / "b.raw"), title="Quad spin", detail="74 MB · rotating sensor", is_rotating=True),
    ]
    dlg = WelcomeDialog(None, items=items)
    assert dlg.list.count() == 2
    assert dlg.open_btn.isEnabled()
    dlg.list.setCurrentRow(1)
    dlg._open_selected()                                 # acts like clicking "Open example"
    assert dlg.chosen_path == str(tmp_path / "b.raw")


def test_welcome_dialog_empty_disables_open(app):
    from gottlux.app.welcome import WelcomeDialog
    dlg = WelcomeDialog(None, items=[])
    assert not dlg.open_btn.isEnabled()
    assert dlg.chosen_path is None


def test_main_window_examples_menu_populates(app, tmp_path, monkeypatch):
    pytest.importorskip("pyqtgraph")
    from gottlux.app.main import MainWindow
    (tmp_path / "demo_clip.raw").write_bytes(b"\x00" * 1024)
    monkeypatch.setenv("GOTTLUX_EXAMPLES", str(tmp_path))
    win = MainWindow()
    win._populate_examples_menu()
    labels = [a.text() for a in win.examples_menu.actions() if a.text()]
    assert any("demo clip" in s for s in labels)
    assert any("Open examples folder" in s for s in labels)
