"""
Tests for the system-wide light/dark theme (gottlux.app.style + its consumers).

Covers the switch itself — the palette rebound in place, the application stylesheet
regenerated, the choice persisted and read back — and the parts that only *look* right if
the sweep to paint-time palette reads actually happened: the painted icon cache handing out
a different pixmap per theme, the main window's toolbar toggle flipping the persisted
choice, and a real render of the Timeline preview, its lanes and a legend coming out light
under the light theme.

Everything runs offscreen. The persisted setting is redirected to an in-memory stand-in for
``QSettings`` so no test writes to the user's real settings, and the dark theme is restored
after every test — the QApplication is shared with the rest of the suite.

Two deliberate economies keep the module fast. ``apply_theme(None, …)`` does every part of a
switch (palette, stylesheet string, pyqtgraph, icon cache, persistence, notification) except
the application step, which most assertions do not need; and where that step *is* the point,
it is aimed at a scoped widget or recorded off the shared ``QApplication``. Restyling the
real application repolishes every widget the whole test session has left alive — thousands,
by the time this module runs — which is a property of the shared process, not of the theme
switch a single-window app performs.
"""
import glob
import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from gottlux.app import icons, style  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _FakeSettings:
    """A ``QSettings`` stand-in backed by a plain dict (one store per test)."""

    def __init__(self, store):
        self._store = store

    def value(self, key, default=None, **_kw):
        return self._store.get(key, default)

    def setValue(self, key, value):
        self._store[key] = value


@pytest.fixture(autouse=True)
def settings_store(monkeypatch, app):
    """Redirect the persisted choice into memory, and leave the suite on the dark theme."""
    store = {}
    monkeypatch.setattr(style, "settings", lambda: _FakeSettings(store))
    yield store
    style.load_theme("dark")


@pytest.fixture
def app_sheets(monkeypatch, app):
    """Record what a switch hands the shared ``QApplication`` (see the module docstring)."""
    seen = []
    monkeypatch.setattr(app, "setStyleSheet", seen.append, raising=False)
    return seen


def _lightness(image, x, y):
    return image.pixelColor(x, y).lightnessF()


# ------------------------------------------------------------------ the palette table
def test_both_themes_define_every_palette_role():
    assert set(style.theme_names()) == {"dark", "light"}
    for name, palette in style.THEMES.items():
        assert set(palette) == set(style.PALETTE_KEYS), name
        for role, value in palette.items():
            assert QtGui.QColor(value).isValid(), f"{name}/{role} = {value!r}"


def test_dark_theme_is_the_original_instrument_palette():
    """The dark palette is the historical one, verbatim — the switch adds a theme, it does
    not restyle the instrument."""
    dark = style.THEMES["dark"]
    assert (dark["BG"], dark["BG2"], dark["PANEL"]) == ("#0e1116", "#161b22", "#1b2230")
    assert (dark["FG"], dark["MUTED"]) == ("#d7dde7", "#8b97a7")
    assert (dark["ACCENT"], dark["ACCENT2"]) == ("#39c5cf", "#f78166")
    assert dark["ACCENT_TEXT"] == "#04181a" and dark["BORDER"] == "#2b3340"


def test_light_theme_is_light_and_keeps_the_accent_hue():
    light, dark = style.THEMES["light"], style.THEMES["dark"]
    assert QtGui.QColor(light["BG"]).lightnessF() > 0.85     # a genuinely light page
    assert QtGui.QColor(light["FG"]).lightnessF() < 0.25     # dark text on it
    # same accent family, just taken down until it can carry text on a white panel
    assert abs(QtGui.QColor(light["ACCENT"]).hue() - QtGui.QColor(dark["ACCENT"]).hue()) <= 12
    # the dim roles have to stay readable, not merely visible
    for role in ("MUTED", "ACCENT", "ACCENT2", "BAD"):
        assert QtGui.QColor(light[role]).lightnessF() < 0.55, role


# ------------------------------------------------------------------ applying
def test_apply_theme_changes_stylesheet_and_palette_in_place():
    """A switch rewrites the palette in place and re-dresses its target with the sheet the
    new palette generates. Aimed at a scoped widget — the same ``setStyleSheet`` call the
    application takes in production, without repolishing the whole test session."""
    target = QtWidgets.QWidget()
    try:
        style.apply_theme(target, "dark")
        dark_sheet = target.styleSheet()
        dark_bg, dark_fg = style.BG, style.FG

        assert style.apply_theme(target, "light") == "light"
        assert style.THEME == "light"
        assert style.BG != dark_bg and style.FG != dark_fg
        assert (style.BG, style.FG) == (style.THEMES["light"]["BG"],
                                        style.THEMES["light"]["FG"])
        assert target.styleSheet() != dark_sheet
        assert style.BG in target.styleSheet() and style.FG in target.styleSheet()
        assert target.styleSheet() == style.STYLESHEET == style.build_stylesheet("light")

        style.apply_theme(target, "dark")
        assert (style.BG, style.FG) == (dark_bg, dark_fg)
        assert target.styleSheet() == dark_sheet
    finally:
        target.deleteLater()


def test_apply_theme_dresses_the_application(app_sheets):
    """The production call: ``apply_theme(app, …)`` hands the QApplication the regenerated
    sheet, so the whole GUI — every open window and dialog — re-reads it."""
    app = QtWidgets.QApplication.instance()
    style.apply_theme(app, "light")
    assert app_sheets == [style.build_stylesheet("light")]
    style.apply_theme(app, "dark")
    assert app_sheets[-1] == style.build_stylesheet("dark")


def test_load_theme_rebinds_the_palette_without_an_application():
    style.load_theme("light")
    assert (style.THEME, style.BG) == ("light", style.THEMES["light"]["BG"])
    assert style.STYLESHEET == style.build_stylesheet("light")
    assert style.PREVIOUS_BG == style.THEMES["dark"]["BG"]


def test_unknown_theme_falls_back_to_the_default():
    assert style.apply_theme(None, "solarized-banana") == "dark"
    assert style.THEME == "dark"


def test_toggle_theme_alternates():
    assert style.toggle_theme(None) == "light"
    assert style.toggle_theme(None) == "dark"


def test_step_moves_away_from_the_page_in_both_themes():
    """The helper the lane blocks / nav-cube faces shade with: lighter on dark, darker on
    light — a bare lighter() would wash the light theme out to flat white."""
    style.load_theme("dark")
    assert style.step(style.PANEL, 150).lightnessF() > QtGui.QColor(style.PANEL).lightnessF()
    style.load_theme("light")
    assert style.step(style.PANEL, 150).lightnessF() < QtGui.QColor(style.PANEL).lightnessF()


# ------------------------------------------------------------------ persistence
def test_persisted_theme_round_trips(settings_store):
    assert style.current_theme() == "dark"           # nothing stored yet → the default
    style.apply_theme(None, "light")
    assert settings_store[style.SETTINGS_KEY] == "light"
    assert style.current_theme() == "light"
    # a fresh launch adopts it without being told which theme to use
    style.load_theme(None)
    assert style.THEME == "light"


def test_persisting_is_optional(settings_store):
    style.apply_theme(None, "light", persist=False)
    assert style.SETTINGS_KEY not in settings_store
    assert style.current_theme() == "dark"


def test_stored_junk_falls_back_to_the_default(settings_store):
    settings_store[style.SETTINGS_KEY] = "chartreuse"
    assert style.current_theme() == "dark"


def test_theme_changed_signal_carries_the_new_name():
    seen = []
    conn = style.notifier().themeChanged.connect(seen.append)
    try:
        style.apply_theme(None, "light")
        style.apply_theme(None, "dark")
    finally:
        style.notifier().themeChanged.disconnect(conn)
    assert seen == ["light", "dark"]


# ------------------------------------------------------------------ icons
def test_icon_cache_invalidates_across_themes(app):
    style.apply_theme(None, "dark")
    dark_icon = icons.icon("play")
    dark_px = dark_icon.pixmap(QtCore.QSize(16, 16)).toImage()

    style.apply_theme(None, "light")
    light_icon = icons.icon("play")
    light_px = light_icon.pixmap(QtCore.QSize(16, 16)).toImage()

    assert light_icon is not dark_icon                 # the cache is keyed by theme
    assert light_px != dark_px                         # and the mark is actually repainted
    assert light_px.pixelColor(7, 8).name() == style.THEMES["light"]["FG"]
    assert dark_px.pixelColor(7, 8).name() == style.THEMES["dark"]["FG"]


def test_palette_role_tint_follows_the_theme(app):
    """An icon tinted for an accent-filled button asks for the ROLE, so it is resolved per
    paint instead of frozen at construction."""
    style.apply_theme(None, "dark")
    ic = icons.icon("play", color="ACCENT_TEXT")       # a solid fill: no antialiased edges
    dark_name = ic.pixmap(QtCore.QSize(16, 16)).toImage().pixelColor(7, 8).name()
    style.apply_theme(None, "light")
    light_name = ic.pixmap(QtCore.QSize(16, 16)).toImage().pixelColor(7, 8).name()
    assert dark_name == style.THEMES["dark"]["ACCENT_TEXT"]
    assert light_name == style.THEMES["light"]["ACCENT_TEXT"]


def test_theme_icon_shows_the_theme_it_switches_to():
    style.load_theme("dark")
    assert icons.theme_icon() is icons.icon("sun")
    style.load_theme("light")
    assert icons.theme_icon() is icons.icon("moon")


def test_clear_cache_drops_everything(app):
    first = icons.icon("play")
    icons.clear_cache()
    assert icons.icon("play") is not first


# ------------------------------------------------------------------ the toolbar toggle
def test_main_window_theme_action_flips_the_persisted_choice(app_sheets):
    pytest.importorskip("pyqtgraph")
    from gottlux.app.main import MainWindow
    style.load_theme("dark")
    win = MainWindow()
    try:
        assert win.act_theme.text() == "Light theme"        # names what it switches TO

        win.act_theme.trigger()
        assert style.THEME == "light" and style.current_theme() == "light"
        assert win.act_theme.text() == "Dark theme"
        assert app_sheets[-1] == style.build_stylesheet("light")

        win.act_theme.trigger()
        assert style.THEME == "dark" and style.current_theme() == "dark"
        assert win.act_theme.text() == "Light theme"
    finally:
        win.close()


def test_quick_viewer_has_the_same_toggle(app_sheets):
    pytest.importorskip("pyqtgraph")
    from gottlux.app.quickview import QuickViewWindow
    style.load_theme("dark")
    win = QuickViewWindow()
    try:
        assert win.act_theme is not None
        assert win._toggle_theme() == "light"
        assert style.current_theme() == "light"       # shared with the full suite
        assert app_sheets[-1] == style.build_stylesheet("light")
    finally:
        win.close()


# ------------------------------------------------------------------ rendering
def _rendered(widget, w, h):
    """The widget's own painting at (*w*, *h*), over the theme's page colour, as a QImage.

    ``DrawChildren`` alone runs the widget's ``paintEvent`` without Qt first stamping the
    window background over the page fill, so what is measured is what the painter drew.
    """
    widget.resize(w, h)
    pm = QtGui.QPixmap(w, h)
    pm.fill(QtGui.QColor(style.BG))
    widget.render(pm, QtCore.QPoint(), QtGui.QRegion(),
                  QtWidgets.QWidget.RenderFlag.DrawChildren)
    return pm.toImage()


def test_timeline_preview_and_legend_render_light_under_the_light_theme(app):
    """The headline of the sweep: painters that used to hold hardcoded dark colours now
    read the palette, so a real render comes out light-on-dark or dark-on-light to match."""
    pytest.importorskip("pyqtgraph")
    from gottlux.app.legend import FrequencyLegend
    from gottlux.app.timeline import TimelineEditor

    editor = TimelineEditor()
    legend = FrequencyLegend()
    legend.set_band(80.0, 800.0)
    try:
        style.apply_theme(None, "dark")
        assert _lightness(_rendered(editor.preview, 240, 120), 4, 4) < 0.25
        assert _lightness(_rendered(editor.lanes, 240, 140), 4, 60) < 0.25

        style.apply_theme(None, "light")
        light_preview = _rendered(editor.preview, 240, 120)
        light_lanes = _rendered(editor.lanes, 240, 140)
        light_legend = _rendered(legend, 240, 44)
        assert _lightness(light_preview, 4, 4) > 0.85       # the viewport's own fill
        assert _lightness(light_lanes, 4, 60) > 0.85        # the lane strip behind the ruler
        assert _lightness(light_legend, 236, 40) > 0.85     # legend page, clear of the bar
        # and the lettering inverted with it, instead of staying pale-on-pale
        letters = {light_legend.pixelColor(x, y).name()
                   for y in range(4, 20) for x in range(0, 60)}
        assert style.FG in letters
    finally:
        editor.deleteLater()
        legend.deleteLater()


def test_arrange_stage_retakes_its_chrome_on_a_switch(app):
    """The composer/Timeline cell stage owns colours Qt never revisits (a scene brush and
    the canvas frame's pen), so it follows the switch through its own themeChanged slot."""
    pytest.importorskip("pyqtgraph")
    from gottlux.app.canvas import CanvasArrangeView
    view = CanvasArrangeView()
    try:
        style.apply_theme(None, "light")
        assert view.gscene.backgroundBrush().color().name() == style.THEMES["light"]["BG"]
        style.apply_theme(None, "dark")
        assert view.gscene.backgroundBrush().color().name() == style.THEMES["dark"]["BG"]
    finally:
        view.deleteLater()


def test_restyle_plots_moves_only_views_still_on_the_old_background(app):
    pg = pytest.importorskip("pyqtgraph")
    host = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(host)
    style.apply_theme(None, "dark")
    followed = pg.GraphicsLayoutWidget()                 # takes the theme background
    custom = pg.GraphicsLayoutWidget()
    custom.setBackground("#0b0f0b")                      # a deliberate scope canvas
    lay.addWidget(followed)
    lay.addWidget(custom)
    try:
        style.apply_theme(None, "light")
        style.restyle_plots(host)
        assert followed.backgroundBrush().color().name() == style.THEMES["light"]["BG"]
        assert custom.backgroundBrush().color().name() == "#0b0f0b"
    finally:
        host.deleteLater()


# ------------------------------------------------------------------ the sweep, as a rule
def test_no_module_binds_palette_constants_at_import_time():
    """``from gottlux.app.style import FG`` freezes a colour at import and goes stale on a
    switch. Every GUI module must reach the palette through the module (``style.FG``)."""
    import gottlux.app
    root = os.path.dirname(os.path.abspath(gottlux.app.__file__))
    pattern = re.compile(r"from\s+gottlux\.app\.style\s+import\s+(.+)")
    offenders = []
    for path in sorted(glob.glob(os.path.join(root, "*.py"))):
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                m = pattern.search(line)
                if not m:
                    continue
                names = {n.strip() for n in m.group(1).split(",")}
                frozen = names & set(style.PALETTE_KEYS)
                if frozen:
                    offenders.append(f"{os.path.basename(path)}:{lineno} {', '.join(frozen)}")
    assert not offenders, ("palette constants bound at import time (use style.X):\n"
                           + "\n".join(offenders))
