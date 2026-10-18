"""
style.py — the instrument look for the gottlux GUI, in a dark and a light theme.

One Qt stylesheet + a few pyqtgraph defaults give the whole app a calm, high-contrast,
"instrument" feel that keeps attention on the data. Result *figures* (matplotlib, via
:mod:`gottlux.viz`) deliberately stay light/academic — this theme is only for the live UI.

**Two themes, one palette shape.** :data:`THEMES` holds one entry per theme; both define
exactly the same roles (:data:`PALETTE_KEYS`), so every consumer reads the same names and
neither theme can grow a colour the other lacks. ``'dark'`` is the original instrument
palette; ``'light'`` is its daylight counterpart — the same teal accent hue, light panels,
and muted/dim roles pushed dark enough to stay legible on a bright surface.

**Switching at run time.** :func:`apply_theme` rewrites the module-level colour attributes
**in place**, regenerates :data:`STYLESHEET`, applies it to the ``QApplication``, re-points
pyqtgraph, drops the painted-icon cache, persists the choice (``QSettings``, organization
``GottLUX``) and emits :attr:`notifier().themeChanged`. Because the constants are rebound
on the module rather than replaced by new module objects, anything that reads them at
*paint* time — ``style.FG``, ``style.ACCENT``, … — picks up the new theme on its next
repaint with no further wiring.

That last part is the one rule for consumers: **read colours as attributes**
(``from gottlux.app import style`` … ``style.FG``), never ``from gottlux.app.style import
FG``, which binds the value once and then goes stale on a theme switch.

:func:`current_theme` reads the persisted choice (``'dark'`` by default) and
:func:`load_theme` adopts a palette without needing a ``QApplication`` — the boot splash
uses it to paint in the right theme before the app style is applied.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

#: Every colour role a theme defines — and every module-level attribute a theme switch
#: rewrites. Both themes carry all of them; :func:`load_theme` asserts nothing is missing.
PALETTE_KEYS = ("BG", "BG2", "PANEL", "FG", "MUTED", "ACCENT", "ACCENT_HOVER",
                "ACCENT_TEXT", "ACCENT2", "GOOD", "WARN", "BAD", "BORDER",
                "SELECT", "HANDLE", "PLAYHEAD")

#: The palettes. Role by role:
#:
#: ``BG``/``BG2``/``PANEL``  the three surfaces — page, recessed group, raised control
#: ``FG``/``MUTED``          primary and secondary text
#: ``ACCENT``                the instrument teal: focus, selection, checked state
#: ``ACCENT_HOVER``          the accent under the pointer (the ``#primary`` button)
#: ``ACCENT_TEXT``           text/icons sitting ON an accent-filled button
#: ``ACCENT2``               the warm counter-accent (playheads, secondary curves)
#: ``GOOD``/``WARN``/``BAD`` status colours (also the record/capture dots)
#: ``BORDER``                every hairline and groove
#: ``SELECT``/``HANDLE``     the In/Out range line and its drag handles
#: ``PLAYHEAD``              the bright cursor line drawn across the selection strip
THEMES = {
    # The original instrument panel, unchanged.
    "dark": {
        "BG": "#0e1116", "BG2": "#161b22", "PANEL": "#1b2230",
        "FG": "#d7dde7", "MUTED": "#8b97a7",
        "ACCENT": "#39c5cf", "ACCENT_HOVER": "#54d6df", "ACCENT_TEXT": "#04181a",
        "ACCENT2": "#f78166",
        "GOOD": "#3fb950", "WARN": "#d29922", "BAD": "#f85149",
        "BORDER": "#2b3340",
        "SELECT": "#00e5ff", "HANDLE": "#ff2bd6", "PLAYHEAD": "#f0f3f7",
    },
    # Daylight: the same 184° accent hue darkened until it carries text on a white panel,
    # surfaces inverted (the page is the mid tone, controls are the bright one), and the
    # muted/dim roles taken far enough down to stay readable rather than merely visible.
    "light": {
        "BG": "#f2f5f8", "BG2": "#e6ebf1", "PANEL": "#ffffff",
        "FG": "#16202b", "MUTED": "#5a6a7a",
        "ACCENT": "#0b7078", "ACCENT_HOVER": "#0e8b95", "ACCENT_TEXT": "#ffffff",
        "ACCENT2": "#b4471f",
        "GOOD": "#1a7f37", "WARN": "#8a6100", "BAD": "#cf222e",
        "BORDER": "#c2ccd7",
        "SELECT": "#0090a3", "HANDLE": "#c01f96", "PLAYHEAD": "#10161d",
    },
}

#: The theme used when nothing has been persisted yet.
DEFAULT_THEME = "dark"

#: Where the choice lives inside the ``GottLUX`` settings.
SETTINGS_KEY = "appearance/theme"

# Palette — rebound in place by load_theme(); read these as attributes (style.FG), never
# as from-imports, or a theme switch will leave the value behind.
BG = BG2 = PANEL = FG = MUTED = ACCENT = ACCENT_HOVER = ACCENT_TEXT = ACCENT2 = ""
GOOD = WARN = BAD = BORDER = SELECT = HANDLE = PLAYHEAD = ""

#: The name of the palette currently loaded into the module attributes.
THEME = DEFAULT_THEME

#: The background the *previous* palette used — what :func:`restyle_plots` recognises as
#: "this view was still showing the old theme" and is therefore safe to repaint.
PREVIOUS_BG = ""
#: The foreground (axis/tick/title ink) the previous palette used — see :func:`restyle_plots`.
PREVIOUS_FG = ""

# One explicit app-wide font stack so text metrics (and hence layouts) are the same on
# Windows and Linux instead of whatever the platform default resolves to.
FONT_STACK = '"Segoe UI", "Noto Sans", "DejaVu Sans", sans-serif'


def theme_names() -> tuple:
    """Every theme :func:`apply_theme` accepts, in offer order."""
    return tuple(THEMES)


def resolve(name) -> str:
    """*name* if it is a known theme, else :data:`DEFAULT_THEME` (never raises)."""
    return name if name in THEMES else DEFAULT_THEME


def _arrow(direction: str, color: str) -> str:
    """Path (forward slashes, for a stylesheet ``url()``) of a small triangle SVG.

    Qt stylesheets do not draw the CSS "zero-size box with one coloured border" triangle:
    they paint the box, so the spin-box steppers and menu indicators showed as two identical
    squares. Qt does load images, so each arrow is a tiny SVG written once per colour into
    the temp folder (rebuilt on demand, so a cleaned temp folder costs nothing)."""
    import os
    import tempfile
    pts = {"up": "1,7 5,2 9,7", "down": "1,3 5,8 9,3"}[direction]
    folder = os.path.join(tempfile.gettempdir(), "gottlux_qss")
    path = os.path.join(folder, f"{direction}_{color.lstrip('#')}.svg")
    if not os.path.exists(path):
        try:
            os.makedirs(folder, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
                        f'<polygon points="{pts}" fill="{color}"/></svg>')
        except OSError:
            return ""
    return path.replace("\\", "/")


def build_stylesheet(name: str = DEFAULT_THEME) -> str:
    """The whole Qt stylesheet for theme *name* (nothing is applied or stored, apart from the
    arrow images :func:`_arrow` caches in the temp folder)."""
    p = THEMES[resolve(name)]
    bg, bg2, panel, fg, muted = p["BG"], p["BG2"], p["PANEL"], p["FG"], p["MUTED"]
    accent, hover, border = p["ACCENT"], p["ACCENT_HOVER"], p["BORDER"]
    accent_text = p["ACCENT_TEXT"]
    up, down, up_off, down_off, menu = (_arrow("up", fg), _arrow("down", fg), _arrow("up", muted),
                                        _arrow("down", muted), _arrow("down", muted))
    return f"""
QWidget {{ background: {bg}; color: {fg}; font-size: 12px;
           font-family: {FONT_STACK};
           selection-background-color: {accent}; }}
QMainWindow, QDialog {{ background: {bg}; }}
QLabel {{ background: transparent; }}
QLabel#muted {{ color: {muted}; }}
QLabel#h1 {{ font-size: 15px; font-weight: 600; color: {fg}; }}
QLabel#h2 {{ font-size: 12px; font-weight: 600; color: {accent}; }}
QGroupBox {{ border: 1px solid {border}; border-radius: 6px; margin-top: 14px;
             padding-top: 8px; background: {bg2}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
                    color: {accent}; font-weight: 600; }}
QPushButton {{ background: {panel}; border: 1px solid {border}; border-radius: 5px;
               padding: 5px 12px; }}
QPushButton:hover {{ border-color: {accent}; }}
QPushButton:pressed {{ background: {bg2}; }}
QPushButton#primary {{ background: {accent}; color: {accent_text}; font-weight: 600;
                       border: none; }}
QPushButton#primary:hover {{ background: {hover}; }}
QPushButton:disabled {{ color: {muted}; border-color: {bg2}; }}
/* Tool buttons match the push-button look (they host the painted vector icons — see
   icons.py). The ::menu-indicator is the ONE dropdown arrow for every menu button, a
   small SVG like the spin arrows below (see _arrow) — no hand-rolled "▾" in labels. */
QToolButton {{ background: {panel}; border: 1px solid {border}; border-radius: 5px;
               padding: 4px 8px; }}
QToolButton:hover {{ border-color: {accent}; }}
QToolButton:pressed {{ background: {bg2}; }}
QToolButton:checked {{ border-color: {accent}; color: {accent}; }}
QToolButton:disabled {{ color: {muted}; border-color: {bg2}; }}
QToolButton[popupMode="1"], QToolButton[popupMode="2"],
QToolButton[popupMode="MenuButtonPopup"], QToolButton[popupMode="InstantPopup"] {{
    padding-right: 18px; }}
QToolButton::menu-indicator {{ image: url({menu}); width: 9px; height: 9px;
    subcontrol-origin: padding; subcontrol-position: center right; right: 5px; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {panel}; border: 1px solid {border}; border-radius: 4px; padding: 3px 6px; }}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {accent}; }}
QComboBox QAbstractItemView {{ background: {panel}; border: 1px solid {border};
                               selection-background-color: {accent}; }}
/* Spin-box step buttons. Qt requires these to be styled explicitly once the box itself is
   styled, otherwise the UP button collapses to a zero hit-area (clicks do nothing). Both
   buttons get a real width, a divider, and a triangle arrow image. */
QSpinBox, QDoubleSpinBox {{ padding-right: 20px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border; subcontrol-position: top right; width: 17px;
    border-left: 1px solid {border}; border-bottom: 1px solid {border};
    border-top-right-radius: 4px; background: {panel}; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; subcontrol-position: bottom right; width: 17px;
    border-left: 1px solid {border}; border-top: 1px solid {border};
    border-bottom-right-radius: 4px; background: {panel}; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {accent}; }}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{ background: {bg2}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({up}); width: 9px; height: 9px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({down}); width: 9px; height: 9px; }}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{ image: url({up_off}); }}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{ image: url({down_off}); }}
QSlider::groove:horizontal {{ height: 4px; background: {border}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {accent}; width: 14px; margin: -6px 0;
                              border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 2px; }}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {border};
                        border-radius: 3px; background: {panel}; }}
QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent}; }}
QTabBar::tab {{ background: {bg2}; padding: 7px 16px; border: 1px solid {border};
                border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px;
                color: {muted}; }}
QTabBar::tab:selected {{ background: {panel}; color: {accent}; }}
QTabWidget::pane {{ border: 1px solid {border}; top: -1px; }}
QProgressBar {{ border: 1px solid {border}; border-radius: 4px; background: {panel};
                text-align: center; height: 16px; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 3px; }}
QStatusBar {{ background: {bg2}; color: {muted}; border-top: 1px solid {border}; }}
QPlainTextEdit, QTextEdit {{ background: {bg2}; border: 1px solid {border};
                             font-family: Consolas, monospace; font-size: 11px; }}
QScrollBar:vertical {{ background: {bg}; width: 11px; }}
QScrollBar::handle:vertical {{ background: {border}; border-radius: 5px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QSplitter::handle {{ background: {border}; }}
QToolTip {{ background: {panel}; color: {fg}; border: 1px solid {accent}; padding: 4px; }}
"""


#: The stylesheet for the palette currently loaded (regenerated by :func:`load_theme`).
STYLESHEET = ""


# --------------------------------------------------------------------- persistence
def settings() -> QtCore.QSettings:
    """The ``QSettings`` the theme choice lives in (organization ``GottLUX``).

    Named explicitly rather than relying on the application having called
    ``setOrganizationName`` first, so the quick viewer, the full suite and a bare
    ``QApplication`` in a test all read and write the same key. Tests monkeypatch this.
    """
    return QtCore.QSettings("GottLUX", "gottlux")


def current_theme() -> str:
    """The persisted theme name — :data:`DEFAULT_THEME` when nothing is stored (or the
    stored value names a theme that no longer exists)."""
    try:
        stored = settings().value(SETTINGS_KEY, DEFAULT_THEME)
    except Exception:                      # pragma: no cover — no settings backend
        return DEFAULT_THEME
    return resolve(None if stored is None else str(stored))


def set_current_theme(name: str) -> str:
    """Persist *name* as the theme later launches start in; returns what was stored."""
    name = resolve(name)
    try:
        settings().setValue(SETTINGS_KEY, name)
    except Exception:                      # pragma: no cover — no settings backend
        pass
    return name


# --------------------------------------------------------------------- notification
_NOTIFIER = None


def notifier():
    """The object whose ``themeChanged(str)`` signal fires after every :func:`apply_theme`.

    A module-level singleton, so a window can react to a switch made anywhere (the quick
    viewer's corner toggle, the main window's toolbar action, a test) without the two
    knowing about each other.
    """
    global _NOTIFIER
    if _NOTIFIER is None:
        class _ThemeNotifier(QtCore.QObject):
            themeChanged = QtCore.Signal(str)

        _NOTIFIER = _ThemeNotifier()
    return _NOTIFIER


# --------------------------------------------------------------------- applying
def load_theme(name=None) -> str:
    """Adopt a palette into the module attributes and rebuild :data:`STYLESHEET`.

    The half of a theme switch that needs no ``QApplication``: the constants are rebound
    **in place** so every paint-time reader (``style.FG`` …) sees the new values at once.
    *name* defaults to the persisted choice. Returns the theme actually loaded.
    """
    global STYLESHEET, THEME, PREVIOUS_BG, PREVIOUS_FG
    name = resolve(current_theme() if name is None else name)
    palette = THEMES[name]
    missing = [k for k in PALETTE_KEYS if k not in palette]
    if missing:                            # pragma: no cover — guards a mistyped theme
        raise KeyError(f"theme {name!r} is missing colour role(s): {', '.join(missing)}")
    PREVIOUS_BG = BG or palette["BG"]
    PREVIOUS_FG = FG or palette["FG"]
    globals().update({k: palette[k] for k in PALETTE_KEYS})
    STYLESHEET = build_stylesheet(name)
    THEME = name
    return name


def apply_theme(app, name=None, persist=True) -> str:
    """Switch *app* to theme *name* (default: the persisted choice) and return its name.

    Everything a live switch needs, in one call: the palette is rebound, the stylesheet
    regenerated and handed to *app* — normally the ``QApplication``, so the sheet reaches
    every open window — pyqtgraph re-pointed at the new chrome colours, the painted-icon
    cache dropped (icons resolve their colours per theme), the choice persisted, and
    ``notifier().themeChanged`` emitted so open windows can re-theme the parts Qt cannot
    repaint on its own (plot backgrounds, 3-D canvases, hand-styled widgets).

    ``app=None`` performs the whole switch except that one step, for callers that only
    need the palette moved.
    """
    name = load_theme(name)
    if app is not None:
        app.setStyleSheet(STYLESHEET)
    try:
        import pyqtgraph as pg
        pg.setConfigOptions(antialias=True, imageAxisOrder="row-major",
                            background=BG, foreground=FG, useOpenGL=False)
    except Exception:
        pass
    try:
        from gottlux.app import icons
        icons.clear_cache()
    except Exception:                      # pragma: no cover — icons import can't fail here
        pass
    if persist:
        set_current_theme(name)
    notifier().themeChanged.emit(name)
    return name


def apply_app_style(app, name=None) -> str:
    """Apply the instrument stylesheet + pyqtgraph defaults to a fresh ``QApplication``.

    The startup entry point (kept under its historical name): with no *name* it adopts
    whatever theme the user last chose.
    """
    return apply_theme(app, name)


def toggle_theme(app) -> str:
    """Flip between dark and light, apply, and return the new theme's name."""
    return apply_theme(app, "light" if THEME == "dark" else "dark")


# --------------------------------------------------------------------- helpers
def is_dark(name=None) -> bool:
    """Whether *name* (default: the loaded theme) is a dark-on-light-text palette."""
    return QtGui.QColor(THEMES[resolve(name or THEME)]["BG"]).lightnessF() < 0.5


def step(color, percent: int = 130) -> QtGui.QColor:
    """*color* moved one step **away** from the page: lighter on dark, darker on light.

    Chrome that needs to read as "raised off the panel" (a lane block, a cube face) can
    then be expressed once and stay correct in both themes, where a bare ``lighter()``
    would wash a light theme out to flat white.
    """
    c = QtGui.QColor(color)
    return c.lighter(percent) if is_dark() else c.darker(percent)


def repolish(root) -> None:
    """Re-run the style engine over *root* and every child, after the sheet changed.

    Qt only re-evaluates a widget's stylesheet rules when its style is *unpolished*;
    without this, widgets already on screen keep the colours they were built with.
    """
    if root is None:
        return
    st = root.style()
    for w in [root] + root.findChildren(QtWidgets.QWidget):
        st.unpolish(w)
        st.polish(w)
        w.update()


def restyle_plots(root, previous_bg=None) -> None:
    """Re-point every pyqtgraph plot under *root* at the new chrome background.

    pyqtgraph reads its background from the global config **once**, when the view is
    built, so plots that already exist keep the old colour across a switch. Only views
    still showing *previous_bg* (which defaults to :data:`PREVIOUS_BG`, the background the
    outgoing theme used) are touched — a view given a deliberately different canvas, like
    the radar scope or a 3-D background preset, keeps exactly what it was given.
    """
    if root is None:
        return
    try:
        import pyqtgraph as pg
    except Exception:                      # pragma: no cover — pyqtgraph is optional
        return
    old = QtGui.QColor(previous_bg or PREVIOUS_BG).name()
    for w in root.findChildren(pg.GraphicsView):
        if w.backgroundBrush().color().name() == old:
            w.setBackground(BG)
    _reink_axes(root, pg)


def _reink_axes(root, pg) -> None:
    """Re-colour axis lines, tick labels, axis labels and plot titles that still carry the
    outgoing theme's foreground. pyqtgraph bakes ``foreground`` into each AxisItem when it is
    built, so after dark → light the ticks and titles stayed light grey on a light canvas
    (nearly invisible in the Fusion lab and the workbench spectrum). Items given their own
    colour are left alone."""
    old_fg = QtGui.QColor(PREVIOUS_FG).name() if PREVIOUS_FG else None
    if not old_fg:
        return
    for gv in root.findChildren(pg.GraphicsView):
        scene = gv.scene()
        if scene is None:
            continue
        for item in scene.items():
            if isinstance(item, pg.AxisItem):
                if item.pen().color().name() == old_fg:
                    item.setPen(FG)
                if item.textPen().color().name() == old_fg:
                    item.setTextPen(FG)
                if item.labelStyle.get("color") in (None, PREVIOUS_FG, old_fg):
                    item.labelStyle["color"] = FG
                    item._updateLabel()
            elif isinstance(item, pg.PlotItem):
                title = item.titleLabel
                if title.opts.get("color") in (None, PREVIOUS_FG, old_fg) and title.text:
                    title.setText(title.text, color=FG)


def refresh_window(win) -> None:
    """Bring one already-built window up to date with the loaded theme.

    The two things Qt will not do by itself after the application stylesheet changes:
    re-point the pyqtgraph canvases, and unpolish/polish so every styled widget re-reads
    the sheet. A window's own toggle calls this after :func:`apply_theme`; chrome beyond
    that (a 3-D background preset, a hand-styled panel) belongs to the widget that owns
    it, which follows the switch through its own ``themeChanged`` slot.
    """
    if win is None:
        return
    restyle_plots(win)
    repolish(win)


load_theme(DEFAULT_THEME)          # module import leaves the historical dark palette loaded
