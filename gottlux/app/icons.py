"""
icons.py — the painted vector icon system for the GUI (zero extra dependencies).

Historically every "icon" in the app was a Unicode glyph set as button *text* (the play
triangle U+25B6, the scissors dingbat, the down-arrow-to-bar, …). Those glyphs are resolved
through the platform's font-fallback lottery: on Windows the play triangle lands in the Segoe
UI *Emoji* color-bitmap font (oversized, blurry and off-baseline at 12 px on fractional DPI),
the pause bars come from a *different* font than the play triangle (so the transport button
changed size on every toggle), and several of the dingbats are tofu boxes on a stock Linux
install. This module replaces all of that with a tiny :class:`QIconEngine` that *paints* each
mark as an antialiased vector shape:

* Every shape is drawn in a **unit box** and scaled to whatever rect Qt asks for, so the same
  icon is crisp at 16 px in a tool button and at 256 px in the task bar.
* :meth:`~VectorIconEngine.pixmap` renders at ``size × devicePixelRatio`` and stamps the
  ratio on the pixmap, so icons stay pixel-crisp at 100 / 125 / 150 / 200 % display scaling
  on Windows and Linux alike.
* Colors come from the instrument palette in :mod:`gottlux.app.style`: normal marks in the
  foreground color, disabled in the muted grey, checked/on state in the accent cyan, and the
  record/capture dots in the alarm red — automatically, per icon mode/state. The palette is
  read at *paint* time (``style.FG``, never a from-import), so a live theme switch recolors
  every icon already on screen; the :func:`icon` cache is keyed by theme as well, and
  :func:`clear_cache` drops it outright when the theme changes.

Public API: :func:`icon` (cached name → ``QIcon``), :func:`app_icon` (the GottLUX "event
burst" mark used as the window/app icon), :func:`theme_icon` (the sun/moon mark for the
light/dark toggle), :data:`ICON_NAMES`, :func:`clear_cache`, and the small layout helper
:func:`freeze_width` (fix a toggling button's width so play↔pause never shifts the layout).
"""
from __future__ import annotations

import math

from PySide6 import QtCore, QtGui

from gottlux.app import style

# Icons whose "hot" element (the dot) is the alarm red rather than the accent cyan.
_HOT_RED = frozenset({"record", "capture"})


def _pen(col, w, cap=QtCore.Qt.RoundCap, join=QtCore.Qt.RoundJoin):
    """A non-cosmetic pen with width *w* in unit-box coordinates (scales with the icon)."""
    pen = QtGui.QPen(col)
    pen.setWidthF(w)
    pen.setCapStyle(cap)
    pen.setJoinStyle(join)
    return pen


def _tri(p, pts, col):
    """Fill the triangle with unit-box vertices *pts* in *col*."""
    path = QtGui.QPainterPath()
    path.moveTo(*pts[0])
    for x, y in pts[1:]:
        path.lineTo(x, y)
    path.closeSubpath()
    p.fillPath(path, col)


def _arrowhead(p, x, y, deg, col, s=0.20):
    """A filled arrowhead at (*x*, *y*) pointing along screen angle *deg* (0° = +x)."""
    p.save()
    p.translate(x, y)
    p.rotate(deg)
    _tri(p, [(0.6 * s, 0.0), (-0.5 * s, 0.55 * s), (-0.5 * s, -0.55 * s)], col)
    p.restore()


# --------------------------------------------------------------------- the shape library
# Each painter draws inside the unit box [0,1]×[0,1] on an already-scaled painter.
# ``col`` is the main stroke/fill color; ``hot`` the emphasis color (accent, or alarm red
# for record/capture) — both already resolved for the icon's mode/state.

def _play(p, col, hot):
    _tri(p, [(0.32, 0.18), (0.32, 0.82), (0.82, 0.50)], col)


def _pause(p, col, hot):
    for x in (0.28, 0.57):
        p.fillRect(QtCore.QRectF(x, 0.20, 0.15, 0.60), col)


def _stop(p, col, hot):
    path = QtGui.QPainterPath()
    path.addRoundedRect(QtCore.QRectF(0.25, 0.25, 0.50, 0.50), 0.06, 0.06)
    p.fillPath(path, col)


def _record(p, col, hot):
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(hot)
    p.drawEllipse(QtCore.QPointF(0.5, 0.5), 0.30, 0.30)


def _capture(p, col, hot):
    p.setBrush(QtCore.Qt.NoBrush)
    p.setPen(_pen(col, 0.08))
    p.drawEllipse(QtCore.QPointF(0.5, 0.5), 0.34, 0.34)
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(hot)
    p.drawEllipse(QtCore.QPointF(0.5, 0.5), 0.16, 0.16)


def _film(p, col, hot):
    """A clapper board (video / MP4 actions)."""
    p.setPen(QtCore.Qt.NoPen)
    body = QtGui.QPainterPath()
    body.addRoundedRect(QtCore.QRectF(0.14, 0.46, 0.72, 0.34), 0.05, 0.05)
    p.fillPath(body, col)
    p.save()
    p.translate(0.15, 0.44)
    p.rotate(-14)
    p.fillRect(QtCore.QRectF(0.0, -0.17, 0.72, 0.17), col)
    p.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)  # slash stripes on the board
    for x in (0.15, 0.34, 0.53):
        p.fillRect(QtCore.QRectF(x, -0.17, 0.05, 0.17), col)
    p.restore()


def _save(p, col, hot):
    """The classic floppy: clipped-corner shell, shutter, label."""
    shell = QtGui.QPainterPath()
    shell.moveTo(0.20, 0.20)
    shell.lineTo(0.68, 0.20)
    shell.lineTo(0.80, 0.32)
    shell.lineTo(0.80, 0.80)
    shell.lineTo(0.20, 0.80)
    shell.closeSubpath()
    p.setBrush(QtCore.Qt.NoBrush)
    p.setPen(_pen(col, 0.075, join=QtCore.Qt.MiterJoin))
    p.drawPath(shell)
    p.fillRect(QtCore.QRectF(0.34, 0.20, 0.26, 0.17), col)      # shutter
    p.setPen(_pen(col, 0.06, join=QtCore.Qt.MiterJoin))
    p.drawRect(QtCore.QRectF(0.33, 0.53, 0.34, 0.27))           # label


def _export(p, col, hot):
    """An arrow dropping into a tray (file-producing exports)."""
    p.setBrush(QtCore.Qt.NoBrush)
    p.setPen(_pen(col, 0.09))
    tray = QtGui.QPainterPath()
    tray.moveTo(0.16, 0.60)
    tray.lineTo(0.16, 0.82)
    tray.lineTo(0.84, 0.82)
    tray.lineTo(0.84, 0.60)
    p.drawPath(tray)
    p.drawLine(QtCore.QPointF(0.50, 0.12), QtCore.QPointF(0.50, 0.40))
    _tri(p, [(0.33, 0.38), (0.67, 0.38), (0.50, 0.62)], col)


def _sync(p, col, hot):
    """Two circular arrows (refresh / recompute / rotate)."""
    r = QtCore.QRectF(0.20, 0.20, 0.60, 0.60)
    p.setBrush(QtCore.Qt.NoBrush)
    p.setPen(_pen(col, 0.085, cap=QtCore.Qt.FlatCap))
    for start in (150.0, 330.0):                    # two arcs, swept clockwise
        arc = QtGui.QPainterPath()
        arc.arcMoveTo(r, start)
        arc.arcTo(r, start, -120.0)
        p.drawPath(arc)
    # arrowheads at the arc ends (angles 30° and 210°), tangent to the sweep
    for ang in (30.0, 210.0):
        a = math.radians(ang)
        x, y = 0.5 + 0.30 * math.cos(a), 0.5 - 0.30 * math.sin(a)
        deg = math.degrees(math.atan2(math.cos(a), math.sin(a)))
        _arrowhead(p, x, y, deg, col)


def _cut(p, col, hot):
    """Scissors: crossing blades + two ring handles."""
    p.setPen(_pen(col, 0.08))
    p.drawLine(QtCore.QPointF(0.68, 0.14), QtCore.QPointF(0.34, 0.66))
    p.drawLine(QtCore.QPointF(0.32, 0.14), QtCore.QPointF(0.66, 0.66))
    p.setBrush(QtCore.Qt.NoBrush)
    p.setPen(_pen(col, 0.07))
    p.drawEllipse(QtCore.QPointF(0.27, 0.78), 0.105, 0.105)
    p.drawEllipse(QtCore.QPointF(0.73, 0.78), 0.105, 0.105)


def _split(p, col, hot):
    """Two panes side-by-side (split / compare view)."""
    p.setBrush(QtCore.Qt.NoBrush)
    p.setPen(_pen(col, 0.07, join=QtCore.Qt.MiterJoin))
    p.drawRoundedRect(QtCore.QRectF(0.14, 0.24, 0.30, 0.52), 0.04, 0.04)
    p.drawRoundedRect(QtCore.QRectF(0.56, 0.24, 0.30, 0.52), 0.04, 0.04)


def _add(p, col, hot):
    p.setPen(_pen(col, 0.12))
    p.drawLine(QtCore.QPointF(0.50, 0.20), QtCore.QPointF(0.50, 0.80))
    p.drawLine(QtCore.QPointF(0.20, 0.50), QtCore.QPointF(0.80, 0.50))


def _close(p, col, hot):
    p.setPen(_pen(col, 0.11))
    p.drawLine(QtCore.QPointF(0.26, 0.26), QtCore.QPointF(0.74, 0.74))
    p.drawLine(QtCore.QPointF(0.74, 0.26), QtCore.QPointF(0.26, 0.74))


def _arrow_left(p, col, hot):
    _tri(p, [(0.66, 0.22), (0.66, 0.78), (0.26, 0.50)], col)


def _arrow_right(p, col, hot):
    _tri(p, [(0.34, 0.22), (0.34, 0.78), (0.74, 0.50)], col)


def _arrow_up(p, col, hot):
    _tri(p, [(0.22, 0.66), (0.78, 0.66), (0.50, 0.26)], col)


def _arrow_down(p, col, hot):
    _tri(p, [(0.22, 0.34), (0.78, 0.34), (0.50, 0.74)], col)


def _chevron_right(p, col, hot):
    p.setBrush(QtCore.Qt.NoBrush)
    p.setPen(_pen(col, 0.11))
    path = QtGui.QPainterPath()
    path.moveTo(0.38, 0.22)
    path.lineTo(0.68, 0.50)
    path.lineTo(0.38, 0.78)
    p.drawPath(path)


def _target(p, col, hot):
    """A dot in a crosshair ring (add-this-box / add-keyframe actions)."""
    p.setBrush(QtCore.Qt.NoBrush)
    p.setPen(_pen(col, 0.075))
    p.drawEllipse(QtCore.QPointF(0.5, 0.5), 0.27, 0.27)
    for (x0, y0, x1, y1) in ((0.50, 0.05, 0.50, 0.17), (0.50, 0.83, 0.50, 0.95),
                             (0.05, 0.50, 0.17, 0.50), (0.83, 0.50, 0.95, 0.50)):
        p.drawLine(QtCore.QPointF(x0, y0), QtCore.QPointF(x1, y1))
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(hot)
    p.drawEllipse(QtCore.QPointF(0.5, 0.5), 0.11, 0.11)


def _sun(p, col, hot):
    """A sun: filled disc + eight rays — 'switch to the light theme'."""
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(col)
    p.drawEllipse(QtCore.QPointF(0.5, 0.5), 0.21, 0.21)
    p.setPen(_pen(col, 0.075))
    for i in range(8):
        a = math.radians(i * 45.0)
        dx, dy = math.cos(a), math.sin(a)
        p.drawLine(QtCore.QPointF(0.5 + 0.30 * dx, 0.5 + 0.30 * dy),
                   QtCore.QPointF(0.5 + 0.43 * dx, 0.5 + 0.43 * dy))


def _moon(p, col, hot):
    """A crescent — 'switch to the dark theme'. Drawn as a disc minus an offset disc."""
    path = QtGui.QPainterPath()
    path.addEllipse(QtCore.QPointF(0.50, 0.50), 0.34, 0.34)
    bite = QtGui.QPainterPath()
    bite.addEllipse(QtCore.QPointF(0.68, 0.38), 0.31, 0.31)
    p.fillPath(path.subtracted(bite), col)


def _gottlux_mark(p, col, hot):
    """The GottLUX app mark: a bright event dot with two arc rings ("event burst") on the
    instrument plate — recognisable at 16 px in the task bar and at 256 px."""
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(QtGui.QColor(style.BG))
    p.drawRoundedRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0), 0.18, 0.18)
    accent, warm = QtGui.QColor(style.ACCENT), QtGui.QColor(style.ACCENT2)
    p.setBrush(QtCore.Qt.NoBrush)
    p.setPen(_pen(accent, 0.075))
    inner = QtCore.QRectF(0.25, 0.25, 0.50, 0.50)
    arc = QtGui.QPainterPath()
    arc.arcMoveTo(inner, 300.0)
    arc.arcTo(inner, 300.0, 250.0)
    p.drawPath(arc)
    p.setPen(_pen(warm, 0.065))
    outer = QtCore.QRectF(0.11, 0.11, 0.78, 0.78)
    arc2 = QtGui.QPainterPath()
    arc2.arcMoveTo(outer, 120.0)
    arc2.arcTo(outer, 120.0, 210.0)
    p.drawPath(arc2)
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(accent)
    p.drawEllipse(QtCore.QPointF(0.5, 0.5), 0.12, 0.12)


_PAINTERS = {
    "play": _play,
    "pause": _pause,
    "stop": _stop,
    "record": _record,
    "capture": _capture,
    "film": _film,
    "save": _save,
    "export": _export,
    "sync": _sync,
    "cut": _cut,
    "split": _split,
    "add": _add,
    "close": _close,
    "arrow-left": _arrow_left,
    "arrow-right": _arrow_right,
    "arrow-up": _arrow_up,
    "arrow-down": _arrow_down,
    "chevron-right": _chevron_right,
    "target": _target,
    "sun": _sun,
    "moon": _moon,
    "gottlux": _gottlux_mark,
}

#: Every icon name :func:`icon` accepts.
ICON_NAMES = tuple(sorted(_PAINTERS))


# --------------------------------------------------------------------- the engine
class VectorIconEngine(QtGui.QIconEngine):
    """Paints one named shape from the library, normalized to whatever rect Qt asks for.

    ``paint`` translates/scales into the target rect's largest centered square, so shapes are
    authored once in the unit box. ``pixmap`` renders at ``size × devicePixelRatio`` and tags
    the pixmap with the ratio, which is what makes the icons crisp on fractional-DPI screens.
    """

    def __init__(self, name: str, color=None):
        super().__init__()
        self._name = name
        # A palette *role* name ("ACCENT_TEXT") is kept as a role and looked up per paint,
        # so an icon tinted for an accent-filled button follows a theme switch; anything
        # else is a fixed colour.
        self._role = color if isinstance(color, str) and color in style.PALETTE_KEYS else None
        self._color = None if (color is None or self._role) else QtGui.QColor(color)

    def clone(self):
        return VectorIconEngine(self._name, self._role or self._color)

    # ----- palette resolution (mode/state → colors) -----
    def _colors(self, mode, state):
        """The (main, emphasis) colors for this mode/state, read from the palette *now* —
        an engine outlives a theme switch, so the values must not be captured earlier."""
        if self._role is not None:                     # a palette role: monochrome, live
            c = QtGui.QColor(getattr(style, self._role))
            return c, QtGui.QColor(c)
        if self._color is not None:                    # explicit override: monochrome icon
            return QtGui.QColor(self._color), QtGui.QColor(self._color)
        if mode == QtGui.QIcon.Disabled:
            c = QtGui.QColor(style.MUTED)
            return c, c
        base = QtGui.QColor(style.ACCENT) if state == QtGui.QIcon.On else QtGui.QColor(style.FG)
        hot = QtGui.QColor(style.BAD) if self._name in _HOT_RED else QtGui.QColor(style.ACCENT)
        return base, hot

    # ----- QIconEngine API -----
    def paint(self, painter, rect, mode, state):
        col, hot = self._colors(mode, state)
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        side = float(min(rect.width(), rect.height()))
        painter.translate(rect.x() + (rect.width() - side) / 2.0,
                          rect.y() + (rect.height() - side) / 2.0)
        painter.scale(side, side)
        _PAINTERS[self._name](painter, col, hot)
        painter.restore()

    def pixmap(self, size, mode, state):
        app = QtGui.QGuiApplication.instance()
        dpr = float(app.devicePixelRatio()) if app is not None else 1.0
        return self.scaledPixmap(size, mode, state, dpr)

    def scaledPixmap(self, size, mode, state, scale):
        scale = max(float(scale), 1.0)
        pm = QtGui.QPixmap(int(round(size.width() * scale)), int(round(size.height() * scale)))
        pm.setDevicePixelRatio(scale)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        self.paint(p, QtCore.QRect(0, 0, size.width(), size.height()), mode, state)
        p.end()
        return pm


# --------------------------------------------------------------------- public API
_CACHE: dict = {}


def icon(name: str, color=None) -> QtGui.QIcon:
    """The cached ``QIcon`` for *name* (see :data:`ICON_NAMES`).

    By default the engine follows the icon mode/state: foreground, muted when disabled,
    accent when checked/on, alarm red dots for record/capture. *color* forces one colour
    instead — either a literal (``"#ff3b3b"``, a ``QColor``) or, better, the **name of a
    palette role** from :data:`gottlux.app.style.PALETTE_KEYS`, e.g. ``"ACCENT_TEXT"`` for
    a mark sitting on a ``#primary`` accent button: a role is resolved at paint time and so
    survives a theme switch, where a literal is frozen.

    The cache is keyed by the active theme as well as by name and override, so an icon
    handed out under one theme can never be mistaken for the other theme's — and
    :func:`clear_cache` (called on every switch) drops the lot regardless.
    """
    if name not in _PAINTERS:
        raise KeyError(f"unknown icon {name!r}; known: {', '.join(ICON_NAMES)}")
    if color is None or (isinstance(color, str) and color in style.PALETTE_KEYS):
        tint = color
    else:
        tint = QtGui.QColor(color).name(QtGui.QColor.HexArgb)
    key = (style.THEME, name, tint)
    ic = _CACHE.get(key)
    if ic is None:
        ic = QtGui.QIcon(VectorIconEngine(name, color))
        _CACHE[key] = ic
    return ic


def clear_cache() -> None:
    """Drop every cached ``QIcon`` — what a theme switch calls (see
    :func:`gottlux.app.style.apply_theme`)."""
    _CACHE.clear()


def app_icon() -> QtGui.QIcon:
    """The GottLUX application/window icon (the painted 'event burst' mark)."""
    return icon("gottlux")


def theme_icon() -> QtGui.QIcon:
    """The mark for the light/dark toggle: it shows the theme the toggle switches **to** —
    a sun while the dark theme is active, a crescent moon while the light one is."""
    return icon("sun" if style.THEME == "dark" else "moon")


def freeze_width(button, texts):
    """Fix *button*'s width to the widest of *texts* so toggling its label (play ↔ pause,
    behind ↔ ahead) never changes its geometry and the row never jitters."""
    old = button.text()
    w = 0
    for t in texts:
        button.setText(t)
        w = max(w, button.sizeHint().width())
    button.setText(old)
    button.setFixedWidth(w)
