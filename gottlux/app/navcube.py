"""
navcube.py — a CAD-style orientation cube for the OpenGL views.

A small clickable gizmo (like the navigation cube in CAD tools): an isometric cube whose three
visible faces snap the camera to the **Top / Front / Side** orthographic views, with a compact
row of buttons beneath for the three hidden faces (**Bottom / Back / Left**) and an **Iso**
home. Bind it to a pyqtgraph ``GLViewWidget`` with :meth:`bind`; clicking a face calls the
view's ``setCameraPosition(elevation=…, azimuth=…)`` (distance preserved).

Embedded as a corner overlay on every 3-D tab so the scene can be re-oriented with one click
instead of mouse-orbiting to a known angle.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from gottlux.app import icons
from gottlux.app import style

#: How far each visible face sits off the panel surface — the shading that makes the flat
#: isometric outline read as a solid. Applied through :func:`gottlux.app.style.step`, which
#: lifts on a dark theme and deepens on a light one, so the cube has depth in both.
_FACE_STEP = {"Top": 175, "Side": 135, "Front": 110}

# Standard orthographic + isometric camera presets for pyqtgraph GLViewWidget
# (elevation = degrees above the xy-plane, azimuth = degrees around z).
VIEWS = {
    "Top":    (90.0, -90.0),
    "Bottom": (-90.0, -90.0),
    "Front":  (0.0, 90.0),
    "Back":   (0.0, -90.0),
    "Side":   (0.0, 0.0),
    "Left":   (0.0, 180.0),
    "Iso":    (30.0, 45.0),
}


class NavCube(QtWidgets.QWidget):
    """A clickable isometric orientation cube + hidden-face buttons. Emits :attr:`viewPicked`."""

    viewPicked = QtCore.Signal(float, float)      # (elevation, azimuth)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._view = None
        self.setFixedSize(116, 116)
        self.setToolTip("Orientation cube — click a face to snap the view (Top / Front / Side); "
                        "buttons below give the hidden faces, an isometric home, and a 90° rotate.")
        self._faces = {}          # name -> QPolygonF (built in paintEvent)
        self._build_buttons()

    def bind(self, glview):
        """Bind to a GLViewWidget; clicks will set its camera orientation."""
        self._view = glview
        self.viewPicked.connect(self._apply)
        return self

    def _apply(self, el, az):
        if self._view is not None:
            try:
                self._view.setCameraPosition(elevation=el, azimuth=az)
            except Exception:
                pass

    def _flip90(self):
        """Rotate the current view 90° about the vertical axis (a quarter-turn in azimuth).

        Relative to wherever the camera is now, so repeated clicks step 90° → 180° → 270° → 0°.
        Elevation and distance are preserved.
        """
        if self._view is None:
            return
        try:
            az = float(self._view.opts.get("azimuth", 0.0))
            self._view.setCameraPosition(azimuth=(az + 90.0) % 360.0)
        except Exception:
            pass

    # ------------------------------------------------------------------ hidden-face buttons
    def _build_buttons(self):
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(2, 2, 2, 2); v.setSpacing(2)
        v.addStretch(1)
        row = QtWidgets.QWidget()
        # these are deliberately tiny fixed-size buttons — drop the app-wide button padding
        row.setStyleSheet("QToolButton { padding: 0px; }")
        hb = QtWidgets.QHBoxLayout(row); hb.setContentsMargins(0, 0, 0, 0); hb.setSpacing(2)
        for name, label in (("Iso", "⌂"), ("Bottom", "B"), ("Back", "K"), ("Left", "L")):
            b = QtWidgets.QToolButton(); b.setText(label); b.setFixedSize(20, 18)
            b.setToolTip(f"{name} view")
            b.clicked.connect(lambda _=False, n=name: self.viewPicked.emit(*VIEWS[n]))
            hb.addWidget(b)
        flip = QtWidgets.QToolButton(); flip.setIcon(icons.icon("sync"))
        flip.setFixedSize(20, 18)                       # matches the letter buttons beside it
        # icon size from the font metrics so the mark sits on the same optical size as the letters
        s = flip.fontMetrics().height()
        flip.setIconSize(QtCore.QSize(s, s))
        flip.setToolTip("Rotate the view 90° (click again for 180° / 270° / back to 0°).")
        flip.clicked.connect(self._flip90)
        hb.addWidget(flip)
        v.addWidget(row, 0, QtCore.Qt.AlignHCenter)

    # ------------------------------------------------------------------ painting
    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        cx, cy, r = self.width() / 2, 44, 30
        # isometric cube vertices
        T = QtCore.QPointF(cx, cy - r)
        R = QtCore.QPointF(cx + 0.87 * r, cy - 0.5 * r)
        L = QtCore.QPointF(cx - 0.87 * r, cy - 0.5 * r)
        C = QtCore.QPointF(cx, cy)
        Bm = QtCore.QPointF(cx, cy + r)
        L2 = QtCore.QPointF(cx - 0.87 * r, cy + 0.5 * r)
        R2 = QtCore.QPointF(cx + 0.87 * r, cy + 0.5 * r)
        top = QtGui.QPolygonF([T, R, C, L])
        front = QtGui.QPolygonF([L, C, Bm, L2])
        side = QtGui.QPolygonF([C, R, R2, Bm])
        self._faces = {"Top": top, "Front": front, "Side": side}
        pen = QtGui.QPen(QtGui.QColor(style.BORDER)); pen.setWidth(1)
        p.setPen(pen)
        for name, poly in (("Top", top), ("Front", front), ("Side", side)):
            p.setBrush(style.step(style.BG2, _FACE_STEP[name]))
            p.drawPolygon(poly)
        p.setPen(QtGui.QColor(style.FG))
        f = p.font(); f.setPointSize(7); p.setFont(f)
        p.drawText(top.boundingRect(), QtCore.Qt.AlignCenter, "TOP")
        p.drawText(front.boundingRect(), QtCore.Qt.AlignCenter, "FRONT")
        p.drawText(side.boundingRect(), QtCore.Qt.AlignCenter, "SIDE")
        p.end()

    def mousePressEvent(self, ev):
        pt = ev.position() if hasattr(ev, "position") else QtCore.QPointF(ev.pos())
        for name, poly in self._faces.items():
            if poly.containsPoint(pt, QtCore.Qt.OddEvenFill):
                self.viewPicked.emit(*VIEWS[name])
                return
        super().mousePressEvent(ev)


def navcube_container(glview):
    """Wrap *glview* in a holder that floats a bound :class:`NavCube` at its top-right corner.

    Uses a single-cell ``QGridLayout`` with the cube as an overlapping *sibling* of the GL view
    (the supported way to composite an ordinary widget over a ``QOpenGLWidget``), so it renders
    correctly as a corner gizmo. Returns ``(holder_widget, cube)``.
    """
    cube = NavCube().bind(glview)
    holder = QtWidgets.QWidget()
    grid = QtWidgets.QGridLayout(holder)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.addWidget(glview, 0, 0)
    grid.addWidget(cube, 0, 0, QtCore.Qt.AlignTop | QtCore.Qt.AlignRight)
    return holder, cube
