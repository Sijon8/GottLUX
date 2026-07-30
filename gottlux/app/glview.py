"""glview.py — a GLViewWidget whose middle-mouse pan still works looking straight down.

pyqtgraph maps the middle-drag "grab" to ``GLViewWidget.pan(..., relative='view-upright')``,
which builds its screen basis from ``cross(z, camera_vector)``. At the **Top / Bottom**
orthographic views (elevation = ±90°) the camera vector is parallel to *z*, that cross product
collapses to a zero vector, and the scene simply refuses to pan — exactly the "the top view
breaks the middle-mouse grab" symptom.

:class:`GLView` overrides :meth:`pan` to substitute the *analytic limit* of that basis (derived
purely from the azimuth) whenever the camera is within half a degree of straight up/down. The
limit is continuous with pyqtgraph's own formula away from the poles, so panning behaves
identically everywhere else and merely keeps working at the poles.

Used by every 3-D OpenGL tab (the event-rate tower and the space-time view). Falls back to the
stock widget when OpenGL is unavailable (``GLView is None``).
"""
from __future__ import annotations

import numpy as np

try:
    import pyqtgraph.opengl as _gl
    from PySide6 import QtGui
    _HAVE_GL = True
except Exception:                                  # pragma: no cover - no OpenGL in this env
    _HAVE_GL = False


if _HAVE_GL:

    class GLView(_gl.GLViewWidget):
        """``GLViewWidget`` with a pole-safe ``view-upright`` pan (see module docstring)."""

        # Within this many degrees of straight up/down, use the analytic in-plane basis instead of
        # the cross-product (which is numerically degenerate there).
        _POLE_DEG = 89.5

        def pan(self, dx, dy, dz, relative="global"):
            opts = self.opts
            if str(relative) == "view-upright" and abs(float(opts["elevation"])) >= self._POLE_DEG:
                cVec = opts["center"] - self.cameraPosition()
                dist = cVec.length()
                xDist = dist * 2.0 * np.tan(0.5 * np.radians(float(opts["fov"])))
                xScale = xDist / max(self.width(), 1)
                az = np.radians(float(opts["azimuth"]))
                # The limit of cross(z, cVec)/|…| and cross(xVec, z) as elevation → ±90°, which is
                # what pyqtgraph itself uses just shy of the pole — so the pan stays continuous.
                xVec = QtGui.QVector3D(float(np.sin(az)), float(-np.cos(az)), 0.0)
                yVec = QtGui.QVector3D(float(-np.cos(az)), float(-np.sin(az)), 0.0)
                zVec = QtGui.QVector3D(0.0, 0.0, 1.0)
                opts["center"] = opts["center"] + xVec * xScale * dx + yVec * xScale * dy \
                    + zVec * xScale * dz
                self.update()
            else:
                super().pan(dx, dy, dz, relative=relative)

else:                                              # pragma: no cover
    GLView = None
