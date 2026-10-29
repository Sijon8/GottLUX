"""
colorschemes.py — colormap and palette pickers, and the editor that adds your own schemes.

Every colormap picker in the GUI is a :class:`ColormapCombo` and every ON/OFF palette picker
a :class:`PaletteCombo`. Both list the colour registry (:mod:`gottlux.core.colors`) with a
swatch per entry, repopulate themselves the moment a scheme is added or removed anywhere,
and end with **"Add a colour scheme…"**, which opens :class:`ColorSchemesDialog` and selects
the new scheme. Setting a name the list does not show — a colormap a saved composition
used, say — adds that name instead of silently keeping the previous one (which would
otherwise be written back over the saved setting).

:class:`ColorSchemesDialog` (also *View → Colour schemes…*) creates a colormap from colour
stops (pick, add, remove, reverse, even out), starts one from any existing map, creates
ON/OFF/background palettes, and imports or exports schemes as ``.json`` to share them.
User schemes are saved at once to :func:`gottlux.core.colors.schemes_path`.
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from gottlux.core import colors

_ADD_TEXT = "Add a colour scheme…"
_ADD_ROLE = QtCore.Qt.UserRole + 7


# ====================================================================================
# Swatches
# ====================================================================================
_icon_cache: dict = {}


def gradient_image(name, w=96, h=14) -> QtGui.QImage:
    """A horizontal ramp of colormap *name* as a ``QImage``."""
    t = colors.lut_u8(name)
    idx = np.linspace(0, 255, max(int(w), 2)).astype(int)
    row = np.ascontiguousarray(np.repeat(t[idx][None, :, :], max(int(h), 1), axis=0))
    img = QtGui.QImage(row.data, row.shape[1], row.shape[0], 3 * row.shape[1],
                       QtGui.QImage.Format_RGB888)
    return img.copy()


def gradient_icon(name, w=48, h=12) -> QtGui.QIcon:
    key = ("cmap", name, w, h, tuple(colors.lut_u8(name)[::32].ravel()))
    ic = _icon_cache.get(key)
    if ic is None:
        ic = _icon_cache[key] = QtGui.QIcon(QtGui.QPixmap.fromImage(gradient_image(name, w, h)))
    return ic


def palette_icon(name, w=48, h=12) -> QtGui.QIcon:
    pal = colors.palette(name)
    key = ("pal", name, w, h, tuple(pal.values()))
    ic = _icon_cache.get(key)
    if ic is None:
        pm = QtGui.QPixmap(w, h)
        pm.fill(QtGui.QColor(pal["bg"]))
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = h * 0.36
        for cx, key_ in ((w * 0.33, "on"), (w * 0.66, "off")):
            p.setBrush(QtGui.QColor(pal[key_])); p.setPen(QtCore.Qt.NoPen)
            p.drawEllipse(QtCore.QPointF(cx, h / 2), r, r)
        p.end()
        ic = _icon_cache[key] = QtGui.QIcon(pm)
    return ic


def pg_colormap(name):
    """A pyqtgraph ``ColorMap`` for any registry colormap (user schemes included)."""
    import pyqtgraph as pg
    t = colors.lut_u8(name)
    return pg.ColorMap(np.linspace(0.0, 1.0, len(t)), t)


# ====================================================================================
# The pickers
# ====================================================================================
class _SchemeCombo(QtWidgets.QComboBox):
    """Shared behaviour: registry-backed items, live refresh, the 'Add…' entry."""

    kind = "colormap"

    def __init__(self, parent=None, *, include_add=True, names=None):
        super().__init__(parent)
        self._fixed = list(names) if names is not None else None
        self._include_add = include_add
        self._extra = []                     # names set from outside the registry list
        self._last = ""
        self.setIconSize(QtCore.QSize(48, 12))
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.currentIndexChanged.connect(self._on_index)
        self._populate()
        colors.subscribe(self._refresh)
        self.destroyed.connect(lambda *_: colors.unsubscribe(self._refresh))

    # --- the list ------------------------------------------------------------------
    def _names(self) -> list:
        raise NotImplementedError

    def _icon(self, name) -> QtGui.QIcon:
        raise NotImplementedError

    def _populate(self, keep=None):
        keep = keep if keep is not None else self.currentText()
        b = QtCore.QSignalBlocker(self)
        self.clear()
        names = self._names()
        for n in self._extra:
            if n not in names:
                names.append(n)
        for n in names:
            self.addItem(self._icon(n), n)
        if self._include_add:
            self.insertSeparator(self.count())
            self.addItem(_ADD_TEXT)
            self.setItemData(self.count() - 1, True, _ADD_ROLE)
        i = self.findText(keep) if keep else -1
        self.setCurrentIndex(i if i >= 0 else 0)
        self._last = self.currentText()
        del b

    def _refresh(self):
        try:
            self._populate(self._last)
        except RuntimeError:                 # the C++ side is gone
            colors.unsubscribe(self._refresh)

    def names(self) -> list:
        """The selectable scheme names (no separator, no 'Add…' entry)."""
        return [self.itemText(i) for i in range(self.count())
                if self.itemText(i) and not self.itemData(i, _ADD_ROLE)]

    # --- selection -----------------------------------------------------------------
    def setCurrentText(self, name):          # noqa: N802 (Qt API)
        """Select *name*; a valid scheme the list does not show yet is added for it."""
        name = str(name)
        if self.findText(name) < 0 and name and name != _ADD_TEXT and self._valid(name):
            self._extra.append(name)
            self._populate(keep=self._last)
        i = self.findText(name)
        if i >= 0:
            self.setCurrentIndex(i)

    def _valid(self, name) -> bool:
        raise NotImplementedError

    def _on_index(self, i):
        if self.itemData(i, _ADD_ROLE):
            b = QtCore.QSignalBlocker(self)
            j = self.findText(self._last)
            self.setCurrentIndex(j if j >= 0 else 0)
            del b
            QtCore.QTimer.singleShot(0, self._open_editor)
            return
        self._last = self.itemText(i)

    def _open_editor(self):
        dlg = ColorSchemesDialog(self.window(), start=self.kind)
        dlg.exec()
        made = dlg.last_created(self.kind)
        if made:
            self.setCurrentText(made)


class ColormapCombo(_SchemeCombo):
    """A colormap picker over the colour registry (built-ins, then user schemes)."""

    kind = "colormap"

    def _names(self):
        return list(self._fixed) if self._fixed is not None else colors.colormap_names()

    def _icon(self, name):
        return gradient_icon(name)

    def _valid(self, name):
        return colors.has_colormap(name)


class PaletteCombo(_SchemeCombo):
    """An ON/OFF/background palette picker over the colour registry."""

    kind = "palette"

    def _names(self):
        return list(self._fixed) if self._fixed is not None else colors.palette_names()

    def _icon(self, name):
        return palette_icon(name)

    def _valid(self, name):
        return colors.has_palette(name)


# ====================================================================================
# The editor
# ====================================================================================
class _Gradient(QtWidgets.QWidget):
    """A live preview strip of colour stops."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stops, self.positions = ["#000000", "#ffffff"], None
        self.setMinimumHeight(34)

    def set_stops(self, stops, positions=None):
        self.stops, self.positions = list(stops), positions
        self.update()

    def paintEvent(self, _ev):               # noqa: N802 (Qt API)
        p = QtGui.QPainter(self)
        r = self.rect().adjusted(1, 1, -1, -1)
        g = QtGui.QLinearGradient(r.left(), 0, r.right(), 0)
        n = len(self.stops)
        pos = self.positions or [i / max(n - 1, 1) for i in range(n)]
        for x, c in zip(pos, self.stops, strict=False):
            g.setColorAt(float(x), QtGui.QColor(c))
        p.fillRect(r, g)
        p.setPen(self.palette().color(QtGui.QPalette.Mid))
        p.drawRect(r)
        p.end()


class _ColorButton(QtWidgets.QPushButton):
    """A swatch button: click to pick a colour."""

    colorChanged = QtCore.Signal(str)

    def __init__(self, color="#ffffff", parent=None):
        super().__init__(parent)
        self.setFixedSize(34, 24)
        self.set_color(color)
        self.clicked.connect(self._pick)

    def set_color(self, c):
        self._c = colors.rgb01_to_hex(colors.hex_to_rgb01(c))
        self.setStyleSheet(f"QPushButton {{ background: {self._c}; border: 1px solid #888;"
                           " border-radius: 3px; }")
        self.setToolTip(self._c)

    def color(self) -> str:
        return self._c

    def _pick(self):
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._c), self, "Pick a colour")
        if c.isValid():
            self.set_color(c.name())
            self.colorChanged.emit(self._c)


class ColorSchemesDialog(QtWidgets.QDialog):
    """Create, edit, delete, import and export colour schemes (colormaps and palettes)."""

    def __init__(self, parent=None, start="colormap"):
        super().__init__(parent)
        self.setWindowTitle("Colour schemes — GottLUX")
        self.resize(720, 440)
        self._created = {"colormap": None, "palette": None}
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_maps(), "Colormaps")
        self.tabs.addTab(self._build_palettes(), "ON / OFF palettes")
        self.tabs.setCurrentIndex(1 if start == "palette" else 0)

        path = QtWidgets.QLabel(f"Your schemes are saved in {colors.schemes_path()}")
        path.setObjectName("muted"); path.setWordWrap(True)
        path.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        imp = QtWidgets.QPushButton("Import…")
        imp.setToolTip("Add the schemes in a shared .json file.")
        imp.clicked.connect(self._import)
        exp = QtWidgets.QPushButton("Export…")
        exp.setToolTip("Save your schemes (or the selected one) to a .json file to share.")
        exp.clicked.connect(self._export)
        close = QtWidgets.QPushButton("Close"); close.setDefault(True)
        close.clicked.connect(self.accept)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(imp); bottom.addWidget(exp); bottom.addStretch(1); bottom.addWidget(close)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.tabs, 1)
        lay.addWidget(path)
        lay.addLayout(bottom)
        self._fill_maps()
        self._fill_pals()

    def last_created(self, kind) -> str | None:
        """The scheme of *kind* this dialog last saved (``None`` if none)."""
        return self._created.get(kind)

    # ------------------------------------------------------------------ colormaps
    def _build_maps(self):
        w = QtWidgets.QWidget()
        self.map_list = QtWidgets.QListWidget()
        self.map_list.setIconSize(QtCore.QSize(64, 12))
        self.map_list.currentRowChanged.connect(self._load_map)
        self.map_name = QtWidgets.QLineEdit()
        self.map_name.setPlaceholderText("name of your colormap")
        self.map_preview = _Gradient()
        self._stop_box = QtWidgets.QHBoxLayout()
        self._stop_box.setSpacing(4)
        self._stops: list = []
        stops_w = QtWidgets.QWidget(); stops_w.setLayout(self._stop_box)
        add = QtWidgets.QPushButton("+ stop"); add.setToolTip("Add a colour stop at the end.")
        add.clicked.connect(lambda: self._add_stop(self._stops[-1].color() if self._stops
                                                   else "#ffffff"))
        rm = QtWidgets.QPushButton("− stop"); rm.setToolTip("Remove the last colour stop.")
        rm.clicked.connect(self._remove_stop)
        rev = QtWidgets.QPushButton("Reverse"); rev.setToolTip("Flip the ramp end to end.")
        rev.clicked.connect(self._reverse)
        self.map_save = QtWidgets.QPushButton("Save colormap")
        self.map_save.setObjectName("primary")
        self.map_save.setToolTip("Save under the name above (a new name adds a scheme; your "
                                 "own scheme's name replaces it).")
        self.map_save.clicked.connect(self._save_map)
        self.map_del = QtWidgets.QPushButton("Delete")
        self.map_del.setToolTip("Delete the selected scheme of yours (built-ins stay).")
        self.map_del.clicked.connect(self._delete_map)
        self.map_hint = QtWidgets.QLabel("")
        self.map_hint.setObjectName("muted"); self.map_hint.setWordWrap(True)
        tools = QtWidgets.QHBoxLayout()
        for b in (add, rm, rev):
            tools.addWidget(b)
        tools.addStretch(1)
        right = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()
        form.addRow("Name", self.map_name)
        form.addRow("Preview", self.map_preview)
        form.addRow("Stops", stops_w)
        right.addLayout(form)
        right.addLayout(tools)
        right.addWidget(self.map_hint)
        right.addStretch(1)
        act = QtWidgets.QHBoxLayout()
        act.addStretch(1); act.addWidget(self.map_del); act.addWidget(self.map_save)
        right.addLayout(act)
        h = QtWidgets.QHBoxLayout(w)
        h.addWidget(self.map_list, 2)
        h.addLayout(right, 3)
        return w

    def _fill_maps(self, select=None):
        b = QtCore.QSignalBlocker(self.map_list)
        self.map_list.clear()
        for n in colors.colormap_names():
            it = QtWidgets.QListWidgetItem(gradient_icon(n, 64, 12),
                                           n + ("" if colors.is_user_colormap(n) else
                                                "   (built-in)"))
            it.setData(QtCore.Qt.UserRole, n)
            self.map_list.addItem(it)
        del b
        names = colors.colormap_names()
        row = names.index(select) if select in names else 0
        self.map_list.setCurrentRow(row)
        self._load_map(row)

    def _map_at(self, row):
        it = self.map_list.item(row)
        return it.data(QtCore.Qt.UserRole) if it is not None else None

    def _load_map(self, row):
        name = self._map_at(row)
        if not name:
            return
        stops = colors.colormap_stops(name)
        if stops is None:                    # a matplotlib map: start from 9 samples of it
            stops = [colors.rgb01_to_hex(c) for c in colors.lut(name)[::32]]
            stops.append(colors.rgb01_to_hex(colors.lut(name)[-1]))
        user = colors.is_user_colormap(name)
        self.map_name.setText(name if user else f"my {name}")
        self.map_del.setEnabled(user)
        self.map_hint.setText("" if user else
                              "A built-in stays as it is: edit the stops and save under "
                              "a new name to make your own version.")
        self._set_stops(stops)

    def _set_stops(self, stops):
        while self._stops:
            self._stop_box.removeWidget(self._stops[-1])
            self._stops.pop().deleteLater()
        for c in stops:
            self._add_stop(c, update=False)
        self._update_preview()

    def _add_stop(self, color, update=True):
        if len(self._stops) >= 16:
            return
        btn = _ColorButton(color)
        btn.colorChanged.connect(lambda *_: self._update_preview())
        self._stops.append(btn)
        self._stop_box.addWidget(btn)
        if update:
            self._update_preview()

    def _remove_stop(self):
        if len(self._stops) > 2:
            btn = self._stops.pop()
            self._stop_box.removeWidget(btn)
            btn.deleteLater()
            self._update_preview()

    def _reverse(self):
        self._set_stops([b.color() for b in self._stops][::-1])

    def _update_preview(self):
        self.map_preview.set_stops([b.color() for b in self._stops])

    def _save_map(self):
        name = self.map_name.text().strip()
        try:
            name = colors.add_colormap(name, [b.color() for b in self._stops])
        except (ValueError, OSError) as e:
            QtWidgets.QMessageBox.warning(self, "Save colormap", str(e))
            return
        self._created["colormap"] = name
        self._fill_maps(select=name)

    def _delete_map(self):
        name = self._map_at(self.map_list.currentRow())
        if name and colors.is_user_colormap(name):
            colors.remove_colormap(name)
            self._fill_maps()

    # ------------------------------------------------------------------ palettes
    def _build_palettes(self):
        w = QtWidgets.QWidget()
        self.pal_list = QtWidgets.QListWidget()
        self.pal_list.setIconSize(QtCore.QSize(48, 14))
        self.pal_list.currentRowChanged.connect(self._load_pal)
        self.pal_name = QtWidgets.QLineEdit()
        self.pal_name.setPlaceholderText("name of your palette")
        self.pal_btn = {k: _ColorButton() for k in ("on", "off", "bg")}
        self.pal_save = QtWidgets.QPushButton("Save palette")
        self.pal_save.setObjectName("primary")
        self.pal_save.clicked.connect(self._save_pal)
        self.pal_del = QtWidgets.QPushButton("Delete")
        self.pal_del.clicked.connect(self._delete_pal)
        form = QtWidgets.QFormLayout()
        form.addRow("Name", self.pal_name)
        form.addRow("ON events", self.pal_btn["on"])
        form.addRow("OFF events", self.pal_btn["off"])
        form.addRow("Background", self.pal_btn["bg"])
        note = QtWidgets.QLabel("Palettes colour the polarity looks (trails, time surface, "
                                "stroboscope, space-time, radar). On a dark background ON "
                                "and OFF add up to white where they overlap.")
        note.setObjectName("muted"); note.setWordWrap(True)
        right = QtWidgets.QVBoxLayout()
        right.addLayout(form); right.addWidget(note); right.addStretch(1)
        act = QtWidgets.QHBoxLayout()
        act.addStretch(1); act.addWidget(self.pal_del); act.addWidget(self.pal_save)
        right.addLayout(act)
        h = QtWidgets.QHBoxLayout(w)
        h.addWidget(self.pal_list, 2)
        h.addLayout(right, 3)
        return w

    def _fill_pals(self, select=None):
        b = QtCore.QSignalBlocker(self.pal_list)
        self.pal_list.clear()
        for n in colors.palette_names():
            it = QtWidgets.QListWidgetItem(palette_icon(n, 48, 14),
                                           n + ("" if colors.is_user_palette(n) else
                                                "   (built-in)"))
            it.setData(QtCore.Qt.UserRole, n)
            self.pal_list.addItem(it)
        del b
        names = colors.palette_names()
        row = names.index(select) if select in names else 0
        self.pal_list.setCurrentRow(row)
        self._load_pal(row)

    def _load_pal(self, row):
        it = self.pal_list.item(row)
        if it is None:
            return
        name = it.data(QtCore.Qt.UserRole)
        pal = colors.palette(name)
        for k, btn in self.pal_btn.items():
            btn.set_color(pal[k])
        user = colors.is_user_palette(name)
        self.pal_name.setText(name if user else f"my {name}")
        self.pal_del.setEnabled(user)

    def _save_pal(self):
        try:
            name = colors.add_palette(self.pal_name.text().strip(),
                                      *(self.pal_btn[k].color() for k in ("on", "off", "bg")))
        except (ValueError, OSError) as e:
            QtWidgets.QMessageBox.warning(self, "Save palette", str(e))
            return
        self._created["palette"] = name
        self._fill_pals(select=name)

    def _delete_pal(self):
        it = self.pal_list.currentItem()
        if it is not None and colors.is_user_palette(it.data(QtCore.Qt.UserRole)):
            colors.remove_palette(it.data(QtCore.Qt.UserRole))
            self._fill_pals()

    # ------------------------------------------------------------------ files
    def _import(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import colour schemes", "", "Colour schemes (*.json)")
        if not path:
            return
        try:
            added = colors.import_schemes(path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Import colour schemes",
                                          f"Could not import {path}:\n{e}")
            return
        self._fill_maps(); self._fill_pals()
        QtWidgets.QMessageBox.information(
            self, "Import colour schemes",
            f"Added {len(added)} scheme(s): {', '.join(added) or 'none'}.")

    def _export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export colour schemes", "gottlux-colors.json", "Colour schemes (*.json)")
        if not path:
            return
        if self.tabs.currentIndex() == 0:
            sel = self._map_at(self.map_list.currentRow())
            maps = [n for n in colors.colormap_names() if colors.is_user_colormap(n)] or [sel]
            pals = [n for n in colors.palette_names() if colors.is_user_palette(n)]
        else:
            it = self.pal_list.currentItem()
            pals = [n for n in colors.palette_names() if colors.is_user_palette(n)] or (
                [it.data(QtCore.Qt.UserRole)] if it is not None else [])
            maps = [n for n in colors.colormap_names() if colors.is_user_colormap(n)]
        try:
            colors.export_schemes(path, maps, pals)
        except OSError as e:
            QtWidgets.QMessageBox.warning(self, "Export colour schemes", str(e))
            return
        QtWidgets.QMessageBox.information(
            self, "Export colour schemes",
            f"Saved {len(maps)} colormap(s) and {len(pals)} palette(s) to\n{path}")
