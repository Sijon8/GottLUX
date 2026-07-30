"""
uikit.py — small, generic Qt layout helpers for a responsive instrument UI.

Every analysis panel is the same shape: a big growable view on the left, a column of
controls on the right. Historically each one hardcoded that column to a *fixed* pixel width
(196–480 px). That is the root of two complaints: on a laptop the fixed column crowds the
view (and in the side-by-side/split view it became unusable), while on a 4K panel it wastes
space. These helpers replace that pattern with a *draggable* splitter whose control deck has
a sensible **minimum** width, is scroll-wrapped so it never clips, and can be collapsed —
so one arrangement works on a small display and a large one.

Nothing here is gottlux-specific; it is plain Qt.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


def with_progress(parent, title, work, label="Working…"):
    """Run ``work(progress_cb)`` behind a modal progress dialog and return its result.

    *work* receives a ``progress_cb(fraction)`` it should call as it advances (e.g. the streamed
    ``.raw`` writers call it once per block). The dialog pumps the event loop on each callback so
    the bar repaints and the window stays responsive during a long synchronous write — which is
    what stops a multi-GB cut/merge from looking frozen. Best-effort: any dialog error is ignored
    and *work* still runs.
    """
    dlg = None
    try:
        dlg = QtWidgets.QProgressDialog(label, None, 0, 1000, parent)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(QtCore.Qt.WindowModal)
        dlg.setMinimumWidth(380)
        dlg.setAutoClose(False); dlg.setAutoReset(False)
        dlg.setCancelButton(None)                 # a half-written .raw is worse than waiting
        dlg.setValue(0)
        dlg.show()
        QtWidgets.QApplication.processEvents()
    except Exception:
        dlg = None

    def cb(frac):
        if dlg is None:
            return
        try:
            dlg.setValue(int(max(0.0, min(float(frac), 1.0)) * 1000))
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

    try:
        return work(cb)
    finally:
        if dlg is not None:
            dlg.close()


def scrollable(widget, *, horizontal=False, frameless=True):
    """Wrap *widget* in a resizable QScrollArea so its controls scroll instead of clipping.

    Vertical scrolling appears on demand; horizontal is off by default (the deck keeps its
    width via the splitter's minimum, rather than scrolling sideways).
    """
    sa = QtWidgets.QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(widget)
    if frameless:
        sa.setFrameShape(QtWidgets.QFrame.NoFrame)
    sa.setHorizontalScrollBarPolicy(
        QtCore.Qt.ScrollBarAsNeeded if horizontal else QtCore.Qt.ScrollBarAlwaysOff)
    return sa


def reserve_lines(label, n_lines, *, pad_px=6):
    """Pin a wrapping QLabel to a constant height of *n_lines* text lines.

    A live readout that word-wraps several metrics onto a label re-wraps as a value's width
    changes (e.g. the event rate spiking from ``0.5`` to ``12.34 Mev/s``): the wrapped line
    count flips between 1 and 2, the label's height changes, and that reflows the surrounding
    layout every frame — the visible "twitch" where a metric fights between staying inline and
    dropping to a new line. Reserving a fixed height (content wraps *inside* a fixed box,
    top-aligned) decouples the label's footprint from its content, so the layout never moves.
    """
    label.setWordWrap(True)
    h = label.fontMetrics().lineSpacing() * int(n_lines) + pad_px
    label.setMinimumHeight(h)
    label.setMaximumHeight(h)
    label.setSizePolicy(label.sizePolicy().horizontalPolicy(), QtWidgets.QSizePolicy.Fixed)
    label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
    return label


def plot_with_deck(view, deck, *, min_view=240, min_deck=300, init_deck=340,
                   scroll=True, collapsible=True):
    """A horizontal splitter: *view* grows to fill, *deck* is a min-width control column.

    The deck is given a minimum width (and, with ``scroll=True``, wrapped in a scroll area so
    its controls never clip on a short/narrow display). The splitter starts with the deck at
    ``init_deck`` px and the view taking the rest, the divider is draggable, and the deck can
    be collapsed to give the view the whole pane. Returns the :class:`QSplitter`.

    Pass ``scroll=False`` for a deck that is already a ``QTabWidget`` (its tabs manage their
    own space; a scroll area around tabs is awkward).
    """
    view.setMinimumWidth(min_view)
    deck.setMinimumWidth(min_deck)
    wrap = scrollable(deck) if scroll else deck
    wrap.setMinimumWidth(min_deck)

    sp = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    sp.addWidget(view)
    sp.addWidget(wrap)
    sp.setStretchFactor(0, 1)        # the view soaks up extra space when the window grows
    sp.setStretchFactor(1, 0)
    sp.setCollapsible(0, False)
    sp.setCollapsible(1, collapsible)
    # A large left value means "view takes whatever is left after the deck's initial width".
    sp.setSizes([10_000, init_deck])
    return sp
