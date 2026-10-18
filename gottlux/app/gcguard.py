"""
gcguard.py — run Python's cyclic garbage collection on the GUI thread only.

CPython's cyclic collector runs inside whichever thread happens to allocate when a generation
threshold is crossed. GottLUX runs Python work on other threads — the screen recorder's
encoder, the Studio export worker, the decode and sampling loaders — so a collection can land
on one of them. If the garbage it frees includes Qt objects (windows and panels held in
reference cycles through signal lambdas), their C++ destructors then run off the GUI thread,
and Qt crashes the process with an access violation or an abort.

:func:`install` switches automatic collection off and collects on a GUI-thread timer instead
(pyqtgraph's :class:`~pyqtgraph.util.garbage_collector.GarbageCollector`: the same generation
thresholds, checked every *interval_s*). Reference counting still frees everything that is not
in a cycle immediately; only cycles wait for the timer.
"""
from __future__ import annotations

_GUARD = None


def install(interval_s: float = 0.5):
    """Start GUI-thread garbage collection (idempotent). Call after the QApplication exists,
    from the GUI thread. Returns the collector."""
    global _GUARD
    if _GUARD is None:
        from pyqtgraph.util.garbage_collector import GarbageCollector
        _GUARD = GarbageCollector(interval=interval_s)
    return _GUARD


def uninstall() -> None:
    """Stop the timer and hand collection back to CPython (tests use this)."""
    global _GUARD
    import gc
    if _GUARD is not None:
        _GUARD.timer.stop()
        _GUARD = None
    gc.enable()
