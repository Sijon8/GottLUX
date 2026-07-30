"""
io_evt21.py — compatibility shim: a module-level ``io_evt21.load`` API mapped onto the
gottlux streaming decoder.

The ported EBS rotation modules (and the ``gottlux-calibrate`` path) call
``io_evt21.load(raw)`` and expect the classic ``ev`` dict (``x, y, p, t[µs], width, height,
n, n_on, fmt, meta``) backed by a decode-once memmap cache. The gottlux
:func:`gottlux.io.cache.load` already returns exactly that contract, so this shim simply
forwards to it — one decoder for the whole system.
"""
from __future__ import annotations

from gottlux.io import cache as _cache


def load(path, cache_dir=None, force=False, progress=None) -> dict:
    """Decode-once into the streaming memmap cache; return the ``ev`` dict (memmap-backed)."""
    return _cache.load(path, cache_dir=cache_dir, force=force, progress=progress)


def decode(path) -> dict:
    """Full in-memory decode (small files / tests)."""
    from gottlux.io import decode as _d
    return _d.decode(path)
