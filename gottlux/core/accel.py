"""
accel.py — optional Numba JIT acceleration with a transparent pure-NumPy fallback.

The performance-critical inner loops in gottlux (time-surface scatter, per-pixel event
binning, refractory filtering) are written once and JIT-compiled by Numba when it is
available. If Numba is *not* installed, the very same call sites still work — they fall
back to a slower NumPy implementation — so the suite never hard-depends on a compiler.

Use :data:`HAVE_NUMBA` to branch, and :func:`njit` as a decorator that is a no-op when
Numba is missing.
"""
from __future__ import annotations

try:
    from numba import njit, prange       # noqa: F401
    HAVE_NUMBA = True
except Exception:                          # pragma: no cover - portability fallback
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        """No-op stand-in for :func:`numba.njit` when Numba is unavailable."""
        # Support both @njit and @njit(...) usage.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(fn):
            return fn
        return wrap

    def prange(*args, **kwargs):           # plain range fallback
        return range(*args, **kwargs)


def warmup():
    """Trigger Numba compilation of the hot kernels up front (hides first-call latency).

    Safe to call from a background thread at GUI start; a no-op without Numba.
    """
    if not HAVE_NUMBA:
        return
    try:
        import numpy as np
        from gottlux.core.accumulate import _time_surface_kernel
        x = np.zeros(1, np.int64); y = np.zeros(1, np.int64); t = np.zeros(1, np.float64)
        _time_surface_kernel(x, y, t, 1, 1)
    except Exception:
        pass
