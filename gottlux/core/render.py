"""
render.py — the one canonical event-frame render pipeline.

The same sequence — *window → live filters → accumulate(mode) → tone-map(expr/gamma/scale)* —
was duplicated across the Live viewer, the Multi-clip panes, and the Range lab (in both their
on-screen ``_render`` and their faithful ``capture_frame``). That duplication is exactly where
subtle drift and bugs hide, so it lives here once. Pure NumPy (no Qt, no matplotlib).

:func:`render_frame` returns ``(disp, levels, vmax, win)``:

* ``disp``   — the tone-mapped array ready to display / colourize,
* ``levels`` — the ``(lo, hi)`` display range for it,
* ``vmax``   — the white-point actually used (so a "static / frozen scale" view can hold it),
* ``win``    — the :class:`~gottlux.io.recording.EventWindow` (for the view's readout).
"""
from __future__ import annotations

from gottlux.core import tonemap
from gottlux.core.accumulate import accumulate_frame


def render_frame(rec, t, dt, *, mode="count", expr="sqrt", gamma=0.5,
                 vmax_ref=None, filters=None, back=False):
    """Render one event frame through the shared pipeline. See module docstring.

    *t* is the cursor time and *dt* the integration window. ``back=False`` (default) integrates
    AHEAD of the cursor — ``[t, t+dt]`` — while ``back=True`` integrates BEHIND it — ``[t-dt, t]``.
    Either way the newest events sit at the window's upper edge, which is the time-surface 'now'.
    """
    if back:
        a, b = max(rec.t_start_s, t - dt), min(t, rec.t_stop_s)
    else:
        a, b = max(rec.t_start_s, t), min(t + dt, rec.t_stop_s)
    win = rec.window(a, b)
    if filters is not None:
        win = filters.apply(win)
    frame = accumulate_frame(win, mode=mode, tau=dt, ref_time_us=b * 1e6)
    if mode in ("time_surface", "binary"):
        return frame, (0.0, 1.0), None, win
    if mode == "polarity_ratio":
        return frame, (-1.0, 1.0), None, win
    if mode == "polarity":
        disp, vmax = tonemap.compress_signed(frame, expr=expr, vmax=vmax_ref, gamma=gamma)
        return disp, (-1.0, 1.0), vmax, win
    disp, vmax = tonemap.compress(frame, expr=expr, vmax=vmax_ref, gamma=gamma)
    return disp, (0.0, 1.0), vmax, win
