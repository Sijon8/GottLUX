"""
gottlux.export_tools — hand-written standalone-script templates for ``--export-tool``.

Each tool in this package is a *pair of self-contained scripts* — one Python, one MATLAB —
that re-implements one gottlux analysis with **no gottlux import**:

* the Python script depends only on ``numpy`` + ``h5py`` (+ ``scipy`` where noted);
  ``matplotlib`` is strictly optional (plots are skipped without it);
* the MATLAB script uses only base MATLAB (``h5read``/``h5info``, no toolboxes).

Both operate on any GottLUX-exported HDF5 event file (``gottlux INPUT --to-hdf5``, or the
``data.h5`` inside an exported bundle) — the Metavision-compatible compound ``CD/events``
layout *and* the plain parallel ``x/y/p/t`` fallback.

Templates are plain strings parameterized by simple ``{placeholder}`` tokens; the exporter
(:mod:`gottlux.run.tool_export`) bakes the user's current CLI values (window, ROI, band,
sensor geometry, …) in as ordinary variables at the top of the generated script, so the
recipient can edit them with a text editor. The math is a deliberate, honest simplification
of the corresponding gottlux module (named in each script's header) — enough to reproduce
the headline number, small enough to read in one sitting.

Registry: :data:`TOOLS` maps tool name → :class:`ExportTool`; :func:`render` substitutes
placeholders (and refuses to emit a script with any left unresolved).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ExportTool:
    """One exportable tool: a Python + MATLAB template pair and its baked-parameter manifest."""
    name: str
    description: str        # one line, shown by '--export-tool list'
    module: str             # the gottlux module the math is ported from
    outputs: tuple          # (filename, what-it-holds) pairs the scripts write
    params: tuple           # (placeholder_key, human description) of the baked knobs
    py_template: str
    m_template: str


# ======================================================================================
# The shared event-file loaders, injected into every template as {loader}. They are the
# only non-trivial code all the tools share; keeping one copy here keeps the hand-written
# templates focused on their actual math while every GENERATED script stays self-contained.
# ======================================================================================
PY_LOADER = '''\
def load_events(path):
    """Read a GottLUX-exported HDF5 event file -> (x, y, p, t_s, width, height).

    Accepts the Metavision-compatible compound 'CD/events' layout gottlux writes, a
    root-level compound 'events' dataset, and plain parallel x/y/p/t datasets (at the
    root or under an 'events/' group). Times come back in SECONDS, sorted and re-zeroed
    to the first event; p is 0/1 (1 = ON).
    """
    import h5py
    with h5py.File(path, "r") as f:
        node = None
        for key in ("CD/events", "events"):
            if key in f:
                node = f[key]
                break
        if node is not None and hasattr(node, "dtype") and node.dtype.names:
            rows = node[...]                              # compound layout
            x = rows["x"].astype(np.int64)
            y = rows["y"].astype(np.int64)
            p = (rows["p"] > 0).astype(np.uint8)
            t = rows["t"].astype(np.float64)
        else:                                             # plain parallel datasets
            grp = node if node is not None else f
            x = np.asarray(grp["x"], np.int64)
            y = np.asarray(grp["y"], np.int64)
            p = (np.asarray(grp["p"]) > 0).astype(np.uint8)
            t = np.asarray(grp["t"], np.float64)
        try:
            width = int(f.attrs["width"])
            height = int(f.attrs["height"])
        except Exception:
            width = int(x.max()) + 1 if x.size else 1
            height = int(y.max()) + 1 if y.size else 1
    order = np.argsort(t, kind="stable")
    x, y, p, t = x[order], y[order], p[order], t[order]
    t_s = (t - (t[0] if t.size else 0.0)) / 1e6           # microseconds -> zero-based seconds
    return x, y, p, t_s, width, height


def input_path(default="data.h5"):
    """The events file: argv[1] if given, else data.h5 beside this script."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), default)


def try_pyplot():
    """Matplotlib is OPTIONAL: return pyplot (Agg backend) or None — never fail."""
    try:
        import matplotlib
        matplotlib.use("Agg")                             # write files, never open a window
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None
'''

M_LOADER = '''\
function [x, y, p, t_s, w, h] = gl_load_events(path)
% Read a GottLUX-exported HDF5 event file (compound CD/events, or plain x/y/p/t).
% Returns times in SECONDS, sorted and re-zeroed to the first event; p is 0/1.
try
    ev = h5read(path, '/CD/events');                 % Metavision-compatible compound layout
    x = double(ev.x); y = double(ev.y); p = double(ev.p > 0); t = double(ev.t);
catch
    try
        x = double(h5read(path, '/events/x')); y = double(h5read(path, '/events/y'));
        p = double(h5read(path, '/events/p') > 0); t = double(h5read(path, '/events/t'));
    catch
        x = double(h5read(path, '/x')); y = double(h5read(path, '/y'));
        p = double(h5read(path, '/p') > 0); t = double(h5read(path, '/t'));
    end
end
try
    w = double(h5readatt(path, '/', 'width'));
    h = double(h5readatt(path, '/', 'height'));
catch
    w = max(x) + 1;
    h = max(y) + 1;
end
[t, order] = sort(t);
x = x(order); y = y(order); p = p(order);
t_s = (t - t(1)) / 1e6;
end
'''


# ======================================================================================
# Registry — importing the tool modules here keeps `from gottlux.export_tools import TOOLS`
# the single lookup the exporter, CLI listing, and tests all share.
# ======================================================================================
from gottlux.export_tools import (  # noqa: E402
    centroid_tracker,
    event_frames,
    event_rate,
    flicker_map,
    region_spectrum,
    viz_config,
)

TOOLS: dict[str, ExportTool] = {
    t.name: t for t in (event_frames.TOOL, event_rate.TOOL, region_spectrum.TOOL,
                        flicker_map.TOOL, centroid_tracker.TOOL, viz_config.TOOL)
}

#: Any {identifier} brace token — what render() must leave none of.
_PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


def render(template: str, mapping: dict) -> str:
    """Substitute ``{key}`` tokens from *mapping*; refuse to emit an unresolved script.

    Only tokens whose key is in *mapping* are touched (templates contain no other brace
    constructs — that is a reviewed property of the hand-written templates, enforced here).
    """
    out = template
    for key, val in mapping.items():
        out = out.replace("{" + key + "}", str(val))
    left = sorted(set(m.group(0) for m in _PLACEHOLDER.finditer(out)))
    if left:
        raise KeyError(f"unresolved template placeholder(s): {', '.join(left)}")
    return out
