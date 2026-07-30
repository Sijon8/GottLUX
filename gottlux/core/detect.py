"""
detect.py — spatial target isolation (blob detection) on the foreground stream.

Given a window of events and a foreground keep-mask (from :mod:`gottlux.core.background`
and/or :mod:`gottlux.core.filters`), this accumulates events into successive
``accum_dt``-second frames, clusters each frame (morphological close → connected
components → minimum-area gate), and emits one :class:`Detection` per surviving blob with
its centroid, bounding box, pixel area and event count.

The blob's apparent size (bbox extent / diagonal) is what downstream ranging turns into a
distance; the centroid is what tracking links over time and geometry turns into bearing /
elevation. This is the spatial half of detection — the temporal-frequency half (does the
blob *flutter* like a rotor?) lives in :mod:`gottlux.core.frequency` and the detectors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Detection:
    """A single blob detected in one accumulation frame."""
    t: float            # frame-centre time (s)
    cx: float           # centroid x (px)
    cy: float           # centroid y (px)
    x0: int             # bbox left
    y0: int             # bbox top
    x1: int             # bbox right (exclusive)
    y1: int             # bbox bottom (exclusive)
    area: int           # connected-component pixel area
    n_events: int       # events inside the blob

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def diag(self) -> float:
        return float(np.hypot(self.width, self.height))

    @property
    def bbox(self):
        return (self.x0, self.y0, self.x1, self.y1)


def detect_blobs(win, keep_mask=None, accum_dt: float = 0.02, min_pixels: int = 60,
                 dilation: int = 3, erode: int = 1):
    """Isolate moving targets frame-by-frame. Returns ``(event_keep_mask, detections)``.

    Parameters
    ----------
    win : EventWindow
    keep_mask : array | None
        Boolean foreground mask over *win*'s events (``True`` = candidate). ``None`` = all.
    accum_dt : float
        Frame integration time (s).
    min_pixels : int
        Minimum connected-component area to accept.
    dilation, erode : int
        Morphology iterations bridging gaps then trimming spurs before labeling.
    """
    x = np.asarray(win.x); y = np.asarray(win.y); t = win.t_s
    W, H = win.width, win.height
    n = len(t)
    cand = np.ones(n, bool) if keep_mask is None else np.asarray(keep_mask, bool)
    se = ndimage.generate_binary_structure(2, 2)
    event_keep = np.zeros(n, bool)
    dets: list[Detection] = []

    ci = np.where(cand)[0]
    if ci.size == 0:
        return event_keep, dets
    order = np.argsort(t[ci], kind="stable")
    ci = ci[order]
    ct, cx_, cy_ = t[ci], x[ci], y[ci]
    t_lo = float(ct[0]); t_hi = float(ct[-1])
    edges = np.arange(t_lo, t_hi + accum_dt, accum_dt)
    lo_all = np.searchsorted(ct, edges[:-1])
    hi_all = np.searchsorted(ct, edges[1:])
    for fi in range(len(edges) - 1):
        lo, hi = lo_all[fi], hi_all[fi]
        if hi - lo < min_pixels:
            continue
        fx, fy = cx_[lo:hi].astype(np.int64), cy_[lo:hi].astype(np.int64)
        img = np.zeros((H, W), bool)
        img[fy, fx] = True
        bw = img
        if dilation:
            bw = ndimage.binary_dilation(bw, iterations=dilation)
        if erode:
            bw = ndimage.binary_erosion(bw, iterations=erode)
        lab, nlab = ndimage.label(bw, structure=se)
        if not nlab:
            continue
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, nlab + 1))
        good = np.where(sizes >= min_pixels)[0] + 1
        if not len(good):
            continue
        lab_ev = lab[fy, fx]
        in_good = np.isin(lab_ev, good)
        event_keep[ci[lo:hi][in_good]] = True
        tc = 0.5 * (edges[fi] + edges[fi + 1])
        for g in good:
            sel = lab_ev == g
            if sel.sum() < min_pixels:
                continue
            sx, sy = fx[sel], fy[sel]
            dets.append(Detection(
                t=float(tc), cx=float(sx.mean()), cy=float(sy.mean()),
                x0=int(sx.min()), y0=int(sy.min()),
                x1=int(sx.max()) + 1, y1=int(sy.max()) + 1,
                area=int(sizes[g - 1]), n_events=int(sel.sum())))
    return event_keep, dets


def cluster_frame(x, y, W, H, min_pixels: int = 40, dilation: int = 2, erode: int = 1):
    """Cluster a single set of (already time-windowed) event coordinates into blobs.

    The streaming primitive used by detectors: rasterize ``(x, y)`` to a binary image,
    morphologically close, connected-component label, gate by area, and return a list of
    ``(cx, cy, x0, y0, x1, y1, area)`` tuples (strongest area first). Faster than
    :func:`detect_blobs` when you already hold one frame's events.
    """
    if len(x) < min_pixels:
        return []
    xi = np.asarray(x).astype(np.int64)
    yi = np.asarray(y).astype(np.int64)
    img = np.zeros((H, W), bool)
    img[yi, xi] = True
    bw = img
    if dilation:
        bw = ndimage.binary_dilation(bw, iterations=dilation)
    if erode:
        bw = ndimage.binary_erosion(bw, iterations=erode)
    se = ndimage.generate_binary_structure(2, 2)
    lab, nlab = ndimage.label(bw, structure=se)
    if not nlab:
        return []
    objs = ndimage.find_objects(lab)
    areas = np.bincount(lab.ravel())
    out = []
    for li in range(1, nlab + 1):
        if areas[li] < min_pixels:
            continue
        sy, sx = objs[li - 1]
        out.append((0.5 * (sx.start + sx.stop), 0.5 * (sy.start + sy.stop),
                    int(sx.start), int(sy.start), int(sx.stop), int(sy.stop), int(areas[li])))
    out.sort(key=lambda c: -c[-1])
    return out


def detections_to_arrays(dets) -> dict:
    """Pack a list of :class:`Detection` into parallel NumPy arrays (for CSV / plotting)."""
    if not dets:
        return {k: np.zeros(0) for k in
                ("t", "cx", "cy", "x0", "y0", "x1", "y1", "area", "n_events", "diag")}
    return dict(
        t=np.array([d.t for d in dets]),
        cx=np.array([d.cx for d in dets]), cy=np.array([d.cy for d in dets]),
        x0=np.array([d.x0 for d in dets]), y0=np.array([d.y0 for d in dets]),
        x1=np.array([d.x1 for d in dets]), y1=np.array([d.y1 for d in dets]),
        area=np.array([d.area for d in dets]),
        n_events=np.array([d.n_events for d in dets]),
        diag=np.array([d.diag for d in dets]))
