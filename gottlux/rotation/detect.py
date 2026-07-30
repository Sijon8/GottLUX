"""
detect.py  --  Mode-agnostic target isolation, pinhole ranging, trajectory.

isolate_target() takes a precomputed `drop_mask` (from background.py: the frozen
rotation reference for ROTATION mode, or the persistent-pixel mask for STARING
mode) so the same clustering core serves both modes.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage


def focal_px(fov_deg, sensor_px=320):
    return (sensor_px / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)


def estimate_range_m(size_px, fov_deg, phys_m, sensor_px=320):
    if phys_m <= 0:
        return np.full_like(np.asarray(size_px, float), np.nan)
    return phys_m * focal_px(fov_deg, sensor_px) / np.maximum(size_px, 1.0)


def isolate_target(ev, drop_mask, accum_dt=0.01, min_pixels=80, dilation=3, erode=1):
    """Return (keep_mask over events, dets[N,7]=t,cx,cy,area,nev,dx,dy)."""
    W, H = ev["width"], ev["height"]
    x, y, t = np.asarray(ev["x"]), np.asarray(ev["y"]), np.asarray(ev["t"]) / 1e6
    N = len(t)
    cand = ~drop_mask
    se = ndimage.generate_binary_structure(2, 2)
    keep = np.zeros(N, bool)
    dets = []
    ci = np.where(cand)[0]
    order = np.argsort(t[ci], kind="stable")
    ci = ci[order]
    ct, cx_, cy_ = t[ci], x[ci], y[ci]
    edges = np.arange(0, t.max() + accum_dt, accum_dt)
    lo_all = np.searchsorted(ct, edges[:-1]); hi_all = np.searchsorted(ct, edges[1:])
    for fi in range(len(edges) - 1):
        lo, hi = lo_all[fi], hi_all[fi]
        if hi - lo < min_pixels:
            continue
        fx, fy = cx_[lo:hi], cy_[lo:hi]
        img = np.zeros((H, W), bool); img[fy, fx] = True
        bw = ndimage.binary_erosion(ndimage.binary_dilation(img, iterations=dilation), iterations=erode)
        lab, nlab = ndimage.label(bw, structure=se)
        if not nlab:
            continue
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, nlab + 1))
        good = np.where(sizes >= min_pixels)[0] + 1
        if not len(good):
            continue
        lab_ev = lab[fy, fx]
        keep[ci[lo:hi][np.isin(lab_ev, good)]] = True
        for g in good:
            sel = lab_ev == g
            if sel.sum() < min_pixels:
                continue
            sx, sy = fx[sel], fy[sel]
            dets.append((0.5 * (edges[fi] + edges[fi + 1]), float(sx.mean()), float(sy.mean()),
                         float(sizes[g - 1]), int(sel.sum()),
                         float(sx.max() - sx.min() + 1), float(sy.max() - sy.min() + 1)))
    return keep, (np.array(dets) if dets else np.zeros((0, 7)))


def build_trajectory(dets, cfg, tel=None, prefix="cam1"):
    """Return dict of trajectory arrays with bearing/elevation/range.

    ROTATION: bearing = pan azimuth(t) + intra-FOV correction.
    STARING : bearing = relative angle within FOV from pixel x (boresight = 0)."""
    if len(dets) == 0:
        return {}
    W, H = cfg.sensor_w, cfg.sensor_h
    deg_per_px = cfg.fov_deg / W
    elev = (H / 2 - dets[:, 2]) * deg_per_px          # cy (vertical) -> elevation about the HEIGHT centre
    rng = estimate_range_m(dets[:, 6], cfg.fov_deg, cfg.target_diag_m, W)
    if cfg.mode == "rotation" and tel is not None:
        pan = np.rad2deg(np.interp(dets[:, 0] - tel.offset, tel.t, tel.azimuth_unwrapped()))
        az = np.mod(pan + cfg.az_sign * (dets[:, 1] - W / 2) * deg_per_px, 360.0)
    else:
        az = cfg.az_sign * (dets[:, 1] - W / 2) * deg_per_px   # relative bearing
    # altitude Z above boresight plane (V26 kinematic ranging): Z = D * tan(elev)
    altitude_z = rng * np.tan(np.deg2rad(elev))
    return dict(t=dets[:, 0], azimuth_deg=az, elev_deg=elev, range_m=rng,
                altitude_z_m=altitude_z, cx=dets[:, 1], cy=dets[:, 2],
                dx=dets[:, 5], dy=dets[:, 6], area=dets[:, 3], n_events=dets[:, 4])


def derotate_events(ev, keep, cfg, tel=None):
    """World azimuth per KEPT event (for point clouds / exports)."""
    x = np.asarray(ev["x"])[keep].astype(np.float64)
    t = np.asarray(ev["t"])[keep] / 1e6
    deg_per_px = cfg.fov_deg / cfg.sensor_w
    if cfg.mode == "rotation" and tel is not None:
        pan = np.rad2deg(np.interp(t - tel.offset, tel.t, tel.azimuth_unwrapped()))
        return np.mod(pan + cfg.az_sign * (x - cfg.sensor_w / 2) * deg_per_px, 360.0)
    return cfg.az_sign * (x - cfg.sensor_w / 2) * deg_per_px
