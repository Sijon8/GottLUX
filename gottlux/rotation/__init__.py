"""
gottlux.rotation — the rotating-payload rotation/fusion/tracker/viz suite, built on the
gottlux :class:`~gottlux.io.recording.Recording` data model.

This subpackage carries everything specific to a *spinning* sensor:

* **geometry / rotation** — de-rotation to a world frame, the frozen *N-rotation* background
  reference + ``masksweep`` drop-off, and accumulation-independent **per-pass centroiding**
  (the ~0.03° bearing-SE result);
* **dual-EBS** — co-registration, **fusion**, and timing-vs-boresight **calibration**;
* **trackers** — the regime-split suite (``nearest, single, kalman, cmax, staring_kvf,
  hummingbird, drone_fft``), also exposed as detectors in the unified registry;
* **metrics** — quantitative coverage / revisit / localization figures of merit;
* **viz** — the rotation-analysis figure & video suite (radar map + sweep, MTI, rate-surface
  video, validation video, panorama sweep video, mask-sweep, tracking report).

The ported modules speak the classic EBS ``ev`` dict (``x, y, p, t[µs], width, height, n``).
:func:`ev_dict` bridges a Recording/``EventWindow`` into that contract, and
:func:`build_context` runs the shared front-end (background → isolate → trajectory →
centroids) so the viz and trackers can be driven straight from a Recording.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def ev_dict(src) -> dict:
    """Adapt a :class:`Recording` or :class:`EventWindow` into the EBS ``ev`` dict contract.

    The arrays are taken as views (``np.asarray`` on a memmap does not copy), so this is
    cheap even for multi-GB recordings.
    """
    x = np.asarray(src.x)
    y = np.asarray(src.y)
    p = np.asarray(src.p)
    t = np.asarray(src.t)
    return {"x": x, "y": y, "p": p, "t": t,
            "width": int(src.width), "height": int(src.height),
            "n": int(x.shape[0]), "n_on": int((p == 1).sum()),
            "fmt": getattr(src, "fmt", "unknown")}


def resolve_cfg(rec, cfg=None):
    """Fill a Config's runtime geometry from a Recording and resolve ``mode="auto"``."""
    from gottlux.config import Config
    cfg = cfg if cfg is not None else Config()
    cfg.sensor_w = int(rec.width)
    cfg.sensor_h = int(rec.height)
    if cfg.fov_deg is None:
        cfg.fov_deg = cfg.resolved_fov() or 50.0
    if cfg.mode == "auto":
        cfg.mode = "rotation" if rec.is_rotating else "staring"
    return cfg


@dataclass
class RotationContext:
    """The shared rotation front-end's output: everything the viz/trackers/metrics need."""
    rec: object
    ev: dict
    cfg: object
    tel: Optional[object]
    hot: object
    drop: object
    keep: Optional[object] = None
    dets: Optional[object] = None
    traj: Optional[dict] = None
    centroids: Optional[object] = None      # per_pass_centroids array (P, 7)

    @property
    def mode(self):
        return self.cfg.mode


def build_context(rec, cfg=None, isolate=True) -> RotationContext:
    """Run the shared rotation front-end on a Recording.

    Steps: hot-pixel mask → background drop mask (frozen N-rotation reference in rotation
    mode, persistent-pixel mask in staring mode) → (optionally) target isolation → trajectory
    (bearing/elev/range) → accumulation-independent per-pass centroids. Returns a
    :class:`RotationContext` that the viz suite, trackers, and metrics all consume.
    """
    from gottlux.rotation import background, centroid, detect
    cfg = resolve_cfg(rec, cfg)
    ev = ev_dict(rec.all())
    tel = rec.telemetry if cfg.mode == "rotation" else None
    hot = background.hot_pixel_mask(ev, cfg.hot_pixel_pct)
    if cfg.mode == "rotation" and tel is not None and cfg.use_ref_mask:
        ref_end = background.reference_end_time(tel, cfg.mask_rotations)
        ref = background.build_reference(ev, tel, ref_end, n_phase=cfg.n_phase)
        drop = background.rotation_drop_mask(ev, tel, ref, cfg.n_phase, hot)
    elif cfg.mode == "rotation" and tel is not None:
        drop = hot[ev["y"], ev["x"]]
    else:
        drop = background.staring_drop_mask(ev, hot, cfg.bg_window_s)
    ctx = RotationContext(rec=rec, ev=ev, cfg=cfg, tel=tel, hot=hot, drop=drop)
    if isolate:
        keep, dets = detect.isolate_target(ev, drop, cfg.accum_dt, cfg.min_pixels,
                                           cfg.cluster_dilation, cfg.cluster_erode)
        ctx.keep, ctx.dets = keep, dets
        ctx.traj = detect.build_trajectory(dets, cfg, tel)
        ctx.centroids = (centroid.per_pass_centroids(ev, keep, cfg, tel)
                         if bool(np.any(keep)) else np.zeros((0, 7)))
    return ctx
