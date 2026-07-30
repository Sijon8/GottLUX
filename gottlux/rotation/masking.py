"""
masking.py — rotational background masking as a first-class product: suppress the static world,
keep only relative-motion objects, and turn them into a volumetric (azimuth, elevation, range) map.

This is the original MATLAB rotational-subtraction recipe, productionized. A spinning sensor re-images the static
scene at the same rotation *phase* every revolution, so a frozen phase-space reference (built from
the first *N* revolutions) predicts and removes all static clutter; what survives is moving objects
(drones, birds, wind-blown vegetation). Three deliverables:

* :func:`sweep` — data-rate reduction vs *N* (the >90% reduction headline + how it trades against
  self-masking of a slow target),
* :func:`extract_movers` — the surviving movers, de-rotated to world bearing/elevation and ranged by
  the pinhole model — a single-EBS volumetric moving-object map,
* :func:`run_masking` — bundles both into a :class:`MaskingResult` for the headless analysis.

The de-rotation reuses :func:`gottlux.rotation.rotor_scan.event_world_azimuth`; the static-reference
mask is :mod:`gottlux.rotation.background`. Range is apparent-size (pinhole) — coarse on a
pure-rotation sensor (no triangulation baseline); see ``docs/ROTATIONAL_EBS_METHODS.md`` §3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gottlux.rotation import background as bg
from gottlux.rotation.detect import estimate_range_m
from gottlux.rotation.rotor_scan import event_world_azimuth


# ====================================================================================
# Records
# ====================================================================================
@dataclass
class MaskMetric:
    """Data-rate reduction (and mover concentration) for one reference depth *N*."""
    n_rotations: int
    reduction_pct: float        # 100·(1 − kept/total) — the data-rate reduction
    n_kept: int
    mover_cells: int            # de-rotated (az,elev) cells holding survivors (fewer = cleaner)
    concentration: float        # peak/mean survivor density (higher = movers stand out)


@dataclass
class MovingObject:
    """One surviving moving object in one revolution (a de-rotated, ranged detection)."""
    rev: int
    t_s: float
    bearing_deg: float
    elev_deg: float
    range_m: Optional[float]
    n_events: int
    size_px: float
    above_horizon: bool

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class MaskingResult:
    sweep: list                      # list[MaskMetric]
    movers: list                     # list[MovingObject]
    n_rotations: int                 # the reference depth used for the mover extraction
    reduction_pct: float             # reduction at n_rotations
    fov_deg: float
    t_rot_s: float
    target_size_m: float

    def headline(self) -> dict:
        ah = [m for m in self.movers if m.above_horizon]
        brg = [m.bearing_deg for m in ah]
        rng = [m.range_m for m in ah if m.range_m is not None and np.isfinite(m.range_m)]
        return {
            "n_rotations": self.n_rotations,
            "reduction_pct": round(self.reduction_pct, 1),
            "reduction_sweep": {m.n_rotations: round(m.reduction_pct, 1) for m in self.sweep},
            "n_movers": len(self.movers),
            "n_above_horizon": len(ah),
            "above_horizon_bearing_span_deg": ([round(min(brg), 1), round(max(brg), 1)]
                                               if brg else None),
            "above_horizon_range_median_m": (round(float(np.median(rng)), 1) if rng else None),
        }

    def as_dict(self) -> dict:
        return {
            "n_rotations": self.n_rotations, "reduction_pct": round(self.reduction_pct, 1),
            "fov_deg": self.fov_deg, "t_rot_s": self.t_rot_s, "target_size_m": self.target_size_m,
            "sweep": [vars(m) for m in self.sweep],
            "movers": [m.as_dict() for m in self.movers],
            "headline": self.headline(),
        }


# ====================================================================================
# Core
# ====================================================================================
def keep_mask_cumulative(ev, tel, cfg, *, hot=None, skip_first_rev=True):
    """Cumulative 3D-voxel masking (the cumulative first-occurrence masking recipe): keep only the **first occurrence** of each
    ``(rotation-phase, x, y)`` voxel.

    A static feature lights the same voxel every revolution → kept once, then suppressed; a mover
    enters fresh voxels each revolution → kept every time. Unlike the frozen first-*N* reference this
    needs no "reference window", so it is robust to a **quiet start** (sparse early revolutions) and a
    target that appears late. Returns a boolean keep-mask over events.
    """
    if hot is None:
        hot = bg.hot_pixel_mask(ev, cfg.hot_pixel_pct)
    x = np.asarray(ev["x"]).astype(np.int64)
    y = np.asarray(ev["y"]).astype(np.int64)
    t = np.asarray(ev["t"], np.float64) / 1e6
    T = float(getattr(tel, "T_rot", 0.0)) or (t[-1] - t[0]) / 2.0
    npz = int(cfg.n_phase); W = int(cfg.sensor_w); H = int(cfg.sensor_h)
    phase = tel.phase_at(t) if tel is not None else np.mod(t / T, 1.0)
    theta = np.clip((phase * npz).astype(np.int64), 0, npz - 1)
    cycle = (tel.revolution_at(t).astype(np.int64) if tel is not None
             else np.floor((t - t[0]) / T).astype(np.int64))
    vid = (theta * W + x) * H + y                          # flat voxel id
    first = np.full(npz * W * H, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(first, vid, cycle)
    keep = cycle <= first[vid]                             # first time this voxel ever lit
    if skip_first_rev:
        keep &= (cycle > cycle.min())                     # drop the seed revolution
    keep &= ~hot[y, x]
    return keep


def keep_mask(ev, tel, cfg, n_rotations, hot=None):
    """Boolean over events: True = a *surviving* (moving-object) event after the frozen
    *N*-rotation static reference is subtracted. ``n_rotations=0`` keeps everything but hot pixels."""
    if hot is None:
        hot = bg.hot_pixel_mask(ev, cfg.hot_pixel_pct)
    y = np.asarray(ev["y"]); x = np.asarray(ev["x"])
    if n_rotations <= 0:
        return ~hot[y, x]
    ref = bg.build_reference(ev, tel, bg.reference_end_time(tel, n_rotations), n_phase=cfg.n_phase)
    return ~bg.rotation_drop_mask(ev, tel, ref, cfg.n_phase, hot)


def _derotate(ev, tel, cfg):
    """(world_az_deg, elev_deg, rev, t_s) for every event."""
    x = np.asarray(ev["x"], float); y = np.asarray(ev["y"], float)
    t = np.asarray(ev["t"], float) / 1e6
    waz = event_world_azimuth(x, t, cfg, tel)
    elev = (cfg.sensor_h / 2.0 - y) * (cfg.fov_deg / cfg.sensor_w)
    rev = (tel.revolution_at(t).astype(int) if tel is not None else np.zeros(x.size, int))
    return waz, elev, rev, t


def sweep(ev, tel, cfg, n_list=(0, 1, 2, 3, 4), *, hot=None, deroted=None) -> list:
    """Data-rate reduction (+ mover concentration) for each reference depth in *n_list*."""
    if hot is None:
        hot = bg.hot_pixel_mask(ev, cfg.hot_pixel_pct)
    waz, elev, rev, _ = deroted if deroted is not None else _derotate(ev, tel, cfg)
    n = int(ev["n"])
    out = []
    for N in n_list:
        keep = keep_mask(ev, tel, cfg, N, hot=hot)
        nk = int(keep.sum())
        # de-rotated survivor density on a coarse (az,elev) grid → concentration of movers
        if nk:
            H, _, _ = np.histogram2d(waz[keep], elev[keep], bins=[180, 60],
                                     range=[[0, 360], [-30, 30]])
            occ = H[H > 0]
            cells = int((H >= max(0.02 * nk / max(occ.size, 1), 5)).sum())
            conc = float(H.max() / occ.mean()) if occ.size else 0.0
        else:
            cells, conc = 0, 0.0
        out.append(MaskMetric(int(N), round(100 * (1 - nk / n), 2), nk, cells, round(conc, 1)))
    return out


def extract_movers(ev, tel, cfg, keep, *, bin_deg=2.0, min_events=150, deroted=None) -> list:
    """De-rotate the survivors and emit one :class:`MovingObject` per (revolution, azimuth cluster).

    Each populated ``(rev, azimuth-bin)`` cell with enough surviving events is a moving object at that
    bearing; its elevation is the event-cloud centroid and its range is the pinhole estimate from the
    vertical (elevation) extent of its compact core. This is generic — it returns *every* mover (the
    drone and any wind-blown clutter); the drone is the persistent above-horizon one.
    """
    waz, elev, rev, t = deroted if deroted is not None else _derotate(ev, tel, cfg)
    keep = np.asarray(keep, bool)
    wz = waz[keep]; el = elev[keep]; rv = rev[keep]; tt = t[keep]
    yk = np.asarray(ev["y"], float)[keep]
    if wz.size == 0:
        return []
    nb = int(np.ceil(360.0 / bin_deg))
    cell = rv.astype(np.int64) * 100000 + np.clip((wz / bin_deg).astype(int), 0, nb - 1)
    order = np.argsort(cell, kind="stable")
    cid = cell[order]
    bounds = np.flatnonzero(np.diff(cid)) + 1
    starts = np.concatenate(([0], bounds)); stops = np.concatenate((bounds, [cid.size]))
    movers = []
    for s, e in zip(starts, stops):
        if e - s < min_events:
            continue
        idx = order[s:e]
        az_c = float(np.median(wz[idx])); el_c = float(np.median(el[idx]))
        # compact core in elevation for the size/range (reject a few stragglers)
        yy = yk[idx]
        size = float(np.percentile(yy, 90) - np.percentile(yy, 10))
        rng = (float(estimate_range_m(max(size, 1.0), cfg.fov_deg, cfg.target_size_m, cfg.sensor_w))
               if cfg.target_size_m and cfg.target_size_m > 0 else None)
        movers.append(MovingObject(
            rev=int(rv[idx[0]]), t_s=float(np.median(tt[idx])), bearing_deg=round(az_c, 2),
            elev_deg=round(el_c, 2), range_m=(round(rng, 1) if rng else None),
            n_events=int(e - s), size_px=round(size, 1), above_horizon=bool(el_c > 0)))
    movers.sort(key=lambda m: (m.rev, m.bearing_deg))
    return movers


def run_masking(rec, cfg, *, n_list=(0, 1, 2, 3, 4), bin_deg=2.0,
                min_events=150) -> MaskingResult:
    """Run the full masking analysis on a Recording: the N-sweep + the moving-object volumetric map
    at ``cfg.mask_rotations``. Telemetry is required (real, or synthesized via ``--assume_spin``)."""
    from gottlux.rotation import ev_dict, resolve_cfg
    cfg = resolve_cfg(rec, cfg)
    tel = rec.telemetry
    if tel is None:
        raise ValueError("masking needs rotation telemetry (real, or --assume_spin)")
    ev = ev_dict(rec.all())
    hot = bg.hot_pixel_mask(ev, cfg.hot_pixel_pct)
    deroted = _derotate(ev, tel, cfg)
    metrics = sweep(ev, tel, cfg, n_list, hot=hot, deroted=deroted)
    N = int(getattr(cfg, "mask_rotations", 2) or 2)
    keep = keep_mask(ev, tel, cfg, N, hot=hot)
    movers = extract_movers(ev, tel, cfg, keep, bin_deg=bin_deg, min_events=min_events,
                            deroted=deroted)
    red = next((m.reduction_pct for m in metrics if m.n_rotations == N), float("nan"))
    return MaskingResult(sweep=metrics, movers=movers, n_rotations=N, reduction_pct=red,
                         fov_deg=cfg.fov_deg, t_rot_s=round(float(getattr(tel, "T_rot", 0.0)), 5),
                         target_size_m=cfg.target_size_m)


def _ang_diff(a, b):
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def link_mover_tracks(movers, *, gate_deg=8.0, init_gate_deg=30.0, min_revs=3):
    """Link per-revolution movers into **coherent tracks** (greedy nearest-neighbour with bearing
    continuity + motion prediction), so a real object is followed across revolutions instead of the
    per-revolution "densest mover" jumping between the drone and clutter. The first association of a
    track uses a generous gate (motion unknown); later ones a tight predicted gate. Returns tracks
    (lists of :class:`MovingObject`, by revolution) with ``≥ min_revs`` points, longest first — the
    drone is typically the most persistent coherent track. No elevation assumption."""
    by_rev: dict = {}
    for m in movers:
        by_rev.setdefault(int(m.rev), []).append(m)
    open_tr: list = []
    for rev in sorted(by_rev):
        used = set()
        for d in sorted(by_rev[rev], key=lambda m: -m.n_events):
            best_i, best_cost = None, np.inf
            for i, tr in enumerate(open_tr):
                if i in used:
                    continue
                pred = tr["last_b"] + tr["off"] * (rev - tr["last_rev"])
                g = gate_deg if len(tr["d"]) >= 2 else init_gate_deg
                cost = abs(_ang_diff(d.bearing_deg, pred))
                if cost <= g and cost < best_cost:
                    best_cost, best_i = cost, i
            if best_i is None:
                open_tr.append({"d": [d], "last_b": d.bearing_deg, "last_rev": rev, "off": 0.0})
                used.add(len(open_tr) - 1)
            else:
                tr = open_tr[best_i]; tr["d"].append(d)
                rr = np.array([x.rev for x in tr["d"]], float)
                bb = np.rad2deg(np.unwrap(np.deg2rad([x.bearing_deg for x in tr["d"]])))
                tr["off"] = float(np.polyfit(rr, bb, 1)[0]) if rr.size >= 2 else 0.0
                tr["last_b"], tr["last_rev"] = d.bearing_deg, rev
                used.add(best_i)
    tracks = [sorted(tr["d"], key=lambda m: m.rev) for tr in open_tr
              if len(set(x.rev for x in tr["d"])) >= min_revs]
    tracks.sort(key=lambda tk: -len(tk))
    return tracks


def densest_track(result):
    """The drone track by the **competitive-maximum** rule (the original MATLAB recipe): after masking,
    the drone is the *densest mover* — per revolution, the surviving moving cluster with the **most
    events** — with **no elevation/above-horizon assumption** (high sensor tilt or a low-flying drone
    can put it below the horizon). Returns the per-revolution :class:`MovingObject` list (the track)."""
    by_rev: dict = {}
    for m in result.movers:
        by_rev.setdefault(int(m.rev), []).append(m)
    track = []
    for rev in sorted(by_rev):
        track.append(max(by_rev[rev], key=lambda m: m.n_events))   # greatest event count = the target
    return track


def _summarize_cluster(cluster):
    revs = sorted(set(x.rev for x in cluster))
    bper = {r: float(np.median([x.bearing_deg for x in cluster if x.rev == r])) for r in revs}
    drift = (max(bper.values()) - min(bper.values())) if len(revs) >= 2 else 0.0
    rng = [x.range_m for x in cluster if x.range_m is not None and np.isfinite(x.range_m)]
    return {
        "bearing_deg": round(float(np.median([x.bearing_deg for x in cluster])), 2),
        "elev_deg": round(float(np.median([x.elev_deg for x in cluster])), 2),
        "range_m": (round(float(np.median(rng)), 1) if rng else None),
        "range_iqr_m": ([round(float(np.percentile(rng, 25)), 1),
                         round(float(np.percentile(rng, 75)), 1)] if rng else None),
        "n_revs_present": len(revs), "n_detections": len(cluster), "revs": revs,
        "bearing_drift_deg": round(float(drift), 1),     # bearing change across revs = translation
        "translating": bool(drift > 5.0),                # a flying drone marches; rooted clutter doesn't
    }


def primary_mover(result, *, az_window=None, gate_deg=6.0, min_revs=2):
    """Pick the drone candidate: the persistent above-horizon mover.

    With *az_window* ``(lo, hi)`` (the known target arc) the whole arc's above-horizon movers are
    taken as the target and its per-revolution bearing **track + drift** reported — robust for a
    *translating* drone (whose bearing marches across revolutions and would otherwise be split by
    fixed-bearing clustering). Without a window, falls back to the most-persistent fixed-bearing
    cluster — a weaker heuristic that can latch onto **rooted vegetation** (persistent + above-horizon
    too); use :func:`persistent_candidates` to see all options, or confirm by eye / a staring dwell.
    """
    ah = [m for m in result.movers if m.above_horizon]
    if az_window:
        lo, hi = az_window
        ah = [m for m in ah if lo <= m.bearing_deg <= hi]
    if not ah:
        return None
    if az_window:
        cluster = ah                                     # the known arc is the target
    else:
        ah.sort(key=lambda m: m.bearing_deg)
        clusters, cur = [], [ah[0]]
        for m in ah[1:]:
            if abs(_ang_diff(m.bearing_deg, cur[-1].bearing_deg)) <= gate_deg:
                cur.append(m)
            else:
                clusters.append(cur); cur = [m]
        clusters.append(cur)
        cluster = max(clusters, key=lambda c: len(set(x.rev for x in c)))
    if len(set(x.rev for x in cluster)) < min_revs:
        return None
    return _summarize_cluster(cluster)


def persistent_candidates(result, *, gate_deg=6.0, min_revs=2, top_k=4) -> list:
    """All persistent above-horizon mover clusters, ranked by revolutions present (drone candidates).
    Surfacing several is the honest move — the most-persistent one can be rooted clutter, so the
    driver shows the set and lets cross-file consistency / the ``translating`` flag / a staring dwell
    disambiguate."""
    ah = sorted((m for m in result.movers if m.above_horizon), key=lambda m: m.bearing_deg)
    if not ah:
        return []
    clusters, cur = [], [ah[0]]
    for m in ah[1:]:
        if abs(_ang_diff(m.bearing_deg, cur[-1].bearing_deg)) <= gate_deg:
            cur.append(m)
        else:
            clusters.append(cur); cur = [m]
    clusters.append(cur)
    cand = [_summarize_cluster(c) for c in clusters if len(set(x.rev for x in c)) >= min_revs]
    cand.sort(key=lambda d: -d["n_revs_present"])
    return cand[:top_k]


def run_multifile(items, cfg, *, n_list=(0, 1, 2, 3, 4), assume_spin_hz=None,
                  bearing_gate_deg=10.0, az_window=None) -> dict:
    """Run masking across several rotating clips and cross-check the drone.

    *items* is a list of Recordings or path strings (paths are loaded with ``cfg.camera``). Each clip
    is masked, its persistent above-horizon mover (:func:`primary_mover`) extracted, and the results
    compared: a drone is "consistent" if **every** clip yields a persistent above-horizon mover — the
    basis for cueing. If the rig didn't move between captures, those bearings should also agree
    (within *bearing_gate_deg*). Telemetry-less clips get synthesized telemetry from *assume_spin_hz*
    (``0`` = estimate from the event-rate autocorrelation).

    Returns ``{"per_file": [...], "consistent": bool, "bearings_agree": bool|None, ...}``.
    """
    import gottlux as eb
    from gottlux.config import Config
    from gottlux.io.telemetry import Telemetry, estimate_spin_period_s

    per_file = []
    for it in items:
        rec = it if hasattr(it, "telemetry") else eb.load(it, camera=cfg.camera, mode="auto",
                                                          progress=lambda f: None)
        if rec.telemetry is None and assume_spin_hz is not None:
            per = (1.0 / assume_spin_hz) if assume_spin_hz > 0 else \
                estimate_spin_period_s(rec.t.astype(float) / 1e6)[0]
            if per:
                rec.attach_telemetry(Telemetry.from_spin(rec.duration_s, per), refine=False)
        c = Config.from_dict(cfg.to_dict())          # per-file copy (resolve_cfg mutates geometry)
        res = run_masking(rec, c, n_list=n_list)
        per_file.append({"name": rec.name, "result": res,
                         "primary": primary_mover(res, az_window=az_window),
                         "candidates": persistent_candidates(res)})

    prims = [f["primary"] for f in per_file]
    consistent = bool(per_file) and all(p is not None for p in prims)
    bearings = [p["bearing_deg"] for p in prims if p]
    bearings_agree = (max(abs(_ang_diff(b, bearings[0])) for b in bearings) <= bearing_gate_deg
                      if len(bearings) >= 2 else None)
    return {"per_file": per_file, "consistent": consistent, "bearings_agree": bearings_agree,
            "bearings_deg": [round(b, 1) for b in bearings],
            "n_files": len(per_file)}


def movers_table(result) -> dict:
    """Flatten the moving objects into a dict-of-columns for CSV/Parquet."""
    m = result.movers
    return {
        "rev": [o.rev for o in m], "t_s": [round(o.t_s, 4) for o in m],
        "bearing_deg": [o.bearing_deg for o in m], "elev_deg": [o.elev_deg for o in m],
        "range_m": [o.range_m for o in m], "n_events": [o.n_events for o in m],
        "size_px": [o.size_px for o in m], "above_horizon": [int(o.above_horizon) for o in m],
    }
