"""
mask_sweep.py  --  Successive-mask comparison & data drop-off quantification.

Builds the FROZEN reference from the first N revolutions for N = 0,1,..,mask_levels
(N=0 = raw, no rotational mask) and, for each level:
  * counts the surviving events (the "data drop-off" of the masking technique),
  * renders the de-rotated 360 panorama (shared color scale, so the panels
    visibly dim as more is masked),
  * records the surviving event-rate vs time.

Outputs (all PNG/CSV, Python only):
  <tag>_panorama_maskN.png        one panorama per level (unique)
  <tag>_panorama_masksweep.png    montage of all levels (before vs after masks)
  <tag>_masksweep_rate.png        event-rate vs time, raw + each successive mask
  <tag>_masksweep_dropoff.png     surviving events & % vs number of masks
  <tag>_masksweep_counts.csv      level, events, percent, panorama_events
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gottlux.rotation import background
from gottlux.rotation.viz.panorama import _world_coords


def render_mask_sweep(ev, tel, cfg, hot, out_dir, tag, az_bins=900, y_bins=None,
                      max_pano_pts=3_000_000):
    W, H = cfg.sensor_w, cfg.sensor_h
    if y_bins is None: y_bins = min(int(H), 1080)        # match the sensor height (any size)
    t = np.asarray(ev["t"]) / 1e6
    x = np.asarray(ev["x"]); y = np.asarray(ev["y"])
    raw_n = len(t)
    n_phase = cfg.n_phase
    ph = (tel.phase_at(t) * n_phase).astype(np.int64) % n_phase
    hotmask = hot[y, x]
    az, ypx = _world_coords(ev, cfg, tel)      # vertical axis = raw sensor row (px)
    rng = np.random.default_rng(0)

    bin_s = 0.05
    edges = np.arange(0, t.max() + bin_s, bin_s); ctr = 0.5 * (edges[:-1] + edges[1:])
    raw_rate = np.histogram(t, bins=edges)[0] / bin_s

    levels = list(range(0, max(1, cfg.mask_levels) + 1))   # 0 = raw
    counts, pcts, pano_counts, rates, panos, titles = [], [], [], [], [], []
    for N in levels:
        if N == 0:
            keep = np.ones(raw_n, bool); label = "raw (no mask)"
        else:
            ref_end = background.reference_end_time(tel, N)
            ref = background.build_reference(ev, tel, ref_end, n_phase=n_phase)
            keep = ~(ref[ph, y, x] | hotmask)
            label = f"{N}-rotation mask"
        c = int(keep.sum())
        counts.append(c); pcts.append(100.0 * c / raw_n)
        rates.append(np.histogram(t[keep], bins=edges)[0] / bin_s)
        idx = np.where(keep)[0]
        pano_counts.append(len(idx))
        if len(idx) > max_pano_pts:
            idx = rng.choice(idx, max_pano_pts, replace=False)
        Hh, _, _ = np.histogram2d(ypx[idx], az[idx], bins=[y_bins, az_bins],
                                  range=[[0, H], [0, 360]])
        panos.append(np.log1p(Hh)); titles.append(f"{label}: {c:,} ev ({100.0*c/raw_n:.1f}%)")

    vmax = np.percentile(panos[0][panos[0] > 0], 99) if (panos[0] > 0).any() else 1.0
    arts = []

    # --- unique per-level panoramas ---
    for N, pano, ttl in zip(levels, panos, titles):
        fig, ax = plt.subplots(figsize=(14, 4.4))
        im = ax.imshow(pano, origin="lower", aspect="auto", cmap="inferno",
                       extent=[0, 360, 0, H], vmax=vmax)
        fig.colorbar(im, ax=ax, pad=0.01, label="log(1+events)")
        ax.set_xlabel("world azimuth [deg]"); ax.set_ylabel("sensor Y [px]")
        ax.set_title(f"{tag} panorama -- {ttl}"); ax.set_xticks(np.arange(0, 361, 45))
        ax.invert_yaxis()
        p = os.path.join(out_dir, f"{tag}_panorama_mask{N}.png")
        fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
        arts.append((p, f"panorama after {N}-rotation mask ({pano_counts[levels.index(N)]:,} events)"))

    # --- montage of all levels (shared color scale shows the drop-off) ---
    nL = len(levels)
    fig, axes = plt.subplots(nL, 1, figsize=(14, 2.4 * nL))
    axes = np.atleast_1d(axes)
    for ax, pano, ttl in zip(axes, panos, titles):
        im = ax.imshow(pano, origin="lower", aspect="auto", cmap="inferno",
                       extent=[0, 360, 0, H], vmax=vmax)
        ax.set_ylabel("sensor Y [px]"); ax.set_title(ttl, fontsize=9)
        ax.set_xticks(np.arange(0, 361, 60)); ax.invert_yaxis()
    axes[-1].set_xlabel("world azimuth [deg]")
    fig.suptitle(f"{tag}: 360 panorama vs successive masking (shared scale)", y=1.0)
    p = os.path.join(out_dir, f"{tag}_panorama_masksweep.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    arts.append((p, "ALL panoramas vs successive masks (montage)"))

    # --- event-rate vs time: raw + each successive mask ---
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(ctr, raw_rate, "k", lw=1.0, label=f"raw (N=0): {counts[0]:,}")
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, nL - 1))
    for i, N in enumerate(levels[1:]):
        ax.plot(ctr, rates[i + 1], color=cmap[i], lw=0.9,
                label=f"N={N}: {counts[i+1]:,} ({pcts[i+1]:.1f}%)")
    for hl in tel.hall_t + tel.offset:
        ax.axvline(hl, color="r", ls="--", lw=0.3, alpha=0.3)
    ax.set_xlabel("time [s]"); ax.set_ylabel("events / s")
    ax.set_title(f"{tag}: event rate drop-off with successive masks")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    p = os.path.join(out_dir, f"{tag}_masksweep_rate.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    arts.append((p, "event-rate vs time across successive masks"))

    # --- drop-off vs number of masks ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(levels, counts, color="steelblue", alpha=0.8)
    ax1.set_xlabel("number of masking rotations (N)"); ax1.set_ylabel("surviving events", color="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(levels, pcts, "ro-", lw=1.5); ax2.set_ylabel("% of raw retained", color="r")
    for N, c, pc in zip(levels, counts, pcts):
        ax1.text(N, c, f"{c:,}\n{pc:.1f}%", ha="center", va="bottom", fontsize=8)
    ax1.set_title(f"{tag}: data retained vs successive masking")
    p = os.path.join(out_dir, f"{tag}_masksweep_dropoff.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    arts.append((p, "data drop-off vs number of masks"))

    # --- CSV ---
    csvp = os.path.join(out_dir, f"{tag}_masksweep_counts.csv")
    np.savetxt(csvp, np.column_stack([levels, counts, pcts, pano_counts]),
               delimiter=",", header="n_masks,surviving_events,percent_retained,panorama_events",
               comments="", fmt=["%d", "%d", "%.3f", "%d"])
    arts.append((csvp, "per-level event counts / percentages"))

    summary = dict(masksweep_levels=levels, masksweep_event_counts=counts,
                   masksweep_event_pct=[round(p, 2) for p in pcts],
                   masksweep_panorama_events=pano_counts)
    return summary, arts
