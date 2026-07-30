"""
masking_viz.py — figures for the rotational background-masking analysis
(:mod:`gottlux.rotation.masking`).

One comprehensive panel: (A) data-rate reduction & mover concentration vs reference depth N,
(B) the de-rotated moving-target map (azimuth × elevation density of survivors, with detected
movers marked), (C) a tactical radar of the moving objects (bearing × range, drone = above-horizon
highlighted), and (D) the single-EBS volumetric (azimuth, elevation, range) point cloud.

Pure matplotlib (Agg-safe). Drone vs other movers is shown by the above-horizon flag — honest about
the fact that wind-blown clutter also survives (see ``docs/ROTATIONAL_EBS_METHODS.md`` §4).
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)


def masking_figure(result, waz_keep, elev_keep, title="Rotational masking → volumetric moving-object map"):
    """Four-panel masking report. *waz_keep, elev_keep* are the de-rotated survivor coordinates
    (world azimuth deg, elevation deg) at the chosen reference depth."""
    sweep = result.sweep
    movers = result.movers
    ah = [m for m in movers if m.above_horizon]
    other = [m for m in movers if not m.above_horizon]

    fig = plt.figure(figsize=(15, 9), facecolor="w")

    # A — reduction & concentration vs N
    axA = fig.add_subplot(2, 2, 1)
    N = [m.n_rotations for m in sweep]
    axA.plot(N, [m.reduction_pct for m in sweep], "o-", color="tab:blue", label="data-rate reduction %")
    axA.axhline(90, ls=":", color="gray"); axA.set_ylabel("reduction %", color="tab:blue")
    axA.set_xlabel("reference rotations N"); axA.set_xticks(N)
    axB = axA.twinx()
    axB.plot(N, [m.concentration for m in sweep], "s--", color="tab:red", label="mover concentration")
    axB.set_ylabel("survivor concentration (peak/mean)", color="tab:red")
    axA.set_title("A. Data-rate reduction & mover amplification vs N")
    axA.text(0.02, 0.04, f"used N={result.n_rotations} → {result.reduction_pct:.1f}% reduction",
             transform=axA.transAxes, fontsize=9,
             bbox=dict(boxstyle="round", fc="#e3f2fd", ec="tab:blue"))

    # B — de-rotated moving-target map (azimuth × elevation density)
    axM = fig.add_subplot(2, 2, 2)
    H, xe, ye = np.histogram2d(np.asarray(waz_keep), np.asarray(elev_keep),
                               bins=[180, 80], range=[[0, 360], [-25, 25]])
    axM.imshow(np.log1p(H.T), origin="lower", aspect="auto", cmap="inferno",
               extent=[0, 360, -25, 25])
    axM.axhline(0, color="w", lw=0.4, alpha=0.4)
    if ah:
        axM.scatter([m.bearing_deg for m in ah], [m.elev_deg for m in ah], s=40,
                    facecolors="none", edgecolors="cyan", lw=1.3, label="mover ↑horizon")
    if other:
        axM.scatter([m.bearing_deg for m in other], [m.elev_deg for m in other], s=20,
                    facecolors="none", edgecolors="#888", lw=0.8, label="mover ↓horizon")
    axM.set_xlabel("world azimuth [deg]"); axM.set_ylabel("elevation [deg]")
    axM.set_title("B. Moving-object map (static world removed)")
    axM.legend(fontsize=8, loc="upper right")

    # C — tactical radar of the movers (bearing × range)
    axR = fig.add_subplot(2, 2, 3, projection="polar")
    axR.set_theta_zero_location("N"); axR.set_theta_direction(-1)
    axR.set_facecolor((0.05, 0.05, 0.05)); axR.grid(color=(0.3, 0.8, 0.3), alpha=0.5)
    rngs = [m.range_m for m in movers if m.range_m is not None and np.isfinite(m.range_m)]
    rmax = float(np.nanpercentile(rngs, 95) * 1.2) if rngs else 1.0
    axR.set_ylim(0, max(rmax, 1.0))
    for grp, c, lab in ((ah, "tab:cyan", "↑horizon (drone)"), (other, "#888", "↓horizon")):
        b = [np.deg2rad(m.bearing_deg) for m in grp if m.range_m]
        r = [m.range_m for m in grp if m.range_m]
        if b:
            axR.scatter(b, r, c=c, s=55, edgecolors="w", linewidths=0.4, label=lab)
    axR.set_title("C. Moving-object radar (bearing × range)", color="k", pad=14)
    axR.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.1, -0.1))

    # D — volumetric (azimuth, elevation, range) point cloud
    axV = fig.add_subplot(2, 2, 4, projection="3d")
    if ah:
        bb = [m.bearing_deg for m in ah]; ee = [m.elev_deg for m in ah]
        rr = [(m.range_m if m.range_m else np.nan) for m in ah]
        p = axV.scatter(bb, rr, ee, c=[m.t_s for m in ah], cmap="viridis", s=40)
        fig.colorbar(p, ax=axV, pad=0.1, shrink=0.6, label="t [s]")
    axV.set_xlabel("azimuth [deg]"); axV.set_ylabel("range [m]"); axV.set_zlabel("elevation [deg]")
    axV.set_title("D. Volumetric map (az, elev, range) — above-horizon movers")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def radar_track_figure(result, track, title="Drone track (densest mover) — radar + moving-object map"):
    """Verification figure: the per-revolution **densest-mover track** overlaid on (left) a tactical
    polar radar (θ = bearing, r = range, colour = revolution time; track connected) and (right) the
    azimuth×elevation moving-object map (all movers faint, the track in colour). No elevation gate —
    the track is the greatest-event mover each revolution, wherever it is. *track* is the output of
    :func:`gottlux.rotation.masking.densest_track`."""
    movers = result.movers
    tb = np.array([m.bearing_deg for m in track], float)
    tr = np.array([(m.range_m if m.range_m else np.nan) for m in track], float)
    te = np.array([m.elev_deg for m in track], float)
    tt = np.array([m.t_s for m in track], float)
    fig = plt.figure(figsize=(14, 6.2), facecolor="w")

    # left — tactical radar
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_facecolor((0.05, 0.05, 0.05)); ax.grid(color=(0.3, 0.8, 0.3), alpha=0.5)
    ax.tick_params(colors="0.3")
    have_r = np.isfinite(tr).any()
    rr = tr.copy()
    if not have_r:
        rr = np.arange(1, len(tb) + 1, dtype=float)        # fallback: revolution order
    ax.set_ylim(0, max(float(np.nanpercentile(rr, 95) * 1.2) if np.isfinite(rr).any() else 1.0, 1.0))
    if len(tb):
        ax.plot(np.deg2rad(tb), rr, "-", color=(0, 0.6, 0), lw=1.0, alpha=0.7)
        sc = ax.scatter(np.deg2rad(tb), rr, c=tt, cmap="viridis", s=90, edgecolors="w", linewidths=0.5)
        fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.75, label="time [s]")
    ax.set_title("θ = bearing · r = range" + ("" if have_r else " (rev order — range N/A)"),
                 fontsize=10, pad=14)

    # right — moving-object map with the track
    axm = fig.add_subplot(1, 2, 2)
    if movers:
        axm.scatter([m.bearing_deg for m in movers], [m.elev_deg for m in movers], s=8,
                    c="#bbb", label="all movers")
    if len(tb):
        axm.plot(tb, te, "-", color="tab:red", lw=1.0, alpha=0.6)
        s2 = axm.scatter(tb, te, c=tt, cmap="viridis", s=70, edgecolors="k", linewidths=0.4,
                         zorder=3, label="drone track (densest mover/rev)")
    axm.axhline(0, color="gray", lw=0.5)
    axm.set_xlim(0, 360); axm.set_xlabel("world bearing [deg]"); axm.set_ylabel("elevation [deg]")
    axm.set_title("moving-object map (track = greatest-event mover per rev)")
    axm.grid(alpha=0.3); axm.legend(fontsize=8, loc="upper right")

    brg = (f"bearing {np.nanmin(tb):.1f}–{np.nanmax(tb):.1f}°" if len(tb) else "no track")
    fig.suptitle(f"{title}   ·   {len(track)} revolutions · {brg}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def tracks_radar_figure(tracks, movers, title="Linked moving-object tracks — verify the drone"):
    """All coherent linked tracks on a radar + the azimuth×elevation map. The most persistent track
    (drone candidate) is bold/labelled; others are thin. No elevation gate — you verify which track
    is the drone. *tracks* is :func:`gottlux.rotation.masking.link_mover_tracks`."""
    cmap = plt.cm.tab10
    fig = plt.figure(figsize=(14, 6.4), facecolor="w")
    # left — radar
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_facecolor((0.05, 0.05, 0.05)); ax.grid(color=(0.3, 0.8, 0.3), alpha=0.5); ax.tick_params(colors="0.4")
    allr = [m.range_m for t in tracks for m in t if m.range_m and np.isfinite(m.range_m)]
    have_r = len(allr) > 0
    rmax = max(float(np.nanpercentile(allr, 95) * 1.2) if have_r else 1.0, 1.0)
    ax.set_ylim(0, rmax)
    for k, tk in enumerate(tracks):
        b = np.deg2rad([m.bearing_deg for m in tk])
        r = np.array([(m.range_m if m.range_m else rmax * 0.5) for m in tk], float)
        lw, ms = (2.4, 80) if k == 0 else (1.0, 30)
        ax.plot(b, r, "-", color=cmap(k % 10), lw=lw, alpha=0.85 if k == 0 else 0.5)
        ax.scatter(b, r, color=cmap(k % 10), s=ms, edgecolors="w", linewidths=0.4, zorder=3)
    ax.set_title("θ = bearing · r = range" + ("" if have_r else " (rev order)"), fontsize=10, pad=14)
    # right — az/elev map with tracks
    axm = fig.add_subplot(1, 2, 2)
    if movers:
        axm.scatter([m.bearing_deg for m in movers], [m.elev_deg for m in movers], s=6,
                    c="#ccc", zorder=0)
    for k, tk in enumerate(tracks):
        bb = [m.bearing_deg for m in tk]; ee = [m.elev_deg for m in tk]
        lab = (f"DRONE? {np.median(bb):.0f}° · {len(tk)} revs" if k == 0 else None)
        axm.plot(bb, ee, "o-", color=cmap(k % 10), lw=(2.0 if k == 0 else 0.8),
                 ms=(6 if k == 0 else 3), label=lab, zorder=3 if k == 0 else 1)
    axm.axhline(0, color="gray", lw=0.5); axm.set_xlim(0, 360)
    axm.set_xlabel("world bearing [deg]"); axm.set_ylabel("elevation [deg]")
    axm.set_title("moving-object map + linked tracks (no elevation gate)")
    axm.grid(alpha=0.3); axm.legend(fontsize=9, loc="upper right")
    dom = (f"dominant track: bearing≈{np.median([m.bearing_deg for m in tracks[0]]):.1f}°, "
           f"{len(tracks[0])} revolutions" if tracks else "no coherent track")
    fig.suptitle(f"{title}   ·   {len(tracks)} coherent tracks · {dom}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def multifile_figure(multi, title="Cross-file masking — is the drone the consistent mover?"):
    """Compare masking across clips (output of :func:`gottlux.rotation.masking.run_multifile`):
    one column per clip — top: bearing×range radar of the movers (drone candidate ringed); bottom:
    azimuth×elevation moving-object map. A verdict banner states whether every clip yields a
    persistent above-horizon mover (and whether the bearings agree)."""
    files = multi["per_file"]
    n = max(len(files), 1)
    fig = plt.figure(figsize=(5.2 * n, 8.4), facecolor="w")
    for i, f in enumerate(files):
        movers = f["result"].movers
        prim = f["primary"]
        ah = [m for m in movers if m.above_horizon]
        oth = [m for m in movers if not m.above_horizon]
        # top — radar
        axr = fig.add_subplot(2, n, i + 1, projection="polar")
        axr.set_theta_zero_location("N"); axr.set_theta_direction(-1)
        axr.set_facecolor((0.05, 0.05, 0.05)); axr.grid(color=(0.3, 0.8, 0.3), alpha=0.5)
        rngs = [m.range_m for m in movers if m.range_m]
        axr.set_ylim(0, max(float(np.nanpercentile(rngs, 95) * 1.2) if rngs else 1.0, 1.0))
        for grp, c in ((oth, "#888"), (ah, "tab:cyan")):
            b = [np.deg2rad(m.bearing_deg) for m in grp if m.range_m]
            r = [m.range_m for m in grp if m.range_m]
            if b:
                axr.scatter(b, r, c=c, s=40, edgecolors="w", linewidths=0.3)
        if prim and prim["range_m"]:
            axr.scatter([np.deg2rad(prim["bearing_deg"])], [prim["range_m"]], s=240,
                        facecolors="none", edgecolors="yellow", linewidths=2.0)
        cand = (f"drone≈{prim['bearing_deg']:.0f}° r≈{prim['range_m']}m · {prim['n_revs_present']} revs"
                if prim else "no persistent mover")
        axr.set_title(f"{f['name'][:22]}\n{cand}", color="k", fontsize=9, pad=12)
        # bottom — az/elev moving-object map
        axm = fig.add_subplot(2, n, n + i + 1)
        if oth:
            axm.scatter([m.bearing_deg for m in oth], [m.elev_deg for m in oth], s=10,
                        c="#999", label="↓horizon")
        if ah:
            axm.scatter([m.bearing_deg for m in ah], [m.elev_deg for m in ah], s=22,
                        c="tab:cyan", edgecolors="k", linewidths=0.3, label="↑horizon")
        if prim:
            axm.scatter([prim["bearing_deg"]], [prim["elev_deg"]], s=200, facecolors="none",
                        edgecolors="orange", linewidths=2.0)
        axm.axhline(0, color="gray", lw=0.5); axm.set_xlim(0, 360); axm.set_xlabel("bearing [deg]")
        axm.set_ylabel("elev [deg]"); axm.grid(alpha=0.3); axm.legend(fontsize=7, loc="upper right")
    verdict = ("✓ drone is the consistent above-horizon mover in all clips"
               if multi["consistent"] else "✗ not a persistent above-horizon mover in every clip")
    if multi.get("bearings_agree") is not None:
        verdict += ("  ·  bearings agree across clips" if multi["bearings_agree"]
                    else "  ·  bearings differ (target or rig moved between captures)")
    fig.suptitle(f"{title}\n{verdict}   ({multi['n_files']} clips · bearings {multi['bearings_deg']})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
