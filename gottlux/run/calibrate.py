"""
calibrate.py — the ``gottlux-calibrate`` console script (ported from EBS ``ebs-calibrate``).

Two jobs, both for turning the system's *relative* measurements into something absolute:

* **rel → metres** (default): the tracking report emits a unitless ``rel_distance`` per track
  point (monotonic with true range). Given the known closest/farthest distance for *this*
  flight, map the proxy onto metres (linear, since proxy ∝ range) and plot calibrated range.

      gottlux-calibrate <tag>_tracks.csv --near 30 --far 300

* **inter-camera offset** (``--intercam FOLDER``): measure the constant time offset between the
  dual-EBS cameras from their correlated event-rate timelines (and, target permitting, separate
  timing skew from boresight misalignment).

      gottlux-calibrate --intercam path/to/capture_folder
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def calibrate(csv, near, far, track=None, out=None, lo_pct=1.0, hi_pct=99.0):
    """Map ``rel_distance`` → metres using the file's proxy extremes and known near/far.
    Returns ``(plot_path, calibrated_csv_path)``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.genfromtxt(csv, delimiter=",", names=True)
    tid = np.atleast_1d(d["track_id"]).astype(int)
    t = np.atleast_1d(d["t_s"]).astype(float)
    rel = np.atleast_1d(d["rel_distance"]).astype(float)
    good = np.isfinite(rel)
    if good.sum() < 2:
        raise SystemExit("not enough finite rel_distance values to calibrate")
    rmin = float(np.percentile(rel[good], lo_pct))
    rmax = float(np.percentile(rel[good], hi_pct))
    if rmax - rmin < 1e-9:
        raise SystemExit("relative distance is ~constant; cannot calibrate (size never changed)")

    def to_m(r):
        return near + (np.clip(r, rmin, rmax) - rmin) / (rmax - rmin) * (far - near)

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="w")
    ids = [int(track)] if track is not None else sorted(set(tid[good].tolist()))
    for i in ids:
        m = good & (tid == i)
        if m.any():
            ax.plot(t[m], to_m(rel[m]), "-o", ms=2.5, lw=1.3, label=f"track #{i}")
    ax.axhline(near, color="g", ls=":", lw=1.2, label=f"near {near:g} m")
    ax.axhline(far, color="r", ls=":", lw=1.2, label=f"far {far:g} m")
    ax.set_xlabel("time [s]"); ax.set_ylabel("calibrated distance [m]")
    ax.set_title(f"Calibrated range — {os.path.basename(csv)}  (near {near:g} / far {far:g} m)")
    ax.grid(True, ls="--", alpha=0.4); ax.legend(fontsize=8)
    out = out or (os.path.splitext(csv)[0] + "_calibrated_range.png")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

    cout = os.path.splitext(csv)[0] + "_calibrated_range.csv"
    arr = np.column_stack([tid[good], t[good], to_m(rel[good])])
    np.savetxt(cout, arr, delimiter=",", header="track_id,t_s,distance_m", comments="", fmt="%.4f")
    return out, cout


def _intercam(folder):
    """Measure the inter-camera time offset from a capture folder (cam0 + cam1)."""
    import glob

    import gottlux as eb
    from gottlux.rotation import calibrate as cal
    raws = {c: eb.load(folder, camera=c) for c in ("cam0", "cam1")}
    t0 = np.asarray(raws["cam0"].t) / 1e6
    t1 = np.asarray(raws["cam1"].t) / 1e6
    res = cal.measure_intercam_offset(t0, t1)
    print(f"[calibrate] inter-camera offset = {res['offset_s']*1e3:+.2f} ms "
          f"(add to cam1 to align with cam0); peak corr = {res['peak_corr']:.3f}")
    # target-based separation (needs a telemetry CSV in the folder)
    if glob.glob(os.path.join(folder, "*.csv")):
        try:
            pr = cal.measure_offset_from_passes(folder)
            fb = cal.fit_timing_boresight(pr)
            print(f"[calibrate] matched passes = {pr['n']}, "
                  f"bearing offset = {pr['dbearing_deg']:+.2f} deg (cam0 − cam1)")
            if fb.get("separable"):
                print(f"[calibrate] timing skew = {fb['dt_ms']:+.2f} ± {fb['dt_se_ms']:.2f} ms; "
                      f"boresight = {fb['dphi_deg']:+.2f} ± {fb['dphi_se_deg']:.2f} deg")
            else:
                print(f"[calibrate] timing/boresight not separable here "
                      f"({fb.get('reason', 'omega too constant')})")
        except Exception as e:
            print(f"[calibrate] target-based separation skipped: {e}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="gottlux-calibrate",
        description="Map a tracker's relative-distance proxy to metres (per-flight near/far), "
                    "or measure the dual-EBS inter-camera time offset.")
    ap.add_argument("csv", nargs="?", help="a <tag>_tracks.csv produced by the tracking report")
    ap.add_argument("--near", type=float, help="closest known distance for this file [m]")
    ap.add_argument("--far", type=float, help="farthest known distance for this file [m]")
    ap.add_argument("--track", type=int, default=None, help="restrict to a single track id")
    ap.add_argument("--out", default=None, help="output PNG path (default: alongside the CSV)")
    ap.add_argument("--intercam", metavar="FOLDER", default=None,
                    help="measure inter-camera offset from a capture folder instead")
    a = ap.parse_args(argv)
    if a.intercam:
        _intercam(a.intercam)
        return 0
    if not a.csv or a.near is None or a.far is None:
        ap.error("rel→metres mode needs CSV, --near and --far (or use --intercam FOLDER)")
    png, cout = calibrate(a.csv, a.near, a.far, a.track, a.out)
    print(f"[calibrate] plot  -> {png}")
    print(f"[calibrate] table -> {cout}")
    return 0


if __name__ == "__main__":
    main()
