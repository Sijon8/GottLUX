"""
calibrate.py  --  Inter-sensor temporal co-registration for the dual-EBS rig.

Each .raw is zero-based to ITS OWN first event, and each camera's clock starts
at a slightly different instant -> a small constant time offset between the two
streams. Because both sensors view the SAME rotating world, their event-rate
timelines (one background-sweep peak per revolution) are highly correlated; the
lag that maximizes their cross-correlation measures the inter-stream offset.

NOTE (for the paper): with a (near) constant rotation rate omega, a pure timing
offset (dt) and a fixed boresight angular offset (d_phi) are degenerate -- both
produce a constant bearing shift d_phi = omega*dt. The cross-correlation lag is
the quantity you must correct for fusion regardless; separating the two requires
regressing the per-revolution bearing offset against the (slightly varying)
per-revolution omega. See measure_intercam_offset() and the paper notes.
"""
from __future__ import annotations
import numpy as np


def _norm_rate(t, edges):
    r = np.histogram(t, edges)[0].astype(np.float64)
    r -= r.mean()
    s = r.std()
    return r / s if s > 0 else r


def measure_intercam_offset(t0, t1, bin_s=0.002, max_lag_s=0.5):
    """Lag (s) to ADD to camera-1 timestamps to align them with camera-0.

    t0, t1 are event-time arrays (s), each zero-based to its own stream.
    Returns dict(offset_s, peak_corr, lags, corr)."""
    tmax = min(float(t0[-1]), float(t1[-1]))
    edges = np.arange(0, tmax, bin_s)
    r0 = _norm_rate(np.asarray(t0), edges)
    r1 = _norm_rate(np.asarray(t1), edges)
    n = len(r0)
    corr = np.correlate(r0, r1, mode="full") / n     # peak where r1 shifted to match r0
    lags = np.arange(-(n - 1), n) * bin_s
    m = np.abs(lags) <= max_lag_s
    cl, ll = corr[m], lags[m]
    k = int(np.argmax(cl))
    # parabolic sub-bin refinement
    delta = 0.0
    if 0 < k < len(cl) - 1:
        y0, y1, y2 = cl[k - 1], cl[k], cl[k + 1]
        den = (y0 - 2 * y1 + y2)
        if den != 0:
            delta = 0.5 * (y0 - y2) / den
    offset = ll[k] + delta * bin_s
    return dict(offset_s=float(offset), peak_corr=float(cl[k]),
                lags=ll, corr=cl, bin_s=bin_s)


def measure_offset_from_passes(base, fov0=58.0, fov1=50.0, gate_deg=25.0, min_events=50):
    """Measure the inter-camera offset from the DRONE's per-pass crossing time
    (the common, target-based signal). For each shared revolution, compare the
    accumulation-independent per-pass centroid time/bearing of cam0 vs cam1.
    Returns dict(n, dt_ms, dt_std_ms, dbearing_deg).

    fov0/fov1 are the **horizontal** fields of view (deg) of the wide (cam0) and narrow (cam1)
    optics; cam0 defaults to the GenX320 + 1.8 mm rig's 58° (its 76° figure is the diagonal)."""
    import glob, os
    from gottlux.config import Config
    from gottlux.rotation import io_evt21, background, detect
    from gottlux.io.telemetry import Telemetry
    from gottlux.rotation.centroid import per_pass_centroids

    csv = (glob.glob(os.path.join(base, "data_*.csv")) or glob.glob(os.path.join(base, "*.csv")))[0]

    def cam_passes(prefix, fov, gate_center=None):
        raw = glob.glob(os.path.join(base, f"{prefix}*.raw"))[0]
        ev = io_evt21.load(raw); tel = Telemetry(csv)
        t = np.asarray(ev["t"]) / 1e6; tel.refine_offset_to_events(t)
        cfg = Config(mode="rotation"); cfg.fov_deg = fov
        w = cfg.resolved_sensor_wh()[0]            # sensor width from the active profile
        hot = background.hot_pixel_mask(ev, cfg.hot_pixel_pct)
        ref = background.build_reference(ev, tel, float(tel.hall_t[0] + tel.offset), n_phase=cfg.n_phase)
        drop = background.rotation_drop_mask(ev, tel, ref, cfg.n_phase, hot)
        keep, _ = detect.isolate_target(ev, drop, cfg.accum_dt, cfg.min_pixels)
        if gate_center is not None:                # gate kept events to the drone bearing
            dpp = fov / w
            pan = np.rad2deg(np.interp(t[keep] - tel.offset, tel.t, tel.azimuth_unwrapped()))
            bear = np.mod(pan + cfg.az_sign * (np.asarray(ev["x"])[keep] - w / 2) * dpp, 360.0)
            d = np.abs(np.rad2deg(np.angle(np.exp(1j * np.deg2rad(bear - gate_center)))))
            idx = np.where(keep)[0][d < gate_deg]
            keep = np.zeros_like(keep); keep[idx] = True
        P = per_pass_centroids(ev, keep, cfg, tel, min_events=min_events)
        rev = tel.revolution_at(P[:, 0]).astype(int) if len(P) else np.array([], int)
        return P, rev, tel

    P1, r1, tel1 = cam_passes("cam1", fov1)
    gc = float(np.median(P1[:, 1])) if len(P1) else 330.0
    P0, r0, _ = cam_passes("cam0", fov0, gate_center=gc)
    hall = tel1.hall_t
    m1 = {int(rr): P1[i] for i, rr in enumerate(r1)}
    rev_m, omega_m, dts, dbs = [], [], [], []
    for i, rr in enumerate(r0):
        rr = int(rr)
        if rr in m1 and 0 <= rr < len(hall) - 1:
            dts.append(P0[i, 0] - m1[rr][0])
            dbs.append(np.rad2deg(np.angle(np.exp(1j * np.deg2rad(P0[i, 1] - m1[rr][1])))))
            omega_m.append(360.0 / (hall[rr + 1] - hall[rr]))   # per-revolution omega
            rev_m.append(rr)
    dts, dbs = np.array(dts), np.array(dbs)
    omega_m, rev_m = np.array(omega_m), np.array(rev_m)
    return dict(n=len(dts), rev=rev_m, omega=omega_m, dbearing=dbs, dt=dts,
                dt_ms=float(np.median(dts) * 1e3) if len(dts) else float("nan"),
                dt_std_ms=float(np.std(dts) * 1e3) if len(dts) > 1 else float("nan"),
                dbearing_deg=float(np.median(dbs)) if len(dbs) else float("nan"))


def fit_timing_boresight(pass_result):
    """Separate timing skew from boresight misalignment using the per-pass
    bearing offset vs per-pass omega:  d_bearing = omega*dt + d_phi.
    Slope -> dt (timing, s); intercept -> d_phi (boresight, deg). Reports SEs and
    whether the omega spread is sufficient to separate them."""
    w = np.asarray(pass_result.get("omega", []))
    db = np.asarray(pass_result.get("dbearing", []))
    if len(w) < 3 or np.ptp(w) < 1e-6:
        return dict(separable=False, reason="insufficient passes or omega spread")
    # weighted linear regression db_deg = (dt_s)*w_deg_s + dphi_deg
    A = np.column_stack([w, np.ones_like(w)])
    coef, res, *_ = np.linalg.lstsq(A, db, rcond=None)
    dt_s, dphi = float(coef[0]), float(coef[1])
    pred = A @ coef
    dof = max(len(w) - 2, 1)
    sigma2 = float(np.sum((db - pred) ** 2) / dof)
    cov = sigma2 * np.linalg.inv(A.T @ A)
    dt_se, dphi_se = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
    # condition: is omega spread enough to constrain dt? (relative)
    omega_rel_spread = float(np.ptp(w) / np.mean(w))
    separable = omega_rel_spread > 0.05 and abs(dt_s) > 2 * dt_se
    return dict(separable=bool(separable), dt_ms=dt_s * 1e3, dt_se_ms=dt_se * 1e3,
                dphi_deg=dphi, dphi_se_deg=dphi_se, omega_rel_spread=omega_rel_spread,
                omega_mean=float(np.mean(w)))


if __name__ == "__main__":
    import sys, glob, os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gottlux.rotation import io_evt21

    base = sys.argv[1]
    r0 = glob.glob(os.path.join(base, "cam0*.raw"))[0]
    r1 = glob.glob(os.path.join(base, "cam1*.raw"))[0]
    e0 = io_evt21.load(r0); e1 = io_evt21.load(r1)
    t0 = np.asarray(e0["t"]) / 1e6; t1 = np.asarray(e1["t"]) / 1e6
    res = measure_intercam_offset(t0, t1)
    print(f"inter-camera offset = {res['offset_s']*1e3:+.2f} ms  "
          f"(add to cam1 to align with cam0)   peak corr = {res['peak_corr']:.3f}")

    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    ax[0].plot(res["lags"] * 1e3, res["corr"], "b"); ax[0].axvline(res["offset_s"] * 1e3, color="r", ls="--")
    ax[0].set_xlabel("lag [ms]"); ax[0].set_ylabel("norm. cross-corr")
    ax[0].set_title(f"cam0 x cam1 event-rate cross-correlation -> offset {res['offset_s']*1e3:+.2f} ms")
    bin_s = 0.01
    tmax = min(t0[-1], t1[-1]); edges = np.arange(0, tmax, bin_s); ctr = 0.5 * (edges[:-1] + edges[1:])
    a0 = np.histogram(t0, edges)[0] / bin_s; a1 = np.histogram(t1, edges)[0] / bin_s
    ax[1].plot(ctr, a0 / a0.max(), "k", lw=0.7, label="cam0")
    ax[1].plot(ctr, a1 / a1.max(), "r", lw=0.7, label="cam1 (raw)")
    ax[1].plot(ctr + res["offset_s"], a1 / a1.max(), "g", lw=0.7, label="cam1 (aligned)")
    ax[1].set_xlim(0, min(6, tmax)); ax[1].legend(); ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("norm rate")
    ax[1].set_title("event-rate alignment (first 6 s)")
    fig.tight_layout(); out = os.path.join(base, "intercam_offset.png"); fig.savefig(out, dpi=130)
    print("saved", out)

    print("\n--- target-based (drone per-pass) inter-camera offset ---")
    pr = measure_offset_from_passes(base)
    print(f"  matched passes : {pr['n']}")
    print(f"  bearing offset : {pr['dbearing_deg']:+.2f} deg  (cam0 - cam1)  <- co-registration constant")
    fb = fit_timing_boresight(pr)
    print("\n--- timing vs boresight separation (d_bearing = omega*dt + d_phi) ---")
    if fb.get("separable"):
        print(f"  TIMING skew   : {fb['dt_ms']:+.2f} +/- {fb['dt_se_ms']:.2f} ms")
        print(f"  BORESIGHT off : {fb['dphi_deg']:+.2f} +/- {fb['dphi_se_deg']:.2f} deg")
    else:
        print(f"  NOT separable on this capture ({fb.get('reason','omega too constant')}); "
              f"omega rel-spread = {fb.get('omega_rel_spread', 0):.3f}")
        if "dt_ms" in fb:
            print(f"  (unconstrained fit: dt={fb['dt_ms']:+.2f}+/-{fb['dt_se_ms']:.2f} ms, "
                  f"d_phi={fb['dphi_deg']:+.2f}+/-{fb['dphi_se_deg']:.2f} deg)")
    if pr["n"] >= 3:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.scatter(pr["omega"], pr["dbearing"], c="b", s=40)
        if "dt_ms" in fb:
            ww = np.linspace(pr["omega"].min(), pr["omega"].max(), 50)
            ax2.plot(ww, fb["dt_ms"] / 1e3 * ww + fb["dphi_deg"], "r--",
                     label=f"dt={fb['dt_ms']:+.1f}ms, d_phi={fb['dphi_deg']:+.2f}deg")
            ax2.legend()
        ax2.set_xlabel("per-revolution omega [deg/s]"); ax2.set_ylabel("cam0-cam1 bearing offset [deg]")
        ax2.set_title("Timing vs boresight separation"); ax2.grid(alpha=.3)
        out2 = os.path.join(base, "timing_vs_boresight.png"); fig2.savefig(out2, dpi=130)
        print("saved", out2)
