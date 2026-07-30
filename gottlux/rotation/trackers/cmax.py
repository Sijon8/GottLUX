r"""
trackers/cmax.py  --  Contrast-Maximization (CMax) velocity estimation + tracker.

Event-based vision background
-----------------------------
An event camera emits an asynchronous stream of events e_i = (x_i, y_i, t_i, p_i):
a pixel coordinate, a microsecond timestamp, and a polarity (+/-1) firing when
the log-brightness at that pixel crosses a threshold. There are no frames -- only
these sparse, high-temporal-resolution edge signals.

If a brightness pattern moves rigidly across the sensor with image-plane velocity
(vx, vy) [px/s], then the events it generates lie along a *trail* in (x, y, t)
space. Warping every event back to a common reference time t_ref along the
candidate velocity,

        x'_i = x_i + vx * (t_ref - t_i)
        y'_i = y_i + vy * (t_ref - t_i)

and accumulating the warped events into a 2-D histogram produces the
"Image of Warped Events" (IWE). When the candidate velocity matches the true
motion, the trails collapse onto sharp edges and the IWE becomes high-contrast
(few pixels, large counts). When it is wrong, events smear out (many pixels,
low counts). Contrast Maximization (Gallego et al., CVPR 2018) therefore
estimates motion by MAXIMIZING a contrast/focus objective of the IWE -- most
commonly its variance:

        v* = argmax_v  Var( IWE(v) )

This module implements the full CMax pipeline (simulator, IWE builder, focus
losses, optimizer, dense flow field, motion insight extraction, multi-scale
coarse-to-fine refinement, and a velocity Kalman smoother), and wraps the core
velocity estimator into a platform Tracker (`name = "cmax"`).

Two ways to use it
------------------
1. As a registered tracker on real EBS data::

       python -m gottlux.rotation <capture-or-raw> --tracker cmax --analyses detect

   For each detector window the tracker takes the events in a region of
   interest around the detected target, runs CMax to estimate the target's
   image-plane velocity, motion-compensates the centroid (the sharpened IWE
   centroid is a better position estimate than the raw blob centroid), and
   reports a track in the standard bearing/elevation/range schema -- annotated
   with the per-window velocity (vx, vy) [px/s] and the contrast gain.

2. As a self-contained demo on SYNTHETIC data (no hardware needed)::

       python -m gottlux.rotation.trackers.cmax

   This runs the end-to-end pipeline on a simulated moving/rotating bar with a
   known ground-truth velocity, recovers it with CMax, builds a dense flow
   field, extracts object insights (translation + angular velocity + flow
   clustering), and saves the four diagnostic plots.

Dependencies: numpy + scipy only for the core/tracker; matplotlib (plots) and
scikit-learn (optional flow clustering) are imported lazily so this module
loads even when they are absent.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

from gottlux.rotation.trackers import register
from gottlux.rotation.trackers.base import Tracker


# =========================================================================== #
#  1. EVENT DATA SIMULATOR                                                    #
# =========================================================================== #
def simulate_events(shape=(180, 240), v=(60.0, -25.0), duration=0.3,
                    n_edge_events=6000, motion="translate", omega=2.0,
                    n_features=140, pos_noise_px=0.6, t_jitter_s=2e-4,
                    spurious_frac=0.05, seed=0):
    """Synthesize an event stream from a moving textured object, known motion.

    The object carries a fixed set of `n_features` "edge features" rigidly
    attached to its body (the analogue of the persistent brightness edges a
    real scene would have). Each event is one feature firing at one instant as
    the body moves -- so events from the same feature lie along a trail in
    (x, y, t), and warping by the TRUE motion collapses each trail back to a
    sharp point (high IWE contrast). This persistence is what makes Contrast
    Maximization work; a cloud of independent one-shot points would not align.

    Parameters
    ----------
    shape : (H, W) sensor size in pixels.
    v     : ground-truth translational velocity (vx, vy) [px/s] (used for
            ``motion='translate'``; a small fraction also drifts the rotating
            object's centre).
    duration : stream length [s]. Timestamps are returned in *microseconds*.
    n_edge_events : number of "real" (signal) events.
    motion : 'translate' (rigid drift) or 'rotate' (spin about the centre).
    omega  : angular velocity [rad/s] for the rotating case.
    n_features : number of persistent body features that emit events.
    pos_noise_px : Gaussian std added to event positions (sensor noise).
    t_jitter_s   : Gaussian std added to timestamps (latency jitter).
    spurious_frac: fraction of EXTRA uniformly-random noise events to add.

    Returns
    -------
    events : (N, 4) float array of [x, y, t_us, polarity], time-sorted.
    gt     : dict of ground-truth parameters.
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    vx, vy = v
    t = rng.uniform(0.0, duration, n_edge_events)
    fi = rng.integers(0, n_features, n_edge_events)      # which feature fired

    if motion == "rotate":
        # textured disk spinning rigidly about its centre at `omega` rad/s. The
        # local velocity of a rigid rotation is tangential: |v| = omega * r,
        # which is what the per-patch dense flow / angular-velocity fit recover.
        cx0, cy0 = W * 0.5, H * 0.5
        R = 0.34 * min(H, W)
        f_r = R * np.sqrt(rng.uniform(0, 1, n_features))         # body features
        f_a0 = rng.uniform(0, 2 * np.pi, n_features)
        a = f_a0[fi] + omega * t                                 # rigid spin
        bx = cx0 + 0.15 * vx * t + f_r[fi] * np.cos(a)
        by = cy0 + 0.15 * vy * t + f_r[fi] * np.sin(a)
    else:
        # rigid translating textured patch (single global velocity == v). 2-D
        # texture (not one straight edge) makes BOTH velocity components
        # observable -- a straight edge has the classic aperture problem and
        # only constrains motion normal to itself.
        bw, bh = 0.34 * W, 0.34 * H
        cx0, cy0 = W * 0.28, H * 0.58
        f_ox = rng.uniform(-bw / 2, bw / 2, n_features)          # body features
        f_oy = rng.uniform(-bh / 2, bh / 2, n_features)
        bx = cx0 + vx * t + f_ox[fi]
        by = cy0 + vy * t + f_oy[fi]

    # sensor + timing noise
    bx = bx + rng.normal(0, pos_noise_px, bx.size)
    by = by + rng.normal(0, pos_noise_px, by.size)
    t = t + rng.normal(0, t_jitter_s, t.size)
    pol = rng.choice([-1.0, 1.0], bx.size)

    # ~5% spurious events uniformly scattered in the volume
    n_sp = int(spurious_frac * n_edge_events)
    sx = rng.uniform(0, W, n_sp)
    sy = rng.uniform(0, H, n_sp)
    st = rng.uniform(0, duration, n_sp)
    sp = rng.choice([-1.0, 1.0], n_sp)

    x = np.concatenate([bx, sx])
    y = np.concatenate([by, sy])
    tt = np.concatenate([t, st])
    p = np.concatenate([pol, sp])

    # clip to sensor, time-sort, convert time to microseconds
    keep = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (tt >= 0) & (tt <= duration)
    x, y, tt, p = x[keep], y[keep], tt[keep], p[keep]
    o = np.argsort(tt)
    events = np.column_stack([x[o], y[o], tt[o] * 1e6, p[o]])
    gt = dict(v=(vx, vy), motion=motion, omega=omega, shape=shape,
              duration=duration, n_signal=int(keep[:n_edge_events].sum()))
    return events, gt


# =========================================================================== #
#  2. IMAGE OF WARPED EVENTS (IWE) BUILDER                                     #
# =========================================================================== #
def warp_events(x, y, t, vx, vy, t_ref):
    """Warp events to t_ref under a constant image-plane velocity (vx, vy).

    Physics: an event at time t generated by a feature moving at (vx, vy) would
    have been at (x + vx*(t_ref - t), y + vy*(t_ref - t)) at the reference time.
    t and t_ref are in SECONDS; velocity in px/s.
    """
    dt = t_ref - t
    return x + vx * dt, y + vy * dt


def build_iwe(x, y, t, v, t_ref, bins, rng, blur_sigma=1.0):
    """Accumulate warped events into a 2-D histogram (the IWE).

    `bins` = (ny, nx); `rng` = [[ymin, ymax], [xmin, xmax]]. Optional Gaussian
    smoothing makes the focus objective differentiable / less jagged for the
    optimizer (and suppresses single-pixel overfitting of the variance).
    """
    vx, vy = v
    xw, yw = warp_events(x, y, t, vx, vy, t_ref)
    iwe, _, _ = np.histogram2d(yw, xw, bins=bins, range=rng)
    if blur_sigma and blur_sigma > 0:
        iwe = ndimage.gaussian_filter(iwe, blur_sigma)
    return iwe


# =========================================================================== #
#  3. CONTRAST / FOCUS LOSS                                                    #
# =========================================================================== #
def focus_loss(v, x, y, t, t_ref, bins, rng, blur_sigma=1.0, kind="variance"):
    """Scalar focus score of the IWE at velocity v (HIGHER == better focus).

    'variance'   : Var(IWE)  -- the canonical CMax objective; sharp edges =>
                   a few bright pixels and many empty ones => high variance.
    'msq'        : mean of IWE^2 -- the "magnitude/mean-square" focus measure.
    """
    iwe = build_iwe(x, y, t, v, t_ref, bins, rng, blur_sigma)
    if kind == "msq":
        return float(np.mean(iwe ** 2))
    return float(np.var(iwe))


# =========================================================================== #
#  4. VELOCITY OPTIMIZER                                                       #
# =========================================================================== #
def estimate_velocity(x, y, t, t_ref, bins, rng, vmax=400.0, grid=7,
                      blur_sigma=1.0, kind="variance", method="Nelder-Mead",
                      maxiter=80):
    """Find v=(vx, vy) maximizing the focus loss.

    A coarse grid search seeds a local optimizer (the variance landscape is
    non-convex and has a broad basin, so a good initial guess matters). Returns
    the best velocity and an info dict including the optimization trajectory
    (focus value per objective evaluation) for the convergence plot.
    """
    traj = []

    def neg(v):
        val = focus_loss(v, x, y, t, t_ref, bins, rng, blur_sigma, kind)
        traj.append(val)
        return -val

    # --- coarse grid search for initialization ---
    vs = np.linspace(-vmax, vmax, grid)
    best_v, best_l = np.zeros(2), -np.inf
    for vx in vs:
        for vy in vs:
            l = focus_loss((vx, vy), x, y, t, t_ref, bins, rng, blur_sigma, kind)
            traj.append(l)
            if l > best_l:
                best_l, best_v = l, np.array([vx, vy])

    # --- local refinement ---
    opts = dict(maxiter=maxiter, xatol=0.5, fatol=1e-6)
    if method == "Nelder-Mead":
        # Seed an explicit, properly-scaled simplex. scipy's default perturbs a
        # zero-valued start coordinate by only 2.5e-4, which would trap a
        # component initialized at 0 (e.g. vy) -- so set the step from the grid.
        step = max(8.0, 1.2 * vmax / (grid - 1))
        opts["initial_simplex"] = np.array(
            [best_v, best_v + [step, 0.0], best_v + [0.0, step]], float)
    res = minimize(neg, best_v, method=method, options=opts)
    v_ref = np.asarray(res.x)
    l_ref = focus_loss(v_ref, x, y, t, t_ref, bins, rng, blur_sigma, kind)
    if l_ref >= best_l:
        best_v, best_l = v_ref, l_ref

    return best_v, dict(focus=best_l, grid_best=best_v.copy(),
                        loss_traj=traj, success=bool(res.success))


def multiscale_estimate_velocity(x, y, t, t_ref, bins, rng, vmax=400.0,
                                 levels=3, grid=7, blur_sigma=3.0, kind="variance"):
    """BONUS: coarse-to-fine ("pyramid") velocity search.

    Start with a broad velocity range and a heavily-blurred IWE (coarse,
    captures the global basin), then iteratively zoom the search window around
    the current best while *sharpening* the IWE (reducing blur) to refine.
    Mirrors a spatial image pyramid, but the scale here is velocity resolution /
    IWE blur rather than spatial downsampling. Returns (v, loss_trajectory).
    """
    center = np.zeros(2)
    span = np.array([2.0 * vmax, 2.0 * vmax])
    traj = []
    for L in range(levels):
        bs = blur_sigma * (0.5 ** L) + 0.5          # sharpen as we descend
        vxs = np.linspace(center[0] - span[0] / 2, center[0] + span[0] / 2, grid)
        vys = np.linspace(center[1] - span[1] / 2, center[1] + span[1] / 2, grid)
        best_l = -np.inf
        for vx in vxs:
            for vy in vys:
                l = focus_loss((vx, vy), x, y, t, t_ref, bins, rng, bs, kind)
                traj.append(l)
                if l > best_l:
                    best_l, center = l, np.array([vx, vy])
        span = span / (grid / 2.0)                  # zoom into the winning cell
    return center, traj


# =========================================================================== #
#  5. DENSE VELOCITY VECTOR FIELD                                             #
# =========================================================================== #
def dense_flow(x, y, t, shape, patch_size=40, t_ref=None, vmax=400.0,
               min_events=60, blur_sigma=1.0):
    """Per-patch CMax -> dense flow field of shape (H/ps, W/ps, 2).

    The sensor is tiled into `patch_size` x `patch_size` cells; CMax is run
    independently on the events inside each cell to recover that region's local
    velocity. Patches with too few events are left NaN.
    """
    H, W = shape
    if t_ref is None:
        t_ref = float(t.mean())
    ny, nx = H // patch_size, W // patch_size
    flow = np.full((ny, nx, 2), np.nan)
    counts = np.zeros((ny, nx), int)
    for j in range(ny):
        for i in range(nx):
            x0, x1 = i * patch_size, (i + 1) * patch_size
            y0, y1 = j * patch_size, (j + 1) * patch_size
            m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            counts[j, i] = int(m.sum())
            if m.sum() < min_events:
                continue
            v, _ = estimate_velocity(
                x[m], y[m], t[m], t_ref, bins=(patch_size, patch_size),
                rng=[[y0, y1], [x0, x1]], vmax=vmax, grid=7, blur_sigma=blur_sigma)
            flow[j, i] = v
    return flow, counts


# =========================================================================== #
#  6. OBJECT INSIGHT EXTRACTION                                               #
# =========================================================================== #
def _kmeans_np(X, k, iters=60, seed=0):
    """Tiny pure-numpy k-means fallback (used when scikit-learn is absent)."""
    rng = np.random.default_rng(seed)
    if len(X) <= k:
        return np.arange(len(X)), X.copy()
    c = X[rng.choice(len(X), k, replace=False)]
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, lab):
            break
        lab = new
        c = np.array([X[lab == j].mean(0) if np.any(lab == j) else c[j]
                      for j in range(k)])
    return lab, c


def _cluster_flow(features, n_clusters=2, method="kmeans", eps=20.0):
    """Cluster per-patch flow vectors. Prefers scikit-learn; falls back to
    a numpy k-means so the module works without sklearn installed."""
    try:
        if method == "dbscan":
            from sklearn.cluster import DBSCAN
            return DBSCAN(eps=eps, min_samples=3).fit_predict(features)
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=n_clusters, n_init=10,
                      random_state=0).fit_predict(features)
    except Exception:
        lab, _ = _kmeans_np(features, n_clusters)
        return lab


def fit_rigid_motion(centers, vectors):
    """Least-squares fit of a 2-D rigid flow field: v = v0 + omega x r.

    For a planar rigid body, the velocity at position (x, y) is
        vx = v0x - omega * (y - yc)
        vy = v0y + omega * (x - xc)
    where omega is the scalar angular velocity about the centroid (xc, yc).
    We solve the stacked linear system for [v0x, v0y, omega] by least squares.
    Returns (v0, omega, centroid).
    """
    c = centers.mean(0)
    dx = centers[:, 0] - c[0]
    dy = centers[:, 1] - c[1]
    n = len(centers)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    A[0::2, 0] = 1.0           # vx rows
    A[0::2, 2] = -dy
    b[0::2] = vectors[:, 0]
    A[1::2, 1] = 1.0           # vy rows
    A[1::2, 2] = dx
    b[1::2] = vectors[:, 1]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[:2], float(sol[2]), c


def object_insights(flow, patch_size, n_clusters=2, verbose=True):
    """From the dense flow field, extract translation, rotation, and segments.

    Returns a dict with the global translation/rotation and a per-cluster
    breakdown. Prints a human-readable summary when `verbose`.
    """
    ny, nx = flow.shape[:2]
    jj, ii = np.mgrid[0:ny, 0:nx]
    centers = np.column_stack([(ii.ravel() + 0.5) * patch_size,
                               (jj.ravel() + 0.5) * patch_size]).astype(float)
    vecs = flow.reshape(-1, 2)
    good = np.isfinite(vecs).all(1)
    centers, vecs = centers[good], vecs[good]
    out = dict(n_patches=int(good.sum()))
    if len(vecs) == 0:
        if verbose:
            print("[insights] no valid flow patches.")
        return out

    out["mean_translation"] = vecs.mean(0)
    v0, omega, c = fit_rigid_motion(centers, vecs)
    out["global_v0"] = v0
    out["global_omega"] = omega
    out["centroid"] = c

    k = min(n_clusters, len(vecs))
    labels = _cluster_flow(vecs, n_clusters=k) if len(vecs) >= 2 else np.zeros(len(vecs), int)
    out["labels"] = labels
    out["objects"] = []
    if verbose:
        print(f"[insights] {len(vecs)} valid patches; global "
              f"translation=({v0[0]:.1f}, {v0[1]:.1f}) px/s, omega={omega:+.3f} rad/s")
    for lab in sorted(set(labels)):
        if lab < 0:                       # DBSCAN noise label
            continue
        m = labels == lab
        if m.sum() < 2:
            continue
        tv = vecs[m].mean(0)
        lv0, lomega, _ = fit_rigid_motion(centers[m], vecs[m])
        out["objects"].append(dict(label=int(lab), n=int(m.sum()),
                                   translation=tv, omega=lomega))
        if verbose:
            print(f"  Object {lab}: translation=({tv[0]:.1f}, {tv[1]:.1f}) px/s, "
                  f"angular={lomega:+.3f} rad/s  ({m.sum()} patches)")
    return out


# =========================================================================== #
#  BONUS: velocity Kalman smoother (tracks v over successive time windows)     #
# =========================================================================== #
class VelocityKalman:
    """Constant-velocity (random-walk) Kalman filter on the 2-D velocity vector.

    State = [vx, vy]; we observe a noisy CMax velocity per window and smooth it
    over time. `q` is the process-noise variance (how fast the true velocity may
    change), `r` the measurement-noise variance (how noisy each CMax estimate is).
    """

    def __init__(self, q=80.0, r=40.0):
        self.x = None
        self.P = np.eye(2) * 1e4
        self.q, self.r = q, r

    def update(self, z):
        z = np.asarray(z, float)
        if self.x is None:
            if np.all(np.isfinite(z)):
                self.x = z.copy()
            return z
        self.P = self.P + np.eye(2) * self.q                 # predict (random walk)
        if np.all(np.isfinite(z)):                           # update
            S = self.P + np.eye(2) * self.r
            K = self.P @ np.linalg.inv(S)
            self.x = self.x + K @ (z - self.x)
            self.P = (np.eye(2) - K) @ self.P
        return self.x.copy()


# =========================================================================== #
#  7. VISUALIZATION  (matplotlib imported lazily; figures saved to disk)       #
# =========================================================================== #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_event_cloud(events, path, title="raw events (x, y, t)"):
    plt = _plt()
    x, y, t, p = events[:, 0], events[:, 1], events[:, 2] / 1e6, events[:, 3]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sub = np.linspace(0, len(x) - 1, min(4000, len(x))).astype(int)
    ax.scatter(x[sub], t[sub], y[sub], c=np.where(p[sub] > 0, "r", "b"), s=3, alpha=0.5)
    ax.set_xlabel("x [px]"); ax.set_ylabel("t [s]"); ax.set_zlabel("y [px]")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_iwe_comparison(events, shape, v_gt, v_est, path, blur_sigma=1.0):
    plt = _plt()
    H, W = shape
    x, y, t = events[:, 0], events[:, 1], events[:, 2] / 1e6
    t_ref = float(t.mean())
    bins, rng = (H, W), [[0, H], [0, W]]
    iwe0 = build_iwe(x, y, t, (0, 0), t_ref, bins, rng, blur_sigma)
    iwe_g = build_iwe(x, y, t, v_gt, t_ref, bins, rng, blur_sigma)
    iwe_e = build_iwe(x, y, t, v_est, t_ref, bins, rng, blur_sigma)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    fmt = lambda vv: f"({float(vv[0]):.1f}, {float(vv[1]):.1f})"
    for a, im, ti in zip(ax, [iwe0, iwe_g, iwe_e],
                         [f"unwarped  Var={iwe0.var():.1f}",
                          f"ground truth v={fmt(v_gt)}  Var={iwe_g.var():.1f}",
                          f"estimated v={fmt(v_est)}  Var={iwe_e.var():.1f}"]):
        a.imshow(im, cmap="inferno", origin="lower"); a.set_title(ti, fontsize=9); a.axis("off")
    fig.suptitle("Image of Warped Events: sharpening at the correct velocity")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_flow_quiver(flow, events, shape, patch_size, path):
    plt = _plt()
    H, W = shape
    x, y, t = events[:, 0], events[:, 1], events[:, 2] / 1e6
    iwe = build_iwe(x, y, t, (0, 0), float(t.mean()), (H, W), [[0, H], [0, W]], 1.0)
    ny, nx = flow.shape[:2]
    jj, ii = np.mgrid[0:ny, 0:nx]
    cx = (ii + 0.5) * patch_size
    cy = (jj + 0.5) * patch_size
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(iwe, cmap="gray", origin="lower")
    ax.quiver(cx, cy, flow[..., 0], flow[..., 1], color="cyan",
              angles="xy", scale_units="xy", scale=8.0, width=0.004)
    ax.set_title("Dense velocity field (per-patch CMax) over IWE")
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_convergence(loss_traj, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.maximum.accumulate(loss_traj), "k-", lw=1.5, label="best-so-far")
    ax.plot(loss_traj, ".", ms=3, alpha=0.4, label="per-evaluation")
    ax.set_xlabel("objective evaluation"); ax.set_ylabel("focus (IWE variance)")
    ax.set_title("CMax optimization convergence"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


# =========================================================================== #
#  TRACKER:  CMax velocity estimation on the real EBS event stream            #
# =========================================================================== #
@register
class CMaxTracker(Tracker):
    uses_events = True
    name = "cmax"
    description = ("Contrast-Maximization tracker: per-window event focus "
                   "maximization for image-plane velocity + motion-compensated centroid.")
    params = dict(
        window_s=0.02,          # event time window per estimate [s]
        roi_px=36,              # half-size of the ROI around each detection [px]
        vmax_px_s=1200.0,       # velocity search half-range [px/s]
        grid=7,                 # coarse-grid resolution for CMax initialization
        blur_sigma=1.5,         # IWE Gaussian smoothing [px]
        maxiter=60,             # local-optimizer iteration cap
        min_events=120,         # skip windows with fewer ROI events
        max_events_per_win=4000,# subsample cap per window (compute bound)
        max_windows=200,        # cap number of detector windows processed
        kalman_q=120.0,         # velocity-smoother process noise
        kalman_r=60.0,          # velocity-smoother measurement noise
        min_track_len=3,
    )

    def track(self, traj, cfg, tel=None, ev=None):
        if ev is None or not traj:
            return {"tracks": []}
        P = self.params
        W, H = ev["width"], ev["height"]
        x = np.asarray(ev["x"]); y = np.asarray(ev["y"])
        ti = np.asarray(ev["t"])

        # build a time-sorted view for fast windowed slicing (events usually
        # arrive monotonically; only pay for an argsort if they do not).
        if bool(np.all(ti[1:] >= ti[:-1])):
            ts = ti.astype(np.float64) / 1e6
            torder = None
        else:
            torder = np.argsort(ti, kind="stable")
            ts = ti[torder].astype(np.float64) / 1e6

        # detector measurements (centroids) that seed the per-window ROIs
        td = np.asarray(traj["t"], float)
        cx = np.asarray(traj["cx"], float)
        cy = np.asarray(traj["cy"], float)
        tr_rng = np.asarray(traj.get("range_m", np.full_like(td, np.nan)), float)
        o = np.argsort(td)
        td, cx, cy, tr_rng = td[o], cx[o], cy[o], tr_rng[o]
        if td.size > P["max_windows"]:
            sel = np.unique(np.linspace(0, td.size - 1, P["max_windows"]).round().astype(int))
            td, cx, cy, tr_rng = td[sel], cx[sel], cy[sel], tr_rng[sel]

        half = P["window_s"] / 2.0
        roi = P["roi_px"]
        vk = VelocityKalman(P["kalman_q"], P["kalman_r"])
        sub_rng = np.random.default_rng(0)
        rec = []         # (t, cx_ref, cy_ref, vx, vy, gain, range)

        for k in range(td.size):
            t0, t1 = td[k] - half, td[k] + half
            i0 = int(np.searchsorted(ts, t0))
            i1 = int(np.searchsorted(ts, t1))
            if i1 - i0 < P["min_events"]:
                rec.append((td[k], cx[k], cy[k], np.nan, np.nan, np.nan, tr_rng[k]))
                continue
            idx = slice(i0, i1) if torder is None else torder[i0:i1]
            xs = x[idx].astype(np.float64)
            ys = y[idx].astype(np.float64)
            tw = ts[i0:i1]
            x0, x1 = cx[k] - roi, cx[k] + roi
            y0, y1 = cy[k] - roi, cy[k] + roi
            m = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            if int(m.sum()) < P["min_events"]:
                rec.append((td[k], cx[k], cy[k], np.nan, np.nan, np.nan, tr_rng[k]))
                continue
            ex, ey, et = xs[m], ys[m], tw[m]
            if ex.size > P["max_events_per_win"]:
                pick = sub_rng.choice(ex.size, P["max_events_per_win"], replace=False)
                ex, ey, et = ex[pick], ey[pick], et[pick]

            bins = (int(2 * roi), int(2 * roi))
            box = [[y0, y1], [x0, x1]]
            v, _info = estimate_velocity(ex, ey, et, td[k], bins, box,
                                         vmax=P["vmax_px_s"], grid=P["grid"],
                                         blur_sigma=P["blur_sigma"], maxiter=P["maxiter"])
            vsm = vk.update(v)
            # contrast gain: how much sharper the IWE is at v vs unwarped
            c1 = focus_loss(v, ex, ey, et, td[k], bins, box, P["blur_sigma"])
            c0 = focus_loss((0.0, 0.0), ex, ey, et, td[k], bins, box, P["blur_sigma"])
            gain = c1 / c0 if c0 > 0 else np.nan
            # motion-compensated centroid (sharpened position estimate at t_ref)
            xw, yw = warp_events(ex, ey, et, vsm[0], vsm[1], td[k])
            cxr = float(np.clip(xw.mean(), 0, W - 1))
            cyr = float(np.clip(yw.mean(), 0, H - 1))
            rec.append((td[k], cxr, cyr, float(vsm[0]), float(vsm[1]), float(gain), tr_rng[k]))

        rec = np.asarray(rec, float)
        if rec.shape[0] < P["min_track_len"]:
            print(f"[track:{self.name}] too few usable windows ({rec.shape[0]})")
            return {"tracks": []}

        tt, CX, CY = rec[:, 0], rec[:, 1], rec[:, 2]
        deg_per_px = cfg.fov_deg / W
        if cfg.mode == "rotation" and tel is not None:
            pan = np.rad2deg(np.interp(tt - tel.offset, tel.t, tel.azimuth_unwrapped()))
            az = np.mod(pan + cfg.az_sign * (CX - W / 2) * deg_per_px, 360.0)
        else:
            az = cfg.az_sign * (CX - W / 2) * deg_per_px
        elev = (H / 2 - CY) * deg_per_px

        speed = np.hypot(rec[:, 3], rec[:, 4])
        nfin = int(np.isfinite(speed).sum())
        med_speed = float(np.nanmedian(speed)) if nfin else float("nan")
        med_gain = float(np.nanmedian(rec[:, 5])) if nfin else float("nan")
        track = dict(id=0, t=tt, azimuth_deg=az, elev_deg=elev, range_m=rec[:, 6],
                     cx=CX, cy=CY, vx_px_s=rec[:, 3], vy_px_s=rec[:, 4],
                     speed_px_s=speed, contrast_gain=rec[:, 5])
        print(f"[track:{self.name}] 1 track from {td.size} windows "
              f"({nfin} with a velocity lock); median speed {med_speed:.0f} px/s, "
              f"median contrast gain {med_gain:.2f}x")
        return {"tracks": [track]}


# =========================================================================== #
#  __main__ :  end-to-end CMax pipeline on SYNTHETIC data                      #
# =========================================================================== #
def _demo(outdir=None):
    import os
    outdir = outdir or os.path.join(os.getcwd(), "cmax_demo_out")
    os.makedirs(outdir, exist_ok=True)
    print("=" * 70)
    print("  CONTRAST-MAXIMIZATION velocity estimation -- synthetic demo")
    print("=" * 70)
    shape = (180, 240)
    bins, rng = (shape[0], shape[1]), [[0, shape[0]], [0, shape[1]]]

    # ----------------------------------------------------------------- #
    # PART A -- a rigidly TRANSLATING edge: validate the global CMax     #
    #           velocity estimate against the known ground truth.        #
    # ----------------------------------------------------------------- #
    v_gt = (70.0, -30.0)
    duration = 0.30
    ev_t, gt = simulate_events(shape=shape, v=v_gt, duration=duration,
                               n_edge_events=12000, motion="translate",
                               spurious_frac=0.05, seed=1)
    print(f"\n[A] translating texture: {len(ev_t)} events "
          f"({gt['n_signal']} signal + ~5% spurious), gt v={v_gt} px/s")
    x, y, t = ev_t[:, 0], ev_t[:, 1], ev_t[:, 2] / 1e6
    t_ref = float(t.mean())
    v_est, info = estimate_velocity(x, y, t, t_ref, bins, rng, vmax=300, grid=9, blur_sigma=2.0)
    v_ms, _ = multiscale_estimate_velocity(x, y, t, t_ref, bins, rng, vmax=300, levels=3, grid=9)
    err = np.hypot(v_est[0] - v_gt[0], v_est[1] - v_gt[1])
    print(f"    GROUND TRUTH velocity      : ({v_gt[0]:6.1f}, {v_gt[1]:6.1f}) px/s")
    print(f"    CMax estimate (single)     : ({v_est[0]:6.1f}, {v_est[1]:6.1f}) px/s   focus={info['focus']:.1f}")
    print(f"    CMax estimate (multiscale) : ({v_ms[0]:6.1f}, {v_ms[1]:6.1f}) px/s")
    print(f"    error |v_est - v_gt|       : {err:.1f} px/s ({100*err/np.hypot(*v_gt):.1f}% of speed)")

    # bonus) Kalman-smooth the velocity estimated over successive time windows.
    # Windows must be wide enough that the object actually MOVES a few pixels
    # inside each (CMax cannot resolve a velocity from sub-pixel displacement),
    # so we use overlapping ~0.12 s windows rather than many tiny slices.
    vk = VelocityKalman(q=60.0, r=50.0)
    win, stride = 0.12, 0.05
    raw, sm = [], np.array([np.nan, np.nan])
    for c in np.arange(win / 2, duration - win / 2 + 1e-9, stride):
        m = (t >= c - win / 2) & (t < c + win / 2)
        if m.sum() < 200:
            sm = vk.update([np.nan, np.nan]); continue
        vw, _ = estimate_velocity(x[m], y[m], t[m], c, bins, rng, vmax=300, grid=7, blur_sigma=2.0)
        raw.append(vw); sm = vk.update(vw)
    raw = np.array(raw)
    print(f"    per-window raw mean v      : ({raw[:,0].mean():6.1f}, {raw[:,1].mean():6.1f}) px/s "
          f"(std {raw[:,0].std():.1f}, {raw[:,1].std():.1f})")
    print(f"    Kalman-smoothed final v    : ({sm[0]:6.1f}, {sm[1]:6.1f}) px/s")

    # ----------------------------------------------------------------- #
    # PART B -- a spinning textured disk: dense flow + angular velocity. #
    # ----------------------------------------------------------------- #
    omega_gt = 2.5
    ev_r, gtr = simulate_events(shape=shape, v=(0.0, 0.0), duration=0.12,
                                n_edge_events=24000, motion="rotate",
                                omega=omega_gt, spurious_frac=0.05, seed=2)
    print(f"\n[B] spinning disk: {len(ev_r)} events, gt omega={omega_gt} rad/s")
    xr, yr, tr = ev_r[:, 0], ev_r[:, 1], ev_r[:, 2] / 1e6
    patch = 30
    flow, counts = dense_flow(xr, yr, tr, shape, patch_size=patch,
                              t_ref=float(tr.mean()), vmax=260, min_events=40)
    insights = object_insights(flow, patch, n_clusters=2)
    om = insights.get("global_omega", float("nan"))
    print(f"    recovered global omega     : {om:+.3f} rad/s (gt {omega_gt:+.3f})")

    # ----------------------------------------------------------------- #
    # Visualization                                                      #
    # ----------------------------------------------------------------- #
    p1 = plot_event_cloud(ev_t, os.path.join(outdir, "1_event_cloud.png"))
    p2 = plot_iwe_comparison(ev_t, shape, v_gt, v_est, os.path.join(outdir, "2_iwe_compare.png"), blur_sigma=1.0)
    p3 = plot_flow_quiver(flow, ev_r, shape, patch, os.path.join(outdir, "3_flow_quiver.png"))
    p4 = plot_convergence(info["loss_traj"], os.path.join(outdir, "4_convergence.png"))
    print("\nplots saved:")
    for p in (p1, p2, p3, p4):
        print(f"  {p}")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
