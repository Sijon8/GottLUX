"""
hummingbird_gallery.py — renderings of the bundled hummingbird-fight recording, made with
the GottLUX library (decode, denoise) plus NumPy / Matplotlib / imageio.

    python examples/hummingbird_gallery.py [recording.raw] [out_dir] [--no-video]

Writes, into *out_dir* (default ``examples/hummingbird_gallery_out``):

* ``hb1_light_painting.png``  every denoised event of the 3 s clip on one frame; brightness
                              = events per pixel (log), colour = time of the pixel's last
                              event, labelled with the dominant bird's position every 0.3 s.
* ``hb2_spacetime.png``       the close pass and hover (0.55–1.45 s) as an (x, t, y) event
                              cloud: each wingbeat is a stripe along the time axis.
* ``hb3_stroboscope.png``     a stroboscope made from events: during the hover the wingbeat
                              frequency f0 is measured, and events are folded by wingbeat phase
                              (t·f0 mod 1) into 8 frames of one averaged wing stroke.
* ``hb4_wingbeat.png``        the wingbeat line over the whole clip: a spectrogram of the event
                              rate inside a box that follows the dominant bird.
* ``hb_slowmo.mp4``           11x slow motion (3 ms of recording per frame at 30 fps) of the
                              denoised events (ON orange, OFF blue) with the event-rate trace
                              and a playhead.
* ``hb_orbit.mp4``            a camera orbit around the space-time cloud of figure 2.

Method (first principles): an event camera reports per-pixel log-brightness changes. A
beating wing modulates the brightness of the pixels it crosses twice per stroke cycle, so
the event rate inside a box around the bird is periodic at the wingbeat frequency; its
spectrum peaks at f0. Folding event times by the phase t·f0 mod 1 stacks ~30 strokes onto
one cycle, which is the same trick a stroboscope plays with a flash.

Denoise: GottLUX ``filter_window`` with the hot-pixel cut (top 0.1 % of pixels by count)
and the background-activity filter (an event is kept only if an 8-neighbour fired within
2 ms). On this clip that keeps 203,346 of 1,434,823 events.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REC = os.path.join(HERE, "data", "Humming_Bird_Fight_merged_2.raw")

# Dark chart surface + ink (dataviz reference palette); ON/OFF = the validated
# orange/blue pair (CVD ΔE 26.8 on the dark surface).
SURFACE = "#1a1a19"
INK = "#f0efec"
INK2 = "#c3c2b7"
ON = np.array([0xd9, 0x59, 0x26]) / 255
OFF = np.array([0x39, 0x87, 0xe5]) / 255
# single-hue sequential ramp (blue 650 → 100 → white) for time and magnitude
RAMP = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6", "#e6f0fc"]


def _ramp(name="gl_blue"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, RAMP)


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": "#4a4945", "font.size": 10, "axes.titlesize": 11,
        "axes.titlecolor": INK, "font.family": "DejaVu Sans",
    })


# ---------------------------------------------------------------------------- data
def load_clean(path):
    import gottlux.io as gio
    from gottlux.core.denoise import FilterSettings, filter_window
    rec = gio.load(path)
    win = filter_window(rec.all(), FilterSettings(enabled=True, hot_pixel=True, hot_pct=99.9,
                                                 baf=True, baf_dt_us=2000))
    ev = {k: np.asarray(getattr(win, k)) for k in ("x", "y", "p", "t")}
    ev["W"], ev["H"] = int(rec.width), int(rec.height)
    ev["n_raw"] = rec.n
    return rec, ev


def track_bird(ev, step_us=10_000):
    """Dominant-bird centroid every *step_us*: the densest 40 px neighbourhood of denoised
    events in a 20 ms window (8 px cells, 5×5 cell box)."""
    from scipy import ndimage
    x, y, t = ev["x"], ev["y"], ev["t"]
    out = []
    for a0 in range(0, int(t[-1]), step_us):
        a, b = np.searchsorted(t, [a0, a0 + 2 * step_us])
        if b - a < 50:
            continue
        img = np.zeros((ev["H"] // 8 + 1, ev["W"] // 8 + 1))
        np.add.at(img, (y[a:b] // 8, x[a:b] // 8), 1)
        iy, ix = np.unravel_index(np.argmax(ndimage.uniform_filter(img, 3)), img.shape)
        sel = (np.abs(x[a:b] // 8 - ix) <= 2) & (np.abs(y[a:b] // 8 - iy) <= 2)
        if sel.sum() >= 40:
            out.append(((a0 + step_us) / 1e6, x[a:b][sel].mean(), y[a:b][sel].mean(), sel.sum()))
    tr = np.array(out)
    return {"t": tr[:, 0], "cx": tr[:, 1], "cy": tr[:, 2], "n": tr[:, 3]}


def bird_signal(ev, track, t0, t1, fs=2000.0, half=24):
    """Event-count series (fs Hz) inside a ±*half* px box that follows the bird."""
    x, y, t = ev["x"], ev["y"], ev["t"]
    a, b = np.searchsorted(t, [t0 * 1e6, t1 * 1e6])
    ts = t[a:b] / 1e6
    bx = np.interp(ts, track["t"], track["cx"]); by = np.interp(ts, track["t"], track["cy"])
    inb = (np.abs(x[a:b] - bx) < half) & (np.abs(y[a:b] - by) < half)
    n = int(round((t1 - t0) * fs))
    return np.histogram(ts[inb], bins=n, range=(t0, t1))[0].astype(float)


def wingbeat_f0(sig, fs=2000.0, lo=40.0, hi=110.0):
    """Peak of the zero-padded spectrum in [lo, hi] Hz, refined by a parabola."""
    s = (sig - sig.mean()) * np.hanning(sig.size)
    nfft = 1 << 17
    P = np.abs(np.fft.rfft(s, nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fs)
    band = np.where((f >= lo) & (f <= hi))[0]
    k = band[np.argmax(P[band])]
    d = 0.5 * (P[k - 1] - P[k + 1]) / (P[k - 1] - 2 * P[k] + P[k + 1])
    return float(f[k] + d * (f[1] - f[0])), f, P


# ---------------------------------------------------------------------------- figures
def fig_light_painting(ev, track, out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    x, y, t = ev["x"], ev["y"], ev["t"] / 1e6
    W, H = ev["W"], ev["H"]
    flat = y.astype(np.int64) * W + x
    cnt = np.bincount(flat, minlength=W * H).astype(float)
    last = np.zeros(W * H); np.maximum.at(last, flat, t)
    bright = np.clip(np.log1p(cnt) / np.log1p(np.percentile(cnt[cnt > 0], 99.5)), 0, 1)
    bright[cnt < 3] = 0.0              # residual shot noise: isolated pixels, 1-2 events
    cmap = _ramp()
    rgb = cmap(Normalize(0, t[-1])(last))[:, :3] * bright[:, None]
    rgb = rgb.reshape(H, W, 3)
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ax.imshow(rgb, interpolation="bilinear")
    # direct labels: the dominant bird's position every 0.3 s (time is never colour-only)
    for tl in np.arange(0.3, t[-1], 0.3):
        k = np.argmin(np.abs(track["t"] - tl))
        if abs(track["t"][k] - tl) < 0.02 and track["n"][k] > 200:
            ax.plot(track["cx"][k], track["cy"][k], "o", ms=5, mfc="none", mec=INK, mew=1)
            ax.annotate(f"{tl:.1f} s", (track["cx"][k], track["cy"][k]), xytext=(6, -12),
                        textcoords="offset points", color=INK, fontsize=8)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_xlabel("sensor x (px)"); ax.set_ylabel("sensor y (px)")
    ax.set_title(f"Light painting — {ev['t'].size:,} denoised events over "
                 f"{t[-1]:.2f} s\nbrightness = events per pixel (log) · "
                 "colour = time of the pixel's last event", loc="left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, t[-1]))
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("time of last event (s)"); cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=170); plt.close(fig)


def _cloud(ev, t0, t1, max_pts=160_000, seed=0, min_voxel=4):
    """Events in [t0, t1) that sit in a dense space-time voxel (4 px × 4 px × 5 ms holding
    at least *min_voxel* events), subsampled to *max_pts* — the birds without the speckle."""
    x, y, p, t = ev["x"], ev["y"], ev["p"], ev["t"]
    a, b = np.searchsorted(t, [t0 * 1e6, t1 * 1e6])
    vox = ((t[a:b] - t[a]) // 5000).astype(np.int64) * 10_000 + (y[a:b] // 4) * 100 + x[a:b] // 4
    _, inv, cnt = np.unique(vox, return_inverse=True, return_counts=True)
    idx = np.arange(a, b)[cnt[inv] >= min_voxel]
    if idx.size > max_pts:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_pts, replace=False))
    return x[idx], y[idx], p[idx], t[idx] / 1e6


def _draw_cloud(ax, cx, cy, cp, ct, W, H):
    col = np.where(cp[:, None] == 1, ON, OFF)
    ax.scatter(cx, ct, cy, c=col, s=0.35, alpha=0.55, linewidths=0, depthshade=False)
    ax.set_xlim(0, W); ax.set_zlim(H, 0); ax.set_ylim(ct.min(), ct.max())
    ax.set_xlabel("x (px)", labelpad=6); ax.set_ylabel("time (s)", labelpad=10)
    ax.set_zlabel("y (px)", labelpad=6)
    ax.set_box_aspect((1, 2.4, 1), zoom=1.25)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor(SURFACE); a.pane.set_edgecolor("#3a3936")
        a._axinfo["grid"]["color"] = "#2c2b29"


def fig_spacetime(ev, out, t0=0.55, t1=1.45):
    import matplotlib.pyplot as plt
    cx, cy, cp, ct = _cloud(ev, t0, t1)
    fig = plt.figure(figsize=(9.6, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    _draw_cloud(ax, cx, cy, cp, ct, ev["W"], ev["H"])
    ax.view_init(elev=18, azim=-62)
    fig.suptitle(f"Space-time event cloud, {t0:.2f}–{t1:.2f} s ({cx.size:,} events in dense "
                 "voxels) — each wingbeat is a fin along the time axis", x=0.01, y=0.97,
                 ha="left", fontsize=11, color=INK)
    fig.text(0.02, 0.03, "■ ON (brighter)", color=ON, fontsize=9)
    fig.text(0.16, 0.03, "■ OFF (darker)", color=OFF, fontsize=9)
    fig.subplots_adjust(left=0, right=1, bottom=0.02, top=0.95)
    fig.savefig(out, dpi=160); plt.close(fig)


def fig_stroboscope(ev, track, out, t0=1.00, t1=1.45, nbin=8):
    import matplotlib.pyplot as plt
    fs = 2000.0
    sig = bird_signal(ev, track, t0, t1, fs)
    f0, _, _ = wingbeat_f0(sig, fs)
    k = (track["t"] >= t0) & (track["t"] <= t1)
    bx, by = np.median(track["cx"][k]), np.median(track["cy"][k])
    half = 34
    x0, y0 = int(max(bx - half, 0)), int(max(by - half, 0))
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    x, y, p, t = ev["x"], ev["y"], ev["p"], ev["t"]
    a, b = np.searchsorted(t, [t0 * 1e6, t1 * 1e6])
    sel = slice(a, b)
    xs, ys, ps, ts = x[sel], y[sel], p[sel], t[sel] / 1e6
    inb = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    xs, ys, ps, ts = xs[inb] - x0, ys[inb] - y0, ps[inb], ts[inb]
    phase = (ts * f0) % 1.0
    pb = np.minimum((phase * nbin).astype(int), nbin - 1)
    frames = []
    for i in range(nbin):
        on = np.zeros((2 * half, 2 * half)); off = np.zeros_like(on)
        m = pb == i
        np.add.at(on, (ys[m & (ps == 1)], xs[m & (ps == 1)]), 1)
        np.add.at(off, (ys[m & (ps == 0)], xs[m & (ps == 0)]), 1)
        frames.append((on, off))
    vmax = np.percentile(np.concatenate([np.r_[a.ravel(), b.ravel()] for a, b in frames]), 99.5)
    fig, axes = plt.subplots(2, nbin // 2, figsize=(9.6, 5.4))
    for i, (ax, (on, off)) in enumerate(zip(axes.ravel(), frames, strict=True)):
        rgb = (np.clip(on / vmax, 0, 1)[..., None] * ON + np.clip(off / vmax, 0, 1)[..., None] * OFF)
        ax.imshow(np.clip(rgb, 0, 1), interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"phase {i * 360 // nbin}°–{(i + 1) * 360 // nbin}°", fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    n_strokes = int((t1 - t0) * f0)
    fig.suptitle(f"Event stroboscope — hover {t0:.2f}–{t1:.2f} s, wingbeat f0 = {f0:.1f} Hz "
                 f"(period {1e3 / f0:.1f} ms); {n_strokes} strokes folded onto one cycle",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.015, f"{2 * half}×{2 * half} px crop at sensor ({x0}, {y0}) · "
             "orange = ON events, blue = OFF events, per phase bin", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out, dpi=170); plt.close(fig)
    return f0


def fig_wingbeat(ev, track, out, fs=2000.0, win_s=0.25, hop_s=0.01):
    import matplotlib.pyplot as plt
    t_end = ev["t"][-1] / 1e6
    sig = bird_signal(ev, track, 0.0, t_end, fs)
    n = int(win_s * fs); hop = int(hop_s * fs)
    starts = np.arange(0, sig.size - n, hop)
    win = np.hanning(n)
    S = np.array([np.abs(np.fft.rfft((sig[s:s + n] - sig[s:s + n].mean()) * win, 4 * n)) ** 2
                  for s in starts]).T
    f = np.fft.rfftfreq(4 * n, 1 / fs)
    keep = (f >= 20) & (f <= 200)
    S = S[keep]; f = f[keep]
    tc = (starts + n / 2) / fs
    # whiten each 0.25 s column by its own median so a tonal line stands out even while the
    # bird darts past (the fly-by at 0.78 s is broadband and would set the colour scale)
    Sdb = 10 * np.log10(S / np.median(S, axis=0, keepdims=True) + 1e-9)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.6, 6.2), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    im = ax.pcolormesh(tc, f, Sdb, cmap=_ramp(), vmin=0, vmax=np.percentile(Sdb, 99.5),
                       shading="auto", rasterized=True)
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Wingbeat over the whole clip: spectrogram of the event rate in a 48 px box\n"
                 "that follows the densest bird (0.25 s windows, each column whitened)",
                 loc="left")
    for k, lbl in ((1, "f0"), (2, "2·f0")):
        ax.axhline(71.6 * k, color=INK2, lw=0.6, ls=(0, (2, 4)))
        ax.text(tc[-1], 71.6 * k + 3, f"{lbl} of the hover ({71.6 * k:.0f} Hz)",
                ha="right", va="bottom", fontsize=8, color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("dB above the column median"); cb.outline.set_visible(False)
    rate = np.histogram(ev["t"] / 1e6, bins=int(t_end / 0.01), range=(0, t_end))[0] / 0.01
    ax2.fill_between(np.arange(rate.size) * 0.01 + 0.005, rate / 1e3, color=RAMP[3], lw=0)
    ax2.set_ylabel("denoised\nkev/s"); ax2.set_xlabel("time (s)")
    ax2.set_xlim(0, t_end)
    pos, pos2 = ax.get_position(), ax2.get_position()
    ax2.set_position([pos2.x0, pos2.y0, pos.width, pos2.height])   # align under the colourbar
    for a in (ax, ax2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.savefig(out, dpi=160); plt.close(fig)


# ---------------------------------------------------------------------------- video
def _font(size):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def video_slowmo(ev, out, step_ms=3.0, trail_ms=6.0, scale=2, fps=30):
    """Slow motion: each frame advances *step_ms* of recording time (3 ms at 30 fps = 11x
    slower); events fade
    with an exponential trail (time constant *trail_ms*); a rate strip carries a playhead."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    x, y, p, t = ev["x"], ev["y"], ev["p"], ev["t"]
    W, H = ev["W"], ev["H"]
    t_end = t[-1] / 1e3
    strip_h = 70
    rate = np.histogram(t / 1e6, bins=600, range=(0, t_end / 1e3))[0].astype(float)
    rate /= rate.max()
    font, small = _font(18), _font(13)
    on = np.zeros((H, W)); off = np.zeros((H, W))
    decay = np.exp(-step_ms / trail_ms)
    speed = (1000.0 / fps) / step_ms
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8,
                                macro_block_size=8, pixelformat="yuv420p")
    i0 = 0
    for k, fe in enumerate(np.arange(step_ms, t_end, step_ms)):
        i1 = np.searchsorted(t, fe * 1e3)
        on *= decay; off *= decay
        m = p[i0:i1] == 1
        np.add.at(on, (y[i0:i1][m], x[i0:i1][m]), 1.0)
        np.add.at(off, (y[i0:i1][~m], x[i0:i1][~m]), 1.0)
        i0 = i1
        rgb = (np.clip(on / 2.0, 0, 1)[..., None] * ON + np.clip(off / 2.0, 0, 1)[..., None] * OFF)
        img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(img).resize((W * scale, H * scale), Image.BILINEAR)
        canvas = Image.new("RGB", (W * scale, H * scale + strip_h), SURFACE)
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        y0 = H * scale + 8; sw = W * scale - 16
        for j, r in enumerate(rate):
            xx = 8 + int(j * sw / rate.size)
            d.line([(xx, y0 + 50), (xx, y0 + 50 - int(r * 40))], fill=RAMP[2])
        px = 8 + int(fe / t_end * sw)
        d.line([(px, y0 + 4), (px, y0 + 54)], fill=INK, width=2)
        d.text((10, 8), f"t = {fe / 1e3:.3f} s", fill=INK, font=font)
        d.text((10, 32), f"{speed:.0f}× slower · {trail_ms:g} ms trail", fill=INK2, font=small)
        d.text((W * scale - 150, 8), "ON", fill=tuple((ON * 255).astype(int)), font=small)
        d.text((W * scale - 110, 8), "OFF", fill=tuple((OFF * 255).astype(int)), font=small)
        writer.append_data(np.asarray(canvas))
    writer.close()


def video_orbit(ev, out, t0=0.55, t1=1.45, n_frames=168, fps=24):
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    cx, cy, cp, ct = _cloud(ev, t0, t1, max_pts=90_000)
    fig = plt.figure(figsize=(7.2, 5.4), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    _draw_cloud(ax, cx, cy, cp, ct, ev["W"], ev["H"])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8,
                                macro_block_size=8, pixelformat="yuv420p")
    for k in range(n_frames):
        ax.view_init(elev=18 + 10 * np.sin(2 * np.pi * k / n_frames), azim=-62 + 360 * k / n_frames)
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    writer.close(); plt.close(fig)


# ---------------------------------------------------------------------------- main
def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    argv = list(sys.argv[1:] if argv is None else argv)
    video = "--no-video" not in argv
    argv = [a for a in argv if not a.startswith("--")]
    path = argv[0] if argv else DEFAULT_REC
    out = argv[1] if len(argv) > 1 else os.path.join(HERE, "hummingbird_gallery_out")
    os.makedirs(out, exist_ok=True)
    _style()
    rec, ev = load_clean(path)
    track = track_bird(ev)
    print(f"{rec.name}: kept {ev['t'].size:,} of {ev['n_raw']:,} events after denoise")
    fig_light_painting(ev, track, os.path.join(out, "hb1_light_painting.png"))
    fig_spacetime(ev, os.path.join(out, "hb2_spacetime.png"))
    f0 = fig_stroboscope(ev, track, os.path.join(out, "hb3_stroboscope.png"))
    print(f"hover wingbeat f0 = {f0:.2f} Hz")
    fig_wingbeat(ev, track, os.path.join(out, "hb4_wingbeat.png"))
    if video:
        video_slowmo(ev, os.path.join(out, "hb_slowmo.mp4"))
        video_orbit(ev, os.path.join(out, "hb_orbit.mp4"))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
