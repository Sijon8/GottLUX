"""
ladder_report.py — export a paper-ready rotor-ladder study.

Bundles, into one folder:

* **full-resolution PNGs** of the 3-D space-time event cloud rendered from several camera
  angles (the "entire scene from various angles"),
* the **rotor-ladder measurement figure** (the ``(t, x)`` staircase + the comb autocorrelation)
  and the **region spectrum** plot,
* a **measurements table** (JSON + CSV), and
* a self-contained, **compilable LaTeX report** that lays the figures out with captions and
  labels and explains how the rotor-ladder algorithm works (with the equations).

The writer is pure — NumPy + matplotlib + PIL + the :mod:`gottlux.io.export` helpers, no Qt.
The multi-angle scene renders are produced by the caller (the Space-time view, which owns the
GL context) and handed in as RGB arrays, so this module is testable without Qt/OpenGL.
"""
from __future__ import annotations

import datetime
import os

import numpy as np

from gottlux.io import export
from gottlux.rotation.rotor_ladder import ladder_figure, ladder_signature


# --------------------------------------------------------------------- small helpers
def _slug(s) -> str:
    """A filename-safe, LaTeX-safe (no underscores) slug."""
    return "".join(c if c.isalnum() else "-" for c in str(s).lower()).strip("-") or "x"


def _latex_escape(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def _fmt(v) -> str:
    if v is None:
        return "--"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _save_png(rgb, path) -> str:
    rgb = np.asarray(rgb).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except Exception:                                   # pragma: no cover - PIL fallback
        import matplotlib.pyplot as plt
        plt.imsave(path, rgb)
    return path


def _spectrum_figure(spectrum, band=None, title="Region spectrum (box)"):
    import matplotlib.pyplot as plt
    f = np.asarray(spectrum.freqs, float)
    p = np.maximum(np.asarray(spectrum.power, float), 1e-12)
    fig, ax = plt.subplots(figsize=(7.5, 4.0), facecolor="w")
    ax.semilogy(f, p, color="#1f4e8c", lw=1.2)
    if band:
        ax.axvspan(band[0], band[1], color="#39c5cf", alpha=0.15,
                   label=f"band {band[0]:g}-{band[1]:g} Hz")
    pk = float(getattr(spectrum, "peak_freq", np.nan))
    if np.isfinite(pk):
        ax.axvline(pk, color="#c62828", ls="--", lw=1.0)
        ax.annotate(f"peak {pk:.0f} Hz (SNR {getattr(spectrum, 'snr', float('nan')):.1f})",
                    xy=(pk, getattr(spectrum, "peak_power", p.max())), fontsize=9, color="#c62828")
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("power")
    ax.set_title(title); ax.grid(True, ls="--", alpha=0.3)
    if band:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- the algorithm writeup
_ALGO = r"""
\section{How the rotor-ladder detector works}

When the event sensor \emph{spins} (here $\approx 1\,\mathrm{Hz}$) and its field of view sweeps
across a multirotor, the rotor's high blade-pass frequency is \textbf{spatially demodulated} by
the sweep: each blade-pass burst of events lands at a slightly different sensor column than the
last, so the rotor draws a regularly spaced \textbf{ladder / staircase} of bursts across the
sweep direction. A building edge is swept too, but has no high-frequency burst structure, so it
leaves a \emph{continuous} streak rather than a comb; unstructured noise leaves neither.

\paragraph{Geometry.} Let the sensor spin at angular rate $\Omega$ (rad/s) and let the pixel
angular scale be $\beta = \mathrm{FOV}/W$ (rad/px). A target at world azimuth
$\theta_d(t)=\theta_0+\Omega_d t$ images at sensor column
\[
  x(t) = x_c + \frac{\theta_d(t)-\Omega t}{\beta}
       = \mathrm{const} + \frac{\Omega_d-\Omega}{\beta}\,t ,
\]
so it drifts at the \textbf{sweep velocity} $v=\mathrm{d}x/\mathrm{d}t=(\Omega_d-\Omega)/\beta$
[px/s] ($\approx -\Omega/\beta$ since $\Omega_d \ll \Omega$). The rotor emits event bursts at the
blade-pass frequency $f$, i.e. at times $\tau_k=\tau_0+k/f$, so burst $k$ lands at
\[
  x_k = x_c + v\,\tau_k \qquad\Longrightarrow\qquad \Delta x = x_{k+1}-x_k = \frac{v}{f} .
\]

\paragraph{The measurement.} Two facts fall straight out and are the whole algorithm:
\[
  \boxed{\; f = \frac{|v|}{\Delta x} \;}
  \qquad\text{and}\qquad
  \Omega_d = \Omega - \beta\,v .
\]
The spin turns a hard 80--800\,Hz \emph{temporal} measurement into an easy ${\sim}10\,$px
\emph{spatial} one. GottLUX measures it cheaply: a robust line fit of the event cloud $x$ vs
$t$ gives the sweep velocity $v$; a one-dimensional histogram of $x$ along the sweep direction
is autocorrelated, and the comb step $\Delta x$ is the lag whose harmonic energy (peaks at
$\Delta x$, $2\Delta x$, $3\Delta x$) is greatest and whose implied $f=|v|/\Delta x$ lies in the
rotor band. The normalized comb peak is the \emph{comb strength}; a detection requires an
in-band, harmonically structured comb riding on a coherent linear drift.

\paragraph{Recurrence (relative motion).} The ladder \emph{recurs} every revolution. A
stationary drone repeats an identical ladder; a moving one shifts by a fixed azimuth offset
$\Delta\Theta$ per revolution, giving its angular rate
$\Omega_d \approx \Delta\Theta / T_{\mathrm{rot}}$.
"""


def _fig_block(fname, caption, label, width=r"0.9\textwidth") -> str:
    return ("\\begin{figure}[H]\n\\centering\n"
            "\\includegraphics[width=" + width + "]{" + fname + "}\n"
            "\\caption{" + caption + "}\n"
            "\\label{fig:" + label + "}\n"
            "\\end{figure}")


def _summary_lines(result, track, band) -> list:
    if result is None:
        return [r"No rotor-ladder measurement was computed (insufficient events in the box)."]
    out = []
    verdict = (r"a \textbf{rotor ladder was detected}" if result.detected
               else r"\textbf{no clear rotor ladder} was found")
    band_s = (f"{band[0]:g}--{band[1]:g}\\,Hz" if band else "the rotor band")
    out.append("In the analysed box (" + str(result.n_events) + " events), " + verdict + ".")
    if result.blade_hz:
        out.append(r"The measured blade-pass frequency is $f=" + f"{result.blade_hz:g}"
                   + r"\,\mathrm{Hz}$, from a comb step $\Delta x=" + f"{result.step_px:g}"
                   + r"\,\mathrm{px}$ and a sweep velocity $v=" + f"{result.drift_px_s:g}"
                   + r"\,\mathrm{px/s}$ (comb strength " + f"{result.comb_strength:g}"
                   + ", search band " + band_s + ").")
    if track is not None and getattr(track, "n_passes", 0):
        out.append("Across " + str(track.n_passes) + " revolutions the blade frequency was stable ("
                   + f"{track.blade_hz_stability:g}" + r"), with a per-revolution azimuth offset of "
                   + f"{track.azimuth_offset_per_rev_px:g}" + r"\,px (relative motion).")
    return out


def _table_lines(result, track, band) -> list:
    d = result.as_dict()
    rows = [("Detected", _fmt(d.get("detected"))),
            ("In band", _fmt(d.get("in_band"))),
            (r"Blade-pass frequency $f$ (Hz)", _fmt(d.get("blade_hz"))),
            (r"Comb step $\Delta x$ (px)", _fmt(d.get("step_px"))),
            (r"Sweep velocity $v$ (px/s)", _fmt(d.get("drift_px_s"))),
            ("Comb strength", _fmt(d.get("comb_strength"))),
            ("Gappiness", _fmt(d.get("gappiness"))),
            ("Score", _fmt(d.get("score"))),
            ("Events in box", _fmt(d.get("n_events")))]
    if band:
        rows.append(("Search band (Hz)", f"{band[0]:g}--{band[1]:g}"))
    if track is not None and getattr(track, "n_passes", 0):
        rows += [("Revolutions", _fmt(track.n_passes)),
                 (r"Median $f$ across revs (Hz)", _fmt(track.median_blade_hz)),
                 ("Frequency stability", _fmt(track.blade_hz_stability)),
                 ("Azimuth offset / rev (px)", _fmt(track.azimuth_offset_per_rev_px)),
                 ("Confidence", _fmt(track.confidence))]
    out = [r"\begin{table}[H]", r"\centering", r"\begin{tabular}{ll}", r"\toprule",
           r"Quantity & Value \\", r"\midrule"]
    out += [k + " & " + v + r" \\" for k, v in rows]
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Rotor-ladder measured quantities.}", r"\label{tab:ladder}", r"\end{table}"]
    return out


def _build_latex(scene_files, ladder_fn, spec_fn, result, track, meta, band, title) -> str:
    name = meta.get("recording") or meta.get("name") or "recording"
    date = meta.get("date") or datetime.date.today().isoformat()
    L = [r"\documentclass[11pt]{article}",
         r"\usepackage[margin=1in]{geometry}",
         r"\usepackage{graphicx}",
         r"\usepackage{amsmath}",
         r"\usepackage{booktabs}",
         r"\usepackage{float}",
         r"\usepackage[hidelinks]{hyperref}",
         r"\graphicspath{{./}}",
         r"\title{Rotor-ladder study: " + _latex_escape(name) + "}",
         r"\author{GottLUX}",
         r"\date{" + _latex_escape(date) + "}",
         r"\begin{document}", r"\maketitle",
         r"\section{Summary}"]
    L += _summary_lines(result, track, band)
    if meta.get("sensor_px") or meta.get("window_s"):
        bits = []
        if meta.get("sensor_px"):
            bits.append("sensor " + _latex_escape(meta["sensor_px"]) + " px")
        if meta.get("window_s"):
            w = meta["window_s"]
            bits.append(f"window [{w[0]:g}, {w[1]:g}]\\,s")
        L.append("\\par\\smallskip\\noindent\\emph{Context:} " + " $\\cdot$ ".join(bits) + ".")
    L.append(_ALGO)
    if scene_files:
        L.append(r"\section{Scene --- 3-D space-time event cloud}")
        L.append("The accumulated event scene rendered from several camera angles; the rotor's "
                 "swept burst structure (the ladder) appears as the diagonal comb in the cloud.")
        for label, fn in scene_files:
            L.append(_fig_block(fn, "3-D space-time scene --- " + _latex_escape(label) + " view.",
                                "scene-" + _slug(label)))
    if ladder_fn or spec_fn:
        L.append(r"\section{Measurements and plots}")
        if ladder_fn:
            L.append(_fig_block(
                ladder_fn,
                r"Rotor-ladder measurement: (left) the $(t,x)$ event cloud with the fitted sweep "
                r"drift $v$; (right) the autocorrelation of the sweep-coordinate histogram, whose "
                r"first peak is the comb step $\Delta x$. The implied blade-pass frequency is "
                r"$f=|v|/\Delta x$.", "ladder"))
        if spec_fn:
            L.append(_fig_block(
                spec_fn,
                r"Region (box) temporal power spectrum with the rotor band shaded and the in-band "
                r"peak marked --- the direct temporal view of the same tone the ladder recovers "
                r"spatially.", "spectrum"))
    if result is not None:
        L.append(r"\section{Measured quantities}")
        L += _table_lines(result, track, band)
    L.append(r"\end{document}")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------- the public entry point
def save_ladder_study(out_dir, *, scenes=None, x=None, t=None, result=None, spectrum=None,
                      track=None, meta=None, band=None, dpi=300, title=None) -> list:
    """Write a full rotor-ladder study bundle into *out_dir*; return the paths written.

    Parameters
    ----------
    out_dir : str            destination folder (created if missing).
    scenes : dict | None     ``{angle_label: RGB ndarray (H,W,3) uint8}`` — full-res scene renders.
    x, t : array | None      the swept events (sweep column px, time s) the ladder is measured on.
    result : LadderResult | None   computed from *x, t* if not supplied.
    spectrum : Spectrum | None     optional region spectrum for the FFT plot.
    track : LadderTrack | None     optional cross-revolution recurrence result.
    meta : dict | None       context (recording, sensor_px, window_s, …) for the report header.
    band : (lo, hi) | None   rotor search band (Hz), for the figures/table/text.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(meta or {})
    lo, hi = (band if band else (80.0, 800.0))
    written = []

    # 1) full-resolution scene PNGs (one per camera angle)
    scene_files = []
    for label, rgb in (scenes or {}).items():
        fn = "scene-" + _slug(label) + ".png"
        _save_png(rgb, os.path.join(out_dir, fn))
        scene_files.append((label, fn)); written.append(os.path.join(out_dir, fn))

    # 2) the rotor-ladder measurement figure
    if result is None and x is not None and t is not None:
        result = ladder_signature(np.asarray(x, float), np.asarray(t, float), f_lo=lo, f_hi=hi)
    ladder_fn = None
    if x is not None and t is not None and result is not None:
        fig = ladder_figure(np.asarray(x, float), np.asarray(t, float), result, title=title)
        written += export.save_figure(fig, os.path.join(out_dir, "rotor-ladder"), dpi=dpi,
                                      formats=("png", "pdf"), close=True)
        ladder_fn = "rotor-ladder.png"

    # 3) the region-spectrum figure
    spec_fn = None
    if spectrum is not None and getattr(spectrum, "freqs", None) is not None \
            and np.size(spectrum.freqs):
        fig = _spectrum_figure(spectrum, band=(lo, hi))
        written += export.save_figure(fig, os.path.join(out_dir, "spectrum"), dpi=dpi,
                                      formats=("png", "pdf"), close=True)
        spec_fn = "spectrum.png"

    # 4) the measurements table (JSON + CSV)
    measurements = {"meta": meta, "band_hz": [lo, hi],
                    "ladder": (result.as_dict() if result is not None else None),
                    "track": (track.__dict__ if track is not None else None)}
    written += export.save_json(measurements, os.path.join(out_dir, "measurements.json"))
    if result is not None:
        flat = {k: [v] for k, v in result.as_dict().items() if not isinstance(v, (list, dict))}
        written += export.save_table(flat, os.path.join(out_dir, "measurements"))

    # 5) the labelled, compilable LaTeX report
    tex_path = os.path.join(out_dir, "rotor-ladder-report.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(_build_latex(scene_files, ladder_fn, spec_fn, result, track, meta, (lo, hi), title))
    written.append(tex_path)
    return written


# ====================================================================================
# The 360° survey report (rotor_scan): classify a target, then map every bearing it recurs at.
# ====================================================================================
def _scan_summary_lines(result, meta) -> list:
    sig = result.template_signature
    out = []
    if result.f_template_hz:
        verdict = r"a \textbf{rotor signature was found and keyed}"
    else:
        verdict = r"\textbf{no rotor ladder} was found to key the survey on"
    out.append("Across the rotation, " + verdict + ".")
    if sig:
        out.append(r"The template blade-pass frequency is $f=" + f"{sig.blade_hz:g}"
                   + r"\,\mathrm{Hz}$, i.e. a rotor rate of $" + f"{sig.rotor_hz:g}"
                   + r"\,\mathrm{Hz}$ (" + f"{sig.rpm:.0f}" + r"\,RPM) for an assumed "
                   + f"{sig.n_blades}" + r"-blade, " + f"{sig.prop_diameter_m*1000:.0f}"
                   + r"\,mm prop --- a tip speed of $" + f"{sig.tip_speed_mps:g}"
                   + r"\,\mathrm{m/s}$ (Mach " + f"{sig.tip_mach:g}" + ").")
    matched = result.matched
    if matched:
        brg = sorted(round(d.bearing_deg, 1) for d in matched)
        out.append("The same signature recurs at " + str(len(matched))
                   + r" bearings spanning $" + f"{min(brg):g}" + r"^\circ$ to $"
                   + f"{max(brg):g}" + r"^\circ$, over " + str(len(set(d.rev for d in matched)))
                   + " revolutions.")
    for tr in result.tracks:
        motion = ("stationary" if abs(tr.omega_deg_s) < 0.5
                  else f"moving at {tr.omega_deg_s:+g} deg/s")
        rng = (f", range $\\approx {tr.range_m:g}$ m" if tr.range_m else "")
        out.append(r"\emph{Track:} a rotor at bearing $" + f"{tr.bearing_deg:g}"
                   + r"^\circ$" + rng + r", recurring over " + f"{tr.n_passes}"
                   + r" revolutions with a per-revolution azimuth offset of $"
                   + f"{tr.bearing_offset_per_rev_deg:+g}" + r"^\circ$ --- " + motion
                   + r" (frequency stability " + f"{tr.blade_hz_stability:g}" + ").")
    return out


def _prop_table(result) -> list:
    sig = result.template_signature
    rows = [(r"Blade-pass frequency $f$ (Hz)", _fmt(sig.blade_hz if sig else None)),
            ("Blades per rotor (assumed)", _fmt(sig.n_blades if sig else None)),
            (r"Rotor rate $f_\mathrm{rot}=f/N$ (Hz)", _fmt(sig.rotor_hz if sig else None)),
            ("Rotor rate (RPM)", _fmt(sig.rpm if sig else None)),
            (r"Prop diameter $D$ (mm)", _fmt(sig.prop_diameter_m * 1000 if sig else None)),
            (r"Tip speed $\pi D f_\mathrm{rot}$ (m/s)", _fmt(sig.tip_speed_mps if sig else None)),
            ("Tip Mach", _fmt(sig.tip_mach if sig else None)),
            (r"Template bearing ($^\circ$)", _fmt(result.template_bearing_deg)),
            ("Template range (m)", _fmt(result.template_range_m))]
    out = [r"\begin{table}[H]", r"\centering", r"\begin{tabular}{ll}", r"\toprule",
           r"Propeller quantity & Value \\", r"\midrule"]
    out += [k + " & " + v + r" \\" for k, v in rows]
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Propeller signature extracted from the analysis box (template).}",
            r"\label{tab:prop}", r"\end{table}"]
    return out


def _tracks_table(result) -> list:
    if not result.tracks:
        return [r"\emph{No track recurred across $\geq 2$ revolutions.}"]
    out = [r"\begin{table}[H]", r"\centering", r"\begin{tabular}{rrrrrr}", r"\toprule",
           r"Bearing ($^\circ$) & Passes & $f$ (Hz) & Stab. & Offset ($^\circ$/rev) & "
           r"$\Omega_d$ ($^\circ$/s) \\", r"\midrule"]
    for tr in result.tracks:
        out.append(f"{tr.bearing_deg:g} & {tr.n_passes} & {_fmt(tr.median_blade_hz)} & "
                   f"{tr.blade_hz_stability:g} & {tr.bearing_offset_per_rev_deg:+g} & "
                   f"{tr.omega_deg_s:+g} " + r"\\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Cross-revolution rotor tracks. The per-revolution azimuth offset is the "
            r"target's relative motion; $\Omega_d=$ offset$/T_\mathrm{rot}$.}",
            r"\label{tab:tracks}", r"\end{table}"]
    return out


_SCAN_INTRO = r"""
\section{What this study does}
A single analysis box is placed on the target and its rotor-ladder signature is measured (the
blade-pass frequency $f$, recovered from the comb spacing). That frequency is then used as a
\emph{matched key}: the entire $360^\circ$ revolution is de-rotated to a world-azimuth frame and
swept, testing every populated $(\text{revolution},\text{azimuth})$ cell for the same comb. Cells
whose implied $f$ lands within tolerance of the template are reported as the same rotor, giving a
map of \emph{where else} the signature occurs, the target's bearing and range at each, and --- by
linking the recurrences across revolutions --- its per-revolution azimuth offset (relative motion).
"""


def _build_scan_latex(result, figs, meta, title) -> str:
    name = meta.get("recording") or meta.get("name") or "recording"
    date = meta.get("date") or datetime.date.today().isoformat()
    L = [r"\documentclass[11pt]{article}",
         r"\usepackage[margin=1in]{geometry}", r"\usepackage{graphicx}",
         r"\usepackage{amsmath}", r"\usepackage{booktabs}", r"\usepackage{float}",
         r"\usepackage[hidelinks]{hyperref}", r"\graphicspath{{./}}",
         r"\title{Rotor-ladder 360\textdegree{} survey: " + _latex_escape(name) + "}",
         r"\author{GottLUX}", r"\date{" + _latex_escape(date) + "}",
         r"\begin{document}", r"\maketitle", r"\section{Summary}"]
    L += _scan_summary_lines(result, meta)
    if meta.get("sensor_px") or meta.get("t_rot_s") or meta.get("fov_deg"):
        bits = []
        if meta.get("sensor_px"):
            bits.append("sensor " + _latex_escape(meta["sensor_px"]) + " px")
        if result.fov_deg:
            bits.append(f"FOV {result.fov_deg:g}\\textdegree")
        if result.t_rot_s:
            bits.append(f"$T_{{\\mathrm{{rot}}}}={result.t_rot_s:g}$\\,s")
        if result.sweep_px_s:
            bits.append(f"sweep $|v|={abs(result.sweep_px_s):g}$\\,px/s")
        L.append(r"\par\smallskip\noindent\emph{Context:} " + " $\\cdot$ ".join(bits) + ".")
    L.append(_SCAN_INTRO)
    L.append(_ALGO)
    L.append(r"\section{Propeller signature}")
    L.append("The propeller quantities derived from the measured blade-pass frequency (rotor rate, "
             "RPM and tip speed follow from the assumed blade count and prop diameter; ranging uses "
             "the known target size).")
    L += _prop_table(result)
    if result.template_signature is not None and figs.get("ladder"):
        L.append(_fig_block(figs["ladder"],
                 r"Template rotor-ladder measurement on the analysis box: (left) the $(t,x)$ event "
                 r"cloud with the fitted sweep drift $v$; (right) the comb autocorrelation whose "
                 r"first peak is the rung spacing $\Delta x$. The blade-pass frequency is "
                 r"$f=|v|/\Delta x$.", "ladder"))
    if figs.get("scan"):
        L.append(r"\section{The 360\textdegree{} survey}")
        L.append("Every scanned cell's comb is plotted against its world bearing; the shaded band is "
                 "the template frequency $\\pm$ tolerance. Filled points are cells matching the "
                 "template rotor (sized by comb strength); open points are other in-band combs.")
        L.append(_fig_block(figs["scan"], r"Rotor-ladder $360^\circ$ survey: blade-pass frequency "
                 r"vs world bearing for every scanned cell.", "scan"))
    if figs.get("radar"):
        L.append(r"\section{Range and bearing}")
        L.append("Projecting the matched detections onto a polar map gives a target-acquisition "
                 "picture: bearing from the de-rotated azimuth, range from the pinhole model on the "
                 "known target size, with each track's bearing march drawn in.")
        L.append(_fig_block(figs["radar"], r"Target-acquisition radar of the matched rotor "
                 r"detections: $\theta=$ bearing, $r=$ range (m), colour $=$ blade frequency.",
                 "radar", width=r"0.7\textwidth"))
    if figs.get("recur"):
        L.append(r"\section{Cross-revolution recurrence and relative motion}")
        L.append("A rotor that recurs at the same bearing each revolution is stationary; one whose "
                 "bearing marches by a fixed offset per revolution is moving, and that offset over "
                 "the rotation period is its relative angular rate --- the \\emph{unique offset from "
                 "the spin}.")
        L.append(_fig_block(figs["recur"], r"Bearing vs revolution for each linked track (left, with "
                 r"the fitted per-revolution offset) and blade-frequency stability (right).",
                 "recur"))
    L += _tracks_table(result)
    # reproduce
    cmd = meta.get("reproduce")
    if cmd:
        L.append(r"\section{Reproduce}")
        L.append(r"\begin{verbatim}" + "\n" + cmd + "\n" + r"\end{verbatim}")
    # optional verification-against-truth block
    truth = meta.get("truth")
    if truth:
        L.append(r"\section{Verification against ground truth}")
        L.append("This study was generated on a synthetic rotating scene with a fully known "
                 "target, to validate recovery before processing real data.")
        L += _truth_table(truth, result)
    L.append(r"\end{document}")
    return "\n".join(L) + "\n"


def _truth_table(truth, result) -> list:
    tr = result.tracks[0] if result.tracks else None
    rows = [("Quantity", "Truth", "Recovered"),
            (r"Blade-pass $f$ (Hz)", _fmt(truth.get("blade_hz")), _fmt(result.f_template_hz)),
            ("Rotor RPM", _fmt(truth.get("rpm")),
             _fmt(result.template_signature.rpm if result.template_signature else None)),
            (r"Range (m)", _fmt(truth.get("range_m")), _fmt(tr.range_m if tr else None)),
            (r"Bearing offset ($^\circ$/rev)", _fmt(truth.get("drift_deg_per_rev")),
             _fmt(tr.bearing_offset_per_rev_deg if tr else None)),
            (r"Relative rate $\Omega_d$ ($^\circ$/s)", _fmt(truth.get("omega_d_deg_s")),
             _fmt(tr.omega_deg_s if tr else None))]
    out = [r"\begin{table}[H]", r"\centering", r"\begin{tabular}{lrr}", r"\toprule",
           rows[0][0] + " & " + rows[0][1] + " & " + rows[0][2] + r" \\", r"\midrule"]
    out += [k + " & " + a + " & " + b + r" \\" for k, a, b in rows[1:]]
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Recovered rotor quantities vs planted ground truth.}",
            r"\label{tab:truth}", r"\end{table}"]
    return out


def save_scan_report(out_dir, result, *, cfg=None, meta=None, title=None, dpi=300) -> list:
    """Write the full 360° rotor-ladder survey bundle into *out_dir*; return the paths written.

    Figures (survey map, radar, recurrence, template ladder) + JSON + detections/tracks CSV + a
    self-contained, compilable LaTeX report (theory + the figures + tables + a reproduce command,
    and a ground-truth comparison if ``meta['truth']`` is supplied).
    """
    from gottlux.rotation import rotor_scan as rs
    from gottlux.rotation.rotor_ladder import ladder_figure
    from gottlux.rotation.viz import rotor_ladder_viz as viz
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(meta or {})
    written, figs = [], {}

    def _fig(builder, stem, key):
        try:
            fig = builder()
            paths = export.save_figure(fig, os.path.join(out_dir, stem), dpi=dpi,
                                       formats=("png", "pdf"), close=True)
            written.extend(paths)
            figs[key] = stem + ".png"
        except Exception as e:                       # one bad figure must not sink the report
            print(f"[gottlux] rotor-ladder figure '{stem}' failed: {e}")

    _fig(lambda: viz.scan_map_figure(result), "rotor_ladder_scan", "scan")
    _fig(lambda: viz.radar_ladder_figure(result), "rotor_ladder_radar", "radar")
    _fig(lambda: viz.recurrence_figure(result), "rotor_ladder_recurrence", "recur")
    if result.template_events is not None and result.template is not None:
        x0, t0 = result.template_events
        _fig(lambda: ladder_figure(x0, t0, result.template,
                                   title=f"Template — f={result.f_template_hz:g} Hz"),
             "rotor_ladder_signature", "ladder")

    written += export.save_json(result.as_dict(), os.path.join(out_dir, "rotor_ladder.json"))
    written += export.save_table(rs.detections_table(result),
                                 os.path.join(out_dir, "rotor_ladder_detections"))
    if result.tracks:
        written += export.save_table(rs.tracks_table(result),
                                     os.path.join(out_dir, "rotor_ladder_tracks"))

    tex_path = os.path.join(out_dir, "rotor-ladder-360-report.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(_build_scan_latex(result, figs, meta, title))
    written.append(tex_path)
    return written
