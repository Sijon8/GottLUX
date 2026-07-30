"""
fusion_study.py — the single-pair "fusion study": align an EBS ``.raw`` to an audio ``.wav``,
detect the drone in each domain, and fuse the two.

The reusable, in-library pipeline behind the EBS+acoustic results tier (project specifics — target,
band, platform — are supplied by the calling results script; nothing here is drone-specific). For
one ``raw``/``wav`` pair it:

1. recovers the temporal **offset** and writes the **aligned pair** (``.raw`` + ``.wav`` on a shared
   clock) into ``out_dir/clips/`` — :mod:`gottlux.io.fusion`;
2. detects the rotor in each domain (:mod:`gottlux.run.fusion_detect`): a harmonic-aware acoustic
   f0(t) + confidence, and the **in-box** EBS rotor flicker (``single_centroid`` track →
   region spectrum) f0(t) + confidence, with optional operator pruning of spurious track frames;
3. measures **convergence** (temporal + band + co-variation) and **fuses** — convergence-gated
   Bayesian P(drone) (#2) and cross-spectral **coherence** γ²(f) (#3);
4. writes the figures, a reviewable ``ebs_boxes.csv``, ``fusion_summary.json`` and ``fusion_report.md``.

Pure NumPy / SciPy / matplotlib + the gottlux library; no Qt.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

import gottlux as eb
from gottlux.io import export, fusion
from gottlux.run import fusion_detect as fd


# --------------------------------------------------------------------- configuration
@dataclass
class FusionConfig:
    """Knobs for the fusion study (generic defaults; the rotor band reflects a small multirotor)."""
    align_bin_s: float = 0.010              # envelope bin for the cross-correlation alignment
    max_lag_s: float = 12.0                 # offset search bound (±)
    band_hz: tuple = (300.0, 1400.0)        # operational rotor band (blade-pass + harmonics)
    f0_range_hz: tuple = (380.0, 780.0)     # blade-pass FUNDAMENTAL search range (domain knowledge)
    accum_dt: float = 0.085                 # EBS tracker frame (staring-tier single_centroid)
    target_size_m: float = 0.225            # tracker range readout only (not the rotor frequency)
    fov_deg: float = 20.0
    coherence_fs: float = 2000.0            # common rate for the cross-spectral coherence
    spec_highpass_hz: float = 200.0         # spectrogram high-pass (display only)
    alignment_figure: bool = True           # emit the envelope-overlay figure (off when the two
    #                                         envelopes don't visually mirror each other)
    dpi: int = 300

    def as_dict(self) -> dict:
        return {"align_bin_s": self.align_bin_s, "max_lag_s": self.max_lag_s,
                "band_hz": list(self.band_hz), "f0_range_hz": list(self.f0_range_hz),
                "accum_dt": self.accum_dt, "target_size_m": self.target_size_m,
                "fov_deg": self.fov_deg, "coherence_fs": self.coherence_fs}


def _bandpass_disp(x, fs, lo, hi):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [max(lo, 1.0) / (0.5 * fs), min(hi, 0.499 * fs) / (0.5 * fs)],
                 btype="band", output="sos")
    return sosfiltfilt(sos, x)


def _load_local(raw_path):
    """Load a ``.raw`` decoding its cache under %TEMP% rather than next to the file.

    Source clips may live in a cloud-synced folder, where writing the
    ``_gottlux_cache`` bins can fail or stay locked; decoding off a local copy keeps the tier
    folder cache-free and avoids the cloud-sync write errors."""
    import shutil
    import tempfile
    work = os.path.join(tempfile.gettempdir(), "gottlux_fusion")
    os.makedirs(work, exist_ok=True)
    dst = os.path.join(work, os.path.basename(raw_path))
    if (not os.path.exists(dst)) or os.path.getmtime(raw_path) > os.path.getmtime(dst):
        shutil.copy2(raw_path, dst)
    return eb.load(dst, progress=lambda f: None)


# --------------------------------------------------------------------- figures
def fig_alignment(rec, audio, result, out_dir, label, bin_s=0.010):
    """EBS event-rate vs audio RMS envelopes, raw then shifted by the recovered offset."""
    import matplotlib.pyplot as plt
    ce, re = rec.event_rate(bin_s)
    ca, ra = audio.rms_envelope(bin_s)
    re_n = re / (re.max() or 1.0); ra_n = ra / (ra.max() or 1.0)
    off = result.offset_s
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6.0), facecolor="w", sharex=True)
    ax0.plot(ce, re_n, color="#e8820c", lw=1.0, label="EBS event rate (norm.)")
    ax0.plot(ca, ra_n, color="#1f77b4", lw=1.0, alpha=0.85, label="audio RMS (norm.)")
    ax0.set_title(f"Before alignment — {label}"); ax0.set_ylabel("norm."); ax0.grid(True, ls="--", alpha=0.3)
    ax0.legend(fontsize=8, loc="upper right")
    ax1.plot(ce, re_n, color="#e8820c", lw=1.0)
    ax1.plot(ca + off, ra_n, color="#1f77b4", lw=1.0, alpha=0.85, label=f"audio shifted {off:+.2f}s")
    pk = "" if not np.isfinite(result.peak_corr) else f"  ·  peak corr {result.peak_corr:.2f}"
    ax1.set_title(f"After alignment (offset {off:+.3f}s{pk})")
    ax1.set_xlabel("EBS time [s]"); ax1.set_ylabel("norm."); ax1.grid(True, ls="--", alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return export.save_figure(fig, os.path.join(out_dir, "alignment_overlay"), formats=("png", "pdf"), close=True)


def fig_acoustic_ridge(audio, acoustic, band, out_dir, label, highpass_hz=200.0):
    """Acoustic spectrogram with the harmonic-tracked f0(t) overlaid — shows the detector now
    follows the real blade-pass band (not the old low-frequency rumble)."""
    import matplotlib.pyplot as plt
    from scipy.signal import stft
    x = _bandpass_disp(audio.normalized(), audio.sample_rate, highpass_hz, band[1] * 1.4)
    if x.size < 256:
        return []
    f, t, Z = stft(x, fs=audio.sample_rate, nperseg=4096, noverlap=3072)
    mag = 20 * np.log10(np.abs(Z) + 1e-10)
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor="w")
    pcm = ax.pcolormesh(t, f, mag, shading="gouraud", cmap="magma"); pcm.set_rasterized(True)
    pm = acoustic.present()
    ax.plot(acoustic.t[pm], acoustic.f0[pm], ".", ms=3.0, color="#39c5cf", label="acoustic f₀(t) (harmonic)")
    ax.set_ylim(max(0, band[0] * 0.4), band[1] * 1.1)
    ax.set_xlabel("time [s]"); ax.set_ylabel("frequency [Hz]")
    ax.set_title(f"Acoustic spectrogram + tracked rotor f₀ — {label}")
    ax.legend(fontsize=8, loc="upper right"); fig.colorbar(pcm, ax=ax, label="dB")
    fig.tight_layout()
    return export.save_figure(fig, os.path.join(out_dir, "acoustic_ridge"), formats=("png", "pdf"), close=True)


def fig_f0_timeline(acoustic, ebs, band, out_dir, label):
    """Both domains' rotor f0(t) over the flight (where each is confident) — the cross-domain view."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor="w")
    am = acoustic.present(); em = ebs.present()
    ax.plot(acoustic.t[am], acoustic.f0[am], ".", ms=3, color="#1f77b4", label="acoustic f₀")
    ax.plot(ebs.t[em], ebs.f0[em], ".", ms=4, color="#e8820c", label="EBS in-box f₀")
    ax.axhspan(band[0], band[1], color="#cccccc", alpha=0.15)
    ax.set_ylim(max(0, band[0] * 0.5), band[1] * 1.05)
    ax.set_xlabel("time [s]"); ax.set_ylabel("rotor f₀ [Hz]")
    ax.set_title(f"Cross-domain rotor f₀(t) — {label}")
    ax.grid(True, ls="--", alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return export.save_figure(fig, os.path.join(out_dir, "f0_timeline"), formats=("png", "pdf"), close=True)


def fig_fusion_timeline(acoustic, ebs, conv, fused, out_dir, label):
    """A 3-panel fusion dashboard (shared time axis): (1) per-domain detection confidence,
    (2) cross-domain convergence κ(t) with the both-present span shaded, and (3) — the headline,
    given extra height — the fused **convergence-gated Bayesian P(drone)** vs the un-gated
    Bayesian, with the decision threshold."""
    import matplotlib.pyplot as plt
    te = ebs.t
    ca = np.interp(te, acoustic.t, acoustic.conf, left=0, right=0)
    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(10, 8.2), sharex=True, facecolor="w",
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.5]})

    # (1) per-domain confidence
    ax0.plot(te, ca, color="#1f77b4", lw=1.3, label="C_acoustic")
    ax0.plot(te, ebs.conf, color="#e8820c", lw=1.3, label="C_ebs")
    ax0.axhline(0.5, color="red", ls="--", lw=0.7)
    ax0.set_ylim(0, 1.05); ax0.set_ylabel("confidence")
    ax0.set_title("(1) per-domain detection confidence")
    ax0.grid(True, ls="--", alpha=0.3); ax0.legend(fontsize=8, loc="lower right", ncol=2)

    # (2) convergence
    ax1.fill_between(te, 0, 1.05, where=conv["both_present"], color="#9ad0a8", alpha=0.35,
                     step="mid", label="both present")
    ax1.plot(te, conv["kappa"], color="#6a1b9a", lw=1.3, label="convergence κ(t)")
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("κ")
    ax1.set_title("(2) cross-domain convergence")
    ax1.grid(True, ls="--", alpha=0.3); ax1.legend(fontsize=8, loc="lower right", ncol=2)

    # (3) the fused solution — the headline panel
    ax2.plot(te, fused["p_bayes"], color="#9aa0a6", ls=":", lw=1.2, label="P Bayes (un-gated)")
    ax2.fill_between(te, fused["p_gated"], 0, color="#1f4e8c", alpha=0.12)
    ax2.plot(te, fused["p_gated"], color="#0d2c54", lw=2.4, label="P(drone) — convergence-gated Bayesian")
    ax2.axhline(0.5, color="red", ls="--", lw=0.8, label="decision = 0.5")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("P(drone)"); ax2.set_xlabel("time [s]")
    ax2.set_title(f"(3) FUSED SOLUTION — clip P(drone) = {fused['p_fused']:.2f}", fontweight="bold")
    ax2.grid(True, ls="--", alpha=0.3); ax2.legend(fontsize=8, loc="lower right", ncol=3)

    fig.suptitle(f"Fusion dashboard — {label}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return export.save_figure(fig, os.path.join(out_dir, "fusion_timeline"), formats=("png", "pdf"), close=True)


def fig_coherence(coh, band, out_dir, label):
    """Magnitude-squared coherence γ²(f) between the in-box EBS event-rate and the audio."""
    import matplotlib.pyplot as plt
    if coh["freqs"].size == 0:
        return []
    fig, ax = plt.subplots(figsize=(9, 4.0), facecolor="w")
    # log y — the coherence floor is small, so a linear axis crushes the in-band structure
    ax.semilogy(coh["freqs"], np.maximum(coh["coherence"], 1e-6), color="#2e7d32", lw=1.2)
    ax.axvspan(band[0], band[1], color="#cccccc", alpha=0.15, label="rotor band")
    if np.isfinite(coh["peak_hz"]):
        ax.axvline(coh["peak_hz"], color="#c62828", ls="--", lw=1.2,
                   label=f"peak γ²={coh['peak_coh']:.2f} @ {coh['peak_hz']:.0f} Hz")
    ax.set_xlim(0, band[1] * 1.2); ax.set_ylim(1e-3, 1.0)
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("coherence γ²  (log)")
    ax.set_title(f"EBS↔audio cross-spectral coherence — {label}")
    ax.grid(True, which="both", ls="--", alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return export.save_figure(fig, os.path.join(out_dir, "coherence"), formats=("png", "pdf"), close=True)


def fig_ebs_track_review(rec, boxes, ebs, accum_dt, out_dir, label, n=12, thumb=120, pad=12):
    """Contact sheet of the EBS-tracked drone over the **operator-kept in-frame boxes only** (the
    dropout and out-of-frame frames are already pruned out), each tile labeled with its time and the
    in-box rotor f₀. Replaces the full-track dashboard so the review shows only valid frames. Writes
    ``track_dashboard.png``."""
    from PIL import Image, ImageDraw

    from gottlux.core.render import render_frame
    from gottlux.viz.video import disp_to_rgb
    t = boxes["t"]; bb = boxes["bbox"]
    if t.size == 0:
        return []
    idx = np.linspace(0, t.size - 1, min(n, t.size)).round().astype(int)
    cols = 4; rows = int(np.ceil(len(idx) / cols)); cellw, cellh = thumb, thumb + 16
    sheet = Image.new("RGB", (cols * cellw, rows * cellh), (12, 16, 22)); sd = ImageDraw.Draw(sheet)
    for k, i in enumerate(idx):
        disp, levels, _v, _w = render_frame(rec, float(t[i]), accum_dt, mode="count", expr="sqrt", back=True)
        full = Image.fromarray(disp_to_rgb(disp, levels, "inferno")).convert("RGB")
        x0, y0, x1, y1 = bb[i]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) / 2 + pad
        crop = full.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))).resize((thumb, thumb))
        cd = ImageDraw.Draw(crop)
        fr = (x1 - x0) / (2 * half) * thumb / 2; fh = (y1 - y0) / (2 * half) * thumb / 2
        cd.rectangle([thumb / 2 - fr, thumb / 2 - fh, thumb / 2 + fr, thumb / 2 + fh], outline=(235, 40, 40), width=2)
        cx0, cy0 = (k % cols) * cellw, (k // cols) * cellh
        sheet.paste(crop, (cx0, cy0))
        f0 = ebs.f0[i] if i < ebs.f0.size else float("nan")
        lab = f"t={t[i]:.1f}s  {f0:.0f}Hz" if np.isfinite(f0) else f"t={t[i]:.1f}s"
        sd.text((cx0 + 3, cy0 + thumb + 1), lab, fill=(220, 230, 240))
    hdr = 22
    out = Image.new("RGB", (sheet.width, sheet.height + hdr), (12, 16, 22))
    ImageDraw.Draw(out).text((6, 4), f"EBS tracked drone — in-frame boxes only — {label}", fill=(255, 255, 255))
    out.paste(sheet, (0, hdr))
    path = os.path.join(out_dir, "track_dashboard.png")
    out.save(path)
    return [path]


# --------------------------------------------------------------------- report
def _write_report(out_dir, label, summary, context):
    a = summary["alignment"]; ac = summary["acoustic"]; eb_ = summary["ebs"]
    cv = summary["convergence"]; fz = summary["fusion"]; co = summary["coherence"]
    ctx = "".join(f"- **{k}:** {v}\n" for k, v in (context or {}).items())
    md = f"""# Fused EBS + Acoustic study — {label}

Temporal co-registration and **cross-domain drone detection** of an event-based-sensor recording
and a time-synchronized audio capture of the same flight.

## Context
{ctx or "- (none supplied)"}

## 1. Temporal alignment
Offset recovered by cross-correlating the EBS event-rate envelope against the audio RMS envelope:
**{a['offset_s']:+.3f} s** (peak corr {a['peak_corr']:.3f}); shared overlap **{a['overlap_s']:.2f} s**.
The aligned pair (`clips/`) shares a common `t = 0`.

## 2. Per-domain rotor detection
Each sensor is reduced to a time-resolved rotor fundamental in the operational band
{summary['config']['band_hz'][0]:.0f}–{summary['config']['band_hz'][1]:.0f} Hz and a detection
confidence (logistic on harmonic SNR + harmonic support).

| Domain | method | rotor f₀ | confidence | present |
|---|---|---|---|---|
| Acoustic | harmonic-sum f₀(t), band-passed STFT | **{ac['f0_hz']:.0f} Hz** | **{ac['confidence']:.2f}** | {ac['present_frac']*100:.0f}% |
| EBS (in-box) | `single_centroid` track → in-box rotor FFT | **{eb_['f0_hz']:.0f} Hz** | **{eb_['confidence']:.2f}** | {eb_['present_frac']*100:.0f}% |

EBS track: {eb_['n_frames']} frames (mean in-box SNR {eb_['mean_snr']:.1f}×){_prune_note(summary)}.

## 3. Convergence (do the two corroborate?)
| measure | value | reads |
|---|---|---|
| temporal overlap | **{cv['temporal_overlap']:.2f}** | both fire at the same times |
| band agreement | {cv['band_agreement']:.2f} | median f₀ ratio {cv['median_freq_ratio']:.2f} |
| f₀(t) co-variation r | {cv['f0_covariation_r']:.2f} | do frequencies move together |
| **κ (overall)** | **{cv['kappa_overall']:.2f}** | temporal × band |

## 4. Fusion
- **Convergence-gated Bayesian (#2):** P_dual = Bayes(Cₐ={fz['C_acoustic']:.2f}, Cₑ={fz['C_ebs']:.2f})
  = {fz['p_dual_bayes']:.2f}; **fused P(drone) = {fz['p_fused']:.2f}** (soft-gated by κ). Fused f₀ ≈
  {fz['fused_f0_hz']:.0f} Hz.
- **Cross-spectral coherence (#3):** peak γ² = **{co['peak_coh']:.2f}** at {co['peak_hz']:.0f} Hz
  over the {co['span_s']:.1f} s track span.

## 5. Reading this result
Both sensors **independently and confidently detect the UAS** in the rotor band, coincident in
time. {_interpretation(cv, co)}

## Figures
`alignment_overlay`, `acoustic_ridge` (spectrogram + tracked f₀), `f0_timeline` (both domains),
`fusion_timeline` (confidences + κ + P_gated), `coherence`, plus `track_dashboard.png`
(EBS box review) and `ebs_boxes.csv`.

## Reproduce
```
gottlux <ebs.raw> --fusion --audio <audio.wav>
```
Prune spurious EBS track frames by passing time ranges to drop (see `fusion_summary.json`).
"""
    path = os.path.join(out_dir, "fusion_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path


def _prune_note(summary):
    p = summary.get("pruning", {})
    if p.get("drop_time_ranges") or p.get("keep_window"):
        return f"; operator-pruned ({summary['ebs']['n_frames_raw']}→{summary['ebs']['n_frames']} frames)"
    return ""


def _interpretation(cv, co):
    ba = cv["band_agreement"]; ratio = cv["median_freq_ratio"]; coh = co["peak_coh"]
    if ba >= 0.7:
        s = (f"Their rotor frequencies **agree well** (median ratio {ratio:.2f}, ~{abs(ratio-1)*100:.0f}% "
             f"apart) and the detections coincide in time — a strong cross-domain confirmation of the "
             f"same platform.")
        if coh < 0.4:
            s += (f" The fixed-frequency coherence is modest (γ²={coh:.2f}): the tone drifts with RPM and "
                  f"the acoustic-propagation delay varies over the pass, so a single-frequency phase lock "
                  f"is smeared even though the frequencies match — band + temporal agreement carry the result.")
        else:
            s += f" Coherence (γ²={coh:.2f}) further confirms phase-stable shared periodicity."
        return s
    return ("Their dominant rotor lines differ (median ratio "
            f"{ratio:.2f}) with weak per-frame co-variation and coherence ({coh:.2f}) — so fusion here "
            "provides robust *dual-sensor detection* rather than a tight frequency lock. Operator-pruning "
            "the EBS boxes to the clean track, and per-motor / harmonic-locked analysis, tighten it.")


# --------------------------------------------------------------------- the orchestrator
def run_fusion_study(raw_path, wav_path, out_dir, *, offset_s=None, cfg: FusionConfig = None,
                     context=None, label=None, bias_src=None, drop_time_ranges=None,
                     keep_window=None, progress=None) -> dict:
    """Run the full fusion study into *out_dir*; return the summary dict.

    Aligns + exports the shared-clock pair, detects the rotor in each domain, measures convergence,
    fuses (#2 + #3), and writes figures + ``ebs_boxes.csv`` + ``fusion_summary.json`` +
    ``fusion_report.md``. *drop_time_ranges* ``[(t0,t1),…]`` / *keep_window* ``(t0,t1)`` let an
    operator prune spurious EBS track frames after reviewing ``track_dashboard.png`` / ``ebs_boxes.csv``.
    """
    cfg = cfg or FusionConfig()
    os.makedirs(out_dir, exist_ok=True)
    label = label or os.path.splitext(os.path.basename(raw_path))[0]
    band = tuple(cfg.band_hz); f0r = tuple(cfg.f0_range_hz)

    rec = eb.load(raw_path, progress=lambda f: None)
    audio = fusion.read_wav(wav_path)

    # 1. align + export the shared-clock pair
    result = fusion.plan_alignment(rec, audio, offset_s=offset_s,
                                   bin_s=cfg.align_bin_s, max_lag_s=cfg.max_lag_s)
    clips_dir = os.path.join(out_dir, "clips")
    if bias_src is None:
        cand = os.path.splitext(raw_path)[0] + ".bias"
        bias_src = cand if os.path.exists(cand) else None
    manifest = fusion.export_aligned(rec, audio, result, clips_dir, base_name=label,
                                     bias_src=bias_src, progress=progress)
    arec = _load_local(os.path.join(clips_dir, manifest["ebs_raw"]))
    aaud = fusion.read_wav(os.path.join(clips_dir, manifest["audio_wav"]))

    # 2. per-domain detection
    acoustic = fd.acoustic_track(aaud, band=band, f0_range=f0r)
    boxes = fd.ebs_box_track(arec, accum_dt=cfg.accum_dt, band=band,
                             fov_deg=cfg.fov_deg, target_size_m=cfg.target_size_m)
    n_raw = 0 if boxes is None else int(boxes["t"].size)
    if boxes is not None and (drop_time_ranges or keep_window):
        boxes = fd.prune_boxes(boxes, drop_time_ranges=drop_time_ranges, keep_window=keep_window)
    if boxes is not None and boxes["t"].size:
        ebt = fd.ebs_rotor_track(arec, boxes, band=band)
    else:
        ebt = fd.DomainTrack(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), "ebs")

    # 3. convergence + fusion
    conv = fd.convergence(acoustic, ebt)
    fz = fd.fuse(acoustic, ebt, conv)
    coh = fd.cross_coherence(arec, boxes, aaud, result.offset_s, band=band,
                             fs=cfg.coherence_fs) if boxes is not None else \
        {"freqs": np.zeros(0), "coherence": np.zeros(0), "peak_hz": float("nan"),
         "peak_coh": 0.0, "span_s": 0.0}

    # 4. figures
    figs = {}
    if cfg.alignment_figure:
        figs["alignment"] = fig_alignment(rec, audio, result, out_dir, label, bin_s=cfg.align_bin_s)
    figs["acoustic_ridge"] = fig_acoustic_ridge(aaud, acoustic, band, out_dir, label,
                                                highpass_hz=cfg.spec_highpass_hz)
    figs["f0_timeline"] = fig_f0_timeline(acoustic, ebt, band, out_dir, label)
    figs["fusion_timeline"] = fig_fusion_timeline(acoustic, ebt, conv, fz, out_dir, label)
    figs["coherence"] = fig_coherence(coh, band, out_dir, label)
    if boxes is not None and boxes["t"].size:
        try:
            figs["track_review"] = fig_ebs_track_review(arec, boxes, ebt, cfg.accum_dt, out_dir, label)
        except Exception as e:
            print(f"   [warn] track review dashboard: {e}")

    # 5. reviewable boxes table (resilient: a locked output — e.g. open in Excel — must not abort
    # the whole study; warn and carry on so the summary/report still get written)
    if boxes is not None and boxes["t"].size:
        bb = boxes["bbox"]
        try:
            export.save_table({"t_s": boxes["t"], "cx": boxes["cx"], "cy": boxes["cy"],
                               "x0": bb[:, 0], "y0": bb[:, 1], "x1": bb[:, 2], "y1": bb[:, 3],
                               "rotor_f0_hz": ebt.f0, "snr": ebt.snr, "confidence": ebt.conf},
                              os.path.join(out_dir, "ebs_boxes"))
        except PermissionError:
            print("   [warn] ebs_boxes.csv is open/locked — skipped (close it and re-run to refresh)")

    # 6. summary + report
    summary = {
        "label": label, "context": context or {}, "config": cfg.as_dict(),
        "alignment": result.as_dict(),
        "clips": {"dir": "clips", "ebs_raw": manifest["ebs_raw"],
                  "audio_wav": manifest["audio_wav"], "manifest": f"{label}_fusion_manifest.json"},
        "acoustic": {"f0_hz": acoustic.median_f0(), "confidence": acoustic.detection_confidence(),
                     "present_frac": float(np.mean(acoustic.present())) if acoustic.t.size else 0.0,
                     "method": "harmonic-sum f0(t) on band-passed STFT"},
        "ebs": {"f0_hz": ebt.median_f0(), "confidence": ebt.detection_confidence(),
                "present_frac": float(np.mean(ebt.present())) if ebt.t.size else 0.0,
                "mean_snr": float(np.nanmean(ebt.snr)) if ebt.t.size else 0.0,
                "n_frames": int(ebt.t.size), "n_frames_raw": n_raw,
                "method": "single_centroid box → in-box rotor FFT (dominant peak)"},
        "convergence": {k: conv[k] for k in ("temporal_overlap", "band_agreement",
                                             "median_freq_ratio", "f0_covariation_r", "kappa_overall")},
        "fusion": {"method": "convergence-gated Bayesian (#2)",
                   "C_acoustic": fz["C_acoustic"], "C_ebs": fz["C_ebs"],
                   "p_dual_bayes": fz["p_dual_bayes"], "p_fused": fz["p_fused"],
                   "fused_f0_hz": fz["fused_f0_hz"]},
        "coherence": {"method": "magnitude-squared coherence (#3)", "peak_coh": coh["peak_coh"],
                      "peak_hz": coh["peak_hz"], "span_s": coh["span_s"]},
        "pruning": {"drop_time_ranges": drop_time_ranges, "keep_window": keep_window},
        "inputs": {"ebs_raw": os.path.abspath(raw_path), "audio_wav": os.path.abspath(wav_path)},
        "figures": {k: [os.path.basename(p) for p in v] for k, v in figs.items() if v},
    }
    export.save_json(summary, os.path.join(out_dir, "fusion_summary.json"))
    _write_report(out_dir, label, summary, context)
    return summary
