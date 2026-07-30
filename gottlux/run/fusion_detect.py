"""
fusion_detect.py — drone detection in each domain (acoustic + EBS) and their fusion.

The detection/fusion substrate behind :mod:`gottlux.run.fusion_study`. Two sensors observe the same
flight; each yields a **time-resolved rotor fundamental** f0(t) and a **confidence** that a drone
is present, and the two are then **fused**:

* :func:`acoustic_track` — harmonic-aware acoustic f0(t): band-pass → STFT → per-frame
  **harmonic-sum** over candidate fundamentals (the rotor's f0, 2f0, 3f0… comb), continuity-tracked,
  with a harmonic SNR / harmonic count / spectral-flatness → ``C_acoustic``.
* :func:`ebs_box_track` + :func:`ebs_rotor_track` — track the drone (``single_centroid``, the
  staring-tier tracker), take the **in-box** event-rate rotor spectrum per frame → EBS f0(t),
  SNR, harmonic score → ``C_ebs``. Operator pruning (drop spurious time ranges) is supported.
* :func:`convergence` — how well the two domains agree (frequency + temporal coincidence), κ∈[0,1].
* :func:`fuse` — **convergence-gated Bayesian** P(drone) = Bayes(Cₐ,Cₑ)·κ, and a joint fused f0.
* :func:`cross_coherence` — magnitude-squared **coherence** γ²(f) between the in-box EBS event-rate
  and the audio (resampled to a common rate): phase-stable shared periodicity, the headline synthesis.

Pure NumPy / SciPy + the gottlux library; no Qt, no matplotlib. All physical context (band, blade
count, tracker preset) is passed in — nothing here is project-specific.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Logistic confidence calibration (defaults: ~6 dB harmonic SNR + 2 harmonics → ~0.5).
_A_SNR = 0.30          # per dB of harmonic SNR
_B_HARM = 0.55         # per supported harmonic
_C_TONAL = 1.2         # per unit of tonality (1 - spectral flatness)
_D_BIAS = 2.6          # offset


def _logistic(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _bandpass(x, fs, lo, hi, order=4):
    # second-order sections (sosfiltfilt): a transfer-function Butterworth bandpass goes
    # numerically unstable → NaN when the normalized cutoffs are tiny (a kHz band at a 96 kHz
    # sample rate), which is exactly the acoustic case here.
    from scipy.signal import butter, sosfiltfilt
    lo = max(lo, 1.0); hi = min(hi, 0.499 * fs)
    sos = butter(order, [lo / (0.5 * fs), hi / (0.5 * fs)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def spectral_flatness(p):
    """Wiener entropy (geometric mean / arithmetic mean) of a power spectrum slice, in [0, 1].
    Low = tonal (peaky), high = flat (broadband noise)."""
    p = np.asarray(p, float)
    p = p[p > 0]
    if p.size < 4:
        return 1.0
    return float(np.exp(np.mean(np.log(p))) / (np.mean(p) + 1e-30))


def harmonic_f0(freqs, power, *, f0_range=(380.0, 780.0), band=(300.0, 1400.0), n_harm=5,
                cand_step=2.0):
    """Estimate a rotor fundamental from a power spectrum by **harmonic sum**.

    Scores every candidate f0 in *f0_range* by Σ_{h=1..n_harm} P(h·f0) (interp), so the estimate is
    supported by the whole rotor comb rather than a single bin — which is what stops the naive
    argmax from latching onto a strong low-frequency shoulder or a lone harmonic. The **same**
    estimator is applied to the acoustic STFT and the EBS in-box spectrum so the two domains are
    compared like-for-like. Returns ``(f0, snr_over_floor, harmonic_count)``.
    """
    freqs = np.asarray(freqs, float); power = np.maximum(np.asarray(power, float), 0.0)
    if freqs.size < 4:
        return float("nan"), 0.0, 0.0
    bm = (freqs >= band[0]) & (freqs <= band[1])
    floor = float(np.median(power[bm])) + 1e-30 if np.any(bm) else float(np.median(power)) + 1e-30
    cand = np.arange(f0_range[0], f0_range[1] + 1e-6, cand_step)
    score = np.zeros(cand.size)
    for h in range(1, n_harm + 1):
        score += np.interp(h * cand, freqs, power, left=0.0, right=0.0)
    cf = float(cand[int(np.argmax(score))])
    hvals = np.array([np.interp(h * cf, freqs, power, left=0.0, right=0.0) for h in range(1, n_harm + 1)])
    snr = float(hvals.sum() / (floor * n_harm))
    harm = int(np.sum(hvals > 3.0 * floor))
    return cf, snr, harm


# ====================================================================================
# Acoustic
# ====================================================================================
@dataclass
class DomainTrack:
    """A per-frame rotor estimate from one sensor domain."""
    t: np.ndarray                       # frame centers (s, on the shared clock)
    f0: np.ndarray                      # rotor fundamental per frame (Hz; nan if none)
    snr: np.ndarray                     # harmonic SNR (× over broadband floor)
    harm: np.ndarray                    # supported-harmonic count
    conf: np.ndarray                    # per-frame detection confidence [0,1]
    domain: str = ""
    extra: dict = field(default_factory=dict)

    def present(self, thresh=0.5):
        return self.conf >= thresh

    def median_f0(self, thresh=0.5):
        m = self.present(thresh) & np.isfinite(self.f0)
        return float(np.median(self.f0[m])) if np.any(m) else float("nan")

    def detection_confidence(self, thresh=0.5):
        """Clip-level confidence: the high quantile of per-frame confidence (a drone need only be
        present for part of the pass)."""
        return float(np.nanquantile(self.conf, 0.90)) if self.conf.size else 0.0


def acoustic_track(audio, *, band=(300.0, 1400.0), f0_range=(380.0, 780.0), n_harm=5,
                   frame_s=0.10, hop_s=0.05, highpass_hz=200.0, cand_step=2.0) -> DomainTrack:
    """Harmonic-aware, time-resolved acoustic rotor fundamental f0(t) + confidence.

    Band-passes the audio, takes an STFT, and for each frame scores every candidate fundamental in
    *f0_range* by the **harmonic sum** Σ_{h=1..n_harm} S(h·f0) — so the estimate is supported by the
    whole rotor comb, not a single bin (which is how the naive peak latched onto low-frequency
    rumble). The winning f0 per frame is continuity-tracked; per-frame confidence comes from the
    harmonic SNR, the supported-harmonic count, and the spectral tonality.
    """
    from scipy.signal import stft
    fs = audio.sample_rate
    x = audio.normalized()
    if x.size < int(2 * frame_s * fs):
        return DomainTrack(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), "acoustic")
    x = _bandpass(x, fs, max(highpass_hz, band[0] * 0.8), band[1] * 1.8)
    nper = int(frame_s * fs)
    nov = nper - int(hop_s * fs)
    f, t, Z = stft(x, fs=fs, nperseg=nper, noverlap=nov)
    S = (np.abs(Z) ** 2)                                  # power [n_freq, n_frame]
    band_m = (f >= band[0] * 0.5) & (f <= band[1] * 1.2)
    nF = t.size
    f0 = np.full(nF, np.nan); snr = np.zeros(nF); harm = np.zeros(nF); flat = np.ones(nF)
    for j in range(nF):
        col = S[:, j]
        cf, s, h = harmonic_f0(f, col, f0_range=f0_range, band=band, n_harm=n_harm,
                               cand_step=cand_step)
        f0[j] = cf; snr[j] = s; harm[j] = h
        flat[j] = spectral_flatness(col[band_m])
    f0 = _ridge(f0, snr)
    conf = _logistic(_A_SNR * (10 * np.log10(np.maximum(snr, 1e-3))) + _B_HARM * harm
                     + _C_TONAL * (1.0 - flat) - _D_BIAS)
    return DomainTrack(t, f0, snr, harm, conf, "acoustic", {"flatness": flat})


def _ridge(f0, weight, win=5):
    """Continuity-clean a fundamental track: a weighted median filter that pulls octave/spurious
    jumps back toward the locally-consistent value (weight = SNR, so strong frames dominate)."""
    n = f0.size
    if n < win:
        return f0
    out = f0.copy()
    half = win // 2
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        seg = f0[a:b]; w = weight[a:b]
        m = np.isfinite(seg) & (w > 0)
        if np.any(m):
            out[i] = float(np.median(seg[m]))
    return out


# ====================================================================================
# EBS — track the drone, take the in-box rotor spectrum
# ====================================================================================
def ebs_box_track(rec, *, accum_dt=0.085, detector="single_centroid", band=(300.0, 1200.0),
                  fov_deg=20.0, target_size_m=0.225):
    """Track the drone with the staring-tier tracker and return its per-frame boxes.

    Returns ``{t, bbox (n,4), cx, cy, det}`` for the primary track (or ``None`` if nothing tracked).
    FOV / target size only feed the (unused-here) range readout; the rotor frequency comes purely
    from the in-box event timing, so they do not affect the spectra.
    """
    from gottlux.config import Config
    from gottlux.run import performance_report as pr
    from gottlux.run.track_study import track_of
    cfg = Config(mode="staring")
    cfg.detector = detector; cfg.accum_dt = accum_dt
    cfg.fov_deg = fov_deg; cfg.target_size_m = target_size_m
    cfg.sensor_w = rec.width; cfg.sensor_h = rec.height
    cfg.freq_lo, cfg.freq_hi = band
    det = pr._run_detector(rec, cfg)
    tk = track_of(det)
    if tk is None:
        return None
    return {"t": tk["t"], "bbox": tk["bbox"], "cx": tk["cx"], "cy": tk["cy"], "det": det}


def prune_boxes(boxes, *, drop_time_ranges=None, keep_window=None):
    """Operator pruning of a box track: drop frames in any ``drop_time_ranges`` ``[(t0,t1),…]`` and
    /or keep only frames within ``keep_window`` ``(t0,t1)``. Returns a new boxes dict."""
    t = boxes["t"]
    keep = np.ones(t.size, bool)
    for (a, b) in (drop_time_ranges or []):
        keep &= ~((t >= a) & (t <= b))
    if keep_window is not None:
        keep &= (t >= keep_window[0]) & (t <= keep_window[1])
    return {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in boxes.items()}


def ebs_rotor_track(rec, boxes, *, band=(300.0, 1400.0), fwin=0.3, min_snr=3.0) -> DomainTrack:
    """In-box rotor spectrum per tracked frame → EBS rotor f0(t), SNR, harmonic score, confidence.

    For each box, the spectrum of the events inside it over a trailing *fwin*-second window is taken
    (``track_study.box_spectrum`` → ``frequency.region_spectrum``); the EBS observable is its
    **dominant in-band modulation peak** (the optical blade-pass flicker is one strong line, not the
    rich harmonic comb acoustics has — so a harmonic-sum estimator that assumes a comb just splits
    it to a sub-harmonic). Confidence comes from that peak's SNR + harmonic score. Frames with too
    few events return ``nan`` f0 / zero conf.
    """
    from gottlux.run.track_study import box_spectrum
    full = rec.window(rec.t_start_s, rec.t_stop_s)
    fx = np.asarray(full.x); fy = np.asarray(full.y); fts = full.t_s; ftus = np.asarray(full.t)
    t, bb = boxes["t"], boxes["bbox"]
    n = t.size
    f0 = np.full(n, np.nan); snr = np.zeros(n); harm = np.zeros(n)
    for i in range(n):
        sp = box_spectrum(full, fx, fy, fts, ftus, float(t[i]), bb[i], band, fwin=fwin)
        if sp is not None and np.isfinite(sp.peak_freq):
            f0[i] = sp.peak_freq; snr[i] = sp.snr; harm[i] = sp.harmonic_score
    f0 = _ridge(f0, snr)
    # harmonic_score from region_spectrum is ~[0,1]; scale to act like a small harmonic count
    conf = _logistic(_A_SNR * (10 * np.log10(np.maximum(snr, 1e-3))) + _B_HARM * (harm * 3.0)
                     - _D_BIAS)
    conf[snr < min_snr] *= 0.3                           # damp low-SNR frames
    return DomainTrack(t, f0, snr, harm, conf, "ebs")


# ====================================================================================
# Convergence + fusion
# ====================================================================================
def _bayes(pa, pe):
    num = pa * pe
    den = num + (1 - pa) * (1 - pe)
    return np.where(den > 0, num / den, 0.0)


_GATE_FLOOR = 0.4          # soft convergence gate: P_gated = P_bayes·(floor + (1-floor)·κ)


def _harmonic_ratio(fa, fe):
    """``fe`` expressed as a ratio to the nearest harmonic/sub-harmonic of ``fa`` (so an EBS line
    that sits on a harmonic of the acoustic fundamental still reads as ~1.0). ``nan`` if invalid."""
    if not (np.isfinite(fa) and np.isfinite(fe)) or fa <= 0 or fe <= 0:
        return np.nan
    targets = [fa * 1, fa * 2, fa * 3, fa / 2, fa / 3]
    tgt = min(targets, key=lambda tt: abs(np.log(fe / tt)))
    return fe / tgt


def convergence(acoustic: DomainTrack, ebs: DomainTrack, *, sigma_frac=0.20, conf_thresh=0.5):
    """Honest cross-domain agreement between the two rotor tracks.

    Reports, transparently, several distinct things rather than one tuned number:

    * **temporal overlap** — Jaccard of the "drone present" frames (do both fire at the same times?);
    * **band agreement** — exp(−(Δlog f / σ)²) on the *median* fundamentals, harmonic-aware (are
      both in the same rotor band / harmonic family?);
    * **f0 co-variation** — Pearson r of the two f0(t) tracks over co-present frames (do the
      frequencies rise/fall *together* with RPM? — robust to a constant comb-line offset);
    * a per-frame instantaneous κ(t) (for the timeline figure).

    The headline ``kappa_overall = temporal_overlap · band_agreement`` (frequency co-variation is
    reported alongside but not folded in, since here it is weak — see the tier RESULTS caveats).
    """
    te, fe, ce = ebs.t, ebs.f0, ebs.conf
    fa_i = np.interp(te, acoustic.t, acoustic.f0, left=np.nan, right=np.nan)
    ca_i = np.interp(te, acoustic.t, acoustic.conf, left=0.0, right=0.0)
    kappa = np.zeros(te.size); ratio = np.full(te.size, np.nan)
    for i in range(te.size):
        r = _harmonic_ratio(fa_i[i], fe[i])
        ratio[i] = r
        if np.isfinite(r):
            kappa[i] = float(np.exp(-(np.log(r) / sigma_frac) ** 2))
    a_on = ca_i >= conf_thresh; e_on = ce >= conf_thresh
    both = a_on & e_on; either = a_on | e_on
    temporal_overlap = float(np.sum(both) / np.sum(either)) if np.any(either) else 0.0
    sel = both & np.isfinite(fa_i) & np.isfinite(fe)
    if np.sum(sel) >= 5:
        med_ratio = _harmonic_ratio(float(np.median(fa_i[sel])), float(np.median(fe[sel])))
        band_agreement = float(np.exp(-(np.log(med_ratio) / sigma_frac) ** 2))
        covar_r = (float(np.corrcoef(fa_i[sel], fe[sel])[0, 1])
                   if np.std(fa_i[sel]) > 0 and np.std(fe[sel]) > 0 else float("nan"))
    else:
        med_ratio = float("nan"); band_agreement = 0.0; covar_r = float("nan")
    return {"t": te, "kappa": kappa, "freq_ratio": ratio, "both_present": both,
            "temporal_overlap": temporal_overlap, "band_agreement": band_agreement,
            "median_freq_ratio": med_ratio, "f0_covariation_r": covar_r,
            "kappa_overall": temporal_overlap * band_agreement}


def fuse(acoustic: DomainTrack, ebs: DomainTrack, conv: dict):
    """Convergence-gated Bayesian fusion (#2): clip-level **P(drone)** + a per-frame timeline.

    Clip level: P_dual = Bayes(Cₐ, Cₑ) from each domain's detection confidence; the **fused**
    P = P_dual · (floor + (1−floor)·κ_overall) — a *soft* convergence gate (``floor`` keeps a clear
    independent dual-detection from being zeroed by an imperfect frequency lock, while genuine
    convergence lifts it to the full Bayesian value). Per frame: P_bayes(t) and the soft-gated
    P_gated(t) using the instantaneous κ(t), for the timeline figure. The fused f0 is the SNR-
    weighted median of the two domains over co-present frames (reported with the median ratio, since
    here the two lines differ — see RESULTS)."""
    te = ebs.t
    ca_i = np.interp(te, acoustic.t, acoustic.conf, left=0.0, right=0.0)
    fa_i = np.interp(te, acoustic.t, acoustic.f0, left=np.nan, right=np.nan)
    sa_i = np.interp(te, acoustic.t, acoustic.snr, left=0.0, right=0.0)
    p_bayes = _bayes(ca_i, ebs.conf)
    p_gated = p_bayes * (_GATE_FLOOR + (1 - _GATE_FLOOR) * conv["kappa"])

    Ca = acoustic.detection_confidence(); Ce = ebs.detection_confidence()
    p_dual = float(_bayes(Ca, Ce))
    p_fused = p_dual * (_GATE_FLOOR + (1 - _GATE_FLOOR) * conv["kappa_overall"])

    sel = conv["both_present"] & np.isfinite(fa_i) & np.isfinite(fe := ebs.f0)
    if np.any(sel):
        vals = np.r_[fa_i[sel], ebs.f0[sel]]; w = np.r_[sa_i[sel], ebs.snr[sel]]
        order = np.argsort(vals); cw = np.cumsum(w[order])
        fused_f0 = float(vals[order][np.searchsorted(cw, 0.5 * cw[-1])]) if cw[-1] > 0 else float("nan")
    else:
        fused_f0 = float("nan")
    return {"t": te, "p_bayes": p_bayes, "p_gated": p_gated,
            "C_acoustic": Ca, "C_ebs": Ce, "p_dual_bayes": p_dual, "p_fused": p_fused,
            "fused_f0_hz": fused_f0}


def cross_coherence(rec, boxes, audio, offset_s, *, band=(300.0, 1200.0), fs=2000.0,
                    nperseg=2048):
    """Magnitude-squared coherence γ²(f) between the in-box EBS event-rate and the audio.

    Builds a uniform in-box event-rate at *fs* (counting events that fall inside the time-nearest
    tracked box), resamples the audio onto the **same EBS clock and rate** (using the recovered
    *offset_s*), band-passes both, and computes coherence over their shared span. A coherence peak
    in *band* is phase-stable shared periodicity — the strongest 'both sensors, one rotor' evidence;
    it is robust to a constant acoustic-propagation delay (that is only linear phase). Returns
    ``{freqs, coherence, peak_hz, peak_coh, span_s}``.
    """
    from scipy.signal import coherence as _coh
    from scipy.signal import resample_poly
    t, bb = boxes["t"], boxes["bbox"]
    if t.size < 2:
        return {"freqs": np.zeros(0), "coherence": np.zeros(0), "peak_hz": float("nan"),
                "peak_coh": 0.0, "span_s": 0.0}
    t0, t1 = float(t[0]), float(t[-1])                   # EBS-clock span of the (pruned) track
    # in-box event rate at fs over [t0, t1]
    ev = rec.window(t0, t1)
    ex = np.asarray(ev.x); ey = np.asarray(ev.y); et = ev.t_s
    fi = np.clip(np.searchsorted(t, et) - 1, 0, t.size - 1)
    bx = bb[fi]
    inside = (ex >= bx[:, 0]) & (ex < bx[:, 2]) & (ey >= bx[:, 1]) & (ey < bx[:, 3])
    edges = np.arange(t0, t1, 1.0 / fs)
    if edges.size < nperseg:
        return {"freqs": np.zeros(0), "coherence": np.zeros(0), "peak_hz": float("nan"),
                "peak_coh": 0.0, "span_s": t1 - t0}
    r_ebs = np.histogram(et[inside], bins=edges)[0].astype(float)
    # audio on the EBS clock: audio-time = EBS-time - offset; slice [t0-off, t1-off], resample to fs
    a0 = (t0 - offset_s); a1 = (t1 - offset_s)
    aw = audio.window(a0, a1)
    if aw.n < 16:
        return {"freqs": np.zeros(0), "coherence": np.zeros(0), "peak_hz": float("nan"),
                "peak_coh": 0.0, "span_s": t1 - t0}
    from math import gcd
    g = gcd(int(round(fs)), int(aw.sample_rate))
    a_rs = resample_poly(aw.normalized(), int(round(fs)) // g, aw.sample_rate // g)
    n = min(r_ebs.size, a_rs.size)
    r_ebs = r_ebs[:n] - np.mean(r_ebs[:n]); a_rs = a_rs[:n]
    r_ebs = _bandpass(r_ebs, fs, band[0] * 0.8, band[1] * 1.5)
    a_rs = _bandpass(a_rs, fs, band[0] * 0.8, band[1] * 1.5)
    f, Cxy = _coh(r_ebs, a_rs, fs=fs, nperseg=min(nperseg, n))
    m = (f >= band[0]) & (f <= band[1])
    if not np.any(m):
        return {"freqs": f, "coherence": Cxy, "peak_hz": float("nan"), "peak_coh": 0.0,
                "span_s": t1 - t0}
    k = int(np.argmax(Cxy[m]))
    return {"freqs": f, "coherence": Cxy, "peak_hz": float(f[m][k]),
            "peak_coh": float(Cxy[m][k]), "span_s": t1 - t0}
