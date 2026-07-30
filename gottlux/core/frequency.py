"""
frequency.py — the temporal-frequency engine (the heart of flutter/flicker detection).

A spinning rotor or a beating wing modulates scene brightness *periodically*, so the
events it produces arrive in periodic bursts at a characteristic frequency (plus
harmonics). Everything that distinguishes a drone or an insect from something that is
merely moving lives in that temporal signature. This module is the toolbox for measuring
it, four ways:

1. :func:`region_spectrum`  — the workhorse. Bin a region's events to a regular series at
   ``fs``, FFT it, find the in-band spectral peak and its SNR, and score the harmonic comb
   (a rotor shows a fundamental *and* its 2×/3× overtones — a strong discriminator).
2. :func:`lomb_scargle`     — a periodogram that works directly on irregular event *times*
   without binning; better when the event rate is low/uneven.
3. :func:`spectrogram`      — frequency-vs-time for a region, to watch a signature evolve
   (a drone spinning up, a maneuver, a wingbeat changing).
4. :func:`flicker_map`      — the showpiece. A 2-D map over the sensor: each spatial cell's
   dominant in-band flicker frequency and its SNR, computed in one vectorized FFT. Lets you
   literally *see* a rotor disk light up at 200 Hz while the static background stays dark.

All frequencies are in Hz. "SNR" throughout is the dimensionless ratio of the in-band
spectral peak power to a robust (median) noise floor — comparable across regions and runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ====================================================================================
# Binning a (sub)stream of events into a regular time series
# ====================================================================================
def bin_signal(t_us, fs: float, t0_us=None, t1_us=None):
    """Bin event *times* (µs) into a regular event-count series sampled at *fs* Hz.

    Returns ``(signal, t_axis_s)`` where ``signal[k]`` is the number of events in the k-th
    ``1/fs``-second bin. This regular series is what the FFT / spectrogram consume.
    """
    t_us = np.asarray(t_us, np.float64)
    if t_us.size == 0:
        return np.zeros(0), np.zeros(0)
    lo = float(t_us[0] if t0_us is None else t0_us)
    hi = float(t_us[-1] if t1_us is None else t1_us)
    if hi <= lo:
        return np.zeros(0), np.zeros(0)
    dt_us = 1e6 / fs
    n = int((hi - lo) / dt_us)
    if n < 2:
        return np.zeros(0), np.zeros(0)
    edges = lo + np.arange(n + 1) * dt_us
    sig = np.histogram(t_us, bins=edges)[0].astype(np.float64)
    centers = (edges[:-1] + 0.5 * dt_us - lo) / 1e6
    return sig, centers


# ====================================================================================
# Single-region spectrum + harmonic comb score
# ====================================================================================
@dataclass
class Spectrum:
    """Result of :func:`region_spectrum`."""
    freqs: np.ndarray         # Hz
    power: np.ndarray         # spectral power (same length as freqs)
    peak_freq: float          # Hz of the strongest in-band line (nan if none)
    peak_power: float
    snr: float                # peak / median-noise-floor (dimensionless)
    harmonic_score: float     # 0..1, strength of the fundamental's overtone comb
    band: tuple               # (fmin, fmax) used
    n_events: int

    @property
    def detected(self) -> bool:
        return np.isfinite(self.peak_freq)


def _window(n: int) -> np.ndarray:
    """Hann window (reduces spectral leakage) of length *n*."""
    return np.hanning(n) if n > 1 else np.ones(n)


def whiten_power(power: np.ndarray, method: str = "none", win: int | None = None) -> np.ndarray:
    """Normalize a power spectrum to **emphasize peaks** over a colored noise floor.

    EBS event noise is not white — the spectrum slopes — so a raw peak can be buried under
    low-frequency haze. Whitening divides (or standardizes) each bin by a *local* estimate of
    the floor, so a real spectral line stands up sharply regardless of where it sits.

    * ``"none"``   — return unchanged.
    * ``"median"`` — divide by a sliding-median floor (robust spectral whitening). A flat
      noise region becomes ≈1; lines become their true peak-to-floor ratio.
    * ``"zscore"`` — subtract a sliding mean and divide by a sliding std (clipped at 0): the
      spectrum becomes "sigmas above local noise", which makes a faint comb obvious.
    """
    p = np.asarray(power, np.float64)
    if method == "none" or p.size < 8:
        return power
    if win is None:
        win = max(5, (p.size // 20) | 1)        # odd, ~5% of the spectrum
    try:
        from scipy.ndimage import median_filter, uniform_filter
    except Exception:
        return power
    if method == "median":
        floor = median_filter(p, size=win, mode="nearest")
        return p / (floor + 1e-20)
    if method == "zscore":
        mean = uniform_filter(p, size=win, mode="nearest")
        var = np.maximum(uniform_filter(p * p, size=win, mode="nearest") - mean * mean, 1e-20)
        return np.maximum((p - mean) / np.sqrt(var), 0.0)
    return power


def region_spectrum(t_us, fs: float = 2000.0, fmin: float = 10.0, fmax: float = 800.0,
                    detrend: bool = True, window: bool = True,
                    n_harmonics: int = 3, normalize: str = "none",
                    derotate_hz: float = 0.0) -> Spectrum:
    """Compute the temporal power spectrum of a region's event stream.

    Parameters
    ----------
    t_us : array
        Event times (µs) of all events inside the region/ROI over the analysis window.
    fs : float
        Sampling rate the stream is binned to (must exceed ``2*fmax``).
    fmin, fmax : float
        The flutter pass-band searched for the dominant line.
    detrend, window : bool
        Subtract the mean / apply a Hann window before the FFT.
    n_harmonics : int
        How many overtones of the fundamental to include in the harmonic-comb score.
    normalize : str
        Spectral whitening to emphasize peaking — ``"none"`` (default, raw power),
        ``"median"`` or ``"zscore"`` (see :func:`whiten_power`). Applied before the in-band
        peak pick, so it both sharpens the display *and* helps the peak survive colored noise.
    derotate_hz : float
        **Rotational-data compensation.** On a *spinning* sensor a region's events arrive in
        once-per-revolution bursts, so the raw spectrum is dominated by the rotation frequency
        (~1 Hz) and its harmonics + leakage — the "FFT gravity". Setting ``derotate_hz`` to a
        cutoff (e.g. a few × the spin rate, well below ``fmin``) subtracts a moving-average
        low-pass of that bandwidth, i.e. high-passes away the slow rotation envelope, leaving the
        high-frequency flutter. ``0`` disables it (falls back to mean ``detrend``).
    """
    sig, _ = bin_signal(t_us, fs)
    n = sig.size
    if n < 8:
        return Spectrum(np.zeros(0), np.zeros(0), np.nan, 0.0, 0.0, 0.0, (fmin, fmax),
                        int(np.size(t_us)))
    if derotate_hz and derotate_hz > 0:
        # remove the slow rotation envelope (the spin-frequency gravity + harmonics): subtract a
        # moving-average low-pass of bandwidth ~derotate_hz → a high-pass that keeps the flutter.
        L = max(3, int(round(fs / float(derotate_hz))))
        if L < n:
            sig = sig - np.convolve(sig, np.ones(L) / L, mode="same")
        else:
            sig = sig - sig.mean()
    elif detrend:
        sig = sig - sig.mean()
    if window:
        sig = sig * _window(n)
    spec = np.abs(np.fft.rfft(sig)) / n
    if spec.size > 2:
        spec[1:-1] *= 2.0                      # one-sided amplitude correction
    power = spec ** 2
    if normalize != "none":
        power = whiten_power(power, normalize)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        return Spectrum(freqs, power, np.nan, 0.0, 0.0, 0.0, (fmin, fmax), int(np.size(t_us)))
    band_power = power[band]
    bi = int(np.argmax(band_power))
    peak_freq = float(freqs[band][bi])
    peak_power = float(band_power[bi])
    # robust noise floor: median power excluding DC and the immediate peak neighborhood
    noise_pool = power[1:].copy()
    noise = float(np.median(noise_pool)) if noise_pool.size else 0.0
    snr = peak_power / (noise + 1e-12)
    harmonic_score = _harmonic_comb(freqs, power, peak_freq, fmax, n_harmonics)
    return Spectrum(freqs, power, peak_freq, peak_power, snr, harmonic_score,
                    (fmin, fmax), int(np.size(t_us)))


def _harmonic_comb(freqs, power, f0, fmax, n_harmonics) -> float:
    """Fraction of the fundamental's overtones (2·f0, 3·f0, …) that show a local power
    bump. A rotor's periodic-but-non-sinusoidal modulation lights up this comb; broadband
    motion does not. Returns 0..1."""
    if not np.isfinite(f0) or f0 <= 0 or freqs.size < 4:
        return 0.0
    df = freqs[1] - freqs[0]
    base = _local_power(freqs, power, f0, df)
    hits = 0
    total = 0
    for k in range(2, n_harmonics + 2):
        fk = k * f0
        if fk > fmax or fk > freqs[-1]:
            break
        total += 1
        if _local_power(freqs, power, fk, df) > 0.25 * base:
            hits += 1
    return hits / total if total else 0.0


def _local_power(freqs, power, f, df) -> float:
    """Max power within ±1.5 bins of frequency *f*."""
    lo = np.searchsorted(freqs, f - 1.5 * df)
    hi = np.searchsorted(freqs, f + 1.5 * df) + 1
    seg = power[lo:hi]
    return float(seg.max()) if seg.size else 0.0


# ====================================================================================
# Lomb–Scargle (irregularly-sampled event times, no binning)
# ====================================================================================
def lomb_scargle(t_us, fmin: float = 10.0, fmax: float = 800.0, n_freq: int = 512,
                 max_events: int = 6000, seed: int = 0):
    """Lomb–Scargle periodogram of an event point-process directly from its *times*.

    Treats each event as a unit impulse at its (µs→s) time and evaluates the normalized
    periodogram on a linear frequency grid. Returns ``(freqs_hz, power, peak_freq_hz)``.
    Robust when the event rate is too low/uneven for clean binning.

    ``scipy.signal.lombscargle`` costs O(n_freq · n_events), so for dense streams the events
    are uniformly subsampled to *max_events* first (random subsampling of a periodic point
    process preserves its period). For dense, regularly-binnable data prefer the much faster
    :func:`region_spectrum`; Lomb–Scargle's advantage is sparse / uneven sampling.
    """
    from scipy.signal import lombscargle
    t = np.asarray(t_us, np.float64) / 1e6
    if t.size < 8:
        return np.zeros(0), np.zeros(0), np.nan
    if t.size > max_events:                    # cap cost for dense streams
        idx = np.sort(np.random.default_rng(seed).choice(t.size, max_events, replace=False))
        t = t[idx]
    t = t - t[0]
    y = np.ones_like(t)
    y = y - y.mean()
    if not np.any(y):                          # all identical -> add tiny jitter signal
        y = np.sin(2 * np.pi * t)
    freqs = np.linspace(max(fmin, 1e-3), fmax, n_freq)
    ang = 2 * np.pi * freqs
    power = lombscargle(t, y, ang, normalize=True)
    pf = float(freqs[int(np.argmax(power))]) if power.size else np.nan
    return freqs, power, pf


# ====================================================================================
# Spectrogram (frequency vs time) for a region
# ====================================================================================
def spectrogram(t_us, fs: float = 2000.0, nperseg: int = 256, overlap: float = 0.75,
                fmax: float | None = None):
    """STFT spectrogram of a region's binned event stream.

    Returns ``(t_s, f_hz, S)`` with ``S`` shaped ``(n_freq, n_time)`` in power. Use to watch
    a flutter signature evolve (spin-up, maneuver, wing-loading changes).
    """
    from scipy.signal import spectrogram as _spec
    sig, _ = bin_signal(t_us, fs)
    if sig.size < nperseg:
        nperseg = max(16, sig.size // 4)
    if sig.size < 8:
        return np.zeros(0), np.zeros(0), np.zeros((0, 0))
    sig = sig - sig.mean()
    noverlap = int(nperseg * overlap)
    f, tt, S = _spec(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                     scaling="spectrum", mode="magnitude")
    if fmax is not None:
        keep = f <= fmax
        f, S = f[keep], S[keep]
    return tt, f, S ** 2


# ====================================================================================
# Flicker map — the 2-D "where is it flickering, and how fast?" image
# ====================================================================================
@dataclass
class FlickerMap:
    """Result of :func:`flicker_map` — per-cell flicker statistics over the sensor."""
    dominant_freq: np.ndarray    # (Gh, Gw) Hz; nan where insufficient events
    snr: np.ndarray              # (Gh, Gw) peak/noise ratio
    band_power: np.ndarray       # (Gh, Gw) in-band peak power
    event_count: np.ndarray      # (Gh, Gw) events per cell
    cell: int                    # spatial cell size (px)
    fs: float
    band: tuple                  # (fmin, fmax)
    shape: tuple                 # (H, W) of the source sensor

    def upsampled_freq(self) -> np.ndarray:
        """Dominant-frequency map nearest-neighbor upsampled back to full sensor (H, W)."""
        return np.kron(self.dominant_freq, np.ones((self.cell, self.cell)))[: self.shape[0], : self.shape[1]]


def flicker_map(source, fmin: float = 80.0, fmax: float = 800.0, fs: float = 2000.0,
                cell: int = 8, t0: float | None = None, t1: float | None = None,
                min_events_per_cell: int = 40, snr_min: float = 0.0,
                mem_budget_elems: int = 200_000_000) -> FlickerMap:
    """Compute a per-cell dominant-flicker-frequency map over the sensor.

    Accepts a :class:`~gottlux.io.recording.Recording` or
    :class:`~gottlux.io.recording.EventWindow`. Spatially bins the sensor into
    ``cell × cell`` cells, builds the per-cell event-count time series in one pass, FFTs
    every cell at once, and records the in-band peak frequency + SNR per cell.

    To stay within ``mem_budget_elems`` (the cells×time-bins matrix), the time sampling is
    automatically coarsened if needed (a note is printed). Reduce *cell* for finer spatial
    detail, lengthen ``[t0, t1]`` for finer frequency resolution.
    """
    # Resolve to a window of arrays.
    if hasattr(source, "window"):                          # a Recording
        win = source.window(t0, t1)
    else:                                                  # an EventWindow
        win = source
    x = np.asarray(win.x); y = np.asarray(win.y); t = np.asarray(win.t, np.float64)
    H, W = int(win.height), int(win.width)
    Gw = (W + cell - 1) // cell
    Gh = (H + cell - 1) // cell
    n_cells = Gw * Gh

    out_freq = np.full((Gh, Gw), np.nan, np.float32)
    out_snr = np.zeros((Gh, Gw), np.float32)
    out_pow = np.zeros((Gh, Gw), np.float32)
    out_cnt = np.zeros((Gh, Gw), np.int64)
    if x.size < 8:
        return FlickerMap(out_freq, out_snr, out_pow, out_cnt, cell, fs, (fmin, fmax), (H, W))

    lo = float(t[0] if t0 is None else t0 * 1e6)
    hi = float(t[-1] if t1 is None else t1 * 1e6)
    span = max(hi - lo, 1.0)
    # auto-coarsen fs so n_cells * n_tbins stays within budget
    n_tbins = int(span / (1e6 / fs))
    if n_cells * n_tbins > mem_budget_elems and n_tbins > 0:
        fs_eff = fs * (mem_budget_elems / (n_cells * n_tbins))
        fs_eff = max(fs_eff, 2.2 * fmax)                   # never violate Nyquist for the band
        n_tbins = int(span / (1e6 / fs_eff))
        print(f"[flicker_map] coarsened fs {fs:.0f}->{fs_eff:.0f} Hz to fit memory budget "
              f"({n_cells} cells x {n_tbins} bins).")
        fs = fs_eff
    if n_tbins < 8:
        return FlickerMap(out_freq, out_snr, out_pow, out_cnt, cell, fs, (fmin, fmax), (H, W))

    dt_us = 1e6 / fs
    tb = np.clip(((t - lo) / dt_us).astype(np.int64), 0, n_tbins - 1)
    cx = np.clip(x // cell, 0, Gw - 1).astype(np.int64)
    cy = np.clip(y // cell, 0, Gh - 1).astype(np.int64)
    cell_idx = cy * Gw + cx
    combo = cell_idx * n_tbins + tb
    counts = np.bincount(combo, minlength=n_cells * n_tbins).reshape(n_cells, n_tbins)
    cell_counts = counts.sum(axis=1)

    # FFT every cell at once (rows with too few events are skipped to save work).
    active = np.where(cell_counts >= min_events_per_cell)[0]
    if active.size:
        sig = counts[active].astype(np.float64)
        sig -= sig.mean(axis=1, keepdims=True)
        sig *= _window(n_tbins)[None, :]
        spec = np.abs(np.fft.rfft(sig, axis=1)) / n_tbins
        if spec.shape[1] > 2:
            spec[:, 1:-1] *= 2.0
        power = spec ** 2
        freqs = np.fft.rfftfreq(n_tbins, d=1.0 / fs)
        band = (freqs >= fmin) & (freqs <= fmax)
        if band.any():
            bpow = power[:, band]
            bfreqs = freqs[band]
            bi = np.argmax(bpow, axis=1)
            rows = np.arange(active.size)
            peak_power = bpow[rows, bi]
            peak_freq = bfreqs[bi]
            noise = np.median(power[:, 1:], axis=1) + 1e-12
            snr = peak_power / noise
            ay = active // Gw
            ax = active % Gw
            keep = snr >= snr_min
            out_freq[ay[keep], ax[keep]] = peak_freq[keep].astype(np.float32)
            out_snr[ay[keep], ax[keep]] = snr[keep].astype(np.float32)
            out_pow[ay[keep], ax[keep]] = peak_power[keep].astype(np.float32)
    out_cnt = cell_counts.reshape(Gh, Gw)
    return FlickerMap(out_freq, out_snr, out_pow, out_cnt, cell, fs, (fmin, fmax), (H, W))


def flicker_map_max(rec, fmin: float = 80.0, fmax: float = 800.0, fs: float = 2000.0,
                    cell: int = 8, window_s: float = 1.0, hop_s: float | None = None,
                    t0: float | None = None, t1: float | None = None,
                    min_events_per_cell: int = 30, progress=None) -> FlickerMap:
    """Whole-recording flicker map by tiling short windows and keeping each cell's best.

    A single FFT over the entire recording both wastes time (frequency resolution far finer
    than any flutter needs) and *smears* a target that moves between cells. Instead this
    slides a ``window_s`` analysis window in ``hop_s`` steps and, for every cell, retains the
    window in which that cell showed the strongest in-band flutter (max SNR). The result is a
    sharp "peak flutter ever seen here, and at what frequency" image — the right summary for a
    whole flight. Each window is small, so this is fast and bounded in memory.
    """
    hop_s = hop_s if hop_s is not None else window_s
    t_lo = rec.t_start_s if t0 is None else t0
    t_hi = rec.t_stop_s if t1 is None else t1
    H, W = rec.height, rec.width
    Gw = (W + cell - 1) // cell
    Gh = (H + cell - 1) // cell
    best_snr = np.zeros((Gh, Gw), np.float32)
    best_freq = np.full((Gh, Gw), np.nan, np.float32)
    best_pow = np.zeros((Gh, Gw), np.float32)
    tot_cnt = np.zeros((Gh, Gw), np.int64)
    starts = np.arange(t_lo, max(t_hi - window_s, t_lo) + 1e-9, hop_s)
    for k, s in enumerate(starts):
        fm = flicker_map(rec, fmin=fmin, fmax=fmax, fs=fs, cell=cell,
                         t0=s, t1=min(s + window_s, t_hi),
                         min_events_per_cell=min_events_per_cell)
        better = fm.snr > best_snr
        best_snr[better] = fm.snr[better]
        best_freq[better] = fm.dominant_freq[better]
        best_pow[better] = fm.band_power[better]
        tot_cnt += fm.event_count
        if progress:
            try:
                progress((k + 1) / len(starts))
            except Exception:
                pass
    return FlickerMap(best_freq, best_snr, best_pow, tot_cnt, cell, fs, (fmin, fmax), (H, W))


# ====================================================================================
# Non-uniform DFT — spectrum straight from event times, on a chosen frequency grid
# ====================================================================================
def nudft(t_us, freqs_hz, weights=None, chunk: int = 4096) -> np.ndarray:
    """Non-uniform DFT *power* at arbitrary frequencies, directly from event times.

    Treats each event as a unit impulse at its (µs→s) time and evaluates
    ``|Σ wₖ·exp(-2πi f tₖ)|² / N²`` on the supplied frequency grid. Unlike
    :func:`region_spectrum` there is **no binning and no Nyquist ceiling tied to a sample
    rate** — you pick the exact frequencies to probe, so you can zoom a fine grid onto a
    suspected tone or probe above what a modest ``fs`` would allow. Cost is
    ``O(n_freq · n_events)``; events are evaluated in chunks to bound memory.
    """
    t = np.asarray(t_us, np.float64) / 1e6
    f = np.asarray(freqs_hz, np.float64)
    n = t.size
    if n < 4 or f.size == 0:
        return np.zeros(f.shape, np.float64)
    t = t - t[0]
    if weights is None:
        w = np.ones(n)                            # unit impulses (a point process)
    else:
        w = np.asarray(weights, np.float64)
        w = w - w.mean()                          # centre only genuine (non-uniform) weights
    acc = np.zeros(f.size, np.complex128)
    for i in range(0, n, chunk):                  # chunk events to cap the (n_freq × chunk) matrix
        ts = t[i:i + chunk]
        ws = w[i:i + chunk]
        acc += (np.exp(-2j * np.pi * np.outer(f, ts)) * ws).sum(axis=1)
    return (np.abs(acc) / n) ** 2


def nufft_spectrum(t_us, fmin: float = 10.0, fmax: float = 800.0, n_freq: int = 600,
                   normalize: str = "none", max_events: int = 8000, seed: int = 0) -> Spectrum:
    """A :class:`Spectrum` computed by direct non-uniform DFT (no binning).

    Drop-in alternative to :func:`region_spectrum` for the workbench's "non-uniform FFT"
    option: same return type (so the same plot/readout/harmonic-comb code works), but the
    transform is evaluated exactly at a dense linear grid over ``[fmin, fmax]``. Dense
    streams are uniformly subsampled to *max_events* first (random subsampling preserves a
    periodic point-process's period) to keep the cost bounded.
    """
    t = np.asarray(t_us, np.float64)
    if t.size < 8:
        return Spectrum(np.zeros(0), np.zeros(0), np.nan, 0.0, 0.0, 0.0, (fmin, fmax),
                        int(np.size(t)))
    if t.size > max_events:
        idx = np.sort(np.random.default_rng(seed).choice(t.size, max_events, replace=False))
        t = t[idx]
    freqs = np.linspace(max(fmin, 1e-3), fmax, int(n_freq))
    power = nudft(t, freqs)
    if normalize != "none":
        power = whiten_power(power, normalize)
    bi = int(np.argmax(power))
    peak_freq = float(freqs[bi])
    peak_power = float(power[bi])
    noise = float(np.median(power)) if power.size else 0.0
    snr = peak_power / (noise + 1e-12)
    harmonic = _harmonic_comb(freqs, power, peak_freq, fmax, 3)
    return Spectrum(freqs, power, peak_freq, peak_power, snr, harmonic, (fmin, fmax),
                    int(np.size(t)))


# ====================================================================================
# Inter-event-interval frequency — a very-low-compute periodicity lens
# ====================================================================================
def isi_frequency(t_us, fmin: float = 5.0, fmax: float = 2000.0, n_bins: int = 64):
    """Estimate a dominant periodicity from inter-event intervals (no FFT, O(n log n)).

    Sorts the region's event times, takes successive differences (the inter-spike
    intervals), maps each to an instantaneous rate ``1/Δt`` and finds the dominant rate from
    a log-spaced histogram. A periodically-bursting source (rotor, wingbeat) concentrates
    its intervals, so the histogram develops a sharp mode; broadband motion spreads flat.

    Returns ``(freq_hz, strength)`` where *strength* ∈ [0, 1] is how concentrated the ISI
    distribution is around its mode (a cheap confidence). ``freq_hz`` is ``nan`` if there is
    too little to judge. This is a fast first look — confirm a candidate with a real
    spectrum; it is most useful per-pixel / tiny-region where an FFT is overkill.
    """
    t = np.sort(np.asarray(t_us, np.float64)) / 1e6
    if t.size < 8:
        return float("nan"), 0.0
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size < 4:
        return float("nan"), 0.0
    f = 1.0 / dt
    f = f[(f >= fmin) & (f <= fmax)]
    if f.size < 4:
        return float("nan"), 0.0
    edges = np.logspace(np.log10(fmin), np.log10(fmax), int(n_bins) + 1)
    h, _ = np.histogram(f, bins=edges)
    if h.sum() == 0:
        return float("nan"), 0.0
    centers = np.sqrt(edges[:-1] * edges[1:])
    bi = int(np.argmax(h))
    peak_f = float(centers[bi])
    strength = float(h[bi] / h.sum())             # fraction of intervals in the modal bin
    return peak_f, strength


# ====================================================================================
# Two-point space-time measurement (the "drop two points and read it off" tool)
# ====================================================================================
def measure_between(p0, p1, cycles: float = 1.0) -> dict:
    """Geometry + an implied frequency from two picked space-time points.

    Each point is ``(x_px, y_px, t_s)``. If you drop one point on a flutter stripe and the
    next on the following stripe, the time gap is one period, so ``f = cycles / Δt``. Also
    returns the in-image separation and the apparent image-plane speed between them — useful
    for sanity-checking a track or reading a wingbeat straight off the 3-D cloud.
    """
    x0, y0, t0 = float(p0[0]), float(p0[1]), float(p0[2])
    x1, y1, t1 = float(p1[0]), float(p1[1]), float(p1[2])
    dt = t1 - t0
    dx = x1 - x0
    dy = y1 - y0
    dr = float(np.hypot(dx, dy))
    out = dict(dt_s=dt, dx_px=dx, dy_px=dy, dr_px=dr,
               freq_hz=(abs(cycles / dt) if dt != 0 else float("inf")),
               period_s=(abs(dt / cycles) if cycles else float("nan")),
               speed_px_s=(dr / abs(dt) if dt != 0 else float("inf")),
               cycles=cycles)
    return out
