"""
fusion.py — pair an event-based ``.raw`` with a time-synchronized audio ``.wav``.

Two sensors (an EBS camera and an audio recorder) started independently record the same scene;
to study them together their clocks must be brought onto **one common timeline**. This module is
the project-agnostic substrate for that:

* :func:`read_wav` / :class:`AudioClip` — load a ``.wav`` (PCM 8/16/24/32-bit or float) to a
  mono float track, with an RMS **envelope** at any bin size.
* :func:`read_hdf5_events` / :func:`hdf5_to_raw` — read a Prophesee **HDF5** event file
  (``CD/events``) and re-emit it as an EVT2.1 ``.raw`` the rest of gottlux can open. The
  reading itself now lives in :mod:`gottlux.io.hdf5` (which the loader also uses to open
  ``.h5`` directly); the wrapper here keeps the original call and :class:`FusionError`
  contract. ECF-codec HDF5 (the Metavision "compress on save" output, HDF5 filter 36559)
  needs the codec registered — ``pip install gottlux[hdf5]`` — else that case raises a
  clear :class:`FusionError` pointing back to the source ``.raw``.
* :func:`estimate_offset` — recover the temporal offset between the two streams by
  cross-correlating the EBS **event-rate** envelope against the audio **RMS** envelope
  (de-trended, so the shared closest-point-of-approach swell lines up, not the DC level).
* :func:`export_aligned` — write the **aligned pair**: an EBS ``.raw`` and a ``.wav`` both
  cropped to the overlap and re-zeroed to a shared ``t = 0`` (plus a ``fusion_manifest.json``).

Nothing here is drone- or project-specific; the calling results script supplies the physical
context (target, blade band, …). Pure NumPy + stdlib ``wave`` (+ optional scipy/h5py); no Qt,
no matplotlib.
"""
from __future__ import annotations

import os
import wave
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gottlux.io import writer


class FusionError(RuntimeError):
    """A fusion I/O problem stated in terms the caller can act on (e.g. an SDK-locked HDF5)."""


# ====================================================================================
# Audio
# ====================================================================================
@dataclass
class AudioClip:
    """A mono audio track on its own zero-based clock.

    ``samples`` is float64 in physical amplitude units (whatever the file stored, cast to float —
    integer PCM is *not* rescaled, so round-trips are exact). ``sample_rate`` is in Hz.
    """
    samples: np.ndarray
    sample_rate: int
    source_path: str = ""
    subtype: str = "int16"          # how it was stored ('int16'|'int24'|'int32'|'uint8'|'float32')

    @property
    def n(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n / self.sample_rate if self.sample_rate else 0.0

    def normalized(self) -> np.ndarray:
        """Samples scaled to peak |1.0| (float64); all-zero stays all-zero."""
        peak = float(np.max(np.abs(self.samples))) if self.n else 0.0
        return self.samples / peak if peak > 0 else self.samples.astype(np.float64)

    def rms_envelope(self, bin_s: float = 0.010):
        """Return ``(centers_s, rms)`` — the RMS amplitude per *bin_s*-second bin (peak-normalized).

        This is the loudness-over-time curve that swells as a noisy source approaches and fades as
        it leaves — the feature aligned against the EBS event rate.
        """
        if self.n == 0 or self.sample_rate <= 0:
            return np.zeros(0), np.zeros(0)
        x = self.normalized()
        spb = max(1, int(round(self.sample_rate * bin_s)))
        m = (len(x) // spb) * spb
        if m == 0:
            return np.zeros(0), np.zeros(0)
        blocks = x[:m].reshape(-1, spb)
        rms = np.sqrt(np.mean(blocks * blocks, axis=1))
        centers = (np.arange(rms.shape[0]) + 0.5) * (spb / self.sample_rate)
        return centers, rms

    def window(self, t0: Optional[float] = None, t1: Optional[float] = None) -> "AudioClip":
        """A re-zeroed sub-clip covering ``[t0, t1)`` seconds (defaults to the whole clip)."""
        i0 = 0 if t0 is None else max(0, int(round(t0 * self.sample_rate)))
        i1 = self.n if t1 is None else min(self.n, int(round(t1 * self.sample_rate)))
        i1 = max(i1, i0)
        return AudioClip(self.samples[i0:i1].copy(), self.sample_rate,
                         source_path=self.source_path, subtype=self.subtype)


_SUBTYPE_BY_WIDTH = {1: "uint8", 2: "int16", 3: "int24", 4: "int32"}


def read_wav(path: str) -> AudioClip:
    """Load *path* to a mono :class:`AudioClip` (float64 samples).

    Handles PCM 8/16/24/32-bit via the stdlib ``wave`` module; falls back to
    ``scipy.io.wavfile`` for formats ``wave`` rejects (e.g. IEEE-float). Multi-channel audio is
    down-mixed to the mean of its channels.
    """
    path = os.path.abspath(path)
    try:
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            sw = w.getsampwidth()
            n = w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError):
        return _read_wav_scipy(path)

    if sw == 3:                                     # 24-bit packed little-endian → int32
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        x = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
        x = np.where(x >= (1 << 23), x - (1 << 24), x).astype(np.float64)
    elif sw == 1:                                   # 8-bit PCM is unsigned, centered at 128
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0
    else:
        dt = {2: "<i2", 4: "<i4"}.get(sw)
        if dt is None:
            return _read_wav_scipy(path)
        x = np.frombuffer(raw, dtype=dt).astype(np.float64)

    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return AudioClip(np.ascontiguousarray(x), int(sr), source_path=path,
                     subtype=_SUBTYPE_BY_WIDTH.get(sw, "int16"))


def _read_wav_scipy(path: str) -> AudioClip:
    try:
        from scipy.io import wavfile
    except Exception as e:                          # pragma: no cover
        raise FusionError(f"cannot read {path!r}: unsupported WAV and scipy is unavailable ({e})")
    sr, data = wavfile.read(path)
    data = np.asarray(data)
    sub = {np.dtype("float32"): "float32", np.dtype("float64"): "float32",
           np.dtype("int16"): "int16", np.dtype("int32"): "int32",
           np.dtype("uint8"): "uint8"}.get(data.dtype, "float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.uint8:
        data = data.astype(np.float64) - 128.0
    return AudioClip(data.astype(np.float64), int(sr), source_path=path, subtype=sub)


def write_wav(path: str, samples, sample_rate: int, subtype: str = "int16") -> str:
    """Write a mono *samples* array to a PCM ``.wav`` at *sample_rate*. Returns the path.

    Float input that is already peak-normalized to ``|x| ≤ 1`` is scaled up to the integer
    subtype's full range; float input outside ``[-1, 1]`` is taken to already be in integer PCM
    units and written as-is (so ``read_wav`` → ``write_wav`` round-trips a PCM clip exactly).
    Integer input is written as-is.
    """
    if not path.lower().endswith(".wav"):
        path += ".wav"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    x = np.asarray(samples)
    sw = {"uint8": 1, "int16": 2, "int24": 3, "int32": 4, "float32": 2}.get(subtype, 2)
    if np.issubdtype(x.dtype, np.floating):
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if 0.0 < peak <= 1.0:                       # normalized floats → scale to full-scale PCM
            scale = {1: 127, 2: 32767, 3: (1 << 23) - 1, 4: (1 << 31) - 1}[sw]
            x = x * scale                           # else: already integer-valued PCM, keep as-is
    if sw == 1:
        buf = np.clip(x + 128.0, 0, 255).astype(np.uint8).tobytes()
    elif sw == 3:
        xi = np.clip(np.round(x), -(1 << 23), (1 << 23) - 1).astype(np.int32)
        b = np.empty((xi.shape[0], 3), np.uint8)
        b[:, 0] = xi & 0xFF; b[:, 1] = (xi >> 8) & 0xFF; b[:, 2] = (xi >> 16) & 0xFF
        buf = b.tobytes()
    else:
        dt = {2: "<i2", 4: "<i4"}[sw]
        lim = {2: 32767, 4: (1 << 31) - 1}[sw]
        buf = np.clip(np.round(x), -lim - 1, lim).astype(dt).tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(sw)
        w.setframerate(int(sample_rate))
        w.writeframes(buf)
    return path


# ====================================================================================
# Prophesee HDF5 events  →  the gottlux event model / a .raw
# ====================================================================================
def read_hdf5_events(path: str) -> dict:
    """Read a Prophesee HDF5 event file into ``{x, y, p, t_us, width, height, meta, time_shift}``.

    A thin backwards-compatible wrapper over :func:`gottlux.io.hdf5.read_events` (where the
    HDF5 reading now lives, shared with the decode cache and the loader). Accepts the
    Metavision layout — a compound ``CD/events`` dataset with fields ``x, y, p, t`` (plus
    optional root attributes ``geometry``/``format``/``time_shift``) — and plain ``x/y/p/t``
    datasets. ``t_us`` is returned re-zeroed to its own minimum.

    Raises :class:`FusionError` if h5py is missing, the layout is unknown, or the event
    dataset is stored with Prophesee's ECF codec (HDF5 filter 36559) and no registered
    codec can decompress it (``pip install gottlux[hdf5]``); convert from the original
    ``.raw`` instead.
    """
    from gottlux.io import hdf5 as _hdf5
    d = _hdf5.read_events(path)
    return {"x": d["x"], "y": d["y"], "p": d["p"], "t_us": d["t"],
            "width": d["width"], "height": d["height"], "meta": d["meta"],
            "time_shift": d["time_shift"]}


def hdf5_to_raw(h5_path: str, out_raw: str) -> int:
    """Convert a Prophesee HDF5 event file to an EVT2.1 ``.raw``. Returns events written.

    The output is a normal gottlux-readable recording (events re-zeroed). See
    :func:`read_hdf5_events` for the ECF-codec caveat.
    """
    d = read_hdf5_events(h5_path)
    if not out_raw.lower().endswith(".raw"):
        out_raw += ".raw"
    return writer.write_raw(out_raw, d["x"], d["y"], d["p"], d["t_us"],
                            width=d["width"], height=d["height"], meta=d["meta"] or None)


# ====================================================================================
# Envelopes & temporal alignment
# ====================================================================================
def ebs_rate_envelope(rec, bin_s: float = 0.010):
    """Return ``(centers_s, rate_hz)`` — the EBS event rate vs time at *bin_s* resolution.

    Thin wrapper over :meth:`Recording.event_rate` so the pipeline and the GUI build the EBS
    alignment feature exactly the same way.
    """
    return rec.event_rate(bin_s)


def _smooth(a, win_s, bin_s):
    k = max(1, int(round(win_s / bin_s)))
    if k <= 1:
        return np.asarray(a, float)
    return np.convolve(np.asarray(a, float), np.ones(k) / k, mode="same")


def _detrend_norm(a, win_s, bin_s):
    a = np.asarray(a, float)
    a = a - _smooth(a, win_s, bin_s)          # drop the slow drift → keep CPA-scale structure
    s = float(np.std(a))
    return a / s if s > 1e-12 else a


@dataclass
class AlignResult:
    """The recovered temporal alignment between an EBS recording and an audio clip.

    ``offset_s`` is the value to **add to audio timestamps** to place them on the EBS timeline:
    a physical instant at audio-time ``tau`` occurs at EBS-time ``tau + offset_s``. ``peak_corr``
    is the normalized cross-correlation at that lag (a quality/confidence score in roughly
    ``[-1, 1]``). ``overlap_s`` is the duration the two streams share once aligned.
    """
    offset_s: float
    peak_corr: float
    bin_s: float
    max_lag_s: float
    overlap_s: float = 0.0
    lags_s: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    corr: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    def as_dict(self) -> dict:
        return {"offset_s": float(self.offset_s), "peak_corr": float(self.peak_corr),
                "bin_s": float(self.bin_s), "max_lag_s": float(self.max_lag_s),
                "overlap_s": float(self.overlap_s)}


def estimate_offset(ebs_rate, aud_rms, *, bin_s: float = 0.010, max_lag_s: float = 12.0,
                    smooth_s: float = 0.30, detrend_s: float = 2.0) -> AlignResult:
    """Cross-correlate two equal-bin envelopes to recover the audio→EBS time offset.

    Both inputs are 1-D arrays sampled at *bin_s* (the EBS event-rate and audio RMS envelopes,
    each starting at its own ``t = 0``). They are lightly smoothed and de-trended (so the shared
    closest-point-of-approach swell drives the match rather than the DC level), then correlated;
    the lag of the peak — constrained to ``±max_lag_s`` — is the offset. Returns an
    :class:`AlignResult` (``overlap_s`` is filled in later by :func:`plan_alignment`).
    """
    e = _detrend_norm(_smooth(ebs_rate, smooth_s, bin_s), detrend_s, bin_s)
    a = _detrend_norm(_smooth(aud_rms, smooth_s, bin_s), detrend_s, bin_s)
    if e.size < 2 or a.size < 2:
        return AlignResult(0.0, 0.0, bin_s, max_lag_s)
    full = np.correlate(e, a, mode="full") / max(e.size, a.size)
    lags = np.arange(-a.size + 1, e.size)
    lag_cap = int(round(max_lag_s / bin_s))
    mask = np.abs(lags) <= lag_cap
    sub = np.where(mask, full, -np.inf)
    k = int(np.argmax(sub))
    return AlignResult(offset_s=float(lags[k] * bin_s), peak_corr=float(full[k]),
                       bin_s=bin_s, max_lag_s=max_lag_s,
                       lags_s=lags.astype(np.float64) * bin_s, corr=full)


def plan_alignment(rec, audio: AudioClip, *, offset_s: Optional[float] = None,
                   bin_s: float = 0.010, max_lag_s: float = 12.0) -> AlignResult:
    """Decide the EBS↔audio alignment, auto-estimating *offset_s* if not given.

    Computes both envelopes at *bin_s*, runs :func:`estimate_offset` (unless *offset_s* is
    supplied — e.g. a manual nudge from the GUI), and fills in the shared ``overlap_s``.
    """
    _, e_rate = ebs_rate_envelope(rec, bin_s)
    _, a_rms = audio.rms_envelope(bin_s)
    if offset_s is None:
        res = estimate_offset(e_rate, a_rms, bin_s=bin_s, max_lag_s=max_lag_s)
    else:
        res = AlignResult(float(offset_s), float("nan"), bin_s, max_lag_s)
    E = rec.duration_s
    A = audio.duration_s
    lo = max(0.0, res.offset_s)                      # audio spans EBS-time [offset, offset+A]
    hi = min(E, res.offset_s + A)
    res.overlap_s = max(0.0, hi - lo)
    return res


def export_aligned(rec, audio: AudioClip, result: AlignResult, out_dir: str, *,
                   base_name: str = "aligned", bias_src: Optional[str] = None,
                   progress=None) -> dict:
    """Write the aligned EBS ``.raw`` + audio ``.wav`` (shared ``t = 0``) and a manifest.

    Both streams are cropped to their temporal overlap and re-zeroed so the output pair plays on
    one clock. The EBS clip is cut with :func:`gottlux.io.writer.cut_clip`; the audio is sliced
    and re-written at its original sample rate/subtype. A ``.bias`` sidecar (if *bias_src* is
    given and exists) is copied next to the ``.raw``. Returns the manifest dict (also written as
    ``<base_name>_fusion_manifest.json``).
    """
    import shutil

    from gottlux.io import export

    os.makedirs(out_dir, exist_ok=True)
    off = float(result.offset_s)
    E, A = rec.duration_s, audio.duration_s
    lo = max(0.0, off)                               # common window, expressed on the EBS clock
    hi = min(E, off + A)
    if hi <= lo:
        raise FusionError(f"no temporal overlap at offset {off:+.3f}s "
                          f"(EBS {E:.2f}s, audio {A:.2f}s)")

    raw_out = os.path.join(out_dir, f"{base_name}.raw")
    n_ev = writer.cut_clip(rec, raw_out, t0=lo, t1=hi, progress=progress)

    aud_clip = audio.window(lo - off, hi - off)      # same physical window, on the audio clock
    wav_out = os.path.join(out_dir, f"{base_name}.wav")
    write_wav(wav_out, aud_clip.samples, aud_clip.sample_rate, subtype=audio.subtype)

    sidecars = []
    if bias_src and os.path.exists(bias_src):
        dst = os.path.join(out_dir, f"{base_name}.bias")
        try:
            shutil.copy2(bias_src, dst)
            sidecars.append(os.path.basename(dst))
        except OSError:
            pass

    manifest = {
        "base_name": base_name,
        "ebs_raw": os.path.basename(raw_out),
        "audio_wav": os.path.basename(wav_out),
        "sidecars": sidecars,
        "alignment": result.as_dict(),
        "common_window_on_ebs_s": [round(lo, 6), round(hi, 6)],
        "common_window_on_audio_s": [round(lo - off, 6), round(hi - off, 6)],
        "aligned_duration_s": round(hi - lo, 6),
        "ebs": {"source": rec.source_path, "n_events": int(n_ev),
                "width": rec.width, "height": rec.height, "fmt": rec.fmt,
                "src_duration_s": round(E, 6)},
        "audio": {"source": audio.source_path, "sample_rate": audio.sample_rate,
                  "subtype": audio.subtype, "src_duration_s": round(A, 6)},
    }
    export.save_json(manifest, os.path.join(out_dir, f"{base_name}_fusion_manifest.json"))
    return manifest
