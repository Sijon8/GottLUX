/*
 * analysis.js — GottLUX Web frequency analysis (port of the essentials of
 * gottlux/core/frequency.py: bin_signal, region_spectrum incl. peak/SNR and
 * harmonic-comb score, and flicker_map). Vanilla JS; loadable from the main
 * thread, a Web Worker (importScripts) or Node (require).
 *
 * Honest simplifications vs the Python engine:
 *   - FFT is a radix-2 iterative complex FFT; the binned signal is zero-padded
 *     to the next power of two (finer frequency grid, same peak).
 *   - "SNR" keeps the Python definition: in-band peak power over the median
 *     power of all non-DC bins.
 *   - no whitening / Lomb-Scargle / NUDFT — the desktop suite has those.
 */
(function (global) {
  "use strict";

  // ------------------------------------------------------------------
  // Radix-2 iterative in-place complex FFT
  // ------------------------------------------------------------------
  function fft(re, im) {
    var n = re.length;
    if (n < 2 || (n & (n - 1)) !== 0) throw new Error("fft: length must be a power of two");
    // bit reversal permutation
    for (var i = 1, j = 0; i < n; i++) {
      var bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        var tr = re[i]; re[i] = re[j]; re[j] = tr;
        var ti = im[i]; im[i] = im[j]; im[j] = ti;
      }
    }
    for (var len = 2; len <= n; len <<= 1) {
      var ang = -2 * Math.PI / len;
      var wr = Math.cos(ang), wi = Math.sin(ang);
      for (var s = 0; s < n; s += len) {
        var cr = 1, ci = 0;
        var half = len >> 1;
        for (var k = 0; k < half; k++) {
          var i0 = s + k, i1 = s + k + half;
          var xr = re[i1] * cr - im[i1] * ci;
          var xi = re[i1] * ci + im[i1] * cr;
          re[i1] = re[i0] - xr; im[i1] = im[i0] - xi;
          re[i0] += xr;         im[i0] += xi;
          var ncr = cr * wr - ci * wi;
          ci = cr * wi + ci * wr;
          cr = ncr;
        }
      }
    }
  }

  function nextPow2(n) {
    var p = 1;
    while (p < n) p <<= 1;
    return p;
  }

  function hann(n) {
    // parity with np.hanning(n): 0.5 - 0.5*cos(2*pi*k/(n-1))
    var w = new Float64Array(n);
    if (n === 1) { w[0] = 1; return w; }
    for (var k = 0; k < n; k++) w[k] = 0.5 - 0.5 * Math.cos(2 * Math.PI * k / (n - 1));
    return w;
  }

  function median(arr) {
    if (!arr.length) return 0;
    var a = Float64Array.from(arr);
    a.sort();
    var m = a.length >> 1;
    return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
  }

  // One-sided power spectrum of a mean-subtracted, Hann-windowed count signal.
  // sig: Float64Array of nBins counts. Returns {power: Float64Array(K), df}.
  function powerSpectrum(sig, nBins, fs) {
    var mean = 0;
    for (var i = 0; i < nBins; i++) mean += sig[i];
    mean /= nBins;
    var w = hann(nBins);
    var N = nextPow2(nBins);
    var re = new Float64Array(N), im = new Float64Array(N);
    for (var j = 0; j < nBins; j++) re[j] = (sig[j] - mean) * w[j];
    fft(re, im);
    var K = (N >> 1) + 1;
    var power = new Float64Array(K);
    for (var k = 0; k < K; k++) {
      var amp = Math.hypot(re[k], im[k]) / nBins;      // np: |rfft(sig)| / n
      if (k > 0 && k < K - 1) amp *= 2;                // one-sided correction
      power[k] = amp * amp;
    }
    return { power: power, df: fs / N };
  }

  // ---- harmonic comb (port of _harmonic_comb / _local_power) ----------------
  function localPower(power, df, f) {
    var lo = Math.max(0, Math.ceil((f - 1.5 * df) / df));
    var hi = Math.min(power.length - 1, Math.floor((f + 1.5 * df) / df) + 1);
    var best = 0;
    for (var k = lo; k <= hi; k++) if (power[k] > best) best = power[k];
    return best;
  }

  function harmonicComb(power, df, f0, fmax, nHarmonics) {
    if (!isFinite(f0) || f0 <= 0 || power.length < 4) return 0;
    var base = localPower(power, df, f0);
    var fNyq = (power.length - 1) * df;
    var hits = 0, total = 0;
    for (var k = 2; k < nHarmonics + 2; k++) {
      var fk = k * f0;
      if (fk > fmax || fk > fNyq) break;
      total++;
      if (localPower(power, df, fk) > 0.25 * base) hits++;
    }
    return total ? hits / total : 0;
  }

  // ------------------------------------------------------------------
  // Region spectrum (port of region_spectrum essentials)
  //   tSec : Float64Array of the ROI's event times, SECONDS, ascending
  //   opts : { fmin, fmax, fs?, nHarmonics? }
  // ------------------------------------------------------------------
  function regionSpectrum(tSec, opts) {
    opts = opts || {};
    var fmin = opts.fmin !== undefined ? opts.fmin : 10;
    var fmax = opts.fmax !== undefined ? opts.fmax : 800;
    var fs = opts.fs || Math.max(2000, 2.5 * fmax);
    var nEv = tSec.length;
    var empty = { freqs: new Float32Array(0), power: new Float32Array(0),
                  peakFreq: NaN, peakPower: 0, snr: 0, harmonicScore: 0,
                  band: [fmin, fmax], nEvents: nEv, fs: fs, nBins: 0 };
    if (nEv === 0) return empty;
    var lo = tSec[0], hi = tSec[nEv - 1];
    var nBins = Math.floor((hi - lo) * fs);          // bin_signal parity
    if (nBins < 8) return empty;
    var sig = new Float64Array(nBins);
    for (var i = 0; i < nEv; i++) {
      var b = Math.floor((tSec[i] - lo) * fs);
      if (b >= 0 && b < nBins) sig[b] += 1;          // histogram edge parity: drop past-end
    }
    var ps = powerSpectrum(sig, nBins, fs);
    var power = ps.power, df = ps.df;
    var kLo = Math.max(0, Math.ceil(fmin / df));
    var kHi = Math.min(power.length - 1, Math.floor(fmax / df));
    if (kHi < kLo) return empty;
    var bi = kLo;
    for (var k = kLo; k <= kHi; k++) if (power[k] > power[bi]) bi = k;
    var peakFreq = bi * df, peakPower = power[bi];
    var noise = median(power.subarray(1));           // robust floor: median excl. DC
    var snr = peakPower / (noise + 1e-12);
    var harm = harmonicComb(power, df, peakFreq, fmax, opts.nHarmonics || 3);
    // downsample the display arrays to <= 4096 points to keep postMessage light
    var K = power.length;
    var step = Math.max(1, Math.ceil(K / 4096));
    var nOut = Math.ceil(K / step);
    var outF = new Float32Array(nOut), outP = new Float32Array(nOut);
    for (var o = 0; o < nOut; o++) {
      var s = o * step, e = Math.min(s + step, K), mx = 0;
      for (var q = s; q < e; q++) if (power[q] > mx) mx = power[q];   // max-decimate
      outF[o] = s * df;
      outP[o] = mx;
    }
    return { freqs: outF, power: outP, peakFreq: peakFreq, peakPower: peakPower,
             snr: snr, harmonicScore: harm, band: [fmin, fmax],
             nEvents: nEv, fs: fs, nBins: nBins, df: df };
  }

  // ------------------------------------------------------------------
  // Flicker map (port of flicker_map essentials)
  //   args: { x, y, t (seconds, ascending window slice), width, height,
  //           cell?, fs?, fmin, fmax, minEventsPerCell?, onProgress? }
  //   Returns { freq: Float32Array(Gh*Gw) (NaN = none), snr: Float32Array,
  //             count: Float64Array, gw, gh, cell, band, fs }
  // ------------------------------------------------------------------
  var FLICKER_BUDGET = 32e6;      // cells x time-bins elements (mem cap)

  function flickerMap(args) {
    var x = args.x, y = args.y, t = args.t;
    var W = args.width, H = args.height;
    var cell = args.cell || 8;
    var fmin = args.fmin, fmax = args.fmax;
    var fs = args.fs || Math.max(2000, 2.5 * fmax);
    var minEv = args.minEventsPerCell !== undefined ? args.minEventsPerCell : 40;
    var onProgress = args.onProgress || null;
    var gw = Math.ceil(W / cell), gh = Math.ceil(H / cell);
    var nCells = gw * gh;
    var freq = new Float32Array(nCells);
    freq.fill(NaN);
    var snr = new Float32Array(nCells);
    var cnt = new Float64Array(nCells);
    var out = { freq: freq, snr: snr, count: cnt, gw: gw, gh: gh, cell: cell,
                band: [fmin, fmax], fs: fs };
    var n = t.length;
    if (n < 8) return out;
    var lo = t[0], hi = t[n - 1];
    var span = Math.max(hi - lo, 1e-6);
    var nT = Math.floor(span * fs);
    if (nCells * nT > FLICKER_BUDGET && nT > 0) {    // auto-coarsen fs (mem budget)
      fs = Math.max(fs * FLICKER_BUDGET / (nCells * nT), 2.2 * fmax);
      nT = Math.floor(span * fs);
      out.fs = fs;
    }
    if (nT < 8) return out;

    var counts = new Float32Array(nCells * nT);
    for (var i = 0; i < n; i++) {
      var tb = Math.floor((t[i] - lo) * fs);
      if (tb < 0) tb = 0; else if (tb >= nT) tb = nT - 1;
      var cx = Math.min((x[i] / cell) | 0, gw - 1);
      var cy = Math.min((y[i] / cell) | 0, gh - 1);
      var ci = cy * gw + cx;
      counts[ci * nT + tb] += 1;
      cnt[ci] += 1;
    }

    var w = hann(nT);
    var N = nextPow2(nT);
    var re = new Float64Array(N), im = new Float64Array(N);
    var df = fs / N;
    var K = (N >> 1) + 1;
    var kLo = Math.max(1, Math.ceil(fmin / df));
    var kHi = Math.min(K - 1, Math.floor(fmax / df));
    var scratch = new Float64Array(K - 1);
    out.spectrumBins = K - 1;   // non-DC one-sided bins (noise peak/median ~ ln of this)
    var done = 0, active = 0;
    for (var c = 0; c < nCells; c++) if (cnt[c] >= minEv) active++;
    for (var ci2 = 0; ci2 < nCells; ci2++) {
      if (cnt[ci2] < minEv) continue;
      var row = counts.subarray(ci2 * nT, ci2 * nT + nT);
      var mean = 0;
      for (var a = 0; a < nT; a++) mean += row[a];
      mean /= nT;
      re.fill(0); im.fill(0);
      for (var b2 = 0; b2 < nT; b2++) re[b2] = (row[b2] - mean) * w[b2];
      fft(re, im);
      var bi = -1, bp = -1;
      for (var k2 = 1; k2 < K; k2++) {
        var amp = Math.hypot(re[k2], im[k2]) / nT;
        if (k2 < K - 1) amp *= 2;
        var pw = amp * amp;
        scratch[k2 - 1] = pw;
        if (k2 >= kLo && k2 <= kHi && pw > bp) { bp = pw; bi = k2; }
      }
      if (bi >= 0) {
        var noise = median(scratch) + 1e-12;
        freq[ci2] = bi * df;
        snr[ci2] = bp / noise;
      }
      done++;
      if (onProgress && (done & 63) === 0) onProgress(done / Math.max(active, 1));
    }
    if (onProgress) onProgress(1);
    return out;
  }

  // Binary search helpers (np.searchsorted parity) — shared with the viewer.
  function lowerBound(t, v) {
    var lo = 0, hi = t.length;
    while (lo < hi) {
      var mid = (lo + hi) >>> 1;
      if (t[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  function upperBound(t, v) {
    var lo = 0, hi = t.length;
    while (lo < hi) {
      var mid = (lo + hi) >>> 1;
      if (t[mid] <= v) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  var GottLUXAnalysis = {
    fft: fft, nextPow2: nextPow2, hann: hann, median: median,
    regionSpectrum: regionSpectrum, flickerMap: flickerMap,
    lowerBound: lowerBound, upperBound: upperBound
  };

  global.GottLUXAnalysis = GottLUXAnalysis;
  if (typeof module !== "undefined" && module.exports) module.exports = GottLUXAnalysis;
})(typeof self !== "undefined" ? self : this);
