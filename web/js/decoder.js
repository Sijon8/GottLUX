/*
 * decoder.js — GottLUX Web port of gottlux/io/decode.py (DECODER_VERSION 5).
 *
 * Pure vanilla JS, loadable from the main thread (<script>), a Web Worker
 * (importScripts) or Node (require) — it attaches `GottLUXDecoder` to the
 * global scope and also exports via module.exports when present.
 *
 * Formats:
 *   EVT2.1 — 64-bit words, read as two little-endian Uint32 halves (no BigInt
 *            in the hot loop). Type nibble in bits 60..63 (high half >>> 28),
 *            TIME_HIGH accumulator (running max, unit 64 us), 32-px vector
 *            mask expansion from the low half.
 *   EVT2.0 — 32-bit words, single CD events.
 *   EVT3   — 16-bit stateful words (ADDR_Y / ADDR_X / VECT_BASE_X / VECT_12 /
 *            VECT_8 / TIME_LOW / TIME_HIGH with 12-bit epoch wrap).
 *
 * Output of decode(buffer):
 *   { x: Uint16Array, y: Uint16Array, p: Uint8Array,
 *     t: Float64Array (seconds, zero-based, sorted), t0_us, width, height,
 *     n, fmt, meta }
 *
 * Parity notes vs decode.py:
 *   - header parse, format detection, geometry, TIME_HIGH running max,
 *     "CD before first TIME_HIGH is junk", strip_preroll, geometry inference
 *     and x/y clipping are ported 1:1.
 *   - tie ORDER of events sharing one timestamp can differ from NumPy's
 *     bit-major + stable-sort order; every count / sum / span is identical
 *     (verified against Python-decoded fixtures by selftest.html).
 */
(function (global) {
  "use strict";

  var END_MARK = [0x25, 0x20, 0x65, 0x6e, 0x64, 0x0a]; // "% end\n"

  // ------------------------------------------------------------------
  // Header / format / geometry  (decode.parse_header / detect_format / geometry)
  // ------------------------------------------------------------------
  function parseHeader(bytes) {
    var lim = Math.min(bytes.length, 8192);
    var end = -1;
    outer:
    for (var i = 0; i + END_MARK.length <= lim; i++) {
      for (var j = 0; j < END_MARK.length; j++) {
        if (bytes[i + j] !== END_MARK[j]) continue outer;
      }
      end = i;
      break;
    }
    if (end < 0) throw new Error("No '% end' header terminator found (not a Prophesee .raw?)");
    var head = "";
    for (var k = 0; k < end; k++) {
      var c = bytes[k];
      head += c < 128 ? String.fromCharCode(c) : "�";
    }
    var meta = {};
    var lines = head.split("\n");
    for (var li = 0; li < lines.length; li++) {
      var line = lines[li].trim();
      if (line.charAt(0) !== "%") continue;
      var body = line.slice(1).trim();
      var sp = body.indexOf(" ");
      if (sp > 0) meta[body.slice(0, sp)] = body.slice(sp + 1);
    }
    return { meta: meta, offset: end + END_MARK.length };
  }

  function detectFormat(meta) {
    var raw = (String(meta.format || "") + " " + String(meta.evt || ""))
      .toLowerCase().replace(/ /g, "").replace(/\./g, "");
    if (raw.indexOf("evt3") >= 0) return "evt3";
    if (raw.indexOf("evt21") >= 0) return "evt21";
    if (raw.indexOf("evt2") >= 0) return "evt2";   // EVT2.0 (also matches 'evt20')
    return "evt21";                                // legacy default
  }

  function geometry(meta) {
    var w = null, h = null;
    var toks = String(meta.format || "").split(";");
    for (var i = 0; i < toks.length; i++) {
      if (toks[i].indexOf("width=") === 0) w = parseInt(toks[i].split("=")[1], 10);
      else if (toks[i].indexOf("height=") === 0) h = parseInt(toks[i].split("=")[1], 10);
    }
    if ((w === null || h === null || isNaN(w) || isNaN(h)) && meta.geometry) {
      var g = String(meta.geometry).toLowerCase().split("x");
      if (g.length === 2) { w = parseInt(g[0], 10); h = parseInt(g[1], 10); }
    }
    if (w === null || isNaN(w)) w = null;
    if (h === null || isNaN(h)) h = null;
    return [w, h];
  }

  function popcount32(v) {
    v = v - ((v >>> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
    return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
  }

  // ------------------------------------------------------------------
  // EVT2.1 — 64-bit words as (lo, hi) Uint32 pairs
  // ------------------------------------------------------------------
  function decodeEvt21(u, nWords, progress) {
    // pass 1: count expanded events so arrays are allocated once
    var hi = 0, seen = false, count = 0, i, h, typ, raw;
    for (i = 0; i < nWords; i++) {
      h = u[2 * i + 1];
      typ = h >>> 28;
      if (typ === 0x8) {
        raw = h & 0x0fffffff;
        if (raw > hi) hi = raw;
        seen = true;
      } else if (typ <= 1) {
        if (seen) count += popcount32(u[2 * i]);
      }
      if (progress && (i & 0x1fffff) === 0) progress(0.05 + 0.25 * (i / nWords), "scanning");
    }
    var x = new Uint16Array(count), y = new Uint16Array(count),
        p = new Uint8Array(count), t = new Float64Array(count);
    // pass 2: fill
    hi = 0; seen = false;
    var k = 0, sorted = true, lastT = -Infinity;
    for (i = 0; i < nWords; i++) {
      h = u[2 * i + 1];
      typ = h >>> 28;
      if (typ === 0x8) {
        raw = h & 0x0fffffff;
        if (raw > hi) hi = raw;
        seen = true;
      } else if (typ <= 1 && seen) {
        var m = u[2 * i] >>> 0;
        if (m !== 0) {
          var tUs = hi * 64 + ((h >>> 22) & 0x3f);
          var xb = (h >>> 11) & 0x7ff;
          var yb = h & 0x7ff;
          if (tUs < lastT) sorted = false;
          lastT = tUs;
          while (m !== 0) {
            var b = 31 - Math.clz32(m & -m);   // lowest set bit
            x[k] = xb + b; y[k] = yb; p[k] = typ; t[k] = tUs;
            k++;
            m &= m - 1;
          }
        }
      }
      if (progress && (i & 0x1fffff) === 0) progress(0.30 + 0.55 * (i / nWords), "decoding");
    }
    return { x: x, y: y, p: p, t: t, sorted: sorted };
  }

  // ------------------------------------------------------------------
  // EVT2.0 — 32-bit words, single CD events
  // ------------------------------------------------------------------
  function decodeEvt2(u, nWords, progress) {
    var hi = 0, seen = false, count = 0, i, w, typ, raw;
    for (i = 0; i < nWords; i++) {
      w = u[i];
      typ = w >>> 28;
      if (typ === 0x8) {
        raw = w & 0x0fffffff;
        if (raw > hi) hi = raw;
        seen = true;
      } else if (typ <= 1 && seen) count++;
      if (progress && (i & 0x3fffff) === 0) progress(0.05 + 0.25 * (i / nWords), "scanning");
    }
    var x = new Uint16Array(count), y = new Uint16Array(count),
        p = new Uint8Array(count), t = new Float64Array(count);
    hi = 0; seen = false;
    var k = 0, sorted = true, lastT = -Infinity;
    for (i = 0; i < nWords; i++) {
      w = u[i];
      typ = w >>> 28;
      if (typ === 0x8) {
        raw = w & 0x0fffffff;
        if (raw > hi) hi = raw;
        seen = true;
      } else if (typ <= 1 && seen) {
        var tUs = hi * 64 + ((w >>> 22) & 0x3f);
        if (tUs < lastT) sorted = false;
        lastT = tUs;
        x[k] = (w >>> 11) & 0x7ff;
        y[k] = w & 0x7ff;
        p[k] = typ & 1;
        t[k] = tUs;
        k++;
      }
      if (progress && (i & 0x3fffff) === 0) progress(0.30 + 0.55 * (i / nWords), "decoding");
    }
    return { x: x, y: y, p: p, t: t, sorted: sorted };
  }

  // ------------------------------------------------------------------
  // EVT3 — 16-bit stateful words (full state machine port of chunk_evt3)
  // ------------------------------------------------------------------
  function decodeEvt3(u, nWords, progress) {
    var i, w, typ, data, count = 0;
    for (i = 0; i < nWords; i++) {
      typ = u[i] >>> 12;
      if (typ === 0x2) count++;
      else if (typ === 0x4) count += popcount32(u[i] & 0xfff);
      else if (typ === 0x5) count += popcount32(u[i] & 0xff);
      if (progress && (i & 0x3fffff) === 0) progress(0.05 + 0.25 * (i / nWords), "scanning");
    }
    var x = new Uint16Array(count), y = new Uint16Array(count),
        p = new Uint8Array(count), t = new Float64Array(count);
    // state (init_state('evt3')): epoch, last_raw, th_cum, tl, y, base_x_next, base_pol
    var epoch = 0, lastRaw = 0, th = 0, tl = 0, yRow = 0, baseX = 0, basePol = 0;
    var k = 0, sorted = true, lastT = -Infinity, tUs, m, b;
    for (i = 0; i < nWords; i++) {
      w = u[i];
      typ = w >>> 12;
      data = w & 0x0fff;
      switch (typ) {
        case 0x8:                                   // TIME_HIGH (12-bit, epoch-wrapping)
          if (data < lastRaw) epoch++;
          lastRaw = data;
          th = epoch * 4096 + data;
          break;
        case 0x6:                                   // TIME_LOW
          tl = data;
          break;
        case 0x0:                                   // ADDR_Y
          yRow = data & 0x7ff;
          break;
        case 0x2:                                   // ADDR_X: single event
          tUs = th * 4096 + tl;
          if (tUs < lastT) sorted = false;
          lastT = tUs;
          x[k] = data & 0x7ff; y[k] = yRow; p[k] = (data >>> 11) & 1; t[k] = tUs;
          k++;
          break;
        case 0x3:                                   // VECT_BASE_X
          baseX = data & 0x7ff;
          basePol = (data >>> 11) & 1;
          break;
        case 0x4:                                   // VECT_12
          m = data & 0xfff;
          if (m !== 0) {
            tUs = th * 4096 + tl;
            if (tUs < lastT) sorted = false;
            lastT = tUs;
            while (m !== 0) {
              b = 31 - Math.clz32(m & -m);
              x[k] = baseX + b; y[k] = yRow; p[k] = basePol; t[k] = tUs;
              k++;
              m &= m - 1;
            }
          }
          baseX += 12;                              // base x advances by the vector width
          break;
        case 0x5:                                   // VECT_8
          m = data & 0xff;
          if (m !== 0) {
            tUs = th * 4096 + tl;
            if (tUs < lastT) sorted = false;
            lastT = tUs;
            while (m !== 0) {
              b = 31 - Math.clz32(m & -m);
              x[k] = baseX + b; y[k] = yRow; p[k] = basePol; t[k] = tUs;
              k++;
              m &= m - 1;
            }
          }
          baseX += 8;
          break;
      }
      if (progress && (i & 0x3fffff) === 0) progress(0.30 + 0.55 * (i / nWords), "decoding");
    }
    return { x: x, y: y, p: p, t: t, sorted: sorted };
  }

  // ------------------------------------------------------------------
  // strip_preroll — identical logic to decode.strip_preroll (t in integer us)
  // ------------------------------------------------------------------
  function prerollDrop(t) {
    var n = t.length;
    if (n < 200) return 0;
    var span = t[n - 1] - t[0];
    if (span <= 0) return 0;
    var look = Math.max(10, Math.floor(n / 500));
    var jBest = 0, gBest = -Infinity;
    for (var j = 0; j < look; j++) {
      var g = t[j + 1] - t[j];
      if (g > gBest) { gBest = g; jBest = j; }
    }
    if (gBest >= 3000000 && gBest >= Math.floor(span / 2)) return jBest + 1;
    return 0;
  }

  function stableSortByT(x, y, p, t) {
    var n = t.length;
    var idx = new Uint32Array(n);
    for (var i = 0; i < n; i++) idx[i] = i;
    // Array#sort on a plain array copy of indices, ties broken by index => stable
    var arr = Array.prototype.slice.call(idx);
    arr.sort(function (a, b) { return (t[a] - t[b]) || (a - b); });
    var x2 = new Uint16Array(n), y2 = new Uint16Array(n),
        p2 = new Uint8Array(n), t2 = new Float64Array(n);
    for (var j = 0; j < n; j++) {
      var s = arr[j];
      x2[j] = x[s]; y2[j] = y[s]; p2[j] = p[s]; t2[j] = t[s];
    }
    return { x: x2, y: y2, p: p2, t: t2 };
  }

  // ------------------------------------------------------------------
  // Full decode of an ArrayBuffer  (decode.decode equivalent)
  // ------------------------------------------------------------------
  function decode(buffer, progress) {
    var bytes = new Uint8Array(buffer);
    var hdr = parseHeader(bytes);
    var meta = hdr.meta, off = hdr.offset;
    var fmt = detectFormat(meta);
    var g = geometry(meta);
    var width = g[0], height = g[1];
    if (progress) progress(0.02, "header");

    var wordBytes = fmt === "evt3" ? 2 : (fmt === "evt2" ? 4 : 8);
    var payloadLen = bytes.length - off;
    var nWords = Math.floor(payloadLen / wordBytes);
    var align = fmt === "evt3" ? 2 : 4;
    var view;
    if (off % align === 0) {
      view = fmt === "evt3"
        ? new Uint16Array(buffer, off, nWords)
        : new Uint32Array(buffer, off, nWords * (wordBytes / 4));
    } else {                                       // unaligned header end: copy payload
      var copy = buffer.slice(off, off + nWords * wordBytes);
      view = fmt === "evt3" ? new Uint16Array(copy) : new Uint32Array(copy);
    }

    var d;
    if (fmt === "evt21") d = decodeEvt21(view, nWords, progress);
    else if (fmt === "evt2") d = decodeEvt2(view, nWords, progress);
    else d = decodeEvt3(view, nWords, progress);

    var x = d.x, y = d.y, p = d.p, t = d.t;       // t in integer microseconds here
    var n = t.length;
    if (n === 0) {
      return { x: x, y: y, p: p, t: t, t0_us: 0, n: 0,
               width: width || 320, height: height || 320, fmt: fmt, meta: meta };
    }
    if (!d.sorted) {                               // rare: stable time sort (argsort parity)
      if (progress) progress(0.88, "sorting");
      var s = stableSortByT(x, y, p, t);
      x = s.x; y = s.y; p = s.p; t = s.t;
    }
    if (progress) progress(0.9, "finalizing");

    var drop = prerollDrop(t);
    if (drop > 0) {
      x = x.slice(drop); y = y.slice(drop); p = p.slice(drop); t = t.slice(drop);
      n = t.length;
    }
    if (width === null) {
      var mx = 0;
      for (var i1 = 0; i1 < n; i1++) if (x[i1] > mx) mx = x[i1];
      width = mx + 1;
    }
    if (height === null) {
      var my = 0;
      for (var i2 = 0; i2 < n; i2++) if (y[i2] > my) my = y[i2];
      height = my + 1;
    }
    var wm = width - 1, hm = height - 1;
    for (var i3 = 0; i3 < n; i3++) {               // np.clip(x, 0, width-1) parity
      if (x[i3] > wm) x[i3] = wm;
      if (y[i3] > hm) y[i3] = hm;
    }
    var t0 = t[0];
    for (var i4 = 0; i4 < n; i4++) t[i4] = (t[i4] - t0) / 1e6;   // zero-base -> seconds
    if (progress) progress(1.0, "done");
    return { x: x, y: y, p: p, t: t, t0_us: t0, n: n,
             width: width, height: height, fmt: fmt, meta: meta };
  }

  var GottLUXDecoder = {
    parseHeader: parseHeader,
    detectFormat: detectFormat,
    geometry: geometry,
    popcount32: popcount32,
    decode: decode,
    FORMAT_LABEL: { evt21: "EVT2.1", evt2: "EVT2.0", evt3: "EVT3" },
    DECODER_VERSION: 5
  };

  global.GottLUXDecoder = GottLUXDecoder;
  if (typeof module !== "undefined" && module.exports) module.exports = GottLUXDecoder;
})(typeof self !== "undefined" ? self : this);
