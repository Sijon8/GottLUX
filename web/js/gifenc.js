/*
 * gifenc.js — a small, dependency-free animated-GIF (GIF89a) encoder for the web Studio.
 *
 *   var gif = GottLUXGif.encode(frames, width, height, { delayMs: 40, colors: 128 });
 *   // frames: array of RGBA Uint8ClampedArray (width*height*4); returns a Uint8Array
 *
 * One global palette for the whole clip (median cut over pixels sampled from up to 8
 * frames, so colours do not shimmer from frame to frame), each pixel mapped to its nearest
 * palette entry through a 15-bit (5-5-5) cache, LZW-compressed with the variable code
 * width GIF requires, looping forever (NETSCAPE2.0). Frame delays are in centiseconds, so
 * delayMs is rounded to 10 ms. MIT, part of GottLUX.
 */
(function (global) {
  "use strict";

  function medianCut(samples, nColors) {
    // samples: Uint8Array of RGB triples
    var n = samples.length / 3, idx = new Uint32Array(n);
    for (var i = 0; i < n; i++) idx[i] = i;
    var boxes = [{ lo: 0, hi: n }];
    function rangeOf(b) {
      var mn = [255, 255, 255], mx = [0, 0, 0];
      for (var i = b.lo; i < b.hi; i++) {
        var o = idx[i] * 3;
        for (var c = 0; c < 3; c++) { var v = samples[o + c]; if (v < mn[c]) mn[c] = v; if (v > mx[c]) mx[c] = v; }
      }
      var r = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]], ch = r[0] >= r[1] ? (r[0] >= r[2] ? 0 : 2) : (r[1] >= r[2] ? 1 : 2);
      return { ch: ch, span: r[ch] };
    }
    while (boxes.length < nColors) {
      var best = -1, bi = -1, info = null;
      for (var b = 0; b < boxes.length; b++) {
        if (boxes[b].hi - boxes[b].lo < 2) continue;
        var ri = rangeOf(boxes[b]), score = ri.span * Math.sqrt(boxes[b].hi - boxes[b].lo);
        if (score > best) { best = score; bi = b; info = ri; }
      }
      if (bi < 0 || best <= 0) break;
      var box = boxes[bi], ch = info.ch;
      var sub = Array.prototype.slice.call(idx.subarray(box.lo, box.hi));
      sub.sort(function (a, c) { return samples[a * 3 + ch] - samples[c * 3 + ch]; });
      for (var k = 0; k < sub.length; k++) idx[box.lo + k] = sub[k];
      var mid = box.lo + (sub.length >> 1);
      boxes.splice(bi, 1, { lo: box.lo, hi: mid }, { lo: mid, hi: box.hi });
    }
    var pal = [];
    boxes.forEach(function (b) {
      var s = [0, 0, 0], m = b.hi - b.lo;
      for (var i = b.lo; i < b.hi; i++) { var o = idx[i] * 3; s[0] += samples[o]; s[1] += samples[o + 1]; s[2] += samples[o + 2]; }
      if (m) pal.push([Math.round(s[0] / m), Math.round(s[1] / m), Math.round(s[2] / m)]);
    });
    return pal;
  }

  function buildPalette(frames, nColors) {
    var picks = Math.min(8, frames.length), per = 9000, buf = new Uint8Array(picks * per * 3), p = 0;
    for (var f = 0; f < picks; f++) {
      var fr = frames[Math.floor(f * (frames.length - 1) / Math.max(picks - 1, 1))];
      var npx = fr.length / 4, step = Math.max(1, Math.floor(npx / per));
      for (var i = 0; i < npx && p < buf.length; i += step) {
        buf[p++] = fr[i * 4]; buf[p++] = fr[i * 4 + 1]; buf[p++] = fr[i * 4 + 2];
      }
    }
    var pal = medianCut(buf.subarray(0, p), nColors);
    var size = 2; while (size < pal.length) size <<= 1;
    while (pal.length < size) pal.push([0, 0, 0]);
    return pal;
  }

  function Mapper(pal) {
    var cache = new Int16Array(32768); cache.fill(-1);
    this.map = function (r, g, b) {
      var key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3), v = cache[key];
      if (v >= 0) return v;
      var best = 1e9, bi = 0;
      for (var i = 0; i < pal.length; i++) {
        var dr = pal[i][0] - r, dg = pal[i][1] - g, db = pal[i][2] - b, d = dr * dr * 2 + dg * dg * 4 + db * db * 3;
        if (d < best) { best = d; bi = i; }
      }
      cache[key] = bi;
      return bi;
    };
  }

  function ByteSink() {
    var buf = new Uint8Array(1 << 20), n = 0;
    this.byte = function (b) {
      if (n >= buf.length) { var nb = new Uint8Array(buf.length * 2); nb.set(buf); buf = nb; }
      buf[n++] = b;
    };
    this.bytes = function (arr) { for (var i = 0; i < arr.length; i++) this.byte(arr[i]); };
    this.u16 = function (v) { this.byte(v & 255); this.byte((v >> 8) & 255); };
    this.result = function () { return buf.slice(0, n); };
  }

  // LZW for GIF: variable code width from minCode+1 to 12 bits, clear when the table fills
  function lzw(out, indices, minCode) {
    var clear = 1 << minCode, eoi = clear + 1, next = eoi + 1, size = minCode + 1;
    var cur = 0, shift = 0, block = [], table = new Map();
    function flushBlock() { if (block.length) { out.byte(block.length); out.bytes(block); block = []; } }
    function emit(code) {
      cur |= code << shift; shift += size;
      while (shift >= 8) { block.push(cur & 255); cur >>>= 8; shift -= 8; if (block.length === 255) flushBlock(); }
    }
    out.byte(minCode);
    emit(clear);
    var prefix = indices[0];
    for (var i = 1; i < indices.length; i++) {
      var k = indices[i], key = prefix * 4096 + k, code = table.get(key);
      if (code !== undefined) { prefix = code; continue; }
      emit(prefix);
      if (next === 4096) {
        emit(clear); table.clear(); next = eoi + 1; size = minCode + 1;
      } else {
        if (next >= (1 << size)) size++;
        table.set(key, next++);
      }
      prefix = k;
    }
    emit(prefix);
    emit(eoi);
    if (shift > 0) { block.push(cur & 255); if (block.length === 255) flushBlock(); }
    flushBlock();
    out.byte(0);
  }

  function encode(frames, width, height, opts) {
    opts = opts || {};
    var nColors = Math.max(8, Math.min(256, opts.colors || 128));
    var delayCs = Math.max(2, Math.round((opts.delayMs || 40) / 10));
    var pal = buildPalette(frames, nColors), bits = Math.max(1, Math.log2(pal.length) | 0);
    var mapper = new Mapper(pal), out = new ByteSink();
    out.bytes([71, 73, 70, 56, 57, 97]);                        // "GIF89a"
    out.u16(width); out.u16(height);
    out.byte(0x80 | ((bits - 1) << 4) | (bits - 1)); out.byte(0); out.byte(0);
    pal.forEach(function (c) { out.byte(c[0]); out.byte(c[1]); out.byte(c[2]); });
    out.bytes([0x21, 0xff, 11]); out.bytes([78, 69, 84, 83, 67, 65, 80, 69, 50, 46, 48]);   // NETSCAPE2.0
    out.bytes([3, 1]); out.u16(opts.loop === undefined ? 0 : opts.loop); out.byte(0);
    var idx = new Uint8Array(width * height);
    frames.forEach(function (fr, fi) {
      for (var p = 0, q = 0; p < idx.length; p++, q += 4) idx[p] = mapper.map(fr[q], fr[q + 1], fr[q + 2]);
      out.bytes([0x21, 0xf9, 4, 0x04]); out.u16(delayCs); out.byte(0); out.byte(0);   // disposal 1
      out.byte(0x2c); out.u16(0); out.u16(0); out.u16(width); out.u16(height); out.byte(0);
      lzw(out, idx, Math.max(2, bits));
      if (opts.onProgress) opts.onProgress((fi + 1) / frames.length);
    });
    out.byte(0x3b);
    return out.result();
  }

  global.GottLUXGif = { encode: encode, buildPalette: buildPalette };
  if (typeof module !== "undefined" && module.exports) module.exports = global.GottLUXGif;
})(typeof self !== "undefined" ? self : this);
