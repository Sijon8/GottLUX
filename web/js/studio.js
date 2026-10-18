/*
 * studio.js — the web Studio page: load a clip, pick a look, frame the shot, export a GIF
 * (built-in encoder, js/gifenc.js) or a video (MediaRecorder: MP4 where the browser can,
 * else WebM). The shot lives in the URL (#shot=…) so a link reopens the exact shot.
 * Rendering: js/looks.js (a port of gottlux/studio). Method notes: docs/STUDIO.md.
 */
(function () {
  "use strict";
  var S = window.GottLUXStudio, SPEC = S.SPEC;
  var $ = function (id) { return document.getElementById(id); };

  // hosted-clip recipes: spans/crops picked on the sample clips (see docs/STUDIO.md)
  var WIDE = [0, 40, 320, 280];
  var RECIPES = [
    { name: "Wing-beat stroboscope", clip: "hummingbird_fight.raw",
      shot: { look: "strobe", t0: 0.60, t1: 1.05, crop: [0, 110, 68, 178], fps: 16, width: 360,
              bloom: 0.8, params: { bins: 24, cycles: 1, width: 0.09 } } },
    { name: "Dart with light trails", clip: "hummingbird_fight.raw",
      shot: { look: "trails", t0: 0.15, t1: 0.70, crop: WIDE, speed: 0.12, bloom: 0.7,
              legend: true, params: { tau_ms: 5 } } },
    { name: "3-D orbit of one second", clip: "hummingbird_fight.raw",
      shot: { look: "spacetime", t0: 0.15, t1: 1.05, speed: 0.25, speed_label: false,
              params: { depth_ms: 900, freeze: true, orbit_deg: 360, time_scale: 2.2,
                        point_px: 1, brightness: 1.6, elevation: 18 } } },
    { name: "Waterfall of two birds", clip: "hummingbird_fight.raw",
      shot: { look: "waterfall", t0: 0.0, t1: 1.2, crop: WIDE, speed: 0.25, width: 400,
              params: { history_ms: 600, rows: 180 } } },
    { name: "Light painting", clip: "hummingbird_fight.raw",
      shot: { look: "paint", t0: 0.0, t1: 2.1, crop: WIDE, speed: 0.6, bloom: 0.5, speed_label: false } },
    { name: "Wing-beat hue", clip: "hummingbird_fight.raw",
      shot: { look: "flicker", t0: 0.2, t1: 1.2, crop: WIDE, speed: 0.25, fps: 20,
              params: { band_lo: 20, band_hi: 120 } } },
    { name: "Rotor hue (quadcopter)", clip: "quadcopter_staring.raw",
      shot: { look: "flicker", t0: 0.0, t1: 1.1, speed: 0.35, fps: 20, width: 420,
              params: { band_lo: 90, band_hi: 700, window_ms: 200 } } },
    // the hosted sweeps are 1.5 revolutions — too short to measure the spin, so it is given
    { name: "Radar sweep (rotating sensor)", clip: "quadcopter_rotating_a.raw",
      shot: { look: "radar", t0: 0.0, t1: 1.46, speed: 0.5, fps: 25, width: 420, bloom: 0.4,
              params: { period_s: 0.9388, persistence_ms: 300 } } }
  ];

  var state = { rec: null, clip: null, shot: S.defaultShot(), renderer: null, frame: 0,
                playing: true, cropMode: false, evKey: null, events: null, exporting: false,
                rate: null, manifest: [] };

  // ------------------------------------------------------------------ loading
  var worker = new Worker("js/decode_worker.js"), job = 0, pending = null;
  worker.onmessage = function (e) {
    var d = e.data;
    if (!pending || d.id !== pending.id) return;
    if (d.type === "progress") overlay("decoding — " + (d.stage || ""), 0.35 + 0.65 * d.frac);
    else if (d.type === "done") { var p = pending; pending = null; hideOverlay(); setRecording(d.result, p); }
    else if (d.type === "error") { pending = null; overlay("decode failed: " + d.message, 0); }
  };
  function overlay(msg, frac) {
    $("load-overlay").classList.remove("hidden");
    $("load-stage").textContent = msg;
    $("load-bar").style.width = (Math.max(0, Math.min(1, frac)) * 100).toFixed(1) + "%";
  }
  function hideOverlay() { $("load-overlay").classList.add("hidden"); }

  function loadBuffer(buf, info) {
    overlay("decoding…", 0.35);
    pending = { id: ++job, info: info };
    worker.postMessage({ id: pending.id, buffer: buf }, [buf]);
  }
  async function fetchClip(file, then) {
    overlay("downloading " + file, 0);
    var resp = await fetch("data/" + file);
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    var total = +resp.headers.get("Content-Length") || 0, reader = resp.body.getReader(), chunks = [], got = 0;
    for (;;) {
      var r = await reader.read();
      if (r.done) break;
      chunks.push(r.value); got += r.value.length;
      overlay("downloading " + file + " — " + (got / 1048576).toFixed(1) + " MB", total ? 0.35 * got / total : 0.15);
    }
    var buf = new Uint8Array(got), o = 0;
    chunks.forEach(function (c) { buf.set(c, o); o += c.length; });
    loadBuffer(buf.buffer, { file: file, then: then });
  }

  function setRecording(res, p) {
    state.rec = { x: res.x, y: res.y, p: res.p, t: res.t, width: res.width, height: res.height };
    state.clip = p.info.file || null;
    state.evKey = null;
    var n = res.t.length, dur = n ? res.t[n - 1] : 0;
    var bins = 600, rate = new Float32Array(bins);
    for (var i = 0; i < n; i++) rate[Math.min(bins - 1, Math.floor(res.t[i] / Math.max(dur, 1e-9) * bins))] += 1;
    var mx = 0; for (var b = 0; b < bins; b++) mx = Math.max(mx, rate[b]);
    for (var c = 0; c < bins; c++) rate[c] /= mx || 1;
    state.rate = rate; state.dur = dur;
    $("view-hint").hidden = true;
    ["btn-play", "btn-crop", "btn-full", "btn-export", "btn-share", "btn-recipe"].forEach(function (id) { $(id).disabled = false; });
    if (p.info.then) applyShot(p.info.then);
    else {
      if (state.shot.t0 === null || state.shot.t1 > dur) { state.shot.t0 = 0; state.shot.t1 = Math.min(1, dur); }
      syncControls(); rebuild();
    }
  }

  function loadManifest() {
    fetch("data/manifest.json").then(function (r) { if (!r.ok) throw new Error(); return r.json(); })
      .then(function (man) {
        state.manifest = man.clips || [];
        $("manifest-note").textContent = "";
        state.manifest.forEach(function (clip) {
          var card = document.createElement("div");
          card.className = "clip-card";
          card.innerHTML = '<div class="row1"><span class="title"></span><span class="badge"></span></div><div class="desc"></div>';
          card.querySelector(".title").textContent = clip.title || clip.file;
          card.querySelector(".badge").textContent = clip.regime || "clip";
          if (clip.regime === "rotating") card.querySelector(".badge").classList.add("rotating");
          card.querySelector(".desc").textContent = clip.description || "";
          card.addEventListener("click", function () {
            document.querySelectorAll(".clip-card.active").forEach(function (c) { c.classList.remove("active"); });
            card.classList.add("active");
            state.shot.t0 = null; state.shot.t1 = null; state.shot.crop = null;
            fetchClip(clip.file).catch(function (e) { overlay("download failed: " + e.message, 0); });
          });
          $("clip-list").appendChild(card);
        });
        fromHash();
      })
      .catch(function () { $("manifest-note").textContent = "No sample manifest (running locally?) — drop a .raw file instead."; });
  }

  $("dropzone").addEventListener("click", function () { $("file-input").click(); });
  $("file-input").addEventListener("change", function () { var f = $("file-input").files[0]; if (f) openFile(f); });
  document.addEventListener("dragover", function (e) { e.preventDefault(); });
  document.addEventListener("drop", function (e) { e.preventDefault(); if (e.dataTransfer.files.length) openFile(e.dataTransfer.files[0]); });
  function openFile(f) {
    state.shot.t0 = null; state.shot.t1 = null; state.shot.crop = null;
    f.arrayBuffer().then(function (b) { loadBuffer(b, { file: null, name: f.name }); });
  }

  // recipes
  RECIPES.forEach(function (rc) {
    var b = document.createElement("button");
    b.className = "recipe"; b.textContent = rc.name;
    b.addEventListener("click", function () {
      var shot = Object.assign(S.defaultShot(), rc.shot, { params: Object.assign({}, rc.shot.params || {}) });
      if (state.clip === rc.clip && state.rec) applyShot(shot);
      else fetchClip(rc.clip, shot).catch(function (e) { overlay("download failed: " + e.message, 0); });
    });
    $("recipe-list").appendChild(b);
  });

  // ------------------------------------------------------------------ controls
  Object.keys(SPEC.looks).forEach(function (name) {
    var o = document.createElement("option"); o.value = name;
    o.textContent = SPEC.looks[name].title + (SPEC.looks[name].heavy ? " (slower)" : "");
    $("look").appendChild(o);
  });
  Object.keys(SPEC.palettes).forEach(function (n) { var o = document.createElement("option"); o.value = o.textContent = n; $("palette").appendChild(o); });
  Object.keys(SPEC.presets).forEach(function (n) {
    var p = SPEC.presets[n], o = document.createElement("option");
    o.value = n; o.textContent = n + " · " + p[0].toUpperCase() + " " + p[1] + "px " + p[2] + "fps";
    $("preset").appendChild(o);
  });
  var videoMime = ["video/mp4;codecs=avc1.42E01E", "video/mp4", "video/webm;codecs=vp9", "video/webm"]
    .filter(function (m) { return window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(m); });
  var formats = ["gif"];
  if (videoMime.some(function (m) { return m.indexOf("mp4") >= 0; })) formats.push("mp4");
  if (videoMime.some(function (m) { return m.indexOf("webm") >= 0; })) formats.push("webm");
  formats.forEach(function (f) { var o = document.createElement("option"); o.value = o.textContent = f; $("format").appendChild(o); });
  $("preset").value = "gif";

  function buildParams() {
    var host = $("params"); host.innerHTML = "";
    var look = SPEC.looks[state.shot.look], P = S.resolve(state.shot.look, state.shot.params);
    $("look-desc").textContent = look.description;
    look.params.forEach(function (p) {
      var row = document.createElement("label"), input;
      row.title = p.help || "";
      row.appendChild(document.createTextNode(p.label + (p.unit ? " (" + p.unit + ")" : "") + " "));
      if (p.kind === "choice") {
        input = document.createElement("select");
        p.choices.forEach(function (c) { var o = document.createElement("option"); o.value = o.textContent = c; input.appendChild(o); });
        input.value = P[p.key];
      } else if (p.kind === "bool") {
        row.className = "check"; input = document.createElement("input"); input.type = "checkbox"; input.checked = P[p.key];
      } else {
        input = document.createElement("input"); input.type = "number";
        input.min = p.lo; input.max = p.hi; input.step = p.step || (p.kind === "int" ? 1 : "any"); input.value = P[p.key];
      }
      input.addEventListener(p.kind === "float" || p.kind === "int" ? "input" : "change", function () {
        var v = p.kind === "bool" ? input.checked : (p.kind === "choice" ? input.value : Number(input.value));
        state.shot.params[p.key] = v; changed();
      });
      if (p.kind === "bool") row.insertBefore(input, row.firstChild);
      row.appendChild(input);
      host.appendChild(row);
    });
  }

  function syncControls() {
    var s = state.shot;
    $("look").value = s.look; buildParams();
    $("t0").value = s.t0 === null ? "" : (+s.t0).toFixed(3); $("t1").value = s.t1 === null ? "" : (+s.t1).toFixed(3);
    var best = null; Array.prototype.forEach.call($("speed").options, function (o) {
      if (best === null || Math.abs(Math.log(o.value / s.speed)) < Math.abs(Math.log(best.value / s.speed))) best = o;
    });
    $("speed").value = best.value;
    $("denoise").checked = s.denoise; $("palette").value = s.palette;
    $("bloom").value = s.bloom; $("bloom-v").textContent = s.bloom; $("gamma").value = s.gamma; $("gamma-v").textContent = s.gamma;
    $("title").value = s.title;
    ["stamp", "speed_label", "legend", "rate_strip", "pingpong"].forEach(function (k) { $(k).checked = !!s[k]; });
    $("format").value = formats.indexOf(s.format) >= 0 ? s.format : "gif";
    $("width").value = s.width; $("fps").value = s.fps;
    $("crop-readout").textContent = "crop: " + (s.crop ? "x " + s.crop[0] + "–" + s.crop[2] + ", y " + s.crop[1] + "–" + s.crop[3] : "full frame");
  }

  function readControls() {
    var s = state.shot;
    s.look = $("look").value;
    s.t0 = $("t0").value === "" ? null : Number($("t0").value);
    s.t1 = $("t1").value === "" ? null : Number($("t1").value);
    s.speed = Number($("speed").value); s.denoise = $("denoise").checked; s.palette = $("palette").value;
    s.bloom = Number($("bloom").value); s.gamma = Number($("gamma").value); s.title = $("title").value;
    ["stamp", "speed_label", "legend", "rate_strip", "pingpong"].forEach(function (k) { s[k] = $(k).checked; });
    s.format = $("format").value; s.width = Math.max(64, Number($("width").value) || 480); s.fps = Math.max(1, Number($("fps").value) || 25);
    $("bloom-v").textContent = s.bloom; $("gamma-v").textContent = s.gamma;
  }

  ["t0", "t1", "speed", "denoise", "palette", "bloom", "gamma", "title", "stamp", "speed_label",
   "legend", "rate_strip", "pingpong", "format", "width", "fps"].forEach(function (id) {
    $(id).addEventListener(id === "title" || id === "bloom" || id === "gamma" ? "input" : "change", function () { readControls(); changed(); });
  });
  $("look").addEventListener("change", function () { state.shot.look = $("look").value; state.shot.params = {}; buildParams(); changed(); });
  $("preset").addEventListener("change", function () {
    var p = SPEC.presets[$("preset").value];
    state.shot.format = formats.indexOf(p[0]) >= 0 ? p[0] : (p[0] === "gif" ? "gif" : formats[formats.length - 1]);
    state.shot.width = p[1]; state.shot.fps = p[2]; syncControls(); changed();
  });

  function applyShot(shot) {
    state.shot = Object.assign(S.defaultShot(), shot);
    state.shot.params = Object.assign({}, shot.params || {});
    syncControls(); rebuild(); writeHash();
  }

  var rebuildTimer = null;
  function changed() { clearTimeout(rebuildTimer); rebuildTimer = setTimeout(rebuild, 200); writeHash(); }

  // ------------------------------------------------------------------ preview
  var preview = $("preview"), pctx = preview.getContext("2d"), work = document.createElement("canvas");

  function previewShot() {
    var s = Object.assign({}, state.shot), box = $("studio-view").clientWidth || 480;
    s.width = Math.min(s.width, 640, Math.max(240, box));
    return s;
  }

  // scale the preview canvas (CSS px) to fit the stage width and ~2/3 of the window height
  function fitPreview() {
    var box = $("studio-view"), W = preview.width, H = preview.height;
    var k = Math.min((box.clientWidth - 2) / W, Math.max(200, window.innerHeight * 0.66) / H);
    preview.style.width = Math.floor(W * k) + "px"; preview.style.height = Math.floor(H * k) + "px";
  }

  function rebuild() {
    if (!state.rec) return;
    var s = previewShot(), look = S.LOOKS[s.look], P = S.resolve(s.look, s.params);
    var mem = look.memory ? look.memory(P) : 0;
    var key = JSON.stringify([s.t0, s.t1, s.crop, s.denoise, s.baf_us, mem]);
    try {
      if (key !== state.evKey) {
        var t = state.rec.t, rt1 = t.length ? t[t.length - 1] : 0;
        var t0 = s.t0 === null ? 0 : Math.max(0, Math.min(s.t0, rt1));
        var t1 = s.t1 === null ? Math.min(t0 + 1, rt1) : Math.max(t0, Math.min(s.t1, rt1));
        state.events = S.prepareEvents(state.rec, s, t0, t1, mem); state.evKey = key;
      }
      state.renderer = new S.Renderer(state.rec, s, state.events);
    } catch (err) { $("readout").textContent = "render failed: " + err.message; return; }
    var n = state.renderer.length();
    $("frame").max = Math.max(0, n - 1);
    state.frame = Math.min(state.frame, n - 1);
    drawStrip(); show();
  }

  // draw frame i of renderer r onto ctx at the renderer's output size (+ strip), decorated
  function paint(r, i, canvas) {
    var ctx = canvas.getContext("2d"), fr = r.draw(i), shot = r.shot;
    var W = r.outW, H = r.outH, Ht = H + r.stripH;
    if (canvas.width !== W || canvas.height !== Ht) { canvas.width = W; canvas.height = Ht; }
    work.width = fr.w; work.height = fr.h;
    work.getContext("2d").putImageData(new ImageData(fr.data, fr.w, fr.h), 0, 0);
    var bg = r.colors.bg;
    ctx.fillStyle = "rgb(" + Math.round(bg[0] * 153) + "," + Math.round(bg[1] * 153) + "," + Math.round(bg[2] * 153) + ")";
    ctx.fillRect(0, 0, W, Ht);
    ctx.imageSmoothingEnabled = shot.resample !== "nearest";
    ctx.drawImage(work, 0, 0, W, H);
    if (shot.bloom > 0 && "filter" in ctx) {
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, W, H); ctx.clip();
      ctx.globalCompositeOperation = "screen"; ctx.globalAlpha = Math.min(1, shot.bloom);
      ctx.filter = "blur(" + Math.max(1.5, W / 140).toFixed(1) + "px)";
      ctx.drawImage(canvas, 0, 0, W, H, 0, 0, W, H);
      ctx.restore();
    }
    var dark = (bg[0] + bg[1] + bg[2]) / 3 < 0.5, ink = dark ? "#f0efec" : "#141413", dim = dark ? "#b4b2aa" : "#52514e";
    var fs = Math.max(11, Math.floor(W / 34)), pad = Math.max(6, fs >> 1);
    function text(str, x, y, f, col, align) {
      ctx.font = f + "px system-ui, 'Segoe UI', sans-serif"; ctx.textAlign = align || "left"; ctx.textBaseline = "top";
      ctx.fillStyle = dark ? "#000" : "#fff"; ctx.fillText(str, x + 1, y + 1);
      ctx.fillStyle = col; ctx.fillText(str, x, y);
    }
    var y = pad;
    if (shot.stamp) { text(r.stamp(fr.k), pad, y, fs, ink); y += Math.round(fs * 1.25); }
    if (shot.speed_label && shot.look !== "strobe") text(S.speedText(shot.speed), pad, y, Math.round(fs * 0.78), dim);
    if (shot.title) text(shot.title, W - pad, pad, fs, ink, "right");
    if (shot.legend && r.usesPolarity()) {
      var sw = Math.max(8, fs >> 1), sf = Math.round(fs * 0.78), x = W - pad, yb = H - pad - sw;
      [["OFF", "off"], ["ON", "on"]].forEach(function (e) {
        ctx.font = sf + "px system-ui, sans-serif"; var tw = ctx.measureText(e[0]).width;
        x -= tw; text(e[0], x, yb - 3, sf, dim); x -= sw + 4;
        var c = r.colors[e[1]]; ctx.fillStyle = "rgb(" + (c[0] * 255 | 0) + "," + (c[1] * 255 | 0) + "," + (c[2] * 255 | 0) + ")";
        ctx.fillRect(x, yb, sw, sw); x -= pad;
      });
    }
    if (r.stripH) {
      var rate = stripRate(r), hh = r.stripH - 4, oc = r.colors.on;
      ctx.fillStyle = "rgb(" + (oc[0] * 255 | 0) + "," + (oc[1] * 255 | 0) + "," + (oc[2] * 255 | 0) + ")";
      for (var xi = 0; xi < rate.length; xi++) { var hgt = Math.max(1, Math.round(rate[xi] * hh)); if (rate[xi] > 0) ctx.fillRect(xi, H + 2 + hh - hgt, 1, hgt); }
      var px = Math.round((fr.tau - r.t0) / Math.max(r.t1 - r.t0, 1e-9) * (W - 1));
      ctx.fillStyle = ink; ctx.fillRect(px - 1, H, 2, r.stripH);
    }
    return fr;
  }

  function stripRate(r) {
    if (r._rate) return r._rate;
    var ev = r.ev, sp = ev.span(r.t0, r.t1), bins = r.outW, a = new Float32Array(bins), mx = 0;
    for (var i = sp[0]; i < sp[1]; i++) { var b = Math.min(bins - 1, Math.floor((ev.t[i] - r.t0) / Math.max(r.t1 - r.t0, 1e-9) * bins)); a[b] += 1; }
    for (var j = 0; j < bins; j++) mx = Math.max(mx, a[j]);
    for (var k = 0; k < bins; k++) a[k] /= mx || 1;
    r._rate = a; return a;
  }

  function show() {
    var r = state.renderer; if (!r) return;
    if (state.cropMode) { drawOverview(); fitPreview(); return; }
    var fr = paint(r, state.frame, preview);
    fitPreview();
    $("frame").value = state.frame;
    var s = state.shot, full = Math.round(s.width * (r.outH + r.stripH) / r.outW);
    $("readout").textContent = "frame " + (state.frame + 1) + "/" + r.length() + " · t = " + fr.tau.toFixed(3) + " s · " +
      (r.length() / s.fps).toFixed(1) + " s at " + s.fps + " fps · " + (s.look === "strobe" ? "phase loop" : S.speedText(s.speed)) +
      " · export " + s.width + "×" + full + " " + s.format.toUpperCase();
    drawStripHead(fr.tau);
  }

  var lastTick = 0;
  function loop(ts) {
    requestAnimationFrame(loop);
    var r = state.renderer;
    if (!r || !state.playing || state.cropMode || state.exporting) return;
    if (ts - lastTick < 1000 / Math.min(state.shot.fps, 30)) return;
    lastTick = ts;
    state.frame = (state.frame + 1) % r.length();
    show();
  }
  requestAnimationFrame(loop);

  $("btn-play").addEventListener("click", function () { state.playing = !state.playing; $("btn-play").textContent = state.playing ? "Pause" : "Play"; });
  $("frame").addEventListener("input", function () { state.frame = Number($("frame").value); show(); });

  // ------------------------------------------------------------------ crop
  var cropCanvas = $("crop-overlay"), drag = null;
  function drawOverview() {
    var rec = state.rec, W = rec.width, H = rec.height, s = state.shot;
    var i0 = lowerB(rec.t, s.t0 || 0), i1 = lowerB(rec.t, s.t1 === null ? 1e9 : s.t1);
    var cnt = new Float32Array(W * H), mx = 0;
    for (var i = i0; i < i1; i++) cnt[rec.y[i] * W + rec.x[i]] += 1;
    var vals = Array.prototype.filter.call(cnt, function (v) { return v > 0; }).sort(function (a, b) { return a - b; });
    var hi = vals.length ? vals[Math.floor(0.99 * (vals.length - 1))] : 1;
    var img = new ImageData(W, H);
    for (var p = 0; p < W * H; p++) { var g = Math.sqrt(Math.min(1, cnt[p] / hi)) * 255; img.data[p * 4] = g * 0.9; img.data[p * 4 + 1] = g * 0.95; img.data[p * 4 + 2] = g; img.data[p * 4 + 3] = 255; }
    preview.width = W; preview.height = H; pctx.putImageData(img, 0, 0);
    if (s.crop) { pctx.strokeStyle = "#39c5cf"; pctx.lineWidth = 1; pctx.strokeRect(s.crop[0] + 0.5, s.crop[1] + 0.5, s.crop[2] - s.crop[0] - 1, s.crop[3] - s.crop[1] - 1); }
    $("readout").textContent = "Drag a box around the subject (sensor " + W + "×" + H + ").";
    mx = 0;
  }
  function lowerB(t, v) { var lo = 0, hi = t.length; while (lo < hi) { var m = (lo + hi) >>> 1; if (t[m] < v) lo = m + 1; else hi = m; } return lo; }
  function sensorXY(e) {
    var rc = preview.getBoundingClientRect();
    return [(e.clientX - rc.left) / rc.width * preview.width, (e.clientY - rc.top) / rc.height * preview.height];
  }
  $("btn-crop").addEventListener("click", function () {
    state.cropMode = !state.cropMode; $("btn-crop").classList.toggle("on", state.cropMode);
    if (state.cropMode) { drawOverview(); fitPreview(); } else rebuild();
  });
  $("btn-full").addEventListener("click", function () { state.shot.crop = null; state.cropMode = false; $("btn-crop").classList.remove("on"); syncControls(); changed(); });
  preview.addEventListener("pointerdown", function (e) { if (!state.cropMode) return; drag = sensorXY(e); preview.setPointerCapture(e.pointerId); });
  preview.addEventListener("pointermove", function (e) {
    if (!drag) return; var q = sensorXY(e); drawOverview();
    pctx.strokeStyle = "#39c5cf"; pctx.setLineDash([3, 2]);
    pctx.strokeRect(Math.min(drag[0], q[0]), Math.min(drag[1], q[1]), Math.abs(q[0] - drag[0]), Math.abs(q[1] - drag[1]));
    pctx.setLineDash([]);
  });
  preview.addEventListener("pointerup", function (e) {
    if (!drag) return; var q = sensorXY(e), c = S.clampCrop([drag[0], drag[1], q[0], q[1]], state.rec.width, state.rec.height);
    drag = null;
    if (c) { state.shot.crop = c; state.cropMode = false; $("btn-crop").classList.remove("on"); syncControls(); changed(); }
  });

  // ------------------------------------------------------------------ span strip
  var strip = $("span-strip"), sctx = strip.getContext("2d"), sdrag = null;
  function drawStrip() {
    if (!state.rate) return;
    var w = strip.clientWidth || 600, h = strip.clientHeight || 48;
    strip.width = w; strip.height = h;
    sctx.fillStyle = "#0e151b"; sctx.fillRect(0, 0, w, h);
    var r = state.renderer, t0 = r ? r.t0 : 0, t1 = r ? r.t1 : state.dur;
    sctx.fillStyle = "rgba(57,197,207,0.16)";
    sctx.fillRect(t0 / state.dur * w, 0, (t1 - t0) / state.dur * w, h);
    sctx.fillStyle = "#39c5cf";
    for (var x = 0; x < w; x++) { var v = state.rate[Math.min(state.rate.length - 1, Math.floor(x / w * state.rate.length))]; sctx.fillRect(x, h - v * (h - 4), 1, v * (h - 4)); }
    $("span-readout").textContent = "span " + t0.toFixed(3) + " – " + t1.toFixed(3) + " s of " + state.dur.toFixed(3) + " s · drag on the strip to change it";
  }
  function drawStripHead(tau) {
    drawStrip();
    var w = strip.width; sctx.fillStyle = "#f0efec"; sctx.fillRect(tau / Math.max(state.dur, 1e-9) * w - 1, 0, 2, strip.height);
  }
  function stripT(e) { var rc = strip.getBoundingClientRect(); return Math.max(0, Math.min(1, (e.clientX - rc.left) / rc.width)) * state.dur; }
  strip.addEventListener("pointerdown", function (e) { if (!state.rec) return; sdrag = stripT(e); strip.setPointerCapture(e.pointerId); });
  strip.addEventListener("pointerup", function (e) {
    if (sdrag === null) return; var t = stripT(e), a = Math.min(sdrag, t), b = Math.max(sdrag, t); sdrag = null;
    if (b - a < 0.01) { a = Math.max(0, t - 0.25); b = Math.min(state.dur, t + 0.25); }
    state.shot.t0 = +a.toFixed(3); state.shot.t1 = +b.toFixed(3); syncControls(); changed();
  });
  window.addEventListener("resize", function () { drawStrip(); fitPreview(); });

  // ------------------------------------------------------------------ export
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function fileName(ext) {
    var stem = (state.clip || "recording").replace(/\.raw$/, ""), r = state.renderer;
    return stem + "_" + state.shot.look + "_" + r.t0.toFixed(2) + "-" + r.t1.toFixed(2) + "s." + ext;
  }
  function progress(f) { $("export-progress").classList.remove("hidden"); $("export-bar").style.width = (f * 100).toFixed(1) + "%"; }
  function deliver(blob, name, isGif) {
    var url = URL.createObjectURL(blob), res = $("export-result");
    res.innerHTML = "";
    var a = document.createElement("a"); a.href = url; a.download = name; a.textContent = "Download " + name + " (" + (blob.size / 1048576).toFixed(2) + " MB)";
    res.appendChild(a);
    if (isGif) { var img = document.createElement("img"); img.src = url; img.alt = name; img.className = "gif-result"; res.appendChild(img); }
    a.click();
  }

  $("btn-export").addEventListener("click", async function () {
    if (!state.rec || state.exporting) return;
    readControls();
    state.exporting = true; $("btn-export").disabled = true;
    try {
      var r = new S.Renderer(state.rec, Object.assign({}, state.shot)), canvas = document.createElement("canvas");
      var n = r.length();
      if (state.shot.format === "gif") {
        var frames = [];
        for (var i = 0; i < n; i++) {
          paint(r, i, canvas);
          frames.push(canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data);
          if (i % 4 === 0) { progress(0.7 * (i + 1) / n); await sleep(0); }
        }
        var bytes = GottLUXGif.encode(frames, canvas.width, canvas.height, {
          delayMs: 1000 / state.shot.fps, colors: state.shot.gif_colors,
          onProgress: function (f) { progress(0.7 + 0.3 * f); } });
        deliver(new Blob([bytes], { type: "image/gif" }), fileName("gif"), true);
      } else {
        var mime = videoMime.filter(function (m) { return m.indexOf(state.shot.format) >= 0; })[0];
        paint(r, 0, canvas);
        var stream = canvas.captureStream(0), track = stream.getVideoTracks()[0], chunks = [];
        var rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8e6 });
        rec.ondataavailable = function (e) { if (e.data.size) chunks.push(e.data); };
        var stopped = new Promise(function (res) { rec.onstop = res; });
        rec.start();
        for (var j = 0; j < n; j++) {
          paint(r, j, canvas);
          if (track.requestFrame) track.requestFrame();
          progress((j + 1) / n);
          await sleep(1000 / state.shot.fps);
        }
        rec.stop(); await stopped;
        deliver(new Blob(chunks, { type: mime.split(";")[0] }), fileName(state.shot.format), false);
      }
    } catch (err) {
      $("export-result").textContent = "Export failed: " + err.message;
    } finally {
      state.exporting = false; $("btn-export").disabled = false; $("export-progress").classList.add("hidden");
    }
  });

  // ------------------------------------------------------------------ share / recipe
  function writeHash() {
    if (!state.clip) return;
    var data = { clip: state.clip, shot: state.shot };
    history.replaceState(null, "", "#shot=" + encodeURIComponent(JSON.stringify(data)));
  }
  function fromHash() {
    var m = /#shot=(.*)$/.exec(location.hash);
    if (!m) return;
    try {
      var data = JSON.parse(decodeURIComponent(m[1]));
      if (data.clip && data.shot) fetchClip(data.clip, data.shot).catch(function () {});
    } catch (e) { /* a broken link just opens the empty Studio */ }
  }
  $("btn-share").addEventListener("click", function () {
    writeHash();
    var done = function () { $("export-result").textContent = state.clip ? "Link copied — it reopens this exact shot." : "Links work for the hosted sample clips; local files stay on this machine."; };
    if (navigator.clipboard) navigator.clipboard.writeText(location.href).then(done, done); else done();
  });
  $("btn-recipe").addEventListener("click", function () {
    var shot = Object.assign({ source: state.clip || "" }, state.shot);
    var blob = new Blob([JSON.stringify({ shot: shot }, null, 2)], { type: "application/json" });
    var a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = (state.clip || "recording").replace(/\.raw$/, "") + "_" + shot.look + ".shot.json"; a.click();
  });

  syncControls();
  loadManifest();
  window.GottLUXStudioPage = { state: state, applyShot: applyShot, paint: paint, rebuild: rebuild };
})();
