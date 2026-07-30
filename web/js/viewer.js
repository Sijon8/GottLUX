/*
 * viewer.js — GottLUX Web application: clip picker, canvas accumulation
 * renderer, transport, event-rate timeline, ROI spectrum and flicker map.
 * Decoding and heavy analysis run in Web Workers; this file owns the UI.
 */
(function () {
  "use strict";
  const A = GottLUXAnalysis;
  const $ = (id) => document.getElementById(id);

  // ---------------- DOM ----------------
  const view = $("view"), viewerWrap = $("viewer-wrap"), viewHint = $("view-hint");
  const roiTip = $("roi-tip");
  const loadOverlay = $("load-overlay"), loadStage = $("load-stage"), loadBar = $("load-bar");
  const btnPlay = $("btn-play"), tCur = $("t-cur"), tDur = $("t-dur");
  const speedIn = $("speed"), speedOut = $("speed-out");
  const accumIn = $("accum"), accumOut = $("accum-out");
  const loopIn = $("loop"), cmapSel = $("cmap"), toneSel = $("tone"), modeSel = $("mode");
  const rateCanvas = $("rate-strip");
  const fminIn = $("fmin"), fmaxIn = $("fmax"), spanSel = $("span");
  const btnSpectrum = $("btn-spectrum"), btnClearRoi = $("btn-clear-roi");
  const btnFlicker = $("btn-flicker"), btnOverlay = $("btn-overlay");
  const flickerProgress = $("flicker-progress"), flickerBar = $("flicker-bar");
  const specCanvas = $("spectrum"), specReadout = $("spec-readout");
  const legendWrap = $("flicker-legend"), legendCanvas = $("legend-canvas"), legendLabel = $("legend-label");
  const clipList = $("clip-list"), manifestNote = $("manifest-note");
  const dropzone = $("dropzone"), fileInput = $("file-input");

  // ---------------- state ----------------
  let rec = null;                 // {x,y,p,t,n,width,height,fmt,meta,t0_us,duration,title}
  let rate = null;                // {bins, nb, binW, peak, base(offscreen canvas)}
  let cursor = 0, playing = false, lastTs = null;
  let roi = null, roiDraft = null, dragStart = null;
  let flicker = null, flickerColors = null, overlayOn = true;
  let lastSpectrum = null, lastSpectrumMeta = null;
  let dirty = true, scale = 1;
  let frame = null, off = null, octx = null, img = null;
  let jobId = 0, decodeJob = null, spectrumJob = 0, flickerJob = 0;

  // ---------------- colormaps ----------------
  const STOPS = {
    inferno: [[0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110], [147, 38, 103],
              [186, 54, 85], [221, 81, 58], [243, 120, 25], [252, 165, 10],
              [246, 215, 70], [252, 255, 164]],
    viridis: [[68, 1, 84], [71, 24, 106], [72, 40, 120], [62, 74, 137], [49, 104, 142],
              [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
              [180, 222, 44], [253, 231, 37]],
    gray: [[0, 0, 0], [255, 255, 255]]
  };
  const POL_ON = [255, 112, 66], POL_OFF = [64, 148, 255];   // ON warm / OFF cool

  function buildLUT(stops) {
    const lut = new Uint8Array(768);
    const m = stops.length - 1;
    for (let i = 0; i < 256; i++) {
      const f = (i / 255) * m, s = Math.min(Math.floor(f), m - 1), r = f - s;
      for (let c = 0; c < 3; c++)
        lut[i * 3 + c] = Math.round(stops[s][c] + (stops[s + 1][c] - stops[s][c]) * r);
    }
    return lut;
  }
  const LUTS = { inferno: buildLUT(STOPS.inferno), viridis: buildLUT(STOPS.viridis),
                 gray: buildLUT(STOPS.gray) };

  function hueRGB(frac) {           // 0 -> blue(240deg) ... 1 -> red(0deg), full s/v
    const h = (1 - Math.min(Math.max(frac, 0), 1)) * 240 / 60;
    const i = Math.floor(h), f = h - i;
    const q = 1 - f, t = f;
    switch (i % 6) {
      case 0: return [255, t * 255, 0];
      case 1: return [q * 255, 255, 0];
      case 2: return [0, 255, t * 255];
      case 3: return [0, q * 255, 255];
      default: return [t * 255, 0, 255];
    }
  }

  // ---------------- workers ----------------
  const decodeWorker = new Worker("js/decode_worker.js");
  const analysisWorker = new Worker("js/analysis_worker.js");

  decodeWorker.onmessage = (e) => {
    const d = e.data;
    if (!decodeJob || d.id !== decodeJob.id) return;
    if (d.type === "progress") {
      setOverlay(`decoding — ${d.stage || ""}`, 0.35 + 0.65 * d.frac);
    } else if (d.type === "done") {
      const job = decodeJob; decodeJob = null;
      hideOverlay();
      setRecording(d.result, job.title, job.suggested);
    } else if (d.type === "error") {
      decodeJob = null;
      failOverlay("decode failed: " + d.message);
    }
  };

  analysisWorker.onmessage = (e) => {
    const d = e.data;
    if (d.type === "progress" && d.id === flickerJob) {
      flickerBar.style.width = (d.frac * 100).toFixed(1) + "%";
      return;
    }
    if (d.type === "error") {
      if (d.id === flickerJob) { flickerProgress.classList.add("hidden"); btnFlicker.disabled = false; }
      specReadout.innerHTML = `<span class="lbl">analysis error: ${d.message}</span>`;
      return;
    }
    if (d.type !== "done") return;
    if (d.cmd === "spectrum" && d.id === spectrumJob) {
      lastSpectrum = d.result;
      drawSpectrum();
      updateSpecReadout();
    } else if (d.cmd === "flicker" && d.id === flickerJob) {
      flickerProgress.classList.add("hidden");
      btnFlicker.disabled = false;
      flicker = d.result;
      buildFlickerColors();
      overlayOn = true;
      btnOverlay.disabled = false;
      btnOverlay.textContent = "Hide overlay";
      updateLegend();
      dirty = true;
    }
  };

  // ---------------- overlay / progress ----------------
  function setOverlay(stage, frac) {
    loadOverlay.classList.remove("hidden");
    loadStage.textContent = stage;
    loadBar.style.width = (Math.min(Math.max(frac, 0), 1) * 100).toFixed(1) + "%";
  }
  function hideOverlay() { loadOverlay.classList.add("hidden"); }
  function failOverlay(msg) {
    setOverlay(msg, 0);
    setTimeout(hideOverlay, 4500);
  }

  // ---------------- clip picker ----------------
  function loadManifest() {
    fetch("data/manifest.json")
      .then((r) => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then((man) => {
        const clips = (man && man.clips) || [];
        if (!clips.length) throw new Error("empty manifest");
        manifestNote.textContent = "";
        clips.forEach((clip) => {
          const card = document.createElement("div");
          card.className = "clip-card";
          const badgeCls = clip.regime === "rotating" ? "badge rotating" : "badge";
          card.innerHTML =
            `<div class="row1"><span class="title"></span>` +
            `<span class="${badgeCls}">${clip.regime || "clip"}</span></div>` +
            `<div class="desc"></div>` +
            `<div class="stats">${(+clip.duration_s).toFixed(1)} s · ${(+clip.size_mb).toFixed(1)} MB</div>`;
          card.querySelector(".title").textContent = clip.title || clip.file;
          card.querySelector(".desc").textContent = clip.description || "";
          card.addEventListener("click", () => {
            document.querySelectorAll(".clip-card.active").forEach((c) => c.classList.remove("active"));
            card.classList.add("active");
            fetchClip(clip).catch((err) => failOverlay("download failed: " + err.message));
          });
          clipList.appendChild(card);
        });
      })
      .catch(() => {
        manifestNote.textContent =
          "No sample manifest found (running locally?) — drop a .raw file below instead.";
      });
  }

  async function fetchClip(clip) {
    setOverlay("downloading " + clip.file, 0);
    const resp = await fetch("data/" + clip.file);
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const total = +resp.headers.get("Content-Length") ||
                  (clip.size_mb ? clip.size_mb * 1048576 : 0);
    const reader = resp.body.getReader();
    const chunks = [];
    let recvd = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      recvd += value.length;
      setOverlay(`downloading ${clip.file} — ${(recvd / 1048576).toFixed(1)} MB`,
                 total ? 0.35 * (recvd / total) : 0.15);
    }
    const buf = new Uint8Array(recvd);
    let o = 0;
    for (const c of chunks) { buf.set(c, o); o += c.length; }
    loadArrayBuffer(buf.buffer, clip.title || clip.file, clip.suggested || null);
  }

  function loadArrayBuffer(buffer, title, suggested) {
    setOverlay("decoding…", 0.35);
    const id = ++jobId;
    decodeJob = { id, title, suggested };
    decodeWorker.postMessage({ id, buffer }, [buffer]);
  }

  // ---------------- local files ----------------
  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) openFile(fileInput.files[0]);
    fileInput.value = "";
  });
  ["dragover", "dragenter"].forEach((ev) =>
    document.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("hover"); }));
  ["dragleave", "dragend"].forEach((ev) =>
    document.addEventListener(ev, () => dropzone.classList.remove("hover")));
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("hover");
    if (e.dataTransfer.files.length) openFile(e.dataTransfer.files[0]);
  });
  function openFile(file) {
    setOverlay("reading " + file.name, 0.05);
    file.arrayBuffer()
      .then((buf) => loadArrayBuffer(buf, file.name, null))
      .catch((err) => failOverlay("read failed: " + err.message));
  }

  // ---------------- recording setup ----------------
  function setRecording(res, title, suggested) {
    rec = res;
    rec.title = title;
    rec.duration = rec.n ? rec.t[rec.n - 1] : 0;
    if (!rec.n) { failOverlay("no events decoded from this file"); rec = null; return; }
    cursor = 0; playing = false; lastTs = null;
    roi = null; roiDraft = null;
    flicker = null; flickerColors = null;
    lastSpectrum = null; lastSpectrumMeta = null;
    legendWrap.classList.add("hidden");
    btnOverlay.disabled = true;
    frame = new Float32Array(rec.width * rec.height);
    off = document.createElement("canvas");
    off.width = rec.width; off.height = rec.height;
    octx = off.getContext("2d");
    img = octx.createImageData(rec.width, rec.height);
    if (suggested) {
      if (suggested.accum_ms) {
        accumIn.value = Math.min(Math.max(Math.round(suggested.accum_ms), 1), 200);
        accumIn.dispatchEvent(new Event("input"));
      }
      if (suggested.band_hz && suggested.band_hz.length === 2) {
        fminIn.value = Math.round(suggested.band_hz[0]);
        fmaxIn.value = Math.round(suggested.band_hz[1]);
      }
    }
    computeRate();
    fillInfo();
    layout();
    viewHint.style.display = "none";
    btnPlay.disabled = false;
    btnFlicker.disabled = false;
    btnSpectrum.disabled = true;
    btnClearRoi.disabled = true;
    setPlaying(true);
    specReadout.innerHTML =
      `<span class="hint">Drag a box on the viewer to analyze a region.</span>`;
    drawSpectrum();
    dirty = true;
  }

  function fillInfo() {
    $("inf-n").textContent = rec.n.toLocaleString("en-US");
    $("inf-dur").textContent = rec.duration.toFixed(3) + " s";
    $("inf-res").textContent = rec.width + " × " + rec.height;
    $("inf-fmt").textContent = GottLUXDecoder.FORMAT_LABEL[rec.fmt] || rec.fmt;
    $("inf-mean").textContent = fmtRate(rec.n / Math.max(rec.duration, 1e-9));
    $("inf-peak").textContent = rate ? fmtRate(rate.peak / rate.binW) : "—";
    tDur.textContent = rec.duration.toFixed(3);
  }

  function fmtRate(v) {
    if (v >= 1e6) return (v / 1e6).toFixed(2) + " Mev/s";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + " kev/s";
    return v.toFixed(0) + " ev/s";
  }

  // ---------------- event-rate strip ----------------
  function computeRate() {
    const nb = Math.min(2000, Math.max(100, Math.floor(rec.duration / 0.005) || 100));
    const binW = rec.duration / nb || 1;
    const bins = new Float32Array(nb);
    const t = rec.t, n = rec.n;
    for (let i = 0; i < n; i++) {
      let b = (t[i] / binW) | 0;
      if (b >= nb) b = nb - 1;
      bins[b]++;
    }
    let peak = 0;
    for (let i = 0; i < nb; i++) if (bins[i] > peak) peak = bins[i];
    rate = { bins, nb, binW, peak, base: null };
  }

  function renderRateBase() {
    const w = rateCanvas.width, h = rateCanvas.height;
    const base = document.createElement("canvas");
    base.width = w; base.height = h;
    const ctx = base.getContext("2d");
    ctx.fillStyle = "#0e1116";
    ctx.fillRect(0, 0, w, h);
    if (rate && rate.peak > 0) {
      ctx.fillStyle = "rgba(57,197,207,0.55)";
      const perPx = rate.nb / w;
      for (let px = 0; px < w; px++) {
        const b0 = Math.floor(px * perPx), b1 = Math.max(b0 + 1, Math.floor((px + 1) * perPx));
        let m = 0;
        for (let b = b0; b < b1 && b < rate.nb; b++) if (rate.bins[b] > m) m = rate.bins[b];
        const bh = Math.round(Math.sqrt(m / rate.peak) * (h - 6));
        if (bh > 0) ctx.fillRect(px, h - bh - 2, 1, bh);
      }
    }
    ctx.strokeStyle = "#1c2733";
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
    rate.base = base;
  }

  function drawRate() {
    if (!rate) return;
    if (!rate.base || rate.base.width !== rateCanvas.width) renderRateBase();
    const ctx = rateCanvas.getContext("2d");
    ctx.drawImage(rate.base, 0, 0);
    const w = rateCanvas.width, h = rateCanvas.height;
    // Fractional x (no rounding): at deep slow motion (0.005×) the cursor moves
    // ~83 µs per frame, and the anti-aliased sub-pixel drift keeps the marker
    // visibly advancing instead of freezing for whole seconds between pixels.
    const cx = (cursor / Math.max(rec.duration, 1e-9)) * w;
    ctx.fillStyle = "rgba(57,197,207,0.10)";
    ctx.fillRect(0, 0, cx, h);
    ctx.strokeStyle = "#f78166";
    ctx.beginPath();
    ctx.moveTo(cx, 1);
    ctx.lineTo(cx, h - 1);
    ctx.stroke();
  }

  let rateDrag = false;
  function rateSeek(e) {
    if (!rec) return;
    const r = rateCanvas.getBoundingClientRect();
    const f = Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1);
    cursor = f * rec.duration;
    dirty = true;
  }
  rateCanvas.addEventListener("mousedown", (e) => { rateDrag = true; rateSeek(e); });
  window.addEventListener("mousemove", (e) => { if (rateDrag) rateSeek(e); });
  window.addEventListener("mouseup", () => { rateDrag = false; });

  // ---------------- layout / sizing ----------------
  function layout() {
    if (rec) {
      const availW = Math.max(viewerWrap.clientWidth - 8, 64);
      const targetH = Math.min(Math.max(window.innerHeight * 0.52, 320), 620);
      scale = Math.max(1, Math.min(Math.floor(availW / rec.width), Math.floor(targetH / rec.height)));
      view.width = rec.width * scale;
      view.height = rec.height * scale;
      view.style.maxWidth = "100%";
      view.getContext("2d").imageSmoothingEnabled = false;
    }
    rateCanvas.width = Math.max(rateCanvas.clientWidth, 100);
    rateCanvas.height = Math.max(rateCanvas.clientHeight, 40);
    if (rate) rate.base = null;
    specCanvas.width = Math.max(specCanvas.clientWidth, 100);
    specCanvas.height = Math.max(specCanvas.clientHeight, 80);
    drawSpectrum();
    dirty = true;
  }
  window.addEventListener("resize", layout);

  // ---------------- frame rendering ----------------
  function computeFrame() {
    const wSec = (+accumIn.value) / 1000;
    const t = rec.t;
    const i0 = A.lowerBound(t, cursor - wSec);
    const i1 = A.upperBound(t, cursor);
    const count = i1 - i0;
    const stride = count > 6e6 ? Math.ceil(count / 6e6) : 1;
    frame.fill(0);
    const X = rec.x, Y = rec.y, P = rec.p, W = rec.width;
    if (modeSel.value === "polarity") {
      for (let i = i0; i < i1; i += stride) frame[Y[i] * W + X[i]] += P[i] ? 1 : -1;
    } else {
      for (let i = i0; i < i1; i += stride) frame[Y[i] * W + X[i]] += 1;
    }
    return count;
  }

  function whitePoint() {
    // auto white point: ~99.5th percentile of |nonzero| values (sampled)
    const n = frame.length;
    const step = Math.max(1, (n / 16384) | 0);
    const vals = [];
    for (let i = 0; i < n; i += step) {
      const v = frame[i];
      if (v > 0) vals.push(v); else if (v < 0) vals.push(-v);
    }
    if (!vals.length) return 1;
    vals.sort((a, b) => a - b);
    return Math.max(vals[Math.floor(0.995 * (vals.length - 1))], 1);
  }

  function paint() {
    const data = img.data;
    const n = frame.length;
    const white = whitePoint();
    const tone = toneSel.value;
    const logW = Math.log1p(white);
    const polar = modeSel.value === "polarity";
    const lut = LUTS[cmapSel.value] || LUTS.inferno;
    const fc = (flicker && overlayOn) ? flickerColors : null;
    const gw = fc ? flicker.gw : 0, cell = fc ? flicker.cell : 1;
    const W = rec.width;
    for (let i = 0; i < n; i++) {
      const v = frame[i];
      let r, g, b;
      if (polar) {
        const a = v < 0 ? -v : v;
        let u = tone === "log" ? Math.log1p(a) / logW
              : tone === "sqrt" ? Math.sqrt(a / white) : a / white;
        if (u > 1) u = 1;
        const c = v >= 0 ? POL_ON : POL_OFF;
        r = c[0] * u; g = c[1] * u; b = c[2] * u;
      } else {
        let u = v <= 0 ? 0
              : tone === "log" ? Math.log1p(v) / logW
              : tone === "sqrt" ? Math.sqrt(v / white) : v / white;
        if (u > 1) u = 1; else if (u < 0) u = 0;
        const li = (u * 255) | 0;
        r = lut[li * 3]; g = lut[li * 3 + 1]; b = lut[li * 3 + 2];
      }
      if (fc) {
        const px = i % W, py = (i / W) | 0;
        const ci = (((py / cell) | 0) * gw + ((px / cell) | 0)) * 4;
        const al = fc[ci + 3] / 255;
        if (al > 0) {
          r = r * (1 - al) + fc[ci] * al;
          g = g * (1 - al) + fc[ci + 1] * al;
          b = b * (1 - al) + fc[ci + 2] * al;
        }
      }
      const o = i * 4;
      data[o] = r; data[o + 1] = g; data[o + 2] = b; data[o + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
  }

  function draw() {
    if (!rec) return;
    computeFrame();
    paint();
    const ctx = view.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, view.width, view.height);
    ctx.drawImage(off, 0, 0, view.width, view.height);
    const box = roiDraft || roi;
    if (box) {
      ctx.strokeStyle = roiDraft ? "rgba(247,129,102,0.9)" : "rgba(57,197,207,0.95)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(box[0] * scale + 0.5, box[1] * scale + 0.5,
                     (box[2] - box[0]) * scale, (box[3] - box[1]) * scale);
    }
    drawRate();
    tCur.textContent = cursor.toFixed(3);
  }

  function tick(ts) {
    requestAnimationFrame(tick);
    if (!rec) return;
    if (playing) {
      const dt = lastTs === null ? 0 : (ts - lastTs) / 1000;
      cursor += dt * speed;
      if (cursor > rec.duration) {
        if (loopIn.checked) cursor = 0;
        else { cursor = rec.duration; setPlaying(false); }
      }
      dirty = true;
    }
    lastTs = ts;
    if (dirty) { dirty = false; draw(); }
  }
  requestAnimationFrame(tick);

  // ---------------- transport ----------------
  function setPlaying(on) {
    playing = on;
    btnPlay.textContent = on ? "Pause" : "Play";
  }
  btnPlay.addEventListener("click", () => setPlaying(!playing));
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space") return;
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "button" || tag === "textarea") return;
    e.preventDefault();
    if (rec) setPlaying(!playing);
  });
  // Speed slider is logarithmic: position 0..1 maps to SPEED_MIN..SPEED_MAX so
  // the deep-slow-motion end (0.005–0.1×) gets as much travel as 0.1–2×.
  // Speeds snap to 2 significant figures so the readout shows exactly the
  // speed in use (0.005×, 0.05×, 0.25×, 1×, …).
  const SPEED_MIN = 0.005, SPEED_MAX = 2;
  const SPEED_LOG = Math.log(SPEED_MAX / SPEED_MIN);
  const SPEED_PRESETS = [1, 0.1, 0.01];
  let speed = 0.25;
  function sliderToSpeed(pos) {
    return +(SPEED_MIN * Math.exp(pos * SPEED_LOG)).toPrecision(2);
  }
  function speedToSlider(v) {
    return Math.min(Math.max(Math.log(v / SPEED_MIN) / SPEED_LOG, 0), 1);
  }
  speedIn.addEventListener("input", () => {
    speed = sliderToSpeed(+speedIn.value);
    speedOut.textContent = speed + "×";
  });
  speedOut.addEventListener("click", () => {
    const i = SPEED_PRESETS.indexOf(speed);
    speedIn.value = speedToSlider(SPEED_PRESETS[(i + 1) % SPEED_PRESETS.length]);
    speedIn.dispatchEvent(new Event("input"));
  });
  accumIn.addEventListener("input", () => {
    accumOut.textContent = accumIn.value + " ms";
    dirty = true;
  });
  [cmapSel, toneSel, modeSel].forEach((el) =>
    el.addEventListener("change", () => { dirty = true; }));

  // ---------------- ROI drag ----------------
  function viewToSensor(e) {
    const r = view.getBoundingClientRect();
    const fx = rec.width / r.width, fy = rec.height / r.height;
    return [
      Math.min(Math.max(Math.floor((e.clientX - r.left) * fx), 0), rec.width - 1),
      Math.min(Math.max(Math.floor((e.clientY - r.top) * fy), 0), rec.height - 1)
    ];
  }
  view.addEventListener("mousedown", (e) => {
    if (!rec) return;
    e.preventDefault();
    dragStart = viewToSensor(e);
    roiDraft = [dragStart[0], dragStart[1], dragStart[0] + 1, dragStart[1] + 1];
    dirty = true;
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragStart || !rec) return;
    const p = viewToSensor(e);
    roiDraft = [Math.min(dragStart[0], p[0]), Math.min(dragStart[1], p[1]),
                Math.max(dragStart[0], p[0]) + 1, Math.max(dragStart[1], p[1]) + 1];
    roiTip.hidden = false;
    roiTip.textContent = `ROI ${roiDraft[0]},${roiDraft[1]} ` +
      `${roiDraft[2] - roiDraft[0]}×${roiDraft[3] - roiDraft[1]} px`;
    dirty = true;
  });
  window.addEventListener("mouseup", () => {
    if (!dragStart || !rec) return;
    const box = roiDraft;
    dragStart = null; roiDraft = null;
    roiTip.hidden = true;
    if (box && (box[2] - box[0]) >= 3 && (box[3] - box[1]) >= 3) {
      roi = box;
      btnSpectrum.disabled = false;
      btnClearRoi.disabled = false;
      runSpectrum();
    }
    dirty = true;
  });
  btnClearRoi.addEventListener("click", () => {
    roi = null;
    lastSpectrum = null; lastSpectrumMeta = null;
    btnSpectrum.disabled = true;
    btnClearRoi.disabled = true;
    specReadout.innerHTML = `<span class="hint">Drag a box on the viewer to analyze a region.</span>`;
    drawSpectrum();
    dirty = true;
  });
  btnSpectrum.addEventListener("click", runSpectrum);

  // ---------------- analysis span ----------------
  function analysisSpan() {
    const dur = rec.duration;
    const v = spanSel.value;
    let w;
    if (v === "clip") return [0, dur];
    if (v === "auto") { if (dur <= 3) return [0, dur]; w = 1; }
    else w = parseFloat(v);
    if (dur <= w) return [0, dur];
    const t0 = Math.min(Math.max(cursor - w / 2, 0), dur - w);
    return [t0, t0 + w];
  }

  function bandInputs() {
    let lo = Math.max(+fminIn.value || 30, 0.5);
    let hi = Math.max(+fmaxIn.value || 800, lo + 1);
    return [lo, hi];
  }

  // ---------------- region spectrum ----------------
  function runSpectrum() {
    if (!rec || !roi) return;
    const [s0, s1] = analysisSpan();
    const [fmin, fmax] = bandInputs();
    const i0 = A.lowerBound(rec.t, s0), i1 = A.upperBound(rec.t, s1);
    const X = rec.x, Y = rec.y, T = rec.t;
    const [x0, y0, x1, y1] = roi;
    let m = 0;
    for (let i = i0; i < i1; i++)
      if (X[i] >= x0 && X[i] < x1 && Y[i] >= y0 && Y[i] < y1) m++;
    const ts = new Float64Array(m);
    let k = 0;
    for (let i = i0; i < i1; i++)
      if (X[i] >= x0 && X[i] < x1 && Y[i] >= y0 && Y[i] < y1) ts[k++] = T[i];
    lastSpectrumMeta = { span: [s0, s1], roi: roi.slice(), band: [fmin, fmax], nEvents: m };
    if (m < 16) {
      lastSpectrum = null;
      drawSpectrum();
      specReadout.innerHTML = `<span class="lbl">only ${m} events in ROI over ` +
        `${s0.toFixed(2)}–${s1.toFixed(2)} s — enlarge the ROI or span.</span>`;
      return;
    }
    specReadout.innerHTML = `<span class="lbl">computing spectrum (${m.toLocaleString("en-US")} events)…</span>`;
    spectrumJob = ++jobId;
    analysisWorker.postMessage({ id: spectrumJob, cmd: "spectrum", t: ts, fmin, fmax },
                               [ts.buffer]);
  }

  function updateSpecReadout() {
    const s = lastSpectrum, meta = lastSpectrumMeta;
    if (!s || !meta) return;
    if (!isFinite(s.peakFreq)) {
      specReadout.innerHTML = `<span class="lbl">no in-band peak found.</span>`;
      return;
    }
    specReadout.innerHTML =
      `<span class="lbl">peak</span> <span class="val">${s.peakFreq.toFixed(1)} Hz</span>` +
      ` · <span class="lbl">SNR</span> <span class="val">${s.snr.toFixed(1)}</span>` +
      ` · <span class="lbl">harmonics</span> <span class="val">${(s.harmonicScore * 100).toFixed(0)}%</span>` +
      ` · <span class="lbl">${s.nEvents.toLocaleString("en-US")} ev · ` +
      `${meta.span[0].toFixed(2)}–${meta.span[1].toFixed(2)} s · ` +
      `ROI ${meta.roi[2] - meta.roi[0]}×${meta.roi[3] - meta.roi[1]} px</span>`;
  }

  function niceStep(range, maxTicks) {
    const raw = range / maxTicks;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const q of [1, 2, 5, 10]) if (raw <= q * mag) return q * mag;
    return 10 * mag;
  }

  function drawSpectrum() {
    const ctx = specCanvas.getContext("2d");
    const W = specCanvas.width, H = specCanvas.height;
    ctx.fillStyle = "#0e151b";
    ctx.fillRect(0, 0, W, H);
    const s = lastSpectrum, meta = lastSpectrumMeta;
    const ml = 46, mr = 10, mt = 8, mb = 24;
    const pw = W - ml - mr, ph = H - mt - mb;
    if (pw < 40 || ph < 30) return;
    if (!s || !s.freqs.length) {
      ctx.fillStyle = "#8b97a7";
      ctx.font = "12px ui-monospace, Consolas, monospace";
      ctx.textAlign = "center";
      ctx.fillText(rec ? "no spectrum yet — drag an ROI" : "load a clip first", W / 2, H / 2);
      return;
    }
    const [fmin, fmax] = s.band;
    const fLo = 0, fHi = Math.min(s.fs / 2, fmax * 1.2);
    let pMax = 0;
    for (let i = 0; i < s.freqs.length; i++)
      if (s.freqs[i] >= fLo && s.freqs[i] <= fHi && s.power[i] > pMax) pMax = s.power[i];
    if (pMax <= 0) pMax = 1;
    const fx = (f) => ml + ((f - fLo) / (fHi - fLo)) * pw;
    const fy = (p) => mt + ph - (p / pMax) * ph;
    // band shading
    ctx.fillStyle = "rgba(57,197,207,0.07)";
    ctx.fillRect(fx(Math.max(fmin, fLo)), mt,
                 fx(Math.min(fmax, fHi)) - fx(Math.max(fmin, fLo)), ph);
    // axes + ticks
    ctx.strokeStyle = "#1c2733";
    ctx.fillStyle = "#8b97a7";
    ctx.font = "10px ui-monospace, Consolas, monospace";
    ctx.textAlign = "center";
    const step = niceStep(fHi - fLo, 8);
    ctx.beginPath();
    for (let f = Math.ceil(fLo / step) * step; f <= fHi; f += step) {
      ctx.moveTo(fx(f) + 0.5, mt);
      ctx.lineTo(fx(f) + 0.5, mt + ph);
      if (fx(f) < W - mr - 30) ctx.fillText(String(Math.round(f)), fx(f), H - 9);
    }
    ctx.stroke();
    ctx.fillText("Hz", W - mr - 8, H - 9);
    ctx.save();
    ctx.translate(11, mt + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("power (norm)", 0, 0);
    ctx.restore();
    ctx.textAlign = "right";
    ctx.fillText("1.0", ml - 5, mt + 8);
    ctx.fillText("0", ml - 5, mt + ph);
    // spectrum trace
    ctx.strokeStyle = "#39c5cf";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < s.freqs.length; i++) {
      const f = s.freqs[i];
      if (f < fLo || f > fHi) continue;
      const xpx = fx(f), ypx = fy(s.power[i]);
      if (!started) { ctx.moveTo(xpx, ypx); started = true; }
      else ctx.lineTo(xpx, ypx);
    }
    ctx.stroke();
    // peak marker
    if (isFinite(s.peakFreq) && s.peakFreq >= fLo && s.peakFreq <= fHi) {
      const px = fx(s.peakFreq), py = fy(Math.min(s.peakPower, pMax));
      ctx.fillStyle = "#f78166";
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, 2 * Math.PI);
      ctx.fill();
      ctx.textAlign = px > W - 80 ? "right" : "left";
      ctx.fillText(s.peakFreq.toFixed(1) + " Hz", px + (px > W - 80 ? -7 : 7),
                   Math.max(py - 6, mt + 10));
    }
  }

  // ---------------- flicker map ----------------
  btnFlicker.addEventListener("click", () => {
    if (!rec) return;
    const [s0, s1] = analysisSpan();
    const [fmin, fmax] = bandInputs();
    const i0 = A.lowerBound(rec.t, s0), i1 = A.upperBound(rec.t, s1);
    if (i1 - i0 < 100) {
      specReadout.innerHTML = `<span class="lbl">too few events in the analysis span for a flicker map.</span>`;
      return;
    }
    const xs = rec.x.slice(i0, i1), ys = rec.y.slice(i0, i1), ts = rec.t.slice(i0, i1);
    btnFlicker.disabled = true;
    flickerProgress.classList.remove("hidden");
    flickerBar.style.width = "0%";
    flickerJob = ++jobId;
    flicker = null; flickerColors = null;
    flickerWindow = [s0, s1];
    analysisWorker.postMessage(
      { id: flickerJob, cmd: "flicker", x: xs, y: ys, t: ts,
        width: rec.width, height: rec.height, cell: 8,
        fmin, fmax, minEventsPerCell: 30 },
      [xs.buffer, ys.buffer, ts.buffer]);
  });
  let flickerWindow = null;

  function buildFlickerColors() {
    // per-cell RGBA cache: hue = frequency position in band, alpha = SNR strength
    const f = flicker;
    const nc = f.gw * f.gh;
    flickerColors = new Uint8ClampedArray(nc * 4);
    const [lo, hi] = f.band;
    const range = Math.max(hi - lo, 1e-9);
    // Noise gate: for exponential (Poisson-noise) power, peak/median over K bins
    // is ~1.44*ln(K); demand ~e^-3 false-positive headroom above that.
    const K = f.spectrumBins || 512;
    const thr = 1.44 * (Math.log(K) + 3);
    for (let i = 0; i < nc; i++) {
      const fr = f.freq[i], sn = f.snr[i];
      if (!isFinite(fr) || sn < thr) continue;
      const rgb = hueRGB((fr - lo) / range);
      const al = Math.min((sn - thr) / (2 * thr), 1) * 0.85;
      const o = i * 4;
      flickerColors[o] = rgb[0]; flickerColors[o + 1] = rgb[1];
      flickerColors[o + 2] = rgb[2]; flickerColors[o + 3] = Math.round(al * 255);
    }
  }

  function updateLegend() {
    legendWrap.classList.remove("hidden");
    const ctx = legendCanvas.getContext("2d");
    const w = legendCanvas.width, h = legendCanvas.height;
    for (let px = 0; px < w; px++) {
      const rgb = hueRGB(px / (w - 1));
      ctx.fillStyle = `rgb(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0})`;
      ctx.fillRect(px, 0, 1, h);
    }
    const [lo, hi] = flicker.band;
    legendLabel.textContent =
      `${Math.round(lo)}–${Math.round(hi)} Hz · cell ${flicker.cell} px · ` +
      `window ${flickerWindow[0].toFixed(2)}–${flickerWindow[1].toFixed(2)} s`;
  }

  btnOverlay.addEventListener("click", () => {
    overlayOn = !overlayOn;
    btnOverlay.textContent = overlayOn ? "Hide overlay" : "Show overlay";
    legendWrap.classList.toggle("hidden", !overlayOn);
    dirty = true;
  });

  // ---------------- boot ----------------
  speedIn.dispatchEvent(new Event("input"));
  accumIn.dispatchEvent(new Event("input"));
  layout();
  loadManifest();
})();
