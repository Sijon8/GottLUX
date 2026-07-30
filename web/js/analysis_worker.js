/*
 * analysis_worker.js — off-thread frequency analysis for GottLUX Web.
 *
 * In:  { id, cmd:"spectrum", t (Float64Array seconds, transferred), fmin, fmax }
 *      { id, cmd:"flicker",  x, y, t (window slices, transferred), width,
 *        height, cell, fmin, fmax, minEventsPerCell }
 * Out: { type:"progress", id, frac }
 *      { type:"done", id, cmd, result }
 *      { type:"error", id, message }
 */
/* global importScripts, GottLUXAnalysis */
importScripts("analysis.js");

self.onmessage = function (e) {
  var d = e.data, id = d.id;
  try {
    if (d.cmd === "spectrum") {
      var spec = GottLUXAnalysis.regionSpectrum(d.t, { fmin: d.fmin, fmax: d.fmax });
      self.postMessage({ type: "done", id: id, cmd: "spectrum", result: spec },
                       [spec.freqs.buffer, spec.power.buffer]);
    } else if (d.cmd === "flicker") {
      var lastPost = 0;
      var fm = GottLUXAnalysis.flickerMap({
        x: d.x, y: d.y, t: d.t, width: d.width, height: d.height,
        cell: d.cell, fmin: d.fmin, fmax: d.fmax,
        minEventsPerCell: d.minEventsPerCell,
        onProgress: function (frac) {
          var now = Date.now();
          if (now - lastPost > 80 || frac >= 1) {
            lastPost = now;
            self.postMessage({ type: "progress", id: id, frac: frac });
          }
        }
      });
      self.postMessage({ type: "done", id: id, cmd: "flicker", result: fm },
                       [fm.freq.buffer, fm.snr.buffer, fm.count.buffer]);
    } else {
      throw new Error("unknown cmd: " + d.cmd);
    }
  } catch (err) {
    self.postMessage({ type: "error", id: id, message: String(err && err.message || err) });
  }
};
