/*
 * decode_worker.js — runs GottLUXDecoder.decode off the main thread.
 *
 * In:  { id, buffer }                        (buffer is transferred in)
 * Out: { type:"progress", id, frac, stage }
 *      { type:"done", id, result }           (event arrays transferred out)
 *      { type:"error", id, message }
 */
/* global importScripts, GottLUXDecoder */
importScripts("decoder.js");

self.onmessage = function (e) {
  var id = e.data.id;
  try {
    var lastPost = 0;
    var res = GottLUXDecoder.decode(e.data.buffer, function (frac, stage) {
      var now = Date.now();
      if (now - lastPost > 60 || frac >= 1) {
        lastPost = now;
        self.postMessage({ type: "progress", id: id, frac: frac, stage: stage });
      }
    });
    self.postMessage(
      { type: "done", id: id,
        result: { x: res.x, y: res.y, p: res.p, t: res.t, t0_us: res.t0_us,
                  n: res.n, width: res.width, height: res.height,
                  fmt: res.fmt, meta: res.meta } },
      [res.x.buffer, res.y.buffer, res.p.buffer, res.t.buffer]
    );
  } catch (err) {
    self.postMessage({ type: "error", id: id, message: String(err && err.message || err) });
  }
};
