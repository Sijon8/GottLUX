/*
 * looks.js — the Studio's visualization methods ("looks") and frame renderer, in the
 * browser. A port of gottlux/studio (looks.py, render.py, shot.py); method notes are in
 * docs/STUDIO.md. The look/palette/preset/colormap SPEC block below is generated from the
 * Python package by scripts/sync_web_studio.py — edit the Python, then re-run it.
 *
 * Exposes GottLUXStudio = { SPEC, LOOKS, Renderer, defaultShot, resolve, speedText, ... }.
 * Times are seconds on the recording clock (the decoder's t, already zero-based).
 */
(function (global) {
  "use strict";

  /*SPECS-BEGIN*/
  var SPEC = {
 "looks": {
  "trails": {
   "title": "Polarity trails",
   "description": "Every event leaves a light that fades as exp(-age/T): ON events in one colour, OFF in the other. Motion draws comet tails; a beating wing draws alternating ON/OFF bands.",
   "native": true,
   "heavy": false,
   "params": [
    {
     "key": "tau_ms",
     "label": "Trail T",
     "default": 6.0,
     "lo": 0.2,
     "hi": 500.0,
     "step": 0.1,
     "kind": "float",
     "choices": [],
     "unit": "ms",
     "group": "Look",
     "help": "Fade time constant: an event's light drops to 37 % after T."
    },
    {
     "key": "saturate",
     "label": "Saturate at",
     "default": 2.5,
     "lo": 0.2,
     "hi": 50.0,
     "step": 0.1,
     "kind": "float",
     "choices": [],
     "unit": "",
     "group": "Look",
     "help": "Summed (faded) events per pixel that reach full brightness."
    },
    {
     "key": "polarity",
     "label": "Polarity",
     "default": "both",
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.0,
     "kind": "choice",
     "choices": [
      "both",
      "on",
      "off"
     ],
     "unit": "",
     "group": "Look",
     "help": "Show both polarities or only one."
    }
   ]
  },
  "heat": {
   "title": "Accumulated frame",
   "description": "The classic event frame: events per pixel in a sliding window [τ − W, τ], tone-mapped through a colormap, with one exposure for the whole shot.",
   "native": true,
   "heavy": false,
   "params": [
    {
     "key": "window_ms",
     "label": "Window W",
     "default": 10.0,
     "lo": 0.5,
     "hi": 1000.0,
     "step": 0.5,
     "kind": "float",
     "choices": [],
     "unit": "ms",
     "group": "Look",
     "help": "Accumulation window behind each frame."
    },
    {
     "key": "colormap",
     "label": "Colormap",
     "default": "inferno",
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.0,
     "kind": "choice",
     "choices": [
      "ice",
      "ember",
      "inferno",
      "magma",
      "viridis",
      "cividis",
      "gray",
      "turbo"
     ],
     "unit": "",
     "group": "Look",
     "help": ""
    },
    {
     "key": "tone",
     "label": "Tone map",
     "default": "sqrt",
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.0,
     "kind": "choice",
     "choices": [
      "linear",
      "sqrt",
      "log"
     ],
     "unit": "",
     "group": "Look",
     "help": "Compresses bright regions so faint motion stays visible."
    },
    {
     "key": "signed",
     "label": "ON − OFF",
     "default": false,
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.0,
     "kind": "bool",
     "choices": [],
     "unit": "",
     "group": "Look",
     "help": "Show the polarity balance in the palette's ON/OFF colours instead."
    }
   ]
  },
  "surface": {
   "title": "Time surface",
   "description": "Each pixel shows how recently it fired: exp(−(τ − T_last)/T), where T_last is the time of the pixel's latest event. Edges glow at the front of a moving object and fade behind it (the 'surface of active events').",
   "native": true,
   "heavy": false,
   "params": [
    {
     "key": "tau_ms",
     "label": "Decay T",
     "default": 25.0,
     "lo": 1.0,
     "hi": 1000.0,
     "step": 0.5,
     "kind": "float",
     "choices": [],
     "unit": "ms",
     "group": "Look",
     "help": ""
    },
    {
     "key": "color",
     "label": "Colour by",
     "default": "polarity",
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.0,
     "kind": "choice",
     "choices": [
      "polarity",
      "colormap"
     ],
     "unit": "",
     "group": "Look",
     "help": "Polarity of the last event, or a colormap of recency."
    },
    {
     "key": "colormap",
     "label": "Colormap",
     "default": "ice",
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.0,
     "kind": "choice",
     "choices": [
      "ice",
      "ember",
      "inferno",
      "magma",
      "viridis",
      "cividis",
      "gray",
      "turbo"
     ],
     "unit": "",
     "group": "Look",
     "help": ""
    }
   ]
  },
  "paint": {
   "title": "Light painting",
   "description": "A long exposure that builds up over the shot: every event since the start, brightness = log(events per pixel), hue = when the pixel last fired (early → late along the colormap). Flight paths draw themselves.",
   "native": true,
   "heavy": false,
   "params": [
    {
     "key": "colormap",
     "label": "Time colormap",
     "default": "ice",
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.0,
     "kind": "choice",
     "choices": [
      "ice",
      "ember",
      "inferno",
      "magma",
      "viridis",
      "cividis",
      "gray",
      "turbo"
     ],
     "unit": "",
     "group": "Look",
     "help": ""
    },
    {
     "key": "min_count",
     "label": "Min events",
     "default": 2,
     "lo": 1,
     "hi": 20,
     "step": 1,
     "kind": "int",
     "choices": [],
     "unit": "",
     "group": "Look",
     "help": "Pixels with fewer events stay dark (drops residual shot noise)."
    },
    {
     "key": "fade_s",
     "label": "Fade after",
     "default": 0.0,
     "lo": 0.0,
     "hi": 30.0,
     "step": 0.05,
     "kind": "float",
     "choices": [],
     "unit": "s",
     "group": "Look",
     "help": "0 = keep everything; otherwise only the last N seconds are painted."
    }
   ]
  },
  "spacetime": {
   "title": "Space-time cloud",
   "description": "The events of the last D milliseconds as points in a 3-D box (x, y, time); the front face is 'now'. A beating wing becomes a row of fins along the time axis, a moving target a tube. Set Orbit to swing the camera around.",
   "native": false,
   "heavy": true,
   "params": [
    {
     "key": "depth_ms",
     "label": "Depth D",
     "default": 300.0,
     "lo": 5.0,
     "hi": 3000.0,
     "step": 5.0,
     "kind": "float",
     "choices": [],
     "unit": "ms",
     "group": "Look",
     "help": "Length of the time axis."
    },
    {
     "key": "time_scale",
     "label": "Time stretch",
     "default": 1.6,
     "lo": 0.3,
     "hi": 6.0,
     "step": 0.1,
     "kind": "float",
     "choices": [],
     "unit": "",
     "group": "Look",
     "help": "Length of the time axis relative to the sensor's larger side."
    },
    {
     "key": "azimuth",
     "label": "Azimuth",
     "default": -55.0,
     "lo": -180.0,
     "hi": 180.0,
     "step": 1.0,
     "kind": "float",
     "choices": [],
     "unit": "°",
     "group": "Camera",
     "help": ""
    },
    {
     "key": "elevation",
     "label": "Elevation",
     "default": 22.0,
     "lo": -80.0,
     "hi": 80.0,
     "step": 1.0,
     "kind": "float",
     "choices": [],
     "unit": "°",
     "group": "Camera",
     "help": ""
    },
    {
     "key": "orbit_deg",
     "label": "Orbit",
     "default": 0.0,
     "lo": -720.0,
     "hi": 720.0,
     "step": 5.0,
     "kind": "float",
     "choices": [],
     "unit": "°",
     "group": "Camera",
     "help": "Total camera rotation over the shot (360 = one full orbit)."
    },
    {
     "key": "freeze",
     "label": "Freeze time",
     "default": false,
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.0,
     "kind": "bool",
     "choices": [],
     "unit": "",
     "group": "Camera",
     "help": "Hold the cloud at the shot's end [t1 − D, t1] and let only the camera move (with Orbit: a turntable of one moment)."
    },
    {
     "key": "perspective",
     "label": "Perspective",
     "default": 0.35,
     "lo": 0.0,
     "hi": 0.9,
     "step": 0.05,
     "kind": "float",
     "choices": [],
     "unit": "",
     "group": "Camera",
     "help": ""
    },
    {
     "key": "color",
     "label": "Colour by",
     "default": "polarity",
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.0,
     "kind": "choice",
     "choices": [
      "polarity",
      "age"
     ],
     "unit": "",
     "group": "Look",
     "help": ""
    },
    {
     "key": "colormap",
     "label": "Age colormap",
     "default": "ice",
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.0,
     "kind": "choice",
     "choices": [
      "ice",
      "ember",
      "inferno",
      "magma",
      "viridis",
      "cividis",
      "gray",
      "turbo"
     ],
     "unit": "",
     "group": "Look",
     "help": ""
    },
    {
     "key": "box",
     "label": "Box",
     "default": true,
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.0,
     "kind": "bool",
     "choices": [],
     "unit": "",
     "group": "Look",
     "help": "Draw the space-time box."
    },
    {
     "key": "point_px",
     "label": "Point size",
     "default": 2,
     "lo": 1,
     "hi": 4,
     "step": 1,
     "kind": "int",
     "choices": [],
     "unit": "px",
     "group": "Look",
     "help": ""
    },
    {
     "key": "brightness",
     "label": "Brightness",
     "default": 1.0,
     "lo": 0.1,
     "hi": 5.0,
     "step": 0.1,
     "kind": "float",
     "choices": [],
     "unit": "",
     "group": "Look",
     "help": ""
    },
    {
     "key": "max_points",
     "label": "Max points",
     "default": 120000,
     "lo": 5000,
     "hi": 2000000,
     "step": 5000,
     "kind": "int",
     "choices": [],
     "unit": "",
     "group": "Look",
     "help": "Subsample above this many events per frame (keeps rendering quick)."
    }
   ]
  },
  "strobe": {
   "title": "Event stroboscope",
   "description": "For a periodic target (wingbeat, rotor): measure its frequency f0, fold every event by its phase t·f0 mod 1, and play the phase bins as frames. Dozens of cycles stack onto one, like a strobe light freezing a fan. Crop to the target first so f0 is measured on it.",
   "native": true,
   "heavy": false,
   "params": [
    {
     "key": "f0_hz",
     "label": "f0 (0 = auto)",
     "default": 0.0,
     "lo": 0.0,
     "hi": 2000.0,
     "step": 0.1,
     "kind": "float",
     "choices": [],
     "unit": "Hz",
     "group": "Look",
     "help": "Fold frequency; 0 measures it from the event-rate spectrum in the band."
    },
    {
     "key": "band_lo",
     "label": "Search from",
     "default": 15.0,
     "lo": 1.0,
     "hi": 2000.0,
     "step": 1.0,
     "kind": "float",
     "choices": [],
     "unit": "Hz",
     "group": "Look",
     "help": ""
    },
    {
     "key": "band_hi",
     "label": "Search to",
     "default": 150.0,
     "lo": 2.0,
     "hi": 4000.0,
     "step": 1.0,
     "kind": "float",
     "choices": [],
     "unit": "Hz",
     "group": "Look",
     "help": ""
    },
    {
     "key": "bins",
     "label": "Frames per cycle",
     "default": 24,
     "lo": 4,
     "hi": 120,
     "step": 1,
     "kind": "int",
     "choices": [],
     "unit": "",
     "group": "Look",
     "help": ""
    },
    {
     "key": "width",
     "label": "Phase window",
     "default": 0.08,
     "lo": 0.01,
     "hi": 0.5,
     "step": 0.01,
     "kind": "float",
     "choices": [],
     "unit": "",
     "group": "Look",
     "help": "Fraction of a cycle each frame collects (> 1/bins overlaps → smoother)."
    },
    {
     "key": "cycles",
     "label": "Loops",
     "default": 2,
     "lo": 1,
     "hi": 20,
     "step": 1,
     "kind": "int",
     "choices": [],
     "unit": "",
     "group": "Look",
     "help": "How many times the cycle repeats in the output."
    }
   ]
  },
  "waterfall": {
   "title": "Waterfall (kymograph)",
   "description": "The live frame on top and, below it, the last H milliseconds collapsed along one axis and stacked by time (newest row at the top). A moving target draws a slanted track, a beating wing a ladder of stripes.",
   "native": true,
   "heavy": false,
   "params": [
    {
     "key": "axis",
     "label": "Collapse",
     "default": "y",
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.0,
     "kind": "choice",
     "choices": [
      "y",
      "x"
     ],
     "unit": "",
     "group": "Look",
     "help": "'y': history of each column (strip below the frame); 'x': history of each row (strip to the right)."
    },
    {
     "key": "history_ms",
     "label": "History H",
     "default": 400.0,
     "lo": 10.0,
     "hi": 10000.0,
     "step": 10.0,
     "kind": "float",
     "choices": [],
     "unit": "ms",
     "group": "Look",
     "help": ""
    },
    {
     "key": "rows",
     "label": "History px",
     "default": 160,
     "lo": 32,
     "hi": 1024,
     "step": 8,
     "kind": "int",
     "choices": [],
     "unit": "",
     "group": "Look",
     "help": ""
    },
    {
     "key": "tau_ms",
     "label": "Frame trail",
     "default": 6.0,
     "lo": 0.2,
     "hi": 200.0,
     "step": 0.1,
     "kind": "float",
     "choices": [],
     "unit": "ms",
     "group": "Look",
     "help": ""
    }
   ]
  },
  "flicker": {
   "title": "Flicker hue",
   "description": "Where the scene flickers and how fast: each cell's dominant frequency in a sliding window (hue, turbo colormap over the band), opacity = spectral SNR above the scene's noise floor, over dimmed live trails. Rotors and wings light up in their own colour.",
   "native": true,
   "heavy": true,
   "params": [
    {
     "key": "window_ms",
     "label": "Window",
     "default": 250.0,
     "lo": 50.0,
     "hi": 2000.0,
     "step": 10.0,
     "kind": "float",
     "choices": [],
     "unit": "ms",
     "group": "Look",
     "help": "Spectrum window; longer = finer frequency resolution (1/window)."
    },
    {
     "key": "cell",
     "label": "Cell",
     "default": 8,
     "lo": 2,
     "hi": 32,
     "step": 1,
     "kind": "int",
     "choices": [],
     "unit": "px",
     "group": "Look",
     "help": ""
    },
    {
     "key": "band_lo",
     "label": "Band from",
     "default": 15.0,
     "lo": 1.0,
     "hi": 4000.0,
     "step": 1.0,
     "kind": "float",
     "choices": [],
     "unit": "Hz",
     "group": "Look",
     "help": ""
    },
    {
     "key": "band_hi",
     "label": "Band to",
     "default": 150.0,
     "lo": 2.0,
     "hi": 5000.0,
     "step": 1.0,
     "kind": "float",
     "choices": [],
     "unit": "Hz",
     "group": "Look",
     "help": ""
    },
    {
     "key": "underlay",
     "label": "Trails",
     "default": 0.5,
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.05,
     "kind": "float",
     "choices": [],
     "unit": "",
     "group": "Look",
     "help": "Brightness of the live trails underneath."
    }
   ]
  },
  "radar": {
   "title": "Radar sweep (rotating sensor)",
   "description": "For a spinning sensor: every event is placed at its world bearing on a 360° dial — the rotation phase t/P plus the pixel's offset across the field of view — with the sensor row as radius (sky inside, ground at the rim, like a fisheye looking up). Events fade like radar phosphor; the wedge is where the sensor is looking now.",
   "native": false,
   "heavy": false,
   "params": [
    {
     "key": "period_s",
     "label": "Spin period (0 = auto)",
     "default": 0.0,
     "lo": 0.0,
     "hi": 30.0,
     "step": 0.0001,
     "kind": "float",
     "choices": [],
     "unit": "s",
     "group": "Look",
     "help": "Rotation period; 0 estimates it from the event-rate periodicity (needs two or more revolutions)."
    },
    {
     "key": "fov_deg",
     "label": "Field of view",
     "default": 58.0,
     "lo": 5.0,
     "hi": 180.0,
     "step": 0.5,
     "kind": "float",
     "choices": [],
     "unit": "°",
     "group": "Look",
     "help": "Horizontal field of view of the sensor (GenX320 default 58°)."
    },
    {
     "key": "direction",
     "label": "Rotation",
     "default": "cw",
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.0,
     "kind": "choice",
     "choices": [
      "cw",
      "ccw"
     ],
     "unit": "",
     "group": "Look",
     "help": "Direction the boresight turns on the dial."
    },
    {
     "key": "mirror",
     "label": "Column sign",
     "default": "auto",
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.0,
     "kind": "choice",
     "choices": [
      "auto",
      "+1",
      "-1"
     ],
     "unit": "",
     "group": "Look",
     "help": "Sign of the pixel-column → bearing offset; 'auto' picks the sharper panorama (the wrong sign smears static objects over twice the field of view)."
    },
    {
     "key": "persistence_ms",
     "label": "Persistence",
     "default": 250.0,
     "lo": 5.0,
     "hi": 20000.0,
     "step": 5.0,
     "kind": "float",
     "choices": [],
     "unit": "ms",
     "group": "Look",
     "help": "Phosphor fade time constant."
    },
    {
     "key": "radius",
     "label": "Radius",
     "default": "ground-out",
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.0,
     "kind": "choice",
     "choices": [
      "ground-out",
      "sky-out"
     ],
     "unit": "",
     "group": "Look",
     "help": "Which end of the sensor's rows sits at the rim."
    },
    {
     "key": "max_points",
     "label": "Max points",
     "default": 250000,
     "lo": 10000,
     "hi": 5000000,
     "step": 10000,
     "kind": "int",
     "choices": [],
     "unit": "",
     "group": "Look",
     "help": "Subsample above this many events per frame (keeps rendering quick)."
    },
    {
     "key": "saturate",
     "label": "Saturate at",
     "default": 3.0,
     "lo": 0.2,
     "hi": 100.0,
     "step": 0.1,
     "kind": "float",
     "choices": [],
     "unit": "",
     "group": "Look",
     "help": ""
    },
    {
     "key": "dial",
     "label": "Dial",
     "default": true,
     "lo": 0.0,
     "hi": 1.0,
     "step": 0.0,
     "kind": "bool",
     "choices": [],
     "unit": "",
     "group": "Look",
     "help": "Range rings, 30° bearing ticks and the field-of-view wedge."
    }
   ]
  }
 },
 "palettes": {
  "fire-ice": {
   "on": "#ff7a3d",
   "off": "#3f97ff",
   "bg": "#05070a"
  },
  "ember": {
   "on": "#ffb347",
   "off": "#c2185b",
   "bg": "#07040a"
  },
  "aurora": {
   "on": "#7cf5c4",
   "off": "#8a6cff",
   "bg": "#03060a"
  },
  "gottlux": {
   "on": "#39c5cf",
   "off": "#39c5cf",
   "bg": "#05080b"
  },
  "mono": {
   "on": "#ffffff",
   "off": "#ffffff",
   "bg": "#000000"
  },
  "paper": {
   "on": "#1a1a19",
   "off": "#1a1a19",
   "bg": "#f6f5f1"
  },
  "print": {
   "on": "#c0392b",
   "off": "#1f5fbf",
   "bg": "#fcfcfb"
  }
 },
 "presets": {
  "gif-small": [
   "gif",
   320,
   20
  ],
  "gif": [
   "gif",
   480,
   25
  ],
  "gif-large": [
   "gif",
   640,
   25
  ],
  "mp4": [
   "mp4",
   720,
   30
  ],
  "mp4-hd": [
   "mp4",
   1080,
   30
  ],
  "mp4-smooth": [
   "mp4",
   1080,
   60
  ],
  "webm": [
   "webm",
   720,
   30
  ]
 },
 "colormaps": {
  "ice": [
   [
    0.03920000046491623,
    0.10199999809265137,
    0.20000000298023224
   ],
   [
    0.048900000751018524,
    0.16660000383853912,
    0.32600000500679016
   ],
   [
    0.05920000001788139,
    0.2354000061750412,
    0.4602999985218048
   ],
   [
    0.07639999687671661,
    0.28839999437332153,
    0.5536999702453613
   ],
   [
    0.09709999710321426,
    0.33320000767707825,
    0.6259999871253967
   ],
   [
    0.11909999698400497,
    0.37929999828338623,
    0.6990000009536743
   ],
   [
    0.14319999516010284,
    0.42750000953674316,
    0.7731000185012817
   ],
   [
    0.17260000109672546,
    0.476500004529953,
    0.8424000144004822
   ],
   [
    0.2467000037431717,
    0.5315999984741211,
    0.8715999722480774
   ],
   [
    0.3208000063896179,
    0.5867000222206116,
    0.9009000062942505
   ],
   [
    0.4034000039100647,
    0.6389999985694885,
    0.9172999858856201
   ],
   [
    0.4878000020980835,
    0.6906999945640564,
    0.9311000108718872
   ],
   [
    0.5932000279426575,
    0.7556999921798706,
    0.9487000107765198
   ],
   [
    0.715499997138977,
    0.8314999938011169,
    0.9693999886512756
   ],
   [
    0.8277999758720398,
    0.9000999927520752,
    0.9861999750137329
   ],
   [
    0.9139000177383423,
    0.949999988079071,
    0.9930999875068665
   ],
   [
    1.0,
    1.0,
    1.0
   ]
  ],
  "ember": [
   [
    0.0706000030040741,
    0.015699999406933784,
    0.015699999406933784
   ],
   [
    0.14810000360012054,
    0.026799999177455902,
    0.01850000023841858
   ],
   [
    0.23080000281333923,
    0.038600001484155655,
    0.021400000900030136
   ],
   [
    0.3188000023365021,
    0.05660000070929527,
    0.023499999195337296
   ],
   [
    0.4207000136375427,
    0.09059999883174896,
    0.023499999195337296
   ],
   [
    0.522599995136261,
    0.12449999898672104,
    0.023499999195337296
   ],
   [
    0.6226000189781189,
    0.17319999635219574,
    0.029100000858306885
   ],
   [
    0.7214999794960022,
    0.23080000281333923,
    0.03790000081062317
   ],
   [
    0.8203999996185303,
    0.28839999437332153,
    0.046799998730421066
   ],
   [
    0.8736000061035156,
    0.3817000091075897,
    0.07419999688863754
   ],
   [
    0.9253000020980835,
    0.47620001435279846,
    0.1023000031709671
   ],
   [
    0.965399980545044,
    0.5688999891281128,
    0.14880000054836273
   ],
   [
    0.9801999926567078,
    0.6574000120162964,
    0.23589999973773956
   ],
   [
    0.9948999881744385,
    0.7459999918937683,
    0.3230000138282776
   ],
   [
    1.0,
    0.8201000094413757,
    0.45559999346733093
   ],
   [
    1.0,
    0.8865000009536743,
    0.6121000051498413
   ],
   [
    1.0,
    0.9528999924659729,
    0.7685999870300293
   ]
  ],
  "inferno": [
   [
    0.001500000013038516,
    0.0005000000237487257,
    0.013899999670684338
   ],
   [
    0.037700001150369644,
    0.02590000070631504,
    0.13220000267028809
   ],
   [
    0.12290000170469284,
    0.04749999940395355,
    0.2815999984741211
   ],
   [
    0.23149999976158142,
    0.036400001496076584,
    0.39239999651908875
   ],
   [
    0.3352000117301941,
    0.060100000351667404,
    0.4284999966621399
   ],
   [
    0.4350000023841858,
    0.09709999710321426,
    0.4320000112056732
   ],
   [
    0.5346999764442444,
    0.13249999284744263,
    0.41670000553131104
   ],
   [
    0.6340000033378601,
    0.16899999976158142,
    0.38370001316070557
   ],
   [
    0.7299000024795532,
    0.21279999613761902,
    0.33390000462532043
   ],
   [
    0.817300021648407,
    0.2709999978542328,
    0.2703999876976013
   ],
   [
    0.8902999758720398,
    0.3479999899864197,
    0.19830000400543213
   ],
   [
    0.9442999958992004,
    0.44279998540878296,
    0.12039999663829803
   ],
   [
    0.9771000146865845,
    0.5508000254631042,
    0.03909999877214432
   ],
   [
    0.9879000186920166,
    0.6676999926567078,
    0.05829999968409538
   ],
   [
    0.9761000275611877,
    0.7900000214576721,
    0.19599999487400055
   ],
   [
    0.9487000107765198,
    0.9104999899864197,
    0.3953000009059906
   ],
   [
    0.9883999824523926,
    0.9983999729156494,
    0.6449000239372253
   ]
  ],
  "magma": [
   [
    0.001500000013038516,
    0.0005000000237487257,
    0.013899999670684338
   ],
   [
    0.03550000116229057,
    0.0284000001847744,
    0.12520000338554382
   ],
   [
    0.10790000110864639,
    0.06430000066757202,
    0.267300009727478
   ],
   [
    0.20489999651908875,
    0.06289999932050705,
    0.4115000069141388
   ],
   [
    0.31040000915527344,
    0.06970000267028809,
    0.4832000136375427
   ],
   [
    0.40860000252723694,
    0.10790000110864639,
    0.5041000247001648
   ],
   [
    0.506600022315979,
    0.1459999978542328,
    0.5077999830245972
   ],
   [
    0.607200026512146,
    0.17980000376701355,
    0.49950000643730164
   ],
   [
    0.7099999785423279,
    0.21279999613761902,
    0.4772000014781952
   ],
   [
    0.8108999729156494,
    0.25290000438690186,
    0.439300000667572
   ],
   [
    0.8996000289916992,
    0.31459999084472656,
    0.39100000262260437
   ],
   [
    0.9585000276565552,
    0.41130000352859497,
    0.36000001430511475
   ],
   [
    0.9857000112533569,
    0.5281000137329102,
    0.37940001487731934
   ],
   [
    0.9958000183105469,
    0.6463000178337097,
    0.4413999915122986
   ],
   [
    0.996999979019165,
    0.7623999714851379,
    0.5288000106811523
   ],
   [
    0.9927999973297119,
    0.8772000074386597,
    0.6330999732017517
   ],
   [
    0.9871000051498413,
    0.9914000034332275,
    0.7494999766349792
   ]
  ],
  "viridis": [
   [
    0.2669999897480011,
    0.004900000058114529,
    0.3294000029563904
   ],
   [
    0.28189998865127563,
    0.08969999849796295,
    0.4124000072479248
   ],
   [
    0.27959999442100525,
    0.17059999704360962,
    0.47999998927116394
   ],
   [
    0.2606000006198883,
    0.24690000712871552,
    0.5228000283241272
   ],
   [
    0.23170000314712524,
    0.3181000053882599,
    0.5447999835014343
   ],
   [
    0.2011999934911728,
    0.38370001316070557,
    0.5543000102043152
   ],
   [
    0.17430000007152557,
    0.4449999928474426,
    0.5577999949455261
   ],
   [
    0.15049999952316284,
    0.5044000148773193,
    0.5573999881744385
   ],
   [
    0.12870000302791595,
    0.5633000135421753,
    0.5511999726295471
   ],
   [
    0.1200999990105629,
    0.6222000122070312,
    0.5349000096321106
   ],
   [
    0.15389999747276306,
    0.6801999807357788,
    0.5041999816894531
   ],
   [
    0.2393999993801117,
    0.7355999946594238,
    0.45570001006126404
   ],
   [
    0.36070001125335693,
    0.7860000133514404,
    0.387800008058548
   ],
   [
    0.5062999725341797,
    0.8288000226020813,
    0.3003999888896942
   ],
   [
    0.6680999994277954,
    0.8619999885559082,
    0.19629999995231628
   ],
   [
    0.8353000283241272,
    0.8859999775886536,
    0.10260000079870224
   ],
   [
    0.9932000041007996,
    0.9061999917030334,
    0.14390000700950623
   ]
  ],
  "cividis": [
   [
    0.0,
    0.13510000705718994,
    0.30480000376701355
   ],
   [
    0.0,
    0.1761000007390976,
    0.4075999855995178
   ],
   [
    0.09560000151395798,
    0.2176000028848648,
    0.436599999666214
   ],
   [
    0.19030000269412994,
    0.26159998774528503,
    0.4262999892234802
   ],
   [
    0.2597000002861023,
    0.3050999939441681,
    0.4228000044822693
   ],
   [
    0.3206000030040741,
    0.34860000014305115,
    0.42590001225471497
   ],
   [
    0.3774000108242035,
    0.39239999651908875,
    0.4348999857902527
   ],
   [
    0.4318000078201294,
    0.43700000643730164,
    0.44999998807907104
   ],
   [
    0.48510000109672546,
    0.48249998688697815,
    0.47040000557899475
   ],
   [
    0.5440999865531921,
    0.5289000272750854,
    0.47189998626708984
   ],
   [
    0.6051999926567078,
    0.5767999887466431,
    0.4643999934196472
   ],
   [
    0.6679999828338623,
    0.6262000203132629,
    0.44929999113082886
   ],
   [
    0.7324000000953674,
    0.6773999929428101,
    0.42570000886917114
   ],
   [
    0.7985000014305115,
    0.7305999994277954,
    0.39259999990463257
   ],
   [
    0.8664000034332275,
    0.7860000133514404,
    0.3465999960899353
   ],
   [
    0.9366999864578247,
    0.8438000082969666,
    0.2809000015258789
   ],
   [
    0.9957000017166138,
    0.9093000292778015,
    0.21780000627040863
   ]
  ],
  "gray": [
   [
    0.0,
    0.0,
    0.0
   ],
   [
    0.058800000697374344,
    0.058800000697374344,
    0.058800000697374344
   ],
   [
    0.12160000205039978,
    0.12160000205039978,
    0.12160000205039978
   ],
   [
    0.1843000054359436,
    0.1843000054359436,
    0.1843000054359436
   ],
   [
    0.24709999561309814,
    0.24709999561309814,
    0.24709999561309814
   ],
   [
    0.30979999899864197,
    0.30979999899864197,
    0.30979999899864197
   ],
   [
    0.3725000023841858,
    0.3725000023841858,
    0.3725000023841858
   ],
   [
    0.43529999256134033,
    0.43529999256134033,
    0.43529999256134033
   ],
   [
    0.49799999594688416,
    0.49799999594688416,
    0.49799999594688416
   ],
   [
    0.5608000159263611,
    0.5608000159263611,
    0.5608000159263611
   ],
   [
    0.6234999895095825,
    0.6234999895095825,
    0.6234999895095825
   ],
   [
    0.6862999796867371,
    0.6862999796867371,
    0.6862999796867371
   ],
   [
    0.7490000128746033,
    0.7490000128746033,
    0.7490000128746033
   ],
   [
    0.8118000030517578,
    0.8118000030517578,
    0.8118000030517578
   ],
   [
    0.8744999766349792,
    0.8744999766349792,
    0.8744999766349792
   ],
   [
    0.9373000264167786,
    0.9373000264167786,
    0.9373000264167786
   ],
   [
    1.0,
    1.0,
    1.0
   ]
  ],
  "turbo": [
   [
    0.1899999976158142,
    0.07180000096559525,
    0.2321999967098236
   ],
   [
    0.2483000010251999,
    0.24140000343322754,
    0.6129000186920166
   ],
   [
    0.2757999897003174,
    0.41100001335144043,
    0.8794000148773193
   ],
   [
    0.26249998807907104,
    0.5697000026702881,
    0.9976999759674072
   ],
   [
    0.16529999673366547,
    0.7261999845504761,
    0.9315999746322632
   ],
   [
    0.09290000051259995,
    0.8587999939918518,
    0.7724000215530396
   ],
   [
    0.18490000069141388,
    0.9448000192642212,
    0.6071000099182129
   ],
   [
    0.4122999906539917,
    0.9926999807357788,
    0.3982999920845032
   ],
   [
    0.6331999897956848,
    0.9919999837875366,
    0.2393999993801117
   ],
   [
    0.795199990272522,
    0.9301999807357788,
    0.20389999449253082
   ],
   [
    0.9266999959945679,
    0.8202000260353088,
    0.2257000058889389
   ],
   [
    0.991599977016449,
    0.6840999722480774,
    0.2071000039577484
   ],
   [
    0.9858999848365784,
    0.504800021648407,
    0.13369999825954437
   ],
   [
    0.9261999726295471,
    0.3246999979019165,
    0.058400001376867294
   ],
   [
    0.8240000009536743,
    0.19179999828338623,
    0.019700000062584877
   ],
   [
    0.6754000186920166,
    0.08980000019073486,
    0.0044999998062849045
   ],
   [
    0.4796000123023987,
    0.015799999237060547,
    0.010599999688565731
   ]
  ]
 }
};
  /*SPECS-END*/

  // ------------------------------------------------------------------ colour helpers
  function hexRgb(c) {
    c = c.replace("#", "");
    return [parseInt(c.slice(0, 2), 16) / 255, parseInt(c.slice(2, 4), 16) / 255,
            parseInt(c.slice(4, 6), 16) / 255];
  }
  var LUTS = {};
  function lut(name) {
    if (LUTS[name]) return LUTS[name];
    var stops = SPEC.colormaps[name] || SPEC.colormaps.inferno;
    var out = new Float32Array(256 * 3), n = stops.length - 1;
    for (var i = 0; i < 256; i++) {
      var x = i / 255 * n, j = Math.min(Math.floor(x), n - 1), f = x - j;
      for (var c = 0; c < 3; c++) out[i * 3 + c] = stops[j][c] * (1 - f) + stops[j + 1][c] * f;
    }
    LUTS[name] = out;
    return out;
  }
  function tone(v, mode) {
    v = v < 0 ? 0 : (v > 1 ? 1 : v);
    if (mode === "sqrt") return Math.sqrt(v);
    if (mode === "log") return Math.log1p(30 * v) / Math.log1p(30);
    return v;
  }
  function clamp01(v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }

  // ------------------------------------------------------------------ events
  function lowerBound(t, v) { var lo = 0, hi = t.length; while (lo < hi) { var m = (lo + hi) >>> 1; if (t[m] < v) lo = m + 1; else hi = m; } return lo; }
  function upperBound(t, v) { var lo = 0, hi = t.length; while (lo < hi) { var m = (lo + hi) >>> 1; if (t[m] <= v) lo = m + 1; else hi = m; } return lo; }

  function Events(x, y, on, t, W, H, t0, t1) {
    this.x = x; this.y = y; this.on = on; this.t = t; this.W = W; this.H = H;
    this.t0 = t0; this.t1 = t1; this.n = t.length;
    var f = new Int32Array(this.n);
    for (var i = 0; i < this.n; i++) f[i] = y[i] * W + x[i];
    this.flat = f;
  }
  Events.prototype.span = function (a, b) { return [lowerBound(this.t, a), upperBound(this.t, b)]; };

  // last event per pixel in [i0, i1): time (NaN = none) and polarity
  Events.prototype.lastEvent = function (i0, i1) {
    var N = this.W * this.H, T = new Float64Array(N), P = new Uint8Array(N);
    T.fill(NaN);
    for (var i = i0; i < i1; i++) { var f = this.flat[i]; T[f] = this.t[i]; P[f] = this.on[i]; }
    return { T: T, P: P };
  };

  function lockedScale(samples, pct, floor) {
    var vals = [];
    samples.forEach(function (s) { for (var i = 0; i < s.length; i++) if (s[i] > 0) vals.push(s[i]); });
    if (!vals.length) return 1;
    vals.sort(function (a, b) { return a - b; });
    var pos = pct / 100 * (vals.length - 1), lo = Math.floor(pos), hi = Math.min(lo + 1, vals.length - 1);
    var v = vals[lo] + (vals[hi] - vals[lo]) * (pos - lo);          // numpy's linear percentile
    return Math.max(v, floor || 1e-6);
  }
  function sampleIdx(r, n) {
    var out = [], m = Math.min(n || 12, r.nFrames);
    for (var i = 0; i < m; i++) out.push(Math.floor(i * (r.nFrames - 1) / Math.max(m - 1, 1) + 1e-9));
    return out;
  }

  // ------------------------------------------------------------------ looks
  // render(ev, k, tau, P, st, colors, [w, h]) → layers [{a: Float32Array(w*h), c: [r,g,b] | Float32Array(w*h*3)}]
  var LOOKS = {};

  function trailsLayers(ev, tau, P, colors) {
    var T = 1e-3 * P.tau_ms, sp = ev.span(tau - 5 * T, tau), N = ev.W * ev.H;
    var on = new Float32Array(N), off = new Float32Array(N);
    for (var i = sp[0]; i < sp[1]; i++) {
      var w = Math.exp(-(tau - ev.t[i]) / T);
      if (ev.on[i]) on[ev.flat[i]] += w; else off[ev.flat[i]] += w;
    }
    var s = P.saturate, L = [];
    for (var j = 0; j < N; j++) { on[j] = clamp01(on[j] / s); off[j] = clamp01(off[j] / s); }
    if (P.polarity !== "off") L.push({ a: on, c: colors.on });
    if (P.polarity !== "on") L.push({ a: off, c: colors.off });
    return L;
  }

  LOOKS.trails = {
    memory: function (P) { return 5e-3 * P.tau_ms; },
    render: function (ev, k, tau, P, st, colors) { return trailsLayers(ev, tau, P, colors); }
  };

  LOOKS.heat = {
    memory: function (P) { return 1e-3 * P.window_ms; },
    usesPolarity: function (P) { return !!P.signed; },
    raw: function (ev, tau, P) {
      var sp = ev.span(tau - 1e-3 * P.window_ms, tau), N = ev.W * ev.H;
      var a = new Float32Array(N), b = new Float32Array(N);
      for (var i = sp[0]; i < sp[1]; i++) { if (ev.on[i]) a[ev.flat[i]] += 1; else b[ev.flat[i]] += 1; }
      return [a, b];
    },
    prepare: function (ev, r, P) {
      var self = this, s = sampleIdx(r).map(function (i) {
        var ab = self.raw(ev, r.times[i], P), out = new Float32Array(ab[0].length);
        for (var j = 0; j < out.length; j++) out[j] = P.signed ? Math.abs(ab[0][j] - ab[1][j]) : ab[0][j] + ab[1][j];
        return out;
      });
      return { scale: lockedScale(s, 99.5) };
    },
    render: function (ev, k, tau, P, st, colors) {
      var ab = this.raw(ev, tau, P), N = ab[0].length, s = st.scale;
      if (P.signed) {
        var pa = new Float32Array(N), pb = new Float32Array(N);
        for (var i = 0; i < N; i++) { var d = (ab[0][i] - ab[1][i]) / s; pa[i] = tone(d, P.tone); pb[i] = tone(-d, P.tone); }
        return [{ a: pa, c: colors.on }, { a: pb, c: colors.off }];
      }
      var v = new Float32Array(N), col = new Float32Array(N * 3), L = lut(P.colormap);
      for (var j = 0; j < N; j++) {
        var x = tone((ab[0][j] + ab[1][j]) / s, P.tone); v[j] = x;
        var q = Math.min(255, (x * 255) | 0) * 3; col[j * 3] = L[q]; col[j * 3 + 1] = L[q + 1]; col[j * 3 + 2] = L[q + 2];
      }
      return [{ a: v, c: col }];
    }
  };

  LOOKS.surface = {
    memory: function (P) { return 5e-3 * P.tau_ms; },
    usesPolarity: function (P) { return P.color === "polarity"; },
    render: function (ev, k, tau, P, st, colors) {
      var T = 1e-3 * P.tau_ms, sp = ev.span(tau - 5 * T, tau), le = ev.lastEvent(sp[0], sp[1]);
      var N = ev.W * ev.H, v = new Float32Array(N);
      for (var i = 0; i < N; i++) v[i] = isNaN(le.T[i]) ? 0 : Math.exp(-(tau - le.T[i]) / T);
      if (P.color === "colormap") {
        var col = new Float32Array(N * 3), L = lut(P.colormap);
        for (var j = 0; j < N; j++) { var q = Math.min(255, (v[j] * 255) | 0) * 3; col[j * 3] = L[q]; col[j * 3 + 1] = L[q + 1]; col[j * 3 + 2] = L[q + 2]; }
        return [{ a: v, c: col }];
      }
      var a = new Float32Array(N), b = new Float32Array(N);
      for (var m = 0; m < N; m++) { if (le.P[m]) a[m] = v[m]; else b[m] = v[m]; }
      return [{ a: a, c: colors.on }, { a: b, c: colors.off }];
    }
  };

  LOOKS.paint = {
    usesPolarity: function () { return false; },
    sp: function (ev, tau, P) { var a = P.fade_s <= 0 ? ev.t0 : Math.max(ev.t0, tau - P.fade_s); return ev.span(a, tau); },
    prepare: function (ev, r, P) {
      var sp = this.sp(ev, ev.t1, P), c = new Float32Array(ev.W * ev.H);
      for (var i = sp[0]; i < sp[1]; i++) c[ev.flat[i]] += 1;
      return { scale: lockedScale([c], 99.0) };
    },
    render: function (ev, k, tau, P, st) {
      var sp = this.sp(ev, tau, P), N = ev.W * ev.H, cnt = new Float32Array(N), last = new Float64Array(N);
      for (var i = sp[0]; i < sp[1]; i++) { cnt[ev.flat[i]] += 1; last[ev.flat[i]] = ev.t[i]; }
      var a = new Float32Array(N), col = new Float32Array(N * 3), L = lut(P.colormap);
      var span = Math.max(ev.t1 - ev.t0, 1e-9), ls = Math.log1p(st.scale);
      for (var j = 0; j < N; j++) {
        if (cnt[j] < P.min_count) continue;
        a[j] = clamp01(Math.log1p(cnt[j]) / ls);
        var fr = 0.15 + 0.85 * clamp01((last[j] - ev.t0) / span), q = Math.min(255, (fr * 255) | 0) * 3;
        col[j * 3] = L[q]; col[j * 3 + 1] = L[q + 1]; col[j * 3 + 2] = L[q + 2];
      }
      return [{ a: a, c: col }];
    }
  };

  LOOKS.spacetime = {
    native: false,
    memory: function (P) { return 1e-3 * P.depth_ms; },
    usesPolarity: function (P) { return P.color === "polarity"; },
    canvas: function () { return [4, 3]; },
    camera: function (ev, k, frames, P) {
      var D = P.time_scale * Math.max(ev.W, ev.H);
      return { az: (P.azimuth + P.orbit_deg * k / Math.max(frames, 1)) * Math.PI / 180,
               el: P.elevation * Math.PI / 180, D: D,
               R: 0.5 * Math.sqrt(ev.W * ev.W + ev.H * ev.H + D * D) };
    },
    project: function (X, Y, Z, cam, P, w, h, out) {
      var ca = Math.cos(cam.az), sa = Math.sin(cam.az), ce = Math.cos(cam.el), se = Math.sin(cam.el);
      var x1 = X * ca + Z * sa, z1 = -X * sa + Z * ca;
      var y2 = Y * ce - z1 * se, z2 = Y * se + z1 * ce;
      var f = 1 / (1 + P.perspective * z2 / cam.R), s = 0.94 * Math.min(w, h) / (2 * cam.R);
      out[0] = w / 2 + s * x1 * f; out[1] = h / 2 - s * y2 * f; out[2] = f;
    },
    accum: function (ev, k, tau, P, st, w, h, colors) {
      if (P.freeze) tau = ev.t1;
      var sp = ev.span(tau - 1e-3 * P.depth_ms, tau), n = sp[1] - sp[0];
      var stride = Math.max(1, Math.ceil(n / P.max_points)), cam = this.camera(ev, k, st.frames, P);
      var acc = new Float32Array(w * h), rgb = colors ? new Float32Array(w * h * 3) : null;
      var ps = P.point_px | 0, o = [0, 0, 0], L = P.color === "age" ? lut(P.colormap) : null;
      for (var i = sp[0]; i < sp[1]; i += stride) {
        var age = (tau - ev.t[i]) / (1e-3 * P.depth_ms);
        this.project(ev.x[i] - ev.W / 2, -(ev.y[i] - ev.H / 2), age * cam.D - cam.D / 2, cam, P, w, h, o);
        var u = o[0] | 0, v = o[1] | 0;
        if (u < 0 || v < 0 || u >= w - ps || v >= h - ps) continue;
        var wt = (1 - 0.6 * age) * o[2], cr, cg, cb;
        if (rgb) {
          if (L) { var q = Math.min(255, (clamp01(1 - age) * 255) | 0) * 3; cr = L[q]; cg = L[q + 1]; cb = L[q + 2]; }
          else { var c = ev.on[i] ? colors.on : colors.off; cr = c[0]; cg = c[1]; cb = c[2]; }
        }
        for (var dy = 0; dy < ps; dy++) for (var dx = 0; dx < ps; dx++) {
          var f = (v + dy) * w + u + dx;
          acc[f] += wt;
          if (rgb) { rgb[f * 3] += wt * cr; rgb[f * 3 + 1] += wt * cg; rgb[f * 3 + 2] += wt * cb; }
        }
      }
      return { acc: acc, rgb: rgb };
    },
    prepare: function (ev, r, P) {
      var st = { frames: r.nFrames, t1: ev.t1 }, self = this;
      var s = sampleIdx(r, 10).map(function (i) { return self.accum(ev, i, r.times[i], P, st, r.drawW, r.drawH, null).acc; });
      st.scale = lockedScale(s, 98.0);
      return st;
    },
    box: function (ev, k, P, st, w, h) {
      var cam = this.camera(ev, k, st.frames, P), img = new Float32Array(w * h), o = [0, 0, 0];
      var hx = ev.W / 2, hy = ev.H / 2, hz = cam.D / 2, c = [];
      [-1, 1].forEach(function (sx) { [-1, 1].forEach(function (sy) { [-1, 1].forEach(function (sz) { c.push([sx * hx, sy * hy, sz * hz]); }); }); });
      for (var i = 0; i < 8; i++) for (var j = i + 1; j < 8; j++) {
        var diff = 0; for (var d = 0; d < 3; d++) if (Math.abs(c[i][d] - c[j][d]) > 1e-9) diff++;
        if (diff !== 1) continue;
        var front = c[i][2] < 0 && c[j][2] < 0;
        for (var s = 0; s <= 400; s++) {
          var f = s / 400;
          this.project(c[i][0] * (1 - f) + c[j][0] * f, c[i][1] * (1 - f) + c[j][1] * f,
                       c[i][2] * (1 - f) + c[j][2] * f, cam, P, w, h, o);
          var u = o[0] | 0, v = o[1] | 0;
          if (u >= 0 && v >= 0 && u < w && v < h) img[v * w + u] = front ? 0.55 : 0.22;
        }
      }
      return img;
    },
    render: function (ev, k, tau, P, st, colors, size) {
      var w = size[0], h = size[1], r = this.accum(ev, k, tau, P, st, w, h, colors);
      var a = new Float32Array(w * h), col = new Float32Array(w * h * 3);
      for (var i = 0; i < w * h; i++) {
        var acc = r.acc[i]; if (acc <= 0) continue;
        a[i] = clamp01(P.brightness * acc / st.scale);
        col[i * 3] = r.rgb[i * 3] / acc; col[i * 3 + 1] = r.rgb[i * 3 + 1] / acc; col[i * 3 + 2] = r.rgb[i * 3 + 2] / acc;
      }
      var L = [{ a: a, c: col }];
      if (P.box) {
        var g = (colors.bg[0] + colors.bg[1] + colors.bg[2]) / 3 < 0.5 ? 0.75 : 0.25;
        L.unshift({ a: this.box(ev, k, P, st, w, h), c: [g, g, g] });
      }
      return L;
    },
    stamp: function (k, tau, P, st) {
      if (P.freeze) return "t = " + (st.t1 - 1e-3 * P.depth_ms).toFixed(3) + " – " + st.t1.toFixed(3) + " s";
      return "t = " + tau.toFixed(3) + " s";
    }
  };

  // strongest line of the event rate (+ the ON − OFF rate when polarities are given) in
  // [lo, hi]; a peak whose half frequency keeps ≥ 30 % of its power is taken as 2·f0
  function estimateF0(t, i0, i1, t0, t1, lo, hi, on) {
    var fs = Math.max(2000, 4 * hi), n = Math.max(Math.floor((t1 - t0) * fs), 8);
    var N = 1; while (N < n * 16) N <<= 1;
    function power(signed) {
      var re = new Float64Array(N), im = new Float64Array(N);
      for (var i = i0; i < i1; i++) {
        var b = Math.floor((t[i] - t0) / (t1 - t0) * n);
        if (b === n && t[i] === t1) b = n - 1;                    // numpy's closed last bin
        if (b >= 0 && b < n) re[b] += signed ? (on[i] ? 1 : -1) : 1;
      }
      var mean = 0; for (var j = 0; j < n; j++) mean += re[j]; mean /= n;
      for (var m = 0; m < n; m++) re[m] = (re[m] - mean) * (0.5 - 0.5 * Math.cos(2 * Math.PI * m / Math.max(n - 1, 1)));
      global.GottLUXAnalysis.fft(re, im);
      var P = new Float64Array((N >> 1) + 1), mx = 1e-30;
      for (var k = 0; k <= N >> 1; k++) { P[k] = re[k] * re[k] + im[k] * im[k]; if (k > 0 && P[k] > mx) mx = P[k]; }
      for (var k2 = 0; k2 < P.length; k2++) P[k2] /= mx;
      return P;
    }
    var P = power(false);
    if (on) { var Q = power(true); for (var z = 0; z < P.length; z++) P[z] += Q[z]; }
    var df = fs / N, best = -1, bi = -1;
    for (var q = Math.ceil(lo / df - 1e-9); q * df <= hi && q < P.length; q++) if (P[q] > best) { best = P[q]; bi = q; }
    if (bi < 1) return NaN;
    var half = Math.round(bi / 2);
    if (half * df >= lo && half > 1) {
      var bj = half - 2;
      for (var hh = half - 2; hh <= half + 2; hh++) if (P[hh] > P[bj]) bj = hh;
      if (P[bj] >= 0.3 * P[bi]) bi = bj;
    }
    var den = P[bi - 1] - 2 * P[bi] + P[bi + 1], d = den ? 0.5 * (P[bi - 1] - P[bi + 1]) / den : 0;
    return (bi + d) * df;
  }

  LOOKS.strobe = {
    nFrames: function (r, P) { return P.bins * P.cycles; },
    prepare: function (ev, r, P) {
      var sp = ev.span(ev.t0, ev.t1);
      var f0 = P.f0_hz || estimateF0(ev.t, sp[0], sp[1], ev.t0, ev.t1, P.band_lo, P.band_hi, ev.on);
      if (!(f0 > 0)) f0 = 0.5 * (P.band_lo + P.band_hi);
      var ph = new Float64Array(sp[1] - sp[0]);
      for (var i = sp[0]; i < sp[1]; i++) { var x = ev.t[i] * f0; ph[i - sp[0]] = x - Math.floor(x); }
      var st = { f0: f0, sp: sp, phase: ph }, self = this, samples = [];
      for (var b = 0; b < P.bins; b++) { var r2 = self.raw(ev, (b + 0.5) / P.bins, P, st); samples.push(r2[0]); }
      st.scale = lockedScale(samples, 99.5);
      return st;
    },
    raw: function (ev, phi, P, st) {
      var N = ev.W * ev.H, a = new Float32Array(N), b = new Float32Array(N), half = 0.5 * P.width;
      for (var i = st.sp[0]; i < st.sp[1]; i++) {
        var d = st.phase[i - st.sp[0]] - phi + 0.5; d = Math.abs(d - Math.floor(d) - 0.5);
        if (d >= half) continue;
        var w = 1 - d / half;
        if (ev.on[i]) a[ev.flat[i]] += w; else b[ev.flat[i]] += w;
      }
      return [a, b];
    },
    render: function (ev, k, tau, P, st, colors) {
      var ab = this.raw(ev, ((k % P.bins) + 0.5) / P.bins, P, st), s = st.scale;
      for (var i = 0; i < ab[0].length; i++) { ab[0][i] = clamp01(ab[0][i] / s); ab[1][i] = clamp01(ab[1][i] / s); }
      return [{ a: ab[0], c: colors.on }, { a: ab[1], c: colors.off }];
    },
    stamp: function (k, tau, P, st) {
      return "f0 = " + st.f0.toFixed(1) + " Hz · phase " + Math.round(360 * ((k % P.bins) + 0.5) / P.bins) + "°";
    }
  };

  LOOKS.waterfall = {
    memory: function (P) { return 1e-3 * Math.max(P.history_ms, 5 * P.tau_ms); },
    canvas: function (W, H, P) { return P.axis === "y" ? [W, H + P.rows] : [W + P.rows, H]; },
    hist: function (ev, tau, P) {
      var Hs = 1e-3 * P.history_ms, sp = ev.span(tau - Hs, tau), L = P.axis === "y" ? ev.W : ev.H;
      var a = new Float32Array(P.rows * L), b = new Float32Array(P.rows * L);
      for (var i = sp[0]; i < sp[1]; i++) {
        var age = Math.min(P.rows - 1, Math.max(0, Math.floor((tau - ev.t[i]) / Hs * P.rows)));
        var f = age * L + (P.axis === "y" ? ev.x[i] : ev.y[i]);
        if (ev.on[i]) a[f] += 1; else b[f] += 1;
      }
      return [a, b, L];
    },
    prepare: function (ev, r, P) {
      var self = this, s = sampleIdx(r).map(function (i) {
        var h = self.hist(ev, r.times[i], P), o = new Float32Array(h[0].length);
        for (var j = 0; j < o.length; j++) o[j] = h[0][j] + h[1][j];
        return o;
      });
      return { hscale: lockedScale(s, 99.5) };
    },
    render: function (ev, k, tau, P, st, colors) {
      var tr = trailsLayers(ev, tau, { tau_ms: P.tau_ms, saturate: 2.5, polarity: "both" }, colors);
      var h = this.hist(ev, tau, P), L = h[2], cs = this.canvas(ev.W, ev.H, P), cw = cs[0], ch = cs[1];
      var out = [], hs = st.hscale;
      for (var layer = 0; layer < 2; layer++) {
        var full = new Float32Array(cw * ch), img = tr[layer].a, hist = h[layer];
        for (var y = 0; y < ev.H; y++) for (var x = 0; x < ev.W; x++) full[y * cw + x] = img[y * ev.W + x];
        for (var r = 0; r < P.rows; r++) for (var p = 0; p < L; p++) {
          var v = clamp01(hist[r * L + p] / hs);
          if (P.axis === "y") full[(ev.H + r) * cw + p] = v; else full[p * cw + ev.W + r] = v;
        }
        out.push({ a: full, c: tr[layer].c });
      }
      var g = new Float32Array(cw * ch);
      if (P.axis === "y") for (var gx = 0; gx < cw; gx++) g[ev.H * cw + gx] = 0.5;
      else for (var gy = 0; gy < ch; gy++) g[gy * cw + ev.W] = 0.5;
      out.push({ a: g, c: [0.6, 0.6, 0.6] });
      return out;
    }
  };

  LOOKS.flicker = {
    memory: function (P) { return 1e-3 * P.window_ms; },
    usesPolarity: function () { return false; },
    render: function (ev, k, tau, P, st, colors) {
      var L = trailsLayers(ev, tau, { tau_ms: 6, saturate: 2.5, polarity: "both" }, colors);
      L.forEach(function (l) { for (var i = 0; i < l.a.length; i++) l.a[i] *= P.underlay; });
      var sp = ev.span(tau - 1e-3 * P.window_ms, tau);
      if (sp[1] - sp[0] < 50 || !global.GottLUXAnalysis) return L;
      var lo = P.band_lo, hi = Math.max(P.band_hi, lo + 1);
      var fm = global.GottLUXAnalysis.flickerMap({
        x: ev.x.subarray(sp[0], sp[1]), y: ev.y.subarray(sp[0], sp[1]), t: ev.t.subarray(sp[0], sp[1]),
        width: ev.W, height: ev.H, cell: P.cell, fmin: lo, fmax: hi, fs: Math.max(2000, 4 * hi),
        minEventsPerCell: 12 });
      // alpha from the noise floor (median valid cell SNR) to the 99.5th percentile (≥ 3× floor)
      var vals = []; for (var c = 0; c < fm.snr.length; c++) if (!isNaN(fm.freq[c])) vals.push(fm.snr[c]);
      if (!vals.length) return L;
      vals.sort(function (a, b) { return a - b; });
      var floor = vals[Math.floor((vals.length - 1) / 2)];
      var ref = Math.max(vals[Math.floor(0.995 * (vals.length - 1))], 3 * floor);
      var lf = Math.log1p(floor), lr = Math.max(Math.log1p(ref), lf + 1e-6), T = lut("turbo");
      var N = ev.W * ev.H, a = new Float32Array(N), col = new Float32Array(N * 3);
      for (var y = 0; y < ev.H; y++) for (var x = 0; x < ev.W; x++) {
        var ci = Math.min((y / P.cell) | 0, fm.gh - 1) * fm.gw + Math.min((x / P.cell) | 0, fm.gw - 1);
        if (isNaN(fm.freq[ci])) continue;
        var i2 = y * ev.W + x, q = Math.min(255, (clamp01((fm.freq[ci] - lo) / (hi - lo)) * 255) | 0) * 3;
        a[i2] = 0.85 * clamp01((Math.log1p(Math.max(fm.snr[ci], 0)) - lf) / (lr - lf));
        col[i2 * 3] = T[q]; col[i2 * 3 + 1] = T[q + 1]; col[i2 * 3 + 2] = T[q + 2];
      }
      L.push({ a: a, c: col });
      return L;
    },
    stamp: function (k, tau, P) { return "t = " + tau.toFixed(3) + " s · hue " + P.band_lo.toFixed(0) + "–" + P.band_hi.toFixed(0) + " Hz"; }
  };

  // rotation period from the event-rate autocorrelation (coarse: the desktop refines it
  // further from the per-revolution passes; set the period explicitly for exact parity)
  function estimatePeriod(t) {
    var n = t.length; if (n < 100) return null;
    var dur = t[n - 1] - t[0], bin = 0.005, nb = Math.floor(dur / bin) + 1;
    if (dur < 0.4) return null;
    var rate = new Float64Array(nb);
    for (var i = 0; i < n; i++) rate[Math.min(nb - 1, Math.floor((t[i] - t[0]) / bin))] += 1;
    var mean = 0; for (var j = 0; j < nb; j++) mean += rate[j]; mean /= nb;
    for (var k = 0; k < nb; k++) rate[k] -= mean;
    var c0 = 0; for (var a = 0; a < nb; a++) c0 += rate[a] * rate[a];
    if (c0 <= 0) return null;
    var lo = Math.ceil(0.2 / bin), hi = Math.min(Math.floor(5 / bin), Math.floor(nb / 2)), best = -1, bl = -1, acf = [];
    for (var lag = 0; lag <= hi + 1; lag++) {
      var s2 = 0; for (var q = 0; q + lag < nb; q++) s2 += rate[q] * rate[q + lag];
      acf.push(s2 / c0);
      if (lag >= lo && lag <= hi && s2 / c0 > best) { best = s2 / c0; bl = lag; }
    }
    if (bl < 0) return null;
    var den = acf[bl - 1] - 2 * acf[bl] + acf[bl + 1], d = den ? 0.5 * (acf[bl - 1] - acf[bl + 1]) / den : 0;
    return (bl + d) * bin;
  }

  LOOKS.radar = {
    native: false,
    memory: function (P) { return 3e-3 * P.persistence_ms; },
    canvas: function () { return [1, 1]; },
    bearing: function (t, x, W, tref, st, P) {
      var b = st.direction * 360 * (t - tref) / st.period + st.mirror * (x / W - 0.5) * P.fov_deg;
      b %= 360; return b < 0 ? b + 360 : b;
    },
    prepare: function (ev, r, P) {
      var period = P.period_s, tref = r.rec.t.length ? r.rec.t[0] : 0;
      if (!(period > 0)) period = estimatePeriod(r.rec.t) || estimatePeriod(ev.t) || 1.0;
      var st = { period: period, direction: P.direction === "cw" ? 1 : -1, mirror: 1, t_ref: tref };
      if (P.mirror === "auto") {
        var sp = ev.span(ev.t0, ev.t1), step = Math.max(1, Math.floor((sp[1] - sp[0]) / 400000));
        var rows = Math.max(ev.H >> 1, 1), score = {};
        [1, -1].forEach(function (m) {
          var h = new Float64Array(720 * rows), tmp = { period: period, direction: st.direction, mirror: m };
          for (var i = sp[0]; i < sp[1]; i += step) {
            var b = LOOKS.radar.bearing(ev.t[i], ev.x[i], ev.W, tref, tmp, P);
            h[Math.min(719, Math.floor(b * 2)) * rows + Math.min(rows - 1, Math.floor(ev.y[i] / ev.H * rows))] += 1;
          }
          var ss = 0; for (var k = 0; k < h.length; k++) ss += h[k] * h[k];
          score[m] = ss;
        });
        st.mirror = score[1] >= score[-1] ? 1 : -1;
      } else st.mirror = Number(P.mirror);
      var w = r.drawW, hgt = r.drawH, c = (w - 1) / 2;
      st.R = 0.47 * Math.min(w, hgt); st.r0 = 0.18 * st.R;
      st.rad = new Float32Array(w * hgt); st.ang = new Float32Array(w * hgt);
      var rings = new Float32Array(w * hgt);
      for (var yy = 0; yy < hgt; yy++) for (var xx = 0; xx < w; xx++) {
        var o = yy * w + xx, dx = xx - c, dy = yy - c, rad = Math.sqrt(dx * dx + dy * dy);
        var ang = Math.atan2(dx, -dy) * 180 / Math.PI; if (ang < 0) ang += 360;
        st.rad[o] = rad; st.ang[o] = ang;
        [0, 0.5, 1].forEach(function (f) { if (Math.abs(rad - (st.r0 + f * (st.R - st.r0))) < 0.7) rings[o] = 0.25; });
        var tk = ((ang + 15) % 30) - 15;
        if (Math.abs(tk) < 0.5 && rad > st.R * 0.94 && rad <= st.R) rings[o] = 0.5;
      }
      st.rings = rings;
      var self = this, s = sampleIdx(r, 8).map(function (i) {
        var ab = self.accum(ev, r.times[i], P, st, w, hgt), out = new Float32Array(ab[0].length);
        for (var j = 0; j < out.length; j++) out[j] = ab[0][j] + ab[1][j];
        return out;
      });
      st.scale = lockedScale(s, 97.0);
      return st;
    },
    accum: function (ev, tau, P, st, w, h) {
      var T = 1e-3 * P.persistence_ms, sp = ev.span(tau - 3 * T, tau);
      var step = Math.max(1, Math.ceil((sp[1] - sp[0]) / P.max_points)), c = (w - 1) / 2;
      var a = new Float32Array(w * h), b = new Float32Array(w * h);
      for (var i = sp[0]; i < sp[1]; i += step) {
        var br = this.bearing(ev.t[i], ev.x[i], ev.W, st.t_ref, st, P) * Math.PI / 180;
        var row = (ev.y[i] + 0.5) / ev.H, rr = st.r0 + (st.R - st.r0) * (P.radius === "ground-out" ? row : 1 - row);
        var u = Math.trunc(c + rr * Math.sin(br)), v = Math.trunc(c - rr * Math.cos(br));   // numpy astype(int)
        if (u < 0 || v < 0 || u >= w || v >= h) continue;
        var wt = Math.exp(-(tau - ev.t[i]) / T);
        if (ev.on[i]) a[v * w + u] += wt; else b[v * w + u] += wt;
      }
      return [a, b];
    },
    render: function (ev, k, tau, P, st, colors, size) {
      var w = size[0], h = size[1], ab = this.accum(ev, tau, P, st, w, h), s = st.scale / P.saturate * 3;
      var L = [];
      if (P.dial) {
        var bs = (st.direction * 360 * (tau - st.t_ref) / st.period) % 360; if (bs < 0) bs += 360;
        var wedge = new Float32Array(w * h), half = P.fov_deg / 2;
        for (var o = 0; o < w * h; o++) {
          if (st.rad[o] > st.R || st.rad[o] < st.r0) continue;
          var d = Math.abs(((st.ang[o] - bs + 540) % 360) - 180);
          if (d <= half) wedge[o] = 0.10;
          if (Math.abs(d - half) < 0.6) wedge[o] += 0.35;
        }
        var g = (colors.bg[0] + colors.bg[1] + colors.bg[2]) / 3 < 0.5 ? 0.75 : 0.25;
        L.push({ a: wedge, c: colors.on }, { a: st.rings, c: [g, g, g] });
      }
      for (var q = 0; q < w * h; q++) { ab[0][q] = clamp01(ab[0][q] / s); ab[1][q] = clamp01(ab[1][q] / s); }
      L.push({ a: ab[0], c: colors.on }, { a: ab[1], c: colors.off });
      return L;
    },
    stamp: function (k, tau, P, st) { return "t = " + tau.toFixed(3) + " s · " + st.period.toFixed(3) + " s/rev"; }
  };

  // ------------------------------------------------------------------ shot + renderer
  function defaultShot() {
    return { look: "trails", params: {}, t0: null, t1: null, crop: null, speed: 0.1, fps: 25,
             width: 480, palette: "fire-ice", denoise: true, baf_us: 2000, bloom: 0, gamma: 1,
             resample: "bilinear", stamp: true, speed_label: true, title: "", legend: false,
             rate_strip: false, pingpong: false, format: "gif", gif_colors: 128, quality: 8 };
  }

  function resolve(look, params) {
    var P = {};
    SPEC.looks[look].params.forEach(function (p) {
      var v = params && params[p.key] !== undefined ? params[p.key] : p.default;
      if (p.kind === "bool") v = !!v;
      else if (p.kind === "choice") v = p.choices.indexOf(v) >= 0 ? v : p.default;
      else { v = Math.min(p.hi, Math.max(p.lo, Number(v))); if (p.kind === "int") v = Math.round(v); }
      P[p.key] = v;
    });
    return P;
  }

  function speedText(s) {
    if (Math.abs(s - 1) < 1e-6) return "real time";
    var x = s < 1 ? 1 / s : s, num = (x >= 10 || Math.abs(x - Math.round(x)) < 0.05) ? x.toFixed(0) : x.toFixed(1);
    return num + "× " + (s < 1 ? "slower" : "faster");
  }

  function clampCrop(crop, W, H) {
    if (!crop) return null;
    var x0 = Math.max(0, Math.min(Math.round(Math.min(crop[0], crop[2])), W));
    var x1 = Math.max(0, Math.min(Math.round(Math.max(crop[0], crop[2])), W));
    var y0 = Math.max(0, Math.min(Math.round(Math.min(crop[1], crop[3])), H));
    var y1 = Math.max(0, Math.min(Math.round(Math.max(crop[1], crop[3])), H));
    if (x1 - x0 < 4 || y1 - y0 < 4) return null;
    if (x0 === 0 && y0 === 0 && x1 === W && y1 === H) return null;
    return [x0, y0, x1, y1];
  }

  // span, crop, denoise (hot-pixel cut at the 99.9th percentile + background-activity filter)
  function prepareEvents(rec, shot, t0, t1, lookback) {
    var crop = clampCrop(shot.crop, rec.width, rec.height) || [0, 0, rec.width, rec.height];
    var i0 = lowerBound(rec.t, Math.max(0, t0 - lookback)), i1 = upperBound(rec.t, t1);
    var W = crop[2] - crop[0], H = crop[3] - crop[1], keep = [];
    for (var i = i0; i < i1; i++) {
      var x = rec.x[i], y = rec.y[i];
      if (x >= crop[0] && x < crop[2] && y >= crop[1] && y < crop[3]) keep.push(i);
    }
    if (shot.denoise && keep.length) {
      var SW = rec.width, cnt = new Int32Array(rec.width * rec.height);
      keep.forEach(function (j) { cnt[rec.y[j] * SW + rec.x[j]]++; });
      var nz = []; for (var c = 0; c < cnt.length; c++) if (cnt[c] > 0) nz.push(cnt[c]);
      nz.sort(function (a, b) { return a - b; });
      var thr = Math.max(nz[Math.floor(0.999 * (nz.length - 1))], 1);
      var Tl = new Float64Array(rec.width * rec.height); Tl.fill(-1e18);
      var dt = shot.baf_us * 1e-6, kept = [];
      keep.forEach(function (j) {
        var xj = rec.x[j], yj = rec.y[j], tj = rec.t[j], ok = false;
        for (var dy = -1; dy <= 1 && !ok; dy++) {
          var yy = yj + dy; if (yy < 0 || yy >= rec.height) continue;
          for (var dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            var xx = xj + dx; if (xx < 0 || xx >= SW) continue;
            if (tj - Tl[yy * SW + xx] <= dt) { ok = true; break; }
          }
        }
        if (ok && cnt[yj * SW + xj] < thr) kept.push(j);
        Tl[yj * SW + xj] = tj;
      });
      keep = kept;
    }
    var n = keep.length, X = new Int32Array(n), Y = new Int32Array(n), O = new Uint8Array(n), T = new Float64Array(n);
    for (var q = 0; q < n; q++) {
      var e = keep[q];
      X[q] = rec.x[e] - crop[0]; Y[q] = rec.y[e] - crop[1]; O[q] = rec.p[e] ? 1 : 0; T[q] = rec.t[e];
    }
    return new Events(X, Y, O, T, W, H, t0, t1);
  }

  function even(v) { v = Math.max(2, Math.round(v)); return v + (v % 2); }

  function Renderer(rec, shot, events) {
    this.rec = rec; this.shot = shot;
    var look = this.look = LOOKS[shot.look], spec = SPEC.looks[shot.look];
    var P = this.P = resolve(shot.look, shot.params);
    var rt0 = rec.t[0] || 0, rt1 = rec.t[rec.t.length - 1] || 0;
    var t0 = shot.t0 === null || shot.t0 === undefined ? rt0 : Math.max(rt0, Math.min(shot.t0, rt1));
    var t1 = shot.t1 === null || shot.t1 === undefined ? Math.min(t0 + 1, rt1) : Math.max(t0, Math.min(shot.t1, rt1));
    if (t1 <= t0) t1 = Math.min(rt1, t0 + 1e-3);
    this.t0 = t0; this.t1 = t1;
    this.lookback = look.memory ? look.memory(P) : 0;
    this.ev = events || prepareEvents(rec, shot, t0, t1, this.lookback);
    var n = Math.max(1, Math.round((t1 - t0) * shot.fps / Math.max(shot.speed, 1e-6)));
    this.nFrames = (look.nFrames && look.nFrames(this, P)) || n;
    this.times = new Float64Array(this.nFrames);
    for (var i = 0; i < this.nFrames; i++) this.times[i] = t0 + (i + 1) * (t1 - t0) / this.nFrames;
    var cs = look.canvas ? look.canvas(this.ev.W, this.ev.H, P) : [this.ev.W, this.ev.H];
    this.outW = even(shot.width); this.outH = even(this.outW * cs[1] / cs[0]);
    this.native = spec.native;
    this.drawW = this.native ? cs[0] : this.outW; this.drawH = this.native ? cs[1] : this.outH;
    var pal = SPEC.palettes[shot.palette] || SPEC.palettes["fire-ice"];
    this.colors = { on: hexRgb(pal.on), off: hexRgb(pal.off), bg: hexRgb(pal.bg) };
    this.st = look.prepare ? look.prepare(this.ev, this, P) : {};
    this.stripH = shot.rate_strip ? even(Math.max(20, this.outH / 9)) : 0;
  }
  Renderer.prototype.length = function () { var n = this.nFrames; return this.shot.pingpong && n > 2 ? 2 * n - 2 : n; };
  Renderer.prototype.sourceIndex = function (i) { var n = this.nFrames; return i < n ? i : 2 * n - 2 - i; };
  Renderer.prototype.usesPolarity = function () { return this.look.usesPolarity ? this.look.usesPolarity(this.P) : true; };
  Renderer.prototype.stamp = function (k) {
    var tau = this.times[k];
    return this.look.stamp ? this.look.stamp(k, tau, this.P, this.st) : "t = " + tau.toFixed(3) + " s";
  };
  // composited RGBA bytes at the drawing size (gamma applied); the page scales + decorates it
  Renderer.prototype.draw = function (i) {
    var k = this.sourceIndex(i % Math.max(this.length(), 1)), tau = this.times[k];
    var w = this.drawW, h = this.drawH, N = w * h;
    var layers = this.look.render(this.ev, k, tau, this.P, this.st, this.colors, [w, h]);
    var bg = this.colors.bg, add = (bg[0] + bg[1] + bg[2]) / 3 < 0.5;
    var out = new Float32Array(N * 3);
    for (var p = 0; p < N; p++) { out[p * 3] = bg[0]; out[p * 3 + 1] = bg[1]; out[p * 3 + 2] = bg[2]; }
    layers.forEach(function (L) {
      var a = L.a, c = L.c, per = c.length === N * 3;
      for (var q = 0; q < N; q++) {
        var al = a[q]; if (!(al > 0)) continue;
        for (var ch = 0; ch < 3; ch++) {
          var cv = per ? c[q * 3 + ch] : c[ch], o = q * 3 + ch;
          out[o] = add ? out[o] + al * cv : out[o] * (1 - al) + al * cv;
        }
      }
    });
    var g = this.shot.gamma && Math.abs(this.shot.gamma - 1) > 1e-3 ? 1 / this.shot.gamma : 1;
    var bytes = new Uint8ClampedArray(N * 4);
    for (var m = 0; m < N; m++) {
      for (var c2 = 0; c2 < 3; c2++) {
        var v = clamp01(out[m * 3 + c2]); if (g !== 1) v = Math.pow(v, g);
        bytes[m * 4 + c2] = Math.round(v * 255);
      }
      bytes[m * 4 + 3] = 255;
    }
    return { data: bytes, w: w, h: h, k: k, tau: tau };
  };

  global.GottLUXStudio = {
    SPEC: SPEC, LOOKS: LOOKS, Renderer: Renderer, Events: Events, defaultShot: defaultShot,
    resolve: resolve, speedText: speedText, clampCrop: clampCrop, prepareEvents: prepareEvents,
    hexRgb: hexRgb, lut: lut, estimateF0: estimateF0, estimatePeriod: estimatePeriod
  };
  if (typeof module !== "undefined" && module.exports) module.exports = global.GottLUXStudio;
})(typeof self !== "undefined" ? self : this);
