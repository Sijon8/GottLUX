"""
config.py — the single source of truth for a run's parameters.

A :class:`Config` is a plain dataclass: every field is documented, has a sensible
default, and is JSON-serializable so the *exact* configuration of a run is archived in
its manifest (provenance). The GUI, the CLI, and the headless pipeline all construct and
pass around the same object, so there is one place — here — to learn what every knob does.

Sections
--------
* run / mode          — what to do and to which capture
* sensor / optics      — geometry and the pinhole model used for ranging
* accumulation         — how events become frames (the one temporal-window control)
* background           — static-clutter suppression (rotation vs staring)
* filters              — denoise pre-filters (hot-pixel, refractory, rotation-phase)
* detection            — blob isolation thresholds
* flutter detector     — the tunable flicker/frequency detector (drones, insects)
* visualization        — figure/video parameters
* provenance           — where results go
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Optional, Tuple

from gottlux import sensors


# --- Per-camera FOV overrides for a rotating dual-EBS payload rig --------------
# Horizontal FOV (degrees — it spans the sensor width, which is what the bearing/range math
# consumes) for a *named camera whose optic differs from the active sensor profile*. cam0 is the
# default GenX320 + 1.8 mm rig, so it is NOT listed here — it resolves to that profile's 58°
# horizontal FOV (the 76° often quoted for this rig is the *diagonal*; see
# :data:`gottlux.sensors.GENX320`). cam1 is the narrower second optic, kept as an explicit
# override. Any other rig: pick a profile (``sensor=``/``--sensor``), pass ``--fov_deg``, or let
# it be inferred from the recording. An unlisted camera id falls through to the profile FOV.
CAMERA_FOV_DEG = {"cam1": 50.0}


@dataclass
class Config:
    # ============================ run / mode ============================
    mode: str = "auto"
    """Capture geometry: ``"rotation"`` (panning payload, de-rotate to a world frame),
    ``"staring"`` (fixed sensor), or ``"auto"`` (rotation iff telemetry is present)."""

    camera: str = "cam0"
    """Logical camera id (which optic on a multi-camera payload); selects a per-camera FOV
    override from :data:`CAMERA_FOV_DEG`. Distinct from :attr:`sensor`, which names the
    hardware *profile*; the camera override (when recognized) takes precedence over it."""

    label: str = ""
    """Optional human label woven into output file names (e.g. ``"cam0_wide"``)."""

    # ============================ sensor / optics ============================
    sensor: str = sensors.DEFAULT_PROFILE
    """Hardware profile id naming the capture rig (sensor + optics) — the *variable* to change
    for a clip shot on different hardware. Resolves through :mod:`gottlux.sensors`
    (``"genx320"`` default, ``"imx636"``, or any rig you :func:`~gottlux.sensors.register`).
    Supplies the default resolution, pixel pitch, focal length and fields of view; the explicit
    overrides below win over it per field. List the built-ins with ``gottlux --list_sensors``."""

    sensor_w: Optional[int] = None
    """Sensor width in pixels. ``None`` → from the :attr:`sensor` profile, else inferred from
    the recording header/data."""

    sensor_h: Optional[int] = None
    """Sensor height in pixels. ``None`` → from the :attr:`sensor` profile, else inferred."""

    fov_deg: Optional[float] = None
    """Horizontal field of view (degrees). ``None`` → :data:`CAMERA_FOV_DEG`/name, else the
    :attr:`sensor` profile's horizontal FOV. Set it to override the optics for one clip."""

    target_size_m: float = 0.22
    """Known physical size (m) of the target's largest dimension, used by the pinhole
    range model (0.22 m ≈ a 5-inch quad). Set to 0 to disable absolute ranging and use
    the unitless relative-distance proxy only."""

    az_sign: float = -1.0
    """Sign of the intra-FOV (x → bearing) term; +1 or -1 depending on mount handedness."""

    rotor_blades: int = 2
    """Number of blades per rotor, used only to convert the measured **blade-pass** frequency
    into a rotor rate / RPM (``rotor_hz = blade_hz / rotor_blades``). 2 for a typical 5-inch
    racing/quad prop; 3 for many cinematic/photography props. Pure reporting — does not affect
    detection. See :mod:`gottlux.rotation.rotor_scan`."""

    prop_diameter_m: float = 0.127
    """Propeller diameter (m) of the expected multirotor (0.127 m = 5 inch), used only to report
    the rotor **tip speed** (``π·D·rotor_hz``) and a Mach sanity number for the rotor-ladder
    propeller-signature readout. Pure reporting; ranging uses :attr:`target_size_m`."""

    # ============================ accumulation ============================
    accum_dt: float = 0.02
    """The one temporal-window control (seconds): how long an interval of events is
    integrated into a single frame / time-surface. Smaller = sharper motion, noisier."""

    accum_mode: str = "count"
    """How events map to a frame pixel: ``"count"`` (events per pixel), ``"polarity"``
    (ON−OFF), ``"time_surface"`` (exponential most-recent-event time), ``"binary"``."""

    # ============================ background suppression ============================
    suppress_background: bool = True
    """Enable static-clutter suppression before detection."""

    mask_rotations: int = 1
    """ROTATION mode: build the frozen background reference from the first N revolutions
    (freezing avoids the cumulative-erosion failure mode that masks late-arriving targets)."""

    mask_levels: int = 4
    """`masksweep` analysis: compare N = 0..mask_levels successive masks."""

    bg_window_s: float = 1.0
    """STARING mode: seconds of stream used to learn the persistent-pixel background."""

    bg_method: str = "median"
    """STARING background estimator: ``"median"`` or ``"first_window"``."""

    n_phase: int = 360
    """Phase bins per revolution for the rotation background reference."""

    # ============================ denoise filters ============================
    hot_pixel_pct: float = 99.95
    """Pixels above this event-count percentile are treated as hot/stuck and removed."""

    refractory_us: float = 0.0
    """Per-pixel refractory period (µs): drop events arriving < this after the previous
    event at the same pixel. 0 disables. Suppresses high-rate pixel chatter."""

    rotation_phase_filter: bool = False
    """ROTATION mode denoise: keep only events whose per-pixel timing deviates from the
    rotation-locked phase (static scene points recur at the same phase every revolution;
    a moving target does not). See :mod:`gottlux.core.filters`."""

    rpf_r_min: float = 0.5           # min circular concentration to call a pixel "locked"
    rpf_recur_frac: float = 0.5      # min fraction of revolutions a locked pixel must recur
    rpf_min_events: int = 8          # min events for a pixel to be judged at all
    rpf_phase_tol: float = 0.15      # phase deviation (fraction of a rev) that keeps an event

    # ============================ detection (blob isolation) ============================
    min_pixels: int = 60
    """Minimum connected-component size (px) to accept as a detection."""

    cluster_dilation: int = 3
    """Morphological dilation radius applied before connected-components."""

    cluster_erode: int = 1
    """Morphological erosion radius applied before connected-components."""

    # ============================ flutter / flicker detector ============================
    detector: Optional[str] = None
    """Name of a registered detector to run (e.g. ``"drone"``, ``"insect"``), or ``None``.
    Detectors live in :mod:`gottlux.detectors`; list them with ``--list_detectors``."""

    freq_lo: float = 80.0
    """Lower bound (Hz) of the flutter pass-band the detector verifies against."""

    freq_hi: float = 800.0
    """Upper bound (Hz) of the flutter pass-band (≈ rotor range; insects use 10–120 Hz)."""

    fft_fs: float = 2000.0
    """Sampling rate (Hz) the event-count signal is binned to before the FFT.
    Must exceed 2·freq_hi (Nyquist) with comfortable margin."""

    fft_window_s: float = 0.30
    """Trailing window (seconds) of events fed to each per-region FFT."""

    snr_thresh: float = 4.0
    """Minimum spectral peak-to-noise ratio inside the band to accept a flutter signature."""

    spectrum_method: str = "fft"
    """Region/global spectrum transform: ``"fft"`` (binned) or ``"nufft"`` (non-uniform FFT,
    evaluated straight from event times — no sample-rate Nyquist ceiling)."""

    spectrum_normalize: str = "none"
    """Spectral whitening to emphasize peaking: ``"none"``, ``"median"`` (divide by a
    sliding-median floor) or ``"zscore"`` (sigmas above local noise)."""

    # ============================ lab exports (CLI parity with the GUI) ============================
    export_cube: bool = False
    """Also voxelize the (windowed) stream into a space-time event cube ``V[y, x, t]`` and save
    it (NPZ + HDF5 + JSON). The GUI's Export ▾ → event cube, headless."""

    cube_nt: int = 64
    """Number of time slices along the exported event cube's depth axis."""

    cube_bin: int = 1
    """Spatial pixel binning for the exported event cube (1 = full resolution)."""

    make_report: bool = False
    """Also write a first-principles **detection report** (Markdown + JSON) for the detector
    run — the method, every parameter's meaning/value, assumptions, results and interpretation.
    Runs the detector (defaulting to ``drone``) if one was not otherwise requested."""

    # ============================ results metrics (KPIs) ============================
    make_performance: bool = False
    """Also compute the operator **results metrics** (KPIs) and save a paper-ready bundle:
    *tracking range*, *prop-frequency-resolution range*, and *time-to-contact*. Each metric is
    computed and saved independently (a weak one never soils the others). See
    :mod:`gottlux.run.performance_report`."""

    approach_speed_mps: float = 15.0
    """Nominal closing speed (m/s) for the **time-to-contact** capability number
    (warning time = detection range / approach speed). The measured TTC, when a track is
    actually approaching, is computed independently from the range trend."""

    # ============================ visualization / export ============================
    fig_dpi: int = 300
    """Raster DPI for exported journal figures."""

    fig_format: str = "png"
    """Default raster figure format (``png``/``tiff``); vector ``pdf`` is always also saved."""

    colormap: str = "inferno"
    """Default perceptually-uniform colormap for event/frequency imagery."""

    video_fps: int = 25
    """Frame rate for exported analysis videos."""

    make_video: bool = False
    """Also render analysis videos (slower) in headless runs."""

    # ============================ region / time of interest ============================
    roi: Optional[Tuple[int, int, int, int]] = None
    """Restrict analysis to a sensor sub-rectangle ``(x0, y0, x1, y1)`` in pixels."""

    t_start: Optional[float] = None
    """Restrict analysis to events at or after this time (seconds). ``None`` → start."""

    t_stop: Optional[float] = None
    """Restrict analysis to events before this time (seconds). ``None`` → end."""

    # ============================ rotation / fusion / tracking ============================
    use_ref_mask: bool = True
    """ROTATION mode: suppress static clutter with the frozen N-rotation reference mask."""

    assume_spin_hz: Optional[float] = None
    """For a **rotating clip that logged no azimuth telemetry**: synthesize telemetry from a
    steady spin so the rotation analyses (radar / rotor_ladder) can de-rotate. ``None`` = off;
    ``0`` = estimate the period from the event-rate periodicity
    (:func:`gottlux.io.telemetry.estimate_spin_period_s`); a positive value = that spin rate (Hz).
    Recovered bearings are then rotation-phase-relative (absolute North uncalibrated); the period,
    recurrence and relative-motion offset are faithful when the rate is accurate."""

    az_offset_deg: float = 0.0
    """Dual-EBS co-registration constant (deg) subtracted from cam_b bearings during fusion."""

    fuse_gate_deg: float = 15.0
    """Bearing gate (deg) for accepting the wider camera's detections during dual-EBS fusion."""

    track_known_range_m: Optional[float] = None
    """Optional known target range (m) for a flight, used to calibrate the relative-distance
    proxy to metres (the ``gottlux-calibrate`` path)."""

    tracker: Optional[str] = None
    """Comma-separated names of ported EBS trackers to run (e.g. ``"kalman,cmax"``), or
    ``None``. Trackers are also exposed as detectors in the unified registry (§4)."""

    # ---- rotor-ladder 360° scan (gottlux.rotation.rotor_scan / --analyses rotor_ladder) ----
    ladder_f_lo: float = 80.0
    """Lower bound (Hz) of the rotor blade-pass band the 360° survey searches. Kept separate from
    the flutter :attr:`freq_lo` because a multirotor's blade-pass can run well above the flutter
    band's ceiling."""

    ladder_f_hi: float = 1500.0
    """Upper bound (Hz) of the rotor blade-pass band the survey searches. Default 1500 (not 800)
    because fast 5-inch racing props reach ~1 kHz blade-pass; a ceiling below the true tone makes
    the comb pin to the band edge. Raise further for very high-RPM rotors."""

    ladder_bin_deg: float = 3.0
    """World-azimuth bin width (deg) for the 360° rotor-ladder scan. Each (revolution, bin)
    cell that has enough events is tested for the stair-step comb. Smaller = finer bearing
    resolution but fewer events per cell."""

    ladder_min_events: int = 120
    """Minimum events in a scan cell (or analysis box) before the rotor-ladder comb is judged."""

    ladder_blade_tol: float = 0.25
    """Fractional tolerance on the blade-pass frequency for a scan cell to count as the *same*
    rotor signature as the template box (``|f − f_template| ≤ tol·f_template``)."""

    ladder_track_gate_deg: float = 6.0
    """Bearing gate (deg) for linking rotor-ladder detections across revolutions into one track
    (the per-revolution azimuth offset of a linked track is the target's relative motion)."""

    # ============================ rotation viz / video ====================================
    viz_accum_dt: float = 0.02
    """Accumulation window (s) for the rotation **video** renderers (rate-surface, validation,
    tracker overlay). Distinct from :attr:`accum_dt` so videos can integrate differently."""

    rs_grid: int = 80
    """Rate-surface video: spatial grid resolution (cells per axis)."""
    rs_fps: int = 25
    """Rate-surface / radar-sweep / panorama video frame rate (frames/s)."""
    rs_tail_sec: float = 0.25
    """Rate-surface video: trailing fade window (s)."""
    rs_max_points: int = 9000
    """Rate-surface video: max scatter points per frame (caps render cost)."""

    vr_fps: int = 25
    """Validation / tracker-overlay video frame rate (frames/s)."""
    vr_tail_sec: float = 0.4
    """Validation video: trailing fade window (s)."""

    rs_t0: Optional[float] = None
    """Optional start time (s) for the rotation video renderers (``None`` → recording start)."""
    rs_t1: Optional[float] = None
    """Optional stop time (s) for the rotation video renderers (``None`` → recording end)."""

    panorama_video: bool = False
    """Also render the sweeping panorama video in a headless rotation run."""
    pv_fps: int = 25
    """Panorama video frame rate (frames/s)."""
    pv_frame_dt: float = 0.03
    """Panorama video: per-frame accumulation step (s)."""

    # ============================ run targets / provenance ============================
    analyses: Tuple[str, ...] = ("overview", "spectral", "panorama")
    """Which analysis *groups* the headless pipeline runs (each emits several figures).
    See :mod:`gottlux.run.pipeline`."""

    plots: Optional[Tuple[str, ...]] = None
    """If set, output ONLY these specific named figures (e.g. ``("flicker_map", "tracks")``)
    instead of the analysis groups. List available names with ``--list_plots``."""

    output_root: Optional[str] = None
    """Where run folders are written (``None`` → next to the capture)."""

    cache_dir: Optional[str] = None
    """Override the decode cache location (``None`` → auto, relocating off deep paths)."""

    open_when_done: bool = True
    """Open the run's output folder in the file browser when a headless run finishes."""

    # ---------------------------------------------------------------- legacy-name aliases
    # The ported EBS modules and old EBS manifests use a handful of different field names.
    # These read/write properties keep that code working against the one superset Config.
    @property
    def target_diag_m(self) -> float:                 # EBS name for target_size_m
        return self.target_size_m

    @target_diag_m.setter
    def target_diag_m(self, v):
        self.target_size_m = v

    @property
    def rot_phase_filter(self) -> bool:               # EBS name for rotation_phase_filter
        return self.rotation_phase_filter

    @rot_phase_filter.setter
    def rot_phase_filter(self, v):
        self.rotation_phase_filter = v

    @property
    def rs_accum_dt(self) -> float:                   # rate-surface uses the one viz window
        return self.viz_accum_dt

    @property
    def vr_accum_dt(self) -> float:                   # validation uses the one viz window
        return self.viz_accum_dt

    #: Legacy manifest keys → current field names (consumed by :meth:`from_dict`).
    _ALIASES = {
        "target_diag_m": "target_size_m",
        "rot_phase_filter": "rotation_phase_filter",
        "viz_accum_dt": "viz_accum_dt",
    }

    # ---------------------------------------------------------------- optics resolution
    def active_profile(self) -> "sensors.SensorProfile":
        """The selected hardware profile (:attr:`sensor`), resolved through the registry.

        Unknown/empty ids fall back to the default rig, so an old manifest naming a profile
        we no longer ship still loads. This is the *base* hardware; per-clip overrides
        (:attr:`fov_deg`, :attr:`sensor_w`/:attr:`sensor_h`) are applied by the resolvers below.
        """
        return sensors.get(self.sensor)

    def resolved_fov(self) -> Optional[float]:
        """The effective **horizontal** FOV (deg): explicit :attr:`fov_deg`, else a recognized
        per-camera :data:`CAMERA_FOV_DEG` override, else the :attr:`sensor` profile's horizontal
        FOV (``None`` only if the profile has none set)."""
        if self.fov_deg is not None:
            return self.fov_deg
        if self.camera in CAMERA_FOV_DEG:
            return CAMERA_FOV_DEG[self.camera]
        return self.active_profile().fov_horizontal_deg or None

    def resolved_sensor_wh(self) -> Tuple[Optional[int], Optional[int]]:
        """Configured sensor (width, height) in px: explicit overrides, else the profile's.

        Either element may be ``None`` only if both the override and the profile leave it unset
        (then a caller infers it from the decoded recording).
        """
        prof = self.active_profile()
        w = self.sensor_w if self.sensor_w is not None else prof.width_px
        h = self.sensor_h if self.sensor_h is not None else prof.height_px
        return w, h

    def resolved_pixel_pitch_um(self) -> Optional[float]:
        """The active rig's pixel pitch (µm), from the :attr:`sensor` profile."""
        return self.active_profile().pixel_pitch_um

    def optics(self) -> dict:
        """The fully-resolved optics for this run (profile defaults + per-clip overrides).

        A flat dict — ``sensor``, ``width_px``, ``height_px``, ``pixel_pitch_um``,
        ``focal_length_mm``, ``fov_horizontal_deg`` — that the photogrammetry/ranging paths and
        the run manifest can consume without re-deriving the precedence rules.
        """
        prof = self.active_profile()
        w, h = self.resolved_sensor_wh()
        return {
            "sensor": prof.key,
            "sensor_name": prof.name,
            "width_px": w,
            "height_px": h,
            "pixel_pitch_um": prof.pixel_pitch_um,
            "focal_length_mm": prof.focal_length_mm,
            "fov_horizontal_deg": self.resolved_fov(),
            "fov_diagonal_deg": prof.fov_diagonal_deg or None,
            "f_number": prof.f_number,
        }

    # ---------------------------------------------------------------- helpers

    def to_dict(self) -> dict:
        """JSON-serializable snapshot (used in the run manifest)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Reconstruct from a dict, mapping legacy EBS keys and ignoring unknown ones
        (forward/backward compatible — both old EBS and the staring substrate manifests deserialize)."""
        known = {f.name for f in fields(cls)}
        out = {}
        for k, v in d.items():
            key = cls._ALIASES.get(k, k)
            if key in known:
                out[key] = v
        return cls(**out)
