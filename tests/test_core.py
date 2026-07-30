"""Core-compute tests: accumulation modes, the frequency engine, filters, geometry."""
import numpy as np

from gottlux.core import accumulate as acc
from gottlux.core import filters, frequency as fq, geometry as geo
from gottlux.io.recording import Recording


def _periodic_pixel(freq_hz, duration_s=1.0, n_per_burst=20, seed=0):
    """Events at a single pixel firing in bursts at freq_hz (a pure flicker source)."""
    rng = np.random.default_rng(seed)
    period = 1.0 / freq_hz
    ts = []
    for k in range(int(duration_s / period)):
        c = (k + 0.5) * period
        ts.append(c + (rng.random(n_per_burst) - 0.5) * 0.2 * period)
    t = np.sort(np.concatenate(ts)) * 1e6
    x = np.full(t.size, 10, np.uint16); y = np.full(t.size, 10, np.uint16)
    p = np.ones(t.size, np.uint8)
    return Recording.from_events(x, y, p, t.astype(np.int64), width=64, height=64)


def test_region_spectrum_recovers_frequency():
    """A 137 Hz periodic source must be recovered to within a few Hz."""
    rec = _periodic_pixel(137.0, duration_s=1.0, n_per_burst=25)
    sp = fq.region_spectrum(rec.window().t, fs=2000, fmin=20, fmax=500)
    assert sp.detected
    assert abs(sp.peak_freq - 137.0) < 5.0
    assert sp.snr > 5.0


def test_accumulate_modes_shapes():
    rng = np.random.default_rng(4)
    n = 5000
    rec = Recording.from_events(rng.integers(0, 50, n), rng.integers(0, 50, n),
                                rng.integers(0, 2, n), np.sort(rng.integers(0, 500_000, n)),
                                width=50, height=50)
    win = rec.window()
    for mode in ("count", "polarity", "on", "off", "time_surface", "binary"):
        f = acc.accumulate_frame(win, mode=mode)
        assert f.shape == (50, 50)
        assert np.isfinite(f).all()
    assert acc.accumulate_frame(win, mode="binary").max() <= 1.0


def test_flicker_map_localizes_source():
    """The flicker map must light up the cell containing a periodic source."""
    rec = _periodic_pixel(220.0, duration_s=1.0, n_per_burst=30)
    fm = fq.flicker_map(rec, fmin=50, fmax=500, fs=2000, cell=8, min_events_per_cell=20)
    valid = np.isfinite(fm.dominant_freq)
    assert valid.any()
    iy, ix = np.unravel_index(np.nanargmax(np.where(valid, fm.snr, -1)), fm.snr.shape)
    # source is at pixel (10, 10) -> cell (1, 1) for cell=8
    assert ix == 10 // 8 and iy == 10 // 8
    assert abs(fm.dominant_freq[iy, ix] - 220.0) < 12.0


def test_hot_pixel_filter_removes_stuck_pixel():
    rng = np.random.default_rng(5)
    n = 5000
    x = rng.integers(0, 50, n); y = rng.integers(0, 50, n)
    # add a stuck pixel firing 2000 times at (5,5)
    x = np.concatenate([x, np.full(2000, 5)]); y = np.concatenate([y, np.full(2000, 5)])
    p = np.ones(x.size, np.uint8); t = np.sort(rng.integers(0, 500_000, x.size))
    rec = Recording.from_events(x, y, p, t, width=50, height=50)
    win = rec.window()
    keep = filters.hot_pixel_mask(win, pct=99.9)
    # events at (5,5) should be predominantly dropped
    at_stuck = (np.asarray(win.x) == 5) & (np.asarray(win.y) == 5)
    assert keep[at_stuck].mean() < 0.5


def test_geometry_bearing_elevation_centered():
    # a target at sensor centre reads 0 bearing and 0 elevation
    assert abs(geo.pixel_to_bearing(160, 76.0, 320)) < 1e-6
    assert abs(geo.pixel_to_elevation(160, 76.0, 320, 320)) < 1e-6
    # range is monotonic decreasing in apparent size
    r = geo.estimate_range_m(np.array([10.0, 20.0, 40.0]), 76.0, 0.22, 320)
    assert r[0] > r[1] > r[2]
