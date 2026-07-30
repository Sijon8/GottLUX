"""
Offscreen tests for the classic viewer tab (gottlux.app.ebsviewer).

Validates that all eleven view modes render through the GottLUX capture path, the Band/Single/
Stack column-mode expression builds the right column LUTs, and the GottLUX panel/export hooks
(set_recording / capture_frame / sensor_size / capture_clock) work. Headless via Qt 'offscreen'.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from gottlux.io.recording import Recording  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _synth(n=20000, dur_s=1.0, w=320, h=320, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, dur_s, n) * 1e6).astype(np.int64)
    return Recording.from_events(rng.integers(0, w, n), rng.integers(0, h, n),
                                 rng.integers(0, 2, n), t, width=w, height=h, name="clip")


def test_all_view_modes_render(app):
    from gottlux.app.ebsviewer import EBSViewer, VIEW_MODES
    v = EBSViewer()
    v.set_recording(_synth())
    assert len(VIEW_MODES) == 11
    for mode in VIEW_MODES:
        v.vmode.setCurrentText(mode)
        rgb = v.capture_frame(0.5)
        assert rgb is not None, f"{mode} produced no frame"
        assert rgb.ndim == 3 and rgb.shape[2] == 3, f"{mode} bad shape {rgb.shape}"


def test_switching_mode_updates_visualization(app):
    """Regression: every mode switch must change what's shown. The rotation-only views used to
    silently fall back to the *identical* Events frame when no telemetry was present, so the
    selector looked frozen on 'Events'. Each mode must now produce a distinct frame."""
    import hashlib

    from gottlux.app.ebsviewer import EBSViewer, ROTATION_ONLY_MODES, VIEW_MODES
    v = EBSViewer()
    v.set_recording(_synth())           # synthetic clip has no azimuth telemetry
    assert v.tel is None

    def _h(mode):
        v.vmode.setCurrentText(mode)
        rgb = v._render_to_rgb(0.5)
        return hashlib.md5(rgb.tobytes()).hexdigest()

    events = _h("Events")
    seen = {}
    for mode in VIEW_MODES:
        h = _h(mode)
        if mode in ROTATION_ONLY_MODES:
            # without a telemetry file these render from a synthesized spin, never the Events frame
            assert h != events, f"{mode} silently fell back to the Events view"
        seen[mode] = h
    # and the eleven modes are not all collapsed onto one image
    assert len(set(seen.values())) >= 8, "view modes are not producing distinct visualizations"


def test_rotation_views_synthesize_spin_without_telemetry(app):
    """The spin-dependent views must render straight from the events when a clip ships no
    telemetry file: an azimuth track is synthesized (estimated spin, North uncalibrated) so the
    panorama / radar / waterfall actually display instead of showing the 'needs telemetry' card."""
    import hashlib

    from gottlux.app.ebsviewer import EBSViewer, ROTATION_ONLY_MODES
    v = EBSViewer()
    v.set_recording(_synth(dur_s=3.0))
    assert v.tel is None

    for mode in sorted(ROTATION_ONLY_MODES):
        v.vmode.setCurrentText(mode)
        rgb = v._render_to_rgb(1.5)
        assert rgb is not None and rgb.ndim == 3
        # telemetry was synthesized (not a real logged CSV) so the view could render
        assert v.tel is not None and v.tel.synthesized is True
        # and it is NOT the 'needs rotation telemetry' placeholder card
        card = v._render_needs_rotation(mode)
        assert hashlib.md5(rgb.tobytes()).hexdigest() != hashlib.md5(card.tobytes()).hexdigest(), \
            f"{mode} fell back to the placeholder instead of synthesizing spin"

    # the manual 'spin' override changes the assumed period (and rebuilds the synthesized track)
    v.spin_sp.setValue(0.5)
    v.vmode.setCurrentText("Panorama")
    v._render_to_rgb(1.5)
    assert v.tel.synthesized is True
    assert v._synth_period_est == pytest.approx(0.5, rel=1e-6)


def test_column_mode_expression(app):
    """Band / Single / Stack each build the expected boolean column LUT."""
    from gottlux.app.ebsviewer import EBSViewer
    v = EBSViewer()
    v.set_recording(_synth())

    v._col_mode_switch("single", 1); v.col_single_sp.setValue(160)
    assert int(v._col_lut.sum()) == 1

    v._col_mode_switch("stack", 2)
    for c in (10, 20, 30):
        v.col_add_sp.setValue(c); v._col_stack_add()
    assert v.col_stack == [10, 20, 30] and int(v._col_lut.sum()) == 3

    v._col_mode_switch("band", 0)
    v.col_from.setValue(50); v.col_to.setValue(120); v._col_spin()
    assert int(v._col_lut.sum()) == 70
    # full width collapses the LUT to None (a no-op filter)
    v.col_from.setValue(0); v.col_to.setValue(v.W); v._col_spin()
    assert v._col_lut is None


def test_capture_clock_adapter(app):
    """The clock adapter exposes the interface GottLUX's Capture/Export dialogs drive."""
    from gottlux.app.ebsviewer import EBSViewer
    v = EBSViewer()
    v.set_recording(_synth(dur_s=2.0))
    c = v.capture_clock()
    c.set_cursor(1.0)
    assert abs(c.cursor - 1.0) < 0.05
    assert c.t1 == pytest.approx(v.dur, rel=1e-3)
    assert not c.has_selection()          # full-range trim → no selection
    c.toggle(); assert v.timer.isActive()
    c.pause(); assert not v.timer.isActive()


def test_set_recording_and_export_hooks(app):
    from gottlux.app.ebsviewer import EBSViewer
    v = EBSViewer()
    v.set_recording(_synth(w=128, h=128))
    assert v.W == 128 and v.H == 128 and v.dur > 0
    w, h = v.sensor_size()
    assert w > 0 and h > 0
    # capture honours an explicit accumulation and a resize
    rgb = v.capture_frame(0.5, dt=0.05, size=(200, 150))
    assert rgb.shape[:2] == (150, 200)
