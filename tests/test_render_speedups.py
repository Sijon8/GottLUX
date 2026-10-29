"""The render speed-ups are exact: each fast path against the straightforward code it
replaced (kept here as the reference), on inputs that exercise its edges."""
import numpy as np
import pytest


# ------------------------------------------------------------------ Studio composite
def _composite_ref(layers, bg, shape):
    h, w = shape
    out = np.empty((h, w, 3), np.float32)
    out[:] = bg
    additive = float(np.mean(bg)) < 0.5
    for a, col in layers:
        a = np.asarray(a, np.float32)[..., None]
        col = np.asarray(col, np.float32)
        if col.ndim == 1:
            col = col[None, None, :]
        if additive:
            out += a * col
        else:
            out = out * (1.0 - a) + a * col
    return np.clip(out, 0.0, 1.0)


@pytest.mark.parametrize("bg", [(0.02, 0.03, 0.04), (0.97, 0.96, 0.95)])
def test_composite_matches_the_reference(bg):
    from gottlux.studio.render import composite
    rng = np.random.default_rng(1)
    h, w = 37, 53
    layers = [(rng.random((h, w)).astype(np.float32), np.array([1.0, 0.48, 0.24], np.float32)),
              (rng.random((h, w)) * 2.0, np.array([0.25, 0.59, 1.0], np.float32)),
              (rng.random((h, w)).astype(np.float32), rng.random((h, w, 3)).astype(np.float32))]
    bg = np.asarray(bg, np.float32)
    assert np.array_equal(composite(layers, bg, (h, w)), _composite_ref(layers, bg, (h, w)))


# ------------------------------------------------------------------ stroboscope folding
def test_strobe_window_search_matches_weighing_every_event():
    from gottlux.studio.looks import Events, get_look
    rng = np.random.default_rng(2)
    n, W, H = 20000, 40, 30
    t = np.sort(rng.uniform(0, 1.0, n))
    ev = Events(rng.integers(0, W, n), rng.integers(0, H, n), rng.integers(0, 2, n) == 1,
                (t * 1e6).astype(np.int64), W, H, 0.0, 1.0)
    look = get_look("strobe")
    for width in (0.01, 0.08, 0.5):
        P = look.resolve({"f0_hz": 37.0, "bins": 12, "width": width})
        sl = ev.span(ev.t0, ev.t1)
        phase = (ev.t[sl] * 37.0) % 1.0
        order = np.argsort(phase, kind="stable")
        st = {"sl": sl, "phase": phase, "order": order, "phase_sorted": phase[order]}
        for phi in (0.0, 0.004, 0.5, 0.97, 0.999, 1.0):     # windows that wrap both ways
            d = np.abs(((phase - phi + 0.5) % 1.0) - 0.5)
            half = 0.5 * width
            m = d < half
            wgt = (1.0 - d / half) * m
            on = ev.on[sl]
            ref_on = ev.splat(sl, wgt, on & m)
            ref_off = ev.splat(sl, wgt, ~on & m)
            got_on, got_off = look._raw(ev, phi, P, st)
            assert np.array_equal(got_on, ref_on) and np.array_equal(got_off, ref_off)


# ------------------------------------------------------------------ EBS viewer counting
def test_bincount_counts_match_add_at_and_histogram2d():
    pytest.importorskip("PySide6")
    from gottlux.app.ebsviewer import _count, _pano_hist
    rng = np.random.default_rng(3)
    H, W = 48, 64
    ys, xs = rng.integers(0, H, 5000), rng.integers(0, W, 5000)
    ref = np.zeros((H, W), np.float32); np.add.at(ref, (ys, xs), 1.0)
    assert np.array_equal(_count((H, W), ys, xs), ref)
    zs = rng.integers(0, 7, 5000)
    ref3 = np.zeros((H, W, 7), np.float32); np.add.at(ref3, (ys, xs, zs), 1.0)
    assert np.array_equal(_count((H, W, 7), ys, xs, zs), ref3)
    assert np.array_equal(_count((H, W), ys[:0], xs[:0]), np.zeros((H, W), np.float32))
    az = rng.uniform(0, 360, 5000)
    az[:6] = [0.0, 0.5, 359.5, 359.9999999, 360.0, 180.25]
    y = rng.integers(0, H, 5000).astype(float)
    ref_h = np.histogram2d(y, az, bins=[H, 720], range=[[0, H], [0, 360]])[0]
    assert np.array_equal(_pano_hist(y, az, H, 720), ref_h)


# ------------------------------------------------------------------ canvas text
def _draw_texts_ref(frame, texts, t_canvas):
    from PIL import Image, ImageDraw
    from gottlux.core import canvas as cv
    active = [t for t in texts if cv.text_active(t, t_canvas)]
    if not active:
        return frame
    H, W = frame.shape[:2]
    im = Image.fromarray(np.ascontiguousarray(frame, np.uint8))
    for txt in active:
        font = cv._load_font(txt.font_size_px)
        ink = tuple(int(v) for v in txt.color)
        if txt.kind == "slide":
            im = Image.new("RGB", (W, H), tuple(int(v) for v in (txt.bg_color or (0, 0, 0))))
            d = ImageDraw.Draw(im)
            box = d.multiline_textbbox((0, 0), txt.text, font=font, align="center")
            d.multiline_text(((W - box[2] - box[0]) / 2, (H - box[3] - box[1]) / 2),
                             txt.text, fill=ink, font=font, align="center")
            continue
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        box = d.multiline_textbbox((0, 0), txt.text, font=font, align="center")
        tw, th = box[2] - box[0], box[3] - box[1]
        m = int(txt.margin_px)
        x = {"w": m, "e": W - tw - m}.get(txt.anchor, (W - tw) // 2)
        y = {"n": m, "s": H - th - m}.get(txt.anchor, (H - th) // 2)
        if txt.bg_color is not None:
            pad = max(4, int(txt.font_size_px) // 4)
            d.rectangle((x - pad, y - pad, x + tw + pad, y + th + pad),
                        fill=tuple(int(v) for v in txt.bg_color) + (cv._TEXT_BAR_ALPHA,))
        d.multiline_text((x - box[0], y - box[1]), txt.text, fill=ink + (255,),
                         font=font, align="center")
        im = Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")
    return np.asarray(im, np.uint8)


@pytest.mark.parametrize("size", [(320, 240), (333, 221)])
def test_cached_text_items_draw_the_same_pixels(size):
    from gottlux.core import canvas as cv
    W, H = size
    base = (np.random.default_rng(4).random((H, W, 3)) * 255).astype(np.uint8)
    texts = [cv.CanvasText(text="GottLux canvas", anchor="n"),
             cv.CanvasText(text="two\nlines", anchor="e", bg_color=None, font_size_px=20),
             cv.CanvasText(text="Ågp|yÿ — 0.25×", anchor="s", color=(255, 200, 0)),
             cv.CanvasText(text="centre", anchor="center", font_size_px=40),
             cv.CanvasText(text="SLIDE", kind="slide", span=(0.9, 1.1), bg_color=(30, 0, 60)),
             cv.CanvasText(text="after the slide", anchor="w", margin_px=3)]
    for t in (0.5, 1.0):                       # without and with the slide
        assert np.array_equal(cv.draw_texts(base, texts, t), _draw_texts_ref(base, texts, t))
