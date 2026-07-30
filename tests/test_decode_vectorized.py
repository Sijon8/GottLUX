"""
Byte-identical equivalence of the vectorized decoders against the per-bit reference loop.

``chunk_evt21``'s 32-px column mask and ``chunk_evt3``'s 12/8-bit vector words used to be
expanded with a Python-level per-bit boolean-select loop (the decoder's hot loop);
:func:`gottlux.io.decode._bit_positions` now does the same expansion with a handful of
``np.unpackbits`` passes. The decoder's contract is *bit-exactness* (it was validated
against the ECF-HDF5 oracle), so the vectorization must be invisible: same events, same
**bit-major** order (all bit-0 hits, then bit-1, …) that keeps every stable time-sort
downstream byte-identical. These tests pin that equivalence on random masks, a synthetic
EVT3 stream, and the bundled real EVT2.1 clip — and keep an eye on the speed win.
"""
import os
import time

import numpy as np
import pytest

from gottlux.io import decode

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "examples", "data")
QUAD = os.path.join(DATA, "5inch_quadcopter.raw")
needs_quad = pytest.mark.skipif(not os.path.exists(QUAD),
                                reason="bundled example clip missing: 5inch_quadcopter.raw")


def _bit_positions_ref(mask, nbits):
    """The original per-bit boolean-select loop: for each bit b (ascending), the word
    indices whose mask has bit b set — i.e. bit-major ``(bit, word_index)`` order."""
    m = np.asarray(mask)
    bits, idxs = [], []
    for b in range(nbits):
        sel = np.nonzero((m.astype(np.int64) >> b) & 1)[0]
        if sel.size:
            bits.append(np.full(sel.size, b, np.int32))
            idxs.append(sel.astype(np.intp))
    if not bits:
        return np.zeros(0, np.int32), np.zeros(0, np.intp)
    return np.concatenate(bits), np.concatenate(idxs)


def _synth_evt3_words(n_groups=20_000, seed=1):
    """A well-formed random EVT3 stream: TIME_HIGH/TIME_LOW/ADDR_Y interleaved with single
    ADDR_X events and VECT_BASE_X + VECT_12/VECT_8 vector groups (random masks)."""
    rng = np.random.default_rng(seed)
    words, hi = [], 5
    for _ in range(n_groups):
        if rng.random() < 0.05:
            hi += 1
            words.append(0x8000 | (hi & 0xFFF))                       # TIME_HIGH
        words.append(0x6000 | int(rng.integers(0, 4096)))             # TIME_LOW
        words.append(0x0000 | int(rng.integers(0, 320)))              # ADDR_Y
        r = rng.random()
        if r < 0.3:                                                   # single CD event
            words.append(0x2000 | (int(rng.integers(0, 2)) << 11) | int(rng.integers(0, 320)))
        elif r < 0.7:                                                 # 12-px vector
            words.append(0x3000 | (int(rng.integers(0, 2)) << 11) | int(rng.integers(0, 300)))
            words.append(0x4000 | int(rng.integers(0, 4096)))
        else:                                                         # 8-px vector
            words.append(0x3000 | (int(rng.integers(0, 2)) << 11) | int(rng.integers(0, 300)))
            words.append(0x5000 | int(rng.integers(0, 256)))
    return np.array(words, dtype="<u2")


# ====================================================================================
# The expansion primitive
# ====================================================================================
@pytest.mark.parametrize("dtype,nbits", [(np.uint32, 32), (np.uint16, 12), (np.uint8, 8)])
def test_bit_positions_matches_reference(dtype, nbits):
    rng = np.random.default_rng(0)
    cases = [rng.integers(0, 2**nbits, 100_000).astype(dtype),        # random
             np.zeros(1_000, dtype),                                  # nothing set
             np.full(1_000, (1 << nbits) - 1, dtype),                 # everything set
             np.zeros(0, dtype)]                                      # empty chunk
    for m in cases:
        b_vec, i_vec = decode._bit_positions(m, nbits)
        b_ref, i_ref = _bit_positions_ref(m, nbits)
        assert np.array_equal(b_vec, b_ref)
        assert np.array_equal(i_vec, i_ref)


# ====================================================================================
# Whole-decoder equivalence (the reference loop swapped in via monkeypatch)
# ====================================================================================
@needs_quad
def test_evt21_real_file_byte_identical(monkeypatch):
    d_vec = decode.decode(QUAD)
    monkeypatch.setattr(decode, "_bit_positions", _bit_positions_ref)
    d_ref = decode.decode(QUAD)
    assert d_vec["t0_us"] == d_ref["t0_us"]
    for k in ("x", "y", "p", "t"):
        assert d_vec[k].dtype == d_ref[k].dtype
        assert np.array_equal(d_vec[k], d_ref[k]), f"{k} differs from the reference loop"


def test_evt3_synthetic_stream_byte_identical(monkeypatch):
    w = _synth_evt3_words()
    out_vec = decode.chunk_evt3(w, decode.init_state("evt3"))
    monkeypatch.setattr(decode, "_bit_positions", _bit_positions_ref)
    out_ref = decode.chunk_evt3(w, decode.init_state("evt3"))
    assert len(out_vec[3]) > 50_000                 # the stream really expanded into events
    for vec, ref, name in zip(out_vec, out_ref, ("x", "y", "p", "t")):
        assert np.array_equal(vec, ref), f"{name} differs from the reference loop"


# ====================================================================================
# The speed win (informational print; generous bound so timing noise can't flake it)
# ====================================================================================
@needs_quad
def test_evt21_vectorized_not_slower_than_reference(capsys):
    meta, off = decode.parse_header(QUAD)
    with open(QUAD, "rb") as f:
        f.seek(off)
        raw = f.read()
    w = np.frombuffer(raw[: (len(raw) // 8) * 8], dtype="<u8")

    def timed(fn):
        best = np.inf
        for _ in range(2):
            t0 = time.perf_counter()
            out = fn(w, decode.init_state("evt21"))
            best = min(best, time.perf_counter() - t0)
        return best, out

    t_vec, out_vec = timed(decode.chunk_evt21)
    orig = decode._bit_positions
    decode._bit_positions = _bit_positions_ref
    try:
        t_ref, out_ref = timed(decode.chunk_evt21)
    finally:
        decode._bit_positions = orig
    assert all(np.array_equal(a, b) for a, b in zip(out_vec, out_ref))
    with capsys.disabled():
        print(f"\n[chunk_evt21 {len(w):,} words -> {len(out_vec[3]):,} events] "
              f"vectorized {t_vec*1e3:.0f} ms vs per-bit loop {t_ref*1e3:.0f} ms "
              f"(x{t_ref/max(t_vec, 1e-9):.2f})")
    assert t_vec <= t_ref * 1.5                     # must at least hold the line
