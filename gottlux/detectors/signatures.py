"""
signatures.py — the flutter-signature library.

Different fluttering things live in different frequency bands. A :class:`Signature` captures
what a target *class* is expected to look like in the temporal-frequency domain: the search
band, whether to expect a harmonic comb (periodic-but-non-sinusoidal sources like rotors do;
smooth wingbeats less so), and a default per-detection SNR gate. Detectors are parameterized
by a Signature, so retargeting a detector from drones to insects is a one-line change — and a
``custom`` signature lets you dial in any band you like.

Reference bands (approximate, for orientation — tune to your platform):

================  ===============  ================================================
class             band (Hz)        physical source
================  ===============  ================================================
``drone``         80 – 800         multirotor blade-pass freq (rotor RPM × #blades) + overtones
``insect``        30 – 250         flying-insect wingbeat (flies/bees ~150–250, butterflies ~10)
``mosquito``      300 – 800         high wingbeat tone of small dipterans
``hummingbird``   15 – 120         hovering wingbeat
``bird``          3 – 30           flapping flight of larger birds
``custom``        user-defined     anything with a periodic brightness modulation
================  ===============  ================================================
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Signature:
    """A target class's expected temporal-frequency signature."""
    name: str
    freq_lo: float                 # search band lower edge (Hz)
    freq_hi: float                 # search band upper edge (Hz)
    description: str = ""
    expect_harmonics: bool = False
    default_snr: float = 4.0       # default per-detection peak/noise gate
    typical_hz: tuple = ()         # informative "where it usually sits"

    @property
    def band(self) -> tuple:
        return (self.freq_lo, self.freq_hi)

    def nyquist_fs(self, margin: float = 2.5) -> float:
        """A sampling rate that comfortably resolves the band (≥ margin × freq_hi)."""
        return margin * self.freq_hi


_LIBRARY = {
    "drone": Signature("drone", 80.0, 800.0,
                       "Multirotor rotor blade-pass signature (+ harmonics).",
                       expect_harmonics=True, default_snr=4.0, typical_hz=(120, 400)),
    "insect": Signature("insect", 30.0, 250.0,
                        "General flying-insect wingbeat.",
                        expect_harmonics=False, default_snr=3.0, typical_hz=(60, 220)),
    "mosquito": Signature("mosquito", 300.0, 800.0,
                          "High wingbeat tone of small dipterans (mosquitoes/midges).",
                          expect_harmonics=True, default_snr=3.0, typical_hz=(400, 600)),
    "hummingbird": Signature("hummingbird", 15.0, 120.0,
                             "Hovering hummingbird wingbeat.",
                             expect_harmonics=False, default_snr=3.0, typical_hz=(40, 80)),
    "bird": Signature("bird", 3.0, 30.0,
                      "Flapping flight of larger birds.",
                      expect_harmonics=False, default_snr=2.5, typical_hz=(4, 15)),
    "custom": Signature("custom", 10.0, 1000.0,
                        "User-defined band — dial in any periodic flicker.",
                        expect_harmonics=False, default_snr=4.0),
}


def get_signature(name: str) -> Signature:
    """Look up a built-in signature by name (case-insensitive)."""
    key = (name or "custom").lower()
    if key not in _LIBRARY:
        raise KeyError(f"unknown signature {name!r}; available: {sorted(_LIBRARY)}")
    return _LIBRARY[key]


def list_signatures() -> dict:
    """Mapping of signature name → :class:`Signature`."""
    return dict(_LIBRARY)
