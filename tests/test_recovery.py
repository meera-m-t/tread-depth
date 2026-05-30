import numpy as np

from tread_depth.config import Settings
from tread_depth.detect import detect_grooves
from tread_depth.geometry import fit_depth, pixels_per_mm


def _synthetic_asymmetry(depth_mm, px_per_mm, phase, n=40, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    azimuth = np.radians(np.linspace(-15, 30, n))
    clean = depth_mm * px_per_mm * np.sin(azimuth + phase)
    return azimuth, clean + rng.normal(0, noise, n)


def test_recovers_known_depth():
    settings = Settings()
    px_per_mm = 3.9
    for depth in (4.0, 6.0, 8.0):
        for phase in (-0.4, 0.0, 0.6):
            az, asym = _synthetic_asymmetry(depth, px_per_mm, phase)
            out = fit_depth(az, asym, px_per_mm, settings)
            assert abs(out["depth_mm"] - depth) < 0.7, (depth, phase, out["depth_mm"])
            assert out["reliable"]


def test_rejects_glitch_outliers():
    settings = Settings()
    px_per_mm = 3.9
    az, asym = _synthetic_asymmetry(6.0, px_per_mm, 0.2, noise=0.8)
    asym[5] -= 25  # a tracking valley-swap spike
    asym[20] -= 22
    out = fit_depth(az, asym, px_per_mm, settings)
    assert abs(out["depth_mm"] - 6.0) < 1.0
    assert out["n_observations"] < len(asym)  # spikes were dropped


def test_flat_signal_is_low_confidence():
    settings = Settings()
    rng = np.random.default_rng(1)
    az = np.radians(np.linspace(-15, 30, 40))
    asym = rng.normal(0, 1.0, 40)  # no real wall signal
    out = fit_depth(az, asym, 3.9, settings)
    assert not out["reliable"]


def test_detects_four_evenly_spaced_grooves():
    settings = Settings()
    w, h = 1620, 200
    img = np.full((h, w), 180, np.uint8)
    centres = [500, 740, 980, 1220]
    xs = np.arange(w)
    for c in centres:
        img -= (60 * np.exp(-((xs - c) ** 2) / (2 * 12 ** 2))).astype(np.uint8)
    found = detect_grooves(img, settings)
    assert found is not None and len(found) == 4
    assert max(abs(f - c) for f, c in zip(found, centres)) <= 6


def test_pixels_per_mm_uses_assumption():
    settings = Settings(tread_width_mm=175.0, span_to_contact_ratio=0.6)
    # a 400 px groove span -> contact 666 px -> 3.81 px/mm
    assert abs(pixels_per_mm(400, settings) - 3.81) < 0.05
