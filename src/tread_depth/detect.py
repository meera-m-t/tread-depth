from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .config import Settings

_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def intensity_profile(gray: np.ndarray, settings: Settings) -> np.ndarray:
    """Mean column intensity over a horizontal band, lighting-normalised."""
    h = gray.shape[0]
    lo, hi = settings.profile_band
    band = _CLAHE.apply(gray)[int(lo * h):int(hi * h)]
    return gaussian_filter1d(band.mean(axis=0), settings.profile_blur_px)


def detect_grooves(gray: np.ndarray, settings: Settings) -> list[int] | None:
    """Return the x of the N grooves, left to right, or None if not found.

    We over-detect dark valleys, then pick the N-tuple that is most evenly
    spaced -- the circumferential grooves are regular, which cleanly rejects
    siping and shoulder clutter.
    """
    w = gray.shape[1]
    profile = gaussian_filter1d(intensity_profile(gray, settings), 2)
    x0, x1 = int(settings.search_band[0] * w), int(settings.search_band[1] * w)

    inverted = profile.max() - profile
    peaks, props = find_peaks(
        inverted[x0:x1],
        distance=settings.min_peak_distance_px,
        prominence=1.0,
    )
    if len(peaks) < settings.groove_count:
        return None

    xs = peaks + x0
    prom = props["prominences"]
    keep = prom > settings.min_prominence_frac * prom.max()
    xs, prom = xs[keep], prom[keep]
    if len(xs) < settings.groove_count:
        return None

    nominal = np.median(np.diff(np.sort(xs)))
    best, best_key = None, None
    candidates = sorted(zip(xs.tolist(), prom.tolist()))
    for combo in combinations(candidates, settings.groove_count):
        cx = [c[0] for c in combo]
        gaps = np.diff(cx)
        if gaps.min() < (1 - settings.spacing_tolerance) * nominal:
            continue
        if gaps.max() > (1 + settings.spacing_tolerance) * nominal:
            continue
        # most regular spacing wins; break ties toward the strongest grooves
        key = (round(gaps.std() / gaps.mean(), 4), -sum(c[1] for c in combo))
        if best_key is None or key < best_key:
            best_key, best = key, cx
    return best


def detect_valleys(gray: np.ndarray, settings: Settings) -> list[int]:
    """All prominent dark valleys in the search band, left to right.

    Looser than :func:`detect_grooves` (it does not insist on a regular
    N-tuple). Tracking uses it so a groove can still be followed in a frame
    where the strict even-spacing pick would have rejected the layout.
    """
    w = gray.shape[1]
    profile = gaussian_filter1d(intensity_profile(gray, settings), 2)
    x0, x1 = int(settings.search_band[0] * w), int(settings.search_band[1] * w)
    inverted = profile.max() - profile
    peaks, props = find_peaks(
        inverted[x0:x1], distance=settings.min_peak_distance_px, prominence=1.0)
    if len(peaks) == 0:
        return []
    prom = props["prominences"]
    keep = prom > settings.min_prominence_frac * prom.max()
    return sorted((peaks[keep] + x0).tolist())


def wall_asymmetry(profile: np.ndarray, x: int, settings: Settings) -> float | None:
    """Signed asymmetry of the dark valley at column ``x`` (left run - right run).

    Measured at half-depth between the floor minimum and the flanking rib crests.
    Returns None when the groove is too washed out to measure reliably.
    """
    half = settings.wall_half_width_px
    lo, hi = max(0, x - half), min(len(profile), x + half)
    seg = profile[lo:hi].astype(float)
    if seg.size < 5:
        return None

    floor_i = int(np.argmin(seg))
    left_crest = seg[:floor_i + 1].max() if floor_i > 0 else seg[floor_i] + 1
    right_crest = seg[floor_i:].max()
    rib = 0.5 * (left_crest + right_crest)
    if rib - seg[floor_i] < settings.min_groove_contrast:
        return None

    threshold = 0.5 * (rib + seg[floor_i])
    left = floor_i
    while left > 0 and seg[left] < threshold:
        left -= 1
    right = floor_i
    while right < len(seg) - 1 and seg[right] < threshold:
        right += 1
    return (floor_i - left) - (right - floor_i)
