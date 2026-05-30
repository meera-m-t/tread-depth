from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares

from .config import Settings


def pixels_per_mm(groove_span_px: float, settings: Settings) -> float:
    """px/mm from the assumed contact width and groove-span ratio."""
    contact_px = groove_span_px / settings.span_to_contact_ratio
    return contact_px / settings.tread_width_mm


def fit_depth(azimuths: np.ndarray, asymmetry: np.ndarray, px_per_mm: float,
              settings: Settings) -> dict:
    """Robustly fit the asymmetry sinusoid; return depth and fit diagnostics.

    Asymmetry is a smooth function of azimuth, so the depth signal -- a large
    excursion concentrated at the high-slant end of the pan -- is smooth, while
    momentary tracking valley-swaps show up as isolated spikes. We reject points
    that depart from a *local* (azimuth-sorted) median, which removes the spikes
    without discarding the informative excursion, then least-squares fit the
    sinusoid to what remains.
    """
    azimuths = np.asarray(azimuths, float)
    asymmetry = np.asarray(asymmetry, float)

    order = np.argsort(azimuths)
    local = median_filter(asymmetry[order], size=7, mode="nearest")
    resid = asymmetry[order] - local
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
    keep = np.zeros(len(azimuths), bool)
    keep[order[np.abs(resid) < max(8.0, 3.0 * mad)]] = True
    if keep.sum() < settings.min_observations:
        keep[:] = True

    az_in, asym_in = azimuths[keep], asymmetry[keep]
    amp0 = 1.4 * np.std(asym_in)
    best = None
    for phase0 in np.linspace(-1.5, 1.5, 9):
        sol = least_squares(lambda p: p[0] * np.sin(az_in + p[1]) - asym_in,
                            [amp0, phase0])
        if best is None or sol.cost < best.cost:
            best = sol
    amplitude, phase = best.x

    resid_in = amplitude * np.sin(az_in + phase) - asym_in
    rms = float(np.sqrt(np.mean(resid_in ** 2)))
    span_deg = float(np.degrees(az_in.max() - az_in.min()))
    reliable = (int(keep.sum()) >= settings.min_observations
                and span_deg >= settings.min_azimuth_span_deg
                and rms < 0.4 * abs(amplitude) + 2.0
                and abs(amplitude) / px_per_mm >= settings.min_resolved_depth_mm)

    return {
        "depth_mm": abs(amplitude) / px_per_mm,
        "amplitude_px": float(amplitude),
        "baseline_angle_deg": float(np.degrees(phase)),
        "fit_rms_px": rms,
        "azimuth_span_deg": span_deg,
        "n_observations": int(keep.sum()),
        "reliable": bool(reliable),
    }
