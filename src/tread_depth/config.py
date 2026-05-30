from __future__ import annotations

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """All knobs for the tread-depth pipeline."""

    # --- the scale assumption ----------------------------------------------
    # We have no fiducial in the scene, so metric scale comes from one stated
    # assumption: a passenger-tyre contact patch is ~tread_width_mm wide, and
    # the four circumferential grooves span ~span_to_contact_ratio of it. Depth
    # in mm is only as good as this assumption; the *relative* groove pattern is
    # not affected by it.
    tread_width_mm: float = 175.0
    span_to_contact_ratio: float = 0.60

    groove_count: int = 4

    # --- frame sampling ----------------------------------------------------
    # Sample across the pan so each groove is seen over a wide azimuth range.
    frame_start: int = 120
    frame_stop: int = 480
    frame_step: int = 6

    # --- groove detection (vertical intensity profile) ---------------------
    profile_band: tuple[float, float] = (0.45, 0.58)  # vertical crop, fraction of H
    profile_blur_px: float = 3.0
    search_band: tuple[float, float] = (0.30, 0.80)  # horizontal crop, fraction of W
    min_peak_distance_px: int = 45
    min_prominence_frac: float = 0.35  # keep grooves >= this fraction of the strongest
    spacing_tolerance: float = 0.40  # accept a 4-tuple if gaps are within +/- this of the median

    # --- wall-asymmetry measurement ----------------------------------------
    wall_half_width_px: int = 55
    min_groove_contrast: float = 12.0  # skip washed-out grooves (CLAHE units)

    # --- pose / azimuth (relative pose from the essential matrix) ----------
    max_features: int = 6000
    match_ratio: float = 0.80
    min_pose_inliers: int = 40
    # the pan is a smooth arc, so yaw vs frame should be smooth; reject PnP
    # poses whose yaw departs from a robust quadratic trend by more than this
    azimuth_outlier_deg: float = 8.0

    # --- groove tracking across the pan ------------------------------------
    # the tyre translates as the camera pans, so we follow each groove from the
    # reference frame outward, matching to the nearest detected valley within
    # this per-frame gate (< half the groove spacing, so tracks can't swap)
    track_gate_px: int = 45

    # --- robust sinusoid fit -----------------------------------------------
    min_observations: int = 6  # need at least this many frames to trust a groove
    fit_loss_scale_px: float = 2.0  # soft-L1 scale, suppresses detection outliers
    min_azimuth_span_deg: float = 6.0  # below this a groove never leaves face-on
    min_resolved_depth_mm: float = 2.0  # below this the wall signal is at the noise floor

    # --- annotated video ---------------------------------------------------
    output_fps: int = 24
