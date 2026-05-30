from __future__ import annotations

import cv2
import numpy as np

from .config import Settings
from .frames import FrameStore


def _yaw(rotation: np.ndarray) -> float:
    """Yaw (rotation about the vertical) from a rotation matrix."""
    return float(np.arctan2(rotation[0, 2], rotation[2, 2]))


def _build_tracks(frames: FrameStore, reference: int, sift, matcher,
                  settings: Settings):
    ref_kp, ref_desc = sift.detectAndCompute(frames.gray[reference], None)
    ref_xy = np.float32([k.pt for k in ref_kp])
    # tracks[ref_feature_index] = {frame_id: (x, y)}
    tracks: dict[int, dict[int, tuple[float, float]]] = {
        i: {reference: tuple(ref_xy[i])} for i in range(len(ref_kp))}

    for fid in frames.frame_ids:
        if fid == reference:
            continue
        kp, desc = sift.detectAndCompute(frames.gray[fid], None)
        if desc is None or len(kp) < 8:
            continue
        pairs = matcher.knnMatch(ref_desc, desc, k=2)
        good = [(m.queryIdx, m.trainIdx) for m, n in pairs
                if m.distance < settings.match_ratio * n.distance]
        if len(good) < 12:
            continue
        idx = np.array(good)
        pa, pb = ref_xy[idx[:, 0]], np.float32([kp[j].pt for j in idx[:, 1]])
        F, inl = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.999)
        inl = inl.ravel().astype(bool) if inl is not None else np.ones(len(idx), bool)
        for (a, b), keep in zip(idx, inl):
            if keep:
                tracks[a][fid] = kp[b].pt
    return {p: obs for p, obs in tracks.items() if len(obs) >= 3}


def _reject_outliers(azimuths: dict[int, float], settings: Settings) -> dict[int, float]:
    """Drop frames whose yaw departs from a robust quadratic trend.

    A handful of PnP solves return grossly wrong rotations. The pan is a smooth
    arc, so yaw is a smooth function of frame number; gross outliers stand out
    against an iteratively-reweighted quadratic fit and are removed.
    """
    if len(azimuths) < 6:
        return azimuths
    items = sorted(azimuths.items())
    fids = np.array([f for f, _ in items], float)
    yaw = np.array([a for _, a in items], float)

    coef = np.polyfit(fids, yaw, 2)
    for _ in range(3):
        resid = yaw - np.polyval(coef, fids)
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
        keep = np.abs(resid - np.median(resid)) < 2.5 * mad
        if keep.sum() < 4:
            break
        coef = np.polyfit(fids[keep], yaw[keep], 2)

    resid = yaw - np.polyval(coef, fids)
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
    gate = max(np.radians(settings.azimuth_outlier_deg), 2.5 * mad)
    return {int(f): float(y) for f, y in items
            if abs((y - np.polyval(coef, f)) - np.median(resid)) < gate}


def estimate_azimuths(frames: FrameStore, reference: int,
                      settings: Settings) -> dict[int, float]:
    """Map each posed frame id to its azimuth (radians) vs the reference."""
    azimuths = _estimate_raw(frames, reference, settings)
    azimuths = _reject_outliers(azimuths, settings)
    azimuths[reference] = 0.0  # keep the reference even if it was trimmed
    return azimuths


def _estimate_raw(frames: FrameStore, reference: int,
                  settings: Settings) -> dict[int, float]:
    K = frames.intrinsics()
    sift = cv2.SIFT_create(settings.max_features)
    matcher = cv2.BFMatcher()

    tracks = _build_tracks(frames, reference, sift, matcher, settings)
    if not tracks:
        return {reference: 0.0}

    # order frames by proximity to the reference; seed with the nearest one
    order = sorted((f for f in frames.frame_ids if f != reference),
                   key=lambda f: abs(f - reference))
    seed = next((f for f in order if sum(f in t for t in tracks.values()) >= 8), None)
    if seed is None:
        return {reference: 0.0}

    shared = [p for p in tracks if seed in tracks[p]]
    pa = np.float32([tracks[p][reference] for p in shared])
    pb = np.float32([tracks[p][seed] for p in shared])
    E, _ = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, 1.0)
    if E is None or E.shape != (3, 3):
        return {reference: 0.0}
    _, R_seed, t_seed, mask = cv2.recoverPose(E, pa, pb, K)

    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R_seed, t_seed])
    X = cv2.triangulatePoints(P0, P1, pa.T, pb.T)
    X = (X[:3] / X[3]).T
    points3d = {shared[i]: X[i] for i in range(len(shared))
                if mask[i] and 0 < X[i, 2] < 60}

    azimuths = {reference: 0.0, seed: _yaw(R_seed)}
    for fid in order:
        if fid == seed:
            continue
        have = [p for p in tracks if fid in tracks[p] and p in points3d]
        if len(have) < 8:
            continue
        obj = np.float32([points3d[p] for p in have])
        img = np.float32([tracks[p][fid] for p in have])
        ok, rvec, _, _ = cv2.solvePnPRansac(obj, img, K, None,
                                            reprojectionError=3.0)
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            azimuths[fid] = _yaw(R)
    return azimuths
