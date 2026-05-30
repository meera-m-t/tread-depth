from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import detect, geometry
from .config import Settings
from .frames import FrameStore
from .models import GrooveDepth, TreadResult
from .pose import estimate_azimuths


@dataclass
class Observation:
    frame_id: int
    azimuth: float
    asymmetry: float


@dataclass
class GrooveTrack:
    index: int
    positions: dict[int, int] = field(default_factory=dict)  # frame_id -> x
    observations: list[Observation] = field(default_factory=list)

    @property
    def image_x(self) -> int:
        return int(np.median(list(self.positions.values())))


def _pick_reference(frames: FrameStore, settings: Settings) -> tuple[int, float]:
    """Most face-on frame = widest groove span; also returns that span in px."""
    spans: dict[int, float] = {}
    for fid in frames.frame_ids:
        grooves = detect.detect_grooves(frames.gray[fid], settings)
        if grooves:
            spans[fid] = grooves[-1] - grooves[0]
    if not spans:
        raise RuntimeError("could not detect grooves in any frame")
    reference = max(spans, key=spans.get)
    return reference, spans[reference]


def _track_grooves(frames: FrameStore, azimuths: dict[int, float], reference: int,
                   settings: Settings) -> list[GrooveTrack]:
    """Follow each groove outward from the reference frame in both directions.

    The tyre translates across the pan, so rather than rejecting frames whose
    layout shifted, we predict each groove forward and match it to the nearest
    detected valley within ``track_gate_px``. A groove may be missing in a
    frame without breaking the rest of its track.
    """
    seed = detect.detect_grooves(frames.gray[reference], settings)
    if not seed:
        raise RuntimeError(f"no grooves in reference frame {reference}")
    tracks = [GrooveTrack(index=i + 1) for i in range(settings.groove_count)]
    gate = settings.track_gate_px

    def walk(frame_seq: list[int], start: np.ndarray) -> None:
        pos = start.astype(float).copy()
        vel = np.zeros(settings.groove_count)
        for fid in frame_seq:
            valleys = np.array(detect.detect_valleys(frames.gray[fid], settings), float)
            predicted = pos + vel
            new_pos = pos.copy()
            matched = np.zeros(settings.groove_count, bool)
            for j in range(settings.groove_count):
                if valleys.size:
                    k = int(np.argmin(np.abs(valleys - predicted[j])))
                    if abs(valleys[k] - predicted[j]) < gate:
                        new_pos[j], matched[j] = valleys[k], True
            vel = 0.5 * vel + 0.5 * (new_pos - pos)
            pos = new_pos

            if fid not in azimuths:
                continue
            profile = detect.intensity_profile(frames.gray[fid], settings)
            for j in range(settings.groove_count):
                if not matched[j]:
                    continue
                tracks[j].positions[fid] = int(pos[j])
                asym = detect.wall_asymmetry(profile, int(pos[j]), settings)
                if asym is not None:
                    tracks[j].observations.append(
                        Observation(fid, azimuths[fid], asym))

    seed = np.array(seed, float)
    forward = [f for f in frames.frame_ids if f >= reference]
    backward = [f for f in frames.frame_ids if f < reference][::-1]
    walk(forward, seed)
    walk(backward, seed)
    return tracks


def run(video_path: str, settings: Settings | None = None) -> tuple[TreadResult, FrameStore, list[GrooveTrack]]:
    settings = settings or Settings()
    frames = load_frames_for(video_path, settings)

    reference, span_px = _pick_reference(frames, settings)
    px_per_mm = geometry.pixels_per_mm(span_px, settings)
    azimuths = estimate_azimuths(frames, reference, settings)
    tracks = _track_grooves(frames, azimuths, reference, settings)

    grooves: list[GrooveDepth] = []
    for track in tracks:
        obs = track.observations
        if len(obs) < settings.min_observations:
            grooves.append(GrooveDepth(
                index=track.index, image_x=track.image_x, depth_mm=float("nan"),
                amplitude_px=0.0, baseline_angle_deg=0.0, n_observations=len(obs),
                azimuth_span_deg=0.0, fit_rms_px=float("nan"), reliable=False))
            continue
        fit = geometry.fit_depth(
            [o.azimuth for o in obs], [o.asymmetry for o in obs], px_per_mm, settings)
        grooves.append(GrooveDepth(
            index=track.index, image_x=track.image_x, **fit))

    span_deg = float(np.degrees(max(azimuths.values()) - min(azimuths.values())))
    result = TreadResult(
        grooves=grooves,
        px_per_mm=px_per_mm,
        reference_frame=reference,
        azimuth_span_deg=span_deg,
        scale_assumption=(
            f"contact patch {settings.tread_width_mm:.0f} mm wide, grooves span "
            f"{settings.span_to_contact_ratio:.0%} of it"),
        notes=_notes(grooves, span_deg, settings),
    )
    return result, frames, tracks


def _notes(grooves, span_deg, settings) -> list[str]:
    notes = ["depth from visible-wall goniometry; floors are never reconstructed",
             "absolute depth scales with the stated contact-width assumption; "
             "the relative groove pattern does not"]
    weak = [g.index for g in grooves if not g.reliable]
    if weak:
        notes.append(
            f"groove(s) {weak} flagged low-confidence: wall signal near the "
            f"~{settings.min_resolved_depth_mm:.0f} mm resolution floor "
            "(heavily worn, or its wall is not revealed across this pan)")
    return notes


# kept as a thin indirection so tests can swap in a stub store
def load_frames_for(video_path: str, settings: Settings) -> FrameStore:
    from .frames import load_frames
    return load_frames(video_path, settings)
