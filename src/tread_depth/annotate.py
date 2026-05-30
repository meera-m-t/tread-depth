from __future__ import annotations

import cv2
import numpy as np

from .config import Settings
from .frames import FrameStore
from .models import TreadResult
from .pipeline import GrooveTrack

_GREEN = (60, 190, 90)
_AMBER = (40, 170, 235)
_INK = (235, 235, 235)
_DARK = (30, 30, 30)


def _depth_color(reliable: bool):
    return _GREEN if reliable else _AMBER


def _interp_x(track: GrooveTrack, frame_id: int):
    if not track.positions:
        return None
    fids = sorted(track.positions)
    if frame_id < fids[0] or frame_id > fids[-1]:
        return None
    xs = [track.positions[f] for f in fids]
    return int(np.interp(frame_id, fids, xs))


def _banner(img, lines: list[str]):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 28 + 22 * len(lines)), _DARK, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (16, 26 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _INK, 1, cv2.LINE_AA)


def _draw_grooves(img, frame_id, tracks, result, settings):
    h = img.shape[0]
    lo, hi = settings.profile_band
    y0, y1 = int(lo * h), int(hi * h)
    for track, groove in zip(tracks, result.grooves):
        x = _interp_x(track, frame_id)
        if x is None:
            continue
        color = _depth_color(groove.reliable)
        cv2.line(img, (x, y0), (x, y1), color, 2, cv2.LINE_AA)
        label = (f"G{groove.index} {groove.depth_mm:.1f}mm"
                 if groove.reliable else f"G{groove.index} ~{groove.depth_mm:.1f}?")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y0 - 10
        cv2.rectangle(img, (x - tw // 2 - 4, ty - th - 4),
                      (x + tw // 2 + 4, ty + 4), _DARK, -1)
        cv2.putText(img, label, (x - tw // 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def render_video(video_path: str, frames: FrameStore, tracks: list[GrooveTrack],
                 result: TreadResult, settings: Settings, out_path: str) -> None:
    azimuth = {o.frame_id: o.azimuth for tr in tracks for o in tr.observations}
    cap = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"),
        settings.output_fps, (frames.width, frames.height))

    for fid in range(settings.frame_start, settings.frame_stop + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            continue
        _draw_grooves(frame, fid, tracks, result, settings)
        az = azimuth.get(fid)
        _banner(frame, [
            "Tread depth - visible-wall goniometry",
            f"frame {fid}" + (f"   view angle {np.degrees(az):+.0f} deg" if az is not None else ""),
        ])
        writer.write(frame)
    writer.release()
    cap.release()


def render_still(frames: FrameStore, tracks: list[GrooveTrack],
                 result: TreadResult, settings: Settings, out_path: str) -> None:
    img = frames.color[result.reference_frame].copy()
    _draw_grooves(img, result.reference_frame, tracks, result, settings)
    _banner(img, [
        "Tread depth - reference frame",
        f"scale {result.px_per_mm:.2f} px/mm  (assumed contact width "
        f"{settings.tread_width_mm:.0f} mm)",
    ])
    cv2.imwrite(out_path, img)


def render_fit_figure(tracks: list[GrooveTrack], result: TreadResult,
                      out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(result.grooves)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4), squeeze=False)
    for ax, track, groove in zip(axes[0], tracks, result.grooves):
        az = np.degrees([o.azimuth for o in track.observations])
        asym = [o.asymmetry for o in track.observations]
        ax.scatter(az, asym, s=12, alpha=0.5, color="#3b6ea5")
        amp = groove.amplitude_px
        phase = np.radians(groove.baseline_angle_deg)
        xs = np.linspace(min(az), max(az), 80)
        ax.plot(xs, amp * np.sin(np.radians(xs) + phase), color="#c23b22", lw=2)
        tag = "" if groove.reliable else "  (low conf.)"
        ax.set_title(f"groove {groove.index}: {groove.depth_mm:.1f} mm{tag}")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlabel("camera view angle (deg)")
    axes[0][0].set_ylabel("wall asymmetry (px)")
    fig.suptitle("Wall asymmetry vs viewing angle  (amplitude = depth x px/mm)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
