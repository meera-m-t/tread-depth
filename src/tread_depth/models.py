from __future__ import annotations

from pydantic import BaseModel, Field


class GrooveDepth(BaseModel):
    index: int  # 1..N, left to right in the image
    image_x: int  # column of the groove in the reference frame
    depth_mm: float
    amplitude_px: float  # fitted asymmetry amplitude (= depth_mm * px_per_mm)
    baseline_angle_deg: float  # crown angle where the groove sits at face-on
    n_observations: int
    azimuth_span_deg: float  # how far the groove swept across the pan
    fit_rms_px: float
    reliable: bool  # False when the fit is under-constrained (see notes)

    def line(self) -> str:
        flag = "" if self.reliable else "  (low confidence)"
        return (
            f"groove {self.index} @ x={self.image_x:4d}:  "
            f"{self.depth_mm:4.1f} mm   "
            f"[n={self.n_observations:2d}, sweep={self.azimuth_span_deg:4.1f}deg, "
            f"rms={self.fit_rms_px:3.1f}px]{flag}"
        )


class TreadResult(BaseModel):
    grooves: list[GrooveDepth]
    px_per_mm: float
    reference_frame: int
    azimuth_span_deg: float
    scale_assumption: str
    notes: list[str] = Field(default_factory=list)

    def table(self) -> str:
        rows = [g.line() for g in self.grooves]
        head = "tread depth (visible-wall goniometry)\n" + "-" * 38
        foot = (
            f"\nscale: {self.px_per_mm:.2f} px/mm  "
            f"(assumption: {self.scale_assumption})\n"
            f"reference frame {self.reference_frame}, "
            f"camera swept {self.azimuth_span_deg:.1f} deg"
        )
        notes = "\n".join(f"note: {n}" for n in self.notes)
        return "\n".join([head, *rows, foot] + ([notes] if notes else []))
