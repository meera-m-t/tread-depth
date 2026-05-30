from __future__ import annotations

import argparse
from pathlib import Path

from . import annotate
from .config import Settings
from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tread-depth",
        description="Measure tread-groove depth from a single panning video "
                    "using visible-wall goniometry.")
    p.add_argument("video", help="path to the pan video")
    p.add_argument("-o", "--out", default="out", help="output directory")
    p.add_argument("--tread-width-mm", type=float,
                   help="override the assumed contact-patch width (scale)")
    p.add_argument("--no-video", action="store_true",
                   help="skip rendering the annotated video (faster)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    settings = Settings()
    if args.tread_width_mm:
        settings = settings.model_copy(update={"tread_width_mm": args.tread_width_mm})

    result, frames, tracks = run(args.video, settings)

    print(result.table())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tread_depth.json").write_text(result.model_dump_json(indent=2))
    annotate.render_still(frames, tracks, result, settings, str(out / "annotated.png"))
    annotate.render_fit_figure(tracks, result, str(out / "wall_fits.png"))
    if not args.no_video:
        annotate.render_video(args.video, frames, tracks, result, settings,
                              str(out / "output.mp4"))

    print(f"\nwrote results to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
