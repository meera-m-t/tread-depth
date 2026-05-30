# Tread depth from a single panning video

Estimates the depth of each circumferential tread groove from one monocular pan
of a tyre (`Tire_Pan_2.mp4`) and renders an annotated video so the result can be
judged by eye.

## Install & run

```bash
pip install .                       # add [viz,test] for the fit figure + tests
tread-depth Tire_Pan_2.mp4 -o out
pytest
```

Outputs in `out/`: a console table, `tread_depth.json`, `annotated.png`,
`wall_fits.png`, and `output.mp4` (each groove tracked and labelled with its
depth). `--tread-width-mm` overrides the scale assumption; `--no-video` skips
rendering.

## The problem, and why it is hard

Reading tread depth in millimetres from a single uncalibrated handheld video has
two hard parts:

1. **No scale.** There is no fiducial in the scene, so metric depth must rest on
   one stated assumption.
2. **The floor is the worst thing to measure.** The quantity we want is how far
   the groove floor sits below the ribs, but the floor is dark, textureless and
   self-occluded — exactly what defeats the obvious geometric method.

## Approach: one method, chosen on evidence

Three depth cues were considered; one was committed to after a quick empirical
check on the actual footage.

- **Photometric (floor darkness).** Needs uniform diffuse lighting. Here the
  light is mixed (daylight plus garage), so floor/rib brightness is confounded
  by wall shading, siping and dirt. Rejected.
- **Multi-view triangulation / SfM.** The textbook answer, and it was tried
  first — but it **fails on this clip**. The textureless, self-occluded floor
  bottoms yield no matchable features, so the reconstructed tread cross-section
  is a flat noise band with no dip at the grooves; depth reads ~0 mm. Geometry
  ends up measuring the wall tops, not the floor. Rejected *because it was
  tested*, not on a hunch.
- **Visible-wall goniometry (committed).** Sidesteps the floor entirely.

### How the committed method works

A circumferential groove reads as a dark valley. When the tread is tilted, the
groove's near **wall** becomes visible and the dark valley goes lopsided. A
little projective geometry gives

```
asymmetry_px  ≈  depth · sin(view_angle) · px_per_mm
```

so as the camera pans and the view angle sweeps, the asymmetry of each groove
traces a sinusoid whose **amplitude is the depth**. We fit that sinusoid per
groove. The view angle comes from relative camera pose (essential matrix to
seed, then PnP against a shared point set, so all frames share one consistent
frame — no bundle adjustment, no metric reconstruction). Scale (`px_per_mm`)
comes from the one stated assumption: the contact-patch width.

The point of the method is that it only ever uses the directly-visible **walls**
(which carry texture and slant), never the unreconstructable floor.

## Pipeline

```
frames  → load + subsample, approximate intrinsics
detect  → locate the 4 evenly-spaced grooves; measure per-frame wall asymmetry
pose    → per-frame camera view angle from essential + PnP
geometry→ scale from the width assumption; robust sinusoid fit → depth
pipeline→ track grooves across the pan, fit each, return TreadResult
```

All tunables live in `config.py` (pydantic). Results are pydantic models with
provenance (observation count, view-angle sweep, fit residual) so a reader can
weigh each number.

## Result on `Tire_Pan_2.mp4`

| groove | depth | note |
|--------|-------|------|
| 1 | ~1 mm | low confidence — wall barely revealed across this pan |
| 2 | 6.1 mm | |
| 3 | 6.0 mm | |
| 4 | 5.8 mm | |

Scale 3.90 px/mm (assumed 175 mm contact width). Camera swept ~45°.

**What to trust.** The *relative* groove pattern is robust. The *absolute* depth
scales with the contact-width assumption and carries a few mm of model
uncertainty (the sinusoid under-fits the saturating wall-visibility curve, which
is why `wall_fits.png` shows elevated residuals on the deep grooves). Grooves
whose wall is barely revealed across the pan are flagged low-confidence rather
than reported as confident shallow readings.

## Assumptions

- **Scale:** contact patch ≈ 175 mm wide; the four grooves span ≈ 60 % of it.
  Absolute depth is linear in this; change it with `--tread-width-mm`.
- **Intrinsics:** focal length ≈ image width, principal point centred. Only the
  recovered *rotation* matters here, and rotation is insensitive to focal error.
- **Model:** asymmetry is treated as `depth · sin(view_angle)`; the real
  wall-visibility curve saturates, so absolute depth is approximate.

## Limitations and what would improve it

- A fiducial or a known tread spec would remove the scale assumption.
- A forward wall-visibility model (accounting for saturation and the crown
  profile) would tighten absolute depth and lower the fit residual.
- Grooves that stay near face-on through the pan are under-observed; a wider arc
  or a second pass from another angle would constrain them.

## Tests

`pytest` covers the parts provable without the messy video: synthetic
known-depth recovery across depths and phases, rejection of tracking-glitch
outliers, low-confidence flagging of a flat (no-signal) groove, and groove
detection on a synthetic profile.

## Scope

Per the brief this is **one** committed method and a clean, reviewable slice —
not a full production pipeline. Absolute accuracy is explicitly not the target;
the scale assumption is stated rather than hidden.
