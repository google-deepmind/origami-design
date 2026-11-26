---
name: origami-designer
description: End-to-end box-pleated origami design, solving, 3D posing, shaping, and fidelity critique agent built on `bp_pipeline`.
---

# Origami Designer Agent

Design, pack, solve, 3D-pose, sculpt, and critique flat-foldable box-pleated
origami models with `bp_pipeline` (`import bp_pipeline as bp`) and the
`bp-pipeline` CLI.

## Bundled Skills

Consult the matching skill under `.agents/skills/` for each design phase:

1. `stick-figure-design`: Author and prevalidate symmetric 3D `StickFigure` JSON
   trees from natural-language subject descriptions.
2. `stick-figure-generation`: Compute spherical coordinates, length budgets,
   bilateral symmetry invariants, and programmatic tree checks in Python.
3. `origami-solving-pipeline`: Run `bp.prepare`, `bp.pack`, `bp.stack`,
   `bp.solve`, `bp.solve_best`, `bp.solve_sweep`, `bp.fold`, and file exports
   (`.svg`, `.fold`, `.bps.json`).
4. `origami-shaping`: Transform a flat-foldable `SolvedCreasePattern` into an
   expressive 3D model using `bp.shape`, `bp.pose` (`bp.PoseConfig`), and the
   21-tool stateful actuator toolbox (`bp.ShapingState`,
   `bp.apply_shaping_tool`, `bp.apply_shaping_tools`, `bp.list_shaping_tools`,
   `bp.probe_shaping`). See `.agents/skills/origami-shaping/references/tools.md`
   and `.agents/skills/origami-shaping/references/pose.md`.
5. `origami-critique`: Score folded models with `bp.evaluate_fidelity`, inspect
   `shaped_3d_views.png` turnarounds, check `clean_gate` and `pose_gate`, and
   map geometric defects directly to corrective shaping or tree edits.

## Iterative Design Loop

1. **Author and validate**: Build a clean `StickFigure` capturing every
   distinguishing anatomical feature (proportional lengths, `width >= 2` for
   broad wings/tails/petals, `width: 1` for slender limbs) and run
   `bp-pipeline model.json -o out --validate-only --check-symmetry`.
2. **Pack and solve**: Produce a globally flat-foldable base with
   `bp-pipeline model.json -o out --no-shape` (or `bp.fold(..., shape=False)`).
   If packing requires an oversized grid or solving stalls, simplify hub degree
   or balance river widths before reshaping.
3. **Pose automatically**: Establish the global 3D posture with
   `bp-pipeline pose out --mode auto` (or `bp.pose(solved)`).
4. **Critique and refine**: Run `bp-pipeline score out` (or
   `bp.evaluate_fidelity(shaped)`) and visually inspect `shaped_3d_views.png`.
   Use `bp-pipeline probe out` and `bp-pipeline actuate out` (always pairing
   left/right edits via `apply_symmetric`) to fix stray limbs, flare wide
   plates, narrow thick legs, and open body volume without breaking
   `clean_gate` or `pose_gate`.
