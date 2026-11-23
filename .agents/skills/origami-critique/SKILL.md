---
name: origami-critique
description: Quantitative and visual critique of folded 3D box-pleated origami models using `bp.evaluate_fidelity` and multi-view turnaround renders. Use when scoring a solved or shaped crease pattern, checking `clean_gate` and `pose_gate`, or diagnosing pose, width, separation, symmetry, and fold-strain defects.
---

# Origami Fidelity Critique

Evaluate every posed or shaped crease pattern by combining the deterministic
3D fidelity report (`bp.evaluate_fidelity` or `bp-pipeline score`) with visual
inspection of `shaped_3d_views.png` (`bp.save_3d_turnaround`).

## 1. Running Fidelity Evaluation

```python
import bp_pipeline as bp

result = bp.fold(stick_figure_dict, preset='demo', shape=False)
posed = bp.pose(result.solved, mode='figure', feedback_rounds=1)
report = bp.evaluate_fidelity(posed)

score = report.fidelity
clean_ok = report.clean_gate.passed
pose_ok = report.pose_gate.passed
components = report.components
```

From the command line:

```bash
bp-pipeline score out/model
bp-pipeline score out/model --reset   # score the unshaped base for comparison
```

## 2. Hard Quality Gates

Always check the two structural gates before optimizing the headline score:

- **`clean_gate`**: Passes when every internal vertex satisfies 2D
  Kawasaki/Maekawa flat-foldability (`non_flat_foldable_vertices == 0`) and
  deterministic-folder vertex strain is at most `0.85%`. Note that intentional
  3D Belcastro-Hull cuts (such as partial-fold pleats or squashes) preserve low
  vertex strain even when 2D Kawasaki is relaxed locally; reject any edit that
  pushes `fold_vertex_error_percent` above `0.85%`.
- **`pose_gate`**: Passes when `components['direction'] >= 0.70`, at most `2`
  flaps are stray (`direction_error_deg > 60` degrees), and `side_crossings` is
  empty (no paired limb tip or midpoint crosses or crowds the sagittal midline).

## 3. Fidelity Components and Weights

The headline `fidelity` score (`0` to `10`) is a weighted geometric mean of
component scores in `[0, 1]` (each floored at `0.05` so a single collapsed
dimension cannot hide behind high averages):

| Component | Weight | What it measures |
| :--- | ---: | :--- |
| `depth` | 2.0 | 3D principal-axis extent relative to the stick figure (`1.0` for planar designs) |
| `separation` | 1.5 | Tip non-clumping and left/right mirrored pair spacing |
| `width` | 1.5 | Mid-flap cross-section width compared to design `stick.width` |
| `pose` | 1.0 | Fraction of joints within `0.20` and `0.35` figure sizes after robust alignment |
| `silhouette` | 1.0 | Multi-view recall, wide-plate coverage, precision, and absence of side holes |
| `cleanliness` | 1.0 | Kawasaki flat-foldability and low 3D folder strain |
| `direction` | 0.5 | Length-weighted share of sticks within `45` degrees of target direction |
| `articulation` | 0.5 | Realized knee/elbow/hock bend angle along multi-segment paired limbs |
| `symmetry` | 0.5 | Left/right mirror position and width agreement across bilateral pairs |
| `volume` | 0.0 | Local body breadth diagnostic (reported in `components`, unweighted) |

## 4. Defect-to-Tool Remedy Guide

Use the diagnostic lists in `report` alongside `shaped_3d_views.png` to choose
targeted actuator steps (see `origami-shaping` for full tool parameters):

- **High `direction_error_deg` or `stray_flaps`**: Re-pose globally with
  `rotation_3d`, or steer individual flaps with `bend_to_direction`,
  `crimp_fold`, or `reverse_fold` (wrapped in `apply_symmetric` for pairs).
- **Non-empty `side_crossings` or `stacked_pairs`**: Increase outward azimuth
  in `rotation_3d` toward `+90` (left) and `-90` (right), or separate adjacent
  rays with `spread_fork` / `radial_fan`.
- **Low `articulation` on multi-segment limbs**: Add knee or elbow bends with
  `thin_limbs_3d`, `crimp_fold(mode='3d')`, or `curve_flap`.
- **Thick planks (`widths[name] >> stick.width`)**: Apply `simple_fold` with
  `narrow=True` (via `apply_symmetric` for pairs) or `narrow_central_flap` on
  centerline heads, horns, and tails.
- **Narrow plates (`widths[name] << stick.width` on `width >= 2` flaps)**:
  Widen and open wings, fins, petals, or fan tails with `squash_fold`,
  `fan_out_fold`, or `open_flap`.
- **Flat body (`depth` or `volume` low)**: Tent the midline with `fold_spine`,
  or expand body rivers with `open_flap` and `inscribed_flap`.
- **Asymmetry (`asymmetric_width_pairs` or low `symmetry`)**: Revert the
  one-sided edit and re-apply through `apply_symmetric(..., min_symmetry=0.85)`.
