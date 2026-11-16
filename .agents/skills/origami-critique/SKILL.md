---
name: origami-critique
description: Critiques a folded 3D origami model produced by `bp-pipeline` by inspecting rendered turnaround views (`shaped_3d_views.png`, `pipeline_stages.png`) alongside objective 3D fidelity metrics (`bp.evaluate_fidelity` / `bp-pipeline score`), scoring on an anchored 0-10 rubric, and diagnosing whether defects belong to stick-figure design, solving, or shaping.
---

# Visual & Geometric Critique of Folded Origami Models

Evaluate every folded model by combining **multi-angle visual inspection**
(`shaped_3d_views.png` and `pipeline_stages.png`) with **objective 3D fidelity
gates** (`bp.evaluate_fidelity` or `bp-pipeline score`). Always inspect all
camera angles (`Iso`, `Side`, `Front`, `Top`): a model that reads well from one
angle can still be a flat slab or a stack of overlapping legs from the side.

## 1. Objective Fidelity & Pose Gates (`bp.evaluate_fidelity`)

Compute the deterministic 3D fidelity report before scoring visually:

```python
import bp_pipeline as bp

result = bp.fold('cat', preset='demo', shape=True)
fidelity_report = bp.evaluate_fidelity(result)
assert fidelity_report['clean_gate']['locally_flat_foldable']
assert 'pose_gate' in fidelity_report
```

Check two hard geometric gates in `fidelity_report`:

*   **`clean_gate.passed`**: Requires `100%` local flat-foldability (or valid
    3D Belcastro-Hull relief vertices) and deterministic 3D fold axial strain
    `fold_vertex_error_percent <= 0.25%`.
*   **`pose_gate.passed`**: Fails when `components.direction < 0.70`, when more
    than 2 flaps stray over `60°` from their target stick-figure directions
    (`stray_flaps`), or when bilateral limbs cross the sagittal midline
    (`side_crossings`). Also inspect `stacked_pairs` (mirrored L/R limbs folded
    into the same plane) and `body_thickness`.

## 2. Anchored 0–10 Visual Rubric

Score what is **actually visible** in the rendered 3D turnaround views:

*   **0 — Unrecognizable**: Crumpled ball, flat unshaped sheet, or severe mesh
    tearing.
*   **2 — Abstract**: Overall mass exists, but appendages are indistinguishable.
*   **4 — Poor**: Recognizable attempt in some views, but missing key anatomy,
    wrong pose, or severe asymmetry.
*   **5 — Basic**: Most parts present, but stick-thin or plank-wide flaps, flat
    body strip in side view, or unarticulated limbs.
*   **6 — Fair**: Clearly recognizable in the hero (`Iso`) view with countable
    appendages and plausible pose, but lacking 3D body volume or joint bends.
*   **7 — Solid (all gates required)**:
    1. Hero `Iso` view reads as the subject at a glance without labels;
    2. Correct visible appendage count (mirrored L/R limbs separated, not
       stacked as one);
    3. Limbs articulated (visible knee/elbow/hock bends) and naturally splayed;
    4. Visible 3D body volume in `Side` view (not a paper-thin strip);
    5. `pose_gate.passed` and `clean_gate.passed` both hold.
    *(Missing 1 gate caps the score at `6.5`; missing 2+ caps at `6.0`.)*
*   **8 — Good**: Strong 3D resemblance from every angle with shaped details
    (fanned wings/tails, tapered limbs, wedged head/claws, arched spine).
*   **10 — Exceptional**: Exhibition-grade silhouette, proportions, and 3D
    shaping across all views.

Record structured critique notes per iteration (`agent_score.json`):

```jsonc
{
  "iteration": 2,
  "score": 6.5,
  "fidelity": 7.42,
  "reasons": [
    "Arched tail and raised pincers read clearly in Iso and Side views",
    "Walking legs 2-3 still overlap in Side view"
  ],
  "defects": [
    {
      "what": "Forelegs lack knee bend and taper",
      "stage": "shaping",
      "plan": "Narrow on base with clip_pattern_fold, pose, then bend_to_direction"
    }
  ]
}
```

## 3. Routing Defects Back to the Right Pipeline Stage

| Observed Defect | Stage to Revisit | Action |
| :--- | :--- | :--- |
| Wrong appendage count, missing anatomy, or poor proportions even in `stick_figure_views.png` | `stick-figure-design` | Adjust stick lengths, widths, and branching hubs, then re-pack and re-solve |
| Appendage too short (`length == 1`) to accept a bend or taper | `stick-figure-design` | Increase stick `length >= 2` (or `>= 3` for subdivided limbs) |
| Mirrored L/R flaps fail symmetrically or central flaps are trapped | `origami-solving-pipeline` | Re-solve with `remove_central_crease=True` or `bp.solve_sweep(packed)` |
| Right parts exist, but limbs are wide planks, stacked, or unbent | `origami-shaping` | Run `bp-pipeline probe`, narrow thin limbs on the base, apply `bp.pose`, and refine with `bend_to_direction` / `squash_fold` / `open_flap` |
| Folded mesh self-intersects or crumples despite local flat-foldability | `origami-shaping` | Reduce partial `fold_percent` / `open_percent` magnitudes and check `fold_vertex_error_percent` |
