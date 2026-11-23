---
name: origami-shaping
description: Shapes and 3D-poses a solved box-pleated origami base into an expressive 3D model. Covers automatic posing (`bp.pose`, `bp.PoseConfig`), one-shot geometric shaping (`bp.shape`), stateful iterative actuation (`bp.ShapingState`, `bp.apply_shaping_tool`), stick-figure joint subdivision, and visual/fidelity verification loops.
---

# Origami Shaping and 3D Posing

A solved box-pleated base (`SolvedCreasePattern`) allocates every flap and
structural river on a flat-foldable square sheet, with flaps initially stacked
along axial grid directions. `bp_pipeline` provides three complementary ways to
turn a solved base into a finished 3D origami model:

1. **One-shot geometric shaping (`bp.shape`)**: Inscribes 3D simple folds and
   river bends directly from the unmerged `StickFigure`.
2. **Automatic closed-loop posing (`bp.pose`, `bp.PoseConfig`)**: Combines
   simultaneous 3D rotation, root-frame angle feedback, side-crossing repair,
   and deadline-bounded anatomical search (see `references/pose.md`).
3. **Stateful actuator toolbox (`bp.ShapingState`, `bp.apply_shaping_tool`)**:
   Applies any of 19 orthogonal 2D/3D shaping operators step by step while
   preserving foldability and 3D partial-fold overlays (see
   `references/tools.md`).

## 1. Automatic Shaping and Posing (`bp.shape` and `bp.pose`)

```python
import bp_pipeline as bp

design = bp.prepare(stick_figure_dict)
packed = bp.pack(design)
solved = bp.solve_best(packed)

# Fast one-shot 3D geometric shaping:
shaped = bp.shape(
    solved,
    bp.ShapingTaskConfig(
        fold_strategy_3d=bp.FoldStrategy3D.DEFAULT,
        clip_slope_thresh=3.0,
        max_slope_thresh=10.0,
    ),
)

# Closed-loop pose recipe with configurable mode and budget:
posed = bp.pose(
    solved,
    bp.PoseConfig(mode='figure', narrow=False, feedback_rounds=1),
)
mesh_3d = bp.build_folded_mesh_3d(posed)
```

## 2. Iterative Actuation with `bp.ShapingState`

Every actuator accepts an immutable `bp.ShapingState` and returns an updated
state plus a `bp.ShapingToolResult` without mutating prior states, making
speculative branching and rollback trivial:

```python
import bp_pipeline as bp

tools = bp.list_shaping_tools()
state = bp.ShapingState.from_solved(solved)
probe_info = bp.probe_shaping(state, ['left_leg'])

state, step = bp.apply_shaping_tool(
    state,
    'apply_symmetric',
    {
        'tool': 'simple_fold',
        'element_name': 'left_leg',
        'arguments': {'angle': 45.0, 'depth': 0.0, 'verify_fold': False},
    },
)
assert step.ok, step.message
custom_shaped = state.to_shaped(solved)
```

Batch multiple steps with `bp.apply_shaping_tools(state, calls)`. Full
parameter signatures for all 19 actuators live in `references/tools.md`.

## 3. CLI Shaping Subcommands

Run interactive posing and shaping against any `bp-pipeline` output directory,
`.bps.json`, serialized stage JSON, `StickFigure` JSON, or example name:

```bash
# 1. Pack and solve a base without shaping:
bp-pipeline cat -o out/cat --no-shape

# 2. Inspect elements, orientation tree, and applicable tools:
bp-pipeline tools out/cat
bp-pipeline probe out/cat --elements left_front_leg,tail

# 3. Run an automatic pose recipe (modes: auto, figure, search, greedy):
bp-pipeline pose out/cat --mode auto --time-budget 60

# 4. Apply targeted single or batched actuator edits on top of the current state:
bp-pipeline actuate out/cat --tool fold_spine --args '{"fold_percent": 0.65}'
bp-pipeline actuate out/cat --calls calls.json

# 5. Evaluate 0-10 fidelity and inspect out/cat/shaped_3d_views.png:
bp-pipeline score out/cat
```

Session metadata is stored compactly in `shaping_state.json` alongside
`shaped_cp.svg` (and `flat_backbone.svg` when a partial-fold 3D overlay is
active). Pass `--reset` to `actuate`, `probe`, or `score` to return to the
unshaped base.

## 4. Stick-Figure Subdivision Guidelines

Linear chains of flap sticks meeting at degree-2 joints merge into one packed
flap of length `sum(length)` during `bp.pack`, then articulate at each joint
during `bp.shape`, `bp.pose`, or `tree_edit`:

- **Preserve total length and space joints**: Subdivided segment lengths must
  sum to the packed flap length; prefer `length >= 2` per segment (and
  `length >= 3` on proximal diagonal `width: 1` limbs) so adjacent simple-fold
  creases do not collide.
- **Keep rivers unsplit and hubs intact**: Never subdivide river sticks or add
  new degree-3+ branching hubs when editing an already-packed model.
- **Preserve bilateral symmetry**: Subdivide left and right partners identically
  with `azimuth_R = -azimuth_L` and `elevation_R = elevation_L`.
- **Use visible bend angles**: Separate consecutive segment angles by at least
  `35` to `90` degrees so bends are neither skipped nor heavily slope-clipped.

## 5. SVG Fold Conventions and Visual Checks

- **Colors**: `Fold.MOUNTAIN` is red (`#ff0000`), `Fold.VALLEY` is blue
  (`#0000ff`), `Fold.BORDER` is black (`#000000`), `Fold.HINGE` is green
  (packing only), and `Fold.TRIANGLE` marks ridge/pleat annotations.
- **Partial dihedral angles**: Stroke opacity (`fold_percent` in `(0, 1]`)
  sets the target dihedral angle to `fold_percent * 180` degrees (`1.0` for
  flat creases, `0.35` to `0.75` for open 3D pleats, squashes, and spines).
- **Turnaround verification**: Always evaluate 3D models from
  `shaped_3d_views.png` (`bp.save_3d_turnaround`) and `bp.evaluate_fidelity`
  (see `origami-critique`) rather than flat 180-degree previews; global
  flat-foldability checks apply to the solved base, not partial-angle 3D poses.
