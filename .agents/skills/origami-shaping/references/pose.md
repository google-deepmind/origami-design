# Automatic Pose Recipes (`bp.pose`)

`bp.pose` synthesizes a clean 3D pose on any flat-foldable `SolvedCreasePattern`
using `bp.PoseConfig` and returns an immutable `bp.ShapedCreasePattern` whose
`diagnostics` dictionary summarizes the score, gate status, and applied steps.

## 1. Configuration (`bp.PoseConfig`)

- `mode: bp.PoseMode = 'auto'` — Recipe strategy (`'auto'`, `'figure'`,
  `'search'`, or `'greedy'`).
- `narrow: bool = False` — Narrow width-1 outer limbs and central flaps when the
  narrowed pattern remains clean.
- `time_budget_s: float = 60.0` — Wall-clock budget in seconds for `'search'`
  refinement (`'figure'` runs to completion; `'auto'` allocates remaining budget
  to `'search'` when `'figure'` has not yet passed both `clean_gate` and
  `pose_gate`).
- `feedback_rounds: int = 2` — Maximum closed-loop root-frame direction feedback
  rounds during `'figure'` posing.
- `min_gain: float = 0.05` — Minimum fidelity improvement required to accept an
  incremental step during `'search'`.

Keyword arguments on `bp.pose(solved, mode=..., narrow=..., time_budget_s=...,
feedback_rounds=...)` override fields of `config` directly.

## 2. Choosing a Pose Mode

| Mode | 3D folder calls | Best for |
| :--- | :--- | :--- |
| `'auto'` | Fast when clean | Default end-to-end posing: runs `'figure'`, escalating to `'search'` in remaining budget only if `pose_gate` or `clean_gate` is not yet met |
| `'figure'` | 2 to 5 | Fast, accurate alignment to stick-figure `(azimuth, elevation)` with symmetric feedback and crossing repair |
| `'search'` | Budget-bounded | Exploring plate flaring (`squash_fold`, `fan_out_fold`, `open_flap`), tapering, and body relief |
| `'greedy'` | 0 | Millisecond flat-foldable posing in tight inner loops without 3D folder evaluation |

## 3. How Closed-Loop Posing Works

1. **Anatomical role classification (`pose_plan`)**: Classifies sticks into
   `root`, `river`, `thin_limb`, `limb`, `segment`, `wide_tip`, and `tip_down`
   and adds subtle symmetric outward azimuth bias on steep downward legs so
   mirrored feet do not cross at the midline (automatically retrying unbiased
   stick-figure angles and reporting `unbiased_fallback: true` in `diagnostics`
   if the bias introduces fold strain).
2. **Simultaneous 3D rotation (`rotation_3d`)**: Rotates non-root elements from
   the root stick's frame, retrying order conflicts and checking a flaps-only
   fallback when river bends introduce vertex strain.
3. **Symmetric root-frame feedback**: Measures each flap's realized direction in
   the root stick's own frame and applies damped, bilaterally symmetrized
   `(az, el)` corrections across up to `feedback_rounds` rounds.
4. **Side-crossing repair**: If any paired flap crosses or crowds the sagittal
   mid-plane, nudges the pair outward toward `+90` / `-90` degrees.
5. **Gate-preserving local refinement (`search`)**: Tests symmetry-wrapped
   candidate actuators (`bend_to_direction`, `open_flap`, `squash_fold`,
   `fan_out_fold`, `simple_fold(narrow=True)`, `narrow_central_flap`,
   `curve_flap`, `inscribed_flap`, `fold_spine`), keeping only edits that raise
   fidelity by at least `min_gain` without regressing `clean_gate` or
   `pose_gate`.

## 4. Hand-Refining After `bp.pose`

Start an interactive session from a posed pattern to sculpt subject-specific
details:

- **Python**: `state = bp.ShapingState.from_solved(solved,
  pattern=posed.creases)` followed by `bp.apply_shaping_tool(state, ...)`.
- **CLI**: Run `bp-pipeline pose out/model --mode figure`, inspect
  `out/model/shaped_3d_views.png` and `bp-pipeline score out/model`, then chain
  `bp-pipeline actuate out/model --tool ... --args ...` on top of the saved
  `shaping_state.json`.
