---
name: origami-shaping
description: Shapes a solved box-pleated origami base into an expressive 3D model. Covers stick-figure joint subdivision, automated 3D geometric shaping (`bp.shape`), deterministic pose recipes (`bp.pose`), the 28-tool stateful shaping actuator toolbox (`bp.ShapingState`, `bp.apply_shaping_tools`), and objective 3D fidelity scoring (`bp.evaluate_fidelity`).
---

# Origami Shaping for Box-Pleated Models

A solved box-pleated **base** (`SolvedCreasePattern`) allocates every appendage
flap and structural river on a flat-foldable square sheet, but its flaps
initially lie flat along the axial grid directions. **Shaping** transforms this
uniaxial base into a lifelike, 3D-posed origami model by introducing:

1.  **3D Joint Subdivisions** in the unmerged `StickFigure`
    (`DesignTree.source`), which bend limbs, necks, tails, and horns at specific
    3D spherical angles while keeping their total packed lengths unchanged.
2.  **3D Geometric Simple Folds & River Bends** inscribed directly into the flap
    and river regions of the crease pattern (`bp.shape(...)`).
3.  **Partial Fold-Angle Assignments** (encoding target dihedral fold angles via
    crease opacity / `fold_percent`) for 3D geometric folding.

--------------------------------------------------------------------------------

## 1. Two-Stage Shaping Architecture

In `bp_pipeline`, shaping operates across two coupled representations:

```
Unmerged StickFigure (3D pose + joint bends)
        │
        ▼  merge_linear_chains()
Merged StickFigure ──► bp.pack() ───► PackedCreasePattern
        │                                     │
        │                                     ▼  bp.solve_best()
        │                            SolvedCreasePattern
        │                                     │
        └───────────► bp.shape(...) ◄─────────┘
                            │
                            ▼
               Shaped CreasePattern (SVG / FOLD)
```

-   **Why `merge_linear_chains()` matters**: During packing, any linear chain of
    flap sticks connected at degree-2 joints (e.g., `upper_arm` $\to$ `forearm`
    $\to$ `hand`) is merged into a single flap of length $L_{\text{total}} =
    \sum L_i$. During shaping (`bp.shape`), the unmerged tree's intermediate
    degree-2 joints tell the geometric folder **where** along the flap's ridge
    line to place simple folds, and at **what 3D angle**!

--------------------------------------------------------------------------------

## 2. Stage A: Stick-Figure Subdivision Shaping (Pose & Joint Articulation)

When you have a valid `StickFigure` (or an existing `PackedCreasePattern`) and
want to articulate its 3D pose without changing its 2D packing layout:

### Invariants for Subdivision Shaping

1.  **Conserve Merged Stick Lengths & Space Joints by `length >= 2`**:
    -   Every linear chain of non-river leaf sticks that merges into a single
        stick in `tree.merge_linear_chains()` **must sum to the exact same
        integer length** as the original stick, and each subdivided segment
        should have `length >= 2` so adjacent simple folds do not collide.
    -   Example: A straight `left_leg` of `length: 5` can be subdivided into:
        -   `left_thigh` (`start: "hips"`, `end: "left_knee"`, `length: 3`,
            `elevation_angle: -45`, `azimuth_angle: 90`, `width: 1`)
        -   `left_shin` (`start: "left_knee"`, `end: "left_foot"`, `length: 2`,
            `elevation_angle: -85`, `azimuth_angle: 90`, `width: 1`)
        -   Total length $= 3 + 2 = 5$. The 2D packing grid size is
            **unchanged**!
2.  **Never Subdivide or Re-angle Rivers**:
    -   Do **not** split internal river sticks (`stick.is_river == True`) into
        multiple sticks, and do not change their `length`. Rivers correspond to
        fixed-width spacer bands in the crease pattern.
3.  **Preserve Branching Hubs**:
    -   Do not add new degree-$\ge 3$ branching joints when shaping an
        already-packed model, as a new branch turns an internal segment into a
        river and invalidates the packing.
4.  **Maintain Strict Bilateral Symmetry**:
    -   Every subdivision on `left_X` must be mirrored identically on `right_X`
        with `elevation_angle` $\psi_R = \psi_L$ and `azimuth_angle` $\phi_R =
        -\phi_L$.
5.  **Enforce Distinct Joint Angles**:
    -   Consecutive sticks in a subdivided chain should have distinct
        `(elevation_angle, azimuth_angle)` values (differing by at least
        $37^\circ\text{–}90^\circ$) so the joint produces a visible 3D bend
        without triggering slope clipping (`clip_slope_thresh=3.0` clips bends
        in $(12^\circ, 37^\circ)$ and `max_slope_thresh=10.0` skips bends
        $\lesssim 12^\circ$).
6.  **Keep Multi-River Spines Co-Axial & Basal On-Axis Segments Collinear**:
    -   When a model uses 2 on-axis rivers (`body` and `neck`), keep both rivers
        co-axial in `(azimuth_angle, elevation_angle)` (e.g., `(0.0, 0.0)`) so
        rotating the shaping root does not assign a non-zero 3D fold angle to an
        open river bank.
    -   Because the 2D kinematic folder (`compute_folded_state`) folds every
        shaping crease flat by $180^\circ$ regardless of `fold_percent`, any
        non-zero elevation bend at the base of an on-axis chain (`tail_base`,
        `horn_base`) or an odd number of segments ($K = 3 \implies 2$ horizontal
        folds) reflects the appendage $180^\circ$ back into the torso in flat 2D
        silhouettes. Keep the basal segment collinear ($0^\circ$ bend) with the
        spine (placing the upward elevation bend $\Delta\psi \ge 37^\circ$ on
        `tail_tip` / `horn_tip`) or use $K = 2$ segments with $L_2 > L_1$, and
        always evaluate 3D poses using `shaped_3d_views.png` rather than a flat
        $180^\circ$ 2D reflection.
7.  **Use `width >= 2` Spine & `length >= 2` Flaps on 5-Flap Crowns**:
    -   When an anterior hub (`crown`) has 5 flaps (e.g., `snout` +
        `left_ear`/`right_ear` + 2-segment antlers), use `width >= 2` (e.g.,
        `width: 2`) on all symmetry-axis sticks (preserving even parity) and
        `length >= 2` on all terminal flaps so ear flaps are not stretched into
        `1×4` split-circle strips whose shaping creases collide.
8.  **Coplanar Pure-Azimuth Pincers (`elevation_angle: 0.0°`) vs. 3D Knee
    Clearance**:
    -   For **pure-azimuth** horizontal limbs (pincer claws `claw_upper` $\to$
        `claw_lower`, jaws/mandibles), keeping `elevation_angle: 0.0°` coplanar
        with the body preserves a vertical sheet normal (`[0, 0, -1]`) after the
        base fold so the elbow azimuth bend projects at its full angle in 2D. Do
        **not** flatten walking or raptorial leg knees (`femur` $\to$ `tibia`)
        to `0°` elevation.
    -   On any subdivided limb of width $W$, size the proximal segment so the
        hub's simple-fold triangle clears the first degree-2 joint:
        $L_{\text{upper}} > W \cdot \cot(\Delta\phi_{\text{base}} / 2)$. Note
        that a flap's packed axial direction (`0°`, `±90°`, or `180°`) depends
        on whether `bp.pack()` places it on a side border (`±90°`) or wraps it
        onto the top/bottom border (`0°` / `180°`) — e.g., `azimuth_angle: 122°`
        has $\Delta\phi_{\text{base}} = 32^\circ$ on a side border vs.
        $58^\circ$ on a top/bottom border. For $W = 1$ subdivided limbs at
        diagonal azimuths ($30^\circ\text{–}60^\circ$ or
        $120^\circ\text{–}150^\circ$), prefer $L_{\text{upper}} \ge 3$ so the
        hub fold clears the knee joint under either border placement; otherwise
        `bp.shape` discards the colliding limb pair
        (`shaping.discarded_elements`).

--------------------------------------------------------------------------------

## 3. Stage B: Automated 3D Geometric Shaping (`bp.shape`)

Once you have a `SolvedCreasePattern` (from `bp.solve_best(packed)` or
`bp.fold(...)`), run `bp.shape(...)` to inscribe 3D geometric shaping creases
into the base crease pattern:

```python
import bp_pipeline as bp

design = bp.prepare(stick_figure_dict)
packed = bp.pack(design)
solved = bp.solve_best(packed)

# Run automated 3D geometric shaping on the solved base:
shaped = bp.shape(solved)
shaped_svg = shaped.to_svg()       # or bp.to_svg(shaped)
shaped_fold = bp.to_fold(shaped)

# Build the 3D hinge-folded mesh (FoldedMesh3D) at partial dihedral angles:
mesh_3d = bp.build_folded_mesh_3d(shaped)
folded_3d_dict = mesh_3d.to_fold_dict('model')

# Compute the 2D folded model in FOLD format (strict=True is default; pass
# strict=False to fall back to kinematic 2D face folding when local 3D shaping
# creases introduce multi-layer taco-tortilla overlaps):
folded_shaped = shaped.flat_fold(strict=False)
```

### How Automated 3D Geometric Shaping Works Under the Hood

1.  **Central-Crease Adjustment**:
    -   If the solver used `remove_central_crease=True` to eliminate redundant
        centerline pleats, `repack_cp_with_shortened_flaps` and
        `adjust_shaping_elements_for_central_crease_removal` automatically
        adjust flap and river bounding regions before inscribing shaping folds.
2.  **Shaping Element Extraction**:
    -   `convert_packing_elements_to_shaping_elements` maps each packed flap
        (`Flap`) and river (`River`) into `ShapingFlap` and `ShapingRiver`
        regions on the solved crease pattern.
3.  **3D Geometric Folding Heuristics (`FoldStrategy3D`)**:
    -   Walks the unmerged `StickFigure` (`shape_with_unmerged_tree=True`) from
        the shaping root in 3D posture space (`PostureMatrix`).
    -   At each joint between a parent stick and child stick (including rivers
        and subdivided limb chains), computes the 3D simple-fold rotation axis
        and signed dihedral angle (`SimpleFoldInfo`) and projects the resulting
        cut line across the folded faces into the flap or river region.
    -   Supports three 3D folding strategies (`fold_strategy_3d`):
        -   `bp.FoldStrategy3D.DEFAULT`: Single 3D physical simple fold per
            joint.
        -   `bp.FoldStrategy3D.ROTATE_BEND`: Decomposes 3D rotations into an
            in-plane fold followed by an out-of-plane bend.
        -   `bp.FoldStrategy3D.SHALLOW_FOLDS`: Splits acute in-plane rotations
            into two shallow folds followed by an out-of-plane bend.
    -   Slope clamping:
        -   `clip_slope_thresh` (default `3.0`): Simple folds with slope between
            `clip_slope_thresh` and `max_slope_thresh` are clipped to
            `clip_slope_thresh` to prevent extreme acute slivers.
        -   `max_slope_thresh` (default `10.0`): Simple folds steeper than
            `max_slope_thresh` are omitted to preserve clean mesh geometry.
4.  **Local Flat-Foldability Filtering
    (`filter_by_local_flat_foldability=True`)**:
    -   Verifies Kawasaki's and Maekawa's theorems at interior vertices of the
        shaped crease pattern (`shaped_cp.local_flat_foldability()`).

### Recommended `ShapingTaskConfig` Settings

```python
import bp_pipeline as bp

shaping_config = bp.ShapingTaskConfig(
    apply_simple_fold_shaping=True,
    fold_strategy_3d=bp.FoldStrategy3D.DEFAULT,
    shape_with_unmerged_tree=True,
    filter_by_local_flat_foldability=True,
    clip_slope_thresh=3.0,
    max_slope_thresh=10.0,
    cp_tolerance=0.001,
)
```

--------------------------------------------------------------------------------

## 4. Crease Opacity & Partial Fold Angles in SVG

In the origami SVG format used by `bp_pipeline`:

-   **Stroke Color** encodes fold direction:
    -   `red` (`#ff0000`): Mountain fold (`Fold.MOUNTAIN`, $-180^\circ$ at full
        fold).
    -   `blue` (`#0000ff`): Valley fold (`Fold.VALLEY`, $+180^\circ$ at full
        fold).
    -   `black` (`#000000`): Paper boundary (`Fold.BORDER`).
    -   `green` (`#00ff00`): Unassigned hinge (`Fold.HINGE`, Packing CP only —
        must not appear in a solved/shaped CP).
    -   `yellow` / `cyan` / `magenta`: Axial ridge and pleat annotations during
        solving (`Fold.TRIANGLE`).
-   **Stroke Opacity (`stroke-opacity` / `fold_percent` $\in [0.0, 1.0]$)**
    encodes the target dihedral angle:
    $$\theta_{\text{target}} = \text{fold\_percent} \times 180^\circ$$
    -   Base structural pleats typically use `fold_percent = 1.0` ($180^\circ$
        flat fold).
    -   3D shaping folds (e.g., opening a body cavity, flaring wings at
        $90^\circ$, or gently curving a tail) can use intermediate
        `fold_percent` values such as `0.5` ($90^\circ$) or `0.25` ($45^\circ$).

--------------------------------------------------------------------------------

## 5. Interpreting `report.json` & Visual Troubleshooting

-   **Target `report.json` acceptance bar (`bp-pipeline` CLI)**:
    -   Require `solving.is_globally_flat_foldable == true` (globally
        flat-foldable uniaxial base) **and** `shaping.discarded_elements == []`
        (all 3D limb/river shaping folds retained). Note that on symmetric
        trees, if one side of a left/right mirror pair fails local
        flat-foldability during `bp.shape()`, both sides are discarded together
        to preserve bilateral symmetry — so `shaping.is_locally_flat_foldable:
        true` alone does **not** guarantee all limbs were shaped unless
        `shaping.discarded_elements` is empty.
    -   `shaping.is_globally_flat_foldable: false` (e.g., `"taco-tortilla
        conflict"` or `"Local violations found"`) or `global_ff_timed_out: true`
        is **expected** on complex 3D-shaped models because `flat_folder` checks
        simultaneous $180^\circ$ flat folding rather than partial 3D dihedral
        angles (`fold_percent * ±180°`).
-   **Judging 3D poses (`shaped_3d_views.png` vs. flat 2D folding)**:
    -   *Best practice*: Always inspect `shaped_3d_views.png` (the 4-angle
        shaded 3D hinge-folded turnaround: `Iso`, `Side`, `Front`, `Top` emitted
        by `bp-pipeline` or `bp.save_3d_turnaround(shaped, ...)`) with your
        image/file viewer tool. It evaluates partial dihedral fold angles
        (`fold_percent * ±180°`), whereas 2D flat/kinematic previews collapse
        every shaping crease flat by `180°`.
-   **Limbs point straight up/down with no joints**:
    -   *Root cause*: `StickFigure` has single un-subdivided sticks per limb.
    -   *Fix*: Subdivide limb sticks into 2–3 segments (`is_river: false`)
        conserving total length, then re-run `bp.shape()`.
-   **Left and right limbs bend in opposite 3D directions**:
    -   *Root cause*: Symmetric flap angles were given the same sign instead of
        opposite signs.
    -   *Fix*: Ensure $\phi_R = -\phi_L$ in `StickFigure`.
-   **`local_flat_foldability()` fails after shaping**:
    -   *Root cause*: Two shaping folds are too close together or a simple fold
        slope is too steep.
    -   *Fix*: Set `clip_slope_thresh=2.5, max_slope_thresh=6.0`, or space
        subdivision joints at least 2 grid units apart.
-   **Central flaps look too wide or trapped**:
    -   *Root cause*: Solver ran with `remove_central_crease=False` on a model
        with narrow central features.
    -   *Fix*: Solve with `remove_central_crease=True` and
        `disallow_trapping_edge_flaps=True` before shaping.

--------------------------------------------------------------------------------

## 6. Stage C: Stateful Actuator Toolbox & Pose Recipes (`bp.pose`, `bp.ShapingState`)

When `bp.shape(solved)` discards colliding limbs or when you want thin narrowed
legs, fanned wings/tails, bilateral wedge squashes, concentric carapace relief,
or fine-grained joint articulation, drive the **stateful shaping toolbox**
(`bp_pipeline.shaping.toolbox`) and **pose recipes** (`bp.pose`):

```python
import bp_pipeline as bp

# 1. Start from the deterministic pose recipe (or directly from `solved`):
posed, pose_report = bp.pose(solved, search=False, return_report=True)
fidelity_report = bp.evaluate_fidelity(posed)

# 2. Inspect available actuators and refine step by step with `ShapingState`:
available_tools = bp.list_shaping_tools()
state = bp.ShapingState.from_solved(solved)
probe_outcomes = bp.probe_shaping(state, ['left_leg'])
state, results = bp.apply_shaping_tools(
    state,
    [
        {
            'name': 'simple_fold',
            'arguments': {
                'element_name': 'left_leg',
                'angle': 45.0,
                'depth': 0.0,
                'narrow': True,
            },
        },
        {
            'name': 'simple_fold',
            'arguments': {
                'element_name': 'right_leg',
                'angle': -45.0,
                'depth': 0.0,
                'narrow': True,
            },
        },
    ],
)
custom_shaped = state.to_shaped(solved)
custom_fidelity = bp.evaluate_fidelity(custom_shaped)
```

### CLI Subcommands (`bp-pipeline tools | probe | pose | actuate | score`)

You can run iterative shaping directly from the terminal against an output
directory created by `bp-pipeline <design.json> -o <run_dir> --no-shape`:

```bash
# 1. Inspect the BFS element table, flap orientations, and 28 tool names:
bp-pipeline tools <run_dir>

# 2. Apply the deterministic pose recipe and render shaped_3d_views.png:
bp-pipeline pose <run_dir>

# 3. Probe which basic folds and 1/n narrowings succeed on target elements:
bp-pipeline probe <run_dir> --elements left_leg,right_leg

# 4. Apply a batch of actuator calls (pass --reset to start from solved base):
bp-pipeline actuate <run_dir> --calls calls.json

# 5. Evaluate objective 0-10 3D fidelity, pose_gate, and clean_gate:
bp-pipeline score <run_dir>
```

### Key Composition Rules for Multi-Tool Shaping

1.  **Straight Narrowing Before `rotation_3d`**:
    -   `clip_pattern_fold(element_name=..., template_name='fold', enable_narrow=True, narrow_n_times=2)`
        halves a flap or river straight without a zig-zag bend (`narrow_n_times=3`
        narrows to $1/3$ width). Call it on the unshaped base for 1–3 thin limb
        pairs per round **before** `rotation_3d`.
2.  **Bilateral Squash Opening (`squash_fold`)**:
    -   `squash_fold(element_name=..., open_percent=0.45)` opens symmetric
        3D kites across both mirror ridges of a flap (heads, jaws, hoods, thighs,
        palms). Apply flat squashes (`open_percent=1.0`) before `rotation_3d` or
        3D squashes (`open_percent=0.35..0.60`) on flaps with plain pleats.
3.  **Automatic 3D Overlay Replay (`flat_backbone`)**:
    -   `ShapingState` tracks a 100%-Kawasaki 2D `flat_backbone` alongside 3D
        partial-angle overlays (`open_flap`, `inscribed_flap`, `fold_spine_3d`),
        replaying 3D relief creases automatically after subsequent 2D folds and
        verifying `vertex_error <= 1.0%` in the deterministic 3D folder.
4.  **Detailed Reference Manuals**:
    -   Full 28-tool parameter table, preconditions, and recipes:
        [references/tools.md](references/tools.md)
    -   Objective 3D fidelity components and pose-search architecture:
        [references/pose.md](references/pose.md)
