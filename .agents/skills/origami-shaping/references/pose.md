# Pose Recipe & Objective 3-D Fidelity (`bp.pose`, `bp.evaluate_fidelity`)

Reference for automated pose planning (`bp_pipeline.shaping.recipes`) and
deterministic 3-D geometry evaluation (`bp_pipeline.shaping.fidelity`).

--------------------------------------------------------------------------------

## 1. Objective 3-D Fidelity (`fidelity.py`)

`fidelity.py` evaluates how closely the folded 3-D model (`folder.solve_fold`)
matches the target stick figure (`tree.json`) without relying on camera
viewpoints or vision-model judgement.

### Public API

*   `fidelity.score_pattern(tree, packing, pattern, shaped_elements=()) -> dict`
    Scores an in-memory `(tree, packing, pattern)` triple using the
    deterministic folder (`folder.solve_fold(pattern,
    face_triangulation=False)`).
*   `bp.evaluate_fidelity(target) -> dict` Scores a `SolvedCreasePattern`,
    `ShapedCreasePattern`, or `PipelineResult` directly.
*   `fidelity.render_rotation(report) -> list[list[float]] | None` Returns a
    proper 3×3 rotation matrix (`det = +1`) that transforms folded mesh vertices
    into the upright stick-figure frame (`x` right, `y` forward, `z` up) before
    rendering the 7 overview views.

### Exact Box-Pleating Joint Localization (`paper_layout`)

In box-pleating, each flap corresponds to an L-infinity square contour of radius
$r$ around its packed tip center $(x_0, y_0)$. Unlike `shape_similarity.py`
(which maps stick endpoints to rectangle corners, distorting internal chain
joints), `fidelity.paper_layout` maps every stick endpoint at axial distance $d$
from its chain tip to the exact inset contour: $$\text{fraction} = 1 -
\frac{d}{\text{total\_length}}, \qquad \text{inset} = r \cdot \text{fraction}.$$
A point on the crease pattern belongs to that joint's locus when its Chebyshev
distance from $(x_0, y_0)$ equals $\text{inset}$ inside the element's bounding
box. Averaging the 3-D coordinates of FOLD faces incident to each joint's paper
locus recovers exact 3-D joint coordinates: on unshaped bases (`spider_simple`,
`scorpion_p3`, `scorpion_p3_lc`, `BullMoose`, `moose_d1`), every stick's folded
3-D length matches its stick-figure length to within $10^{-2}$ (`ratio = 1.00`).

### Subscores (`[0, 1]`) and Composite Score (`[1, 10]`)

1.  **Cauchy-Reweighted Similarity Alignment**: Finds the proper similarity
    transform ($s, R \in SO(3), t$, `allow_reflection=False`) aligning folded
    3-D joints to stick-figure 3-D joints. Five IRLS steps with Cauchy weights
    prevent one unshaped limb from dragging the body frame away from the posed
    limbs.
2.  **`direction`** (weight `0.24`): Length-weighted mean of $\max(0, \cos
    \Delta\theta_i)$ across all sticks after alignment.
3.  **`position`** (weight `0.16`): $\exp(-1.5 \cdot \text{NRMSE})$ of aligned
    joint coordinates normalized by the stick-figure bounding-box diagonal.
4.  **`symmetry`** (weight `0.14`): Mirror consistency of matched `(L, R)` stick
    pairs across the sagittal plane ($x \mapsto -x$).
5.  **`separation`** (weight `0.14`): Combines:
    *   **L/R pair separation**: fraction of symmetric `(L, R)` pairs whose
        folded distal separation is at least 25% of their target 3-D separation
        (detects `stacked_pairs` when mirrored legs fold into the exact same
        plane).
    *   **Tip de-clumping**: penalizes distinct flap tips that remain within
        $15\%$ of the target mean tip spacing (`clumped_tips`).
6.  **`extent`** (weight `0.16`): Geometric mean of 3-D `depth` and `width`
    extent ratios relative to the target stick figure. An unshaped flat base has
    `depth = 0.0` (`extent = 0`), cleanly separating flat pancake folds from
    genuine 3-D volume.
7.  **`coverage`** (weight `0.10`): Length-weighted fraction of non-root sticks
    touched by shaping tools (`60%`) plus the fraction of multi-segment chains
    that are non-collinear in 3-D (`40%`).
8.  **`length_preservation`** (weight `0.06`): Penalizes sticks whose normalized
    folded length collapses below their target length.

### Upright Camera Frame (`render_rotation`)

`folder.solve_fold` fixes FOLD face 0 at the identity matrix. Whenever a 3-D
fold touches or separates face 0 from the body (for example when face 0 lies on
a hindleg in `moose_20261023_1` or `BullMoose`), the native render rolls onto
its side in all 7 camera views.

`fidelity.render_rotation(report)` returns a proper rotation matrix $R \in
SO(3)$:

*   **`render_frame = "stick_figure"`**: uses the Cauchy-reweighted Umeyama
    rotation when folded joints span at least two independent directions
    ($\sigma_2 / \sigma_1 > 0.08$).
*   **`render_frame = "root"`**: when the folded model is nearly collinear (such
    as an unshaped base where all flaps lie along the $y$-axis), aligns the root
    stick direction to its target direction and maps the root face normal to
    $+x$ so the profile view (`view_2`) shows the full silhouette.

--------------------------------------------------------------------------------

## 2. Search-Based Pose Recipe (`bp-pipeline pose <run_dir>`, `recipes.py`)

`bp-pipeline pose <run_dir>` runs `recipes.search_pose(state, time_budget_s=120.0)` by default
(pass `--no-search` for the legacy greedy `recipes.pose` fallback and
`--time-budget <seconds>` to adjust the wall-time budget) and writes
`pose_report.json` (`mode`, `base_fidelity`, `fidelity`, `fidelity_trajectory`,
`candidates_tried`, `candidates_kept`, `groups`, `choices`, `seconds`,
`budget_exhausted`, `elements`).

`recipes.pose_plan(state)` inspects the element table (`shaping.list_elements`)
and stick-figure connectivity (`tree.json`) to assign each element a structural
role and ordered candidate tool calls:

Role              | Condition                                                                     | Candidate Tools
:---------------- | :---------------------------------------------------------------------------- | :--------------
`ROLE_WIDE_TIP`   | Terminal flap (`T`/`F`), `width >= 2`, `length >= 2`                          | `fan_out_fold_heuristic(±1)` → `fan_out_fold(horizontal/vertical)` → `squash_fold(ridge_index=0)` → `simple_fold(fold_percent=0.5)`
`ROLE_SEGMENT`    | Walk of $\ge 2$ non-root sticks (`length >= 2`) with $\ge 15^\circ$ bend      | Whole-chain `rotation_3d` at stick-figure `(el, az)` → fallback per-segment `crimp_fold` / `reverse_fold`
`ROLE_LIMB`       | Non-chain flap (`T`/`F`/`S`), `width > 1`, `length >= 2`, bend $\ge 15^\circ$ | `simple_fold(narrow=False)` → `crimp_fold(depth=0.5)` → `reverse_fold`
`ROLE_THIN_LIMB`  | Non-chain flap (`T`/`F`/`S`), `width <= 1`, `length >= 2`                     | `simple_fold(narrow=True)` → `thin_limbs_3d` (narrow + 3-D knee/bend) → `rotation_3d`
`ROLE_TIP_DOWN`   | Terminal flap (`T`/`F`), `length >= 2`, downward `el <= -15`                  | `reverse_fold` → `simple_fold(fold_percent=0.5)`
`ROLE_RIVER`      | Non-root river (`R`)                                                          | Grouped into `body` (`rotation_3d` on rivers and `shape_angle_width(only_listed=True)`)
`ROLE_SKIP`       | Root stick or `length < 2`                                                    | None (included in chain `rotation_3d` when leading to a wide tip)

`recipes.pose_groups(state, plans)` groups elements in priority order:
1. **Axis chains (`kind='chain'`)**: tails, necks, and asymmetric claws (`rot3d`, `rot3d+squash`, `rot3d_all`, `legacy`).
2. **Coupled wide L/R pairs (`kind='pair'`, `width >= 1`)**: pedipalps/chelae and antler beams/palms (`rot3d` + `fan_out_fold_heuristic`/`fan_out_fold`, `rot3d+squash`, `rot3d_all`, `legacy`).
3. **Coupled thin L/R pairs (`kind='pair'`, `width < 1`)**: walking legs and quadruped legs (`rot3d`, `narrow`, `thin_limbs_3d`).
4. **Single flaps (`kind='single'`)**: unpaired flaps (`head`, `bell`, `tail`).
5. **Body rivers (`kind='body'`)**: river arch & volume candidates (`rot3d_body`, `saw_listed`).

Each group candidate is validated with the deterministic folder (`fidelity_evaluator`) and kept only if it strictly improves fidelity.

--------------------------------------------------------------------------------

## 3. Post-Pose Refinement & Species Guidelines (Scorpion, Moose, Fiddler Crab)

After running the default `bp-pipeline pose <run_dir>` (`recipes.search_pose`), inspect
`pose_report.json` (`choices`, `unshaped`, `fidelity_trajectory`) and refine any
remaining elements in manual batches guided by `bp-pipeline score <run_dir>`:

### Core Structural Rules

1.  **Coupled Left/Right Pairs (`pair:<stem>`)**:
    *   When `central_fold_removed=False`, applying the same 2-D `simple_fold`
        sign to mirrored flaps (`leg_L1`, `leg_R1`) folds both flaps into the
        exact same 3-D plane (`stacked_pairs > 0`, 4 visible legs instead of 8).
    *   Always use **opposite signs** (`+angle` on L, `-angle` on R) when
        `central_fold_removed=False`, and **same signs** when
        `central_fold_removed=True`.
    *   To lift or splay legs out of the sagittal plane in 3-D, include
        symmetric `(L, R)` pairs in the single `rotation_3d` call with opposite
        `az` signs before narrowing distal segments.
2.  **Multi-Segment Chains (`chain:<root>`)**:
    *   Compute inter-segment bend from the 3-D unit-vector angle
        $\arccos(\hat{u}_{\text{parent}} \cdot \hat{u}_{\text{child}})$, not
        from raw $|\text{el} - \text{parent\_el}|$. At the apex of a scorpion
        tail arch (`tail_3 -> tail_4`), azimuth flips $180^\circ \to 0^\circ$
        while elevation stays near $+70^\circ$: raw $\Delta\text{el}$ is only
        $-5^\circ$, but the true 3-D turn is $35^\circ$.
    *   Do not follow proximal limb narrowing (`simple_fold(narrow=True)`) with
        a 2-D `reverse_fold` on the terminal segment (`leg_R4_2`): either
        flat-foldability rejects it or it inscribes a full-width pleat that
        flattens the body. Instead, pose multi-segment limbs via `rotation_3d`
        on the chain or `simple_fold(narrow=True, fold_percent=0.55)` on the
        longest segment.
    *   When a chain has a `length == 1` proximal segment (`leg_L1_1`), still
        traverse through it so its `length >= 2` distal segment (`leg_L1_2`) is
        grouped and shaped.

### Species-Specific Pose Recipes

*   **Scorpion**:
    *   **Tail (`tail_1..tail_5`, `stinger`)**: Include all segments with
        `length >= 2` in a single `rotation_3d` call with progressive `el`
        ($35^\circ \to 75^\circ \to 85^\circ \to 70^\circ \to 25^\circ \to
        -40^\circ$), or alternate `crimp_fold` on `tail_1`, `tail_3`, `tail_5`
        (`depth=0.4`, `angle=45..60`) plus `reverse_fold` on `stinger`.
    *   **Pedipalps (`arm_L/R`, `pincer_L1/L2`, `pincer_R1/R2`)**: Bend arms
        forward with opposite-sign `simple_fold` (`±45°`, `fold_percent=0.6`);
        spread width-2 pincers with `fan_out_fold_heuristic(±1)` or
        opposite-sign `simple_fold` (`±35°`, `fold_percent=0.5`).
    *   **Walking legs (`leg_L1..L4`, `leg_R1..R4`)**: Use `thin_limbs_3d` with
        per-pair `az`/`el` overrides (`±35°`, `±70°`, `±110°`, `±145°`) or
        narrow longest segments with `simple_fold(narrow=True)` and opposite L/R
        signs (`central_fold_removed=False`).
    *   **Body volume (`body_front`, `body_rear` / `prosoma`, `mesosoma`)**:
        Arch the river spine via `rotation_3d` on both body rivers (`d_el=+25°`
        or `-35°`) or `shape_angle_width(updates=[...], only_listed=True)`.
*   **Moose**:
    *   **Palmate antlers (`antler_beam_L/R`, `antler_palm_L/R`,
        `antler_tine_*`)**: Spread wide palm flaps (`d >= 2`) with
        `fan_out_fold_heuristic(±1)` or `simple_fold(narrow=False,
        fold_percent=0.45)`; splay tines at staggered angles (`±25°`, `±45°`,
        `±65°`) without narrowing the palm.
    *   **Long legs with knees/hocks (`foreleg_upper/lower`,
        `hindleg_upper/lower`)**: Apply `thin_limbs_3d` (narrow + 3-D bend/knee)
        or `rotation_3d` across upper/lower leg chains (`el=-75..-85`,
        `az=±15..25`).
    *   **Neck, head, snout, bell (`neck`, `head`, `snout`, `bell`)**: Pose
        `neck` and `head` in `rotation_3d` (`el=40`, `el=-15`), `reverse_fold`
        on `snout` (`-45°`), and `simple_fold(narrow=True)` on `bell`; arch
        torso/neck rivers with `shape_angle_width` (`d_el=-35°`) for shoulder
        hump volume.
*   **Fiddler Crab**:
    *   **Asymmetric claws (`major_claw` vs. `minor_claw`)**: Keep `major_claw`
        (`d >= 2`) wide with `fan_out_fold_heuristic(±1)` or
        `simple_fold(narrow=False, angle=40, fold_percent=0.45)`; shape
        `minor_claw` (`d <= 1`) with `simple_fold(narrow=False, angle=-45,
        fold_percent=0.6)`.
    *   **Eye stalks (`eye_L`, `eye_R`)**: Lift vertically with `rotation_3d`
        (`el=80, az=±15`) or opposite-sign `simple_fold(narrow=True, angle=±75,
        fold_percent=0.5)`.
    *   **Walking legs (`walk_L1..L4`, `walk_R1..R4`)**: Use `thin_limbs_3d` with
        radial `az` overrides (`±40°`, `±70°`, `±110°`, `±140°`) or splay
        laterally with opposite L/R `simple_fold(narrow=True)` signs.
    *   **Carapace (`body` / `carapace` rivers)**: Arch carapace rivers with
        `shape_angle_width` (`d_el=-25°`) or `rotation_3d`; never narrow rivers.
