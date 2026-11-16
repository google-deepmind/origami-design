# Shaping tool catalogue

Reference for the 28 shaping actuators in `bp_pipeline.shaping.toolbox`
(exposed via `bp.ShapingState`, `bp.apply_shaping_tool`,
`bp.apply_shaping_tools`, and `bp-pipeline actuate`).

All tools have the signature `tool(state, **arguments) -> new_state` and raise
`ValueError` with an explanation when the operation is impossible or breaks
local flat-foldability. `bp.apply_shaping_tool(state, name, arguments)` never
raises; it returns `(state, ToolResult)` and records the call in
`state.history`. The workbench `bp-pipeline actuate <run_dir> --calls calls.json` applies a
JSON list `[{"name": ..., "arguments": {...}}, ...]`.

## Reading the element table (`bp-pipeline tools <run_dir>`)

```
central_fold_removed=False
tail_1: hips->t1 r=2 θ=180 ϕ=30 d=2.0 t=S
```

*   `r` length, `θ` azimuth (rotation about z in the xy plane), `ϕ` elevation,
    `d` width, `t` type: **R** river, **T** flap with triangular tip, **F** flap
    with rectangular tip, **S** sub-flap (a stick inside a merged chain, no
    tip).
*   Elements are listed breadth-first from the root; **do not shape the root
    stick** (`flap_orientation.root_stick`).
*   `flap_orientation.orientations` tells whether a flap points UP (θ=0) or DOWN
    (θ=180) in the folded, unshaped base. Shaping angles are relative to that.
*   `central_fold_removed`: if **True**, θ should have the **same sign** for the
    left/right copies of a symmetric element; if **False**, **opposite signs**.
*   Do not shape elements of length 1. Use `depth: 0` as the default. Narrowing
    (`narrow`, `enable_narrow`) is for thin appendages (insect legs, antennae,
    elements with `d < 1`); never narrow rivers (type R).

## Angles: two different kinds (read this first)

The angle investigation found that most wrong limb orientations came from mixing
these up:

*   **Absolute directions** (`rotation_3d` `az`/`el`, `shape_angle_width`,
    `bend_to_direction`, `recipes.figure_pose`): the direction the element
    should point, in the stick figure's frame (x forward at az=0, y left, z up).
    These use the same numbers as the element table's θ/ϕ: left limbs have az >
    0, right limbs az < 0, and legs pointing down have el < 0. They are measured
    from the stick figure's root stick (`ToolConfig.use_designer_root_3d`,
    default on). Never pass parent-relative angles here.
*   **In-plane turns** (`simple_fold`, `crimp_fold`, `reverse_fold`,
    `crimp_fold_3d`, `reverse_fold_3d`, `angle`): a turn within the flap's
    current folded plane. That plane is usually oblique. The sign's direction
    depends on the flap's layer, and the realized turn is smaller than requested
    (median 14–21° for 45°). On mirrored pairs, the 3-D folds need **opposite**
    signs and `simple_fold` mostly the **same** sign, but don't rely on that
    rule. Don't guess signs: use `bend_to_direction` (or `apply_symmetric`),
    which measures the folded result.
*   Always check with numbers: `fidelity_report.json` → `pose_gate`,
    `direction_error_deg`, `stray_flaps`; `recipes.realized_directions(state)`
    gives each stick's target and realized az/el.
*   **Default posing path:** `bp-pipeline pose <run_dir>` now starts with
    `recipes.figure_pose` (all elements at their stick-figure angles, with
    measured feedback). Then narrow and refine with `bend_to_direction`.

## Tools

Tool                     | Arguments                                                                                                                                                                                                                                                                                                                                                                                                                                      | What it does / when to use it
:----------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------
`simple_fold`            | `element_name`, `angle` ∈ [-180,-8]∪[8,180](deg; sign = zig-zag phase/side; (-8,8) is clipped to ±8), `depth` ∈ [0,1) (fraction of the element where the fold starts, 0 = base), `narrow` (halve width; flaps only), `fold_percent` ∈ [0,1] (1 = fully folded 180°; smaller leaves the fold open for a 3-D effect; start from `1 -                                                                                                             | ϕ
`simple_fold_heuristic`  | `element_name_list` (empty = all)                                                                                                                                                                                                                                                                                                                                                                                                              | Angles computed from the tree. **Use before any other tool**; cannot narrow. May add nothing when angles are below the small-angle threshold.
`crimp_fold`             | `element_name`, `angle` ∈ (-180,-8]∪[8,180) (180 not allowed), `depth` ∈ (0,1)                                                                                                                                                                                                                                                                                                                                                                 | Compound fold that changes the angle **and shortens** the flap. Knees, elbows, necks (herons, swans, dragons), insect tails/abdomens, scorpion tail segments. Tries both mountain/valley orders; when neither is flat-foldable (edge/corner flaps, flaps next to rivers: ~30% of gallery flaps) it applies `crimp_fold_3d` at the same angle/depth and says so in the message.
`reverse_fold`           | `element_name`, `angle` ∈ [-180,180], `orientation` ∈ {horizontal, vertical}, `fold_type` ∈ {valley, mountain}                                                                                                                                                                                                                                                                                                                                 | Inside/outside reverse fold. Heads, beaks, tails, stingers, horns, ears, limb ends (works on children of narrowed flaps using the element's doubled grid). Tries the other orientation and both fold assignments as fallbacks, then `reverse_fold_3d` (depth 0.5) when no zig-zag assignment is flat-foldable (the zig-zag works on only ~20% of gallery flaps; the message says which was applied).
`clip_pattern_fold`      | `element_name` (flap **or river**), `step` ≥ 0 (grid units along the element before the pattern starts; < element length; `step < 0.5` auto-falls back to `0.5` if the proximal joint collides), `template_name` ∈ {fold (default), turn_and_narrow, bend_when_turn_and_narrow, pleat_to_hide}, `bend_angle` ∈ [-180,180], `rotate_angle` ∈ [-180,180](fold template), `enable_narrow`, `narrow_n_times` ≥ 2 (width → 1/n), `narrow_symmetric` | Draws a template on every layer of the element with the crease-pattern drawing machine. **The straight narrowing tool**: `clip_pattern_fold(element_name, template_name='fold', enable_narrow=True, narrow_n_times=2)` narrows **without a bend** (no zig-zag kink) and is the only tool that narrows rivers (neck, thorax) and narrows to 1/n. Measured on the benchmark bases (88 elements): works on 94% of elements, folded width 0.49–0.56× for n=2 and 0.33–0.5× for n=3 — vs. 0.9× for `simple_fold_narrow(±15–20°)` as used in the benchmark champions. `step=1` narrows only beyond the first grid unit (the distal part; no effect on length-2 flaps). Call it **on the base, before `rotation_3d`** (narrowing a posed flap fails with `ff:` or narrows only to 0.8×), **a few elements at a time**: narrowing all 14 thin elements of the Pegasus before one `rotation_3d` left 8 strays (fidelity 4.3); one or two per limb pair, re-posed from the base, kept fidelity ≥ pose-only at 0% vertex error. Costs +500–700 creases per narrowed flap (+3700 for a neck river), so later folds are slower. `fold` needs at least one of bend/rotate/narrow; `global_refine` is not accepted. Axis flaps on a central-crease-removed base still fail (`ff: 99% of vertices`): use `narrow_central_flap`.
`rotation_3d`            | `rotations`: list of `{element_name, el ∈ [-90,90], az ∈ [-180,180], depth ∈ [0,1)}`, `require_all=False`                                                                                                                                                                                                                                                                                                                                      | Rotates flaps in 3-D with the geometric folding heuristics. `az`/`el` are **absolute** directions in the stick figure's frame, measured from the stick figure's root, which is never rotated. Baseline: the stick figure's own θ/ϕ (`recipes.figure_pose` does this for every element with measured feedback); nudge angles near 0/90/180 by ~10°. Rotations don't compose across calls: each call assumes the parents sit at their stick-figure angles, so pose everything in one call from the base. Can run after narrowing *other* flaps (degenerate folded edges are skipped), but refuses already-narrowed elements themselves. Fails if 0 creases are added; when a subset of elements is skipped for flat-foldability, names them in `ToolResult.message` (`Rotated [...]; left out: ...`) or fails if `require_all=True`.
`shape_angle_width`      | `updates`: list of `{element_name, el?, az?, narrow=False, depth=0}`, `only_listed=False`                                                                                                                                                                                                                                                                                                                                                      | Rotation + width change across the tree (combined heuristics, shallow-fold strategy). Only narrows sticks explicitly listed with `narrow=True` (unlisted `d < 1` sticks are kept at width 1 so they do not trigger `ROTATE_BEND` assertions). Effective on body rivers (e.g. `el: -35` on `body_rear`) for whole-model 3-D arch and body thickness. Once per round on the unshaped base. By default every stick is shaped to its stick-figure pose; on big trees (≈30 sticks) that can give a flat-foldable pattern the folder cannot fold, so pass `only_listed=True` to shape just the listed elements (e.g. `updates=[{'element_name': 'prosoma', 'el': -35}], only_listed=True`: +182 creases, folds, body volume). For thin + 3-D limbs use `thin_limbs_3d` (narrowing through this tool asserts or breaks flat-foldability on every bench base).
`thin_limbs_3d`          | `limbs`: list of `{proximal, distal?, narrow_angle?, bend?, knee?, az?, el?, distal_az?, distal_el?}` (`distal` defaults to the next stick of the merged chain; `bend`/`knee` = `{angle, depth, fold_percent, fold_order?}` or `null`; `az`/`el` override the target direction, e.g. to fan legs wider than the stick figure)                                                                                                                  | **Thin limbs posed in 3-D** (legs, antennae). Per limb: `simple_fold(proximal, narrow_angle, narrow=True)` (thin + in-plane turn), an open `simple_fold` (`fold_percent` < 1) on the narrowed proximal segment that lifts it out of the paper plane, and an open knee fold on the distal segment. Missing steps are searched per limb by folding with the deterministic folder and measuring each segment's direction against the stick figure (`limb_direction_errors`: frame fitted on joints outside the limbs). ~30 folds per call (1–4 min); the chosen plan is in the message and replaces the arguments in the history, so replays are fast. Call it **after** posing the body/tail/other chains (`rotation_3d`, `shape_angle_width`) and **instead of** `rotation_3d` on those limbs.
`fan_out_fold`           | `element_name` (flap), `orientation` ∈ {horizontal, vertical}, `move_delta` (grid units, typically ±1, 0.5, or 0.25), `open_percent=0.5` ∈ (0, 1], `fold_type?` ∈ {mountain, valley}                                                                                                                                                                                                                                                           | Moves border crease points to fan a flap out: wings, tail feathers, fins, pincers. Uses sequential multi-pleat fanning (`_fan_incrementally`) constrained by cooperative timeout (2.0s per pleat) and tight per-pleat solve rectangles. Conforms to the $3.0^\circ$ geometric tilt limit ($\theta \approx \arctan(\delta/L) \le 3.0^\circ$): long pleats ($L \ge 12$, e.g. wings) use $\delta = 0.5$, while short pleats ($L = 5$, e.g. tails) use adaptive scaling $\delta \in (1.0, 0.5, 0.25)$. For axis flaps spanning mirror-symmetric crease patterns, `_mirror_half` performs bilateral mirror completion across the symmetry axis (e.g. tail fanning 4 pleats symmetrically with deltas `[-1.0, -0.25, 1.0, 0.25]`). Widens wings up to **×2.45** (up to +14 creases per wing on grid-26 avian bases) and tails up to **×1.26–1.51**. Candidates with fold vertex error > 1.0% (`MAX_RENDERABLE_VERTEX_ERROR`) are rejected. Default `open_percent=0.5` leaves the fan open in 3-D.
`fan_out_fold_heuristic` | `element_name`, `move_delta`                                                                                                                                                                                                                                                                                                                                                                                                                   | Same, picking orientation/fold type automatically and growing multi-pleat subsets. Use the **same** `move_delta` sign (e.g. both `+1` or both `-1`) on mirrored left/right flaps (`chela_L`/`chela_R`), and turn fanned wings afterward with `reverse_fold_3d` rather than `rotation_3d` on the fanned flaps.
`geometric_simple_fold`  | `folds`: list of `{element_names (empty = all intersecting), start_x, start_y, end_x, end_y ∈ [0,1] in folded-form coordinates, fold_type}`; `sequential`                                                                                                                                                                                                                                                                                      | Folds along explicit cut lines in the folded form. Expert tool when the named-element tools cannot express the fold.
`squash_fold`            | `element_name` (flap), `ridge_index` (0–3), `starting_position_index`, `ending_position_index`, `open_percent=0.45`                                                                                                                                                                                                                                                                                                                            | Opens a **kite** between two pleats crossing a center-to-corner ridge of the flap (squash sink), by default on **both mirror ridges** of the flap at once (across the paper axis for axis-centered flaps, across the flap's own length axis otherwise), so one call gives a symmetric result on heads, jaws, thighs, wings, hoods. `open_percent=1.0` is a *flat* kite: the folded width does not change (1.03×, 3–4 s); the visible effect — a 3-D wedge / flare of the flap — comes from `open_percent` 0.35–0.6, which opens the kite and its continuation pleats (benchmark bases: median 1.15×, head/tail-dock 1.1–1.4×, cobra neck/hood/jaw and trike pes 2–3×; 15–20 s). Needs **plain parallel pleats** on the ridge: call it **before** narrowing the same flap (`simple_fold_narrow(fang_tip)` left the dragon snout without parallel pleats → fails) and before `rotation_3d` when possible (flat squash first, 3-D opening after, see composition order below). Width-0.5/1 flaps with ≤ 2 ridge segments and rivers cannot be squashed. The call now fails instead of returning a +2-crease no-op; the message names the ridge, segments and reason.
`tree_edit`              | `tree` (full stick figure dict), `stick_name_mapping` (edited stick → original merged stick), `narrow`                                                                                                                                                                                                                                                                                                                                         | Shapes by **re-posing a subdivided stick figure** without repacking: split sticks by adding joints (sub-lengths must sum to the original), rotate sub-sticks freely; then the heuristic shaper is applied. The packing must stay equivalent.
`level_shifter`          | `element_name` (stick width > 1), `position_percent` ∈ [0,1]                                                                                                                                                                                                                                                                                                                                                                                   | Widens part of a flap with a level shifter (requires a base packed with `PackingTaskConfig(use_level_shifter=True)` so `sacrificial_flaps` is non-empty). `bp-pipeline tools <run_dir>` lists `level_shifter_positions` per element.
`reverse_fold_3d`        | `element_name`, `angle` ∈ (-180,180] (sign picks side), `depth` ∈ [0,1](0.5 = midpoint), `fold_percent` ∈ (0,1] (1 = flat), `verify_fold=True`                                                                                                                                                                                                                                                                                                 | **Local reverse fold** cut in the folded state: distal part of the element turns by `angle`° in its own folded plane. Creases stay on the element only, works on narrowed limbs, and composes with `rotation_3d` before/after. Tries depth fallbacks (±0.15, ±0.3) and M/V variants. Use for: heads, beaks, stingers, horns, tails, knees of narrowed legs, any sharp direction change within a single flap. Like `reverse_fold` but local (no full-paper pleats) and works after narrowing/3-D rotation.
`crimp_fold_3d`          | `element_name`, `angle` ∈ (-180,180) (sign picks side), `depth` ∈ [0,1](0.4 default), `gap` (distance between cuts; default: auto), `fold_percent` ∈ (0,1] (1 = flat), `verify_fold=True`                                                                                                                                                                                                                                                      | **Local crimp** (two opposite reverse folds): direction change + shortening within the element. Creases stay on the element, works on narrowed limbs. Use for: elbows, knees, zigzag tails, scorpion stinger barbs, any bend that must also shorten the limb. Composes with `rotation_3d` / `thin_limbs_3d`.
`open_flap`              | `element_name` (flap), `fold_percent` ∈ (0,1] (0.3 default), `ridge_only=False`                                                                                                                                                                                                                                                                                                                                                                | **Primary plate opener for wide flaps (`width >= 2`)**: searches clean geometric crease subsets on the flap (axial pleats within each circle square or rectangle, border-connected pleats, and diagonal ridges), verifies each with the deterministic folder, and keeps the folding pattern that spreads the flap widest (92.3% verified pass rate on bench wide flaps with width ratio >= 1.15). Works on raw bases without pre-shaping; `ridge_only=True` opens only center-to-corner / tip diagonal ridges for a tent-like plate.
`inscribed_flap`         | `element_name` (wide flap), `depths` (list of grid depths, default `[1]`), `fold_order` (`'MV'` or `'VM'`), `fold_percent` ∈ (0,1] (default 0.5)                                                                                                                                                                                                                                                                                               | **Concentric inscribed relief for broad bodies/carapaces**: cuts concentric shrunk polygon rings into wide body/carapace flaps, stepping down z-height into layered relief without altering the outer boundary. Composes cleanly with distal tail/limb posing.
`curve_flap`             | `element_name` (proximal flap or chain), `n_segments` (default 3), `total_bend_deg` (default 90.0), `direction` (`'up'`, `'down'`, `'left'`, `'right'`), `method` (`'auto'`, `'crimp_3d'`, `'reverse_3d'`)                                                                                                                                                                                                                                     | **Smooth C-curve bending for multi-segment chains** (scorpion tails, necks, trunks): distributes `n_segments` coordinated bend stations along a linear flap chain, preserving world-space curvature parity (crimps keep sign, reverse folds alternate sign).
`pleat_to_hide`          | `element_name`, `step` ≥ 0, `pleat_distance` (default 0.5), `span` (default auto)                                                                                                                                                                                                                                                                                                                                                              | **Hides excess flap/river length** using balanced interior pleats (shortening flaps while preserving flat-foldability).
`simple_fold_narrow`     | `element_name`, `angle` (default 45.0), `depth` (default 0.0), `fold_order` (`'MV'`/`'VM'`), `fallback_clip_narrow=True`                                                                                                                                                                                                                                                                                                                       | **Automatic flap/river narrowing with fallbacks**: tries `simple_fold(..., narrow=True)` on flaps and automatically falls back to `clip_pattern_fold(..., enable_narrow=True)` on interior flaps and rivers.
`narrow_central_flap`    | `element_name`, `tip_width` ∈ [0,1) (default 0.25 = fraction of the folded width kept at the far end; 0 = pointed), `start_offset` ≥ 0 grid units into the parent (default 0; 1 also narrows the flap's base and chamfers the parent's top corners), `fold_type` (`auto`/`valley`/`mountain`)                                                                                                                                                  | **Symmetric narrowing of a flap that spans x = 0.5** (heads/snouts on the axis after `remove_central_crease=True`, where `simple_fold(narrow=True)`/`crimp_fold_3d` act on one paper half only). Two mirrored taper folds through every layer in the 2-D folded form; flat-foldable by construction, CP mirror-symmetric (`mirror_match` in the report). Apply **before** `rotation_3d`/3-D tools; does not mark the flap as narrowed, so `rotation_3d` still rotates it. Cobra head: `tip_width=0.2, start_offset=1` → head 1.45 vs hood 2.5 grid units.
`apply_symmetric`        | `tool`, `left_element`, `right_element?` (auto-resolved if empty), directional parameters (`angle`, `fold_order`, `depth`, etc.)                                                                                                                                                                                                                                                                                                               | **Bilateral 3-D symmetry coordinator**: locks parameters across left/right pairs, tests candidate pairs `(angle, -angle)` vs `(angle, angle)` in the 3-D folder, measures mirror error across the sagittal midplane, and picks the candidate maximizing bilateral symmetry (eliminating sky-legs). Clean rollback if either side fails.
`fold_spine_3d`          | `spine_fold_percent` ∈ (0,1] (0.6 default), `fold_type` (`'valley'` or `'mountain'`)                                                                                                                                                                                                                                                                                                                                                           | **Sagittal midplane body fold**: cuts an open valley/mountain spine crease along the paper symmetry axis (`x = 0.5`), curving the torso/mesosoma/carapace into a 3-D volumetric dome or book fold.
`spread_fork`            | `element_a`, `element_b`, `spread_angle` (default 35.0°), `depth` (default 0.2), `narrow=False`, `fold_percent=1.0`, `fold_percent_b=None`, `out_of_plane=False`                                                                                                                                                                                                                                                                               | **Spreads co-planar forks/pincers into open V-gap**: useful for chelae dactyl/pollex pincers, crab claws, or split brow tines/antlers. `out_of_plane=True` applies differential fold percent between jaws (0.75 vs 1.0) to create authentic 3-D vertical daylight separation between pincers/claws.
`bend_to_direction`      | `element_name`, `az?`, `el?` (absolute target; default: the stick's own θ/ϕ), `measure?` (stick to aim; default `element_name`), `tools` ⊆ {crimp_fold_3d, reverse_fold_3d, simple_fold}, `angles`, `depths`, `tolerance=15`, `max_candidates=36`, `chosen?` (replay)                                                                                                                                                                          | **Bends a flap toward an absolute direction by measurement.** It tries the in-plane bend tools with both signs, several sizes and depths, folds each candidate, and keeps the one that points `measure` closest to (az, el), with a penalty for moving other sticks off. Works on narrowed limbs, where `rotation_3d` doesn't. Use it for knees and feet after narrowing, and to fix single strays reported in `stray_flaps`. The chosen call is recorded in the history, so replays skip the search.
`radial_fan`             | `elements` (≥ 2 sibling rays sharing a joint), `arc_deg?`, `center_az?`, `base_el?`, `dome_el=0`, `fan_wide=True`, `open_percent=1.0`, `verify_fold=True`                                                                                                                                                                                                                                                                                      | Poses sibling rays along a circular arc and fans the broad ones (feathers, frill rays, tentacles). Rejects a single flap (`radial_fan requires at least 2 ray elements`): for circular wings, split each wing into feather sub-rays in the stick figure.
`open_river_3d`          | `element_name` (river), `fold_percent` ∈ (0.15, 0.95] (default 0.65), `max_vertex_error=0.35`, `verify_fold=True`                                                                                                                                                                                                                                                                                                                             | Partially opens interior parallel accordion pleats of a wide river for 3-D trunk/neck volume while leaving perimeter hinges and paper borders untouched, verifying Belcastro-Hull 3-D foldability and `vertex_error <= max_vertex_error`.

## Known Traps & Best Practices

*   **`simple_fold_narrow` with the small turns used in benchmark (±15–20°) barely
    narrows** (0.9× of the base width on the Pegasus legs/muzzle; 0.7× at best).
    For a thin limb, horn, ear, tongue, tail tip or neck use
    `clip_pattern_fold(template_name='fold', enable_narrow=True,
    narrow_n_times=2)` (0.5×) or `narrow_n_times=3` (0.33–0.5×) on the base,
    then `rotation_3d` of everything. Narrow 1–3 elements per round (both sides
    of a pair), re-pose from the base and check `stray_flaps`: wholesale
    narrowing of every thin element before or after the pose broke the Pegasus
    (fidelity 8.2 → 4.3 / 5.5). `fidelity.measure` can report width 0.0 for a
    clip-narrowed section (its contour points fail to locate on the finely split
    faces): judge the render, not that number.
*   **Squash needs plain pleats.** A flat squash (`open_percent=1.0`) does not
    change the folded width; the wedge/flare comes from `open_percent` 0.35–0.6.
    Squash a flap before narrowing it and before the 3-D tools touch its ridges;
    after `simple_fold_narrow` on the same chain (dragon snout → fang_tip) or on
    posed thighs it fails with "No parallel pleats" / "Could not construct
    point". It is bilateral by default (both mirror ridges of the flap),
    deterministic, and fails rather than returning a no-op.

*   **`rotation_3d` drops flaps silently.** The call succeeds and only the
    message says `left out: <flap> (its fold breaks flat-foldability ...)`.
    Wings fanned flat before posing are always left out: they stay folded along
    the body, the pose gate still passes (44.8 deg < 45), and the wings read as
    crossing/stacked (`stacked_pairs: wing_L/wing_R`). Fix: pose with
    `recipes.figure_pose(state, from_current=True, bend_left_out=True)` after
    the flat fans; it keeps the fans and bends left-out flaps with
    `bend_to_direction` (Pegasus: 44.8 deg stacked -> 29 deg spread, fidelity
    6.48 -> 7.06, width kept at 2.47x).
*   **Narrowing axis flaps on a central-crease-removed base fails.** With
    `central_fold_removed=True`, every simple fold or narrowing on a flap that
    sits on the symmetry axis (heads, hoods, necks, tail chains) leaves
    degree-2 kinks on x = 0.5 (no spine crease to fold along): Cobra 5/12
    elements narrowable vs 12/12 with the crease kept. Re-solving with the
    crease kept makes narrowing work but folds every axis flap in half (Cobra
    hood 0.53x -> 0.11-0.26x its design width). Use local 3-D tools on axis
    heads (`crimp_fold_3d`, `reverse_fold_3d`) or a per-flap central-crease
    removal (not implemented yet).
*   **Narrowing one stick narrows its merged chain.** A later `rotation_3d`
    listing any stick of that chain fails as a whole unless
    `allow_narrowed=True` (after narrows with |angle| <= 20).

## Recipes (scorpion, moose, crab)

*   **Bilateral symmetry & standing quadruped pose (`recipes.pose`)**:
    `recipes.pose(state, spine_fold_percent=0.65)` derives an element plan from
    the stick figure and poses bilateral pairs together using `apply_symmetric`
    and sagittal whole-chain 3-D rotation. Evaluates candidate rotations in 2-D
    microsecond ranking (Rodrigues / spherical alignment + local
    flat-foldability), capping 3-D simulation folds to at most 2 per bilateral
    pair (testing rank #1, falling back to #2 only on fold failure,
    early-exiting on score >= 0.85). Delivers >= 90% completeness (>= 18/21
    elements posed on canonical quadruped), `sym >= 0.80`, `separation >= 0.65`,
    zero asymmetric leg spikes, and slashes latency to < 12s median (< 20s max
    across all 7 bench bases).
*   **Tail / neck / multi-segment chains**: Pose with one `rotation_3d` call
    across all chain segments (`metasoma_1..5` + `telson_stinger`, or `neck` +
    `head`). Avoid 2-D `simple_fold`/`crimp_fold` on the root chain (holds FOLD
    face 0) or full-width chain strips.
*   **Thin + 3-D legs with knees**: pose the body and the other chains first,
    then one `thin_limbs_3d` call over all legs:

    ```python
    state, result = bp.apply_shaping_tool(state, 'rotation_3d', {'rotations': [
        {'element_name': n, 'el': state.tree[n].elevation_angle,
         'az': state.tree[n].azimuth_angle}
        for n in ('metasoma_2', 'metasoma_3', 'metasoma_4', 'metasoma_5',
                  'pedipalp_hum_L', 'pedipalp_pat_L', 'chela_L',
                  'pedipalp_hum_R', 'pedipalp_pat_R', 'chela_R')]})
    state, result = bp.apply_shaping_tool(state, 'thin_limbs_3d', {'limbs': [
        {'proximal': f'leg_{side}{i}_femur'} for side in 'LR' for i in (1, 2, 3, 4)]})
    print(result.message)  # Plan: leg_L1_femur: narrow -90.0, bend {...}, knee {...} (error 3/19 deg); ...
    # Fan a leg wider than the stick figure: {'proximal': 'leg_L1_femur', 'az': 40, 'el': 20}
    # Replay / tweak one limb without searching (fields from result.arguments):
    # {'proximal': 'leg_L1_femur', 'narrow_angle': -90.0,
    #  'bend': {'angle': 180.0, 'depth': 0.5, 'fold_percent': 0.8},
    #  'knee': {'angle': 180.0, 'depth': 0.25, 'fold_percent': 0.65}}
    ```

    Manual alternatives: narrow the proximal segment
    (`simple_fold(..., narrow=True)`) and bend the distal knee (`simple_fold`
    or `reverse_fold` on the distal segment, segments of length ≥ 2); or
    rotate the proximal segments with `rotation_3d` first and narrow + bend the
    distal segments. Never `rotation_3d` a narrowed limb.
*   **Mirrored left/right 3-D legs (`rotation_3d` dropping one side)**: If
    `rotation_3d` reports `left out: <limb_L> (its fold breaks flat-foldability
    ...)` while `<limb_R>` succeeds on a base with `central_fold_removed=False`,
    re-solve the existing packing with `SolverConfig(remove_central_crease=True)` (~1 s, no re-pack needed); both sides then
    succeed symmetrically.
*   **Wide tips / broad plates (pincers, palms, claws, heads)**: Call
    `open_flap` directly on wide flaps (`width >= 2`, e.g.
    `open_flap(element_name='chela_L', fold_percent=0.3)`), which picks the
    cleanest folding pleat/ridge subset to spread the flap into a broad 3-D
    plate (`×1.47–2.38` breadth on 11/15 bench flaps). Alternatives:
    `fan_out_fold_heuristic(move_delta=+1)` (same sign on both `chela_L` and
    `chela_R` **before** `rotation_3d`), or
    `squash_fold(element_name=..., open_percent=0.3)` (`ridge_index=None` tries
    all ridges automatically).
*   **Local 3-D bends on narrowed limbs (hooves, feet, stinger tip)**: After
    `thin_limbs_3d`, add detail folds with `reverse_fold_3d` or `crimp_fold_3d`
    on individual segments — they work on already-narrowed or 3-D-rotated flaps:

    ```python
    # After thin_limbs_3d: bend the stinger tip down
    state, result = bp.apply_shaping_tool(state, 'reverse_fold_3d', {
        'element_name': 'telson_stinger', 'angle': -60, 'depth': 0.7})
    # Crimp a hoof for a tight knee+shortening:
    state, result = bp.apply_shaping_tool(state, 'crimp_fold_3d', {
        'element_name': 'leg_L1_tarsus', 'angle': 45, 'depth': 0.5})
    ```
*   **Body volume / arch**: `simple_fold` on body rivers with small
    `fold_percent` (0.3–0.5) to arch the back, `fold_spine_3d` along `x = 0.5`
    (`spine_fold_percent=0.65, fold_type='valley'`) to tent the dorsal trunk
    into a V-hull (`z_relief ≈ 0.31`), or `shape_angle_width` on a body river
    (e.g. `updates=[{'element_name': 'body_rear', 'el': -35}]`) on the base
    pattern for whole-model 3-D depth and body thickness. Never narrow rivers.
*   **Multi-tool 3-D + 2-D composition order (critical for
    `vertex_error ≤ 1.0%`)**:

    1.  **Never call partial-open `squash_fold(..., open_percent < 1.0)` or
        `fan_out_fold(..., open_percent < 1.0)` BEFORE `rotation_3d`**: partial
        `fold_percent` creases on the unposed sheet compound hinge strain up to
        `4%-10%` when `rotation_3d` cuts across adjacent chain segments.
    2.  **Flat 2-D squashes (`open_percent=1.0`) BEFORE `rotation_3d` + 3-D
        opening AFTER `rotation_3d`**: interior flat squashes
        (`squash_fold(element_name, ridge_index=r, starting_position_index=1,
        ending_position_index=N-1, open_percent=1.0)`) preserve `100%` Kawasaki
        flat-foldability (`verr = 0.00%` after `rotation_3d`). Then apply
        `squash_fold(..., open_percent=0.40)`,
        `fan_out_fold(..., open_percent=0.60)`,
        `open_flap(..., fold_percent=0.35)`, `fold_spine_3d`,
        `spread_fork(..., out_of_plane=True)`, `curve_flap(..., n_segments=3)`,
        `pleat_to_hide`, and `apply_symmetric` in any order —
        `ShapingState.flat_backbone` automatically runs 2-D tools on the 100%
        flat-foldable backbone, replays 3-D overlays, and enforces both
        Belcastro-Hull 3-D foldability and `vertex_error ≤ 1.0%`.
    3.  **Lang's 2-axis-endpoint packing rule**: In a symmetric stick figure,
        the symmetry axis `x = 0.5` meets the square boundary at only two points
        — `(0.5, 0.0)` and `(0.5, 1.0)` — so **at most two merged chains** can
        pack as unified single-apex central flaps (e.g., cranial S-chain + tail,
        or neck-head chain + tail). A third central leaf forces corner-split
        rabbit-ear packing.
    4.  **Fast deterministic replay of `apply_symmetric`**:
        `ToolResult.arguments` records `left_arguments` and `right_arguments`;
        passing those back into `apply_symmetric` skips the candidate search and
        replays in ~1 s.

## Failure modes and what to do

Message                                | Meaning                                                                                             | Try
:------------------------------------- | :-------------------------------------------------------------------------------------------------- | :--
`not flat-foldable after shaping: NN%` | The fold conflicts with existing creases at some vertices.                                          | Other sign/angle (±10°), different `depth`, the other orientation, or `bp-pipeline probe <run_dir>` to see which basic folds work on that element.
`Backtracking detected`                | The zig-zag would run back over itself (element too short for the angle/depth).                     | Smaller angle, `depth` 0, or shape the parent element instead.
`No shaping creases found`             | The tool had nothing to add (small angles below threshold, squash on a flap without usable ridges). | Larger angle; another ridge; `simple_fold` instead of heuristics.
`Element X not found`                  | Name must match the element table (unmerged names).                                                 | Copy the name from `bp-pipeline tools <run_dir>`.
`cannot be narrowed` / `not a flap`    | Tool restricted to flaps.                                                                           | Use `clip_pattern_fold` for rivers.

Reset to the base pattern with `bp-pipeline actuate <run_dir> --reset --calls ...`; every call is logged
in `shaping_history.jsonl` and `shaping_state.json` so a sequence can be
replayed or trimmed.
