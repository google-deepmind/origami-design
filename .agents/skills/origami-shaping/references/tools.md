# Shaping Actuator Toolbox Reference

All 21 tools returned by `bp.list_shaping_tools()` can be called via
`bp.apply_shaping_tool(state, name, arguments)`,
`bp.apply_shaping_tools(state, calls)`, `bp-pipeline actuate`, or as keyword
functions on `bp_pipeline.shaping.toolbox`.

Every tool treats `bp.ShapingState` as immutable, checks foldability before
accepting a pattern, and automatically replays active 3D partial-fold overlays
(`state.flat_backbone`) when applying 2D backbone folds.

## 1. Core 2D and 3D Fold Primitives

### `simple_fold`

2D zig-zag simple fold, tree-angle heuristic fold, or width-halving narrow.

- `element_name: str` — Target flap or river.
- `angle: float | None = None` — Turn angle in degrees; `None` derives the
  heuristic angle from `state.tree` (or `45.0` when `narrow=True`).
- `depth: float = 0.0` — Fractional offset from the base in `[0, 1)`.
- `narrow: bool = False` — Halve flap/river width (falls back to clip-pattern
  narrowing on rivers and interior flaps, and records grid doubling).
- `fold_percent: float = 1.0` — Dihedral fraction in `(0, 1]`.
- `fold_order: str | None = None` — `'MV'`, `'VM'`, or `None` (heuristic).
- `verify_fold: bool = False` — Require clean 3D folding when `True`.

### `crimp_fold`

Double-crease crimp fold switching cleanly between 2D zig-zag and 3D layer cuts.

- `element_name: str`, `angle: float`, `depth: float = 0.4`
- `gap: float | None = None` — Separation between 3D crimp cut lines (routes
  `mode='auto'` to `'3d'` with a note; raises `ValueError` if `mode='2d'`).
- `fold_percent: float = 1.0`
- `mode: str = 'auto'` — `'auto'` (tries `'2d'` then `'3d'`), `'2d'`, or `'3d'`.
- `verify_fold: bool = True`

### `reverse_fold`

Inside/outside reverse fold supporting 2D zig-zag and local 3D folded-form cuts.

- `element_name: str`, `angle: float`, `depth: float | None = None` — Defaults
  to `0.0` in `'2d'` mode and `0.5` in `'3d'` mode.
- `fold_type: str = 'valley'` — `'valley'` or `'mountain'` (2D mode).
- `orientation: str = 'horizontal'` — `'horizontal'` or `'vertical'` (2D mode).
- `fold_percent: float = 1.0`
- `mode: str = 'auto'` — `'auto'`, `'2d'`, or `'3d'`.
- `verify_fold: bool = True`

### `geometric_simple_fold`

Cuts across the 2D flat-folded state along explicit normalized segments.

- `folds: list[dict]` — Items with `start_x`, `start_y`, `end_x`, `end_y`,
  `fold_type` (`'mountain'` | `'valley'`), and optional `element_names:
  list[str]` restricting affected layers.
- `sequential: bool = False` — Recompute folded graph between lines when `True`.
- `verify_fold: bool = False`

## 2. Clip Patterns, Tapering, Fanning, and Relief

### `clip_pattern_fold`

Layer-aware template creases for narrowing, turning, bending, and tucking.

- `element_name: str`
- `template_name: str = 'fold'` — One of `'fold'`, `'turn_and_narrow'`,
  `'bend_when_turn_and_narrow'`, or `'pleat_to_hide'`.
- `step: float = 0.0` — Grid-unit offset into the element.
- `bend_angle: float = 0.0`, `rotate_angle: float = 0.0` — Degrees.
- `enable_narrow: bool = False`, `narrow_n_times: int = 2`,
  `narrow_symmetric: bool | None = None`
- `pleat_distance: float = 0.5`, `length: float | None = None` — Used by
  `'pleat_to_hide'` to shorten or sink tips.
- `global_refine: bool = False`, `verify_fold: bool = True`

### `narrow_central_flap`

Symmetrically tapers both halves of a flap spanning the `x = 0.5` axis (heads,
beaks, snouts, central horns, tails) with mirror-polarity verification.

- `element_name: str`
- `tip_width: float = 0.25` — Target tip fraction in `[0, 1)`.
- `start_offset: float = 0.0` — Base fraction where tapering starts (`[0, 0.9)`).
- `fold_type: str = 'auto'` — `'auto'`, `'mountain'`, or `'valley'`.
- `require_symmetric: bool = True`, `verify_fold: bool = True`

### `fan_out_fold`

Flares accordion border pleats of a flap into a broad fan, wing, shell, or fin.

- `element_name: str`
- `move_delta: float = 1.0` — Signed grid shift along the border.
- `orientation: str | None = None` — `'horizontal'`, `'vertical'`, or `None`
  (auto-selects valid border orientation and widens widest first).
- `fold_type: str | None = None` — `'mountain'`, `'valley'`, or `None`.
- `spread: bool = False` — Graded radial fan spread across pleats.
- `open_percent: float = 1.0` — Partial opening in `(0, 1]` for 3D relief.
- `verify_fold: bool = True`

### `squash_fold`

Squashes one diagonal ridge (or a symmetric mirror-ridge pair) into a wide kite
plate, ideal for ears, fins, petals, hoods, and wings.

- `element_name: str`
- `ridge_index: int | None = None` — Explicit ridge index or `None` (auto-pairs
  mirror ridges, then tries longest ridges).
- `starting_position_index: int | None = None`,
  `ending_position_index: int | None = None`
- `open_percent: float = 0.45` — `1.0` stays flat; `0.35` to `0.55` spreads 3D.
- `verify_fold: bool = True`

### `inscribed_flap`

Adds concentric mountain/valley stepped rectangle pleats inside a flap or river
for layered shields, shells, or torso volume.

- `element_name: str`
- `depths: int | list[int] = (1,)` — Grid-unit inset depth(s).
- `fold_order: str = 'MV'` — `'MV'` or `'VM'`.
- `fold_percent: float = 0.5`, `verify_fold: bool = True`

### `open_flap`

Partially opens existing interior pleats or diagonal ridges of a flap or river
to inflate flat layers into a 3D tube, belly, or curved plate.

- `element_name: str` — Target flap or river.
- `fold_percent: float = 0.35` — Target dihedral fraction in `(0, 1)`.
- `ridge_only: bool = False` — Restrict opening to diagonal ridges on flaps.
- `verify_fold: bool = True`

### `fold_spine`

Partially folds the sagittal `x = 0.5` spine to tent the body or arch the back.

- `fold_percent: float = 0.65` — Dihedral fraction in `(0, 1]`.
- `element_name: str | None = None` — Specific midline element or `None`
  (auto-detects proximal midline river/root span).
- `fold_type: str = 'mountain'` — `'mountain'` or `'valley'`.
- `verify_fold: bool = True`

## 3. 3D Direction, Articulation, and Multi-Element Tools

### `rotation_3d`

Simultaneously rotates multiple elements toward figure-frame `(az, el)` angles,
automatically retrying order conflicts and optional combined narrowing.

- `rotations: list[dict]` — Items `{element_name, az?, el?, depth=0.0,
  narrow=False}`.
- `require_all: bool = False` — Raise if any requested element was skipped.
- `allow_narrowed: bool = False` — Allow cutting already-narrowed flaps.
- `include_unlisted: bool = False` — Also shape unlisted tree sticks.

### `bend_to_direction`

Closed-loop 3D crimp/reverse search steering one flap to a target `(az, el)`
measured in the root stick's frame.

- `element_name: str`
- `az: float | None = None`, `el: float | None = None` — Defaults to tree target.
- `depth: float = 0.45`, `measure: str | None = None`
- `tolerance: float = 12.0`, `max_candidates: int = 12`
- `chosen: dict | None = None` — Replay token `{tool, angle, depth}`.

### `thin_limbs_3d`

Narrows width-1 limbs and jointly searches symmetric 3D hip/shoulder and
knee/elbow angles in batched folder passes.

- `limbs: list[dict]` — Items `{proximal, distal=None}` (plus optional replay
  fields `narrow_angle`, `bend`, `knee`).

### `curve_flap`

Applies graduated multi-station bends along a tail, neck, trunk, or horn.

- `element_name: str`
- `n_segments: int = 3`, `total_bend_deg: float = 90.0`
- `direction: str = 'up'` — `'up'`, `'down'`, `'left'`, `'right'`, `'forward'`,
  `'backward'`, `'positive'`, or `'negative'`.
- `method: str = 'auto'` — `'auto'`, `'crimp'`, `'reverse'`, or `'tree_edit'`.
- `fold_percent: float = 1.0`, `verify_fold: bool = True`

### `spread_fork`

Separates two neighboring or forked flaps (ears, horns, pincers, antennae) to
maximize 3D tip distance.

- `element_a: str`, `element_b: str`
- `spread_angle: float = 35.0`, `depth: float = 0.25`
- `fold_percent: float = 1.0`, `out_of_plane: bool = False`
- `verify_fold: bool = True`

### `radial_fan`

Distributes two or more rays (petals, tentacles, tail feathers, spider legs)
evenly across a 3D azimuth arc with optional dome elevation and wide-ray flare.

- `elements: list[str]` (at least 2 names)
- `arc_deg: float | None = None`, `center_az: float | None = None`
- `base_el: float | None = None`, `dome_el: float = 0.0`
- `fan_wide: bool = True`, `open_percent: float = 0.65`,
  `verify_fold: bool = True`

### `apply_symmetric`

Applies any single-element actuator to a left/right mirror pair, automatically
resolving the partner, mirroring signed parameters, and verifying symmetry.

- `tool: str` — Inner single-element tool name (e.g. `'crimp_fold'`,
  `'squash_fold'`, `'open_flap'`, `'bend_to_direction'`, `'simple_fold'`).
- `element_name: str`, `partner_name: str | None = None`
- `arguments: dict | None = None` — Keyword arguments passed to `tool`.
- `min_symmetry: float | None = None` — Minimum required pair symmetry in
  `[0, 1]`.
- `left_arguments: dict | None = None`, `right_arguments: dict | None = None`

### `tree_edit`

Subdivides flap sticks or adjusts angles and widths while preserving packing
equivalence, then shapes the updated tree onto the current base.

- `subdivisions: dict[str, list[float]] | None = None` — Stick name to segment
  lengths summing to the stick's original length.
- `azimuths: dict[str, float] | None = None`,
  `elevations: dict[str, float] | None = None`,
  `widths: dict[str, float] | None = None`
- `tree: dict | None = None`, `stick_name_mapping: dict[str, str] | None = None`
- `narrow: bool = False`, `only_edited: bool = False`,
  `verify_fold: bool = False`

### `level_shifter`

Adjusts the uniaxial elevation of a width-2 flap or river packed with
sacrificial level-shifter pockets (`bp.PackingConfig(use_level_shifter=True)` /
`--use-level-shifter`), splicing `2x2` 45-degree diamonds so shifted axial
creases land on integer elevations (`+2` for `'up'`, `-1` for `'down'`) while
preserving global flat-foldability. Solve with `record_stages=True` when
re-positioning an already-widened base, and apply before adding 2D shaping
creases.

- `element_name: str` — Target width-2 flap or river (mirrored pairs shift
  symmetrically).
- `position_percent: float = 0.5` — Relative slot position in `[0, 1]`.
- `direction: str = 'auto'` — `'auto'`, `'up'`, or `'down'`.
- `verify_fold: bool = True`

### `shape_flap_template`

Applies a single deterministic sculpting operation to auxiliary
`shaping_flap: true` pockets packed by `TailTemplate` or `KneeTemplate` without
posing or bending non-shaping sticks (`shaping_flap` sticks carry no pose target
and no fidelity term).

- `element_name: str = ''` — Optional template stick name (plus its mirror
  partner when symmetric); naming a `TailTemplate` connecting river or
  `tip_flap` selects the whole tail, and naming a `KneeTemplate` normal flap or
  river selects that knee (plus its mirror partner). Empty targets all template
  `shaping_flap` sticks.
- `style: str = 'armor'` — `'armor'`
  (`open_flap(fold_percent=open_percent, ridge_only=False)`), `'taper'`
  (`narrow_central_flap` on central tail blades spanning `x = 0.5`
  (`contains_central_crease`) with `length >= 2` when `central_crease_removed`
  is True, or width-halving `simple_fold` on length-1, off-axis, or split
  blades), or `'tuck'` (width-halving `simple_fold`).
- `open_percent: float = 0.45` — `fold_percent` in `(0, 1)` forwarded to
  `open_flap` for `'armor'` (smaller values open the flap wider).
- `verify_fold: bool = True` — Require clean 3D folding
  (`vertex_error <= MAX_RENDERABLE_VERTEX_ERROR`, `1.0%`).
