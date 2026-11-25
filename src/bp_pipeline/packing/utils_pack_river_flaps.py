# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for packing flaps enclosed by rivers."""

from collections.abc import Mapping
import copy
import logging

import numpy as np

from ..stick_figures import packing_utils
from ..stick_figures import stick as stick_lib
from ..stick_figures import stick_figure as stick_figure_lib
from . import config as config_lib
from . import river as river_lib
from . import scoring
from . import tiling_solver
from . import types
from . import utils_packing_flaps
from . import utils_tracker
from .debug import debug_types


def _get_river_side_mask(
    river: river_lib.River, side_name: str, grid_size: int
) -> np.ndarray:
  """Returns a boolean mask for cells on a given side of a straight river.

  Args:
    river: The river to mask.
    side_name: The side to mask. Must be 'top', 'bottom', 'left', or 'right'.
    grid_size: The size of the grid.

  Returns:
    A boolean mask for cells on the given side of the river.
  """
  mask = np.zeros((grid_size, grid_size), dtype=bool)
  if river.orientation == river_lib.RiverOrientation.HORIZONTAL:
    if side_name == 'top':
      if river.offset > 0:
        mask[: river.offset, :] = True
    elif side_name == 'bottom':
      if river.offset + river.width < grid_size:
        mask[river.offset + river.width :, :] = True
  elif river.orientation == river_lib.RiverOrientation.VERTICAL:
    if side_name == 'left':
      if river.offset > 0:
        mask[:, : river.offset] = True
    elif side_name == 'right':
      if river.offset + river.width < grid_size:
        mask[:, river.offset + river.width :] = True
  return mask


def get_enclosed_pocket_mask(
    pocket: packing_utils.FlapPocket,
    tracker_with_all_rivers: utils_tracker.CreasePatternTracker,
    tree: stick_figure_lib.StickFigure,
    stack: list[types.RiverStackFrame],
    current_river_cand: types.RiverCandidate,
    *,
    preplaced_joint_to_side: Mapping[str, Mapping[str, str]] | None = None,
) -> np.ndarray | None:
  """Computes mask for a pocket enclosed by multiple rivers.

  Args:
    pocket: The flap pocket.
    tracker_with_all_rivers: Tracker with all rivers added.
    tree: The stick figure.
    stack: The river stack.
    current_river_cand: The current river candidate.
    preplaced_joint_to_side: Optional joint-to-side map for pre-placed rivers on
      `tracker_with_all_rivers` that do not appear in `stack`.

  Returns:
    A boolean mask for the pocket, or None if any of the boundary rivers is not
    found in the stack.
  """
  grid_size = tracker_with_all_rivers.grid_size
  combined_mask = np.ones((grid_size, grid_size), dtype=bool)

  boundary_rivers_in_tracker = pocket['boundary_rivers'].intersection(
      tracker_with_all_rivers.packed_rivers.keys()
  )
  all_boundary_rivers = boundary_rivers_in_tracker
  if current_river_cand['stick_name'] not in all_boundary_rivers:
    all_boundary_rivers.add(current_river_cand['stick_name'])

  for river_name in all_boundary_rivers:
    if river_name == current_river_cand['stick_name']:
      river = current_river_cand['river']
      joint_to_side = current_river_cand.get('joint_to_side')
    else:
      frame = next(
          (
              f
              for f in reversed(stack)
              if f.applied_candidate['stick_name'] == river_name
          ),
          None,
      )
      if frame is not None:
        river = frame.applied_candidate['river']
        joint_to_side = frame.applied_candidate.get('joint_to_side')
      elif (
          preplaced_joint_to_side is not None
          and river_name in preplaced_joint_to_side
          and river_name in tracker_with_all_rivers.packed_rivers
      ):
        river = tracker_with_all_rivers.packed_rivers[river_name]
        joint_to_side = dict(preplaced_joint_to_side[river_name])
      else:
        logging.warning(
            'Could not find boundary river %s in stack to compute mask.',
            river_name,
        )
        return None

    mask, _ = get_pocket_mask_and_cells(
        pocket,
        tree[river_name],
        river,
        grid_size,
        joint_to_side,
        tracker_with_all_rivers,
    )
    if mask is None:
      return None
    combined_mask &= mask
  return combined_mask


def _get_pocket_joint_name(
    pocket: packing_utils.FlapPocket,
    stick: stick_lib.Stick,
) -> str | None:
  """Returns river joint shared with pocket, or None if 0 or 2 joints."""
  pocket_joints = set()
  for f in pocket['flaps']:
    pocket_joints.add(f.start)
    pocket_joints.add(f.end)
  shared_with_start = stick.start in pocket_joints
  shared_with_end = stick.end in pocket_joints
  if shared_with_start and not shared_with_end:
    return stick.start
  elif shared_with_end and not shared_with_start:
    return stick.end
  return None


def _get_enclosed_pockets_to_pack(
    stick: stick_lib.Stick,
    river: types.RiverPlacementAlias,
    tracker: utils_tracker.CreasePatternTracker,
    pockets_info: list[packing_utils.FlapPocket],
    joint_to_side: dict[str, str] | None,
    side_areas: dict[str, float] | None,
) -> list[packing_utils.FlapPocket]:
  """Returns list of flap pockets enclosed by river, sorted by area.

  Args:
    stick: The river stick.
    river: The river object.
    tracker: Crease pattern tracker with river added.
    pockets_info: All flap pockets.
    joint_to_side: For straight rivers, maps joint to river side.
    side_areas: For straight rivers, maps side to area.

  Returns:
    List of flap pockets to pack.
  """
  pockets_to_pack = []
  rivers_in_tracker_now = set(tracker.packed_rivers.keys())
  for pocket in pockets_info:
    is_newly_enclosed = stick.name in pocket['boundary_rivers'] and pocket[
        'boundary_rivers'
    ].issubset(rivers_in_tracker_now)
    already_packed_by_prev_river = any(
        f.name in tracker.packed_polygons
        and not f.name.endswith(types.RIVER_SF1_SUFFIX)
        and not f.name.endswith(types.RIVER_SF2_SUFFIX)
        for f in pocket['flaps']
    )
    if is_newly_enclosed and not already_packed_by_prev_river:
      pockets_to_pack.append(pocket)

  if joint_to_side and side_areas and isinstance(river, river_lib.River):
    pockets_to_pack.sort(
        key=lambda p, side_areas=side_areas, joint_to_side=joint_to_side: packing_utils.get_pocket_area(
            p, stick, side_areas, joint_to_side
        )
    )
  return pockets_to_pack


def get_pocket_mask_and_cells(
    pocket: packing_utils.FlapPocket,
    stick: stick_lib.Stick,
    river: types.RiverPlacementAlias,
    grid_size: int,
    joint_to_side: dict[str, str] | None,
    tracker: utils_tracker.CreasePatternTracker,
) -> tuple[np.ndarray | None, list[tuple[int, int]]]:
  """Computes mask and river-adjacent cells for a pocket.

  Args:
    pocket: The flap pocket.
    stick: The river stick bordering the pocket.
    river: The river object.
    grid_size: Grid size.
    joint_to_side: For straight rivers, maps joint to river side.
    tracker: Tracker with river added.

  Returns:
    Tuple of (pocket_mask, pocket_river_adj_cells).
  """
  pocket_mask = None
  pocket_joint_name = _get_pocket_joint_name(pocket, stick)
  pocket_river_adj_cells = []

  if pocket_joint_name and joint_to_side:
    side_name = joint_to_side[pocket_joint_name]
    if isinstance(river, river_lib.River):
      pocket_mask = _get_river_side_mask(river, side_name, grid_size)
    elif isinstance(river, river_lib.LShapedRiver):
      inner_mask, outer_mask = river.get_area_masks(grid_size)
      pocket_mask = inner_mask if side_name == 'inner' else outer_mask
    elif isinstance(river, river_lib.HuggingRiver):
      hugging_mask, opposite_mask = river.get_area_masks(tracker.occupancy_grid)
      pocket_mask = hugging_mask if side_name == 'hugging' else opposite_mask
    river_id = stick.name
    if river_id in tracker.river_banks:
      for row, col in tracker.river_banks[river_id]:
        cell_state = tracker.grid[row, col]
        if (
            cell_state
            and cell_state.type
            == utils_packing_flaps.GridCellType.RIVER_ADJACENT
            and cell_state.source_id == pocket_joint_name
        ):
          pocket_river_adj_cells.append((row, col))
  return pocket_mask, pocket_river_adj_cells


def _pack_flaps_in_pocket(
    pocket: packing_utils.FlapPocket,
    stick: stick_lib.Stick,
    river: types.RiverPlacementAlias,
    grid_size: int,
    joint_to_side: dict[str, str] | None,
    side_areas: dict[str, float] | None,
    tracker: utils_tracker.CreasePatternTracker,
    traversal_plan: list[types.TraversalStep],
    fast_packing_debug: bool,
    is_symmetric: bool,
    symmetry_axes: dict[str, bool] | None,
    debug: debug_types.DebugConfig,
    scoring_config: scoring.ScoringConfig = scoring.ScoringConfig(),
    config: config_lib.PackingConfig = config_lib.PackingConfig(),
    sacrificial_polygons: dict[str, types.PolygonPlacement] | None = None,
    base_square_dims_override: dict[str, tuple[int, int]] | None = None,
) -> dict[str, types.PolygonPlacement] | None:
  """Packs flaps in a single pocket and returns placements if successful.

  Args:
    pocket: The flap pocket to pack.
    stick: The river stick.
    river: The river object.
    grid_size: Grid size.
    joint_to_side: For straight rivers, maps joint to river side.
    side_areas: For straight rivers, maps side to area.
    tracker: Tracker with river added.
    traversal_plan: Traversal plan for packing.
    fast_packing_debug: If True, use fast packing algorithm.
    is_symmetric: If True, enforce symmetry.
    symmetry_axes: If symmetric, stick to axis mapping.
    debug: Debug configuration object.
    scoring_config: Function used to score potential flap placements.
    config: Configuration object that holds solver parameters.
    sacrificial_polygons: If provided, a dict of sacrificial polygon IDs to
      their placements, threaded through to the tiling solver.
    base_square_dims_override: If provided, overrides locally computed dims.

  Returns:
    Dict of packed flaps if successful, else None.
  """

  flaps_to_pack_in_pocket = pocket['flaps']
  if not flaps_to_pack_in_pocket:
    return {}
  pocket_mask = None
  pocket_joint_name = _get_pocket_joint_name(pocket, stick)
  pocket_river_adj_cells = []

  if pocket_joint_name and joint_to_side and side_areas:
    side_name = joint_to_side[pocket_joint_name]
    if isinstance(river, river_lib.River):
      pocket_mask = _get_river_side_mask(river, side_name, grid_size)
    elif isinstance(river, river_lib.LShapedRiver):
      inner_mask, outer_mask = river.get_area_masks(grid_size)
      pocket_mask = inner_mask if side_name == 'inner' else outer_mask
    elif isinstance(river, river_lib.HuggingRiver):
      hugging_mask, opposite_mask = river.get_area_masks(tracker.occupancy_grid)
      pocket_mask = hugging_mask if side_name == 'hugging' else opposite_mask
    river_id = stick.name
    if river_id in tracker.river_banks:
      for r, c in tracker.river_banks[river_id]:
        cell_state = tracker.grid[r, c]
        if (
            cell_state
            and cell_state.type
            == utils_packing_flaps.GridCellType.RIVER_ADJACENT
            and cell_state.source_id == pocket_joint_name
        ):
          pocket_river_adj_cells.append((r, c))

  placements = tiling_solver.pack_flaps(
      tracker=tracker,
      flaps_to_pack=flaps_to_pack_in_pocket,
      traversal_plan=traversal_plan,
      debug=debug,
      fast_packing_debug=fast_packing_debug,
      is_symmetric=is_symmetric,
      symmetry_axes=symmetry_axes,
      confine_to_mask=pocket_mask,
      scoring_config=scoring_config,
      river_adjacent_cells=pocket_river_adj_cells
      if config.enforce_proximity_constraints
      else None,
      config=config,
      sacrificial_polygons=sacrificial_polygons,
      base_square_dims_override=base_square_dims_override,
  )
  return placements


def _precheck_pocket_packing(
    pockets_to_pack: list[packing_utils.FlapPocket],
    stick: stick_lib.Stick,
    river: types.RiverPlacementAlias,
    grid_size: int,
    joint_to_side: dict[str, str] | None,
    side_areas: dict[str, float] | None,
    tracker: utils_tracker.CreasePatternTracker,
    traversal_plan: list[types.TraversalStep],
    is_symmetric: bool,
    symmetry_axes: dict[str, bool] | None,
    debug: debug_types.DebugConfig,
    scoring_config: scoring.ScoringConfig = scoring.ScoringConfig(),
    config: config_lib.PackingConfig = config_lib.PackingConfig(),
) -> bool:
  """Performs a fast pre-check to see if all pockets can be packed.

  Args:
    pockets_to_pack: List of flap pockets to pack.
    stick: The river stick.
    river: The river object.
    grid_size: Grid size.
    joint_to_side: For straight rivers, maps joint to river side.
    side_areas: For straight rivers, maps side to area.
    tracker: Tracker with river added.
    traversal_plan: Traversal plan for packing.
    is_symmetric: If True, enforce symmetry.
    symmetry_axes: If symmetric, stick to axis mapping.
    debug: Debug configuration object.
    scoring_config: Function used to score potential flap placements.
    config: Configuration object that holds solver parameters.

  Returns:
    True if all pockets can be packed, else False.
  """
  tracker_precheck = copy.deepcopy(tracker)
  for pocket in pockets_to_pack:
    placements = _pack_flaps_in_pocket(
        pocket,
        stick,
        river,
        grid_size,
        joint_to_side,
        side_areas,
        tracker_precheck,
        traversal_plan,
        fast_packing_debug=True,
        is_symmetric=is_symmetric,
        symmetry_axes=symmetry_axes,
        debug=debug,
        scoring_config=scoring_config,
        config=config,
    )
    if placements is None:
      return False
    else:
      for flap_id, placement in placements.items():
        try:
          tracker_precheck.add_polygon(flap_id, placement)
        except ValueError:
          return False
  return True


def pack_enclosed_flaps(
    tracker: utils_tracker.CreasePatternTracker,
    stick: stick_lib.Stick,
    river: types.RiverPlacementAlias,
    pockets_info: list[packing_utils.FlapPocket],
    traversal_plan: list[types.TraversalStep],
    grid_size: int,
    joint_to_side: dict[str, str] | None,
    side_areas: dict[str, float] | None,
    fast_packing_debug: bool,
    is_symmetric: bool,
    symmetry_axes: dict[str, bool] | None,
    debug: debug_types.DebugConfig,
    scoring_config: scoring.ScoringConfig = scoring.ScoringConfig(),
    config: config_lib.PackingConfig = config_lib.PackingConfig(),
    sacrificial_polygons: dict[str, types.PolygonPlacement] | None = None,
    base_square_dims_override: dict[str, tuple[int, int]] | None = None,
) -> dict[str, types.PolygonPlacement] | None:
  """Packs flaps in pockets enclosed by river.

  Args:
    tracker: Tracker with river added.
    stick: The river stick.
    river: The river object.
    pockets_info: All flap pockets.
    traversal_plan: Traversal plan for packing.
    grid_size: Grid size.
    joint_to_side: For straight rivers, maps joint to river side.
    side_areas: For straight rivers, maps side to area.
    fast_packing_debug: If True, use fast packing algorithm.
    is_symmetric: If True, enforce symmetry.
    symmetry_axes: If symmetric, stick to axis mapping.
    debug: Debug configuration object.
    scoring_config: Function used to score potential flap placements.
    config: Configuration object that holds solver parameters.
    sacrificial_polygons: If provided, a dict of sacrificial polygon IDs to
      their placements, threaded through to pocket packing.
    base_square_dims_override: If provided, overrides locally computed dims.

  Returns:
    Dict of packed flaps if successful, else None.
  """
  pockets_to_pack = _get_enclosed_pockets_to_pack(
      stick, river, tracker, pockets_info, joint_to_side, side_areas
  )

  # Pre-check if all pockets can be packed with initial packing
  if not fast_packing_debug and len(pockets_to_pack) > 1:
    if not _precheck_pocket_packing(
        pockets_to_pack,
        stick,
        river,
        grid_size,
        joint_to_side,
        side_areas,
        tracker,
        traversal_plan,
        is_symmetric,
        symmetry_axes,
        debug=debug,
        scoring_config=scoring_config,
        config=config,
    ):
      return None

  flaps_for_this_cand = {}
  pocket_placements = []
  for pocket in pockets_to_pack:
    placements = _pack_flaps_in_pocket(
        pocket,
        stick,
        river,
        grid_size,
        joint_to_side,
        side_areas,
        tracker,
        traversal_plan,
        fast_packing_debug,
        is_symmetric=is_symmetric,
        symmetry_axes=symmetry_axes,
        debug=debug,
        scoring_config=scoring_config,
        config=config,
        sacrificial_polygons=sacrificial_polygons,
        base_square_dims_override=base_square_dims_override,
    )
    if placements is None:
      return None
    pocket_placements.append(placements)

  sacrificial_ids = frozenset(sacrificial_polygons or {})
  for placements in pocket_placements:
    for flap_id, placement in placements.items():
      if flap_id in sacrificial_ids and flap_id in flaps_for_this_cand:
        existing = flaps_for_this_cand[flap_id]
        existing_area = existing.size[0] * existing.size[1]
        new_area = placement.size[0] * placement.size[1]
        if new_area <= existing_area:
          continue
      if flap_id in tracker.packed_polygons:
        tracker.remove_by_id(flap_id)
      try:
        tracker.add_polygon(flap_id, placement)
        flaps_for_this_cand[flap_id] = placement
      except ValueError as e:
        logging.warning('Failed to add polygon %s to tracker: %s', flap_id, e)
        return None
  return flaps_for_this_cand
