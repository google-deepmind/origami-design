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

"""Solver utilities specific to packing rivers."""

from collections.abc import Callable, Collection, Generator, Iterator, MutableSequence
import copy
import dataclasses
import functools
import logging
import math
from typing import Any, Literal

import numpy as np

from .. import timeout
from .. import utils_tree
from ..stick_figures import packing_utils
from ..stick_figures import stick as stick_lib
from ..stick_figures import stick_figure
from ..stick_figures import traversal
from . import config as config_lib
from . import hugging_river_generator
from . import level_shifter
from . import river as river_lib
from . import river_adjacency
from . import scoring
from . import tiling_solver
from . import tiling_solver_local
from . import types
from . import utils_backtracking
from . import utils_pack_river_flaps
from . import utils_packing_flaps
from . import utils_packing_svg
from . import utils_tracker
from .debug import debug_types


def _is_border_cell(row: int, col: int, grid_size: int) -> bool:
  """Returns whether (row, col) lies on the outer border of the grid."""
  return row == 0 or col == 0 or row == grid_size - 1 or col == grid_size - 1


def _pack_pocket_not_enclosed_by_rivers(
    pocket: packing_utils.FlapPocket,
    traversal_plan: list[types.TraversalStep],
    tracker: utils_tracker.CreasePatternTracker,
    is_symmetric: bool,
    symmetry_axes: dict[str, bool] | None,
    debug: debug_types.DebugConfig,
    pocket_mask: np.ndarray,
    pocket_river_adj_cells: list[tuple[int, int]],
    config: config_lib.PackingConfig,
    scoring_config: scoring.ScoringConfig = scoring.ScoringConfig(),
    base_square_dims_override: dict[str, tuple[int, int]] | None = None,
) -> Iterator[utils_tracker.CreasePatternTracker]:
  """Packs flaps within a pocket defined by a mask.

  Args:
    pocket: The flap pocket to pack.
    traversal_plan: The traversal plan for all flaps and rivers.
    tracker: The current tracker state.
    is_symmetric: Whether packing is symmetric.
    symmetry_axes: Symmetry axes for sticks.
    debug: Debug configuration object.
    pocket_mask: Mask defining the area of the pocket.
    pocket_river_adj_cells: List of river-adjacent cells for packing constraint.
    config: Configuration object that holds solver parameters.
    scoring_config: Configuration for scoring placements.
    base_square_dims_override: If provided, overrides locally computed dims.

  Yields:
    An iterator of CreasePatternTracker objects, each containing only the
    packed pocket flaps for a successful packing.
  """

  pocket_flap_names = pocket['flap_names']
  pocket_traversal = [
      step for step in traversal_plan if step['stick'].name in pocket_flap_names
  ]
  pocket_dims = {}
  for step in pocket_traversal:
    stick = step['stick']
    if base_square_dims_override and stick.name in base_square_dims_override:
      pocket_dims[stick.name] = base_square_dims_override[stick.name]
    else:
      dim = utils_packing_flaps.get_flap_dim(stick, config.use_level_shifter)
      pocket_dims[stick.name] = (dim, dim)
  pocket_occupancy_mask = tracker.occupancy_grid | ~pocket_mask

  pocket_packing_result = tiling_solver.pack_flaps_with_adjacency(
      traversal_plan=pocket_traversal,
      base_square_dims=pocket_dims,
      grid_size_for_tracker=tracker.grid_size,
      grid_width=tracker.grid_size,
      grid_height=tracker.grid_size,
      debug=debug,
      is_symmetric=is_symmetric,
      symmetry_axes=symmetry_axes,
      allow_paired_axis_placement=False,
      enforce_proximity_constraints=True,
      occupancy_mask=pocket_occupancy_mask,
      scoring_config=scoring_config,
      river_adjacent_cells=pocket_river_adj_cells
      if pocket_river_adj_cells
      else None,
      config=config,
  )
  for packing_result in pocket_packing_result:
    timeout.maybe_timeout_cooperatively()
    yield packing_result[0]


def _get_adjacent_border_cells(
    placement: types.PolygonPlacement,
    grid_size: int,
    occupancy_grid: np.ndarray,
) -> list[tuple[int, int]]:
  """Returns list of unoccupied border cells adjacent to a placement.

  Args:
    placement: The placement of a polygon.
    grid_size: The size of the grid.
    occupancy_grid: The occupancy grid.

  Returns:
    A list of (row, col) tuples of unoccupied border cells adjacent to
    placement.
  """
  adjacent_cells = set()
  offset_x, offset_y = placement.offset
  width, height = placement.size
  # top neighbors
  if offset_y > 0:
    for col in range(offset_x, offset_x + width):
      if 0 <= col < grid_size:
        adjacent_cells.add((offset_y - 1, col))
  # bottom neighbors
  if offset_y + height < grid_size:
    for col in range(offset_x, offset_x + width):
      if 0 <= col < grid_size:
        adjacent_cells.add((offset_y + height, col))
  # left neighbors
  if offset_x > 0:
    for row in range(offset_y, offset_y + height):
      if 0 <= row < grid_size:
        adjacent_cells.add((row, offset_x - 1))
  # right neighbors
  if offset_x + width < grid_size:
    for row in range(offset_y, offset_y + height):
      if 0 <= row < grid_size:
        adjacent_cells.add((row, offset_x + width))

  adjacent_border_cells = []
  for row, col in sorted(adjacent_cells):
    if not occupancy_grid[row, col] and _is_border_cell(row, col, grid_size):
      adjacent_border_cells.append((row, col))
  return adjacent_border_cells


def _get_next_river_start_cell_from_pocket(
    pocket_tracker: utils_tracker.CreasePatternTracker,
    packed_boundary_river: (
        river_lib.River | river_lib.LShapedRiver | river_lib.HuggingRiver
    ),
    grid_size: int,
    *,
    is_symmetric: bool,
) -> tuple[int, int] | None:
  """Computes start cell for the next river based on packed pocket border cells.

  Args:
    pocket_tracker: Tracker containing packed flaps in the pocket.
    packed_boundary_river: The river bordering the packed pocket.
    grid_size: The size of the grid.
    is_symmetric: Whether packing is symmetric.

  Returns:
    The start cell for the next river, or None if no border cells are found.
  """
  border_cells_unique = set()
  for placement in pocket_tracker.packed_polygons.values():
    border_cells_unique.update(
        (y, x)
        for (y, x) in _get_adjacent_border_cells(
            placement, grid_size, pocket_tracker.occupancy_grid
        )
        if not is_symmetric or x < grid_size // 2
    )

  return min(
      border_cells_unique,
      key=lambda cell: river_lib.calculate_counter_clockwise_distance(
          cell, packed_boundary_river.start_cell, grid_size  # pyrefly: ignore[bad-argument-type]
      ),
      default=None,
  )


def _maybe_mirror_pocket_placements(
    packed_polygons: dict[str, types.PolygonPlacement],
    tracker: utils_tracker.CreasePatternTracker,
    template_options: types.TemplateRiverOptions | None,
) -> dict[str, types.PolygonPlacement]:
  """Mirrors asymmetric enclosed-pocket placements when configured."""
  if (
      template_options is None
      or not template_options.mirror_pocket_flaps
      or template_options.full_tree is None
      or not packed_polygons
  ):
    return packed_polygons
  full_tree = template_options.full_tree
  with_mirrors = dict(packed_polygons)
  for f_name, f_pl in packed_polygons.items():
    if f_name in full_tree:
      sym_name = full_tree[f_name].symmetric_stick_name
      if (
          sym_name
          and sym_name != f_name
          and sym_name not in tracker.packed_polygons
          and sym_name not in with_mirrors
      ):
        with_mirrors[sym_name] = (
            utils_packing_svg.get_mirrored_polygon_placement(
                f_pl, tracker.grid_size
            )
        )
  return with_mirrors


def pack_intermediate_pocket(
    pocket: packing_utils.FlapPocket,
    tree: stick_figure.StickFigure,
    traversal_plan: list[types.TraversalStep],
    tracker: utils_tracker.CreasePatternTracker,
    stack: list[types.RiverStackFrame],
    is_symmetric: bool,
    symmetry_axes: dict[str, bool] | None,
    debug: debug_types.DebugConfig,
    config: config_lib.PackingConfig,
    scoring_config: scoring.ScoringConfig = scoring.ScoringConfig(),
    base_square_dims_override: dict[str, tuple[int, int]] | None = None,
    pocket_tiling_cache: (
        dict[
            Any,
            list[
                tuple[tuple[int, int] | None, dict[str, types.PolygonPlacement]]
            ],
        ]
        | None
    ) = None,
    *,
    template_options: types.TemplateRiverOptions | None = None,
) -> Iterator[tuple[tuple[int, int] | None, dict[str, types.PolygonPlacement]]]:
  """Packs an intermediate pocket and yields solutions.

  Args:
    pocket: The flap pocket to pack.
    tree: The stick figure.
    traversal_plan: The traversal plan for all flaps and rivers.
    tracker: The current tracker state.
    stack: The backtracking stack.
    is_symmetric: Whether packing is symmetric.
    symmetry_axes: Symmetry axes for sticks.
    debug: Debug configuration object.
    config: Configuration object that holds solver parameters.
    scoring_config: Configuration for scoring placements.
    base_square_dims_override: If provided, overrides locally computed dims.
    pocket_tiling_cache: Optional cache mapping pocket states to solutions.
    template_options: Optional explicit options for pre-placed shaping-flap
      template rivers.

  Yields:
    An iterator of tuples, each containing:
    - tuple[int, int] | None: The start cell for the next river.
    - dict[str, types.PolygonPlacement]: A dictionary mapping packed flap IDs
      to their placements.
  """
  if not tracker.packed_rivers:
    pocket_adj_cells_on_border = []
    pocket_flap_border_cells = []
    for flap_name, placement in tracker.packed_polygons.items():
      if flap_name in pocket['flap_names']:
        pocket_adj_cells_on_border.extend(
            _get_adjacent_border_cells(
                placement, tracker.grid_size, tracker.occupancy_grid
            )
        )
        ox, oy = placement.offset
        w, h = placement.size
        for row in range(max(0, oy), min(tracker.grid_size, oy + h)):
          for col in range(max(0, ox), min(tracker.grid_size, ox + w)):
            if _is_border_cell(row, col, tracker.grid_size):
              pocket_flap_border_cells.append((row, col))
    anchor_cell = max(
        pocket_flap_border_cells,
        key=lambda cell: river_lib.get_border_cell_counter_clockwise_index(
            cell, tracker.grid_size
        ),
        default=None,
    )
    if anchor_cell is not None:
      start_cell = min(
          pocket_adj_cells_on_border,
          key=lambda cell: river_lib.calculate_counter_clockwise_distance(
              cell, anchor_cell, tracker.grid_size
          ),
          default=None,
      )
    else:
      start_cell = None
    yield start_cell, {}
    return

  if config.legacy_iteration_order:
    packed_boundary_river_names = [
        pocket['boundary_rivers']
        .intersection(tracker.packed_rivers.keys())
        .pop()
    ]
  else:
    packed_boundary_river_names = [
        r for r in tracker.packed_rivers if r in pocket['boundary_rivers']
    ]
  if not packed_boundary_river_names:
    return
  pocket_mask = None
  pocket_river_adj_cells: list[tuple[int, int]] = []
  preplaced_j2s = (
      template_options.preplaced_joint_to_side
      if template_options is not None
      else None
  )
  for r_name in packed_boundary_river_names:
    frame = next(
        (
            frame_item
            for frame_item in stack
            if frame_item.applied_candidate['stick_name'] == r_name
        ),
        None,
    )
    if frame is not None:
      joint_to_side = frame.applied_candidate.get('joint_to_side')
      river = frame.applied_candidate['river']
    elif (
        preplaced_j2s is not None
        and r_name in preplaced_j2s
        and r_name in tracker.packed_rivers
    ):
      joint_to_side = dict(preplaced_j2s[r_name])
      river = tracker.packed_rivers[r_name]
    else:
      return
    r_mask, r_adj_cells = utils_pack_river_flaps.get_pocket_mask_and_cells(
        pocket,
        tree[r_name],
        river,
        tracker.grid_size,
        joint_to_side,
        tracker,
    )
    if r_mask is None:
      return
    pocket_mask = r_mask if pocket_mask is None else (pocket_mask & r_mask)
    for cell in r_adj_cells:
      if cell not in pocket_river_adj_cells:
        pocket_river_adj_cells.append(cell)

  if pocket_mask is None:
    return

  if template_options is not None and template_options.reorder_pocket_flaps:
    boundary_placed = pocket['boundary_rivers'].intersection(
        tracker.packed_rivers.keys()
    )
    if boundary_placed and len(pocket['flap_names']) > 1:
      b_river = min(boundary_placed)
      p_indices = [
          idx
          for idx, st in enumerate(traversal_plan)
          if st['stick'].name in pocket['flap_names']
      ]
      if len(p_indices) > 1:
        first_c = traversal_plan[p_indices[0]]['constraints'].get(b_river, {})
        last_c = traversal_plan[p_indices[-1]]['constraints'].get(b_river, {})
        if (
            first_c.get('type') != types.ConstraintType.EQUAL
            and last_c.get('type') == types.ConstraintType.EQUAL
        ):
          traversal_plan = list(traversal_plan)
          rev_steps = [traversal_plan[idx] for idx in reversed(p_indices)]
          for idx, rev_st in zip(p_indices, rev_steps, strict=True):
            traversal_plan[idx] = rev_st

  cache_key = None
  if (
      config.memoize_pocket_tiling
      and pocket_tiling_cache is not None
      and isinstance(pocket_mask, np.ndarray)
      and isinstance(tracker.occupancy_grid, np.ndarray)
  ):
    gs = tracker.grid_size
    border_mask = np.zeros((gs, gs), dtype=bool)
    border_mask[0, :] = border_mask[-1, :] = border_mask[:, 0] = border_mask[
        :, -1
    ] = True
    pocket_mask_bool = pocket_mask.astype(bool)
    relevant_occ = (
        tracker.occupancy_grid.astype(bool) & (pocket_mask_bool | border_mask)
    ).tobytes()
    dims_key = (
        tuple(sorted(base_square_dims_override.items()))
        if base_square_dims_override
        else None
    )
    river_start = (
        river.start_cell if (river and hasattr(river, 'start_cell')) else None
    )
    cache_key = (
        tuple(sorted(pocket['flap_names'])),
        gs,
        is_symmetric,
        river_start,
        pocket_mask_bool.tobytes(),
        relevant_occ,
        tuple(sorted(pocket_river_adj_cells)),
        dims_key,
    )
    if cache_key in pocket_tiling_cache:
      for cached_start, cached_placements in pocket_tiling_cache[cache_key]:
        yield cached_start, copy.deepcopy(cached_placements)
      return

  collected_solutions: list[
      tuple[tuple[int, int] | None, dict[str, types.PolygonPlacement]]
  ] = []

  pocket_tracker_iterator = _pack_pocket_not_enclosed_by_rivers(
      pocket,
      traversal_plan,
      tracker,
      is_symmetric,
      symmetry_axes,
      debug,
      pocket_mask,
      pocket_river_adj_cells,
      config,
      scoring_config=scoring_config,
      base_square_dims_override=base_square_dims_override,
  )

  for pocket_tracker in pocket_tracker_iterator:
    timeout.maybe_timeout_cooperatively()
    packed_polygons = {}
    for flap_id, placement in pocket_tracker.packed_polygons.items():
      if flap_id not in tracker.packed_polygons:
        packed_polygons[flap_id] = placement

    start_cell_for_next_river = _get_next_river_start_cell_from_pocket(
        pocket_tracker,
        river,
        tracker.grid_size,
        is_symmetric=is_symmetric,
    )

    collected_solutions.append(
        (start_cell_for_next_river, copy.deepcopy(packed_polygons))
    )
    yield start_cell_for_next_river, _maybe_mirror_pocket_placements(
        packed_polygons, tracker, template_options
    )

    if config.stretch_intermediate_pocket_flaps and len(packed_polygons) == 1:
      flap_id, placement = next(iter(packed_polygons.items()))
      ox, oy = placement.offset
      w, h = placement.size
      gs = tracker.grid_size
      x_end = min(gs, ox + w)
      max_col = gs // 2 if is_symmetric else gs
      if ox <= 0 and 0 < x_end <= max_col:
        candidate_rects: list[tuple[int, int, int]] = []
        r0_base, r1_base = max(0, oy), min(gs, oy + h)
        if x_end < max_col and r0_base < r1_base:
          east_slice = (slice(r0_base, r1_base), slice(x_end, max_col))
          if (
              pocket_mask[east_slice].all()
              and not pocket_tracker.occupancy_grid[east_slice].any()
          ):
            candidate_rects.append((oy, h, max_col - ox))
          if (ox + w) % 2 == 1:
            col_slice = (slice(r0_base, r1_base), ox + w)
            if (
                pocket_mask[col_slice].all()
                and not pocket_tracker.occupancy_grid[col_slice].any()
            ):
              candidate_rects.append((oy, h, w + 1))
        if 0 < oy + h < gs:
          south_slice = (slice(oy + h, gs), slice(0, x_end))
          if (
              pocket_mask[south_slice].all()
              and not pocket_tracker.occupancy_grid[south_slice].any()
          ):
            candidate_rects.append((oy, gs - oy, w))
        if 0 < oy and oy + h <= gs:
          north_slice = (slice(0, oy), slice(0, x_end))
          if (
              pocket_mask[north_slice].all()
              and not pocket_tracker.occupancy_grid[north_slice].any()
          ):
            candidate_rects.append((0, oy + h, w))

        yielded_starts_and_sizes: set[
            tuple[tuple[int, int] | None, int, int]
        ] = {(start_cell_for_next_river, w, h)}
        for cur_oy, cur_h, cur_w in candidate_rects:
          stretched_placement = dataclasses.replace(
              placement,
              offset=(ox, cur_oy),
              size=(cur_w, cur_h),
          )
          stretched_tracker = copy.deepcopy(pocket_tracker)
          stretched_tracker.remove_by_id(flap_id)
          stretched_tracker.add_polygon(flap_id, stretched_placement)
          stretched_start = _get_next_river_start_cell_from_pocket(
              stretched_tracker,
              river,
              gs,
              is_symmetric=is_symmetric,
          )
          key = (stretched_start, cur_w, cur_h)
          if key not in yielded_starts_and_sizes:
            yielded_starts_and_sizes.add(key)
            stretched_sol = {flap_id: stretched_placement}
            collected_solutions.append(
                (stretched_start, copy.deepcopy(stretched_sol))
            )
            yield stretched_start, _maybe_mirror_pocket_placements(
                stretched_sol, tracker, template_options
            )
    elif (
        config.stretch_intermediate_pocket_flaps
        and is_symmetric
        and packed_polygons
        and all(
            symmetry_axes and symmetry_axes.get(fid) for fid in packed_polygons
        )
    ):
      gs = tracker.grid_size
      min_ox = min(p.offset[0] for p in packed_polygons.values())
      max_w = gs - 2 * min_ox
      ordered_items = sorted(
          packed_polygons.items(), key=lambda kv: kv[1].offset[1]
      )
      aligned_polygons: dict[str, types.PolygonPlacement] = {}
      can_align = True
      for idx, (fid, p) in enumerate(ordered_items):
        _, p_oy = p.offset
        _, p_h = p.size
        cur_oy = p_oy
        cur_h = p_h
        if idx == len(ordered_items) - 1 and 0 < p_oy + p_h < gs:
          south_slice = (slice(p_oy + p_h, gs), slice(min_ox, gs // 2))
          if (
              pocket_mask[south_slice].all()
              and not pocket_tracker.occupancy_grid[south_slice].any()
          ):
            cur_h = gs - p_oy
        r0, r1 = max(0, cur_oy), min(gs, cur_oy + cur_h)
        if r0 < r1 and not pocket_mask[r0:r1, min_ox : gs // 2].all():
          can_align = False
          break
        aligned_polygons[fid] = dataclasses.replace(
            p, offset=(min_ox, cur_oy), size=(max_w, cur_h)
        )
      if can_align and aligned_polygons != packed_polygons:
        aligned_tracker = copy.deepcopy(tracker)
        try:
          for fid, ap in aligned_polygons.items():
            aligned_tracker.add_polygon(fid, ap)
          aligned_start = _get_next_river_start_cell_from_pocket(
              aligned_tracker,
              river,
              gs,
              is_symmetric=is_symmetric,
          )
          collected_solutions.append(
              (aligned_start, copy.deepcopy(aligned_polygons))
          )
          yield aligned_start, aligned_polygons
        except ValueError:
          pass

  if cache_key is not None and pocket_tiling_cache is not None:
    pocket_tiling_cache[cache_key] = collected_solutions


def find_adjacent_start_cell(
    stick: stick_lib.Stick,
    tree: stick_figure.StickFigure,
    tracker: utils_tracker.CreasePatternTracker,
    grid_size: int,
    debug: debug_types.DebugConfig,
) -> tuple[int, int] | None:
  """Finds border start cell if river must be adjacent to a placed one.

  Args:
    stick: The river stick.
    tree: The stick figure to process.
    tracker: The current tracker state.
    grid_size: The size of the grid.
    debug: Debug configuration object.

  Returns:
    The start cell info if found, otherwise None.
  """
  all_candidate_cells = []
  angularly_adjacent_rivers = []
  if stick.name in tracker.packed_rivers:
    return None
  for placed_river_name in tracker.packed_rivers:
    if placed_river_name not in tree:
      continue
    placed_stick = tree[placed_river_name]
    if river_adjacency.share_joint_with_no_sticks_between(
        stick, placed_stick, tree
    ):
      angularly_adjacent_rivers.append(placed_river_name)
      if placed_river_name not in tracker.river_banks:
        continue
      shared_joints = {stick.start, stick.end} & {
          placed_stick.start,
          placed_stick.end,
      }
      for row, col in tracker.river_banks[placed_river_name]:
        if not _is_border_cell(row, col, grid_size):
          continue
        cell_state = tracker.grid[row, col]
        if (
            cell_state
            and cell_state.type
            is utils_packing_flaps.GridCellType.RIVER_ADJACENT
            and cell_state.source_id in shared_joints
        ):
          all_candidate_cells.append((row, col))

  if not all_candidate_cells:
    return None

  # Sort based on last placed river in angularly_adjacent_rivers
  anchor_river_name = angularly_adjacent_rivers[-1]
  placed_river_start_cell = tracker.packed_rivers[anchor_river_name].start_cell

  unique_candidate_cells = sorted(list(set(all_candidate_cells)))
  cell_distances = []
  for cell in unique_candidate_cells:
    dist = river_lib.calculate_counter_clockwise_distance(
        cell, placed_river_start_cell, grid_size
    )
    cell_distances.append({'cell': cell, 'dist': dist})
  # sort list of dicts by distance
  cell_distances.sort(key=lambda x: x['dist'])

  if debug.is_active():
    debug.record_step(
        tracker,
        'CHOOSE_START_CELL',
        f'River {stick.name}: Found adjacent start cell candidates, sorting CCW'
        f' from {anchor_river_name} (start={placed_river_start_cell}):'
        f' {cell_distances}',
    )
  start_cell_info = cell_distances[0]['cell']
  if debug.is_active():
    debug.record_step(
        tracker,
        'CHOOSE_START_CELL',
        f'River {stick.name}: Chose start cell {start_cell_info} for'
        ' adjacency.',
    )
  return start_cell_info


def find_possible_straight_river_placements(
    grid_size: int, width: int, is_symmetric: bool
) -> list[river_lib.River]:
  """Generates all possible straight river placements.

  Args:
    grid_size: The size of the grid to search.
    width: The width of the river to generate.
    is_symmetric: Whether the river placement is symmetric.

  Returns:
    A list of all possible river placements.
  """
  rivers = []
  for orientation, max_offset in [
      (river_lib.RiverOrientation.HORIZONTAL, grid_size),
      (
          river_lib.RiverOrientation.VERTICAL,
          grid_size if not is_symmetric else math.ceil(grid_size / 2),
      ),
  ]:
    for offset in range(1, max_offset - width):
      rivers.append(
          river_lib.River(
              width=width,
              offset=offset,
              orientation=orientation,
              grid_size=grid_size,
          )
      )
  return rivers


def find_possible_lshape_river_placements(
    grid_size: int, width: int, is_symmetric: bool
) -> list[river_lib.LShapedRiver]:
  """Generates all possible L-shaped river placements.

  Args:
    grid_size: The size of the grid to search.
    width: The width of the river to generate.
    is_symmetric: Whether the river placement is symmetric.

  Returns:
    A list of all possible L-shaped river placements.
  """
  if is_symmetric:
    max_turn_x = math.ceil(grid_size / 2)
  else:
    max_turn_x = grid_size
  rivers = []
  for l_shape_type in river_lib.LShapeType:
    for turn_x in range(max_turn_x - width + 1):
      for turn_y in range(grid_size - width + 1):
        rivers.append(
            river_lib.LShapedRiver(
                width=width,
                grid_size=grid_size,
                l_shape_type=l_shape_type,
                turn_x=turn_x,
                turn_y=turn_y,
            )
        )
  return rivers


def check_river_symmetry_constraints(
    river: types.RiverPlacementAlias,
    grid_size: int,
    is_on_axis: bool,
) -> bool:
  """Checks if a river placement satisfies symmetry constraints.

  If a river is on the symmetry axis, it must be parallel to the symmetry line.
  If it is off the symmetry axis, it must not cross the symmetry line.

  Args:
    river: The river to check.
    grid_size: The size of the grid this river is on.
    is_on_axis: Whether the river is on the symmetry axis.

  Returns:
    True if the river placement satisfies the symmetry constraints, False
    otherwise.
  """
  if is_on_axis:
    if isinstance(river, river_lib.HuggingRiver):
      return river.is_vertically_symmetric()
    elif isinstance(river, river_lib.LShapedRiver):
      if river.l_shape_type in [
          river_lib.LShapeType.TOP_LEFT,
          river_lib.LShapeType.BOTTOM_LEFT,
      ]:
        return river.turn_x + river.width > grid_size / 2.0
      else:  # TOP_RIGHT or BOTTOM_RIGHT
        return river.turn_x < grid_size / 2.0
    elif river.orientation == river_lib.RiverOrientation.VERTICAL:
      return False
    else:  # HORIZONTAL
      return True
  else:  # not on axis
    if isinstance(river, river_lib.HuggingRiver):
      for r in river.rectangles:
        if r.x_start + r.x_len > grid_size / 2.0:
          return False
      return True
    elif isinstance(river, river_lib.LShapedRiver):
      if river.l_shape_type in [
          river_lib.LShapeType.TOP_LEFT,
          river_lib.LShapeType.BOTTOM_LEFT,
      ]:
        return river.turn_x + river.width <= grid_size / 2.0
      else:  # TOP_RIGHT or BOTTOM_RIGHT
        return False
    elif river.orientation == river_lib.RiverOrientation.HORIZONTAL:
      return False  # crosses symmetry line
    else:  # VERTICAL
      return river.offset + river.width <= grid_size / 2.0


def validate_river_candidate(
    occupancy_grid: np.ndarray,
    river: types.RiverPlacementAlias,
    area_reqs: tuple[float, float],
) -> str | None:
  """Checks if a river placement satisfies area constraints.

  Args:
    occupancy_grid: The occupancy grid to check.
    river: The river to check.
    area_reqs: The area requirements for the river.

  Returns:
    None if placement is valid, otherwise a string detailing failure reason.
  """
  a1, a2 = river.compute_unoccupied_areas_on_river_sides(occupancy_grid)

  r1, r2 = area_reqs
  # Check if the computed areas satisfy requirements in either order.
  if not ((a1 >= r1 and a2 >= r2) or (a1 >= r2 and a2 >= r1)):
    return f'Area mismatch: got=({a1:.0f},{a2:.0f}), req=({r1:.0f},{r2:.0f})'
  return None


def add_river_candidate(
    tracker: utils_tracker.CreasePatternTracker, candidate: types.RiverCandidate
) -> list[str]:
  """Adds a river candidate to the tracker.

  Args:
    tracker: The CreasePatternTracker to modify.
    candidate: The candidate to add.

  Returns:
    A list of object IDs added to the tracker.
  """
  tracker.add_river(candidate['stick_name'], candidate['river'])
  added_object_ids = [candidate['stick_name']]
  if 'joint_to_side' in candidate:
    add_river_adjacent_cells(
        tracker,
        candidate['stick_name'],
        candidate['river'],
        candidate['joint_to_side'],
        candidate.get('level_shifter_placements'),
    )
  try:
    for flap_id, placement in candidate['packed_flaps'].items():
      if flap_id in tracker.packed_polygons:
        tracker.remove_by_id(flap_id)
        added_object_ids = [oid for oid in added_object_ids if oid != flap_id]
      tracker.add_polygon(flap_id, placement)
      added_object_ids.append(flap_id)
    return added_object_ids
  except ValueError as e:
    for object_id in added_object_ids:
      tracker.remove_by_id(object_id)
    raise e


def remove_river_candidate(
    tracker: utils_tracker.CreasePatternTracker, object_ids: list[str]
):
  """Removes river candidates from the tracker.

  Args:
    tracker: The CreasePatternTracker to modify.
    object_ids: A list of object IDs to remove.
  """
  for object_id in object_ids:
    tracker.remove_by_id(object_id)


def _is_assignment_valid_with_adj_constraints(
    assignment: dict[str, str],
    adjacent_river_constraints: dict[str, river_adjacency.Side],
) -> bool:
  """Checks if a joint-to-side assignment is valid with adjacent river constraints.

  Args:
    assignment: The joint-to-side assignment to check.
    adjacent_river_constraints: A dictionary mapping joint names to the side
      name ('top', 'bottom', 'left', 'right', 'inner', 'outer') that they
      *cannot* be assigned to.

  Returns:
    True if the assignment is valid, False otherwise.
  """
  for joint, side in assignment.items():
    if (
        joint in adjacent_river_constraints
        and adjacent_river_constraints[joint].value == side
    ):
      return False
  return True


def _get_side_assignment(
    stick: stick_lib.Stick,
    area_sides_reqs: dict[str, float],
    side_areas: dict[str, float],
    side1_name: str,
    side2_name: str,
    adjacent_river_constraints: dict[str, river_adjacency.Side],
    variant_type: river_lib.PathVariantType | None = None,
) -> dict[str, str] | None:
  """Gets joint-to-side assignment based on areas and constraints.

  Args:
    stick: The river stick.
    area_sides_reqs: The area requirements for each side.
    side_areas: The computed areas on each side.
    side1_name: The name of side 1.
    side2_name: The name of side 2.
    adjacent_river_constraints: The adjacent river constraints.
    variant_type: The path variant type for hugging rivers.

  Returns:
    The joint-to-side assignment if valid, otherwise None.
  """
  a1 = side_areas[side1_name]
  a2 = side_areas[side2_name]
  req_area_start = area_sides_reqs[stick.start]
  req_area_end = area_sides_reqs[stick.end]

  is_straight_hugging = variant_type == river_lib.PathVariantType.STRAIGHT

  cand1_req_start_ok = (
      a1 == 0
      if req_area_start == 0 and not is_straight_hugging
      else a1 >= req_area_start
  )
  cand1_req_end_ok = (
      a2 == 0
      if req_area_end == 0 and not is_straight_hugging
      else a2 >= req_area_end
  )
  cand1_valid = cand1_req_start_ok and cand1_req_end_ok

  cand2_req_start_ok = (
      a2 == 0
      if req_area_start == 0 and not is_straight_hugging
      else a2 >= req_area_start
  )
  cand2_req_end_ok = (
      a1 == 0
      if req_area_end == 0 and not is_straight_hugging
      else a1 >= req_area_end
  )
  cand2_valid = cand2_req_start_ok and cand2_req_end_ok

  # Heuristic: assign smaller pocket to smaller area side if possible.
  prefer_cand1 = (req_area_start < req_area_end and a1 < a2) or (
      req_area_start > req_area_end and a2 < a1
  )
  prefer_cand2 = (req_area_start < req_area_end and a2 < a1) or (
      req_area_start > req_area_end and a1 < a2
  )

  assignment1 = {stick.start: side1_name, stick.end: side2_name}
  assignment2 = {stick.start: side2_name, stick.end: side1_name}

  cand1_valid = cand1_valid and _is_assignment_valid_with_adj_constraints(
      assignment1, adjacent_river_constraints
  )
  cand2_valid = cand2_valid and _is_assignment_valid_with_adj_constraints(
      assignment2, adjacent_river_constraints
  )

  if prefer_cand1 and cand1_valid:
    return assignment1
  elif prefer_cand2 and cand2_valid:
    return assignment2
  elif cand1_valid:
    return assignment1
  elif cand2_valid:
    return assignment2
  else:
    return None


def check_placement_adjacency_constraints(
    river: types.RiverPlacementAlias,
    step: types.TraversalStep,
    tracker: utils_tracker.CreasePatternTracker,
    tree: stick_figure.StickFigure,
    debug: debug_types.DebugConfig,
) -> bool:
  """Checks if river placement satisfies adjacency constraints.

  Args:
    river: The river to check.
    step: The traversal step.
    tracker: The CreasePatternTracker.
    tree: The stick figure.
    debug: The debug config.

  Returns:
    True if the river placement satisfies the adjacency constraints, False
    otherwise.
  """
  stick_cand = step['stick']
  river_cand_for_debug: types.RiverCandidate = {
      'stick_name': stick_cand.name,
      'river': river,
      'packed_flaps': {},
  }
  for placed_river_name, placed_river in tracker.packed_rivers.items():
    if placed_river_name not in tree:
      continue
    stick_placed = tree[placed_river_name]
    is_geom_adj = river_adjacency.are_rivers_geometrically_adjacent(
        river, placed_river
    )
    if is_geom_adj:
      logging.info(
          'River %s candidate is geometrically adjacent to placed river %s',
          stick_cand.name,
          placed_river_name,
      )
    if river_adjacency.share_joint_with_no_sticks_between(
        stick_cand, stick_placed, tree
    ):
      is_multi_river_star_joint = False
      cfg = getattr(tracker, 'config', None)
      if getattr(cfg, 'star_hub_rivers', False):
        shared_joints = {stick_cand.start, stick_cand.end} & {
            stick_placed.start,
            stick_placed.end,
        }
        is_multi_river_star_joint = any(
            sum(1 for s in tree.joints[j].sticks if s.is_river) >= 3
            for j in shared_joints
            if j in tree.joints
        )
      if not is_geom_adj and not is_multi_river_star_joint:
        if debug:
          debug.record_step(
              tracker,
              'FAIL',
              f'River {stick_cand.name}: adjacency constraint failed with river'
              f' {placed_river_name}. Rivers for angularly adjacent sticks'
              ' must be geometrically adjacent.',
              cand=river_cand_for_debug,
          )
        return False
  return True


def _infer_zero_area_hugging_sides(
    river: river_lib.HuggingRiver,
    stick: stick_lib.Stick,
    tracker: utils_tracker.CreasePatternTracker,
    tree: stick_figure.StickFigure,
    grid: np.ndarray,
    hugging_joint: str | None,
) -> tuple[dict[str, str], dict[str, float]] | None:
  """Infers hugging river sides from adjacent pre-placed rivers when areas=0."""
  a1, a2 = river.compute_unoccupied_areas_on_river_sides(grid)
  side_areas = {'hugging': float(a1), 'opposite': float(a2)}
  inferred: dict[str, set[str]] = {}
  for r_name, r_obj in tracker.packed_rivers.items():
    if r_name not in tree or r_name == stick.name:
      continue
    r_stick = tree[r_name]
    shared = {stick.start, stick.end} & {r_stick.start, r_stick.end}
    if not shared:
      continue
    jn = next(iter(shared))
    test_occ = grid.copy()
    for rect in r_obj.rectangles:
      x0, y0 = int(rect.x_start), int(rect.y_start)
      xl, yl = int(rect.x_len), int(rect.y_len)
      test_occ[max(0, y0) : y0 + yl, max(0, x0) : x0 + xl] = False
    h_after, o_after = river.compute_unoccupied_areas_on_river_sides(test_occ)
    if h_after > a1 and o_after == a2:
      inferred.setdefault(jn, set()).add('hugging')
    elif o_after > a2 and h_after == a1:
      inferred.setdefault(jn, set()).add('opposite')
  j_start, j_end = stick.start, stick.end
  for s_start, s_end in (('hugging', 'opposite'), ('opposite', 'hugging')):
    cand_map = {j_start: s_start, j_end: s_end}
    if (
        hugging_joint is not None
        and cand_map.get(hugging_joint) != 'hugging'
    ):
      continue
    if inferred and all(
        cand_map[jn] in sides
        for jn, sides in inferred.items()
        if jn in cand_map
    ):
      return cand_map, side_areas
  return None


def _assign_river_sides(
    river: types.RiverPlacementAlias,
    step: types.TraversalStep,
    tracker: utils_tracker.CreasePatternTracker,
    tree: stick_figure.StickFigure,
    debug: debug_types.DebugConfig,
    hugging_joint: str | None = None,
    occupancy_grid: np.ndarray | None = None,
    *,
    infer_zero_area_hugging_sides: bool = False,
) -> tuple[dict[str, str], dict[str, float]] | None:
  """Assigns river sides to joints and returns side assignment and areas.

  Args:
    river: The river to assign sides to.
    step: The traversal step.
    tracker: The CreasePatternTracker.
    tree: The stick figure.
    debug: The debug config.
    hugging_joint: If river is hugging, the joint on the hugging side.
    occupancy_grid: Occupancy grid to use for area computation. If None,
      tracker.occupancy_grid is used.
    infer_zero_area_hugging_sides: If True, infers hugging river sides from
      adjacent pre-placed rivers when remaining branch area requirements are 0.

  Returns:
    The selected joint-to-side assignment if valid, otherwise None.
  """
  grid = (
      occupancy_grid if occupancy_grid is not None else tracker.occupancy_grid
  )
  stick = step['stick']
  variant_type = None
  if isinstance(river, river_lib.River):
    if river.orientation == river_lib.RiverOrientation.HORIZONTAL:
      side1_name, side2_name = 'top', 'bottom'
    else:
      side1_name, side2_name = 'left', 'right'
    a1, a2 = river.compute_unoccupied_areas_on_river_sides(grid)
    side_areas = {side1_name: a1, side2_name: a2}
  elif isinstance(river, river_lib.LShapedRiver):
    side1_name, side2_name = 'inner', 'outer'
    a1, a2 = river.compute_unoccupied_areas_on_river_sides(grid)
    side_areas = {side1_name: a1, side2_name: a2}
  elif isinstance(river, river_lib.HuggingRiver):
    variant_type = river.variant_type
    side1_name, side2_name = 'hugging', 'opposite'
    a1, a2 = river.compute_unoccupied_areas_on_river_sides(grid)
    side_areas = {side1_name: a1, side2_name: a2}
    area_sides_reqs = step['area_sides']
    if hugging_joint is not None:
      if hugging_joint not in area_sides_reqs:
        return None
      other_joint = stick.start if stick.end == hugging_joint else stick.end
      if other_joint not in area_sides_reqs:
        return None
      joint_to_side = {hugging_joint: 'hugging', other_joint: 'opposite'}
      req_hugging = area_sides_reqs[hugging_joint]
      req_opposite = area_sides_reqs[other_joint]
      is_straight_hugging = variant_type == river_lib.PathVariantType.STRAIGHT
      hugging_ok = (
          a1 >= req_hugging
          if is_straight_hugging
          else (a1 == 0 if req_hugging == 0 else a1 >= req_hugging)
      )
      opposite_ok = (
          a2 >= req_opposite
          if is_straight_hugging
          else (a2 == 0 if req_opposite == 0 else a2 >= req_opposite)
      )
      if hugging_ok and opposite_ok:
        return joint_to_side, side_areas  # pyrefly: ignore[bad-return]
      elif (
          infer_zero_area_hugging_sides
          and area_sides_reqs
          and all(v == 0 for v in area_sides_reqs.values())
      ):
        return _infer_zero_area_hugging_sides(
            river, stick, tracker, tree, grid, hugging_joint
        )
      else:
        return None
    else:
      side1_name, side2_name = 'hugging', 'opposite'
  else:
    return None

  area_sides_reqs = step['area_sides']
  adjacent_river_constraints = river_adjacency.get_adjacent_river_constraints(
      river, stick, tracker, tree
  )
  joint_to_side = _get_side_assignment(
      stick,
      area_sides_reqs,
      side_areas,  # pyrefly: ignore[bad-argument-type]
      side1_name,
      side2_name,
      adjacent_river_constraints,
      variant_type=variant_type,
  )

  if joint_to_side is None:
    if (
        infer_zero_area_hugging_sides
        and isinstance(river, river_lib.HuggingRiver)
        and area_sides_reqs
        and all(v == 0 for v in area_sides_reqs.values())
    ):
      inferred = _infer_zero_area_hugging_sides(
          river, stick, tracker, tree, grid, hugging_joint
      )
      if inferred is not None:
        return inferred
    if debug.save_history or debug.live_plot:
      river_cand_for_debug: types.RiverCandidate = {
          'stick_name': stick.name,
          'river': river,
          'packed_flaps': {},
      }
      debug.record_step(
          tracker,
          'FAIL',
          f'River {stick.name}: Could not assign river sides to joints.',
          cand=river_cand_for_debug,
      )
    logging.warning('Could not assign river sides to joints.')
    return None
  return joint_to_side, side_areas  # pyrefly: ignore[bad-return]


def _get_sf_adjacent_cells(
    placement: types.PolygonPlacement,
    side: str,
    grid_size: int,
) -> list[tuple[tuple[int, int], str]]:
  """Returns cells adjacent to a sacrificial flap placement.

  Args:
    placement: The polygon placement of the sacrificial flap.
    side: The river side name to assign to adjacent cells (e.g. 'top').
    grid_size: The size of the grid.

  Returns:
    A list of ((row, col), side) tuples for cells adjacent to the placement.
  """
  cells = []
  offset_x, offset_y = placement.offset
  w, h = placement.size
  if offset_y > 0:
    for col in range(offset_x, offset_x + w):
      if 0 <= col < grid_size:
        cells.append(((offset_y - 1, col), side))
  if offset_y + h < grid_size:
    for col in range(offset_x, offset_x + w):
      if 0 <= col < grid_size:
        cells.append(((offset_y + h, col), side))
  if offset_x > 0:
    for row in range(offset_y, offset_y + h):
      if 0 <= row < grid_size:
        cells.append(((row, offset_x - 1), side))
  if offset_x + w < grid_size:
    for row in range(offset_y, offset_y + h):
      if 0 <= row < grid_size:
        cells.append(((row, offset_x + w), side))
  return cells


def add_river_adjacent_cells(
    tracker: utils_tracker.CreasePatternTracker,
    river_id: str,
    river: types.RiverPlacementAlias,
    joint_to_side: dict[str, str],
    level_shifter_placements: (
        tuple[types.PolygonPlacement, types.PolygonPlacement] | None
    ) = None,
):
  """Adds cells adjacent to river banks to tracker.

  This info is used for debug visualization and for river adjacency checks.
  It populates tracker.grid with RIVER_ADJACENT cells and tracker.river_banks
  with cells adjacent to river banks.

  Args:
    tracker: The CreasePatternTracker to modify.
    river_id: The ID of the river.
    river: The river object.
    joint_to_side: A dictionary mapping joint names to river side names (e.g.
      'top', 'bottom', 'inner', 'outer').
    level_shifter_placements: Placements of sacrificial flaps (sf1, sf2) if any.
  """
  grid_size = tracker.grid_size
  side_to_joint = {v: k for k, v in joint_to_side.items()}
  adjacent_cells_with_side = []

  if isinstance(river, river_lib.River):
    if river.orientation == river_lib.RiverOrientation.HORIZONTAL:
      if river.offset > 0:
        adjacent_cells_with_side.extend(
            [((river.offset - 1, x), 'top') for x in range(grid_size)]
        )
      if river.offset + river.width < grid_size:
        adjacent_cells_with_side.extend([
            ((river.offset + river.width, x), 'bottom')
            for x in range(grid_size)
        ])
    else:  # VERTICAL
      if river.offset > 0:
        adjacent_cells_with_side.extend(
            [((y, river.offset - 1), 'left') for y in range(grid_size)]
        )
      if river.offset + river.width < grid_size:
        adjacent_cells_with_side.extend([
            ((y, river.offset + river.width), 'right') for y in range(grid_size)
        ])
  elif isinstance(river, river_lib.LShapedRiver):
    river_mask = np.zeros((grid_size, grid_size), dtype=bool)
    for rectangle in river.rectangles:
      utils_packing_flaps.set_grid_occupation(river_mask, rectangle, True)
    inner_mask, outer_mask = river.get_area_masks(grid_size)
    for y in range(grid_size):
      for x in range(grid_size):
        if inner_mask[y, x] or outer_mask[y, x]:
          is_adjacent = False
          for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor_y, neighbor_x = y + dy, x + dx
            if (
                0 <= neighbor_y < grid_size
                and 0 <= neighbor_x < grid_size
                and river_mask[neighbor_y, neighbor_x]
            ):
              is_adjacent = True
              break
          if is_adjacent:
            side = 'inner' if inner_mask[y, x] else 'outer'
            adjacent_cells_with_side.append(((y, x), side))
  elif isinstance(river, river_lib.HuggingRiver):
    hugging_adj, opposite_adj = river.get_adjacent_cells_by_side(
        tracker.occupancy_grid
    )
    for cell in hugging_adj:
      adjacent_cells_with_side.append((cell, 'hugging'))
    for cell in opposite_adj:
      adjacent_cells_with_side.append((cell, 'opposite'))

  if level_shifter_placements:
    sf1, sf2 = level_shifter_placements
    side1, side2 = None, None
    if sf1.river_sf_info and sf2.river_sf_info:
      side1 = sf1.river_sf_info.river_side
      side2 = sf2.river_sf_info.river_side
    elif isinstance(river, river_lib.River):
      if river.orientation == river_lib.RiverOrientation.HORIZONTAL:
        side1, side2 = 'top', 'bottom'
      else:
        side1, side2 = 'left', 'right'
    elif isinstance(river, river_lib.LShapedRiver):
      side1, side2 = 'inner', 'outer'
    elif isinstance(river, river_lib.HuggingRiver):
      side1, side2 = 'hugging', 'opposite'

    if side1 and side2:
      adjacent_cells_with_side.extend(
          _get_sf_adjacent_cells(sf1, side1, grid_size)
      )
      adjacent_cells_with_side.extend(
          _get_sf_adjacent_cells(sf2, side2, grid_size)
      )

  river_bank_cells = []
  for (y, x), side_name in adjacent_cells_with_side:
    if side_name in side_to_joint:
      cell = tracker.grid[y, x]
      joint_name = side_to_joint[side_name]
      match cell:
        case utils_packing_flaps.EmptyGridCellState:
          tracker.grid[y, x] = utils_packing_flaps.GridCellState(
              utils_packing_flaps.GridCellType.RIVER_ADJACENT, joint_name
          )
          river_bank_cells.append((y, x))
        case utils_packing_flaps.GridCellState(
            utils_packing_flaps.GridCellType.RIVER_ADJACENT,
            source_id,
        ) if (
            source_id == joint_name
        ):
          river_bank_cells.append((y, x))
  tracker.river_banks[river_id] = river_bank_cells


def attempt_flap_packing(
    river: river_lib.River | river_lib.LShapedRiver,
    step: types.TraversalStep,
    grid_size: int,
    tracker: utils_tracker.CreasePatternTracker,
    pockets_info: list[packing_utils.FlapPocket],
    traversal_plan: list[types.TraversalStep],
    debug: debug_types.DebugConfig,
    joint_to_side: dict[str, str],
    side_areas: dict[str, float],
    fast_packing_debug: bool,
    is_symmetric: bool,
    symmetry_axes: dict[str, bool] | None,
    scoring_config: scoring.ScoringConfig,
    config: config_lib.PackingConfig,
    sf_pair: (
        tuple[types.PolygonPlacement, types.PolygonPlacement] | None
    ) = None,
    base_square_dims_override: dict[str, tuple[int, int]] | None = None,
) -> dict[str, types.PolygonPlacement] | None:
  """Attempts to pack flaps for a given river candidate.

  Args:
    river: The river to pack flaps for.
    step: The traversal step.
    grid_size: The grid size.
    tracker: The CreasePatternTracker.
    pockets_info: The flap pockets info.
    traversal_plan: The traversal plan.
    debug: The debug config.
    joint_to_side: The joint-to-side assignment.
    side_areas: The side areas.
    fast_packing_debug: Whether to use fast packing debug.
    is_symmetric: Whether the stick figure is symmetric.
    symmetry_axes: The symmetry axes.
    scoring_config: Function used to score potential flap placements.
    config: Configuration object that holds solver parameters.
    sf_pair: A tuple of two placements for sacrificial flaps.
    base_square_dims_override: If provided, overrides locally computed dims.

  Returns:
    The packed flaps if successful, otherwise None.
  """
  stick = step['stick']
  tracker_copy = copy.deepcopy(tracker)
  river_cand_for_debug: types.RiverCandidate = {
      'stick_name': stick.name,
      'river': river,
      'packed_flaps': {},
  }
  try:
    tracker_copy.add_river(stick.name, river)
    if joint_to_side:
      add_river_adjacent_cells(
          tracker_copy, stick.name, river, joint_to_side, sf_pair
      )
  except ValueError as e:
    if debug and debug.is_active():
      debug.record_step(
          tracker,
          'FAIL',
          f'River {stick.name}: add_river failed ({e})',
          cand=river_cand_for_debug,
      )
    return None

  sf_placements = {}
  if sf_pair:
    try:
      p1, p2 = sf_pair
      sf1_name = f'{stick.name}{types.RIVER_SF1_SUFFIX}'
      sf2_name = f'{stick.name}{types.RIVER_SF2_SUFFIX}'
      tracker_copy.add_polygon(sf1_name, p1)
      tracker_copy.add_polygon(sf2_name, p2)
      sf_placements = {sf1_name: p1, sf2_name: p2}
      if debug and debug.is_active():
        debug.record_step(
            tracker_copy,
            'INFO',
            f'River {stick.name}: Added sacrificial flaps {sf1_name} and'
            f' {sf2_name}.',
            cand=river_cand_for_debug,
        )
    except ValueError as e:
      if debug and debug.is_active():
        debug.record_step(
            tracker,
            'FAIL',
            f'River {stick.name}: add_polygon for SF failed ({e})',
            cand=river_cand_for_debug,
        )
      return None

  area_info_str_parts = []
  if joint_to_side and side_areas:
    for joint, side in joint_to_side.items():
      req = step['area_sides'][joint]
      comp = side_areas[side]
      area_info_str_parts.append(f'{joint}: req={req:.0f}, comp={comp:.0f}')
  area_info_str = '; '.join(area_info_str_parts)
  debug.record_step(
      tracker_copy,
      'INFO',
      f'River {stick.name}: Found valid placement with areas'
      f' ({area_info_str}), packing flaps...',
      cand=river_cand_for_debug,
  )

  packed_flaps = utils_pack_river_flaps.pack_enclosed_flaps(
      tracker_copy,
      stick,
      river,
      pockets_info,
      traversal_plan,
      grid_size,
      joint_to_side,
      side_areas,
      fast_packing_debug,
      is_symmetric,
      symmetry_axes,
      debug=debug,
      scoring_config=scoring_config,
      config=config,
      sacrificial_polygons=sf_placements or None,
      base_square_dims_override=base_square_dims_override,
  )
  if packed_flaps is not None:
    all_placements = {**sf_placements, **packed_flaps}
    if sf_pair and config.local_tiling_config is not None:
      tiled_tracker = (
          tiling_solver_local.find_perfect_tiling_from_local_expansions(
              tracker_copy,
              is_symmetric=is_symmetric,
              config=config.local_tiling_config,
          )
      )
      if tiled_tracker is not None:
        all_placements.update(tiled_tracker.packed_polygons)
    return all_placements
  return None


def _try_to_add_straight(
    river_width: int,
    grid_size: int,
    orientation: river_lib.RiverOrientation,
    cell_coord: int,
) -> list[river_lib.River]:
  """Generates straight river placements touching a border cell.

  It generates rivers of a given orientation such that at least one cell
  of the river's edge falls on `cell_coord`.

  Args:
    river_width: The width of the river.
    grid_size: The grid size.
    orientation: The orientation of rivers to generate.
    cell_coord: The row or column coordinate of the border cell that the river
      must touch.

  Returns:
    A list of river placements.
  """
  placements = []
  for i in range(river_width):
    offset = cell_coord - i
    if 0 <= offset <= grid_size - river_width:
      placements.append(
          river_lib.River(river_width, offset, orientation, grid_size)
      )
  return placements


def _try_to_add_l_shapes(
    river_width: int,
    grid_size: int,
    dim: Literal['x', 'y'],
    cell_coord: int,
    l_shape_types: Collection[river_lib.LShapeType],
) -> list[river_lib.LShapedRiver]:
  """Generates L-shaped river placements touching a border cell.

  Args:
    river_width: The width of the river.
    grid_size: The grid size.
    dim: 'x' or 'y', indicating if cell_coord is a column or row index.
    cell_coord: The row or column coordinate of the border cell that the river
      must touch.
    l_shape_types: The list of L-shape types to generate.

  Returns:
    A list of river placements.
  """
  placements = []
  coord_range = range(max(0, cell_coord - river_width + 1), cell_coord + 1)
  if dim == 'y':  # for left/right border, cell_coord is row
    for turn_y in coord_range:
      for turn_x in range(grid_size):
        for river_type in l_shape_types:
          placements.append(
              river_lib.LShapedRiver(
                  river_width, grid_size, river_type, turn_x, turn_y
              )
          )
  else:  # dim == 'x', for top/bottom border, cell_coord is col
    for turn_x in coord_range:
      for turn_y in range(grid_size):
        for river_type in l_shape_types:
          placements.append(
              river_lib.LShapedRiver(
                  river_width, grid_size, river_type, turn_x, turn_y
              )
          )
  return placements


def generate_placements_from_start_cell(
    start_cell_info: tuple[int, int],
    river_width: int,
    grid_size: int,
    include_straight_rivers: bool,
    include_lshape_rivers: bool,
    *,
    legacy_iteration_order: bool = False,
) -> list[river_lib.River | river_lib.LShapedRiver]:
  """Generates river candidates based on a starting border cell.

  This function generates river placements (straight or L-shaped) that must
  touch the given starting cell on one of the grid borders. It is used
  when river adjacency constraints require a new river to start from a
  specific border cell adjacent to an existing river.

  Args:
    start_cell_info: The (row, col) of the border cell to start from.
    river_width: The width of the rivers to generate.
    grid_size: The grid size.
    include_straight_rivers: Whether to include straight rivers in search.
    include_lshape_rivers: Whether to include L-shaped rivers in search.
    legacy_iteration_order: Whether to deduplicate via `list(set(...))` instead
      of insertion-order `dict.fromkeys(...)`.

  Returns:
    A list of river placements.
  """
  placements = []
  row, col = start_cell_info

  if col == 0:  # Left border
    if include_straight_rivers:
      placements.extend(
          _try_to_add_straight(
              river_width,
              grid_size,
              river_lib.RiverOrientation.HORIZONTAL,
              row,
          )
      )
    if include_lshape_rivers:
      placements.extend(
          _try_to_add_l_shapes(
              river_width,
              grid_size,
              'y',
              row,
              [river_lib.LShapeType.BOTTOM_LEFT, river_lib.LShapeType.TOP_LEFT],
          )
      )
  if row == grid_size - 1:  # Bottom border
    if include_straight_rivers:
      placements.extend(
          _try_to_add_straight(
              river_width,
              grid_size,
              river_lib.RiverOrientation.VERTICAL,
              col,
          )
      )
    if include_lshape_rivers:
      placements.extend(
          _try_to_add_l_shapes(
              river_width,
              grid_size,
              'x',
              col,
              [river_lib.LShapeType.TOP_LEFT, river_lib.LShapeType.TOP_RIGHT],
          )
      )
  if col == grid_size - 1:  # Right border
    if include_straight_rivers:
      placements.extend(
          _try_to_add_straight(
              river_width,
              grid_size,
              river_lib.RiverOrientation.HORIZONTAL,
              row,
          )
      )
    if include_lshape_rivers:
      placements.extend(
          _try_to_add_l_shapes(
              river_width,
              grid_size,
              'y',
              row,
              [
                  river_lib.LShapeType.BOTTOM_RIGHT,
                  river_lib.LShapeType.TOP_RIGHT,
              ],
          )
      )
  if row == 0:  # Top border
    if include_straight_rivers:
      placements.extend(
          _try_to_add_straight(
              river_width,
              grid_size,
              river_lib.RiverOrientation.VERTICAL,
              col,
          )
      )
    if include_lshape_rivers:
      placements.extend(
          _try_to_add_l_shapes(
              river_width,
              grid_size,
              'x',
              col,
              [
                  river_lib.LShapeType.BOTTOM_LEFT,
                  river_lib.LShapeType.BOTTOM_RIGHT,
              ],
          )
      )
  if legacy_iteration_order:
    return list(set(placements))
  return list(dict.fromkeys(placements))


def find_river_candidates(
    step: types.TraversalStep,
    grid_size: int,
    tracker: utils_tracker.CreasePatternTracker,
    tree: stick_figure.StickFigure,
    pockets_info: list[packing_utils.FlapPocket],
    traversal_plan: list[types.TraversalStep],
    debug: debug_types.DebugConfig,
    fast_packing_debug: bool = False,
    is_symmetric: bool = False,
    symmetry_axes: dict[str, bool] | None = None,
    scoring_config: scoring.ScoringConfig = scoring.ScoringConfig(),
    pocket_start_cell: tuple[int, int] | None = None,
    hugging_joint: str | None = None,
    config: config_lib.PackingConfig = config_lib.PackingConfig(),
    base_square_dims_override: dict[str, tuple[int, int]] | None = None,
    *,
    template_options: types.TemplateRiverOptions | None = None,
) -> Iterator[types.RiverCandidate]:
  """Generates and validates river candidates for a given traversal step.

  A river candidate is only yielded if flaps on its sides can be successfully
  packed with it.

  Args:
    step: The traversal step containing stick and constraint info.
    grid_size: The size of the grid.
    tracker: The current tracker state.
    tree: The stick figure to process.
    pockets_info: Precomputed information about flap pockets.
    traversal_plan: The traversal plan.
    debug: Debug configuration object.
    fast_packing_debug: If True, use fast packing for flaps.
    is_symmetric: If True, symmetry constraints are applied.
    symmetry_axes: A dict indicating which stick names are on a symmetry axis.
    scoring_config: Function used to score potential flap placements.
    pocket_start_cell: If provided, use this as start cell for hugging river.
    hugging_joint: If river is hugging, the joint on the hugging side.
    config: Configuration object that holds solver parameters.
    base_square_dims_override: If provided, overrides locally computed dims.
    template_options: Optional explicit options for pre-placed shaping-flap
      template rivers.

  Yields:
    Valid RiverCandidate dictionaries.
  """
  stick = step['stick']
  if not stick.is_river:
    return
  if template_options is not None and step.get('area_sides'):
    refreshed_plan = utils_tree.compute_constraints_along_traversal(
        tree,
        config=config,
        tracker=tracker,
        pockets_info=pockets_info,
    )
    placed_all = set(tracker.packed_polygons) | set(tracker.packed_rivers)
    step = copy.deepcopy(step)
    for r_step in refreshed_plan:
      if r_step['stick'].name == stick.name:
        step['area_sides'] = dict(r_step['area_sides'])
        break
    for side_joint in list(step['area_sides']):
      visited = {stick.name}
      frontier = [side_joint]
      unplaced_on_branch = False
      while frontier and not unplaced_on_branch:
        jn = frontier.pop()
        for b_stick in tree.joints[jn].sticks:
          if b_stick.name in visited:
            continue
          visited.add(b_stick.name)
          if b_stick.name not in placed_all:
            unplaced_on_branch = True
            break
          frontier.append(b_stick.other_joint_name(jn))
      if not unplaced_on_branch:
        step['area_sides'][side_joint] = 0.0
  river_width = int(stick.length)
  placements: list[types.RiverPlacementAlias] = []

  grid_for_area_computation = tracker.occupancy_grid
  if is_symmetric:
    grid_for_area_computation = grid_for_area_computation.copy()
    grid_for_area_computation[:, grid_size // 2 :] = 1

  start_cell_info = pocket_start_cell or find_adjacent_start_cell(
      stick, tree, tracker, grid_size, debug
  )
  hugging_sf = None
  effective_occupancy = tracker.occupancy_grid
  if start_cell_info:
    hug_boundaries = False
    if is_symmetric and symmetry_axes and symmetry_axes[stick.name]:
      hug_boundaries = True
    if is_symmetric and not hug_boundaries and config.symmetry_line_obstacles:
      effective_occupancy = grid_for_area_computation

    effective_start_cell = start_cell_info
    if config.use_level_shifter and stick.requires_level_shifter:
      sf_result = level_shifter.place_hugging_side_sacrificial_flap(
          start_cell_info, grid_size, tracker.occupancy_grid
      )
      if sf_result is None:
        return
      hugging_sf, effective_start_cell = sf_result
      effective_occupancy = effective_occupancy.copy()
      offset_x, offset_y = hugging_sf.offset
      size_w, size_h = hugging_sf.size
      effective_occupancy[
          offset_y : offset_y + size_h, offset_x : offset_x + size_w
      ] = True

    hug_side = river_lib.HugSide.LEFT
    if debug.is_active():
      debug.record_step(
          tracker,
          'INFO',
          f'River {stick.name}: Generating hugging river candidates from'
          f" {effective_start_cell} with hug_side='{hug_side}'"
          f" area_reqs={step['area_sides']}",
      )
    cand_rivers = list(
        hugging_river_generator.generate_hugging_river_candidates(
            grid_size,
            river_width,
            effective_occupancy,
            effective_start_cell,
            hug_side,
            hug_boundaries=hug_boundaries,
            area_reqs=step['area_sides'],
            allow_off_grid=config.easy_mode,
            retry_opposite_hug_side=bool(
                template_options and template_options.retry_opposite_hug_side
            ),
        )
    )
    if config.reject_degenerate_hugging_rivers:
      # Separated layers split a single multi-width river around opposite sides
      # of a flap, violating constant river width, so they are filtered out.
      # Hairpin turns are deprioritized behind non-hairpin candidates so tight
      # asymmetric grids (where a width-1 inner pocket flap forces path0 to
      # fold around a 1-cell corner) remain packable when needed.
      cand_rivers = [
          r
          for r in cand_rivers
          if not isinstance(r, river_lib.HuggingRiver)
          or not hugging_river_generator.has_separated_layers(r.subriver_paths)
      ]
      non_hairpin = [
          r
          for r in cand_rivers
          if not isinstance(r, river_lib.HuggingRiver)
          or not hugging_river_generator.has_hairpin_turn(r.subriver_paths)
      ]
      hairpin = [
          r
          for r in cand_rivers
          if isinstance(r, river_lib.HuggingRiver)
          and hugging_river_generator.has_hairpin_turn(r.subriver_paths)
      ]
      cand_rivers = non_hairpin + hairpin
    placements.extend(cand_rivers)

    has_unplaced_sibling_at_star_joint = (
        config.star_hub_rivers
        and pocket_start_cell is None
        and any(
            sum(1 for s in tree.joints[j].sticks if s.is_river) >= 3
            and any(
                s.is_river
                and s.name != stick.name
                and s.name not in tracker.packed_rivers
                for s in tree.joints[j].sticks
            )
            for j in (stick.start, stick.end)
            if j in tree.joints
        )
    )
    if has_unplaced_sibling_at_star_joint and not config.easy_mode:
      if config.include_straight_rivers:
        placements.extend(
            find_possible_straight_river_placements(
                grid_size, river_width, is_symmetric
            )
        )
      if config.include_lshape_rivers:
        placements.extend(
            find_possible_lshape_river_placements(
                grid_size, river_width, is_symmetric
            )
        )
  else:
    if config.easy_mode:
      logging.warning(
          'River %s: Easy mode active but no start cell found. Skipping'
          ' non-hugging placements.',
          stick.name,
      )
      return

    if tracker.packed_rivers:
      logging.warning(
          'River %s must be adjacent to a packed river, but no'
          ' adjacency was found. Skipping non-hugging placements.',
          stick.name,
      )
      return
    if config.include_straight_rivers:
      placements.extend(
          find_possible_straight_river_placements(
              grid_size, river_width, is_symmetric
          )
      )
    if config.include_lshape_rivers:
      placements.extend(
          find_possible_lshape_river_placements(
              grid_size, river_width, is_symmetric
          )
      )

  if is_symmetric and symmetry_axes:
    is_on_axis = symmetry_axes[stick.name]
    placements = [
        p
        for p in placements
        if check_river_symmetry_constraints(p, grid_size, is_on_axis)
    ]

  if config.remove_central_crease and is_symmetric:
    is_on_axis = bool(symmetry_axes and symmetry_axes.get(stick.name))
    if not is_on_axis:
      half = grid_size // 2
      placements = [
          p
          for p in placements
          if all(
              int(r.x_start) + int(r.x_len) <= half - 1 for r in p.rectangles
          )
      ]

  yielded_any = False
  for river in placements:
    timeout.maybe_timeout_cooperatively()
    area_reqs = tuple(step['area_sides'].values())
    river_cand_for_debug = {
        'stick_name': stick.name,
        'river': river,
        'packed_flaps': {},
    }

    validation_error = validate_river_candidate(
        grid_for_area_computation, river, area_reqs  # pyrefly: ignore[bad-argument-type]
    )
    if validation_error:

      if debug.is_active():
        debug.record_step(
            tracker,
            'FAIL',
            f'River {stick.name}: {validation_error}',
            cand=river_cand_for_debug,
        )
      continue

    if not start_cell_info and not check_placement_adjacency_constraints(
        river, step, tracker, tree, debug
    ):
      continue

    assignment = _assign_river_sides(
        river,
        step,
        tracker,
        tree,
        debug,
        hugging_joint,
        occupancy_grid=grid_for_area_computation,
        infer_zero_area_hugging_sides=bool(
            template_options and template_options.infer_zero_area_hugging_sides
        ),
    )
    if assignment is None:
      continue
    joint_to_side, side_areas = assignment

    sf_pair = None
    if hugging_sf is not None and isinstance(river, river_lib.HuggingRiver):
      temp_grid = effective_occupancy.copy()
      for path in river.subriver_paths:
        coords = np.array(path)
        valid = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < grid_size)
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < grid_size)
        )
        temp_grid[coords[valid, 0], coords[valid, 1]] = True
      opposite_sf = next(
          level_shifter.find_opposite_sacrificial_slot(
              hugging_sf, river, temp_grid, is_symmetric=is_symmetric
          ),
          None,
      )
      if opposite_sf is None:
        continue
      side_to_joint = {v: k for k, v in joint_to_side.items()}
      hugging_sf_tagged = dataclasses.replace(
          hugging_sf,
          river_sf_info=types.RiverSfInfo(
              river_name=stick.name,
              river_side='hugging',
              parent_joint=side_to_joint.get('hugging', ''),
          ),
      )
      opposite_sf = dataclasses.replace(
          opposite_sf,
          river_sf_info=types.RiverSfInfo(
              river_name=stick.name,
              river_side='opposite',
              parent_joint=side_to_joint.get('opposite', ''),
          ),
      )
      sf_pair = (hugging_sf_tagged, opposite_sf)
    elif (
        config.use_level_shifter
        and stick.requires_level_shifter
        and isinstance(river, (river_lib.River, river_lib.LShapedRiver))
    ):
      sf_pair = next(
          level_shifter.find_aligned_sacrificial_slots(
              river, tracker.occupancy_grid, is_symmetric=is_symmetric
          ),
          None,
      )
      if sf_pair is not None:
        p1, p2 = sf_pair
        side_to_joint = {v: k for k, v in joint_to_side.items()}
        if isinstance(river, river_lib.River):
          if river.orientation == river_lib.RiverOrientation.HORIZONTAL:
            side1, side2 = 'top', 'bottom'
          else:
            side1, side2 = 'left', 'right'
        else:
          inner_mask, _ = river.get_area_masks(river.grid_size)
          c1, r1 = p1.offset
          if inner_mask[max(r1, 0), max(c1, 0)]:
            side1, side2 = 'inner', 'outer'
          else:
            side1, side2 = 'outer', 'inner'
        p1 = dataclasses.replace(
            p1,
            river_sf_info=types.RiverSfInfo(
                river_name=stick.name,
                river_side=side1,
                parent_joint=side_to_joint.get(side1, ''),
            ),
        )
        p2 = dataclasses.replace(
            p2,
            river_sf_info=types.RiverSfInfo(
                river_name=stick.name,
                river_side=side2,
                parent_joint=side_to_joint.get(side2, ''),
            ),
        )
        sf_pair = (p1, p2)

    packed_flaps = attempt_flap_packing(
        river,  # pyrefly: ignore[bad-argument-type]
        step,
        grid_size,
        tracker,
        pockets_info,
        traversal_plan,
        debug,
        joint_to_side,
        side_areas,
        fast_packing_debug,
        is_symmetric,
        symmetry_axes,
        scoring_config,
        config,
        sf_pair=sf_pair,
        base_square_dims_override=base_square_dims_override,
    )

    if packed_flaps is not None:
      river_cand: types.RiverCandidate = {
          'stick_name': stick.name,
          'river': river,
          'packed_flaps': packed_flaps,
      }
      if joint_to_side:
        river_cand['joint_to_side'] = joint_to_side
      if sf_pair:
        river_cand['level_shifter_placements'] = sf_pair
      yielded_any = True
      yield river_cand
  if (
      not yielded_any
      and pocket_start_cell is not None
      and template_options is not None
      and template_options.retry_without_pocket_start_cell
  ):
    yield from find_river_candidates(
        step,
        grid_size,
        tracker,
        tree,
        pockets_info,
        traversal_plan,
        debug,
        fast_packing_debug,
        is_symmetric,
        symmetry_axes,
        scoring_config,
        None,
        None,
        config,
        base_square_dims_override=base_square_dims_override,
        template_options=template_options,
    )


def process_solution_and_backtrack(
    *,
    stack: MutableSequence[types.RiverStackFrame],
    tracker: utils_tracker.CreasePatternTracker,
    solutions: MutableSequence[utils_tracker.CreasePatternTracker],
    first_river_step_index: int,
    max_solutions: int,
    debug: debug_types.DebugConfig,
    exhaust_solutions: bool = False,
) -> int | None:
  """Processes a solution and backtracks to find more diverse solutions.

  When a solution is found, it is added to the results. If more solutions
  are needed (max_solutions > 1), it then backtracks.

  If exhaust_solutions is False (default), it backtracks to the point *before*
  placing rivers that come after the first one. The search for the next
  solution will then start by trying the next placement option for the
  *first* river. This strategy ensures that if multiple solutions are
  returned, they differ in the placement of at least the first river,
  thus improving solution diversity.

  If exhaust_solutions is True, it performs a standard backtracking step,
  checking for alternative placements for the last placed river. This allows
  finding all possible solutions (up to max_solutions).

  Args:
    stack: The backtracking stack.
    tracker: The current tracker state containing the solution.
    solutions: The list of solutions found so far.
    first_river_step_index: The index of the first river step in traversal plan.
    max_solutions: The maximum number of solutions to return.
    debug: Debug configuration object.
    exhaust_solutions: Whether to exhaust all solutions.

  Returns:
    The next step_index to continue search from (if backtracking), or None if
    search should terminate.
  """
  solutions.append(copy.deepcopy(tracker))
  timeout.maybe_timeout_cooperatively()
  if len(solutions) == max_solutions:
    return None

  if not exhaust_solutions and first_river_step_index != -1:
    # Backtrack to explore options for the first river
    while stack and stack[-1].step_index > first_river_step_index:
      frame_to_remove = stack.pop()
      remove_river_candidate(tracker, frame_to_remove.object_ids)
  step_index = utils_backtracking.try_to_backtrack(
      stack,
      functools.partial(add_river_candidate, tracker),
      functools.partial(remove_river_candidate, tracker),
      tracker,
      debug,
  )
  return step_index


def iter_solve_rivers_iteratively(
    tree: stick_figure.StickFigure,
    *,
    debug: debug_types.DebugConfig,
    grid_size: int | None = None,
    fast_packing_debug: bool = False,
    is_symmetric: bool = False,
    symmetry_axes: dict[str, bool] | None = None,
    scoring_config: scoring.ScoringConfig = scoring.ScoringConfig(),
    max_solutions: int | None = 1,
    config: config_lib.PackingConfig = config_lib.PackingConfig(),
    initial_tracker: utils_tracker.CreasePatternTracker | None = None,
    ignore_sticks: set[str] | None = None,
    pockets_info: list[packing_utils.FlapPocket] | None = None,
    default_traversal_plan: list[types.TraversalStep] | None = None,
    base_square_dims_override: dict[str, tuple[int, int]] | None = None,
    accept_fn: (
        Callable[[utils_tracker.CreasePatternTracker], bool] | None
    ) = None,
    template_options: types.TemplateRiverOptions | None = None,
) -> Generator[utils_tracker.CreasePatternTracker, bool | None, None]:
  """Lazily yields river and enclosed-pocket trackers using backtracking.

  Callers that validate downstream leaf pockets (`remaining_flaps`) after each
  yielded tracker can either pass an `accept_fn` predicate or send a boolean
  acceptance decision back into the generator via `.send(accepted)`. When a
  tracker is rejected (`False`), the solver performs a local backtrack on the
  most recently placed river and does not count the tracker toward
  `max_solutions`.

  Args:
    tree: The stick figure to pack.
    debug: Debug configuration object.
    grid_size: The grid size to use for packing.
    fast_packing_debug: Whether to enable fast packing debug mode.
    is_symmetric: Whether symmetry is enforced.
    symmetry_axes: Mapping of stick names to whether they lie on symmetry axis.
    scoring_config: Scoring configuration for candidate placement.
    max_solutions: Maximum number of accepted river solutions to yield.
    config: Packing configuration parameters.
    initial_tracker: Optional starting tracker with pre-packed polygons.
    ignore_sticks: Optional set of stick names to skip during river packing.
    pockets_info: Precomputed flap pockets.
    default_traversal_plan: Precomputed traversal plan.
    base_square_dims_override: Optional override for base square dimensions.
    accept_fn: Optional predicate returning True when a yielded tracker is
      accepted by downstream pocket packing.
    template_options: Optional explicit options for pre-placed shaping-flap
      template rivers.

  Yields:
    Candidate `CreasePatternTracker` instances with rivers and enclosed pockets
    placed.
  """
  try:
    if pockets_info is None:
      pockets_info = packing_utils.precompute_flap_pockets(tree, config=config)
    if default_traversal_plan is not None:
      traversal_plan = default_traversal_plan
    else:
      traversal_plan = utils_tree.compute_constraints_along_traversal(
          tree,
          config=config,
          tracker=initial_tracker,
          pockets_info=pockets_info,
      )
    all_sticks_traversal = list(
        traversal.packing_traversal(
            tree,
            start_stick_name=config.start_stick_name,
            mode=config.traversal_mode,
        )
    )
    stick_to_idx = {s.name: i for i, s in enumerate(all_sticks_traversal)}
    pockets_info.sort(
        key=lambda p: min(stick_to_idx[f.name] for f in p['flaps']),
    )
  except ValueError:
    return
  if initial_tracker:
    tracker = copy.deepcopy(initial_tracker)
  else:
    tracker = utils_tracker.CreasePatternTracker(grid_size)  # pyrefly: ignore[bad-argument-type]
  stack: list[types.RiverStackFrame] = []
  step_index = 0
  accepted_count = 0
  pocket_tiling_cache: (
      dict[
          Any,
          list[
              tuple[tuple[int, int] | None, dict[str, types.PolygonPlacement]]
          ],
      ]
      | None
  ) = (
      {} if config.memoize_pocket_tiling else None
  )
  enclosed_tiling_cache: (
      dict[Any, dict[str, types.PolygonPlacement] | None] | None
  ) = ({} if config.memoize_pocket_tiling else None)

  first_river_step_index = -1
  for i, step in enumerate(traversal_plan):
    if step['stick'].is_river:
      first_river_step_index = i
      break

  try:
    while True:
      timeout.maybe_timeout_cooperatively()
      if step_index == len(traversal_plan):
        candidate_tracker = copy.deepcopy(tracker)
        if accept_fn is not None and not accept_fn(candidate_tracker):
          accepted = False
        else:
          sent_accepted = yield candidate_tracker
          accepted = True if sent_accepted is None else bool(sent_accepted)
        if accepted:
          accepted_count += 1
          if max_solutions is not None and accepted_count >= max_solutions:
            return
        timeout.maybe_timeout_cooperatively()
        if (
            accepted
            and not config.exhaust_solutions
            and first_river_step_index != -1
        ):
          while stack and stack[-1].step_index > first_river_step_index:
            frame_to_remove = stack.pop()
            remove_river_candidate(tracker, frame_to_remove.object_ids)
        step_index = utils_backtracking.try_to_backtrack(
            stack,
            functools.partial(add_river_candidate, tracker),
            functools.partial(remove_river_candidate, tracker),
            tracker,
            debug,
        )
        if step_index is None:
          return
        continue

      current_step = traversal_plan[step_index]
      if ignore_sticks and current_step['stick'].name in ignore_sticks:
        step_index += 1
        continue
      if not current_step['stick'].is_river:
        step_index += 1
        continue

      stick = current_step['stick']
      stick_name = stick.name
      pocket_start_cell = None
      hugging_joint = None
      is_first_river = not tracker.packed_rivers

      pocket_to_pack = None
      is_star_hub = getattr(config, 'star_hub_rivers', False)
      for pocket in pockets_info:
        boundary_match = (
            stick_name in pocket['boundary_rivers']
            and pocket['boundary_rivers'].intersection(
                tracker.packed_rivers.keys()
            )
            and (
                not is_star_hub
                or (pocket['boundary_rivers'] - {stick_name}).issubset(
                    tracker.packed_rivers.keys()
                )
            )
        )
        if boundary_match:
          if any(f.name in tracker.packed_polygons for f in pocket['flaps']):
            continue
          pocket_to_pack = pocket
          break
        elif is_first_river and pocket['flap_names'].intersection(
            tracker.packed_polygons.keys()
        ):
          pocket_to_pack = pocket
          break

      logging.info(
          'Step %d, Attempting to place river %s',
          step_index,
          current_step['stick'].name,
      )
      if pocket_to_pack:

        def _get_candidates_with_pocket_packing(
            pocket: packing_utils.FlapPocket,
            step_for_river: types.TraversalStep,
        ) -> Iterator[types.RiverCandidate]:
          """Generates candidates with pocket packing."""
          stick = step_for_river['stick']
          pocket_packing_iterator = pack_intermediate_pocket(
              pocket,
              tree,
              traversal_plan,
              tracker,
              stack,
              is_symmetric,
              symmetry_axes,
              debug,
              config,
              scoring_config=scoring_config,
              base_square_dims_override=base_square_dims_override,
              pocket_tiling_cache=pocket_tiling_cache,
              template_options=template_options,
          )
          is_star_hub = getattr(config, 'star_hub_rivers', False)
          if is_star_hub:
            has_off_axis_flap = any(
                not (
                    is_symmetric and symmetry_axes and symmetry_axes.get(f.name)
                )
                for f in pocket['flaps']
            )
          for (
              p_start_cell,
              p_packed_placements,
          ) in pocket_packing_iterator:
            timeout.maybe_timeout_cooperatively()
            if (
                is_star_hub
                and not is_first_river
                and has_off_axis_flap
                and p_start_cell is None
            ):
              continue
            current_step_for_iter = copy.deepcopy(step_for_river)
            tracker_with_pocket = copy.deepcopy(tracker)
            for f_id, f_placement in p_packed_placements.items():
              tracker_with_pocket.add_polygon(f_id, f_placement)

            p_hugging_joint = None
            if p_start_cell:
              p_pocket_joints = set()
              for f in pocket['flaps']:
                p_pocket_joints.add(f.start)
                p_pocket_joints.add(f.end)
              p_shared_joints = {stick.start, stick.end} & p_pocket_joints
              if p_shared_joints:
                if config.legacy_iteration_order:
                  p_hugging_joint = p_shared_joints.pop()
                else:
                  p_hugging_joint = (
                      stick.start
                      if stick.start in p_pocket_joints
                      else stick.end
                  )

            river_iter = find_river_candidates(
                current_step_for_iter,
                grid_size,  # pyrefly: ignore[bad-argument-type]
                tracker_with_pocket,
                tree,
                pockets_info,
                traversal_plan,
                debug,
                fast_packing_debug,
                is_symmetric,
                symmetry_axes,
                scoring_config,
                p_start_cell,
                p_hugging_joint,
                config,
                base_square_dims_override=base_square_dims_override,
                template_options=template_options,
            )
            if is_star_hub:
              pocket_fully_enclosed = pocket['boundary_rivers'].issubset(
                  set(tracker.packed_rivers.keys()) | {stick.name}
              )
            else:
              pocket_fully_enclosed = True
            for river_cand in river_iter:
              timeout.maybe_timeout_cooperatively()
              flaps_in_pocket_are_packed = any(
                  f_name in tracker.packed_polygons
                  for f_name in pocket['flap_names']
              )
              if is_first_river and flaps_in_pocket_are_packed:
                river_cand['packed_flaps'].update(p_packed_placements)
                yield river_cand
              elif is_star_hub and not pocket_fully_enclosed:
                yield river_cand
              else:
                # Re-tile pocket in the space between all its boundary rivers
                tracker_with_all_rivers = copy.deepcopy(tracker)
                tracker_with_all_rivers.add_river(
                    river_cand['stick_name'], river_cand['river']
                )
                if 'joint_to_side' in river_cand:
                  add_river_adjacent_cells(
                      tracker_with_all_rivers,
                      river_cand['stick_name'],
                      river_cand['river'],
                      river_cand['joint_to_side'],
                      river_cand.get('level_shifter_placements'),
                  )

                sf_placements = {}
                for sf_id, sf_placement in river_cand['packed_flaps'].items():
                  if sf_placement.river_sf_info is None:
                    continue
                  if sf_id in tracker_with_all_rivers.packed_polygons:
                    continue
                  tracker_with_all_rivers.add_polygon(sf_id, sf_placement)
                  sf_placements[sf_id] = sf_placement
                if not sf_placements:
                  sf_placements = None

                # The flaps in the current pocket are going to be re-tiled, so
                # they should be removed from the occupancy grid.
                for flap_name in pocket['flap_names']:
                  if flap_name in tracker_with_all_rivers.packed_polygons:
                    tracker_with_all_rivers.remove_by_id(flap_name)

                enclosed_mask = utils_pack_river_flaps.get_enclosed_pocket_mask(
                    pocket,
                    tracker_with_all_rivers,
                    tree,
                    stack,
                    river_cand,
                    preplaced_joint_to_side=(
                        template_options.preplaced_joint_to_side
                        if template_options is not None
                        else None
                    ),
                )

                if enclosed_mask is None:
                  continue

                enclosed_cache_key = None
                if (
                    config.memoize_pocket_tiling
                    and enclosed_tiling_cache is not None
                    and isinstance(enclosed_mask, np.ndarray)
                    and isinstance(
                        tracker_with_all_rivers.occupancy_grid, np.ndarray
                    )
                ):
                  if sf_placements:
                    sf_key = tuple(
                        sorted(
                            (k, p.offset, p.size)
                            for k, p in sf_placements.items()
                        )
                    )
                  else:
                    sf_key = None
                  enclosed_mask_bool = enclosed_mask.astype(bool)
                  relevant_enclosed = (
                      tracker_with_all_rivers.occupancy_grid.astype(bool)
                      & enclosed_mask_bool
                  ).tobytes()
                  enclosed_cache_key = (
                      tuple(sorted(pocket['flap_names'])),
                      grid_size,
                      is_symmetric,
                      enclosed_mask_bool.tobytes(),
                      relevant_enclosed,
                      sf_key,
                  )
                  if enclosed_cache_key in enclosed_tiling_cache:
                    cached_retiled = enclosed_tiling_cache[enclosed_cache_key]
                    if cached_retiled is not None:
                      river_cand['packed_flaps'].update(
                          copy.deepcopy(cached_retiled)
                      )
                      yield river_cand
                    continue

                retiled_placements = tiling_solver.pack_flaps(
                    tracker=tracker_with_all_rivers,
                    flaps_to_pack=pocket['flaps'],
                    traversal_plan=traversal_plan,
                    debug=debug,
                    is_symmetric=is_symmetric,
                    symmetry_axes=symmetry_axes,
                    confine_to_mask=enclosed_mask,
                    scoring_config=scoring_config,
                    config=config,
                    sacrificial_polygons=sf_placements,
                )
                if (
                    enclosed_cache_key is not None
                    and enclosed_tiling_cache is not None
                ):
                  enclosed_tiling_cache[enclosed_cache_key] = copy.deepcopy(
                      retiled_placements
                  )
                if retiled_placements is not None:
                  river_cand['packed_flaps'].update(retiled_placements)
                  yield river_cand

        candidate_iterator = _get_candidates_with_pocket_packing(
            pocket_to_pack, current_step
        )
      else:
        candidate_iterator = find_river_candidates(
            current_step,
            grid_size,  # pyrefly: ignore[bad-argument-type]
            tracker,
            tree,
            pockets_info,
            traversal_plan,
            debug,
            fast_packing_debug,
            is_symmetric,
            symmetry_axes,
            scoring_config,
            pocket_start_cell,
            hugging_joint,
            config,
            base_square_dims_override=base_square_dims_override,
            template_options=template_options,
        )

      if utils_backtracking.try_to_advance(
          candidate_iterator,
          functools.partial(add_river_candidate, tracker),
          stack,
          types.RiverStackFrame,
          step_index,
          tracker,
          debug,
          stick=stick,
          config=config,
      ):
        river_stick_name = current_step['stick'].name
        logging.info(
            'River %s and its enclosed flaps placed.', river_stick_name
        )
        step_index += 1
        continue

      else:  # advance failed, need to backtrack
        if debug:
          debug.record_step(
              tracker,
              'BACKTRACK',
              f"No candidates found for {current_step['stick'].name}."
              ' Backtracking...',
              cand=None,
          )
        if config.easy_mode:
          logging.info(
              'Easy mode enabled: backtracking disabled. Stopping search.'
          )
          step_index = None
        else:
          step_index = utils_backtracking.try_to_backtrack(
              stack,
              functools.partial(add_river_candidate, tracker),
              functools.partial(remove_river_candidate, tracker),
              tracker,
              debug,
          )
        if step_index is None:
          return  # Backtracking failed
  except timeout.CooperativeTimeoutError:
    logging.warning(
        'Cooperative timeout reached in iter_solve_rivers_iteratively after %d'
        ' accepted solutions.',
        accepted_count,
    )
    raise


def solve_rivers_iteratively(
    tree: stick_figure.StickFigure,
    *,
    debug: debug_types.DebugConfig,
    grid_size: int | None = None,
    fast_packing_debug: bool = False,
    is_symmetric: bool = False,
    symmetry_axes: dict[str, bool] | None = None,
    scoring_config: scoring.ScoringConfig = scoring.ScoringConfig(),
    max_solutions: int = 1,
    config: config_lib.PackingConfig = config_lib.PackingConfig(),
    initial_tracker: utils_tracker.CreasePatternTracker | None = None,
    ignore_sticks: set[str] | None = None,
    pockets_info: list[packing_utils.FlapPocket] | None = None,
    default_traversal_plan: list[types.TraversalStep] | None = None,
    base_square_dims_override: dict[str, tuple[int, int]] | None = None,
    template_options: types.TemplateRiverOptions | None = None,
) -> list[utils_tracker.CreasePatternTracker]:
  """Packs adjacent rivers and enclosed flaps using iterative backtracking."""
  solutions: list[utils_tracker.CreasePatternTracker] = []
  try:
    for tracker_sol in iter_solve_rivers_iteratively(
        tree,
        debug=debug,
        grid_size=grid_size,
        fast_packing_debug=fast_packing_debug,
        is_symmetric=is_symmetric,
        symmetry_axes=symmetry_axes,
        scoring_config=scoring_config,
        max_solutions=max_solutions,
        config=config,
        initial_tracker=initial_tracker,
        ignore_sticks=ignore_sticks,
        pockets_info=pockets_info,
        default_traversal_plan=default_traversal_plan,
        base_square_dims_override=base_square_dims_override,
        template_options=template_options,
    ):
      solutions.append(tracker_sol)
  except timeout.CooperativeTimeoutError:
    if not solutions:
      raise
  return solutions
