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

"""Functions for generating hugging river candidates with path variations."""

from collections.abc import Iterator, Sequence

import numpy as np

from .. import rectangle as shaping_utils
from . import river as river_lib
from . import utils_packing_flaps


def has_hairpin_turn(
    subriver_paths: Sequence[Sequence[tuple[int, int]]],
) -> bool:
  """Returns True if any subriver path makes a 180-degree hairpin U-turn."""
  for path in subriver_paths:
    if len(path) < 4:
      continue
    for i in range(len(path) - 3):
      dr1 = path[i + 1][0] - path[i][0]
      dc1 = path[i + 1][1] - path[i][1]
      dr3 = path[i + 3][0] - path[i + 2][0]
      dc3 = path[i + 3][1] - path[i + 2][1]
      if dr1 == -dr3 and dc1 == -dc3 and (dr1 != 0 or dc1 != 0):
        return True
  return False


def has_separated_layers(
    subriver_paths: Sequence[Sequence[tuple[int, int]]],
) -> bool:
  """Returns True if any consecutive river layers are spatially disconnected."""
  if len(subriver_paths) <= 1:
    return False
  for i in range(len(subriver_paths) - 1):
    layer_a = set(subriver_paths[i])
    layer_b = subriver_paths[i + 1]
    has_contact = False
    for r, c in layer_b:
      for dr, dc in (
          (-1, 0),
          (1, 0),
          (0, -1),
          (0, 1),
          (-1, -1),
          (-1, 1),
          (1, -1),
          (1, 1),
      ):
        if (r + dr, c + dc) in layer_a:
          has_contact = True
          break
      if has_contact:
        break
    if not has_contact:
      return True
  return False


def has_hairpin_or_separated_layers(
    subriver_paths: Sequence[Sequence[tuple[int, int]]],
) -> bool:
  """Returns True if river has hairpins or disconnected layers."""
  return has_hairpin_turn(subriver_paths) or has_separated_layers(
      subriver_paths
  )


def extend_straight_to_boundary(
    start_node: tuple[int, int],
    direction: river_lib.Direction,
    grid_size: int,
    occupancy_grid: np.ndarray,
    current_path_prefix: list[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
  """Extends path from start_node in direction until boundary or obstacle.

  If current_path_prefix is not empty, start_node must be adjacent to the
  last cell in current_path_prefix to avoid gaps or diagonal moves; otherwise,
  this function will return None.

  Args:
    start_node: The cell (y, x) from which to start extending.
    direction: The direction to extend in.
    grid_size: The grid size.
    occupancy_grid: Boolean grid of occupied cells.
    current_path_prefix: The list of cells already in the path before extension.
      Can be an empty list if path starts with extension.

  Returns:
    The extended path as a list of cells, or None if a collision occurs or if
    start_node is not adjacent to current_path_prefix[-1].
  """
  path_suffix = []
  y, x = start_node
  if current_path_prefix:
    py, px = current_path_prefix[-1]
    if abs(y - py) + abs(x - px) > 1:
      return None  # start_node not adjacent to last cell in prefix
  dy, dx = direction.delta
  visited_cells = set(current_path_prefix)
  while 0 <= y < grid_size and 0 <= x < grid_size:
    if occupancy_grid[y, x] or (y, x) in visited_cells:
      return None  # Collision
    path_suffix.append((y, x))
    visited_cells.add((y, x))
    y += dy
    x += dx
  return tuple(current_path_prefix + path_suffix)


def generate_turn_away_variant(
    state: river_lib.NeighborState,
    direction: river_lib.Direction,
    away_turn: int,
    grid_size: int,
    occupancy_grid: np.ndarray,
    path_prefix: list[tuple[int, int]],
    visited_in_prefix: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
  """Generates a path variant by turning away from hugging side.

  Explores path that turns 90 degrees away from hugging side and continues
  straight to boundary.

  Args:
    state: The neighbor state at the current cell.
    direction: The current direction of hugging.
    away_turn: The turn direction away from hugging side (-1 or 1).
    grid_size: The grid size.
    occupancy_grid: Boolean grid of occupied cells.
    path_prefix: The list of cells already in the path before variant.
    visited_in_prefix: A set of cells in path_prefix.

  Returns:
    The variant path as a list of cells, or None if variant is not possible.
  """
  if away_turn == -1:
    away_direction = direction.turn_left()
  else:
    away_direction = direction.turn_right()
  away_cell_padded = state.opposite_cell
  away_cell = (away_cell_padded[0] - 1, away_cell_padded[1] - 1)
  if (
      not state.is_opposite_occupied
      and away_cell not in visited_in_prefix
      and 0 <= away_cell[0] < grid_size
      and 0 <= away_cell[1] < grid_size
  ):
    variant_path = extend_straight_to_boundary(
        away_cell, away_direction, grid_size, occupancy_grid, path_prefix
    )
    if variant_path is not None:
      return variant_path
  return None


def generate_continue_straight_variant(
    state: river_lib.NeighborState,
    direction: river_lib.Direction,
    grid_size: int,
    occupancy_grid: np.ndarray,
    path_prefix: list[tuple[int, int]],
    visited_in_prefix: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
  """Generates a path variant by continuing straight.

  If front is not occupied, explores path that continues straight to boundary.

  Args:
    state: The neighbor state at the current cell.
    direction: The current direction of hugging.
    grid_size: The grid size.
    occupancy_grid: Boolean grid of occupied cells.
    path_prefix: The list of cells already in the path before variant.
    visited_in_prefix: A set of cells in path_prefix.

  Returns:
    The variant path as a list of cells, or None if variant is not possible.
  """
  if not state.is_front_occupied:
    straight_cell_padded = state.front_cell
    straight_cell = (
        straight_cell_padded[0] - 1,
        straight_cell_padded[1] - 1,
    )
    if (
        0 <= straight_cell[0] < grid_size
        and 0 <= straight_cell[1] < grid_size
        and not occupancy_grid[straight_cell]
        and straight_cell not in visited_in_prefix
    ):
      variant_path = extend_straight_to_boundary(
          straight_cell,
          direction,
          grid_size,
          occupancy_grid,
          path_prefix,
      )
      if variant_path is not None:
        return variant_path
  return None


def generate_corner_variant(
    state: river_lib.NeighborState,
    direction: river_lib.Direction,
    hug_side: river_lib.HugSide,
    padded_grid: np.ndarray,
    grid_size: int,
    occupancy_grid: np.ndarray,
    path_prefix: list[tuple[int, int]],
    visited_in_prefix: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
  """Generates a path variant when a corner is detected.

  Args:
    state: The neighbor state at the current cell.
    direction: The current direction of hugging.
    hug_side: river_lib.HugSide.LEFT or river_lib.HugSide.RIGHT.
    padded_grid: The padded occupancy grid.
    grid_size: The grid size.
    occupancy_grid: Boolean grid of occupied cells.
    path_prefix: The list of cells already in the path before variant.
    visited_in_prefix: A set of cells in path_prefix.

  Returns:
    The variant path as a list of cells, or None if variant is not possible.
  """
  if state.is_front_occupied:  # inner corner
    # For inner corners, hugging algorithm turns away from hugging side.
    # Here we generate a variant that explores turning towards hugging side.
    if hug_side == river_lib.HugSide.LEFT:
      variant_direction = direction.turn_left()
    else:
      variant_direction = direction.turn_right()
    variant_cell_padded = state.hugging_cell
    variant_cell = (variant_cell_padded[0] - 1, variant_cell_padded[1] - 1)
    is_cell_occupied = state.is_hugging_occupied
  else:  # outer corner
    turn_cell_y, turn_cell_x = state.front_cell
    state_at_turn = river_lib.get_neighbor_states(
        turn_cell_y, turn_cell_x, direction, hug_side, padded_grid
    )
    if hug_side == river_lib.HugSide.LEFT:
      variant_direction = direction.turn_right()
    else:
      variant_direction = direction.turn_left()
    variant_cell_padded = state_at_turn.opposite_cell
    variant_cell = (variant_cell_padded[0] - 1, variant_cell_padded[1] - 1)
    is_cell_occupied = state_at_turn.is_opposite_occupied

  if (
      not is_cell_occupied
      and variant_cell not in visited_in_prefix
      and 0 <= variant_cell[0] < grid_size
      and 0 <= variant_cell[1] < grid_size
  ):
    variant_path = extend_straight_to_boundary(
        variant_cell,
        variant_direction,
        grid_size,
        occupancy_grid,
        path_prefix,
    )
    if variant_path is not None:
      return variant_path
  return None


def generate_path_variants(
    grid_size: int,
    occupancy_grid: np.ndarray,
    start_cell: tuple[int, int],
    hug_side: river_lib.HugSide,
    hug_boundaries: bool = False,
    area_reqs: dict[str, float] | None = None,
) -> Iterator[tuple[tuple[tuple[int, int], ...], river_lib.PathVariantType]]:
  """Generates hugging path and its variants.

  Args:
    grid_size: The size of grid this river is on.
    occupancy_grid: The occupancy grid from a CreasePatternTracker.
    start_cell: The starting cell (y, x) for path generation.
    hug_side: river_lib.HugSide.LEFT or river_lib.HugSide.RIGHT.
    hug_boundaries: If True, boundaries are treated as obstacles to hug.
    area_reqs: The area requirements for the river.

  Yields:
    Tuples of ((path), variant_type) representing path candidates.
  """
  try:
    standard_river = river_lib.HuggingRiver.from_contour(
        grid_size=grid_size,
        width=1,
        occupancy_grid=occupancy_grid,
        start_cell=start_cell,
        hug_side=hug_side,
        hug_boundaries=hug_boundaries,
    )
    for path in standard_river.subriver_paths:
      yield tuple(path), river_lib.PathVariantType.STANDARD
  except (river_lib.HuggingError, ValueError):
    return

  padded_grid = np.pad(
      occupancy_grid, 1, 'constant', constant_values=hug_boundaries
  )
  y, x = start_cell[0] + 1, start_cell[1] + 1
  has_zero_req = area_reqs is not None and any(
      r == 0 for r in area_reqs.values()
  )
  try:
    direction = river_lib.get_initial_direction((y, x), padded_grid, hug_side)
  except river_lib.HuggingError:
    return

  path_prefix = []
  visited_in_prefix = set()
  turn = -1 if hug_side == river_lib.HugSide.LEFT else 1
  away_turn = -turn

  for _ in range(grid_size * grid_size + 8):
    if not (1 <= y <= grid_size and 1 <= x <= grid_size):
      break
    cell = (y - 1, x - 1)
    if cell in visited_in_prefix:
      break
    path_prefix.append(cell)
    visited_in_prefix.add(cell)

    state = river_lib.get_neighbor_states(
        y, x, direction, hug_side, padded_grid
    )

    # 1. Turn-Away variant
    if not has_zero_req:
      variant_path = generate_turn_away_variant(
          state,
          direction,
          away_turn,
          grid_size,
          occupancy_grid,
          path_prefix,
          visited_in_prefix,
      )
      if variant_path:
        yield variant_path, river_lib.PathVariantType.TURN_AWAY

    # 2. Continue-Straight variant
    variant_path = generate_continue_straight_variant(
        state,
        direction,
        grid_size,
        occupancy_grid,
        path_prefix,
        visited_in_prefix,
    )
    if variant_path:
      yield variant_path, river_lib.PathVariantType.STRAIGHT

    # Apply hugging logic to advance to the next cell.
    try:
      next_y, next_x, next_direction = river_lib.apply_hugging_step(
          y,
          x,
          direction,
          hug_side,
          padded_grid,
      )
    except river_lib.HuggingError:
      return  # Hugging failed

    # If hugging direction changed, explore additional variants based on turn
    # type.
    if next_direction != direction and not has_zero_req:
      variant_path = generate_corner_variant(
          state,
          direction,
          hug_side,
          padded_grid,
          grid_size,
          occupancy_grid,
          path_prefix,
          visited_in_prefix,
      )
      if variant_path:
        yield variant_path, river_lib.PathVariantType.CORNER

    y, x, direction = next_y, next_x, next_direction


def generate_hugging_river_candidates(
    grid_size: int,
    width: int,
    occupancy_grid: np.ndarray,
    start_cell: tuple[int, int],
    hug_side: river_lib.HugSide,
    hug_boundaries: bool = False,
    area_reqs: dict[str, float] | None = None,
    allow_off_grid: bool = False,
    *,
    retry_opposite_hug_side: bool = False,
) -> Iterator[river_lib.HuggingRiver]:
  """Generates hugging river candidates including path variants.

  This function generates hugging river candidates. Beyond the standard hugging
  river that follows the contour of occupied cells, it explores path variations:
  - 'Turn-Away' variants: at each step, a path can deviate by turning 90
    degrees away from the hugged obstacle and then continue straight to
    boundary.
  - 'Continue-Straight' variants: when hugging logic dictates a turn, a path
    can instead continue straight if possible, then straight to boundary.

  If a side of the river has a zero-area requirement, path generation is
  restricted to 'Continue-Straight' and standard variants.

  After the first layer is generated, the hugging logic is applied to grow the
  river to full width by adding layers hugging previous ones.

  Args:
    grid_size: The size of grid this river is on.
    width: The width of river.
    occupancy_grid: The occupancy grid from a CreasePatternTracker.
    start_cell: The starting cell (y, x) for path generation. Must be an
      unoccupied cell adjacent to an occupied cell.
    hug_side: river_lib.HugSide.LEFT or river_lib.HugSide.RIGHT.
    hug_boundaries: If True, boundaries are treated as obstacles to hug.
    area_reqs: The area requirements for the river.
    allow_off_grid: If True, allows the river to temporarily go outside the grid
      boundaries. This is useful for "easy mode" packing where rivers might need
      to wrap around corners outside the current bounds.
    retry_opposite_hug_side: If True, retries candidate generation with the
      opposite hug side when `hug_side` yields no candidates.

  Yields:
    HuggingRiver objects representing path candidates.
  """
  if allow_off_grid:
    # Enough room for all subriver layers to grow off-grid.
    expansion = width
    expanded_occupancy = np.pad(occupancy_grid, expansion)
    expanded_grid_size = expanded_occupancy.shape[0]

    expanded_start_cell = (start_cell[0] + expansion, start_cell[1] + expansion)
  else:
    expansion = 0
    expanded_grid_size = grid_size
    expanded_occupancy = occupancy_grid
    expanded_start_cell = start_cell

  generated_path0_tuples = set()
  candidate_count = 0

  for path0_tuple, variant_type in generate_path_variants(
      expanded_grid_size,
      expanded_occupancy,
      expanded_start_cell,
      hug_side,
      hug_boundaries,
      area_reqs,
  ):
    if path0_tuple in generated_path0_tuples:
      continue
    generated_path0_tuples.add(path0_tuple)

    river = _try_grow_river_from_path(
        path0_tuple,
        variant_type,
        width,
        grid_size,
        expanded_grid_size,
        expanded_occupancy,
        expanded_start_cell,
        hug_side,
        expansion,
        hug_boundaries=hug_boundaries,
    )
    if river is not None:
      candidate_count += 1
      yield river

  if candidate_count == 0:
    for direction in river_lib.Direction:
      path = extend_straight_to_boundary(
          expanded_start_cell,
          direction,
          expanded_grid_size,
          expanded_occupancy,
          [],
      )
      if path is None or len(path) <= 1:
        continue
      if path in generated_path0_tuples:
        continue
      generated_path0_tuples.add(path)

      river = _try_grow_river_from_path(
          path,
          river_lib.PathVariantType.STRAIGHT,
          width,
          grid_size,
          expanded_grid_size,
          expanded_occupancy,
          expanded_start_cell,
          hug_side,
          expansion,
          initial_direction=direction,
      )
      if river is not None:
        candidate_count += 1
        yield river

  if candidate_count == 0 and retry_opposite_hug_side:
    alt_side = (
        river_lib.HugSide.RIGHT
        if hug_side == river_lib.HugSide.LEFT
        else river_lib.HugSide.LEFT
    )
    yield from generate_hugging_river_candidates(
        grid_size,
        width,
        occupancy_grid,
        start_cell,
        alt_side,
        hug_boundaries=hug_boundaries,
        area_reqs=area_reqs,
        allow_off_grid=allow_off_grid,
        retry_opposite_hug_side=False,
    )


def _find_next_border_start_ccw(
    prev_start: tuple[int, int],
    grid_size: int,
    occupancy_grid: np.ndarray,
    expansion: int = 0,
) -> tuple[int, int]:
  """Finds the next unoccupied cell CCW along the inner border.

  Args:
    prev_start: The start cell of the previous subriver layer (y, x).
    grid_size: The size of the expanded grid.
    occupancy_grid: Occupancy grid including previous layers.
    expansion: The expansion offset. The border is the original grid boundary.

  Returns:
    The first unoccupied border cell found going CCW from prev_start.

  Raises:
    river_lib.HuggingError: If no available border cell is found.
  """
  top = expansion
  left = expansion
  bottom = grid_size - 1 - expansion
  right = grid_size - 1 - expansion

  border_cells = []
  for col in range(left, right + 1):
    border_cells.append((top, col))
  for row in range(top + 1, bottom + 1):
    border_cells.append((row, right))
  for col in range(right - 1, left - 1, -1):
    border_cells.append((bottom, col))
  for row in range(bottom - 1, top, -1):
    border_cells.append((row, left))

  try:
    idx = border_cells.index(prev_start)
  except ValueError as exc:
    raise river_lib.HuggingError(
        f'Previous start cell {prev_start} is not on grid border.'
    ) from exc

  n = len(border_cells)
  for offset in range(1, n):
    candidate = border_cells[(idx - offset) % n]
    if not occupancy_grid[candidate[0], candidate[1]]:
      return candidate

  raise river_lib.HuggingError(
      'No available border cell found CCW from %s.' % (prev_start,)
  )


def _try_grow_river_from_path(
    path0_tuple: tuple[tuple[int, int], ...],
    variant_type: river_lib.PathVariantType,
    width: int,
    grid_size: int,
    expanded_grid_size: int,
    expanded_occupancy: np.ndarray,
    expanded_start_cell: tuple[int, int],
    hug_side: river_lib.HugSide,
    expansion: int,
    initial_direction: river_lib.Direction | None = None,
    hug_boundaries: bool = False,
) -> river_lib.HuggingRiver | None:
  """Tries to grow a width-1 path into a full-width hugging river.

  Args:
    path0_tuple: The width-1 path for layer 0.
    variant_type: The path variant type.
    width: The target river width.
    grid_size: The original grid size.
    expanded_grid_size: The expanded grid size.
    expanded_occupancy: The expanded occupancy grid.
    expanded_start_cell: The start cell in expanded coordinates.
    hug_side: The hugging side.
    expansion: The expansion offset.
    initial_direction: If provided, overrides the direction computed by
      get_initial_direction. Used for straight fallback paths where the growth
      direction should be perpendicular to the path direction.
    hug_boundaries: If True, boundaries are treated as obstacles to hug.

  Returns:
    A HuggingRiver if growth succeeds, None otherwise.
  """
  try:
    path0 = list(path0_tuple)
    subriver_paths: list[list[tuple[int, int]]] = [path0]
    obstacles_plus_river = expanded_occupancy.copy()
    layer_grid = np.zeros_like(expanded_occupancy)
    for y, x in path0:
      obstacles_plus_river[y, x] = True
      layer_grid[y, x] = True

    if initial_direction is not None:
      direction = initial_direction
    else:
      padded_occupancy_grid = np.pad(
          expanded_occupancy, 1, 'constant', constant_values=False
      )
      start_cell_padded = (
          expanded_start_cell[0] + 1,
          expanded_start_cell[1] + 1,
      )
      direction = river_lib.get_initial_direction(
          start_cell_padded, padded_occupancy_grid, hug_side
      )
    curr_start_cell = expanded_start_cell
    for i in range(1, width):
      if expansion > 0:
        curr_start_cell = _find_next_border_start_ccw(
            curr_start_cell,
            expanded_grid_size,
            obstacles_plus_river,
            expansion=expansion,
        )
      else:
        curr_start_cell = river_lib.get_next_start_cell(
            curr_start_cell,
            direction,
            hug_side,
            expanded_grid_size,
            obstacles_plus_river,
            i,
        )
      if expansion > 0:
        hugging_target = obstacles_plus_river
      else:
        hugging_target = layer_grid
      path_i, direction = river_lib.hugging_path_for_layer(
          expanded_grid_size,
          hugging_target,
          curr_start_cell,
          hug_side,
          hug_boundaries=hug_boundaries if expansion > 0 else False,
      )
      if expansion > 0 and hug_boundaries:
        path_i = river_lib.get_mirrored_path_for_symmetry(
            path_i, expanded_grid_size
        )
      if expansion == 0:
        layer_grid = np.zeros_like(expanded_occupancy)
      for y, x in path_i:
        rect = shaping_utils.Rectangle(
            name='', x_start=x, y_start=y, x_len=1, y_len=1
        )
        if not utils_packing_flaps.is_unoccupied(obstacles_plus_river, rect):
          raise river_lib.HuggingError(
              f'Hugging river layer {i} collides with obstacle at {(y, x)}'
          )
        utils_packing_flaps.set_grid_occupation(
            obstacles_plus_river, rect, True
        )
        if expansion == 0:
          utils_packing_flaps.set_grid_occupation(layer_grid, rect, True)
      subriver_paths.append(path_i)

    if expansion > 0:
      shifted_subriver_paths = []
      for path in subriver_paths:
        shifted_path = [(y - expansion, x - expansion) for y, x in path]
        clipped_path = [
            (y, x)
            for y, x in shifted_path
            if 0 <= y < grid_size and 0 <= x < grid_size
        ]
        if not clipped_path:
          raise river_lib.HuggingError(
              'Off-grid river path has no on-grid cells after clipping.'
          )
        shifted_subriver_paths.append(clipped_path)
      subriver_paths = shifted_subriver_paths
    return river_lib.HuggingRiver(
        grid_size=grid_size,
        width=width,
        subriver_paths=subriver_paths,
        hug_side=hug_side,
        variant_type=variant_type,
    )
  except (ValueError, river_lib.HuggingError):
    return None
