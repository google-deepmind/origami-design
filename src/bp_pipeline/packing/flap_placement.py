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

"""Functions for finding and validating flap placements."""

from collections.abc import Callable, Iterator
import itertools
import math
from typing import TypedDict

from ..crease_pattern import point as point_lib
from ..stick_figures import stick as stick_lib
from . import config as config_lib
from . import scoring
from . import types
from . import utils_packing_flaps
from . import utils_packing_svg
from . import utils_tracker


def _placement_touches_cells(
    placement: types.PolygonPlacement, cells: list[tuple[int, int]]
) -> bool:
  """Checks if placement overlaps with any cell in cells list.

  Args:
    placement: The placement to check.
    cells: A list of (x, y) coordinates of cells to check against.

  Returns:
    True if the placement overlaps with any cell in cells list, False
    otherwise.
  """
  offset_x, offset_y = placement.offset
  width, height = placement.size
  for row, column in cells:
    if (
        offset_y <= row < offset_y + height
        and offset_x <= column < offset_x + width
    ):
      return True
  return False


def check_symmetry_constraints(
    is_symmetric: bool,
    symmetry_axes: dict[str, bool] | None,
    current_stick_name: str,
    nx: int,
    new_width: int,
    grid_size: int,
    treat_as_on_axis: bool = False,
    remove_central_crease: bool = False,
) -> bool:
  """Checks if a rectangle placement satisfies symmetry constraints.

  If symmetry is enabled, and the rectangle is on a symmetry axis, its center
  must lie on the grid's vertical center line. If it is not on a symmetry
  axis, it must lie entirely to the left of the center line.

  When remove_central_crease is True, non-axis rectangles must stop one column
  before the axis to reserve space for central crease diagonals.

  Args:
      is_symmetric: If True, symmetry constraints are applied.
      symmetry_axes: A dict indicating which stick names are on a symmetry axis.
      current_stick_name: The identifier for the new polygon being placed.
      nx: The x-coordinate of the top-left corner of the placement.
      new_width: The width of the placement.
      grid_size: The size of the grid.
      treat_as_on_axis: If True, validates the placement under the assumption
        that it lies on a symmetry axis (i.e. centered on grid_size / 2.0),
        regardless of the value in symmetry_axes.
      remove_central_crease: If True, non-axis placements must stop one column
        before the symmetry axis.

  Returns:
      True if the placement is valid according to symmetry rules, False
      otherwise.
  """
  if not is_symmetric or not symmetry_axes:
    return True

  is_on_axis = symmetry_axes[current_stick_name] or treat_as_on_axis

  if is_on_axis:
    return math.isclose(
        nx + new_width / 2.0,
        grid_size / 2.0,
        abs_tol=point_lib.DEFAULT_TOLERANCE,
    )
  else:
    axis_boundary = grid_size / 2.0 - (1 if remove_central_crease else 0)
    return nx + new_width <= axis_boundary


def generate_paired_placements(
    tracker: utils_tracker.CreasePatternTracker,
    current_stick_name: str,
    stick_name_to_dims: dict[str, tuple[int, int]],
    is_symmetric: bool,
    symmetry_axes: dict[str, bool] | None,
    validate_candidate_fn: Callable[..., types.Candidate | None],
) -> list[types.Candidate]:
  """Generates paired placements for a rectangle on the symmetry axis.

  When packing a symmetric crease pattern, if a flap is not on the symmetry
  line of the tree, it can be packed on the symmetry line if its symmetric
  counterpart is also packed on the symmetry line. This function finds pairs of
  valid, non-overlapping placements for rectangles of the same size, where both
  are centered on the grid's vertical symmetry axis.

  Args:
      tracker: The CreasePatternTracker containing the current grid state.
      current_stick_name: The identifier for the new polygon being placed.
      stick_name_to_dims: A mapping from stick name to polygon dimensions.
      is_symmetric: If True, symmetry constraints are applied.
      symmetry_axes: A dict indicating which stick names are on a symmetry axis.
      validate_candidate_fn: The function to use for validating candidates.

  Returns:
      A list of valid paired `Candidate` dictionaries.
  """
  on_axis_slots: list[types.Candidate] = []
  new_width, new_height = stick_name_to_dims[current_stick_name]
  grid_center_x = tracker.grid_size / 2.0
  fixed_nx = int(grid_center_x - new_width / 2.0)
  step_size = 1

  for ny in range(-new_height, tracker.grid_size + 1, step_size):
    candidate = validate_candidate_fn(
        tracker,
        current_stick_name,
        (fixed_nx, ny),
        (new_width, new_height),
        is_symmetric=is_symmetric,
        symmetry_axes=symmetry_axes,
        treat_as_on_axis=True,
    )
    if candidate:
      on_axis_slots.append(candidate)

  paired_candidates = []
  if len(on_axis_slots) >= 2:
    for i, slot_i in enumerate(on_axis_slots):
      for slot_j in on_axis_slots[i + 1 :]:
        p1 = slot_i['placement']
        p2 = slot_j['placement']

        if (p1.offset[1] + p1.size[1] <= p2.offset[1]) or (
            p2.offset[1] + p2.size[1] <= p1.offset[1]
        ):
          paired_candidates.append({
              'type': 'PAIRED_ON_AXIS',
              'stick_name': current_stick_name,
              'placement': p1,
              'partner_placement': p2,
          })
          paired_candidates.append({
              'type': 'PAIRED_ON_AXIS',
              'stick_name': current_stick_name,
              'placement': p2,
              'partner_placement': p1,
          })
  return paired_candidates


class ZoneProperties(TypedDict):
  """Describes the properties of an adjacent placement zone.

  Attributes:
    anchor: The (x, y) coordinate of the zone's anchor point.
    slide_axis: The axis along which the polygon will slide (x or y).
  """

  anchor: tuple[int, int]
  slide_axis: str


def validate_candidate(
    tracker: utils_tracker.CreasePatternTracker,
    current_stick_name: str,
    offset: tuple[int, int],
    size: tuple[int, int],
    is_symmetric: bool = False,
    symmetry_axes: dict[str, bool] | None = None,
    treat_as_on_axis: bool = False,
    remove_central_crease: bool = False,
) -> types.Candidate | None:
  """Validates a single potential placement for a polygon.

  This function checks for five primary conditions:
  1.  The polygon's center must align with the grid's integer coordinates.
  2.  The polygon's center must be within or on the grid boundaries.
  3.  At least one corner of the polygon must be on the grid (or on the border).
  4.  If symmetry is enabled, symmetry constraints must be satisfied:
      - If on a symmetry axis, center must lie on grid's vertical center line.
      - If not on a symmetry axis, polygon cannot cross vertical center line.
  5.  The polygon must not overlap with any existing polygons.

  Args:
      tracker: The CreasePatternTracker containing the current grid state.
      current_stick_name: The identifier for the new polygon being placed.
      offset: The proposed top-left (x, y) coordinate for the placement.
      size: The proposed (width, height) for the placement.
      is_symmetric: If True, symmetry constraints are applied.
      symmetry_axes: A dict indicating which stick names are on a symmetry axis.
      treat_as_on_axis: If True, treat as if on symmetry axis for validation.
      remove_central_crease: If True, non-axis placements must stop one column
        before the symmetry axis.

  Returns:
      A `Candidate` dict if the placement is valid, or None if it is invalid.
  """

  nx, ny = offset
  new_width, new_height = size
  grid_size = tracker.grid_size

  # 1. Check for even size.
  if (new_width | new_height) & 1:
    return None

  # 2. Check if the center is within grid boundaries.
  center_x = nx + new_width / 2.0
  center_y = ny + new_height / 2.0
  if not (0 <= center_x <= grid_size and 0 <= center_y <= grid_size):
    return None

  # 3. Symmetry check
  if not check_symmetry_constraints(
      is_symmetric,
      symmetry_axes,
      current_stick_name,
      nx,
      new_width,
      grid_size,
      treat_as_on_axis,
      remove_central_crease=remove_central_crease,
  ):
    return None

  # 4. Check if at least one corner is on the grid.
  if not (
      (0 <= nx <= grid_size or 0 <= nx + new_width <= grid_size)
      and (0 <= ny <= grid_size or 0 <= ny + new_height <= grid_size)
  ):
    return None

  # 5. Check for collision with other polygons or occupied cells.
  occupancy_grid = tracker.occupancy_grid
  container_height, container_width = occupancy_grid.shape
  x_start = max(0, nx)
  y_start = max(0, ny)
  x_end = min(container_width, nx + new_width)
  y_end = min(container_height, ny + new_height)
  if (
      x_start < x_end
      and y_start < y_end
      and occupancy_grid[y_start:y_end, x_start:x_end].any()
  ):
    return None

  placement = types.PolygonPlacement.from_grid_coords(
      offset=offset, size=size, grid_size=grid_size
  )

  # If all checks pass, the candidate is valid.
  valid_candidate: types.Candidate = {
      'type': 'FLAP',
      'stick_name': current_stick_name,
      'placement': placement,
  }
  return valid_candidate


def is_valid_level_shifter_corner(
    *, off_x: int, off_y: int, grid_size: int, size: int
) -> bool:
  """Checks if a corner placement of the given size fits inside the grid."""
  return (
      off_x >= 0
      and off_y >= 0
      and off_x + size <= grid_size
      and off_y + size <= grid_size
  )


def is_valid_ls_center(
    placement: types.PolygonPlacement, grid_size: int, stick_width: int
) -> bool:
  """Checks if a placement center leaves room for level-shifter repacking."""
  min_dist = utils_packing_flaps.level_shifter_center_margin(stick_width)
  cx_int = placement.offset[0] + placement.size[0] // 2
  cy_int = placement.offset[1] + placement.size[1] // 2
  return (
      cx_int >= min_dist
      and cx_int <= grid_size - min_dist
      and cy_int >= min_dist
      and cy_int <= grid_size - min_dist
  )


def is_center_axis_on_paper(
    placement: types.PolygonPlacement, grid_size: int
) -> bool:
  """Checks if the center column of a non-square flap is on the paper.

  For a rectangle at offset (ox, oy) with size (w, h), the vertical center
  column at ox + w//2 must be within the grid bounds [0, grid_size].

  Args:
    placement: The candidate placement.
    grid_size: The size of the grid.

  Returns:
    True if the center column is within the paper bounds.
  """
  ox, _ = placement.offset
  w, _ = placement.size
  center_col = ox + w // 2
  return 0 <= center_col <= grid_size


def find_all_unoccupied_placements(
    tracker: utils_tracker.CreasePatternTracker,
    current_stick_name: str,
    stick_name_to_dims: dict[str, tuple[int, int]],
    step_size: int = 1,
    is_symmetric: bool = False,
    symmetry_axes: dict[str, bool] | None = None,
    remove_central_crease: bool = False,
) -> Iterator[types.Candidate]:
  """Finds all valid placements by scanning the entire grid space.

  This function performs a brute-force search, iterating through every
  possible top-left coordinate for the new polygon within a reasonable
  range of the grid (including off-board placements whose centers are still
  on the grid). It calls a validation function for each potential spot.

  Args:
    tracker: The CreasePatternTracker containing the current grid state.
    current_stick_name: The identifier for the new polygon being placed.
    stick_name_to_dims: A mapping from stick name to polygon dimensions.
    step_size: The step size to use for the grid search.
    is_symmetric: If True, symmetry constraints are applied.
    symmetry_axes: A dict indicating which stick names are on a symmetry axis.
    remove_central_crease: If True, non-axis placements must stop one column
      before the symmetry axis.

  Yields:
      Valid placement candidates.
  """
  new_width, new_height = stick_name_to_dims[current_stick_name]
  if (new_width | new_height) & 1:
    return

  grid_size = tracker.grid_size
  if step_size == 1:
    half_w = new_width // 2
    half_h = new_height // 2
    nx_min = -half_w
    nx_max = grid_size - half_w
    ny_min = -half_h
    ny_max = grid_size - half_h
    if is_symmetric and symmetry_axes:
      if symmetry_axes[current_stick_name]:
        if grid_size & 1:
          return
        nx_min = nx_max = grid_size // 2 - half_w
      else:
        axis_max_nx = (
            grid_size // 2 - (1 if remove_central_crease else 0) - new_width
        )
        nx_max = min(nx_max, axis_max_nx)

    occupancy_grid = tracker.occupancy_grid
    container_height, container_width = occupancy_grid.shape
    rows = occupancy_grid.tolist()
    for nx in range(nx_min, nx_max + 1):
      if not (0 <= nx <= grid_size or 0 <= nx + new_width <= grid_size):
        continue
      x_start = 0 if nx < 0 else nx
      x_end = (
          container_width
          if nx + new_width > container_width
          else nx + new_width
      )
      row_occ = (
          [any(r[x_start:x_end]) for r in rows] if x_start < x_end else None
      )
      for ny in range(ny_min, ny_max + 1):
        if not (0 <= ny <= grid_size or 0 <= ny + new_height <= grid_size):
          continue
        if row_occ is not None:
          y_start = 0 if ny < 0 else ny
          y_end = (
              container_height
              if ny + new_height > container_height
              else ny + new_height
          )
          if y_start < y_end and any(row_occ[y_start:y_end]):
            continue
        yield {
            'type': 'FLAP',
            'stick_name': current_stick_name,
            'placement': types.PolygonPlacement.from_grid_coords(
                offset=(nx, ny),
                size=(new_width, new_height),
                grid_size=grid_size,
            ),
        }
    return

  for nx in range(-new_width, grid_size + 1, step_size):
    for ny in range(-new_height, grid_size + 1, step_size):
      valid_candidate = validate_candidate(
          tracker,
          current_stick_name,
          (nx, ny),
          (new_width, new_height),
          is_symmetric=is_symmetric,
          symmetry_axes=symmetry_axes,
          remove_central_crease=remove_central_crease,
      )
      if valid_candidate:
        yield valid_candidate


def _get_adjacent_zones(
    prev_placement: types.PolygonPlacement, new_size: tuple[int, int]
) -> dict[str, ZoneProperties]:
  """Calculates the four zones where a new polygon can touch an existing one.

  For each of the four faces (top, bottom, left, right) of the existing
  polygon, this function computes an "anchor" placement for the new polygon
  and the axis along which it can "slide".

  Args:
    prev_placement: The existing PolygonPlacement to place against.
    new_size: The (width, height) of the new polygon to be placed.

  Returns:
    A dictionary mapping zone names to their anchor and slide axis properties.
  """
  px, py = prev_placement.offset
  pw, ph = prev_placement.size
  new_width, new_height = new_size

  zones: dict[str, ZoneProperties] = {
      'bottom': {
          'anchor': (px + (pw - new_width) // 2, py + ph),
          'slide_axis': 'x',
      },
      'top': {
          'anchor': (px + (pw - new_width) // 2, py - new_height),
          'slide_axis': 'x',
      },
      'right': {
          'anchor': (px + pw, py + (ph - new_height) // 2),
          'slide_axis': 'y',
      },
      'left': {
          'anchor': (px - new_width, py + (ph - new_height) // 2),
          'slide_axis': 'y',
      },
  }
  return zones


def _generate_sliding_placements(
    tracker: utils_tracker.CreasePatternTracker,
    current_stick_name: str,
    new_size: tuple[int, int],
    anchor_offset: tuple[int, int],
    slide_axis: str,
    slide_range: tuple[int, int],
    increment: int,
    is_symmetric: bool = False,
    symmetry_axes: dict[str, bool] | None = None,
    remove_central_crease: bool = False,
) -> Iterator[types.Candidate]:
  """Generates and validates all placements by 'sliding' along a face.

  Starting from a central anchor point on a face of an existing polygon, this
  function searches for valid placements by moving step-by-step in both
  directions along the specified axis, up to the computed boundaries of the
  face.

  Args:
    tracker: The CreasePatternTracker containing the current grid state.
    current_stick_name: The identifier for the new polygon being placed.
    new_size: The (width, height) of the new polygon.
    anchor_offset: The starting (x, y) top-left coordinate for the search.
    slide_axis: The axis ('x' or 'y') to slide along.
    slide_range: The (min, max) coordinates for the slide axis.
    increment: The step size for the sliding search.
    is_symmetric: If True, symmetry constraints are applied.
    symmetry_axes: A dict indicating which stick names are on a symmetry axis.
    remove_central_crease: If True, non-axis placements must stop one column
      before the symmetry axis.

  Yields:
    Valid `Candidate` placements found along the face.
  """
  slide_idx = 0 if slide_axis == 'x' else 1
  range_min, range_max = slide_range

  # Start at the anchor and search in the positive direction
  current_offset = list(anchor_offset)
  while current_offset[slide_idx] <= range_max:
    candidate = validate_candidate(
        tracker,
        current_stick_name,
        tuple(current_offset),  # pyrefly: ignore[bad-argument-type]
        new_size,
        is_symmetric=is_symmetric,
        symmetry_axes=symmetry_axes,
        remove_central_crease=remove_central_crease,
    )
    if candidate:
      yield candidate
    current_offset[slide_idx] += increment

  # Start *just past* the anchor and search in the negative direction
  current_offset = list(anchor_offset)
  current_offset[slide_idx] -= increment
  while current_offset[slide_idx] >= range_min:
    candidate = validate_candidate(
        tracker,
        current_stick_name,
        tuple(current_offset),  # pyrefly: ignore[bad-argument-type]
        new_size,
        is_symmetric=is_symmetric,
        symmetry_axes=symmetry_axes,
        remove_central_crease=remove_central_crease,
    )
    if candidate:
      yield candidate
    current_offset[slide_idx] -= increment


def find_analytically_valid_placements(
    *,
    tracker: utils_tracker.CreasePatternTracker,
    current_stick_name: str,
    constraints: dict[str, types.Constraint],
    stick_name_to_dims: dict[str, tuple[int, int]],
    increment: int = 1,
    is_symmetric: bool = False,
    symmetry_axes: dict[str, bool] | None = None,
    remove_central_crease: bool = False,
) -> Iterator[types.Candidate]:
  """Finds valid placements for a polygon that must touch existing ones.

  This function iterates through all 'EQUAL 0' distance constraints, which
  signify that the new polygon must be adjacent to an existing one. For each
  such constraint, it generates and validates all possible placements that
  touch any of the four faces of the existing polygon, including "sliding"
  along those faces.

  Args:
      tracker: The CreasePatternTracker containing the current grid state.
      current_stick_name: The identifier for the new polygon being placed.
      constraints: A dictionary of constraints against already-placed polygons.
      stick_name_to_dims: A mapping from stick name to polygon dimensions.
      increment: The step size to use for the sliding search.
      is_symmetric: If True, symmetry constraints are applied.
      symmetry_axes: A dict indicating which stick names are on a symmetry axis.
      remove_central_crease: If True, non-axis placements must stop one column
        before the symmetry axis.

  Yields:
      Unique, valid `Candidate` dictionaries, sorted with placements whose
      centers are on the border appearing first.
  """
  unique_placements = set()
  new_size = stick_name_to_dims[current_stick_name]
  new_width, new_height = new_size
  found_placements_by_adj = False

  for prev_stick_name, constraint in constraints.items():
    if not (
        constraint['type'] == types.ConstraintType.EQUAL
        and constraint['value'] == 0
    ):
      continue

    if prev_stick_name not in tracker.packed_polygons:
      continue
    found_placements_by_adj = True
    prev_placement = tracker.packed_polygons[prev_stick_name]
    px, py = prev_placement.offset
    pw, ph = prev_placement.size

    zones = _get_adjacent_zones(prev_placement, new_size)

    for _, props in zones.items():
      if props['slide_axis'] == 'x':
        slide_range = (px - new_width + 1, px + pw - 1)
      else:  # slide_axis == 'y'
        slide_range = (py - new_height + 1, py + ph - 1)

      for candidate in _generate_sliding_placements(
          tracker=tracker,
          current_stick_name=current_stick_name,
          new_size=new_size,
          anchor_offset=props['anchor'],
          slide_axis=props['slide_axis'],
          slide_range=slide_range,
          increment=increment,
          is_symmetric=is_symmetric,
          symmetry_axes=symmetry_axes,
          remove_central_crease=remove_central_crease,
      ):
        if candidate['placement'] not in unique_placements:
          unique_placements.add(candidate['placement'])
          yield candidate

  if not found_placements_by_adj:
    # This happens if analytical was called but no neighbors are packed yet.
    # Revert to grid scan to place the first flap of a pocket.
    yield from find_all_unoccupied_placements(
        tracker,
        current_stick_name,
        stick_name_to_dims,
        is_symmetric=is_symmetric,
        symmetry_axes=symmetry_axes,
        remove_central_crease=remove_central_crease,
    )


def find_possible_flap_placements(
    *,
    tracker: utils_tracker.CreasePatternTracker,
    stick: stick_lib.Stick,
    current_stick_name: str,
    constraints: dict[str, types.Constraint],
    stick_name_to_dims: dict[str, tuple[int, int]],
    start_on_corner: bool = True,
    is_symmetric: bool = False,
    symmetry_axes: dict[str, bool] | None = None,
    allow_paired_axis_placement: bool = False,
    enforce_proximity_constraints: bool = True,
    max_candidates: int = 1000,
    scoring_config: scoring.ScoringConfig = scoring.ScoringConfig(),
    river_adjacent_cells: list[tuple[int, int]] | None = None,
    config: config_lib.PackingConfig = config_lib.PackingConfig(),
    locked_long_side_flaps: set[str] | None = None,
) -> Iterator[types.Candidate]:
  """Determines which placement strategy to use and finds all valid placements.

  This function acts as a dispatcher. For the very first polygon, it uses a
  special placement function. For all subsequent polygons, it checks if there
  are any 'must touch' (EQUAL 0) constraints and
  `enforce_proximity_constraints` is True. If so, it uses the analytical
  solver; otherwise, it falls back to a brute-force scan of the entire grid.

  Args:
      tracker: The CreasePatternTracker containing the current grid state.
      stick: The stick to place.
      current_stick_name: The identifier for the new polygon being placed.
      constraints: A dictionary of constraints against already-placed polygons.
      stick_name_to_dims: A mapping from stick name to polygon dimensions.
      start_on_corner: If True, the first piece will be placed with its center
        at the grid origin (0,0).
      is_symmetric: If True, symmetry constraints are applied.
      symmetry_axes: A dict indicating which stick names are on a symmetry axis.
      allow_paired_axis_placement: If True, allow paired placement on axis.
      enforce_proximity_constraints: If True, enforce EQUAL 0 constraints by
        calling the analytical solver. Otherwise, use brute-force scan.
      max_candidates: The maximum number of placement candidates to generate.
      scoring_config: The function used to score and sort the candidates.
      river_adjacent_cells: If provided, it contains a list of grid coordinates
        for cells that are adjacent to a river. If it is the first flap of a
        pocket being placed, only placements that occupy at least one of the
        cells in `river_adjacent_cells` will be considered. This forces the flap
        to be placed in contact with a river bank.
      config: The configuration options for packing.
      locked_long_side_flaps: If provided, flap names whose long side is locked.
        Placements are filtered to keep the center axis on paper.

  Yields:
      Valid `Candidate` dictionaries found by the chosen strategy.
  """
  is_level_shifter = config.use_level_shifter and stick.requires_level_shifter
  is_first_polygon = not tracker.packed_polygons
  is_on_axis = (
      is_symmetric
      and symmetry_axes
      and symmetry_axes.get(current_stick_name, False)
  )
  if (
      is_first_polygon
      and start_on_corner
      and not tracker.occupancy_grid.any()
      and not is_on_axis
      and not is_level_shifter
  ):
    yield _place_first_polygon_at_origin(current_stick_name, stick_name_to_dims)
    return

  has_adjacency_constraint = any(
      c['type'] == types.ConstraintType.EQUAL and c['value'] == 0
      for c in constraints.values()
  )

  if has_adjacency_constraint and enforce_proximity_constraints:
    # Get all geometrically valid placements adjacent to the specified polygons.
    candidates = find_analytically_valid_placements(
        tracker=tracker,
        current_stick_name=current_stick_name,
        constraints=constraints,
        stick_name_to_dims=stick_name_to_dims,
        is_symmetric=is_symmetric,
        symmetry_axes=symmetry_axes,
        remove_central_crease=config.remove_central_crease,
    )
  else:
    candidates = find_all_unoccupied_placements(
        tracker,
        current_stick_name,
        stick_name_to_dims,
        is_symmetric=is_symmetric,
        symmetry_axes=symmetry_axes,
        remove_central_crease=config.remove_central_crease,
    )
  if is_first_polygon and river_adjacent_cells:
    candidates = filter(
        lambda c: _placement_touches_cells(
            c['placement'], river_adjacent_cells
        ),
        candidates,
    )

  if is_level_shifter:
    stick_width = round(stick.width)
    candidates = filter(
        lambda c: is_valid_ls_center(
            c['placement'], tracker.grid_size, stick_width
        ),
        candidates,
    )

  if locked_long_side_flaps and current_stick_name in locked_long_side_flaps:
    candidates = filter(
        lambda c: is_center_axis_on_paper(c['placement'], tracker.grid_size),
        candidates,
    )

  candidates = scoring_config.prune_candidates(tracker, candidates)
  yield from scoring_config.sort_candidates(
      tracker, itertools.islice(candidates, max_candidates)
  )

  if (
      is_symmetric
      and symmetry_axes
      and not symmetry_axes[current_stick_name]
      and allow_paired_axis_placement
  ):
    yield from generate_paired_placements(
        tracker,
        current_stick_name,
        stick_name_to_dims,
        is_symmetric,
        symmetry_axes,
        validate_candidate,
    )


def _place_first_polygon_at_origin(
    current_stick_name: str, stick_name_to_dims: dict[str, tuple[int, int]]
) -> types.Candidate:
  """Generates a single placement for the first polygon centered at (0,0).

  Args:
      current_stick_name: The identifier for the new polygon being placed.
      stick_name_to_dims: A mapping from stick name to polygon dimensions.

  Returns:
      A single `Candidate` for the placement at the origin.
  """
  width, height = stick_name_to_dims[current_stick_name]
  placement = types.PolygonPlacement.from_grid_coords(
      offset=(int(-width / 2.0), int(-height / 2.0)),
      size=(width, height),
      grid_size=1,
  )
  candidate: types.Candidate = {
      'type': 'FLAP',
      'stick_name': current_stick_name,
      'placement': placement,
  }
  return candidate


def add_polygon_candidate(
    tracker: utils_tracker.CreasePatternTracker, cand: types.Candidate
) -> list[str]:
  """Adds a polygon candidate to the tracker.

  Args:
    tracker: The CreasePatternTracker to modify.
    cand: The candidate to add.

  Returns:
    A list of object IDs added to the tracker.
  """
  obj_id = cand['stick_name']
  if cand['type'] == 'PAIRED_ON_AXIS':
    partner_id = (
        f'{cand["stick_name"]}{utils_packing_svg.PolygonIDSuffix.COPY.value}'
    )
    tracker.add_polygon(obj_id, cand['placement'])
    tracker.add_polygon(partner_id, cand['partner_placement'])
    return [obj_id, partner_id]
  else:
    tracker.add_polygon(obj_id, cand['placement'])
    return [obj_id]


def remove_polygon_candidate(
    tracker: utils_tracker.CreasePatternTracker, object_ids: list[str]
):
  """Removes polygon candidates from the tracker.

  Args:
    tracker: The CreasePatternTracker to modify.
    object_ids: A list of object IDs to remove.
  """
  for object_id in object_ids:
    tracker.remove_by_id(object_id)
