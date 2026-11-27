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

"""Utilities for scoring polygon placements."""

from collections.abc import Callable, Iterable, Iterator, Sequence
import dataclasses
from typing import Protocol

import numpy as np

from . import river as river_lib
from . import types
from . import utils_tracker


@dataclasses.dataclass(frozen=True)
class PackingContext:
  """A class that holds context about the problem being solved."""

  base_square_dims: dict[str, tuple[int, int]]


class ScoringFn(Protocol):
  """A callable that scores a polygon placement."""

  def __call__(
      self,
      tracker: utils_tracker.CreasePatternTracker,
      placement: types.PolygonPlacement,
      context: PackingContext | None,
  ) -> float | tuple[float, ...]:
    """Scores a (valid) polygon placement."""


def _key_from_scoring_fn(
    tracker: utils_tracker.CreasePatternTracker,
    scoring_fn: ScoringFn,
    context: PackingContext | None,
) -> Callable[[types.Candidate], float | tuple[float, ...]]:
  """Returns a key that can be used for sorting candidates by max score."""

  def key(candidate: types.Candidate):
    score = scoring_fn(tracker, candidate['placement'], context)
    if isinstance(score, Sequence):
      return tuple(-subscore for subscore in score)
    return -score

  return key


class HierarchicalScore(ScoringFn):
  """A class that represents a hierarchical score."""

  def __init__(self, scoring_fns: Sequence[ScoringFn]):
    assert (
        len(scoring_fns) > 1
    ), 'Need at least two scoring functions to make a hierarchical score.'
    self._scoring_fns = scoring_fns

  def __call__(
      self,
      tracker: utils_tracker.CreasePatternTracker,
      placement: types.PolygonPlacement,
      context: PackingContext | None,
  ) -> tuple[float, ...]:
    all_scores = []
    for scoring_fn in self._scoring_fns:
      score = scoring_fn(tracker, placement, context)
      if isinstance(score, Sequence):
        all_scores.extend(score)
      else:
        all_scores.append(score)
    return tuple(all_scores)


def is_center_on_border_score(
    tracker: utils_tracker.CreasePatternTracker,
    placement: types.PolygonPlacement,
    context: PackingContext | None,
) -> float:
  del tracker, context
  return float(placement.center.is_border)


def ccw_border_continuity_score(
    tracker: utils_tracker.CreasePatternTracker,
    placement: types.PolygonPlacement,
    context: PackingContext | None,
) -> float:
  """Scores by proximity in CCW border order to the last placed polygon."""
  del context

  grid_size = tracker.grid_size
  offset_x, offset_y = placement.offset
  width, height = placement.size

  x_min = max(0, offset_x)
  x_max = min(grid_size - 1, offset_x + width - 1)
  y_min = max(0, offset_y)
  y_max = min(grid_size - 1, offset_y + height - 1)

  border_cells = set()
  for x in range(x_min, x_max + 1):
    if y_min == 0:
      border_cells.add((0, x))
    if y_max == grid_size - 1:
      border_cells.add((grid_size - 1, x))
  for y in range(y_min, y_max + 1):
    if x_min == 0:
      border_cells.add((y, 0))
    if x_max == grid_size - 1:
      border_cells.add((y, grid_size - 1))

  if not border_cells:
    return float('-inf')

  border_indices = []
  for cell in border_cells:
    try:
      border_indices.append(
          river_lib.get_border_cell_counter_clockwise_index(cell, grid_size)
      )
    except ValueError:
      continue

  if not border_indices:
    return float('-inf')

  if not tracker.packed_polygons:
    return -float(min(border_indices))

  last_placement = list(tracker.packed_polygons.values())[-1]
  last_ox, last_oy = last_placement.offset
  last_w, last_h = last_placement.size

  last_x_min = max(0, last_ox)
  last_x_max = min(grid_size - 1, last_ox + last_w - 1)
  last_y_min = max(0, last_oy)
  last_y_max = min(grid_size - 1, last_oy + last_h - 1)

  last_border_cells = set()
  for x in range(last_x_min, last_x_max + 1):
    if last_y_min == 0:
      last_border_cells.add((0, x))
    if last_y_max == grid_size - 1:
      last_border_cells.add((grid_size - 1, x))
  for y in range(last_y_min, last_y_max + 1):
    if last_x_min == 0:
      last_border_cells.add((y, 0))
    if last_x_max == grid_size - 1:
      last_border_cells.add((y, grid_size - 1))

  last_indices = []
  for cell in last_border_cells:
    try:
      last_indices.append(
          river_lib.get_border_cell_counter_clockwise_index(cell, grid_size)
      )
    except ValueError:
      continue

  if not last_indices:
    return -float(min(border_indices))

  num_border = 4 * grid_size - 4
  if num_border == 0:
    return 0.0

  ref_idx = max(last_indices)
  min_dist = num_border
  for idx in border_indices:
    dist = (idx - ref_idx) % num_border
    min_dist = min(min_dist, dist)

  return -float(min_dist)


def random_score(
    tracker: utils_tracker.CreasePatternTracker,
    placement: types.PolygonPlacement,
    context: PackingContext | None,
) -> float:
  """Returns a random score."""
  del tracker, placement, context
  return np.random.uniform(0, 1)


@dataclasses.dataclass(frozen=True)
class PerimeterScore(ScoringFn):
  """Scores placements based on adjacency to occupied cells."""

  include_border: bool = True
  restrict_to_subproblem: bool = False

  def _get_grid(
      self,
      tracker: utils_tracker.CreasePatternTracker,
      context: PackingContext | None,
  ) -> np.ndarray:
    """Returns the occupancy grid to use for scoring."""
    if not self.restrict_to_subproblem:
      return tracker.occupancy_grid
    else:
      if context is None:
        raise ValueError(
            'Context is required when use_subproblem_polygons=True.'
        )
      subproblem_polygon_ids = context.base_square_dims.keys()

      grid = np.zeros_like(tracker.occupancy_grid, dtype=bool)
      for poly_id, placement in tracker.packed_polygons.items():
        if poly_id in subproblem_polygon_ids:
          offset_x, offset_y = placement.offset
          width, height = placement.size
          x_start = max(0, offset_x)
          y_start = max(0, offset_y)
          x_end = offset_x + width
          y_end = offset_y + height
          assert (
              x_end > 0 and y_end > 0
          ), f'Polygon {poly_id} is not on the grid: {placement}'
          grid[y_start:y_end, x_start:x_end] = True

      return grid

  def __call__(
      self,
      tracker: utils_tracker.CreasePatternTracker,
      placement: types.PolygonPlacement,
      context: PackingContext | None,
  ) -> float:
    """Calculates the perimeter score for a placement."""
    grid = self._get_grid(tracker, context)
    offset_x, offset_y = placement.offset
    width, height = placement.size
    grid_size = tracker.grid_size

    left = max(0, offset_x)
    right = min(grid_size, offset_x + width)
    top = max(0, offset_y)
    bottom = min(grid_size, offset_y + height)

    border_fill_value = self.include_border

    if top == 0:
      top_edge = np.full(right - left, fill_value=border_fill_value, dtype=bool)
    else:
      top_edge = grid[top - 1, left:right]

    if bottom == grid_size:
      bottom_edge = np.full(
          right - left, fill_value=border_fill_value, dtype=bool
      )
    else:
      bottom_edge = grid[bottom, left:right]

    if left == 0:
      left_edge = np.full(
          bottom - top, fill_value=border_fill_value, dtype=bool
      )
    else:
      left_edge = grid[top:bottom, left - 1]

    if right == grid_size:
      right_edge = np.full(
          bottom - top, fill_value=border_fill_value, dtype=bool
      )
    else:
      right_edge = grid[top:bottom, right]

    all_edges = [top_edge, bottom_edge, left_edge, right_edge]

    total_perimeter = sum(edge.size for edge in all_edges)
    occupied_perimeter = sum(np.count_nonzero(edge) for edge in all_edges)
    empty_perimeter = total_perimeter - occupied_perimeter

    score = float(-empty_perimeter)
    return score


perimeter_score = PerimeterScore(
    include_border=True, restrict_to_subproblem=False
)
subproblem_perimeter_score = PerimeterScore(
    include_border=False, restrict_to_subproblem=True
)


def _chessboard_distance_transform(mask: np.ndarray) -> np.ndarray:
  """Exact chessboard (L_inf) distance to nearest False cell, matching SciPy."""
  rows, cols = mask.shape
  inf = rows + cols + 1
  dist = np.where(mask, inf, 0).astype(np.int32)
  if not np.any(mask):
    return dist
  if np.all(mask):
    return np.full((rows, cols), -1, dtype=np.int32)

  # Vectorized row-by-row 2-pass Rosenfeld-Pfaltz Chamfer DP (exact for L_inf).
  for r in range(rows):
    if r > 0:
      prev = dist[r - 1]
      best_up = prev.copy()
      best_up[1:] = np.minimum(best_up[1:], prev[:-1])
      best_up[:-1] = np.minimum(best_up[:-1], prev[1:])
      dist[r] = np.minimum(dist[r], best_up + 1)
    row = dist[r]
    for c in range(1, cols):
      if row[c - 1] + 1 < row[c]:
        row[c] = row[c - 1] + 1

  for r in range(rows - 1, -1, -1):
    if r + 1 < rows:
      nxt = dist[r + 1]
      best_dn = nxt.copy()
      best_dn[1:] = np.minimum(best_dn[1:], nxt[:-1])
      best_dn[:-1] = np.minimum(best_dn[:-1], nxt[1:])
      dist[r] = np.minimum(dist[r], best_dn + 1)
    row = dist[r]
    for c in range(cols - 2, -1, -1):
      if row[c + 1] + 1 < row[c]:
        row[c] = row[c + 1] + 1
  return dist


def _largest_possible_flap(occupancy_grid: np.ndarray) -> int:
  """Finds the side length of the largest square that can be placed on the grid."""
  if occupancy_grid.size == 0:
    return 0

  if not np.any(occupancy_grid):
    return np.inf  # type: ignore[return-value]

  rows, cols = occupancy_grid.shape

  inverted_mask = ~occupancy_grid
  pad_width = max(rows, cols)
  padded_inverted = np.pad(
      inverted_mask, pad_width=pad_width, mode='constant', constant_values=True
  )

  dist_grid_cells = _chessboard_distance_transform(padded_inverted)

  dist_centers = dist_grid_cells[
      pad_width : pad_width + rows, pad_width : pad_width + cols
  ]

  max_k_cell = np.max(dist_centers)
  s_odd = 2 * max_k_cell - 1

  dist_for_vertices = dist_grid_cells[
      pad_width - 1 : pad_width + rows + 1, pad_width - 1 : pad_width + cols + 1
  ]
  quad_tl = dist_for_vertices[:-1, :-1]
  quad_tr = dist_for_vertices[:-1, 1:]
  quad_bl = dist_for_vertices[1:, :-1]
  quad_br = dist_for_vertices[1:, 1:]

  dist_vertices = np.minimum(
      np.minimum(quad_tl, quad_tr), np.minimum(quad_bl, quad_br)
  )

  max_k_vertex = np.max(dist_vertices)
  s_even = 2 * max_k_vertex

  return max(0, int(s_odd), int(s_even))


def _placement_lt_norm(
    placement: types.PolygonPlacement,
    existing_placement: types.PolygonPlacement,
):
  if placement.center.norm < existing_placement.center.norm:
    return True
  elif placement.center.norm == existing_placement.center.norm:
    if placement.center.x < existing_placement.center.x:
      return True
  return False


def _placement_lt_lexical(
    placement: types.PolygonPlacement,
    existing_placement: types.PolygonPlacement,
):
  if (placement.offset[1], placement.offset[0]) < (
      existing_placement.offset[1],
      existing_placement.offset[0],
  ):
    return True
  return False


class CanonicalFlapOrderScore(ScoringFn):
  """Scoring function that forces repeated flap sizes to be placed in an order."""

  def __init__(
      self,
      lt: Callable[[types.PolygonPlacement, types.PolygonPlacement], bool],
  ):
    self._lt = lt

  def __call__(
      self,
      tracker: utils_tracker.CreasePatternTracker,
      placement: types.PolygonPlacement,
      context: PackingContext | None,
  ) -> float:
    subproblem_polygons = tracker.packed_polygons.keys()
    if context is not None:
      subproblem_polygons &= context.base_square_dims.keys()

    for flap_id in subproblem_polygons:
      existing_placement = tracker.packed_polygons[flap_id]
      if existing_placement.size == placement.size:
        if self._lt(placement, existing_placement):
          return 0.0
    return 1.0


canonical_flap_order_norm_score = CanonicalFlapOrderScore(lt=_placement_lt_norm)
canonical_flap_order_lexical_score = CanonicalFlapOrderScore(
    lt=_placement_lt_lexical
)


class ImpossibleRemainingFlapsScore(ScoringFn):
  """Score that returns 0 if a placement has impossible remaining flaps and 1 otherwise."""

  def _remaining_squares(
      self,
      tracker: utils_tracker.CreasePatternTracker,
      base_square_dims: dict[str, tuple[int, int]],
  ) -> dict[str, tuple[int, int]]:
    """Returns the number of squares that can still be placed."""
    return {
        leaf_id: dims
        for leaf_id, dims in base_square_dims.items()
        if leaf_id not in tracker.packed_polygons
    }

  def __call__(
      self,
      tracker: utils_tracker.CreasePatternTracker,
      placement: types.PolygonPlacement,
      context: PackingContext | None,
  ) -> float:
    """Returns 0 if a placement should be pruned and 1 if it should be kept."""
    if context is None:
      raise ValueError('Context is required for pruning score.')
    largest_remaining_square = max(
        min(dims)
        for dims in self._remaining_squares(
            tracker, context.base_square_dims
        ).values()
    )
    if largest_remaining_square > _largest_possible_flap(
        tracker.occupancy_grid
    ):
      return 0.0
    return 1.0


@dataclasses.dataclass(frozen=True)
class ScoringConfig:
  """A class that holds configuration for scoring."""

  scoring_fn: ScoringFn | None = is_center_on_border_score
  pruning_fn: ScoringFn | None = None
  context: PackingContext | None = None

  def should_keep(
      self,
      tracker: utils_tracker.CreasePatternTracker,
      placement: types.PolygonPlacement,
  ) -> bool:
    """Returns True if the candidate should be kept instead of pruned."""
    if self.pruning_fn is None:
      return True
    score = self.pruning_fn(tracker, placement, self.context)
    if isinstance(score, Sequence):
      return all(bool(s) for s in score)
    else:
      return bool(score)

  def prune_candidates(
      self,
      tracker: utils_tracker.CreasePatternTracker,
      candidates: Iterable[types.Candidate],
  ) -> Iterator[types.Candidate]:
    """Filters candidates based on the configured pruning function."""
    it = iter(candidates)
    if self.pruning_fn is None:
      return it
    return (
        candidate
        for candidate in it
        if self.should_keep(tracker, candidate['placement'])
    )

  def sort_candidates(
      self,
      tracker: utils_tracker.CreasePatternTracker,
      candidates: Iterable[types.Candidate],
  ) -> list[types.Candidate]:
    """Sorts candidates based on the scoring config."""
    candidate_list = list(candidates)
    if self.scoring_fn is not None:
      score_key = _key_from_scoring_fn(tracker, self.scoring_fn, self.context)
      candidate_list.sort(key=score_key)
    return candidate_list
