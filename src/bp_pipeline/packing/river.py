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

"""Defines River objects that can be placed on a CreasePatternTracker."""

import collections
from collections.abc import Sequence
import dataclasses
import enum
import functools
import itertools
from typing import Any, TypedDict

import more_itertools
import numpy as np

from .. import rectangle as shaping_utils
from . import utils_packing_flaps

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W


def _label_4connected(mask: np.ndarray) -> tuple[np.ndarray, int]:
  """Labels 4-connected True components in a 2D boolean array (1-indexed)."""
  rows, cols = mask.shape
  labels = np.zeros((rows, cols), dtype=np.int32)
  if not np.any(mask):
    return labels, 0
  mask_list = mask.tolist()
  cur_label = 0
  for r0 in range(rows):
    row_mask = mask_list[r0]
    for c0 in range(cols):
      if not row_mask[c0] or labels[r0, c0] != 0:
        continue
      cur_label += 1
      labels[r0, c0] = cur_label
      queue = collections.deque([(r0, c0)])
      while queue:
        r, c = queue.popleft()
        if r > 0 and mask_list[r - 1][c] and labels[r - 1, c] == 0:
          labels[r - 1, c] = cur_label
          queue.append((r - 1, c))
        if r + 1 < rows and mask_list[r + 1][c] and labels[r + 1, c] == 0:
          labels[r + 1, c] = cur_label
          queue.append((r + 1, c))
        if c > 0 and mask_list[r][c - 1] and labels[r, c - 1] == 0:
          labels[r, c - 1] = cur_label
          queue.append((r, c - 1))
        if c + 1 < cols and mask_list[r][c + 1] and labels[r, c + 1] == 0:
          labels[r, c + 1] = cur_label
          queue.append((r, c + 1))
  return labels, cur_label


def _get_connected_components_areas(
    forbidden: np.ndarray,
    seeds1: Sequence[tuple[int, int]],
    seeds2: Sequence[tuple[int, int]],
) -> tuple[int, int]:
  """Computes unoccupied area on each side of river reachable from river bank."""
  if not seeds1 and not seeds2:
    return 0, 0
  labels, _ = _label_4connected(~forbidden)
  counts = np.bincount(labels.ravel())
  counts[0] = 0
  area1 = 0
  if seeds1:
    r, c = zip(*seeds1)
    area1 = int(counts[np.unique(labels[list(r), list(c)])].sum())
  area2 = 0
  if seeds2:
    r, c = zip(*seeds2)
    area2 = int(counts[np.unique(labels[list(r), list(c)])].sum())
  return area1, area2


class RiverDict(TypedDict):
  """A dictionary representation of a River."""

  name: str
  rectangles: list[shaping_utils.RectangleDict]
  river_type: str
  width: int
  grid_size: int


def count_corners_in_mask(mask: np.ndarray) -> int:
  """Counts how many of the grid's 4 corner cells are True in the mask.

  Args:
    mask: A boolean mask.

  Returns:
    The number of grid corners that are True in the mask.
  """
  grid_size = mask.shape[0]
  if grid_size == 1:
    return 1 if mask[0, 0] else 0
  return mask[[0, 0, -1, -1], [0, -1, 0, -1]].sum()


class Direction(enum.IntEnum):
  """Enum for cardinal directions.

  Attributes:
    NORTH: North direction.
    EAST: East direction.
    SOUTH: South direction.
    WEST: West direction.
  """

  NORTH = 0
  EAST = 1
  SOUTH = 2
  WEST = 3

  def turn_left(self) -> 'Direction':
    """Returns the direction after turning 90 degrees to the left.

    Returns:
      The new direction.
    """
    return Direction((self.value - 1 + 4) % 4)

  def turn_right(self) -> 'Direction':
    """Returns the direction after turning 90 degrees to the right.

    Returns:
      The new direction.
    """
    return Direction((self.value + 1 + 4) % 4)

  @property
  def delta(self) -> tuple[int, int]:
    """Returns the (dy, dx) tuple for this direction.

    Returns:
      A tuple (dy, dx) representing movement in this direction.
    """
    return DIRS[self.value]


class RiverOrientation(enum.StrEnum):
  HORIZONTAL = 'horizontal'
  VERTICAL = 'vertical'


class LShapeType(enum.StrEnum):
  """Defines corner connection type for L-shaped rivers."""

  TOP_LEFT = 'top_left'
  TOP_RIGHT = 'top_right'
  BOTTOM_LEFT = 'bottom_left'
  BOTTOM_RIGHT = 'bottom_right'


class HugSide(enum.StrEnum):
  LEFT = 'left'
  RIGHT = 'right'


class HuggingError(Exception):
  """Error during hugging process."""


class PathVariantType(enum.StrEnum):
  """Enum for types of hugging river path variants."""

  STANDARD = 'standard'
  TURN_AWAY = 'turn_away'
  STRAIGHT = 'straight'
  CORNER = 'corner'


def _neighbor_free_mask(
    river_mask: np.ndarray, forbidden_grid: np.ndarray
) -> np.ndarray:
  """Returns boolean mask of unoccupied 4-neighbors of river_mask."""
  grid_size = river_mask.shape[0]
  neighbor_mask = np.zeros((grid_size, grid_size), dtype=bool)
  if grid_size > 1:
    neighbor_mask[:-1, :] |= river_mask[1:, :]
    neighbor_mask[1:, :] |= river_mask[:-1, :]
    neighbor_mask[:, :-1] |= river_mask[:, 1:]
    neighbor_mask[:, 1:] |= river_mask[:, :-1]
  neighbor_mask &= ~forbidden_grid
  return neighbor_mask


def _get_seeds_from_river_mask(
    *,
    grid_size: int,
    river_mask: np.ndarray,
    side1_mask: np.ndarray,
    forbidden_grid: np.ndarray,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
  """Finds seeds on river banks and classify them in side1 and side2.

  Args:
    grid_size: The size of the grid.
    river_mask: A boolean mask indicating the river bank.
    side1_mask: A boolean mask indicating the side 1.
    forbidden_grid: A boolean mask indicating the forbidden grid.

  Returns:
    A tuple of two sets of seeds, one for side 1 and one for side 2.
  """
  neighbor_mask = _neighbor_free_mask(
      river_mask[:grid_size, :grid_size],
      forbidden_grid[:grid_size, :grid_size],
  )
  s1_rows, s1_cols = np.where(neighbor_mask & side1_mask)
  s2_rows, s2_cols = np.where(neighbor_mask & ~side1_mask)
  return (
      set(zip(s1_rows.tolist(), s1_cols.tolist())),
      set(zip(s2_rows.tolist(), s2_cols.tolist())),
  )


@dataclasses.dataclass(frozen=True)
class River:
  """A straight river that can be placed on a grid.

  This class represents straight rivers that span the grid horizontally or
  vertically.
  A river is defined by an orientation, an offset, and a width. A river must
  start from a border of the grid and end at a border of the grid. We currently
  do not support rivers looping on themselves.
  If orientation is HORIZONTAL, offset is the y-coordinate of top edge,
  and river spans from y=offset to y=offset+width-1 for all x.
  If orientation is VERTICAL, offset is the x-coordinate of left edge,
  and river spans from x=offset to x=offset+width-1 for all y.

  Attributes:
    width: The width of the river.
    offset: The coordinate of the top (horizontal) or left (vertical) edge.
    orientation: Whether the river is 'horizontal' or 'vertical'.
    grid_size: The size of the grid this river is on.
  """

  width: int
  offset: int
  orientation: RiverOrientation
  grid_size: int

  def __post_init__(self):
    object.__setattr__(self, 'width', int(self.width))
    object.__setattr__(self, 'offset', int(self.offset))
    object.__setattr__(self, 'grid_size', int(self.grid_size))
    if self.width <= 0:
      raise ValueError('River width must be positive.')

  def asdict(self, name: str) -> RiverDict:
    """Converts a River to a dict for S2 serialization."""
    return {
        'name': name,
        'rectangles': [r.asdict() for r in self.rectangles],
        'river_type': 'straight',
        'width': self.width,
        'grid_size': self.grid_size,
    }

  @functools.cached_property
  def rectangle(self) -> shaping_utils.Rectangle:
    """Returns the rectangle occupied by the river."""
    if self.orientation == RiverOrientation.HORIZONTAL:
      return shaping_utils.Rectangle(
          name='',
          x_start=0,
          y_start=self.offset,
          x_len=self.grid_size,
          y_len=self.width,
      )
    else:  # VERTICAL
      return shaping_utils.Rectangle(
          name='',
          x_start=self.offset,
          y_start=0,
          x_len=self.width,
          y_len=self.grid_size,
      )

  @functools.cached_property
  def rectangles(self) -> list[shaping_utils.Rectangle]:
    """Returns a list containing the single rectangle occupied by the river."""
    return [self.rectangle]

  @functools.cached_property
  def bounds(self) -> tuple[int, int, int, int]:
    """Returns offset_x, offset_y, width, height for grid occupation."""
    return (
        int(self.rectangle.x_start),
        int(self.rectangle.y_start),
        int(self.rectangle.x_len),
        int(self.rectangle.y_len),
    )

  @functools.cached_property
  def start_cell(self) -> tuple[int, int]:
    """Returns the start cell of the river."""
    if self.orientation == RiverOrientation.HORIZONTAL:
      return (self.offset, 0)
    else:
      return (0, self.offset)

  def mirror(self, grid_size: int) -> 'River':
    """Returns a mirrored version of the river if it is vertical.

    Args:
      grid_size: The grid size.

    Returns:
      A mirrored version of the river.
    """

    if self.orientation == RiverOrientation.HORIZONTAL:
      return self
    else:
      return River(
          width=self.width,
          offset=grid_size - self.width - self.offset,
          orientation=self.orientation,
          grid_size=grid_size,
      )

  def get_corner_counts_on_river_sides(
      self, occupancy_grid: np.ndarray
  ) -> dict[str, int]:
    """Computes number of available corners on each side of the river.

    For VERTICAL rivers, returns {'left': left_corners, 'right': right_corners}.
    For HORIZONTAL rivers, returns {'top': top_corners, 'bottom':
    bottom_corners}.
    An available corner is an unoccupied grid corner reachable from river bank.

    Args:
      occupancy_grid: The boolean occupancy grid from a CreasePatternTracker.

    Returns:
      A dictionary containing corner counts for each side.
    """
    grid_size = occupancy_grid.shape[0]
    river_mask = np.zeros((grid_size, grid_size), dtype=bool)
    utils_packing_flaps.set_grid_occupation(river_mask, self.rectangle, True)
    forbidden_grid = np.logical_or(occupancy_grid, river_mask)

    side1_mask = np.zeros((grid_size, grid_size), dtype=bool)
    if self.offset > 0:
      if self.orientation == RiverOrientation.VERTICAL:
        side1_mask[:, : self.offset] = True
      else:  # HORIZONTAL
        side1_mask[: self.offset, :] = True
    side1_mask = np.logical_and(side1_mask, ~river_mask)

    seeds1, seeds2 = _get_seeds_from_river_mask(
        grid_size=grid_size,
        river_mask=river_mask,
        side1_mask=side1_mask,
        forbidden_grid=forbidden_grid,
    )
    corners1 = _count_corners_from_seeds(seeds1, grid_size, forbidden_grid)
    corners2 = _count_corners_from_seeds(seeds2, grid_size, forbidden_grid)

    if self.orientation == RiverOrientation.VERTICAL:
      return {'left': corners1, 'right': corners2}
    else:  # HORIZONTAL
      return {'top': corners1, 'bottom': corners2}

  def compute_unoccupied_areas_on_river_sides(
      self, occupancy_grid: np.ndarray
  ) -> tuple[int, int]:
    """Computes unoccupied area on each side of the river.

    For VERTICAL rivers, returns (left_unoccupied_area, right_unoccupied_area).
    For HORIZONTAL rivers, returns (top_unoccupied_area,
    bottom_unoccupied_area).
    Unoccupied area is defined as number of cells reachable from river bank
    without crossing any occupied cells.

    Args:
      occupancy_grid: The boolean occupancy grid from a CreasePatternTracker.

    Returns:
      A tuple containing the two unoccupied areas.
    """
    grid_size = occupancy_grid.shape[0]
    forbidden_grid = occupancy_grid.copy()
    utils_packing_flaps.set_grid_occupation(
        forbidden_grid, self.rectangle, True
    )
    seeds1 = []
    seeds2 = []
    if self.orientation == RiverOrientation.VERTICAL:
      if self.offset > 0:
        seeds1 = [(r, self.offset - 1) for r in range(grid_size)]
      if self.offset + self.width < grid_size:
        seeds2 = [(r, self.offset + self.width) for r in range(grid_size)]
    else:  # HORIZONTAL
      if self.offset > 0:
        seeds1 = [(self.offset - 1, c) for c in range(grid_size)]
      if self.offset + self.width < grid_size:
        seeds2 = [(self.offset + self.width, c) for c in range(grid_size)]
    return _get_connected_components_areas(forbidden_grid, seeds1, seeds2)


@dataclasses.dataclass(frozen=True)
class LShapedRiver:
  """An L-shaped river defined by width, grid_size, type and turn coords.

  L-shaped rivers are defined by a width, grid_size, type, and turn
  coordinates. The type determines the shape of the L-shape, and the turn
  coordinates specify the location of the top-left corner of the turn square.
  The turn square is the section of the grid that is shared between the two
  rectangles of the L-shape.

  Attributes:
    width: The width of the river.
    grid_size: The size of the grid this river is on.
    l_shape_type: The type of L-shape.
    turn_x: The x-coordinate of the top-left corner of the turn square.
    turn_y: The y-coordinate of the top-left corner of the turn square.
  """

  width: int
  grid_size: int
  l_shape_type: LShapeType
  turn_x: int  # x-coord of top-left of turn square
  turn_y: int  # y-coord of top-left of turn square

  def __post_init__(self):
    object.__setattr__(self, 'width', int(self.width))
    object.__setattr__(self, 'grid_size', int(self.grid_size))
    object.__setattr__(self, 'turn_x', int(self.turn_x))
    object.__setattr__(self, 'turn_y', int(self.turn_y))
    if self.width <= 0:
      raise ValueError('River width must be positive.')
    if not (0 <= self.turn_x < self.grid_size) or not (
        0 <= self.turn_y < self.grid_size
    ):
      raise ValueError('Turn coordinates must be within grid.')

  def asdict(self, name: str) -> RiverDict:
    """Converts an LShapedRiver to a dict for S2 serialization."""
    return {
        'name': name,
        'rectangles': [r.asdict() for r in self.rectangles],
        'river_type': 'l_shaped',
        'width': self.width,
        'grid_size': self.grid_size,
    }

  @functools.cached_property
  def rectangles(self) -> list[shaping_utils.Rectangle]:
    """Returns the two rectangles forming the L-shape.

    Returns:
      A list of two rectangles forming the L-shape.
    """
    if self.l_shape_type == LShapeType.BOTTOM_LEFT:
      r1 = shaping_utils.Rectangle(
          name='',
          x_start=0,
          y_start=self.turn_y,
          x_len=self.turn_x + self.width,
          y_len=self.width,
      )
      r2 = shaping_utils.Rectangle(
          name='',
          x_start=self.turn_x,
          y_start=0,
          x_len=self.width,
          y_len=self.turn_y + self.width,
      )
    elif self.l_shape_type == LShapeType.BOTTOM_RIGHT:
      r1 = shaping_utils.Rectangle(
          name='',
          x_start=self.turn_x,
          y_start=self.turn_y,
          x_len=self.grid_size - self.turn_x,
          y_len=self.width,
      )
      r2 = shaping_utils.Rectangle(
          name='',
          x_start=self.turn_x,
          y_start=0,
          x_len=self.width,
          y_len=self.turn_y + self.width,
      )
    elif self.l_shape_type == LShapeType.TOP_LEFT:
      r1 = shaping_utils.Rectangle(
          name='',
          x_start=0,
          y_start=self.turn_y,
          x_len=self.turn_x + self.width,
          y_len=self.width,
      )
      r2 = shaping_utils.Rectangle(
          name='',
          x_start=self.turn_x,
          y_start=self.turn_y,
          x_len=self.width,
          y_len=self.grid_size - self.turn_y,
      )
    elif self.l_shape_type == LShapeType.TOP_RIGHT:
      r1 = shaping_utils.Rectangle(
          name='',
          x_start=self.turn_x,
          y_start=self.turn_y,
          x_len=self.grid_size - self.turn_x,
          y_len=self.width,
      )
      r2 = shaping_utils.Rectangle(
          name='',
          x_start=self.turn_x,
          y_start=self.turn_y,
          x_len=self.width,
          y_len=self.grid_size - self.turn_y,
      )
    else:
      raise ValueError(f'Unknown LShapeType: {self.l_shape_type}')
    return [r1, r2]

  @functools.cached_property
  def start_cell(self) -> tuple[int, int]:
    """Returns the start cell of the river."""
    if self.l_shape_type == LShapeType.BOTTOM_LEFT:
      return (self.turn_y, 0)
    elif self.l_shape_type == LShapeType.TOP_LEFT:
      return (self.turn_y, 0)
    elif self.l_shape_type == LShapeType.BOTTOM_RIGHT:
      return (self.turn_y, self.grid_size - 1)
    elif self.l_shape_type == LShapeType.TOP_RIGHT:
      return (0, self.turn_x)
    raise ValueError(f'Unknown LShapeType: {self.l_shape_type}')

  def mirror(self, grid_size: int) -> 'LShapedRiver':
    """Returns a mirrored version of the L-shaped river.

    Args:
      grid_size: The grid size.

    Returns:
      A mirrored version of the L-shaped river.
    """
    if self.l_shape_type == LShapeType.BOTTOM_LEFT:
      new_type = LShapeType.BOTTOM_RIGHT
    elif self.l_shape_type == LShapeType.BOTTOM_RIGHT:
      new_type = LShapeType.BOTTOM_LEFT
    elif self.l_shape_type == LShapeType.TOP_LEFT:
      new_type = LShapeType.TOP_RIGHT
    elif self.l_shape_type == LShapeType.TOP_RIGHT:
      new_type = LShapeType.TOP_LEFT
    else:
      raise ValueError(f'Unknown LShapeType: {self.l_shape_type}')
    return LShapedRiver(
        width=self.width,
        grid_size=grid_size,
        l_shape_type=new_type,
        turn_x=grid_size - self.width - self.turn_x,
        turn_y=self.turn_y,
    )

  def get_corner_counts_on_river_sides(
      self, occupancy_grid: np.ndarray
  ) -> dict[str, int]:
    """Computes number of available corners on each side of the river.

    Returns {'inner': inner_corners, 'outer': outer_corners}.
    An available corner is an unoccupied grid corner reachable from river bank.

    Args:
      occupancy_grid: The boolean occupancy grid from a CreasePatternTracker.

    Returns:
      A dictionary containing corner counts for inner and outer side.
    """
    grid_size = occupancy_grid.shape[0]
    if grid_size != self.grid_size:
      raise ValueError('Occupancy grid size does not match river grid size.')

    river_mask = np.zeros((grid_size, grid_size), dtype=bool)
    for rect in self.rectangles:
      utils_packing_flaps.set_grid_occupation(river_mask, rect, True)

    inner_mask_area, _ = self.get_area_masks(grid_size)
    forbidden_grid = np.logical_or(occupancy_grid, river_mask)

    inner_seeds, outer_seeds = _get_seeds_from_river_mask(
        grid_size=grid_size,
        river_mask=river_mask,
        side1_mask=inner_mask_area,
        forbidden_grid=forbidden_grid,
    )

    corners1 = _count_corners_from_seeds(inner_seeds, grid_size, forbidden_grid)
    corners2 = _count_corners_from_seeds(outer_seeds, grid_size, forbidden_grid)

    return {'inner': corners1, 'outer': corners2}

  def _get_inner_slice(self) -> tuple[slice, slice]:
    """Returns grid slices for the inner corner of the L-shaped river.

    Returns:
      A tuple of two slices for the inner corner of the L-shaped river.
      The first slice is for the inner corner of the first rectangle, and the
      second slice is for the inner corner of the second rectangle.
    """
    if self.l_shape_type == LShapeType.BOTTOM_LEFT:
      inner_slice = np.s_[0 : self.turn_y, 0 : self.turn_x]
    elif self.l_shape_type == LShapeType.BOTTOM_RIGHT:
      inner_slice = np.s_[
          0 : self.turn_y, self.turn_x + self.width : self.grid_size
      ]
    elif self.l_shape_type == LShapeType.TOP_LEFT:
      inner_slice = np.s_[
          self.turn_y + self.width : self.grid_size, 0 : self.turn_x
      ]
    elif self.l_shape_type == LShapeType.TOP_RIGHT:
      inner_slice = np.s_[
          self.turn_y + self.width : self.grid_size,
          self.turn_x + self.width : self.grid_size,
      ]
    else:
      raise ValueError(f'Unknown LShapeType: {self.l_shape_type}')
    return inner_slice

  def get_area_masks(self, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns masks for inner and outer areas.

    Args:
      grid_size: The grid size.

    Returns:
      A tuple of (inner_mask, outer_mask), where True indicates cells
      belonging to inner and outer areas respectively.
    """
    river_mask = np.zeros((grid_size, grid_size), dtype=bool)
    for rect in self.rectangles:
      utils_packing_flaps.set_grid_occupation(river_mask, rect, True)
    inner_mask = np.zeros((grid_size, grid_size), dtype=bool)
    inner_slice = self._get_inner_slice()
    inner_mask[inner_slice] = True
    outer_mask = ~(river_mask | inner_mask)
    return inner_mask, outer_mask

  def compute_unoccupied_areas_on_river_sides(
      self, occupancy_grid: np.ndarray
  ) -> tuple[int, int]:
    """Computes unoccupied area on each side of the river reachable from river bank.

    It splits grid into inner corner area and outer area, and finds cells
    reachable from river banks in each area via BFS.
    Returns (inner_corner_unoccupied_area, outer_corner_unoccupied_area).

    Args:
      occupancy_grid: The boolean occupancy grid from a CreasePatternTracker.

    Returns:
      A tuple containing the two unoccupied areas.
    """
    grid_size = occupancy_grid.shape[0]
    if grid_size != self.grid_size:
      raise ValueError('Occupancy grid size does not match river grid size.')

    river_mask = np.zeros((grid_size, grid_size), dtype=bool)
    for rect in self.rectangles:
      utils_packing_flaps.set_grid_occupation(river_mask, rect, True)

    inner_mask, _ = self.get_area_masks(grid_size)
    forbidden_grid = np.logical_or(occupancy_grid, river_mask)
    neighbor_mask = _neighbor_free_mask(river_mask, forbidden_grid)
    if not np.any(neighbor_mask):
      return 0, 0
    inner_rows, inner_cols = np.where(neighbor_mask & inner_mask)
    outer_rows, outer_cols = np.where(neighbor_mask & ~inner_mask)
    return _get_connected_components_areas(
        forbidden_grid,
        list(zip(inner_rows.tolist(), inner_cols.tolist())),
        list(zip(outer_rows.tolist(), outer_cols.tolist())),
    )


def get_initial_direction(
    cell: tuple[int, int], padded_grid: np.ndarray, hug_side: HugSide
) -> Direction:
  """Finds initial direction for wall following.

  The initial direction is along an edge of an occupied cell, such that
  the occupied cell is on the left/right as specified by hug_side.

  Args:
    cell: The starting cell (y, x) in padded coordinates.
    padded_grid: The occupancy grid with padding.
    hug_side: HugSide.LEFT or HugSide.RIGHT, to determine scanning direction.

  Returns:
    The initial direction.

  Raises:
    HuggingError: If no adjacent occupied neighbor is found for hugging.
  """
  for direction in Direction:
    y, x = cell
    if hug_side == HugSide.LEFT:
      hugging_dir = direction.turn_left()
    else:
      hugging_dir = direction.turn_right()
    dy, dx = hugging_dir.delta
    if padded_grid[y + dy, x + dx]:
      return direction
  raise HuggingError(
      f'Start cell ({cell[0]-1}, {cell[1]-1}) has no adjacent occupied'
      ' neighbors for hugging.'
  )


def get_next_start_cell(
    current_path_start_cell: tuple[int, int],
    direction: Direction,
    hug_side: HugSide,
    grid_size: int,
    occupancy_grid: np.ndarray,
    next_path_index: int,
) -> tuple[int, int]:
  """Computes the start cell for the next layer of a wide hugging river.

  Args:
    current_path_start_cell: The start cell of the path layer just generated.
    direction: The initial direction of hugging path generation for layer just
      generated.
    hug_side: HugSide.LEFT or HugSide.RIGHT, determining hugging direction.
    grid_size: The size of grid this river is on.
    occupancy_grid: The occupancy grid including cells from previous layers.
    next_path_index: The index of the next path layer to be generated (e.g. 2 if
      we just generated path 1).

  Returns:
    The start cell for the next path layer.

  Raises:
    HuggingError: if next start cell is out of bounds or occupied.
  """
  if hug_side == HugSide.LEFT:
    growth_dir = direction.turn_right()
  else:
    growth_dir = direction.turn_left()
  dy, dx = growth_dir.delta
  next_start_cell = (
      current_path_start_cell[0] + dy,
      current_path_start_cell[1] + dx,
  )

  if not (
      0 <= next_start_cell[0] < grid_size
      and 0 <= next_start_cell[1] < grid_size
  ):
    raise HuggingError(
        f'Next start cell {next_start_cell} for path {next_path_index} is out'
        ' of bounds.'
    )
  if occupancy_grid[next_start_cell[0], next_start_cell[1]]:
    raise HuggingError(
        f'Next start cell {next_start_cell} for path {next_path_index} is'
        ' already occupied by previous path section.'
    )
  return next_start_cell


def hugging_path_for_layer(
    grid_size: int,
    hugging_grid: np.ndarray,
    start_cell: tuple[int, int],
    hug_side: HugSide,
    hug_boundaries: bool = False,
) -> tuple[list[tuple[int, int]], Direction]:
  """Generates path layer hugging hugging_grid.

  Args:
    grid_size: The size of grid this river is on.
    hugging_grid: The grid to hug.
    start_cell: The starting cell (y, x) for path generation.
    hug_side: HugSide.LEFT or HugSide.RIGHT, determining hugging direction.
    hug_boundaries: If True, boundaries are treated as obstacles to hug. If
      True, path generation also stops if it reaches or crosses grid's vertical
      symmetry axis, and the path is mirrored.

  Returns:
    A tuple of (path, direction), where path is the path layer, and direction
    is the initial direction of hugging.

  Raises:
    HuggingError: If path generation fails.
  """
  padded_grid = np.pad(
      hugging_grid, 1, 'constant', constant_values=hug_boundaries
  )
  start_y_padded, start_x_padded = start_cell[0] + 1, start_cell[1] + 1
  try:
    direction = get_initial_direction(
        (start_y_padded, start_x_padded), padded_grid, hug_side
    )
    path = _generate_hugging_path(
        grid_size,
        hugging_grid,
        start_cell,
        hug_side,
        hug_boundaries=hug_boundaries,
    )
  except HuggingError as e:
    raise HuggingError(f'Could not generate path layer: {e}') from e
  if not path:
    raise HuggingError('Could not generate path layer: path is empty.')
  return path, direction


def get_mirrored_path_for_symmetry(
    path: list[tuple[int, int]], grid_size: int
) -> list[tuple[int, int]]:
  """Mirrors riverpath across vertical axis for symmetric hugging.

  Reversing the path ensures continuity for the mirrored path. The original
  path runs from a border to the symmetry axis; the mirrored path must
  continue from the axis to the opposite border.

  Args:
    path: A list of (row, column) tuples representing path generated in one half
      of grid.
    grid_size: The size of the grid.

  Returns:
    A new list containing points from path and mirrored points.

  Raises:
    HuggingError: If a mirrored point is to left of symmetry axis, implying
      path contained points too far to right.
  """
  path_set = set(path)
  mirrored_path = []
  for row, column in reversed(path):
    mirrored_column = grid_size - 1 - column
    if mirrored_column < grid_size // 2:
      raise HuggingError(
          'Symmetric hugging river generation crossed symmetry axis.'
      )
    if (row, mirrored_column) not in path_set:
      mirrored_path.append((row, mirrored_column))
  return path + mirrored_path


def _append_hugging_path_layer(
    grid_size: int,
    hugging_grid: np.ndarray,
    start_cell: tuple[int, int],
    hug_side: HugSide,
    all_paths: list[list[tuple[int, int]]],
    obstacles_plus_river: np.ndarray,
    occupancy_grid_for_collision: np.ndarray | None,
    hug_boundaries: bool = False,
) -> tuple[Direction, np.ndarray]:
  """Generates a path layer, adds to all_paths, updates grids."""
  path, direction = hugging_path_for_layer(
      grid_size,
      hugging_grid,
      start_cell,
      hug_side,
      hug_boundaries=hug_boundaries,
  )
  if hug_boundaries:
    path = get_mirrored_path_for_symmetry(path, grid_size)

  layer_grid = np.zeros_like(hugging_grid)
  for y, x in path:
    rect = shaping_utils.Rectangle(
        name='', x_start=x, y_start=y, x_len=1, y_len=1
    )
    if (
        occupancy_grid_for_collision is not None
        and not utils_packing_flaps.is_unoccupied(
            occupancy_grid_for_collision, rect
        )
    ):
      raise HuggingError(
          f'Hugging river layer {len(all_paths)+1} collides with'
          f' obstacle at {(y, x)}'
      )
    utils_packing_flaps.set_grid_occupation(obstacles_plus_river, rect, True)
    utils_packing_flaps.set_grid_occupation(layer_grid, rect, True)
  all_paths.append(path)
  return direction, layer_grid


@dataclasses.dataclass(frozen=True)
class HuggingRiver:
  """A hugging river represented by a collection of rectangles of same width.

  Hugging rivers follow borders of occupied areas. They are represented
  as a list of rectangles.
  The path generation for width=1 rivers from start_cell
  and hug_side using contour tracing is implemented in `from_contour`.

  Attributes:
    grid_size: The size of grid this river is on.
    width: The width of river.
    subriver_paths: list of paths for each subriver of width 1.
    hug_side: side of hugging, HugSide.LEFT or HugSide.RIGHT.
    is_on_axis: Whether river is on axis of symmetry in symmetric packing.
    variant_type: The type of path variant used to generate the river.
    rectangles: list of rectangles forming river.
    path: The path of the river. If not provided, it is computed from
      occupancy_grid and start_cell in `from_contour`. If provided, it must
      match the path computed from occupancy_grid and start_cell.
  """

  grid_size: int
  width: int
  subriver_paths: list[list[tuple[int, int]]]
  hug_side: HugSide
  is_on_axis: bool = False
  variant_type: PathVariantType = PathVariantType.STANDARD
  rectangles: list[shaping_utils.Rectangle] = dataclasses.field(init=False)
  path: list[tuple[int, int]] = dataclasses.field(init=False)

  def __post_init__(self):
    object.__setattr__(self, 'grid_size', int(self.grid_size))
    object.__setattr__(self, 'width', int(self.width))
    if self.width <= 0:
      raise ValueError('HuggingRiver width must be positive.')
    if len(self.subriver_paths) != self.width:
      raise ValueError('subriver_paths must have length equal to width.')
    if any(not path for path in self.subriver_paths):
      raise ValueError('Subriver paths cannot be empty.')
    if not more_itertools.all_unique(
        itertools.chain.from_iterable(self.subriver_paths)
    ):
      raise ValueError('All points in subriver paths must be unique.')

    all_rects = []
    all_path_cells = []
    for path in self.subriver_paths:
      all_path_cells.extend(path)
      for y, x in path:
        all_rects.append(
            shaping_utils.Rectangle(
                name='', x_start=x, y_start=y, x_len=1, y_len=1
            )
        )
    object.__setattr__(self, 'rectangles', all_rects)
    object.__setattr__(self, 'path', all_path_cells)

    if not self.rectangles:
      raise ValueError('HuggingRiver rectangles cannot be empty.')
    for rect in self.rectangles:
      if not (
          0 <= rect.x_start
          and 0 <= rect.y_start
          and rect.x_start + rect.x_len <= self.grid_size
          and rect.y_start + rect.y_len <= self.grid_size
      ):
        raise ValueError(f'Rectangle {rect} must be within grid.')

  def asdict(self, name: str) -> RiverDict:
    """Converts a HuggingRiver to a dict for S2 serialization."""
    return {
        'name': name,
        'rectangles': [r.asdict() for r in self.rectangles],
        'river_type': 'hugging',
        'width': self.width,
        'grid_size': self.grid_size,
    }

  @functools.cached_property
  def start_cell(self) -> tuple[int, int]:
    """Returns the start cell of the river."""
    return self.subriver_paths[0][0]

  def is_vertically_symmetric(self) -> bool:
    """Checks if river placement is symmetric wrt to vertical centerline."""
    cells = set(self.path)
    for row, col in cells:
      if (row, self.grid_size - 1 - col) not in cells:
        return False
    return True

  def mirror(self, grid_size: int) -> 'HuggingRiver':
    """Returns a mirrored version of the hugging river."""
    new_hug_side = (
        HugSide.RIGHT if self.hug_side == HugSide.LEFT else HugSide.LEFT
    )
    new_subriver_paths = []
    for path in self.subriver_paths:
      new_path = [(y, grid_size - 1 - x) for y, x in path]
      new_subriver_paths.append(new_path)
    return HuggingRiver(
        grid_size=grid_size,
        width=self.width,
        subriver_paths=new_subriver_paths,
        hug_side=new_hug_side,
    )

  def splice(
      self, other: 'HuggingRiver', reverse_subrivers: bool = False
  ) -> 'HuggingRiver':
    """Splices another hugging river into this one."""
    if self.width != other.width:
      raise ValueError('Spliced river must have same width.')

    other_paths = (
        reversed(other.subriver_paths)
        if reverse_subrivers
        else other.subriver_paths
    )
    new_subriver_paths = []
    for current_path, splice_path in zip(self.subriver_paths, other_paths):
      start_point, end_point = splice_path[0], splice_path[-1]

      try:
        start_index = current_path.index(start_point)
        end_index = current_path.index(end_point)
      except ValueError as e:
        raise ValueError(
            'Start or end point of subriver not found in base river.'
        ) from e

      if start_index <= end_index:
        new_path = (
            current_path[:start_index]
            + splice_path
            + current_path[end_index + 1 :]
        )
      else:
        new_path = (
            current_path[:end_index]
            + splice_path[::-1]
            + current_path[start_index + 1 :]
        )
      new_subriver_paths.append(new_path)

    return dataclasses.replace(self, subriver_paths=new_subriver_paths)

  def get_river_mask_and_forbidden_grid(
      self, occupancy_grid: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    """Returns river mask and forbidden grid."""
    river_mask = np.zeros_like(occupancy_grid)
    for rect in self.rectangles:
      utils_packing_flaps.set_grid_occupation(river_mask, rect, True)
    forbidden = np.logical_or(occupancy_grid, river_mask)
    return river_mask, forbidden

  def get_corner_counts_on_river_sides(
      self, occupancy_grid: np.ndarray
  ) -> dict[str, int]:
    """Computes number of available corners on each side of river."""
    _, forbidden = self.get_river_mask_and_forbidden_grid(occupancy_grid)

    hugging_seeds = self.get_seeds_hugging_side(forbidden)
    hugging_corners = _count_corners_from_seeds(
        hugging_seeds, self.grid_size, forbidden
    )

    opposite_seeds = self.get_seeds_opposite_to_hugging(forbidden)
    opposite_corners = _count_corners_from_seeds(
        opposite_seeds, self.grid_size, forbidden
    )
    return {'hugging': hugging_corners, 'opposite': opposite_corners}

  def get_area_masks(
      self, occupancy_grid: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    """Returns masks for hugging and opposite areas."""
    _, forbidden = self.get_river_mask_and_forbidden_grid(occupancy_grid)

    hugging_seeds = self.get_seeds_hugging_side(forbidden)
    hugging_mask = _get_reachable_mask_from_seeds(
        hugging_seeds, self.grid_size, forbidden
    )

    opposite_seeds = self.get_seeds_opposite_to_hugging(forbidden)
    opposite_mask = _get_reachable_mask_from_seeds(
        opposite_seeds, self.grid_size, forbidden
    )
    return hugging_mask, opposite_mask

  def get_river_bank_seeds(
      self, occupancy_grid: np.ndarray
  ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Returns seed cells for BFS on hugging and opposite sides."""
    _, forbidden = self.get_river_mask_and_forbidden_grid(occupancy_grid)
    hugging_seeds = self.get_seeds_hugging_side(forbidden)
    opposite_seeds = self.get_seeds_opposite_to_hugging(forbidden)
    return hugging_seeds, opposite_seeds

  def get_adjacent_cells_by_side(
      self, occupancy_grid: np.ndarray
  ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Returns river-adjacent cells separated by hugging and opposite sides."""
    hugging_mask, opposite_mask = self.get_area_masks(occupancy_grid)
    river_mask, _ = self.get_river_mask_and_forbidden_grid(occupancy_grid)
    hugging_adj = set()
    opposite_adj = set()
    river_coords = np.argwhere(river_mask)
    for row, column in river_coords:
      for delta_row, delta_column in DIRS:
        neighbor_row, neighbor_column = row + delta_row, column + delta_column
        if (
            0 <= neighbor_row < self.grid_size
            and 0 <= neighbor_column < self.grid_size
        ):
          if hugging_mask[neighbor_row, neighbor_column]:
            hugging_adj.add((neighbor_row, neighbor_column))
          elif opposite_mask[neighbor_row, neighbor_column]:
            opposite_adj.add((neighbor_row, neighbor_column))
    return hugging_adj, opposite_adj

  def _get_seeds_for_side(
      self, forbidden: np.ndarray, side: str
  ) -> set[tuple[int, int]]:
    """Returns seed cells for BFS on one side of river."""
    if side == 'hugging':
      path_index = 0
      turn_left = self.hug_side == HugSide.LEFT
    elif side == 'opposite':
      path_index = -1
      turn_left = self.hug_side == HugSide.RIGHT
    else:
      raise ValueError(f"side must be 'hugging' or 'opposite', got {side}")

    seeds = set()
    path = self.subriver_paths[path_index]
    for i in range(len(path) - 1):
      y1, x1 = path[i]
      y2, x2 = path[i + 1]
      move_dy, move_dx = y2 - y1, x2 - x1

      if abs(y1 - y2) + abs(x1 - x2) > 1:
        break

      if turn_left:
        offset_y, offset_x = -move_dx, move_dy
      else:
        offset_y, offset_x = move_dx, -move_dy

      potential_seeds = [
          (y1 + offset_y, x1 + offset_x),
          (y2 + offset_y, x2 + offset_x),
      ]

      for seed_y, seed_x in potential_seeds:
        if (
            0 <= seed_y < self.grid_size
            and 0 <= seed_x < self.grid_size
            and not forbidden[seed_y, seed_x]
        ):
          seeds.add((seed_y, seed_x))
    return seeds

  def get_seeds_opposite_to_hugging(
      self, forbidden: np.ndarray
  ) -> set[tuple[int, int]]:
    """Returns seed cells for BFS on side opposite to hugging."""
    return self._get_seeds_for_side(forbidden, 'opposite')

  def get_seeds_hugging_side(
      self, forbidden: np.ndarray
  ) -> set[tuple[int, int]]:
    """Returns seed cells for BFS on hugging side."""
    return self._get_seeds_for_side(forbidden, 'hugging')

  def compute_unoccupied_areas_on_river_sides(
      self, occupancy_grid: np.ndarray
  ) -> tuple[int, int]:
    """Computes unoccupied area on each side of river reachable from river bank."""
    _, forbidden = self.get_river_mask_and_forbidden_grid(occupancy_grid)
    hugging_seeds = self.get_seeds_hugging_side(forbidden)
    opposite_seeds = self.get_seeds_opposite_to_hugging(forbidden)
    if not hugging_seeds and not opposite_seeds:
      return 0, 0
    return _get_connected_components_areas(
        forbidden, list(hugging_seeds), list(opposite_seeds)
    )

  @classmethod
  def from_contour(
      cls,
      grid_size: int,
      width: int,
      occupancy_grid: np.ndarray,
      start_cell: tuple[int, int],
      hug_side: HugSide,
      hug_boundaries: bool = False,
      variant_type: PathVariantType = PathVariantType.STANDARD,
  ) -> 'HuggingRiver':
    """Creates a HuggingRiver by tracing border of occupied cells."""
    if width == 1:
      path = _generate_hugging_path(
          grid_size,
          occupancy_grid,
          start_cell,
          hug_side,
          hug_boundaries=hug_boundaries,
      )
      if hug_boundaries:
        path = get_mirrored_path_for_symmetry(path, grid_size)
      return cls(
          grid_size=grid_size,
          width=width,
          subriver_paths=[path],
          hug_side=hug_side,
          is_on_axis=hug_boundaries,
          variant_type=variant_type,
      )

    obstacles_plus_river = occupancy_grid.copy()
    subriver_paths: list[list[tuple[int, int]]] = []

    current_start_cell = start_cell
    direction, layer_grid = _append_hugging_path_layer(
        grid_size,
        occupancy_grid,
        current_start_cell,
        hug_side,
        subriver_paths,
        obstacles_plus_river,
        None,
        hug_boundaries=hug_boundaries,
    )

    for i in range(1, width):
      current_start_cell = get_next_start_cell(
          current_start_cell,
          direction,
          hug_side,
          grid_size,
          obstacles_plus_river,
          i + 1,
      )
      direction, layer_grid = _append_hugging_path_layer(
          grid_size,
          layer_grid,
          current_start_cell,
          hug_side,
          subriver_paths,
          obstacles_plus_river,
          occupancy_grid,
          hug_boundaries=hug_boundaries,
      )

    return cls(
        grid_size=grid_size,
        width=width,
        subriver_paths=subriver_paths,
        hug_side=hug_side,
        is_on_axis=hug_boundaries,
        variant_type=variant_type,
    )


@dataclasses.dataclass
class NeighborState:
  """Occupancy state of cells relative to current position and direction."""

  hugging_cell: tuple[int, int]
  front_cell: tuple[int, int]
  opposite_cell: tuple[int, int]
  is_hugging_occupied: bool
  is_front_occupied: bool
  is_opposite_occupied: bool


def _is_cell_occupied(cell: tuple[int, int], padded_grid: np.ndarray) -> bool:
  """Checks if a cell is occupied in padded grid, with bounds check."""
  y, x = cell
  return (
      0 <= y < padded_grid.shape[0]
      and 0 <= x < padded_grid.shape[1]
      and padded_grid[cell]
  )


def get_neighbor_states(
    y: int,
    x: int,
    direction: Direction,
    hug_side: HugSide,
    padded_grid: np.ndarray,
) -> NeighborState:
  """Gets coordinates and occupancy of neighbors relative to hugging direction."""
  if hug_side == HugSide.LEFT:
    hugging_dir = direction.turn_left()
    opposite_dir = direction.turn_right()
  else:
    hugging_dir = direction.turn_right()
    opposite_dir = direction.turn_left()

  hugging_dy, hugging_dx = hugging_dir.delta
  hugging_cell = (y + hugging_dy, x + hugging_dx)
  front_dy, front_dx = direction.delta
  front_cell = (y + front_dy, x + front_dx)
  opposite_dy, opposite_dx = opposite_dir.delta
  opposite_cell = (y + opposite_dy, x + opposite_dx)

  return NeighborState(
      hugging_cell=hugging_cell,
      front_cell=front_cell,
      opposite_cell=opposite_cell,
      is_hugging_occupied=_is_cell_occupied(hugging_cell, padded_grid),
      is_front_occupied=_is_cell_occupied(front_cell, padded_grid),
      is_opposite_occupied=_is_cell_occupied(opposite_cell, padded_grid),
  )


def apply_hugging_step(
    y: int,
    x: int,
    direction: Direction,
    hug_side: HugSide,
    padded_grid: np.ndarray,
) -> tuple[int, int, Direction]:
  """Applies one step of hugging logic based on rules and neighbor states."""
  state = get_neighbor_states(y, x, direction, hug_side, padded_grid)
  if hug_side == HugSide.LEFT:
    hugging_dir = direction.turn_left()
    opposite_dir = direction.turn_right()
  else:
    hugging_dir = direction.turn_right()
    opposite_dir = direction.turn_left()

  if not state.is_hugging_occupied:
    direction = hugging_dir
    dy, dx = direction.delta
    y += dy
    x += dx
  elif state.is_front_occupied:
    if not state.is_opposite_occupied:
      direction = opposite_dir
      y, x = state.opposite_cell
    else:
      raise HuggingError(
          f'Hugging path generation failed at cell {(y - 1, x - 1)}. The set of'
          ' polygons could not be strictly hugged'
      )
  else:
    y, x = state.front_cell
  return y, x, direction


def _generate_hugging_path(
    grid_size: int,
    occupancy_grid: np.ndarray,
    start_cell: tuple[int, int],
    hug_side: HugSide,
    hug_boundaries: bool = False,
) -> list[tuple[int, int]]:
  """Generates a width-1 river path by hugging occupied cells."""
  padded_grid = np.pad(
      occupancy_grid, 1, 'constant', constant_values=hug_boundaries
  )
  current_y, current_x = start_cell[0] + 1, start_cell[1] + 1

  if padded_grid[current_y, current_x]:
    raise HuggingError(f'Start cell {start_cell} must be unoccupied.')

  try:
    direction = get_initial_direction(
        (current_y, current_x), padded_grid, hug_side
    )
  except HuggingError as e:
    raise HuggingError(
        f'Could not find initial direction for start cell {start_cell}.'
    ) from e

  path = []
  visited_path_cells = set()
  symmetry_col = grid_size // 2

  for _ in range(grid_size * grid_size + 8):
    if not hug_boundaries and not (
        1 <= current_y <= grid_size and 1 <= current_x <= grid_size
    ):
      break

    current_path_cell = (current_y - 1, current_x - 1)

    if hug_boundaries and current_path_cell[1] >= symmetry_col:
      break

    if current_path_cell in visited_path_cells:
      break
    path.append(current_path_cell)
    visited_path_cells.add(current_path_cell)

    current_y, current_x, direction = apply_hugging_step(
        current_y,
        current_x,
        direction,
        hug_side,
        padded_grid,
    )
  return path


def is_mirror_correct(
    river: River | LShapedRiver | HuggingRiver, grid_size: int
) -> bool:
  """Checks if mirror implementation is correct by comparing grids."""
  grid = np.zeros((grid_size, grid_size), dtype=bool)
  for rect in river.rectangles:
    utils_packing_flaps.set_grid_occupation(grid, rect, True)

  mirrored_grid_by_flipping = np.fliplr(grid)

  mirrored_river = river.mirror(grid_size)
  grid_from_mirrored_river = np.zeros((grid_size, grid_size), dtype=bool)
  for rect in mirrored_river.rectangles:
    utils_packing_flaps.set_grid_occupation(
        grid_from_mirrored_river, rect, True
    )

  return np.array_equal(mirrored_grid_by_flipping, grid_from_mirrored_river)


def get_border_cell_counter_clockwise_index(
    cell: tuple[int, int], grid_size: int
) -> int:
  """Returns counter-clockwise index of a border cell."""
  row, col = cell
  if (
      col == 0 and row >= 0 and row < grid_size - 1
  ):  # left edge excluding bottom-left
    return row
  if (
      row == grid_size - 1 and col >= 0 and col < grid_size - 1
  ):  # bottom edge excluding bottom-right
    return grid_size - 1 + col
  if (
      col == grid_size - 1 and row > 0 and row <= grid_size - 1
  ):  # right edge excluding top-right
    return grid_size - 1 + grid_size - 1 + (grid_size - 1 - row)
  if (
      row == 0 and col > 0 and col <= grid_size - 1
  ):  # top edge excluding top-left
    return grid_size - 1 + grid_size - 1 + grid_size - 1 + (grid_size - 1 - col)
  raise ValueError(f'Cell {cell} is not on border of grid {grid_size}')


def calculate_counter_clockwise_distance(
    cell: tuple[int, int], start_cell: tuple[int, int], grid_size: int
) -> int:
  """Returns counter-clockwise distance from start_cell to cell."""
  num_border_cells = 4 * grid_size - 4
  if num_border_cells == 0:  # grid_size 1
    return 0
  start_idx = get_border_cell_counter_clockwise_index(start_cell, grid_size)
  cell_idx = get_border_cell_counter_clockwise_index(cell, grid_size)
  return (cell_idx - start_idx + num_border_cells) % num_border_cells


def _count_corners_from_seeds(
    seeds: set[tuple[int, int]], grid_size: int, forbidden: np.ndarray
) -> int:
  """Counts corners in mask of cells reachable from seeds via BFS."""
  if not seeds:
    return 0
  mask = _get_reachable_mask_from_seeds(seeds, grid_size, forbidden)
  return count_corners_in_mask(mask)


def _get_reachable_mask_from_seeds(
    seeds: set[tuple[int, int]], grid_size: int, forbidden: np.ndarray
) -> np.ndarray:
  """Returns mask of cells reachable from seeds via BFS."""
  if not seeds:
    return np.zeros((grid_size, grid_size), dtype=bool)
  free = ~forbidden
  seed_rows, seed_cols = zip(*seeds)
  rows_arr = np.fromiter(seed_rows, dtype=np.intp, count=len(seeds))
  cols_arr = np.fromiter(seed_cols, dtype=np.intp, count=len(seeds))
  if not np.all(free[rows_arr, cols_arr]):
    free = free.copy()
    free[rows_arr, cols_arr] = True
  labels, _ = _label_4connected(free)
  seed_labels = np.unique(labels[rows_arr, cols_arr])
  seed_labels = seed_labels[seed_labels != 0]
  return np.isin(labels, seed_labels)


_RECONSTRUCTED_RIVER_FACTORY: Any = None


def register_reconstructed_river_factory(factory: Any) -> None:
  """Registers a factory for reconstructing extended river instances."""
  global _RECONSTRUCTED_RIVER_FACTORY
  _RECONSTRUCTED_RIVER_FACTORY = factory


def get_reconstructed_river_factory() -> Any:
  """Returns the registered factory for reconstructed rivers, if any."""
  return _RECONSTRUCTED_RIVER_FACTORY


PackingRiver = River | LShapedRiver | HuggingRiver
