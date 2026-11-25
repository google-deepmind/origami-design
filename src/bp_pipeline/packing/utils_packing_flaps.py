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

"""Utilities for packing in box pleating with flaps."""

import dataclasses
import enum
import logging
import math
import numpy as np
from .. import rectangle as shaping_utils
from ..stick_figures import stick as stick_lib


class GridCellType(enum.Enum):
  """The type of object that occupied a cell on the metadata grid."""

  EMPTY = 'EMPTY'
  POLYGON = 'POLYGON'
  RIVER = 'RIVER'
  BLOCKED = 'BLOCKED'

  RIVER_ADJACENT = 'RIVER_ADJACENT'


_EMPTY_ID = ''
_BLOCKED_ID = '__blocked__'


@dataclasses.dataclass(frozen=True, slots=True)
class GridCellState:
  type: GridCellType
  source_id: str

  def __post_init__(self):
    match self.type:
      case GridCellType.EMPTY:
        if self.source_id != _EMPTY_ID:
          raise ValueError(f'source_id must be "{_EMPTY_ID}" for EMPTY cells.')
      case GridCellType.BLOCKED:
        if self.source_id != _BLOCKED_ID:
          raise ValueError(
              f'source_id must be "{_BLOCKED_ID}" for BLOCKED cells.'
          )
      case _:
        if self.source_id in (_EMPTY_ID, _BLOCKED_ID):
          raise ValueError(
              'source_id must not be the source_id of EMPTY or BLOCKED cells.'
          )


# pylint: disable=invalid-name
EmptyGridCellState = GridCellState(type=GridCellType.EMPTY, source_id=_EMPTY_ID)
BlockedGridCellState = GridCellState(
    type=GridCellType.BLOCKED, source_id=_BLOCKED_ID
)
# pylint: enable=invalid-name


def level_shifter_pocket_depth(width: int) -> int:
  """Returns sacrificial pocket depth in grid cells (`0` when `width <= 1`).

  Args:
    width: Target integer flap or river width.

  Returns:
    Required sacrificial pocket depth (`2 * (width - 1)` for `width == 2`).

  Raises:
    ValueError: If `width > 2` (sacrificial pockets support `width == 2`).
  """
  w = int(width)
  if w <= 1:
    return 0
  if w != 2:
    raise ValueError(
        'Sacrificial-pocket level shifters currently support width == 2,'
        f' got width={width}.'
    )
  return 2 * (w - 1)


def level_shifter_center_margin(width: int) -> int:
  """Returns the minimum center-to-border margin (`pocket_depth // 2`)."""
  return level_shifter_pocket_depth(width) // 2


def get_flap_dim(
    stick: stick_lib.Stick, use_level_shifter: bool = False
) -> int:
  """Calculates flap dimension based on stick length and level shifter."""
  dim = stick.length * 2
  if use_level_shifter and stick.requires_level_shifter:
    dim += level_shifter_pocket_depth(round(stick.width))
  return dim


def set_grid_occupation(
    grid: np.ndarray,
    rect: shaping_utils.Rectangle,
    occupied: bool,
    metadata_grid: np.ndarray | None = None,
    state: GridCellState = EmptyGridCellState,
) -> None:
  """Marks or unmarks a rectangle area in a NumPy grid using slicing.

  This is a more performant and concise version of the original loop-based
  function. It automatically handles boundary clipping.

  Optionally updates a second, parallel 'metadata_grid' with a given state if
  provided.

  Args:
    grid: NumPy array (dtype=bool) to mark. The shape is (height, width).
    rect: The rectangle to set occupation for.
    occupied: True to mark as occupied, False to unmark.
    metadata_grid: Optional parallel NumPy array (dtype=object) to update. Its
      elements are expected to be GridCellState objects.
    state: a GridCellState to set for the rectangle area.
  """
  container_height, container_width = grid.shape
  if not all(
      isinstance(coord, int) or math.isclose(coord, round(coord), abs_tol=1e-9)
      for coord in (rect.x_start, rect.y_start, rect.x_len, rect.y_len)
  ):
    logging.error(
        'Attempt to place rectangle with non-integer coordinates: %s', rect
    )
  x = int(rect.x_start)
  y = int(rect.y_start)
  width = int(rect.x_len)
  height = int(rect.y_len)

  x_start = max(0, x)
  y_start = max(0, y)
  x_end = min(container_width, x + width)
  y_end = min(container_height, y + height)

  if x_start >= x_end or y_start >= y_end:
    return

  grid[y_start:y_end, x_start:x_end] = occupied

  if metadata_grid is not None:
    metadata_grid[y_start:y_end, x_start:x_end] = state


def is_unoccupied(
    grid: np.ndarray,
    rect: shaping_utils.Rectangle,
) -> bool:
  """Checks if a rectangle area is clear within container boundaries.

  Args:
    grid: Grid to check.
    rect: The rectangle to check for overlap.

  Returns:
    True if the rectangle area is clear within container boundaries, False
    otherwise.
  """
  container_height, container_width = grid.shape
  x_start = max(0, int(rect.x_start))
  y_start = max(0, int(rect.y_start))
  x_end = min(container_width, int(rect.x_start + rect.x_len))
  y_end = min(container_height, int(rect.y_start + rect.y_len))

  if x_start >= x_end or y_start >= y_end:
    return True

  return not grid[y_start:y_end, x_start:x_end].any()


def compute_unfilled_area(
    container_width: int,
    container_height: int,
    packing_placement: list[tuple[tuple[int, int], int, int]],
    initial_mask: np.ndarray | None = None,
) -> int:
  """Calculates the area of the grid that is not occupied by the rectangles.

  Handles overlapping rectangles by marking coverage on a grid.
  Optionally takes an initial_mask to account for pre-blocked areas.

  Args:
      container_width (int): The width of the container.
      container_height (int): The height of the container.
      packing_placement (list of tuples): A list of ((width, height), x, y)
        tuples representing the rectangles to be placed.
      initial_mask (np.ndarray, optional): An initial mask representing
        pre-blocked areas. True indicates a blocked cell.

  Returns:
      int: The area of the grid that is not occupied by the rectangles,
           considering both placed rectangles and initial masked areas.
  """
  grid = np.zeros((container_height, container_width), dtype=bool)
  if initial_mask is not None:
    height = min(container_height, initial_mask.shape[0])
    width = min(container_width, initial_mask.shape[1])
    grid[:height, :width] = initial_mask[:height, :width]

  for (width, height), x, y in packing_placement:
    rect = shaping_utils.Rectangle(
        name='',
        x_start=float(x),
        y_start=float(y),
        x_len=float(width),
        y_len=float(height),
    )
    set_grid_occupation(grid, rect, occupied=True)

  return container_width * container_height - np.count_nonzero(grid)  # pyrefly: ignore[bad-return]
