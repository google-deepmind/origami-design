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

r"""Data structures for representing triangles in a paper grid.

Grid is represented as a 2D array of cells, each cell has two diagonals which
form 4 triangles. This module contains data structures to represent a position
of a triangle in a cell and the triangle itself.

See diagram below that illustrates a cell with 4 triangles:

+-------------+
|\    UP    / |
| \        /  |
|   \     /   |
|LEFT \ /RIGHT|
|    /  \     |
|   /    \    |
|  / DOWN  \  |
| /         \ |
+-------------+

Here is an illustration of a 2x2 grid with all the relevant coordinates:

Cell of a grid identified by its center with coordinates (row, col). Triangles
(TOP, RIGHT, BOTTOM, LEFT) are identified by (x,y) where x axis is directed down
and y axis is directed to the right.
Triangle reference points are shown below:
 TOP: (row-0.25, col)
 LEFT: (row, col-0.25)
 RIGHT: (row, col+0.25)
 BOTTOM: (row+0.25, col)
"""

import dataclasses
import enum
import immutabledict
import numpy as np


class CellTriangle(enum.IntEnum):
  """Represents one of the 4 triangles in a cell with two diagonals."""

  BOTTOM = 0
  RIGHT = 1
  LEFT = 2
  TOP = 3


#  This map is used to translate center of a triangle to its corresponding
#  triangle enum value. Sparse lookup table looks like this:
#  |0     1        2
#  __________________
# 0|      TOP
# 1|LEFT         RIGHT
# 2|     BOTTOM
#
_TRIANGLE_MAP = immutabledict.immutabledict({
    (1, 0): CellTriangle.LEFT,
    (0, 1): CellTriangle.TOP,
    (1, 2): CellTriangle.RIGHT,
    (2, 1): CellTriangle.BOTTOM,
})

_REVERSE_TRIANGLE_MAP = immutabledict.immutabledict(
    {v: k for k, v in _TRIANGLE_MAP.items()}
)

TRIANGLE_ORDER = (
    CellTriangle.LEFT,
    CellTriangle.TOP,
    CellTriangle.RIGHT,
    CellTriangle.BOTTOM,
)

TRIANGLE_INDEX = immutabledict.immutabledict({
    CellTriangle.LEFT: 0,
    CellTriangle.TOP: 1,
    CellTriangle.RIGHT: 2,
    CellTriangle.BOTTOM: 3,
})


@dataclasses.dataclass(kw_only=True, frozen=True)
class TriangleCellPosition:
  """Represents a position of a cell's triangle in a grid.

  Attributes:
    row: Row index of the cell in the grid.
    col: Column index of the cell in the grid.
    z_sign: Used to identify which direction a surface of a triangle as a 2D
      shape in 3D space is pointed to. Use 1 as a default orientation. -1 will
      indicate that triangle was flipped. 0 means there is no information about
      the orientation.
    tr: One of the 4 triangles in the cell.
    basis_matrix: The basis matrix of the triangle. Can be None if not needed.
  """

  row: int
  col: int
  z_sign: int
  tr: CellTriangle
  basis_matrix: np.ndarray | None = None

  def key(self) -> tuple[int, int, CellTriangle]:
    return (self.row, self.col, self.tr)

  @classmethod
  def from_key(
      cls, key: tuple[int, int, CellTriangle]
  ) -> "TriangleCellPosition":
    return TriangleCellPosition(row=key[0], col=key[1], z_sign=1, tr=key[2])

  @classmethod

  def from_coords(
      cls,
      x: float,
      y: float,
      z: float,
      basis_matrix: np.ndarray | None = None,
  ) -> "TriangleCellPosition":
    """Returns a triangle cell position from x,y,z coordinates.

    Args:
      x: x coordinate of the center of the triangle in a row-col coordinate
        system.
      y: y coordinate of the center of the triangle in a row-col coordinate
        system.
      z: which side triangle is turned on: (0 for N/A, >0 for up, <0 for down)
      basis_matrix: The basis matrix of the triangle. Can be None.

    Returns:
      A TriangleCellPosition object.
    """
    cell_row, triangle_row = divmod(int(round((0.25 + x) * 4)), 4)
    cell_col, triangle_col = divmod(int(round((0.25 + y) * 4)), 4)
    if (triangle_row, triangle_col) not in _TRIANGLE_MAP:
      raise ValueError(
          f"unexpected point: {(triangle_row, triangle_col)} from x,y=({x},{y})"
      )
    tr = _TRIANGLE_MAP[(triangle_row, triangle_col)]
    if np.isclose(z, 0.0):
      z_sign = 0
    elif z > 0:
      z_sign = 1
    else:
      z_sign = -1

    return TriangleCellPosition(
        row=cell_row,
        col=cell_col,
        z_sign=z_sign,
        tr=tr,
        basis_matrix=basis_matrix,
    )

  def triangle_index_in_grid(self, num_cols: int) -> int:
    if num_cols <= self.col:
      raise ValueError(
          f"Invalid number of columns: {num_cols} for cell ({self.row},"
          f" {self.col})"
      )
    return self.row * (4 * num_cols) + 4 * self.col + TRIANGLE_INDEX[self.tr]

  def center_point(self) -> tuple[float, float, float]:
    """Returns the center point of the triangle in the row-col coordinate system."""
    trian_row, trian_col = _REVERSE_TRIANGLE_MAP[self.tr]
    x = self.row + trian_row / 4 - 0.25
    y = self.col + trian_col / 4 - 0.25

    return (x, y, 0.0)

  def vertices_2d(
      self,
  ) -> tuple[
      tuple[float, float],
      tuple[float, float],
      tuple[float, float],
  ]:
    """Returns the 3 vertices of the triangle as (X, Y) for 2D plotting."""
    center = (float(self.col), -float(self.row))
    top_left = (center[0] - 0.5, center[1] + 0.5)
    top_right = (center[0] + 0.5, center[1] + 0.5)
    bottom_left = (center[0] - 0.5, center[1] - 0.5)
    bottom_right = (center[0] + 0.5, center[1] - 0.5)

    match self.tr:
      case CellTriangle.TOP:
        return (top_left, top_right, center)
      case CellTriangle.RIGHT:
        return (top_right, bottom_right, center)
      case CellTriangle.BOTTOM:
        return (bottom_right, bottom_left, center)
      case CellTriangle.LEFT:
        return (bottom_left, top_left, center)
      case _:
        raise ValueError(f"Invalid CellTriangle value: {self.tr}")
