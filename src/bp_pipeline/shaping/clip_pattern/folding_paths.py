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

"""Generates folding paths for a paper grid.

For a given grid size, where each cell has 4 triangles as discribed in
triangle_geometry.py, this module provides functions to generate folding paths
which all start at the same triangle and visit all triangles.
DiagonalFoldingPaths look like diagonal slices which start from every cell in
first row and every cell in first column and follow the diagonal down until
reach the boundary.

This class uses a simple list of paths instead of tree/graph data structure. It
allows to process each path independently. In addition, there is a mask list
which is set to True on triangles which are
visited for the first time. It helps to remove duplicated triangles in the
paths if needed.
"""

from typing import TypeAlias
import numpy as np

from . import triangle_geometry as triangle_geometry_lib
from . import world_space

TriangleCellPosition: TypeAlias = triangle_geometry_lib.TriangleCellPosition


class DiagonalFoldingPaths:
  """Generator of paths for left-diag-slices."""

  _DO_LEFT = np.array([0, -0.25, 0], dtype=np.float32)
  _DO_RIGHT = np.array([0, 0.25, 0], dtype=np.float32)
  _DO_UP = np.array([-0.25, 0, 0], dtype=np.float32)
  _DO_DOWN = np.array([0.25, 0, 0], dtype=np.float32)
  _DO_RIGHT_CELL = np.array([0, 1, 0], dtype=np.float32)
  _DO_DOWN_CELL = np.array([1, 0, 0], dtype=np.float32)

  def _traverse_direction(self, grid: world_space.World, direction: np.ndarray):
    grid.transform(translate=direction)
    grid.add_point_at_origin()
    grid.pop_transformation()

  def _traverse_directions(
      self, grid: world_space.World, directions: list[np.ndarray]
  ):
    for direction in directions:
      self._traverse_direction(grid, direction)

  def _traverse_left_up_right(self, grid: world_space.World):
    self._traverse_directions(
        grid, [self._DO_LEFT, self._DO_UP, self._DO_RIGHT]
    )

  def _traverse_up_right(self, grid: world_space.World):
    self._traverse_directions(grid, [self._DO_UP, self._DO_RIGHT])

  def _traverse_left_down(self, grid: world_space.World):
    self._traverse_directions(grid, [self._DO_LEFT, self._DO_DOWN])

  def _traverse_up_left_down(self, grid: world_space.World):
    self._traverse_directions(grid, [self._DO_UP, self._DO_LEFT, self._DO_DOWN])

  def get_paths_diag_slices_global_coord(
      self, num_rows: int, num_cols: int
  ) -> list[list[world_space.Point3D]]:
    """Returns paths for left-diag-slices."""

    paths = []
    # Move horizontally in the first row.
    for col in range(num_cols + 1):
      grid = world_space.World()
      # set world to the row-col coordinate system
      grid.transform(rotate_deg=-90, rotate_axis=(0, 0, 1))
      transform_numb = 0
      cur_row = 0
      cur_col = 0
      for _ in range(col):
        self._traverse_left_up_right(grid)
        grid.transform(translate=self._DO_RIGHT_CELL)
        transform_numb += 1
        cur_col += 1
      # Follow triangles on the diagonal, use while to support non-square grid.
      while cur_row < num_rows and cur_col < num_cols:
        self._traverse_left_down(grid)
        grid.transform(translate=self._DO_DOWN_CELL)
        transform_numb += 1
        cur_row += 1
        if cur_row >= num_rows:
          break
        self._traverse_up_right(grid)

        grid.transform(translate=self._DO_RIGHT_CELL)
        transform_numb += 1
        cur_col += 1

      for _ in range(transform_numb):
        grid.pop_transformation()
      assert grid.transform_stack_size == 2
      paths.append(grid.get_global_points())

    # Move vertically in the first column.
    for row in range(1, num_rows + 1):
      grid = world_space.World()
      # set world to the row-col coordinate system
      grid.transform(rotate_deg=-90, rotate_axis=(0, 0, 1))
      transform_numb = 0
      cur_row = 0
      cur_col = 0
      for i in range(row):
        if i == 0:
          self._traverse_left_down(grid)
        else:
          self._traverse_up_left_down(grid)
        grid.transform(translate=self._DO_DOWN_CELL)
        transform_numb += 1
        cur_row += 1
      # Follow triangles on the diagonal, use while to support non-square grid.
      while cur_row < num_rows and cur_col < num_cols:
        self._traverse_up_right(grid)
        grid.transform(translate=self._DO_RIGHT_CELL)
        transform_numb += 1
        cur_col += 1
        if cur_col >= num_cols:
          break
        self._traverse_left_down(grid)
        grid.transform(translate=self._DO_DOWN_CELL)
        transform_numb += 1
        cur_row += 1

      for _ in range(transform_numb):
        grid.pop_transformation()
      assert grid.transform_stack_size == 2
      paths.append(grid.get_global_points())
    return paths

  def get_paths_diag_slices(
      self, num_rows: int, num_cols: int
  ) -> list[list[TriangleCellPosition]]:
    """Returns paths for left-diag-slices as TriangleCellPosition."""
    data = self.get_paths_diag_slices_global_coord(num_rows, num_cols)
    paths = []
    for d in data:
      grid = world_space.World()
      for p in d:
        grid.add_point(p.x, p.y, p.z)
      grid.transform(rotate_deg=-90, rotate_axis=(0, 0, 1))
      path = []
      for p in grid.get_points_in_current_coords():
        path.append(TriangleCellPosition.from_coords(x=p.x, y=p.y, z=p.z))
      paths.append(path)
    return paths


def get_triangle_index_in_grid_for_paths(
    paths: list[list[TriangleCellPosition]],
    num_cols: int,
) -> list[list[int]]:
  """Returns a list of triangle indices for each path."""
  triangle_indices = []
  for path in paths:
    triangle_indices.append(
        [p.triangle_index_in_grid(num_cols=num_cols) for p in path]
    )
  return triangle_indices


def get_triangles_masks_for_paths(
    paths: list[list[TriangleCellPosition]],
) -> list[list[bool]]:
  """Returns a masks where True means triangle is visited for the first time."""
  triangles_set = set()
  triangles_masks = []
  for path in paths:
    mask = []
    for p in path:
      if p not in triangles_set:
        triangles_set.add(p)
        mask.append(True)
      else:
        mask.append(False)
    triangles_masks.append(mask)
  return triangles_masks
