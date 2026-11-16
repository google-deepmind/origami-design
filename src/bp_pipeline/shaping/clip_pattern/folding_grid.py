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

"""Geometrical Folding Algorithm for a paper grid with diagonals.

Paper grid can have vertical, horizontal, left and right diagonal creases.
Each cell of a grid is represented as 4 triangles. Output of the folding process
is position of all those triangles.

Folding algorithm:
1.  Build paths which all start from left triangle of cell (0,0) union of which
traverses all triangles.
2. Fold each path according to a simple algorithm: when path crosses a crease,
rotate coordinates accordingly and continue movement according to the path.
3. Join results from all paths, so that each triangle from the flat state is
represented once.

Algorithm also computes z_sign to store if triangle was flipped. +1 means it
faces same direction as in the flat state, -1 means it faces opposite direction.
"""

import dataclasses
from typing import TypeAlias

import numpy as np

from . import drawing
from . import folding_paths
from . import paper_grid as paper_grid_lib
from . import triangle_geometry
from . import world_space

PaperGridData: TypeAlias = paper_grid_lib.PaperGridData
EdgeAssignment: TypeAlias = paper_grid_lib.EdgeAssignment
EdgeKind: TypeAlias = paper_grid_lib.EdgeKind
CellTriangle: TypeAlias = triangle_geometry.CellTriangle
TriangleCellPosition: TypeAlias = triangle_geometry.TriangleCellPosition


# Re-export drawing functions for convenience.
plot_triangles_default = drawing.plot_triangles_default
draw_triangles_on_plot = drawing.draw_triangles_on_plot


@dataclasses.dataclass()
class FoldingGrid:
  """A folding grid.

  Attributes:
    paper_grid: The paper grid to be folded, including crease assignments.
    folding_paths: A list of paths, where each path is a list of
      TriangleCellPosition. The paths should cover all triangles in the grid.
    paths_masks: A list of boolean masks, one for each path in
      folding_paths. The mask indicates whether the triangle at that
      position in the path is the first time it is visited across all paths.
      This is used to when each triangle should be processed only once.
  """

  paper_grid: PaperGridData
  folding_paths: list[list[TriangleCellPosition]] | None = None
  paths_masks: list[list[bool]] | None = None

  @classmethod
  def from_paper_grid(cls, paper_grid: PaperGridData) -> "FoldingGrid":
    """Initializes prerequisites for folding the given grid."""
    return FoldingGrid(
        paper_grid=paper_grid,
        folding_paths=None,
        paths_masks=None,
    )

  def initialize_folding_paths(self) -> None:
    self.folding_paths = (
        folding_paths.DiagonalFoldingPaths().get_paths_diag_slices(
            self.paper_grid.num_rows, self.paper_grid.num_cols
        )
    )
    self.paths_masks = folding_paths.get_triangles_masks_for_paths(
        self.folding_paths
    )

  def get_edge_assignment_between_cells(
      self,
      row1: int,
      col1: int,
      tr1: CellTriangle,
      row2: int,
      col2: int,
      tr2: CellTriangle,
  ) -> EdgeAssignment:
    """Returns the edge assignment between two triangles.

    The two triangles must be adjacent, i.e. share an edge. If they are not
    adjacent, or if they are the same triangle, a ValueError is raised.

    Args:
      row1: Row index of the first triangle.
      col1: Column index of the first triangle.
      tr1: Triangle type of the first triangle.
      row2: Row index of the second triangle.
      col2: Column index of the second triangle.
      tr2: Triangle type of the second triangle.

    Returns:
      The edge assignment of the shared edge.
    """
    if not 0 <= row1 < self.paper_grid.num_rows:
      raise ValueError(f"Invalid row index: {row1}")
    if not 0 <= col1 < self.paper_grid.num_cols:
      raise ValueError(f"Invalid column index: {col1}")
    if not 0 <= row2 < self.paper_grid.num_rows:
      raise ValueError(f"Invalid row index: {row2}")
    if not 0 <= col2 < self.paper_grid.num_cols:
      raise ValueError(f"Invalid column index: {col2}")
    if row1 == row2 and col1 == col2 and tr1 == tr2:
      raise ValueError(f"Same cell: ({row1}, {col1})")
    if row1 == row2 and col1 == col2:
      if (tr1 == CellTriangle.TOP and tr2 == CellTriangle.RIGHT) or (
          tr1 == CellTriangle.RIGHT and tr2 == CellTriangle.TOP
      ):
        # right diag
        return self.paper_grid.get_cell(row1, col1).get_edge_assignment(
            EdgeKind.RIGHT_DIAG
        )
      if (tr1 == CellTriangle.RIGHT and tr2 == CellTriangle.BOTTOM) or (
          tr1 == CellTriangle.BOTTOM and tr2 == CellTriangle.RIGHT
      ):
        # left diag
        return self.paper_grid.get_cell(row1, col1).get_edge_assignment(
            EdgeKind.LEFT_DIAG
        )
      if (tr1 == CellTriangle.BOTTOM and tr2 == CellTriangle.LEFT) or (
          tr1 == CellTriangle.LEFT and tr2 == CellTriangle.BOTTOM
      ):
        # right diag
        return self.paper_grid.get_cell(row1, col1).get_edge_assignment(
            EdgeKind.RIGHT_DIAG
        )
      if (tr1 == CellTriangle.LEFT and tr2 == CellTriangle.TOP) or (
          tr1 == CellTriangle.TOP and tr2 == CellTriangle.LEFT
      ):
        # left diag
        return self.paper_grid.get_cell(row1, col1).get_edge_assignment(
            EdgeKind.LEFT_DIAG
        )
      raise ValueError(
          f"In same cell no edge between ({row1}, {col1}, {tr1}) and ({row2},"
          f" {col2}, {tr2})"
      )
    if row1 == row2 and col2 == col1 + 1:
      if tr1 == CellTriangle.RIGHT and tr2 == CellTriangle.LEFT:
        return self.paper_grid.get_cell(row1, col1).get_edge_assignment(
            EdgeKind.RIGHT
        )
      else:
        raise ValueError("Triangles does not share an edge.")
    if row1 == row2 and col1 == col2 + 1:
      if tr2 == CellTriangle.RIGHT and tr1 == CellTriangle.LEFT:
        return self.paper_grid.get_cell(row1, col2).get_edge_assignment(
            EdgeKind.RIGHT
        )
      else:
        raise ValueError("Triangles does not share an edge.")
    if col1 == col2 and row2 == row1 + 1:
      if tr1 == CellTriangle.BOTTOM and tr2 == CellTriangle.TOP:
        return self.paper_grid.get_cell(row1, col1).get_edge_assignment(
            EdgeKind.BOTTOM
        )
      else:
        raise ValueError("Triangles does not share an edge.")
    if col1 == col2 and row1 == row2 + 1:
      if tr2 == CellTriangle.BOTTOM and tr1 == CellTriangle.TOP:
        return self.paper_grid.get_cell(row2, col1).get_edge_assignment(
            EdgeKind.BOTTOM
        )
      else:
        raise ValueError("Triangles does not share an edge.")
    raise ValueError(
        f"No edge between ({row1}, {col1}, {tr1}) and ({row2}, {col2}, {tr2})"
    )

  def fold_path(
      self,
      path: list[TriangleCellPosition],
      do_not_fold: bool = False,
      compute_basis: bool = False,
  ) -> list[TriangleCellPosition]:
    """Folds grid according to a path.

    The path is a list of triangles, each adjacent to the previous one.
    The folding process iterates through the path and applies rotations
    based on the edge assignments between triangles according to the
    self.paper_grid.
    In the paper grid some edges are folded (180 degrees), some are flat (0
    degrees).

    Properties of the setup are such that folded triangle can always be
    represented by the same type as some not-folded triangle.

    Args:
      path: A list of triangle cell positions representing the path to fold.
      do_not_fold: If True, all folding angles are set to 0, the function
        returns the triangles which just follow the path.
      compute_basis: If True, saves the basis matrix for each triangle.

    Returns:
      A list of TriangleCellPosition after folding in the same sequence as
      triangles in the provided path.
    """

    path_points = np.stack([np.array(p.center_point()) for p in path], axis=0)
    directions = np.diff(path_points, axis=0)
    norms = np.linalg.norm(directions, axis=1)
    if np.any(np.isclose(norms, 0.0)):
      raise ValueError("A direction vector cannot be a zero vector.")
    creases = np.zeros((directions.shape[0], 3))
    creases[:, 0] = -directions[:, 1] / norms
    creases[:, 1] = directions[:, 0] / norms
    assignments = np.array([
        self.get_edge_assignment_between_cells(
            path[i].row,
            path[i].col,
            path[i].tr,
            path[i - 1].row,
            path[i - 1].col,
            path[i - 1].tr,
        )
        for i in range(1, len(path))
    ])

    if do_not_fold:
      thetas = np.zeros(len(assignments))
    else:
      thetas = np.where(assignments == EdgeAssignment.FOLDED, 180.0, 0.0)
    transformations = []

    def add_norm_point(world: world_space.World):
      world.transform(translate=(0, 0, 1))
      world.add_point_at_origin()
      world.pop_transformation()

    grid = world_space.World()
    # set world to the row-col coordinate system
    grid.transform(rotate_deg=-90, rotate_axis=(0, 0, 1))
    grid.transform(translate=np.array(path[0].center_point()))
    transformations.append(grid.peek_transform())
    add_norm_point(grid)
    steps = len(directions)
    for i in range(steps):
      grid.transform(translate=directions[i] / 2)
      grid.transform(rotate_deg=thetas[i], rotate_axis=creases[i])
      grid.transform(translate=directions[i] / 2)
      transformations.append(grid.peek_transform())
      add_norm_point(grid)

    for _ in range(steps):
      grid.pop_transformation()
      grid.pop_transformation()
      grid.pop_transformation()

    # pop initial translate to start position:
    grid.pop_transformation()
    assert grid.transform_stack_size == 2

    folded_grid_points = grid.get_points_in_current_coords()
    for point in folded_grid_points:
      if not np.isclose(point.z, 1.0) and not np.isclose(point.z, -1.0):
        raise ValueError("figure is not flat after folding (bug in the code).")
    folded_positions = [
        TriangleCellPosition.from_coords(
            x=point.x,
            y=point.y,
            z=point.z,
            basis_matrix=transformations[index] if compute_basis else None,
        )
        for index, point in enumerate(folded_grid_points)
    ]
    return folded_positions

  def folded_grid(
      self, *, do_not_fold: bool = False
  ) -> list[TriangleCellPosition]:
    """Folds grid according to the folding paths.

    This method folds all paths in `self.folding_paths` by calling `fold_path`
    for each path and returns a list of all folded triangles in the grid. When
    triangles are mentioned in multiple paths, only
    first occurrence of the triangle is returned.

    Args:
      do_not_fold: If True, the grid is not folded, and the function returns the
        triangles in their original positions.

    Returns:
      A list of all triangle cell positions of the grid after folding.
    """
    if self.folding_paths is None:
      self.initialize_folding_paths()
    folded_triangles = []
    for path, triangles_mask in zip(self.folding_paths, self.paths_masks):
      folded_path = self.fold_path(path, do_not_fold=do_not_fold)
      folded_triangles.extend([
          folded_triangle
          for folded_triangle, mask in zip(folded_path, triangles_mask)
          if mask
      ])
    return folded_triangles
