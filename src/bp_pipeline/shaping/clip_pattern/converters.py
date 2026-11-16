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

"""Helpers to convert data usable by geometric simulator."""

import dataclasses
import numpy as np
from ...crease_pattern import crease
from ...crease_pattern import crease_pattern
from ...crease_pattern.parser import svg as parser
from .. import folder_3d as fold_lib
from . import paper_grid


@dataclasses.dataclass(frozen=True)
class FoldGridParams:
  """Basic parameters of a grid that represents a fold."""

  num_rows: int
  num_cols: int
  min_coords: tuple[float, float]
  max_coords: tuple[float, float]
  width: float


def grid_params_from_fold(
    fold_data: fold_lib.FOLD, grid_size: int
) -> FoldGridParams:
  """Returns basic grid parameters for a given FOLD object and grid size."""
  min_coords = np.min(np.array(fold_data.vertices_coords), axis=0)
  max_coords = np.max(np.array(fold_data.vertices_coords), axis=0)
  cell_size = (max_coords - min_coords) / grid_size
  if not np.isclose(cell_size[0], cell_size[1]):
    raise ValueError('Not supported structure')
  return FoldGridParams(
      grid_size, grid_size, min_coords, max_coords, cell_size[0]
  )


@dataclasses.dataclass(frozen=True)
class GridPoint:
  """Represents a cell in a table-like grid."""

  row: int
  col: int

  @classmethod
  def from_top_left_coords(
      cls,
      coords: tuple[float, float],
      grid_params: FoldGridParams,
      atol: float = 0.05,
  ) -> 'GridPoint':
    """Converts a SVG coordinate (top-left point of a cell) to a GridPoint."""
    row_float = (coords[1] - grid_params.min_coords[1]) / grid_params.width
    col_float = (coords[0] - grid_params.min_coords[0]) / grid_params.width
    row = round(row_float)
    col = round(col_float)
    if not np.isclose(row_float, row, atol=atol) or not np.isclose(
        col_float, col, atol=atol
    ):
      raise ValueError(
          f'Coordinates {coords} do not align with the grid. Rounded row/col:'
          f' ({row}, {col}), Float row/col: ({row_float}, {col_float}). Either'
          ' crease pattern is not supported or try to change the grid_size'
          f' (currently it is {grid_params.num_rows}x{grid_params.num_cols}) or'
          ' try to set larger atol parameter if it is a precision problem.'
      )
    return GridPoint(row, col)


def crease_pattern_to_paper_grid(
    pattern: crease_pattern.CreasePattern, grid_size: int
) -> paper_grid.PaperGridData:
  svg = parser.to_svg(pattern, grid=grid_size)
  return svg_to_paper_grid(svg, grid_size)


def packing_cp_to_hinges_paper_grid(
    packing_cp: crease_pattern.CreasePattern, grid_size: int
) -> paper_grid.PaperGridData:
  """Saves hinges from packing crease pattern to a paper grid."""
  hinges = [c for c in packing_cp.creases if c.fold == crease.Fold.HINGE]
  new_hinges = []
  for h in hinges:
    new_hinges.append(
        crease_pattern.crease.Crease(h.p1, h.p2, fold=crease.Fold.MOUNTAIN)
    )
  borders = [c for c in packing_cp.creases if c.fold == crease.Fold.BORDER]
  svg = parser.to_svg(
      crease_pattern.CreasePattern(creases=new_hinges + borders), grid=grid_size
  )
  return svg_to_paper_grid(svg, grid_size)


def svg_to_paper_grid(svg: str, grid_size: int) -> paper_grid.PaperGridData:
  fold_data = fold_lib.FOLD.from_svg(svg_str=svg, vert_tol=1e-3)
  grid = convert_fold_to_grid(fold_data, grid_size=grid_size)
  return grid


def convert_fold_to_grid(
    fold_data: fold_lib.FOLD, grid_size: int
) -> paper_grid.PaperGridData:
  """Converts a FOLD object to a PaperGridData object."""
  grid_params = grid_params_from_fold(fold_data, grid_size)
  grid = paper_grid.PaperGridData.from_default(
      num_rows=grid_params.num_rows, num_cols=grid_params.num_cols
  )
  for k, edge_nodes in enumerate(fold_data.edges_vertices):

    assignment = fold_data.edges_assignment[k]
    if assignment not in ('M', 'V', 'B'):
      raise ValueError('Only M, V, B creases are supported')
    if assignment == 'B':
      continue
    vert1 = fold_data.vertices_coords[edge_nodes[0]]
    vert2 = fold_data.vertices_coords[edge_nodes[1]]
    cell1 = GridPoint.from_top_left_coords(vert1, grid_params)
    cell2 = GridPoint.from_top_left_coords(vert2, grid_params)
    if cell1.row == cell2.row and cell1.col == cell2.col:
      raise ValueError(f'Two vertices are the same for edge {edge_nodes}')
    if cell1.row == cell2.row and cell1.col != cell2.col:
      if cell1.row == 0:
        continue
      start_col = min(cell1.col, cell2.col)
      end_col = max(cell1.col, cell2.col)
      for i in range(start_col, end_col):
        grid.set_edge_assignment(
            row=cell1.row - 1,
            col=i,
            edge_kind=paper_grid.EdgeKind.BOTTOM,
            edge_assignment=paper_grid.EdgeAssignment.FOLDED,
        )
    if cell1.col == cell2.col and cell1.row != cell2.row:
      if cell1.col == 0:
        continue
      start_row = min(cell1.row, cell2.row)
      end_row = max(cell1.row, cell2.row)
      for i in range(start_row, end_row):
        grid.set_edge_assignment(
            row=i,
            col=cell1.col - 1,
            edge_kind=paper_grid.EdgeKind.RIGHT,
            edge_assignment=paper_grid.EdgeAssignment.FOLDED,
        )
    if cell1.col != cell2.col and cell1.row != cell2.row:
      # Assert it is a diagonal, then walk through cells of it.
      if abs(cell1.col - cell2.col) != abs(cell1.row - cell2.row):
        raise ValueError('Only 45 degree creases are supported')
      steps = abs(cell1.col - cell2.col)
      main_cell, other_cell = (
          (cell1, cell2) if cell1.col < cell2.col else (cell2, cell1)
      )
      if other_cell.row - main_cell.row > 0:
        for i in range(steps):
          grid.set_edge_assignment(
              row=main_cell.row + i,
              col=main_cell.col + i,
              edge_kind=paper_grid.EdgeKind.LEFT_DIAG,
              edge_assignment=paper_grid.EdgeAssignment.FOLDED,
          )
      else:
        for i in range(steps):
          grid.set_edge_assignment(
              row=main_cell.row - i - 1,
              col=main_cell.col + i,
              edge_kind=paper_grid.EdgeKind.RIGHT_DIAG,
              edge_assignment=paper_grid.EdgeAssignment.FOLDED,
          )

  return grid
