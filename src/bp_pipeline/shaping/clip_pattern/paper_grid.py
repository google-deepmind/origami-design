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

"""Paper grid class.

Represents a crease pattern as a table where each cell is a square and can have
predefined set of creases.
"""

import dataclasses
import enum
import xml.etree.ElementTree as ET


class CutPattern(enum.Enum):
  """Cut pattern to add to the SVG."""

  NONE = 0
  DIAGONAL = 1


@dataclasses.dataclass(kw_only=True, frozen=True)
class SvgRenderingOptions:
  """Options for rendering the SVG."""

  cell_width: int
  cell_height: int
  cut_pattern: CutPattern = CutPattern.NONE

  @classmethod
  def from_default(cls) -> "SvgRenderingOptions":
    return SvgRenderingOptions(
        cell_width=50,
        cell_height=50,
        cut_pattern=CutPattern.NONE,
    )

  @classmethod
  def from_cut_pattern(cls, cut_pattern: CutPattern) -> "SvgRenderingOptions":
    return SvgRenderingOptions(
        cell_width=50,
        cell_height=50,
        cut_pattern=cut_pattern,
    )


SVG_RENDERING_OPTIONS_DEFAULT = SvgRenderingOptions.from_default()
SVG_RENDERING_OPTIONS_DIAGONAL = SvgRenderingOptions.from_cut_pattern(
    CutPattern.DIAGONAL
)


class EdgeAssignment(enum.Enum):
  """Edge assignment for a cell."""

  UNASSIGNED = 0
  FLAT = 1
  FOLDED = 2


def color_for_edge_assignment(edge_assignment: EdgeAssignment) -> str:
  if edge_assignment == EdgeAssignment.FLAT:
    return "yellow"
  elif edge_assignment == EdgeAssignment.FOLDED:
    return "red"
  elif edge_assignment == EdgeAssignment.UNASSIGNED:
    return "black"
  else:
    raise ValueError(f"Unknown edge assignment: {edge_assignment}")


class EdgeKind(enum.Enum):
  """Edge kind for a cell."""

  BOTTOM = 0
  RIGHT = 1
  RIGHT_DIAG = 2
  LEFT_DIAG = 3


@dataclasses.dataclass(kw_only=True, frozen=True)
class CellPosition:
  row: int
  col: int


@dataclasses.dataclass(kw_only=True)
class CellEdge:
  """Represents an edge in a cell."""

  kind: EdgeKind
  state: EdgeAssignment


@dataclasses.dataclass(kw_only=True)
class GridCell:
  """Represents a cell in a grid."""

  # NOTE: refactor edge to be a dict with support of richer properties.
  edges: dict[EdgeKind, CellEdge]
  original_position: CellPosition

  @classmethod
  def from_default(cls, row: int, col: int) -> "GridCell":
    return GridCell(
        edges={
            EdgeKind.BOTTOM: CellEdge(
                kind=EdgeKind.BOTTOM,
                state=EdgeAssignment.FLAT,
            ),
            EdgeKind.RIGHT: CellEdge(
                kind=EdgeKind.RIGHT, state=EdgeAssignment.FLAT
            ),
            EdgeKind.RIGHT_DIAG: CellEdge(
                kind=EdgeKind.RIGHT_DIAG, state=EdgeAssignment.FLAT
            ),
            EdgeKind.LEFT_DIAG: CellEdge(
                kind=EdgeKind.LEFT_DIAG, state=EdgeAssignment.FLAT
            ),
        },
        original_position=CellPosition(row=row, col=col),
    )

  def set_edge_assignment(
      self,
      *,
      edge_kind: EdgeKind,
      edge_assignment: EdgeAssignment,
  ):
    """Sets the edge assignment for a given cell and edge kind."""
    self.edges[edge_kind].state = edge_assignment

  def get_edge_assignment(self, edge_kind: EdgeKind) -> EdgeAssignment:
    """Returns the edge assignment for a given cell and edge kind."""
    return self.edges[edge_kind].state


@dataclasses.dataclass(kw_only=True)
class PaperGridData:
  """Represents crease pattern as a grid of cells."""

  num_rows: int
  num_cols: int
  cells: list[list[GridCell]]

  @classmethod
  def from_default(cls, num_rows: int, num_cols: int) -> "PaperGridData":
    """Creates a PaperGridData from a default grid."""
    return PaperGridData(
        num_rows=num_rows,
        num_cols=num_cols,
        cells=[
            [GridCell.from_default(row=row, col=col) for col in range(num_cols)]
            for row in range(num_rows)
        ],
    )

  def get_cell(self, row: int, col: int) -> GridCell:
    """Returns the cell at the given row and column."""
    if row < 0 or row >= self.num_rows:
      raise ValueError(f"Invalid row: {row}")
    if col < 0 or col >= self.num_cols:
      raise ValueError(f"Invalid col: {col}")
    return self.cells[row][col]

  def set_cell(self, cell: GridCell) -> None:
    """Returns the cell at the given row and column."""
    row = cell.original_position.row
    col = cell.original_position.col
    if row < 0 or row >= self.num_rows:
      raise ValueError(f"Invalid row: {row}")
    if col < 0 or col >= self.num_cols:
      raise ValueError(f"Invalid col: {col}")
    self.cells[row][col] = cell

  def set_edge_assignment(
      self,
      *,
      row: int,
      col: int,
      edge_kind: EdgeKind,
      edge_assignment: EdgeAssignment,
  ):
    """Sets the edge assignment for a given cell and edge kind."""
    cell = self.get_cell(row, col)
    cell.set_edge_assignment(
        edge_kind=edge_kind, edge_assignment=edge_assignment
    )

  def encode_to_svg_string_default(self) -> str:
    """Returns a svg string which represents the grid crease pattern."""
    return self.encode_to_svg_string(SVG_RENDERING_OPTIONS_DEFAULT)

  def encode_to_svg_string_diagonal_cut(self) -> str:
    """Returns a svg string which represents the grid crease pattern."""
    return self.encode_to_svg_string(SVG_RENDERING_OPTIONS_DIAGONAL)

  def encode_to_svg_string(self, options: SvgRenderingOptions) -> str:
    """Returns a svg string which represents the grid crease pattern."""
    cell_width = options.cell_width
    cell_height = options.cell_height
    svg = ET.Element(
        "svg",
        width=str(self.num_cols * cell_width),
        height=str(self.num_rows * cell_height),
        xmlns="http://www.w3.org/2000/svg",
    )

    ET.SubElement(
        svg,
        "line",
        x1="0",
        y1="0",
        x2="0",
        y2=f"{cell_width*self.num_rows}",
        stroke="black",
        stroke_width="1",
    )
    ET.SubElement(
        svg,
        "line",
        x1="0",
        y1="0",
        x2=f"{cell_width*self.num_cols}",
        y2="0",
        stroke="black",
        stroke_width="1",
    )
    for row in range(self.num_rows):
      for col in range(self.num_cols):
        cell = self.get_cell(row, col)
        color = {}
        color[EdgeKind.BOTTOM] = color_for_edge_assignment(
            cell.get_edge_assignment(EdgeKind.BOTTOM)
        )
        color[EdgeKind.RIGHT] = color_for_edge_assignment(
            cell.get_edge_assignment(EdgeKind.RIGHT)
        )
        color[EdgeKind.LEFT_DIAG] = color_for_edge_assignment(
            cell.get_edge_assignment(EdgeKind.LEFT_DIAG)
        )
        color[EdgeKind.RIGHT_DIAG] = color_for_edge_assignment(
            cell.get_edge_assignment(EdgeKind.RIGHT_DIAG)
        )
        if row == self.num_rows - 1:
          color[EdgeKind.BOTTOM] = "black"
        if col == self.num_cols - 1:
          color[EdgeKind.RIGHT] = "black"
        # BOTTOM EDGE
        ET.SubElement(
            svg,
            "line",
            x1=f"{cell_width*col}",
            y1=f"{cell_height*(row+1)}",
            x2=f"{cell_width*(col+1)}",
            y2=f"{cell_height*(row+1)}",
            stroke=color[EdgeKind.BOTTOM],
            stroke_width="1",
        )
        # RIGHT EDGE
        ET.SubElement(
            svg,
            "line",
            x1=f"{cell_width*(col+1)}",
            y1=f"{cell_height*row}",
            x2=f"{cell_width*(col+1)}",
            y2=f"{cell_height*(row+1)}",
            stroke=color[EdgeKind.RIGHT],
            stroke_width="1",
        )
        # LEFT_DIAG BOTTOM
        diag_color = color[EdgeKind.LEFT_DIAG]
        if options.cut_pattern == CutPattern.DIAGONAL:
          diag_color = "rgb(0,255,0)"
        ET.SubElement(
            svg,
            "line",
            x1=f"{cell_width*(col+0.5)}",
            y1=f"{cell_height*(row+0.5)}",
            x2=f"{cell_width*(col+1)}",
            y2=f"{cell_height*(row+1)}",
            stroke=diag_color,
            stroke_width="1",
        )
        # LEFT_DIAG TOP
        diag_color = color[EdgeKind.LEFT_DIAG]
        if options.cut_pattern == CutPattern.DIAGONAL:
          if row != 0 and col != 0:
            diag_color = "rgb(0,255,0)"
        ET.SubElement(
            svg,
            "line",
            x1=f"{cell_width*(col)}",
            y1=f"{cell_height*(row)}",
            x2=f"{cell_width*(col+0.5)}",
            y2=f"{cell_height*(row+0.5)}",
            stroke=diag_color,
            stroke_width="1",
        )
        # RIGHT_DIAG BOTTOM
        diag_color = color[EdgeKind.RIGHT_DIAG]
        ET.SubElement(
            svg,
            "line",
            x1=f"{cell_width*col}",
            y1=f"{cell_height*(row+1)}",
            x2=f"{cell_width*(col+0.5)}",
            y2=f"{cell_height*(row+0.5)}",
            stroke=diag_color,
            stroke_width="1",
        )

        # RIGHT_DIAG TOP
        diag_color = color[EdgeKind.RIGHT_DIAG]
        ET.SubElement(
            svg,
            "line",
            x1=f"{cell_width*(col+0.5)}",
            y1=f"{cell_height*(row+0.5)}",
            x2=f"{cell_width*(col+1)}",
            y2=f"{cell_height*(row)}",
            stroke=diag_color,
            stroke_width="1",
        )

    return ET.tostring(svg, "unicode")
