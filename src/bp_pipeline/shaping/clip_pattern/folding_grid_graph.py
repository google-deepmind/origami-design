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

"""Functions to convert paper grid to graph and manipulate it."""

import networkx as nx

from . import folding_grid
from . import paper_grid
from . import triangle_geometry


def triangle_key(
    row: int, col: int, tr: triangle_geometry.CellTriangle
) -> tuple[int, int, triangle_geometry.CellTriangle]:
  """Returns a unique key for a triangle in the grid."""
  return triangle_geometry.TriangleCellPosition(
      row=row, col=col, tr=tr, z_sign=1
  ).key()


def triangle_from_key(
    key: tuple[int, int, triangle_geometry.CellTriangle],
) -> triangle_geometry.TriangleCellPosition:
  return triangle_geometry.TriangleCellPosition(
      row=key[0], col=key[1], tr=key[2], z_sign=1
  )


def grid_to_nx_graph(grid: paper_grid.PaperGridData) -> nx.Graph:
  """Converts a paper grid to a networkx graph.

  Each node in the graph represents a TriangleCellPosition in the grid.
  Edges connect adjacent triangles.

  Args:
    grid: The paper grid to convert.

  Returns:
    A networkx graph representing the grid.
  """
  graph = nx.Graph()
  edges = []
  # Case 1:  Next triangleis within a square
  for row in range(grid.num_rows):
    for col in range(grid.num_cols):
      edges.extend([
          (
              triangle_key(row, col, triangle_geometry.CellTriangle.TOP),
              triangle_key(row, col, triangle_geometry.CellTriangle.RIGHT),
          ),
          (
              triangle_key(row, col, triangle_geometry.CellTriangle.RIGHT),
              triangle_key(row, col, triangle_geometry.CellTriangle.BOTTOM),
          ),
          (
              triangle_key(row, col, triangle_geometry.CellTriangle.BOTTOM),
              triangle_key(row, col, triangle_geometry.CellTriangle.LEFT),
          ),
          (
              triangle_key(row, col, triangle_geometry.CellTriangle.LEFT),
              triangle_key(row, col, triangle_geometry.CellTriangle.TOP),
          ),
      ])
  # Case 2: Next triangle is on the right.
  for row in range(grid.num_rows):
    for col in range(grid.num_cols - 1):
      edges.append(
          (
              triangle_key(row, col, triangle_geometry.CellTriangle.RIGHT),
              triangle_key(row, col + 1, triangle_geometry.CellTriangle.LEFT),
          ),
      )
  # Case 3: Next triangle is below.
  for col in range(grid.num_cols):
    for row in range(grid.num_rows - 1):
      edges.append(
          (
              triangle_key(row, col, triangle_geometry.CellTriangle.BOTTOM),
              triangle_key(row + 1, col, triangle_geometry.CellTriangle.TOP),
          ),
      )
  # networkx graph automatically adds nodes from edges.
  graph.add_edges_from(edges)
  return graph


def _get_edges_with_folded_assignment(
    grid: paper_grid.PaperGridData, row: int, col: int
) -> list[
    tuple[
        tuple[int, int, triangle_geometry.CellTriangle],
        tuple[int, int, triangle_geometry.CellTriangle],
    ]
]:
  """Returns graph edges with folded assignment for a given cell."""
  cell = grid.get_cell(row, col)
  folded_edges = []
  if (
      col < grid.num_cols - 1
      and cell.get_edge_assignment(folding_grid.EdgeKind.RIGHT)
      == folding_grid.EdgeAssignment.FOLDED
  ):
    folded_edges.append((
        triangle_key(row, col, triangle_geometry.CellTriangle.RIGHT),
        triangle_key(row, col + 1, triangle_geometry.CellTriangle.LEFT),
    ))
  if (
      row < grid.num_rows - 1
      and cell.get_edge_assignment(folding_grid.EdgeKind.BOTTOM)
      == folding_grid.EdgeAssignment.FOLDED
  ):
    folded_edges.append((
        triangle_key(row, col, triangle_geometry.CellTriangle.BOTTOM),
        triangle_key(row + 1, col, triangle_geometry.CellTriangle.TOP),
    ))
  if (
      cell.get_edge_assignment(folding_grid.EdgeKind.RIGHT_DIAG)
      == folding_grid.EdgeAssignment.FOLDED
  ):
    folded_edges.extend([
        (
            triangle_key(row, col, triangle_geometry.CellTriangle.TOP),
            triangle_key(row, col, triangle_geometry.CellTriangle.RIGHT),
        ),
        (
            triangle_key(row, col, triangle_geometry.CellTriangle.LEFT),
            triangle_key(row, col, triangle_geometry.CellTriangle.BOTTOM),
        ),
    ])
  if (
      cell.get_edge_assignment(folding_grid.EdgeKind.LEFT_DIAG)
      == folding_grid.EdgeAssignment.FOLDED
  ):
    folded_edges.extend([
        (
            triangle_key(row, col, triangle_geometry.CellTriangle.TOP),
            triangle_key(row, col, triangle_geometry.CellTriangle.LEFT),
        ),
        (
            triangle_key(row, col, triangle_geometry.CellTriangle.RIGHT),
            triangle_key(row, col, triangle_geometry.CellTriangle.BOTTOM),
        ),
    ])
  return folded_edges


def disconnect_where_folded(
    graph: nx.Graph, grid: paper_grid.PaperGridData
) -> nx.Graph:
  """Removes edges from graph that correspond to folded creases in the grid.

  If an edge in `grid` has `EdgeAssignment.FOLDED`, then in `graph` connection
  between triangles separated by this edge will be removed.

  Args:
    graph: The graph to remove edges from.
    grid: The paper grid with crease assignments.

  Returns:
    The updated graph with corresponding edges removed.
  """
  edges_to_remove = []
  for row in range(grid.num_rows):
    for col in range(grid.num_cols):
      edges_to_remove.extend(_get_edges_with_folded_assignment(grid, row, col))
  graph.remove_edges_from(edges_to_remove)
  return graph


def grid_to_nx_graph_disconnect_where_folded(
    grid: paper_grid.PaperGridData,
) -> nx.Graph:
  """Converts a paper grid to a networkx graph and removes "folded" edges.

  Args:
    grid: The paper grid to convert to graph. Edges corresponding to folded
      creases will be removed from the graph.

  Returns:
    A networkx graph representing the grid with "folded" edges removed.
  """
  graph = grid_to_nx_graph(grid)
  return disconnect_where_folded(graph, grid)


def get_one_node_from_each_component(
    graph: nx.Graph,
) -> list[tuple[int, int, triangle_geometry.CellTriangle]]:
  """Finds one arbitrary node from each connected component of a graph.

  Args:
      graph (nx.Graph): A networkx graph object. It can be connected or
        disconnected.

  Returns:
      list: A list of nodes, where each node belongs to a different
            connected component of the input graph. Returns an empty
            list if the graph has no nodes.
  """
  components = nx.connected_components(graph)
  representative_nodes = [next(iter(component)) for component in components]
  return representative_nodes


def get_folding_path(
    graph: nx.Graph,
    fixed_start_node: tuple[int, int, triangle_geometry.CellTriangle],
    end_node: tuple[int, int, triangle_geometry.CellTriangle],
) -> list[tuple[int, int, triangle_geometry.CellTriangle]]:
  """Finds the shortest path between two triangle nodes in the graph.

  Result can be used as a folding path, where the first node is the start of the
  folding which remains fixed, and every other other triangle in the path if
  folded towards it.

  Args:
    graph: The networkx graph representing the folding grid.
    fixed_start_node: The starting triangle key.
    end_node: The ending triangle key.

  Returns:
    A list of triangle keys representing the shortest path from
    `fixed_start_node` to `end_node`.
  """
  return nx.shortest_path(graph, source=fixed_start_node, target=end_node)
