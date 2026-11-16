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

"""Shape folded flap."""

from collections.abc import Sequence
import dataclasses
from typing import Any, Callable

import immutabledict
import networkx as nx
import numpy as np

from ...crease_pattern import crease
from . import drawing_machine
from . import drawing_machine_utils
from . import folding_grid
from . import folding_grid_graph
from . import paper_grid
from . import triangle_geometry
from . import world_space
from ... import timeout as timeout_lib

_ALIGN_MATRIX_FOR_TRIANGLE = immutabledict.immutabledict({
    triangle_geometry.CellTriangle.TOP: np.array(
        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    ),
    triangle_geometry.CellTriangle.RIGHT: np.array(
        [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    ),
    triangle_geometry.CellTriangle.BOTTOM: np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    ),
    triangle_geometry.CellTriangle.LEFT: np.array(
        [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    ),
})


def _keys_to_triangles(keys):
  return [folding_grid_graph.triangle_from_key(key) for key in keys]


def _convex_hull_for_triangle_group(
    triangle_group: list[triangle_geometry.TriangleCellPosition],
) -> np.ndarray:
  """Returns the convex hull of the vertices of the triangle group."""
  triangle_group_vertices = []
  for t in triangle_group:
    triangle_group_vertices.extend(t.vertices_2d())
  return drawing_machine_utils.convex_hull(
      triangle_group_vertices + np.array([0.5, -0.5])
  )


def _aligned_folded_transformation(
    align_triangle: triangle_geometry.TriangleCellPosition,
    folded_align_triangle: triangle_geometry.TriangleCellPosition,
    grid_size: int,
) -> np.ndarray:
  """Returns the transformation to draw shaping pattern at the right place.

  Args:
    align_triangle: The triangle for which shaping pattern was originally
      aligned.
    folded_align_triangle: The triangle from a section for which we will draw
      the shaping pattern. It must have basis_matrix, computed by folding path
      from align_triangle to folded_align_triangle.
    grid_size: The size of the grid on which the pattern is drawn.

  Returns:
    The transformation matrix to be used in CreasePatternDrawingMachine before
    drawing the shaping crease lines.
  """
  canvas = world_space.World()
  canvas.transform(scale=(1 / grid_size, 1 / grid_size, 1))
  canvas.transform(translate=(0.5, 0.5, 0))
  canvas.transform(scale=(1, -1, 1))

  tmp_canvas = world_space.World()
  tmp_canvas.push_local_transform(folded_align_triangle.basis_matrix)
  tmp_canvas.transform(rotate_deg=90, rotate_axis=(0, 0, 1))
  tmp_canvas.push_local_transform(_ALIGN_MATRIX_FOR_TRIANGLE[align_triangle.tr])

  canvas.push_local_transform(tmp_canvas.peek_transform())
  canvas.transform(translate=(-0.5, -0.25, 0))
  return canvas.peek_transform()


_TRIANGLE_VERTEX_OFFSETS = {
    triangle_geometry.CellTriangle.TOP: np.array(
        [[-0.5, 0.5], [0.5, 0.5], [0.0, 0.0]]
    ),
    triangle_geometry.CellTriangle.RIGHT: np.array(
        [[0.5, 0.5], [0.5, -0.5], [0.0, 0.0]]
    ),
    triangle_geometry.CellTriangle.BOTTOM: np.array(
        [[0.5, -0.5], [-0.5, -0.5], [0.0, 0.0]]
    ),
    triangle_geometry.CellTriangle.LEFT: np.array(
        [[-0.5, -0.5], [-0.5, 0.5], [0.0, 0.0]]
    ),
}


def _get_polygon_points_fast(
    graph: nx.Graph,
    main_triangle_key: tuple[int, int, triangle_geometry.CellTriangle],
    grid_size: int,
) -> np.ndarray:
  """Returns polygon points using vectorized numpy operations.

  Args:
    graph: The graph representing connectivity of triangles.
    main_triangle_key: The key of the triangle whose connected component defines
      the polygon.
    grid_size: The size of the grid on which the pattern is drawn.

  Returns:
    An array of polygon vertices.

  Avoids creating TriangleCellPosition objects by computing triangle vertices
  directly from key tuples using a pre-computed offset lookup table.
  """
  keys = nx.node_connected_component(graph, main_triangle_key)
  # Build all triangle vertices directly from keys, avoiding object creation.
  # Each key is (row, col, CellTriangle). vertices_2d returns vertices with
  # center = (col, -row), so we compute centers and add offsets per triangle
  # type.
  n = len(keys)
  all_vertices = np.empty((n * 3, 2), dtype=np.float64)
  i = 0
  for row, col, tr in keys:
    center_x = float(col)
    center_y = -float(row)
    offsets = _TRIANGLE_VERTEX_OFFSETS[tr]
    all_vertices[i] = (center_x + offsets[0, 0], center_y + offsets[0, 1])
    all_vertices[i + 1] = (center_x + offsets[1, 0], center_y + offsets[1, 1])
    all_vertices[i + 2] = (center_x + offsets[2, 0], center_y + offsets[2, 1])
    i += 3
  # Apply the offset that _convex_hull_for_triangle_group adds.
  all_vertices[:, 0] += 0.5
  all_vertices[:, 1] -= 0.5
  # Compute convex hull.
  hull_points = drawing_machine_utils.convex_hull(all_vertices)
  # Scale and flip y in one vectorized operation.
  hull_points[:, 0] /= grid_size
  hull_points[:, 1] /= -grid_size
  return hull_points


def get_polygon_points(
    graph: nx.Graph,
    main_triangle_key: tuple[int, int, triangle_geometry.CellTriangle],
    grid_size: int,
    fast: bool = True,
) -> np.ndarray:
  """Returns the polygon points for the given main triangle.

  Args:
    graph: The graph representing connectivity of triangles.
    main_triangle_key: The key of the triangle whose connected component defines
      the polygon.
    grid_size: The size of the grid on which the pattern is drawn.
    fast: If True, uses a vectorized numpy implementation. If False, uses the
      original implementation based on TriangleCellPosition objects.

  Returns:
    An array of polygon vertices.
  """
  if fast:
    return _get_polygon_points_fast(graph, main_triangle_key, grid_size)
  segment_triangles = _keys_to_triangles(
      nx.node_connected_component(graph, main_triangle_key)
  )
  return np.array([
      np.array([v[0] / grid_size, -v[1] / grid_size])
      for v in _convex_hull_for_triangle_group(segment_triangles)
  ])


@dataclasses.dataclass(frozen=True)
class PrecomputedData:
  """Precomputed data for clip pattern shaping.

  This data depends only on grids and align_triangles, not on draw_pattern.
  It can be computed once and reused across multiple calls to
  get_clip_pattern_shaping_lines_for_precomputed_data.
  """

  hinges_graph: nx.Graph
  base_graph: nx.Graph
  representative_nodes: list[tuple[int, int, triangle_geometry.CellTriangle]]
  polygon_points: dict[
      tuple[int, int, triangle_geometry.CellTriangle], np.ndarray
  ]
  align_triangle_to_main_triangles: dict[
      tuple[int, int, triangle_geometry.CellTriangle],
      list[tuple[int, int, triangle_geometry.CellTriangle]],
  ]


def get_precomputed_data(
    grid_size: int,
    hinges_grid: paper_grid.PaperGridData,
    grid: paper_grid.PaperGridData,
    align_triangles: list[tuple[int, int, triangle_geometry.CellTriangle]],
) -> PrecomputedData:
  """Precomputes data for clip pattern shaping.

  The returned data can be cached and reused across multiple calls to
  get_clip_pattern_shaping_lines_for_precomputed_data.

  Args:
    grid_size: The size of the grid on which the pattern is drawn.
    hinges_grid: The paper grid which stores hinges lines.
    grid: The paper grid that stores crease pattern of a base.
    align_triangles: The list of align triangle keys for which to precompute.

  Returns:
    A PrecomputedData instance.
  """
  hinges_graph = folding_grid_graph.grid_to_nx_graph_disconnect_where_folded(
      hinges_grid
  )
  base_graph = folding_grid_graph.grid_to_nx_graph_disconnect_where_folded(grid)
  base_graph = folding_grid_graph.disconnect_where_folded(
      base_graph, hinges_grid
  )
  representative_nodes = folding_grid_graph.get_one_node_from_each_component(
      base_graph
  )

  polygon_points_map = {}
  for node in representative_nodes:
    polygon_points_map[node] = get_polygon_points(base_graph, node, grid_size)

  align_triangle_to_main_triangles = {}
  for align_key in align_triangles:
    hinges_triangle_keys = nx.node_connected_component(hinges_graph, align_key)
    intersected = set(representative_nodes) & set(hinges_triangle_keys)
    align_triangle_to_main_triangles[align_key] = sorted(
        intersected, key=lambda t: (t[0], t[1], t[2].value)
    )

  return PrecomputedData(
      hinges_graph=hinges_graph,
      base_graph=base_graph,
      representative_nodes=representative_nodes,
      polygon_points=polygon_points_map,
      align_triangle_to_main_triangles=align_triangle_to_main_triangles,
  )


def _get_drawing_machine_transformation(
    hinges_graph: nx.Graph,
    main_triangle_key: tuple[int, int, triangle_geometry.CellTriangle],
    align_triangle: triangle_geometry.TriangleCellPosition,
    grid: paper_grid.PaperGridData,
    grid_size: int,
) -> np.ndarray:
  """Returns the transformation to draw shaping pattern at the right place.

  Args:
    hinges_graph: Graph representation of the paper grid where edges
      corresponding to hinge creases were deleted.
    main_triangle_key: The key of the triangle which is in the segment for which
      we position CreasePatternDrawingMachine now. Any other triangle in the
      same segment would produce the same output.
    align_triangle: The triangle for which shaping pattern was originally
      aligned.
    grid: The paper grid that stores crease pattern of a base.
    grid_size: The size of the grid on which the pattern is drawn.
  """
  path_keys = folding_grid_graph.get_folding_path(
      graph=hinges_graph,
      fixed_start_node=main_triangle_key,
      end_node=align_triangle.key(),
  )
  path_triangles = _keys_to_triangles(path_keys)
  folded_triangles = folding_grid.FoldingGrid.from_paper_grid(grid).fold_path(
      path_triangles, do_not_fold=False, compute_basis=True
  )
  folded_align_triangle = folded_triangles[-1]
  return _aligned_folded_transformation(
      align_triangle, folded_align_triangle, grid_size
  )


def get_clip_pattern_shaping_lines_for_precomputed_data(
    grid_size: int,
    grid: paper_grid.PaperGridData,
    align_triangle: triangle_geometry.TriangleCellPosition,
    draw_pattern: Callable[[drawing_machine.CreasePatternDrawingMachine], None],
    precomputed_data: PrecomputedData,
    disable_clipping: bool = False,
    bounding_rectangles: Sequence[Any] | None = None,
    global_refine: bool = False,
) -> list[drawing_machine.Line]:
  """Returns the shaping lines using precomputed data.

  This is the core implementation that uses precomputed data. Use
  get_precomputed_data() to create the precomputed data, and cache it across
  calls for performance.

  Args:
    grid_size: The size of the grid on which the pattern is drawn.
    grid: The paper grid that stores crease pattern of a base.
    align_triangle: The triangle which sets position and orientation for
      CreasePatternDrawingMachine on a folded base.
    draw_pattern: The function to draw crease with the given
      CreasePatternDrawingMachine.
    precomputed_data: Precomputed data from get_precomputed_data().
    disable_clipping: Whether to disable clipping. Useful for debugging.
    bounding_rectangles: Optional sequence of rectangles to scope the shaping
      lines to. If provided, lines and machine placements outside these
      rectangles are pruned to avoid face explosion / crease bloat on untouched
      regions.
    global_refine: If True, disables localized pruning and performs global
      crease placement across the entire base.
  """
  main_triangles_keys = precomputed_data.align_triangle_to_main_triangles[
      align_triangle.key()
  ]
  if bounding_rectangles is not None and not global_refine:
    filtered_keys = []
    margin = 1.5 / grid_size
    for key in main_triangles_keys:
      row, col, tr = key
      cp = triangle_geometry.TriangleCellPosition(
          row=row, col=col, z_sign=1, tr=tr
      ).center_point()
      cx = (cp[1] + 0.5) / grid_size
      cy = (cp[0] + 0.5) / grid_size
      if any(
          (r.x_start - margin <= cx <= r.x_start + r.x_len + margin)
          and (r.y_start - margin <= cy <= r.y_start + r.y_len + margin)
          for r in bounding_rectangles
      ):
        filtered_keys.append(key)
    if filtered_keys:
      main_triangles_keys = filtered_keys

  all_creases = []
  margin = 1.5 / grid_size
  for main_triangle_key in main_triangles_keys:
    timeout_lib.maybe_timeout_cooperatively()
    if disable_clipping:
      polygon_points = np.array([])
    else:
      polygon_points = precomputed_data.polygon_points[main_triangle_key]

    drawing_machine_transformation = _get_drawing_machine_transformation(
        precomputed_data.hinges_graph,
        main_triangle_key,
        align_triangle,
        grid,
        grid_size,
    )
    machine = drawing_machine.CreasePatternDrawingMachine()
    machine.canvas.push_local_transform(drawing_machine_transformation)
    draw_pattern(machine)
    clipped_lines = machine.get_clipped_lines(polygon_points)
    min_len = crease.DEFAULT_TOLERANCE * (1.0 + 1e-6)
    for line in clipped_lines:
      if np.linalg.norm(line.p1[:2] - line.p2[:2]) > min_len:
        all_creases.append(line)
  return all_creases


def get_clip_pattern_shaping_lines(
    grid_size: int,
    hinges_grid: paper_grid.PaperGridData,
    grid: paper_grid.PaperGridData,
    align_triangle: triangle_geometry.TriangleCellPosition,
    draw_pattern: Callable[[drawing_machine.CreasePatternDrawingMachine], None],
    disable_clipping: bool = False,
    bounding_rectangles: Sequence[Any] | None = None,
    global_refine: bool = False,
) -> list[drawing_machine.Line]:
  """Returns the shaping lines for a section specified by align_triangle.


  This is a convenience function that computes precomputed data and then calls
  get_clip_pattern_shaping_lines_for_precomputed_data. If you need to call this
  for multiple align_triangles with the same grids, use get_precomputed_data()
  and get_clip_pattern_shaping_lines_for_precomputed_data() directly for better
  performance.

  Args:
    grid_size: The size of the grid on which the pattern is drawn.
    hinges_grid: The paper grid which stores hinges lines. Shaping will be
      applied to one of the regions. Region is surrounded by hinge lines or
      borders.
    grid: The paper grid that stores crease pattern of a base. Together with
      hinges_grid, it specifies sections which remain flat. align_triangle
      belongs to one of the section, which belongs to one of the regions (flap
      or river). Shaping pattern will be drawn and clipped.
    align_triangle: The triangle which sets position and orientation for
      CreasePatternDrawingMachine on a folded base. It is used to determine
      which flap or river function needs to shape and how to position
      CreasePatternDrawingMachine for every flat section of the region.
    draw_pattern: The function to draw crease with the given
      CreasePatternDrawingMachine.
    disable_clipping: Whether to disable clipping. Useful for debugging.
    bounding_rectangles: Optional sequence of rectangles to scope the shaping.
    global_refine: If True, disables localized pruning.
  """
  precomputed_data = get_precomputed_data(
      grid_size=grid_size,
      hinges_grid=hinges_grid,
      grid=grid,
      align_triangles=[align_triangle.key()],
  )
  return get_clip_pattern_shaping_lines_for_precomputed_data(
      grid_size=grid_size,
      grid=grid,
      align_triangle=align_triangle,
      draw_pattern=draw_pattern,
      precomputed_data=precomputed_data,
      disable_clipping=disable_clipping,
      bounding_rectangles=bounding_rectangles,
      global_refine=global_refine,
  )
