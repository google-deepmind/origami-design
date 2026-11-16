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

"""Clip pattern shaping adapter for CreasePattern."""

from collections.abc import Callable, Sequence
import dataclasses
import enum
import functools
from typing import Any
import networkx as nx
import numpy as np
from ...crease_pattern import crease_pattern
from . import drawing_machine
from . import clip_pattern_shaping
from . import clip_patterns
from .. import shaping_utils
from . import converters as geom_converters
from . import folding_grid_graph
from . import paper_grid
from . import triangle_geometry
from ... import timeout as timeout_lib


class TemplateName(enum.StrEnum):
  FOLD = "fold"
  TEST = "test"
  TURN_AND_NARROW = "turn_and_narrow"
  BEND_WHEN_TURN_AND_NARROW = "bend_when_turn_and_narrow"
  PLEAT_TO_HIDE = "pleat_to_hide"


def add_lines_as_creases(
    cp: crease_pattern.CreasePattern,
    lines: list[drawing_machine.Line],
) -> crease_pattern.CreasePattern:
  """Adds shaping creases to the crease pattern.

  Args:
    cp: The crease pattern to add creases to.
    lines: The lines to add as creases.

  Returns:
    A new crease pattern with the added creases.
  """
  creases = [line.to_crease() for line in lines]
  return crease_pattern.CreasePattern(creases=cp.creases + creases)


def get_clip_pattern_template_draw_pattern(
    *,
    template_name: TemplateName,
    margin: float,
    step: float,
    bend_angle: float = 0.0,
    length: float | None = None,
    pleat_distance: float = 0.25,
) -> Callable[[drawing_machine.CreasePatternDrawingMachine], None]:
  """Returns a function that draws a clip pattern template."""

  def draw_pattern(
      machine: drawing_machine.CreasePatternDrawingMachine,
  ) -> None:
    machine.canvas.transform(translate=(0, margin, 0))
    machine.canvas.transform(translate=(0, step, 0))
    match template_name:
      case TemplateName.TEST:
        clip_patterns.rotate(machine, 90)
      case TemplateName.TURN_AND_NARROW:
        clip_patterns.turn_and_narrow(machine, length=length)
      case TemplateName.BEND_WHEN_TURN_AND_NARROW:
        clip_patterns.bend_when_turn_and_narrow(machine, bend_angle)
      case TemplateName.PLEAT_TO_HIDE:
        base_length = length if length is not None else 10.0
        effective_length = base_length - step - margin
        clip_patterns.pleat_to_hide(
            machine, length=effective_length, distance=pleat_distance
        )
      case _:
        raise ValueError(f"Unexpected template name: {template_name}")

  return draw_pattern


def get_clip_pattern_fold_draw_pattern(
    *,
    margin: float,
    step: float,
    bend_angle: float,
    rotate_angle: float,
    enable_narrow: bool,
    narrow_symmetric: bool,
    narrow_n_times: int | None = None,
    narrow_length: float | None = None,
) -> Callable[[drawing_machine.CreasePatternDrawingMachine], None]:
  """Returns a function that draws a clip pattern fold.

  Args:
    margin: The margin step to use between subpatterns.
    step: Move along by step value before applying the fold pattern. Must be >=0
      and less than the length of the flap.
    bend_angle: The angle in degrees between [-180, 180] to bend the flap (3D).
      Value 0 means flap remains straight. Angle determines a fold angle of a
      crease.
    rotate_angle: The angle in degrees between [-180, 180] to rotate the flap.
      Value 0 means flap remains straight.
    enable_narrow: Set to True to apply narrowing to the flap.
    narrow_symmetric: Whether to apply symmetric narrowing or asymmetric.
    narrow_n_times: The number of times to narrow the flap. Width will be
      1/narrow_n_times of the original width.
    narrow_length: The length of the flap to apply narrowing to. If None, the
      narrowing will be applied to the whole flap (not a river).

  Returns:
    A callable that takes a CreasePatternDrawingMachine and draws a pattern.
  """

  def _noop_step(
      machine: drawing_machine.CreasePatternDrawingMachine,
      step: float,
  ):
    clip_patterns.noop_step(machine, step)
    return step

  def draw_pattern(machine: drawing_machine.CreasePatternDrawingMachine):
    accum_step = _noop_step(machine, step)
    if not np.isclose(bend_angle, 0):
      if np.isclose(step, 0):
        # Small step is required to avoid collision with the hinge.
        accum_step += _noop_step(machine, margin)
      clip_patterns.bend(machine, bend_angle)
      accum_step += _noop_step(machine, margin)
    else:
      if np.isclose(step, 0):
        # Small step is required to avoid collision with the hinge.
        accum_step += _noop_step(machine, margin)
    if not np.isclose(rotate_angle, 0):
      clip_patterns.rotate(machine, rotate_angle)
      accum_step += _noop_step(machine, 1.0)  # required after a rotation
      accum_step += _noop_step(machine, margin)
    if enable_narrow:
      length = None
      if narrow_length is not None:
        length = round(narrow_length - accum_step, 8)
      if narrow_n_times is None or narrow_n_times < 2:
        n_times = 2
      else:
        n_times = narrow_n_times
      clip_patterns.narrow(
          machine, length=length, symmetric=narrow_symmetric, n_times=n_times
      )

  return draw_pattern


def find_align_triangles(
    hinges_grid: paper_grid.PaperGridData,
) -> list[tuple[int, int, triangle_geometry.CellTriangle]]:
  """Finds the align triangle for every fragment in the packing.

  Hinges split crease pattern into fragments. For the shaping algorithm to work
  on a fragment, we need a triangle, we call it align triangle, whose base is on
  the hinge. We sort the triangles in each fragment and return the first one to
  ensure deterministic behavior.

  Args:
    hinges_grid: Paper grid built from the packing. Folded creases are hinges.

  Returns:
    A list of align triangles, one for each fragment.
  """

  hinges_graph_connected = folding_grid_graph.grid_to_nx_graph(hinges_grid)
  hinges_graph = folding_grid_graph.grid_to_nx_graph_disconnect_where_folded(
      hinges_grid
  )
  components = nx.connected_components(hinges_graph)
  grid_center = (hinges_grid.num_rows / 2, hinges_grid.num_cols / 2)
  align_triangles = []
  for component in components:
    timeout_lib.maybe_timeout_cooperatively()
    candidate_nodes = [
        node
        for node in component
        if hinges_graph.degree(node) < hinges_graph_connected.degree(node)
    ]
    if candidate_nodes:
      align_triangles.append(
          min(
              candidate_nodes,
              # Sort by distance to paper center, then by triangle type
              # to break ties.
              key=lambda x: (
                  round(
                      (x[0] - grid_center[0]) ** 2
                      + (x[1] - grid_center[1]) ** 2,
                      8,
                  ),
                  -x[2],
              ),
          )
      )
  return align_triangles


@dataclasses.dataclass(frozen=True)
class ClipPatternShapingAdapter:
  """Adapter for shaping."""

  grid_size: int
  base_grid: paper_grid.PaperGridData
  hinges_grid: paper_grid.PaperGridData
  align_triangles: list[tuple[int, int, triangle_geometry.CellTriangle]]

  # Precomputed normalized (cx, cy) centers for each entry in align_triangles.
  # Avoids recomputing center_point() in call to get_align_triangle_for_rect.
  _align_triangle_centers: list[tuple[float, float]] = dataclasses.field(
      default_factory=list, repr=False
  )

  @classmethod
  def _compute_centers(
      cls,
      align_triangles: list[tuple[int, int, triangle_geometry.CellTriangle]],
      grid_size: int,
  ) -> list[tuple[float, float]]:
    """Precomputes normalized triangle centers for fast rectangle lookup."""
    centers = []
    for r, c, d in align_triangles:
      cp = triangle_geometry.TriangleCellPosition(
          row=r, col=c, z_sign=1, tr=d
      ).center_point()
      # Replicates the axis-swap, offset, and normalization done in
      # get_align_triangle_for_rect so that the hot loop is just comparisons.
      cx = (cp[1] + 0.5) / grid_size
      cy = (cp[0] + 0.5) / grid_size
      centers.append((cx, cy))
    return centers

  @classmethod
  def from_base_cp_and_packing_svg(
      cls,
      base_cp: crease_pattern.CreasePattern,
      packing_svg: str,
  ) -> "ClipPatternShapingAdapter":
    """Creates a new instance from a base crease pattern and packing svg."""

    grid_size = base_cp.grid_size
    base_grid = geom_converters.crease_pattern_to_paper_grid(
        pattern=base_cp, grid_size=grid_size
    )
    hinges_grid = geom_converters.svg_to_paper_grid(
        svg=packing_svg, grid_size=grid_size
    )
    align_triangles = find_align_triangles(hinges_grid)
    centers = cls._compute_centers(align_triangles, grid_size)
    return ClipPatternShapingAdapter(
        grid_size=grid_size,
        base_grid=base_grid,
        hinges_grid=hinges_grid,
        align_triangles=align_triangles,
        _align_triangle_centers=centers,
    )

  @classmethod
  def from_crease_patterns(
      cls,
      base_cp: crease_pattern.CreasePattern,
      packing_cp: crease_pattern.CreasePattern,
  ) -> "ClipPatternShapingAdapter":
    """Creates a new instance from a crease pattern.

    Args:
      base_cp: The base crease pattern.
      packing_cp: The packing crease pattern.

    Returns:
      A new ClipPatternShapingAdapter instance.
    """
    grid_size = base_cp.grid_size
    base_grid = geom_converters.crease_pattern_to_paper_grid(
        pattern=base_cp, grid_size=grid_size
    )
    hinges_grid = geom_converters.packing_cp_to_hinges_paper_grid(
        packing_cp=packing_cp, grid_size=grid_size
    )
    align_triangles = find_align_triangles(hinges_grid)
    centers = cls._compute_centers(align_triangles, grid_size)

    return ClipPatternShapingAdapter(
        grid_size=grid_size,
        base_grid=base_grid,
        hinges_grid=hinges_grid,
        align_triangles=align_triangles,
        _align_triangle_centers=centers,
    )

  @functools.cached_property
  def _hinges_graph(self) -> nx.Graph:
    """Graph derived from hinges_grid with folded edges disconnected."""
    return folding_grid_graph.grid_to_nx_graph_disconnect_where_folded(
        self.hinges_grid
    )

  @functools.cached_property
  def _base_graph(self) -> nx.Graph:
    """Graph derived from base_grid with folded edges (from both grids) disconnected."""
    graph = folding_grid_graph.grid_to_nx_graph_disconnect_where_folded(
        self.base_grid
    )
    return folding_grid_graph.disconnect_where_folded(graph, self.hinges_grid)

  @functools.cached_property
  def _precomputed_data(self) -> clip_pattern_shaping.PrecomputedData:
    """Precomputed data for clip pattern shaping."""
    return clip_pattern_shaping.get_precomputed_data(
        grid_size=self.grid_size,
        hinges_grid=self.hinges_grid,
        grid=self.base_grid,
        align_triangles=self.align_triangles,
    )

  def get_clip_pattern_shaping_lines(
      self,
      rectangles: list[shaping_utils.Rectangle],
      draw_pattern: Callable[
          [drawing_machine.CreasePatternDrawingMachine], None
      ],
      bounding_rectangles: Sequence[Any] | None = None,
      global_refine: bool = False,
  ) -> list[drawing_machine.Line]:
    """Applies shaping to the rectangle.

    Args:
      rectangles: The rectangle to apply shaping to. A list to support rivers.
      draw_pattern: The draw pattern to use for shaping.
      bounding_rectangles: Optional sequence of rectangles to scope shaping.
      global_refine: If True, disables localized pruning.

    Returns:
      A list of lines representing the shaping creases.

    Raises:
      ValueError: If no align triangle is found within the rectangle.
    """
    align_triangle = None
    for rectangle in rectangles:
      try:
        align_triangle = self.get_align_triangle_for_rect(rectangle)
        break
      except ValueError:
        continue
    if align_triangle is None:
      raise ValueError(
          f"No align triangle found within the rectangle. Rect: {rectangles}"
      )
    return clip_pattern_shaping.get_clip_pattern_shaping_lines_for_precomputed_data(
        grid_size=self.grid_size,
        grid=self.base_grid,
        align_triangle=triangle_geometry.TriangleCellPosition.from_key(
            align_triangle
        ),
        draw_pattern=draw_pattern,
        precomputed_data=self._precomputed_data,
        bounding_rectangles=bounding_rectangles
        if bounding_rectangles is not None
        else rectangles,
        global_refine=global_refine,
    )

  def get_align_triangle_for_rect(
      self, rect: shaping_utils.Rectangle
  ) -> tuple[int, int, triangle_geometry.CellTriangle]:
    """Returns the align triangle for the fragment covered by the rectangle.

    Args:
      rect: The rectangle to find the align triangle for.

    Returns:
      The align triangle for the rectangle as a tuple.
      Meaning of the tuple elements: (row, col, {TOP, BOTTOM, LEFT, RIGHT}),
      which determines a square in the grid and which out of 4 triangles to use.
      4 triangles are formed by drawing two diagonals on the square.

    Raises:
      ValueError: If no align triangle is found within the rectangle.
    """
    for (r, c, d), (cx, cy) in zip(
        self.align_triangles, self._align_triangle_centers
    ):
      if (
          cx >= rect.x_start
          and cx <= rect.x_start + rect.x_len
          and cy >= rect.y_start
          and cy <= rect.y_start + rect.y_len
      ):
        return (r, c, d)
    col_min = max(0, int(np.floor(rect.x_start * self.grid_size)))
    col_max = min(
        self.grid_size - 1,
        int(np.ceil((rect.x_start + rect.x_len) * self.grid_size)) - 1,
    )
    row_min = max(0, int(np.floor(rect.y_start * self.grid_size)))
    row_max = min(
        self.grid_size - 1,
        int(np.ceil((rect.y_start + rect.y_len) * self.grid_size)) - 1,
    )
    for r in range(row_min, row_max + 1):
      for c in range(col_min, col_max + 1):
        for d in triangle_geometry.CellTriangle:
          node = (r, c, d)
          if node not in self._hinges_graph:
            continue
          cp = triangle_geometry.TriangleCellPosition(
              row=r, col=c, z_sign=1, tr=d
          ).center_point()
          cx = (cp[1] + 0.5) / self.grid_size
          cy = (cp[0] + 0.5) / self.grid_size
          if (
              cx >= rect.x_start
              and cx <= rect.x_start + rect.x_len
              and cy >= rect.y_start
              and cy <= rect.y_start + rect.y_len
          ):
            comp = nx.node_connected_component(self._hinges_graph, node)
            for cand in self.align_triangles:
              if cand in comp:
                return cand
    raise ValueError(
        f"No align triangle found within the rectangle. Rect: {rect}"
    )
