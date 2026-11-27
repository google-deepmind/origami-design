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

"""Geometric folding heuristics for 3D shaping."""

import collections
from collections.abc import Sequence
import dataclasses
import logging
import math
import typing
from typing import NotRequired, TypedDict, cast

import numpy as np

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import point as point_utils
from ..packing import packing_output as packing_output_utils
from ..packing import types as packing_types
from ..solving import hinges as hinge_utils
from ..stick_figures import stick_figure as sf_utils
from . import geometric_fold_3d_heuristics_utils
from . import geometric_simple_fold
from . import geometric_utils
from . import heuristics
from . import heuristics_3d
from . import heuristics_utils
from . import shaping_elements
from . import shaping_utils
from . import types as shaping_types

_NINETY_DEG_SLOPE_TOLERANCE = 0.1
_MAX_UNROTATED_PASSES = 3


class ElementShapingResult(TypedDict):
  """Result of shaping a single element."""

  simple_fold_info: heuristics_3d.SimpleFoldInfo
  shaping_element: shaping_elements.ShapingElement
  shaping_creases: Sequence[crease_utils.Crease]


class Heuristics3DShapingResult(TypedDict):
  """Result of heuristics 3D shaping."""

  shaping_creases: Sequence[crease_utils.Crease]
  # Mapping from element name to a list of element shaping results.
  detailed_shaping_info: dict[str, list[ElementShapingResult]]
  # Mapping from unshaped element name to a structured reason string.
  skip_reasons: NotRequired[dict[str, str]]


@dataclasses.dataclass(frozen=True)
class InscribeHParams:
  """Parameters for inscribing a simple fold plus some other hparams.

  Attributes:
    avoid_hinges: Whether to attempt to avoid hinges by inscribing deeper.
    inscribe_depth: The depth (in grid units) to inscribe the simple fold
      unrelated to avoid_hinges.
    hinge_inscribe_depth: The depth to additionally inscribe when avoiding
      hinges.
    ninety_deg_slope_tolerance: The tolerance for ninety degree angles. Angles
      close (under this tolerance) to ninety degrees are treated as ninety
      degrees.
    max_slope_thresh: The maximum slope for a simple fold. Anything above this
      threshold is removed.
    clip_slope_thresh: The threshold to clip the slope of a simple fold to if
      it's not above max_slope_thresh.
  """

  avoid_hinges: bool = False
  inscribe_depth: float = 0.0
  hinge_inscribe_depth: float = (
      geometric_fold_3d_heuristics_utils.INSCRIBE_FRACTION
  )
  ninety_deg_slope_tolerance: float = _NINETY_DEG_SLOPE_TOLERANCE
  max_slope_thresh: float = heuristics_3d.MAX_SLOPE
  clip_slope_thresh: float = 5.0


@dataclasses.dataclass(frozen=True)
class GeometricFolding3DHParams:
  """Parameters for geometric folding 3D shaping.

  Attributes:
    shape_subset: The subset of elements to shape.
    default_inscribe_hparams: The default inscribe hparams for all elements.
    check_ff_violations: Whether to check for flap folding violations.
    strategy: The 3D folding strategy to use.
    per_flap_strategy: Per-flap overrides for the folding strategy.
    per_flap_hinge_inscribe_depth: Per-flap overrides for hinge inscribe depth.
    measure_base_directions: Whether to measure initial flap directions along or
      against the root directly from the flat-folded base instead of relying
      solely on hinge-inferred fold states.
    priority: Element names to shape first so high-priority elements win when
      later elements are discarded for local flat-foldability violations.
    direction_graph_data: Optional precomputed graph data of the unshaped base
      for `measure_base_directions` (defaults to the graph of `base_cp`). Use
      when flat pre-shaping has already altered flap boundaries.
    use_tree_root: When True, anchor 3D shaping postures and reference frames at
      `stick_figure.root` rather than the heuristic root selector.
  """

  shape_subset: Sequence[str] | None = None
  default_inscribe_hparams: InscribeHParams = InscribeHParams()
  check_ff_violations: bool = False
  strategy: heuristics_3d.FoldStrategy3D = heuristics_3d.FoldStrategy3D.DEFAULT
  per_flap_strategy: dict[str, heuristics_3d.FoldStrategy3D] = (
      dataclasses.field(default_factory=dict)
  )
  per_flap_hinge_inscribe_depth: dict[str, float] = dataclasses.field(
      default_factory=dict
  )
  measure_base_directions: bool = True
  priority: Sequence[str] = ()
  direction_graph_data: geometric_simple_fold.GraphData | None = None
  use_tree_root: bool = False


def _get_rotation_axis_slope(
    simple_fold_direction: np.ndarray,
) -> float | None:
  """Returns the slope of the simple fold direction."""
  if math.isclose(simple_fold_direction[0], 0.0):
    return None
  return simple_fold_direction[1] / simple_fold_direction[0]


def _clip_to_slope_one(simple_fold_direction: np.ndarray) -> np.ndarray:
  """Returns a vector with slope 1 or -1.

  The sign of the returned vector matches the sign of the input vector.

  Args:
      simple_fold_direction: The simple fold direction to clip.

  Returns:
      A vector with slope 1 or -1.
  """
  return np.array([
      float(np.sign(simple_fold_direction[0])),
      float(np.sign(simple_fold_direction[1])),
      0.0,
  ])


def _clip_slope_to_max(
    simple_fold_direction: np.ndarray, max_slope: float
) -> np.ndarray:
  """Returns a vector with slope at most max_slope."""
  # Normalize so that first component is 1.
  simple_fold_direction = simple_fold_direction / simple_fold_direction[0]
  return np.array([
      1.0,
      np.clip(simple_fold_direction[1], -max_slope, max_slope),
      0.0,
  ])


def _get_opposite_hinge_points_no_elevation(
    boundary: shaping_elements.ShapingBoundary,
    grid_size: int,
) -> tuple[point_utils.Point, point_utils.Point]:
  """Returns two points on the boundary that are on paper."""
  cur_p, edge_index = boundary.creases[0].p1_init, 0
  if not shaping_utils.check_point_on_grid(cur_p, grid_size):
    raise ValueError(f'Point {cur_p} is not on the grid with size {grid_size}.')
  for _ in range(boundary.num_walk_steps_upper_bound(grid_size)):
    next_p, edge_index = boundary.walk(cur_p, edge_index, grid_size)
    if not shaping_utils.check_point_on_grid(next_p, grid_size):
      raise ValueError(
          f'Point {next_p} is not on the grid with size {grid_size}.'
      )

    if cur_p.on_paper and next_p.on_paper:
      return cur_p, next_p
    else:
      cur_p = next_p

  raise ValueError(
      'No opposite hinge points found on boundary: {boundary} with grid size:'
      f' {grid_size}.'
  )


def get_opposite_hinge_points(
    boundary: shaping_elements.ShapingBoundary,
    grid_size: int,
    elevation_diff: int,
    creases: list[crease_utils.Crease],
) -> tuple[point_utils.Point, point_utils.Point]:
  """Returns two points on the boundary with opposite elevations."""
  if elevation_diff <= 0:
    raise ValueError('Elevation difference must be positive.')
  if elevation_diff == 1:
    # Use basic method that ignores elevation
    return _get_opposite_hinge_points_no_elevation(boundary, grid_size)

  def boundary_iter():
    """Walks around the boundary."""
    cur_p, edge_index = boundary.creases[0].p1_init, 0
    cur_elevation = geometric_fold_3d_heuristics_utils.check_point_elevation(
        cur_p, creases
    )
    for _ in range(boundary.num_walk_steps_upper_bound(grid_size)):
      yield cur_p, cur_elevation
      cur_p, edge_index, cur_elevation = (
          geometric_fold_3d_heuristics_utils.get_next_point_edge_index_and_elevation(
              cur_p, edge_index, creases, boundary, grid_size
          )
      )

  window = collections.deque()
  for p2, p2_el in boundary_iter():
    window.append((p2, p2_el))
    if len(window) <= elevation_diff:
      continue

    p1, p1_el = window.popleft()
    if p1_el is None or p2_el is None:
      continue
    if not p1.on_paper or not p2.on_paper:
      continue
    if abs(p1_el - p2_el) == elevation_diff:
      return p1, p2

  raise ValueError(f'No opposite hinge points found on boundary: {boundary}')


def _get_cut_line_from_anchor_and_direction(
    anchor: point_utils.Point,
    simple_fold_direction: np.ndarray,
    graph_data: geometric_simple_fold.GraphData,
    folded_verts: list[tuple[float, float] | None],
) -> tuple[tuple[float, float], tuple[float, float]]:
  """Returns the cut line from an anchor point and fold direction.

  The cut line starting point is determined by mapping the anchor point
  on the unfolded crease pattern to a point on the folded crease pattern.
  The cut line ending point is determined by adding the simple fold direction
  vector to the starting point.

  Args:
      anchor: The anchor point on the unfolded crease pattern lying on the
        boundary of the target shaping element.
      simple_fold_direction: A vector indicating the direction of the fold.
      graph_data: The graph data for the crease pattern.
      folded_verts: The folded vertices of the crease pattern.

  Returns:
      A tuple containing the start and end points of the cut line in folded
      space.
  """
  if any(v is None for v in folded_verts):
    raise ValueError(f'Unresolved folded vertices: {folded_verts}.')
  cut_line_start = geometric_utils.get_folded_pos(
      point=anchor,
      unfolded_coords=graph_data['vertices_coords'],
      folded_coords=folded_verts,  # pyrefly: ignore[bad-argument-type]
      edge_vertices=graph_data['edge_vertices'],
      face_vertices=graph_data['face_vertices'],
  )
  if cut_line_start is None:
    raise ValueError(f'Cut line start is None for anchor: {anchor}.')
  cut_line_end = (
      cut_line_start[0] + simple_fold_direction[0],
      cut_line_start[1] + simple_fold_direction[1],
  )
  return cut_line_start, cut_line_end


def _get_root_element_orientation(
    root_element: shaping_elements.ShapingElement,
    stick_figure: sf_utils.StickFigure,
    graph_data: geometric_simple_fold.GraphData,
    folded_verts: list[tuple[float, float] | None],
) -> np.ndarray:
  """Returns a 3d vector from the root element's start to end joint.

  This vector points from the root element's start joint to its end joint in 3D.
  It is computed by finding the direction of the boundary depth vector
  corresponding to the element's start joint.

  Args:
      root_element: The root element of the shaping element tree.
      stick_figure: The stick figure.
      graph_data: The graph data for the crease pattern.
      folded_verts: The folded vertices of the crease pattern.

  Returns:
      A unit vector in 3d pointing from a point on the root start joint with
      elevation 0 to a point on the root end joint with elevation 0.
  """
  start, end = shaping_elements.get_unfolded_root_vector(
      root_element, stick_figure
  )

  # Get folded positions of start and end points.
  start_folded = geometric_utils.get_folded_pos(
      point=start,
      unfolded_coords=graph_data['vertices_coords'],
      folded_coords=folded_verts,  # pyrefly: ignore[bad-argument-type]
      edge_vertices=graph_data['edge_vertices'],
      face_vertices=graph_data['face_vertices'],
  )
  end_folded = geometric_utils.get_folded_pos(
      point=end,
      unfolded_coords=graph_data['vertices_coords'],
      folded_coords=folded_verts,  # pyrefly: ignore[bad-argument-type]
      edge_vertices=graph_data['edge_vertices'],
      face_vertices=graph_data['face_vertices'],
  )

  if start_folded is None or end_folded is None:
    raise ValueError(
        f'Could not map root element {root_element.name!r} to folded space.'
    )
  return np.array([
      end_folded[0] - start_folded[0],
      end_folded[1] - start_folded[1],
      0.0,
  ])


_REFERENCE_FRACTIONS = ((0.3137, 0.4271), (0.6173, 0.2719), (0.2311, 0.7193))


def root_reference_point(
    root_element: shaping_elements.ShapingElement,
) -> tuple[float, float] | None:
  """Returns a deterministic interior paper point of `root_element`.

  Uses off-grid fractional offsets so the point avoids box-pleating creases and
  pins the posture frame normal (`+z`) to the top side of a consistent root
  face even after pre-shaping changes face ordering.

  Args:
    root_element: Root shaping flap or river.

  Returns:
    `(x, y)` inside `root_element`, or `None` if no candidate lies strictly
    inside.
  """
  if isinstance(root_element, shaping_elements.ShapingRiver):
    rectangles = list(root_element.covering_rectangles)
  else:
    rectangles = [root_element.to_legacy_rectangle()]
  for rectangle in rectangles:
    for fx, fy in _REFERENCE_FRACTIONS:
      x = rectangle.x_start + fx * rectangle.x_len
      y = rectangle.y_start + fy * rectangle.y_len
      point = point_utils.Point(x=x, y=y, require_on_paper=False)
      if root_element.is_internal_point(point, strict=True):
        return (x, y)
  return None


def _point_in_face(point: np.ndarray, polygon: np.ndarray) -> bool:
  """Returns whether 2D `point` lies inside `polygon` via even-odd ray cast."""
  inside = False
  num_verts = len(polygon)
  for i in range(num_verts):
    (x1, y1), (x2, y2) = polygon[i], polygon[(i + 1) % num_verts]
    if (y1 > point[1]) != (y2 > point[1]):
      if point[0] < x1 + (point[1] - y1) * (x2 - x1) / (y2 - y1):
        inside = not inside
  return inside


def _is_root_reference_flipped(
    root_element: shaping_elements.ShapingElement,
    graph_data: geometric_simple_fold.GraphData,
) -> bool:
  """Returns whether the face containing `root_reference_point` is flipped.

  `prepare_graph_data` orients its folded frame with face 0 face-up, whose
  identity can change when pre-shaping inserts creases. Anchoring the posture
  normal to `root_reference_point` keeps handedness and mountain/valley signs
  invariant to face reindexing.

  Args:
    root_element: Root shaping element.
    graph_data: Flat-folded graph data of the crease pattern.

  Returns:
    `True` if the root reference face is flipped relative to face 0.
  """
  reference = root_reference_point(root_element)
  if reference is None or not graph_data['vertices_coords']:
    return False
  vertices = np.asarray(graph_data['vertices_coords'], dtype=float)
  point = np.asarray(reference, dtype=float)
  if float(vertices.max()) > 1.5 >= float(point.max()):
    point = point * root_element.grid_size
  orientations = graph_data['face_orientations']
  for index, face in enumerate(graph_data['face_vertices']):
    if _point_in_face(point, vertices[face]):
      return bool(orientations[index])
  return False


def _rotate_simple_fold_direction_by_model_orientation(
    simple_fold_direction: np.ndarray,
    root_element: shaping_elements.ShapingElement,
    stick_figure: sf_utils.StickFigure,
    graph_data: geometric_simple_fold.GraphData,
    folded_verts: list[tuple[float, float] | None],
    *,
    root_flipped: bool | None = None,
) -> np.ndarray:
  """Maps a cut-line direction from the posture frame into folded 2D space.

  In the posture frame, the root stick points along `+y` and the root reference
  face normal defines `+z` (`+x = +y x +z`). When `root_flipped` is false and
  the folded root points upward (`uy < 0` in top-down `(x, y)` coordinates),
  mapping `+x` onto the right side of the folded root requires negating `x`
  before rotating `(0, 1)` onto `(ux, uy)` — matching the legacy 180-degree
  x-axis flip for `(0, -1)` while leaving `uy >= 0` (`(0, +1)` and horizontal
  roots) unnegated.

  Args:
    simple_fold_direction: Direction vector in the posture frame (`z` ignored).
    root_element: Root element of the shaping element tree.
    stick_figure: Stick figure defining root orientation.
    graph_data: Flat-folded graph data for the crease pattern.
    folded_verts: Flat-folded vertex coordinates.
    root_flipped: Precomputed `_is_root_reference_flipped` flag, or `None` to
      compute on demand.

  Returns:
    3D direction vector `(dx, dy, 0.0)` in the flat-folded frame.
  """
  root_orientation = _get_root_element_orientation(
      root_element, stick_figure, graph_data, folded_verts
  )
  logging.info('root_element_orientation: %s', root_orientation)
  if root_flipped is None:
    root_flipped = _is_root_reference_flipped(root_element, graph_data)
  px = float(simple_fold_direction[0])
  py = float(simple_fold_direction[1])
  rx, ry = float(root_orientation[0]), float(root_orientation[1])
  norm = math.hypot(rx, ry)
  if norm < 1e-12:
    if not root_flipped:
      px = -px
    return np.array([px, py, 0.0])
  ux, uy = rx / norm, ry / norm
  if not root_flipped and uy < 0:
    px = -px
  rotated = np.array([uy * px + ux * py, -ux * px + uy * py, 0.0])
  logging.info('new_simple_fold_direction: %s', rotated)
  return rotated


def _element_folded_direction(
    element: shaping_elements.ShapingFlap,
    graph_data: geometric_simple_fold.GraphData,
    folded_verts: list[tuple[float, float] | None],
) -> np.ndarray | None:
  """Returns the proximal-to-distal 2D direction of a flap on the folded base."""
  try:
    boundary = shaping_elements.get_shaping_boundary(
        element, boundary_index=None
    )
    tips = boundary.tips
    point, edge_index = boundary.creases[0].p1_init, 0
    for _ in range(boundary.num_walk_steps_upper_bound(element.grid_size)):
      point, edge_index = boundary.walk(point, edge_index, element.grid_size)
      if point.on_paper and all(not point.isclose(tip) for tip in tips):
        break
    if not point.on_paper:
      return None
    edge_index = boundary.get_edge_index(point)
    if edge_index is None:
      return None
    inner = boundary.depth_unit_vectors[edge_index].point_add(point)
    start_folded = geometric_utils.get_folded_pos(
        point=point,
        unfolded_coords=graph_data['vertices_coords'],
        folded_coords=folded_verts,
        edge_vertices=graph_data['edge_vertices'],
        face_vertices=graph_data['face_vertices'],
    )
    end_folded = geometric_utils.get_folded_pos(
        point=inner,
        unfolded_coords=graph_data['vertices_coords'],
        folded_coords=folded_verts,
        edge_vertices=graph_data['edge_vertices'],
        face_vertices=graph_data['face_vertices'],
    )
  except (AssertionError, ValueError, IndexError, KeyError) as error:
    logging.info('No folded direction for %s: %s', element.name, error)
    return None
  if start_folded is None or end_folded is None:
    return None
  return np.array(
      [end_folded[0] - start_folded[0], end_folded[1] - start_folded[1]],
      dtype=float,
  )


def _measured_base_signs(
    elements: Sequence[shaping_elements.ShapingElement],
    root_element: shaping_elements.ShapingElement,
    stick_figure: sf_utils.StickFigure,
    graph_data: geometric_simple_fold.GraphData,
) -> dict[str, float]:
  """Measures `+1` / `-1` alignment of non-root flaps with the folded root."""
  folded_verts = graph_data['folded_vertices']
  try:
    root_dir = _get_root_element_orientation(
        root_element, stick_figure, graph_data, folded_verts
    )[:2]
  except (AssertionError, ValueError, IndexError, KeyError) as error:
    logging.warning(
        'No measured base directions (root orientation: %s).', error
    )
    return {}
  root_norm = float(np.linalg.norm(root_dir))
  if root_norm < 1e-9:
    return {}
  root_unit = root_dir / root_norm

  signs: dict[str, float] = {}
  for element in elements:
    # Only measure branching flaps: river bank indices do not distinguish the
    # proximal bank, and interior degree-<=2 chain sub-flaps inherit their
    # parent flap's direction directly.
    if element.name == root_element.name or not isinstance(
        element, shaping_elements.ShapingFlap
    ):
      continue
    if element.name in stick_figure:
      stick = stick_figure[element.name]
      if all(
          stick_figure.joints[j].degree <= 2 for j in (stick.start, stick.end)
      ):
        continue
    direction = _element_folded_direction(element, graph_data, folded_verts)
    if direction is None:
      continue
    dir_norm = float(np.linalg.norm(direction))
    if dir_norm < 1e-9:
      continue
    cosine = float(np.dot(direction, root_unit)) / dir_norm
    if abs(cosine) >= 0.7:
      signs[element.name] = math.copysign(1.0, cosine)
  return signs


def _get_shaping_creases(
    base_cp: cp_utils.CreasePattern,
    anchor: point_utils.Point,
    simple_fold_direction: np.ndarray,
    fold_assignment: crease_utils.Fold,
    limiting_polygon: (
        shaping_elements.ShapingFlap | shaping_elements.ShapingRiver
    ),
    graph_data: geometric_simple_fold.GraphData,
    folded_verts: list[tuple[float, float] | None],
    rotation_angle: float,
    visualize: bool = False,
    extra_limiting_rectangles: Sequence[shaping_utils.Rectangle] = (),
) -> Sequence[crease_utils.Crease]:
  """Returns creases from a geometric simple fold.

  The creases are generated for a given anchor point and fold direction.

  Args:
      base_cp: The base crease pattern.
      anchor: The anchor point for the simple fold.
      simple_fold_direction: The simple fold direction for the simple fold.
      fold_assignment: The fold assignment for the simple fold.
      limiting_polygon: The limiting polygon for the simple fold.
      graph_data: The graph data for the base crease pattern.
      folded_verts: The folded vertices for the base crease pattern.
      rotation_angle: The rotation angle for the simple fold.
      visualize: Whether to visualize the cut line on the folded crease pattern.
      extra_limiting_rectangles: Other rectangles that the creases can propagate
        to.

  Returns:
      The shaping creases for the element.
  """
  logging.info('anchor: %s', anchor)
  logging.info('simple_fold_direction: %s', simple_fold_direction)

  cut_line_start, cut_line_end = _get_cut_line_from_anchor_and_direction(
      anchor=anchor,
      simple_fold_direction=simple_fold_direction,
      graph_data=graph_data,
      folded_verts=folded_verts,
  )

  if (
      geometric_fold_3d_heuristics_utils.tuple_dist(
          cut_line_start, cut_line_end
      )
      < crease_utils.DEFAULT_TOLERANCE
  ):
    logging.warning(
        'Cut line start and end points are too close for element: %s. '
        'Cut line start: %s, cut line end: %s. '
        'Skipping.',
        limiting_polygon.name,
        cut_line_start,
        cut_line_end,
    )
    return []

  if visualize:
    import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

    fig = geometric_fold_3d_heuristics_utils.visualize_cut_line_on_folded_cp(
        base_cp,
        cut_line_start,
        cut_line_end,
        limiting_polygon,
    )
    plt.close(fig)

  if not extra_limiting_rectangles:
    limiting_rectangles = limiting_polygon
  else:
    if isinstance(limiting_polygon, shaping_elements.ShapingFlap):
      limiting_rectangles = [limiting_polygon.to_legacy_rectangle()]
    else:
      limiting_rectangles = list(limiting_polygon.covering_legacy_rectangles)
    limiting_rectangles.extend(extra_limiting_rectangles)

  shaping_creases = geometric_simple_fold.get_geometric_fold_creases(
      crease_pattern=base_cp,
      cut_line_start=cut_line_start,
      cut_line_end=cut_line_end,
      fold_assignment=fold_assignment,
      limiting_polygon=limiting_rectangles,
      graph_data=graph_data,
  )

  # Adjust fold percent to match the rotation angle.
  fold_percent = abs(rotation_angle / 180)
  shaping_creases = [
      dataclasses.replace(crease, fold_percent=fold_percent)
      for crease in shaping_creases
  ]

  return shaping_creases


def _clip_by_min_inscribe_depth(inscribe_depth: float, grid_size: int) -> float:
  """Returns the minimum inscribe depth for a given grid size."""
  return max(inscribe_depth, point_utils.DEFAULT_TOLERANCE * grid_size + 1e-5)


def _get_shaping_creases_with_heuristics(
    *,
    base_cp: cp_utils.CreasePattern,
    anchor: point_utils.Point,
    simple_fold_direction: np.ndarray,
    fold_assignment: crease_utils.Fold,
    limiting_polygon: (
        shaping_elements.ShapingFlap | shaping_elements.ShapingRiver
    ),
    graph_data: geometric_simple_fold.GraphData,
    folded_verts: list[tuple[float, float] | None],
    hinge_info_list: Sequence[packing_types.HingeInfo],
    rotation_angle: float,
    root_element: shaping_elements.ShapingElement,
    stick_figure: sf_utils.StickFigure,
    visualize: bool = False,
    inscribe_hparams: InscribeHParams = InscribeHParams(),
    extra_limiting_rectangles: Sequence[shaping_utils.Rectangle] = (),
    root_flipped: bool | None = None,
) -> tuple[
    Sequence[crease_utils.Crease], shaping_elements.ShapingElement, bool
]:
  """Returns the shaping creases for a given anchor point and fold direction.

  Some heuristics are included to improve visual quality and reduce
  flat-foldability violations.

  Args:
      base_cp: The base crease pattern.
      anchor: The anchor point for the simple fold.
      simple_fold_direction: The simple fold direction for the simple fold.
      fold_assignment: The fold assignment for the simple fold.
      limiting_polygon: The limiting polygon for the simple fold.
      graph_data: The graph data for the base crease pattern.
      folded_verts: The folded vertices for the base crease pattern.
      hinge_info_list: The list of hinge info.
      rotation_angle: The rotation angle for the simple fold.
      root_element: The root element of the shaping element tree.
      stick_figure: The stick figure.
      visualize: Whether to visualize the cut line on the folded crease pattern.
      inscribe_hparams: Parameters controlling cut line inscription depth.
      extra_limiting_rectangles: Other rectangles to propagate creases to.
      root_flipped: Whether the root element face orientation is flipped.

  Returns:
      A tuple containing:
        - The shaping creases for the element.
        - The shaping element used (after shrinking/inscribing).
  """
  # Filter out vertical simple fold directions.
  simple_fold_slope = _get_rotation_axis_slope(simple_fold_direction)
  if simple_fold_slope is None:
    logging.warning(
        'Simple fold direction is vertical for element: %s. Skipping.',
        limiting_polygon.name,
    )
    return [], limiting_polygon, False

  # Filter out nearly vertical simple fold directions.
  if (
      abs(_get_rotation_axis_slope(simple_fold_direction))  # pyrefly: ignore[bad-argument-type]
      > inscribe_hparams.max_slope_thresh
  ):
    logging.warning(
        'Slope of simple fold direction is too large for element: %s. '
        'Slope: %s. Skipping.',
        limiting_polygon.name,
        _get_rotation_axis_slope(simple_fold_direction),
    )
    return [], limiting_polygon, False
  # Otherwise, clip slope if still too large.
  elif (
      abs(_get_rotation_axis_slope(simple_fold_direction))  # pyrefly: ignore[bad-argument-type]
      > inscribe_hparams.clip_slope_thresh
  ):
    logging.warning(
        'Slope of simple fold direction is too large for element: %s. '
        'Slope: %s. Clipping.',
        limiting_polygon.name,
        _get_rotation_axis_slope(simple_fold_direction),
    )
    simple_fold_direction = _clip_slope_to_max(
        simple_fold_direction, inscribe_hparams.clip_slope_thresh
    )

  # Filter out creases with rotation angle 0.
  if math.isclose(rotation_angle, 0.0):
    logging.warning(
        'Rotation angle is close to 0 for element: %s. Skipping.',
        limiting_polygon.name,
    )
    return [], limiting_polygon, False

  # Clip angles near 90o to 90o
  if (
      abs(abs(simple_fold_slope) - 1.0)
      < inscribe_hparams.ninety_deg_slope_tolerance
  ):
    logging.info(
        'Clipping simple fold direction to slope 1 for element: %s',
        limiting_polygon.name,
    )
    simple_fold_direction = _clip_to_slope_one(simple_fold_direction)

  # Rotate simple_fold_direction to match how the flap is folded in 3D.
  simple_fold_direction = _rotate_simple_fold_direction_by_model_orientation(
      simple_fold_direction,
      root_element,
      stick_figure,
      graph_data,
      folded_verts,
      root_flipped=root_flipped,
  )

  bank_index = (
      limiting_polygon.get_outer_bank_index()
      if isinstance(limiting_polygon, shaping_elements.ShapingRiver)
      else None
  )

  # Increase inscribe depth if element has hinge.
  # NOTE: we inscribe with a hinge even if inscribe_depth > 0.0
  # because a close by fold is assumed to already be inscribed
  # due to the hinge.
  if inscribe_hparams.avoid_hinges and shaping_elements.has_assigned_hinge(
      limiting_polygon, hinge_info_list, bank_index=bank_index
  ):
    inscribe_depth = (
        inscribe_hparams.inscribe_depth + inscribe_hparams.hinge_inscribe_depth
    )
  else:
    inscribe_depth = inscribe_hparams.inscribe_depth

  # Inscribe cut line if inscribe depth is specified.
  inscribe_success = True
  if inscribe_depth > 0.0:
    # Get min inscribe depth depending on grid size and point tolerance.
    # Otherwise the inscribed point will be snapped back to the grid.
    inscribe_depth = _clip_by_min_inscribe_depth(
        inscribe_depth, grid_size=base_cp.grid_size
    )
    logging.info(
        'Inscribing cut line for element: %s with depth: %s',
        limiting_polygon.name,
        inscribe_depth,
    )
    anchor, limiting_polygon, inscribe_success = (
        geometric_fold_3d_heuristics_utils.inscribe_cut_line(
            anchor, limiting_polygon, inscribe_depth=inscribe_depth
        )
    )
    logging.info(
        'Inscribed cut line for element: %s. New anchor: %s, new limiting'
        ' polygon: %s',
        limiting_polygon.name,
        anchor,
        limiting_polygon,
    )

  creases = _get_shaping_creases(
      base_cp=base_cp,
      anchor=anchor,
      simple_fold_direction=simple_fold_direction,
      fold_assignment=fold_assignment,
      limiting_polygon=limiting_polygon,
      graph_data=graph_data,
      folded_verts=folded_verts,
      visualize=visualize,
      rotation_angle=rotation_angle,
      extra_limiting_rectangles=extra_limiting_rectangles,
  )
  return creases, limiting_polygon, inscribe_success


def shape_element_with_3d_geometric_folding_heuristics(
    *,
    element: shaping_elements.ShapingElement,
    base_cp: cp_utils.CreasePattern,
    simple_fold_info: heuristics_3d.SimpleFoldInfo,
    hinge_info_list: Sequence[packing_types.HingeInfo],
    root_element: shaping_elements.ShapingElement,
    stick_figure: sf_utils.StickFigure,
    inscribe_hparams: InscribeHParams = InscribeHParams(),
    visualize: bool = False,
    graph_data: geometric_simple_fold.GraphData | None = None,
    extra_limiting_rectangles: Sequence[shaping_utils.Rectangle] = (),
    root_flipped: bool | None = None,
) -> ElementShapingResult:
  """Shapes a single element with 3D geometric folding heuristics.

  The heuristic proceeds in several steps:
  1. The folded state of the base crease pattern is computed.
  2. The boundary of the shaping element is determined.
  3. Two points on the boundary with opposite elevations are found.
  4. Apply a two geometric simple folds with each point as the anchor.
    Only one of the folds will typically succeed (i.e. produce creases
    that intersect the target element).

  Args:
      element: The shaping element to shape.
      base_cp: The base crease pattern.
      simple_fold_info: The simple fold info for the element. Computed by
        heuristics_3d.get_flap_post_dict().
      hinge_info_list: The list of hinge info.
      root_element: The root element of the shaping element tree.
      stick_figure: The stick figure.
      inscribe_hparams: Parameters controlling cut line inscription depth.
      visualize: Whether to visualize the cut line on the folded crease pattern.
      graph_data: Precomputed graph data for the crease pattern. If None, it
        will be computed from base_cp.
      extra_limiting_rectangles: Other rectangles to propagate creases to.
      root_flipped: Optional override indicating whether the root face is
        flipped.

  Returns:
      A list of ElementShapingResult objects, one for each successful fold
      attempt.
  """
  # Skip elements with no fold.
  if math.isclose(simple_fold_info.rotation_angle, 0.0):
    logging.info(
        'No fold for element: %s. Rotation angle: %s. Skipping.',
        element.name,
        simple_fold_info.rotation_angle,
    )
    return ElementShapingResult(
        simple_fold_info=simple_fold_info,
        shaping_element=element,
        shaping_creases=[],
    )

  # 1. Compute Folded State
  # NOTE: optimize runtime by caching graph data.
  if graph_data is None:
    graph_data = geometric_simple_fold.prepare_graph_data(base_cp)
  folded_verts = graph_data['folded_vertices']

  # 2. Get Shaping Boundary
  match element:
    case shaping_elements.ShapingFlap():
      shaping_boundary = shaping_elements.get_flap_boundary(
          element, element.grid_size
      )
    case shaping_elements.ShapingRiver():
      shaping_boundary = element.get_boundary_from_index(
          element.get_outer_bank_index()
      )
    case _ as unreachable:
      typing.assert_never(unreachable)

  # Compute points on the boundary of the flap on opposite elevations.
  elevations = element.elevations
  if len(elevations) < 2:
    elevation_diff = 1
  else:
    elevation_diff = max(elevations) - min(elevations)
  p1, p2 = get_opposite_hinge_points(
      shaping_boundary, element.grid_size, elevation_diff, element.creases
  )

  # 3. Try geometric folding with both hinge points as the anchor.
  if root_flipped is None:
    root_flipped = _is_root_reference_flipped(root_element, graph_data)
  toward_normal = (simple_fold_info.rotation_angle > 0) != root_flipped
  fold_assignment = (
      crease_utils.Fold.VALLEY if toward_normal else crease_utils.Fold.MOUNTAIN
  )

  simple_fold_direction = simple_fold_info.rotation_axis_local
  logging.info('simple_fold_direction: %s', simple_fold_direction)
  logging.info('slope: %s', _get_rotation_axis_slope(simple_fold_direction))

  # If close to 180o, round to 180o.
  if abs(simple_fold_direction[1]) < 0.1:
    simple_fold_direction = np.array([1.0, 0.0, 0.0])

  shaping_creases: Sequence[crease_utils.Crease] = []
  shaped_element = element
  inscribe_success = False
  for anchor in (p1, p2):
    shaping_creases, shaped_element, inscribe_success = (
        _get_shaping_creases_with_heuristics(
            base_cp=base_cp,
            anchor=anchor,
            simple_fold_direction=simple_fold_direction,
            fold_assignment=fold_assignment,
            limiting_polygon=element,
            hinge_info_list=hinge_info_list,
            graph_data=graph_data,
            folded_verts=folded_verts,
            root_element=root_element,
            stick_figure=stick_figure,
            visualize=visualize,
            inscribe_hparams=inscribe_hparams,
            rotation_angle=simple_fold_info.rotation_angle,
            extra_limiting_rectangles=extra_limiting_rectangles,
            root_flipped=root_flipped,
        )
    )
    if shaping_creases:
      break

  if not inscribe_success:
    return ElementShapingResult(
        simple_fold_info=simple_fold_info,
        shaping_element=element,
        shaping_creases=[],
    )
  return ElementShapingResult(
      simple_fold_info=simple_fold_info,
      shaping_element=shaped_element,
      shaping_creases=shaping_creases,
  )


def get_shaping_creases_with_3d_geometric_folding_heuristics(
    *,
    base_cp: cp_utils.CreasePattern,
    stick_figure: sf_utils.StickFigure,
    shaping_flaps: Sequence[shaping_elements.ShapingFlap],
    shaping_rivers: Sequence[shaping_elements.ShapingRiver],
    hinge_info_list: Sequence[packing_types.HingeInfo],
    central_crease_removed: bool,
    shape_with_unmerged_tree: bool,
    merge_converter: dict[str, str] | None = None,
    visualize: bool = False,
    hparams: GeometricFolding3DHParams = GeometricFolding3DHParams(),
    graph_data: geometric_simple_fold.GraphData | None = None,
    shaping_data: heuristics_utils.ShapingData | None = None,
    sacrificial_flaps: Sequence[packing_types.SacrificialFlap] = (),
) -> Heuristics3DShapingResult:
  """Returns shaping creases using 3D geometric folding heuristics.

  Aligns the stick figure root to zero azimuth and elevation, resolves flap fold
  states (optionally cross-checked against measured folded-base directions),
  computes local 3D simple-fold rotations with fixed-point propagation for
  unrotated parents, and maps each cut line back to the unfolded paper.

  Args:
    base_cp: Base crease pattern to shape.
    stick_figure: Target 3D stick figure.
    shaping_flaps: Candidate shaping flaps.
    shaping_rivers: Candidate shaping rivers.
    hinge_info_list: Hinge placement and assignment info.
    central_crease_removed: Whether the central symmetry crease was removed.
    shape_with_unmerged_tree: Whether to shape sub-elements of merged chains.
    merge_converter: Mapping from unmerged stick names to merged stick names.
    visualize: Whether to plot candidate cut lines on the folded state.
    hparams: Hyperparameters controlling heuristic behavior.
    graph_data: Optional precomputed flat-folded graph data for `base_cp`.
    shaping_data: Optional precomputed `ShapingData`.
    sacrificial_flaps: Sacrificial flaps sharing creases with active flaps.

  Returns:
    A `Heuristics3DShapingResult` with accepted shaping creases and per-element
    diagnostic entries.

  Raises:
    ValueError: If the flat-folded graph of `base_cp` cannot be constructed.
  """
  logging.info(
      'Getting shaping creases for %s with 3D geometric folding heuristics.',
      stick_figure.name,
  )

  is_input_symmetric = stick_figure.is_symmetric_tree
  symmetric_partners: dict[str, str] = {}
  if is_input_symmetric:
    for s in stick_figure.children:
      if (
          s.symmetric_stick_name is not None
          and s.symmetric_stick_name != s.name
      ):
        symmetric_partners[s.name] = s.symmetric_stick_name
    if not shape_with_unmerged_tree:
      merged_sf, _ = stick_figure.merge_linear_chains()
      if merged_sf.is_symmetric_tree:
        for s in merged_sf.children:
          if (
              s.symmetric_stick_name is not None
              and s.symmetric_stick_name != s.name
          ):
            symmetric_partners[s.name] = s.symmetric_stick_name

  root = heuristics.get_first_stick_for_shaping(
      stick_figure, use_tree_root=hparams.use_tree_root
  )
  stick_figure = dataclasses.replace(stick_figure, root=root.name)
  stick_figure = stick_figure.align_root_stick_to_target_azimuth_elevation()

  initial_flaps_by_name = {flap.name: flap for flap in shaping_flaps}
  if shaping_data is None:
    shaping_data = heuristics_utils.get_heuristics_shaping_data(
        stick_figure=stick_figure,
        shaping_flaps=list(shaping_flaps),
        shaping_rivers=list(shaping_rivers),
        hinge_info_list=list(hinge_info_list),
        central_crease_removed=central_crease_removed,
        shape_with_unmerged_tree=shape_with_unmerged_tree,
        merge_converter=merge_converter,
        sacrificial_flaps=sacrificial_flaps,
        use_tree_root=hparams.use_tree_root,
    )
  shaping_flaps = shaping_data['shaping_flaps']
  shaping_rivers = shaping_data['shaping_rivers']
  flap_info_dict = shaping_data['flap_info']
  outer_bank_indices = shaping_data['outer_bank_indices']

  shaping_rivers = [
      shaping_river.update_outer_bank_index(
          cast(shaping_types.BankIndex, outer_bank_indices[shaping_river.name])
      )
      for shaping_river in shaping_rivers
  ]

  flap_strategies = {flap.name: hparams.strategy for flap in shaping_flaps}
  flap_strategies.update(hparams.per_flap_strategy)

  elements = list(shaping_flaps) + list(shaping_rivers)
  root_element = shaping_elements.get_element_by_name(
      stick_figure.root, shaping_flaps, shaping_rivers
  )
  if hparams.shape_subset is not None:
    elements = [
        element for element in elements if element.name in hparams.shape_subset
    ]
  if hparams.priority:
    priority_rank = {name: i for i, name in enumerate(hparams.priority)}
    elements.sort(key=lambda e: priority_rank.get(e.name, len(priority_rank)))
  shaping_flaps_by_name = {flap.name: flap for flap in shaping_flaps}
  elements_by_name = {element.name: element for element in elements}
  if is_input_symmetric:
    for elem_name in elements_by_name:
      if elem_name.endswith(packing_types.COPY_SUFFIX):
        base_name = elem_name[: -len(packing_types.COPY_SUFFIX)]
        if (
            base_name in elements_by_name
            and base_name not in symmetric_partners
            and elem_name not in symmetric_partners
        ):
          symmetric_partners[base_name] = elem_name
          symmetric_partners[elem_name] = base_name
    symmetric_partners = {
        k: v
        for k, v in symmetric_partners.items()
        if k in elements_by_name and v in elements_by_name
    }

  if graph_data is None:
    graph_data = geometric_simple_fold.prepare_graph_data(base_cp)
  if not graph_data['vertices_coords']:
    raise ValueError(
        'The flat-folded graph of the pattern could not be built (sweep-line'
        ' graph creation failed); the 3D heuristics cannot place cut lines.'
    )

  root_flipped = _is_root_reference_flipped(root_element, graph_data)
  base_signs = None
  if hparams.measure_base_directions:
    base_signs = _measured_base_signs(
        list(shaping_flaps) + list(shaping_rivers),
        root_element,
        stick_figure,
        hparams.direction_graph_data or graph_data,
    )

  def _diagnose_empty_info(info: heuristics_3d.SimpleFoldInfo) -> str:
    direction = info.rotation_axis_local
    if abs(direction[1]) >= 0.1:
      slope = _get_rotation_axis_slope(direction)
      if slope is None:
        return 'vertical_axis'
      if abs(slope) > hparams.default_inscribe_hparams.max_slope_thresh:
        return 'steep_slope'
    return 'no_creases'

  def _shape_elements_once(
      simple_fold_info_dict: dict[str, Sequence[heuristics_3d.SimpleFoldInfo]],
  ) -> tuple[
      list[crease_utils.Crease],
      dict[str, list[ElementShapingResult]],
      list[tuple[object, ...]],
      dict[str, str],
  ]:
    accepted_creases_by_element: dict[str, list[crease_utils.Crease]] = {}
    discarded_partners: dict[str, str] = {}
    detailed_shaping_info: dict[str, list[ElementShapingResult]] = {}
    pending_warnings: list[tuple[object, ...]] = []
    skip_reasons: dict[str, str] = {}
    current_cp = base_cp

    def _rollback_partner(partner_name: str, failed_name: str) -> None:
      nonlocal current_cp, accepted_creases_by_element
      if partner_name in accepted_creases_by_element:
        accepted_creases_by_element.pop(partner_name, None)
        detailed_shaping_info.pop(partner_name, None)
        skip_reasons[partner_name] = 'flat_foldability'
        pending_warnings.append((
            (
                'Flat foldability violation detected after shaping element: %s.'
                ' Discarding creases for this element (symmetric partner %s'
                ' failed).'
            ),
            partner_name,
            failed_name,
        ))
        rebuilt_cp = base_cp
        if hparams.check_ff_violations:
          surviving: dict[str, list[crease_utils.Crease]] = {}
          for kept_name, kept_creases in accepted_creases_by_element.items():
            if not kept_creases:
              surviving[kept_name] = kept_creases
              continue
            cand_cp = rebuilt_cp.update_with_shaping_creases(kept_creases)
            if cand_cp.local_flat_foldability():
              rebuilt_cp = cand_cp
              surviving[kept_name] = kept_creases
            else:
              detailed_shaping_info.pop(kept_name, None)
              skip_reasons[kept_name] = 'flat_foldability'
          accepted_creases_by_element = surviving
        current_cp = rebuilt_cp
      else:
        discarded_partners[partner_name] = failed_name

    for element in elements:
      if element.name in discarded_partners:
        skip_reasons[element.name] = 'flat_foldability'
        pending_warnings.append((
            (
                'Flat foldability violation detected after shaping element: %s.'
                ' Discarding creases for this element (symmetric partner %s'
                ' failed).'
            ),
            element.name,
            discarded_partners[element.name],
        ))
        continue
      if element.name not in simple_fold_info_dict:
        skip_reasons[element.name] = 'no_fold_info'
        pending_warnings.append((
            'No simple fold info found for element: %s. Skipping.',
            element.name,
        ))
        continue
      simple_fold_infos = simple_fold_info_dict[element.name]
      if not simple_fold_infos or all(
          math.isclose(info.rotation_angle, 0.0, abs_tol=1e-3)
          for info in simple_fold_infos
      ):
        skip_reasons[element.name] = 'zero_angle'
        continue
      element_creases: list[crease_utils.Crease] = []
      element_had_assertion_failure = False
      empty_reason = 'no_creases'
      extra_limiting_rectangles: list[shaping_utils.Rectangle] = []
      for sac_flap in flap_info_dict[element.name].sacrificial_flaps:
        sac_elem = shaping_flaps_by_name.get(
            sac_flap
        ) or initial_flaps_by_name.get(sac_flap)
        if sac_elem is not None:
          extra_limiting_rectangles.append(sac_elem.to_legacy_rectangle())
      hinge_inscribe_depth = hparams.per_flap_hinge_inscribe_depth.get(
          element.name,
          hparams.default_inscribe_hparams.hinge_inscribe_depth,
      )
      for simple_fold_info in simple_fold_infos:
        if math.isclose(simple_fold_info.rotation_angle, 0.0, abs_tol=1e-3):
          continue
        try:
          element_output = shape_element_with_3d_geometric_folding_heuristics(
              element=element,
              base_cp=base_cp,
              simple_fold_info=simple_fold_info,
              hinge_info_list=hinge_info_list,
              root_element=root_element,
              stick_figure=stick_figure,
              visualize=visualize,
              inscribe_hparams=dataclasses.replace(
                  hparams.default_inscribe_hparams,
                  inscribe_depth=simple_fold_info.inscribe_depth,
                  hinge_inscribe_depth=hinge_inscribe_depth,
              ),
              graph_data=graph_data,
              extra_limiting_rectangles=extra_limiting_rectangles,
              root_flipped=root_flipped,
          )
          if not element_output['shaping_creases']:
            empty_reason = _diagnose_empty_info(simple_fold_info)
          element_creases.extend(element_output['shaping_creases'])
          if element.name not in detailed_shaping_info:
            detailed_shaping_info[element.name] = []
          detailed_shaping_info[element.name].append(element_output)
        except AssertionError as e:
          element_had_assertion_failure = True
          skip_reasons[element.name] = 'geometry_failed'
          pending_warnings.append((
              (
                  'Failed to shape element: %s with 3D geometric folding'
                  ' heuristics. Simple fold info: %s. Skipping. Error: %s'
              ),
              element.name,
              simple_fold_info.rotation_axis_local,
              e,
          ))
      if element_had_assertion_failure and element.name in symmetric_partners:
        detailed_shaping_info.pop(element.name, None)
        pending_warnings.append((
            (
                'Flat foldability violation detected after shaping element: %s.'
                ' Discarding creases for this element (symmetric failure).'
            ),
            element.name,
        ))
        _rollback_partner(symmetric_partners[element.name], element.name)
        continue
      if not element_creases:
        skip_reasons.setdefault(element.name, empty_reason)
      if hparams.check_ff_violations and element_creases:
        updated_cp = current_cp.update_with_shaping_creases(element_creases)
        if not updated_cp.local_flat_foldability():
          pending_warnings.append((
              (
                  'Flat foldability violation detected after shaping element:'
                  ' %s. Discarding creases for this element.'
              ),
              element.name,
          ))
          detailed_shaping_info.pop(element.name, None)
          skip_reasons[element.name] = 'flat_foldability'
          if element.name in symmetric_partners:
            _rollback_partner(symmetric_partners[element.name], element.name)
          continue
        current_cp = updated_cp
      accepted_creases_by_element[element.name] = element_creases

    shaping_creases: list[crease_utils.Crease] = []
    for creases in accepted_creases_by_element.values():
      shaping_creases.extend(creases)
    return (
        shaping_creases,
        detailed_shaping_info,
        pending_warnings,
        skip_reasons,
    )

  unrotated: set[str] = set()
  if hparams.shape_subset is not None:
    unrotated = {stick.name for stick in stick_figure.children} - set(
        hparams.shape_subset
    )
  shaping_creases: list[crease_utils.Crease] = []
  detailed_shaping_info: dict[str, list[ElementShapingResult]] = {}
  warnings_to_log: list[tuple[object, ...]] = []
  skip_reasons: dict[str, str] = {}
  for _ in range(_MAX_UNROTATED_PASSES):
    simple_fold_info_dict = heuristics_3d.get_flap_post_dict(
        stick_figure,
        flap_info_dict,
        make_post_hinge_top=True,
        default_strategy=hparams.strategy,
        flap_strategies=flap_strategies,
        max_slope_thresh=hparams.default_inscribe_hparams.max_slope_thresh,
        unrotated=unrotated,
        base_signs=base_signs,
        use_tree_root=hparams.use_tree_root,
    )
    shaping_creases, detailed_shaping_info, warnings_to_log, skip_reasons = (
        _shape_elements_once(simple_fold_info_dict)
    )
    shaped = {
        name
        for name, infos in detailed_shaping_info.items()
        if any(info['shaping_creases'] for info in infos)
    }
    left_out = {
        element.name
        for element in elements
        if element.name not in shaped
        and element.name in flap_info_dict
        and any(
            not math.isclose(info.rotation_angle, 0.0, abs_tol=1e-3)
            for info in simple_fold_info_dict.get(element.name, ())
        )
    }
    newly_unrotated = {
        name
        for name in left_out - unrotated
        if shaped.intersection(
            heuristics.get_descendants_names(flap_info_dict, name)
        )
    }
    if not newly_unrotated:
      break
    unrotated |= newly_unrotated

  for warning_args in warnings_to_log:
    logging.warning(*warning_args)

  return Heuristics3DShapingResult(
      shaping_creases=shaping_creases,
      detailed_shaping_info=detailed_shaping_info,
      skip_reasons=skip_reasons,
  )


def shape_cp_with_3d_geometric_folding_heuristics_pipeline_wrapper(
    *,
    base_cp: cp_utils.CreasePattern,
    stick_figure: sf_utils.StickFigure,
    packing_output: packing_output_utils.PackingOutput,
    solving_output: hinge_utils.SolverResult,
    solving_grid_size: int,
    central_crease_removed: bool,
    shape_with_unmerged_tree: bool = True,
    merge_converter: dict[str, str] | None = None,
    visualize: bool = False,
    hparams: GeometricFolding3DHParams = GeometricFolding3DHParams(),
    light_mode: bool = False,
    shaping_flaps: Sequence[shaping_elements.ShapingFlap] | None = None,
    shaping_rivers: Sequence[shaping_elements.ShapingRiver] | None = None,
    graph_data: geometric_simple_fold.GraphData | None = None,
    shaping_data: heuristics_utils.ShapingData | None = None,
) -> cp_utils.CreasePattern:
  """Shapes a crease pattern using 3D geometric folding heuristics."""
  if shaping_flaps is None or shaping_rivers is None:
    packing_cp = packing_output.packing_cp
    if packing_cp is None:
      raise ValueError('packing_output.packing_cp must not be None.')
    shaping_flaps, shaping_rivers = (
        shaping_elements.convert_packing_elements_to_shaping_elements(
            packing_rectangles=packing_output.flap_rectangles,  # pyrefly: ignore[bad-argument-type]
            packing_rivers=packing_output.packing_rivers,  # pyrefly: ignore[bad-argument-type]
            packing_grid_size=packing_cp.grid_size,
            solving_grid_size=solving_grid_size,
            base_cp=base_cp,
        )
    )

  hinge_info_list = heuristics.copy_hinge_info_list(packing_output.hinges)
  hinge_info_list = heuristics.update_hinge_info_from_solver_result(
      hinge_info_list, solving_output  # pyrefly: ignore[bad-argument-type]
  )

  shaping_result = get_shaping_creases_with_3d_geometric_folding_heuristics(
      base_cp=base_cp,
      stick_figure=stick_figure,
      shaping_flaps=shaping_flaps,
      shaping_rivers=shaping_rivers,
      hinge_info_list=hinge_info_list,
      central_crease_removed=central_crease_removed,
      shape_with_unmerged_tree=shape_with_unmerged_tree,
      merge_converter=merge_converter,
      visualize=visualize,
      hparams=hparams,
      graph_data=graph_data,
      shaping_data=shaping_data,
      sacrificial_flaps=packing_output.sacrificial_flaps or [],
  )

  return base_cp.update_with_shaping_creases(
      shaping_result['shaping_creases'],
      split_shaping_by_shaping=not light_mode,
  )
