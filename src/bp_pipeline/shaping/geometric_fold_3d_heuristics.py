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
from typing import Any, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import point as point_utils
from ..packing import packing_output as packing_output_utils
from ..packing import types as packing_types
from ..solving import hinges as hinge_utils
from ..stick_figures import angle_utils
from ..stick_figures import stick_figure as sf_utils
from . import geometric_fold_3d_heuristics_utils
from . import geometric_simple_fold
from . import geometric_utils
from . import heuristics
from . import heuristics_3d
from . import heuristics_utils
from . import shaping_elements
from . import shaping_utils
from . import simple_fold_shaping
from . import types as shaping_types
from . import zigzag_shaping

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
    measure_base_directions: Start every element from its direction measured on
      the folded base (along or against the root) instead of the hinge-inferred
      fold state (see `_measured_base_signs`).
    direction_graph_data: Graph data of the unshaped base, used for
      `measure_base_directions` (default: the graph of `base_cp`). A flat
      pre-shaping (wing fans) moves the flap boundaries the measurement starts
      from: on Pegasus the fanned wing_mid_L then read as pointing the other way
        and was rotated to (155, -41) for (90, 36).
    priority: Elements to shape first. Elements are shaped one by one and an
      element whose creases break flat-foldability together with the earlier
      ones is left out, so the order decides which of two conflicting elements
      (often a mirrored pair) survives.
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
  direction_graph_data: Any = None


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
  assert shaping_utils.check_point_on_grid(
      cur_p, grid_size
  ), f'Point {cur_p} is not on the grid with size {grid_size}.'
  for _ in range(boundary.num_walk_steps_upper_bound(grid_size)):
    next_p, edge_index = boundary.walk(cur_p, edge_index, grid_size)
    assert shaping_utils.check_point_on_grid(
        next_p, grid_size
    ), f'Point {next_p} is not on the grid with size {grid_size}.'

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
  assert all(v is not None for v in folded_verts), folded_verts
  cut_line_start = geometric_utils.get_folded_pos(
      point=anchor,
      unfolded_coords=graph_data['vertices_coords'],
      folded_coords=folded_verts,  # pyrefly: ignore[bad-argument-type]
      edge_vertices=graph_data['edge_vertices'],
      face_vertices=graph_data['face_vertices'],
  )
  assert (
      cut_line_start is not None
  ), f'Cut line start is None for anchor: {anchor}.'
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

  return np.array([
      end_folded[0] - start_folded[0],
      end_folded[1] - start_folded[1],
      0.0,
  ])


_REFERENCE_FRACTIONS = ((0.3137, 0.4271), (0.6173, 0.2719), (0.2311, 0.7193))

_FLIP_CACHE: dict[int, tuple[Any, bool]] = {}

def root_reference_point(
    root_element: shaping_elements.ShapingElement,
) -> tuple[float, float] | None:
  """A fixed paper point inside the root element (element coordinates).

  The 3-D heuristics' "up" is the paper's top side at this point (see
  `_root_reference_flipped`); `shaping.measured_directions` measures in the
  same convention. The offsets avoid grid lines (creases).

  Args:
    root_element: The root shaping element.

  Returns:
    (x, y), or None when no candidate lies inside the element.
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
  inside = False
  for i in range(len(polygon)):
    (x1, y1), (x2, y2) = polygon[i], polygon[(i + 1) % len(polygon)]
    if (y1 > point[1]) != (y2 > point[1]):
      if point[0] < x1 + (point[1] - y1) * (x2 - x1) / (y2 - y1):
        inside = not inside
  return inside

def _root_reference_flipped(
    root_element: shaping_elements.ShapingElement,
    graph_data: geometric_simple_fold.GraphData,
) -> bool:
  """Whether the root's reference face is flipped in `graph_data`.

  The folded frame of `prepare_graph_data` has face 0 face-up, and face 0 is
  an arbitrary face that changes whenever creases are added: on Pegasus a
  flat wing fan made the new face 0 lie opposite to the old one relative to
  the root, so the posture frame (normal = folded +z) flipped and the whole
  pose was computed in the mirror-image convention (12 of 16 paired flaps
  on the wrong side after `fan_out_fold_heuristic` then `rotation_3d`). The
  posture normal is therefore pinned to the paper's top side at
  `root_reference_point`; when that face is flipped in the folded frame the
  posture frame is the folded frame mirrored (cut lines reflected across the
  root axis, mountain and valley swapped).

  Args:
    root_element: The root shaping element.
    graph_data: Graph data of the pattern the cut lines are drawn on.

  Returns:
    True when the reference face is flipped relative to face 0.
  """
  cached = _FLIP_CACHE.get(id(graph_data))
  if cached is not None and cached[0] is graph_data:
    return cached[1]
  flipped = False
  reference = root_reference_point(root_element)
  if reference is not None:
    vertices = np.asarray(graph_data['vertices_coords'], dtype=float)
    point = np.asarray(reference, dtype=float)
    if vertices.max() > 1.5 >= point.max():
      point = point * root_element.grid_size
    for index, face in enumerate(graph_data['face_vertices']):
      if _point_in_face(point, vertices[face]):
        flipped = bool(graph_data['face_orientations'][index])
        break
  if len(_FLIP_CACHE) > 16:
    _FLIP_CACHE.clear()
  _FLIP_CACHE[id(graph_data)] = (graph_data, flipped)
  return flipped

def _rotate_simple_fold_direction_by_model_orientation(
    simple_fold_direction: np.ndarray,
    root_element: shaping_elements.ShapingElement,
    stick_figure: sf_utils.StickFigure,
    graph_data: geometric_simple_fold.GraphData,
    folded_verts: list[tuple[float, float] | None],
) -> np.ndarray:
  """Maps a direction from the posture frame into the folded frame.

  The posture frame (`heuristics_3d.get_flap_post_dict`) has the root stick
  along +y and the folded base's +z as its normal, so it is the folded frame
  turned about z until the root's folded start->end direction lies on +y. The
  inverse turn, about z, maps posture directions onto the folded paper.

  This used `angle_utils.get_rotation_matrix_to_align(root, +y)`: for a root
  folded along -y that returns a 180 degree turn about the x axis, which on
  the paper plane is the reflection y -> -y. Every cut line was mirrored, so
  each in-plane rotation went to the other side (left flaps to the right):
  measured on the Pegasus base (root `back` along -y), every single-flap
  rotation realized the requested |az| and el with the azimuth sign flipped.

  Args:
    simple_fold_direction: Direction in the posture frame (z ignored).
    root_element: The root element of the shaping element tree.
    stick_figure: The stick figure.
    graph_data: The graph data for the crease pattern.
    folded_verts: The folded vertices of the crease pattern.

  Returns:
    The direction in folded coordinates (z = 0).
  """
  root_element_orientation = _get_root_element_orientation(
      root_element, stick_figure, graph_data, folded_verts
  )
  logging.info('root_element_orientation: %s', root_element_orientation)
  simple_fold_direction = np.asarray(simple_fold_direction, dtype=float)
  if _root_reference_flipped(root_element, graph_data):
    simple_fold_direction = simple_fold_direction * np.array([-1.0, 1.0, 1.0])
  angle = math.atan2(-root_element_orientation[0], root_element_orientation[1])
  cosine, sine = math.cos(angle), math.sin(angle)
  rotation_matrix = np.array(
      [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
  )
  new_simple_fold_direction = rotation_matrix @ np.asarray(
      simple_fold_direction, dtype=float
  )
  new_simple_fold_direction[2] = 0.0
  logging.info('new_simple_fold_direction: %s', new_simple_fold_direction)
  return new_simple_fold_direction

def _element_folded_direction(
    element: shaping_elements.ShapingElement,
    graph_data: geometric_simple_fold.GraphData,
    folded_verts: list[tuple[float, float] | None],
) -> np.ndarray | None:
  """Proximal-to-distal direction of an element in the folded base (2-D).

  Same construction as `shaping_elements.get_unfolded_root_vector`: a point
  on the element's proximal boundary and one depth step into the element,
  both mapped to the folded state. Reliable for flaps only (see
  `_measured_base_signs`).

  Args:
    element: A shaping flap or river.
    graph_data: The graph data for the crease pattern.
    folded_verts: The folded vertices of the crease pattern.

  Returns:
    The folded direction, or None when it cannot be determined.
  """
  try:
    bank_index = None
    if isinstance(element, shaping_elements.ShapingRiver):
      bank_index = shaping_elements.flip_bank_index(
          element.get_outer_bank_index()
      )
    boundary = shaping_elements.get_shaping_boundary(
        element, boundary_index=bank_index
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
    folded = [
        geometric_utils.get_folded_pos(
            point=p,
            unfolded_coords=graph_data['vertices_coords'],
            folded_coords=folded_verts,
            edge_vertices=graph_data['edge_vertices'],
            face_vertices=graph_data['face_vertices'],
        )
        for p in (point, inner)
    ]
  except (AssertionError, ValueError, IndexError, KeyError, TypeError) as e:
    logging.info('No folded direction for %s: %s', element.name, e)
    return None
  if any(f is None for f in folded):
    return None
  return np.array(
      [folded[1][0] - folded[0][0], folded[1][1] - folded[0][1]], dtype=float
  )

def _measured_base_signs(
    elements: Sequence[shaping_elements.ShapingElement],
    root_element: shaping_elements.ShapingElement,
    stick_figure: sf_utils.StickFigure,
    graph_data: geometric_simple_fold.GraphData,
) -> dict[str, float]:
  """+1/-1: each element points along/against the root in the folded base.

  `heuristics_3d.get_flap_post_dict` infers each element's base direction
  from hinge fold states. On the Pegasus base that inference had 3 of 6
  wings pointing forward while the folded base has them pointing backward,
  so their rotations were computed from the wrong starting direction. The
  folded state is the ground truth; elements whose folded direction is not
  clearly along the root axis are left to the inference.

  Args:
    elements: Shaping elements (unmerged).
    root_element: The root element.
    stick_figure: The stick figure.
    graph_data: Graph data of the base crease pattern.

  Returns:
    `{element name: +1.0 | -1.0}`.
  """
  folded_verts = graph_data['folded_vertices']
  try:
    root_direction = _get_root_element_orientation(
        root_element, stick_figure, graph_data, folded_verts
    )[:2]
  except (AssertionError, ValueError, IndexError, KeyError, TypeError) as e:
    logging.warning('No measured base directions (root orientation: %s).', e)
    return {}
  if np.linalg.norm(root_direction) < 1e-9:
    return {}
  root_direction = root_direction / np.linalg.norm(root_direction)
  signs = {}
  for element in elements:
    # Rivers keep the inferred direction: the bank conventions of non-root
    # rivers do not identify their proximal bank (measured on Pegasus:
    # `withers` read as pointing backward, which turned it by 168 degrees).
    if element.name == root_element.name or not isinstance(
        element, shaping_elements.ShapingFlap
    ):
      continue
    # Segments that continue a chain (both joints shared by at most one other
    # stick) ride on their parent segment and inherit its direction; their
    # inner sub-flap boundary is not a reliable proximal boundary (measured on
    # scorpion leg_L4_tarsus: an inverted sign made the femur/tarsus folds
    # unfoldable in 3-D, 8.8% strain).
    if element.name in stick_figure:
      stick = stick_figure[element.name]
      if all(
          stick_figure.joints[j].degree <= 2 for j in (stick.start, stick.end)
      ):
        continue
    direction = _element_folded_direction(element, graph_data, folded_verts)
    if direction is None or np.linalg.norm(direction) < 1e-9:
      continue
    cosine = float(np.dot(direction, root_direction)) / float(
        np.linalg.norm(direction)
    )
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
  toward_normal = simple_fold_info.rotation_angle > 0
  if _root_reference_flipped(root_element, graph_data):
    toward_normal = not toward_normal
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
  """Returns shaping creases using 3D geometric folding heuristics."""
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

  root = heuristics.get_first_stick_for_shaping(stick_figure)
  stick_figure = dataclasses.replace(stick_figure, root=root.name)
  stick_figure = stick_figure.align_root_stick_to_target_azimuth_elevation()

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
    first = list(hparams.priority)
    elements.sort(
        key=lambda e: first.index(e.name) if e.name in first else len(first)
    )
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
        ' graph creation failed); the 3-D heuristics cannot place cut lines.'
    )

  base_signs = None
  if hparams.measure_base_directions:
    base_signs = _measured_base_signs(
        list(shaping_flaps) + list(shaping_rivers),
        root_element,
        stick_figure,
        hparams.direction_graph_data or graph_data,
    )

  def _shape_elements_once(
      simple_fold_info_dict: dict[str, Sequence[heuristics_3d.SimpleFoldInfo]],
  ) -> tuple[list[crease_utils.Crease], dict[str, list[ElementShapingResult]]]:
    accepted_creases_by_element: dict[str, list[crease_utils.Crease]] = {}
    discarded_partners: dict[str, str] = {}
    detailed_shaping_info: dict[str, list[ElementShapingResult]] = {}
    current_cp = base_cp

    def _rollback_partner(partner_name: str, failed_name: str) -> None:
      nonlocal current_cp, accepted_creases_by_element
      if partner_name in accepted_creases_by_element:
        accepted_creases_by_element.pop(partner_name, None)
        detailed_shaping_info.pop(partner_name, None)
        logging.warning(
            'Flat foldability violation detected after shaping element: %s. '
            'Discarding creases for this element (symmetric partner %s failed).',
            partner_name,
            failed_name,
        )
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
          accepted_creases_by_element = surviving
        current_cp = rebuilt_cp
      else:
        discarded_partners[partner_name] = failed_name

    for element in elements:
      if element.name in discarded_partners:
        continue
      if element.name not in simple_fold_info_dict:
        continue
      simple_fold_infos = simple_fold_info_dict[element.name]
      element_creases: list[crease_utils.Crease] = []
      element_had_assertion_failure = False
      extra_limiting_rectangles = [
          shaping_flaps_by_name[sac_flap].to_legacy_rectangle()
          for sac_flap in flap_info_dict[element.name].sacrificial_flaps
      ]
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
          )
          element_creases.extend(element_output['shaping_creases'])
          if element.name not in detailed_shaping_info:
            detailed_shaping_info[element.name] = []
          detailed_shaping_info[element.name].append(element_output)
        except AssertionError as e:
          element_had_assertion_failure = True
          logging.warning(
              'Failed to shape element: %s with 3D geometric folding'
              ' heuristics. Simple fold info: %s. Skipping. Error: %s',
              element.name,
              simple_fold_info.rotation_axis_local,
              e,
          )
      if element_had_assertion_failure and element.name in symmetric_partners:
        detailed_shaping_info.pop(element.name, None)
        _rollback_partner(symmetric_partners[element.name], element.name)
        continue
      if hparams.check_ff_violations and element_creases:
        updated_cp = current_cp.update_with_shaping_creases(element_creases)
        if not updated_cp.local_flat_foldability():
          detailed_shaping_info.pop(element.name, None)
          if element.name in symmetric_partners:
            _rollback_partner(symmetric_partners[element.name], element.name)
          continue
        current_cp = updated_cp
      accepted_creases_by_element[element.name] = element_creases

    shaping_creases: list[crease_utils.Crease] = []
    for creases in accepted_creases_by_element.values():
      shaping_creases.extend(creases)
    return shaping_creases, detailed_shaping_info

  unrotated: set[str] = set()
  if hparams.shape_subset is not None:
    unrotated = {stick.name for stick in stick_figure.children} - set(
        hparams.shape_subset
    )
  shaping_creases: list[crease_utils.Crease] = []
  detailed_shaping_info: dict[str, list[ElementShapingResult]] = {}
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
    )
    shaping_creases, detailed_shaping_info = _shape_elements_once(
        simple_fold_info_dict
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

  return Heuristics3DShapingResult(
      shaping_creases=shaping_creases,
      detailed_shaping_info=detailed_shaping_info,
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
    assert packing_cp is not None, 'Packing CP is None.'
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


def get_zigzag_shaping_params_from_3d_creases(
    *,
    element: shaping_elements.ShapingElement,
    simple_fold_info: heuristics_3d.SimpleFoldInfo,
    shaping_creases_3d: Sequence[crease_utils.Crease],
    central_crease_removed: bool,
) -> zigzag_shaping.ZigzagShapingParamsForElement | None:
  """Returns the zigzag shaping params matching 3D geometric folding heuristics.

  This function first computes the shaping creases using 3D heuristics. Then it
  computes the target flap angle and other zigzag parameters by visually
  matching the 3D heuristic creases with creases generated by the zigzag
  shaping algorithm.

  Args:
      element: The shaping element.
      simple_fold_info: The simple fold info for the element.
      shaping_creases_3d: The shaping creases from 3D heuristics.
      central_crease_removed: Whether the central crease has been removed.

  Returns:
      The zigzag shaping params for the element.
  """
  if not shaping_creases_3d:
    logging.info(
        'No shaping creases found for element %s using 3d heuristics.',
        element.name,
    )
    return None

  if simple_fold_info.rotation_axis_depth is None:
    logging.warning(
        'Rotation axis depth is None for element %s. Cannot compute target'
        ' zigzag depth.',
        element.name,
    )
    return None

  target_zigzag_depth = abs(simple_fold_info.rotation_axis_depth)
  target_flap_angle = shaping_utils.fold_angle_from_zigzag_depth(
      target_zigzag_depth
  )
  fold_percent = abs(simple_fold_info.rotation_angle / 180)

  best_params = simple_fold_shaping.infer_fold_order_phase_from_creases(
      target_creases=shaping_creases_3d,
      element=element,
      target_flap_angle=target_flap_angle,
      inscribe_depth=simple_fold_info.inscribe_depth,
      fold_percent=fold_percent,
      central_crease_removed=central_crease_removed,
  )

  return best_params


def geometric_simple_fold_creases_from_zigzag_params(
    base_cp: cp_utils.CreasePattern,
    rectangles: Sequence[shaping_elements.ShapingFlap],
    element_shaping_params: Sequence[
        zigzag_shaping.ZigzagShapingParamsForElement
    ],
    root_element: shaping_elements.ShapingElement,
    stick_figure: sf_utils.StickFigure,
    *,
    rivers: Sequence[shaping_elements.ShapingRiver] | None = None,
    graph_data: geometric_simple_fold.GraphData | None = None,
) -> Sequence[crease_utils.Crease]:
  """Returns creases from a geometric simple fold based on zigzag parameters."""
  if rivers is None:
    rivers = []

  if graph_data is None:
    graph_data = geometric_simple_fold.prepare_graph_data(base_cp)

  shaping_creases = []

  for param in element_shaping_params:
    element_name = param.element_name
    element = shaping_elements.get_element_by_name(
        element_name, list(rectangles), list(rivers)
    )
    zigzag_params = param.zigzag_params

    # Resolve zigzag depth
    if zigzag_params.target_flap_angle is not None:
      target_zigzag_depth = shaping_utils.zigzag_depth_from_fold_angle(
          zigzag_params.target_flap_angle
      )
    else:
      target_zigzag_depth = zigzag_params.target_zigzag_depth
    assert (
        target_zigzag_depth is not None
    ), f'Target zigzag depth is None for element: {element_name}.'
    simple_fold_direction = (
        np.array([1.0, target_zigzag_depth, 0.0])
        if zigzag_params.even_phase
        else np.array([1.0, -target_zigzag_depth, 0.0])
    )
    rotation_angle = (
        zigzag_params.fold_percent * 180
        if zigzag_params.fold_order[0] == crease_utils.Fold.MOUNTAIN
        else -1 * zigzag_params.fold_percent * 180
    )
    simple_fold_info = heuristics_3d.SimpleFoldInfo(
        rotation_axis_local=simple_fold_direction,
        rotation_angle=rotation_angle,
        inscribe_depth=zigzag_params.depth,
        # Dummy values
        posture_matrix_before_shaping=heuristics_3d.PostureMatrix.from_matrix(
            np.eye(3)
        ),
        posture_matrix_after_shaping=heuristics_3d.PostureMatrix.from_matrix(
            np.eye(3)
        ),
        required_flip=False,
        rotation_axis=np.array([0.0, 0.0, 0.0]),
    )

    result = shape_element_with_3d_geometric_folding_heuristics(
        element=element,
        base_cp=base_cp,
        simple_fold_info=simple_fold_info,
        hinge_info_list=[],
        root_element=root_element,
        stick_figure=stick_figure,
        graph_data=graph_data,
        inscribe_hparams=InscribeHParams(
            avoid_hinges=False,
            inscribe_depth=zigzag_params.depth,
        ),
    )
    shaping_creases.extend(result['shaping_creases'])

  return shaping_creases


@dataclasses.dataclass(frozen=True)
class FoldedAxis:
  """An element's axis in the folded state.

  Attributes:
    base: Folded position of the element's proximal end (on its center line).
    tip: Folded position of the element's distal end.
    width: Extent of the element's folded footprint across the axis.
  """

  base: np.ndarray
  tip: np.ndarray
  width: float

  @property
  def length(self) -> float:
    """Length of the folded axis from base to tip."""
    return float(np.linalg.norm(self.tip - self.base))

  @property
  def direction(self) -> np.ndarray:
    """Unit direction vector from base to tip."""
    return (self.tip - self.base) / max(self.length, 1e-12)


def element_folded_points(
    graph_data: geometric_simple_fold.GraphData,
    limiting_polygon: geometric_simple_fold.LimitingPolygon,
) -> np.ndarray:
  """Folded points of the face regions inside `limiting_polygon`."""
  points = geometric_simple_fold.clipped_folded_points(
      graph_data, limiting_polygon
  )
  return np.array(points, dtype=float).reshape(-1, 2)


def get_element_folded_axis(
    graph_data: geometric_simple_fold.GraphData,
    limiting_polygon: geometric_simple_fold.LimitingPolygon,
    reference_point: Sequence[float] | np.ndarray,
    *,
    min_aspect: float = 1.2,
) -> FoldedAxis:
  """Returns the folded axis of an element, oriented away from a reference.

  The axis is the principal direction of the element's folded footprint; if the
  footprint is nearly square (aspect < `min_aspect`), the direction from the
  reference point to the footprint center is used instead.

  Args:
    graph_data: Folded-state graph of the current crease pattern.
    limiting_polygon: The element's region (unfolded coordinates).
    reference_point: Folded point on the proximal side (e.g. the center of the
      parent element's footprint).
    min_aspect: Minimum footprint aspect ratio to trust the principal axis.

  Returns:
    The folded axis.

  Raises:
    ValueError: If the element has no folded faces.
  """
  points = element_folded_points(graph_data, limiting_polygon)
  if len(points) < 3:
    raise ValueError('The element has no faces in the folded state.')
  reference = np.asarray(reference_point, dtype=float)
  mean = points.mean(axis=0)
  centered = points - mean
  eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
  axis = eigenvectors[:, 1]
  extent = centered @ axis
  across = centered @ np.array([-axis[1], axis[0]])
  aspect = (extent.max() - extent.min()) / max(
      across.max() - across.min(), 1e-12
  )
  if aspect < min_aspect or eigenvalues[1] <= 0:
    axis = mean - reference
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
      raise ValueError('Cannot orient the element axis.')
    axis = axis / norm
  if np.dot(mean - reference, axis) < 0:
    axis = -axis
  normal = np.array([-axis[1], axis[0]])
  extent = centered @ axis
  across = centered @ normal
  center = mean + normal * (across.max() + across.min()) / 2
  return FoldedAxis(
      base=center + axis * extent.min(),
      tip=center + axis * extent.max(),
      width=float(across.max() - across.min()),
  )


def _rotate(vector: np.ndarray, angle_deg: float) -> np.ndarray:
  angle = math.radians(angle_deg)
  c, s = math.cos(angle), math.sin(angle)
  return np.array(
      [c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]]
  )


def _directed_line(
    point: np.ndarray, direction: np.ndarray, inner_point: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float]]:
  """Line through `point` along ±`direction` with `inner_point` on its left."""
  cross = direction[0] * (inner_point[1] - point[1]) - direction[1] * (
      inner_point[0] - point[0]
  )
  if cross < 0:
    direction = -direction
  return (
      (float(point[0]), float(point[1])),
      (float(point[0] + direction[0]), float(point[1] + direction[1])),
  )


def snap_to_points(
    point: np.ndarray, points: np.ndarray | None, max_distance: float
) -> np.ndarray:
  """The nearest of `points` within `max_distance` of `point`, else `point`.

  Cut lines through existing folded vertices avoid sliver faces.

  Args:
    point: The target point to snap.
    points: Array of candidate points, or None.
    max_distance: Maximum distance threshold for snapping.

  Returns:
    The nearest snapped point or the original point.
  """
  if points is None or not len(points):  # pylint: disable=g-explicit-length-test
    return point
  distances = np.linalg.norm(points - point, axis=1)
  index = int(np.argmin(distances))
  if distances[index] <= max_distance:
    return points[index]
  return point


def line_span(
    line: tuple[tuple[float, float], tuple[float, float]],
    points: np.ndarray,
    margin: float,
) -> tuple[float, float]:
  """Extent of `points` along `line` (from its start), widened by margin."""
  start = np.asarray(line[0])
  direction = np.asarray(line[1]) - start
  direction = direction / np.linalg.norm(direction)
  along = (points - start) @ direction
  return float(along.min() - margin), float(along.max() + margin)


def reverse_fold_cut_line(
    axis: FoldedAxis,
    *,
    angle: float,
    depth: float,
    snap_points: np.ndarray | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
  """Cut line of a flat reverse fold turning the distal part by `angle`.

  Reflecting the distal part across a line at angle a/2 to the axis turns it
  by a (counter-clockwise in folded coordinates for a > 0).

  Args:
    axis: The element's folded axis.
    angle: Turn in degrees, 0 < |angle| <= 180.
    depth: Fraction of the element (0 = base, 1 = tip) where the cut line
      crosses the center line.
    snap_points: Folded vertices to snap the crossing point to (within a quarter
      of the element's width or length).

  Returns:
    The directed cut line, with the distal part on its left.
  """
  if not 0 < abs(angle) <= 180:
    raise ValueError(f'angle must be in [-180, 0) or (0, 180]; got {angle}.')
  point = snap_to_points(
      axis.base + depth * (axis.tip - axis.base),
      snap_points,
      0.25 * min(axis.width, axis.length),
  )
  direction = _rotate(axis.direction, angle / 2)
  return _directed_line(point, direction, axis.tip)


def crimp_fold_cut_lines(
    axis: FoldedAxis,
    *,
    angle: float,
    depth: float,
    gap: float | None = None,
    margin: float = 1.25,
    snap_points: np.ndarray | None = None,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
  """Two cut lines of a flat crimp turning the distal part by `angle`.

  The lines are tilted by ±angle/4 from the perpendicular of the axis, so the
  composition of the two reflections turns the distal part by `angle`
  (counter-clockwise for > 0). By default the gap along the center line makes
  them meet `margin` half-widths from the axis, just outside the flap on the
  inside of the bend.

  Args:
    axis: The element's folded axis.
    angle: Turn in degrees, 0 < |angle| < 180.
    depth: Fraction of the element where the first line crosses the axis.
    gap: Distance between the lines along the axis (folded units).
    margin: Meeting point distance in half-widths when `gap` is None; the
      default gap is also at least 15 % of the element's length.
    snap_points: Folded vertices to snap the first crossing point to.

  Returns:
    `[first, second]` directed lines with the strip between them on the left.
  """
  if not 0 < abs(angle) < 180:
    raise ValueError(f'angle must be in (-180, 0) or (0, 180); got {angle}.')
  if gap is None:
    gap = max(
        margin * axis.width * math.tan(math.radians(abs(angle) / 4)),
        0.15 * axis.length,
    )
  first_point = snap_to_points(
      axis.base + depth * (axis.tip - axis.base),
      snap_points,
      0.25 * min(axis.width, axis.length),
  )
  second_point = first_point + gap * axis.direction
  normal = _rotate(axis.direction, 90)
  # Folding along the first line, then along the image of the second, maps
  # the distal part by R1 o R2: a rotation by 2 * (phi1 - phi2) = angle. The
  # lines converge on the outer side of the bend (the inner side shortens).
  first = _directed_line(first_point, _rotate(normal, angle / 4), second_point)
  second = _directed_line(
      second_point, _rotate(normal, -angle / 4), first_point
  )
  return [first, second]
