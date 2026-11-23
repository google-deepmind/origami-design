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

"""Utility functions for applying geometric simple folds to crease patterns."""

import collections
from collections.abc import Callable, Sequence
import itertools
import logging
import math
from typing import TypedDict

from .. import timeout as timeout_lib
from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import point as point_utils
from ..third_party.flat_folder import folded_state
from ..third_party.flat_folder import graph_creation
from ..third_party.flat_folder import graph_topology
from ..third_party.flat_folder import math_utils
from . import shaping_elements
from . import shaping_utils

Point = math_utils.Point
Edge = math_utils.Edge
type Matrix2x2 = tuple[tuple[float, float], tuple[float, float]]
type LimitingPolygon = (
    shaping_utils.Rectangle
    | shaping_elements.ShapingFlap
    | shaping_elements.ShapingRiver
    | list[shaping_utils.Rectangle]
)

_PLANAR_SNAP_TOL = 1e-8
_CLIP_TOL = 1e-9
_MIN_POLYGON_AREA = 1e-12
_MAX_SWEEP_FAILURES = 3


def interpolate(*, point1: Point, point2: Point, t: float) -> Point:
  """Linearly interpolates between two 2D points."""
  return (
      point1[0] + t * (point2[0] - point1[0]),
      point1[1] + t * (point2[1] - point1[1]),
  )


class GraphData(TypedDict):
  """Graph data for flat-folding simulations.

  Attributes:
    vertices_coords: Unfolded coordinates of vertices.
    face_vertices: Vertex index cycle per face.
    edge_vertices: Vertex index pairs per edge.
    edge_assignments: Fold assignment (`'M'`, `'V'`, `'B'`, `'U'`) per edge.
    edge_key_to_faces: Map from sorted vertex pair to adjacent face indices.
    folded_vertices: Flat-folded 2D coordinates of vertices.
    face_orientations: Per-face orientation (`True` when flipped relative to
      face 0).
  """

  vertices_coords: list[Point]
  face_vertices: list[list[int]]
  edge_vertices: list[list[int]]
  edge_assignments: list[str]
  edge_key_to_faces: dict[tuple[int, int], list[int]]
  folded_vertices: list[Point | None]
  face_orientations: list[bool | None]


def _segments_strictly_cross(
    p1: Point, p2: Point, p3: Point, p4: Point
) -> bool:
  """Returns whether segments `(p1, p2)` and `(p3, p4)` cross strictly."""
  if (
      max(p1[0], p2[0]) < min(p3[0], p4[0]) - _PLANAR_SNAP_TOL
      or max(p3[0], p4[0]) < min(p1[0], p2[0]) - _PLANAR_SNAP_TOL
      or max(p1[1], p2[1]) < min(p3[1], p4[1]) - _PLANAR_SNAP_TOL
      or max(p3[1], p4[1]) < min(p1[1], p2[1]) - _PLANAR_SNAP_TOL
  ):
    return False

  def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

  d1 = _cross(p1, p2, p3)
  d2 = _cross(p1, p2, p4)
  d3 = _cross(p3, p4, p1)
  d4 = _cross(p3, p4, p2)
  tol = 1e-12
  return ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and (
      (d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)
  )


def _graph_from_planar_lines(
    lines: Sequence[tuple[Point, Point]],
) -> tuple[list[Point], list[Edge], list[list[int]]]:
  """Builds a planar graph directly from pre-split segments.

  Used as a fallback when sweep-line subdivision fails repeatedly due to
  floating-point endpoint ordering degeneracies. Returns empty lists if any
  vertex lies strictly inside an edge or any pair of edges crosses strictly.

  Args:
    lines: Segments as pairs of `(x, y)` coordinates.

  Returns:
    Tuple `(vertices, edge_vertices, edge_lines)`, or `([], [], [])` if the
    input segments are not already planar.
  """
  logging.warning(
      'Sweep-line graph creation failed; building planar graph directly from'
      ' %d segments.',
      len(lines),
  )
  vertices: list[Point] = []
  grid: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
  inv_cell = 1.0 / _PLANAR_SNAP_TOL

  def _intern_vertex(point: Point) -> int:
    x, y = float(point[0]), float(point[1])
    cx = math.floor(x * inv_cell)
    cy = math.floor(y * inv_cell)
    for gx in (cx - 1, cx, cx + 1):
      for gy in (cy - 1, cy, cy + 1):
        for vid in grid.get((gx, gy), ()):
          vx, vy = vertices[vid]
          if math.hypot(x - vx, y - vy) < _PLANAR_SNAP_TOL:
            return vid
    vid = len(vertices)
    vertices.append((x, y))
    grid[(cx, cy)].append(vid)
    return vid

  edge_vertices: list[Edge] = []
  edge_lines: list[list[int]] = []
  edge_index: dict[tuple[int, int], int] = {}
  for line_idx, (p1, p2) in enumerate(lines):
    timeout_lib.maybe_timeout_cooperatively()
    v1 = _intern_vertex(p1)
    v2 = _intern_vertex(p2)
    if v1 == v2:
      continue
    key = (v1, v2) if v1 < v2 else (v2, v1)
    existing = edge_index.get(key)
    if existing is not None:
      edge_lines[existing].append(line_idx)
    else:
      edge_index[key] = len(edge_vertices)
      edge_vertices.append([v1, v2])
      edge_lines.append([line_idx])

  # Reject T-junctions (a non-incident vertex lying strictly inside an edge).
  for v1, v2 in edge_vertices:
    timeout_lib.maybe_timeout_cooperatively()
    ax, ay = vertices[v1]
    bx, by = vertices[v2]
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    length = math.sqrt(len_sq)
    x_lo = min(ax, bx) - _PLANAR_SNAP_TOL
    x_hi = max(ax, bx) + _PLANAR_SNAP_TOL
    y_lo = min(ay, by) - _PLANAR_SNAP_TOL
    y_hi = max(ay, by) + _PLANAR_SNAP_TOL
    for vid, (px, py) in enumerate(vertices):
      if vid in (v1, v2):
        continue
      if px < x_lo or px > x_hi or py < y_lo or py > y_hi:
        continue
      rx, ry = px - ax, py - ay
      t = (rx * dx + ry * dy) / len_sq
      if (
          _CLIP_TOL < t < 1.0 - _CLIP_TOL
          and abs(rx * dy - ry * dx) / length < _PLANAR_SNAP_TOL
      ):
        return [], [], []

  # Reject proper interior X-crossings between non-adjacent edges.
  num_edges = len(edge_vertices)
  for i in range(num_edges):
    timeout_lib.maybe_timeout_cooperatively()
    a1, a2 = edge_vertices[i]
    pa1, pa2 = vertices[a1], vertices[a2]
    for j in range(i + 1, num_edges):
      b1, b2 = edge_vertices[j]
      if a1 == b1 or a1 == b2 or a2 == b1 or a2 == b2:
        continue
      if _segments_strictly_cross(pa1, pa2, vertices[b1], vertices[b2]):
        return [], [], []

  return vertices, edge_vertices, edge_lines


def fast_create_graph(
    lines: Sequence[tuple[Point, Point]],
) -> tuple[list[Point], list[Edge], list[list[int]]]:
  """Builds a planar subdivision graph, falling back on sweep-line errors."""
  if not lines:
    return [], [], []
  line_list = list(lines)
  min_length = math_utils.min_line_length(line_list)
  if min_length == 0:
    min_length = 1.0

  vertex_count = 0
  edge_count = 0
  stability_count = 0
  fail_count = 0
  result_vertices: list[Point] = []
  result_edge_vertices: list[Edge] = []
  result_edge_lines: list[list[int]] = []

  for exponent in range(3, 23):
    timeout_lib.maybe_timeout_cooperatively()
    epsilon = min_length / (2**exponent)
    if epsilon < math_utils.FLOAT_EPSILON:
      break
    try:
      new_vertices, new_edge_vertices, new_edge_lines = (
          graph_creation._create_graph_from_lines_eps(  # pylint: disable=protected-access
              line_list, epsilon
          )
      )
    except (ValueError, IndexError):
      fail_count += 1
      if fail_count >= _MAX_SWEEP_FAILURES:
        return _graph_from_planar_lines(line_list)
      vertex_count = 0
      edge_count = 0
      stability_count = 0
      continue

    if (
        len(new_vertices) == vertex_count
        and len(new_edge_vertices) == edge_count
    ):
      stability_count += 1
    else:
      vertex_count = len(new_vertices)
      edge_count = len(new_edge_vertices)
      stability_count = 1

    if stability_count >= 2:
      return new_vertices, new_edge_vertices, new_edge_lines
    result_vertices = new_vertices
    result_edge_vertices = new_edge_vertices
    result_edge_lines = new_edge_lines

  if not result_vertices and fail_count > 0:
    return _graph_from_planar_lines(line_list)
  return result_vertices, result_edge_vertices, result_edge_lines


def fold_to_assignment(fold: crease_utils.Fold) -> str:
  """Converts a `Fold` enum into a FOLD-format single-letter assignment."""
  if fold == crease_utils.Fold.MOUNTAIN:
    return 'M'
  if fold == crease_utils.Fold.VALLEY:
    return 'V'
  if fold == crease_utils.Fold.BORDER:
    return 'B'
  return 'U'


def prepare_graph_data(crease_pattern: cp_utils.CreasePattern) -> GraphData:
  """Prepares graph data from a crease pattern for flat-folding simulations."""
  lines: list[tuple[Point, Point]] = []
  assignments: list[str] = []
  for c in crease_pattern.creases:
    timeout_lib.maybe_timeout_cooperatively()
    lines.append(((c.p1.x, c.p1.y), (c.p2.x, c.p2.y)))
    assignments.append(fold_to_assignment(c.fold))

  vertices_coords, edge_vertices, edge_lines = fast_create_graph(lines)
  edge_assignments = []
  for indices in edge_lines:
    timeout_lib.maybe_timeout_cooperatively()
    edge_assignments.append(assignments[indices[0]] if indices else 'U')

  _, face_vertices = graph_topology.graph_to_adjacency_and_faces(
      vertices_coords, edge_vertices
  )
  edge_key_to_faces: dict[tuple[int, int], list[int]] = collections.defaultdict(
      list
  )
  for face_index, vertices in enumerate(face_vertices):
    timeout_lib.maybe_timeout_cooperatively()
    for i, vertex1 in enumerate(vertices):
      vertex2 = vertices[(i + 1) % len(vertices)]
      edge_key = (vertex1, vertex2) if vertex1 < vertex2 else (vertex2, vertex1)
      edge_key_to_faces[edge_key].append(face_index)

  folded_vertices, face_orientations = folded_state.compute_folded_state(
      vertices_coords, face_vertices, edge_vertices, edge_assignments
  )
  return {
      'vertices_coords': vertices_coords,
      'face_vertices': face_vertices,
      'edge_vertices': edge_vertices,
      'edge_assignments': edge_assignments,
      'edge_key_to_faces': dict(edge_key_to_faces),
      'folded_vertices': folded_vertices,
      'face_orientations': face_orientations,
  }


def _limiting_polygon_rectangles(
    limiting_polygon: LimitingPolygon,
) -> list[shaping_utils.Rectangle]:
  """Converts a `LimitingPolygon` into a list of legacy `Rectangle`s."""
  if isinstance(limiting_polygon, list):
    return limiting_polygon
  if isinstance(limiting_polygon, shaping_elements.ShapingRiver):
    return [
        shaping_utils.Rectangle.from_shaping_rect(r)
        for r in limiting_polygon.covering_rectangles
    ]
  if isinstance(limiting_polygon, shaping_elements.ShapingFlap):
    return [limiting_polygon.to_legacy_rectangle()]
  if isinstance(limiting_polygon, shaping_utils.Rectangle):
    return [limiting_polygon]
  raise ValueError(
      'limiting_polygon must be a list of Rectangles, a Rectangle, a '
      f'ShapingFlap, or a ShapingRiver. Got: {type(limiting_polygon)}.'
  )


def check_inside_limiting_polygon(
    point: Point, limiting_polygon: LimitingPolygon | None
) -> bool:
  """Returns whether `point` (unfolded coordinates) lies inside the polygon."""
  if limiting_polygon is None:
    return True
  pt = point_utils.Point(x=point[0], y=point[1], require_on_paper=False)
  if isinstance(limiting_polygon, list):
    return any(polygon.is_internal_point(pt) for polygon in limiting_polygon)
  if isinstance(
      limiting_polygon,
      (
          shaping_elements.ShapingRiver,
          shaping_utils.Rectangle,
          shaping_elements.ShapingFlap,
      ),
  ):
    return limiting_polygon.is_internal_point(pt)
  raise ValueError(
      'limiting_polygon must be a list of Rectangles, a Rectangle, a '
      f'ShapingFlap, or a ShapingRiver. Got: {type(limiting_polygon)}.'
  )


def fit_2d_isometry(
    source: Sequence[Point], target: Sequence[Point]
) -> tuple[Matrix2x2, Point, Point]:
  """Fits a least-squares 2D isometry (with reflection) from source to target.

  Args:
    source: Points in the source 2D frame.
    target: Corresponding points in the target 2D frame.

  Returns:
    `(rotation, source_mean, target_mean)` mapping each point `p` to
    `rotation @ (p - source_mean) + target_mean`.

  Raises:
    ValueError: If `source` is empty or has a different length than `target`.
  """
  n = len(source)
  if n == 0 or n != len(target):
    raise ValueError(
        'source and target must be non-empty and share the same length; got'
        f' {n} and {len(target)}.'
    )
  sx = sum(p[0] for p in source) / n
  sy = sum(p[1] for p in source) / n
  tx = sum(p[0] for p in target) / n
  ty = sum(p[1] for p in target) / n

  h00 = h01 = h10 = h11 = 0.0
  for (px, py), (qx, qy) in zip(source, target, strict=True):
    a, b = px - sx, py - sy
    c, d = qx - tx, qy - ty
    h00 += a * c
    h01 += a * d
    h10 += b * c
    h11 += b * d

  best_score = -math.inf
  best_rotation: Matrix2x2 = ((1.0, 0.0), (0.0, 1.0))
  for sign in (1.0, -1.0):
    s_h10, s_h11 = sign * h10, sign * h11
    angle = math.atan2(h01 - s_h10, h00 + s_h11)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    score = (h00 + s_h11) * cos_a + (h01 - s_h10) * sin_a
    if score > best_score:
      best_score = score
      best_rotation = ((cos_a, -sin_a * sign), (sin_a, cos_a * sign))
  return best_rotation, (sx, sy), (tx, ty)


def apply_2d_isometry(
    point: Point,
    rotation: Matrix2x2,
    source_mean: Point,
    target_mean: Point,
) -> Point:
  """Applies a fitted 2D isometry `(rotation, source_mean, target_mean)`."""
  dx, dy = point[0] - source_mean[0], point[1] - source_mean[1]
  return (
      rotation[0][0] * dx + rotation[0][1] * dy + target_mean[0],
      rotation[1][0] * dx + rotation[1][1] * dy + target_mean[1],
  )


def _clip_half_plane(
    points: Sequence[Point],
    inside_fn: Callable[[Point], bool],
    intersect_fn: Callable[[Point, Point], Point],
) -> list[Point]:
  """Clips a polygon against a single half-plane (Sutherland-Hodgman step)."""
  if not points:
    return []
  clipped: list[Point] = []
  prev = points[-1]
  prev_in = inside_fn(prev)
  for cur in points:
    cur_in = inside_fn(cur)
    if cur_in:
      if not prev_in:
        clipped.append(intersect_fn(prev, cur))
      clipped.append(cur)
    elif prev_in:
      clipped.append(intersect_fn(prev, cur))
    prev, prev_in = cur, cur_in
  return clipped


def _clip_polygon_to_rectangle(
    polygon: Sequence[Point],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> list[Point]:
  """Clips a 2D polygon against `[x_min, x_max] x [y_min, y_max]`."""
  pts = _clip_half_plane(
      polygon,
      lambda p: p[0] >= x_min - _CLIP_TOL,
      lambda a, b: (
          x_min,
          a[1] + (x_min - a[0]) / (b[0] - a[0]) * (b[1] - a[1]),
      ),
  )
  pts = _clip_half_plane(
      pts,
      lambda p: p[0] <= x_max + _CLIP_TOL,
      lambda a, b: (
          x_max,
          a[1] + (x_max - a[0]) / (b[0] - a[0]) * (b[1] - a[1]),
      ),
  )
  pts = _clip_half_plane(
      pts,
      lambda p: p[1] >= y_min - _CLIP_TOL,
      lambda a, b: (
          a[0] + (y_min - a[1]) / (b[1] - a[1]) * (b[0] - a[0]),
          y_min,
      ),
  )
  return _clip_half_plane(
      pts,
      lambda p: p[1] <= y_max + _CLIP_TOL,
      lambda a, b: (
          a[0] + (y_max - a[1]) / (b[1] - a[1]) * (b[0] - a[0]),
          y_max,
      ),
  )


def _polygon_area(polygon: Sequence[Point]) -> float:
  """Computes the unsigned area of a 2D polygon via the shoelace formula."""
  if len(polygon) < 3:
    return 0.0
  twice_area = 0.0
  for i, (x1, y1) in enumerate(polygon):
    x2, y2 = polygon[(i + 1) % len(polygon)]
    twice_area += x1 * y2 - x2 * y1
  return abs(twice_area) * 0.5


def clipped_folded_points(
    graph_data: GraphData, limiting_polygon: LimitingPolygon
) -> list[Point]:
  """Returns folded vertices of face regions clipped to `limiting_polygon`.

  Clipping each unfolded face polygon against the rectangles of
  `limiting_polygon` handles sub-flaps of merged chains whose faces span
  multiple sub-elements without internal hinge creases.

  Args:
    graph_data: Folded-state graph of the crease pattern.
    limiting_polygon: Target region in unfolded coordinates.

  Returns:
    Folded 2D coordinates of the clipped face polygon corners.
  """
  rects = _limiting_polygon_rectangles(limiting_polygon)
  vertices = graph_data['vertices_coords']
  folded = graph_data['folded_vertices']
  points: list[Point] = []
  for face in graph_data['face_vertices']:
    timeout_lib.maybe_timeout_cooperatively()
    folded_pts: list[Point] = []
    skip_face = False
    for v in face:
      fp = folded[v]
      if fp is None:
        skip_face = True
        break
      folded_pts.append(fp)
    if skip_face or not folded_pts:
      continue
    unfolded_pts = [vertices[v] for v in face]
    rotation, u_mean, f_mean = fit_2d_isometry(unfolded_pts, folded_pts)
    for rect in rects:
      x_min = min(rect.x_start, rect.x_start + rect.x_len)
      x_max = max(rect.x_start, rect.x_start + rect.x_len)
      y_min = min(rect.y_start, rect.y_start + rect.y_len)
      y_max = max(rect.y_start, rect.y_start + rect.y_len)
      clipped = _clip_polygon_to_rectangle(
          unfolded_pts, x_min, y_min, x_max, y_max
      )
      if _polygon_area(clipped) <= _MIN_POLYGON_AREA:
        continue
      for unfolded_pt in clipped:
        points.append(apply_2d_isometry(unfolded_pt, rotation, u_mean, f_mean))
  return points


def get_face_intersections_in_unfolded_plane(
    face_vertex_indices: list[int],
    folded_vertices: list[Point | None],
    unfolded_vertices: list[Point],
    *,
    cut_line_start: Point,
    cut_line_end: Point,
) -> list[Point]:
  """Computes intersections of a single face with the folded-space cut line."""
  intersections: list[Point] = []
  line_crease: crease_utils.Crease | None = None
  for i, vertex1_index in enumerate(face_vertex_indices):
    vertex2_index = face_vertex_indices[(i + 1) % len(face_vertex_indices)]
    point1_folded = folded_vertices[vertex1_index]
    point2_folded = folded_vertices[vertex2_index]
    if point1_folded is None or point2_folded is None:
      continue

    segment_p1 = point_utils.Point(
        x=point1_folded[0], y=point1_folded[1], require_on_paper=False
    )
    segment_p2 = point_utils.Point(
        x=point2_folded[0], y=point2_folded[1], require_on_paper=False
    )
    # Narrowing can emit folded sub-faces with corners closer than Point's
    # default tolerance even when graph_creation kept them distinct.
    if segment_p1.isclose(segment_p2):
      continue
    segment_crease = crease_utils.Crease(
        segment_p1, segment_p2, fold=crease_utils.Fold.BORDER
    )
    if line_crease is None:
      line_p1 = point_utils.Point(
          x=cut_line_start[0], y=cut_line_start[1], require_on_paper=False
      )
      line_p2 = point_utils.Point(
          x=cut_line_end[0], y=cut_line_end[1], require_on_paper=False
      )
      line_crease = crease_utils.Crease(
          line_p1, line_p2, fold=crease_utils.Fold.BORDER
      )
    intersection = segment_crease.vertex_with_crease(
        line_crease,
        extend_lines=True,
        include_external=True,
    )
    if intersection is not None:
      if intersection.isclose(segment_p1, atol=point_utils.DEFAULT_TOLERANCE):
        intersection = segment_p1
      elif intersection.isclose(segment_p2, atol=point_utils.DEFAULT_TOLERANCE):
        intersection = segment_p2

    if (
        intersection
        and segment_crease.point_status(intersection)
        == crease_utils.PointStatus.EXTERNAL
    ):
      intersection = None

    if intersection is None:
      continue
    dx_folded = point2_folded[0] - point1_folded[0]
    dy_folded = point2_folded[1] - point1_folded[1]
    if abs(dx_folded) > abs(dy_folded):
      t = (
          (intersection.x - point1_folded[0]) / dx_folded
          if abs(dx_folded) > point_utils.DEFAULT_TOLERANCE
          else 0.0
      )
    else:
      t = (
          (intersection.y - point1_folded[1]) / dy_folded
          if abs(dy_folded) > point_utils.DEFAULT_TOLERANCE
          else 0.0
      )
    t = max(0.0, min(1.0, t))
    point_unfolded = interpolate(
        point1=unfolded_vertices[vertex1_index],
        point2=unfolded_vertices[vertex2_index],
        t=t,
    )
    if not any(
        math_utils.distance(existing_point, point_unfolded)
        < point_utils.DEFAULT_TOLERANCE
        for existing_point in intersections
    ):
      intersections.append(point_unfolded)
  return intersections


def compute_intersections_and_selected_face_indices(
    *,
    vertices_coords: list[Point],
    face_vertices: list[list[int]],
    folded_vertices: list[Point | None],
    face_orientations: list[bool | None],
    cut_line_start: Point,
    cut_line_end: Point,
    limiting_polygon: LimitingPolygon | None,
) -> tuple[dict[int, tuple[Point, Point]], list[int]]:
  """Calculates face-cut intersections and identifies selected face indices."""
  face_cuts: dict[int, tuple[Point, Point]] = {}
  selected_face_indices: list[int] = []
  for face_index, face in enumerate(face_vertices):
    timeout_lib.maybe_timeout_cooperatively()
    if face_orientations[face_index] is None:
      continue
    intersections = get_face_intersections_in_unfolded_plane(
        face,
        folded_vertices,
        vertices_coords,
        cut_line_start=cut_line_start,
        cut_line_end=cut_line_end,
    )
    if len(intersections) >= 3:
      intersections = [
          pt
          for pt in intersections
          if check_inside_limiting_polygon(pt, limiting_polygon)
      ]
    if len(intersections) >= 2:
      best_pair = max(
          itertools.combinations(intersections, 2),
          key=lambda pair: math_utils.distance(pair[0], pair[1]),
      )
      face_cuts[face_index] = best_pair
      selected_face_indices.append(face_index)
  return face_cuts, selected_face_indices


def create_new_creases_from_cuts(
    *,
    selected_face_indices: list[int],
    face_cuts: dict[int, tuple[Point, Point]],
    face_orientations: list[bool | None],
    fold_assignment: crease_utils.Fold,
) -> dict[int, crease_utils.Crease]:
  """Creates unfolded creases from per-face cut chords and face orientations."""
  new_creases_dict: dict[int, crease_utils.Crease] = {}
  for face_idx in set(selected_face_indices):
    timeout_lib.maybe_timeout_cooperatively()
    p1, p2 = face_cuts[face_idx]
    assignment = (
        fold_assignment.flip()
        if face_orientations[face_idx]
        else fold_assignment
    )
    try:
      new_creases_dict[face_idx] = crease_utils.Crease(
          point_utils.Point(x=p1[0], y=p1[1]),
          point_utils.Point(x=p2[0], y=p2[1]),
          fold=assignment,
      )
    except ValueError as e:
      logging.exception('Failed to create crease for face %d: %s', face_idx, e)
  return new_creases_dict


def _restrict_creases_to_limiting_polygon(
    creases: Sequence[crease_utils.Crease],
    limiting_polygon: LimitingPolygon,
) -> list[crease_utils.Crease]:
  """Clips `creases` to the rectangles covering `limiting_polygon`."""
  rects = _limiting_polygon_rectangles(limiting_polygon)
  restricted_lists = shaping_utils.restrict_creases_to_rectangles(
      rects, creases
  )
  restricted: list[crease_utils.Crease] = []
  for crease_list in restricted_lists:
    if len(crease_list) > 1:
      raise ValueError(
          'Expected at most one clipped segment per face cut; got'
          f' {len(crease_list)}.'
      )
    restricted.extend(crease_list)
  return restricted


def get_geometric_fold_creases(
    *,
    crease_pattern: cp_utils.CreasePattern,
    cut_line_start: Point,
    cut_line_end: Point,
    fold_assignment: crease_utils.Fold = crease_utils.Fold.MOUNTAIN,
    limiting_polygon: LimitingPolygon | None = None,
    graph_data: GraphData | None = None,
) -> Sequence[crease_utils.Crease]:
  """Returns the creases resulting from a geometric simple fold."""
  if graph_data is None:
    graph_data = prepare_graph_data(crease_pattern)
  face_cuts, selected_face_indices = (
      compute_intersections_and_selected_face_indices(
          vertices_coords=graph_data['vertices_coords'],
          face_vertices=graph_data['face_vertices'],
          folded_vertices=graph_data['folded_vertices'],
          face_orientations=graph_data['face_orientations'],
          cut_line_start=cut_line_start,
          cut_line_end=cut_line_end,
          limiting_polygon=limiting_polygon,
      )
  )
  new_creases_dict = create_new_creases_from_cuts(
      selected_face_indices=selected_face_indices,
      face_cuts=face_cuts,
      face_orientations=graph_data['face_orientations'],
      fold_assignment=fold_assignment,
  )
  creases: Sequence[crease_utils.Crease] = list(new_creases_dict.values())
  if limiting_polygon is not None:
    creases = _restrict_creases_to_limiting_polygon(creases, limiting_polygon)
  return creases


def get_segment_cut_creases(
    *,
    graph_data: GraphData,
    cut_line_start: Point,
    cut_line_end: Point,
    span: tuple[float, float],
    limiting_polygon: LimitingPolygon,
    fold_assignment: crease_utils.Fold = crease_utils.Fold.VALLEY,
) -> list[crease_utils.Crease]:
  """Cuts faces intersecting a bounded distance `span` along a folded line.

  Each candidate face chord is clipped to `limiting_polygon` first, and kept if
  its clipped midpoint maps to a signed distance along `(cut_line_start,
  cut_line_end)` inside `span`. This cleanly cuts sub-flaps whose faces extend
  beyond `limiting_polygon` while sparing unrelated layers elsewhere on the
  infinite line.

  Args:
    graph_data: Folded-state graph of the crease pattern.
    cut_line_start: Start of the cut line in folded coordinates.
    cut_line_end: A second distinct point on the cut line in folded coordinates.
    span: `(min_dist, max_dist)` signed distance bounds from `cut_line_start`.
    limiting_polygon: Unfolded region restricting the generated creases.
    fold_assignment: Fold assignment for face-up faces (flipped on flipped
      faces).

  Returns:
    Cut creases clipped to `limiting_polygon` whose midpoints fall in `span`.

  Raises:
    ValueError: If `cut_line_start` and `cut_line_end` coincide.
  """
  dx = cut_line_end[0] - cut_line_start[0]
  dy = cut_line_end[1] - cut_line_start[1]
  norm = math.hypot(dx, dy)
  if norm < _MIN_POLYGON_AREA:
    raise ValueError('cut_line_start and cut_line_end must be distinct.')
  ux, uy = dx / norm, dy / norm

  vertices = graph_data['vertices_coords']
  folded = graph_data['folded_vertices']
  orientations = graph_data['face_orientations']
  creases: list[crease_utils.Crease] = []

  for face_index, face in enumerate(graph_data['face_vertices']):
    timeout_lib.maybe_timeout_cooperatively()
    if orientations[face_index] is None:
      continue
    hits: list[tuple[Point, float, int]] = []
    for i, v1 in enumerate(face):
      v2 = face[(i + 1) % len(face)]
      p1, p2 = folded[v1], folded[v2]
      if p1 is None or p2 is None:
        continue
      d1 = ux * (p1[1] - cut_line_start[1]) - uy * (p1[0] - cut_line_start[0])
      d2 = ux * (p2[1] - cut_line_start[1]) - uy * (p2[0] - cut_line_start[0])
      if abs(d1) < _CLIP_TOL and abs(d2) < _CLIP_TOL:
        continue
      vertex = -1
      if abs(d1) < _CLIP_TOL:
        t, vertex = 0.0, v1
      elif abs(d2) < _CLIP_TOL:
        t, vertex = 1.0, v2
      elif (d1 < 0) != (d2 < 0):
        t = d1 / (d1 - d2)
      else:
        continue
      fx = p1[0] + t * (p2[0] - p1[0])
      fy = p1[1] + t * (p2[1] - p1[1])
      along = ux * (fx - cut_line_start[0]) + uy * (fy - cut_line_start[1])
      unfolded = interpolate(point1=vertices[v1], point2=vertices[v2], t=t)
      if not any(
          math_utils.distance(prev_pt, unfolded) < point_utils.DEFAULT_TOLERANCE
          for prev_pt, _, _ in hits
      ):
        hits.append((unfolded, along, vertex))

    if len(hits) < 2:
      continue
    (q1, s1, w1), (q2, s2, w2) = max(
        itertools.combinations(hits, 2),
        key=lambda pair: math_utils.distance(pair[0][0], pair[1][0]),
    )
    # Skip chords that coincide with an already-creased edge of the face.
    if w1 >= 0 and w2 >= 0:
      step = (face.index(w1) - face.index(w2)) % len(face)
      if step in (1, len(face) - 1):
        continue

    assignment = (
        fold_assignment.flip() if orientations[face_index] else fold_assignment
    )
    try:
      raw_crease = crease_utils.Crease(
          point_utils.Point(x=q1[0], y=q1[1]),
          point_utils.Point(x=q2[0], y=q2[1]),
          fold=assignment,
      )
    except ValueError:
      continue
    chord_len_sq = (q2[0] - q1[0]) ** 2 + (q2[1] - q1[1]) ** 2
    if chord_len_sq <= 1e-24:
      continue
    for clipped in _restrict_creases_to_limiting_polygon(
        [raw_crease], limiting_polygon
    ):
      mid_x = (clipped.p1.x + clipped.p2.x) * 0.5
      mid_y = (clipped.p1.y + clipped.p2.y) * 0.5
      t_mid = (
          (mid_x - q1[0]) * (q2[0] - q1[0]) + (mid_y - q1[1]) * (q2[1] - q1[1])
      ) / chord_len_sq
      along_mid = s1 + t_mid * (s2 - s1)
      if span[0] <= along_mid <= span[1]:
        creases.append(clipped)
  return creases


def shape_cp_geometric(
    crease_pattern: cp_utils.CreasePattern,
    *,
    cut_line_start: Point,
    cut_line_end: Point,
    fold_assignment: crease_utils.Fold = crease_utils.Fold.MOUNTAIN,
    limiting_polygon: LimitingPolygon | None = None,
    graph_data: GraphData | None = None,
) -> cp_utils.CreasePattern:
  """Applies a simple fold defined by a cut line in the folded state."""
  new_creases = get_geometric_fold_creases(
      crease_pattern=crease_pattern,
      cut_line_start=cut_line_start,
      cut_line_end=cut_line_end,
      fold_assignment=fold_assignment,
      limiting_polygon=limiting_polygon,
      graph_data=graph_data,
  )
  return crease_pattern.update_with_shaping_creases(new_creases)
