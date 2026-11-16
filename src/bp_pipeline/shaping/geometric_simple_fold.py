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
from collections.abc import Sequence
import dataclasses
import itertools
import logging
import math
from typing import Any, TypeAlias, TypedDict

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

LimitingPolygon: TypeAlias = (
    shaping_utils.Rectangle
    | shaping_elements.ShapingFlap
    | shaping_elements.ShapingRiver
    | list[shaping_utils.Rectangle]
)


def interpolate(*, point1: Point, point2: Point, t: float) -> Point:
  """Linearly interpolates between two points."""
  return (
      point1[0] + t * (point2[0] - point1[0]),
      point1[1] + t * (point2[1] - point1[1]),
  )


class GraphData(TypedDict):
  """Graph data for folding simulations.

  Attributes:
      vertices_coords: Unfolded coordinates of vertices.
      face_vertices: list of vertex indices per face.
      edge_vertices: list of vertex index pairs per edge.
      edge_assignments: Fold assignment (M/V/B/U) for each edge.
      edge_key_to_faces: Map from edge tuple to list of adjacent face indices.
      folded_vertices: Folded coordinates of vertices.
      face_orientations: List of face orientations (flipped or not).
  """

  vertices_coords: list[Point]
  face_vertices: list[list[int]]
  edge_vertices: list[list[int]]
  edge_assignments: list[str]
  edge_key_to_faces: dict[tuple[int, int], list[int]]
  folded_vertices: list[Point | None]
  face_orientations: list[bool | None]


def prepare_graph_data(crease_pattern: cp_utils.CreasePattern) -> GraphData:
  """Prepares graph data from a crease pattern for folding simulations."""
  lines = []
  assignments = []
  for c in crease_pattern.creases:
    timeout_lib.maybe_timeout_cooperatively()
    lines.append(((c.p1.x, c.p1.y), (c.p2.x, c.p2.y)))
    if c.fold == crease_utils.Fold.MOUNTAIN:
      assignments.append('M')
    elif c.fold == crease_utils.Fold.VALLEY:
      assignments.append('V')
    elif c.fold == crease_utils.Fold.BORDER:
      assignments.append('B')
    else:
      assignments.append('U')

  vertices_coords, edge_vertices, edge_lines = _fast_create_graph(lines)
  edge_assignments = []
  for indices in edge_lines:
    timeout_lib.maybe_timeout_cooperatively()
    if not indices:
      edge_assignments.append('U')
    else:
      edge_assignments.append(assignments[indices[0]])

  _, face_vertices = graph_topology.graph_to_adjacency_and_faces(
      vertices_coords, edge_vertices
  )
  edge_key_to_faces = collections.defaultdict(list)
  for face_index, vertices in enumerate(face_vertices):
    timeout_lib.maybe_timeout_cooperatively()
    for i, vertex1 in enumerate(vertices):
      vertex2 = vertices[(i + 1) % len(vertices)]
      edge_key = tuple(sorted((vertex1, vertex2)))
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


def _check_inside_limiting_polygon(
    point: Point, limiting_polygon: LimitingPolygon | None
) -> bool:
  """Checks if a point is inside a limiting polygon."""
  if limiting_polygon is None:
    return True
  pt = point_utils.Point(x=point[0], y=point[1], require_on_paper=False)
  if isinstance(limiting_polygon, list):
    return any(polygon.is_internal_point(pt) for polygon in limiting_polygon)
  elif isinstance(
      limiting_polygon,
      (
          shaping_elements.ShapingRiver,
          shaping_utils.Rectangle,
          shaping_elements.ShapingFlap,
      ),
  ):
    return limiting_polygon.is_internal_point(pt)
  else:
    raise ValueError(
        'limiting_polygon must be a list of Rectangles, a Rectangle, a '
        f'ShapingFlap or a ShapingRiver. Got: {type(limiting_polygon)}.'
    )


def get_face_intersections_in_unfolded_plane(
    face_vertex_indices: list[int],
    folded_vertices: list[Point | None],
    unfolded_vertices: list[Point],
    *,
    cut_line_start: Point,
    cut_line_end: Point,
) -> list[Point]:
  """Computes intersections of a single face with the cut line."""
  intersections = []
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

    if intersection is not None:
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

      point1_unfolded = unfolded_vertices[vertex1_index]
      point2_unfolded = unfolded_vertices[vertex2_index]
      point_unfolded = interpolate(
          point1=point1_unfolded, point2=point2_unfolded, t=t
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
  face_cuts = {}  # face_idx -> (p1_unfolded, p2_unfolded)
  selected_face_indices = []
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
          intersection
          for intersection in intersections
          if _check_inside_limiting_polygon(intersection, limiting_polygon)
      ]

    if len(intersections) >= 2:
      best_pair = max(
          (
              (intersection_1, intersection_2)
              for intersection_1, intersection_2 in itertools.combinations(
                  intersections, 2
              )
          ),
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
  """Creates new creases from face cuts, selected_face_indices, and orientations."""
  faces_to_process = set(selected_face_indices)

  new_creases_dict = {}
  for face_idx in faces_to_process:
    timeout_lib.maybe_timeout_cooperatively()
    intersections = face_cuts[face_idx]
    is_flipped = face_orientations[face_idx]

    current_assignment = fold_assignment
    if is_flipped:
      current_assignment = (
          crease_utils.Fold.VALLEY
          if fold_assignment == crease_utils.Fold.MOUNTAIN
          else crease_utils.Fold.MOUNTAIN
      )

    try:
      new_creases_dict[face_idx] = crease_utils.Crease(
          point_utils.Point(x=intersections[0][0], y=intersections[0][1]),
          point_utils.Point(x=intersections[1][0], y=intersections[1][1]),
          fold=current_assignment,
      )
    except ValueError as e:
      logging.exception('Failed to create crease for face %d: %s', face_idx, e)
      continue

  return new_creases_dict


def _restrict_creases_to_limiting_polygon(
    creases: Sequence[crease_utils.Crease],
    limiting_polygon: LimitingPolygon,
) -> Sequence[crease_utils.Crease]:
  """Restricts creases to the given limiting polygon."""
  if isinstance(limiting_polygon, list):
    rects = limiting_polygon
  elif isinstance(limiting_polygon, shaping_elements.ShapingRiver):
    rects = [
        shaping_utils.Rectangle.from_shaping_rect(r)
        for r in limiting_polygon.covering_rectangles
    ]
  elif isinstance(limiting_polygon, shaping_elements.ShapingFlap):
    rects = [limiting_polygon.to_legacy_rectangle()]
  elif isinstance(limiting_polygon, shaping_utils.Rectangle):
    rects = [limiting_polygon]
  else:
    raise ValueError(
        'limiting_polygon must be a list of Rectangles, a Rectangle, a '
        f'ShapingFlap or a ShapingRiver. Got: {type(limiting_polygon)}.'
    )
  restricted_crease_lists = shaping_utils.restrict_creases_to_rectangles(
      rects, creases
  )
  restricted_creases = []
  for crease_list in restricted_crease_lists:
    assert len(crease_list) <= 1
    restricted_creases.extend(crease_list)
  return restricted_creases


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
  vertices_coords = graph_data['vertices_coords']
  face_vertices = graph_data['face_vertices']
  folded_vertices = graph_data['folded_vertices']
  face_orientations = graph_data['face_orientations']

  face_cuts, selected_face_indices = (
      compute_intersections_and_selected_face_indices(
          vertices_coords=vertices_coords,
          face_vertices=face_vertices,
          folded_vertices=folded_vertices,
          face_orientations=face_orientations,
          cut_line_start=cut_line_start,
          cut_line_end=cut_line_end,
          limiting_polygon=limiting_polygon,
      )
  )

  new_creases_dict = create_new_creases_from_cuts(
      selected_face_indices=selected_face_indices,
      face_cuts=face_cuts,
      face_orientations=face_orientations,
      fold_assignment=fold_assignment,
  )
  creases = list(new_creases_dict.values())

  if limiting_polygon is not None:
    creases = _restrict_creases_to_limiting_polygon(creases, limiting_polygon)

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


def _fast_create_graph(
    lines: list[Any],
) -> tuple[list[Point], list[Edge], list[list[int]]]:
  """Computes graph representation, early-exiting on repeated sweep-line errors."""
  min_line_length = math_utils.min_line_length(lines)
  if min_line_length == 0:
    min_line_length = 1.0

  vertex_count, edge_count, stability_count = 0, 0, 0
  result_vertices: list[Point] = []
  result_edge_vertices: list[Edge] = []
  result_edge_lines: list[list[int]] = []
  fail_count = 0

  for exponent_i in range(20):
    exponent = exponent_i + 3
    epsilon = min_line_length / (2**exponent)
    if epsilon < math_utils.FLOAT_EPSILON:
      break
    try:
      new_vertices, new_edge_vertices, new_edge_lines = (
          graph_creation._create_graph_from_lines_eps(  # pylint: disable=protected-access
              lines, epsilon
          )
      )
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
      result_vertices, result_edge_vertices, result_edge_lines = (
          new_vertices,
          new_edge_vertices,
          new_edge_lines,
      )
    except (ValueError, IndexError):
      fail_count += 1
      if fail_count >= 3:
        return _graph_from_planar_lines(lines)
      vertex_count, edge_count, stability_count = 0, 0, 0

  return result_vertices, result_edge_vertices, result_edge_lines


def _graph_from_planar_lines(
    lines: list[Any],
) -> tuple[list[Point], list[Edge], list[list[int]]]:
  """Graph of lines that already meet only at their endpoints.

  Fallback when the sweep line fails: on the Pegasus base re-solved with
  remove_central_crease, `graph_creation._create_graph_from_lines_eps` raised
  "Missing Endpoint" at every epsilon even for the single crease
  ((0.8696, 0.4565), (0.8913, 0.4783)) on its own, which left every 3-D
  heuristic with an empty graph. Solver and shaping crease patterns are split
  at every crossing, so their creases can be used as edges directly.

  Args:
    lines: Lines as pairs of (x, y) points.

  Returns:
    (vertices, edge_vertices, edge_lines), or empty lists when two lines
    cross away from their endpoints (the lines are not planar).
  """
  logging.warning(
      'Sweep-line graph creation failed; building the graph from the creases'
      ' directly (%d lines).',
      len(lines),
  )
  index: dict[tuple[float, float], int] = {}
  vertices: list[Point] = []
  edge_vertices: list[Edge] = []
  edge_lines: list[list[int]] = []
  edge_index: dict[tuple[int, int], int] = {}

  def vertex(point: Any) -> int:
    key = (round(float(point[0]), 9), round(float(point[1]), 9))
    if key not in index:
      index[key] = len(vertices)
      vertices.append((float(point[0]), float(point[1])))
    return index[key]

  for line_index, (p1, p2) in enumerate(lines):
    a, b = vertex(p1), vertex(p2)
    if a == b:
      continue
    key = (min(a, b), max(a, b))
    if key in edge_index:
      edge_lines[edge_index[key]].append(line_index)
      continue
    edge_index[key] = len(edge_vertices)
    edge_vertices.append([a, b])
    edge_lines.append([line_index])
  # Planarity: no vertex may lie strictly inside another edge.
  for a, b in edge_vertices:
    (ax, ay), (bx, by) = vertices[a], vertices[b]
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    length = math.sqrt(length2)
    for px, py in vertices:
      rx, ry = px - ax, py - ay
      t = (rx * dx + ry * dy) / length2
      if 1e-9 < t < 1 - 1e-9 and abs(rx * dy - ry * dx) / length < 1e-9:
        return [], [], []
  return vertices, edge_vertices, edge_lines


_REGION_SIDE_TOLERANCE = 1e-7


def _limiting_polygon_rectangles(
    limiting_polygon: LimitingPolygon,
) -> list[shaping_utils.Rectangle]:
  """Converts a `LimitingPolygon` into legacy `Rectangle`s."""
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
      f'ShapingFlap or a ShapingRiver. Got: {type(limiting_polygon)}.'
  )


def _clip_polygon_to_rectangle(
    polygon: Sequence[Point],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> list[Point]:
  """Clips a 2-D polygon against `[x_min, x_max] x [y_min, y_max]`."""

  def _clip_half_plane(
      points: Sequence[Point],
      inside_fn,
      intersect_fn,
  ) -> list[Point]:
    if not points:
      return []
    clipped = []
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

  eps = 1e-9
  pts = list(polygon)
  pts = _clip_half_plane(
      pts,
      lambda p: p[0] >= x_min - eps,
      lambda a, b: (
          x_min,
          a[1] + (x_min - a[0]) / (b[0] - a[0]) * (b[1] - a[1]),
      ),
  )
  pts = _clip_half_plane(
      pts,
      lambda p: p[0] <= x_max + eps,
      lambda a, b: (
          x_max,
          a[1] + (x_max - a[0]) / (b[0] - a[0]) * (b[1] - a[1]),
      ),
  )
  pts = _clip_half_plane(
      pts,
      lambda p: p[1] >= y_min - eps,
      lambda a, b: (
          a[0] + (y_min - a[1]) / (b[1] - a[1]) * (b[0] - a[0]),
          y_min,
      ),
  )
  pts = _clip_half_plane(
      pts,
      lambda p: p[1] <= y_max + eps,
      lambda a, b: (
          a[0] + (y_max - a[1]) / (b[1] - a[1]) * (b[0] - a[0]),
          y_max,
      ),
  )
  return pts


def _polygon_area(polygon: Sequence[Point]) -> float:
  """Unsigned area of a 2-D polygon."""
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
  """Folded points of face regions clipped to `limiting_polygon`.

  Clipping each unfolded face against the rectangles of `limiting_polygon`
  handles sub-flaps of merged chains whose faces span multiple sub-elements
  without internal hinge creases (so no unsplit face centroid falls inside the
  sub-flap's rectangle).

  Args:
    graph_data: Folded-state graph of the crease pattern.
    limiting_polygon: Target region in unfolded coordinates.

  Returns:
    Folded coordinates of the clipped face polygon vertices.
  """
  rects = _limiting_polygon_rectangles(limiting_polygon)
  vertices = graph_data['vertices_coords']
  folded = graph_data['folded_vertices']
  points: list[Point] = []
  for face in graph_data['face_vertices']:
    if any(folded[v] is None for v in face):
      continue
    unfolded_pts = [vertices[v] for v in face]
    folded_pts = [folded[v] for v in face]
    rotation, u_mean, f_mean = _fit_isometry(unfolded_pts, folded_pts)
    for rect in rects:
      x_min = min(rect.x_start, rect.x_start + rect.x_len)
      x_max = max(rect.x_start, rect.x_start + rect.x_len)
      y_min = min(rect.y_start, rect.y_start + rect.y_len)
      y_max = max(rect.y_start, rect.y_start + rect.y_len)
      clipped = _clip_polygon_to_rectangle(
          unfolded_pts, x_min, y_min, x_max, y_max
      )
      if _polygon_area(clipped) <= 1e-12:
        continue
      for ux, uy in clipped:
        dx, dy = ux - u_mean[0], uy - u_mean[1]
        points.append((
            rotation[0][0] * dx + rotation[0][1] * dy + f_mean[0],
            rotation[1][0] * dx + rotation[1][1] * dy + f_mean[1],
        ))
  return points


def check_inside_limiting_polygon(
    point: Point, limiting_polygon: LimitingPolygon
) -> bool:
  """Returns whether `point` (unfolded coordinates) is in the polygon."""
  return _check_inside_limiting_polygon(point, limiting_polygon)


def _signed_side(point: Point, line_start: Point, line_end: Point) -> float:
  """Signed distance of `point` from the line; > 0 on its left."""
  dx, dy = line_end[0] - line_start[0], line_end[1] - line_start[1]
  norm = (dx * dx + dy * dy) ** 0.5
  return (dx * (point[1] - line_start[1]) - dy * (point[0] - line_start[0])) / (
      norm
  )


def _fit_isometry(
    source: Sequence[Point], target: Sequence[Point]
) -> tuple[list[list[float]], Point, Point]:
  """Least-squares 2-D isometry (reflections allowed) mapping source to target.

  Args:
    source: Points in the source frame.
    target: Matching points in the target frame.

  Returns:
    `(rotation, source_mean, target_mean)` such that a source point `p` maps to
    `rotation @ (p - source_mean) + target_mean`.
  """
  n = len(source)
  sx = sum(p[0] for p in source) / n
  sy = sum(p[1] for p in source) / n
  tx = sum(p[0] for p in target) / n
  ty = sum(p[1] for p in target) / n
  # Cross-covariance H = sum (s - s_mean)(t - t_mean)^T.
  h = [[0.0, 0.0], [0.0, 0.0]]
  for (px, py), (qx, qy) in zip(source, target, strict=True):
    a, b = px - sx, py - sy
    c, d = qx - tx, qy - ty
    h[0][0] += a * c
    h[0][1] += a * d
    h[1][0] += b * c
    h[1][1] += b * d
  # Best proper rotation and best reflection; keep whichever fits better.
  best = None
  for sign in (1.0, -1.0):
    # R = [[c, -s * sign], [s, c * sign]] (sign -1: reflection); maximize
    # sum(t^T R s) = c * (h00 + h11') + s * (h01 - h10') with h1j' = sign * h1j.
    h00, h01 = h[0][0], h[0][1]
    h10, h11 = sign * h[1][0], sign * h[1][1]
    angle = math.atan2(h01 - h10, h00 + h11)
    score = (h00 + h11) * math.cos(angle) + (h01 - h10) * math.sin(angle)
    c, s = math.cos(angle), math.sin(angle)
    rotation = [[c, -s * sign], [s, c * sign]]
    if best is None or score > best[0]:
      best = (score, rotation)
  assert best is not None
  return best[1], (sx, sy), (tx, ty)


def get_segment_cut_creases(
    *,
    graph_data: GraphData,
    cut_line_start: Point,
    cut_line_end: Point,
    span: tuple[float, float],
    limiting_polygon: LimitingPolygon,
    fold_assignment: crease_utils.Fold = crease_utils.Fold.VALLEY,
) -> list[crease_utils.Crease]:
  """Like `get_geometric_fold_creases`, for faces cut inside a span of the line.

  Each candidate face chord is clipped to `limiting_polygon` first, and kept if
  the clipped segment's midpoint lies within `span` (signed distances along the
  line from `cut_line_start`, in units of the folded form), so sub-flaps whose
  faces extend outside `limiting_polygon` are cut cleanly while layers of other
  flaps elsewhere on the infinite line are left alone.

  Args:
    graph_data: Folded-state graph of the crease pattern.
    cut_line_start: Start of the cut line (folded coordinates).
    cut_line_end: A second point on the cut line.
    span: `(min, max)` distance along the line where faces are cut.
    limiting_polygon: Only faces inside it (unfolded coordinates) are cut.
    fold_assignment: Assignment for face-up faces (flipped on flipped faces).

  Returns:
    The cut creases, clipped to the limiting polygon.
  """
  vertices = graph_data['vertices_coords']
  folded = graph_data['folded_vertices']
  orientations = graph_data['face_orientations']
  dx = cut_line_end[0] - cut_line_start[0]
  dy = cut_line_end[1] - cut_line_start[1]
  norm = math.hypot(dx, dy)
  ux, uy = dx / norm, dy / norm
  creases = []
  for face_index, face in enumerate(graph_data['face_vertices']):
    if orientations[face_index] is None:
      continue
    points = []  # (unfolded point, distance along the line, vertex or -1)
    for i, v1 in enumerate(face):
      v2 = face[(i + 1) % len(face)]
      p1, p2 = folded[v1], folded[v2]
      if p1 is None or p2 is None:
        continue
      d1 = ux * (p1[1] - cut_line_start[1]) - uy * (p1[0] - cut_line_start[0])
      d2 = ux * (p2[1] - cut_line_start[1]) - uy * (p2[0] - cut_line_start[0])
      if abs(d1) < 1e-9 and abs(d2) < 1e-9:
        continue
      vertex = -1
      if abs(d1) < 1e-9:
        t, vertex = 0.0, v1
      elif abs(d2) < 1e-9:
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
          math_utils.distance(q, unfolded) < point_utils.DEFAULT_TOLERANCE
          for q, _, _ in points
      ):
        points.append((unfolded, along, vertex))
    if len(points) < 2:
      continue
    (q1, s1, w1), (q2, s2, w2) = max(
        itertools.combinations(points, 2),
        key=lambda pair: math_utils.distance(pair[0][0], pair[1][0]),
    )
    if w1 >= 0 and w2 >= 0:
      k1, k2 = face.index(w1), face.index(w2)
      if (k1 - k2) % len(face) in (1, len(face) - 1):
        # The chord is a face edge lying on the cut line (a folded edge that
        # is already creased there): no new crease.
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
    clipped_creases = _restrict_creases_to_limiting_polygon(
        [raw_crease], limiting_polygon
    )
    chord_len_sq = (q2[0] - q1[0]) ** 2 + (q2[1] - q1[1]) ** 2
    if chord_len_sq <= 1e-24:
      continue
    for clipped in clipped_creases:
      mid_x = (clipped.p1.x + clipped.p2.x) * 0.5
      mid_y = (clipped.p1.y + clipped.p2.y) * 0.5
      t_mid = (
          (mid_x - q1[0]) * (q2[0] - q1[0]) + (mid_y - q1[1]) * (q2[1] - q1[1])
      ) / chord_len_sq
      along_mid = s1 + t_mid * (s2 - s1)
      if span[0] <= along_mid <= span[1]:
        creases.append(clipped)
  return creases


class InvertedRegionVariant(TypedDict):
  """One M/V assignment of a reverse/crimp fold.

  Attributes:
    name: Short description of the assignment (e.g. `simple`, `flipped`).
    crease_pattern: The full crease pattern with the fold applied.
    num_new_creases: Number of new (cut) crease segments.
    num_flipped_creases: Number of existing creases flipped inside the region.
    region_faces: Number of faces in the inverted region.
  """

  name: str
  crease_pattern: cp_utils.CreasePattern
  num_new_creases: int
  num_flipped_creases: int
  region_faces: int


def get_inverted_region_variants(
    crease_pattern: cp_utils.CreasePattern,
    *,
    cut_lines: Sequence[tuple[Point, Point]],
    limiting_polygon: LimitingPolygon,
    graph_data: GraphData | None = None,
    fold_percent: float = 1.0,
    spans: Sequence[tuple[float, float] | None] | None = None,
    proximal_polygon: LimitingPolygon | None = None,
    target_polygon: LimitingPolygon | None = None,
) -> list[InvertedRegionVariant]:
  """Reverse fold (one cut line) or crimp (two) by inverting a region.

  The crease positions of a reverse fold are those of a geometric simple fold
  through every layer of the target flap (the distal part is reflected across
  the cut line in the folded state); only the mountain/valley assignment
  differs. The region R enclosed by the cut lines (to the left of every
  directed cut line, in folded coordinates) keeps its layer order instead of
  being turned over, so:

  * every existing crease with both faces in R flips mountain <-> valley;
  * the new creases of one connected cut polyline share one assignment in the
    crease pattern (Maekawa at each cut vertex: the proximal half of the
    crossed crease keeps `a`, the distal half becomes `not a`, so the two new
    segments must be equal).

  With one cut line R is the distal part of the flap (reverse fold); with two
  cut lines (the second distal to the first and directed the other way) R is
  the strip between them (crimp). R is found by a flood fill from the cut
  across the other edges; reaching a face on the outer side of a cut line
  means some layer of the flap was not cut (it lies outside
  `limiting_polygon`), and a ValueError is raised instead of a partial fold.

  Args:
    crease_pattern: The current crease pattern.
    cut_lines: Directed cut lines `(start, end)` in folded coordinates; the
      region to invert lies on the left of each.
    limiting_polygon: Only faces inside it (unfolded coordinates) are cut.
    graph_data: Precomputed `prepare_graph_data(crease_pattern)`.
    fold_percent: Fold percent of the new creases (1 = flat).
    spans: Optional `(min, max)` per cut line: only faces whose cut chord
      midpoint lies within that distance range along the line are cut (see
      `get_segment_cut_creases`); None cuts along the whole line.
    proximal_polygon: Region (unfolded coordinates) that must stay outside the
      inverted region, e.g. the parent element: reaching it means the flap is
      connected to the rest of the model beyond the cut (as for chains packed as
      full-width strips), and a ValueError is raised.
    target_polygon: Faces inside it are never treated as proximal (sub-flap
      rectangles of a chain can lie inside their parent's rectangle).

  Returns:
    The candidate assignments (2-4); the caller checks which one folds.

  Raises:
    ValueError: If a cut line cuts nothing, the cut lines cross inside the
      flap, or the cut does not separate the region from the rest.
  """
  if not cut_lines:
    raise ValueError('At least one cut line is required.')
  if graph_data is None:
    graph_data = prepare_graph_data(crease_pattern)
  tolerance = crease_pattern.tolerance

  # 1. Cut creases per line (positions; assignment = simple fold).
  cut_creases_per_line = []
  for index, (start, end) in enumerate(cut_lines):
    span = spans[index] if spans else None
    if span is None:
      creases = get_geometric_fold_creases(
          crease_pattern=crease_pattern,
          cut_line_start=start,
          cut_line_end=end,
          fold_assignment=crease_utils.Fold.VALLEY,
          limiting_polygon=limiting_polygon,
          graph_data=graph_data,
      )
    else:
      creases = get_segment_cut_creases(
          graph_data=graph_data,
          cut_line_start=start,
          cut_line_end=end,
          span=span,
          limiting_polygon=limiting_polygon,
      )
    if not creases:
      raise ValueError(f'Cut line {index} does not cut the target flap.')
    cut_creases_per_line.append(list(creases))
  all_cut = []
  for creases in cut_creases_per_line:
    all_cut.extend(creases)

  # 2. Split base and cut creases at their intersections.
  tips, _ = cp_utils.get_tips_and_grid_creases(
      [*crease_pattern.creases, *all_cut], grid_size=-1
  )
  base_split = cp_utils.split_creases(
      crease_pattern.creases, all_cut, tips, tolerance=tolerance
  )
  cut_split_per_line = [
      cp_utils.split_creases(
          creases,
          [*crease_pattern.creases, *all_cut],
          tips,
          tolerance=tolerance,
      )
      for creases in cut_creases_per_line
  ]
  creases = list(base_split)
  line_of_crease = [-1] * len(base_split)
  for index, split in enumerate(cut_split_per_line):
    creases.extend(split)
    line_of_crease.extend([index] * len(split))

  # 3. Graph of the split pattern; cut edges are flat for the folded state,
  # which then equals the current one up to an isometry.
  lines = [((c.p1.x, c.p1.y), (c.p2.x, c.p2.y)) for c in creases]
  vertices, edge_vertices, edge_lines = _fast_create_graph(lines)
  if not vertices:
    raise ValueError(
        'The cut leaves faces too small to build the folded graph (try'
        ' another depth).'
    )
  edge_crease = [indices[0] if indices else -1 for indices in edge_lines]
  assignments = []
  for crease_index in edge_crease:
    if crease_index < 0:
      assignments.append('U')
      continue
    if line_of_crease[crease_index] >= 0:
      assignments.append('F')
      continue
    fold = creases[crease_index].fold
    if fold == crease_utils.Fold.MOUNTAIN:
      assignments.append('M')
    elif fold == crease_utils.Fold.VALLEY:
      assignments.append('V')
    elif fold == crease_utils.Fold.BORDER:
      assignments.append('B')
    else:
      assignments.append('U')
  _, face_vertices = graph_topology.graph_to_adjacency_and_faces(
      vertices, edge_vertices
  )
  folded, _ = folded_state.compute_folded_state(
      vertices, face_vertices, edge_vertices, assignments
  )

  # Register the new folded frame onto the current one (shared vertices).
  current = {}
  for point, folded_point in zip(
      graph_data['vertices_coords'], graph_data['folded_vertices']
  ):
    if folded_point is not None:
      current[(round(point[0], 6), round(point[1], 6))] = folded_point
  source, target = [], []
  for point, folded_point in zip(vertices, folded):
    key = (round(point[0], 6), round(point[1], 6))
    if folded_point is not None and key in current:
      source.append(folded_point)
      target.append(current[key])
  if len(source) < 3:
    raise ValueError('Could not register the cut pattern onto the folded form.')
  rotation, source_mean, target_mean = _fit_isometry(source, target)

  def to_current(p: Point) -> Point:
    x, y = p[0] - source_mean[0], p[1] - source_mean[1]
    return (
        rotation[0][0] * x + rotation[0][1] * y + target_mean[0],
        rotation[1][0] * x + rotation[1][1] * y + target_mean[1],
    )

  # 4. Faces, their folded centroids and sides.
  edge_index = {}
  for index, (v1, v2) in enumerate(edge_vertices):
    edge_index[tuple(sorted((v1, v2)))] = index
  edge_faces = collections.defaultdict(list)
  centroids = []
  face_points = []
  for face_index, face in enumerate(face_vertices):
    points = [folded[v] for v in face]
    if any(p is None for p in points):
      centroids.append(None)
      face_points.append([])
    else:
      points = [to_current(p) for p in points]
      face_points.append(points)
      centroids.append((
          sum(p[0] for p in points) / len(points),
          sum(p[1] for p in points) / len(points),
      ))
    for i, v1 in enumerate(face):
      v2 = face[(i + 1) % len(face)]
      key = tuple(sorted((v1, v2)))
      if key in edge_index:
        edge_faces[edge_index[key]].append(face_index)

  def unfolded_centroid(face_index: int) -> Point:
    face = face_vertices[face_index]
    return (
        sum(vertices[v][0] for v in face) / len(face),
        sum(vertices[v][1] for v in face) / len(face),
    )

  def outside(face_index: int) -> bool:
    if proximal_polygon is not None:
      centroid = unfolded_centroid(face_index)
      if _check_inside_limiting_polygon(centroid, proximal_polygon) and not (
          target_polygon is not None
          and _check_inside_limiting_polygon(centroid, target_polygon)
      ):
        return True
    # Any vertex clearly on the outer side of a cut line (a face straddling a
    # line was not cut there).
    for point in face_points[face_index]:
      for start, end in cut_lines:
        if _signed_side(point, start, end) < -_REGION_SIDE_TOLERANCE:
          return True
    return False

  is_cut_edge = [
      crease_index >= 0 and line_of_crease[crease_index] >= 0
      for crease_index in edge_crease
  ]

  # 5. Flood fill the region from the faces on the inner side of the cuts.
  region = set()
  queue = []
  for index, cut in enumerate(is_cut_edge):
    if not cut:
      continue
    for face_index in edge_faces[index]:
      centroid = centroids[face_index]
      if centroid is None or face_index in region:
        continue
      line = cut_lines[line_of_crease[edge_crease[index]]]
      if _signed_side(centroid, line[0], line[1]) > 0 and not outside(
          face_index
      ):
        region.add(face_index)
        queue.append(face_index)
  if not region:
    raise ValueError('No faces found on the inner side of the cut lines.')
  face_edges = collections.defaultdict(list)
  for index, faces in edge_faces.items():
    for face_index in faces:
      face_edges[face_index].append(index)
  while queue:
    face_index = queue.pop()
    for index in face_edges[face_index]:
      if is_cut_edge[index] or assignments[index] == 'B':
        continue
      for neighbor in edge_faces[index]:
        if neighbor in region:
          continue
        if outside(neighbor):
          raise ValueError(
              'The cut does not separate the folded region from the rest of'
              ' the model: some layer of the flap at the cut lies outside the'
              ' target element (try another depth/angle, or a longer element).'
          )
        region.add(neighbor)
        queue.append(neighbor)

  def full_inversion_variants() -> list[InvertedRegionVariant]:
    # 6. Creases to flip: existing creases whose edges are inside the region.
    flip = set()
    for index, faces in edge_faces.items():
      crease_index = edge_crease[index]
      if crease_index < 0 or is_cut_edge[index]:
        continue
      if assignments[index] not in ('M', 'V'):
        continue
      if len(faces) == 2 and faces[0] in region and faces[1] in region:
        flip.add(crease_index)

    # 7. Cut polylines per line. Along a polyline, Maekawa at each cut vertex
    # relates consecutive segments: with S = (M - V) of the other creases there
    # (after the flips), |S + x1 + x2| = 2, so x2 = x1 if S = 0, x2 = -x1 if
    # |S| = 2, and x1 = x2 = -sign(S) if |S| = 4 (x = +1 mountain, -1 valley).
    vertex_edges = collections.defaultdict(list)
    for index, (v1, v2) in enumerate(edge_vertices):
      vertex_edges[v1].append(index)
      vertex_edges[v2].append(index)

    def sign(index: int) -> int:
      crease_index = edge_crease[index]
      value = 1 if assignments[index] == 'M' else -1
      return -value if crease_index in flip else value

    # relation[(c1, c2)] = +1 (equal), -1 (opposite); forced[c] = +1/-1.
    relations = collections.defaultdict(list)
    forced: dict[int, int] = {}
    for vertex, incident in vertex_edges.items():
      cut_here = [e for e in incident if is_cut_edge[e]]
      if not cut_here:
        continue
      others = [e for e in incident if not is_cut_edge[e]]
      if any(assignments[e] == 'B' for e in others):
        continue  # Paper border: no constraint.
      total = sum(sign(e) for e in others if assignments[e] in ('M', 'V'))
      cut_creases = sorted({edge_crease[e] for e in cut_here})
      if len(cut_creases) == 2:
        c1, c2 = cut_creases
        if total == 0:
          relations[c1].append((c2, 1))
          relations[c2].append((c1, 1))
        elif abs(total) == 2:
          relations[c1].append((c2, -1))
          relations[c2].append((c1, -1))
        elif abs(total) == 4:
          forced[c1] = forced[c2] = -1 if total > 0 else 1
        else:
          raise ValueError(
              f'Cut vertex {vertices[vertex]} cannot satisfy Maekawa.'
          )
      elif len(cut_creases) == 1 and abs(total) == 1:
        # The cut ends inside the paper on a crease (odd degree): x = -total.
        forced[cut_creases[0]] = -total

    def to_fold(value: int) -> crease_utils.Fold:
      return (
          crease_utils.Fold.MOUNTAIN if value > 0 else crease_utils.Fold.VALLEY
      )

    def to_value(fold: crease_utils.Fold) -> int:
      return 1 if fold == crease_utils.Fold.MOUNTAIN else -1

    components = []  # (line index, {crease: relative sign}, simple-fold value)
    seen_creases = set()
    for i, line_index in enumerate(line_of_crease):
      if line_index < 0 or i in seen_creases:
        continue
      relative = {i: 1}
      comp_queue = [i]
      seen_creases.add(i)
      while comp_queue:
        cur_crease = comp_queue.pop()
        for other, relation in relations[cur_crease]:
          value = relative[cur_crease] * relation
          if other in relative:
            if relative[other] != value:
              raise ValueError('The cut polyline has inconsistent M/V parity.')
            continue
          relative[other] = value
          seen_creases.add(other)
          comp_queue.append(other)
      components.append((line_index, relative, to_value(creases[i].fold)))

    def component_values(
        start: int, relative: dict[int, int]
    ) -> dict[int, int]:
      return {c: start * r for c, r in relative.items()}

    def respects_forced(values: dict[int, int]) -> bool:
      return all(forced.get(c, v) == v for c, v in values.items())

    # 8. Candidate assignments: each component starts from its simple-fold
    # assignment or the opposite one.
    def build(starts: Sequence[int]) -> cp_utils.CreasePattern | None:
      result = []
      for i, crease in enumerate(creases):
        if line_of_crease[i] >= 0:
          continue
        result.append(crease.flip_fold() if i in flip else crease)
      for (_, relative, _), start in zip(components, starts, strict=True):
        values = component_values(start, relative)
        if not respects_forced(values):
          return None
        for c, value in values.items():
          result.append(
              dataclasses.replace(
                  creases[c], fold=to_fold(value), fold_percent=fold_percent
              )
          )
      return cp_utils.CreasePattern(
          creases=result, light_mode=True, tolerance=tolerance
      )

    simple = [c[2] for c in components]
    flipped_starts = [-v for v in simple]

    # Maekawa leaves two starts per polyline; big-little-big (the crimp check
    # of `vertex_flat_foldability_percentage`) may rule one out. The cut
    # vertices of different polylines are disjoint, so two full builds tell
    # which starts are locally flat-foldable for every polyline at once.
    def point_key(p: point_utils.Point) -> tuple[float, float]:
      return (round(p.x, 6), round(p.y, 6))

    component_points = [
        {point_key(creases[c].p1) for c in relative}
        | {point_key(creases[c].p2) for c in relative}
        for _, relative, _ in components
    ]
    valid: list[set[int]] = [set() for _ in components]
    for starts in (simple, flipped_starts):
      pattern = build(starts)
      if pattern is None:
        continue
      _, _, bad_vertices = pattern.vertex_flat_foldability_percentage()
      bad = {point_key(v.point) for v in bad_vertices}
      for k, points in enumerate(component_points):
        if not points & bad:
          valid[k].add(starts[k])

    def prefer(starts: Sequence[int]) -> list[int]:
      return [
          start if (start in valid[k] or not valid[k]) else -start
          for k, start in enumerate(starts)
      ]

    choices = [('simple', prefer(simple)), ('flipped', prefer(flipped_starts))]
    if len(cut_lines) > 1:
      # Crimp: also the choices with only one of the lines flipped.
      choices.append((
          'simple/flipped',
          prefer([v if c[0] == 0 else -v for c, v in zip(components, simple)]),
      ))
      choices.append((
          'flipped/simple',
          prefer([-v if c[0] == 0 else v for c, v in zip(components, simple)]),
      ))
    built_variants = []
    seen = set()
    for name, starts in choices:
      signature = tuple(starts)
      if signature in seen:
        continue
      seen.add(signature)
      pattern = build(starts)
      if pattern is None:
        continue
      built_variants.append(
          InvertedRegionVariant(
              name=name,
              crease_pattern=pattern,
              num_new_creases=sum(len(c[1]) for c in components),
              num_flipped_creases=len(flip),
              region_faces=len(region),
          )
      )
    return built_variants

  def multi_layer_variants() -> list[InvertedRegionVariant]:
    """Spine-partition and pleat-crimp variants for non-alternating stacks.

    Narrowing (`simple_fold(..., narrow=True)`) creates half-grid pleats whose
    M/V signs repeat in pairs (`MMVV`), so flipping every interior crease in R
    violates big-little-big at one side of each pleat pair. Partitioning R into
    two half-stacks across a single axial spine line (`x = x0` or `y = y0`)
    flips only the spine creases between the two half-stacks while each
    half-stack folds as a coherent packet; for two cut lines (crimp), opposite
    folded-space simple folds (`pleat_mv` / `pleat_vm`, 0 flips) are also
    checked.

    Returns:
      List of inverted region variants valid for multi-layer spines.
    """
    cut_face_in_r: dict[int, int] = {}
    for index, cut in enumerate(is_cut_edge):
      if not cut:
        continue
      crease_index = edge_crease[index]
      for face_index in edge_faces[index]:
        if face_index in region:
          cut_face_in_r[crease_index] = face_index

    u_centroids = {
        face_index: unfolded_centroid(face_index) for face_index in region
    }
    spine_xs = sorted({
        round(creases[edge_crease[idx]].p1.x, 6)
        for idx, faces in edge_faces.items()
        if len(faces) == 2
        and faces[0] in region
        and faces[1] in region
        and edge_crease[idx] >= 0
        and abs(creases[edge_crease[idx]].p1.x - creases[edge_crease[idx]].p2.x)
        < 1e-6
    })
    spine_ys = sorted({
        round(creases[edge_crease[idx]].p1.y, 6)
        for idx, faces in edge_faces.items()
        if len(faces) == 2
        and faces[0] in region
        and faces[1] in region
        and edge_crease[idx] >= 0
        and abs(creases[edge_crease[idx]].p1.y - creases[edge_crease[idx]].p2.y)
        < 1e-6
    })
    partitions: list[tuple[str, dict[int, int]]] = []
    if spine_xs:
      mid_x = 0.5 * (
          min(u_centroids[f][0] for f in region)
          + max(u_centroids[f][0] for f in region)
      )
      for x0 in sorted(spine_xs, key=lambda x: abs(x - mid_x)):
        sigma = {f: 1 if u_centroids[f][0] < x0 else -1 for f in region}
        if len(set(sigma.values())) == 2:
          partitions.append((f'spine_x={x0:.3f}', sigma))
    if spine_ys:
      mid_y = 0.5 * (
          min(u_centroids[f][1] for f in region)
          + max(u_centroids[f][1] for f in region)
      )
      for y0 in sorted(spine_ys, key=lambda y: abs(y - mid_y)):
        sigma = {f: 1 if u_centroids[f][1] < y0 else -1 for f in region}
        if len(set(sigma.values())) == 2:
          partitions.append((f'spine_y={y0:.3f}', sigma))

    candidates: list[InvertedRegionVariant] = []
    for part_name, sigma in partitions:
      found_for_spine = False
      for global_sign in (1, -1):
        flip_set = {
            edge_crease[idx]
            for idx, faces in edge_faces.items()
            if edge_crease[idx] >= 0
            and not is_cut_edge[idx]
            and assignments[idx] in ('M', 'V')
            and len(faces) == 2
            and faces[0] in region
            and faces[1] in region
            and sigma[faces[0]] != sigma[faces[1]]
        }
        result = []
        num_new = 0
        for i, crease in enumerate(creases):
          if line_of_crease[i] >= 0:
            face_index = cut_face_in_r.get(i)
            if face_index is None:
              continue
            fold_val = (
                crease.fold
                if global_sign * sigma[face_index] > 0
                else crease.fold.flip()
            )
            result.append(
                dataclasses.replace(
                    crease, fold=fold_val, fold_percent=fold_percent
                )
            )
            num_new += 1
          else:
            result.append(crease.flip_fold() if i in flip_set else crease)
        pattern = cp_utils.CreasePattern(
            creases=result, light_mode=True, tolerance=tolerance
        )
        pct, _, _ = pattern.vertex_flat_foldability_percentage()
        if math.isclose(pct, 1.0):
          suffix = '+' if global_sign > 0 else '-'
          candidates.append(
              InvertedRegionVariant(
                  name=f'{part_name}{suffix}',
                  crease_pattern=pattern,
                  num_new_creases=num_new,
                  num_flipped_creases=len(flip_set),
                  region_faces=len(region),
              )
          )
          found_for_spine = True
      if found_for_spine:
        break

    if len(cut_lines) == 2:
      c0, c1 = cut_creases_per_line[0], cut_creases_per_line[1]
      for pleat_name, flip0, flip1 in (
          ('pleat_mv', True, False),
          ('pleat_vm', False, True),
      ):
        creases0 = [
            dataclasses.replace(
                c.flip_fold() if flip0 else c, fold_percent=fold_percent
            )
            for c in c0
        ]
        creases1 = [
            dataclasses.replace(
                c.flip_fold() if flip1 else c, fold_percent=fold_percent
            )
            for c in c1
        ]
        pattern = crease_pattern.update_with_shaping_creases(
            [*creases0, *creases1]
        )
        pct, _, _ = pattern.vertex_flat_foldability_percentage()
        if math.isclose(pct, 1.0):
          candidates.append(
              InvertedRegionVariant(
                  name=pleat_name,
                  crease_pattern=pattern,
                  num_new_creases=len(creases0) + len(creases1),
                  num_flipped_creases=0,
                  region_faces=len(region),
              )
          )
    return candidates

  full_error: ValueError | None = None
  variants: list[InvertedRegionVariant] = []
  try:
    variants = full_inversion_variants()
  except ValueError as error:
    full_error = error
  if any(
      math.isclose(
          v['crease_pattern'].vertex_flat_foldability_percentage()[0], 1.0
      )
      for v in variants
  ):
    return variants
  fallback_variants = multi_layer_variants()
  if fallback_variants:
    return fallback_variants
  if full_error is not None:
    raise full_error
  return variants
