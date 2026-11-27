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

"""Utility functions for geometric shaping."""

from typing import Sequence

from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import point as point_utils
from bp_pipeline.shaping import geometric_simple_fold
from bp_pipeline.third_party.flat_folder import folded_state
from bp_pipeline.third_party.flat_folder import math_utils
import numpy as np


def _sort_points_ccw(points: Sequence[tuple[float, float]]) -> np.ndarray:
  """Sorts a list of 2D points in counter-clockwise order."""
  pts = np.array(points)
  centroid = np.mean(pts, axis=0)

  def get_angle(p):
    return np.arctan2(p[1] - centroid[1], p[0] - centroid[0])

  sorted_points = sorted(pts, key=get_angle)
  return np.array(sorted_points)


def _point_in_polygon_2d(
    px: float,
    py: float,
    poly: np.ndarray,
    tol: float,
) -> bool:
  """Checks whether `(px, py)` lies inside or within `tol` of `poly`."""
  n = len(poly)
  if n < 3:
    return False
  inside = False
  tol_sq = max(tol, 0.0) ** 2
  x1, y1 = float(poly[-1, 0]), float(poly[-1, 1])
  for i in range(n):
    x2, y2 = float(poly[i, 0]), float(poly[i, 1])
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq > 0.0:
      t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
      dist_sq = (px - (x1 + t * dx)) ** 2 + (py - (y1 + t * dy)) ** 2
      if dist_sq <= tol_sq:
        return True
    if (y1 > py) != (y2 > py):
      x_intersect = x1 + (py - y1) * dx / (y2 - y1)
      if px < x_intersect:
        inside = not inside
    x1, y1 = x2, y2
  return inside


def _check_point_in_face(
    point: np.ndarray,
    face_poly_2d: np.ndarray,
    tol: float = crease_utils.DEFAULT_TOLERANCE,
) -> bool:
  """Checks if the point is in the face."""
  ccw_corners = _sort_points_ccw(face_poly_2d)
  if _point_in_polygon_2d(float(point[0]), float(point[1]), ccw_corners, tol):
    return True
  border_edges = []
  for i in range(len(ccw_corners)):
    corner_p1 = point_utils.Point(
        x=ccw_corners[i][0],
        y=ccw_corners[i][1],
        require_on_paper=False,
    )
    corner_p2 = point_utils.Point(
        x=ccw_corners[(i + 1) % len(ccw_corners)][0],
        y=ccw_corners[(i + 1) % len(ccw_corners)][1],
        require_on_paper=False,
    )
    if corner_p1.isclose(corner_p2):
      continue
    border_edges.append(
        crease_utils.Crease(
            corner_p1,
            corner_p2,
            fold=crease_utils.Fold.BORDER,
        )
    )
  pt = point_utils.Point(x=point[0], y=point[1], require_on_paper=False)
  return any(
      crease.point_status(pt) != crease_utils.PointStatus.EXTERNAL
      for crease in border_edges
  )


def _signed_triangle_area_2d(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
  return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))


def _barycentric_coords_triangle(
    *,
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> tuple[float, float, float] | None:
  """Computes the barycentric coordinates of a point in a triangle."""
  area = _signed_triangle_area_2d(a, b, c)
  if abs(area) < 1e-12:
    return None
  u = _signed_triangle_area_2d(p, b, c) / area
  v = _signed_triangle_area_2d(a, p, c) / area
  w = 1.0 - u - v
  return u, v, w


def _interpolate_in_face(
    point_tuple: tuple[float, float],
    face_indices: list[int],
    unfolded_coords: list[tuple[float, float]],
    folded_coords: list[folded_state.Point],
) -> folded_state.Point | None:
  """Interpolates the folded position of a point located within a face."""
  if len(face_indices) < 3:
    return None
  a_idx = face_indices[0]
  a_unfolded = unfolded_coords[a_idx]
  a_folded = folded_coords[a_idx]
  for i in range(1, len(face_indices) - 1):
    b_idx = face_indices[i]
    c_idx = face_indices[i + 1]
    b_unfolded = unfolded_coords[b_idx]
    c_unfolded = unfolded_coords[c_idx]
    bary = _barycentric_coords_triangle(
        p=point_tuple, a=a_unfolded, b=b_unfolded, c=c_unfolded
    )
    if bary is None:
      continue
    u, v, w = bary
    if (
        u >= -point_utils.DEFAULT_TOLERANCE
        and v >= -point_utils.DEFAULT_TOLERANCE
        and w >= -point_utils.DEFAULT_TOLERANCE
    ):
      b_folded = folded_coords[b_idx]
      c_folded = folded_coords[c_idx]
      return (
          u * a_folded[0] + v * b_folded[0] + w * c_folded[0],
          u * a_folded[1] + v * b_folded[1] + w * c_folded[1],
      )
  return None


def get_folded_pos(
    *,
    point: point_utils.Point,
    unfolded_coords: list[tuple[float, float]],
    folded_coords: list[folded_state.Point],
    edge_vertices: list[list[int]],
    face_vertices: list[list[int]] | None = None,
) -> folded_state.Point:
  """Gets the folded position of a point in the crease pattern."""
  point_tuple = (point.x, point.y)

  min_distance_to_vertex = float('inf')
  best_vertex_index = -1
  for i, coord in enumerate(unfolded_coords):
    distance = math_utils.distance(coord, point_tuple)
    if distance < min_distance_to_vertex:
      min_distance_to_vertex = distance
      best_vertex_index = i

  if min_distance_to_vertex < point_utils.DEFAULT_TOLERANCE:
    return folded_coords[best_vertex_index]

  for u, v in edge_vertices:
    point1 = unfolded_coords[u]
    point2 = unfolded_coords[v]

    segment_distance = math_utils.distance(point1, point2)
    if segment_distance < point_utils.DEFAULT_TOLERANCE:
      continue

    distance1 = math_utils.distance(point1, point_tuple)
    distance2 = math_utils.distance(point2, point_tuple)

    if (
        abs(distance1 + distance2 - segment_distance)
        < point_utils.DEFAULT_TOLERANCE
    ):
      interpolation_factor = distance1 / segment_distance
      folded_coord_u = folded_coords[u]
      folded_coord_v = folded_coords[v]
      return geometric_simple_fold.interpolate(
          point1=folded_coord_u, point2=folded_coord_v, t=interpolation_factor
      )

  if face_vertices is not None:
    point_2d = np.array(point_tuple)
    for face_indices in face_vertices:
      face_poly_2d = np.array([unfolded_coords[i] for i in face_indices])
      if _check_point_in_face(point_2d, face_poly_2d):
        result = _interpolate_in_face(
            point_tuple, face_indices, unfolded_coords, folded_coords
        )
        if result is not None:
          return result

  raise ValueError(f'Point {point} is not on any vertex, edge, or face.')
