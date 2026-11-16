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

"""Functions useful to manipulate lines during crease pattern drawing.

For example, cropping lines to polygons.
"""

from matplotlib import path
import numpy as np
from scipy import spatial


def is_point_on_segment(
    point: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
    atol: float = 1e-9,
) -> bool:
  """Checks if a point lies on a line segment.

  This check is based on vector cross and dot products for robustness.

  Args:
    point: The point to check.
    seg_start: The starting point of the segment.
    seg_end: The ending point of the segment.
    atol: The tolerance for floating point comparisons.

  Returns:
    True if the point lies on the segment, False otherwise.
  """
  vec_sp = point - seg_start
  vec_se = seg_end - seg_start
  cross_product_2d = vec_sp[0] * vec_se[1] - vec_sp[1] * vec_se[0]
  if abs(cross_product_2d) > atol:
    return False
  dot_product = np.dot(vec_sp, vec_se)
  if dot_product < -atol:
    return False
  squared_length_se = np.dot(vec_se, vec_se)
  if dot_product > squared_length_se + atol:
    return False

  return True


def cross_2d(v1: np.ndarray, v2: np.ndarray) -> float:
  """Computes the 2D cross product of two vectors."""
  return v1[0] * v2[1] - v1[1] * v2[0]


def _get_segment_intersection(
    line1: np.ndarray, line2: np.ndarray, atol: float = 1e-9
) -> np.ndarray | None:
  """Finds the intersection of two line segments, if it exists.

  Args:
    line1: A 2x2 NumPy array representing the first line segment.
    line2: A 2x2 NumPy array representing the second line segment.
    atol: The tolerance for floating point comparisons.

  Returns:
    The intersection point as a NumPy array, or None if there is no
    intersection or if lines are parallel.
  """

  p1, p2 = line1
  p3, p4 = line2

  v1 = p2 - p1
  v2 = p4 - p3

  denom = cross_2d(v1, v2)

  if abs(denom) < atol:
    return None

  p3_minus_p1 = p3 - p1

  t = cross_2d(p3_minus_p1, v2) / denom
  u = cross_2d(p3_minus_p1, v1) / denom

  if (-atol <= t <= 1 + atol) and (-atol <= u <= 1 + atol):
    return p1 + t * v1

  return None


def clip_line_to_polygon_rich(
    line: np.ndarray, polygon: np.ndarray, atol: float = 1e-9
) -> dict[str, list[np.ndarray]]:
  """Analyzes a line segment against a polygon.

  Classifies every portion of the original line as inside, outside, or on the
  border of the polygon.

  Args:
    line: A 2x2 NumPy array representing the line segment, e.g., np.array([[x1,
      y1], [x2, y2]]).
    polygon: An Nx2 NumPy array of vertices defining the polygon, ordered
      clockwise or counter-clockwise.
    atol: The tolerance for floating point comparisons.

  Returns:
    A dictionary with three keys: "inside", "outside", "on_border".
    Each key holds a list of line segments (2x2 NumPy arrays)
    corresponding to that classification.
  """
  if len(polygon) < 3:
    return {"inside": [], "outside": [line], "on_border": []}

  line_start, line_end = line
  polygon_path = path.Path(polygon)

  event_points = {tuple(line_start), tuple(line_end)}
  polygon_edges = np.roll(polygon, -1, axis=0)
  for i, polygon_node in enumerate(polygon):
    poly_edge = np.array([polygon_node, polygon_edges[i]])
    intersection = _get_segment_intersection(line, poly_edge, atol=atol)
    if intersection is not None:
      event_points.add(tuple(intersection))
    for vertex in poly_edge:
      if is_point_on_segment(vertex, line_start, line_end, atol=atol):
        event_points.add(tuple(vertex))
  sorted_points = sorted(
      [np.array(p) for p in event_points],
      key=lambda p: np.linalg.norm(p - line_start),
  )
  unique_points = []
  if sorted_points:
    unique_points.append(sorted_points[0])
    for i in range(1, len(sorted_points)):
      if np.linalg.norm(sorted_points[i] - sorted_points[i - 1]) > atol:
        unique_points.append(sorted_points[i])
  if len(unique_points) < 2:
    is_on_border = any(
        is_point_on_segment(
            line_start, polygon_node, polygon_edges[i], atol=atol
        )
        for i, polygon_node in enumerate(polygon)
    )
    if is_on_border:
      return {"inside": [], "outside": [], "on_border": [line]}
    elif polygon_path.contains_point(line_start):
      return {"inside": [line], "outside": [], "on_border": []}
    else:
      return {"inside": [], "outside": [line], "on_border": []}
  elementary_segments = [
      np.array([unique_points[i], unique_points[i + 1]])
      for i in range(len(unique_points) - 1)
  ]
  classified_segments = []
  for seg in elementary_segments:
    # Segments don't intersect polygon edges, so we can just take the midpoint.
    midpoint = np.mean(seg, axis=0)
    is_on_border = any(
        is_point_on_segment(midpoint, polygon_node, polygon_edges[i], atol=atol)
        for i, polygon_node in enumerate(polygon)
    )
    if is_on_border:
      classification = "on_border"
    elif polygon_path.contains_point(midpoint):
      classification = "inside"
    else:
      classification = "outside"
    classified_segments.append({"segment": seg, "type": classification})

  if not classified_segments:
    return {"inside": [], "outside": [], "on_border": []}

  merged_results = {"inside": [], "outside": [], "on_border": []}
  current_type = classified_segments[0]["type"]
  merged_start_point = classified_segments[0]["segment"][0]

  for i in range(1, len(classified_segments)):
    next_type = classified_segments[i]["type"]
    if next_type != current_type:
      merged_end_point = classified_segments[i - 1]["segment"][1]
      merged_segment = np.array([merged_start_point, merged_end_point])
      merged_results[current_type].append(merged_segment)
      current_type = next_type
      merged_start_point = classified_segments[i]["segment"][0]

  final_end_point = classified_segments[-1]["segment"][1]
  final_segment = np.array([merged_start_point, final_end_point])
  merged_results[current_type].append(final_segment)

  return merged_results


def clip_lines_to_polygon(
    lines: np.ndarray,
    polygon: np.ndarray,
    atol: float = 1e-9,
    fast: bool = False,
) -> list[tuple[int, np.ndarray]]:
  """Clips multiple line segments to a polygon, returning only inside portions.

  Args:
    lines: An Mx2x2 NumPy array of M line segments, each defined by two 2D
      endpoints, i.e., lines[i] = [[x1, y1], [x2, y2]].
    polygon: An Nx2 NumPy array of vertices defining the polygon, ordered
      clockwise or counter-clockwise.
    atol: The tolerance for floating point comparisons.
    fast: If True, uses an optimized batch implementation that creates the Path
      and polygon edges once for all lines and uses squared distances to avoid
      np.linalg.norm overhead. If False (default), uses the reference
      implementation that delegates per-line to clip_line_to_polygon_rich.

  Returns:
    A list of (line_index, segment) tuples where line_index is the index into
    the input lines array and segment is a 2x2 NumPy array [[x1,y1],[x2,y2]]
    representing an inside portion of that line.
  """
  if fast:
    return _clip_lines_to_polygon_fast(lines, polygon, atol)
  return _clip_lines_to_polygon_reference(lines, polygon, atol)


def _clip_lines_to_polygon_reference(
    lines: np.ndarray,
    polygon: np.ndarray,
    atol: float = 1e-9,
) -> list[tuple[int, np.ndarray]]:
  """Reference implementation: clips lines by calling clip_line_to_polygon_rich.

  Args:
    lines: An Mx2x2 NumPy array of M line segments.
    polygon: An Nx2 NumPy array of polygon vertices.
    atol: The tolerance for floating point comparisons.

  Returns:
    A list of (line_index, segment) tuples for inside portions.
  """
  if len(polygon) < 3 or len(lines) == 0:
    return []

  result = []
  for line_idx, line in enumerate(lines):
    rich = clip_line_to_polygon_rich(line, polygon, atol=atol)
    for segment in rich["inside"]:
      result.append((line_idx, segment))
  return result


def _clip_lines_to_polygon_fast(
    lines: np.ndarray,
    polygon: np.ndarray,
    atol: float = 1e-9,
) -> list[tuple[int, np.ndarray]]:
  """Optimized batch implementation of clip_lines_to_polygon.

  This version:
    - Creates the Path and polygon edges once for all lines.
    - Skips outside/on_border classification.
    - Uses squared distances to avoid np.linalg.norm overhead.

  Args:
    lines: An Mx2x2 NumPy array of M line segments.
    polygon: An Nx2 NumPy array of polygon vertices.
    atol: The tolerance for floating point comparisons.

  Returns:
    A list of (line_index, segment) tuples for inside portions.
  """
  if len(polygon) < 3 or len(lines) == 0:
    return []

  polygon_path = path.Path(polygon)
  polygon_next = np.roll(polygon, -1, axis=0)
  num_edges = len(polygon)
  atol_sq = atol * atol

  result = []
  for line_idx in range(len(lines)):
    line = lines[line_idx]
    line_start, line_end = line[0], line[1]

    # Collect event points (as tuples for dedup via set).
    event_points = {(line_start[0], line_start[1]),
                    (line_end[0], line_end[1])}

    for i in range(num_edges):
      poly_edge = np.array([polygon[i], polygon_next[i]])
      intersection = _get_segment_intersection(line, poly_edge, atol=atol)
      if intersection is not None:
        event_points.add((intersection[0], intersection[1]))
      for vertex in poly_edge:
        if is_point_on_segment(vertex, line_start, line_end, atol=atol):
          event_points.add((vertex[0], vertex[1]))

    # Sort by squared distance from line_start (avoids sqrt).
    pts = [np.array(p) for p in event_points]
    start = line_start
    pts.sort(
        key=lambda p, s=start: (p[0] - s[0]) ** 2 + (p[1] - s[1]) ** 2
    )

    # Deduplicate close points using squared distance.
    unique = [pts[0]]
    for i in range(1, len(pts)):
      d = pts[i] - unique[-1]
      if d[0]*d[0] + d[1]*d[1] > atol_sq:
        unique.append(pts[i])

    if len(unique) < 2:
      # Single point — check if inside (not on border).
      if polygon_path.contains_point(line_start):
        result.append((line_idx, line))
      continue

    # Classify segments and merge consecutive "inside" runs.
    run_start = None
    prev_inside = False
    for i in range(len(unique) - 1):
      midpoint = (unique[i] + unique[i + 1]) * 0.5
      is_inside = polygon_path.contains_point(midpoint)
      if is_inside:
        if not prev_inside:
          run_start = unique[i]
        prev_inside = True
      else:
        if prev_inside:
          result.append(
              (line_idx, np.array([run_start, unique[i]])))
        prev_inside = False
    if prev_inside:
      result.append(
          (line_idx, np.array([run_start, unique[-1]])))

  return result


def convex_hull(points: np.ndarray) -> np.ndarray:
  """Computes the convex hull of 2D points using the Monotone Chain algorithm.

  Args:
      points (np.ndarray): An array of shape (N, 2) representing N points.

  Returns:
      np.ndarray: An array of shape (M, 2) representing the M points
                  that form the convex hull, ordered along the boundary.
  """
  if len(points) < 3:
    sorted_indices = np.lexsort((points[:, 1], points[:, 0]))
    return points[sorted_indices]
  hull = spatial.ConvexHull(points)
  return points[hull.vertices]
