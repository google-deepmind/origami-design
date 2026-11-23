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

"""Folded-state reverse and crimp folds by planar region inversion.

Provides folded-footprint axis measurement, directed cut-line placement, and
mountain/valley assignment solvers that invert the planar region enclosed by
one cut line (reverse fold) or two cut lines (crimp fold) while preserving
local vertex flat-foldability.
"""

import collections
from collections.abc import Sequence
import dataclasses
import math
from typing import TypedDict

import numpy as np

from .. import timeout as timeout_lib
from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..third_party.flat_folder import folded_state
from ..third_party.flat_folder import graph_topology
from . import geometric_simple_fold

Point = geometric_simple_fold.Point
type CutLine = tuple[Point, Point]

_REGION_SIDE_TOL = 1e-7
_VERTEX_MATCH_TOL = 1e-5
_AXIAL_EDGE_TOL = 1e-6
_MIN_NORM = 1e-12


@dataclasses.dataclass(frozen=True)
class FoldedAxis:
  """An element's oriented centerline axis in the flat-folded 2D state.

  Attributes:
    base: Folded position of the element's proximal end along its centerline.
    tip: Folded position of the element's distal end along its centerline.
    width: Extent of the element's folded footprint perpendicular to the axis.
  """

  base: np.ndarray
  tip: np.ndarray
  width: float

  @property
  def length(self) -> float:
    """Euclidean distance from `base` to `tip`."""
    return float(np.linalg.norm(self.tip - self.base))

  @property
  def direction(self) -> np.ndarray:
    """Unit 2D direction vector from `base` to `tip`."""
    return (self.tip - self.base) / max(self.length, _MIN_NORM)


class InvertedRegionVariant(TypedDict):
  """One candidate mountain/valley assignment for a reverse or crimp fold.

  Attributes:
    name: Short identifier of the assignment (e.g. `'simple'`, `'flipped'`,
      `'spine_x=0.500+'`, `'pleat_mv'`).
    crease_pattern: Full crease pattern with the fold applied.
    num_new_creases: Number of newly inserted cut crease segments.
    num_flipped_creases: Number of existing creases flipped inside the region.
    region_faces: Number of planar faces inside the inverted region.
    is_flat_foldable: Whether every interior vertex satisfies local
      flat-foldability (Maekawa, Kawasaki, and crimp conditions).
  """

  name: str
  crease_pattern: cp_utils.CreasePattern
  num_new_creases: int
  num_flipped_creases: int
  region_faces: int
  is_flat_foldable: bool


def element_folded_points(
    graph_data: geometric_simple_fold.GraphData,
    limiting_polygon: geometric_simple_fold.LimitingPolygon,
) -> np.ndarray:
  """Returns an `(N, 2)` array of folded points inside `limiting_polygon`."""
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
  """Computes an element's folded axis oriented away from `reference_point`.

  Uses the principal axis of the element's folded footprint when its aspect
  ratio is at least `min_aspect`; falls back to the direction from
  `reference_point` to the footprint centroid for near-square footprints.

  Args:
    graph_data: Folded-state graph of the current crease pattern.
    limiting_polygon: Unfolded region of the target element.
    reference_point: Proximal folded 2D point (e.g. parent footprint center).
    min_aspect: Minimum footprint aspect ratio required to trust PCA direction.

  Returns:
    The oriented `FoldedAxis`.

  Raises:
    ValueError: If the element has fewer than 3 folded points or coincides with
      `reference_point` when near-square.
  """
  points = element_folded_points(graph_data, limiting_polygon)
  if points.shape[0] < 3:
    raise ValueError('The element has no faces in the folded state.')
  reference = np.asarray(reference_point, dtype=float)
  mean = points.mean(axis=0)
  centered = points - mean
  eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
  axis = eigenvectors[:, 1]
  extent = centered @ axis
  across = centered @ np.array([-axis[1], axis[0]])
  aspect = float(extent.max() - extent.min()) / max(
      float(across.max() - across.min()), _MIN_NORM
  )
  if aspect < min_aspect or eigenvalues[1] <= 0:
    delta = mean - reference
    norm = float(np.linalg.norm(delta))
    if norm < _MIN_NORM:
      raise ValueError('Cannot orient the element axis.')
    axis = delta / norm
  if float(np.dot(mean - reference, axis)) < 0:
    axis = -axis
  normal = np.array([-axis[1], axis[0]])
  extent = centered @ axis
  across = centered @ normal
  center = mean + normal * float(across.max() + across.min()) * 0.5
  return FoldedAxis(
      base=center + axis * float(extent.min()),
      tip=center + axis * float(extent.max()),
      width=float(across.max() - across.min()),
  )


def _rotate_2d(vector: np.ndarray, angle_deg: float) -> np.ndarray:
  """Rotates a 2D vector counter-clockwise by `angle_deg` degrees."""
  rad = math.radians(angle_deg)
  cos_a, sin_a = math.cos(rad), math.sin(rad)
  return np.array([
      cos_a * vector[0] - sin_a * vector[1],
      sin_a * vector[0] + cos_a * vector[1],
  ])


def _directed_line(
    point: np.ndarray, direction: np.ndarray, inner_point: np.ndarray
) -> CutLine:
  """Orients the line through `point` so `inner_point` lies on its left."""
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
  """Snaps `point` to the nearest entry of `points` within `max_distance`."""
  if points is None or points.size == 0:
    return point
  distances = np.linalg.norm(points - point, axis=1)
  nearest = int(np.argmin(distances))
  if float(distances[nearest]) <= max_distance:
    return points[nearest]
  return point


def line_span(
    line: CutLine, points: np.ndarray, margin: float
) -> tuple[float, float]:
  """Returns the signed projection extent of `points` along `line` +/- pad."""
  if points.size == 0:
    raise ValueError('points must be non-empty.')
  start = np.asarray(line[0], dtype=float)
  delta = np.asarray(line[1], dtype=float) - start
  norm = float(np.linalg.norm(delta))
  if norm < _MIN_NORM:
    raise ValueError('Cut line endpoints must be distinct.')
  along = (points - start) @ (delta / norm)
  return float(along.min() - margin), float(along.max() + margin)


def reverse_fold_cut_line(
    axis: FoldedAxis,
    *,
    angle: float,
    depth: float,
    snap_points: np.ndarray | None = None,
) -> CutLine:
  """Returns the directed cut line for a flat reverse fold turning by `angle`.

  Args:
    axis: Oriented folded axis of the target element.
    angle: Counter-clockwise turn angle in degrees (`0 < |angle| <= 180`).
    depth: Fractional position along `axis` (`0` = base, `1` = tip).
    snap_points: Optional folded vertices to snap the cut anchor toward.

  Returns:
    Directed cut line `(start, end)` with the distal region on its left.

  Raises:
    ValueError: If `angle` lies outside `[-180, 0) | (0, 180]`.
  """
  if not 0 < abs(angle) <= 180:
    raise ValueError(f'angle must be in [-180, 0) or (0, 180]; got {angle}.')
  anchor = snap_to_points(
      axis.base + depth * (axis.tip - axis.base),
      snap_points,
      0.25 * min(axis.width, axis.length),
  )
  return _directed_line(anchor, _rotate_2d(axis.direction, angle / 2), axis.tip)


def crimp_fold_cut_lines(
    axis: FoldedAxis,
    *,
    angle: float,
    depth: float,
    gap: float | None = None,
    margin: float = 1.25,
    snap_points: np.ndarray | None = None,
) -> list[CutLine]:
  """Returns `[first, second]` directed cut lines for a flat crimp fold.

  Tilts two cut lines by `±angle / 4` from the normal to `axis` so composing
  their reflections rotates the distal portion by `angle` degrees while keeping
  the strip between them on the left of both directed lines.

  Args:
    axis: Oriented folded axis of the target element.
    angle: Counter-clockwise turn angle in degrees (`0 < |angle| < 180`).
    depth: Fractional position along `axis` where the first line crosses.
    gap: Centerline spacing between the two lines in folded units.
    margin: Meeting-point distance in half-widths when `gap` is `None` (floored
      at 15% of `axis.length`).
    snap_points: Optional folded vertices to snap the first anchor toward.

  Returns:
    Two directed cut lines `[first, second]`.

  Raises:
    ValueError: If `angle` lies outside `(-180, 0) | (0, 180)`.
  """
  if not 0 < abs(angle) < 180:
    raise ValueError(f'angle must be in (-180, 0) or (0, 180); got {angle}.')
  if gap is None:
    gap = max(
        margin * axis.width * math.tan(math.radians(abs(angle) / 4)),
        0.15 * axis.length,
    )
  first_pt = snap_to_points(
      axis.base + depth * (axis.tip - axis.base),
      snap_points,
      0.25 * min(axis.width, axis.length),
  )
  second_pt = first_pt + gap * axis.direction
  normal = _rotate_2d(axis.direction, 90.0)
  first = _directed_line(first_pt, _rotate_2d(normal, angle / 4), second_pt)
  second = _directed_line(second_pt, _rotate_2d(normal, -angle / 4), first_pt)
  return [first, second]


def _signed_side(point: Point, line_start: Point, line_end: Point) -> float:
  """Signed distance of `point` from `(line_start, line_end)` (> 0 on left)."""
  dx, dy = line_end[0] - line_start[0], line_end[1] - line_start[1]
  norm = math.hypot(dx, dy)
  if norm < _MIN_NORM:
    raise ValueError('Cut line endpoints must be distinct.')
  return (
      dx * (point[1] - line_start[1]) - dy * (point[0] - line_start[0])
  ) / norm


class _PointLookup:
  """Tolerance-safe 2D spatial hash grid avoiding decimal rounding cliffs."""

  def __init__(self, tolerance: float = _VERTEX_MATCH_TOL) -> None:
    self._tol = tolerance
    self._inv = 1.0 / tolerance
    self._entries: dict[tuple[int, int], list[tuple[Point, int]]] = (
        collections.defaultdict(list)
    )

  def add(self, point: Point, index: int) -> None:
    key = (math.floor(point[0] * self._inv), math.floor(point[1] * self._inv))
    self._entries[key].append((point, index))

  def find(self, point: Point) -> int | None:
    """Returns index of the nearest point within tolerance, or `None`."""
    cx = math.floor(point[0] * self._inv)
    cy = math.floor(point[1] * self._inv)
    best_idx: int | None = None
    best_dist = self._tol
    for gx in (cx - 1, cx, cx + 1):
      for gy in (cy - 1, cy, cy + 1):
        for candidate, idx in self._entries.get((gx, gy), ()):
          dist = math.hypot(point[0] - candidate[0], point[1] - candidate[1])
          if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


@dataclasses.dataclass(frozen=True)
class _CutGraph:
  """Subdivided planar graph with cut creases registered to the base fold."""

  creases: list[crease_utils.Crease]
  cut_creases_per_line: list[list[crease_utils.Crease]]
  line_of_crease: list[int]
  vertices: list[Point]
  edge_vertices: list[geometric_simple_fold.Edge]
  edge_crease: list[int]
  assignments: list[str]
  is_cut_edge: list[bool]
  face_vertices: list[list[int]]
  edge_faces: dict[int, list[int]]
  face_points: list[list[Point]]
  centroids: list[Point | None]

  def unfolded_centroid(self, face_index: int) -> Point:
    face = self.face_vertices[face_index]
    n = len(face)
    return (
        sum(self.vertices[v][0] for v in face) / n,
        sum(self.vertices[v][1] for v in face) / n,
    )


def _is_cp_flat_foldable(pattern: cp_utils.CreasePattern) -> bool:
  """Returns whether every interior vertex of `pattern` is flat-foldable."""
  pct, _, _ = pattern.vertex_flat_foldability_percentage()
  return math.isclose(pct, 1.0)


def _build_registered_cut_graph(
    crease_pattern: cp_utils.CreasePattern,
    cut_lines: Sequence[CutLine],
    limiting_polygon: geometric_simple_fold.LimitingPolygon,
    graph_data: geometric_simple_fold.GraphData,
    spans: Sequence[tuple[float, float] | None] | None,
) -> _CutGraph:
  """Cuts the pattern, subdivides intersections, and aligns folded frames."""
  cut_creases_per_line: list[list[crease_utils.Crease]] = []
  all_cut: list[crease_utils.Crease] = []
  for idx, (start, end) in enumerate(cut_lines):
    timeout_lib.maybe_timeout_cooperatively()
    span = spans[idx] if spans is not None else None
    if span is None:
      line_creases = list(
          geometric_simple_fold.get_geometric_fold_creases(
              crease_pattern=crease_pattern,
              cut_line_start=start,
              cut_line_end=end,
              fold_assignment=crease_utils.Fold.VALLEY,
              limiting_polygon=limiting_polygon,
              graph_data=graph_data,
          )
      )
    else:
      line_creases = geometric_simple_fold.get_segment_cut_creases(
          graph_data=graph_data,
          cut_line_start=start,
          cut_line_end=end,
          span=span,
          limiting_polygon=limiting_polygon,
      )
    if not line_creases:
      raise ValueError(f'Cut line {idx} does not cut the target flap.')
    cut_creases_per_line.append(line_creases)
    all_cut.extend(line_creases)

  tolerance = crease_pattern.tolerance
  combined = [*crease_pattern.creases, *all_cut]
  tips, _ = cp_utils.get_tips_and_grid_creases(combined, grid_size=-1)
  base_split = cp_utils.split_creases(
      crease_pattern.creases, all_cut, tips, tolerance=tolerance
  )
  creases = list(base_split)
  line_of_crease = [-1] * len(base_split)
  for idx, line_creases in enumerate(cut_creases_per_line):
    split = cp_utils.split_creases(
        line_creases, combined, tips, tolerance=tolerance
    )
    creases.extend(split)
    line_of_crease.extend([idx] * len(split))

  lines = [((c.p1.x, c.p1.y), (c.p2.x, c.p2.y)) for c in creases]
  vertices, edge_vertices, edge_lines = geometric_simple_fold.fast_create_graph(
      lines
  )
  if not vertices:
    raise ValueError(
        'The cut leaves faces too small to build the folded graph (try another'
        ' depth).'
    )

  edge_crease = [indices[0] if indices else -1 for indices in edge_lines]
  assignments: list[str] = []
  for crease_idx in edge_crease:
    if crease_idx < 0:
      assignments.append('U')
    elif line_of_crease[crease_idx] >= 0:
      assignments.append('F')
    else:
      assignments.append(
          geometric_simple_fold.fold_to_assignment(creases[crease_idx].fold)
      )

  _, face_vertices = graph_topology.graph_to_adjacency_and_faces(
      vertices, edge_vertices
  )
  folded, _ = folded_state.compute_folded_state(
      vertices, face_vertices, edge_vertices, assignments
  )

  # Align the subdivided folded state onto `graph_data` via shared vertices.
  lookup = _PointLookup()
  base_folded: list[Point] = []
  for pt, f_pt in zip(
      graph_data['vertices_coords'], graph_data['folded_vertices']
  ):
    if f_pt is not None:
      lookup.add(pt, len(base_folded))
      base_folded.append(f_pt)

  source: list[Point] = []
  target: list[Point] = []
  for pt, f_pt in zip(vertices, folded):
    if f_pt is None:
      continue
    match_idx = lookup.find(pt)
    if match_idx is not None:
      source.append(f_pt)
      target.append(base_folded[match_idx])
  if len(source) < 3:
    raise ValueError('Could not register the cut pattern onto the folded form.')
  rotation, source_mean, target_mean = geometric_simple_fold.fit_2d_isometry(
      source, target
  )

  edge_index = {
      (v1, v2) if v1 < v2 else (v2, v1): idx
      for idx, (v1, v2) in enumerate(edge_vertices)
  }
  edge_faces: dict[int, list[int]] = collections.defaultdict(list)
  face_points: list[list[Point]] = []
  centroids: list[Point | None] = []
  for face_idx, face in enumerate(face_vertices):
    raw_pts: list[Point] = []
    missing = False
    for v in face:
      fp = folded[v]
      if fp is None:
        missing = True
        break
      raw_pts.append(
          geometric_simple_fold.apply_2d_isometry(
              fp, rotation, source_mean, target_mean
          )
      )
    if missing or not raw_pts:
      face_points.append([])
      centroids.append(None)
    else:
      face_points.append(raw_pts)
      centroids.append((
          sum(p[0] for p in raw_pts) / len(raw_pts),
          sum(p[1] for p in raw_pts) / len(raw_pts),
      ))
    for i, v1 in enumerate(face):
      v2 = face[(i + 1) % len(face)]
      key = (v1, v2) if v1 < v2 else (v2, v1)
      e_idx = edge_index.get(key)
      if e_idx is not None:
        edge_faces[e_idx].append(face_idx)

  is_cut_edge = [
      crease_idx >= 0 and line_of_crease[crease_idx] >= 0
      for crease_idx in edge_crease
  ]
  return _CutGraph(
      creases=creases,
      cut_creases_per_line=cut_creases_per_line,
      line_of_crease=line_of_crease,
      vertices=vertices,
      edge_vertices=edge_vertices,
      edge_crease=edge_crease,
      assignments=assignments,
      is_cut_edge=is_cut_edge,
      face_vertices=face_vertices,
      edge_faces=edge_faces,
      face_points=face_points,
      centroids=centroids,
  )


def _flood_fill_inverted_region(
    cut_graph: _CutGraph,
    cut_lines: Sequence[CutLine],
    proximal_polygon: geometric_simple_fold.LimitingPolygon | None,
    target_polygon: geometric_simple_fold.LimitingPolygon | None,
) -> set[int]:
  """Flood-fills faces enclosed on the left side of all `cut_lines`."""

  def _is_outside(face_idx: int) -> bool:
    if proximal_polygon is not None:
      centroid = cut_graph.unfolded_centroid(face_idx)
      in_proximal = geometric_simple_fold.check_inside_limiting_polygon(
          centroid, proximal_polygon
      )
      in_target = (
          target_polygon is not None
          and geometric_simple_fold.check_inside_limiting_polygon(
              centroid, target_polygon
          )
      )
      if in_proximal and not in_target:
        return True
    for pt in cut_graph.face_points[face_idx]:
      for start, end in cut_lines:
        if _signed_side(pt, start, end) < -_REGION_SIDE_TOL:
          return True
    return False

  region: set[int] = set()
  stack: list[int] = []
  for edge_idx, is_cut in enumerate(cut_graph.is_cut_edge):
    if not is_cut:
      continue
    line = cut_lines[cut_graph.line_of_crease[cut_graph.edge_crease[edge_idx]]]
    for face_idx in cut_graph.edge_faces[edge_idx]:
      centroid = cut_graph.centroids[face_idx]
      if centroid is None or face_idx in region:
        continue
      if _signed_side(centroid, line[0], line[1]) > 0 and not _is_outside(
          face_idx
      ):
        region.add(face_idx)
        stack.append(face_idx)
  if not region:
    raise ValueError('No faces found on the inner side of the cut lines.')

  face_edges: dict[int, list[int]] = collections.defaultdict(list)
  for edge_idx, faces in cut_graph.edge_faces.items():
    for face_idx in faces:
      face_edges[face_idx].append(edge_idx)

  while stack:
    timeout_lib.maybe_timeout_cooperatively()
    face_idx = stack.pop()
    for edge_idx in face_edges[face_idx]:
      if cut_graph.is_cut_edge[edge_idx] or cut_graph.assignments[edge_idx] == (
          'B'
      ):
        continue
      for neighbor in cut_graph.edge_faces[edge_idx]:
        if neighbor in region:
          continue
        if _is_outside(neighbor):
          raise ValueError(
              'The cut does not separate the folded region from the rest of'
              ' the model: some layer of the flap at the cut lies outside the'
              ' target element (try another depth/angle, or a longer element).'
          )
        region.add(neighbor)
        stack.append(neighbor)
  return region


def _full_inversion_variants(
    cut_graph: _CutGraph,
    region: set[int],
    num_cut_lines: int,
    fold_percent: float,
    tolerance: float,
) -> list[InvertedRegionVariant]:
  """Solves Maekawa parity along cut polylines when flipping all of `region`."""
  flip: set[int] = set()
  for edge_idx, faces in cut_graph.edge_faces.items():
    crease_idx = cut_graph.edge_crease[edge_idx]
    if (
        crease_idx >= 0
        and not cut_graph.is_cut_edge[edge_idx]
        and cut_graph.assignments[edge_idx] in ('M', 'V')
        and len(faces) == 2
        and faces[0] in region
        and faces[1] in region
    ):
      flip.add(crease_idx)

  vertex_edges: dict[int, list[int]] = collections.defaultdict(list)
  for edge_idx, (v1, v2) in enumerate(cut_graph.edge_vertices):
    vertex_edges[v1].append(edge_idx)
    vertex_edges[v2].append(edge_idx)

  relations: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
  forced: dict[int, int] = {}
  for vertex, incident in vertex_edges.items():
    cut_here = [e for e in incident if cut_graph.is_cut_edge[e]]
    if not cut_here:
      continue
    others = [e for e in incident if not cut_graph.is_cut_edge[e]]
    if any(cut_graph.assignments[e] == 'B' for e in others):
      continue
    total = 0
    for e in others:
      if cut_graph.assignments[e] not in ('M', 'V'):
        continue
      val = 1 if cut_graph.assignments[e] == 'M' else -1
      total += -val if cut_graph.edge_crease[e] in flip else val
    cut_creases = sorted({cut_graph.edge_crease[e] for e in cut_here})
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
            f'Cut vertex {cut_graph.vertices[vertex]} cannot satisfy Maekawa.'
        )
    elif len(cut_creases) == 1 and abs(total) == 1:
      forced[cut_creases[0]] = -total

  components: list[tuple[int, dict[int, int], int]] = []
  seen_creases: set[int] = set()
  for idx, line_idx in enumerate(cut_graph.line_of_crease):
    if line_idx < 0 or idx in seen_creases:
      continue
    relative = {idx: 1}
    stack = [idx]
    seen_creases.add(idx)
    while stack:
      cur = stack.pop()
      for other, rel in relations[cur]:
        val = relative[cur] * rel
        if other in relative:
          if relative[other] != val:
            raise ValueError('The cut polyline has inconsistent M/V parity.')
          continue
        relative[other] = val
        seen_creases.add(other)
        stack.append(other)
    init_sign = (
        1 if cut_graph.creases[idx].fold == crease_utils.Fold.MOUNTAIN else -1
    )
    components.append((line_idx, relative, init_sign))

  base_result = [
      crease.flip_fold() if i in flip else crease
      for i, crease in enumerate(cut_graph.creases)
      if cut_graph.line_of_crease[i] < 0
  ]

  def _build(starts: Sequence[int]) -> cp_utils.CreasePattern | None:
    result = list(base_result)
    for (_, relative, _), start in zip(components, starts, strict=True):
      for c_idx, rel in relative.items():
        val = start * rel
        if forced.get(c_idx, val) != val:
          return None
        fold = (
            crease_utils.Fold.MOUNTAIN if val > 0 else crease_utils.Fold.VALLEY
        )
        result.append(
            dataclasses.replace(
                cut_graph.creases[c_idx], fold=fold, fold_percent=fold_percent
            )
        )
    return cp_utils.CreasePattern(
        creases=result, light_mode=True, tolerance=tolerance
    )

  simple = [comp[2] for comp in components]
  flipped = [-val for val in simple]
  component_endpoints: list[list[Point]] = []
  for _, relative, _ in components:
    pts: list[Point] = []
    for c_idx in relative:
      c = cut_graph.creases[c_idx]
      pts.append((c.p1.x, c.p1.y))
      pts.append((c.p2.x, c.p2.y))
    component_endpoints.append(pts)

  valid: list[set[int]] = [set() for _ in components]
  evaluated: dict[tuple[int, ...], tuple[cp_utils.CreasePattern, bool]] = {}
  for starts in (simple, flipped):
    timeout_lib.maybe_timeout_cooperatively()
    pattern = _build(starts)
    if pattern is None:
      continue
    pct, _, bad_vertices = pattern.vertex_flat_foldability_percentage()
    evaluated[tuple(starts)] = (pattern, math.isclose(pct, 1.0))
    bad_lookup = _PointLookup()
    for bad_idx, v in enumerate(bad_vertices):
      bad_lookup.add((v.point.x, v.point.y), bad_idx)
    for comp_idx, endpoints in enumerate(component_endpoints):
      if all(bad_lookup.find(pt) is None for pt in endpoints):
        valid[comp_idx].add(starts[comp_idx])

  def _prefer(starts: Sequence[int]) -> list[int]:
    return [
        s if (s in valid[k] or not valid[k]) else -s
        for k, s in enumerate(starts)
    ]

  choices = [('simple', _prefer(simple)), ('flipped', _prefer(flipped))]
  if num_cut_lines > 1:
    choices.append((
        'simple/flipped',
        _prefer([v if c[0] == 0 else -v for c, v in zip(components, simple)]),
    ))
    choices.append((
        'flipped/simple',
        _prefer([-v if c[0] == 0 else v for c, v in zip(components, simple)]),
    ))

  num_new = sum(len(comp[1]) for comp in components)
  variants: list[InvertedRegionVariant] = []
  seen_signatures: set[tuple[int, ...]] = set()
  for name, starts in choices:
    signature = tuple(starts)
    if signature in seen_signatures:
      continue
    seen_signatures.add(signature)
    cached = evaluated.get(signature)
    if cached is not None:
      pattern, is_ff = cached
    else:
      timeout_lib.maybe_timeout_cooperatively()
      pattern = _build(starts)
      if pattern is None:
        continue
      is_ff = _is_cp_flat_foldable(pattern)
    variants.append(
        InvertedRegionVariant(
            name=name,
            crease_pattern=pattern,
            num_new_creases=num_new,
            num_flipped_creases=len(flip),
            region_faces=len(region),
            is_flat_foldable=is_ff,
        )
    )
  return variants


def _spine_partition_variants(
    cut_graph: _CutGraph,
    region: set[int],
    fold_percent: float,
    tolerance: float,
) -> list[InvertedRegionVariant]:
  """Finds axial half-stack spine partitions for narrowed multi-layer stacks."""
  cut_face_in_region: dict[int, int] = {}
  for edge_idx, is_cut in enumerate(cut_graph.is_cut_edge):
    if not is_cut:
      continue
    crease_idx = cut_graph.edge_crease[edge_idx]
    for face_idx in cut_graph.edge_faces[edge_idx]:
      if face_idx in region:
        cut_face_in_region[crease_idx] = face_idx

  u_centroids = {
      face_idx: cut_graph.unfolded_centroid(face_idx) for face_idx in region
  }
  partitions: list[tuple[str, dict[int, int]]] = []
  for axis, label in ((0, 'x'), (1, 'y')):
    spine_coords: set[float] = set()
    for edge_idx, faces in cut_graph.edge_faces.items():
      crease_idx = cut_graph.edge_crease[edge_idx]
      if (
          crease_idx < 0
          or len(faces) != 2
          or faces[0] not in region
          or faces[1] not in region
      ):
        continue
      c = cut_graph.creases[crease_idx]
      c1 = c.p1.x if axis == 0 else c.p1.y
      c2 = c.p2.x if axis == 0 else c.p2.y
      if abs(c1 - c2) < _AXIAL_EDGE_TOL:
        spine_coords.add(round(c1, 6))
    if not spine_coords:
      continue
    coords = [u_centroids[f][axis] for f in region]
    midpoint = 0.5 * (min(coords) + max(coords))
    for threshold in sorted(
        spine_coords, key=lambda v, mid=midpoint: abs(v - mid)
    ):
      sigma = {f: 1 if u_centroids[f][axis] < threshold else -1 for f in region}
      if len(set(sigma.values())) == 2:
        partitions.append((f'spine_{label}={threshold:.3f}', sigma))

  for part_name, sigma in partitions:
    spine_matches: list[InvertedRegionVariant] = []
    for global_sign in (1, -1):
      timeout_lib.maybe_timeout_cooperatively()
      flip_set = {
          cut_graph.edge_crease[edge_idx]
          for edge_idx, faces in cut_graph.edge_faces.items()
          if cut_graph.edge_crease[edge_idx] >= 0
          and not cut_graph.is_cut_edge[edge_idx]
          and cut_graph.assignments[edge_idx] in ('M', 'V')
          and len(faces) == 2
          and faces[0] in region
          and faces[1] in region
          and sigma[faces[0]] != sigma[faces[1]]
      }
      result: list[crease_utils.Crease] = []
      num_new = 0
      for i, crease in enumerate(cut_graph.creases):
        if cut_graph.line_of_crease[i] >= 0:
          face_idx = cut_face_in_region.get(i)
          if face_idx is None:
            continue
          fold = (
              crease.fold
              if global_sign * sigma[face_idx] > 0
              else crease.fold.flip()
          )
          result.append(
              dataclasses.replace(crease, fold=fold, fold_percent=fold_percent)
          )
          num_new += 1
        else:
          result.append(crease.flip_fold() if i in flip_set else crease)
      pattern = cp_utils.CreasePattern(
          creases=result, light_mode=True, tolerance=tolerance
      )
      if _is_cp_flat_foldable(pattern):
        suffix = '+' if global_sign > 0 else '-'
        spine_matches.append(
            InvertedRegionVariant(
                name=f'{part_name}{suffix}',
                crease_pattern=pattern,
                num_new_creases=num_new,
                num_flipped_creases=len(flip_set),
                region_faces=len(region),
                is_flat_foldable=True,
            )
        )
    if spine_matches:
      return spine_matches
  return []


def _pleat_crimp_variants(
    crease_pattern: cp_utils.CreasePattern,
    cut_creases_per_line: Sequence[Sequence[crease_utils.Crease]],
    region_faces: int,
    fold_percent: float,
) -> list[InvertedRegionVariant]:
  """Builds zero-flip opposite-sign pleat variants for two-line crimps."""
  if len(cut_creases_per_line) != 2:
    return []
  c0, c1 = cut_creases_per_line
  variants: list[InvertedRegionVariant] = []
  for pleat_name, flip0, flip1 in (
      ('pleat_mv', True, False),
      ('pleat_vm', False, True),
  ):
    timeout_lib.maybe_timeout_cooperatively()
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
    pattern = crease_pattern.update_with_shaping_creases([*creases0, *creases1])
    if _is_cp_flat_foldable(pattern):
      variants.append(
          InvertedRegionVariant(
              name=pleat_name,
              crease_pattern=pattern,
              num_new_creases=len(creases0) + len(creases1),
              num_flipped_creases=0,
              region_faces=region_faces,
              is_flat_foldable=True,
          )
      )
  return variants


def get_inverted_region_variants(
    crease_pattern: cp_utils.CreasePattern,
    *,
    cut_lines: Sequence[CutLine],
    limiting_polygon: geometric_simple_fold.LimitingPolygon,
    graph_data: geometric_simple_fold.GraphData | None = None,
    fold_percent: float = 1.0,
    spans: Sequence[tuple[float, float] | None] | None = None,
    proximal_polygon: geometric_simple_fold.LimitingPolygon | None = None,
    target_polygon: geometric_simple_fold.LimitingPolygon | None = None,
) -> list[InvertedRegionVariant]:
  """Generates candidate M/V assignments for a folded-state reverse or crimp.

  Inverts the planar region `R` lying to the left of every directed line in
  `cut_lines` (one line = distal region of a reverse fold; two opposing lines =
  central strip of a crimp fold). First attempts full interior inversion with
  Maekawa polyline propagation; if no full-inversion variant is locally
  flat-foldable (e.g. on narrowed double-pleat stacks), falls back to axial
  spine half-stack partitions and zero-flip two-line pleat crimps.

  Args:
    crease_pattern: Current crease pattern to fold.
    cut_lines: Directed cut lines `(start, end)` in folded 2D coordinates.
    limiting_polygon: Unfolded region restricting where cut creases are placed.
    graph_data: Optional precomputed `prepare_graph_data(crease_pattern)`.
    fold_percent: Fold percentage assigned to newly inserted cut creases.
    spans: Optional `(min_dist, max_dist)` projection bounds per cut line.
    proximal_polygon: Optional unfolded region that must remain outside `R`.
    target_polygon: Optional unfolded region exempted from `proximal_polygon`.

  Returns:
    Candidate `InvertedRegionVariant` records with `is_flat_foldable` flags.

  Raises:
    ValueError: If `cut_lines` is empty, `spans` mismatches `cut_lines`, a cut
      line misses the flap, the cut fails to separate `R`, or Maekawa parity
      fails with no flat-foldable multi-layer fallback.
  """
  if not cut_lines:
    raise ValueError('At least one cut line is required.')
  if spans is not None and len(spans) != len(cut_lines):
    raise ValueError(
        f'spans length ({len(spans)}) must match cut_lines ({len(cut_lines)}).'
    )
  if graph_data is None:
    graph_data = geometric_simple_fold.prepare_graph_data(crease_pattern)

  cut_graph = _build_registered_cut_graph(
      crease_pattern, cut_lines, limiting_polygon, graph_data, spans
  )
  region = _flood_fill_inverted_region(
      cut_graph, cut_lines, proximal_polygon, target_polygon
  )

  tolerance = crease_pattern.tolerance
  full_error: ValueError | None = None
  variants: list[InvertedRegionVariant] = []
  try:
    variants = _full_inversion_variants(
        cut_graph, region, len(cut_lines), fold_percent, tolerance
    )
  except ValueError as error:
    full_error = error

  if any(v['is_flat_foldable'] for v in variants):
    return variants

  fallback = [
      *_spine_partition_variants(cut_graph, region, fold_percent, tolerance),
      *_pleat_crimp_variants(
          crease_pattern,
          cut_graph.cut_creases_per_line,
          len(region),
          fold_percent,
      ),
  ]
  if fallback:
    return fallback
  if full_error is not None:
    raise full_error
  return variants
