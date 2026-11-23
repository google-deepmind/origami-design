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

"""Pure-NumPy deterministic 3D origami folder and planar face extractor."""

import bisect
import collections
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import math
import time
import typing
from typing import Any, Literal

import numpy as np

from . import bus
from .crease_pattern import crease as crease_lib
from .crease_pattern import crease_pattern as cp_lib
from .crease_pattern.parser import svg as svg_parser

FoldAssignment = Literal['B', 'M', 'V', 'F', 'U']
FoldTarget = (
    cp_lib.CreasePattern
    | bus.ShapedCreasePattern
    | bus.SolvedCreasePattern
    | bus.PipelineResult
)

_FOLD_PRECEDENCE: Mapping[FoldAssignment, int] = {
    'M': 3,
    'V': 3,
    'F': 2,
    'B': 1,
    'U': 0,
}
_MIN_FACE_DOUBLE_AREA = 1e-12
_COLLINEAR_COS_TOL = 1e-6
_ANGLE_MERGE_TOL_DEG = 1e-4
_BARYCENTRIC_TOL = 1e-9
_EDGE_FALLBACK_TOL = 1e-5


def crease_assignment_and_angle(
    fold: crease_lib.Fold,
    fold_percent: float,
) -> tuple[FoldAssignment, float]:
  """Maps a `Fold` and `fold_percent` to a FOLD assignment and degree angle."""
  pct = max(0.0, min(1.0, float(fold_percent)))
  match fold:
    case crease_lib.Fold.MOUNTAIN | crease_lib.Fold.MOUNTAIN_ZEBRA:
      return ('M', -180.0 * pct)
    case crease_lib.Fold.VALLEY | crease_lib.Fold.VALLEY_ZEBRA:
      return ('V', 180.0 * pct)
    case crease_lib.Fold.BORDER:
      return ('B', 0.0)
    case crease_lib.Fold.HINGE | crease_lib.Fold.TRIANGLE:
      return ('F', 0.0)
    case _ as unreachable:
      typing.assert_never(unreachable)


def _polygon_signed_double_area(
    face: Sequence[int],
    vertices: Sequence[Sequence[float]] | np.ndarray,
) -> float:
  """Returns twice the signed area of a 2D polygon via the shoelace formula."""
  n = len(face)
  if n < 3:
    return 0.0
  area2 = 0.0
  for i in range(n):
    x1, y1 = vertices[face[i]]
    x2, y2 = vertices[face[(i + 1) % n]]
    area2 += float(x1) * float(y2) - float(x2) * float(y1)
  return area2


def extract_planar_faces(
    vertices: Sequence[Sequence[float]] | np.ndarray,
    edges: Sequence[Sequence[int]] | np.ndarray,
) -> list[list[int]]:
  """Extracts minimal bounded planar faces via angular half-edge walking.

  Faces are returned in clockwise `(x, y)` order (counter-clockwise in top-down
  SVG coordinates), matching FOLD hinge conventions so a positive (valley) fold
  angle around a directed face edge `v1 -> v2` lifts the neighbor toward `+z`.

  Args:
    vertices: `(V, 2)` planar vertex coordinates.
    edges: `(E, 2)` undirected vertex-index pairs.

  Returns:
    Bounded face vertex cycles as lists of vertex indices.
  """
  num_verts = len(vertices)
  adj_sets: list[set[int]] = [set() for _ in range(num_verts)]
  for edge in edges:
    u, v = int(edge[0]), int(edge[1])
    if u != v:
      adj_sets[u].add(v)
      adj_sets[v].add(u)

  next_half_edge: dict[tuple[int, int], int | None] = {}
  for v, nb_set in enumerate(adj_sets):
    if not nb_set:
      continue
    x0, y0 = float(vertices[v][0]), float(vertices[v][1])
    neighbors = sorted(
        nb_set,
        key=lambda nb, cx=x0, cy=y0: math.atan2(
            float(vertices[nb][1]) - cy,
            float(vertices[nb][0]) - cx,
        ),
    )
    deg = len(neighbors)
    for idx, u in enumerate(neighbors):
      next_half_edge[(u, v)] = neighbors[(idx - 1) % deg]

  max_steps = len(next_half_edge) + 2
  faces: list[list[int]] = []
  for half_edge in list(next_half_edge):
    w = next_half_edge[half_edge]
    if w is None:
      continue
    u, v = half_edge
    next_half_edge[half_edge] = None
    face = [u, v]
    valid = True
    while w != u:
      if w is None or len(face) > max_steps:
        valid = False
        break
      face.append(w)
      step = (v, w)
      v, w = w, next_half_edge.get(step)
      next_half_edge[step] = None
    closing = (face[-1], face[0])
    if valid and closing in next_half_edge:
      next_half_edge[closing] = None
      if _polygon_signed_double_area(face, vertices) > _MIN_FACE_DOUBLE_AREA:
        faces.append(face[::-1])
  return faces


class _SpatialVertexIndex:
  """Snaps 2D points within `tol` using a non-drifting uniform grid."""

  def __init__(self, tol: float) -> None:
    self._tol = tol
    self._tol_sq = tol * tol
    self._inv_cell = (1.0 / tol) if tol > 0.0 else 0.0
    self._grid: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    self.coords: list[tuple[float, float]] = []

  def add(self, x: float, y: float) -> int:
    """Returns an existing vertex index within `tol`, or inserts `(x, y)`."""
    if self._tol <= 0.0:
      idx = len(self.coords)
      self.coords.append((x, y))
      return idx
    cx = math.floor(x * self._inv_cell)
    cy = math.floor(y * self._inv_cell)
    best_idx: int | None = None
    best_dist_sq = self._tol_sq
    for dx in (-1, 0, 1):
      for dy in (-1, 0, 1):
        for cand in self._grid.get((cx + dx, cy + dy), ()):
          ux, uy = self.coords[cand]
          dist_sq = (x - ux) ** 2 + (y - uy) ** 2
          if dist_sq < best_dist_sq or (
              dist_sq == best_dist_sq
              and best_idx is not None
              and cand < best_idx
          ):
            best_dist_sq = dist_sq
            best_idx = cand
    if best_idx is not None:
      return best_idx
    idx = len(self.coords)
    self.coords.append((x, y))
    self._grid[(cx, cy)].append(idx)
    return idx


@dataclasses.dataclass(frozen=True, slots=True)
class _EdgeRecord:
  v1: int
  v2: int
  assignment: FoldAssignment
  fold_angle: float


def _deduplicate_edges(edges: Sequence[_EdgeRecord]) -> list[_EdgeRecord]:
  """Removes self-loops and merges parallel edges by assignment precedence."""
  by_key: dict[tuple[int, int], int] = {}
  result: list[_EdgeRecord] = []
  for edge in edges:
    if edge.v1 == edge.v2:
      continue
    key = (edge.v1, edge.v2) if edge.v1 < edge.v2 else (edge.v2, edge.v1)
    prev_idx = by_key.get(key)
    if prev_idx is None:
      by_key[key] = len(result)
      result.append(edge)
    elif (
        _FOLD_PRECEDENCE[edge.assignment]
        > _FOLD_PRECEDENCE[result[prev_idx].assignment]
    ):
      result[prev_idx] = edge
  return result


def _split_planar_edges(
    index: _SpatialVertexIndex,
    edges: Sequence[_EdgeRecord],
    tol: float,
) -> list[_EdgeRecord]:
  """Splits edges at all interior X-crossings, T-junctions, and overlaps."""
  if not edges:
    return []

  # Phase 1: sweep x-sorted bounding boxes to insert proper X-crossing points.
  coords = index.coords
  boxes: list[tuple[float, float, float, float, int]] = []
  for idx, edge in enumerate(edges):
    x1, y1 = coords[edge.v1]
    x2, y2 = coords[edge.v2]
    boxes.append((
        min(x1, x2),
        max(x1, x2),
        min(y1, y2),
        max(y1, y2),
        idx,
    ))
  boxes.sort(key=lambda item: item[0])

  num_edges = len(boxes)
  for a in range(num_edges):
    _, max_x1, min_y1, max_y1, idx1 = boxes[a]
    e1 = edges[idx1]
    x1, y1 = coords[e1.v1]
    x2, y2 = coords[e1.v2]
    dx1, dy1 = x2 - x1, y2 - y1
    len1 = math.hypot(dx1, dy1)
    if len1 <= tol:
      continue
    for b in range(a + 1, num_edges):
      min_x2, _, min_y2, max_y2, idx2 = boxes[b]
      if min_x2 > max_x1 + tol:
        break
      if max_y2 < min_y1 - tol or min_y2 > max_y1 + tol:
        continue
      e2 = edges[idx2]
      if e1.v1 == e2.v1 or e1.v1 == e2.v2 or e1.v2 == e2.v1 or e1.v2 == e2.v2:
        continue
      x3, y3 = coords[e2.v1]
      x4, y4 = coords[e2.v2]
      dx2, dy2 = x4 - x3, y4 - y3
      denom = dx1 * dy2 - dy1 * dx2
      if abs(denom) < 1e-14:
        continue
      rx, ry = x3 - x1, y3 - y1
      t1 = (rx * dy2 - ry * dx2) / denom
      t2 = (rx * dy1 - ry * dx1) / denom
      len2 = math.hypot(dx2, dy2)
      d1 = t1 * len1
      d2 = t2 * len2
      if (
          -tol <= d1 <= len1 + tol
          and -tol <= d2 <= len2 + tol
          and (tol < d1 < len1 - tol)
          and (tol < d2 < len2 - tol)
      ):
        index.add(x1 + t1 * dx1, y1 + t1 * dy1)

  # Phase 2: split every edge at all vertices strictly inside its span.
  vert_order = sorted(range(len(coords)), key=lambda v: coords[v][0])
  vert_xs = [coords[v][0] for v in vert_order]

  split_edges: list[_EdgeRecord] = []
  for edge in edges:
    x1, y1 = coords[edge.v1]
    x2, y2 = coords[edge.v2]
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    length = math.sqrt(len_sq)
    if length <= tol:
      continue
    min_x = min(x1, x2) - tol
    max_x = max(x1, x2) + tol
    min_y = min(y1, y2) - tol
    max_y = max(y1, y2) + tol
    dot_limit = tol * length
    cross_limit = tol * length

    lo = bisect.bisect_left(vert_xs, min_x)
    hi = bisect.bisect_right(vert_xs, max_x)
    splits: list[tuple[float, int]] = []
    for pos in range(lo, hi):
      v = vert_order[pos]
      if v == edge.v1 or v == edge.v2:
        continue
      vx, vy = coords[v]
      if vy < min_y or vy > max_y:
        continue
      avx, avy = vx - x1, vy - y1
      if abs(dx * avy - dy * avx) < cross_limit:
        dot = dx * avx + dy * avy
        if dot_limit < dot < len_sq - dot_limit:
          splits.append((dot, v))

    if not splits:
      split_edges.append(edge)
      continue
    splits.sort(key=lambda item: item[0])
    prev = edge.v1
    for _, mid in splits:
      if mid != prev:
        split_edges.append(
            _EdgeRecord(prev, mid, edge.assignment, edge.fold_angle)
        )
        prev = mid
    if prev != edge.v2:
      split_edges.append(
          _EdgeRecord(prev, edge.v2, edge.assignment, edge.fold_angle)
      )
  return _deduplicate_edges(split_edges)


def _merge_collinear_degree2_vertices(
    coords: Sequence[tuple[float, float]],
    edges: Sequence[_EdgeRecord],
) -> tuple[list[tuple[float, float]], list[_EdgeRecord]]:
  """Contracts strictly collinear degree-2 vertices with matching attributes."""
  num_verts = len(coords)
  incident: list[list[int]] = [[] for _ in range(num_verts)]
  active: list[_EdgeRecord | None] = list(edges)
  for idx, edge in enumerate(edges):
    incident[edge.v1].append(idx)
    incident[edge.v2].append(idx)

  for v in range(num_verts):
    inc = [e_idx for e_idx in incident[v] if active[e_idx] is not None]
    if len(inc) != 2:
      continue
    e_a = active[inc[0]]
    e_b = active[inc[1]]
    if e_a is None or e_b is None:
      continue
    if e_a.assignment != e_b.assignment:
      continue
    if abs(e_a.fold_angle - e_b.fold_angle) > _ANGLE_MERGE_TOL_DEG:
      continue
    u = e_a.v2 if e_a.v1 == v else e_a.v1
    w = e_b.v2 if e_b.v1 == v else e_b.v1
    if u == w:
      continue
    vx, vy = coords[v]
    ux, uy = coords[u][0] - vx, coords[u][1] - vy
    wx, wy = coords[w][0] - vx, coords[w][1] - vy
    denom = math.hypot(ux, uy) * math.hypot(wx, wy)
    if denom <= 0.0:
      continue
    if abs((ux * wx + uy * wy) / denom + 1.0) >= _COLLINEAR_COS_TOL:
      continue
    merged = _EdgeRecord(
        u, w, e_a.assignment, 0.5 * (e_a.fold_angle + e_b.fold_angle)
    )
    new_idx = len(active)
    active[inc[0]] = None
    active[inc[1]] = None
    active.append(merged)
    incident[u].append(new_idx)
    incident[w].append(new_idx)

  kept_edges = _deduplicate_edges([e for e in active if e is not None])
  used: set[int] = set()
  for edge in kept_edges:
    used.add(edge.v1)
    used.add(edge.v2)
  ordered = sorted(used)
  remap = {old: new for new, old in enumerate(ordered)}
  new_coords = [coords[old] for old in ordered]
  remapped_edges = [
      _EdgeRecord(remap[e.v1], remap[e.v2], e.assignment, e.fold_angle)
      for e in kept_edges
  ]
  return new_coords, remapped_edges


@dataclasses.dataclass(frozen=True, slots=True)
class FoldGraph:
  """Planar crease graph and extracted bounded faces on the unit square."""

  vertices_2d: np.ndarray
  edges: np.ndarray
  assignments: tuple[FoldAssignment, ...]
  fold_angles: np.ndarray
  faces: tuple[np.ndarray, ...]

  def edge_to_faces(self) -> dict[tuple[int, int], list[int]]:
    """Maps sorted undirected edge vertex pairs to incident face indices."""
    mapping: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for f_idx, face in enumerate(self.faces):
      n = len(face)
      for i in range(n):
        u, v = int(face[i]), int(face[(i + 1) % n])
        key = (u, v) if u < v else (v, u)
        mapping[key].append(f_idx)
    return mapping

  def num_face_components(self, *, check_manifold: bool = True) -> int:
    """Returns connected component count across shared face edges."""
    num_faces = len(self.faces)
    if num_faces == 0:
      return 0
    edge_faces = self.edge_to_faces()
    face_adj: list[list[int]] = [[] for _ in range(num_faces)]
    for key, inc in edge_faces.items():
      if check_manifold and len(inc) > 2:
        raise ValueError(
            f'Non-manifold edge {key} is shared by {len(inc)} faces.'
        )
      if len(inc) >= 2:
        for i in range(len(inc) - 1):
          face_adj[inc[i]].append(inc[i + 1])
          face_adj[inc[i + 1]].append(inc[i])

    visited = [False] * num_faces
    components = 0
    for seed in range(num_faces):
      if visited[seed]:
        continue
      components += 1
      visited[seed] = True
      queue = collections.deque([seed])
      while queue:
        curr = queue.popleft()
        for nb in face_adj[curr]:
          if not visited[nb]:
            visited[nb] = True
            queue.append(nb)
    return components

  def without_dangling(self) -> 'FoldGraph':
    """Drops edges and vertices that do not belong to any bounded face."""
    if not self.faces:
      return FoldGraph(
          vertices_2d=np.zeros((0, 2), dtype=float),
          edges=np.zeros((0, 2), dtype=int),
          assignments=(),
          fold_angles=np.zeros(0, dtype=float),
          faces=(),
      )
    face_edges = self.edge_to_faces()
    active_verts_set: set[int] = set()
    for face in self.faces:
      active_verts_set.update(int(v) for v in face)
    active_verts = sorted(active_verts_set)
    remap = {old: new for new, old in enumerate(active_verts)}

    known_edges: dict[tuple[int, int], tuple[FoldAssignment, float]] = {}
    kept_edges: list[tuple[int, int]] = []
    kept_assign: list[FoldAssignment] = []
    kept_angles: list[float] = []
    for idx, (u_raw, v_raw) in enumerate(self.edges):
      u, v = int(u_raw), int(v_raw)
      key = (u, v) if u < v else (v, u)
      if key not in face_edges:
        continue
      known_edges[key] = (self.assignments[idx], float(self.fold_angles[idx]))
      kept_edges.append((remap[u], remap[v]))
      kept_assign.append(self.assignments[idx])
      kept_angles.append(float(self.fold_angles[idx]))

    # Ensure every face boundary edge is represented even if missing initially.
    for key in face_edges:
      if key not in known_edges:
        kept_edges.append((remap[key[0]], remap[key[1]]))
        kept_assign.append('B')
        kept_angles.append(0.0)

    new_faces = tuple(
        np.array([remap[int(v)] for v in face], dtype=int)
        for face in self.faces
    )
    return FoldGraph(
        vertices_2d=self.vertices_2d[active_verts],
        edges=(
            np.asarray(kept_edges, dtype=int)
            if kept_edges
            else np.zeros((0, 2), dtype=int)
        ),
        assignments=tuple(kept_assign),
        fold_angles=np.asarray(kept_angles, dtype=float),
        faces=new_faces,
    )


def build_fold_graph(
    pattern: cp_lib.CreasePattern,
    *,
    vert_tol: float = 1e-6,
    add_border: bool = False,
    merge_collinear: bool = True,
    remove_dangling: bool = True,
) -> FoldGraph:
  """Builds a clean planar `FoldGraph` directly from a `CreasePattern`."""
  index = _SpatialVertexIndex(vert_tol)
  raw_edges: list[_EdgeRecord] = []

  def _add(
      x1: float,
      y1: float,
      x2: float,
      y2: float,
      assignment: FoldAssignment,
      fold_angle: float,
  ) -> None:
    if math.hypot(x2 - x1, y2 - y1) <= max(vert_tol, 1e-12):
      return
    v1 = index.add(x1, y1)
    v2 = index.add(x2, y2)
    if v1 != v2:
      raw_edges.append(_EdgeRecord(v1, v2, assignment, fold_angle))

  if add_border:
    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for i in range(4):
      c1, c2 = corners[i], corners[(i + 1) % 4]
      _add(c1[0], c1[1], c2[0], c2[1], 'B', 0.0)

  for crease in pattern.creases:
    assign, angle = crease_assignment_and_angle(
        crease.fold, crease.fold_percent
    )
    _add(
        float(crease.p1.x),
        float(crease.p1.y),
        float(crease.p2.x),
        float(crease.p2.y),
        assign,
        angle,
    )

  deduped = _deduplicate_edges(raw_edges)
  split = _split_planar_edges(index, deduped, vert_tol)
  coords: list[tuple[float, float]] = list(index.coords)
  if merge_collinear:
    coords, split = _merge_collinear_degree2_vertices(coords, split)

  vertices_arr = (
      np.asarray(coords, dtype=float)
      if coords
      else np.zeros((0, 2), dtype=float)
  )
  edges_arr = (
      np.asarray([(e.v1, e.v2) for e in split], dtype=int)
      if split
      else np.zeros((0, 2), dtype=int)
  )
  raw_faces = extract_planar_faces(vertices_arr, edges_arr)
  graph = FoldGraph(
      vertices_2d=vertices_arr,
      edges=edges_arr,
      assignments=tuple(e.assignment for e in split),
      fold_angles=np.asarray([e.fold_angle for e in split], dtype=float),
      faces=tuple(np.asarray(face, dtype=int) for face in raw_faces),
  )
  return graph.without_dangling() if remove_dangling else graph


def rotation_matrix_4x4(axis: np.ndarray, angle_rad: float) -> np.ndarray:
  """Returns a 4x4 right-handed Rodrigues rotation matrix around `axis`."""
  norm = float(np.linalg.norm(axis))
  if norm < 1e-12 or abs(angle_rad) < 1e-12:
    return np.eye(4, dtype=float)
  ux = float(axis[0]) / norm
  uy = float(axis[1]) / norm
  uz = float(axis[2]) / norm
  cos_a = math.cos(angle_rad)
  sin_a = math.sin(angle_rad)
  omc = 1.0 - cos_a
  return np.array(
      [
          [
              omc * ux * ux + cos_a,
              omc * ux * uy - sin_a * uz,
              omc * ux * uz + sin_a * uy,
              0.0,
          ],
          [
              omc * ux * uy + sin_a * uz,
              omc * uy * uy + cos_a,
              omc * uy * uz - sin_a * ux,
              0.0,
          ],
          [
              omc * ux * uz - sin_a * uy,
              omc * uy * uz + sin_a * ux,
              omc * uz * uz + cos_a,
              0.0,
          ],
          [0.0, 0.0, 0.0, 1.0],
      ],
      dtype=float,
  )


def _hinge_transform(
    p1: np.ndarray, p2: np.ndarray, angle_deg: float
) -> np.ndarray:
  """Returns the 4x4 rigid transform rotating around hinge `p1 -> p2`."""
  if abs(angle_deg) < 1e-9:
    return np.eye(4, dtype=float)
  rot = rotation_matrix_4x4(p2 - p1, math.radians(angle_deg))
  rot[:3, 3] = p1 - rot[:3, :3] @ p1
  return rot


def _solve_face_transforms(
    graph: FoldGraph,
    *,
    anchor_face: int,
    strict: bool,
) -> tuple[np.ndarray, np.ndarray]:
  """Computes `(F, 4, 4)` SE(3) face transforms via BFS hinge folding."""
  num_faces = len(graph.faces)
  if not (0 <= anchor_face < num_faces):
    raise ValueError(
        f'anchor_face {anchor_face} out of range for {num_faces} faces.'
    )

  edge_angle: dict[tuple[int, int], float] = {}
  for idx, (u_raw, v_raw) in enumerate(graph.edges):
    u, v = int(u_raw), int(v_raw)
    key = (u, v) if u < v else (v, u)
    edge_angle[key] = float(graph.fold_angles[idx])
  edge_faces = graph.edge_to_faces()

  matrices = np.broadcast_to(np.eye(4, dtype=float), (num_faces, 4, 4)).copy()
  reached = np.zeros(num_faces, dtype=bool)
  reached[anchor_face] = True
  queue = collections.deque([anchor_face])
  verts = graph.vertices_2d

  while queue:
    curr = queue.popleft()
    curr_mat = matrices[curr]
    rot_part = curr_mat[:3, :2]
    trans_part = curr_mat[:3, 3]
    face = graph.faces[curr]
    n = len(face)
    for i in range(n):
      v1, v2 = int(face[i]), int(face[(i + 1) % n])
      key = (v1, v2) if v1 < v2 else (v2, v1)
      angle_deg = edge_angle.get(key, 0.0)
      for nxt in edge_faces.get(key, ()):
        if nxt == curr or reached[nxt]:
          continue
        p1 = rot_part @ verts[v1] + trans_part
        p2 = rot_part @ verts[v2] + trans_part
        matrices[nxt] = _hinge_transform(p1, p2, angle_deg) @ curr_mat
        reached[nxt] = True
        queue.append(nxt)

  if strict and not bool(np.all(reached)):
    missing = np.flatnonzero(~reached).tolist()
    raise ValueError(
        f'Disconnected crease pattern: {len(missing)} of {num_faces} faces '
        f'unreachable from face {anchor_face}.'
    )
  return matrices, reached


def _average_vertex_positions(
    vertices_2d: np.ndarray,
    faces: Sequence[np.ndarray],
    matrices: np.ndarray,
    reached: np.ndarray,
) -> np.ndarray:
  """Averages transformed 3D vertex positions across incident reached faces."""
  num_verts = len(vertices_2d)
  sums = np.zeros((num_verts, 3), dtype=float)
  counts = np.zeros(num_verts, dtype=float)
  for f_idx, face in enumerate(faces):
    if not reached[f_idx] or len(face) == 0:
      continue
    mat = matrices[f_idx]
    pts = vertices_2d[face] @ mat[:3, :2].T + mat[:3, 3]
    sums[face] += pts
    counts[face] += 1.0

  out = np.zeros((num_verts, 3), dtype=float)
  valid = counts > 0.0
  out[valid] = sums[valid] / counts[valid, None]
  if not np.all(valid):
    out[~valid, :2] = vertices_2d[~valid]
  return out


def compute_vertex_error(
    vertices_2d: np.ndarray,
    vertices_3d: np.ndarray,
    edges: np.ndarray,
) -> tuple[float, np.ndarray]:
  """Returns `(mean_percent_error, per_vertex_strain)` from edge strains."""
  v2 = np.asarray(vertices_2d, dtype=float)
  v3 = np.asarray(vertices_3d, dtype=float)
  e = np.asarray(edges, dtype=int)
  num_verts = len(v2)
  if num_verts == 0 or len(e) == 0:
    return 0.0, np.zeros(num_verts, dtype=float)

  len_2d = np.linalg.norm(v2[e[:, 0]] - v2[e[:, 1]], axis=1)
  len_3d = np.linalg.norm(v3[e[:, 0]] - v3[e[:, 1]], axis=1)
  valid = len_2d > 1e-12
  strain = np.zeros(len(e), dtype=float)
  strain[valid] = np.abs(len_3d[valid] / len_2d[valid] - 1.0)

  v_err = np.zeros(num_verts, dtype=float)
  v_cnt = np.zeros(num_verts, dtype=float)
  np.add.at(v_err, e[valid, 0], strain[valid])
  np.add.at(v_err, e[valid, 1], strain[valid])
  np.add.at(v_cnt, e[valid, 0], 1.0)
  np.add.at(v_cnt, e[valid, 1], 1.0)

  per_vertex = np.zeros(num_verts, dtype=float)
  has_edges = v_cnt > 0.0
  per_vertex[has_edges] = v_err[has_edges] / v_cnt[has_edges]
  mean_pct = (
      float(per_vertex[has_edges].mean() * 100.0) if np.any(has_edges) else 0.0
  )
  return mean_pct, per_vertex


def _point_in_triangle_2d(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> bool:
  """Returns True if 2D point `p` lies inside or on triangle `(a, b, c)`."""
  u = b - a
  v = c - a
  det = float(u[0] * v[1] - u[1] * v[0])
  if abs(det) < 1e-18:
    return False
  w = p - a
  s = float(w[0] * v[1] - w[1] * v[0]) / det
  t = float(u[0] * w[1] - u[1] * w[0]) / det
  return s >= -1e-12 and t >= -1e-12 and s + t <= 1.0 + 1e-12


def _triangulate_face(
    face: np.ndarray, vertices_2d: np.ndarray
) -> list[tuple[int, int, int]]:
  """Triangulates a simple 2D polygon face via convex fan or ear clipping."""
  k = len(face)
  if k < 3:
    return []
  if k == 3:
    return [(int(face[0]), int(face[1]), int(face[2]))]

  # Orientation sign: negative for CW faces returned by `extract_planar_faces`.
  area_sign = (
      -1.0 if _polygon_signed_double_area(face, vertices_2d) < 0.0 else 1.0
  )

  def _is_convex(i_prev: int, i_curr: int, i_next: int) -> bool:
    e1 = vertices_2d[i_curr] - vertices_2d[i_prev]
    e2 = vertices_2d[i_next] - vertices_2d[i_curr]
    return float(e1[0] * e2[1] - e1[1] * e2[0]) * area_sign >= -1e-15

  ring = [int(v) for v in face]
  if all(_is_convex(ring[i - 1], ring[i], ring[(i + 1) % k]) for i in range(k)):
    return [(ring[0], ring[i], ring[i + 1]) for i in range(1, k - 1)]

  triangles: list[tuple[int, int, int]] = []
  while len(ring) > 3:
    m = len(ring)
    clipped = False
    for i in range(m):
      a, b, c = ring[i - 1], ring[i], ring[(i + 1) % m]
      if not _is_convex(a, b, c):
        continue
      pa, pb, pc = vertices_2d[a], vertices_2d[b], vertices_2d[c]
      e1, e2 = pb - pa, pc - pa
      if abs(float(e1[0] * e2[1] - e1[1] * e2[0])) < 1e-18:
        ring.pop(i)
        clipped = True
        break
      contains_other = False
      for other in ring:
        if other in (a, b, c):
          continue
        if _point_in_triangle_2d(vertices_2d[other], pa, pb, pc):
          contains_other = True
          break
      if not contains_other:
        triangles.append((a, b, c))
        ring.pop(i)
        clipped = True
        break
    if not clipped:
      # Fallback fan for near-degenerate collinear rings.
      triangles.extend(
          (ring[0], ring[i], ring[i + 1]) for i in range(1, len(ring) - 1)
      )
      return triangles
  triangles.append((ring[0], ring[1], ring[2]))
  return triangles


@dataclasses.dataclass(frozen=True, slots=True)
class _TriangleIndex:
  """Precomputed 2D face triangulation for fast pure-NumPy point location."""

  face_boxes: np.ndarray  # (F, 4): min_x, max_x, min_y, max_y
  a: np.ndarray  # (T, 2)
  u: np.ndarray  # (T, 2)
  v: np.ndarray  # (T, 2)
  inv_det: np.ndarray  # (T,)
  tri_face: np.ndarray  # (T,)
  tri_boxes: np.ndarray  # (T, 4)

  @classmethod
  def build(
      cls, vertices_2d: np.ndarray, faces: Sequence[np.ndarray]
  ) -> '_TriangleIndex':
    """Builds a barycentric triangle lookup table over all faces."""
    if not faces:
      empty4 = np.zeros((0, 4), dtype=float)
      empty2 = np.zeros((0, 2), dtype=float)
      return cls(
          face_boxes=empty4,
          a=empty2,
          u=empty2,
          v=empty2,
          inv_det=np.zeros(0, dtype=float),
          tri_face=np.zeros(0, dtype=int),
          tri_boxes=empty4,
      )

    face_boxes = np.array(
        [
            [
                vertices_2d[f, 0].min(),
                vertices_2d[f, 0].max(),
                vertices_2d[f, 1].min(),
                vertices_2d[f, 1].max(),
            ]
            for f in faces
        ],
        dtype=float,
    )
    tris: list[tuple[int, int, int]] = []
    owners: list[int] = []
    for f_idx, face in enumerate(faces):
      for tri in _triangulate_face(face, vertices_2d):
        tris.append(tri)
        owners.append(f_idx)

    if not tris:
      empty2 = np.zeros((0, 2), dtype=float)
      return cls(
          face_boxes=face_boxes,
          a=empty2,
          u=empty2,
          v=empty2,
          inv_det=np.zeros(0, dtype=float),
          tri_face=np.zeros(0, dtype=int),
          tri_boxes=np.zeros((0, 4), dtype=float),
      )

    tri_arr = np.asarray(tris, dtype=int)
    pa = vertices_2d[tri_arr[:, 0]]
    pb = vertices_2d[tri_arr[:, 1]]
    pc = vertices_2d[tri_arr[:, 2]]
    u = pb - pa
    v = pc - pa
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    keep = np.abs(det) > 1e-18
    pa, pb, pc = pa[keep], pb[keep], pc[keep]
    u, v, det = u[keep], v[keep], det[keep]
    tri_face = np.asarray(owners, dtype=int)[keep]
    xs = np.stack([pa[:, 0], pb[:, 0], pc[:, 0]], axis=1)
    ys = np.stack([pa[:, 1], pb[:, 1], pc[:, 1]], axis=1)
    tri_boxes = np.stack(
        [xs.min(axis=1), xs.max(axis=1), ys.min(axis=1), ys.max(axis=1)],
        axis=1,
    )
    return cls(
        face_boxes=face_boxes,
        a=pa,
        u=u,
        v=v,
        inv_det=1.0 / det,
        tri_face=tri_face,
        tri_boxes=tri_boxes,
    )


def _min_distance_to_face_boundary(
    point: np.ndarray, face_pts: np.ndarray
) -> float:
  """Returns minimum Euclidean distance from `point` to a polygon's edges."""
  seg_start = face_pts
  seg_end = np.roll(face_pts, -1, axis=0)
  diff = seg_end - seg_start
  len_sq = np.sum(diff * diff, axis=1)
  rel = point[None, :] - seg_start
  safe_len_sq = np.where(len_sq > 1e-24, len_sq, 1.0)
  t = np.where(
      len_sq > 1e-24,
      np.clip(np.sum(rel * diff, axis=1) / safe_len_sq, 0.0, 1.0),
      0.0,
  )
  proj = seg_start + t[:, None] * diff
  return float(np.min(np.linalg.norm(point[None, :] - proj, axis=1)))


@dataclasses.dataclass(frozen=True)
class FoldedModel:
  """Deterministic 3D geometry of a folded crease pattern on the unit paper.

  Attributes:
    vertices_2d: `(V, 2)` planar vertex positions on the unit paper `[0, 1]^2`.
    vertices_3d: `(V, 3)` folded 3D vertex positions (unit-paper scale).
    faces: Tuple of 1D integer vertex-index arrays, one per bounded face.
    edges: `(E, 2)` integer edge vertex-index pairs.
    assignments: FOLD edge assignments (`'B'`, `'M'`, `'V'`, `'F'`, `'U'`).
    fold_angles: `(E,)` signed target dihedral fold angles in degrees.
    face_matrices: `(F, 4, 4)` rigid SE(3) transforms mapping unit-paper `(x, y,
      0, 1)` points to their folded 3D positions.
    anchor_face: Index of the stationary identity-transform face.
    vertex_error: Mean per-vertex axial edge-length strain in percent.
    non_ff_vertices: Count of interior vertices violating local flat-foldability
      (populated when `check_flat_foldability=True`).
    seconds: Wall-clock duration of folding in seconds.
  """

  vertices_2d: np.ndarray
  vertices_3d: np.ndarray
  faces: tuple[np.ndarray, ...]
  edges: np.ndarray
  assignments: tuple[FoldAssignment, ...]
  fold_angles: np.ndarray
  face_matrices: np.ndarray
  anchor_face: int = 0
  vertex_error: float = 0.0
  non_ff_vertices: int = 0
  seconds: float = 0.0

  def _triangle_index(self) -> _TriangleIndex:
    cached = self.__dict__.get('_tri_index')
    if cached is None:
      cached = _TriangleIndex.build(self.vertices_2d, self.faces)
      object.__setattr__(self, '_tri_index', cached)
    return cached

  def locate_faces(self, points: np.ndarray) -> np.ndarray:
    """Returns the face index containing each `(N, 2)` unit-paper point.

    Points outside every face return `-1`. Points lying on shared creases or
    within vertex-snapping tolerance of a face edge return an incident face.

    Args:
      points: `(N, 2)` array of unit-paper coordinates.

    Returns:
      `(N,)` integer array of face indices (`-1` when outside the sheet).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    result = np.full(len(pts), -1, dtype=int)
    if pts.size == 0 or not self.faces:
      return result

    idx = self._triangle_index()
    if len(idx.tri_face) > 0:
      box_tol = _EDGE_FALLBACK_TOL
      in_box = (
          (pts[:, None, 0] >= idx.tri_boxes[None, :, 0] - box_tol)
          & (pts[:, None, 0] <= idx.tri_boxes[None, :, 1] + box_tol)
          & (pts[:, None, 1] >= idx.tri_boxes[None, :, 2] - box_tol)
          & (pts[:, None, 1] <= idx.tri_boxes[None, :, 3] + box_tol)
      )
      p_idx, t_idx = np.nonzero(in_box)
      if len(p_idx) > 0:
        w = pts[p_idx] - idx.a[t_idx]
        s = (
            w[:, 0] * idx.v[t_idx, 1] - w[:, 1] * idx.v[t_idx, 0]
        ) * idx.inv_det[t_idx]
        t = (
            idx.u[t_idx, 0] * w[:, 1] - idx.u[t_idx, 1] * w[:, 0]
        ) * idx.inv_det[t_idx]
        hits = (
            (s >= -_BARYCENTRIC_TOL)
            & (t >= -_BARYCENTRIC_TOL)
            & (s + t <= 1.0 + _BARYCENTRIC_TOL)
        )
        hit_p = p_idx[hits]
        hit_f = idx.tri_face[t_idx[hits]]
        # Keep the first matching face per point deterministically.
        if len(hit_p) > 0:
          _, first_pos = np.unique(hit_p, return_index=True)
          result[hit_p[first_pos]] = hit_f[first_pos]

    missing = np.flatnonzero(result < 0)
    if len(missing) > 0:
      fb = idx.face_boxes
      tol = _EDGE_FALLBACK_TOL
      for m in missing:
        px, py = float(pts[m, 0]), float(pts[m, 1])
        cands = np.flatnonzero(
            (px >= fb[:, 0] - tol)
            & (px <= fb[:, 1] + tol)
            & (py >= fb[:, 2] - tol)
            & (py <= fb[:, 3] + tol)
        )
        best_face = -1
        best_dist = tol
        for f_raw in cands:
          f = int(f_raw)
          dist = _min_distance_to_face_boundary(
              pts[m], self.vertices_2d[self.faces[f]]
          )
          if dist <= best_dist:
            best_dist = dist
            best_face = f
        result[m] = best_face
    return result

  def locate(self, points: np.ndarray) -> np.ndarray:
    """Maps `(N, 2)` unit-paper points to `(N, 3)` folded positions."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    faces = self.locate_faces(pts)
    out = np.full((len(pts), 3), np.nan, dtype=float)
    found = faces >= 0
    if np.any(found):
      mats = self.face_matrices[faces[found]]
      out[found] = (
          np.einsum('nij,nj->ni', mats[:, :3, :2], pts[found]) + mats[:, :3, 3]
      )
    return out

  def locate_point(
      self, point: Sequence[float] | np.ndarray
  ) -> np.ndarray | None:
    """Maps a single `(x, y)` unit-paper point to `(3,)`, or `None`."""
    folded = self.locate(np.asarray(point, dtype=float).reshape(1, 2))[0]
    return None if bool(np.isnan(folded).any()) else folded

  def locate_groups(
      self, groups: Mapping[str, np.ndarray]
  ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Batches `locate` across named contours, returning `(means, spreads)`.

    Args:
      groups: Map from group name to `(N, 2)` unit-paper sample coordinates.

    Returns:
      `(means, spreads)` where `means[name]` is the `(3,)` centroid and
      `spreads[name]` is the RMS distance around that centroid. Groups with no
      points inside any face are omitted.
    """
    slices: list[tuple[str, int, int]] = []
    arrays: list[np.ndarray] = []
    offset = 0
    for name, raw in groups.items():
      arr = np.asarray(raw, dtype=float).reshape(-1, 2)
      if arr.size:
        arrays.append(arr)
      slices.append((name, offset, offset + len(arr)))
      offset += len(arr)
    folded = (
        self.locate(np.concatenate(arrays, axis=0))
        if arrays
        else np.zeros((0, 3), dtype=float)
    )
    means: dict[str, np.ndarray] = {}
    spreads: dict[str, float] = {}
    for name, start, end in slices:
      chunk = folded[start:end]
      valid = chunk[~np.isnan(chunk[:, 0])]
      if len(valid) == 0:
        continue
      mean = valid.mean(axis=0)
      means[name] = mean
      spreads[name] = math.sqrt(
          float(np.mean(np.sum((valid - mean) ** 2, axis=1)))
      )
    return means, spreads

  def to_fold_dict(self, title: str = 'model') -> dict[str, Any]:
    """Serializes the folded 3D geometry to a FOLD v1.2 `foldedForm` dict."""
    return {
        'file_spec': 1.2,
        'file_creator': 'bp_pipeline',
        'file_title': title,
        'file_classes': ['singleModel'],
        'frame_classes': ['foldedForm'],
        'frame_attributes': ['3D'],
        'vertices_coords': np.round(self.vertices_3d, 6).tolist(),
        'faces_vertices': [face.tolist() for face in self.faces],
        'edges_vertices': self.edges.tolist(),
        'edges_assignment': list(self.assignments),
        'edges_foldAngle': np.round(self.fold_angles, 4).tolist(),
    }


def _unwrap_fold_target(target: FoldTarget) -> cp_lib.CreasePattern:
  """Extracts a `CreasePattern` from a pattern or pipeline stage record."""
  if isinstance(target, bus.PipelineResult):
    if target.shaped is not None:
      return target.shaped.creases
    return target.solved.creases
  if isinstance(target, (bus.ShapedCreasePattern, bus.SolvedCreasePattern)):
    return target.creases
  if isinstance(target, cp_lib.CreasePattern):
    return target
  raise TypeError(
      'Expected CreasePattern, SolvedCreasePattern, ShapedCreasePattern, or '
      f'PipelineResult; got {type(target)!r}.'
  )


def fold_3d(
    target: FoldTarget,
    *,
    anchor_face: int | Callable[[FoldGraph], int] | None = None,
    vert_tol: float = 1e-6,
    add_border: bool = False,
    merge_collinear: bool = True,
    strict: bool = True,
    max_vertex_error: float = 10.0,
    check_flat_foldability: bool = False,
) -> FoldedModel:
  """Deterministically folds a crease pattern in 3D via BFS hinge rotations.

  Args:
    target: Crease pattern or solved/shaped pipeline stage object.
    anchor_face: Index of the fixed identity face, a selector callable taking
      the preprocessed `FoldGraph`, or `None` to anchor at face `0`.
    vert_tol: Vertex snapping and intersection tolerance in unit-paper units.
    add_border: If True, seeds the four `[0, 1]^2` sheet boundary segments.
    merge_collinear: If True, contracts redundant collinear degree-2 vertices.
    strict: If True, enforces a single manifold connected component and raises
      `ValueError` when `vertex_error > max_vertex_error`.
    max_vertex_error: Maximum permitted mean vertex strain (%) in strict mode.
    check_flat_foldability: If True, counts interior flat-foldability violations
      on `target` and stores the count in `FoldedModel.non_ff_vertices`.

  Returns:
    A `FoldedModel` on the unit square with rigid SE(3) face transforms.

  Raises:
    ValueError: If no faces exist, or if manifold, connectivity, or strain
      checks fail in `strict` mode.
  """
  pattern = _unwrap_fold_target(target)
  non_ff = 0
  if check_flat_foldability:
    _, _, non_ff_list = pattern.vertex_flat_foldability_percentage()
    non_ff = len(non_ff_list)

  start = time.perf_counter()
  graph = build_fold_graph(
      pattern,
      vert_tol=vert_tol,
      add_border=add_border,
      merge_collinear=merge_collinear,
      remove_dangling=True,
  )
  if not graph.faces:
    raise ValueError('No bounded planar faces found in crease pattern.')
  if strict:
    components = graph.num_face_components(check_manifold=True)
    if components != 1:
      raise ValueError(
          f'Crease pattern has {components} disconnected face components; '
          'expected a single connected component.'
      )

  if anchor_face is None:
    resolved_anchor = 0
  elif callable(anchor_face):
    resolved_anchor = int(anchor_face(graph))
  else:
    resolved_anchor = int(anchor_face)

  matrices, reached = _solve_face_transforms(
      graph, anchor_face=resolved_anchor, strict=strict
  )
  vertices_3d = _average_vertex_positions(
      graph.vertices_2d, graph.faces, matrices, reached
  )
  vertex_error, _ = compute_vertex_error(
      graph.vertices_2d, vertices_3d, graph.edges
  )
  seconds = time.perf_counter() - start
  if strict and vertex_error > max_vertex_error:
    raise ValueError(
        f'Fold vertex strain ({vertex_error:.4f}%) exceeds '
        f'max_vertex_error ({max_vertex_error:.4f}%).'
    )
  return FoldedModel(
      vertices_2d=graph.vertices_2d,
      vertices_3d=vertices_3d,
      faces=graph.faces,
      edges=graph.edges,
      assignments=graph.assignments,
      fold_angles=graph.fold_angles,
      face_matrices=matrices,
      anchor_face=resolved_anchor,
      vertex_error=vertex_error,
      non_ff_vertices=non_ff,
      seconds=seconds,
  )


def fold_svg(
    svg_string: str,
    *,
    vert_tol: float = 1e-6,
    max_vertex_error: float = 10.0,
) -> FoldedModel:
  """Parses an SVG crease pattern string and folds it in 3D."""
  pattern = svg_parser.parse_svg(svg_string, light_mode=True)
  return fold_3d(
      pattern,
      vert_tol=vert_tol,
      max_vertex_error=max_vertex_error,
  )


__all__ = [
    'FoldAssignment',
    'FoldGraph',
    'FoldTarget',
    'FoldedModel',
    'build_fold_graph',
    'compute_vertex_error',
    'crease_assignment_and_angle',
    'extract_planar_faces',
    'fold_3d',
    'fold_svg',
    'rotation_matrix_4x4',
]
