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

"""Pure-NumPy deterministic 3D origami folder and FOLD topology preprocessor."""

import collections
from collections.abc import Sequence
import dataclasses
import json
import logging
import math
import typing
from typing import Any, Literal
from xml.etree import ElementTree as ET

import networkx as nx
import numpy as np

from ..crease_pattern import crease as crease_lib
from ..crease_pattern import crease_pattern as cp_lib
from ..crease_pattern.parser import svg as svg_parser

_PRECISION = 3

Vertex2D = tuple[float, float]
EdgeVertices = tuple[int, int]
EdgeAssignment = Literal['B', 'M', 'V', 'F', 'U']


@dataclasses.dataclass(kw_only=True, frozen=True)
class _RawData:
  """Container for raw crease segments before FOLD graph construction."""

  vertices: list[Vertex2D]
  borders: list[tuple[int, int]]
  mountains: list[tuple[int, int, float, int]]
  valleys: list[tuple[int, int, float, int]]
  triangulations: list[tuple[int, int, int]]
  has_fold_steps: bool


def _parse_svg(svg_str: str) -> _RawData:
  """Parses an SVG crease pattern string into raw vertex and edge lists."""
  tree = ET.fromstring(svg_str)

  vertices: list[Vertex2D] = []
  vertex_map: dict[Vertex2D, int] = {}
  borders: list[tuple[int, int]] = []
  mountains: list[tuple[int, int, float, int]] = []
  valleys: list[tuple[int, int, float, int]] = []
  triangulations: list[tuple[int, int, int]] = []

  def add_crease(
      edge: tuple[int, int, float, int], fold_type: crease_lib.Fold
  ) -> None:
    match fold_type:
      case crease_lib.Fold.MOUNTAIN | crease_lib.Fold.MOUNTAIN_ZEBRA:
        mountains.append(edge)
      case crease_lib.Fold.VALLEY | crease_lib.Fold.VALLEY_ZEBRA:
        valleys.append(edge)
      case crease_lib.Fold.BORDER:
        borders.append(edge[:2])
      case crease_lib.Fold.TRIANGLE | crease_lib.Fold.HINGE:
        v1, v2, _, step = edge
        triangulations.append((v1, v2, step))
      case _:
        raise ValueError(f'Unsupported fold type: {fold_type}')

  def get_vertex_index(x: float, y: float) -> int:
    coord = (float(x), float(y))
    if coord not in vertex_map:
      vertex_map[coord] = len(vertices)
      vertices.append(coord)
    return vertex_map[coord]

  # Propagate group-level attributes (<g stroke=... opacity=...>) if present.
  inheritable = ('stroke', 'fold-step', 'opacity', 'stroke-opacity')
  for group in tree.iter():
    if group.tag.split('}')[-1] != 'g':
      continue
    if group.get('class') == 'bp-grid-lines':
      continue
    for child in group:
      for attr in inheritable:
        if attr in group.attrib and attr not in child.attrib:
          child.set(attr, group.attrib[attr])

  has_fold_steps = False
  for target_fold_type in (
      crease_lib.Fold.BORDER,
      crease_lib.Fold.MOUNTAIN,
      crease_lib.Fold.VALLEY,
      crease_lib.Fold.TRIANGLE,
  ):
    for elem in tree.iter():
      tag_name = elem.tag.split('}')[-1]
      if tag_name not in {'line', 'path', 'rect', 'polygon', 'polyline'}:
        continue
      stroke = elem.get('stroke', '')
      if not stroke:
        continue
      try:
        fold_type = svg_parser.get_fold(stroke)
      except svg_parser.SVGParseError:
        if tag_name == 'path':
          continue
        raise
      if fold_type == crease_lib.Fold.MOUNTAIN_ZEBRA:
        fold_type = crease_lib.Fold.MOUNTAIN
      elif fold_type == crease_lib.Fold.VALLEY_ZEBRA:
        fold_type = crease_lib.Fold.VALLEY
      elif fold_type == crease_lib.Fold.HINGE:
        fold_type = crease_lib.Fold.TRIANGLE
      if fold_type != target_fold_type:
        continue

      opacity = (
          elem.get('stroke-opacity', '') or elem.get('opacity', '') or '1'
      )
      angle = float(opacity) * math.degrees(fold_type.to_angle())
      if elem.get('fold-step') is not None:
        has_fold_steps = True
      fold_step = int(elem.get('fold-step', 0))

      if tag_name == 'line':
        x1 = float(elem.get('x1', 0))
        y1 = float(elem.get('y1', 0))
        x2 = float(elem.get('x2', 0))
        y2 = float(elem.get('y2', 0))
        v1 = get_vertex_index(x1, y1)
        v2 = get_vertex_index(x2, y2)
        add_crease((v1, v2, angle, fold_step), fold_type)
      elif tag_name == 'rect':
        x0 = float(elem.get('x', 0))
        y0 = float(elem.get('y', 0))
        width = float(elem.get('width', 0))
        height = float(elem.get('height', 0))
        if width < 0 or height < 0:
          raise ValueError(
              'Rect with negative dimensions is not supported:'
              f' width={width}, height={height}'
          )
        if width == 0 or height == 0:
          logging.warning(
              'Rect with zero width or height is not parsed. Skipping.'
          )
          continue
        v0 = get_vertex_index(x0, y0)
        v1 = get_vertex_index(x0 + width, y0)
        v2 = get_vertex_index(x0 + width, y0 + height)
        v3 = get_vertex_index(x0, y0 + height)
        add_crease((v0, v1, angle, fold_step), fold_type)
        add_crease((v1, v2, angle, fold_step), fold_type)
        add_crease((v2, v3, angle, fold_step), fold_type)
        add_crease((v3, v0, angle, fold_step), fold_type)
      else:
        raise ValueError(f'Unsupported element type: {elem.tag}')

  return _RawData(
      vertices=vertices,
      borders=borders,
      mountains=mountains,
      valleys=valleys,
      triangulations=triangulations,
      has_fold_steps=has_fold_steps,
  )


def _parse_crease_pattern(pattern: cp_lib.CreasePattern) -> _RawData:
  """Parses a `CreasePattern` into raw vertex and edge lists."""
  vertices: list[Vertex2D] = []
  vertex_map: dict[Vertex2D, int] = {}
  borders: list[tuple[int, int]] = []
  mountains: list[tuple[int, int, float, int]] = []
  valleys: list[tuple[int, int, float, int]] = []
  triangulations: list[tuple[int, int, int]] = []

  def get_vertex_index(x: float, y: float) -> int:
    coord = (round(x, _PRECISION), round(y, _PRECISION))
    if coord not in vertex_map:
      vertex_map[coord] = len(vertices)
      vertices.append(coord)
    return vertex_map[coord]

  def add_crease(
      edge: tuple[int, int, float, int], fold_type: crease_lib.Fold
  ) -> None:
    match fold_type:
      case crease_lib.Fold.MOUNTAIN | crease_lib.Fold.MOUNTAIN_ZEBRA:
        mountains.append(edge)
      case crease_lib.Fold.VALLEY | crease_lib.Fold.VALLEY_ZEBRA:
        valleys.append(edge)
      case crease_lib.Fold.BORDER:
        borders.append(edge[:2])
      case crease_lib.Fold.TRIANGLE | crease_lib.Fold.HINGE:
        v1, v2, _, step = edge
        triangulations.append((v1, v2, step))
      case _:
        raise ValueError(f'Unsupported fold type: {fold_type}')

  for c in pattern.creases:
    angle = math.degrees(c.fold_angle)
    v1 = get_vertex_index(c.p1.x, c.p1.y)
    v2 = get_vertex_index(c.p2.x, c.p2.y)
    add_crease((v1, v2, angle, c.fold_step), c.fold)

  return _RawData(
      vertices=vertices,
      borders=borders,
      mountains=mountains,
      valleys=valleys,
      triangulations=triangulations,
      has_fold_steps=True,
  )


def _geom_polygon_orientation(
    face_vertex_indices: Sequence[int], all_vertices_coords: Sequence[Vertex2D]
) -> float:
  """Calculates twice the signed polygon area using the shoelace formula."""
  if len(face_vertex_indices) < 3:
    return 0.0
  signed_double_area = 0.0
  num_verts = len(face_vertex_indices)
  for i in range(num_verts):
    p1_idx = face_vertex_indices[i]
    p2_idx = face_vertex_indices[(i + 1) % num_verts]
    x1, y1 = all_vertices_coords[p1_idx]
    x2, y2 = all_vertices_coords[p2_idx]
    signed_double_area += x1 * y2 - x2 * y1
  return signed_double_area


def edges_vertices_to_faces_vertices(
    vertices_coords: Sequence[Vertex2D], edges_vertices: Sequence[EdgeVertices]
) -> list[list[int]]:
  """Converts planar `edges_vertices` to bounded CCW `faces_vertices`."""
  vertices_vertices: list[list[int]] = [[] for _ in range(len(vertices_coords))]
  for v1, v2 in edges_vertices:
    vertices_vertices[v1].append(v2)
    vertices_vertices[v2].append(v1)

  for v in range(len(vertices_coords)):
    if not vertices_vertices[v]:
      continue
    x0, y0 = vertices_coords[v]
    vertices_vertices[v].sort(
        key=lambda nb, cx=x0, cy=y0: math.atan2(
            vertices_coords[nb][1] - cy, vertices_coords[nb][0] - cx
        )
    )

  next_map: dict[tuple[int, int], int | None] = {}
  for v, neighbors in enumerate(vertices_vertices):
    if not neighbors:
      continue
    n = len(neighbors)
    for i, u in enumerate(neighbors):
      next_map[(u, v)] = neighbors[(i - 1) % n]

  faces_vertices: list[list[int]] = []
  for uv in list(next_map):
    u, v = uv
    w = next_map[uv]
    if w is None:
      continue
    next_map[uv] = None
    face = [u, v]
    start = u
    while w != start:
      if w is None:
        logging.warning(
            'Face construction failed due to inconsistent graph structure.'
        )
        break
      face.append(w)
      new_uv = (v, w)
      v, w = w, next_map.get(new_uv)
      next_map[new_uv] = None
    else:
      if (face[-1], face[0]) in next_map:
        next_map[(face[-1], face[0])] = None
      if _geom_polygon_orientation(face, vertices_coords) > 1e-9:
        faces_vertices.append(face[::-1])

  return faces_vertices


def _convert_to_fold(raw: _RawData) -> dict[str, Any]:
  """Converts `_RawData` into initial FOLD arrays."""
  edges_vertices: list[tuple[int, int]] = []
  edges_assignment: list[EdgeAssignment] = []
  edges_fold_angle: list[float] = []
  edges_fold_step: list[int] = []

  for v1, v2 in raw.borders:
    edges_vertices.append((v1, v2))
    edges_assignment.append('B')
    edges_fold_angle.append(0.0)
    edges_fold_step.append(0)

  for v1, v2, angle, step in raw.mountains:
    edges_vertices.append((v1, v2))
    edges_assignment.append('M')
    edges_fold_angle.append(angle)
    edges_fold_step.append(step)

  for v1, v2, angle, step in raw.valleys:
    edges_vertices.append((v1, v2))
    edges_assignment.append('V')
    edges_fold_angle.append(angle)
    edges_fold_step.append(step)

  for v1, v2, step in raw.triangulations:
    edges_vertices.append((v1, v2))
    edges_assignment.append('F')
    edges_fold_angle.append(0.0)
    edges_fold_step.append(step)

  return {
      'vertices_coords': raw.vertices,
      'edges_vertices': edges_vertices,
      'edges_assignment': edges_assignment,
      'edges_foldAngle': edges_fold_angle,
      'faces_vertices': [],
      'edges_fold_step': edges_fold_step,
  }


def _line_intersect(
    v1: Vertex2D, v2: Vertex2D, v3: Vertex2D, v4: Vertex2D
) -> tuple[Vertex2D, float, float] | None:
  """Computes the parametric intersection of two 2D lines."""
  denom = (v1[0] - v2[0]) * (v3[1] - v4[1]) - (v1[1] - v2[1]) * (v3[0] - v4[0])
  if abs(denom) < 1e-10:
    return None
  t1 = (
      (v1[0] - v3[0]) * (v3[1] - v4[1]) - (v1[1] - v3[1]) * (v3[0] - v4[0])
  ) / denom
  t2 = (
      (v1[0] - v3[0]) * (v1[1] - v2[1]) - (v1[1] - v3[1]) * (v1[0] - v2[0])
  ) / denom
  ix = v1[0] + t1 * (v2[0] - v1[0])
  iy = v1[1] + t1 * (v2[1] - v1[1])
  return ((ix, iy), t1, t2)


def find_intersections(
    *,
    vertices_coords: list[list[float]],
    edges_vertices: list[list[int]],
    edges_foldAngle: list[float],  # pylint: disable=invalid-name
    edges_assignment: list[str],
    edges_fold_step: list[int] | None,
    tol: float,
) -> tuple[
    list[list[float]],
    list[list[int]],
    list[float],
    list[str],
    list[int] | None,
]:
  """Finds and splits interior edge-edge intersections within `tol`."""
  has_fold_steps = edges_fold_step is not None
  fold_steps = (
      list(edges_fold_step)
      if edges_fold_step is not None
      else [-1] * len(edges_vertices)
  )
  vertices = [list(v) for v in vertices_coords]
  edges = [list(e) for e in edges_vertices]
  fold_angles = list(edges_foldAngle)
  assignments = list(edges_assignment)

  i = len(edges) - 1
  while i >= 0:
    v1x, v1y = vertices[edges[i][0]][0], vertices[edges[i][0]][1]
    v2x, v2y = vertices[edges[i][1]][0], vertices[edges[i][1]][1]
    min_ix = (v1x if v1x < v2x else v2x) - tol
    max_ix = (v1x if v1x > v2x else v2x) + tol
    min_iy = (v1y if v1y < v2y else v2y) - tol
    max_iy = (v1y if v1y > v2y else v2y) + tol

    j = i - 1
    while j >= 0:
      v3x, v3y = vertices[edges[j][0]][0], vertices[edges[j][0]][1]
      v4x, v4y = vertices[edges[j][1]][0], vertices[edges[j][1]][1]
      if (
          (v3x if v3x > v4x else v4x) < min_ix
          or (v3x if v3x < v4x else v4x) > max_ix
          or (v3y if v3y > v4y else v4y) < min_iy
          or (v3y if v3y < v4y else v4y) > max_iy
      ):
        j -= 1
        continue

      hit = _line_intersect((v1x, v1y), (v2x, v2y), (v3x, v3y), (v4x, v4y))
      if hit is not None:
        intersection, t1, t2 = hit
        dx1, dy1 = v2x - v1x, v2y - v1y
        length1 = math.hypot(dx1, dy1)
        dx2, dy2 = v4x - v3x, v4y - v3y
        length2 = math.hypot(dx2, dy2)

        d1 = t1 * length1
        d2 = t2 * length2
        if (
            d1 < -tol
            or d1 > length1 + tol
            or d2 < -tol
            or d2 > length2 + tol
        ):
          j -= 1
          continue

        seg1_int = tol < d1 < length1 - tol
        seg2_int = tol < d2 < length2 - tol
        if not seg1_int and not seg2_int:
          j -= 1
          continue

        if seg1_int and seg2_int:
          vert_index = len(vertices)
          vertices.append([intersection[0], intersection[1]])
        elif seg1_int:
          vert_index = edges[j][0] if d2 <= tol else edges[j][1]
        else:
          vert_index = edges[i][0] if d1 <= tol else edges[i][1]

        if seg1_int:
          fa_i = fold_angles[i]
          as_i = assignments[i]
          fs_i = fold_steps[i]
          e_i0, e_i1 = edges[i][0], edges[i][1]
          edges[i : i + 1] = [[vert_index, e_i0], [vert_index, e_i1]]
          fold_angles[i : i + 1] = [fa_i, fa_i]
          assignments[i : i + 1] = [as_i, as_i]
          fold_steps[i : i + 1] = [fs_i, fs_i]
          i += 1
          v1x, v1y = vertices[edges[i][0]][0], vertices[edges[i][0]][1]
          v2x, v2y = vertices[edges[i][1]][0], vertices[edges[i][1]][1]
          min_ix = (v1x if v1x < v2x else v2x) - tol
          max_ix = (v1x if v1x > v2x else v2x) + tol
          min_iy = (v1y if v1y < v2y else v2y) - tol
          max_iy = (v1y if v1y > v2y else v2y) + tol

        if seg2_int:
          fa_j = fold_angles[j]
          as_j = assignments[j]
          fs_j = fold_steps[j]
          e_j0, e_j1 = edges[j][0], edges[j][1]
          edges[j : j + 1] = [[vert_index, e_j0], [vert_index, e_j1]]
          fold_angles[j : j + 1] = [fa_j, fa_j]
          assignments[j : j + 1] = [as_j, as_j]
          fold_steps[j : j + 1] = [fs_j, fs_j]
          j += 1
          i += 1

      j -= 1
    i -= 1

  return (
      vertices,
      edges,
      fold_angles,
      assignments,
      fold_steps if has_fold_steps else None,
  )


@dataclasses.dataclass(kw_only=True, frozen=True)
class FOLD:
  """Core FOLD representation for planar crease pattern preprocessing."""

  vertices_coords: list[Vertex2D]
  edges_vertices: list[EdgeVertices]
  edges_assignment: list[EdgeAssignment]
  edges_foldAngle: list[float]  # pylint: disable=invalid-name
  faces_vertices: list[list[int]]
  vertices_vertices: list[list[int]] | None = None
  faceOrders: list[list[int]] | None = None  # pylint: disable=invalid-name
  edges_fold_step: list[int] | None = None

  def __post_init__(self) -> None:
    self.validate()

  @classmethod
  def from_json(cls, json_str: str) -> 'FOLD':
    return cls.from_dict(json.loads(json_str))

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'FOLD':
    return cls(
        vertices_coords=[(float(v[0]), float(v[1])) for v in data['vertices_coords']],
        edges_vertices=[(int(v[0]), int(v[1])) for v in data['edges_vertices']],
        edges_assignment=list(data['edges_assignment']),
        edges_foldAngle=[float(a) for a in data['edges_foldAngle']],
        faces_vertices=[list(f) for f in data.get('faces_vertices', [])],
        faceOrders=data.get('faceOrders'),
        edges_fold_step=data.get('edges_fold_step'),
        vertices_vertices=data.get('vertices_vertices'),
    )

  def to_json(self, indent: int = 2) -> str:
    return json.dumps(
        dataclasses.asdict(
            self,
            dict_factory=lambda items: {k: v for k, v in items if v is not None},
        ),
        indent=indent,
    )

  def validate(self) -> None:
    if not all(len(v) == 2 for v in self.vertices_coords):
      raise ValueError('Vertices must be 2D.')
    num_vertices = len(self.vertices_coords)
    num_edges = len(self.edges_vertices)
    check_arrays: list[Sequence[Any]] = [
        self.edges_assignment,
        self.edges_foldAngle,
    ]
    if self.edges_fold_step is not None:
      check_arrays.append(self.edges_fold_step)
    if not all(len(arr) == num_edges for arr in check_arrays):
      raise ValueError(
          f'Array lengths do not match: {[len(arr) for arr in check_arrays]}.'
      )
    for e1, e2 in self.edges_vertices:
      if not (0 <= e1 < num_vertices and 0 <= e2 < num_vertices) or e1 == e2:
        raise ValueError(f'Edge vertex indices are not valid: {e1}, {e2}.')
    for face in self.faces_vertices:
      if not all(0 <= v < num_vertices for v in face):
        raise ValueError('Face vertex indices are not valid.')
    valid_assignments = typing.get_args(EdgeAssignment)
    if not all(a in valid_assignments for a in self.edges_assignment):
      raise ValueError(
          f'Edge assignments are not valid: {self.edges_assignment}'
      )

  def _get_edge_to_faces(self) -> dict[tuple[int, int], list[int]]:
    edge_to_faces: dict[tuple[int, int], list[int]] = collections.defaultdict(
        list
    )
    for f_idx, face_verts in enumerate(self.faces_vertices):
      for i in range(len(face_verts)):
        v1, v2 = face_verts[i], face_verts[(i + 1) % len(face_verts)]
        edge_key = (v1, v2) if v1 < v2 else (v2, v1)
        edge_to_faces[edge_key].append(f_idx)
    return edge_to_faces

  @property
  def dangling_edges(self) -> list[int]:
    edge_to_faces = self._get_edge_to_faces()
    return [
        i
        for i, (v1, v2) in enumerate(self.edges_vertices)
        if not edge_to_faces.get((v1, v2) if v1 < v2 else (v2, v1))
    ]

  @property
  def dangling_vertices(self) -> list[int]:
    vertices_in_faces: set[int] = set()
    for face_verts in self.faces_vertices:
      vertices_in_faces.update(face_verts)
    return [
        i
        for i in range(len(self.vertices_coords))
        if i not in vertices_in_faces
    ]

  @property
  def num_face_components(self) -> int:
    if not self.faces_vertices:
      return 0
    edge_to_faces = self._get_edge_to_faces()
    for edge_key, faces in edge_to_faces.items():
      if len(faces) > 2:
        raise ValueError(
            f'Non-manifold edge {edge_key} is shared by {len(faces)} faces:'
            f' {faces}.'
        )
    graph = nx.Graph()
    graph.add_nodes_from(range(len(self.faces_vertices)))
    for faces in edge_to_faces.values():
      if len(faces) == 2:
        graph.add_edge(faces[0], faces[1])
    return nx.number_connected_components(graph)

  @property
  def is_fully_connected(self) -> bool:
    return self.num_face_components == 1

  def remove_dangling_edges(self) -> 'FOLD':
    if not self.dangling_edges:
      return self
    dangling_set = set(self.dangling_edges)
    edges_vertices: list[tuple[int, int]] = []
    edges_assignment: list[EdgeAssignment] = []
    edges_fold_angle: list[float] = []
    edges_fold_step: list[int] | None = (
        [] if self.edges_fold_step is not None else None
    )
    for idx in range(len(self.edges_vertices)):
      if idx not in dangling_set:
        edges_vertices.append(self.edges_vertices[idx])
        edges_assignment.append(self.edges_assignment[idx])
        edges_fold_angle.append(self.edges_foldAngle[idx])
        if edges_fold_step is not None and self.edges_fold_step is not None:
          edges_fold_step.append(self.edges_fold_step[idx])
    return dataclasses.replace(
        self,
        edges_vertices=edges_vertices,
        edges_assignment=edges_assignment,
        edges_foldAngle=edges_fold_angle,
        edges_fold_step=edges_fold_step,
    )

  def remove_dangling_vertices(self) -> 'FOLD':
    if not self.dangling_vertices:
      return self
    dangling_set = set(self.dangling_vertices)
    keep = [
        idx
        for idx in range(len(self.vertices_coords))
        if idx not in dangling_set
    ]
    old_to_new = {old: new for new, old in enumerate(keep)}
    vertices_vertices = None
    if self.vertices_vertices is not None:
      vertices_vertices = [
          [
              old_to_new[v]
              for v in self.vertices_vertices[idx]
              if v not in dangling_set
          ]
          for idx in keep
      ]
    return dataclasses.replace(
        self,
        vertices_coords=[self.vertices_coords[idx] for idx in keep],
        edges_vertices=[
            (old_to_new[e[0]], old_to_new[e[1]]) for e in self.edges_vertices
        ],
        faces_vertices=[
            [old_to_new[v] for v in face] for face in self.faces_vertices
        ],
        vertices_vertices=vertices_vertices,
    )

  @classmethod
  def from_raw_data(cls, raw_data: _RawData, vert_tol: float | None) -> 'FOLD':
    fold_data = _convert_to_fold(raw_data)
    fold = FOLD(**fold_data)
    if vert_tol is not None:
      fold = preprocess(fold, vert_tol=vert_tol)
    faces_vertices = edges_vertices_to_faces_vertices(
        fold.vertices_coords, fold.edges_vertices
    )
    if not raw_data.has_fold_steps:
      fold = dataclasses.replace(fold, edges_fold_step=None)
    return dataclasses.replace(fold, faces_vertices=faces_vertices)

  @classmethod
  def from_svg(cls, svg_str: str, vert_tol: float | None = 1e-3) -> 'FOLD':
    return cls.from_raw_data(_parse_svg(svg_str), vert_tol=vert_tol)

  @classmethod
  def from_crease_pattern(
      cls, pattern: cp_lib.CreasePattern, vert_tol: float | None = 1e-3
  ) -> 'FOLD':
    return cls.from_raw_data(_parse_crease_pattern(pattern), vert_tol=vert_tol)


def _remove_duplicate_edges_vertices(fold: FOLD) -> FOLD:
  """Removes duplicate edges using assignment precedence `M, V > F > B > U`."""
  edge_map: dict[tuple[int, int], int] = {}
  new_edges: list[tuple[int, int]] = []
  new_fold_angles: list[float] = []
  new_assignments: list[EdgeAssignment] = []
  new_fold_steps: list[int] = []
  precedence = {'M': 3, 'V': 3, 'F': 2, 'B': 1, 'U': 0}

  for idx, edge in enumerate(fold.edges_vertices):
    sorted_edge = (edge[0], edge[1]) if edge[0] < edge[1] else (edge[1], edge[0])
    current_assignment = fold.edges_assignment[idx]
    step_val = fold.edges_fold_step[idx] if fold.edges_fold_step is not None else -1
    if sorted_edge in edge_map:
      target_idx = edge_map[sorted_edge]
      old_assignment = new_assignments[target_idx]
      old_prec = precedence.get(old_assignment, 0)
      new_prec = precedence.get(current_assignment, 0)
      if new_prec > old_prec:
        new_edges[target_idx] = edge
        new_fold_angles[target_idx] = fold.edges_foldAngle[idx]
        new_assignments[target_idx] = current_assignment
        new_fold_steps[target_idx] = step_val
    else:
      edge_map[sorted_edge] = len(new_edges)
      new_edges.append(edge)
      new_fold_angles.append(fold.edges_foldAngle[idx])
      new_assignments.append(current_assignment)
      new_fold_steps.append(step_val)

  return dataclasses.replace(
      fold,
      edges_vertices=new_edges,
      edges_foldAngle=new_fold_angles,
      edges_assignment=new_assignments,
      edges_fold_step=new_fold_steps if fold.edges_fold_step is not None else None,
  )


def _collapse_nearby_vertices(fold: FOLD, tolerance: float) -> FOLD:
  """Merges vertices within `tolerance` distance using a uniform spatial grid."""
  if tolerance <= 0:
    return fold
  vertices = [list(v) for v in fold.vertices_coords]
  old2new: list[int] = []
  unique_vertices: list[list[float]] = []
  inv_cell = 1.0 / tolerance
  grid: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
  cluster_cell: list[tuple[int, int]] = []
  tol_sq = tolerance * tolerance

  for vx, vy in vertices:
    cx = int(math.floor(vx * inv_cell))
    cy = int(math.floor(vy * inv_cell))
    best_cluster_idx = None
    for dx in (-1, 0, 1):
      for dy in (-1, 0, 1):
        cell_clusters = grid.get((cx + dx, cy + dy))
        if not cell_clusters:
          continue
        for c_idx in cell_clusters:
          uv = unique_vertices[c_idx]
          if (vx - uv[0]) ** 2 + (vy - uv[1]) ** 2 < tol_sq:
            if best_cluster_idx is None or c_idx < best_cluster_idx:
              best_cluster_idx = c_idx

    if best_cluster_idx is not None:
      old2new.append(best_cluster_idx)
      old_cx, old_cy = cluster_cell[best_cluster_idx]
      unique_vertices[best_cluster_idx] = [vx, vy]
      if (old_cx, old_cy) != (cx, cy):
        grid[(old_cx, old_cy)].remove(best_cluster_idx)
        grid[(cx, cy)].append(best_cluster_idx)
        cluster_cell[best_cluster_idx] = (cx, cy)
    else:
      c_idx = len(unique_vertices)
      old2new.append(c_idx)
      unique_vertices.append([vx, vy])
      grid[(cx, cy)].append(c_idx)
      cluster_cell.append((cx, cy))

  new_edges: list[tuple[int, int]] = []
  new_fold_angles: list[float] = []
  new_assignments: list[EdgeAssignment] = []
  new_fold_steps: list[int] | None = (
      [] if fold.edges_fold_step is not None else None
  )
  for idx, edge in enumerate(fold.edges_vertices):
    v1 = old2new[edge[0]]
    v2 = old2new[edge[1]]
    if v1 != v2:
      new_edges.append((v1, v2))
      new_fold_angles.append(fold.edges_foldAngle[idx])
      new_assignments.append(fold.edges_assignment[idx])
      if new_fold_steps is not None and fold.edges_fold_step is not None:
        new_fold_steps.append(fold.edges_fold_step[idx])

  return dataclasses.replace(
      fold,
      vertices_coords=[(v[0], v[1]) for v in unique_vertices],
      edges_vertices=new_edges,
      edges_foldAngle=new_fold_angles,
      edges_assignment=new_assignments,
      edges_fold_step=new_fold_steps,
  )


def _remove_loop_edges(fold: FOLD) -> FOLD:
  """Removes self-loop edges `(v, v)`."""
  new_edges: list[tuple[int, int]] = []
  new_fold_angles: list[float] = []
  new_assignments: list[EdgeAssignment] = []
  new_fold_steps: list[int] | None = (
      [] if fold.edges_fold_step is not None else None
  )
  for idx, edge in enumerate(fold.edges_vertices):
    if edge[0] != edge[1]:
      new_edges.append(edge)
      new_fold_angles.append(fold.edges_foldAngle[idx])
      new_assignments.append(fold.edges_assignment[idx])
      if new_fold_steps is not None and fold.edges_fold_step is not None:
        new_fold_steps.append(fold.edges_fold_step[idx])
  return dataclasses.replace(
      fold,
      edges_vertices=new_edges,
      edges_foldAngle=new_fold_angles,
      edges_assignment=new_assignments,
      edges_fold_step=new_fold_steps,
  )


def _build_vertices_vertices(fold: FOLD) -> list[list[int]]:
  vertices_vertices: list[list[int]] = [
      [] for _ in range(len(fold.vertices_coords))
  ]
  for v1, v2 in fold.edges_vertices:
    if v2 not in vertices_vertices[v1]:
      vertices_vertices[v1].append(v2)
    if v1 not in vertices_vertices[v2]:
      vertices_vertices[v2].append(v1)
  return vertices_vertices


def _remove_stray_vertices(
    fold: FOLD, vertices_vertices: list[list[int]]
) -> tuple[FOLD, list[list[int]]]:
  """Removes isolated degree-0 vertices."""
  old2new: list[int | None] = []
  new_index = 0
  num_strays = 0
  for neighbors in vertices_vertices:
    if not neighbors:
      old2new.append(None)
      num_strays += 1
    else:
      old2new.append(new_index)
      new_index += 1
  if num_strays == 0:
    return fold, vertices_vertices

  new_vertices = [
      coord
      for i, coord in enumerate(fold.vertices_coords)
      if old2new[i] is not None
  ]
  new_edges: list[tuple[int, int]] = []
  new_fold_angles: list[float] = []
  new_assignments: list[EdgeAssignment] = []
  new_fold_steps: list[int] = []
  for idx, edge in enumerate(fold.edges_vertices):
    v1_new = old2new[edge[0]]
    v2_new = old2new[edge[1]]
    if v1_new is not None and v2_new is not None:
      new_edges.append((v1_new, v2_new))
      new_fold_angles.append(fold.edges_foldAngle[idx])
      new_assignments.append(fold.edges_assignment[idx])
      if fold.edges_fold_step is not None:
        new_fold_steps.append(fold.edges_fold_step[idx])

  new_vv = [
      [old2new[n] for n in neighbors if old2new[n] is not None]
      for i, neighbors in enumerate(vertices_vertices)
      if old2new[i] is not None
  ]
  return (
      dataclasses.replace(
          fold,
          vertices_coords=new_vertices,
          edges_vertices=new_edges,
          edges_foldAngle=new_fold_angles,
          edges_assignment=new_assignments,
          edges_fold_step=(
              new_fold_steps if fold.edges_fold_step is not None else None
          ),
      ),
      new_vv,
  )


def _merge_edge(
    fold: FOLD,
    vertices_vertices: list[list[int]],
    v1: int,
    v2: int,
    v3: int,
) -> tuple[bool, FOLD, list[list[int]]]:
  """Merges collinear edges `(v1, v2)` and `(v2, v3)` when compatible."""
  angle_avg = 0.0
  avg_sum = 0
  edge_assignment: EdgeAssignment | None = None
  fold_step: int | None = None
  edge_indices: list[int] = []

  for i in range(len(fold.edges_vertices) - 1, -1, -1):
    edge = fold.edges_vertices[i]
    if v2 in edge and (v1 in edge or v3 in edge):
      cur_step = (
          fold.edges_fold_step[i] if fold.edges_fold_step is not None else -1
      )
      if edge_assignment is None:
        edge_assignment = fold.edges_assignment[i]
        fold_step = cur_step
      elif edge_assignment != fold.edges_assignment[i]:
        return False, fold, vertices_vertices
      elif fold_step != cur_step:
        return False, fold, vertices_vertices
      angle = fold.edges_foldAngle[i]
      if angle:
        angle_avg += angle
        avg_sum += 1
      edge_indices.append(i)

  assert edge_assignment is not None
  edge_index_set = set(edge_indices)
  new_edges: list[tuple[int, int]] = []
  new_fold_angles: list[float] = []
  new_assignments: list[EdgeAssignment] = []
  new_fold_steps: list[int] = []
  for i, edge in enumerate(fold.edges_vertices):
    if i not in edge_index_set:
      new_edges.append(edge)
      new_fold_angles.append(fold.edges_foldAngle[i])
      new_assignments.append(fold.edges_assignment[i])
      if fold.edges_fold_step is not None:
        new_fold_steps.append(fold.edges_fold_step[i])

  new_edges.append((v1, v3))
  new_assignments.append(edge_assignment)
  if fold.edges_fold_step is not None:
    new_fold_steps.append(fold_step if fold_step is not None else -1)
  new_fold_angles.append(angle_avg / avg_sum if avg_sum > 0 else 0.0)

  new_vv = [list(neighbors) for neighbors in vertices_vertices]
  new_vv[v1].pop(new_vv[v1].index(v2))
  new_vv[v1].append(v3)
  new_vv[v3].pop(new_vv[v3].index(v2))
  new_vv[v3].append(v1)

  return (
      True,
      dataclasses.replace(
          fold,
          edges_vertices=new_edges,
          edges_foldAngle=new_fold_angles,
          edges_assignment=new_assignments,
          edges_fold_step=(
              new_fold_steps if fold.edges_fold_step is not None else None
          ),
      ),
      new_vv,
  )


def _remove_redundant_vertices(
    fold: FOLD, vertices_vertices: list[list[int]], epsilon: float = 0.01
) -> tuple[FOLD, list[list[int]]]:
  """Removes collinear degree-2 vertices."""
  old2new: list[int | None] = []
  num_redundant = 0
  new_index = 0

  for i in range(len(vertices_vertices)):
    nbs = vertices_vertices[i]
    if len(nbs) != 2:
      old2new.append(new_index)
      new_index += 1
      continue
    vx, vy = fold.vertices_coords[i]
    n0x, n0y = fold.vertices_coords[nbs[0]]
    n1x, n1y = fold.vertices_coords[nbs[1]]
    v0x, v0y = n0x - vx, n0y - vy
    v1x, v1y = n1x - vx, n1y - vy
    denom = math.sqrt((v0x * v0x + v0y * v0y) * (v1x * v1x + v1y * v1y))
    if denom > 0 and abs((v0x * v1x + v0y * v1y) / denom + 1.0) < epsilon:
      merged, fold, vertices_vertices = _merge_edge(
          fold, vertices_vertices, nbs[0], i, nbs[1]
      )
      if merged:
        num_redundant += 1
        old2new.append(None)
        continue
    old2new.append(new_index)
    new_index += 1

  if num_redundant == 0:
    return fold, vertices_vertices

  new_vertices = [
      coord
      for i, coord in enumerate(fold.vertices_coords)
      if old2new[i] is not None
  ]
  new_edges: list[tuple[int, int]] = []
  new_fold_angles: list[float] = []
  new_assignments: list[EdgeAssignment] = []
  new_fold_steps: list[int] = []
  for idx, edge in enumerate(fold.edges_vertices):
    v1_new = old2new[edge[0]] if edge[0] < len(old2new) else None
    v2_new = old2new[edge[1]] if edge[1] < len(old2new) else None
    if v1_new is not None and v2_new is not None:
      new_edges.append((v1_new, v2_new))
      new_fold_angles.append(fold.edges_foldAngle[idx])
      new_assignments.append(fold.edges_assignment[idx])
      if fold.edges_fold_step is not None:
        new_fold_steps.append(fold.edges_fold_step[idx])

  new_vv = [
      [
          old2new[n]
          for n in neighbors
          if n < len(old2new) and old2new[n] is not None
      ]
      for i, neighbors in enumerate(vertices_vertices)
      if old2new[i] is not None
  ]
  return (
      dataclasses.replace(
          fold,
          vertices_coords=new_vertices,
          edges_vertices=new_edges,
          edges_foldAngle=new_fold_angles,
          edges_assignment=new_assignments,
          edges_fold_step=(
              new_fold_steps if fold.edges_fold_step is not None else None
          ),
      ),
      new_vv,
  )


def _sort_vertices_vertices_ccw(
    fold: FOLD, vertices_vertices: list[list[int]]
) -> list[list[int]]:
  new_vv: list[list[int]] = []
  for i, neighbors in enumerate(vertices_vertices):
    if len(neighbors) <= 1:
      new_vv.append(list(neighbors))
      continue
    cx, cy = fold.vertices_coords[i]
    new_vv.append(
        sorted(
            neighbors,
            key=lambda nb, x0=cx, y0=cy: math.atan2(
                fold.vertices_coords[nb][1] - y0,
                fold.vertices_coords[nb][0] - x0,
            ),
        )
    )
  return new_vv


def preprocess(
    fold: FOLD,
    *,
    vert_tol: float = 0.01,
    collinear_tol: float = 0.01,
) -> FOLD:
  """Merges nearby vertices, splits crossings, and cleans planar topology."""
  has_fold_steps = fold.edges_fold_step is not None
  if not has_fold_steps:
    fold = dataclasses.replace(
        fold, edges_fold_step=[-1] * len(fold.edges_vertices)
    )

  fold = _collapse_nearby_vertices(fold, vert_tol)
  fold = _remove_loop_edges(fold)
  fold = _remove_duplicate_edges_vertices(fold)

  vertices, edges, angles, assignments, fold_steps = find_intersections(
      vertices_coords=[list(v) for v in fold.vertices_coords],
      edges_vertices=[list(v) for v in fold.edges_vertices],
      edges_foldAngle=fold.edges_foldAngle,
      edges_assignment=list(fold.edges_assignment),
      edges_fold_step=fold.edges_fold_step,
      tol=vert_tol,
  )
  fold = dataclasses.replace(
      fold,
      vertices_coords=[(v[0], v[1]) for v in vertices],
      edges_vertices=[(e[0], e[1]) for e in edges],
      edges_foldAngle=angles,
      edges_assignment=typing.cast(list[EdgeAssignment], assignments),
      edges_fold_step=fold_steps,
  )

  fold = _collapse_nearby_vertices(fold, vert_tol)
  fold = _remove_loop_edges(fold)
  fold = _remove_duplicate_edges_vertices(fold)

  vertices_vertices = _build_vertices_vertices(fold)
  fold, vertices_vertices = _remove_stray_vertices(fold, vertices_vertices)
  fold, vertices_vertices = _remove_redundant_vertices(
      fold, vertices_vertices, collinear_tol
  )
  vertices_vertices = _sort_vertices_vertices_ccw(fold, vertices_vertices)

  if not has_fold_steps:
    fold = dataclasses.replace(fold, edges_fold_step=None)
  return dataclasses.replace(fold, vertices_vertices=vertices_vertices)


def rotation_matrix_around_axis(axis: np.ndarray, theta: float) -> np.ndarray:
  """Creates a 4x4 counterclockwise Rodrigues rotation matrix around `axis`."""
  axis = np.asarray(axis, dtype=float)
  norm = float(np.sqrt(np.dot(axis, axis)))
  if np.isclose(norm, 0.0):
    return np.identity(4)
  axis = axis / norm
  a = math.cos(theta / 2.0)
  b, c, d = -axis * math.sin(theta / 2.0)
  aa, bb, cc, dd = a * a, b * b, c * c, d * d
  bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
  return np.array([
      [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0.0],
      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0.0],
      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0.0],
      [0.0, 0.0, 0.0, 1.0],
  ])


def _translation_matrix(v: np.ndarray) -> np.ndarray:
  m = np.identity(4)
  m[:3, 3] = v
  return m


def solve_fold(fold_data: dict[str, Any]) -> list[np.ndarray]:
  """Computes global 4x4 face transformation matrices via BFS hinge folding."""
  faces = fold_data['faces_vertices']
  vertices = np.asarray(fold_data['vertices_coords'], dtype=float)
  edges = fold_data['edges_vertices']
  fold_angles = fold_data['edges_foldAngle']

  if not faces:
    raise ValueError('No faces found in fold_data.')

  edge_map: dict[tuple[int, int], int] = {}
  for i, (v1, v2) in enumerate(edges):
    edge_map[(v1, v2) if v1 < v2 else (v2, v1)] = i

  edge_to_faces: list[list[int]] = [[] for _ in range(len(edges))]
  for f_idx, face_verts in enumerate(faces):
    for i in range(len(face_verts)):
      v1, v2 = face_verts[i], face_verts[(i + 1) % len(face_verts)]
      edge_key = (v1, v2) if v1 < v2 else (v2, v1)
      if edge_key in edge_map:
        edge_to_faces[edge_map[edge_key]].append(f_idx)

  face_matrices: list[np.ndarray | None] = [None] * len(faces)
  visited = {0}
  queue = collections.deque([0])
  face_matrices[0] = np.identity(4)

  while queue:
    curr_f_idx = queue.popleft()
    curr_matrix = face_matrices[curr_f_idx]
    assert curr_matrix is not None

    face_verts = faces[curr_f_idx]
    for i in range(len(face_verts)):
      v1_idx, v2_idx = face_verts[i], face_verts[(i + 1) % len(face_verts)]
      edge_key = (v1_idx, v2_idx) if v1_idx < v2_idx else (v2_idx, v1_idx)
      if edge_key not in edge_map:
        continue
      edge_idx = edge_map[edge_key]
      neighbors = edge_to_faces[edge_idx]
      next_f_idx = next((n for n in neighbors if n != curr_f_idx), None)
      if next_f_idx is not None and next_f_idx not in visited:
        p1_local = np.array(
            [vertices[v1_idx, 0], vertices[v1_idx, 1], 0.0, 1.0]
        )
        p2_local = np.array(
            [vertices[v2_idx, 0], vertices[v2_idx, 1], 0.0, 1.0]
        )
        p1_global = curr_matrix @ p1_local
        p2_global = curr_matrix @ p2_local
        axis = p2_global[:3] - p1_global[:3]
        angle_rad = math.radians(fold_angles[edge_idx])

        pivot = _translation_matrix(p1_global[:3])
        pivot_inv = _translation_matrix(-p1_global[:3])
        rotation = rotation_matrix_around_axis(axis, angle_rad)
        face_matrices[next_f_idx] = pivot @ rotation @ pivot_inv @ curr_matrix
        visited.add(next_f_idx)
        queue.append(next_f_idx)

  none_indices = [i for i, m in enumerate(face_matrices) if m is None]
  if none_indices:
    raise ValueError(
        f'Disconnected faces detected: faces {none_indices} were not reachable'
        f' from face 0 during BFS traversal. Total faces: {len(faces)}.'
    )
  return typing.cast(list[np.ndarray], face_matrices)


def _compute_3d_geometry(
    vertices_2d: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    matrices: Sequence[np.ndarray],
) -> list[tuple[float, float, float]]:
  """Computes mean 3D folded coordinates per vertex across incident faces."""
  num_vertices = len(vertices_2d)
  vertex_sums = np.zeros((num_vertices, 3), dtype=float)
  vertex_counts = np.zeros(num_vertices, dtype=float)

  for f_idx, face_indices in enumerate(faces):
    matrix = matrices[f_idx]
    for v_idx in face_indices:
      coord = np.array(
          [vertices_2d[v_idx][0], vertices_2d[v_idx][1], 0.0, 1.0]
      )
      transformed = matrix @ coord
      vertex_sums[v_idx] += transformed[:3]
      vertex_counts[v_idx] += 1.0

  vertices_3d: list[tuple[float, float, float]] = []
  for v_idx in range(num_vertices):
    if vertex_counts[v_idx] > 0:
      avg = vertex_sums[v_idx] / vertex_counts[v_idx]
      vertices_3d.append((float(avg[0]), float(avg[1]), float(avg[2])))
    else:
      vertices_3d.append(
          (float(vertices_2d[v_idx][0]), float(vertices_2d[v_idx][1]), 0.0)
      )
  return vertices_3d


def compute_vertex_error(
    vertices_2d: Sequence[Sequence[float]],
    vertices_3d: Sequence[Sequence[float]],
    edges: Sequence[Sequence[int]],
) -> tuple[float, np.ndarray]:
  """Computes mean (%) and per-vertex axial edge-length strain errors."""
  vertices_2d_arr = np.asarray(vertices_2d, dtype=float)
  vertices_3d_arr = np.asarray(vertices_3d, dtype=float)
  edges_arr = np.asarray(edges, dtype=int)
  num_vertices = len(vertices_2d_arr)
  if num_vertices == 0 or len(edges_arr) == 0:
    return 0.0, np.zeros((num_vertices, 1), dtype=float)

  len_2d = np.linalg.norm(
      vertices_2d_arr[edges_arr[:, 0]] - vertices_2d_arr[edges_arr[:, 1]],
      axis=1,
  )
  len_3d = np.linalg.norm(
      vertices_3d_arr[edges_arr[:, 0]] - vertices_3d_arr[edges_arr[:, 1]],
      axis=1,
  )
  axial_errors = np.abs(len_3d / len_2d - 1.0)

  vertex_errors = np.zeros(num_vertices, dtype=float)
  vertex_count = np.zeros(num_vertices, dtype=float)
  np.add.at(vertex_errors, edges_arr[:, 0], axial_errors)
  np.add.at(vertex_errors, edges_arr[:, 1], axial_errors)
  np.add.at(vertex_count, edges_arr[:, 0], 1.0)
  np.add.at(vertex_count, edges_arr[:, 1], 1.0)

  per_vertex_errors = (vertex_errors / vertex_count)[:, None]
  mean_error = float(per_vertex_errors.mean() * 100.0)
  return mean_error, per_vertex_errors


def get_fold(
    input_data: str | FOLD,
    *,
    vert_tol: float | None = 1e-3,
    remove_dangling: bool = True,
    simulation_tolerance: float = 1.0,
) -> dict[str, Any]:
  """Folds an SVG string or `FOLD` object deterministically in 3D."""
  fold_obj = (
      input_data
      if isinstance(input_data, FOLD)
      else FOLD.from_svg(input_data, vert_tol=vert_tol)
  )
  if not fold_obj.is_fully_connected:
    raise ValueError(
        f'Crease pattern has {fold_obj.num_face_components} disconnected face'
        ' components. The folder requires all faces to be connected via shared'
        ' edges.'
    )
  if remove_dangling:
    fold_obj = fold_obj.remove_dangling_edges().remove_dangling_vertices()

  fold_dict = json.loads(fold_obj.to_json())
  matrices = solve_fold(fold_dict)
  vertices_2d = fold_dict['vertices_coords']
  faces = fold_dict['faces_vertices']
  vertices_3d = _compute_3d_geometry(vertices_2d, faces, matrices)
  vertex_errors, per_vertex_errors = compute_vertex_error(
      vertices_2d=vertices_2d,
      vertices_3d=vertices_3d,
      edges=fold_dict['edges_vertices'],
  )
  if vertex_errors > simulation_tolerance:
    raise ValueError(
        'Deterministic fold produced higher error than simulation tolerance'
        f' ({vertex_errors} > {simulation_tolerance}).'
    )
  return {
      'vertices_coords': vertices_3d,
      'vertices_coords_2d': vertices_2d,
      'edges_vertices': fold_dict['edges_vertices'],
      'edges_assignment': fold_dict['edges_assignment'],
      'edges_foldAngle': fold_dict['edges_foldAngle'],
      'faces_vertices': faces,
      'simulator_auxs': {
          'matrices': matrices,
          'vertex_errors': vertex_errors,
          'per_vertex_errors': per_vertex_errors,
      },
  }
