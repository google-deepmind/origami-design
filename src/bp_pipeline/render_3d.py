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

"""Pure-Python 3D hinge-rotation mesh folder and Matplotlib shaded renderer."""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import math
import pathlib
from typing import Any, Literal

from matplotlib import axes
from matplotlib import collections as mpl_collections
from matplotlib import patches
import matplotlib.pyplot as plt

from . import bus
from . import folder_3d as fold_3d
from .crease_pattern import crease as crease_lib
from .crease_pattern import crease_pattern as cp_lib

_DEFAULT_PAPER_RGB = (0.85, 0.745, 0.596)

_TURNAROUND_VIEWS: tuple[tuple[str, float, float, float], ...] = (
    ('1. 3D Isometric (3/4 View)', -0.62, 0.38, 0.0),
    ('2. Side Profile View', -1.32, 0.22, 0.0),
    ('3. Front-Oblique View', -0.22, 0.20, 0.0),
    ('4. Top-Down Dorsal View', -0.20, 1.22, 0.0),
)


@dataclasses.dataclass(frozen=True)
class MeshRegion:
  """A packed flap, river, or reserved region that mesh faces can belong to.

  Attributes:
    kind: `'flap'` for a leaf limb, `'river'` for an internal one, or
      `'reserved'` for non-limb paper reserved by packing.
    sticks: Names of the source stick-figure sticks the region realizes, in
      chain order when packing merged several sticks into one limb. Empty for
      `'reserved'` regions.
  """

  kind: Literal['flap', 'river', 'reserved']
  sticks: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class FoldedMesh3D:
  """Planar crease-pattern topology and its 3D BFS hinge-folded geometry.

  The fields from `grid_size` on describe where the mesh came from in the
  pipeline. They keep their empty defaults when the mesh is built from an
  object that does not carry that context.

  Attributes:
    vertices_2d: Flat vertices in `CreasePattern` coordinates minus 0.5, so the
      sheet is centered on the origin and y points down.
    vertices_3d: The same vertices in the folded model.
    faces: Vertex indices around each planar face.
    edges: Vertex index pairs, parallel to the `edges_*` fields.
    edges_assignment: FOLD assignment (`'M'`, `'V'`, `'F'` or `'B'`) per edge.
    edges_fold_angle: Signed fold angle in degrees per edge, valley positive.
    edge_key_angle: Fold angle keyed by sorted vertex index pair.
    grid_size: Box-pleating grid size of the packed layout, if known.
    regions: Flaps and rivers of the packed layout.
    faces_region: Per face, the index into `regions` of the flap or river that
      contains it, or None. Empty when `regions` is empty.
    edges_shaping: Per edge, whether shaping added the crease instead of
      inheriting it from the solved base. Empty unless the mesh is built from a
      `ShapedCreasePattern` (or `PipelineResult` with `.shaped`).
    edges_base_fold_angle: Per edge, signed fold angle in degrees in the solved
      flat-foldable base (`-180.0`, `0.0`, or `180.0`), preserving original
      base signs and magnitudes when shaping bends or flips a base crease.
      Empty unless built from a `ShapedCreasePattern` (or `PipelineResult` with
      `.shaped`).
  """

  vertices_2d: list[tuple[float, float]]
  vertices_3d: list[tuple[float, float, float]]
  faces: list[list[int]]
  edges: list[tuple[int, int]]
  edges_assignment: list[str]
  edges_fold_angle: list[float]
  edge_key_angle: dict[tuple[int, int], float]
  grid_size: int | None = None
  regions: tuple[MeshRegion, ...] = ()
  faces_region: tuple[int | None, ...] = ()
  edges_shaping: tuple[bool, ...] = ()
  edges_base_fold_angle: tuple[float, ...] = ()

  def to_fold_dict(self, title: str = 'model') -> dict[str, Any]:
    """Serializes this mesh to a FOLD dictionary; see `mesh_to_fold_dict`."""
    return mesh_to_fold_dict(self, title)


def _cross3(
    vec_a: tuple[float, float, float],
    vec_b: tuple[float, float, float],
) -> tuple[float, float, float]:
  return (
      vec_a[1] * vec_b[2] - vec_a[2] * vec_b[1],
      vec_a[2] * vec_b[0] - vec_a[0] * vec_b[2],
      vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0],
  )


def _dot3(
    vec_a: tuple[float, float, float],
    vec_b: tuple[float, float, float],
) -> float:
  return vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1] + vec_a[2] * vec_b[2]


def _select_centroid_face(
    vertices_2d: Sequence[tuple[float, float]],
    faces: Sequence[Sequence[int]],
) -> int:
  """Selects the face whose 2D centroid lies closest to the vertex centroid."""
  center_x = sum(v[0] for v in vertices_2d) / len(vertices_2d)
  center_y = sum(v[1] for v in vertices_2d) / len(vertices_2d)
  start_face = 0
  best_dist_sq = float('inf')
  for f_idx, face in enumerate(faces):
    fx = sum(vertices_2d[v][0] for v in face) / len(face)
    fy = sum(vertices_2d[v][1] for v in face) / len(face)
    dist_sq = (fx - center_x) ** 2 + (fy - center_y) ** 2
    if dist_sq < best_dist_sq:
      best_dist_sq = dist_sq
      start_face = f_idx
  return start_face


def _candidate_root_rects(
    packed: bus.PackedCreasePattern,
) -> list[bus.GridRect]:
  """Returns packed rects ordered by preference for the root trunk stick."""
  root_name: str | None = None
  merged_from: Mapping[str, tuple[str, ...]] = {}
  if packed.design is not None:
    root_name = packed.design.source.root
    merged_from = packed.design.merged_from
  elif packed.packed_tree is not None:
    root_name = packed.packed_tree.root

  if root_name is not None:
    exact_rivers: list[bus.GridRect] = []
    for rv in packed.rivers.values():
      if rv.stick == root_name or root_name in merged_from.get(rv.stick, ()):
        exact_rivers.extend(rv.rects)
    if exact_rivers:
      return exact_rivers
    exact_flaps = [
        fl.rect
        for fl in packed.flaps.values()
        if fl.id == fl.stick
        and (
            fl.stick == root_name or root_name in merged_from.get(fl.stick, ())
        )
    ]
    if exact_flaps:
      return exact_flaps

  on_axis_rivers: list[bus.GridRect] = []
  for rv in packed.rivers.values():
    if rv.is_on_axis:
      on_axis_rivers.extend(rv.rects)
  if on_axis_rivers:
    return on_axis_rivers
  mid_x = 0.5 * packed.grid_size
  on_axis_flaps = [
      fl.rect
      for fl in packed.flaps.values()
      if fl.id == fl.stick and fl.rect.x < mid_x < fl.rect.x + fl.rect.width
  ]
  if on_axis_flaps:
    return on_axis_flaps
  all_river_rects: list[bus.GridRect] = []
  for rv in packed.rivers.values():
    all_river_rects.extend(rv.rects)
  return all_river_rects


def _select_anchor_face(
    vertices_2d: Sequence[tuple[float, float]],
    faces: Sequence[Sequence[int]],
    packed: bus.PackedCreasePattern | None,
) -> int:
  """Selects a stationary anchor facet inside the root trunk region."""
  if (
      packed is None
      or packed.grid_size <= 0
      or (not packed.rivers and not packed.flaps)
  ):
    return _select_centroid_face(vertices_2d, faces)

  gs = float(packed.grid_size)
  root_rects = _candidate_root_rects(packed)
  if not root_rects:
    return _select_centroid_face(vertices_2d, faces)

  min_area = 0.08 / (gs * gs)
  band_half = max(2.5 / gs, 0.08)
  candidates: list[tuple[bool, bool, bool, float, float, int]] = []
  for rect in root_rects:
    if rect.width <= 0 or rect.height <= 0:
      continue
    rx0 = rect.x / gs - 0.5 - 1e-4
    ry0 = rect.y / gs - 0.5 - 1e-4
    rx1 = (rect.x + rect.width) / gs - 0.5 + 1e-4
    ry1 = (rect.y + rect.height) / gs - 0.5 + 1e-4
    mid_x = 0.5 * (rx0 + rx1)
    mid_y = 0.5 * (ry0 + ry1)
    horiz_band = rect.width >= rect.height
    for f_idx, face in enumerate(faces):
      face_len = len(face)
      fx = sum(vertices_2d[v][0] for v in face) / face_len
      fy = sum(vertices_2d[v][1] for v in face) / face_len
      if not (rx0 <= fx <= rx1 and ry0 <= fy <= ry1):
        continue
      area2 = 0.0
      vert_len = 0.0
      horiz_len = 0.0
      for i in range(face_len):
        x1, y1 = vertices_2d[face[i]]
        x2, y2 = vertices_2d[face[(i + 1) % face_len]]
        area2 += x1 * y2 - x2 * y1
        edx, edy = abs(x2 - x1), abs(y2 - y1)
        if edx < 1e-4:
          vert_len += edy
        elif edy < 1e-4:
          horiz_len += edx
      area = abs(area2) * 0.5
      if horiz_band:
        spine_dist = abs(fx - 0.5 / gs) + 0.25 * abs(fy - mid_y)
        parallel_len = vert_len
      else:
        spine_dist = abs(fy - 0.5 / gs) + 0.25 * abs(fx - mid_x)
        parallel_len = horiz_len
      candidates.append((
          area < min_area,
          spine_dist > band_half,
          parallel_len < 1e-4,
          round(spine_dist, 4),
          -area,
          f_idx,
      ))

  if not candidates:
    return _select_centroid_face(vertices_2d, faces)
  candidates.sort()
  return candidates[0][5]


def _compute_stick_figure_joints(
    source_tree: Any,
) -> tuple[
    Any,
    dict[str, tuple[float, float, float]],
    set[str],
    set[str],
]:
  """Computes 3D joint positions and root start/end joint partitions."""
  root_stick = next(
      s for s in source_tree.children if s.name == source_tree.root
  )
  adj: dict[str, list[tuple[str, tuple[float, float, float]]]] = (
      collections.defaultdict(list)
  )
  for stick in source_tree.children:
    phi = math.radians(stick.azimuth_angle)
    psi = math.radians(stick.elevation_angle)
    length = float(stick.length)
    delta = (
        -length * math.cos(psi) * math.sin(phi),
        length * math.cos(psi) * math.cos(phi),
        length * math.sin(psi),
    )
    adj[stick.start].append((stick.end, delta))
    adj[stick.end].append((stick.start, (-delta[0], -delta[1], -delta[2])))

  coords: dict[str, tuple[float, float, float]] = {
      root_stick.start: (0.0, 0.0, 0.0)
  }
  queue = collections.deque([root_stick.start])
  while queue:
    curr = queue.popleft()
    cx, cy, cz = coords[curr]
    for nxt, (dx, dy, dz) in adj[curr]:
      if nxt not in coords:
        coords[nxt] = (cx + dx, cy + dy, cz + dz)
        queue.append(nxt)

  def _reachable(seed: str, blocked: str) -> set[str]:
    seen = {seed}
    walk = collections.deque([seed])
    while walk:
      node = walk.popleft()
      for nb, _ in adj[node]:
        if nb != blocked and nb not in seen:
          seen.add(nb)
          walk.append(nb)
    return seen

  return (
      root_stick,
      coords,
      _reachable(root_stick.start, root_stick.end),
      _reachable(root_stick.end, root_stick.start),
  )


def _align_points_to_camera_frame(
    raw_pts_3d: Sequence[tuple[float, float, float]],
    vertices_2d: Sequence[tuple[float, float]],
    anchor_face: Sequence[int],
    packed: bus.PackedCreasePattern | None,
) -> list[tuple[float, float, float]]:
  """Rotates folded 3D points via SO(3) into canonical `(X, Y=Up, Z=Front)`."""
  if (
      packed is None
      or packed.design is None
      or packed.grid_size <= 0
      or not packed.flaps
  ):
    return list(raw_pts_3d)

  design = packed.design
  source = design.source
  try:
    root_stick, sf_coords, start_joints, end_joints = (
        _compute_stick_figure_joints(source)
    )
  except StopIteration:
    return list(raw_pts_3d)

  stick_by_name = {s.name: s for s in source.children}
  gs = float(packed.grid_size)
  flap_samples: list[
      tuple[tuple[float, float, float], tuple[float, float, float]]
  ] = []
  start_xy: list[tuple[float, float]] = []
  end_xy: list[tuple[float, float]] = []

  for flap in packed.flaps.values():
    if flap.id != flap.stick:
      continue
    chain = design.merged_from.get(flap.stick, (flap.stick,))
    leaf_joint: str | None = None
    for s_name in chain:
      stick_obj = stick_by_name.get(s_name)
      if stick_obj is None:
        continue
      for j_name in (stick_obj.end, stick_obj.start):
        joint_obj = source.joints.get(j_name)
        if joint_obj is not None and joint_obj.degree == 1:
          leaf_joint = j_name
          break
      if leaf_joint is not None:
        break
    if leaf_joint is None or leaf_joint not in sf_coords:
      continue

    if flap.circle_centers:
      tx = flap.circle_centers[0].x / gs - 0.5
      ty = flap.circle_centers[0].y / gs - 0.5
    else:
      tx = (flap.rect.x + 0.5 * flap.rect.width) / gs - 0.5
      ty = (flap.rect.y + 0.5 * flap.rect.height) / gs - 0.5

    nearest_v = min(
        range(len(vertices_2d)),
        key=lambda idx, cx=tx, cy=ty: (
            (vertices_2d[idx][0] - cx) ** 2 + (vertices_2d[idx][1] - cy) ** 2
        ),
    )
    flap_samples.append((sf_coords[leaf_joint], raw_pts_3d[nearest_v]))
    if leaf_joint in start_joints:
      start_xy.append((tx, ty))
    elif leaf_joint in end_joints:
      end_xy.append((tx, ty))

  if not flap_samples or not start_xy or not end_xy:
    return list(raw_pts_3d)

  dv_x = sum(p[0] for p in end_xy) / len(end_xy) - (
      sum(p[0] for p in start_xy) / len(start_xy)
  )
  dv_y = sum(p[1] for p in end_xy) / len(end_xy) - (
      sum(p[1] for p in start_xy) / len(start_xy)
  )
  if abs(dv_y) >= abs(dv_x):
    e_along_m = (0.0, 1.0 if dv_y >= 0.0 else -1.0, 0.0)
    u_perp = (1.0, 0.0, 0.0)
  else:
    e_along_m = (1.0 if dv_x >= 0.0 else -1.0, 0.0, 0.0)
    u_perp = (0.0, 1.0, 0.0)

  phi = math.radians(root_stick.azimuth_angle)
  psi = math.radians(root_stick.elevation_angle)
  e_along_sf = (
      -math.cos(psi) * math.sin(phi),
      math.cos(psi) * math.cos(phi),
      math.sin(psi),
  )
  e_left_sf = (-math.cos(phi), -math.sin(phi), 0.0)
  e_norm_sf = _cross3(e_along_sf, e_left_sf)

  root_s = sf_coords[root_stick.start]
  root_e = sf_coords[root_stick.end]
  root_mid = (
      0.5 * (root_s[0] + root_e[0]),
      0.5 * (root_s[1] + root_e[1]),
      0.5 * (root_s[2] + root_e[2]),
  )
  anchor_3d = (
      sum(raw_pts_3d[v][0] for v in anchor_face) / len(anchor_face),
      sum(raw_pts_3d[v][1] for v in anchor_face) / len(anchor_face),
      sum(raw_pts_3d[v][2] for v in anchor_face) / len(anchor_face),
  )

  roll_candidates = (
      (0.0, 0.0, 1.0),
      (0.0, 0.0, -1.0),
      u_perp,
      (-u_perp[0], -u_perp[1], 0.0),
  )
  best_score = float('-inf')
  best_norm_m = roll_candidates[0]
  best_left_m = _cross3(best_norm_m, e_along_m)
  for e_norm_m in roll_candidates:
    e_left_m = _cross3(e_norm_m, e_along_m)
    norm_corr = 0.0
    left_corr = 0.0
    for p_sf, p_m in flap_samples:
      rel_sf = (
          p_sf[0] - root_mid[0],
          p_sf[1] - root_mid[1],
          p_sf[2] - root_mid[2],
      )
      rel_m = (
          p_m[0] - anchor_3d[0],
          p_m[1] - anchor_3d[1],
          p_m[2] - anchor_3d[2],
      )
      norm_corr += _dot3(rel_m, e_norm_m) * _dot3(rel_sf, e_norm_sf)
      left_corr += _dot3(rel_m, e_left_m) * _dot3(rel_sf, e_left_sf)
    score = 2.0 * norm_corr + left_corr
    if score > best_score + 1e-9:
      best_score = score
      best_norm_m = e_norm_m
      best_left_m = e_left_m

  aligned: list[tuple[float, float, float]] = []
  for pt_m in raw_pts_3d:
    c_a = _dot3(pt_m, e_along_m)
    c_l = _dot3(pt_m, best_left_m)
    c_n = _dot3(pt_m, best_norm_m)
    x_sf = c_a * e_along_sf[0] + c_l * e_left_sf[0] + c_n * e_norm_sf[0]
    y_sf = c_a * e_along_sf[1] + c_l * e_left_sf[1] + c_n * e_norm_sf[1]
    z_sf = c_a * e_along_sf[2] + c_l * e_left_sf[2] + c_n * e_norm_sf[2]
    # Right-handed SO(3) mapping to camera frame: +Y is dorsal up, +Z is front.
    aligned.append((-x_sf, z_sf, y_sf))
  return aligned


MeshTarget = (
    cp_lib.CreasePattern
    | bus.ShapedCreasePattern
    | bus.SolvedCreasePattern
    | bus.PipelineResult
)


def _unwrap_mesh_target(
    target: MeshTarget,
    packed: bus.PackedCreasePattern | None,
) -> tuple[cp_lib.CreasePattern, bus.PackedCreasePattern | None]:
  """Extracts `(CreasePattern, PackedCreasePattern | None)` from `target`."""
  if isinstance(target, bus.PipelineResult):
    if isinstance(target.shaped, bus.ShapedCreasePattern):
      return target.shaped.creases, (packed or target.packed)
    return target.solved.creases, (packed or target.packed)
  if isinstance(target, (bus.ShapedCreasePattern, bus.SolvedCreasePattern)):
    return target.creases, (packed or target.packed)
  if isinstance(target, cp_lib.CreasePattern):
    return target, packed
  raise TypeError(
      'Expected CreasePattern, SolvedCreasePattern, ShapedCreasePattern, or'
      f' PipelineResult; got {type(target)!r}'
  )


def _face_regions(
    vertices_2d: Sequence[tuple[float, float]],
    faces: Sequence[Sequence[int]],
    packed: bus.PackedCreasePattern,
) -> tuple[tuple[MeshRegion, ...], tuple[int | None, ...]]:
  """Finds the packed flap or river that contains each face centroid."""
  merged_from = packed.design.merged_from if packed.design is not None else {}
  limbs: list[
      tuple[Literal['flap', 'river'], str, tuple[bus.GridRect, ...]]
  ] = [('flap', f.stick, (f.rect,)) for f in packed.flaps.values()]
  limbs += [('river', r.stick, r.rects) for r in packed.rivers.values()]

  regions: list[MeshRegion] = []
  rects: list[tuple[bus.GridRect, int]] = []
  for kind, stick, limb_rects in limbs:
    region = MeshRegion(kind, tuple(merged_from.get(stick, (stick,))))
    # A limb split into mirrored copies is still a single region.
    if region not in regions:
      regions.append(region)
    rects.extend((rect, regions.index(region)) for rect in limb_rects)
  reserved_idx = len(regions)
  rects.extend((res.rect, reserved_idx) for res in packed.reservations)
  if not rects:
    return (), ()

  grid_size = packed.grid_size
  faces_region: list[int | None] = []
  for face in faces:
    x = (sum(vertices_2d[v][0] for v in face) / len(face) + 0.5) * grid_size
    y = (sum(vertices_2d[v][1] for v in face) / len(face) + 0.5) * grid_size
    faces_region.append(
        next(
            (
                index
                for rect, index in rects
                if rect.x <= x <= rect.x + rect.width
                and rect.y <= y <= rect.y + rect.height
            ),
            None,
        )
    )
  if packed.reservations and reserved_idx in faces_region:
    regions.append(MeshRegion('reserved', ()))
  return (tuple(regions), tuple(faces_region)) if regions else ((), ())


def _merged_interval_length(
    intervals: Sequence[tuple[float, float]],
) -> float:
  """Returns union length of `[lo, hi]` intervals without double-counting."""
  if not intervals:
    return 0.0
  ordered = sorted(intervals)
  total = 0.0
  cur_lo, cur_hi = ordered[0]
  for nxt_lo, nxt_hi in ordered[1:]:
    if nxt_lo <= cur_hi:
      cur_hi = max(cur_hi, nxt_hi)
    else:
      total += cur_hi - cur_lo
      cur_lo, cur_hi = nxt_lo, nxt_hi
  return total + (cur_hi - cur_lo)


# Both endpoints of a folded edge can shift by up to DEFAULT_TOLERANCE
# during planar vertex snapping, giving a 2 * DEFAULT_TOLERANCE corridor.
_BASE_EDGE_SNAP_TOL = 2.0 * crease_lib.DEFAULT_TOLERANCE
# Minimum |cos(theta)| (~16.3 deg) between a mesh edge and a candidate base
# crease; sits above cos(22.5 deg) ~= 0.924 so diagonal creases are rejected.
_BASE_EDGE_MIN_COS = 0.96
# Union overlap ratio required to classify an edge as a solved base crease,
# with a 50% floor when endpoint snapping trims short edges by snap_tol.
_BASE_EDGE_MIN_COVERAGE = 0.9
_BASE_EDGE_SHORT_FLOOR = 0.5


def _classify_base_edges(
    vertices_2d: Sequence[tuple[float, float]],
    edges: Sequence[tuple[int, int]],
    edges_assignment: Sequence[str],
    base: cp_lib.CreasePattern,
) -> tuple[tuple[bool, ...], tuple[float, ...]]:
  """Flags shaping-added edges and extracts signed solved-base fold angles."""
  base_index = crease_lib.CreaseBBoxIndex()
  base_geom: dict[
      int, tuple[float, float, float, float, float, float, bool]
  ] = {}
  for crease_line in base.creases:
    assignment, _ = fold_3d.crease_assignment_and_angle(
        crease_line.fold, crease_line.fold_percent
    )
    if assignment not in ('M', 'V'):
      continue
    bx1, by1 = float(crease_line.p1.x), float(crease_line.p1.y)
    bx2, by2 = float(crease_line.p2.x), float(crease_line.p2.y)
    blen = math.hypot(bx2 - bx1, by2 - by1)
    if blen <= 1e-9:
      continue
    base_index.add(crease_line)
    base_geom[id(crease_line)] = (
        bx1,
        by1,
        bx2,
        by2,
        (bx2 - bx1) / blen,
        (by2 - by1) / blen,
        assignment == 'M',
    )

  snap_tol = _BASE_EDGE_SNAP_TOL
  added: list[bool] = []
  base_angles: list[float] = []
  for (v1, v2), assignment in zip(edges, edges_assignment, strict=True):
    x1, y1 = vertices_2d[v1][0] + 0.5, vertices_2d[v1][1] + 0.5
    x2, y2 = vertices_2d[v2][0] + 0.5, vertices_2d[v2][1] + 0.5
    elen = math.hypot(x2 - x1, y2 - y1)
    if assignment == 'B' or elen <= crease_lib.DEFAULT_TOLERANCE:
      added.append(False)
      base_angles.append(0.0)
      continue
    ex = (x2 - x1) / elen
    ey = (y2 - y1) / elen
    candidates = base_index.query_bbox(
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
        pad=snap_tol,
    )
    all_intervals: list[tuple[float, float]] = []
    m_intervals: list[tuple[float, float]] = []
    v_intervals: list[tuple[float, float]] = []
    for cand in candidates:
      bx1, by1, bx2, by2, ux, uy, is_m = base_geom[id(cand)]
      if abs((x1 - bx1) * uy - (y1 - by1) * ux) > snap_tol:
        continue
      if abs((x2 - bx1) * uy - (y2 - by1) * ux) > snap_tol:
        continue
      if abs(ex * ux + ey * uy) < _BASE_EDGE_MIN_COS:
        continue
      s1 = (bx1 - x1) * ex + (by1 - y1) * ey
      s2 = (bx2 - x1) * ex + (by2 - y1) * ey
      lo = max(0.0, min(s1, s2))
      hi = min(elen, max(s1, s2))
      if hi > lo:
        all_intervals.append((lo, hi))
        if is_m:
          m_intervals.append((lo, hi))
        else:
          v_intervals.append((lo, hi))

    covered = _merged_interval_length(all_intervals)
    min_required = min(
        _BASE_EDGE_MIN_COVERAGE * elen,
        max(_BASE_EDGE_SHORT_FLOOR * elen, elen - snap_tol),
    )
    if covered >= min_required:
      m_cov = _merged_interval_length(m_intervals)
      v_cov = _merged_interval_length(v_intervals)
      added.append(False)
      base_angles.append(-180.0 if m_cov >= v_cov else 180.0)
    else:
      added.append(assignment in ('M', 'V'))
      base_angles.append(0.0)
  return tuple(added), tuple(base_angles)


def build_folded_mesh_3d(
    cp: MeshTarget,
    tol: float = 0.001,
    *,
    packed: bus.PackedCreasePattern | None = None,
) -> FoldedMesh3D:
  """Folds a shaped `CreasePattern` in 3D via BFS dihedral hinge rotations.

  Args:
    cp: Solved or shaped `CreasePattern` (or `ShapedCreasePattern`,
      `SolvedCreasePattern`, or `PipelineResult`) with `fold_percent` encoding
      target dihedral fold angles (`fold_percent * +-180` degrees).
    tol: Planar vertex snapping tolerance in unit-square coordinates.
    packed: Optional `PackedCreasePattern` used to anchor the stationary root
      facet inside the trunk river region and align the folded mesh into the
      canonical turnaround camera frame (`+Y` dorsal up, `+Z` anterior front).
      Automatically inferred when `cp` is a bus stage record.

  Returns:
    A `FoldedMesh3D` containing planar faces and centered 3D folded vertices.

  Raises:
    ValueError: If no bounded planar faces can be extracted from `cp`.
  """
  cp_obj, resolved_packed = _unwrap_mesh_target(cp, packed)
  base_solved_cp: cp_lib.CreasePattern | None = None
  if isinstance(cp, bus.ShapedCreasePattern):
    base_solved_cp = cp.solved.creases
  elif isinstance(cp, bus.PipelineResult) and isinstance(
      cp.shaped, bus.ShapedCreasePattern
  ):
    base_solved_cp = cp.shaped.solved.creases

  def _pick_anchor(graph: fold_3d.FoldGraph) -> int:
    centered_2d = [
        (float(x - 0.5), float(y - 0.5)) for x, y in graph.vertices_2d
    ]
    face_lists = [face.tolist() for face in graph.faces]
    return _select_anchor_face(centered_2d, face_lists, resolved_packed)

  model = fold_3d.fold_3d(
      cp_obj,
      anchor_face=_pick_anchor,
      vert_tol=tol,
      add_border=True,
      merge_collinear=False,
      strict=False,
  )
  vertices_2d = [(float(x - 0.5), float(y - 0.5)) for x, y in model.vertices_2d]
  faces = [face.tolist() for face in model.faces]
  edges = [(int(u), int(v)) for u, v in model.edges]
  edges_assignment = list(model.assignments)
  edges_fold_angle = [float(a) for a in model.fold_angles]
  edge_key_angle = {
      (min(u, v), max(u, v)): angle
      for (u, v), angle in zip(edges, edges_fold_angle, strict=True)
  }
  raw_pts_3d = [
      (float(x - 0.5), float(y - 0.5), float(z))
      for x, y, z in model.vertices_3d
  ]
  aligned_pts_3d = _align_points_to_camera_frame(
      raw_pts_3d,
      vertices_2d,
      faces[model.anchor_face],
      resolved_packed,
  )
  num_v = len(vertices_2d)
  cx = sum(p[0] for p in aligned_pts_3d) / num_v
  cy = sum(p[1] for p in aligned_pts_3d) / num_v
  cz = sum(p[2] for p in aligned_pts_3d) / num_v
  vertices_3d = [(p[0] - cx, p[1] - cy, p[2] - cz) for p in aligned_pts_3d]

  grid_size: int | None = None
  regions: tuple[MeshRegion, ...] = ()
  faces_region: tuple[int | None, ...] = ()
  if resolved_packed is not None and resolved_packed.grid_size > 0:
    grid_size = resolved_packed.grid_size
    regions, faces_region = _face_regions(vertices_2d, faces, resolved_packed)
  edges_shaping: tuple[bool, ...] = ()
  edges_base_fold_angle: tuple[float, ...] = ()
  if base_solved_cp is not None:
    edges_shaping, edges_base_fold_angle = _classify_base_edges(
        vertices_2d, edges, edges_assignment, base_solved_cp
    )

  return FoldedMesh3D(
      vertices_2d=vertices_2d,
      vertices_3d=vertices_3d,
      faces=faces,
      edges=edges,
      edges_assignment=edges_assignment,
      edges_fold_angle=edges_fold_angle,
      edge_key_angle=edge_key_angle,
      grid_size=grid_size,
      regions=regions,
      faces_region=faces_region,
      edges_shaping=edges_shaping,
      edges_base_fold_angle=edges_base_fold_angle,
  )


def mesh_to_fold_dict(mesh: FoldedMesh3D, title: str) -> dict[str, Any]:
  """Serializes a `FoldedMesh3D` to a FOLD dict with both of its states.

  This docstring is the canonical schema specification for the extended
  `shaped_3d.fold` format written by `bp-pipeline -o` and read by `docs/`.

  Key frame and flat frame:
    - Key frame (`frame_classes: ['foldedForm']`, `frame_attributes: ['3D']`):
      3D folded vertices, faces, edges, assignments, and fold angles (FOLD 1.1).
    - `file_frames[0]` (`frame_classes: ['creasePattern']`,
      `frame_attributes: ['2D']`, `frame_inherit: True`, `frame_parent: 0`):
      flat 2D `vertices_coords` on the same vertices, edges, and faces, so a
      reader can unfold the model or re-fold it from `edges_foldAngle`.

  Coordinate and handedness conventions:
    FOLD frames are right-handed (`det = +1`) and list faces counter-clockwise.
    `build_folded_mesh_3d` delegates directly to `folder_3d.fold_3d` (`det = +1`
    proper `SO(3)` rotation of `folder_3d`'s 3D vertices, without reflecting
    `y`). Reversing `folder_3d`'s clockwise face loops makes `faces_vertices`
    counter-clockwise with normals pointing toward `+z`, and storing the flat
    frame as `(1 - x_u, 1 - y_u)` (`[round(0.5 - x, 6), round(0.5 - y, 6)]`, a
    proper 180-degree `SO(2)` rotation of `[0, 1]^2`) keeps faces
    counter-clockwise on `[0, 1]^2`, puts `y_u = 0` at the top when drawn with
    `+y` up, and makes positive fold angles (valleys) fold by the exact stored
    angle into the 3D key frame—matching the bundled `docs/models/` snapshots.

  Custom `bp:` extension keys (omitted when the corresponding `FoldedMesh3D`
  field is empty or `None`, e.g. when built from a bare `CreasePattern`):
    - `file_bp:gridSize` (`int`): box-pleating grid size from the packed layout.
      Present when `mesh.grid_size is not None`.
    - `file_bp:regions` (`list[dict[str, Any]]`): packed flaps, rivers, and
      reserved regions, each `{'kind': 'flap' | 'river' | 'reserved', 'sticks':
      list[str]}` in chain order (empty `sticks` for `'reserved'`). Present
      when `mesh.regions` is non-empty.
    - `faces_bp:region` (`list[int | None]`): parallel to `faces_vertices`;
      index into `file_bp:regions` for the region containing each face centroid,
      or `None`. Present whenever `file_bp:regions` is present.
    - `edges_bp:shaping` (`list[bool]`): parallel to `edges_vertices`; `True`
      for mountain/valley edges added by shaping (edges not covered to at least
      90% of their length, or `max(50%, len - 2 * DEFAULT_TOLERANCE)` on short
      snapped segments, by aligned folded creases of `solved.creases` within
      `2 * DEFAULT_TOLERANCE`). Present when built from a
      `ShapedCreasePattern` (`mesh.edges_shaping` non-empty).
    - `edges_bp:baseFoldAngle` (`list[float]`): parallel to `edges_vertices`;
      signed fold angle in degrees (`-180.0`, `0.0`, or `180.0`) of each edge in
      `solved.creases`, preserving original base orientations when shaping bends
      or flips a base crease. Present when built from a `ShapedCreasePattern`
      (`mesh.edges_base_fold_angle` non-empty).

  Args:
    mesh: The folded mesh to serialize.
    title: Model name used in `file_title`.

  Returns:
    A JSON-serializable FOLD v1.1 dictionary.
  """
  fold: dict[str, Any] = {
      'file_spec': 1.1,
      'file_creator': 'bp_pipeline + render_3d',
      'file_title': f'{title} (3D Shaped Figure)',
      'file_classes': ['singleModel'],
      'frame_classes': ['foldedForm'],
      'frame_attributes': ['3D'],
      'vertices_coords': [
          [round(p[0], 6), round(p[1], 6), round(p[2], 6)]
          for p in mesh.vertices_3d
      ],
      'faces_vertices': [list(reversed(face)) for face in mesh.faces],
      'edges_vertices': [list(e) for e in mesh.edges],
      'edges_assignment': mesh.edges_assignment,
      'edges_foldAngle': [round(a, 4) for a in mesh.edges_fold_angle],
      'file_frames': [{
          'frame_classes': ['creasePattern'],
          'frame_attributes': ['2D'],
          'frame_inherit': True,
          'frame_parent': 0,
          'vertices_coords': [
              [round(0.5 - x, 6), round(0.5 - y, 6)]
              for x, y in mesh.vertices_2d
          ],
      }],
  }
  if mesh.grid_size is not None:
    fold['file_bp:gridSize'] = mesh.grid_size
  if mesh.regions:
    fold['file_bp:regions'] = [
        {'kind': region.kind, 'sticks': list(region.sticks)}
        for region in mesh.regions
    ]
    fold['faces_bp:region'] = list(mesh.faces_region)
  if mesh.edges_shaping:
    fold['edges_bp:shaping'] = list(mesh.edges_shaping)
  if mesh.edges_base_fold_angle:
    fold['edges_bp:baseFoldAngle'] = [
        round(a, 4) for a in mesh.edges_base_fold_angle
    ]
  return fold


def _normalize_paper_rgb(
    color_top: Sequence[int] | None,
) -> tuple[float, float, float]:
  """Normalizes an optional RGB color into a legible lit paper tint."""
  if not color_top or len(color_top) != 3:
    return _DEFAULT_PAPER_RGB
  # Clamp extremes so crease lines and Lambertian facet shading stay clear.
  return (
      max(0.24, min(0.92, float(color_top[0]) / 255.0)),
      max(0.24, min(0.92, float(color_top[1]) / 255.0)),
      max(0.24, min(0.92, float(color_top[2]) / 255.0)),
  )


def draw_folded_mesh_on_ax(
    ax: axes.Axes,
    mesh: FoldedMesh3D,
    title: str,
    *,
    yaw: float = -0.62,
    pitch: float = 0.38,
    roll: float = 0.0,
    color_top: Sequence[int] | None = None,
) -> None:
  """Draws a Lambertian-shaded 3D folded origami mesh view onto `ax`."""
  ax.set_facecolor('#f8fafc')
  ax.set_aspect('equal', adjustable='box')
  ax.set_title(title, fontsize=11, fontweight='bold')
  ax.set_xticks([])
  ax.set_yticks([])
  for spine in ax.spines.values():
    spine.set_edgecolor('#cbd5e1')

  base_rgb = _normalize_paper_rgb(color_top)
  cos_y, sin_y = math.cos(yaw), math.sin(yaw)
  cos_p, sin_p = math.cos(pitch), math.sin(pitch)
  cos_r, sin_r = math.cos(roll), math.sin(roll)

  rotated_pts: list[tuple[float, float, float]] = []
  for px, py, pz in mesh.vertices_3d:
    x1 = px * cos_y + pz * sin_y
    z1 = -px * sin_y + pz * cos_y
    y2 = py * cos_p - z1 * sin_p
    z2 = py * sin_p + z1 * cos_p
    x3 = x1 * cos_r - y2 * sin_r
    y3 = x1 * sin_r + y2 * cos_r
    rotated_pts.append((x3, y3, z2))

  min_x = min(p[0] for p in rotated_pts)
  max_x = max(p[0] for p in rotated_pts)
  min_y = min(p[1] for p in rotated_pts)
  max_y = max(p[1] for p in rotated_pts)
  span_x = max(max_x - min_x, 0.05)
  span_y = max(max_y - min_y, 0.05)
  span = max(span_x, span_y)

  # Subtle ground shadow beneath the model.
  shadow = patches.Ellipse(
      ((min_x + max_x) * 0.5, min_y - span * 0.045),
      width=span_x * 0.85,
      height=span * 0.055,
      facecolor='#0f172a',
      edgecolor='none',
      alpha=0.12,
      zorder=0,
  )
  ax.add_patch(shadow)

  light_1 = (0.42, 0.62, 0.66)
  light_2 = (-0.52, -0.25, 0.81)
  face_items: list[
      tuple[
          float,
          float,
          float,
          list[tuple[float, float]],
          tuple[float, float, float],
          list[tuple[tuple[float, float], tuple[float, float]]],
      ]
  ] = []

  for face in mesh.faces:
    if len(face) < 3:
      continue
    p0, p1, p2 = (
        rotated_pts[face[0]],
        rotated_pts[face[1]],
        rotated_pts[face[2]],
    )
    ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    vx, vy, vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    norm_len = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm_len > 1e-9:
      nx, ny, nz = nx / norm_len, ny / norm_len, nz / norm_len
    else:
      nx, ny, nz = 0.0, 0.0, 1.0

    if nz < 0:
      nx, ny, nz = -nx, -ny, -nz
    dot_1 = max(0.0, nx * light_1[0] + ny * light_1[1] + nz * light_1[2])
    dot_2 = max(0.0, nx * light_2[0] + ny * light_2[1] + nz * light_2[2])
    shade = 0.56 + 0.35 * dot_1 + 0.14 * dot_2
    face_rgb = (
        min(1.0, base_rgb[0] * shade),
        min(1.0, base_rgb[1] * shade),
        min(1.0, base_rgb[2] * shade),
    )

    poly_2d = [(rotated_pts[v][0], rotated_pts[v][1]) for v in face]
    area_2d = 0.0
    face_len = len(poly_2d)
    for i in range(face_len):
      xa, ya = poly_2d[i]
      xb, yb = poly_2d[(i + 1) % face_len]
      area_2d += xa * yb - xb * ya
    area_2d = abs(area_2d) * 0.5

    z_avg = sum(rotated_pts[v][2] for v in face) / face_len
    z_bucket = round(z_avg / 0.002) * 0.002

    crease_segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for i in range(face_len):
      u_idx, v_idx = face[i], face[(i + 1) % face_len]
      key = (min(u_idx, v_idx), max(u_idx, v_idx))
      ang = mesh.edge_key_angle.get(key, 0.0)
      if abs(ang) > 1.0 or key not in mesh.edge_key_angle:
        crease_segs.append((poly_2d[i], poly_2d[(i + 1) % face_len]))

    face_items.append((
        z_bucket,
        area_2d,
        z_avg,
        poly_2d,
        face_rgb,
        crease_segs,
    ))

  face_items.sort(key=lambda item: (item[0], item[1], item[2]))

  polys: list[list[tuple[float, float]]] = []
  face_colors: list[tuple[float, float, float]] = []
  edge_colors: list[tuple[float, float, float]] = []
  crease_lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
  crease_line_colors: list[tuple[float, float, float, float]] = []

  for _, _, _, poly_2d, rgb, crease_segs in face_items:
    polys.append(poly_2d)
    face_colors.append(rgb)
    edge_colors.append((rgb[0] * 0.86, rgb[1] * 0.85, rgb[2] * 0.84))
    dark_rgba = (rgb[0] * 0.56, rgb[1] * 0.54, rgb[2] * 0.52, 0.88)
    for seg in crease_segs:
      crease_lines.append(seg)
      crease_line_colors.append(dark_rgba)

  poly_collection = mpl_collections.PolyCollection(
      polys,
      facecolors=face_colors,
      edgecolors=edge_colors,
      linewidths=0.45,
      zorder=2,
  )
  ax.add_collection(poly_collection)

  if crease_lines:
    line_collection = mpl_collections.LineCollection(
        crease_lines,
        colors=crease_line_colors,
        linewidths=0.75,
        capstyle='round',
        joinstyle='round',
        zorder=3,
    )
    ax.add_collection(line_collection)

  mid_x = (min_x + max_x) * 0.5
  mid_y = (min_y + max_y) * 0.5 - span * 0.02
  half = span * 0.58
  ax.set_xlim(mid_x - half, mid_x + half)
  ax.set_ylim(mid_y - half, mid_y + half)


def render_shaped_3d_turnaround_png(
    out_path: str | pathlib.Path,
    *,
    name: str,
    mesh: FoldedMesh3D,
    color_top: Sequence[int] | None = None,
    dpi: int = 180,
) -> pathlib.Path:
  """Renders a 4-view shaded 3D turnaround PNG (`Iso`, `Side`, `Front`, `Top`)."""
  target_path = pathlib.Path(out_path)
  target_path.parent.mkdir(parents=True, exist_ok=True)
  fig, axs = plt.subplots(1, 4, figsize=(20, 5.2))
  fig.suptitle(
      f'3D Shaped Origami Turnaround: {name} '
      f'({len(mesh.faces)} facets, {len(mesh.edges)} edges)',
      fontsize=13,
      fontweight='bold',
      y=1.02,
  )
  for ax, (view_title, yaw, pitch, roll) in zip(
      axs, _TURNAROUND_VIEWS, strict=True
  ):
    draw_folded_mesh_on_ax(
        ax,
        mesh,
        view_title,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        color_top=color_top,
    )
  fig.tight_layout()
  fig.savefig(target_path, dpi=dpi, bbox_inches='tight')
  plt.close(fig)
  return target_path


def save_3d_turnaround(
    target: FoldedMesh3D | MeshTarget,
    out_path: str | pathlib.Path,
    *,
    name: str | None = None,
    packed: bus.PackedCreasePattern | None = None,
    color_top: Sequence[int] | None = None,
    dpi: int = 180,
) -> FoldedMesh3D:
  """Folds `target` (if needed) and saves a 4-angle shaded 3D turnaround PNG."""
  resolved_packed = packed
  if isinstance(target, FoldedMesh3D):
    mesh = target
  else:
    _, resolved_packed = _unwrap_mesh_target(target, packed)
    mesh = build_folded_mesh_3d(target, packed=resolved_packed)

  source_tree = None
  if resolved_packed is not None:
    if resolved_packed.design is not None:
      source_tree = resolved_packed.design.source
    else:
      source_tree = resolved_packed.packed_tree

  resolved_name = (
      name
      or (
          source_tree.name
          if source_tree is not None and source_tree.name
          else None
      )
      or 'model'
  )
  resolved_color = color_top
  if resolved_color is None and source_tree is not None:
    resolved_color = getattr(source_tree, 'color_top', None)

  render_shaped_3d_turnaround_png(
      out_path,
      name=resolved_name,
      mesh=mesh,
      color_top=resolved_color,
      dpi=dpi,
  )
  return mesh
