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

"""Local flat-foldability repair via best-first crease rotation and flipping."""

import collections
from collections.abc import Sequence
import dataclasses
import heapq
import itertools
import math
from typing import Protocol, Self

from .. import rectangle as rectangle_lib
from .. import timeout as timeout_lib
from . import crease as crease_utils
from . import crease_pattern as cp_utils
from . import point as point_utils
from . import rotation
from . import vertex


class BoundingBox(Protocol):
  """Structural axis-aligned box protocol for constraining local repairs."""

  @property
  def x_start(self) -> float:
    ...

  @property
  def y_start(self) -> float:
    ...

  @property
  def x_len(self) -> float:
    ...

  @property
  def y_len(self) -> float:
    ...

  def clip(self) -> Self | None:
    ...

  def crease_is_within(self, crease: crease_utils.Crease) -> bool:
    ...


def split_border_creases_at_tips(
    creases: Sequence[crease_utils.Crease],
) -> list[crease_utils.Crease]:
  """Splits `Fold.BORDER` segments at interior border tips of other creases.

  When `CreasePattern(..., light_mode=True)` is used after moving or rotating a
  crease endpoint along a merged `Fold.BORDER` edge, `light_mode` skips global
  intersection splitting and would otherwise classify the border tip as an
  internal degree-1 vertex.

  Args:
    creases: Creases of a pattern.

  Returns:
    Creases with `Fold.BORDER` edges split at any interior border endpoints.
  """
  borders = [c for c in creases if c.fold == crease_utils.Fold.BORDER]
  if not borders:
    return list(creases)
  non_borders = [c for c in creases if c.fold != crease_utils.Fold.BORDER]
  border_pts: list[point_utils.Point] = []
  for c in non_borders:
    if c.p1.is_border:
      border_pts.append(c.p1)
    if c.p2.is_border:
      border_pts.append(c.p2)
  if not border_pts:
    return list(creases)
  split_borders: list[crease_utils.Crease] = []
  for bc in borders:
    split_borders.extend(crease_utils.split_crease_at_points(bc, border_pts))
  return non_borders + split_borders


def _build_cp_preserving_mode(
    creases: Sequence[crease_utils.Crease], *, light_mode: bool
) -> cp_utils.CreasePattern:
  if light_mode:
    creases = split_border_creases_at_tips(creases)
  return cp_utils.CreasePattern(creases=list(creases), light_mode=light_mode)


def _point_in_clipped_box(pt: point_utils.Point, box: BoundingBox) -> bool:
  clipped = box.clip()
  if clipped is None:
    return False
  tol = point_utils.DEFAULT_TOLERANCE
  return (
      clipped.x_start - tol <= pt.x <= clipped.x_start + clipped.x_len + tol
      and clipped.y_start - tol <= pt.y <= clipped.y_start + clipped.y_len + tol
  )


@dataclasses.dataclass(kw_only=True)
class CandidateCreasePattern:
  """Priority-queue entry tracking a candidate flat-foldability repair state."""

  cp: cp_utils.CreasePattern
  non_foldable_vertices: list[vertex.Vertex]
  unmodifiable_crease_hashes: frozenset[int]
  crease_hash_to_vertex_mapping: dict[int, list[vertex.Vertex]]
  score: float
  border_distance: float
  num_crease_modifications: int
  candidate_num: int

  @classmethod
  def from_cp(
      cls,
      cp: cp_utils.CreasePattern,
      *,
      unmodifiable_crease_hashes: frozenset[int],
      num_crease_modifications: int,
      candidate_num: int,
  ) -> 'CandidateCreasePattern':
    """Builds a scored candidate record from `cp`."""
    score, _, non_foldable = cp.vertex_flat_foldability_percentage()
    crease_to_vertices: dict[int, list[vertex.Vertex]] = (
        collections.defaultdict(list)
    )
    for vert in cp.vertices:
      timeout_lib.maybe_timeout_cooperatively()
      for c in vert.creases:
        crease_to_vertices[c.hash_value].append(vert)
    border_distance = sum(
        min(v.point.x, 1.0 - v.point.x, v.point.y, 1.0 - v.point.y)
        for v in non_foldable
    )
    return cls(
        cp=cp,
        non_foldable_vertices=non_foldable,
        unmodifiable_crease_hashes=unmodifiable_crease_hashes,
        crease_hash_to_vertex_mapping=crease_to_vertices,
        score=score,
        border_distance=border_distance,
        num_crease_modifications=num_crease_modifications,
        candidate_num=candidate_num,
    )

  def to_tuple(self) -> tuple[float, float, int, int, 'CandidateCreasePattern']:
    """Returns a min-heap sort tuple prioritizing higher foldability score."""
    return (
        -self.score,
        self.border_distance,
        self.num_crease_modifications,
        self.candidate_num,
        self,
    )


def _get_other_connecting_vertex(
    connected_vertex: vertex.Vertex,
    connecting_crease: crease_utils.Crease,
    crease_hash_to_vertex_mapping: dict[int, list[vertex.Vertex]],
) -> vertex.Vertex:
  """Returns the opposite endpoint vertex of `connecting_crease`."""
  verts = crease_hash_to_vertex_mapping[connecting_crease.hash_value]
  if len(verts) != 2:
    raise ValueError(
        f'Crease must connect exactly two vertices (got {len(verts)}).'
    )
  if verts[0].point.isclose(connected_vertex.point):
    return verts[1]
  if verts[1].point.isclose(connected_vertex.point):
    return verts[0]
  raise ValueError(
      'Crease to modify is not connected to the violating vertex point.'
  )


def get_new_cp_with_rotated_crease(
    current_cp: cp_utils.CreasePattern,
    crease_to_be_rotated: crease_utils.Crease,
    violating_vertex: vertex.Vertex,
    crease_hash_to_vertex_mapping: dict[int, list[vertex.Vertex]],
) -> tuple[cp_utils.CreasePattern | None, frozenset[int]]:
  """Rotates `crease_to_be_rotated` about `violating_vertex` to fix Kawasaki.

  Args:
    current_cp: Current crease pattern.
    crease_to_be_rotated: Mountain or valley crease incident to
      `violating_vertex`.
    violating_vertex: Vertex whose alternating sector angles are unbalanced.
    crease_hash_to_vertex_mapping: Map from crease hash to endpoint vertices.

  Returns:
    `(rotated_cp, modified_crease_hashes)`, or `(None, frozenset())` if the
    required rotation passes a neighboring sector or crosses another crease.
  """
  mv_creases = [c for c in violating_vertex.creases if c.mountain_or_valley()]
  if crease_to_be_rotated not in mv_creases:
    return None, frozenset()
  mv_idx = mv_creases.index(crease_to_be_rotated)
  sector_angles = [angle for angle, _, _ in violating_vertex.sector_angles()]
  if len(sector_angles) != len(mv_creases):
    return None, frozenset()

  if mv_idx % 2 == 0:
    angle_delta = sum(sector_angles[::2]) - sum(sector_angles[1::2])
  else:
    angle_delta = sum(sector_angles[1::2]) - sum(sector_angles[::2])
  if math.isclose(angle_delta, 0.0):
    return None, frozenset()

  # Rotating clockwise by `angle_delta / 2` balances odd and even sector sums.
  rotation_delta = 0.5 * angle_delta
  cw_angle = sector_angles[mv_idx]
  ccw_angle = sector_angles[(mv_idx - 1) % len(sector_angles)]
  if rotation_delta >= cw_angle or -rotation_delta >= ccw_angle:
    return None, frozenset()

  rotated_ray = rotation.rotate_crease_clockwise(
      crease_to_rotate=crease_to_be_rotated,
      rotation_angle_degrees=rotation_delta,
      rotation_origin_tip=violating_vertex.point,
      require_on_paper=False,
  )
  adj_vertex = _get_other_connecting_vertex(
      violating_vertex, crease_to_be_rotated, crease_hash_to_vertex_mapping
  )
  adj_idx = adj_vertex.creases.index(crease_to_be_rotated)
  adj_neighbors: list[crease_utils.Crease] = []
  for step in (1, -1):
    cand = adj_vertex.creases[(adj_idx + step) % len(adj_vertex.creases)]
    if cand not in adj_neighbors:
      adj_neighbors.append(cand)

  new_adj_point: point_utils.Point | None = None
  closest_dist = float('inf')
  for neighbor in adj_neighbors:
    timeout_lib.maybe_timeout_cooperatively()
    hit = rotated_ray.vertex_with_crease(neighbor, extend_lines=True)
    if hit is None or neighbor.point_status(hit) not in (
        crease_utils.PointStatus.INTERNAL,
        crease_utils.PointStatus.TIP,
    ):
      continue
    dist = point_utils.distance(violating_vertex.point, hit)
    if dist < closest_dist:
      closest_dist = dist
      new_adj_point = hit
  if new_adj_point is None:
    return None, frozenset()

  if current_cp.light_mode:
    rotated_seg = dataclasses.replace(
        crease_to_be_rotated,
        p1_init=violating_vertex.point,
        p2_init=new_adj_point,
    )
    exempt = {c.hash_value for c in adj_vertex.creases} | {
        c.hash_value for c in violating_vertex.creases
    }
    for other in current_cp.creases:
      if other.fold == crease_utils.Fold.BORDER or other.hash_value in exempt:
        continue
      crossing = rotated_seg.vertex_with_crease(other)
      if (
          crossing is not None
          and not crossing.isclose(new_adj_point)
          and not crossing.isclose(violating_vertex.point)
      ):
        return None, frozenset()

  remaining: dict[int, crease_utils.Crease] = {
      c.hash_value: c for c in current_cp.creases
  }
  modified: list[crease_utils.Crease] = []
  for adj_crease in adj_vertex.creases:
    timeout_lib.maybe_timeout_cooperatively()
    if adj_crease.fold == crease_utils.Fold.BORDER:
      continue
    del remaining[adj_crease.hash_value]
    if adj_crease.p1.isclose(adj_vertex.point):
      anchor_pt = adj_crease.p2
    elif adj_crease.p2.isclose(adj_vertex.point):
      anchor_pt = adj_crease.p1
    else:
      raise ValueError(
          'Incident crease on adjacent vertex does not match its vertex point.'
      )
    if anchor_pt.isclose(new_adj_point):
      continue
    modified.append(
        dataclasses.replace(
            adj_crease, p1_init=anchor_pt, p2_init=new_adj_point
        )
    )

  new_cp = _build_cp_preserving_mode(
      list(remaining.values()) + modified, light_mode=current_cp.light_mode
  )
  return new_cp, frozenset(c.hash_value for c in modified)


def _border_flip_candidate(
    current_cp: cp_utils.CreasePattern,
    target_point: point_utils.Point,
) -> tuple[cp_utils.CreasePattern, frozenset[int], frozenset[int]] | None:
  """Finds a border-ending M/V flip fixing flat-foldability at `target_point`."""
  v_cur = next(
      (v for v in current_cp.vertices if v.point.isclose(target_point)), None
  )
  if (
      v_cur is None
      or not v_cur.check_kawasaki
      or (v_cur.check_maekawa and v_cur.check_crimp_angles)
  ):
    return None
  border_mv = [
      vc
      for vc in v_cur.creases
      if vc.mountain_or_valley()
      and (vc.p2 if vc.p1.isclose(v_cur.point) else vc.p1).is_border
  ]
  if not border_mv:
    return None

  for flip_count in range(1, min(3, len(border_mv) + 1)):
    for comb in itertools.combinations(border_mv, flip_count):
      timeout_lib.maybe_timeout_cooperatively()
      old_hashes = frozenset(c.hash_value for c in comb)
      local_creases = [
          c.flip_fold() if c.hash_value in old_hashes else c
          for c in v_cur.creases
      ]
      v_local = vertex.Vertex(local_creases, v_cur.point)
      if not (
          v_local.check_kawasaki
          and v_local.check_maekawa
          and v_local.check_crimp_angles
      ):
        continue
      flipped_cp = _build_cp_preserving_mode(
          [
              c.flip_fold() if c.hash_value in old_hashes else c
              for c in current_cp.creases
          ],
          light_mode=current_cp.light_mode,
      )
      new_hashes = frozenset(c.flip_fold().hash_value for c in comb)
      return flipped_cp, old_hashes, new_hashes
  return None


def get_flat_foldable(
    cp: cp_utils.CreasePattern,
    *,
    rectangles: Sequence[BoundingBox] | None = None,
    unmodifiable_creases: Sequence[crease_utils.Crease] | None = None,
    max_candidates: int | None = None,
    max_crease_modifications: int | None = None,
) -> cp_utils.CreasePattern | None:
  """Repairs local flat-foldability violations via best-first crease edits.

  Resolves Kawasaki angle imbalances by rotating modifiable mountain/valley
  creases and resolves Maekawa / Big-Little-Big crimp violations by flipping
  border-terminating creases incident to violating vertices.

  Args:
    cp: Crease pattern to repair.
    rectangles: Optional bounding boxes restricting which vertices and creases
      may be modified. Defaults to the full `[0, 1]^2` paper.
    unmodifiable_creases: Creases whose angles must remain fixed.
    max_candidates: Optional cap on candidate patterns added to the search heap.
    max_crease_modifications: Optional maximum edit depth per candidate chain.

  Returns:
    A locally flat-foldable `CreasePattern`, or `None` if no repair is found.
  """
  if rectangles is None:
    boxes: Sequence[BoundingBox] = [
        rectangle_lib.Rectangle(
            name='paper', x_start=0.0, y_start=0.0, x_len=1.0, y_len=1.0
        )
    ]
  else:
    boxes = rectangles
  if cp.light_mode and any(
      v.point.is_border and len(v.creases) == 1 for v in cp.vertices
  ):
    cp = _build_cp_preserving_mode(cp.creases, light_mode=True)

  if unmodifiable_creases is not None:
    matched = cp.get_matching_creases(list(unmodifiable_creases))
    unmodifiable_hashes = frozenset(
        c.hash_value for c in matched or unmodifiable_creases
    )
  else:
    unmodifiable_hashes = frozenset()

  root_cand = CandidateCreasePattern.from_cp(
      cp,
      unmodifiable_crease_hashes=unmodifiable_hashes,
      num_crease_modifications=0,
      candidate_num=0,
  )
  if root_cand.score == 1.0:
    return cp

  min_heap = [root_cand.to_tuple()]
  seen_hashes: set[int] = {cp.hash_value}
  cp_num = 1

  def _enqueue(
      next_cp: cp_utils.CreasePattern,
      next_unmod: frozenset[int],
      next_mods: int,
  ) -> tuple[cp_utils.CreasePattern | None, bool]:
    nonlocal cp_num
    cp_hash = next_cp.hash_value
    if cp_hash in seen_hashes:
      return None, False
    seen_hashes.add(cp_hash)
    if (
        max_crease_modifications is not None
        and next_mods > max_crease_modifications
    ):
      return None, False
    next_cand = CandidateCreasePattern.from_cp(
        next_cp,
        unmodifiable_crease_hashes=next_unmod,
        num_crease_modifications=next_mods,
        candidate_num=cp_num,
    )
    if next_cand.score == 1.0:
      return next_cp, True
    heapq.heappush(min_heap, next_cand.to_tuple())
    cp_num += 1
    if max_candidates is not None and cp_num > max_candidates:
      return None, True
    return None, False

  while min_heap:
    timeout_lib.maybe_timeout_cooperatively()
    _, _, _, _, cand = heapq.heappop(min_heap)

    for bad_vertex in cand.non_foldable_vertices:
      timeout_lib.maybe_timeout_cooperatively()
      in_box = any(_point_in_clipped_box(bad_vertex.point, b) for b in boxes)

      if in_box and bad_vertex.check_kawasaki:
        flip = _border_flip_candidate(cand.cp, bad_vertex.point)
        if flip is not None:
          flipped_cp, old_h, new_h = flip
          updated_unmod = (cand.unmodifiable_crease_hashes - old_h) | new_h
          solved_cp, stop = _enqueue(
              flipped_cp, updated_unmod, cand.num_crease_modifications + 1
          )
          if stop:
            return solved_cp

      within_box: list[crease_utils.Crease] = []
      crossing_box: list[crease_utils.Crease] = []
      for inc in bad_vertex.creases:
        if any(b.crease_is_within(inc) for b in boxes):
          within_box.append(inc)
        elif in_box:
          crossing_box.append(inc)

      for target_crease in within_box + crossing_box:
        if (
            not target_crease.mountain_or_valley()
            or target_crease.hash_value in cand.unmodifiable_crease_hashes
        ):
          continue
        rotated_cp, modified_hashes = get_new_cp_with_rotated_crease(
            cand.cp,
            target_crease,
            bad_vertex,
            cand.crease_hash_to_vertex_mapping,
        )
        if rotated_cp is None:
          continue

        next_unmod = cand.unmodifiable_crease_hashes | modified_hashes
        next_mods = cand.num_crease_modifications + 1
        solved_cp, stop = _enqueue(rotated_cp, next_unmod, next_mods)
        if stop:
          return solved_cp

        flip = _border_flip_candidate(rotated_cp, bad_vertex.point)
        if flip is not None:
          flipped_cp, old_h, new_h = flip
          if not (old_h & next_unmod):
            solved_cp, stop = _enqueue(
                flipped_cp, next_unmod | new_h, next_mods + 1
            )
            if stop:
              return solved_cp

  return None
