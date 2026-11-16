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

"""Flat foldable solver."""

import collections
import dataclasses
import heapq
import itertools
import math
from typing import Sequence

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import point as point_utils
from ..crease_pattern import rotation
from ..crease_pattern import vertex
from . import shaping_utils
from .. import timeout as timeout_lib


def split_border_creases_at_tips(
    creases: Sequence[crease_utils.Crease],
) -> list[crease_utils.Crease]:
  """Splits `Fold.BORDER` creases at border tips of non-border creases.

  When `CreasePattern(..., light_mode=True)` is used after moving or rotating a
  crease endpoint along a merged `Fold.BORDER` edge, `light_mode` skips global
  intersection splitting and leaves the border edge unsplit, causing `Vertex` to
  treat the border tip as an internal degree-1 vertex. Splitting only the border
  edges at border endpoints is O(N) and preserves valid border vertices.

  Args:
    creases: Input creases of a crease pattern.

  Returns:
    Creases with `Fold.BORDER` segments split at any internal border tips.
  """
  border_creases = [c for c in creases if c.fold == crease_utils.Fold.BORDER]
  if not border_creases:
    return list(creases)
  non_border = [c for c in creases if c.fold != crease_utils.Fold.BORDER]
  border_pts = []
  for c in non_border:
    for pt in (c.p1, c.p2):
      if pt.is_border:
        border_pts.append(pt)
  if not border_pts:
    return list(creases)
  split_borders: list[crease_utils.Crease] = []
  for bc in border_creases:
    internal_pts = [
        pt
        for pt in border_pts
        if bc.point_status(pt) == crease_utils.PointStatus.INTERNAL
    ]
    if not internal_pts:
      split_borders.append(bc)
      continue
    pts = sorted(
        internal_pts + [bc.p1, bc.p2],
        key=lambda p: (round(p.x, 7), round(p.y, 7)),
    )
    dedup = [pts[0]]
    for pt in pts[1:]:
      if not pt.isclose(dedup[-1]):
        dedup.append(pt)
    for p_a, p_b in zip(dedup, dedup[1:]):
      split_borders.append(dataclasses.replace(bc, p1_init=p_a, p2_init=p_b))
  return non_border + split_borders


def _build_cp_preserving_mode(
    creases: list[crease_utils.Crease], *, light_mode: bool
) -> cp_utils.CreasePattern:
  if light_mode:
    creases = split_border_creases_at_tips(creases)
  return cp_utils.CreasePattern(creases=creases, light_mode=light_mode)


@dataclasses.dataclass(kw_only=True)
class CandidateCreasePattern:
  """A candidate crease Pattern that is being modified to fix flat foldability."""

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
    """Initializes a CandidateCreasePattern from a Crease Pattern."""
    score, _, non_foldable_vertices = cp.vertex_flat_foldability_percentage()

    crease_hash_to_vertex_mapping: dict[int, list[vertex.Vertex]] = (
        collections.defaultdict(list)
    )
    for v in cp.vertices:
      timeout_lib.maybe_timeout_cooperatively()
      for c in v.creases:
        crease_hash_to_vertex_mapping[c.hash_value].append(v)

    border_distance = sum(
        min(v.point.x, 1.0 - v.point.x, v.point.y, 1.0 - v.point.y)
        for v in non_foldable_vertices
    )

    return cls(
        cp=cp,
        non_foldable_vertices=non_foldable_vertices,
        unmodifiable_crease_hashes=unmodifiable_crease_hashes,
        crease_hash_to_vertex_mapping=crease_hash_to_vertex_mapping,
        score=score,
        border_distance=border_distance,
        num_crease_modifications=num_crease_modifications,
        candidate_num=candidate_num,
    )

  def to_tuple(self) -> tuple[float, float, int, int, 'CandidateCreasePattern']:
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
  """Returns the other vertex connected to the given vertex and crease."""
  connected_vertices = crease_hash_to_vertex_mapping[
      connecting_crease.hash_value
  ]
  if len(connected_vertices) != 2:
    raise ValueError(
        'Crease must be connected to exactly two vertices (got'
        f' {len(connected_vertices)}).'
    )
  if connected_vertices[0].point.isclose(connected_vertex.point):
    return connected_vertices[1]
  elif connected_vertices[1].point.isclose(connected_vertex.point):
    return connected_vertices[0]
  else:
    raise ValueError(
        'Crease to be modified must be connected exactly to the violating'
        ' vertex and one other vertex.'
    )


# NOTE: Have a smarter way of picking which crease to rotate, based
# on crimp angles (see `Vertex.check_crimp_angles`).
def get_new_cp_with_rotated_crease(
    current_cp: cp_utils.CreasePattern,
    crease_to_be_rotated: crease_utils.Crease,
    violating_vertex: vertex.Vertex,
    crease_hash_to_vertex_mapping: dict[int, list[vertex.Vertex]],
) -> tuple[cp_utils.CreasePattern | None, frozenset[int]]:
  """Returns a new CP with a rotated crease and the hashes of the modified creases.

  Args:
    current_cp: The current CP.
    crease_to_be_rotated: The crease to be rotated.
    violating_vertex: The vertex that the crease to be rotated violates.
    crease_hash_to_vertex_mapping: A mapping of crease hashes to the vertices
      that they connect to.

  Returns:
    rotated_cp: The new CP with the rotated crease. If the rotation is invalid,
      this is None instead.
    modified_crease_hashes: The hashes of the creases that were modified in the
      new CP. If the rotation is invalid, this is an empty set instead.
  """
  new_creases: dict[int, crease_utils.Crease] = {
      c.hash_value: c for c in current_cp.creases
  }
  modified_creases: list[crease_utils.Crease] = []

  # Determine which alternating sum each angle belongs to.
  # sum_odd_angle is the sum of the angle made in the clockwise direction
  # of the crease to be modified and every other angle afterwards in the
  # clockwise direction.
  # sum_even_angle is the sum of the angle made in the
  # counter-clockwise direction of the crease to be modified and every other
  # angle afterwards in the counter-clockwise direction.
  crease_to_be_rotated_index = violating_vertex.creases.index(
      crease_to_be_rotated
  )
  sector_angles = [angle for angle, _, _ in violating_vertex.sector_angles()]
  if crease_to_be_rotated_index % 2 == 0:
    sum_odd_angles = sum(sector_angles[::2])
    sum_even_angles = sum(sector_angles[1::2])
  else:
    sum_odd_angles = sum(sector_angles[1::2])
    sum_even_angles = sum(sector_angles[::2])

  angle_delta = sum_odd_angles - sum_even_angles
  if math.isclose(angle_delta, 0):
    # Kawaski is satisfied but this current CP is not flat foldable, meaning
    # that some other flat foldable criteria is being violated.
    return None, frozenset()

  # Rotate by half the angle delta since every n degrees we rotate will increase
  # sum_odd_angle by n/2 and decrease sum_even_angle by
  # n/2. So in order to reduce the angle_delta from n to 0, we rotate clockwise
  # by n/2 degrees.
  rotation_delta = angle_delta / 2
  # The angle made from the crease to be modified, to the adjacent creases in
  # both the clockwise and counter-clockwise directions.
  clockwise_angle = sector_angles[crease_to_be_rotated_index]
  counter_clockwise_angle = sector_angles[
      (crease_to_be_rotated_index - 1) % len(sector_angles)
  ]
  if (rotation_delta >= clockwise_angle) or (
      -rotation_delta >= counter_clockwise_angle
  ):
    # rotation_delta is clockwise degrees to satisfy Kawasaki (negative=CCW). If
    # rotation overlaps/passes adjacent crease, violation is unfixable.
    return None, frozenset()

  rotated_crease = rotation.rotate_crease_clockwise(
      crease_to_rotate=crease_to_be_rotated,
      rotation_angle_degrees=rotation_delta,
      rotation_origin_tip=violating_vertex.point,
      require_on_paper=False,
  )

  # The rotated crease connects the violating vertex and one other (adjacent)
  # vertex. Find this other adjacent vertex. We will have to update this vertex
  # as well, with the newly rotated crease.
  adjacent_vertex = _get_other_connecting_vertex(
      violating_vertex, crease_to_be_rotated, crease_hash_to_vertex_mapping
  )

  # adjacent_creases are the creases in the adjacent_vertex that are
  # adjacent to the rotated crease.
  adjacent_creases: list[crease_utils.Crease] = []
  crease_modified_index = adjacent_vertex.creases.index(crease_to_be_rotated)
  for i in [1, -1]:
    adjacent_crease = adjacent_vertex.creases[
        (crease_modified_index + i) % len(adjacent_vertex.creases)
    ]
    if adjacent_crease not in adjacent_creases:
      adjacent_creases.append(adjacent_crease)

  # The rotated crease should intersect with exactly one of the adjacent_creases
  # of the adjacent_vertex.
  new_adjacent_vertex_point = None
  closest_distance = float('inf')
  for adjacent_crease in adjacent_creases:
    timeout_lib.maybe_timeout_cooperatively()
    intersection_point = rotated_crease.vertex_with_crease(
        adjacent_crease, extend_lines=True
    )
    if (not intersection_point) or adjacent_crease.point_status(
        intersection_point
    ) not in (crease_utils.PointStatus.INTERNAL, crease_utils.PointStatus.TIP):
      continue

    distance = point_utils.distance(violating_vertex.point, intersection_point)
    if distance < closest_distance:
      # The closest intersection point to the rotation origin
      # (violating_vertex.point) is where the rotated crease will end up
      # connecting to, on the adjacent crease.
      closest_distance = distance
      new_adjacent_vertex_point = intersection_point
  if not new_adjacent_vertex_point:
    # This means the intersection point lies outside the range of the adjacent
    # creases, meaning it's not possible to rotate this crease without
    # intersecting with another crease (this crease is connected to the tip of
    # the adjacent crease).
    return None, frozenset()

  if current_cp.light_mode:
    rotated_segment = crease_utils.Crease(
        violating_vertex.point,
        new_adjacent_vertex_point,
        fold=crease_to_be_rotated.fold,
    )
    exempt_hashes = {c.hash_value for c in adjacent_vertex.creases} | {
        c.hash_value for c in violating_vertex.creases
    }
    for other_crease in current_cp.creases:
      if (
          other_crease.fold == crease_utils.Fold.BORDER
          or other_crease.hash_value in exempt_hashes
      ):
        continue
      crossing = rotated_segment.vertex_with_crease(other_crease)
      if (
          crossing is not None
          and not crossing.isclose(new_adjacent_vertex_point)
          and not crossing.isclose(violating_vertex.point)
      ):
        return None, frozenset()

  # Create the new adjacent vertex with the new vertex point, and all creases
  # updated to the new vertex point.
  for adjacent_crease in adjacent_vertex.creases:
    timeout_lib.maybe_timeout_cooperatively()
    if adjacent_crease.fold == crease_utils.Fold.BORDER:
      continue

    # All original non-border creases attached to the adjacent vertex must be
    # removed, since all of them will be replaced by a new corresponding crease
    # that connects to the new_adjacent_vertex_point.
    del new_creases[adjacent_crease.hash_value]

    if adjacent_crease.p1.isclose(adjacent_vertex.point):
      unmodified_point = adjacent_crease.p2
    elif adjacent_crease.p2.isclose(adjacent_vertex.point):
      unmodified_point = adjacent_crease.p1
    else:
      raise ValueError(
          "Crease in adjacent vertex is not connected to the adjacent vertex's"
          ' point.'
      )

    if unmodified_point.isclose(new_adjacent_vertex_point):
      # we moved the vertex close enough to another vertex so this adjacent
      # crease should be eliminated.
      continue
    new_adjacent_crease = crease_utils.Crease(
        unmodified_point, new_adjacent_vertex_point, fold=adjacent_crease.fold
    )
    modified_creases.append(new_adjacent_crease)

  new_cp = _build_cp_preserving_mode(
      list(new_creases.values()) + modified_creases,
      light_mode=current_cp.light_mode,
  )

  return (
      new_cp,
      frozenset({c.hash_value for c in modified_creases}),
  )


def _point_in_clipped_rectangle(
    pt: point_utils.Point, rect: shaping_utils.Rectangle
) -> bool:
  clipped = rect.clip()
  if clipped is None:
    return False
  tol = point_utils.DEFAULT_TOLERANCE
  return (
      clipped.x_start - tol <= pt.x <= clipped.x_end + tol
      and clipped.y_start - tol <= pt.y <= clipped.y_end + tol
  )


def _border_flip_candidates(
    current_cp: cp_utils.CreasePattern,
    target_point: point_utils.Point,
) -> list[tuple[cp_utils.CreasePattern, frozenset[int], frozenset[int]]]:
  """Finds border-terminating M/V fold flips that fix Maekawa/BLB at `target_point`."""
  v_cur = next(
      (v for v in current_cp.vertices if v.point.isclose(target_point)), None
  )
  if (
      v_cur is None
      or not v_cur.check_kawasaki
      or (v_cur.check_maekawa and v_cur.check_crimp_angles)
  ):
    return []
  border_incident: list[crease_utils.Crease] = []
  for vc in v_cur.creases:
    if not vc.mountain_or_valley():
      continue
    other_pt = vc.p2 if vc.p1.isclose(v_cur.point) else vc.p1
    if other_pt.is_border:
      border_incident.append(vc)
  if not border_incident:
    return []

  out: list[tuple[cp_utils.CreasePattern, frozenset[int], frozenset[int]]] = []
  for r_size in range(1, min(3, len(border_incident) + 1)):
    for comb in itertools.combinations(border_incident, r_size):
      timeout_lib.maybe_timeout_cooperatively()
      old_hashes = frozenset(c.hash_value for c in comb)
      flipped_creases = [
          c.flip_fold() if c.hash_value in old_hashes else c
          for c in current_cp.creases
      ]
      flipped_cp = _build_cp_preserving_mode(
          flipped_creases, light_mode=current_cp.light_mode
      )
      v_flipped = next(
          (v for v in flipped_cp.vertices if v.point.isclose(target_point)),
          None,
      )
      if (
          v_flipped is not None
          and v_flipped.check_kawasaki
          and v_flipped.check_maekawa
          and v_flipped.check_crimp_angles
      ):
        new_hashes = frozenset(c.flip_fold().hash_value for c in comb)
        out.append((flipped_cp, old_hashes, new_hashes))
        return out
  return out


def get_flat_foldable(
    cp: cp_utils.CreasePattern,
    *,
    rectangles: Sequence[shaping_utils.Rectangle] | None = None,
    unmodifiable_creases: list[crease_utils.Crease] | None = None,
    max_candidates: int | None = None,
    max_crease_modifications: int | None = None,
) -> cp_utils.CreasePattern | None:
  """Attempts to adjust violating vertex crease angles to restore flat-foldability.

  Supports fixing Kawasaki violations via crease rotations and Maekawa /
  Big-Little-Big crimp-angle violations via M/V flips on border-terminating
  creases.

  Args:
    cp: The crease pattern to be modified.
    rectangles: Only creases within (or incident to vertices within) these
      rectangles are considered for modification. If None, then all creases are
      considered.
    unmodifiable_creases: Creases whose geometry should not be rotated.
    max_candidates: Maximum number of unique candidate CPs that can be added to
      the priority queue. If None, no limit is applied.
    max_crease_modifications: Maximum number of crease modifications (rotations)
      allowed in a single candidate chain. Candidates exceeding this depth are
      not added to the priority queue. If None, no limit is applied.

  Returns:
    A flat foldable Crease Pattern or None if no solution is found.
  """
  if rectangles is None:
    rectangles = [
        shaping_utils.Rectangle(
            name='paper', x_start=0.0, y_start=0.0, x_len=1.0, y_len=1.0
        )
    ]

  if cp.light_mode and any(
      v.point.is_border and len(v.creases) == 1 for v in cp.vertices
  ):
    cp = _build_cp_preserving_mode(cp.creases, light_mode=True)

  if unmodifiable_creases is not None:
    matched_unmod = cp.get_matching_creases(unmodifiable_creases)
    unmodifiable_crease_hashes = frozenset(
        {c.hash_value for c in matched_unmod or unmodifiable_creases}
    )
  else:
    unmodifiable_crease_hashes = frozenset()

  candidate_cp = CandidateCreasePattern.from_cp(
      cp,
      unmodifiable_crease_hashes=unmodifiable_crease_hashes,
      num_crease_modifications=0,
      candidate_num=0,
  )
  if candidate_cp.score == 1.0:
    return cp

  min_heap: list[tuple[float, float, int, int, CandidateCreasePattern]] = [
      candidate_cp.to_tuple()
  ]
  cp_hashes_seen: set[int] = {cp.hash_value}
  cp_num = 1

  def _enqueue(
      next_cp: cp_utils.CreasePattern,
      next_unmod: frozenset[int],
      next_mods: int,
  ) -> tuple[cp_utils.CreasePattern | None, bool]:
    nonlocal cp_num
    next_hash = next_cp.hash_value
    if next_hash in cp_hashes_seen:
      return None, False
    cp_hashes_seen.add(next_hash)
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
    _, _, _, _, candidate_cp = heapq.heappop(min_heap)

    for non_foldable_vertex in candidate_cp.non_foldable_vertices:
      timeout_lib.maybe_timeout_cooperatively()
      vertex_in_rectangle = any(
          _point_in_clipped_rectangle(non_foldable_vertex.point, rect)
          for rect in rectangles
      )
      if vertex_in_rectangle and non_foldable_vertex.check_kawasaki:
        for flipped_cp, old_h, new_h in _border_flip_candidates(
            candidate_cp.cp, non_foldable_vertex.point
        ):
          updated_unmod = (
              (candidate_cp.unmodifiable_crease_hashes - old_h) | new_h
              if old_h & candidate_cp.unmodifiable_crease_hashes
              else candidate_cp.unmodifiable_crease_hashes | new_h
          )
          solved_cp, stop = _enqueue(
              flipped_cp,
              updated_unmod,
              candidate_cp.num_crease_modifications + 1,
          )
          if stop:
            return solved_cp

      creases_in_rect = [
          c
          for c in non_foldable_vertex.creases
          if any(rect.crease_is_within(c) for rect in rectangles)
      ]
      if vertex_in_rectangle:
        creases_crossing_rect = [
            c
            for c in non_foldable_vertex.creases
            if not any(rect.crease_is_within(c) for rect in rectangles)
        ]
      else:
        creases_crossing_rect = []
      for crease_to_be_rotated in creases_in_rect + creases_crossing_rect:
        if not crease_to_be_rotated.mountain_or_valley():
          continue
        if (
            crease_to_be_rotated.hash_value
            in candidate_cp.unmodifiable_crease_hashes
        ):
          continue

        new_cp, modified_crease_hashes = get_new_cp_with_rotated_crease(
            candidate_cp.cp,
            crease_to_be_rotated,
            non_foldable_vertex,
            candidate_cp.crease_hash_to_vertex_mapping,
        )
        if new_cp is None:
          continue

        next_unmod = candidate_cp.unmodifiable_crease_hashes.union(
            modified_crease_hashes
        )
        next_mods = candidate_cp.num_crease_modifications + 1
        solved_cp, stop = _enqueue(new_cp, next_unmod, next_mods)
        if stop:
          return solved_cp

        for flipped_cp, old_h, new_h in _border_flip_candidates(
            new_cp, non_foldable_vertex.point
        ):
          if old_h & next_unmod:
            continue
          flipped_unmod = next_unmod | new_h
          solved_cp, stop = _enqueue(flipped_cp, flipped_unmod, next_mods + 1)
          if stop:
            return solved_cp

  return None
