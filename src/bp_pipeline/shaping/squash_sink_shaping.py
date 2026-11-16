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

"""Shaping functions for applying Squash Sink modifications to ridge creases."""

import dataclasses
import math
from typing import Any

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import point as point_utils
from . import flat_foldable
from . import shaping_utils
from . import vector


def _is_collinear_with_line(
    crease: crease_utils.Crease, line: crease_utils.Crease
) -> bool:
  """Checks collinearity by perpendicular distance (robust to 1e-9 endpoint jitter)."""
  dx = line.p2_init.x - line.p1_init.x
  dy = line.p2_init.y - line.p1_init.y
  length = math.hypot(dx, dy)
  if length <= 1e-9:
    return crease.is_collinear(line)
  ux, uy = dx / length, dy / length
  tol = point_utils.DEFAULT_TOLERANCE
  for pt in (crease.p1_init, crease.p2_init):
    vx = pt.x - line.p1_init.x
    vy = pt.y - line.p1_init.y
    if abs(vx * uy - vy * ux) > tol:
      return False
  return True


def _is_within_segment_on_line(
    crease: crease_utils.Crease, segment: crease_utils.Crease
) -> bool:
  """Checks whether `crease` lies collinearly within `segment` up to tolerance."""
  if not _is_collinear_with_line(crease, segment):
    return False
  dx = segment.p2_init.x - segment.p1_init.x
  dy = segment.p2_init.y - segment.p1_init.y
  length = math.hypot(dx, dy)
  if length <= 1e-9:
    return False
  ux, uy = dx / length, dy / length
  tol = point_utils.DEFAULT_TOLERANCE
  for pt in (crease.p1_init, crease.p2_init):
    proj = (pt.x - segment.p1_init.x) * ux + (pt.y - segment.p1_init.y) * uy
    if proj < -tol or proj > length + tol:
      return False
  return True


def _overlaps_on_line(
    crease: crease_utils.Crease, segment: crease_utils.Crease
) -> bool:
  """Checks whether `crease` and `segment` are collinear and overlap."""
  if not _is_collinear_with_line(crease, segment):
    return False
  dx = segment.p2_init.x - segment.p1_init.x
  dy = segment.p2_init.y - segment.p1_init.y
  length = math.hypot(dx, dy)
  if length <= 1e-9:
    return False
  ux, uy = dx / length, dy / length
  tol = point_utils.DEFAULT_TOLERANCE
  t1 = (crease.p1_init.x - segment.p1_init.x) * ux + (
      crease.p1_init.y - segment.p1_init.y
  ) * uy
  t2 = (crease.p2_init.x - segment.p1_init.x) * ux + (
      crease.p2_init.y - segment.p1_init.y
  ) * uy
  c_min, c_max = min(t1, t2), max(t1, t2)
  return c_max >= -tol and c_min <= length + tol


def get_bisector_point(
    p1: point_utils.Point, p2: point_utils.Point, p3: point_utils.Point
) -> point_utils.Point:
  """Gets an interior point on the interior bisector of angle p2-p1-p3.

  Args:
    p1: The point of the angle, at which we draw the bisector.
    p2: Point on one side of bisector.
    p3: Point on the other side of bisector.

  Returns:
    The point on the interior bisector that also passes through p2-p3 line.
  """
  # Note that an interior point is preferred because it minimizes casework,
  # however simply summing the unit vectors doesn't guarantee an interior point.
  constraint_vec = vector.Vector.from_points(p2, p3)
  p1p2_dist = point_utils.distance(p1, p2)
  p1p3_dist = point_utils.distance(p1, p3)
  ratio = p1p2_dist / (p1p2_dist + p1p3_dist)
  final_point = p2 + ratio * constraint_vec
  return final_point


def get_side_creases(
    point: point_utils.Point,
    cp: cp_utils.CreasePattern,
    ridge_crease: crease_utils.Crease,
) -> list[crease_utils.Crease]:
  """Gets non-ridge creases incident to the point."""
  incident_creases = cp.get_incident_creases(point)
  side_creases = []
  for c in incident_creases:
    if c.fold == crease_utils.Fold.BORDER:
      continue
    is_ridge = _is_collinear_with_line(c, ridge_crease)
    if not is_ridge:
      side_creases.append(c)
  if len(side_creases) < 2 and point.is_border:
    for c in cp.creases:
      if c.fold != crease_utils.Fold.BORDER:
        continue
      if c.point_status(point) == crease_utils.PointStatus.EXTERNAL:
        continue
      for end_pt in (c.p1_init, c.p2_init):
        if not end_pt.isclose(point):
          ray = crease_utils.Crease(
              point, end_pt, fold=crease_utils.Fold.BORDER
          )
          if not _is_collinear_with_line(ray, ridge_crease):
            side_creases.append(ray)
  return side_creases


def get_adjacent_side_creases(
    point: point_utils.Point,
    side_creases: list[crease_utils.Crease],
    toward: point_utils.Point,
) -> list[crease_utils.Crease]:
  """Returns the side creases angularly next to the ridge, one on each side.

  At a flap tip (circle center) every ridge of the flap meets, so the vertex
  has more than two side creases. The squash sink only needs the two that
  bound the faces next to the ridge segment, i.e. the creases closest in
  angle to the ridge direction on either side of it.

  Args:
    point: The vertex on the ridge.
    side_creases: Non-ridge creases incident to `point`.
    toward: A point along the ridge from `point` (the squashed segment's other
      end), giving the ridge direction.

  Returns:
    The two adjacent side creases, or `side_creases` unchanged if there are at
    most two or no crease lies on one side of the ridge.
  """
  if len(side_creases) <= 2:
    return side_creases
  closest = {}  # side (+1 / -1) -> (angle, crease)
  for crease, side, angle in _side_creases_by_side(point, side_creases, toward):
    if side not in closest or angle < closest[side][0]:
      closest[side] = (angle, crease)
  if len(closest) < 2:
    return side_creases
  return [closest[1][1], closest[-1][1]]


def _side_creases_by_side(
    point: point_utils.Point,
    side_creases: list[crease_utils.Crease],
    toward: point_utils.Point,
) -> list[tuple[crease_utils.Crease, int, float]]:
  """Labels each side crease at `point` with its side of the ridge and angle.

  Args:
    point: The vertex on the ridge.
    side_creases: Non-ridge creases incident to `point`.
    toward: A point along the ridge from `point`, giving the ridge direction.

  Returns:
    `(crease, side, angle)` triples; `side` is +1 / -1 (left / right of the
    ridge direction) and `angle` the unsigned angle from the ridge direction.
    Creases collinear with the ridge are left out.
  """
  ridge_x, ridge_y = toward.x - point.x, toward.y - point.y
  labelled = []
  for crease in side_creases:
    other = crease.p2_init if crease.p1_init.isclose(point) else crease.p1_init
    side_x, side_y = other.x - point.x, other.y - point.y
    angle = math.atan2(
        ridge_x * side_y - ridge_y * side_x, ridge_x * side_x + ridge_y * side_y
    )
    if math.isclose(angle, 0.0, abs_tol=1e-9) or math.isclose(
        abs(angle), math.pi, abs_tol=1e-9
    ):
      continue
    labelled.append((crease, 1 if angle > 0 else -1, abs(angle)))
  return labelled


def _is_axis_aligned(crease: crease_utils.Crease) -> bool:
  dx = abs(crease.p2_init.x - crease.p1_init.x)
  dy = abs(crease.p2_init.y - crease.p1_init.y)
  return (
      dx <= point_utils.DEFAULT_TOLERANCE or dy <= point_utils.DEFAULT_TOLERANCE
  )


def get_parallel_side_crease_pairs(
    point_b: point_utils.Point,
    side_creases_b: list[crease_utils.Crease],
    point_c: point_utils.Point,
    side_creases_c: list[crease_utils.Crease],
    toward: point_utils.Point,
) -> tuple[list[crease_utils.Crease], list[crease_utils.Crease]] | None:
  """Picks one parallel (B, C) side-crease pair on each side of the ridge.

  The kite of a squash sink needs the side creases at B and at C on the same
  side of the ridge to be parallel pleats. On a shaped pattern a vertex also
  carries zig-zag, narrowing or kite creases, so the crease angularly nearest
  to the ridge is often not the pleat; this scans every combination instead
  and prefers axis-aligned pleats, then the pair nearest to the ridge.

  Args:
    point_b: Ridge vertex closer to the corner.
    side_creases_b: Non-ridge creases incident to B.
    point_c: Ridge vertex closer to the flap tip.
    side_creases_c: Non-ridge creases incident to C.
    toward: A point along the ridge beyond B on the corner side (A), fixing the
      ridge direction used to tell the two sides apart.

  Returns:
    `([b_plus, b_minus], [c_plus, c_minus])`, or None when one side has no
    parallel pair.
  """
  labelled_b = _side_creases_by_side(point_b, side_creases_b, toward)
  labelled_c = _side_creases_by_side(point_c, side_creases_c, point_b)
  chosen_b: dict[int, crease_utils.Crease] = {}
  chosen_c: dict[int, crease_utils.Crease] = {}
  for side in (1, -1):
    best = None
    for crease_b, side_b, angle_b in labelled_b:
      if side_b != side:
        continue
      for crease_c, side_c, angle_c in labelled_c:
        if side_c != side or not crease_b.is_parallel(crease_c):
          continue
        axis_rank = 0 if _is_axis_aligned(crease_b) else 1
        key = (axis_rank, angle_b + angle_c)
        if best is None or key < best[0]:
          best = (key, crease_b, crease_c)
    if best is None:
      return None
    chosen_b[side] = best[1]
    chosen_c[side] = best[2]
  return [chosen_b[1], chosen_b[-1]], [chosen_c[1], chosen_c[-1]]


def apply_squash_sink_to_segment(
    crease_ab: crease_utils.Crease,
    crease_bc: crease_utils.Crease,
    cp: cp_utils.CreasePattern,
    all_ridge_creases: list[crease_utils.Crease],
) -> tuple[list[crease_utils.Crease], list[crease_utils.Crease]]:
  """Applies the squash sink logic to the ridge segments AB and BC.

  Args:
    crease_ab: The first ridge segment on the squash sink closer to the corner.
    crease_bc: The second ridge segment, closer to the center.
    cp: The full CreasePattern.
    all_ridge_creases: The list of all ridge creases in the CreasePattern.

  Returns:
    A tuple of lists:
    - new_creases: The list of new creases created by the squash sink.
    - creases_to_remove: The list of old creases to be removed.
  """
  # Summary of steps:
  # 1. Get the side creases to B and C, that are not on the ridge.
  # 2. Orient them correctly for convenient calculation.
  # 3. Get bisector points on the side creases, so that we can draw bisector
  #    lines.
  # 4. Find the bisectors' intersections (E, F) with the side creases of B.
  # 5. Complete the kite shape by finding point G on the AB ridge segment.
  # 6. Finalize the list of creases to add and remove.

  point_a, point_b, point_c = (
      crease_ab.p1_init,
      crease_bc.p1_init,
      crease_bc.p2_init,
  )
  fold_bc_direction = crease_bc.fold  # Whether it's a mountain or valley.

  all_side_creases_b = get_side_creases(point_b, cp, all_ridge_creases[0])
  all_side_creases_c = get_side_creases(point_c, cp, all_ridge_creases[0])
  # The kite needs the pleat through B and the pleat through C on each side
  # of the ridge to be parallel. On a shaped pattern the vertices also carry
  # zig-zag / narrowing / kite creases, so pick a parallel pair per side
  # (axis-aligned pleats first); fall back to the two creases angularly next
  # to the ridge (the only candidates on a bare base, where C is often the
  # flap tip at which every ridge of the flap meets).
  pairs = get_parallel_side_crease_pairs(
      point_b, all_side_creases_b, point_c, all_side_creases_c, point_a
  )
  if pairs is not None:
    side_creases_b, side_creases_c = pairs
  else:
    side_creases_b = get_adjacent_side_creases(
        point_b, all_side_creases_b, point_a
    )
    side_creases_c = get_adjacent_side_creases(
        point_c, all_side_creases_c, point_b
    )

  if len(side_creases_b) != 2 or len(side_creases_c) != 2:
    raise ValueError(
        "Expected 2 side creases at the ridge endpoints, got"
        f" {len(side_creases_b)} and {len(side_creases_c)}"
    )

  # Find the side creases incident to B and C. Orient their p1_init to be on the
  # ridge, for convenience.
  # Also arrange b_side_crease_1 to be the side crease that's on the same side
  # as c_side_crease_1, for convenience.
  b_side_crease_1, b_side_crease_2 = (
      side_creases_b[0]
      if side_creases_b[0].p1_init.isclose(point_b)
      else side_creases_b[0].flip_tips(),
      side_creases_b[1]
      if side_creases_b[1].p1_init.isclose(point_b)
      else side_creases_b[1].flip_tips(),
  )
  c_side_crease_1, c_side_crease_2 = (
      side_creases_c[0]
      if side_creases_c[0].p1_init.isclose(point_c)
      else side_creases_c[0].flip_tips(),
      side_creases_c[1]
      if side_creases_c[1].p1_init.isclose(point_c)
      else side_creases_c[1].flip_tips(),
  )
  if not b_side_crease_1.is_parallel(c_side_crease_1):
    b_side_crease_1, b_side_crease_2 = b_side_crease_2, b_side_crease_1

  if not b_side_crease_1.is_parallel(c_side_crease_1):
    raise ValueError(
        "No parallel pleats on the ridge, which are needed for squash sink."
    )

  # Bisector vectors from C.
  # When applying squash sink, the kite-shape uses bisector lines.
  # this is because when the squash opens up the ridge, the pleats on either
  # side lie flat, setting equality on the two angles.
  bisector_point_1 = get_bisector_point(
      point_c, point_b, c_side_crease_1.p2_init
  )
  bisector_line1 = crease_utils.Crease(
      point_c, bisector_point_1, fold=fold_bc_direction
  )
  bisector_point_2 = get_bisector_point(
      point_c, point_b, c_side_crease_2.p2_init
  )
  bisector_line2 = crease_utils.Crease(
      point_c, bisector_point_2, fold=fold_bc_direction
  )

  # The bisector must meet the pleat through B. When the pleat is split into
  # short segments (a crossing vertex close to B), it may meet a later
  # collinear segment instead of the first one: walk the pleat line outward.
  point_e, b_chain_1 = _bisector_hit_on_pleat_line(
      bisector_line1, b_side_crease_1, cp, all_ridge_creases
  )
  point_f, b_chain_2 = _bisector_hit_on_pleat_line(
      bisector_line2, b_side_crease_2, cp, all_ridge_creases
  )

  if point_e is None or point_f is None:
    raise ValueError(
        "Could not construct bisector intersection points for squash sink."
    )
  # if e or f are external, they will be tuples.
  if not isinstance(point_e, point_utils.Point) or not isinstance(
      point_f, point_utils.Point
  ):
    raise ValueError(
        "Warning: Points E and F are external, so the squash will be a"
        " triangle."
    )

  # Find G on AB
  point_g = crease_utils.Crease(
      point_e, point_f, fold=fold_bc_direction
  ).vertex_with_crease(crease_ab, extend_lines=True, include_external=True)
  if point_g is None:
    raise ValueError("Could not construct point for squash sink.")

  if crease_ab.point_status(point_g) is not crease_utils.PointStatus.INTERNAL:
    raise ValueError("Could not construct point for squash sink.")

  # New creases (the kite shape)
  crease_ce = crease_utils.Crease(point_c, point_e, fold=fold_bc_direction)
  crease_cf = crease_utils.Crease(point_c, point_f, fold=fold_bc_direction)
  crease_eg = crease_utils.Crease(point_e, point_g, fold=fold_bc_direction)
  crease_gf = crease_utils.Crease(point_g, point_f, fold=fold_bc_direction)
  crease_ag = crease_utils.Crease(
      point_a, point_g, fold=fold_bc_direction, elevation=crease_ab.elevation
  )
  crease_ae = crease_utils.Crease(
      point_a, point_e, fold=fold_bc_direction.flip()
  )
  crease_af = crease_utils.Crease(
      point_a, point_f, fold=fold_bc_direction.flip()
  )
  crease_gc = crease_utils.Crease(
      point_g, point_c, fold=fold_bc_direction.flip()
  )

  # New creases that replace existing ones: B_S1, B_S2.
  # These are simply removing the portion of B_S1 and B_S2 that was along EB and
  # BF respectively (across every pleat segment between B and E / F).
  new_side_creases: list[crease_utils.Crease] = []
  creases_to_remove = [crease_ab, crease_bc]
  for hit_point, chain in ((point_e, b_chain_1), (point_f, b_chain_2)):
    creases_to_remove.extend(chain)
    last = chain[-1]
    if not hit_point.isclose(last.p2_init):
      new_side_creases.append(
          crease_utils.Crease(
              hit_point,
              last.p2_init,
              fold=last.fold,
              elevation=last.elevation,
          )
      )

  kite_creases = [
      crease_ce,
      crease_cf,
      crease_eg,
      crease_gf,
      crease_ag,
      crease_ae,
      crease_af,
      crease_gc,
  ]
  new_creases = kite_creases + new_side_creases

  return new_creases, creases_to_remove


def _bisector_hit_on_pleat_line(
    bisector: crease_utils.Crease,
    first_segment: crease_utils.Crease,
    cp: cp_utils.CreasePattern,
    all_ridge_creases: list[crease_utils.Crease],
) -> tuple[Any, list[crease_utils.Crease]]:
  """Intersects a kite bisector with the pleat line through B.

  Args:
    bisector: Bisector ray from C (p1 = C).
    first_segment: The pleat segment at B, oriented with `p1_init` at B.
    cp: The crease pattern.
    all_ridge_creases: Ridge creases (never part of a pleat chain).

  Returns:
    `(hit, chain)`: the intersection point (a `Point`, a tuple when it lies
    outside the paper, or None) and the pleat segments from B up to and
    including the one holding the hit; `[first_segment]` when the hit is on
    the first segment or there is no hit.
  """
  hit = bisector.vertex_with_crease(
      first_segment, extend_lines=True, include_external=False
  )
  chain = [first_segment]
  if hit is not None:
    return hit, chain
  # Walk the collinear pleat segments away from B (at most a few steps).
  current = first_segment
  for _ in range(4):
    end = current.p2_init
    next_segment = None
    for candidate in cp.creases:
      if (
          candidate.fold == crease_utils.Fold.BORDER
          or candidate.isclose(current)
          or any(
              _is_collinear_with_line(candidate, r) for r in all_ridge_creases
          )
          or not _is_collinear_with_line(candidate, first_segment)
      ):
        continue
      if candidate.p1_init.isclose(end) and not candidate.p2_init.isclose(
          current.p1_init
      ):
        next_segment = candidate
      elif candidate.p2_init.isclose(end) and not candidate.p1_init.isclose(
          current.p1_init
      ):
        next_segment = candidate.flip_tips()
      if next_segment is not None:
        break
    if next_segment is None:
      return None, chain
    chain.append(next_segment)
    hit = bisector.vertex_with_crease(
        next_segment, extend_lines=True, include_external=False
    )
    if hit is not None:
      return hit, chain
    current = next_segment
  return None, chain


def apply_border_triangle_squash_to_segment(
    crease_bc: crease_utils.Crease,
    cp: cp_utils.CreasePattern,
    all_ridge_creases: list[crease_utils.Crease],
) -> tuple[list[crease_utils.Crease], list[crease_utils.Crease]]:
  """Applies a triangular border squash to a 1-segment border-clipped ridge BC.

  When the theoretical flap corner A lies outside the unit paper so the ridge is
  clipped by the paper border at B (`point_b.is_border`) and ends at an internal
  vertex C (`point_c.is_internal`), the kite's outer half is clipped by the
  paper border, leaving the two 22.5-degree bisector rays `CE` and `CF` from `C`
  to the paper border together with the flipped ridge `CB`.

  Args:
    crease_bc: The ridge segment oriented from border point B to internal C.
    cp: The full CreasePattern.
    all_ridge_creases: Ridge creases along the ridge line.

  Returns:
    `(new_creases, creases_to_remove)`.
  """
  point_b, point_c = crease_bc.p1_init, crease_bc.p2_init
  if not point_b.is_border or not point_c.is_internal:
    raise ValueError(
        "Border triangle squash requires B on paper border and C internal."
    )
  fold_bc = crease_bc.fold
  if not crease_bc.mountain_or_valley():
    raise ValueError("Ridge segment must be mountain or valley.")

  side_c = get_side_creases(point_c, cp, all_ridge_creases[0])
  side_c = get_adjacent_side_creases(point_c, side_c, point_b)
  if len(side_c) != 2:
    raise ValueError(
        f"Expected 2 side creases at C for border squash, got {len(side_c)}."
    )
  c_side_1 = (
      side_c[0] if side_c[0].p1_init.isclose(point_c) else side_c[0].flip_tips()
  )
  c_side_2 = (
      side_c[1] if side_c[1].p1_init.isclose(point_c) else side_c[1].flip_tips()
  )

  p = lambda x, y: point_utils.Point(x=x, y=y)
  border_lines = [
      c for c in cp.creases if c.fold == crease_utils.Fold.BORDER
  ] or [
      crease_utils.Crease(p(0, 0), p(1, 0), fold=crease_utils.Fold.BORDER),
      crease_utils.Crease(p(1, 0), p(1, 1), fold=crease_utils.Fold.BORDER),
      crease_utils.Crease(p(1, 1), p(0, 1), fold=crease_utils.Fold.BORDER),
      crease_utils.Crease(p(0, 1), p(0, 0), fold=crease_utils.Fold.BORDER),
  ]

  bisector_hits: list[point_utils.Point] = []
  for c_side in (c_side_1, c_side_2):
    bp = get_bisector_point(point_c, point_b, c_side.p2_init)
    ray = crease_utils.Crease(point_c, bp, fold=fold_bc)
    hit: point_utils.Point | None = None
    best_dist = float("inf")
    for bc in border_lines:
      pt = ray.vertex_with_crease(bc, extend_lines=True, include_external=False)
      if (
          isinstance(pt, point_utils.Point)
          and bc.point_status(pt) is not crease_utils.PointStatus.EXTERNAL
      ):
        dot = (pt.x - point_c.x) * (bp.x - point_c.x) + (pt.y - point_c.y) * (
            bp.y - point_c.y
        )
        if dot > 1e-6:
          dist = point_utils.distance(point_c, pt)
          if dist < best_dist:
            best_dist = dist
            hit = pt
    if hit is None:
      raise ValueError("Bisector ray did not hit paper border.")
    bisector_hits.append(hit)

  if len(bisector_hits) != 2:
    raise ValueError(f"Expected 2 bisector hits, got {len(bisector_hits)}")
  point_e, point_f = bisector_hits[0], bisector_hits[1]
  crease_ce = crease_utils.Crease(point_c, point_e, fold=fold_bc)
  crease_cb = crease_utils.Crease(
      point_c, point_b, fold=fold_bc.flip(), elevation=crease_bc.elevation
  )
  crease_cf = crease_utils.Crease(point_c, point_f, fold=fold_bc)
  new_creases = [crease_ce, crease_cb, crease_cf]

  for nc in new_creases:
    for oc in cp.creases:
      if (
          oc.fold == crease_utils.Fold.BORDER
          or oc == crease_bc
          or _is_within_segment_on_line(crease_bc, oc)
      ):
        continue
      ip = nc.vertex_with_crease(oc)
      if ip is not None and not ip.isclose(point_c) and not ip.isclose(point_b):
        raise ValueError(
            f"Border triangle crease {nc} intersects {oc} at {ip}."
        )
  return new_creases, [crease_bc]


def get_oriented_ordered_crease_segments_on_ridge(
    cp: cp_utils.CreasePattern,
    ridge: crease_utils.Crease,
    ridge_start_point: point_utils.Point,
) -> list[crease_utils.Crease]:
  """Finds the list of connected, collinear crease segments on the given ridge.

  Args:
      cp: The CreasePattern to search within.
      ridge: A Crease object representing the line to search along. The fold
        assignment of this ridge is not used, only its geometry.
      ridge_start_point: The point to sort the creases by, and arrange the tips
        by. This is usually on one end of the ridge.

  Returns:
      A list of creases, which are ordered along the ridge line, and with tips
      ordered by closeness to ridge_start_point.
  """
  collinear_creases: list[crease_utils.Crease] = []
  vertex_points = [v.point for v in cp.vertices]

  # Orient the ridges in the same direction: set each ridge's p1_init to be
  # the closest point on the ridge start point.
  for c in cp.creases:
    if c.fold == crease_utils.Fold.BORDER:
      continue
    if _is_within_segment_on_line(c, ridge):
      internal_pts = [
          pt
          for pt in vertex_points
          if c.point_status(pt) == crease_utils.PointStatus.INTERNAL
      ]
      if internal_pts:
        pts = sorted(
            internal_pts + [c.p1_init, c.p2_init],
            key=lambda p: point_utils.distance(ridge_start_point, p),
        )
        dedup = [pts[0]]
        for pt in pts[1:]:
          if not pt.isclose(dedup[-1]):
            dedup.append(pt)
        for p_a, p_b in zip(dedup, dedup[1:]):
          if point_utils.distance(p_a, p_b) <= point_utils.DEFAULT_TOLERANCE:
            continue  # Sub-tolerance sliver between two near-duplicate vertices.
          collinear_creases.append(
              dataclasses.replace(c, p1_init=p_a, p2_init=p_b)
          )
      else:
        not_flip_tip = point_utils.distance(
            ridge_start_point, c.p1_init
        ) < point_utils.distance(ridge_start_point, c.p2_init)
        collinear_creases.append(c if not_flip_tip else c.flip_tips())

  # Creases closer to ridge_start_point are sorted to the front.
  collinear_creases.sort(
      key=lambda c: point_utils.distance(ridge_start_point, c.p1_init)
  )
  return collinear_creases


def _remove_crease_or_subsegment(
    creases: list[crease_utils.Crease], target: crease_utils.Crease
) -> None:
  """Removes `target` from `creases`, splitting a containing merged crease if needed."""
  if target in creases:
    creases.remove(target)
    return
  for idx, c in enumerate(creases):
    if c.fold == crease_utils.Fold.BORDER:
      continue
    if _is_within_segment_on_line(target, c):
      creases.pop(idx)
      base_pt = c.p1_init
      pts = sorted(
          [c.p1_init, c.p2_init, target.p1_init, target.p2_init],
          key=lambda p, b=base_pt: point_utils.distance(b, p),
      )
      dedup = [pts[0]]
      for pt in pts[1:]:
        if not pt.isclose(dedup[-1]):
          dedup.append(pt)
      for p_a, p_b in zip(dedup, dedup[1:]):
        sub = dataclasses.replace(c, p1_init=p_a, p2_init=p_b)
        if not (
            (
                sub.p1_init.isclose(target.p1_init)
                and sub.p2_init.isclose(target.p2_init)
            )
            or (
                sub.p1_init.isclose(target.p2_init)
                and sub.p2_init.isclose(target.p1_init)
            )
        ):
          creases.append(sub)
      return


def update_cp(
    cp: cp_utils.CreasePattern,
    new_creases: list[crease_utils.Crease],
    creases_to_remove: list[crease_utils.Crease],
) -> cp_utils.CreasePattern:
  """Returns a new CreasePattern with `new_creases` added and `creases_to_remove` dropped."""
  final_creases = cp.creases.copy()
  for c in creases_to_remove:
    _remove_crease_or_subsegment(final_creases, c)
  final_creases.extend(new_creases)
  if cp.light_mode:
    final_creases = flat_foldable.split_border_creases_at_tips(final_creases)
  return cp_utils.CreasePattern(creases=final_creases, light_mode=cp.light_mode)


def get_all_ridges_from_rectangles(
    rectangles: list[shaping_utils.Rectangle],
) -> list[crease_utils.Crease]:
  """Gets all ridges (center to corner) based on the rectangles."""
  ridges = []
  # Return a separate ridge for each running from circle center to corner.
  for rectangle in rectangles:
    if rectangle.circle_centers is None:
      continue
    elif len(rectangle.circle_centers) == 1:
      # For square rectangles, shaping_elements.get_ridges() will return
      # the diagonal ridge as a single ridge. However, we need separate ridges
      # for each center-to-corner.
      ridges.extend([
          crease_utils.Crease(
              rectangle.circle_centers[0],
              corner,
              fold=crease_utils.Fold.MOUNTAIN,
          )
          for corner in rectangle.corners
      ])
    elif len(rectangle.circle_centers) == 2:
      current_rect_ridges = list(rectangle.get_ridges())
      central_ridge = crease_utils.Crease(
          rectangle.circle_centers[0],
          rectangle.circle_centers[1],
          fold=crease_utils.Fold.MOUNTAIN,
      )
      # central_ridge does not participate in squash sink
      current_rect_ridges.remove(central_ridge)
      ridges.extend(current_rect_ridges)
  return ridges


def get_full_ridge(
    ridge: crease_utils.Crease, ridges: list[crease_utils.Crease]
) -> crease_utils.Crease:
  """Gets the full ridge that the ridge is a segment of."""
  full_ridge = None
  for r in ridges:
    if r.is_overlapping(ridge) or _overlaps_on_line(ridge, r):
      full_ridge = r
  if full_ridge is None:
    raise ValueError(f"Input {ridge} is not a ridge.")
  return full_ridge


def get_start_side_of_ridge(
    ridge_segment: crease_utils.Crease,
    full_ridge: crease_utils.Crease,
    rectangles: list[shaping_utils.Rectangle],
) -> point_utils.Point:
  """Gets the start side of the ridge."""
  # determine which side of the ridge segment is the start side -- ie rectangle
  # corner.
  start_side_of_full_ridge, end_side_of_full_ridge = None, None
  p1, p2 = full_ridge.p1_init, full_ridge.p2_init
  for rectangle in rectangles:
    # since rectangles can be partly external, use circle centers to check
    # if the other point is a rectangle corner.
    centers = rectangle.circle_centers
    for center in centers:
      if p1.isclose(center):
        end_side_of_full_ridge, start_side_of_full_ridge = p1, p2
      elif p2.isclose(center):
        end_side_of_full_ridge, start_side_of_full_ridge = p2, p1
      else:
        continue
      # verify that the start side is one of the rectangle corners.
      if not any(
          start_side_of_full_ridge.isclose(corner)
          for corner in rectangle.corners
      ):
        raise ValueError(f"Input {ridge_segment} is not a ridge.")
      break
  if start_side_of_full_ridge is None or end_side_of_full_ridge is None:
    raise ValueError(f"Input {ridge_segment} is not a ridge.")

  # after knowing which side of the full ridge is the start side (corner),
  # extrapolate it to the ridge segment to also find the start side of the
  # segment.
  p1_closer_to_corner = point_utils.distance(
      ridge_segment.p1_init, start_side_of_full_ridge
  ) < point_utils.distance(ridge_segment.p2_init, start_side_of_full_ridge)
  start_side_of_ridge = (
      ridge_segment.p1_init if p1_closer_to_corner else ridge_segment.p2_init
  )
  return start_side_of_ridge


def shape_cp_with_squash_sink(
    cp: cp_utils.CreasePattern,  # base cp, not the packed cp.
    rectangles: list[shaping_utils.Rectangle],
    ridge: crease_utils.Crease,
) -> cp_utils.CreasePattern:
  """Applies squash sink operations along segmented ridges in a crease pattern.

  Args:
      cp: The input crease pattern, expected to contain segmented ridges.
      rectangles: the packing rectangles of the cp.
      ridge: the ridge to apply squash sink on. If we only want to apply squash
        to one segment, the ridge should be shortened to specify the segment to
        apply to. Since squash sink applies to pairs of adjacent zebra creases,
        this ridge needs to have an even number of creases on it or else we will
        arbitrarily apply squash sink to one end.

  Returns:
      A new CreasePattern with squash sink modifications.
  """
  # 1. find the full ridge that the ridge is a segment of.
  ridges = get_all_ridges_from_rectangles(rectangles)
  full_ridge = get_full_ridge(ridge, ridges)

  # 2. orient the ridge to decide which side to start on. This matters because
  # squash sink has directionality.
  start_side_of_ridge = get_start_side_of_ridge(ridge, full_ridge, rectangles)

  # 3. get the creases on the ridge, in oriented and sorted order.
  ridge_segments = get_oriented_ordered_crease_segments_on_ridge(
      cp, ridge, start_side_of_ridge
  )

  # 4. start from the start side, take pairs of creases and apply squash sink.
  current_cp = cp
  applied_any = False
  for i in range(0, len(ridge_segments) - 1, 2):
    try:
      current_new_creases, current_creases_to_remove = (
          apply_squash_sink_to_segment(
              ridge_segments[i],
              ridge_segments[i + 1],
              current_cp,
              ridge_segments,
          )
      )
      current_cp = update_cp(
          current_cp, current_new_creases, current_creases_to_remove
      )
      applied_any = True
    except ValueError as e:
      print(f"Skipping squash sink on segment {ridge_segments[i]}. Error: {e}")
      continue

  # 5. If no 2-segment kite squash applied and the ridge begins with a
  # border-clipped segment BC (B on border, C internal), apply a border
  # triangular squash.
  if (
      not applied_any
      and ridge_segments
      and ridge_segments[0].p1_init.is_border
      and ridge_segments[0].p2_init.is_internal
  ):
    try:
      tri_new, tri_remove = apply_border_triangle_squash_to_segment(
          ridge_segments[0], current_cp, ridge_segments
      )
      current_cp = update_cp(current_cp, tri_new, tri_remove)
    except ValueError as e:
      print(
          "Skipping border triangle squash on segment"
          f" {ridge_segments[0]}. Error: {e}"
      )

  return current_cp
