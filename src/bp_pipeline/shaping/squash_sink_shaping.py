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

"""Squash-sink shaping primitives along segmented box-pleating flap ridges."""

from collections.abc import Sequence
import math

from absl import logging

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import flat_foldable
from ..crease_pattern import point as point_utils
from . import shaping_utils
from . import vector

_MAX_PLEAT_CHAIN_STEPS = 4


def _line_frame(
    line: crease_utils.Crease,
) -> tuple[float, float, float, float, float] | None:
  """Returns `(x0, y0, ux, uy, length)` for `line`, or `None` if degenerate."""
  x0, y0 = line.p1_init.x, line.p1_init.y
  dx = line.p2_init.x - x0
  dy = line.p2_init.y - y0
  length = math.hypot(dx, dy)
  if length <= 1e-9:
    return None
  return x0, y0, dx / length, dy / length, length


def _collinear_projections(
    crease: crease_utils.Crease, line: crease_utils.Crease
) -> tuple[float, float, float] | None:
  """Returns `(t1, t2, length)` along `line` if `crease` is collinear."""
  frame = _line_frame(line)
  if frame is None:
    return None
  x0, y0, ux, uy, length = frame
  tol = point_utils.DEFAULT_TOLERANCE
  projs: list[float] = []
  for pt in (crease.p1_init, crease.p2_init):
    vx, vy = pt.x - x0, pt.y - y0
    if abs(vx * uy - vy * ux) > tol:
      return None
    projs.append(vx * ux + vy * uy)
  return projs[0], projs[1], length


def _is_collinear_with_line(
    crease: crease_utils.Crease, line: crease_utils.Crease
) -> bool:
  """Returns True if `crease` lies on the infinite line through `line`."""
  frame = _line_frame(line)
  if frame is None:
    return crease.is_collinear(line)
  return _collinear_projections(crease, line) is not None


def _is_within_segment_on_line(
    crease: crease_utils.Crease, segment: crease_utils.Crease
) -> bool:
  """Returns True if `crease` lies collinearly within `segment`."""
  proj = _collinear_projections(crease, segment)
  if proj is None:
    return False
  t1, t2, length = proj
  tol = point_utils.DEFAULT_TOLERANCE
  return -tol <= min(t1, t2) and max(t1, t2) <= length + tol


def _overlaps_on_line(
    crease: crease_utils.Crease, segment: crease_utils.Crease
) -> bool:
  """Returns True if `crease` and `segment` are collinear and overlap."""
  proj = _collinear_projections(crease, segment)
  if proj is None:
    return False
  t1, t2, length = proj
  tol = point_utils.DEFAULT_TOLERANCE
  return max(t1, t2) >= -tol and min(t1, t2) <= length + tol


def _orient_from(
    crease: crease_utils.Crease, origin: point_utils.Point
) -> crease_utils.Crease:
  """Returns `crease` oriented with `p1_init` at `origin`."""
  return crease if crease.p1_init.isclose(origin) else crease.flip_tips()


def _unit_borders() -> list[crease_utils.Crease]:
  b = crease_utils.Fold.BORDER
  corners = [
      point_utils.Point(x=0.0, y=0.0),
      point_utils.Point(x=1.0, y=0.0),
      point_utils.Point(x=1.0, y=1.0),
      point_utils.Point(x=0.0, y=1.0),
  ]
  return [
      crease_utils.Crease(corners[i], corners[(i + 1) % 4], fold=b)
      for i in range(4)
  ]


def get_bisector_point(
    p1: point_utils.Point, p2: point_utils.Point, p3: point_utils.Point
) -> point_utils.Point:
  """Returns the interior angle-bisector intersection of `p2-p1-p3` on p2-p3."""
  d12 = point_utils.distance(p1, p2)
  d13 = point_utils.distance(p1, p3)
  ratio = d12 / (d12 + d13)
  return p2 + ratio * vector.Vector.from_points(p2, p3)


def get_side_creases(
    point: point_utils.Point,
    cp: cp_utils.CreasePattern,
    ridge_crease: crease_utils.Crease,
) -> list[crease_utils.Crease]:
  """Returns non-ridge creases incident to `point` (including border rays)."""
  side = [
      c
      for c in cp.get_incident_creases(point)
      if c.fold != crease_utils.Fold.BORDER
      and not _is_collinear_with_line(c, ridge_crease)
  ]
  if len(side) < 2 and point.is_border:
    for c in cp.creases:
      if (
          c.fold != crease_utils.Fold.BORDER
          or c.point_status(point) == crease_utils.PointStatus.EXTERNAL
      ):
        continue
      for end_pt in (c.p1_init, c.p2_init):
        if not end_pt.isclose(point):
          ray = crease_utils.Crease(
              point, end_pt, fold=crease_utils.Fold.BORDER
          )
          if not _is_collinear_with_line(ray, ridge_crease):
            side.append(ray)
  return side


def _label_creases_by_side(
    point: point_utils.Point,
    side_creases: Sequence[crease_utils.Crease],
    toward: point_utils.Point,
) -> list[tuple[crease_utils.Crease, int, float]]:
  """Labels incident creases with `(crease, side_sign, unsigned_angle)`."""
  rx, ry = toward.x - point.x, toward.y - point.y
  labelled: list[tuple[crease_utils.Crease, int, float]] = []
  for crease in side_creases:
    other = crease.p2_init if crease.p1_init.isclose(point) else crease.p1_init
    sx, sy = other.x - point.x, other.y - point.y
    angle = math.atan2(rx * sy - ry * sx, rx * sx + ry * sy)
    if math.isclose(angle, 0.0, abs_tol=1e-9) or math.isclose(
        abs(angle), math.pi, abs_tol=1e-9
    ):
      continue
    labelled.append((crease, 1 if angle > 0.0 else -1, abs(angle)))
  return labelled


def get_adjacent_side_creases(
    point: point_utils.Point,
    side_creases: list[crease_utils.Crease],
    toward: point_utils.Point,
) -> list[crease_utils.Crease]:
  """Returns the two side creases angularly nearest `point -> toward`."""
  if len(side_creases) <= 2:
    return side_creases
  closest: dict[int, tuple[float, crease_utils.Crease]] = {}
  for crease, side, angle in _label_creases_by_side(
      point, side_creases, toward
  ):
    if side not in closest or angle < closest[side][0]:
      closest[side] = (angle, crease)
  if len(closest) < 2:
    return side_creases
  return [closest[1][1], closest[-1][1]]


def get_parallel_side_crease_pairs(
    point_b: point_utils.Point,
    side_creases_b: Sequence[crease_utils.Crease],
    point_c: point_utils.Point,
    side_creases_c: Sequence[crease_utils.Crease],
    toward: point_utils.Point,
) -> tuple[list[crease_utils.Crease], list[crease_utils.Crease]] | None:
  """Selects one parallel `(B, C)` pleat pair on each side of the ridge.

  Prefers axis-aligned box-pleating pleats over diagonal shaping creases, then
  selects the pair angularly closest to the ridge direction.

  Args:
    point_b: Ridge vertex closer to the flap corner.
    side_creases_b: Non-ridge creases incident to `point_b`.
    point_c: Ridge vertex closer to the flap tip.
    side_creases_c: Non-ridge creases incident to `point_c`.
    toward: Ridge reference point on the corner side of `point_b`.

  Returns:
    `([b_plus, b_minus], [c_plus, c_minus])`, or `None` if either side lacks a
    parallel pair.
  """
  tol = point_utils.DEFAULT_TOLERANCE
  labelled_b = _label_creases_by_side(point_b, side_creases_b, toward)
  labelled_c = _label_creases_by_side(point_c, side_creases_c, point_b)
  chosen_b: dict[int, crease_utils.Crease] = {}
  chosen_c: dict[int, crease_utils.Crease] = {}
  for side in (1, -1):
    best: (
        tuple[tuple[int, float], crease_utils.Crease, crease_utils.Crease]
        | None
    ) = None
    b_on_side = [(cb, ab) for cb, sb, ab in labelled_b if sb == side]
    c_on_side = [(cc, ac) for cc, sc, ac in labelled_c if sc == side]
    for cb, ab in b_on_side:
      if (
          abs(cb.p2_init.x - cb.p1_init.x) <= tol
          or abs(cb.p2_init.y - cb.p1_init.y) <= tol
      ):
        axis_rank = 0
      else:
        axis_rank = 1
      for cc, ac in c_on_side:
        if not cb.is_parallel(cc):
          continue
        key = (axis_rank, ab + ac)
        if best is None or key < best[0]:
          best = (key, cb, cc)
    if best is None:
      return None
    chosen_b[side], chosen_c[side] = best[1], best[2]
  return [chosen_b[1], chosen_b[-1]], [chosen_c[1], chosen_c[-1]]


def _bisector_hit_on_pleat_line(
    bisector: crease_utils.Crease,
    first_segment: crease_utils.Crease,
    cp: cp_utils.CreasePattern,
    all_ridge_creases: Sequence[crease_utils.Crease],
) -> tuple[point_utils.Point | None, list[crease_utils.Crease]]:
  """Intersects `bisector` with the collinear pleat chain starting at `B`."""
  hit = bisector.vertex_with_crease(
      first_segment, extend_lines=True, include_external=False
  )
  chain = [first_segment]
  if hit is not None:
    return hit, chain

  visited: set[int] = {first_segment.hash_value}
  current = first_segment
  for _ in range(_MAX_PLEAT_CHAIN_STEPS):
    end = current.p2_init
    next_seg: crease_utils.Crease | None = None
    for cand in cp.creases:
      if (
          cand.fold == crease_utils.Fold.BORDER
          or cand.hash_value in visited
          or any(_is_collinear_with_line(cand, r) for r in all_ridge_creases)
          or not _is_collinear_with_line(cand, first_segment)
      ):
        continue
      if cand.p1_init.isclose(end) and not cand.p2_init.isclose(
          current.p1_init
      ):
        next_seg = cand
        break
      if cand.p2_init.isclose(end) and not cand.p1_init.isclose(
          current.p1_init
      ):
        next_seg = cand.flip_tips()
        break
    if next_seg is None:
      return None, chain
    visited.add(next_seg.hash_value)
    chain.append(next_seg)
    hit = bisector.vertex_with_crease(
        next_seg, extend_lines=True, include_external=False
    )
    if hit is not None:
      return hit, chain
    current = next_seg
  return None, chain


def apply_squash_sink_to_segment(
    crease_ab: crease_utils.Crease,
    crease_bc: crease_utils.Crease,
    cp: cp_utils.CreasePattern,
    all_ridge_creases: Sequence[crease_utils.Crease],
) -> tuple[list[crease_utils.Crease], list[crease_utils.Crease]]:
  """Constructs a 2-segment kite squash sink over ridge segments `AB` and `BC`.

  Args:
    crease_ab: Outer ridge segment (`A -> B`, closer to the flap corner).
    crease_bc: Inner ridge segment (`B -> C`, closer to the flap tip).
    cp: Full crease pattern.
    all_ridge_creases: All collinear segments along the ridge.

  Returns:
    `(new_creases, creases_to_remove)`.
  """
  point_a = crease_ab.p1_init
  point_b = crease_bc.p1_init
  point_c = crease_bc.p2_init
  fold_bc = crease_bc.fold

  all_b = get_side_creases(point_b, cp, all_ridge_creases[0])
  all_c = get_side_creases(point_c, cp, all_ridge_creases[0])
  pairs = get_parallel_side_crease_pairs(
      point_b, all_b, point_c, all_c, point_a
  )
  if pairs is not None:
    side_b, side_c = pairs
  else:
    side_b = get_adjacent_side_creases(point_b, all_b, point_a)
    side_c = get_adjacent_side_creases(point_c, all_c, point_b)
  if len(side_b) != 2 or len(side_c) != 2:
    raise ValueError(
        f'Expected 2 side creases at B and C, got {len(side_b)} and'
        f' {len(side_c)}.'
    )

  b1, b2 = _orient_from(side_b[0], point_b), _orient_from(side_b[1], point_b)
  c1, c2 = _orient_from(side_c[0], point_c), _orient_from(side_c[1], point_c)
  if not b1.is_parallel(c1):
    b1, b2 = b2, b1
  if not b1.is_parallel(c1):
    raise ValueError('Squash sink requires parallel side pleats at B and C.')

  bis1 = crease_utils.Crease(
      point_c, get_bisector_point(point_c, point_b, c1.p2_init), fold=fold_bc
  )
  bis2 = crease_utils.Crease(
      point_c, get_bisector_point(point_c, point_b, c2.p2_init), fold=fold_bc
  )
  point_e, chain_1 = _bisector_hit_on_pleat_line(
      bis1, b1, cp, all_ridge_creases
  )
  point_f, chain_2 = _bisector_hit_on_pleat_line(
      bis2, b2, cp, all_ridge_creases
  )
  if point_e is None or point_f is None:
    raise ValueError('Could not intersect squash-sink bisectors with B pleats.')

  point_g = crease_utils.Crease(
      point_e, point_f, fold=fold_bc
  ).vertex_with_crease(crease_ab, extend_lines=True, include_external=True)
  if (
      point_g is None
      or crease_ab.point_status(point_g)
      is not crease_utils.PointStatus.INTERNAL
  ):
    raise ValueError('Squash-sink apex G does not lie inside ridge segment AB.')

  flipped = fold_bc.flip()
  new_creases = [
      crease_utils.Crease(point_c, point_e, fold=fold_bc),
      crease_utils.Crease(point_c, point_f, fold=fold_bc),
      crease_utils.Crease(point_e, point_g, fold=fold_bc),
      crease_utils.Crease(point_g, point_f, fold=fold_bc),
      crease_utils.Crease(
          point_a, point_g, fold=fold_bc, elevation=crease_ab.elevation
      ),
      crease_utils.Crease(point_a, point_e, fold=flipped),
      crease_utils.Crease(point_a, point_f, fold=flipped),
      crease_utils.Crease(point_g, point_c, fold=flipped),
  ]
  removed = [crease_ab, crease_bc]
  for hit_pt, chain in ((point_e, chain_1), (point_f, chain_2)):
    removed.extend(chain)
    last = chain[-1]
    if not hit_pt.isclose(last.p2_init):
      new_creases.append(
          crease_utils.Crease(
              hit_pt, last.p2_init, fold=last.fold, elevation=last.elevation
          )
      )
  return new_creases, removed


def apply_border_triangle_squash_to_segment(
    crease_bc: crease_utils.Crease,
    cp: cp_utils.CreasePattern,
    all_ridge_creases: Sequence[crease_utils.Crease],
) -> tuple[list[crease_utils.Crease], list[crease_utils.Crease]]:
  """Constructs a triangular border squash on a border-clipped segment `BC`."""
  point_b, point_c = crease_bc.p1_init, crease_bc.p2_init
  if not point_b.is_border or not point_c.is_internal:
    raise ValueError(
        'Border triangle squash requires B on paper border and C internal.'
    )
  if not crease_bc.mountain_or_valley():
    raise ValueError('Ridge segment must be mountain or valley.')
  fold_bc = crease_bc.fold

  side_c = get_adjacent_side_creases(
      point_c, get_side_creases(point_c, cp, all_ridge_creases[0]), point_b
  )
  if len(side_c) != 2:
    raise ValueError(
        f'Expected 2 side creases at C for border squash, got {len(side_c)}.'
    )
  borders = [
      c for c in cp.creases if c.fold == crease_utils.Fold.BORDER
  ] or _unit_borders()

  hits: list[point_utils.Point] = []
  for raw_side in side_c:
    oriented = _orient_from(raw_side, point_c)
    bis_pt = get_bisector_point(point_c, point_b, oriented.p2_init)
    ray = crease_utils.Crease(point_c, bis_pt, fold=fold_bc)
    best_hit: point_utils.Point | None = None
    best_dist = float('inf')
    for bc in borders:
      pt = ray.vertex_with_crease(bc, extend_lines=True, include_external=False)
      if pt is None or bc.point_status(pt) is crease_utils.PointStatus.EXTERNAL:
        continue
      forward = (pt.x - point_c.x) * (bis_pt.x - point_c.x) + (
          pt.y - point_c.y
      ) * (bis_pt.y - point_c.y)
      if forward > 1e-6:
        dist = point_utils.distance(point_c, pt)
        if dist < best_dist:
          best_dist = dist
          best_hit = pt
    if best_hit is None:
      raise ValueError('Bisector ray did not intersect the paper border.')
    hits.append(best_hit)

  new_creases = [
      crease_utils.Crease(point_c, hits[0], fold=fold_bc),
      crease_utils.Crease(
          point_c, point_b, fold=fold_bc.flip(), elevation=crease_bc.elevation
      ),
      crease_utils.Crease(point_c, hits[1], fold=fold_bc),
  ]
  for nc in new_creases:
    for oc in cp.creases:
      if (
          oc.fold == crease_utils.Fold.BORDER
          or oc == crease_bc
          or _is_within_segment_on_line(crease_bc, oc)
      ):
        continue
      cross = nc.vertex_with_crease(oc)
      if (
          cross is not None
          and not cross.isclose(point_c)
          and not cross.isclose(point_b)
      ):
        raise ValueError(
            f'Border triangle crease {nc} intersects {oc} at {cross}.'
        )
  return new_creases, [crease_bc]


def get_oriented_ordered_crease_segments_on_ridge(
    cp: cp_utils.CreasePattern,
    ridge: crease_utils.Crease,
    ridge_start_point: point_utils.Point,
) -> list[crease_utils.Crease]:
  """Returns collinear ridge segments sorted and oriented from `ridge_start`."""
  vertex_pts = [v.point for v in cp.vertices]
  segments: list[crease_utils.Crease] = []
  for c in cp.creases:
    if c.fold == crease_utils.Fold.BORDER or not _is_within_segment_on_line(
        c, ridge
    ):
      continue
    for sub in crease_utils.split_crease_at_points(c, vertex_pts):
      p1_closer = point_utils.distance(
          ridge_start_point, sub.p1_init
      ) < point_utils.distance(ridge_start_point, sub.p2_init)
      segments.append(sub if p1_closer else sub.flip_tips())
  segments.sort(
      key=lambda seg: point_utils.distance(ridge_start_point, seg.p1_init)
  )
  return segments


def _same_endpoints(a: crease_utils.Crease, b: crease_utils.Crease) -> bool:
  return (a.p1_init.isclose(b.p1_init) and a.p2_init.isclose(b.p2_init)) or (
      a.p1_init.isclose(b.p2_init) and a.p2_init.isclose(b.p1_init)
  )


def _remove_crease_or_subsegment(
    creases: list[crease_utils.Crease], target: crease_utils.Crease
) -> None:
  """Removes `target`, splitting a containing merged crease when needed."""
  if target in creases:
    creases.remove(target)
    return
  for idx, c in enumerate(creases):
    if c.fold == crease_utils.Fold.BORDER or not _is_within_segment_on_line(
        target, c
    ):
      continue
    creases.pop(idx)
    pieces = crease_utils.split_crease_at_points(
        c, (target.p1_init, target.p2_init)
    )
    creases.extend(sub for sub in pieces if not _same_endpoints(sub, target))
    return


def update_cp(
    cp: cp_utils.CreasePattern,
    new_creases: Sequence[crease_utils.Crease],
    creases_to_remove: Sequence[crease_utils.Crease],
) -> cp_utils.CreasePattern:
  """Returns `cp` with `creases_to_remove` dropped and `new_creases` added."""
  updated = list(cp.creases)
  for target in creases_to_remove:
    _remove_crease_or_subsegment(updated, target)
  updated.extend(new_creases)
  if cp.light_mode:
    updated = flat_foldable.split_border_creases_at_tips(updated)
  return cp_utils.CreasePattern(creases=updated, light_mode=cp.light_mode)


def get_all_ridges_from_rectangles(
    rectangles: Sequence[shaping_utils.Rectangle],
) -> list[crease_utils.Crease]:
  """Returns all center-to-corner diagonal ridges of `rectangles`."""
  ridges: list[crease_utils.Crease] = []
  for rect in rectangles:
    if not rect.circle_centers:
      continue
    centers = rect.circle_centers
    for corner in rect.corners:
      nearest = min(
          centers, key=lambda center, c=corner: point_utils.distance(c, center)
      )
      ridges.append(
          crease_utils.Crease(nearest, corner, fold=crease_utils.Fold.MOUNTAIN)
      )
  return ridges


def get_full_ridge(
    ridge: crease_utils.Crease, ridges: Sequence[crease_utils.Crease]
) -> crease_utils.Crease:
  """Returns the full rectangle ridge containing `ridge`."""
  full_ridge: crease_utils.Crease | None = None
  for cand in ridges:
    if cand.is_overlapping(ridge) or _overlaps_on_line(ridge, cand):
      full_ridge = cand
  if full_ridge is None:
    raise ValueError(f'Input {ridge} is not a rectangle ridge.')
  return full_ridge


def get_start_side_of_ridge(
    ridge_segment: crease_utils.Crease,
    full_ridge: crease_utils.Crease,
    rectangles: Sequence[shaping_utils.Rectangle],
) -> point_utils.Point:
  """Returns the endpoint of `ridge_segment` closer to the flap corner."""
  corner_end: point_utils.Point | None = None
  p1, p2 = full_ridge.p1_init, full_ridge.p2_init
  for rect in rectangles:
    if not rect.circle_centers:
      continue
    for center in rect.circle_centers:
      if p1.isclose(center):
        cand_corner = p2
      elif p2.isclose(center):
        cand_corner = p1
      else:
        continue
      if not any(cand_corner.isclose(corner) for corner in rect.corners):
        raise ValueError(f'Input {ridge_segment} is not a rectangle ridge.')
      corner_end = cand_corner
      break
  if corner_end is None:
    raise ValueError(f'Input {ridge_segment} is not a rectangle ridge.')
  if point_utils.distance(
      ridge_segment.p1_init, corner_end
  ) < point_utils.distance(ridge_segment.p2_init, corner_end):
    return ridge_segment.p1_init
  return ridge_segment.p2_init


def shape_cp_with_squash_sink(
    cp: cp_utils.CreasePattern,
    rectangles: Sequence[shaping_utils.Rectangle],
    ridge: crease_utils.Crease,
) -> cp_utils.CreasePattern:
  """Applies kite (or border-triangle) squash sinks along `ridge`."""
  full_ridge = get_full_ridge(ridge, get_all_ridges_from_rectangles(rectangles))
  start_pt = get_start_side_of_ridge(ridge, full_ridge, rectangles)
  segments = get_oriented_ordered_crease_segments_on_ridge(cp, ridge, start_pt)

  current_cp = cp
  applied = False
  for idx in range(0, len(segments) - 1, 2):
    try:
      added, removed = apply_squash_sink_to_segment(
          segments[idx], segments[idx + 1], current_cp, segments
      )
    except ValueError as err:
      logging.debug(
          'Skipping squash sink on segment %s: %s', segments[idx], err
      )
      continue
    current_cp = update_cp(current_cp, added, removed)
    applied = True

  if (
      not applied
      and segments
      and segments[0].p1_init.is_border
      and segments[0].p2_init.is_internal
  ):
    try:
      added, removed = apply_border_triangle_squash_to_segment(
          segments[0], current_cp, segments
      )
    except ValueError as err:
      logging.debug(
          'Skipping border triangle squash on segment %s: %s', segments[0], err
      )
    else:
      current_cp = update_cp(current_cp, added, removed)

  return current_cp
