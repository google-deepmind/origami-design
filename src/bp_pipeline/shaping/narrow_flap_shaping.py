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
"""Flap narrowing by tracing reflected half-grid creases from zigzags."""

import collections
from collections.abc import Iterable, Sequence
import dataclasses
import enum
import math

from absl import logging

from .. import timeout as timeout_lib
from ..crease_pattern import angle_utils
from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import point as point_utils
from ..crease_pattern import rotation
from . import base_rectangle as rect_utils
from . import shaping_utils
from . import vector

type _RectLike = shaping_utils.Rectangle | rect_utils.Rectangle
type _CreaseReplacement = tuple[crease_utils.Crease, crease_utils.Crease]

_GRID_SNAP_TOL = 1e-3
_UNIT_SQUARE_DIAGONAL = math.sqrt(2.0)


class Orientation(enum.StrEnum):
  """Rectangle edge touched by a zigzag segment (canonical clockwise order)."""

  TOP = 'top'
  RIGHT = 'right'
  BOTTOM = 'bottom'
  LEFT = 'left'

  @property
  def is_horizontal_edge(self) -> bool:
    """Whether this orientation is the top or bottom edge of a rectangle."""
    return self in (Orientation.TOP, Orientation.BOTTOM)

  @property
  def ray_target_coord(self) -> float:
    """Paper-boundary coordinate toward which an inward narrowing ray points."""
    return 1.0 if self in (Orientation.TOP, Orientation.LEFT) else 0.0


@dataclasses.dataclass(frozen=True, kw_only=True)
class ZigZagInfo:
  """Metadata linking a zigzag segment to the rectangle border it touches."""

  border_point: point_utils.Point
  non_border_point: point_utils.Point
  orientation: Orientation


@dataclasses.dataclass(frozen=True, kw_only=True)
class ZigZag:
  """Half-grid launch point and initial ray direction for a narrowing crease."""

  middle_point: point_utils.Point
  narrowing_crease_end_point: point_utils.Point

  @classmethod
  def from_zigzag_info(
      cls,
      zigzag_info: ZigZagInfo,
      grid_size: int,
  ) -> 'ZigZag | None':
    """Builds a `ZigZag` at half a grid unit along `zigzag_info`, or `None`."""
    border = zigzag_info.border_point
    interior = zigzag_info.non_border_point
    orientation = zigzag_info.orientation
    half_step = 0.5 / grid_size

    dx = abs(border.x - interior.x)
    dy = abs(border.y - interior.y)
    if (dx <= half_step and dy <= half_step) or (
        math.isclose(dx, half_step) and math.isclose(dy, half_step)
    ):
      return None

    if orientation.is_horizontal_edge:
      b_coord, n_coord = border.x, interior.x
    else:
      b_coord, n_coord = border.y, interior.y

    step = half_step if n_coord > b_coord else -half_step
    mid_coord = b_coord + step
    beyond = mid_coord >= n_coord if step > 0 else mid_coord <= n_coord
    if beyond or math.isclose(mid_coord, n_coord):
      return None

    target = orientation.ray_target_coord
    if orientation.is_horizontal_edge:
      mid_y = point_utils.find_y_on_line(mid_coord, border, interior)
      middle = point_utils.Point(x=mid_coord, y=mid_y)
      end = point_utils.Point(x=middle.x, y=target)
    else:
      mid_x = point_utils.find_x_on_line(mid_coord, border, interior)
      middle = point_utils.Point(x=mid_x, y=mid_coord)
      end = point_utils.Point(x=target, y=middle.y)
    return cls(middle_point=middle, narrowing_crease_end_point=end)


@dataclasses.dataclass(frozen=True, kw_only=True)
class NarrowingPolygon:
  """Connected alternating-MV narrowing crease chain (closed loop or border)."""

  creases: tuple[crease_utils.Crease, ...]

  def __contains__(self, point: point_utils.Point) -> bool:
    """Returns whether `point` is an endpoint of any crease in this chain."""
    return any(
        c.point_status(point) == crease_utils.PointStatus.TIP
        for c in self.creases
    )

  def flip(self) -> 'NarrowingPolygon':
    """Returns a copy with every crease's M/V fold assignment inverted."""
    return NarrowingPolygon(
        creases=tuple(
            dataclasses.replace(c, fold=c.fold.flip()) for c in self.creases
        )
    )


def _zigzag_infos_for_crease(
    rectangle: _RectLike,
    crease: crease_utils.Crease,
    grid_size: int,
) -> list[ZigZagInfo]:
  """Returns rectangle-edge intersections of `crease` that sit on grid nodes."""
  infos: list[ZigZagInfo] = []
  for edge, orientation in zip(rectangle.edges, Orientation, strict=True):
    hit = crease.vertex_with_crease(edge)
    if hit is None:
      continue
    gx = hit.x * grid_size
    gy = hit.y * grid_size
    if not (
        math.isclose(round(gx), gx, abs_tol=_GRID_SNAP_TOL)
        and math.isclose(round(gy), gy, abs_tol=_GRID_SNAP_TOL)
    ):
      continue
    other = crease.p2 if crease.p1.isclose(hit) else crease.p1
    infos.append(
        ZigZagInfo(
            border_point=hit,
            non_border_point=other,
            orientation=orientation,
        )
    )
  return infos


def _find_double_border_neighbor(
    zigzag_creases: Sequence[crease_utils.Crease],
    info1: ZigZagInfo,
    info2: ZigZagInfo,
    crease: crease_utils.Crease,
) -> crease_utils.Crease | None:
  """Finds the adjacent zigzag crease sharing a tip with a two-border crease."""
  if info1.border_point == info2.border_point:
    connecting = (info1.non_border_point,)
  else:
    same_order = info1.border_point.isclose(
        info2.border_point
    ) and info1.non_border_point.isclose(info2.non_border_point)
    swapped_order = info1.non_border_point.isclose(
        info2.border_point
    ) and info1.border_point.isclose(info2.non_border_point)
    if not (same_order or swapped_order):
      raise ValueError(
          f'Double-border zigzag infos have mismatched endpoints: {info1} vs'
          f' {info2}.'
      )
    connecting = (info1.non_border_point, info2.non_border_point)

  for candidate in zigzag_creases:
    if candidate == crease:
      continue
    if any(
        candidate.point_status(pt) == crease_utils.PointStatus.TIP
        for pt in connecting
    ):
      return candidate
  return None


def get_zigzags(
    zigzag_creases: Sequence[crease_utils.Crease],
    rectangle: _RectLike,
    grid_size: int,
) -> tuple[list[ZigZag], list[crease_utils.Crease]]:
  """Extracts narrowing seed `ZigZag`s and their inner half-segments."""
  double_neighbors: dict[int, crease_utils.Crease] = {}
  double_infos: dict[int, list[ZigZagInfo]] = {}
  orientations: dict[int, Orientation] = {}
  by_hash: dict[int, crease_utils.Crease] = {}

  zigzags: list[ZigZag] = []
  inner_segments: list[crease_utils.Crease] = []

  def add_from_info(c: crease_utils.Crease, info: ZigZagInfo) -> bool:
    zz = ZigZag.from_zigzag_info(info, grid_size)
    if zz is None:
      return False
    zigzags.append(zz)
    inner_segments.append(
        dataclasses.replace(
            c, p1_init=zz.middle_point, p2_init=info.non_border_point
        )
    )
    return True

  for crease in zigzag_creases:
    if not rectangle.crease_is_within(crease):
      continue
    by_hash[crease.hash_value] = crease
    infos = _zigzag_infos_for_crease(rectangle, crease, grid_size)
    if len(infos) == 1:
      orientations[crease.hash_value] = infos[0].orientation
      add_from_info(crease, infos[0])
    elif len(infos) == 2:
      neighbor = _find_double_border_neighbor(
          zigzag_creases, infos[0], infos[1], crease
      )
      if neighbor is not None:
        double_neighbors[crease.hash_value] = neighbor
        double_infos[crease.hash_value] = infos

  for crease_hash, neighbor in double_neighbors.items():
    timeout_lib.maybe_timeout_cooperatively()
    target_orientation = orientations.get(neighbor.hash_value)
    if target_orientation is None:
      continue
    for info in double_infos[crease_hash]:
      if info.orientation == target_orientation:
        if add_from_info(by_hash[crease_hash], info):
          break

  return zigzags, inner_segments


def _crease_with_tip(
    point: point_utils.Point,
    creases: Sequence[crease_utils.Crease],
) -> crease_utils.Crease | None:
  """Returns the unique crease in `creases` having `point` as an endpoint."""
  found: crease_utils.Crease | None = None
  for c in creases:
    if not c.is_tip(point):
      continue
    if found is not None:
      raise ValueError(
          f'Multiple creases ({found}, {c}) share endpoint {point}.'
      )
    found = c
  return found


def _paper_border_crease_at(
    point: point_utils.Point, cp: cp_utils.CreasePattern
) -> crease_utils.Crease | None:
  """Returns the paper `BORDER` crease containing `point`, if any."""
  for c in cp.creases:
    if c.fold == crease_utils.Fold.BORDER and c.point_status(point) in (
        crease_utils.PointStatus.TIP,
        crease_utils.PointStatus.INTERNAL,
    ):
      return c
  return None


def _non_rect_border_ridge_at(
    point: point_utils.Point,
    rectangle: _RectLike,
    cp: cp_utils.CreasePattern,
) -> crease_utils.Crease | None:
  """Returns a unique-direction non-edge crease passing through `point`."""
  if not rectangle.is_edge_point(point, extend_edges=True):
    raise ValueError(f'Point {point} is not on rectangle border {rectangle}.')
  ridge: crease_utils.Crease | None = None
  for c in cp.creases:
    if c.point_status(point) not in (
        crease_utils.PointStatus.TIP,
        crease_utils.PointStatus.INTERNAL,
    ):
      continue
    if any(c.is_overlapping(edge) for edge in rectangle.edges):
      continue
    if ridge is None:
      ridge = c
    elif not ridge.is_collinear(c):
      return None
  return ridge


def _closest_intersection(
    start: point_utils.Point,
    ray: crease_utils.Crease,
    candidates: Iterable[crease_utils.Crease],
) -> tuple[crease_utils.Crease | None, point_utils.Point | None, float]:
  """Finds the closest intersection of `ray` with `candidates` past `start`."""
  if ray.point_status(start) != crease_utils.PointStatus.TIP:
    raise ValueError(f'Start point {start} is not an endpoint of {ray}.')
  best_crease: crease_utils.Crease | None = None
  best_point: point_utils.Point | None = None
  best_dist = math.inf
  for c in candidates:
    hit = ray.vertex_with_crease(c)
    if hit is None or hit.isclose(start):
      continue
    dist = point_utils.distance(start, hit)
    if dist < best_dist:
      best_crease, best_point, best_dist = c, hit, dist
  return best_crease, best_point, best_dist


def _extend_from_point(
    origin: point_utils.Point, direction: vector.Vector
) -> point_utils.Point:
  """Returns a point at distance `sqrt(2)` from `origin` along `direction`."""
  ray_vec = direction.normalize * _UNIT_SQUARE_DIAGONAL
  return ray_vec.point_add(dataclasses.replace(origin, require_on_paper=False))


def _rotate_and_clip_crease(
    crease: crease_utils.Crease,
    clockwise_degrees: float,
    pivot: point_utils.Point,
    cp: cp_utils.CreasePattern,
) -> crease_utils.Crease:
  """Rotates `crease` clockwise around `pivot` and clips it to `cp`."""
  rotated = rotation.rotate_crease_clockwise(
      crease_to_rotate=crease,
      rotation_angle_degrees=clockwise_degrees,
      rotation_origin_tip=pivot,
      require_on_paper=False,
  )
  if rotated.p1.isclose(pivot):
    other = rotated.p2
  elif rotated.p2.isclose(pivot):
    other = rotated.p1
  else:
    raise ValueError(f'Rotated crease {rotated} lacks pivot endpoint {pivot}.')

  far_end = _extend_from_point(pivot, vector.Vector.from_points(pivot, other))
  extended = dataclasses.replace(rotated, p1_init=pivot, p2_init=far_end)
  _, hit, _ = _closest_intersection(pivot, extended, cp.creases)
  if hit is None:
    raise ValueError(
        'Rotated zigzag crease does not intersect any crease in the pattern.'
    )
  return dataclasses.replace(crease, p1_init=pivot, p2_init=hit)


def draw_narrowing_crease(
    start_point: point_utils.Point,
    end_point: point_utils.Point,
    rectangle_border_points: Iterable[point_utils.Point],
    rectangle: _RectLike,
    cp: cp_utils.CreasePattern,
    non_border_zigzag_creases: Sequence[crease_utils.Crease],
    prev_clockwise_angle_delta: float | None = None,
    *,
    allow_paper_border_rotation: bool = True,
) -> (
    tuple[
        point_utils.Point,
        point_utils.Point,
        list[_CreaseReplacement],
        float | None,
    ]
    | None
):
  """Steps a narrowing ray one reflection, returning the hit and next ray."""
  incoming = crease_utils.Crease(
      start_point, end_point, fold=crease_utils.Fold.MOUNTAIN
  )
  replacements: list[_CreaseReplacement] = []

  cp_crease, cp_hit, cp_dist = _closest_intersection(
      start_point, incoming, cp.creases
  )
  rect_crease, rect_hit, rect_dist = _closest_intersection(
      start_point, incoming, rectangle.edges
  )
  if cp_dist <= rect_dist or math.isclose(cp_dist, rect_dist):
    ridge, hit = cp_crease, cp_hit
  else:
    ridge, hit = rect_crease, rect_hit
  if hit is None or ridge is None:
    return None

  angle_delta: float | None = None
  should_snap_to_border_vertex = (
      rectangle.is_edge_point(hit)
      and (allow_paper_border_rotation or not hit.is_border)
      and (prev_clockwise_angle_delta is None or not hit.is_border)
      and not incoming.is_vertical
      and not incoming.is_horizontal
  )
  if should_snap_to_border_vertex:
    if prev_clockwise_angle_delta is not None:
      raise ValueError(
          'prev_clockwise_angle_delta must be None for border-ending creases.'
      )
    old_angle = angle_utils.find_angle_from_points(start_point, hit)
    snapped = hit.closest_point(rectangle_border_points)
    snapped_seg = crease_utils.Crease(
        start_point, snapped, fold=crease_utils.Fold.MOUNTAIN
    )
    if not any(snapped_seg.is_overlapping(c) for c in cp.creases):
      hit = snapped

    ridge = (
        _paper_border_crease_at(hit, cp)
        if hit.is_border
        else _non_rect_border_ridge_at(hit, rectangle, cp)
    )
    if ridge is None:
      return None

    new_angle = angle_utils.find_angle_from_points(start_point, hit)
    angle_delta = math.degrees(new_angle - old_angle)

    zz_to_rotate = _crease_with_tip(start_point, non_border_zigzag_creases)
    if zz_to_rotate is None:
      return None
    replacements.append((
        zz_to_rotate,
        _rotate_and_clip_crease(zz_to_rotate, angle_delta, start_point, cp),
    ))

  truncated = crease_utils.Crease(
      start_point, hit, fold=crease_utils.Fold.MOUNTAIN
  )
  reflected = shaping_utils.reflect_crease_over_ridge(
      crease=truncated,
      ridge=ridge,
      precomputed_intersection_point=hit,
  )
  next_end = _extend_from_point(hit, reflected)

  if prev_clockwise_angle_delta is not None:
    next_end = rotation.rotate_point_clockwise(
        point_to_rotate=next_end,
        rotation_angle_degrees=-prev_clockwise_angle_delta,
        rotation_origin_tip=hit,
        require_on_paper=False,
    )
    zz_to_rotate = _crease_with_tip(hit, non_border_zigzag_creases)
    if zz_to_rotate is not None:
      replacements.append((
          zz_to_rotate,
          _rotate_and_clip_crease(
              zz_to_rotate, -prev_clockwise_angle_delta, hit, cp
          ),
      ))

  return hit, next_end, replacements, angle_delta


def _trace_narrowing_path(
    start_point: point_utils.Point,
    end_point: point_utils.Point,
    rectangle_border_points: set[point_utils.Point],
    rectangle: _RectLike,
    cp: cp_utils.CreasePattern,
    non_border_zigzag_creases: Sequence[crease_utils.Crease],
    *,
    max_num_creases: int = 1000,
    allow_paper_border_rotation: bool = True,
) -> tuple[list[point_utils.Point], list[_CreaseReplacement]] | None:
  """Traces a reflected narrowing path until it closes or reaches a border."""
  points: list[point_utils.Point] = [start_point]
  origin = start_point
  replacements: list[_CreaseReplacement] = []
  angle_delta: float | None = None

  for _ in range(max_num_creases + 1):
    timeout_lib.maybe_timeout_cooperatively()
    step = draw_narrowing_crease(
        start_point=start_point,
        end_point=end_point,
        rectangle_border_points=rectangle_border_points,
        rectangle=rectangle,
        cp=cp,
        non_border_zigzag_creases=non_border_zigzag_creases,
        prev_clockwise_angle_delta=angle_delta,
        allow_paper_border_rotation=allow_paper_border_rotation,
    )
    if step is None:
      return None
    hit, end_point, step_replacements, angle_delta = step
    if any(hit.isclose(p) for p in points[1:]):
      return None
    if not (
        rectangle.is_internal_point(hit)
        or rectangle.is_internal_point(points[-1])
    ):
      return None
    points.append(hit)
    replacements.extend(step_replacements)
    start_point = hit
    if hit.isclose(origin) or hit.is_border:
      return points, replacements

  raise ValueError(
      f'Narrowing path exceeded {max_num_creases} reflections in rectangle'
      f' {rectangle.name} (grid_size={cp.grid_size}).'
  )


def _alternating_creases(
    points: Sequence[point_utils.Point],
    initial_fold: crease_utils.Fold,
) -> list[crease_utils.Crease]:
  """Builds consecutive creases along `points` with alternating M/V folds."""
  creases: list[crease_utils.Crease] = []
  fold = initial_fold
  for p1, p2 in zip(points, points[1:]):
    creases.append(crease_utils.Crease(p1, p2, fold=fold))
    fold = fold.flip()
  return creases


def _prune_visited_zigzags(
    zigzags: collections.deque[ZigZag],
    visited: Sequence[point_utils.Point],
) -> None:
  """Drops remaining `ZigZag` seeds whose midpoints lie in `visited`."""
  if not visited:
    return
  kept = [
      zz
      for zz in zigzags
      if not any(zz.middle_point.isclose(pt) for pt in visited)
  ]
  zigzags.clear()
  zigzags.extend(kept)


def _rectangle_border_vertices(
    cp: cp_utils.CreasePattern, rectangle: _RectLike
) -> set[point_utils.Point]:
  """Collects all intersection points between `cp.creases` and `rectangle`."""
  border_points: set[point_utils.Point] = set()
  edges = rectangle.edges
  for crease in cp.creases:
    timeout_lib.maybe_timeout_cooperatively()
    for edge in edges:
      hit = crease.vertex_with_crease(edge)
      if hit is not None:
        border_points.add(hit)
  return border_points


def _narrow_once(
    cp: cp_utils.CreasePattern,
    zigzag_creases: Sequence[crease_utils.Crease],
    rectangles: Sequence[_RectLike],
    grid_size: int,
    light_mode: bool,
    allow_paper_border_rotation: bool,
) -> cp_utils.CreasePattern:
  """Runs one narrowing pass with or without paper-border angle snapping."""
  polygons: list[NarrowingPolygon] = []
  all_replacements: list[_CreaseReplacement] = []

  for raw_rect in rectangles:
    rectangle = raw_rect.clip(xmin=0, ymin=0, xmax=1, ymax=1)
    if rectangle is None:
      raise ValueError(
          f'Rectangle {raw_rect} lies completely outside the unit paper.'
      )

    border_points = _rectangle_border_vertices(cp, rectangle)
    seed_list, inner_segments = get_zigzags(
        zigzag_creases, rectangle, grid_size
    )
    pending = collections.deque(seed_list)

    while pending:
      timeout_lib.maybe_timeout_cooperatively()
      zz = pending.popleft()
      traced = _trace_narrowing_path(
          start_point=zz.middle_point,
          end_point=zz.narrowing_crease_end_point,
          rectangle_border_points=border_points,
          rectangle=rectangle,
          cp=cp,
          non_border_zigzag_creases=inner_segments,
          allow_paper_border_rotation=allow_paper_border_rotation,
      )
      if traced is None:
        continue
      points, replacements = traced
      if len(points) < 2:
        continue

      chain_creases: list[crease_utils.Crease] = []
      if points[-1].isclose(zz.middle_point):
        # Closed loops must have an even number of edges to alternate M/V.
        if len(points) % 2 == 0:
          continue
      elif points[-1].is_border:
        if not rectangle.is_edge_point(points[-1], extend_edges=True):
          continue
        reverse = _trace_narrowing_path(
            start_point=points[1],
            end_point=points[0],
            rectangle_border_points=border_points,
            rectangle=rectangle,
            cp=cp,
            non_border_zigzag_creases=inner_segments,
            allow_paper_border_rotation=allow_paper_border_rotation,
        )
        if reverse is None:
          continue
        rev_points, rev_replacements = reverse
        if (
            len(rev_points) < 2
            or not rev_points[-1].is_border
            or not rectangle.is_edge_point(rev_points[-1], extend_edges=True)
        ):
          continue
        all_replacements.extend(rev_replacements)
        rev_tail = rev_points[1:]
        chain_creases.extend(
            _alternating_creases(rev_tail, crease_utils.Fold.VALLEY)
        )
        _prune_visited_zigzags(pending, rev_tail)
      else:
        raise ValueError(
            'Narrowing path terminated neither on the paper border nor at its'
            ' starting point.'
        )

      all_replacements.extend(replacements)
      chain_creases.extend(
          _alternating_creases(points, crease_utils.Fold.MOUNTAIN)
      )
      _prune_visited_zigzags(pending, points)
      if chain_creases:
        polygons.append(NarrowingPolygon(creases=tuple(chain_creases)))

  if not polygons:
    logging.info(
        'No narrowing polygons traced for rectangles %s.',
        [r.name for r in rectangles],
    )
    return cp

  narrowing_creases: list[crease_utils.Crease] = []
  for poly in polygons:
    narrowing_creases.extend(poly.creases)
  tips, _ = cp_utils.get_tips_and_grid_creases(
      [*cp.creases, *narrowing_creases], grid_size=-1
  )
  split_base = cp_utils.split_creases(cp.creases, narrowing_creases, tips=tips)

  replaced_base: list[crease_utils.Crease] = []
  any_replaced = False
  for crease in split_base:
    timeout_lib.maybe_timeout_cooperatively()
    replacement = next(
        (new for old, new in all_replacements if crease.isclose(old)), None
    )
    if replacement is not None:
      replaced_base.append(replacement)
      any_replaced = True
    else:
      replaced_base.append(crease)

  effective_light = False if any_replaced else light_mode
  narrowed_cp = cp_utils.CreasePattern(
      creases=[*replaced_base, *narrowing_creases],
      light_mode=effective_light,
  )

  # Flip any narrowing polygon incident to a non-flat-foldable vertex.
  flipped = [False] * len(polygons)
  for vertex in narrowed_cp.vertices:
    timeout_lib.maybe_timeout_cooperatively()
    if all(flipped):
      break
    if vertex.flat_foldability():
      continue
    for idx, poly in enumerate(polygons):
      if not flipped[idx] and vertex.point in poly:
        polygons[idx] = poly.flip()
        flipped[idx] = True
        break

  if any(flipped):
    repaired_creases = list(replaced_base)
    for idx, poly in enumerate(polygons):
      if flipped[idx]:
        repaired_creases.extend(poly.creases)
    for idx, poly in enumerate(polygons):
      if not flipped[idx]:
        repaired_creases.extend(poly.creases)
    narrowed_cp = cp_utils.CreasePattern(
        creases=repaired_creases, light_mode=effective_light
    )

  return narrowed_cp


def narrow(
    cp: cp_utils.CreasePattern,
    zigzag_creases: Sequence[crease_utils.Crease],
    rectangles: Sequence[_RectLike],
    grid_size: int | None = None,
    light_mode: bool = False,
    raise_error_on_ff_violation: bool = False,
) -> cp_utils.CreasePattern:
  """Applies half-grid flap narrowing to `rectangles` after zigzag shaping."""
  resolved_grid = cp.grid_size if grid_size is None else grid_size
  narrowed_cp = _narrow_once(
      cp=cp,
      zigzag_creases=zigzag_creases,
      rectangles=rectangles,
      grid_size=resolved_grid,
      light_mode=light_mode,
      allow_paper_border_rotation=True,
  )
  if not narrowed_cp.local_flat_foldability():
    fallback = _narrow_once(
        cp=cp,
        zigzag_creases=zigzag_creases,
        rectangles=rectangles,
        grid_size=resolved_grid,
        light_mode=light_mode,
        allow_paper_border_rotation=False,
    )
    if (
        len(fallback.creases) > len(cp.creases)
        and fallback.local_flat_foldability()
    ):
      narrowed_cp = fallback

  if raise_error_on_ff_violation and not narrowed_cp.local_flat_foldability():
    raise ValueError('Narrowed crease pattern is not locally flat-foldable.')
  return narrowed_cp
