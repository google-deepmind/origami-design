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

"""An object for a Crease in an Origami Crease Pattern."""

import bisect
import collections
from collections.abc import Iterable, Iterator, Sequence
import dataclasses
import enum
import functools
import itertools
import math
from typing import Any, Literal, NotRequired, Optional, Self, TypedDict

from bp_pipeline.crease_pattern import point
from bp_pipeline.crease_pattern import types
import numpy as np

FoldColors = Literal[
    'red', 'blue', 'black', 'yellow', 'orange', 'turquoise', 'green'
]
FoldLinetypes = Literal['1', '2', '3', '4', '5', '6', '7']
DEFAULT_TOLERANCE = 1e-3


class Fold(enum.Enum):
  """The type of fold."""

  MOUNTAIN = enum.auto()
  VALLEY = enum.auto()
  BORDER = enum.auto()
  TRIANGLE = enum.auto()
  # Zebra creases represent lines that change from mountain to valley every time
  # they intersect another crease. The prefix color is the color in the first
  # segment: the segment with the smallest x coordinate, or the smallest y
  # coordinate if the crease is vertical.
  MOUNTAIN_ZEBRA = enum.auto()
  VALLEY_ZEBRA = enum.auto()
  # Hinge creases are used in box pleating to represent creases that are
  # not parallel to the edges of the box. They are not used in Origami.
  HINGE = enum.auto()

  def color(self) -> FoldColors:
    """Returns the color of the fold."""
    match self:
      case Fold.MOUNTAIN:
        return 'red'
      case Fold.VALLEY:
        return 'blue'
      case Fold.BORDER:
        return 'black'
      case Fold.TRIANGLE:
        return 'yellow'
      case Fold.MOUNTAIN_ZEBRA:
        return 'orange'
      case Fold.VALLEY_ZEBRA:
        return 'turquoise'
      case Fold.HINGE:
        return 'green'
      case _:
        raise ValueError(f'Fold {self} is not in the allowed list.')

  def linetype(self) -> FoldLinetypes:
    """Returns the linetype of the fold."""
    match self:
      case Fold.BORDER:
        return '1'
      case Fold.MOUNTAIN:
        return '2'
      case Fold.VALLEY:
        return '3'
      case Fold.TRIANGLE:
        return '4'
      case Fold.MOUNTAIN_ZEBRA:
        return '5'
      case Fold.VALLEY_ZEBRA:
        return '6'
      case Fold.HINGE:
        return '7'
      case _:
        raise ValueError(f'Fold {self} is not in the allowed list.')

  def flip(self) -> 'Fold':
    """Flips M fold to V fold and V fold to M fold."""
    match self:
      case Fold.MOUNTAIN:
        return Fold.VALLEY
      case Fold.VALLEY:
        return Fold.MOUNTAIN
      case Fold.MOUNTAIN_ZEBRA:
        return Fold.VALLEY_ZEBRA
      case Fold.VALLEY_ZEBRA:
        return Fold.MOUNTAIN_ZEBRA
      case _:
        raise ValueError(f'Fold {self} is not a mountain or valley fold.')

  def to_angle(self) -> float:
    """Get fold angle in radians."""
    match self:
      case Fold.MOUNTAIN:
        return -math.pi
      case Fold.VALLEY:
        return math.pi
      case Fold.BORDER | Fold.TRIANGLE:
        return 0.0
      case _:
        raise ValueError(f'Unsupported fold type: {self}')


def fold_one_mountain_one_valley(f1: 'Fold', f2: 'Fold') -> bool:
  """Returns true if the folds are exactly one mountain and one valley."""
  return {f1, f2} == {Fold.VALLEY, Fold.MOUNTAIN}


def one_mountain_one_valley(c1: 'Crease', c2: 'Crease') -> bool:
  """Returns true if two folds are exactly one mountain and one valley."""
  if not fold_one_mountain_one_valley(c1.fold, c2.fold):
    return False
  return math.isclose(
      c1.fold_percent, c2.fold_percent, abs_tol=DEFAULT_TOLERANCE
  )


def equal_fold_and_fold_percent(c1: 'Crease', c2: 'Crease') -> bool:
  """Returns true if two folds and fold percents are equal."""
  return c1.fold == c2.fold and math.isclose(
      c1.fold_percent, c2.fold_percent, abs_tol=DEFAULT_TOLERANCE
  )


class PointStatus(enum.Enum):
  """The status of a point in a crease."""

  EXTERNAL = 1
  INTERNAL = 2
  TIP = 3


class CreaseDict(TypedDict):
  """A dictionary representation of a Crease."""

  p1: dict[str, Any]
  p2: dict[str, Any]
  fold: int
  fold_percent: NotRequired[float]
  fold_step: NotRequired[int]
  elevation: NotRequired[int | None]


def approx_gte(x: float, y: float) -> bool:
  return x >= y - DEFAULT_TOLERANCE


def approx_lte(x: float, y: float) -> bool:
  return x <= y + DEFAULT_TOLERANCE


def strict_less(x: float, y: float) -> bool:
  return x < y - DEFAULT_TOLERANCE


_BBOX_GRID_STEP = 0.0625
_BBOX_CELL_MIN = -4
_BBOX_CELL_MAX = 20

_CreaseBBoxEntry = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    'Crease',
]


_LINE_ANGLE_BINS = 64
_LINE_ANGLE_STEP = math.pi / _LINE_ANGLE_BINS
_LINE_NORMALS: tuple[tuple[float, float], ...] = tuple(
    (-math.sin(i * _LINE_ANGLE_STEP), math.cos(i * _LINE_ANGLE_STEP))
    for i in range(_LINE_ANGLE_BINS)
)


def _canonical_line(
    c: 'Crease', cx: float = 0.5, cy: float = 0.5
) -> tuple[float, float, float]:
  """Returns canonical `(angle, centered_distance, length)` for `c`."""
  dx = c.p2.x - c.p1.x
  dy = c.p2.y - c.p1.y
  length = math.hypot(dx, dy)
  if length == 0.0:
    return 0.0, 0.0, 0.0
  ux, uy = dx / length, dy / length
  if uy < 0.0 or (uy == 0.0 and ux < 0.0):
    ux, uy = -ux, -uy
  if uy == 0.0:
    ux, uy = 1.0, 0.0
  angle = math.atan2(uy, ux)
  nx, ny = -uy, ux
  dist = nx * (c.p1.x - cx) + ny * (c.p1.y - cy)
  return angle, dist, length


class _LineSpatialIndex:
  """Exact Hough-curve spatial index for `Crease.is_collinear` queries.

  Each canonical angle bin `k` has fixed normal `N_k = _LINE_NORMALS[k]`. If
  `c.is_collinear(other)` holds, both endpoints of `other` lie within
  `perp_tol = DEFAULT_TOLERANCE * max_diam` of `c`'s infinite line, bounding
  both the angular difference `asin(2 * perp_tol / other.length)` and the
  projected distance mismatch `|q_dist - N_k . (other.mid - center)| <
  dist_step` at `c`'s nearest angle bin `k`. Indexing `other` along its
  valid angle bins (and placing ultra-short slivers in `short_creases`) yields
  zero false negatives while querying only 3 buckets per crease.
  """

  def __init__(self, creases: Sequence['Crease'] = ()):
    if creases:
      xmin = min(min(c.p1.x, c.p2.x) for c in creases)
      xmax = max(max(c.p1.x, c.p2.x) for c in creases)
      ymin = min(min(c.p1.y, c.p2.y) for c in creases)
      ymax = max(max(c.p1.y, c.p2.y) for c in creases)
      self._cx = 0.5 * (xmin + xmax)
      self._cy = 0.5 * (ymin + ymax)
      max_diam = max(1.0, math.hypot(xmax - xmin, ymax - ymin))
    else:
      self._cx = 0.5
      self._cy = 0.5
      max_diam = math.sqrt(2.0)
    r_max = 0.5 * max_diam
    self._perp_tol = DEFAULT_TOLERANCE * max_diam + 1e-9
    dist_step = max(
        0.02, self._perp_tol + 0.5 * _LINE_ANGLE_STEP * r_max + 1e-6
    )
    self._inv_dist_step = 1.0 / dist_step
    self.buckets: dict[tuple[int, int], list['Crease']] = (
        collections.defaultdict(list)
    )
    self.short_creases: list['Crease'] = []
    for c in creases:
      self.add(c)

  def add(self, c: 'Crease') -> None:
    """Indexes `c` across every canonical angle bin it can be collinear with."""
    angle, _, length = _canonical_line(c, self._cx, self._cy)
    two_perp = 2.0 * self._perp_tol
    if length <= 0.0 or two_perp >= 0.35 * length:
      self.short_creases.append(c)
      return
    delta_angle = math.asin(two_perp / length) + 1e-7
    a_lo = math.ceil((angle - delta_angle) / _LINE_ANGLE_STEP - 0.5 - 1e-9)
    a_hi = math.floor((angle + delta_angle) / _LINE_ANGLE_STEP + 0.5 + 1e-9)
    rel_mx = 0.5 * (c.p1.x + c.p2.x) - self._cx
    rel_my = 0.5 * (c.p1.y + c.p2.y) - self._cy
    inv_dist = self._inv_dist_step
    buckets = self.buckets
    if a_hi - a_lo + 1 >= _LINE_ANGLE_BINS:
      indices: Iterable[int] = range(_LINE_ANGLE_BINS)
    else:
      indices = (a % _LINE_ANGLE_BINS for a in range(a_lo, a_hi + 1))
    for a_mod in indices:
      nx, ny = _LINE_NORMALS[a_mod]
      d_idx = round((nx * rel_mx + ny * rel_my) * inv_dist)
      buckets[(a_mod, d_idx)].append(c)

  def query_collinear(self, c: 'Crease') -> list['Crease']:
    """Returns all indexed creases satisfying `c.is_collinear(other)`."""
    angle, dist, _ = _canonical_line(c, self._cx, self._cy)
    a_raw = round(angle / _LINE_ANGLE_STEP)
    if a_raw >= _LINE_ANGLE_BINS:
      a_mod = 0
      dist = -dist
    else:
      a_mod = a_raw
    d_idx = round(dist * self._inv_dist_step)
    seen: set[int] = set()
    collinear: list['Crease'] = []
    buckets = self.buckets
    for dd in (-1, 0, 1):
      for other in buckets.get((a_mod, d_idx + dd), ()):
        oid = id(other)
        if oid not in seen:
          seen.add(oid)
          if c.is_collinear(other):
            collinear.append(other)
    for other in self.short_creases:
      oid = id(other)
      if oid not in seen:
        seen.add(oid)
        if c.is_collinear(other):
          collinear.append(other)
    return collinear


class _PointTipIndex:
  """Spatial index for fast endpoint query in _has_trivial_extension."""

  def __init__(self, step: float = 0.005):
    self.step = step
    self.buckets: dict[tuple[int, int], list[tuple[point.Point, 'Crease']]] = (
        collections.defaultdict(list)
    )

  def add(self, p: point.Point, c: 'Crease') -> None:
    cx = int(round(p.x / self.step))
    cy = int(round(p.y / self.step))
    self.buckets[(cx, cy)].append((p, c))

  def has_trivial_extension(self, p: point.Point) -> bool:
    """Checks whether point p has exactly two incident creases across buckets."""
    cx = int(round(p.x / self.step))
    cy = int(round(p.y / self.step))
    seen: set[int] = set()
    count = 0
    for dx in (-1, 0, 1):
      for dy in (-1, 0, 1):
        for pt, c in self.buckets.get((cx + dx, cy + dy), ()):
          cid = id(c)
          if cid not in seen:
            seen.add(cid)
            if p.isclose(pt):
              count += 1
              if count > 2:
                return False
    return count == 2


class CreaseBBoxIndex:
  """Exact 2D spatial-grid index over crease bounding boxes."""

  def __init__(self, cell_size: float = _BBOX_GRID_STEP):
    self._inv_cell = 1.0 / cell_size
    self._buckets: dict[tuple[int, int], list[_CreaseBBoxEntry]] = (
        collections.defaultdict(list)
    )

  def _cell_range(
      self, xmin: float, ymin: float, xmax: float, ymax: float
  ) -> tuple[int, int, int, int]:
    inv = self._inv_cell
    ix0 = max(_BBOX_CELL_MIN, min(_BBOX_CELL_MAX, math.floor(xmin * inv)))
    ix1 = max(_BBOX_CELL_MIN, min(_BBOX_CELL_MAX, math.floor(xmax * inv)))
    iy0 = max(_BBOX_CELL_MIN, min(_BBOX_CELL_MAX, math.floor(ymin * inv)))
    iy1 = max(_BBOX_CELL_MIN, min(_BBOX_CELL_MAX, math.floor(ymax * inv)))
    return ix0, ix1, iy0, iy1

  def add(self, c: 'Crease') -> None:
    """Inserts `c` into every grid cell overlapping its bounding box."""
    p1 = c.p1
    p2 = c.p2
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    xmin = x1 if x1 < x2 else x2
    xmax = x2 if x1 < x2 else x1
    ymin = y1 if y1 < y2 else y2
    ymax = y2 if y1 < y2 else y1
    entry: _CreaseBBoxEntry = (x1, y1, x2, y2, xmin, ymin, xmax, ymax, c)
    ix0, ix1, iy0, iy1 = self._cell_range(xmin, ymin, xmax, ymax)
    buckets = self._buckets
    for ix in range(ix0, ix1 + 1):
      for iy in range(iy0, iy1 + 1):
        buckets[(ix, iy)].append(entry)

  def any_overlapping(self, c: 'Crease') -> bool:
    """Returns True iff any indexed crease satisfies `c.is_overlapping(other)`."""
    xmin, ymin, xmax, ymax = c.bounding_box
    pad = DEFAULT_TOLERANCE + 1e-6
    ix0, ix1, iy0, iy1 = self._cell_range(
        xmin - pad, ymin - pad, xmax + pad, ymax + pad
    )
    buckets = self._buckets
    if ix0 == ix1 and iy0 == iy1:
      cell = buckets.get((ix0, iy0))
      if not cell:
        return False
      for _, _, _, _, oxmin, oymin, oxmax, oymax, other in cell:
        if (
            xmax < oxmin - DEFAULT_TOLERANCE
            or xmin > oxmax + DEFAULT_TOLERANCE
            or ymax < oymin - DEFAULT_TOLERANCE
            or ymin > oymax + DEFAULT_TOLERANCE
        ):
          continue
        if c.is_overlapping(other):
          return True
      return False

    seen: set[int] = set()
    for ix in range(ix0, ix1 + 1):
      for iy in range(iy0, iy1 + 1):
        cell = buckets.get((ix, iy))
        if not cell:
          continue
        for _, _, _, _, oxmin, oymin, oxmax, oymax, other in cell:
          oid = id(other)
          if oid in seen:
            continue
          seen.add(oid)
          if (
              xmax < oxmin - DEFAULT_TOLERANCE
              or xmin > oxmax + DEFAULT_TOLERANCE
              or ymax < oymin - DEFAULT_TOLERANCE
              or ymin > oymax + DEFAULT_TOLERANCE
          ):
            continue
          if c.is_overlapping(other):
            return True
    return False

  def collect_split_ts(self, c: 'Crease', tolerance: float) -> list[float]:
    """Collects strict interior intersection parameters along `c`."""
    p1 = c.p1
    p2 = c.p2
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    xmin = x1 if x1 < x2 else x2
    xmax = x2 if x1 < x2 else x1
    ymin = y1 if y1 < y2 else y2
    ymax = y2 if y1 < y2 else y1
    pad = tolerance + 1e-6
    ix0, ix1, iy0, iy1 = self._cell_range(
        xmin - pad, ymin - pad, xmax + pad, ymax + pad
    )
    buckets = self._buckets
    split_ts: list[float] = []
    if ix0 == ix1 and iy0 == iy1:
      cell = buckets.get((ix0, iy0))
      if cell:
        for x3, y3, x4, y4, oxmin, oymin, oxmax, oymax, other in cell:
          if (
              xmax < oxmin - tolerance
              or xmin > oxmax + tolerance
              or ymax < oymin - tolerance
              or ymin > oymax + tolerance
          ):
            continue
          if _creases_share_endpoints(c, other):
            continue
          t = _intersection_t_coords(x1, y1, x2, y2, x3, y3, x4, y4, tolerance)
          if t is not None:
            split_ts.append(t)
      return split_ts

    seen: set[int] = set()
    for ix in range(ix0, ix1 + 1):
      for iy in range(iy0, iy1 + 1):
        cell = buckets.get((ix, iy))
        if not cell:
          continue
        for x3, y3, x4, y4, oxmin, oymin, oxmax, oymax, other in cell:
          oid = id(other)
          if oid in seen:
            continue
          seen.add(oid)
          if (
              xmax < oxmin - tolerance
              or xmin > oxmax + tolerance
              or ymax < oymin - tolerance
              or ymin > oymax + tolerance
          ):
            continue
          if _creases_share_endpoints(c, other):
            continue
          t = _intersection_t_coords(x1, y1, x2, y2, x3, y3, x4, y4, tolerance)
          if t is not None:
            split_ts.append(t)
    return split_ts


def _creases_share_endpoints(c1: 'Crease', c2: 'Crease') -> bool:
  """Returns True if two creases share both endpoints in either orientation."""
  if c1 is c2:
    return True
  return (c1.p1.isclose(c2.p1) and c1.p2.isclose(c2.p2)) or (
      c1.p1.isclose(c2.p2) and c1.p2.isclose(c2.p1)
  )


def _intersection_t_coords(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
    atol: float,
) -> float | None:
  """Returns strict interior parameter `t` along `(x1, y1)->(x2, y2)` if hit."""
  den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
  if -1e-12 < den < 1e-12:
    return None
  t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
  if not (atol < t < 1.0 - atol):
    return None
  u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
  if -atol < u < 1.0 + atol:
    return t
  return None


def prepare_sorted_tips(
    tips: Iterable[point.Point],
) -> tuple[list[point.Point], list[float]]:
  """Returns tips sorted by x-coordinate along with parallel x array."""
  sorted_tips = sorted(tips, key=lambda p: p.x)
  return sorted_tips, [p.x for p in sorted_tips]


def _snap_to_closest_tip(
    px: float,
    py: float,
    sorted_tips: Sequence[point.Point],
    sorted_xs: Sequence[float],
    tol: float,
    require_on_paper: bool,
    endpoint_fallback: point.Point | None = None,
) -> point.Point:
  """Snaps `(px, py)` to the Euclidean-closest tip strictly within `tol`."""
  if sorted_tips and tol > 0.0:
    min_dist_sq = tol * tol
    best_tip: point.Point | None = None
    idx = bisect.bisect_left(sorted_xs, px - tol)
    n_tips = len(sorted_tips)
    while idx < n_tips:
      tip = sorted_tips[idx]
      dx = tip.x - px
      if dx > 0.0 and dx * dx > min_dist_sq:
        break
      dy = tip.y - py
      dy_sq = dy * dy
      if dy_sq <= min_dist_sq:
        dist_sq = dx * dx + dy_sq
        if dist_sq < min_dist_sq:
          min_dist_sq = dist_sq
          best_tip = tip
      idx += 1
    if best_tip is not None:
      if best_tip.require_on_paper == require_on_paper:
        return best_tip
      return point.Point(
          x=best_tip.x, y=best_tip.y, require_on_paper=require_on_paper
      )
  if endpoint_fallback is not None:
    return endpoint_fallback
  return point.Point(x=px, y=py, require_on_paper=require_on_paper)


def split_creases_batch(
    creases_to_split: Sequence['Crease'],
    other_creases: Sequence['Crease'],
    tips: Iterable[point.Point] = (),
    tolerance: float = DEFAULT_TOLERANCE,
) -> list['Crease']:
  """Splits a batch of creases using shared sorted tips and spatial indexing.

  Args:
    creases_to_split: Creases to split into intersection segments.
    other_creases: Intersecting creases queried as supplied without filtering
      `Fold.TRIANGLE`; callers pre-filter when needed (matching
      `CreasePattern.split_creases`).
    tips: Candidate endpoint tips used for snapping split points.
    tolerance: Parametric and Euclidean snapping tolerance.

  Returns:
    Split segments across all `creases_to_split` in input order.
  """
  sorted_tips, sorted_xs = prepare_sorted_tips(tips)
  if len(other_creases) >= 32 and len(creases_to_split) >= 4:
    bbox_index = CreaseBBoxIndex()
    for oc in other_creases:
      bbox_index.add(oc)
    out: list['Crease'] = []
    for c in creases_to_split:
      split_ts = bbox_index.collect_split_ts(c, tolerance)
      out.extend(
          c._segments_from_split_ts(split_ts, sorted_tips, sorted_xs, tolerance)  # pylint: disable=protected-access
      )
    return out

  other_entries: list[_CreaseBBoxEntry] = []
  for oc in other_creases:
    p1 = oc.p1
    p2 = oc.p2
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    other_entries.append((
        x1,
        y1,
        x2,
        y2,
        x1 if x1 < x2 else x2,
        y1 if y1 < y2 else y2,
        x2 if x1 < x2 else x1,
        y2 if y1 < y2 else y1,
        oc,
    ))
  out = []
  for c in creases_to_split:
    out.extend(
        c._split_with_entries(other_entries, sorted_tips, sorted_xs, tolerance)  # pylint: disable=protected-access
    )
  return out


def split_crease_at_points(
    crease: 'Crease',
    points: Iterable[point.Point],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list['Crease']:
  """Splits `crease` at strictly interior candidate points.

  Projects matching interior points along `crease.p1_init -> crease.p2_init`,
  deduplicates nearby points within `tolerance`, and preserves `fold`,
  `fold_percent`, `fold_step`, and `elevation` on every returned sub-segment.

  Args:
    crease: The crease segment to split.
    points: Candidate points; only those strictly inside `crease` are used.
    tolerance: Euclidean tolerance used to drop near-duplicate split points.

  Returns:
    Sub-segments ordered from `crease.p1_init` toward `crease.p2_init`, or
    `[crease]` when no strictly interior split point exists.
  """
  p1, p2 = crease.p1_init, crease.p2_init
  dx = p2.x - p1.x
  dy = p2.y - p1.y
  len_sq = dx * dx + dy * dy
  if len_sq <= 0.0:
    return [crease]
  keyed: list[tuple[float, point.Point]] = []
  for pt in points:
    if crease.point_status(pt) == PointStatus.INTERNAL:
      t = ((pt.x - p1.x) * dx + (pt.y - p1.y) * dy) / len_sq
      if 0.0 < t < 1.0:
        keyed.append((t, pt))
  if not keyed:
    return [crease]
  keyed.sort(key=lambda item: item[0])
  ordered = [p1]
  for _, pt in keyed:
    if math.hypot(pt.x - ordered[-1].x, pt.y - ordered[-1].y) > tolerance:
      ordered.append(pt)
  if math.hypot(p2.x - ordered[-1].x, p2.y - ordered[-1].y) <= tolerance:
    ordered[-1] = p2
  else:
    ordered.append(p2)
  if len(ordered) <= 2:
    return [crease]
  return [
      dataclasses.replace(crease, p1_init=p_a, p2_init=p_b)
      for p_a, p_b in zip(ordered, ordered[1:])
      if math.hypot(p_a.x - p_b.x, p_a.y - p_b.y) > tolerance
  ]


def merge_creases(
    creases: Sequence['Crease'],
    only_trivial: bool = False,
    merge_zebra: bool = False,
) -> list['Crease']:
  """Merges creases into new, potentially longer creases."""
  if len(creases) <= 1:
    return list(creases)

  if len(creases) < 200:
    merged_creases = []
    for c1 in creases:
      # creases are already split into segments, if they overlap with a
      # previously merged crease, they must be a subsegment of the merged crease
      # and can be ignored.
      if any(c1.is_overlapping(c_) for c_ in merged_creases):
        continue
      merged_c = c1.merge_with_segments(creases, only_trivial, merge_zebra)
      if all(not merged_c.is_overlapping(c_) for c_ in merged_creases):
        merged_creases.append(merged_c)
      else:
        merged_creases.append(c1)
    return merged_creases

  line_index = _LineSpatialIndex(creases)

  point_tip_index = None
  if only_trivial:
    point_tip_index = _PointTipIndex()
    for c in creases:
      if c.fold != Fold.TRIANGLE:
        point_tip_index.add(c.p1, c)
        point_tip_index.add(c.p2, c)

  merged_creases = []
  merged_bbox_index = CreaseBBoxIndex()
  for c1 in creases:
    if merged_bbox_index.any_overlapping(c1):
      continue
    collinear = line_index.query_collinear(c1)
    merged_c = c1.merge_with_segments(
        creases,
        only_trivial=only_trivial,
        merge_zebra=merge_zebra,
        collinear_creases=collinear,
        point_tip_index=point_tip_index,
    )
    if not merged_bbox_index.any_overlapping(merged_c):
      merged_creases.append(merged_c)
      merged_bbox_index.add(merged_c)
    else:
      merged_creases.append(c1)
      merged_bbox_index.add(c1)
  return merged_creases


@dataclasses.dataclass(frozen=True, eq=False)
class Crease:
  """Represents a crease in an Origami Crease Pattern.

  A crease is defined by two points, a fold type and a fold percentage.
  The properties `p1` and `p2` provide the crease points sorted by norm.

  Attributes:
    p1_init: The first point defining the crease.
    p2_init: The second point defining the crease.
    p1: The first point in the crease, sorted by norm.
    p2: The second point in the crease, sorted by norm.
    fold: The type of fold.
    fold_percent: How much the crease is folded in simulation, in [0, 1].
    fold_step: The order of the fold, starting from 0.
    elevation: The elevation of a pleat crease (with a constant elevation) in
      box pleating designs. Only axial creases have defined (non-None)
      elevation.

  Raises:
    ValueError: If the tips are too close or if the fold_percent is not in
      [0, 1].
  """

  p1_init: point.Point
  p2_init: point.Point
  fold: Fold = dataclasses.field(kw_only=True)
  fold_percent: float = dataclasses.field(default=1.0, kw_only=True)
  fold_step: int = dataclasses.field(default=0, kw_only=True)
  elevation: int | None = dataclasses.field(default=None, kw_only=True)

  def __post_init__(self):
    if self.p1_init.isclose(self.p2_init):
      raise ValueError(
          f'The tips are too close: {self.p1_init} and {self.p2_init}.'
      )

    if not 0.0 <= self.fold_percent <= 1.0:
      raise ValueError(
          f'fold_percent must be between 0.0 and 1.0, got {self.fold_percent}.'
      )

    if self.fold_step < 0:
      raise ValueError(f'fold_step must be non-negative, got {self.fold_step}.')

  @classmethod
  def from_dict(cls, c_dict: CreaseDict | None) -> Self | None:
    """Converts a dictionary to a Crease object."""
    if c_dict is None:
      return None
    defaults = {}
    for field in dataclasses.fields(cls):
      defaults[field.name] = field.default
    return cls(
        p1_init=point.Point.from_dict(c_dict['p1']),
        p2_init=point.Point.from_dict(c_dict['p2']),
        fold=Fold(c_dict['fold']),
        fold_percent=c_dict.get('fold_percent', defaults['fold_percent']),
        fold_step=c_dict.get('fold_step', defaults['fold_step']),
        elevation=c_dict.get('elevation', defaults['elevation']),
    )

  def asdict(self) -> CreaseDict:
    """Converts a Crease to a dict for JSON serialization."""
    d = CreaseDict(
        p1=self.p1.as_dict(),
        p2=self.p2.as_dict(),
        fold=self.fold.value,
        fold_percent=self.fold_percent,
        fold_step=self.fold_step,
    )
    if self.elevation is not None:
      d['elevation'] = self.elevation
    return d

  @functools.cached_property
  def p1(self) -> point.Point:
    """The first point of the crease, sorted by norm."""
    p1, _ = point.sort_by_norm(self.p1_init, self.p2_init)
    return p1

  @functools.cached_property
  def p2(self) -> point.Point:
    """The second point of the crease, sorted by norm."""
    _, p2 = point.sort_by_norm(self.p1_init, self.p2_init)
    return p2

  def flip_tips(self) -> 'Crease':
    """Flips the tips of the crease."""
    return dataclasses.replace(self, p1_init=self.p2_init, p2_init=self.p1_init)

  def flip_fold(self) -> 'Crease':
    """Flips the fold of the crease."""
    return dataclasses.replace(self, fold=self.fold.flip())

  @functools.cached_property
  def left_point(self) -> point.Point:
    """The crease point closest to the left border."""
    left_p, _ = point.sort_by_x(self.p1_init, self.p2_init)
    return left_p

  @functools.cached_property
  def right_point(self) -> point.Point:
    """The crease point closest to the right border."""
    _, right_p = point.sort_by_x(self.p1_init, self.p2_init)
    return right_p

  @functools.cached_property
  def top_point(self) -> point.Point:
    """The crease point closest to the top border (y-value = 0)."""
    # The y-value starts at 0 at the top of the CP and then y-value increases
    # as we move down the CP.
    top_p, _ = point.sort_by_y(self.p1_init, self.p2_init)
    return top_p

  @functools.cached_property
  def bottom_point(self) -> point.Point:
    """The crease point closest to the bottom border (y-value = 1)."""
    # The y-value starts at 0 at the top of the CP and then y-value increases
    # as we move down the CP.
    _, bottom_p = point.sort_by_y(self.p1_init, self.p2_init)
    return bottom_p

  @functools.cached_property
  def close_to_left_border(self) -> bool:
    """Returns true if the left-most crease point is close to the left border."""
    return self.left_point.close_to_left_border

  @functools.cached_property
  def close_to_right_border(self) -> bool:
    """Returns true if the right-most crease point is close to the right border."""
    return self.right_point.close_to_right_border

  @functools.cached_property
  def close_to_top_border(self) -> bool:
    """Returns true if the top-most crease point is close to the top border."""
    return self.top_point.close_to_top_border

  @functools.cached_property
  def close_to_bottom_border(self) -> bool:
    """Returns true if the bottom-most crease point is close to the bottom border."""
    return self.bottom_point.close_to_bottom_border

  def vertex_with_crease_external(
      self, crease: Self
  ) -> tuple[float, float] | None:
    out = self._vertex_with_crease(crease, True, True)
    if out is None:
      return None
    if not isinstance(out, tuple):
      raise ValueError(f'Expected intersection tuple, got {out!r}')
    out = tuple(out)
    return out

  def vertex_with_crease(
      self,
      crease: Self,
      extend_lines: bool = False,
      include_external: bool = False,
  ) -> point.Point | None:
    """Given another crease, return a common point between the two if it exists.

    Args:
      crease: The other crease to compute the intersection with.
      extend_lines: If true, extend the lines infinitely to find intersections.
      include_external: If true, include intersections that are not on the
        creases, or even outside the grid.

    Returns:
        The common point between the two creases if it exists (otherwise None).
        If include_external is True, then the intersection point may be outside
        the grid and the line segments. In this case, the function returns a
        tuple of (x, y) of the intersection point.
    """
    out = self._vertex_with_crease(
        crease,
        extend_lines=extend_lines,
        include_external=include_external,
    )
    if isinstance(out, tuple):
      out = point.Point(x=out[0], y=out[1], require_on_paper=False)
    return out

  def _vertex_with_crease(
      self,
      crease: Self,
      extend_lines: bool,
      include_external: bool,
  ) -> point.Point | None | tuple[float, float]:
    """Given another crease, return a common point between the two if it exists.

    See: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

    Args:
      crease: The other crease to compute the intersection with.
      extend_lines: If true, extend the lines infinitely to find intersections.
      include_external: If true, include intersections that are not on the
        creases, or even outside the grid.

    Returns:
        The common point between the two creases if it exists (otherwise None).
        If include_external is True, then the intersection point may be outside
        the grid and the line segments. In this case, the function returns a
        tuple of (x, y) of the intersection point.
    """
    if not extend_lines and self.not_in_encompassing_rectangles(crease):
      return None

    if self.p1.isclose(crease.p1) or self.p1.isclose(crease.p2):
      p = self.p1
    elif self.p2.isclose(crease.p1) or self.p2.isclose(crease.p2):
      p = self.p2
    elif self.is_parallel(crease):
      return None
    elif self.is_vertical and self.is_perpendicular(crease):
      p = point.Point(x=self.p1.x, y=crease.p1.y, require_on_paper=False)
    elif self.is_horizontal and self.is_perpendicular(crease):
      p = point.Point(x=crease.p1.x, y=self.p1.y, require_on_paper=False)
    else:
      x1, y1 = self.p1.x, self.p1.y
      x2, y2 = self.p2.x, self.p2.y
      x3, y3 = crease.p1.x, crease.p1.y
      x4, y4 = crease.p2.x, crease.p2.y

      denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
      determinant_1 = x1 * y2 - y1 * x2
      determinant_2 = x3 * y4 - y3 * x4
      if self.is_horizontal:
        y = self.p1.y
      elif crease.is_horizontal:
        y = crease.p1.y
      else:
        y = (
            determinant_1 * (y3 - y4) - determinant_2 * (y1 - y2)
        ) / denominator
      if self.is_vertical:
        x = self.p1.x
      elif crease.is_vertical:
        x = crease.p1.x
      else:
        x = (
            determinant_1 * (x3 - x4) - determinant_2 * (x1 - x2)
        ) / denominator

      p = point.Point(x=x, y=y, require_on_paper=False)

    if not p.on_paper and not include_external:
      return None
    if not extend_lines and (
        self.point_status(p) == PointStatus.EXTERNAL
        or crease.point_status(p) == PointStatus.EXTERNAL
    ):
      return None
    if include_external:
      return (p.x, p.y)
    return point.Point(x=p.x, y=p.y)

  @functools.cached_property
  def midpoint(self) -> point.Point:
    """Returns the midpoint of the crease."""
    return point.midpoint(self.p1, self.p2)

  @functools.cached_property
  def is_valid(self) -> bool:
    """Returns true if the crease is valid.

    For example, if it is a border crease, its midpoint must be on the border.
    """
    if self.fold == Fold.BORDER:
      return self.midpoint.is_border
    else:
      return not self.midpoint.is_border

  def mountain_or_valley(self) -> bool:
    """Returns true if the crease is mountain or valley."""
    return self.fold == Fold.MOUNTAIN or self.fold == Fold.VALLEY

  def isclose(self, other: 'Crease', atol: float = DEFAULT_TOLERANCE) -> bool:
    """Returns true if the creases share the same endpoints."""
    return self.p1.isclose(other.p1, atol=atol) and self.p2.isclose(
        other.p2, atol=atol
    )

  @functools.cached_property
  def is_vertical(self) -> bool:
    """Returns true if the crease is vertical."""
    return math.isclose(self.p1.x, self.p2.x, abs_tol=DEFAULT_TOLERANCE)

  @functools.cached_property
  def is_horizontal(self) -> bool:
    """Returns true if the crease is horizontal."""
    return math.isclose(self.p1.y, self.p2.y, abs_tol=DEFAULT_TOLERANCE)

  @functools.cached_property
  def is_diagonal(self) -> bool:
    """Returns true if the crease is diagonal."""
    return not (self.is_vertical or self.is_horizontal)

  @property
  def internal(self) -> bool:
    """Returns true if the crease is not a border crease."""
    return self.fold != Fold.BORDER

  @functools.cached_property
  def is_overlapping_border(self) -> bool:
    """Returns true if the crease is a border crease."""
    # Both endpoints must be on the border.
    if not (self.p1.is_border and self.p2.is_border):
      return False

    # If the crease is vertical, its x-coordinate must be 0 or 1.
    if self.is_vertical:
      return math.isclose(self.p1.x, 0.0) or math.isclose(self.p1.x, 1.0)

    # If the crease is horizontal, its y-coordinate must be 0 or 1.
    if self.is_horizontal:
      return math.isclose(self.p1.y, 0.0) or math.isclose(self.p1.y, 1.0)

    # A diagonal crease connecting two corners cannot overlap a border edge.
    return False

  @functools.cached_property
  def length(self) -> float:
    """Returns the length of the crease."""
    return math.hypot(self.p1.x - self.p2.x, self.p1.y - self.p2.y)

  @functools.cached_property
  def slope(self) -> np.float64:
    """Returns the slope of the crease. If vertical, returns np.inf."""
    return np.float64(self.p2.y - self.p1.y) / np.float64(self.p2.x - self.p1.x)

  @functools.cached_property
  def y_intercept(self) -> float | None:
    """Returns the b (y-intercept) of the crease."""
    if self.is_vertical:
      return None
    return (self.p2.x * self.p1.y - self.p1.x * self.p2.y) / (
        self.p2.x - self.p1.x
    )

  @functools.cached_property
  def get_equation_a_b_c(self) -> tuple[float, float, float]:
    """Returns the equation of the crease in the form ax + by + c = 0."""
    if self.is_vertical:
      return (1, 0, -self.p1.x)
    else:
      return (float(-self.slope), 1, float(self.slope * self.p1.x - self.p1.y))

  @functools.cached_property
  def normed_vector(self) -> tuple[float, float]:
    """Returns the normalized vector of the crease, from p2 to p1."""
    crease_length = self.length
    return (
        (self.p1.x - self.p2.x) / crease_length,
        (self.p1.y - self.p2.y) / crease_length,
    )

  def is_parallel(self, other: 'Crease') -> bool:
    """Returns true if the creases are parallel."""
    v1x, v1y = self.normed_vector
    v2x, v2y = other.normed_vector
    return math.isclose(v1x * v2y, v1y * v2x, abs_tol=DEFAULT_TOLERANCE)

  def is_perpendicular(self, other: 'Crease') -> bool:
    """Returns true if the creases are perpendicular."""
    v1x, v1y = self.normed_vector
    v2x, v2y = other.normed_vector
    return math.isclose(v1x * v2x + v1y * v2y, 0.0, abs_tol=DEFAULT_TOLERANCE)

  def point_status(self, p: point.Point) -> PointStatus:
    """Returns the status of the point on this crease segment."""
    xmin, ymin, xmax, ymax = self.bounding_box
    tol = DEFAULT_TOLERANCE + 1e-6
    if (
        p.x < xmin - tol
        or p.x > xmax + tol
        or p.y < ymin - tol
        or p.y > ymax + tol
    ):
      return PointStatus.EXTERNAL
    if self.p1.isclose(p) or self.p2.isclose(p):
      return PointStatus.TIP
    elif self.is_intermediate_point(p):
      return PointStatus.INTERNAL
    else:
      return PointStatus.EXTERNAL

  def is_intermediate_point(self, p: point.Point) -> bool:
    """Returns True if the point is on the crease segment, excluding tips."""
    if not point.is_collinear(self.p1, self.p2, p):
      return False

    # For a collinear point, check if it lies between the endpoints using the
    # dot product
    p1_x = p.x - self.p1.x
    p1_y = p.y - self.p1.y
    p1_norm = math.sqrt(p1_x**2 + p1_y**2)
    p2_x = p.x - self.p2.x
    p2_y = p.y - self.p2.y
    p2_norm = math.sqrt(p2_x**2 + p2_y**2)

    cosine = (p1_x * p2_x + p1_y * p2_y) / (p1_norm * p2_norm)

    return strict_less(cosine, 0.0)

  def is_overlapping(self, other: 'Crease', ndigits: int | None = None) -> bool:
    """Returns true if the creases have non-zero overlap.

    If ndigits is set then rounds both creases to the given number of digits
    before comparing them.

    Args:
      other: The other crease to check against.
      ndigits: The number of digits to round to, or None for no rounding.

    Returns:
      True if the creases have non-zero overlap.
    """
    if ndigits is not None:
      return self.round(ndigits).is_overlapping(
          other.round(ndigits), ndigits=None
      )

    if self.not_in_encompassing_rectangles(other):
      return False
    if self.isclose(other):
      return True

    if not self.is_collinear(other):
      return False

    # Check if endpoints of one crease lie within the other crease
    return (
        self.point_status(other.p1) == PointStatus.INTERNAL
        or self.point_status(other.p2) == PointStatus.INTERNAL
        or other.point_status(self.p1) == PointStatus.INTERNAL
        or other.point_status(self.p2) == PointStatus.INTERNAL
    )

  def get_overlap(
      self, other: 'Crease', ndigits: int | None = None
  ) -> Optional['Crease']:
    """Returns the overlap of the creases, or None if there is no overlap.

    If ndigits is set then rounds both creases to the given number of digits
    before comparing them.

    Args:
      other: The other crease to check against.
      ndigits: The number of digits to round to, or None for no rounding.

    Returns:
      The overlap of the creases, or None if there is no overlap.
    """
    if ndigits is not None:
      return self.round(ndigits).get_overlap(other.round(ndigits), ndigits=None)

    if self.isclose(other):
      return self
    if not self.is_overlapping(other):
      return None
    tip_candidates = []
    for tip in [self.p1, self.p2]:
      point_status = other.point_status(tip)
      if (
          point_status == PointStatus.INTERNAL
          or point_status == PointStatus.TIP
      ):
        tip_candidates.append(tip)
    for tip in [other.p1, other.p2]:
      if self.point_status(tip) == PointStatus.INTERNAL:
        tip_candidates.append(tip)
    if len(tip_candidates) != 2:
      raise ValueError(
          f'Expected 2 overlapping tip candidates, got {tip_candidates}'
      )
    return Crease(
        tip_candidates[0],
        tip_candidates[1],
        fold=Fold.MOUNTAIN,
        elevation=self.elevation,
    )

  def is_within(self, other: 'Crease', match_fold: bool = True) -> bool:
    """Returns true if this crease is fully within the other crease.

    Args:
      other: The other crease to check against.
      match_fold: If true, the fold and fold_percent must match.
    """
    return (
        (not match_fold or equal_fold_and_fold_percent(self, other))
        and other.point_status(self.p1) != PointStatus.EXTERNAL
        and other.point_status(self.p2) != PointStatus.EXTERNAL
    )

  def __eq__(self, other: object) -> bool:
    """Returns true if the creases are equal."""
    if not isinstance(other, Crease):
      return NotImplemented

    return (
        self.isclose(other)
        and self.fold == other.fold
        and math.isclose(
            self.fold_percent, other.fold_percent, abs_tol=DEFAULT_TOLERANCE
        )
    )

  def eq_endpoints_and_fold(self, other: object) -> bool:
    """Returns true if the creases endpoints and fold are equal."""
    if not isinstance(other, Crease):
      return NotImplemented

    return self.isclose(other) and self.fold == other.fold

  def _segments_from_split_ts(
      self,
      split_ts: list[float],
      sorted_tips: Sequence[point.Point],
      sorted_xs: Sequence[float],
      tolerance: float,
  ) -> list['Crease']:
    """Builds split segments from raw interior `t` parameters on `p1 -> p2`."""
    p_start = self.p1
    p_end = self.p2
    x1, y1 = p_start.x, p_start.y
    x2, y2 = p_end.x, p_end.y
    require_on_paper = p_start.require_on_paper and p_end.require_on_paper

    split_ts.append(0.0)
    split_ts.append(1.0)
    split_ts.sort()

    unique_ts = [split_ts[0]]
    prev_t = split_ts[0]
    for t in split_ts[1:]:
      if abs(t - prev_t) >= tolerance:
        unique_ts.append(t)
        prev_t = t

    segment_points: list[point.Point] = []
    for t in unique_ts:
      if t == 0.0:
        px, py = x1, y1
        fallback: point.Point | None = p_start
      elif t == 1.0:
        px, py = x2, y2
        fallback = p_end
      else:
        one_minus_t = 1.0 - t
        px = x1 * one_minus_t + x2 * t
        py = y1 * one_minus_t + y2 * t
        fallback = None
      segment_points.append(
          _snap_to_closest_tip(
              px,
              py,
              sorted_tips,
              sorted_xs,
              tolerance,
              require_on_paper,
              endpoint_fallback=fallback,
          )
      )

    segment_pairs = [
        (p_a, p_b)
        for p_a, p_b in itertools.pairwise(segment_points)
        if not p_a.isclose(p_b, atol=tolerance)
    ]

    fold = self.fold
    zebra_fold = fold == Fold.MOUNTAIN_ZEBRA or fold == Fold.VALLEY_ZEBRA
    if zebra_fold:
      fold = Fold.MOUNTAIN if fold == Fold.MOUNTAIN_ZEBRA else Fold.VALLEY
      if self.is_vertical:
        if p_start.y > p_end.y:
          segment_pairs.reverse()
      elif p_start.x > p_end.x:
        segment_pairs.reverse()

    result_creases: list['Crease'] = []
    for p_a, p_b in segment_pairs:
      result_creases.append(
          dataclasses.replace(self, p1_init=p_a, p2_init=p_b, fold=fold)
      )
      if zebra_fold:
        fold = fold.flip()
    return result_creases

  def _split_with_entries(
      self,
      other_entries: Sequence[_CreaseBBoxEntry],
      sorted_tips: Sequence[point.Point],
      sorted_xs: Sequence[float],
      tolerance: float,
  ) -> list['Crease']:
    """Splits against pre-unpacked bounding-box entries."""
    p1 = self.p1
    p2 = self.p2
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    xmin = x1 if x1 < x2 else x2
    xmax = x2 if x1 < x2 else x1
    ymin = y1 if y1 < y2 else y2
    ymax = y2 if y1 < y2 else y1
    split_ts: list[float] = []
    for x3, y3, x4, y4, oxmin, oymin, oxmax, oymax, other in other_entries:
      if (
          xmax < oxmin - tolerance
          or xmin > oxmax + tolerance
          or ymax < oymin - tolerance
          or ymin > oymax + tolerance
      ):
        continue
      if _creases_share_endpoints(self, other):
        continue
      t = _intersection_t_coords(x1, y1, x2, y2, x3, y3, x4, y4, tolerance)
      if t is not None:
        split_ts.append(t)
    return self._segments_from_split_ts(
        split_ts, sorted_tips, sorted_xs, tolerance
    )

  def split_to_segments(
      self,
      creases: list['Crease'],
      tips: Iterable[point.Point] = (),
      cpp: bool = False,
      tolerance: float = DEFAULT_TOLERANCE,
  ) -> list['Crease']:
    """Splits the crease into smaller segments ordered parametrically along it."""
    del cpp
    sorted_tips, sorted_xs = prepare_sorted_tips(tips)
    p1 = self.p1
    p2 = self.p2
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    xmin = x1 if x1 < x2 else x2
    xmax = x2 if x1 < x2 else x1
    ymin = y1 if y1 < y2 else y2
    ymax = y2 if y1 < y2 else y1
    split_ts: list[float] = []
    for c in creases:
      if c.fold == Fold.TRIANGLE:
        continue
      op1 = c.p1
      op2 = c.p2
      x3, y3 = op1.x, op1.y
      x4, y4 = op2.x, op2.y
      oxmin = x3 if x3 < x4 else x4
      oxmax = x4 if x3 < x4 else x3
      oymin = y3 if y3 < y4 else y4
      oymax = y4 if y3 < y4 else y3
      if (
          xmax < oxmin - tolerance
          or xmin > oxmax + tolerance
          or ymax < oymin - tolerance
          or ymin > oymax + tolerance
      ):
        continue
      if _creases_share_endpoints(self, c):
        continue
      t = _intersection_t_coords(x1, y1, x2, y2, x3, y3, x4, y4, tolerance)
      if t is not None:
        split_ts.append(t)
    return self._segments_from_split_ts(
        split_ts, sorted_tips, sorted_xs, tolerance
    )

  def _has_trivial_extension(
      self, p: point.Point, creases: Sequence['Crease']
  ) -> bool:
    """Checks if there are exactly two collinear creases intersecting at p.

    Args:
      p: The intersection point.
      creases: A sequence of creases to check against.

    Returns:
      True if there is a trivial extension of the crease at p.
    """
    intersecting_creases = [
        c
        for c in creases
        if c.point_status(p) == PointStatus.TIP and c.fold != Fold.TRIANGLE
    ]
    return len(intersecting_creases) == 2

  def is_collinear(self, other: 'Crease') -> bool:
    """Returns true if the creases lie on the same infinite line."""
    return point.is_collinear(
        self.p1, self.p2, other.p1
    ) and point.is_collinear(self.p1, self.p2, other.p2)

  def merge_with_segments(
      self,
      creases: Sequence['Crease'],
      only_trivial: bool = False,
      merge_zebra: bool = False,
      collinear_creases: Sequence['Crease'] | None = None,
      point_tip_index: _PointTipIndex | None = None,
  ) -> 'Crease':
    """Merges the crease with collinear segments to form a new, potentially longer crease.

    Args:
      creases: A list of creases to try merge with.
      only_trivial: If true, only merges at connection points where exactly two
        creases meet (a trivial extension).
      merge_zebra: If true, allows merging of alternating Mountain/Valley
        segments into a zebra crease.
      collinear_creases: Optional pre-filtered collinear creases.
      point_tip_index: Optional spatial index for endpoint queries.

    Returns:
      A new Crease object representing the merged segments.

    Raises:
      ValueError: If non-trivial merging is disallowed and overlapping
        creases are found.
    """
    if only_trivial:
      check_creases = (
          collinear_creases if collinear_creases is not None else creases
      )
      for c in check_creases:
        if not c.isclose(self) and self.is_overlapping(c):
          raise ValueError(
              'Overlapping creases are not allowed for trivial merges:'
              f' {self} and {c}.'
          )

    if collinear_creases is None:
      collinear_list = [c for c in creases if self.is_collinear(c)]
    else:
      collinear_list = list(collinear_creases)
    collinear_list.sort(key=_get_min_value)
    ind = collinear_list.index(self)

    p_max, right_zebra, performed_right_walk = self._right_walk(
        merge_zebra=merge_zebra,
        only_trivial=only_trivial,
        collinear_creases=collinear_list,
        creases=creases,
        ind=ind,
        point_tip_index=point_tip_index,
    )
    p_min, left_zebra, fold = self._left_walk(
        merge_zebra=merge_zebra,
        right_zebra=right_zebra,
        performed_right_walk=performed_right_walk,
        only_trivial=only_trivial,
        collinear_creases=collinear_list,
        creases=creases,
        ind=ind,
        point_tip_index=point_tip_index,
    )
    # Decide the fold type of the new crease.
    if p_max.isclose(_get_max_point(self)) and p_min.isclose(
        _get_min_point(self)
    ):
      return self

    if (performed_right_walk and right_zebra) or left_zebra:
      if fold == Fold.MOUNTAIN:
        fold = Fold.MOUNTAIN_ZEBRA
      else:
        fold = Fold.VALLEY_ZEBRA

    return dataclasses.replace(self, p1_init=p_min, p2_init=p_max, fold=fold)

  def _right_walk(
      self,
      merge_zebra: bool,
      only_trivial: bool,
      collinear_creases: Sequence['Crease'],
      creases: Sequence['Crease'],
      ind: int,
      point_tip_index: _PointTipIndex | None = None,
  ) -> tuple[point.Point, bool, bool]:
    """Finds the rightmost point we can walk to from self across collinear segments.

    Args:
      merge_zebra: Whether to merge zebra creases.
      only_trivial: Whether to only merge trivial creases.
      collinear_creases: Sorted collinear creases to walk right on.
      creases: The creases to check for trivial extensions.
      ind: The index of self in the collinear creases.
      point_tip_index: Optional spatial index for endpoint queries.

    Returns:
      A tuple containing:
        The new rightmost point after walking.
        Whether a zebra walk was performed.
        Whether a right walk was performed.
    """
    zebra_walk = False
    p_max = _get_max_point(self)
    if ind == len(collinear_creases) - 1:
      return p_max, False, False

    # Decide the walk type: same fold or zebra
    next_crease = collinear_creases[ind + 1]
    if merge_zebra and one_mountain_one_valley(self, next_crease):
      fold_match_fn = one_mountain_one_valley
      zebra_walk = True
    else:
      fold_match_fn = equal_fold_and_fold_percent

    for i in range(ind + 1, len(collinear_creases)):
      current = collinear_creases[i]
      prev = collinear_creases[i - 1]
      if point_tip_index is not None:
        trivial = not only_trivial or point_tip_index.has_trivial_extension(
            p_max
        )
      else:
        trivial = not only_trivial or self._has_trivial_extension(
            p_max, creases
        )
      if (
          fold_match_fn(prev, current)
          and _get_min_point(current).isclose(p_max)
          and trivial
      ):
        p_max = _get_max_point(current)
      else:
        break
    performed_right_walk = not p_max.isclose(_get_max_point(self))
    return p_max, zebra_walk, performed_right_walk

  def _left_walk(
      self,
      merge_zebra: bool,
      right_zebra: bool,
      performed_right_walk: bool,
      only_trivial: bool,
      collinear_creases: Sequence['Crease'],
      creases: Sequence['Crease'],
      ind: int,
      point_tip_index: _PointTipIndex | None = None,
  ) -> tuple[point.Point, bool, Fold]:
    """Finds the leftmost point we can walk to from self across collinear segments.

    We only walk left if it is the same type of walk as the right walk, or if
    there was no right walk.

    Args:
      merge_zebra: Whether to merge zebra creases.
      right_zebra: Whether a zebra walk was performed on the right.
      performed_right_walk: Whether a right walk was performed.
      only_trivial: Whether to only merge trivial creases.
      collinear_creases: Sorted collinear creases to walk left on.
      creases: The creases to check for trivial extensions.
      ind: The index of self in the collinear creases.
      point_tip_index: Optional spatial index for endpoint queries.

    Returns:
      A tuple containing:
        The new leftmost point after walking.
        Whether a zebra walk was performed.
        The fold type of the new crease.
    """
    fold = self.fold
    p_min = _get_min_point(self)
    zebra_walk = False
    if ind == 0:
      return p_min, False, fold

    # Decide the walk type: same fold or zebra
    prev_crease = collinear_creases[ind - 1]
    if merge_zebra and one_mountain_one_valley(self, prev_crease):
      fold_match_fn = one_mountain_one_valley
      zebra_walk = True
    else:
      fold_match_fn = equal_fold_and_fold_percent

    # Decide if we can walk left.
    if performed_right_walk and (right_zebra != zebra_walk):
      return p_min, False, fold

    # Walk.
    for i in range(ind - 1, -1, -1):
      current = collinear_creases[i]
      next_c = collinear_creases[i + 1]
      if point_tip_index is not None:
        trivial = not only_trivial or point_tip_index.has_trivial_extension(
            p_min
        )
      else:
        trivial = not only_trivial or self._has_trivial_extension(
            p_min, creases
        )
      if (
          fold_match_fn(next_c, current)
          and _get_max_point(current).isclose(p_min)
          and trivial
      ):
        p_min = _get_min_point(current)
        fold = current.fold
      else:
        break
    performed_left_walk = not p_min.isclose(_get_min_point(self))
    return p_min, zebra_walk and performed_left_walk, fold

  @functools.cached_property
  def bounding_box(self) -> tuple[float, float, float, float]:
    """Returns the bounding box (xmin, ymin, xmax, ymax) of the crease."""
    return (
        min(self.p1.x, self.p2.x),
        min(self.p1.y, self.p2.y),
        max(self.p1.x, self.p2.x),
        max(self.p1.y, self.p2.y),
    )

  def not_in_encompassing_rectangles(self, c: 'Crease') -> bool:
    """Checks if the bounding boxes of two creases do not overlap."""
    xmin, ymin, xmax, ymax = self.bounding_box
    c_xmin, c_ymin, c_xmax, c_ymax = c.bounding_box
    return (
        strict_less(xmax, c_xmin)
        or strict_less(ymax, c_ymin)
        or strict_less(c_xmax, xmin)
        or strict_less(c_ymax, ymin)
    )

  def transform(
      self,
      weight: types.Matrix2x2,
      bias: types.Vector2D,
      require_on_paper: bool = True,
  ) -> 'Crease':
    """Applies a linear transformation to the crease's endpoints.

    Args:
      weight: A 2x2 matrix for scaling and rotation.
      bias: A 2D offset vector (b1, b2).
      require_on_paper: If True, the transformed points must be on the paper.

    Returns:
      A new Crease object with the transformed coordinates.
    """
    p1t = self.p1.transform(weight, bias, require_on_paper)
    p2t = self.p2.transform(weight, bias, require_on_paper)
    return dataclasses.replace(self, p1_init=p1t, p2_init=p2t)

  @functools.cached_property
  def fold_angle(self) -> float:
    """The fold angle of the crease in radians."""
    return self.fold_percent * self.fold.to_angle()

  @functools.cached_property
  def hash_value(self) -> int:
    """Returns the hash value of the crease."""
    return hash((
        self.p1,
        self.p2,
        self.fold,
        round(self.fold_percent, 3),
    ))

  @functools.cached_property
  def exact_match(self) -> 'CreaseWithExactMatch':
    """Returns a cached CreaseWithExactMatch wrapper for this crease."""
    return CreaseWithExactMatch(crease=self)

  @functools.cached_property
  def intersects_border(self) -> bool:
    """Returns true if the crease intersects the border."""
    return self.p1.is_border or self.p2.is_border

  def is_tip(self, p: point.Point) -> bool:
    """Returns true if the point is a tip of a crease."""
    return p.isclose(self.p1) or p.isclose(self.p2)

  def intersect_at_tip(self, other: 'Crease') -> bool:
    """Returns true if two creases intersect at a tip."""
    return self.is_tip(other.p1) or self.is_tip(other.p2)

  def mountain_to_valley(self) -> Fold:
    """Returns valley if fold is mountain, or mountain if fold is valley.

    Raises ValueError if fold is not mountain or valley.
    """
    match self.fold:
      case Fold.MOUNTAIN:
        return Fold.VALLEY
      case Fold.VALLEY:
        return Fold.MOUNTAIN
      case other:
        raise ValueError(f'{other} is not a mountain or valley')

  def round(self, ndigits: int) -> 'Crease':
    """Round floating point values of crease points to ndigits."""
    return dataclasses.replace(
        self,
        p1_init=point.Point(
            x=round(self.p1.x, ndigits), y=round(self.p1.y, ndigits)
        ),
        p2_init=point.Point(
            x=round(self.p2.x, ndigits), y=round(self.p2.y, ndigits)
        ),
    )


@dataclasses.dataclass(frozen=True, eq=False)
class CreaseWithExactMatch:
  """A crease with exact match and a __hash__.

  CreaseWithExactMatch wraps the Crease class with a __hash__ and an exact
  __eq__. Crease on the other hand uses isclose to check for equality.
  The class is used when hashing and equality is required to be
  exact, which is useful in box pleating.
  """

  crease: Crease

  def __hash__(self) -> int:
    return self.crease.hash_value

  def __eq__(self, other: object) -> bool:
    if self is other:
      return True
    if not isinstance(other, type(self)):
      return NotImplemented
    if self.crease is other.crease:
      return True
    return (
        self.crease.p1 == other.crease.p1
        and self.crease.p2 == other.crease.p2
        and self.crease.fold == other.crease.fold
        and round(self.crease.fold_percent, 3)
        == round(other.crease.fold_percent, 3)
    )


@dataclasses.dataclass(frozen=True)
class Creases:
  """A set of creases."""

  creases: frozenset[CreaseWithExactMatch]

  def __contains__(self, other: object) -> bool:
    if isinstance(other, Crease):
      return other.exact_match in self.creases
    elif isinstance(other, CreaseWithExactMatch):
      return other in self.creases
    return False

  def __iter__(self) -> Iterator[CreaseWithExactMatch]:
    for c in self.creases:
      yield c

  @classmethod
  def create_from_creases(cls, creases: Iterable[Crease]) -> 'Creases':
    """Creates a Creases object from a sequence of Crease objects."""
    return cls(creases=frozenset(c.exact_match for c in creases))


def _get_min_point(c: Crease) -> point.Point:
  """Returns the point on the crease with the smallest coordinate."""
  if c.is_vertical:
    return c.p1 if c.p1.y < c.p2.y else c.p2
  else:
    return c.p1 if c.p1.x < c.p2.x else c.p2


def _get_max_point(c: Crease) -> point.Point:
  """Returns the point on the crease with the largest coordinate."""
  if c.is_vertical:
    return c.p2 if c.p1.y < c.p2.y else c.p1
  else:
    return c.p2 if c.p1.x < c.p2.x else c.p1


def _get_min_value(c: Crease) -> float:
  """Returns the minimum coordinate value of the crease."""
  p_min = _get_min_point(c)
  return p_min.y if c.is_vertical else p_min.x
