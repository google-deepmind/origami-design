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
"""Dihedral grid reflection and convex clipping for clip-pattern shaping.

Each cell `(row, col)` of a `G x G` box-pleating grid splits along both
diagonals into four quarter-triangles (`CellTriangle`). Folding a base across
horizontal, vertical, and 45-degree grid creases induces an exact integer
dihedral reflection `(R, b)` on every flat sub-polygon of a flap or river. A
template defined in unit-flap coordinates is mapped onto each sub-polygon by
that isometry (flipping mountain/valley when `det(R) < 0`) and clipped via
Cyrus-Beck against the sub-polygon's convex hull.
"""

import collections
from collections.abc import Iterable, Iterator, Sequence
import dataclasses
import enum
import functools
import itertools
import math

import numpy as np

from ... import timeout as timeout_lib
from ...crease_pattern import crease
from ...crease_pattern import crease_pattern
from ...crease_pattern import point
from .. import shaping_utils
from . import templates


class CellTriangle(enum.IntEnum):
  """Quarter-triangle of a unit grid cell split by both diagonals."""

  BOTTOM = 0
  RIGHT = 1
  LEFT = 2
  TOP = 3


class EdgeKind(enum.IntEnum):
  """Crease edge slot owned by cell `(row, col)`."""

  BOTTOM = 0  # Horizontal edge at Y = row + 1 (for row < G - 1).
  RIGHT = 1  # Vertical edge at X = col + 1 (for col < G - 1).
  RIGHT_DIAG = 2  # Anti-diagonal (col, row + 1) -- (col + 1, row).
  LEFT_DIAG = 3  # Main diagonal (col, row) -- (col + 1, row + 1).


type TriangleKey = tuple[int, int, CellTriangle]
type _Mat2 = tuple[tuple[int, int], tuple[int, int]]
type _Isometry = tuple[_Mat2, tuple[int, int]]

_IDENTITY_ISO: _Isometry = (((1, 0), (0, 1)), (0, 0))
_REFLECT_X: _Mat2 = ((-1, 0), (0, 1))
_REFLECT_Y: _Mat2 = ((1, 0), (0, -1))
_REFLECT_LEFT_DIAG: _Mat2 = ((0, 1), (1, 0))
_REFLECT_RIGHT_DIAG: _Mat2 = ((0, -1), (-1, 0))

# Quarter-triangle center offset `(dx, dy)` within `[col, col+1] x [row,
# row+1]`.
_CENTER_OFFSET: dict[CellTriangle, tuple[float, float]] = {
    CellTriangle.TOP: (0.5, 0.25),
    CellTriangle.RIGHT: (0.75, 0.5),
    CellTriangle.BOTTOM: (0.5, 0.75),
    CellTriangle.LEFT: (0.25, 0.5),
}

# Quarter-triangle vertex offsets `(dx, dy)` within `[col, col+1] x [row,
# row+1]`.
_VERTEX_OFFSETS: dict[CellTriangle, tuple[tuple[float, float], ...]] = {
    CellTriangle.TOP: ((0.0, 0.0), (1.0, 0.0), (0.5, 0.5)),
    CellTriangle.RIGHT: ((1.0, 0.0), (1.0, 1.0), (0.5, 0.5)),
    CellTriangle.BOTTOM: ((1.0, 1.0), (0.0, 1.0), (0.5, 0.5)),
    CellTriangle.LEFT: ((0.0, 1.0), (0.0, 0.0), (0.5, 0.5)),
}


def _compose_iso(outer: _Isometry, inner: _Isometry) -> _Isometry:
  """Returns `outer(inner(p))` for two exact integer 2D grid isometries."""
  (a00, a01), (a10, a11) = outer[0]
  ox, oy = outer[1]
  (b00, b01), (b10, b11) = inner[0]
  ix, iy = inner[1]
  mat = (
      (a00 * b00 + a01 * b10, a00 * b01 + a01 * b11),
      (a10 * b00 + a11 * b10, a10 * b01 + a11 * b11),
  )
  vec = (a00 * ix + a01 * iy + ox, a10 * ix + a11 * iy + oy)
  return mat, vec


def _align_frame(key: TriangleKey) -> _Isometry:
  """Maps unit-flap template `(s, t)` to grid-paper `(X, Y)` on `key`."""
  row, col, tri = key
  match tri:
    case CellTriangle.BOTTOM:
      return ((1, 0), (0, -1)), (col, row + 1)
    case CellTriangle.TOP:
      return ((-1, 0), (0, 1)), (col + 1, row)
    case CellTriangle.RIGHT:
      return ((0, -1), (-1, 0)), (col + 1, row + 1)
    case CellTriangle.LEFT:
      return ((0, 1), (1, 0)), (col, row)


def _apply_iso(
    iso: _Isometry, pt: tuple[float, float], inv_g: float
) -> tuple[float, float]:
  """Maps point `pt` by `iso` and rescales from `[0, G]^2` to `[0, 1]^2`."""
  (m00, m01), (m10, m11) = iso[0]
  bx, by = iso[1]
  x, y = pt
  return (m00 * x + m01 * y + bx) * inv_g, (m10 * x + m11 * y + by) * inv_g


def _triangle_center(key: TriangleKey, grid_size: int) -> tuple[float, float]:
  """Returns normalized `(cx, cy)` paper coordinates of `key`'s reference."""
  row, col, tri = key
  dx, dy = _CENTER_OFFSET[tri]
  return (col + dx) / grid_size, (row + dy) / grid_size


def _snap_grid_coord(value: float, grid_size: int, atol: float) -> int:
  """Snaps a `[0, 1]` coordinate to an integer grid line in `[0, grid_size]`."""
  scaled = value * grid_size
  snapped = round(scaled)
  if abs(scaled - snapped) > atol or not 0 <= snapped <= grid_size:
    raise ValueError(
        f'Coordinate {value} does not align with grid of size {grid_size} '
        f'(scaled={scaled}, nearest={snapped}).'
    )
  return snapped


def _convex_hull_ccw(points: np.ndarray) -> np.ndarray:
  """Computes the strict counter-clockwise 2D convex hull (Monotone Chain)."""
  unique = sorted(
      {(round(float(p[0]), 12), round(float(p[1]), 12)) for p in points}
  )
  if len(unique) <= 2:
    return np.asarray(unique, dtype=float)

  def _cross(
      o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
  ) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

  lower: list[tuple[float, float]] = []
  for pt in unique:
    while len(lower) >= 2 and _cross(lower[-2], lower[-1], pt) <= 1e-12:
      lower.pop()
    lower.append(pt)
  upper: list[tuple[float, float]] = []
  for pt in reversed(unique):
    while len(upper) >= 2 and _cross(upper[-2], upper[-1], pt) <= 1e-12:
      upper.pop()
    upper.append(pt)
  return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _clip_segment_to_convex_polygon(
    p1: tuple[float, float],
    p2: tuple[float, float],
    hull_ccw: np.ndarray,
    *,
    atol: float = 1e-9,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
  """Returns the interior sub-segment of `(p1, p2)` inside `hull_ccw`.

  Segments collinear with a polygon edge are excluded so existing base and
  hinge creases along sub-polygon boundaries are never duplicated.

  Args:
    p1: Start point `(x, y)` of the segment.
    p2: End point `(x, y)` of the segment.
    hull_ccw: `(M, 2)` array of strict CCW convex polygon vertices.
    atol: Metric distance tolerance for boundary classification.

  Returns:
    Clipped `((x1, y1), (x2, y2))`, or `None` if no interior span survives.
  """
  num_verts = len(hull_ccw)
  if num_verts < 3:
    return None
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  t_enter = 0.0
  t_exit = 1.0
  for i in range(num_verts):
    ax = float(hull_ccw[i, 0])
    ay = float(hull_ccw[i, 1])
    nxt = (i + 1) % num_verts
    ex = float(hull_ccw[nxt, 0]) - ax
    ey = float(hull_ccw[nxt, 1]) - ay
    edge_len = math.hypot(ex, ey)
    if edge_len <= 1e-15:
      continue
    tol = atol * edge_len
    # Positive when strictly inside the CCW half-plane of edge `i`.
    h1 = ex * (p1[1] - ay) - ey * (p1[0] - ax)
    dh = ex * dy - ey * dx
    if abs(dh) <= tol:
      if h1 <= tol:
        return None
      continue
    t_hit = -h1 / dh
    if dh > 0.0:
      if t_hit > t_enter:
        t_enter = t_hit
    elif t_hit < t_exit:
      t_exit = t_hit
    if t_enter >= t_exit:
      return None
  return (
      (p1[0] + t_enter * dx, p1[1] + t_enter * dy),
      (p1[0] + t_exit * dx, p1[1] + t_exit * dy),
  )


def _range_overlap(
    lo1: float, hi1: float, lo2: float, hi2: float, tol: float = 1e-9
) -> tuple[float, float] | None:
  """Returns non-degenerate intersection of `[lo1, hi1]` and `[lo2, hi2]`."""
  lo = max(lo1, lo2)
  hi = min(hi1, hi2)
  return (lo, hi) if hi - lo > tol else None


def _clip_axis_segment_to_unit_square(
    x1: float, y1: float, x2: float, y2: float, tol: float = 1e-9
) -> tuple[float, float, float, float] | None:
  """Clips an axis-aligned segment to `[0, 1]^2`."""
  if (
      (x1 < 0.0 and x2 < 0.0)
      or (x1 > 1.0 and x2 > 1.0)
      or (y1 < 0.0 and y2 < 0.0)
      or (y1 > 1.0 and y2 > 1.0)
  ):
    return None
  cx1 = min(1.0, max(0.0, x1))
  cy1 = min(1.0, max(0.0, y1))
  cx2 = min(1.0, max(0.0, x2))
  cy2 = min(1.0, max(0.0, y2))
  if math.isclose(cx1, cx2, abs_tol=tol) and math.isclose(
      cy1, cy2, abs_tol=tol
  ):
    return None
  return cx1, cy1, cx2, cy2


@dataclasses.dataclass(frozen=True)
class FoldGrid:
  """Boolean grid of horizontal, vertical, and 45-degree folded cell edges.

  Attributes:
    grid_size: Number of cells per side (`G`).
    folded: `(G, G, 4)` boolean array indexed by `(row, col, EdgeKind)`.
  """

  grid_size: int
  folded: np.ndarray

  @classmethod
  def from_grid_segments(
      cls,
      grid_size: int,
      segments: Iterable[tuple[int, int, int, int]],
  ) -> 'FoldGrid':
    """Builds a `FoldGrid` from integer `(col1, row1, col2, row2)` segments."""
    if grid_size <= 0:
      raise ValueError(f'grid_size must be positive, got {grid_size}.')
    folded = np.zeros((grid_size, grid_size, 4), dtype=bool)
    for c1, r1, c2, r2 in segments:
      if c1 == c2 and r1 == r2:
        raise ValueError(f'Zero-length grid segment at ({c1}, {r1}).')
      if r1 == r2:
        if 0 < r1 < grid_size:
          lo, hi = min(c1, c2), max(c1, c2)
          folded[r1 - 1, max(0, lo) : min(grid_size, hi), EdgeKind.BOTTOM] = (
              True
          )
      elif c1 == c2:
        if 0 < c1 < grid_size:
          lo, hi = min(r1, r2), max(r1, r2)
          folded[max(0, lo) : min(grid_size, hi), c1 - 1, EdgeKind.RIGHT] = True
      else:
        if abs(c2 - c1) != abs(r2 - r1):
          raise ValueError(
              'Only axis-aligned and 45-degree grid creases are supported; '
              f'got ({c1}, {r1}) -> ({c2}, {r2}).'
          )
        if c1 > c2:
          c1, r1, c2, r2 = c2, r2, c1, r1
        steps = c2 - c1
        if r2 > r1:
          for step in range(steps):
            folded[r1 + step, c1 + step, EdgeKind.LEFT_DIAG] = True
        else:
          for step in range(steps):
            folded[r1 - 1 - step, c1 + step, EdgeKind.RIGHT_DIAG] = True
    folded.setflags(write=False)
    return cls(grid_size=grid_size, folded=folded)

  @classmethod
  def from_crease_pattern(
      cls,
      pattern: crease_pattern.CreasePattern,
      grid_size: int | None = None,
      *,
      folds: Sequence[crease.Fold] = (crease.Fold.MOUNTAIN, crease.Fold.VALLEY),
      atol: float = 0.05,
  ) -> 'FoldGrid':
    """Rasterizes grid-aligned creases from `pattern` into a `FoldGrid`."""
    resolved = pattern.grid_size if grid_size is None else grid_size
    allowed = frozenset(folds)
    segments: list[tuple[int, int, int, int]] = []
    for c in pattern.creases:
      if c.fold not in allowed:
        continue
      c1 = _snap_grid_coord(c.p1_init.x, resolved, atol)
      r1 = _snap_grid_coord(c.p1_init.y, resolved, atol)
      c2 = _snap_grid_coord(c.p2_init.x, resolved, atol)
      r2 = _snap_grid_coord(c.p2_init.y, resolved, atol)
      if (c1, r1) != (c2, r2):
        segments.append((c1, r1, c2, r2))
    return cls.from_grid_segments(resolved, segments)

  @classmethod
  def from_packing_layout(
      cls,
      flap_rectangles: Sequence[shaping_utils.Rectangle],
      river_rectangles: Sequence[Sequence[shaping_utils.Rectangle]],
      grid_size: int,
      *,
      atol: float = 0.05,
  ) -> 'FoldGrid':
    """Builds a hinge `FoldGrid` directly from flap and river rectangles."""
    raw_unit: list[tuple[float, float, float, float]] = []
    for r in flap_rectangles:
      x1, y1 = r.x_start, r.y_start
      x2, y2 = x1 + r.x_len, y1 + r.y_len
      for seg in (
          (x1, y1, x2, y1),
          (x2, y1, x2, y2),
          (x2, y2, x1, y2),
          (x1, y2, x1, y1),
      ):
        clipped = _clip_axis_segment_to_unit_square(*seg)
        if clipped is not None:
          raw_unit.append(clipped)

    for river_a, river_b in itertools.combinations(river_rectangles, 2):
      for ra in river_a:
        ax1, ay1 = ra.x_start, ra.y_start
        ax2, ay2 = ax1 + ra.x_len, ay1 + ra.y_len
        for rb in river_b:
          bx1, by1 = rb.x_start, rb.y_start
          bx2, by2 = bx1 + rb.x_len, by1 + rb.y_len
          shared_y: float | None = None
          if math.isclose(ay1, by2, abs_tol=1e-9):
            shared_y = ay1
          elif math.isclose(ay2, by1, abs_tol=1e-9):
            shared_y = ay2
          if shared_y is not None:
            span = _range_overlap(ax1, ax2, bx1, bx2)
            if span is not None:
              clipped = _clip_axis_segment_to_unit_square(
                  span[0], shared_y, span[1], shared_y
              )
              if clipped is not None:
                raw_unit.append(clipped)
          shared_x: float | None = None
          if math.isclose(ax1, bx2, abs_tol=1e-9):
            shared_x = ax1
          elif math.isclose(ax2, bx1, abs_tol=1e-9):
            shared_x = ax2
          if shared_x is not None:
            span = _range_overlap(ay1, ay2, by1, by2)
            if span is not None:
              clipped = _clip_axis_segment_to_unit_square(
                  shared_x, span[0], shared_x, span[1]
              )
              if clipped is not None:
                raw_unit.append(clipped)

    grid_segs: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in raw_unit:
      c1 = _snap_grid_coord(x1, grid_size, atol)
      r1 = _snap_grid_coord(y1, grid_size, atol)
      c2 = _snap_grid_coord(x2, grid_size, atol)
      r2 = _snap_grid_coord(y2, grid_size, atol)
      if (c1, r1) != (c2, r2):
        grid_segs.append((c1, r1, c2, r2))
    return cls.from_grid_segments(grid_size, grid_segs)

  def combined(self, other: 'FoldGrid') -> 'FoldGrid':
    """Returns edge-wise union of two `FoldGrid` instances of equal size."""
    if self.grid_size != other.grid_size:
      raise ValueError(
          f'Grid size mismatch: {self.grid_size} vs {other.grid_size}.'
      )
    merged = np.logical_or(self.folded, other.folded)
    merged.setflags(write=False)
    return FoldGrid(grid_size=self.grid_size, folded=merged)

  def neighbors(
      self, key: TriangleKey
  ) -> Iterator[tuple[TriangleKey, bool, _Isometry]]:
    """Yields `(neighbor_key, is_folded, edge_reflection)` around `key`."""
    row, col, tri = key
    cell = self.folded[row, col]
    rdiag = (_REFLECT_RIGHT_DIAG, (col + row + 1, col + row + 1))
    ldiag = (_REFLECT_LEFT_DIAG, (col - row, row - col))
    match tri:
      case CellTriangle.TOP:
        yield (
            (row, col, CellTriangle.RIGHT),
            bool(cell[EdgeKind.RIGHT_DIAG]),
            rdiag,
        )
        yield (
            (row, col, CellTriangle.LEFT),
            bool(cell[EdgeKind.LEFT_DIAG]),
            ldiag,
        )
        if row > 0:
          yield (
              (row - 1, col, CellTriangle.BOTTOM),
              bool(self.folded[row - 1, col, EdgeKind.BOTTOM]),
              (_REFLECT_Y, (0, 2 * row)),
          )
      case CellTriangle.RIGHT:
        yield (
            (row, col, CellTriangle.TOP),
            bool(cell[EdgeKind.RIGHT_DIAG]),
            rdiag,
        )
        yield (
            (row, col, CellTriangle.BOTTOM),
            bool(cell[EdgeKind.LEFT_DIAG]),
            ldiag,
        )
        if col + 1 < self.grid_size:
          yield (
              (row, col + 1, CellTriangle.LEFT),
              bool(cell[EdgeKind.RIGHT]),
              (_REFLECT_X, (2 * (col + 1), 0)),
          )
      case CellTriangle.BOTTOM:
        yield (
            (row, col, CellTriangle.RIGHT),
            bool(cell[EdgeKind.LEFT_DIAG]),
            ldiag,
        )
        yield (
            (row, col, CellTriangle.LEFT),
            bool(cell[EdgeKind.RIGHT_DIAG]),
            rdiag,
        )
        if row + 1 < self.grid_size:
          yield (
              (row + 1, col, CellTriangle.TOP),
              bool(cell[EdgeKind.BOTTOM]),
              (_REFLECT_Y, (0, 2 * (row + 1))),
          )
      case CellTriangle.LEFT:
        yield (
            (row, col, CellTriangle.BOTTOM),
            bool(cell[EdgeKind.RIGHT_DIAG]),
            rdiag,
        )
        yield (
            (row, col, CellTriangle.TOP),
            bool(cell[EdgeKind.LEFT_DIAG]),
            ldiag,
        )
        if col > 0:
          yield (
              (row, col - 1, CellTriangle.RIGHT),
              bool(self.folded[row, col - 1, EdgeKind.RIGHT]),
              (_REFLECT_X, (2 * col, 0)),
          )

  def connected_components(self) -> list[list[TriangleKey]]:
    """Partitions all `4 * G^2` quarter-triangles across unfolded edges."""
    visited = np.zeros((self.grid_size, self.grid_size, 4), dtype=bool)
    components: list[list[TriangleKey]] = []
    for row in range(self.grid_size):
      for col in range(self.grid_size):
        for tri in CellTriangle:
          if visited[row, col, tri]:
            continue
          visited[row, col, tri] = True
          start: TriangleKey = (row, col, tri)
          queue: collections.deque[TriangleKey] = collections.deque([start])
          comp: list[TriangleKey] = []
          while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt, is_folded, _ in self.neighbors(cur):
              if not is_folded and not visited[nxt[0], nxt[1], nxt[2]]:
                visited[nxt[0], nxt[1], nxt[2]] = True
                queue.append(nxt)
          components.append(comp)
    return components


def find_align_triangles(hinges_grid: FoldGrid) -> list[TriangleKey]:
  """Selects one deterministic hinge-adjacent anchor triangle per fragment."""
  half = 0.5 * hinges_grid.grid_size
  anchors: list[TriangleKey] = []
  for comp in hinges_grid.connected_components():
    timeout_lib.maybe_timeout_cooperatively()
    candidates = [
        node
        for node in comp
        if any(is_folded for _, is_folded, _ in hinges_grid.neighbors(node))
    ]
    if not candidates:
      continue
    anchors.append(
        min(
            candidates,
            key=lambda k: (
                round((k[0] - half) ** 2 + (k[1] - half) ** 2, 8),
                -int(k[2]),
                k[0],
                k[1],
            ),
        )
    )
  return anchors


@dataclasses.dataclass(frozen=True, slots=True)
class _FragmentSection:
  """Single flat sub-polygon inside a hinge-bounded fragment."""

  representative: TriangleKey
  center: tuple[float, float]
  hull_ccw: np.ndarray
  frame: _Isometry
  flip_mv: bool


def _build_fragment_sections(
    base_grid: FoldGrid,
    hinges_grid: FoldGrid,
    align_key: TriangleKey,
    section_hulls: dict[TriangleKey, np.ndarray],
    node_to_rep: dict[TriangleKey, TriangleKey],
) -> tuple[_FragmentSection, ...]:
  """Computes exact folded frames for all flat sections of `align_key`."""
  iso_map: dict[TriangleKey, tuple[_Isometry, bool]] = {
      align_key: (_IDENTITY_ISO, False)
  }
  queue: collections.deque[TriangleKey] = collections.deque([align_key])
  seen_reps: set[TriangleKey] = set()
  while queue:
    cur = queue.popleft()
    seen_reps.add(node_to_rep[cur])
    cur_iso, cur_flip = iso_map[cur]
    for (nxt, hinge_folded, _), (_, base_folded, refl) in zip(
        hinges_grid.neighbors(cur), base_grid.neighbors(cur)
    ):
      if hinge_folded or nxt in iso_map:
        continue
      if base_folded:
        iso_map[nxt] = (_compose_iso(refl, cur_iso), not cur_flip)
      else:
        iso_map[nxt] = (cur_iso, cur_flip)
      queue.append(nxt)

  base_align = _align_frame(align_key)
  ordered_reps = sorted(seen_reps, key=lambda k: (k[0], k[1], int(k[2])))
  sections: list[_FragmentSection] = []
  for rep in ordered_reps:
    rel_iso, flip_mv = iso_map[rep]
    sections.append(
        _FragmentSection(
            representative=rep,
            center=_triangle_center(rep, base_grid.grid_size),
            hull_ccw=section_hulls[rep],
            frame=_compose_iso(rel_iso, base_align),
            flip_mv=flip_mv,
        )
    )
  return tuple(sections)


def _component_hull(comp: Sequence[TriangleKey], grid_size: int) -> np.ndarray:
  """Returns the normalized CCW convex hull of a triangle component."""
  inv_g = 1.0 / grid_size
  pts = np.empty((3 * len(comp), 2), dtype=float)
  idx = 0
  for row, col, tri in comp:
    for dx, dy in _VERTEX_OFFSETS[tri]:
      pts[idx, 0] = (col + dx) * inv_g
      pts[idx, 1] = (row + dy) * inv_g
      idx += 1
  return _convex_hull_ccw(pts)


def _render_sections(
    sections: Sequence[_FragmentSection],
    pattern: Sequence[templates.PatternSegment],
    grid_size: int,
    *,
    disable_clipping: bool = False,
    bounding_rectangles: Sequence[shaping_utils.Rectangle] | None = None,
    global_refine: bool = False,
) -> list[crease.Crease]:
  """Projects `pattern` onto `sections` and clips against each convex hull."""
  if not pattern or not sections:
    return []
  active = sections
  if bounding_rectangles is not None and not global_refine:
    margin = 1.5 / grid_size
    scoped = [
        sec
        for sec in sections
        if any(
            r.x_start - margin <= sec.center[0] <= r.x_start + r.x_len + margin
            and r.y_start - margin
            <= sec.center[1]
            <= r.y_start + r.y_len + margin
            for r in bounding_rectangles
        )
    ]
    if scoped:
      active = scoped

  inv_g = 1.0 / grid_size
  min_len = crease.DEFAULT_TOLERANCE * (1.0 + 1e-6)
  out: list[crease.Crease] = []
  for sec in active:
    timeout_lib.maybe_timeout_cooperatively()
    for seg in pattern:
      p1 = _apply_iso(sec.frame, seg.p1, inv_g)
      p2 = _apply_iso(sec.frame, seg.p2, inv_g)
      if disable_clipping:
        clipped: tuple[tuple[float, float], tuple[float, float]] | None = (
            p1,
            p2,
        )
      else:
        clipped = _clip_segment_to_convex_polygon(p1, p2, sec.hull_ccw)
      if clipped is None:
        continue
      q1, q2 = clipped
      if math.hypot(q2[0] - q1[0], q2[1] - q1[1]) <= min_len:
        continue
      fold = seg.fold
      if sec.flip_mv and fold in (crease.Fold.MOUNTAIN, crease.Fold.VALLEY):
        fold = fold.flip()
      fold_percent = (
          1.0 if fold == crease.Fold.BORDER else seg.angle_deg / 180.0
      )
      out.append(
          crease.Crease(
              point.Point(x=q1[0], y=q1[1], require_on_paper=False),
              point.Point(x=q2[0], y=q2[1], require_on_paper=False),
              fold=fold,
              fold_percent=fold_percent,
          )
      )
  return out


def shape_clip_pattern(
    *,
    base_grid: FoldGrid,
    hinges_grid: FoldGrid,
    align_triangle: TriangleKey,
    pattern: Sequence[templates.PatternSegment],
    disable_clipping: bool = False,
    bounding_rectangles: Sequence[shaping_utils.Rectangle] | None = None,
    global_refine: bool = False,
) -> list[crease.Crease]:
  """Projects and clips `pattern` onto the fragment at `align_triangle`."""
  combined = base_grid.combined(hinges_grid)
  section_hulls: dict[TriangleKey, np.ndarray] = {}
  node_to_rep: dict[TriangleKey, TriangleKey] = {}
  for comp in combined.connected_components():
    rep = comp[0]
    section_hulls[rep] = _component_hull(comp, base_grid.grid_size)
    for node in comp:
      node_to_rep[node] = rep
  sections = _build_fragment_sections(
      base_grid, hinges_grid, align_triangle, section_hulls, node_to_rep
  )
  return _render_sections(
      sections,
      pattern,
      base_grid.grid_size,
      disable_clipping=disable_clipping,
      bounding_rectangles=bounding_rectangles,
      global_refine=global_refine,
  )


def add_shaping_creases(
    cp: crease_pattern.CreasePattern, creases: Sequence[crease.Crease]
) -> crease_pattern.CreasePattern:
  """Returns a new `CreasePattern` combining `cp.creases` with `creases`."""
  return crease_pattern.CreasePattern(creases=[*cp.creases, *creases])


@dataclasses.dataclass(frozen=True)
class ClipPatternShaper:
  """Cached clip-pattern shaping index over a solved box-pleated base.

  Attributes:
    grid_size: Grid resolution `G` of `base_grid` and `hinges_grid`.
    base_grid: Folded-edge mask of the base crease pattern.
    hinges_grid: Folded-edge mask of flap and river hinge boundaries.
    align_triangles: Deterministic anchor triangles, one per hinge fragment.
  """

  grid_size: int
  base_grid: FoldGrid
  hinges_grid: FoldGrid
  align_triangles: tuple[TriangleKey, ...]

  @classmethod
  def from_grids(
      cls, base_grid: FoldGrid, hinges_grid: FoldGrid
  ) -> 'ClipPatternShaper':
    """Constructs a shaper from pre-built base and hinge `FoldGrid`s."""
    if base_grid.grid_size != hinges_grid.grid_size:
      raise ValueError(
          f'Grid size mismatch: {base_grid.grid_size} vs '
          f'{hinges_grid.grid_size}.'
      )
    return cls(
        grid_size=base_grid.grid_size,
        base_grid=base_grid,
        hinges_grid=hinges_grid,
        align_triangles=tuple(find_align_triangles(hinges_grid)),
    )

  @classmethod
  def from_crease_patterns(
      cls,
      base_cp: crease_pattern.CreasePattern,
      packing_cp: crease_pattern.CreasePattern,
  ) -> 'ClipPatternShaper':
    """Constructs a shaper from a solved base CP and its packing CP."""
    grid_size = base_cp.grid_size
    base_grid = FoldGrid.from_crease_pattern(base_cp, grid_size)
    hinges_grid = FoldGrid.from_crease_pattern(
        packing_cp, grid_size, folds=(crease.Fold.HINGE,)
    )
    return cls.from_grids(base_grid, hinges_grid)

  @classmethod
  def from_packing_layout(
      cls,
      base_cp: crease_pattern.CreasePattern,
      flap_rectangles: Sequence[shaping_utils.Rectangle],
      river_rectangles: Sequence[Sequence[shaping_utils.Rectangle]],
  ) -> 'ClipPatternShaper':
    """Constructs a shaper from a solved base CP and packing rectangles."""
    grid_size = base_cp.grid_size
    base_grid = FoldGrid.from_crease_pattern(base_cp, grid_size)
    hinges_grid = FoldGrid.from_packing_layout(
        flap_rectangles, river_rectangles, grid_size
    )
    return cls.from_grids(base_grid, hinges_grid)

  @functools.cached_property
  def _align_centers(self) -> tuple[tuple[float, float], ...]:
    return tuple(
        _triangle_center(key, self.grid_size) for key in self.align_triangles
    )

  @functools.cached_property
  def _node_to_align(self) -> dict[TriangleKey, TriangleKey]:
    align_set = set(self.align_triangles)
    mapping: dict[TriangleKey, TriangleKey] = {}
    for comp in self.hinges_grid.connected_components():
      anchor = next((k for k in self.align_triangles if k in set(comp)), None)
      if anchor is None or anchor not in align_set:
        continue
      for node in comp:
        mapping[node] = anchor
    return mapping

  @functools.cached_property
  def _sections_by_align(
      self,
  ) -> dict[TriangleKey, tuple[_FragmentSection, ...]]:
    """Computes folded fragment sections keyed by alignment triangle."""
    combined = self.base_grid.combined(self.hinges_grid)
    section_hulls: dict[TriangleKey, np.ndarray] = {}
    node_to_rep: dict[TriangleKey, TriangleKey] = {}
    for comp in combined.connected_components():
      rep = comp[0]
      section_hulls[rep] = _component_hull(comp, self.grid_size)
      for node in comp:
        node_to_rep[node] = rep
    return {
        align_key: _build_fragment_sections(
            self.base_grid,
            self.hinges_grid,
            align_key,
            section_hulls,
            node_to_rep,
        )
        for align_key in self.align_triangles
    }

  def get_align_triangle_for_rect(
      self, rect: shaping_utils.Rectangle
  ) -> TriangleKey:
    """Returns the anchor triangle for the hinge fragment covered by `rect`."""
    x_lo, x_hi = rect.x_start, rect.x_start + rect.x_len
    y_lo, y_hi = rect.y_start, rect.y_start + rect.y_len
    for key, (cx, cy) in zip(self.align_triangles, self._align_centers):
      if x_lo <= cx <= x_hi and y_lo <= cy <= y_hi:
        return key
    col_min = max(0, math.floor(x_lo * self.grid_size))
    col_max = min(self.grid_size - 1, math.ceil(x_hi * self.grid_size) - 1)
    row_min = max(0, math.floor(y_lo * self.grid_size))
    row_max = min(self.grid_size - 1, math.ceil(y_hi * self.grid_size) - 1)
    node_map = self._node_to_align
    for row in range(row_min, row_max + 1):
      for col in range(col_min, col_max + 1):
        for tri in CellTriangle:
          key: TriangleKey = (row, col, tri)
          cx, cy = _triangle_center(key, self.grid_size)
          if x_lo <= cx <= x_hi and y_lo <= cy <= y_hi and key in node_map:
            return node_map[key]
    raise ValueError(f'No align triangle found within rectangle: {rect}.')

  def shape_rectangles(
      self,
      rectangles: Sequence[shaping_utils.Rectangle],
      pattern: Sequence[templates.PatternSegment],
      *,
      bounding_rectangles: Sequence[shaping_utils.Rectangle] | None = None,
      global_refine: bool = False,
  ) -> list[crease.Crease]:
    """Projects and clips `pattern` onto the fragment matching `rectangles`."""
    align_key: TriangleKey | None = None
    for rect in rectangles:
      try:
        align_key = self.get_align_triangle_for_rect(rect)
        break
      except ValueError:
        continue
    if align_key is None:
      raise ValueError(
          f'No align triangle found within rectangles: {rectangles}.'
      )
    bounds = rectangles if bounding_rectangles is None else bounding_rectangles
    return _render_sections(
        self._sections_by_align[align_key],
        pattern,
        self.grid_size,
        bounding_rectangles=bounds,
        global_refine=global_refine,
    )
