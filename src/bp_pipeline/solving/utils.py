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

"""Utility functions for box pleating."""

from collections.abc import Collection, Iterable, Mapping, Sequence
import dataclasses
import functools

from bp_pipeline.crease_pattern import crease
from bp_pipeline.crease_pattern import point

_TIP_INV_CELL = 500.0  # cell width 2e-3 > point.DEFAULT_TOLERANCE (1e-3)
_LINE_INV_CELL = 250.0  # bucket width 4e-3 for collinear line families


class TipIndex:
  """Spatial bucket index for O(1) crease tip and tip-intersection lookups.

  When `grid_size` is provided, coordinates are keyed on the exact half-grid
  integer lattice `round(coord * 2 * grid_size)` (since Box Pleating vertices
  lie on `k / G` or `k / (2G)`), while still probing the 3x3 integer
  neighborhood
  so points within `point.DEFAULT_TOLERANCE` across a rounding boundary never
  miss.
  """

  def __init__(
      self,
      creases: Iterable[crease.Crease] = (),
      grid_size: int | None = None,
  ):
    self._scale: float = (
        float(2 * grid_size)
        if grid_size is not None and grid_size > 0
        else _TIP_INV_CELL
    )
    self._creases: list[crease.Crease] = []
    self._buckets: dict[tuple[int, int], list[int]] = {}
    for c in creases:
      self.add(c)

  def add(self, c: crease.Crease) -> None:
    """Adds a crease to the spatial tip index."""
    idx = len(self._creases)
    self._creases.append(c)
    scale = self._scale
    k1 = (round(c.p1.x * scale), round(c.p1.y * scale))
    self._buckets.setdefault(k1, []).append(idx)
    k2 = (round(c.p2.x * scale), round(c.p2.y * scale))
    if k2 != k1:
      self._buckets.setdefault(k2, []).append(idx)

  def creases_at_tip(self, p: point.Point) -> list[crease.Crease]:
    """Returns creases in insertion order that have p as a tip."""
    scale = self._scale
    bx = round(p.x * scale)
    by = round(p.y * scale)
    cand_indices: set[int] = set()
    for dx in (-1, 0, 1):
      for dy in (-1, 0, 1):
        bucket = self._buckets.get((bx + dx, by + dy))
        if bucket:
          cand_indices.update(bucket)
    if not cand_indices:
      return []
    return [
        self._creases[i]
        for i in sorted(cand_indices)
        if self._creases[i].is_tip(p)
    ]

  def has_tip(self, p: point.Point) -> bool:
    """Returns True if any indexed crease has p as a tip."""
    scale = self._scale
    bx = round(p.x * scale)
    by = round(p.y * scale)
    for dx in (-1, 0, 1):
      for dy in (-1, 0, 1):
        bucket = self._buckets.get((bx + dx, by + dy))
        if bucket:
          for i in bucket:
            if self._creases[i].is_tip(p):
              return True
    return False

  def count_at_tip(self, p: point.Point) -> int:
    """Returns the number of indexed creases that have p as a tip."""
    return len(self.creases_at_tip(p))

  def non_overlapping_at_tip(
      self, ref_crease: crease.Crease
  ) -> list[crease.Crease]:
    """Returns creases in insertion order that share a tip without overlapping."""
    scale = self._scale
    b1x = round(ref_crease.p1.x * scale)
    b1y = round(ref_crease.p1.y * scale)
    b2x = round(ref_crease.p2.x * scale)
    b2y = round(ref_crease.p2.y * scale)
    cand_indices: set[int] = set()
    for bx, by in ((b1x, b1y), (b2x, b2y)):
      for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
          bucket = self._buckets.get((bx + dx, by + dy))
          if bucket:
            cand_indices.update(bucket)
    if not cand_indices:
      return []
    result = []
    for i in sorted(cand_indices):
      c = self._creases[i]
      if ref_crease.intersect_at_tip(c) and not ref_crease.is_overlapping(c):
        result.append(c)
    return result


def _line_family_and_intercept(c: crease.Crease) -> tuple[int, float]:
  """Returns (family_id, intercept) for fast collinear bucketing."""
  dx = c.p2.x - c.p1.x
  dy = c.p2.y - c.p1.y
  if abs(dy) <= 1e-5:
    return 0, 0.5 * (c.p1.y + c.p2.y)
  if abs(dx) <= 1e-5:
    return 1, 0.5 * (c.p1.x + c.p2.x)
  if abs(dy - dx) <= 1e-5:
    return 2, 0.5 * ((c.p1.y - c.p1.x) + (c.p2.y - c.p2.x))
  if abs(dy + dx) <= 1e-5:
    return 3, 0.5 * ((c.p1.y + c.p1.x) + (c.p2.y + c.p2.x))
  return 4, 0.0


class LineOverlapIndex:
  """Line-bucketed index for O(1) Crease.is_overlapping queries.

  When `grid_size` is provided, line intercepts (y, x, y - x, y + x) are keyed
  on the exact half-grid integer lattice `round(intercept * 2 * grid_size)`,
  with neighbor bucket probing `(key - 1, key, key + 1)` for tolerance safety.
  """

  def __init__(
      self,
      creases: Iterable[crease.Crease] = (),
      grid_size: int | None = None,
  ):
    self._scale: float = (
        float(2 * grid_size)
        if grid_size is not None and grid_size > 0
        else _LINE_INV_CELL
    )
    self._buckets: tuple[dict[int, list[crease.Crease]], ...] = (
        {},
        {},
        {},
        {},
    )
    self._other: list[crease.Crease] = []
    self._all: list[crease.Crease] = []
    for c in creases:
      self.add(c)

  def add(self, c: crease.Crease) -> None:
    """Adds a crease to its line bucket."""
    self._all.append(c)
    fam, intercept = _line_family_and_intercept(c)
    if fam == 4:
      self._other.append(c)
    else:
      key = round(intercept * self._scale)
      self._buckets[fam].setdefault(key, []).append(c)

  def has_overlap(self, target: crease.Crease) -> bool:
    """Returns True if target overlaps any indexed crease."""
    fam, intercept = _line_family_and_intercept(target)
    if fam == 4:
      return any(target.is_overlapping(c) for c in self._all)
    fam_buckets = self._buckets[fam]
    key = round(intercept * self._scale)
    for k in (key - 1, key, key + 1):
      bucket = fam_buckets.get(k)
      if bucket:
        for c in bucket:
          if target.is_overlapping(c):
            return True
    for c in self._other:
      if target.is_overlapping(c):
        return True
    return False


def deduplicate(creases: list[crease.Crease]) -> list[crease.Crease]:
  """Removes duplicate creases while preserving first-seen order."""
  return [c.crease for c in dict.fromkeys(cr.exact_match for cr in creases)]


def get_interesecting_non_overlapping_creases(
    ref_crease: crease.Crease, creases: Sequence[crease.Crease]
) -> list[crease.Crease]:
  """Returns the creases that intersect but not overlap the ref_crease."""
  intersecting_ridges = []
  for crease_ in creases:
    if not ref_crease.intersect_at_tip(crease_):
      continue
    if not ref_crease.is_overlapping(crease_):
      intersecting_ridges.append(crease_)
  return intersecting_ridges


def split_unassigned_and_central_ridges(
    all_ridges: Sequence[crease.Crease],
) -> tuple[list[crease.Crease], list[crease.Crease]]:
  """Splits the creases into unassigned ridges and central ridges (horizontal or vertical)."""
  unassigned_ridges = []
  central_ridges = []
  for r in all_ridges:
    if r.is_horizontal or r.is_vertical:
      central_ridges.append(r)
    else:
      unassigned_ridges.append(r)
  return unassigned_ridges, central_ridges


@functools.lru_cache(maxsize=100)
def create_grid_points(grid_size: int) -> tuple[point.Point, ...]:
  """Creates grid points for the grid."""
  grid_points = []
  for i in range(grid_size + 1):
    for j in range(grid_size + 1):
      grid_points.append(point.Point(x=i / grid_size, y=j / grid_size))
  return tuple(grid_points)


def creases_to_grid(
    creases: Sequence[crease.Crease],
    grid_points: Collection[point.Point],
    grid_size: int,
) -> list[crease.Crease]:
  """Converts a sequence of creases to a list of grid points."""
  atol = 0.5 / grid_size - 1e-8
  if (
      isinstance(grid_points, Sequence)
      and len(grid_points) == (grid_size + 1) ** 2
  ):
    stride = grid_size + 1
    new_creases = []
    for c in creases:
      p1 = c.p1
      ix1 = round(p1.x * grid_size)
      iy1 = round(p1.y * grid_size)
      if 0 <= ix1 <= grid_size and 0 <= iy1 <= grid_size:
        cand1 = grid_points[ix1 * stride + iy1]
        if p1.isclose(cand1, atol=atol):
          p1 = cand1
      p2 = c.p2
      ix2 = round(p2.x * grid_size)
      iy2 = round(p2.y * grid_size)
      if 0 <= ix2 <= grid_size and 0 <= iy2 <= grid_size:
        cand2 = grid_points[ix2 * stride + iy2]
        if p2.isclose(cand2, atol=atol):
          p2 = cand2
      new_creases.append(
          crease.Crease(
              p1,
              p2,
              fold=c.fold,
              fold_percent=c.fold_percent,
              elevation=c.elevation,
          )
      )
    return new_creases

  new_creases = []
  for c in creases:
    new_points = []
    for p in [c.p1, c.p2]:
      found = False
      for grid_point in grid_points:
        if p.isclose(grid_point, atol=atol):
          found = True
          new_points.append(grid_point)
          break
      if not found:
        new_points.append(p)
    new_creases.append(
        crease.Crease(
            new_points[0],
            new_points[1],
            fold=c.fold,
            fold_percent=c.fold_percent,
            elevation=c.elevation,
        )
    )
  return new_creases


def assign_interleave_crease(
    segments: Sequence[crease.Crease], fold: crease.Fold
) -> list[crease.Crease]:
  """Given segments and fold, assign the first segment with fold.

  Then assign the subsequent segments by interleaving mountain and valley.

  Args:
    segments: The segments to assign folds to.
    fold: The fold to assign to the first segment.

  Returns:
    A list of assigned segments.
  """
  assigned_segments = []
  for h in segments:
    new_segment = crease.Crease(h.p1, h.p2, fold=fold)
    assigned_segments.append(new_segment)
    fold = fold.flip()
  return assigned_segments


def assign_symmetric_crease(
    segments: Sequence[crease.Crease], fold: crease.Fold
) -> list[crease.Crease]:
  """Assigns folds to creases symmetrically.

  Args:
    segments: The segments to assign folds to.
    fold: The fold to assign to the first segment.

  Returns:
    A list of assigned segments and a list of new pleats.
  """
  if len(segments) % 2 != 0:
    raise ValueError(
        f'Expected an even number of symmetric segments, got {len(segments)}'
    )
  assigned_segments = []
  # Interleave at each intersection but the middle one
  for i, h in enumerate(segments):
    new_segment = crease.Crease(h.p1, h.p2, fold=fold)
    assigned_segments.append(new_segment)
    if i != len(segments) // 2 - 1:
      fold = fold.flip()
  return assigned_segments


@dataclasses.dataclass(frozen=True, eq=True)
class HingeAssignmentType:
  """Holds the type of hinge assignment.

  Attributes:
    fold: The fold to assign to the first segment. (ordered by norm but reversed
      if it intersects the border)
    symmetric: Whether to assign folds symmetrically.
  """

  fold: crease.Fold
  symmetric: bool

  def __post_init__(self):
    if self.fold not in (crease.Fold.MOUNTAIN, crease.Fold.VALLEY):
      raise ValueError(f'Unsupported hinge assignment type: {self.fold}')

  def assign_segments(
      self, segments: Sequence[crease.Crease]
  ) -> list[crease.Crease]:
    """Assigns folds to crease segments of hinges or central ridges.

    Args:
      segments: The segments of crease c.

    Returns:
      A list of assigned segments, a list of new pleats, and a list of pleats.
    """
    if segments[0].intersects_border:
      segments = segments[::-1]

    if self.symmetric:
      return assign_symmetric_crease(segments, self.fold)
    else:
      return assign_interleave_crease(segments, self.fold)


def creases_to_paths(creases: list[crease.Crease]) -> list[list[crease.Crease]]:
  """Converts a sequence of creases to a list of paths.

  A path is a sequence of creases that are connected to each other via tips.
  Args:
    creases: The creases to convert to paths.

  Returns:
    list[list[crease.Crease]]
  """
  n = len(creases)
  connectivity_graph: dict[int, list[int]] = {i: [] for i in range(n)}
  for i in range(n):
    c1 = creases[i]
    for j in range(i + 1, n):
      if c1.intersect_at_tip(creases[j]):
        connectivity_graph[i].append(j)
        connectivity_graph[j].append(i)

  components_indices = find_connected_components(connectivity_graph)
  return [
      [creases[idx] for idx in component] for component in components_indices
  ]


def find_connected_components(
    graph: Mapping[int, Collection[int]],
) -> list[list[int]]:
  """Finds all connected components in a graph.

  This function uses a traversal-based approach (iterative DFS) to discover
  all groups of connected nodes. A connected component is a subgraph where
  every node is reachable from every other node in that subgraph.

  Args:
    graph: A dictionary representing the graph as an adjacency list.

  Returns:
    A list of lists, where each inner list contains the nodes of a single
    connected component.
  """
  visited = set()
  components = []
  # Iterate through each node in the graph
  for node in graph:
    if node not in visited:
      # Start a new traversal if the node hasn't been visited
      component = []
      stack = [node]
      visited.add(node)
      while stack:
        current_node = stack.pop()
        component.append(current_node)
        # Check for neighbors
        for neighbor in graph[current_node]:
          if neighbor not in visited:
            visited.add(neighbor)
            stack.append(neighbor)
      # Add the discovered component to the list
      components.append(component)
  return components
