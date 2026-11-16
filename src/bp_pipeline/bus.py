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

"""Immutable pipeline bus representations for Box Pleating origami."""

from collections.abc import Mapping, Sequence
import dataclasses
import enum
import functools
import math
import typing
from typing import Any, Literal
from bp_pipeline import rectangle as rectangle_lib
from bp_pipeline.crease_pattern import crease
from bp_pipeline.crease_pattern import crease_pattern as crease_pattern_lib
from bp_pipeline.crease_pattern import point
from bp_pipeline.crease_pattern import symmetry
from bp_pipeline.crease_pattern.parser import cp_format
from bp_pipeline.crease_pattern.parser import svg as svg_parser
from bp_pipeline.packing import config as packing_config_lib
from bp_pipeline.packing import packing_output as packing_output_lib
from bp_pipeline.packing import river as river_lib
from bp_pipeline.packing import types as packing_types
from bp_pipeline.third_party.flat_folder import fold_cp
from bp_pipeline.third_party.flat_folder import folded_state
from bp_pipeline.third_party.flat_folder import graph_creation
from bp_pipeline.third_party.flat_folder import graph_topology
from bp_pipeline.stick_figures import stick as stick_lib
from bp_pipeline.stick_figures import stick_figure as stick_figure_lib
import immutabledict

if typing.TYPE_CHECKING:
  from bp_pipeline.solving import hinges as hinge_utils


class BaseType(enum.Enum):
  NORMAL = 'normal'
  LS_TEMPLATE = 'ls_template'
  LEVEL_SHIFTED = 'level_shifted'


FlapType = rectangle_lib.FlapType
FlapTipShape = stick_lib.FlapTipShape
RiverKind = Literal['straight', 'l_shaped', 'hugging']


def _require_int(value: int | float, field_name: str) -> int:
  rounded = round(float(value))
  if not math.isclose(float(value), rounded, abs_tol=1e-9):
    raise ValueError(
        f'Grid coordinate {field_name}={value!r} must be an integer.'
    )
  return int(rounded)


@dataclasses.dataclass(frozen=True)
class GridPoint:
  """An immutable point with integer grid coordinates."""

  x: int
  y: int

  def __post_init__(self):
    object.__setattr__(self, 'x', int(self.x))
    object.__setattr__(self, 'y', int(self.y))

  def scaled(self, k: int) -> 'GridPoint':
    return GridPoint(x=self.x * k, y=self.y * k)

  def to_normalized_point(
      self, grid_size: int, *, require_on_paper: bool = True
  ) -> point.Point:
    return point.Point(
        x=self.x / grid_size,
        y=self.y / grid_size,
        require_on_paper=require_on_paper,
    )

  @classmethod
  def from_normalized_point(
      cls, pt: point.Point, grid_size: int
  ) -> 'GridPoint':
    return cls(
        x=_require_int(pt.x * grid_size, 'x'),
        y=_require_int(pt.y * grid_size, 'y'),
    )


@dataclasses.dataclass(frozen=True)
class GridSegment:
  """An immutable segment connecting two grid points."""

  p1: GridPoint
  p2: GridPoint

  def canonical(self) -> 'GridSegment':
    if (self.p2.x, self.p2.y) < (self.p1.x, self.p1.y):
      return GridSegment(p1=self.p2, p2=self.p1)
    return self

  def scaled(self, k: int) -> 'GridSegment':
    return GridSegment(p1=self.p1.scaled(k), p2=self.p2.scaled(k))

  def to_normalized_crease(
      self,
      grid_size: int,
      *,
      fold: crease.Fold = crease.Fold.HINGE,
      require_on_paper: bool = True,
  ) -> crease.Crease:
    return crease.Crease(
        self.p1.to_normalized_point(
            grid_size, require_on_paper=require_on_paper
        ),
        self.p2.to_normalized_point(
            grid_size, require_on_paper=require_on_paper
        ),
        fold=fold,
    )

  @classmethod
  def from_normalized_crease(
      cls, c: crease.Crease, grid_size: int
  ) -> 'GridSegment':
    seg = cls(
        p1=GridPoint.from_normalized_point(c.p1, grid_size),
        p2=GridPoint.from_normalized_point(c.p2, grid_size),
    )
    return seg.canonical()


@dataclasses.dataclass(frozen=True, kw_only=True)
class GridRect:
  """An immutable axis-aligned rectangle on the integer grid."""

  x: int
  y: int
  width: int
  height: int

  def __post_init__(self):
    object.__setattr__(self, 'x', int(self.x))
    object.__setattr__(self, 'y', int(self.y))
    object.__setattr__(self, 'width', int(self.width))
    object.__setattr__(self, 'height', int(self.height))

  @property
  def w(self) -> int:
    return self.width

  @property
  def h(self) -> int:
    return self.height

  def scaled(self, k: int) -> 'GridRect':
    return GridRect(
        x=self.x * k,
        y=self.y * k,
        width=self.width * k,
        height=self.height * k,
    )

  def clipped(self, grid_size: int) -> 'GridRect':
    x0 = max(0, min(grid_size, self.x))
    y0 = max(0, min(grid_size, self.y))
    x1 = max(0, min(grid_size, self.x + self.width))
    y1 = max(0, min(grid_size, self.y + self.height))
    return GridRect(x=x0, y=y0, width=x1 - x0, height=y1 - y0)

  @classmethod
  def from_rectangle(
      cls,
      rect: rectangle_lib.Rectangle,
      grid_size: int | None = None,
  ) -> 'GridRect':
    """Constructs a GridRect from a Rectangle object."""
    raw = cls(
        x=_require_int(rect.x_start, 'x_start'),
        y=_require_int(rect.y_start, 'y_start'),
        width=_require_int(rect.x_len, 'x_len'),
        height=_require_int(rect.y_len, 'y_len'),
    )
    if grid_size is not None:
      return raw.clipped(grid_size)
    return raw

  def to_rectangle(
      self,
      name: str = '',
      circle_centers: Sequence[point.Point] = (),
  ) -> rectangle_lib.Rectangle:
    return rectangle_lib.Rectangle(
        name=name,
        x_start=self.x,
        y_start=self.y,
        x_len=self.width,
        y_len=self.height,
        circle_centers=list(circle_centers) if circle_centers else None,
    )

  def normalized(self, grid_size: int) -> rectangle_lib.Rectangle:
    return self.to_rectangle().normalize_rectangle(grid_size)

  def flap_type(self, grid_size: int) -> FlapType:
    return self.to_rectangle().flap_type(grid_size)


def coalesce_grid_rects(rects: Sequence[GridRect]) -> tuple[GridRect, ...]:
  """Merges contiguous horizontal and vertical GridRects into maximal blocks."""
  if len(rects) <= 1:
    return tuple(rects)
  by_row = sorted(
      (r for r in rects if r.width > 0 and r.height > 0),
      key=lambda r: (r.y, r.height, r.x),
  )
  horiz: list[GridRect] = []
  for r in by_row:
    if (
        horiz
        and horiz[-1].y == r.y
        and horiz[-1].height == r.height
        and horiz[-1].x + horiz[-1].width == r.x
    ):
      prev = horiz[-1]
      horiz[-1] = GridRect(
          x=prev.x, y=prev.y, width=prev.width + r.width, height=prev.height
      )
    else:
      horiz.append(r)
  by_col = sorted(horiz, key=lambda r: (r.x, r.width, r.y))
  vert: list[GridRect] = []
  for r in by_col:
    if (
        vert
        and vert[-1].x == r.x
        and vert[-1].width == r.width
        and vert[-1].y + vert[-1].height == r.y
    ):
      prev = vert[-1]
      vert[-1] = GridRect(
          x=prev.x, y=prev.y, width=prev.width, height=prev.height + r.height
      )
    else:
      vert.append(r)
  return tuple(vert)


@dataclasses.dataclass(frozen=True, kw_only=True)
class DesignTree:
  """Canonical tree ready for packing, with source provenance preserved.

  Attributes:
    source: Unmerged input StickFigure as authored.
    merged: Linear-chain-merged StickFigure computed once via
      `source.merge_linear_chains()`.
    merged_from: Mapping from each merged stick name to the tuple of source
      stick names it was merged from.
    symmetric_partner: Mapping from each stick name in `merged` to its mirrored
      partner stick name (sticks on the symmetry axis map to themselves).
    is_symmetric: Whether `merged` is bilaterally symmetric.
    min_grid_size: Minimum packing grid size (`merged.get_packing_gridsize()`).
    max_grid_size: Maximum packing grid size
      (`calculate_max_grid_size(merged)`).
  """

  source: stick_figure_lib.StickFigure
  merged: stick_figure_lib.StickFigure
  merged_from: Mapping[str, tuple[str, ...]]
  symmetric_partner: Mapping[str, str]
  is_symmetric: bool
  min_grid_size: int
  max_grid_size: int

  @classmethod
  def from_stick_figure(
      cls,
      sf: stick_figure_lib.StickFigure,
  ) -> 'DesignTree':
    """Constructs a DesignTree from a StickFigure."""
    merged, raw_map = sf.merge_linear_chains()
    merged_from = immutabledict.immutabledict(
        {s.name: tuple(raw_map.get(s.name, (s.name,))) for s in merged.children}
    )
    partners: dict[str, str] = {}
    if merged.is_symmetric_tree:
      for s in merged.children:
        if s.symmetric_stick_name is not None:
          partners[s.name] = s.symmetric_stick_name
        elif s.is_on_symmetry_axis:
          partners[s.name] = s.name
    symmetric_partner = immutabledict.immutabledict(partners)

    min_grid_size = merged.get_packing_gridsize()
    max_grid_size = stick_figure_lib.calculate_max_grid_size(merged)

    return cls(
        source=sf,
        merged=merged,
        merged_from=merged_from,
        symmetric_partner=symmetric_partner,
        is_symmetric=merged.is_symmetric_tree,
        min_grid_size=min_grid_size,
        max_grid_size=max_grid_size,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class Flap:
  """A packed flap on the integer box-pleating grid."""

  id: str
  stick: str
  rect: GridRect
  circle_centers: tuple[GridPoint, ...]
  flap_type: FlapType
  tip_shape: FlapTipShape = FlapTipShape.UNSET
  mirror_of: str | None = None

  def to_rectangle(self) -> rectangle_lib.Rectangle:
    centers = [
        point.Point(x=float(c.x), y=float(c.y), require_on_paper=False)
        for c in self.circle_centers
    ]
    return self.rect.to_rectangle(name=self.id, circle_centers=centers)


@dataclasses.dataclass(frozen=True, kw_only=True)
class River:
  """A packed river on the integer box-pleating grid."""

  id: str
  stick: str
  width: int
  kind: RiverKind
  rects: tuple[GridRect, ...]
  mirror_of: str | None = None
  offset: int | None = None
  orientation: river_lib.RiverOrientation | None = None
  l_shape_type: river_lib.LShapeType | None = None
  turn: GridPoint | None = None
  paths: tuple[tuple[GridPoint, ...], ...] = ()
  hug_side: river_lib.HugSide | None = None
  is_on_axis: bool = False
  variant_type: river_lib.PathVariantType = river_lib.PathVariantType.STANDARD
  requires_recovery: bool = False

  def __post_init__(self):
    object.__setattr__(self, 'width', int(self.width))
    if self.offset is not None:
      object.__setattr__(self, 'offset', int(self.offset))

  @classmethod
  def from_packing_river(
      cls,
      river_id: str,
      pr: river_lib.PackingRiver,
      *,
      mirror_of: str | None = None,
  ) -> 'River':
    """Constructs a River bus model from a PackingRiver instance."""
    stick_name = (
        river_id[: -len(packing_types.COPY_SUFFIX)]
        if river_id.endswith(packing_types.COPY_SUFFIX)
        else river_id
    )
    rects = tuple(
        GridRect.from_rectangle(r, grid_size=pr.grid_size)
        for r in pr.rectangles
    )
    match pr:
      case river_lib.River():
        return cls(
            id=river_id,
            stick=stick_name,
            width=pr.width,
            kind='straight',
            rects=rects,
            mirror_of=mirror_of,
            offset=pr.offset,
            orientation=pr.orientation,
        )
      case river_lib.LShapedRiver():
        return cls(
            id=river_id,
            stick=stick_name,
            width=pr.width,
            kind='l_shaped',
            rects=rects,
            mirror_of=mirror_of,
            l_shape_type=pr.l_shape_type,
            turn=GridPoint(x=pr.turn_x, y=pr.turn_y),
        )
      case river_lib.HuggingRiver():
        paths = tuple(
            tuple(GridPoint(x=int(col), y=int(row)) for row, col in path)
            for path in pr.subriver_paths
        )
        return cls(
            id=river_id,
            stick=stick_name,
            width=pr.width,
            kind='hugging',
            rects=coalesce_grid_rects(rects),
            mirror_of=mirror_of,
            paths=paths,
            hug_side=pr.hug_side,
            is_on_axis=pr.is_on_axis,
            variant_type=pr.variant_type,
        )
      case _ if hasattr(pr, 'rectangles'):
        req_recovery = bool(getattr(pr, 'requires_recovery', False))
        if len(rects) == 1:
          r = rects[0]
          if r.width == pr.width and r.height != pr.width:
            orientation = river_lib.RiverOrientation.VERTICAL
            offset = r.x
          else:
            orientation = river_lib.RiverOrientation.HORIZONTAL
            offset = r.y
          return cls(
              id=river_id,
              stick=stick_name,
              width=pr.width,
              kind='straight',
              rects=rects,
              mirror_of=mirror_of,
              offset=offset,
              orientation=orientation,
              requires_recovery=req_recovery,
          )
        return cls(
            id=river_id,
            stick=stick_name,
            width=pr.width,
            kind='l_shaped',
            rects=rects,
            mirror_of=mirror_of,
            requires_recovery=req_recovery,
        )
      case _ as unreachable:
        typing.assert_never(unreachable)

  def to_packing_river(self, grid_size: int) -> river_lib.PackingRiver:
    """Reconstructs a PackingRiver from this River model."""
    match self.kind:
      case 'straight':
        if self.offset is not None and self.orientation is not None:
          return river_lib.River(
              width=self.width,
              offset=self.offset,
              orientation=self.orientation,
              grid_size=grid_size,
          )
      case 'l_shaped':
        if self.l_shape_type is not None and self.turn is not None:
          return river_lib.LShapedRiver(
              width=self.width,
              grid_size=grid_size,
              l_shape_type=self.l_shape_type,
              turn_x=self.turn.x,
              turn_y=self.turn.y,
          )
      case 'hugging':
        if self.paths and self.hug_side is not None:
          return river_lib.HuggingRiver(
              grid_size=grid_size,
              width=self.width,
              subriver_paths=[
                  [(pt.y, pt.x) for pt in path] for path in self.paths
              ],
              hug_side=self.hug_side,
              is_on_axis=self.is_on_axis,
              variant_type=self.variant_type,
          )
      case _ as unreachable:
        typing.assert_never(unreachable)
    factory = river_lib.get_reconstructed_river_factory()
    if factory is not None:
      return factory(
          rectangles=[r.to_rectangle() for r in self.rects],
          width=self.width,
          grid_size=grid_size,
      )
    raise ValueError(
        f'Cannot reconstruct PackingRiver for kind={self.kind!r} without an '
        'extended river factory registered.'
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class Hinge:
  """A hinge segment on the integer box-pleating grid."""

  id: int
  segment: GridSegment
  elements: tuple[str, ...]
  assigned: bool = False

  def __post_init__(self):
    object.__setattr__(self, 'id', int(self.id))

  def to_hinge_info(self, grid_size: int) -> packing_types.HingeInfo:
    return packing_types.HingeInfo(
        coordinate=self.segment.to_normalized_crease(
            grid_size, require_on_paper=False
        ),
        assigned=self.assigned,
        flap_names=typing.cast(tuple[str, str], self.elements),
    )


def filter_and_merge_hinges(
    hinges: Sequence[Hinge],
    cp: crease_pattern_lib.CreasePattern,
    grid_size: int,
) -> tuple[Hinge, ...]:
  """Filters out internal unit-cell edges and merges collinear contiguous hinges."""
  if not hinges:
    return ()
  cp_hinge_segs = [
      GridSegment.from_normalized_crease(c, grid_size)
      for c in cp.creases
      if c.fold == crease.Fold.HINGE
  ]
  vert_cp: dict[int, list[tuple[int, int]]] = {}
  horiz_cp: dict[int, list[tuple[int, int]]] = {}
  for seg in cp_hinge_segs:
    if seg.p1.x == seg.p2.x:
      vert_cp.setdefault(seg.p1.x, []).append(
          (min(seg.p1.y, seg.p2.y), max(seg.p1.y, seg.p2.y))
      )
    elif seg.p1.y == seg.p2.y:
      horiz_cp.setdefault(seg.p1.y, []).append(
          (min(seg.p1.x, seg.p2.x), max(seg.p1.x, seg.p2.x))
      )

  def _on_cp_hinge(seg: GridSegment) -> bool:
    if not cp_hinge_segs:
      return True
    if seg.p1.x == seg.p2.x:
      y0, y1 = min(seg.p1.y, seg.p2.y), max(seg.p1.y, seg.p2.y)
      return any(
          max(y0, cy0) < min(y1, cy1) for cy0, cy1 in vert_cp.get(seg.p1.x, ())
      )
    if seg.p1.y == seg.p2.y:
      x0, x1 = min(seg.p1.x, seg.p2.x), max(seg.p1.x, seg.p2.x)
      return any(
          max(x0, cx0) < min(x1, cx1) for cx0, cx1 in horiz_cp.get(seg.p1.y, ())
      )
    return True

  filtered = [h for h in hinges if _on_cp_hinge(h.segment.canonical())]

  horiz_groups: dict[
      tuple[int, tuple[str, ...], bool], list[tuple[int, int]]
  ] = {}
  vert_groups: dict[
      tuple[int, tuple[str, ...], bool], list[tuple[int, int]]
  ] = {}
  other_hinges: list[Hinge] = []
  for h in filtered:
    seg = h.segment.canonical()
    elems = tuple(dict.fromkeys(h.elements))
    if seg.p1.y == seg.p2.y and seg.p1.x != seg.p2.x:
      horiz_groups.setdefault((seg.p1.y, elems, h.assigned), []).append(
          (seg.p1.x, seg.p2.x)
      )
    elif seg.p1.x == seg.p2.x and seg.p1.y != seg.p2.y:
      vert_groups.setdefault((seg.p1.x, elems, h.assigned), []).append(
          (seg.p1.y, seg.p2.y)
      )
    else:
      other_hinges.append(
          Hinge(id=0, segment=seg, elements=elems, assigned=h.assigned)
      )

  merged_list: list[Hinge] = []
  for (y, elems, assigned), intervals in sorted(horiz_groups.items()):
    intervals.sort()
    cur_start, cur_end = intervals[0]
    for nxt_start, nxt_end in intervals[1:]:
      if nxt_start <= cur_end:
        cur_end = max(cur_end, nxt_end)
      else:
        merged_list.append(
            Hinge(
                id=0,
                segment=GridSegment(
                    p1=GridPoint(x=cur_start, y=y),
                    p2=GridPoint(x=cur_end, y=y),
                ),
                elements=elems,
                assigned=assigned,
            )
        )
        cur_start, cur_end = nxt_start, nxt_end
    merged_list.append(
        Hinge(
            id=0,
            segment=GridSegment(
                p1=GridPoint(x=cur_start, y=y),
                p2=GridPoint(x=cur_end, y=y),
            ),
            elements=elems,
            assigned=assigned,
        )
    )

  for (x, elems, assigned), intervals in sorted(vert_groups.items()):
    intervals.sort()
    cur_start, cur_end = intervals[0]
    for nxt_start, nxt_end in intervals[1:]:
      if nxt_start <= cur_end:
        cur_end = max(cur_end, nxt_end)
      else:
        merged_list.append(
            Hinge(
                id=0,
                segment=GridSegment(
                    p1=GridPoint(x=x, y=cur_start),
                    p2=GridPoint(x=x, y=cur_end),
                ),
                elements=elems,
                assigned=assigned,
            )
        )
        cur_start, cur_end = nxt_start, nxt_end
    merged_list.append(
        Hinge(
            id=0,
            segment=GridSegment(
                p1=GridPoint(x=x, y=cur_start),
                p2=GridPoint(x=x, y=cur_end),
            ),
            elements=elems,
            assigned=assigned,
        )
    )

  merged_list.extend(other_hinges)
  return tuple(
      dataclasses.replace(h, id=idx) for idx, h in enumerate(merged_list)
  )


class HingeIndex:
  """Spatial and ID index over packed `Hinge` segments."""

  def __init__(self, hinges: Sequence[Hinge], grid_size: int):
    self._hinges = tuple(hinges)
    self._grid_size = grid_size
    self._by_id: dict[int, Hinge] = {h.id: h for h in self._hinges}
    self._by_segment: dict[GridSegment, Hinge] = {}
    self._vertical_by_x: dict[int, list[Hinge]] = {}
    self._horizontal_by_y: dict[int, list[Hinge]] = {}
    for h in self._hinges:
      seg = h.segment.canonical()
      self._by_segment.setdefault(seg, h)
      if seg.p1.x == seg.p2.x:
        self._vertical_by_x.setdefault(seg.p1.x, []).append(h)
      elif seg.p1.y == seg.p2.y:
        self._horizontal_by_y.setdefault(seg.p1.y, []).append(h)

  def __getitem__(self, key: int | GridSegment) -> Hinge:
    if isinstance(key, int):
      return self._by_id[key]
    return self._by_segment[key.canonical()]

  def __contains__(self, key: int | GridSegment) -> bool:
    if isinstance(key, int):
      return key in self._by_id
    if isinstance(key, GridSegment):
      return key.canonical() in self._by_segment
    return False

  def find_overlapping(self, target: crease.Crease) -> Hinge | None:
    """Returns the first `Hinge` overlapping `target` in unit-square space."""
    if target.is_vertical:
      x_coord = round(target.p1.x * self._grid_size)
      if math.isclose(target.p1.x * self._grid_size, x_coord, abs_tol=1e-6):
        y_lo = min(target.p1.y, target.p2.y) * self._grid_size
        y_hi = max(target.p1.y, target.p2.y) * self._grid_size
        for h in self._vertical_by_x.get(x_coord, ()):
          seg = h.segment.canonical()
          if max(seg.p1.y, y_lo) < min(seg.p2.y, y_hi) - 1e-6:
            return h
    elif target.is_horizontal:
      y_coord = round(target.p1.y * self._grid_size)
      if math.isclose(target.p1.y * self._grid_size, y_coord, abs_tol=1e-6):
        x_lo = min(target.p1.x, target.p2.x) * self._grid_size
        x_hi = max(target.p1.x, target.p2.x) * self._grid_size
        for h in self._horizontal_by_y.get(y_coord, ()):
          seg = h.segment.canonical()
          if max(seg.p1.x, x_lo) < min(seg.p2.x, x_hi) - 1e-6:
            return h
    return None


@dataclasses.dataclass(frozen=True, kw_only=True)
class PackingDiagnostics:
  """Typed diagnostics from packing search."""

  grid_size: int
  grid_height: int | None = None
  is_symmetric: bool = False
  perfect_tiling_found: bool = True
  unused_area: int = 0
  num_delta_combinations: int | None = None
  reason: str | None = None
  suggestion: str | None = None
  constraint_satisfaction: str | None = None
  satisfied_constraints: tuple[str, ...] = ()
  unsatisfied_constraints: tuple[str, ...] = ()

  def __post_init__(self):
    object.__setattr__(self, 'grid_size', int(self.grid_size))
    if self.grid_height is not None:
      object.__setattr__(self, 'grid_height', int(self.grid_height))
    object.__setattr__(self, 'unused_area', int(self.unused_area))
    if self.num_delta_combinations is not None:
      object.__setattr__(
          self, 'num_delta_combinations', int(self.num_delta_combinations)
      )

  @classmethod
  def from_tiling_details(
      cls,
      details: Mapping[str, Any] | None,
      *,
      default_grid_size: int,
  ) -> 'PackingDiagnostics':
    if details is None:
      return cls(grid_size=default_grid_size)
    return cls(
        grid_size=int(details.get('grid_size', default_grid_size)),
        grid_height=(
            int(details['grid_height'])
            if details.get('grid_height') is not None
            else None
        ),
        is_symmetric=bool(details.get('is_symmetric', False)),
        perfect_tiling_found=bool(details.get('perfect_tiling_found', True)),
        unused_area=int(details.get('unused_area', 0)),
        num_delta_combinations=(
            int(details['num_delta_combinations'])
            if details.get('num_delta_combinations') is not None
            else None
        ),
        reason=details.get('reason'),
        suggestion=details.get('suggestion'),
        constraint_satisfaction=details.get('constraint_satisfaction'),
        satisfied_constraints=tuple(details.get('satisfied_constraints', ())),
        unsatisfied_constraints=tuple(
            details.get('unsatisfied_constraints', ())
        ),
    )

  def to_tiling_details(self) -> dict[str, Any]:
    """Converts diagnostics to a dictionary of tiling details."""
    out: dict[str, Any] = {
        'grid_size': self.grid_size,
        'is_symmetric': self.is_symmetric,
        'perfect_tiling_found': self.perfect_tiling_found,
        'unused_area': self.unused_area,
    }
    if self.grid_height is not None:
      out['grid_height'] = self.grid_height
    if self.num_delta_combinations is not None:
      out['num_delta_combinations'] = self.num_delta_combinations
    if self.reason is not None:
      out['reason'] = self.reason
    if self.suggestion is not None:
      out['suggestion'] = self.suggestion
    if self.constraint_satisfaction is not None:
      out['constraint_satisfaction'] = self.constraint_satisfaction
      out['satisfied_constraints'] = list(self.satisfied_constraints)
      out['unsatisfied_constraints'] = list(self.unsatisfied_constraints)
    return out


def _render_merged_svg(
    cp: crease_pattern_lib.CreasePattern,
    grid_size: int,
    *,
    pretty: bool = True,
    colors: Mapping[crease.Fold, str] | None = None,
    stroke_width: float | int | str | None = None,
    dashed_valley: bool = False,
    include_grid: bool = False,
) -> str:
  """Merges collinear creases and renders `cp` to an SVG string."""
  merged = crease.merge_creases(cp.creases)
  merged_cp = crease_pattern_lib.CreasePattern(merged, light_mode=True)
  effective_stroke_width = (
      stroke_width
      if stroke_width is not None
      else round(max(1, grid_size) / 450.0, 4)
  )
  return svg_parser.to_svg(
      merged_cp,
      grid=grid_size,
      pretty=pretty,
      replace_border=True,
      ndigits=8,
      colors=colors,
      stroke_width=effective_stroke_width,
      dashed_valley=dashed_valley,
      grid_lines=include_grid,
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class PackedCreasePattern:
  """Immutable stage record produced by packing."""

  design: DesignTree | None
  packed_tree: stick_figure_lib.StickFigure | None
  grid_size: int
  creases: crease_pattern_lib.CreasePattern
  flaps: Mapping[str, Flap]
  rivers: Mapping[str, River]
  hinges: tuple[Hinge, ...]
  config: packing_config_lib.PackingConfig
  diagnostics: PackingDiagnostics

  @functools.cached_property
  def hinge_at(self) -> HingeIndex:
    return HingeIndex(self.hinges, self.grid_size)

  @classmethod
  def from_packing_output(
      cls,
      output: packing_output_lib.PackingOutput,
      *,
      design: DesignTree | None,
      grid_size: int | None = None,
  ) -> 'PackedCreasePattern':
    """Creates a PackedCreasePattern from a legacy PackingOutput."""
    if not output.success or output.packing_cp is None:
      raise ValueError(
          'Cannot create PackedCreasePattern from an unsuccessful'
          f' PackingOutput: {output.error_message}'
      )
    cp = output.packing_cp
    resolved_grid_size = grid_size if grid_size is not None else cp.grid_size
    packed_tree = output.stick_figure

    rect_list = output.flap_rectangles or []
    all_flap_ids = {r.name for r in rect_list}
    packed_sticks_by_name = (
        {s.name: s for s in packed_tree.children}
        if packed_tree is not None
        else {}
    )
    flaps: dict[str, Flap] = {}
    for rect in rect_list:
      flap_id = rect.name
      stick_name = (
          flap_id[: -len(packing_types.COPY_SUFFIX)]
          if flap_id.endswith(packing_types.COPY_SUFFIX)
          else flap_id
      )
      grid_rect = GridRect.from_rectangle(rect, grid_size=resolved_grid_size)
      raw_centers = rect.circle_centers or rect.get_circle_centers(
          resolved_grid_size
      )
      circle_centers = tuple(
          GridPoint(
              x=_require_int(c.x, 'center.x'),
              y=_require_int(c.y, 'center.y'),
          )
          for c in raw_centers
      )
      flap_type = grid_rect.flap_type(resolved_grid_size)
      tip_shape = FlapTipShape.UNSET
      if stick_name in packed_sticks_by_name:
        tip_shape = packed_sticks_by_name[stick_name].flap_tip_shape
      mirror_of: str | None = None
      if flap_id.endswith(packing_types.COPY_SUFFIX):
        mirror_of = stick_name
      elif f'{flap_id}{packing_types.COPY_SUFFIX}' in all_flap_ids:
        mirror_of = f'{flap_id}{packing_types.COPY_SUFFIX}'
      elif (
          design is not None
          and stick_name in design.symmetric_partner
          and design.symmetric_partner[stick_name] != stick_name
      ):
        mirror_of = design.symmetric_partner[stick_name]

      flaps[flap_id] = Flap(
          id=flap_id,
          stick=stick_name,
          rect=grid_rect,
          circle_centers=circle_centers,
          flap_type=flap_type,
          tip_shape=tip_shape,
          mirror_of=mirror_of,
      )

    raw_rivers = output.packing_rivers or {}
    rivers: dict[str, River] = {}
    for river_id, pr in raw_rivers.items():
      stick_name = (
          river_id[: -len(packing_types.COPY_SUFFIX)]
          if river_id.endswith(packing_types.COPY_SUFFIX)
          else river_id
      )
      mirror_of = None
      if river_id.endswith(packing_types.COPY_SUFFIX):
        mirror_of = stick_name
      elif f'{river_id}{packing_types.COPY_SUFFIX}' in raw_rivers:
        mirror_of = f'{river_id}{packing_types.COPY_SUFFIX}'
      elif (
          design is not None
          and stick_name in design.symmetric_partner
          and design.symmetric_partner[stick_name] != stick_name
      ):
        mirror_of = design.symmetric_partner[stick_name]
      rivers[river_id] = River.from_packing_river(
          river_id, pr, mirror_of=mirror_of
      )

    raw_hinges: list[Hinge] = []
    for idx, h_info in enumerate(output.hinges or ()):
      if h_info.coordinate is None:
        continue
      raw_hinges.append(
          Hinge(
              id=idx,
              segment=GridSegment.from_normalized_crease(
                  h_info.coordinate, resolved_grid_size
              ),
              elements=tuple(h_info.flap_names),
              assigned=h_info.assigned,
          )
      )
    hinges = filter_and_merge_hinges(raw_hinges, cp, resolved_grid_size)

    return cls(
        design=design,
        packed_tree=packed_tree,
        grid_size=resolved_grid_size,
        creases=cp,
        flaps=immutabledict.immutabledict(flaps),
        rivers=immutabledict.immutabledict(rivers),
        hinges=hinges,
        config=output.config or packing_config_lib.PackingConfig(),
        diagnostics=PackingDiagnostics.from_tiling_details(
            output.details, default_grid_size=resolved_grid_size
        ),
    )

  def to_packing_output(self) -> packing_output_lib.PackingOutput:
    unmerged_sf = (
        self.design.source if self.design is not None else self.packed_tree
    )
    return packing_output_lib.PackingOutput(
        success=True,
        packing_cp=self.creases,
        flap_rectangles=[f.to_rectangle() for f in self.flaps.values()],
        packing_rivers={
            r_id: r.to_packing_river(self.grid_size)
            for r_id, r in self.rivers.items()
        },
        stick_figure=self.packed_tree,
        unmerged_stick_figure=unmerged_sf,
        hinges=[h.to_hinge_info(self.grid_size) for h in self.hinges],
        sacrificial_flaps=None,
        tracker=None,
        details=typing.cast(Any, self.diagnostics.to_tiling_details()),
        config=self.config,
        error_message=None,
    )

  @classmethod
  def from_crease_pattern(
      cls,
      cp: crease_pattern_lib.CreasePattern,
      grid_size: int | None = None,
  ) -> 'PackedCreasePattern':
    """Standalone-CP mode without stick figure metadata."""
    resolved_grid_size = grid_size if grid_size is not None else cp.grid_size
    if resolved_grid_size is None or resolved_grid_size <= 0:
      raise ValueError(
          'grid_size must be positive when constructing PackedCreasePattern'
          ' from a standalone CreasePattern.'
      )
    hinges = tuple(
        Hinge(
            id=idx,
            segment=GridSegment.from_normalized_crease(c, resolved_grid_size),
            elements=(),
            assigned=False,
        )
        for idx, c in enumerate(cp.creases)
        if c.fold == crease.Fold.HINGE
    )
    return cls(
        design=None,
        packed_tree=None,
        grid_size=resolved_grid_size,
        creases=cp,
        flaps=immutabledict.immutabledict(),
        rivers=immutabledict.immutabledict(),
        hinges=hinges,
        config=packing_config_lib.PackingConfig(),
        diagnostics=PackingDiagnostics(grid_size=resolved_grid_size),
    )

  def flap_for_stick(
      self,
      stick_name: str,
      design: DesignTree | None = None,
  ) -> Flap | None:
    """Returns the packed `Flap` corresponding to `stick_name`, if any."""
    active_design = design if design is not None else self.design
    candidates: list[str] = [stick_name]
    if active_design is not None:
      for merged_name, source_chain in active_design.merged_from.items():
        if stick_name in source_chain and merged_name not in candidates:
          candidates.append(merged_name)
      for cand in list(candidates):
        if cand in active_design.symmetric_partner:
          partner = active_design.symmetric_partner[cand]
          if partner not in candidates:
            candidates.append(partner)
    for cand in candidates:
      if cand in self.flaps:
        return self.flaps[cand]
    for flap in self.flaps.values():
      if flap.stick in candidates or flap.mirror_of in candidates:
        return flap
    return None

  def to_svg(
      self,
      *,
      pretty: bool = True,
      colors: Mapping[crease.Fold, str] | None = None,
      stroke_width: float | int | str | None = None,
      dashed_valley: bool = False,
      include_grid: bool = False,
  ) -> str:
    return _render_merged_svg(
        self.creases,
        self.grid_size,
        pretty=pretty,
        colors=colors,
        stroke_width=stroke_width,
        dashed_valley=dashed_valley,
        include_grid=include_grid,
    )


@enum.unique
class SolveStage(enum.Enum):
  """Stages of crease pattern resolution during hinge solving."""

  PACKING_CP = (1, 'Packing CP')
  UNASSIGNED_PLEATS = (2, 'Unassigned Pleats')
  ASSIGNED_PLEATS = (3, 'Assigned Pleats')
  ASSIGNED_RIDGES = (4, 'Assigned Ridges')
  CENTRAL_RIDGES = (5, 'Central Ridges')
  PLUS_UNASSIGNED_PLEATS = (6, '+ Unassigned Pleats')
  FINAL_SOLVING_CP = (7, 'Final Solving CP')

  def __init__(self, stage_index: int, display_name: str):
    self.stage_index = stage_index
    self.display_name = display_name

  @property
  def numbered_name(self) -> str:
    return f'({self.stage_index}) {self.display_name}'


@dataclasses.dataclass(frozen=True, kw_only=True)
class StageSnapshot:
  """Intermediate crease-pattern snapshot captured during hinge solving."""

  creases: crease_pattern_lib.CreasePattern
  grid_size: int
  stage: SolveStage | None = None
  name: str = ''

  def __post_init__(self):
    if not self.name and self.stage is not None:
      object.__setattr__(self, 'name', self.stage.numbered_name)


@dataclasses.dataclass(frozen=True, kw_only=True)
class SolveStats:
  """Diagnostic statistics for a solved crease pattern."""

  time_taken: float = 0.0
  state_score: int = 0
  partition_progress: int = 0
  symmetry_score: float = 0.0


@dataclasses.dataclass(frozen=True, kw_only=True)
class SolvedCreasePattern:
  """Immutable stage record produced by solving."""

  packed: PackedCreasePattern
  creases: crease_pattern_lib.CreasePattern
  scale_factor: int = 1
  base_type: BaseType = BaseType.NORMAL
  central_crease_removed: bool = False
  symmetry: symmetry.Transform = symmetry.Transform.IDENTITY
  config: 'hinge_utils.SolverConfig'
  is_globally_flat_foldable: bool = True
  stats: SolveStats = dataclasses.field(default_factory=SolveStats)
  assigned_hinges: tuple[crease.Crease, ...] = dataclasses.field(
      default=(), compare=False
  )
  assigned_pleats: tuple[crease.Crease, ...] = dataclasses.field(
      default=(), compare=False
  )
  stages: tuple[StageSnapshot, ...] = ()

  @property
  def grid_size(self) -> int:
    return self.packed.grid_size * self.scale_factor

  @classmethod
  def from_solver_result(
      cls,
      result: 'hinge_utils.SolverResult',
      packed: PackedCreasePattern,
      *,
      config: 'hinge_utils.SolverConfig | None' = None,
      scale_factor: int = 1,
      stages: Sequence[StageSnapshot] = (),
  ) -> 'SolvedCreasePattern':
    """Constructs a SolvedCreasePattern from a SolverResult."""
    state = result.final_state
    raw_cfg = config if config is not None else state.solver_config
    clean_cfg = type(raw_cfg).from_dict(
        {k: v for k, v in raw_cfg.asdict().items() if k != 'grid_size'}
    )
    return cls(
        packed=packed,
        creases=result.crease_pattern,
        scale_factor=scale_factor,
        base_type=result.base_type,
        central_crease_removed=bool(state.solver_config.remove_central_crease),
        symmetry=state.transform,
        config=clean_cfg,
        is_globally_flat_foldable=result.is_globally_flat_foldable,
        stats=SolveStats(
            time_taken=result.time_taken,
            state_score=state.state_score,
            partition_progress=state.partition_progress,
            symmetry_score=(
                float(state.symmetry_score())
                if clean_cfg.tie_break_by_hinge_symmetry
                else 0.0
            ),
        ),
        assigned_hinges=tuple(state.hinges),
        assigned_pleats=tuple(state.pleats),
        stages=tuple(stages),
    )

  def to_svg(
      self,
      *,
      pretty: bool = True,
      colors: Mapping[crease.Fold, str] | None = None,
      stroke_width: float | int | str | None = None,
      dashed_valley: bool = False,
      include_grid: bool = False,
  ) -> str:
    return _render_merged_svg(
        self.creases,
        self.grid_size,
        pretty=pretty,
        colors=colors,
        stroke_width=stroke_width,
        dashed_valley=dashed_valley,
        include_grid=include_grid,
    )

  def flat_fold(self) -> dict[str, Any]:
    """Computes the 3D flat-folded state in FOLD format."""
    cp_str = cp_format.to_cp(
        self.creases, scale=1, ndigits=3, grid=self.grid_size
    )
    out = fold_cp.analyze_fold_file(
        filename='pattern.cp',
        doc_content=cp_str,
        file_ext='.cp',
        generate_state_data=True,
    )
    if (
        out.flat_folder_fold is None
        or out.flat_folder_fold.folded_state_fold is None
    ):
      reason = (
          out.flat_folder_fold.conflict_reason
          if out.flat_folder_fold is not None
          else None
      )
      raise ValueError(
          f'Flat-folder failed on crease pattern: {reason or "unknown error"}'
      )
    return dict(out.flat_folder_fold.folded_state_fold)


class ShapingStrategy(enum.Enum):
  """Strategy used to produce a ShapedCreasePattern."""

  DEFAULT = 'default'
  ROTATE_BEND = 'rotate_bend'
  NO_FLIP = 'no_flip'
  SHALLOW_FOLDS = 'shallow_folds'
  TOOLBOX = 'toolbox'
  POSE_FIGURE = 'pose_figure'
  POSE_SEARCH = 'pose_search'
  POSE_GREEDY = 'pose_greedy'
  NONE = 'none'


@dataclasses.dataclass(frozen=True, kw_only=True)
class ShapedCreasePattern:
  """Immutable stage record produced by shaping."""

  solved: SolvedCreasePattern
  creases: crease_pattern_lib.CreasePattern
  diagnostics: Mapping[str, Any] = dataclasses.field(
      default_factory=immutabledict.immutabledict
  )
  is_locally_flat_foldable: bool = True
  strategy: ShapingStrategy = ShapingStrategy.DEFAULT
  time_taken: float = 0.0

  @classmethod
  def from_crease_pattern(
      cls,
      cp: crease_pattern_lib.CreasePattern,
      *,
      solved: SolvedCreasePattern,
      strategy: ShapingStrategy | str = ShapingStrategy.DEFAULT,
      time_taken: float = 0.0,
      diagnostics: Mapping[str, Any] | None = None,
  ) -> 'ShapedCreasePattern':
    """Constructs a ShapedCreasePattern from a crease pattern and solved parent."""
    resolved_strategy = (
        strategy
        if isinstance(strategy, ShapingStrategy)
        else ShapingStrategy(strategy)
    )
    is_ff = cp.local_flat_foldability()
    if diagnostics is None:
      resolved_diagnostics = immutabledict.immutabledict({
          'strategy': resolved_strategy.value,
          'time_taken': time_taken,
          'is_locally_flat_foldable': is_ff,
      })
    else:
      resolved_diagnostics = immutabledict.immutabledict(dict(diagnostics))
    return cls(
        solved=solved,
        creases=cp,
        diagnostics=resolved_diagnostics,
        is_locally_flat_foldable=is_ff,
        strategy=resolved_strategy,
        time_taken=time_taken,
    )

  @property
  def packed(self) -> PackedCreasePattern:
    return self.solved.packed

  @property
  def grid_size(self) -> int:
    return self.solved.grid_size

  def to_svg(
      self,
      *,
      pretty: bool = True,
      colors: Mapping[crease.Fold, str] | None = None,
      stroke_width: float | int | str | None = None,
      dashed_valley: bool = False,
      include_grid: bool = False,
  ) -> str:
    return _render_merged_svg(
        self.creases,
        self.grid_size,
        pretty=pretty,
        colors=colors,
        stroke_width=stroke_width,
        dashed_valley=dashed_valley,
        include_grid=include_grid,
    )

  def flat_fold(self, *, strict: bool = True) -> dict[str, Any]:
    """Computes the flat-folded state in FOLD format.

    Args:
      strict: If True (default), raises ValueError when the global
        layer-ordering solver detects a conflict (such as a taco-tortilla
        conflict caused by local 3D shaping folds). If False, falls back to
        returning the kinematic 2D folded geometry tagged with
        `'bp_pipeline:kinematic_fallback': True` when global layer ordering
        fails.

    Returns:
      A dictionary conforming to the FOLD file format specification representing
      the flat-folded state.
    """
    cp_str = cp_format.to_cp(
        self.creases, scale=1, ndigits=6, grid=self.grid_size
    )
    out = fold_cp.analyze_fold_file(
        filename='pattern.cp',
        doc_content=cp_str,
        file_ext='.cp',
        generate_state_data=True,
    )
    if (
        out.flat_folder_fold is not None
        and out.flat_folder_fold.folded_state_fold is not None
    ):
      return dict(out.flat_folder_fold.folded_state_fold)

    reason = (
        out.flat_folder_fold.conflict_reason
        if out.flat_folder_fold is not None
        else None
    )
    if strict:
      raise ValueError(
          'Flat-folder failed on shaped crease pattern:'
          f' {reason or "unknown error"}'
      )

    lines = []
    assignments = []
    for c in self.creases.creases:
      lines.append(((c.p1.x, c.p1.y), (c.p2.x, c.p2.y)))
      if c.fold == crease.Fold.MOUNTAIN:
        assignments.append('M')
      elif c.fold == crease.Fold.VALLEY:
        assignments.append('V')
      elif c.fold == crease.Fold.BORDER:
        assignments.append('B')
      else:
        assignments.append('U')
    vertices_coords, edge_vertices, edge_lines, _ = (
        graph_creation.create_graph_from_lines(lines)
    )
    edge_assignments = [
        assignments[indices[0]] if indices else 'U' for indices in edge_lines
    ]
    _, face_vertices = graph_topology.graph_to_adjacency_and_faces(
        vertices_coords, edge_vertices
    )
    folded_vertices, face_orientations = folded_state.compute_folded_state(
        vertices_coords, face_vertices, edge_vertices, edge_assignments
    )
    return {
        'file_spec': 1.1,
        'file_creator': 'bp_pipeline (kinematic folded state)',
        'frame_classes': ['foldedForm'],
        'vertices_coords': [
            [float(v[0]), float(v[1])] if v is not None else [0.0, 0.0]
            for v in folded_vertices
        ],
        'faces_vertices': [list(f) for f in face_vertices],
        'edges_vertices': [list(e) for e in edge_vertices],
        'edges_assignment': edge_assignments,
        'bp_pipeline:face_orientations': list(face_orientations),
        'bp_pipeline:kinematic_fallback': True,
        'bp_pipeline:conflict_reason': reason,
    }


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PipelineResult:
  """Typed end-to-end result of `fold(...)`."""

  design: DesignTree
  packed: PackedCreasePattern
  solved: SolvedCreasePattern
  candidates: tuple[SolvedCreasePattern, ...]
  shaped: ShapedCreasePattern | None = None
  shaped_candidates: tuple[ShapedCreasePattern, ...] = ()

  def to_svg(
      self,
      *,
      pretty: bool = True,
      colors: Mapping[crease.Fold, str] | None = None,
      stroke_width: float | int | str | None = None,
      dashed_valley: bool = False,
      include_grid: bool = False,
  ) -> str:
    """Renders the resolved crease pattern to a vector SVG string."""
    target = self.shaped if self.shaped is not None else self.solved
    return target.to_svg(
        pretty=pretty,
        colors=colors,
        stroke_width=stroke_width,
        dashed_valley=dashed_valley,
        include_grid=include_grid,
    )
