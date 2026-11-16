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

"""Classes for representing model elements (flaps and rivers) during shaping."""

from collections.abc import Sequence
import copy
import dataclasses
import functools
import itertools
import logging
import math
import typing
from typing import Optional, cast

from bp_pipeline.crease_pattern import crease as crease_utils
from bp_pipeline.crease_pattern import crease_pattern as cp_utils
from bp_pipeline.crease_pattern import point as point_utils
from bp_pipeline.packing import compute_hinges
from bp_pipeline.packing import compute_ridges
from bp_pipeline.packing import river as river_utils
from bp_pipeline.packing import types as packing_types
from bp_pipeline.shaping import base_rectangle as rect_utils
from bp_pipeline.shaping import shaping_utils
from bp_pipeline.shaping import types as shaping_types
from bp_pipeline.shaping import vector
from bp_pipeline.solving import utils as solving_utils
from bp_pipeline.stick_figures import stick_figure as sf_utils
from bp_pipeline.stick_figures import traversal as tr
from matplotlib import ticker
import matplotlib.pyplot as plt

_NEAR_PAPER_BOUNDING_RECTANGLE = rect_utils.Rectangle(
    name='bounding_rect',
    x_start=-2,
    y_start=-2,
    x_len=5,
    y_len=5,
)


CENTRAL_CREASE_X = 0.5


def _check_continuity(creases: Sequence[crease_utils.Crease]) -> bool:
  """Checks if the creases form a connected component."""
  for crease_1, crease_2 in itertools.pairwise(creases):
    if not crease_1.p2_init.isclose(crease_2.p1_init):
      return False
  return True


def _check_horizontal_or_vertical_parallel(
    crease_1: crease_utils.Crease, crease_2: crease_utils.Crease
) -> bool:
  """Checks if the creases are both horizontal or both vertical."""
  return (crease_1.is_horizontal and crease_2.is_horizontal) or (
      crease_1.is_vertical and crease_2.is_vertical
  )


def _check_horizontal_or_vertical_separation(
    crease_1: crease_utils.Crease, crease_2: crease_utils.Crease
) -> bool:
  """Checks if the horizontal or vertical creases are separated."""
  if crease_1.is_horizontal:
    return not math.isclose(crease_1.p1.y, crease_2.p1.y)
  elif crease_1.is_vertical:
    return not math.isclose(crease_1.p1.x, crease_2.p1.x)
  else:
    raise ValueError(
        f'Creases {crease_1} and {crease_2} are not horizontal or vertical.'
    )


@dataclasses.dataclass(kw_only=True, frozen=True)
class ShapingBoundary:
  """A boundary structured for shaping."""

  creases: Sequence[crease_utils.Crease]
  edge_unit_vectors: Sequence[vector.Vector]
  depth_unit_vectors: Sequence[vector.Vector]

  def __post_init__(self):
    """Checks that the boundary is continuous."""
    if not _check_continuity(self.creases):
      raise ValueError(f'Boundary {self.creases} is not continuous.')

  @functools.cached_property
  def closed(self) -> bool:
    """Whether the boundary is closed (a loop)."""
    return self.creases[-1].p2_init.isclose(self.creases[0].p1_init)

  @functools.cached_property
  def tips(self) -> Sequence[point_utils.Point]:
    """Gets tips of boundary line segments."""
    tips = [crease.p1_init for crease in self.creases]
    if not self.closed:
      tips.append(self.creases[-1].p2_init)
    return tips

  def _point_with_crease(
      self, p: point_utils.Point, crease: crease_utils.Crease
  ) -> bool:
    """Checks if a point is on a crease excluding p2_init."""
    return p.isclose(crease.p1_init) or (
        not p.isclose(crease.p2_init) and crease.is_intermediate_point(p)
    )

  def get_edge_index(self, p: point_utils.Point) -> int | None:
    """Gets the index of the edge p lies on."""
    for i, crease in enumerate(self.creases):
      if self._point_with_crease(p, crease):
        return i
    if not self.closed and p.isclose(self.creases[-1].p2_init):
      # The terminal endpoint of an open boundary belongs to its final crease.
      return len(self.creases) - 1
    return None

  def num_walk_steps_upper_bound(self, grid_size: int) -> int:
    """Returns an upper bound on the number of steps to walk the boundary."""
    return sum(
        math.ceil(shaping_utils.crease_grid_length(crease, grid_size)) + 2
        for crease in self.creases
    )

  def is_corner_point(self, p: point_utils.Point) -> bool:
    """Checks if a point is a corner point of the boundary."""
    return any(p.isclose(tip) for tip in self.tips)

  def walk(
      self,
      p: point_utils.Point,
      edge_index: int,
      grid_size: int,
      snap_to_grid: bool = True,
  ) -> tuple[point_utils.Point, int]:
    """Walks along boundary for one grid step from p."""
    if edge_index >= len(self.creases):
      raise ValueError(
          f'Edge index {edge_index} is out of range for boundary with'
          f' {len(self.creases)} creases.'
      )
    if (
        not self.closed
        and edge_index == len(self.creases) - 1
        and p.isclose(self.creases[-1].p2_init)
    ):
      raise ValueError(
          f'Point {p} is already at the terminal endpoint of open boundary'
          f' with {len(self.creases)} creases.'
      )
    point_status = self.creases[edge_index].point_status(p)
    if not (
        point_status == crease_utils.PointStatus.INTERNAL
        or point_status == crease_utils.PointStatus.TIP
    ):
      raise ValueError(
          f'Point {p} is not an intermediate/tip point on crease {edge_index}'
      )
    next_p = self.edge_unit_vectors[edge_index].point_add(p)

    if self.closed:
      next_edge_index = (edge_index + 1) % len(self.creases)
    else:
      next_edge_index = edge_index + 1

    next_tip = self.tips[next_edge_index]
    cur_next_segment = crease_utils.Crease(
        p, next_p, fold=crease_utils.Fold.HINGE
    )

    if (
        cur_next_segment.point_status(next_tip)
        != crease_utils.PointStatus.EXTERNAL
    ):
      return next_tip, next_edge_index

    if snap_to_grid:
      next_p = shaping_utils.get_closest_grid_line_intersection_to_crease_point(
          cur_next_segment, next_p, grid_size
      )
      assert next_p is not None, (
          f'No grid line intersection found on crease {cur_next_segment}'
          f' with grid size {grid_size}.'
      )

    return next_p, edge_index


@dataclasses.dataclass(kw_only=True, frozen=True)
class ShapingFlap(rect_utils.Rectangle[float]):
  """A flap structured for shaping represented as a rectangle."""

  _circle_centers: Sequence[point_utils.Point]
  grid_size: int

  shaping_history: list[shaping_types.ZigzagParams] = dataclasses.field(
      default_factory=list
  )
  max_shaping_depth: int | None = None

  creases: list[crease_utils.Crease] = dataclasses.field(default_factory=list)

  def __post_init__(self):
    super().__post_init__()

    if not (1 <= len(self.circle_centers) <= 2):
      raise ValueError(
          f'Rectangle {self.name} must have either 1 or 2 circle centers.'
      )
    if len(self.circle_centers) == 2:
      c1, c2 = self.circle_centers
      if c1.isclose(c2):
        raise ValueError(
            f'Circle centers {self.circle_centers} are not distinct.'
        )
    if self.is_square and len(self.circle_centers) != 1:
      raise ValueError(
          f'Square rectangle {self.name} must have exactly one circle center.'
      )
    elif (self.is_wide or self.is_tall) and len(self.circle_centers) != 2:
      raise ValueError(
          f'Wide or tall rectangle {self.name} must have exactly two circle'
          ' centers.'
      )
    if not self.check_center_alignment():
      raise ValueError(
          f'Flap {self.name} is not centered around its circle centers.'
      )
    if any(
        not _NEAR_PAPER_BOUNDING_RECTANGLE.is_internal_point(c)
        for c in self.circle_centers
    ):
      raise ValueError(
          f'Flap {self.name} is not normalized to lie within[-1, 2]^2.'
      )

  def __repr__(self) -> str:
    return (
        f'Flap({self.name}, {self.x_start}, {self.y_start}, {self.x_len},'
        f' {self.y_len}, {self.max_shaping_depth})'
    )

  def __deepcopy__(self, memo: object) -> 'ShapingFlap':
    del memo
    return dataclasses.replace(
        self,
        _circle_centers=list(self._circle_centers),
        shaping_history=list(self.shaping_history),
        creases=list(self.creases),
    )

  @functools.cached_property
  def circle_centers(self) -> Sequence[point_utils.Point]:
    """Gets flap circle centers without requiring them to be on paper."""
    return [
        point_utils.Point(
            x=circle_center.x, y=circle_center.y, require_on_paper=False
        )
        for circle_center in self._circle_centers
    ]

  @functools.cached_property
  def grid_unit_radius(self) -> float:
    """Returns the grid-unit radius of the flap."""
    return self.grid_size * min(self.x_len, self.y_len) / 2.0

  @functools.cached_property
  def radius(self) -> float:
    """Gets the (non-grid unit) flap radius."""
    return math.sqrt(2) * min(self.x_len, self.y_len) / 2.0

  def check_center_alignment(self) -> bool:
    """Checks if the rectangle is centered around its circle centers."""
    if not self.circle_centers:
      return False
    corners = self.corners
    close_centers = [
        min(
            self.circle_centers,
            key=functools.partial(point_utils.distance, corner),
        )
        for corner in corners
    ]
    center_vectors = [
        vector.Vector.from_points(corner, center)
        for corner, center in zip(corners, close_centers)
    ]
    return all(
        math.isclose(abs(center_vector.slope), 1)
        for center_vector in center_vectors
    )

  @classmethod
  def from_rectangle(
      cls,
      rect: rect_utils.Rectangle[float],
      circle_centers: Sequence[point_utils.Point],
      grid_size: int,
      max_shaping_depth: int | None = None,
  ) -> 'ShapingFlap':
    """Creates a shaping flap from a rectangle."""
    return cls(
        name=rect.name,
        x_start=rect.x_start,
        y_start=rect.y_start,
        x_len=rect.x_len,
        y_len=rect.y_len,
        _circle_centers=circle_centers,
        grid_size=grid_size,
        max_shaping_depth=max_shaping_depth,
    )

  def to_rect(self) -> rect_utils.Rectangle[float]:
    """Converts shaping flap to a rectangle."""
    return rect_utils.Rectangle(
        name=self.name,
        x_start=self.x_start,
        y_start=self.y_start,
        x_len=self.x_len,
        y_len=self.y_len,
    )

  def to_legacy_rectangle(self) -> shaping_utils.Rectangle:
    """Converts shaping flap to a legacy rectangle."""
    return shaping_utils.Rectangle(
        name=self.name,
        x_start=self.x_start,
        y_start=self.y_start,
        x_len=self.x_len,
        y_len=self.y_len,
        circle_centers=self.circle_centers,
        radius=self.radius,
    )

  @classmethod
  def from_corners(
      cls,
      corners: Sequence[point_utils.Point],
      name: str,
      circle_centers: Sequence[point_utils.Point],
      grid_size: int,
      max_shaping_depth: int | None = None,
  ) -> 'ShapingFlap':
    """Creates a shaping flap from its corners and circle centers."""
    rect = rect_utils.Rectangle.from_corners(corners, name=name)
    return cls.from_rectangle(
        rect,
        circle_centers,
        grid_size,
        max_shaping_depth=max_shaping_depth,
    )

  @classmethod
  def from_centers(
      cls,
      circle_centers: Sequence[point_utils.Point],
      radius: float,
      grid_size: int,
      name: str,
      max_shaping_depth: int | None = None,
  ) -> 'ShapingFlap':
    """Creates a shaping flap from its circle centers."""
    if radius <= 0:
      raise ValueError(f'Radius {radius} must be non-negative.')

    corner_vectors = [
        radius * vector.Vector(x=-1, y=-1).normalize,
        radius * vector.Vector(x=1, y=-1).normalize,
        radius * vector.Vector(x=1, y=1).normalize,
        radius * vector.Vector(x=-1, y=1).normalize,
    ]
    if len(circle_centers) == 1:
      corners = [
          corner_vector.point_add(circle_centers[0])
          for corner_vector in corner_vectors
      ]
      return cls.from_corners(
          corners,
          name=name,
          circle_centers=circle_centers,
          grid_size=grid_size,
          max_shaping_depth=max_shaping_depth,
      )
    elif len(circle_centers) == 2:
      p1, p2 = circle_centers
      if p1.isclose(p2):
        raise ValueError(f'Circle centers {circle_centers} are not distinct.')

      if not math.isclose(p1.x, p2.x):
        p1, p2 = point_utils.sort_by_x(p1, p2)
        top_left = corner_vectors[0].point_add(p1)
        top_right = corner_vectors[1].point_add(p2)
        bottom_right = corner_vectors[2].point_add(p2)
        bottom_left = corner_vectors[3].point_add(p1)
      elif not math.isclose(p1.y, p2.y):
        p1, p2 = point_utils.sort_by_y(p1, p2)
        top_left = corner_vectors[0].point_add(p1)
        top_right = corner_vectors[1].point_add(p1)
        bottom_right = corner_vectors[2].point_add(p2)
        bottom_left = corner_vectors[3].point_add(p2)
      else:
        raise ValueError(
            f'Circle centers {circle_centers} line on different grid lines.'
        )
      return cls.from_corners(
          [top_left, top_right, bottom_right, bottom_left],
          name=name,
          circle_centers=circle_centers,
          grid_size=grid_size,
          max_shaping_depth=max_shaping_depth,
      )
    else:
      raise ValueError(
          f'Circle centers {circle_centers} must have 1 or 2 centers.'
      )

  @classmethod
  def from_packing_rect(cls) -> Optional['ShapingFlap']:
    pass

  def clip(
      self,
      xmin: float,
      ymin: float,
      xmax: float,
      ymax: float,
  ) -> Optional[rect_utils.Rectangle[float]]:
    """Clips shaping flap as a rectangle to a given region."""
    return self.to_rect().clip(xmin, ymin, xmax, ymax)

  def clip_to_paper(self) -> Optional[rect_utils.Rectangle[float]]:
    """Clips the shaping flap as a rectangle to the paper."""
    return self.to_rect().clip(0, 0, 1, 1)

  def snap_to_grid(self, grid_size: int) -> 'ShapingFlap':
    """Snaps shaping flap to grid."""
    corners = [
        shaping_utils.snap_point_to_grid(c, grid_size) for c in self.corners
    ]
    return ShapingFlap.from_corners(
        corners,
        name=self.name,
        circle_centers=self.circle_centers,
        grid_size=grid_size,
    )

  def get_ridges(self) -> list[crease_utils.Crease]:
    """Gets the ridge creases of the rectangle."""
    if self.circle_centers is None or not self.circle_centers:
      raise ValueError('Circle centers must be defined for ridge computation.')
    corners = self.corners
    if len(self.circle_centers) == 1:
      return [
          crease_utils.Crease(
              corners[0],
              corners[2],
              fold=crease_utils.Fold.MOUNTAIN,
          ),
          crease_utils.Crease(
              corners[1],
              corners[3],
              fold=crease_utils.Fold.MOUNTAIN,
          ),
      ]
    else:
      close_centers = [
          min(
              self.circle_centers,
              key=functools.partial(point_utils.distance, corner),
          )
          for corner in corners
      ]
      return [
          crease_utils.Crease(center, corner, fold=crease_utils.Fold.MOUNTAIN)
          for center, corner in zip(close_centers, corners)
      ] + [
          crease_utils.Crease(
              self.circle_centers[0],
              self.circle_centers[1],
              fold=crease_utils.Fold.MOUNTAIN,
          )
      ]

  @classmethod
  def from_legacy_rectangle_direct(
      cls, rectangle: shaping_utils.Rectangle, grid_size: int
  ) -> 'ShapingFlap':
    """Creates a shaping flap from a legacy rectangle directly."""
    assert rectangle.circle_centers is not None
    assert rectangle.radius is not None
    return cls.from_centers(
        rectangle.circle_centers,
        rectangle.radius,
        grid_size,
        rectangle.name,
    )

  def shrink(
      self, depth: float, copy_shaping_history: bool = True
  ) -> 'ShapingFlap':
    """Shrinks the shaping flap by a given grid-unit normalized depth."""
    new_flap = ShapingFlap.from_centers(
        self.circle_centers,
        self.radius - math.sqrt(2) * depth / self.grid_size,
        self.grid_size,
        self.name,
    )
    new_flap = dataclasses.replace(
        new_flap,
        max_shaping_depth=int(self.max_shaping_depth - depth)
        if self.max_shaping_depth
        else None,
        shaping_history=copy.copy(self.shaping_history)
        if copy_shaping_history
        else [],
    )
    return new_flap.extend_creases(self.creases)

  def get_circumbscribed_river(
      self, river_width: int, name: str
  ) -> 'ShapingRiver':
    """Divides the flap into an outer river of width `river_width` and an inner flap."""
    if river_width <= 0:
      raise ValueError(f'River width {river_width} must be positive.')
    if river_width > self.grid_unit_radius:
      raise ValueError(
          f'River width {river_width} must be smaller than the flaps grid unit'
          f' radius {self.grid_unit_radius}.'
      )

    outer_bank = self.edges
    inner_bank = self.shrink(river_width).edges
    return ShapingRiver(
        name=name,
        bank_1=inner_bank,
        bank_2=outer_bank,
        width=river_width,
        grid_size=self.grid_size,
    )

  def rename_and_copy(
      self, name: str, max_shaping_depth: int | None = None
  ) -> 'ShapingFlap':
    """Renames the shaping flap."""
    if max_shaping_depth is None:
      max_shaping_depth = self.max_shaping_depth
    return dataclasses.replace(
        self, name=name, max_shaping_depth=max_shaping_depth
    )

  @functools.cached_property
  def is_right_of_central_crease(self) -> bool:
    """Checks if the rectangle lies to the right of the central fold."""
    return crease_utils.approx_lte(CENTRAL_CREASE_X, self.x_start)

  @functools.cached_property
  def contains_central_crease(self) -> bool:
    """Checks if the rectangle contains the central crease."""
    return point_utils.in_range(
        CENTRAL_CREASE_X,
        self.x_start,
        self.x_start + self.x_len,
        strict=True,
    )

  @property
  def center(self) -> tuple[float, float]:
    """Returns the center of the rectangle."""
    return (
        self.x_start + self.x_len / 2.0,
        self.y_start + self.y_len / 2.0,
    )

  @functools.cached_property
  def is_bottom_of_central_crease(self) -> bool:
    """Checks if the rectangle lies to the bottom of the central fold."""
    central_crease_line_y = 0.5
    return crease_utils.approx_gte(central_crease_line_y, self.y_start)

  @functools.cached_property
  def borders_central_crease(self) -> bool:
    """Checks if the flap borders the central crease."""
    corner_x_coords = [corner.x for corner in self.corners]
    return any(
        math.isclose(x_coord, CENTRAL_CREASE_X) for x_coord in corner_x_coords
    )

  @functools.cached_property
  def is_central_flap(self) -> bool:
    """Checks if the flap is a central flap."""
    paper = shaping_utils.Rectangle(
        name='paper', x_start=0, y_start=0, x_len=1, y_len=1
    )
    return all(paper.is_internal_point(corner) for corner in self.corners)

  @property
  def elevations(self) -> set[int]:
    """Returns the set of elevations of axial creases in the flap."""
    return {c.elevation for c in self.creases if c.elevation is not None}

  def extend_creases(
      self, creases: Sequence[crease_utils.Crease]
  ) -> 'ShapingFlap':
    """Appends all creases intersecting the flap to the flap's creases list."""
    intersecting_creases = [
        crease for crease in creases if self.intersects_crease(crease)
    ]
    self.creases.extend(intersecting_creases)
    return self


def _get_shared_tip(
    crease_1: crease_utils.Crease, crease_2: crease_utils.Crease
) -> point_utils.Point | None:
  """Returns the shared tip between two creases. Otherwise returns None."""
  if crease_2.is_tip(crease_1.p1):
    return crease_1.p1
  elif crease_2.is_tip(crease_1.p2):
    return crease_1.p2
  else:
    return None


def _check_left_contiguous(
    crease_1: crease_utils.Crease, crease_2: crease_utils.Crease
) -> tuple[bool, crease_utils.Crease]:
  """Checks if crease 2 is left contiguous with crease 1."""
  if crease_1.p1_init.isclose(crease_2.p2_init):
    return True, crease_2
  elif crease_1.p1_init.isclose(crease_2.p1_init):
    return True, crease_2.flip_tips()
  else:
    return False, crease_2


def get_flap_boundary(
    rect: shaping_utils.Rectangle | ShapingFlap,
    grid_size: int,
) -> ShapingBoundary:
  """Gets boundary object representing boundary of the flap."""
  return ShapingBoundary(
      creases=[
          crease_utils.Crease(
              rect.corners[i],
              rect.corners[(i + 1) % 4],
              fold=crease_utils.Fold.HINGE,
          )
          for i in range(4)
      ],
      edge_unit_vectors=rect.get_unit_edge_vectors(grid_size),
      depth_unit_vectors=rect.get_unit_depth_vectors(grid_size),
  )


def _check_right_contiguous(
    crease_1: crease_utils.Crease, crease_2: crease_utils.Crease
) -> tuple[bool, crease_utils.Crease]:
  """Checks if crease 2 is right contiguous with crease 1."""
  if crease_1.p2_init.isclose(crease_2.p1_init):
    return True, crease_2
  elif crease_1.p2_init.isclose(crease_2.p2_init):
    return True, crease_2.flip_tips()
  else:
    return False, crease_2


def _check_points_on_same_horizontal_or_vertical(
    p1: point_utils.Point, p2: point_utils.Point
) -> bool:
  """Checks if the points are on the same horizontal or vertical line."""
  crease = crease_utils.Crease(p1, p2, fold=crease_utils.Fold.HINGE)
  return crease.is_horizontal or crease.is_vertical


def _order_contiguous_creases(
    creases: list[crease_utils.Crease],
) -> list[crease_utils.Crease]:
  """Orders a list of connected creases into a single linear path."""
  creases_copy = copy.copy(creases)
  contiguous_creases = []
  contiguous_creases.append(creases_copy.pop(0))
  l = len(creases_copy)
  for _ in range(l):
    for ind, crease in enumerate(creases_copy):
      is_left_contiguous, oriented_crease = _check_left_contiguous(
          contiguous_creases[0], crease
      )
      if is_left_contiguous:
        contiguous_creases.insert(0, oriented_crease)
        creases_copy.pop(ind)
        break
      is_right_contiguous, oriented_crease = _check_right_contiguous(
          contiguous_creases[-1], crease
      )
      if is_right_contiguous:
        contiguous_creases.append(oriented_crease)
        creases_copy.pop(ind)
        break

  if creases_copy:
    raise ValueError(f'Creases {creases} do not form a linear contiguous path!')
  return contiguous_creases


def _align_banks(
    bank_1: list[crease_utils.Crease],
    bank_2: list[crease_utils.Crease],
) -> tuple[list[crease_utils.Crease], list[crease_utils.Crease]]:
  """Aligns the banks by ordering the creases in each bank."""
  bank_1 = _order_contiguous_creases(bank_1)
  bank_2 = _order_contiguous_creases(bank_2)

  crease_1, crease_2 = bank_1[0], bank_2[0]
  if not (
      _check_points_on_same_horizontal_or_vertical(
          crease_1.p1_init, crease_2.p1_init
      )
      and crease_1.is_parallel(crease_2)
  ):
    bank_2 = [crease.flip_tips() for crease in bank_2[::-1]]
  return bank_1, bank_2


def _line_tuple_to_creases(
    p1: tuple[float, float],
    p2: tuple[float, float],
    grid_size: int | None = None,
) -> crease_utils.Crease:
  """Converts a line tuple to a grid_size normalized Crease object."""
  assert grid_size is not None
  return crease_utils.Crease(
      point_utils.Point(
          x=p1[0] / grid_size, y=p1[1] / grid_size, require_on_paper=False
      ),
      point_utils.Point(
          x=p2[0] / grid_size, y=p2[1] / grid_size, require_on_paper=False
      ),
      fold=crease_utils.Fold.HINGE,
  )


def _transport_point_along_crease(
    point: point_utils.Point,
    crease: crease_utils.Crease,
    displacement: float,
) -> point_utils.Point:
  """Transports a point along a crease by a given grid-unit displacement."""
  if point.isclose(crease.p1):
    displacement_vector = (
        displacement * vector.Vector.from_points(crease.p1, crease.p2).normalize
    )
  elif point.isclose(crease.p2):
    displacement_vector = (
        displacement * vector.Vector.from_points(crease.p2, crease.p1).normalize
    )
  else:
    raise ValueError(f'Point {point} is not a tip of crease {crease}!')
  return displacement_vector.point_add(point)


def _transport_and_stretch_crease_along_vector(
    crease: crease_utils.Crease,
    displacement_vector: vector.Vector,
    stretch_point: point_utils.Point,
) -> crease_utils.Crease:
  """Transports and stretches a horizontal or vertical crease along a vector."""
  if crease.is_horizontal:
    normal_displacement = displacement_vector.y * vector.Vector(x=0, y=1)
  elif crease.is_vertical:
    normal_displacement = displacement_vector.x * vector.Vector(x=1, y=0)
  else:
    raise ValueError(f'Crease {crease} is not horizontal or vertical!')

  if stretch_point.isclose(crease.p1):
    normal_point, stretch_point = crease.p2, crease.p1
  elif stretch_point.isclose(crease.p2):
    normal_point, stretch_point = crease.p1, crease.p2
  else:
    raise ValueError(
        f'Stretch point {stretch_point} must be one of the tips of crease'
        f' {crease}!'
    )

  normally_transported_tip = normal_displacement.point_add(normal_point)
  new_stretched_point = displacement_vector.point_add(stretch_point)
  return crease_utils.Crease(
      normally_transported_tip,
      new_stretched_point,
      fold=crease.fold,
  )


def _validate_bank_endpoints(
    river_name: str,
    bank: Sequence[crease_utils.Crease],
) -> None:
  """Validates that a river bank is closed or starts and ends on the border."""
  start, end = bank[0].p1_init, bank[-1].p2_init
  if not (start.isclose(end) or not (start.is_internal or end.is_internal)):
    raise ValueError(
        f'River {river_name} is not a loop and does not start and end on paper'
        ' border.'
    )


@dataclasses.dataclass(kw_only=True, frozen=True)
class ShapingRiver:
  """A river structured for shaping."""

  name: str
  bank_1: Sequence[crease_utils.Crease]
  bank_2: Sequence[crease_utils.Crease]
  width: float
  grid_size: int

  shaping_history: list[shaping_types.RiverZigzagParams] = dataclasses.field(
      default_factory=list
  )
  _max_shaping_depth: float | None = None
  _ridges: Sequence[crease_utils.Crease] | None = None
  _outer_bank_index: shaping_types.BankIndex | None = None

  creases: list[crease_utils.Crease] = dataclasses.field(default_factory=list)

  def __post_init__(self):
    l1 = len(self.bank_1)
    if l1 != len(self.bank_2):
      raise ValueError(
          f'Banks {self.bank_1} and {self.bank_2} must have the same number'
          ' of creases.'
      )
    if not _check_continuity(self.bank_1):
      raise ValueError(f'Bank {self.bank_1} is not continuous.')
    if not _check_continuity(self.bank_2):
      raise ValueError(f'Bank {self.bank_2} is not continuous.')
    for crease_1, crease_2 in zip(self.bank_1, self.bank_2, strict=True):
      if not _check_horizontal_or_vertical_parallel(crease_1, crease_2):
        raise ValueError(
            f'Creases {crease_1} and {crease_2} are either not horizontal,'
            ' vertical, or parallel.'
        )
      if not _check_horizontal_or_vertical_separation(crease_1, crease_2):
        raise ValueError(
            f'Creases {crease_1} and {crease_2} are not separated.'
        )
    _validate_bank_endpoints(self.name, self.bank_1)
    _validate_bank_endpoints(self.name, self.bank_2)
    bank_1_corners = [crease.p2_init for crease in self.bank_1[:-1]]
    bank_2_corners = [crease.p2_init for crease in self.bank_2[:-1]]
    if self.is_closed:
      bank_1_corners.append(self.bank_1[-1].p2_init)
      bank_2_corners.append(self.bank_2[-1].p2_init)
    for ridge, corner_1, corner_2 in zip(
        self.ridges, bank_1_corners, bank_2_corners, strict=True
    ):
      if not (
          (ridge.p1.isclose(corner_1) and ridge.p2.isclose(corner_2))
          or (ridge.p2.isclose(corner_1) and ridge.p1.isclose(corner_2))
      ):
        raise ValueError(
            f'Ridge {ridge} is not between corners {corner_1} and {corner_2}'
        )
      ridge_vector = vector.Vector.from_points(ridge.p1, ridge.p2)
      if not math.isclose(abs(ridge_vector.slope), 1):
        raise ValueError(
            f'Ridge {ridge} is not at 45o angle between corners of'
            f' {corner_1} and {corner_2}'
        )

  def __repr__(self) -> str:
    return (
        f'River({self.name}, {self.width}, {self.grid_size}, {self.bank_1},'
        f' {self.bank_2})'
    )

  def __deepcopy__(self, memo: object) -> 'ShapingRiver':
    del memo
    return dataclasses.replace(
        self,
        bank_1=list(self.bank_1),
        bank_2=list(self.bank_2),
        shaping_history=list(self.shaping_history),
        _ridges=list(self._ridges) if self._ridges is not None else None,
        creases=list(self.creases),
    )

  @property
  def max_shaping_depth(self) -> float:
    """Gets the maximum shaping depth of the river."""
    return (
        self._max_shaping_depth
        if self._max_shaping_depth is not None
        else self.width
    )

  def get_bank_from_bank_index(
      self, bank_index: shaping_types.BankIndex
  ) -> Sequence[crease_utils.Crease]:
    """Gets the bank corresponding to the given bank index."""
    return self.get_bank_from_index(bank_index)

  @functools.cached_property
  def is_closed(self) -> bool:
    """Checks if the river is closed (i.e. forms a circle)."""
    start, end = self.bank_1[0].p1_init, self.bank_1[-1].p2_init
    return start.isclose(end)

  def get_bank_corners(
      self, bank_index: shaping_types.BankIndex
  ) -> list[point_utils.Point]:
    """Returns the corners of the given bank."""
    bank = self.get_bank_from_index(bank_index)
    if self.is_closed:
      return [crease.p1_init for crease in bank]
    else:
      return [crease.p1_init for crease in bank] + [bank[-1].p2_init]

  @classmethod
  def from_packing_river(
      cls,
      name: str,
      river: (
          river_utils.River
          | river_utils.LShapedRiver
          | river_utils.HuggingRiver
      ),
  ) -> 'ShapingRiver':
    """Creates a ShapingRiver object from a packing river object."""
    grid_size = river.grid_size
    width = river.width
    river_hinges = [
        _line_tuple_to_creases(p1, p2, grid_size)
        for p1, p2 in compute_hinges.get_river_border(
            river.rectangles, grid_size
        )
    ]

    river_ridges = [
        _line_tuple_to_creases(p1, p2, river.grid_size)
        for p1, p2 in compute_ridges.get_river_ridge_lines(river)
    ]
    river_ridges = crease_utils.merge_creases(river_ridges)
    max_ridge_len = width * math.sqrt(2) / grid_size
    full_ridges = []
    for ridge in river_ridges:
      if crease_utils.strict_less(ridge.length, max_ridge_len):
        assert (
            ridge.p1.is_border or ridge.p2.is_border
        ), f'Shortened ridge {ridge} is not on paper border.'
        border_point, inner_point = (
            (ridge.p1, ridge.p2) if ridge.p1.is_border else (ridge.p2, ridge.p1)
        )
        unit_direction_vector = vector.Vector.from_points(
            inner_point, border_point
        ).normalize
        end_point = (unit_direction_vector * max_ridge_len).point_add(
            inner_point
        )
        ridge = crease_utils.Crease(
            inner_point,
            end_point,
            fold=crease_utils.Fold.MOUNTAIN,
        )
      full_ridges.append(ridge)
    river_ridges = full_ridges

    if not river_ridges:
      river_hinges = [
          crease for crease in river_hinges if not crease.is_overlapping_border
      ]
    else:
      reconstructed_river_hinges = []
      for ridge in river_ridges:
        ridge_hinges = [[], []]

        for crease in river_hinges:
          shared_tip = _get_shared_tip(ridge, crease)
          if shared_tip and shared_tip.isclose(ridge.p1):
            ridge_hinges[0].append(crease)
          elif shared_tip and shared_tip.isclose(ridge.p2):
            ridge_hinges[1].append(crease)

        if len(ridge_hinges[0]) + len(ridge_hinges[1]) == 4:
          reconstructed_river_hinges.extend(ridge_hinges[0])
          reconstructed_river_hinges.extend(ridge_hinges[1])
        else:
          assert len(ridge_hinges[0]) == 2 or len(ridge_hinges[1]) == 2, (
              f'Ridge {ridge} should intersect 2 hinges on at least one side of'
              f' the river. Instead got {len(ridge_hinges[0])} and'
              f' {len(ridge_hinges[1])} intersections'
          )
          if len(ridge_hinges[0]) == 2:
            ridge_start, ridge_end, complete_hinge_pair = (
                ridge.p1,
                ridge.p2,
                ridge_hinges[0],
            )
          else:
            ridge_start, ridge_end, complete_hinge_pair = (
                ridge.p2,
                ridge.p1,
                ridge_hinges[1],
            )
          displacement_vector = (
              ridge.length
              * vector.Vector.from_points(ridge_start, ridge_end).normalize
          )
          new_hinge_pair = [
              _transport_and_stretch_crease_along_vector(
                  crease, displacement_vector, ridge_start
              )
              for crease in complete_hinge_pair
          ]

          reconstructed_river_hinges.extend(complete_hinge_pair)
          reconstructed_river_hinges.extend(new_hinge_pair)

      river_hinges = reconstructed_river_hinges

    river_hinges = crease_utils.merge_creases(river_hinges)
    components = solving_utils.creases_to_paths(river_hinges)
    assert len(components) == 2, 'River hinges must have two components.'
    bank_1, bank_2 = components
    bank_1, bank_2 = _align_banks(bank_1, bank_2)

    return cls(
        name=name,
        bank_1=bank_1,
        bank_2=bank_2,
        width=river.width,
        grid_size=river.grid_size,
    )

  @functools.cached_property
  def ridges(self) -> Sequence[crease_utils.Crease]:
    """Gets the ridges of the river boundaries."""
    if self._ridges is not None:
      return self._ridges
    ridges = []
    for crease_1, crease_2 in zip(
        self.bank_1[:-1], self.bank_2[:-1], strict=True
    ):
      p1, p2 = crease_1.p2_init, crease_2.p2_init
      ridges.append(
          crease_utils.Crease(p1, p2, fold=crease_utils.Fold.MOUNTAIN)
      )
    if self.is_closed:
      ridges.append(
          crease_utils.Crease(
              self.bank_1[-1].p2_init,
              self.bank_2[-1].p2_init,
              fold=crease_utils.Fold.MOUNTAIN,
          )
      )
    return ridges

  @functools.cached_property
  def edge_unit_vectors(self) -> list[vector.Vector]:
    """Gets grid unit edge vectors for river."""
    return [
        (1 / self.grid_size)
        * vector.Vector.from_points(
            start=crease.p1_init, end=crease.p2_init
        ).normalize
        for crease in self.bank_1
    ]

  def get_bank_depth_unit_vectors(
      self,
      bank: Sequence[crease_utils.Crease],
      opposite_bank: Sequence[crease_utils.Crease],
  ) -> list[vector.Vector]:
    """Gets grid unit depth vectors corresponding to the boundary."""
    grid_delta = 1 / self.grid_size
    depth_grid_unit_vectors = []
    for crease_1, crease_2 in zip(bank, opposite_bank, strict=True):
      if crease_1.is_horizontal and crease_1.p1.y < crease_2.p1.y:
        depth_grid_unit_vector = vector.Vector(x=0, y=grid_delta)
      elif crease_1.is_horizontal and crease_1.p1.y > crease_2.p1.y:
        depth_grid_unit_vector = vector.Vector(x=0, y=-grid_delta)
      elif crease_1.is_vertical and crease_1.p1.x < crease_2.p1.x:
        depth_grid_unit_vector = vector.Vector(x=grid_delta, y=0)
      elif crease_1.is_vertical and crease_1.p1.x > crease_2.p1.x:
        depth_grid_unit_vector = vector.Vector(x=-grid_delta, y=0)
      else:
        raise ValueError(
            f'Creases {crease_1} and {crease_2} are not horizontal or'
            ' vertical or are too close.'
        )
      depth_grid_unit_vectors.append(depth_grid_unit_vector)
    return depth_grid_unit_vectors

  @functools.cached_property
  def boundary_1(self) -> ShapingBoundary:
    """Gets the boundary corresponding to the first bank."""
    return ShapingBoundary(
        creases=self.bank_1,
        edge_unit_vectors=self.edge_unit_vectors,
        depth_unit_vectors=self.get_bank_depth_unit_vectors(
            self.bank_1, self.bank_2
        ),
    )

  @functools.cached_property
  def boundary_2(self) -> ShapingBoundary:
    """Gets the boundary corresponding to the second bank."""
    return ShapingBoundary(
        creases=self.bank_2,
        edge_unit_vectors=self.edge_unit_vectors,
        depth_unit_vectors=self.get_bank_depth_unit_vectors(
            self.bank_2, self.bank_1
        ),
    )

  @functools.cached_property
  def shorter_bank_index(self) -> shaping_types.BankIndex:
    """Gets the index of the shorter bank."""
    bank_1_length = sum([crease.length for crease in self.bank_1])
    bank_2_length = sum([crease.length for crease in self.bank_2])
    return 'FIRST' if bank_1_length <= bank_2_length else 'SECOND'

  @functools.cached_property
  def longer_bank_index(self) -> shaping_types.BankIndex:
    """Gets the index of the longer bank."""
    return 'SECOND' if self.shorter_bank_index == 'FIRST' else 'FIRST'

  def get_bank_from_index(
      self, index: shaping_types.BankIndex
  ) -> Sequence[crease_utils.Crease]:
    """Gets the bank corresponding to the given index."""
    match index:
      case 'FIRST':
        return self.bank_1
      case 'SECOND':
        return self.bank_2
      case _ as unreachable:
        typing.assert_never(unreachable)

  def get_boundary_from_index(
      self, index: shaping_types.BankIndex
  ) -> ShapingBoundary:
    """Gets the boundary corresponding to the given index."""
    match index:
      case 'FIRST':
        return self.boundary_1
      case 'SECOND':
        return self.boundary_2
      case _ as unreachable:
        typing.assert_never(unreachable)

  def get_inscribed_river_bank(
      self, bank_index: shaping_types.BankIndex, inscribe_depth: float
  ) -> list[crease_utils.Crease]:
    """Gets the inscribed river bank at depth inscribe_depth."""
    if inscribe_depth < 0:
      raise ValueError('Inscribe depth must be greater than 0 for river banks.')

    bank, opposite_bank = (
        (self.bank_2, self.bank_1)
        if bank_index == 'SECOND'
        else (self.bank_1, self.bank_2)
    )

    if inscribe_depth == 0:
      return list(bank)

    ridges = self.ridges
    depth_vectors = self.get_bank_depth_unit_vectors(bank, opposite_bank)
    bank_corners = self.get_bank_corners(bank_index)

    new_bank_tips = []
    if self.is_closed:
      bank_corners = bank_corners[1:] + bank_corners[:1]
    if not self.is_closed:
      new_bank_tips.append(
          (inscribe_depth * depth_vectors[0]).point_add(bank_corners[0])
      )
      new_bank_tips.append(
          (inscribe_depth * depth_vectors[-1]).point_add(bank_corners[-1])
      )
      bank_corners = bank_corners[1:-1]

    new_bank_corners = []
    for corner, ridge in zip(bank_corners, ridges, strict=True):
      new_bank_corners.append(
          _transport_point_along_crease(
              corner, ridge, math.sqrt(2) * inscribe_depth / self.grid_size
          )
      )

    if new_bank_tips:
      new_bank_corners.insert(0, new_bank_tips[0])
      new_bank_corners.append(new_bank_tips[-1])

    new_bank = []
    for corner_1, corner_2 in itertools.pairwise(new_bank_corners):
      new_bank.append(
          crease_utils.Crease(corner_1, corner_2, fold=crease_utils.Fold.HINGE)
      )
    if self.is_closed:
      new_bank.insert(
          0,
          crease_utils.Crease(
              new_bank_corners[-1],
              new_bank_corners[0],
              fold=crease_utils.Fold.HINGE,
          ),
      )

    return new_bank

  def shrink_along_bank(
      self,
      bank_index: shaping_types.BankIndex,
      inscribe_depth: float,
      copy_shaping_history: bool = True,
  ) -> 'ShapingRiver':
    """Shrinks the river inwards along the given bank."""
    if inscribe_depth >= self.width:
      raise ValueError(
          f'Inscribe depth {inscribe_depth} must be less than the river width'
          f' {self.width}.'
      )
    new_bank = self.get_inscribed_river_bank(bank_index, inscribe_depth)
    return ShapingRiver(
        name=self.name,
        bank_1=self.bank_1 if bank_index == 'SECOND' else new_bank,
        bank_2=new_bank if bank_index == 'SECOND' else self.bank_2,
        width=self.width - inscribe_depth,
        grid_size=self.grid_size,
        shaping_history=copy.copy(self.shaping_history)
        if copy_shaping_history
        else [],
    ).extend_creases(self.creases)

  def get_circumbscribed_river(
      self,
      width: float,
      name: str,
      outer_bank_index: shaping_types.BankIndex,
      copy_shaping_history: bool = True,
  ) -> 'ShapingRiver':
    """Divides the river into an outer river of width `width` and an inner river."""
    if width <= 0:
      raise ValueError(
          f'Circumscribing river width must be positive. Got {width}.'
      )
    inscribed_bank = self.get_inscribed_river_bank(outer_bank_index, width)
    outer_bank = self.get_bank_from_index(outer_bank_index)
    return ShapingRiver(
        name=name,
        bank_1=outer_bank if outer_bank_index == 'FIRST' else inscribed_bank,
        bank_2=inscribed_bank if outer_bank_index == 'FIRST' else outer_bank,
        width=width,
        grid_size=self.grid_size,
        shaping_history=copy.copy(self.shaping_history)
        if copy_shaping_history
        else [],
    ).extend_creases(self.creases)

  def rename_and_copy(self, name: str) -> 'ShapingRiver':
    """Renames and copies the river."""
    return dataclasses.replace(self, name=name)

  @functools.cached_property
  def borders_central_crease(self) -> bool:
    """Checks if any river bank corners lie on the central crease."""
    bank_corner_x_coords = (
        [crease.p1_init.x for crease in self.bank_1]
        + [crease.p1_init.x for crease in self.bank_2]
        + [self.bank_1[-1].p2_init.x, self.bank_2[-1].p2_init.x]
    )
    return any(math.isclose(x, CENTRAL_CREASE_X) for x in bank_corner_x_coords)

  @functools.cached_property
  def is_right_of_central_crease(self) -> bool:
    """Checks if the river is right of the central crease."""
    bank_1_points = [crease.p1_init for crease in self.bank_1] + [
        crease.p2_init for crease in self.bank_1
    ]
    bank_2_points = [crease.p1_init for crease in self.bank_2] + [
        crease.p2_init for crease in self.bank_2
    ]
    points = bank_1_points + bank_2_points
    return all(
        point_utils.in_range(point.x, CENTRAL_CREASE_X, 2) for point in points
    )

  def get_joint_bank_index(
      self,
      joint_name: str,
      stick_figure: sf_utils.StickFigure,
      flaps: list[ShapingFlap],
      rivers: list['ShapingRiver'],
  ) -> shaping_types.BankIndex:
    """Gets the index of the bank corresponding to the joint."""
    logging.debug(
        'Getting bank index for joint %s in river %s', joint_name, self.name
    )
    self_stick = stick_figure[self.name]
    start_joint_name, end_joint_name = self_stick.start, self_stick.end
    if joint_name not in (start_joint_name, end_joint_name):
      raise ValueError(
          f'Joint {joint_name} is not a joint in river {self.name}.'
      )

    neighbors_at_joint = [
        stick.name
        for stick in stick_figure.children
        if (stick.start == joint_name or stick.end == joint_name)
        and stick.name != self.name
    ]
    logging.debug('neighbors_at_joint: %s', neighbors_at_joint)
    for bank_index in ('FIRST', 'SECOND'):
      bank_index = cast(shaping_types.BankIndex, bank_index)
      bank = self.get_bank_from_index(bank_index)
      for neighbor_name in neighbors_at_joint:
        element = get_element_by_name(neighbor_name, flaps, rivers)
        if isinstance(element, ShapingFlap):
          is_adjacent = _check_edge_lists_overlap(
              list(bank), list(element.edges)
          )
        else:
          is_adjacent = _check_edge_lists_overlap(
              list(bank), list(element.bank_1)
          ) or _check_edge_lists_overlap(list(bank), list(element.bank_2))
        if is_adjacent:
          return bank_index
    raise ValueError(
        f'Could not determine bank index for joint {joint_name} in river'
        f' {self.name}.'
    )

  def get_outer_bank_index(
      self,
      stick_figure: sf_utils.StickFigure | None = None,
      root_name: str | None = None,
      flaps: list[ShapingFlap] | None = None,
      rivers: list['ShapingRiver'] | None = None,
  ) -> shaping_types.BankIndex:
    """Gets the index of the outer bank of the river from the stick figure."""
    if self._outer_bank_index is not None:
      return self._outer_bank_index

    assert stick_figure is not None, 'Stick figure must be provided.'
    assert root_name is not None, 'Root name must be provided.'
    assert flaps is not None, 'Flaps must be provided.'
    assert rivers is not None, 'Rivers must be provided.'
    self_stick = stick_figure[self.name]
    if self.name == root_name:
      return self.get_joint_bank_index(
          self_stick.end, stick_figure, flaps, rivers
      )
    else:
      marked_joints = set()
      for stick in tr.breadth_first_traversal(
          stick_figure, start_stick_name=root_name
      ):
        if stick.name == self.name:
          assert stick.start in marked_joints or stick.end in marked_joints, (
              f'Joint {stick.start} or {stick.end} not found in marked joints'
              f' {marked_joints}.'
          )
          outer_joint_name = (
              stick.start if stick.start in marked_joints else stick.end
          )
          return self.get_joint_bank_index(
              outer_joint_name, stick_figure, flaps, rivers
          )
        else:
          marked_joints.update([stick.start, stick.end])

    raise RuntimeError(
        f'Could not find outer bank index for river {self.name}.'
    )

  def update_outer_bank_index(
      self, outer_bank_index: shaping_types.BankIndex
  ) -> 'ShapingRiver':
    """Updates the outer bank index of the river."""
    return dataclasses.replace(self, _outer_bank_index=outer_bank_index)

  @functools.cached_property
  def covering_rectangles(self) -> Sequence[rect_utils.Rectangle]:
    """Gets the rectangles covering the river."""
    rectangles = []
    for i, (crease_1, crease_2) in enumerate(
        zip(self.bank_1, self.bank_2, strict=True)
    ):
      points = [
          crease_1.p1_init,
          crease_1.p2_init,
          crease_2.p1_init,
          crease_2.p2_init,
      ]
      x_min, x_max = min(p.x for p in points), max(p.x for p in points)
      y_min, y_max = min([p.y for p in points]), max([p.y for p in points])
      x_start, y_start = x_min, y_min
      x_len, y_len = x_max - x_min, y_max - y_min
      rectangles.append(
          rect_utils.Rectangle(
              x_start=x_start,
              y_start=y_start,
              x_len=x_len,
              y_len=y_len,
              name=f'{self.name}_{i}',
          )
      )
    return rectangles

  @functools.cached_property
  def covering_legacy_rectangles(self) -> Sequence[shaping_utils.Rectangle]:
    """Gets the legacy rectangles covering the river."""
    return [
        shaping_utils.Rectangle(
            name=rect.name,
            x_start=rect.x_start,
            y_start=rect.y_start,
            x_len=rect.x_len,
            y_len=rect.y_len,
        )
        for rect in self.covering_rectangles
    ]

  def is_internal_point(
      self, point: point_utils.Point, strict: bool = False
  ) -> bool:
    """Checks if the river contains the given point."""
    covering_rectangles = self.covering_rectangles
    return any(
        rectangle.is_internal_point(point, strict=strict)
        for rectangle in covering_rectangles
    )

  def intersects_crease(self, crease: crease_utils.Crease) -> bool:
    """Checks if the river intersects the given crease."""
    contains_crease = self.is_internal_point(
        crease.p1_init
    ) and self.is_internal_point(crease.p2_init)
    intersects_border = any(
        rect.intersects_crease(crease) for rect in self.covering_rectangles
    )
    return contains_crease or intersects_border

  @property
  def elevations(self) -> set[int]:
    """Returns the set of elevations of axial creases in the river."""
    return {c.elevation for c in self.creases if c.elevation is not None}

  def extend_creases(
      self, creases: Sequence[crease_utils.Crease]
  ) -> 'ShapingRiver':
    """Appends creases to the river's creases attribute if they intersect."""
    intersecting_creases = [
        crease for crease in creases if self.intersects_crease(crease)
    ]
    self.creases.extend(intersecting_creases)
    return self


ShapingElement = ShapingFlap | ShapingRiver


def avg_bank_y(bank: Sequence[crease_utils.Crease]) -> float:
  """Gets the average y coordinate of the creases in the bank."""
  bank_corners_ys = [crease.p1_init.y for crease in bank] + [bank[-1].p2_init.y]
  return sum(bank_corners_ys) / len(bank_corners_ys)


def _check_edge_lists_overlap(
    edge_list_1: Sequence[crease_utils.Crease],
    edge_list_2: Sequence[crease_utils.Crease],
) -> bool:
  """Checks if two edge lists overlap."""
  return any(
      edge_1.is_overlapping(edge_2)
      for edge_1, edge_2 in itertools.product(edge_list_1, edge_list_2)
  )


def get_element_by_name(
    name: str,
    flaps: list[ShapingFlap],
    rivers: list[ShapingRiver],
) -> ShapingElement:
  """Gets the element corresponding to the given name."""
  element_dict = {element.name: element for element in flaps + rivers}
  if name not in element_dict:
    raise ValueError(f'Element {name} not found.')
  return element_dict[name]


def get_unassigned_hinges_from_elements(
    flaps: Sequence[shaping_utils.Rectangle | ShapingFlap],
    rivers: Sequence[ShapingRiver],
    return_edges: bool = False,
    round_creases: bool = False,
    rounding_digits: int = 5,
) -> list[packing_types.HingeInfo]:
  """Gets unassigned hinges from flaps and rivers."""
  border_creases = []
  for flap in flaps:
    edges = flap.edges
    edges = shaping_utils.restrict_creases_to_paper(list(edges))
    border_creases.extend([(edge, flap.name) for edge in edges])

  for river in rivers:
    for edge in itertools.chain(river.bank_1, river.bank_2):
      border_creases.append((edge, river.name))

  if return_edges:
    edges = []
    for edge in border_creases:
      edges.append(
          packing_types.HingeInfo(
              coordinate=edge[0],
              assigned=False,
              flap_names=tuple([edge[1]]),
          )
      )
    return edges

  hinges = []
  for crease_data_1, crease_data_2 in itertools.combinations(border_creases, 2):
    edge_1, name_1 = crease_data_1
    edge_2, name_2 = crease_data_2
    overlap = edge_1.get_overlap(
        edge_2, ndigits=rounding_digits if round_creases else None
    )
    if overlap:
      hinge = crease_utils.Crease(
          overlap.p1, overlap.p2, fold=crease_utils.Fold.HINGE
      )
      hinges.append(
          packing_types.HingeInfo(
              coordinate=hinge,
              assigned=False,
              flap_names=tuple([name_1, name_2]),
          )
      )

  return hinges


def assign_hinges(
    hinges: list[packing_types.HingeInfo],
    hinge_element_names: list[tuple[str, str]],
) -> list[packing_types.HingeInfo]:
  """Assigns hinges by element name pairs."""
  hinge_dict = {tuple(sorted(hinge.flap_names)): hinge for hinge in hinges}
  for flap_1, flap_2 in hinge_element_names:
    hinge_dict[tuple(sorted([flap_1, flap_2]))].assigned = True
  return hinges


def make_dummy_river(
    name: str,
    grid_size: int,
) -> ShapingRiver:
  """Makes a dummy river for a river in the packing model."""
  return ShapingRiver(
      name=name,
      bank_1=[
          crease_utils.Crease(
              point_utils.Point(x=0, y=0),
              point_utils.Point(x=1, y=0),
              fold=crease_utils.Fold.HINGE,
          )
      ],
      bank_2=[
          crease_utils.Crease(
              point_utils.Point(x=0, y=1),
              point_utils.Point(x=1, y=1),
              fold=crease_utils.Fold.HINGE,
          )
      ],
      width=grid_size,
      grid_size=grid_size,
  )


def convert_packing_rectangles_to_shaping_flaps(
    packing_rectangles: list[shaping_utils.Rectangle],
    packing_grid_size: int,
    solving_grid_size: int,
    base_cp: cp_utils.CreasePattern | None = None,
) -> list[ShapingFlap]:
  """Converts packing rectangles to shaping flaps."""
  assert packing_grid_size > 0, 'Packing grid size must be positive.'
  assert solving_grid_size > 0, 'Solved grid size must be positive.'
  scale_factor = solving_grid_size // packing_grid_size
  shaping_rectangles = [
      rect.normalize_rectangle(
          grid_size=solving_grid_size,
          scale_factor=scale_factor,
      )
      for rect in packing_rectangles
  ]
  shaping_flaps = [
      ShapingFlap.from_centers(
          circle_centers=rect.circle_centers,
          radius=rect.radius,
          grid_size=solving_grid_size,
          name=rect.name,
      )
      for rect in shaping_rectangles
  ]
  if base_cp is not None:
    for flap in shaping_flaps:
      flap.extend_creases(base_cp.creases)
  return shaping_flaps


def convert_packing_rivers_to_shaping_rivers(
    packing_rivers: dict[
        str,
        river_utils.River | river_utils.LShapedRiver | river_utils.HuggingRiver,
    ],
    packing_grid_size: int,
    solving_grid_size: int,
    base_cp: cp_utils.CreasePattern | None = None,
) -> list[ShapingRiver]:
  """Converts packing rivers to shaping rivers."""
  shaping_rivers = []
  scale_factor = solving_grid_size // packing_grid_size
  for name, river in packing_rivers.items():
    try:
      shaping_river = ShapingRiver.from_packing_river(name=name, river=river)
    except (ValueError, AssertionError) as e:
      logging.warning(
          'Failed to convert river %s. Skipping. Error: %s', name, e
      )
      shaping_river = make_dummy_river(name, packing_grid_size)

    width = shaping_river.width * scale_factor
    shaping_river = dataclasses.replace(
        shaping_river, grid_size=solving_grid_size, width=width
    )

    if base_cp is not None:
      shaping_river = shaping_river.extend_creases(base_cp.creases)

    shaping_rivers.append(shaping_river)

  return shaping_rivers


def convert_packing_elements_to_shaping_elements(
    packing_rectangles: list[shaping_utils.Rectangle],
    packing_rivers: dict[
        str,
        river_utils.River | river_utils.LShapedRiver | river_utils.HuggingRiver,
    ],
    packing_grid_size: int,
    solving_grid_size: int,
    base_cp: cp_utils.CreasePattern | None = None,
) -> tuple[list[ShapingFlap], list[ShapingRiver]]:
  """Converts packing elements to shaping elements."""
  shaping_flaps = convert_packing_rectangles_to_shaping_flaps(
      packing_rectangles,
      packing_grid_size,
      solving_grid_size,
      base_cp=base_cp,
  )

  shaping_rivers = convert_packing_rivers_to_shaping_rivers(
      packing_rivers, packing_grid_size, solving_grid_size, base_cp=base_cp
  )

  return shaping_flaps, shaping_rivers


def adjust_shaping_elements_for_central_crease_removal(
    shaping_flaps: list[ShapingFlap],
    shaping_rivers: list[ShapingRiver],
    outer_bank_indices: dict[str, shaping_types.BankIndex],
    central_crease_removed: bool,
) -> tuple[list[ShapingFlap], list[ShapingRiver]]:
  """Adjusts shaping elements when the central crease is removed."""
  if not central_crease_removed:
    return shaping_flaps, shaping_rivers

  new_shaping_flaps = []
  for flap in shaping_flaps:
    if flap.borders_central_crease:
      if crease_utils.approx_lte(float(flap.grid_unit_radius), 1.0):
        continue
      flap = flap.shrink(1)
    new_shaping_flaps.append(flap)

  new_shaping_rivers = []
  for river in shaping_rivers:
    if river.borders_central_crease:
      outer_bank_index = outer_bank_indices[river.name]
      river = river.shrink_along_bank(outer_bank_index, 1)
    new_shaping_rivers.append(river)

  return new_shaping_flaps, new_shaping_rivers


def get_shaping_boundary(
    element: ShapingElement,
    boundary_index: shaping_types.BankIndex | None = None,
) -> ShapingBoundary:
  """Gets the boundary of the shaping element."""
  match element:
    case ShapingFlap():
      return get_flap_boundary(element, element.grid_size)
    case ShapingRiver():
      assert (
          boundary_index is not None
      ), 'Boundary index must be provided for rivers.'
      return element.get_boundary_from_index(boundary_index)
    case _ as unreachable:
      typing.assert_never(unreachable)


def _check_crease_overlap(
    crease: crease_utils.Crease,
    creases: Sequence[crease_utils.Crease],
) -> bool:
  """Returns whether the crease overlaps with any of the creases in the list."""
  return any(crease.is_overlapping(crease_2) for crease_2 in creases)


def has_assigned_hinge(
    element: ShapingElement,
    hinge_info_list: Sequence[packing_types.HingeInfo],
    bank_index: shaping_types.BankIndex | None = None,
) -> bool:
  """Returns whether the element has an assigned hinge."""
  match element:
    case ShapingFlap():
      creases = element.edges
    case ShapingRiver() if bank_index is not None:
      creases = element.get_boundary_from_index(bank_index).creases
    case ShapingRiver():
      creases = list(itertools.chain(element.bank_1, element.bank_2))
    case _ as unreachable:
      typing.assert_never(unreachable)

  for hinge_info in hinge_info_list:
    if not hinge_info.assigned or hinge_info.coordinate is None:
      continue
    if _check_crease_overlap(hinge_info.coordinate, creases):
      return True
  return False


def flip_bank_index(
    bank_index: shaping_types.BankIndex,
) -> shaping_types.BankIndex:
  """Returns the opposite bank index."""
  match bank_index:
    case 'FIRST':
      return 'SECOND'
    case 'SECOND':
      return 'FIRST'
    case _ as unreachable:
      typing.assert_never(unreachable)


def _get_unmerged_to_merged_stick_name_dict(
    merged_to_unmerged_stick_name_dict: dict[str, list[str]],
) -> dict[str, str]:
  """Returns a dictionary mapping unmerged stick names to merged stick names."""
  unmerged_to_merged_stick_name_dict = {}
  for merged_name, unmerged_names in merged_to_unmerged_stick_name_dict.items():
    for unmerged_name in unmerged_names:
      unmerged_to_merged_stick_name_dict[unmerged_name] = merged_name
  return unmerged_to_merged_stick_name_dict


def get_unfolded_root_vector(
    root_element: ShapingElement,
    stick_figure: sf_utils.StickFigure,
) -> tuple[point_utils.Point, point_utils.Point]:
  """Gets the start and end points of the root element in the unfolded pattern."""
  bank_index = None
  if isinstance(root_element, ShapingRiver):
    bank_index = flip_bank_index(root_element.get_outer_bank_index())
  boundary = get_shaping_boundary(root_element, boundary_index=bank_index)

  tips = boundary.tips
  cur_p, edge_index = boundary.creases[0].p1_init, 0
  for _ in range(boundary.num_walk_steps_upper_bound(root_element.grid_size)):
    cur_p, edge_index = boundary.walk(cur_p, edge_index, root_element.grid_size)
    if cur_p.on_paper and all(not cur_p.isclose(tip) for tip in tips):
      break
  assert cur_p.on_paper, f'Root element is not on paper: {root_element}.'
  start = cur_p
  edge_index = boundary.get_edge_index(start)
  assert (
      edge_index is not None
  ), f'Edge index not found for start point {start} on boundary: {boundary}.'
  depth_vector = boundary.depth_unit_vectors[edge_index]
  end = depth_vector.point_add(start)

  if isinstance(root_element, ShapingFlap):
    merged_stick_figure, merged_to_unmerged_stick_name_dict = (
        stick_figure.merge_linear_chains()
    )
    unmerged_to_merged_stick_name_dict = (
        _get_unmerged_to_merged_stick_name_dict(
            merged_to_unmerged_stick_name_dict
        )
    )
    merged_root_name = unmerged_to_merged_stick_name_dict.get(
        root_element.name, root_element.name
    )
    merged_root_stick = merged_stick_figure[merged_root_name]
    assert (
        stick_figure.joints[merged_root_stick.start].degree == 1
        or stick_figure.joints[merged_root_stick.end].degree == 1
    ), f'Flap {merged_root_stick.name} does not have a degree 1 joint.'
    if stick_figure.joints[merged_root_stick.start].degree == 1:
      start, end = end, start

  return start, end


def visualize_shaping_elements(
    shaping_flaps: Sequence[ShapingFlap],
    shaping_rivers: Sequence[ShapingRiver],
    grid_size: int = -1,
    figsize: tuple[float, float] = (10, 10),
) -> tuple[plt.Figure, plt.Axes]:
  """Visualizes the boundaries of shaping flaps and rivers."""
  fig, ax = plt.subplots(figsize=figsize)

  all_xs: list[float] = [0.0, 1.0]
  all_ys: list[float] = [0.0, 1.0]
  for flap in shaping_flaps:
    for c in flap.corners:
      all_xs.append(c.x)
      all_ys.append(c.y)
  for river in shaping_rivers:
    for bank in (river.bank_1, river.bank_2):
      for crease in bank:
        all_xs.extend([crease.p1_init.x, crease.p2_init.x])
        all_ys.extend([crease.p1_init.y, crease.p2_init.y])
  margin = 0.05
  ax.set_xlim(min(all_xs) - margin, max(all_xs) + margin)
  ax.set_ylim(min(all_ys) - margin, max(all_ys) + margin)
  ax.set_aspect('equal', adjustable='box')

  paper_xs = [0, 1, 1, 0, 0]
  paper_ys = [0, 0, 1, 1, 0]
  ax.plot(paper_xs, paper_ys, color='black', lw=2, zorder=0, label='paper')

  colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
  color_index = 0

  for flap in shaping_flaps:
    color = colors[color_index % len(colors)]
    color_index += 1
    corners = flap.corners
    xs = [c.x for c in corners] + [corners[0].x]
    ys = [c.y for c in corners] + [corners[0].y]
    ax.plot(xs, ys, color=color, lw=2, zorder=1, label=f'flap: {flap.name}')
    ax.text(
        flap.x_start + flap.x_len / 2.0,
        flap.y_start + flap.y_len / 2.0,
        flap.name,
        ha='center',
        va='center',
        fontsize=12,
        color='black',
        fontweight='bold',
    )

  for river in shaping_rivers:
    color = colors[color_index % len(colors)]
    color_index += 1

    bank_1_xs = [c.p1_init.x for c in river.bank_1] + [
        river.bank_1[-1].p2_init.x
    ]
    bank_1_ys = [c.p1_init.y for c in river.bank_1] + [
        river.bank_1[-1].p2_init.y
    ]
    ax.plot(
        bank_1_xs,
        bank_1_ys,
        color=color,
        lw=2,
        zorder=2,
        label=f'river: {river.name}',
    )

    bank_2_xs = [c.p1_init.x for c in river.bank_2] + [
        river.bank_2[-1].p2_init.x
    ]
    bank_2_ys = [c.p1_init.y for c in river.bank_2] + [
        river.bank_2[-1].p2_init.y
    ]
    ax.plot(bank_2_xs, bank_2_ys, color=color, lw=2, ls='--', zorder=2)

    rects = river.covering_rectangles
    mid_rect = rects[len(rects) // 2]
    ax.text(
        mid_rect.x_start + mid_rect.x_len / 2.0,
        mid_rect.y_start + mid_rect.y_len / 2.0,
        river.name,
        ha='center',
        va='center',
        fontsize=12,
        color='black',
        fontweight='bold',
    )

  if grid_size > 0:
    tick_interval = 1 / grid_size
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_interval))
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.grid(True, ls='--', color='gray', alpha=0.7, zorder=-1)
  else:
    ax.grid(False)
    ax.axis('off')

  ax.invert_yaxis()
  ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  fig.tight_layout()
  return fig, ax
