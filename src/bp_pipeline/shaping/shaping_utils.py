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

"""Utilities for shaping."""

from collections.abc import Sequence
import dataclasses
import enum
import functools
import hashlib
import itertools
import math
from typing import Any, Optional, Self, TypedDict

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import point as point_utils
from . import base_rectangle as rect_utils
from . import vector

EPS = crease_utils.DEFAULT_TOLERANCE  # threshold


class FlapType(enum.Enum):
  INTERNAL = 'Internal'
  EDGE = 'Edge'
  CORNER = 'Corner'


class RectangleDict(TypedDict):
  """A dictionary representation of a Rectangle."""

  name: str
  bounds: list[float]
  circle_centers: list[float]


@dataclasses.dataclass(kw_only=True, frozen=True)
class Rectangle:
  """A rectangle in the box-pleating grid."""

  name: str
  x_start: float
  y_start: float
  x_len: float
  y_len: float
  circle_centers: Sequence[point_utils.Point] | None = None
  require_on_paper: bool = False
  radius: float | None = None

  def __post_init__(self):
    if self.require_on_paper and not all(c.on_paper for c in self.corners):
      raise ValueError(
          f'Rectangle {self.name} with corners {self.corners}'
          ' is not on the paper.'
      )
    if self.circle_centers is not None and not isinstance(
        self.circle_centers, tuple
    ):
      object.__setattr__(self, 'circle_centers', tuple(self.circle_centers))
    if (
        self.circle_centers is not None
        and not 1 <= len(self.circle_centers) <= 2
    ):
      raise ValueError(
          f'Rectangle {self.name} must have either None, 1, or 2 circle'
          ' centers.'
      )
    if self.circle_centers is not None and len(self.circle_centers) == 2:
      c1, c2 = self.circle_centers
      if c1.isclose(c2):
        raise ValueError(
            f'Circle centers {self.circle_centers} are not distinct.'
        )
    if self.radius is None:
      object.__setattr__(
          self, 'radius', min(self.x_len, self.y_len) * math.sqrt(2) / 2
      )
    if sum([self.is_square, self.is_wide, self.is_tall]) != 1:
      raise ValueError(
          'Rectangle must be exactly one of square, wide, or tall.'
      )

  def asdict(self) -> RectangleDict:
    """Converts a Rectangle to a dict for S2 serialization."""
    cc = []
    if hasattr(self, 'circle_centers') and self.circle_centers:
      for p in self.circle_centers:
        cc.append(p.x)
        cc.append(p.y)
    return {
        'name': self.name,
        'bounds': [self.x_start, self.y_start, self.x_len, self.y_len],
        'circle_centers': cc,
    }

  @classmethod
  def from_dict(cls, r_dict: RectangleDict) -> Self:
    """Converts a dictionary to a Rectangle object."""
    cc = []
    if r_dict['circle_centers']:
      it = iter(r_dict['circle_centers'])
      cc = [
          point_utils.Point(x=x, y=y, require_on_paper=False)
          for x, y in zip(it, it)
      ]
    return cls(
        name=r_dict['name'],
        x_start=r_dict['bounds'][0],
        y_start=r_dict['bounds'][1],
        x_len=r_dict['bounds'][2],
        y_len=r_dict['bounds'][3],
        circle_centers=cc if cc else None,
    )

  @functools.cached_property
  def is_square(self) -> bool:
    """Whether the rectangle is a square."""
    return math.isclose(
        self.x_len, self.y_len, abs_tol=point_utils.DEFAULT_TOLERANCE
    )

  @functools.cached_property
  def is_wide(self) -> bool:
    """Whether the rectangle is wider than it is tall."""
    return self.x_len > self.y_len and not self.is_square

  @functools.cached_property
  def is_tall(self) -> bool:
    """Whether the rectangle is taller than it is wide."""
    return self.y_len > self.x_len and not self.is_square

  def get_borders_touched(
      self, grid_size: int
  ) -> tuple[bool, bool, bool, bool]:
    """Determines which borders of the grid the rectangle touches or extends beyond."""
    if overlaps_paper(
        self.x_start, self.y_start, self.x_len, self.y_len, grid_size
    ):
      touch_left = self.x_start <= point_utils.DEFAULT_TOLERANCE
      touch_right = (
          self.x_start + self.x_len >= grid_size - point_utils.DEFAULT_TOLERANCE
      )
      touch_top = self.y_start <= point_utils.DEFAULT_TOLERANCE
      touch_bottom = (
          self.y_start + self.y_len >= grid_size - point_utils.DEFAULT_TOLERANCE
      )
      return touch_left, touch_right, touch_top, touch_bottom
    return False, False, False, False

  @property
  def center(self) -> tuple[float, float]:
    """Returns the center of the rectangle."""
    return (
        self.x_start + self.x_len / 2.0,
        self.y_start + self.y_len / 2.0,
    )

  @property
  def top_left(self) -> tuple[float, float]:
    """Returns the top-left corner as a tuple."""
    return (self.x_start, self.y_start)

  @property
  def top_right(self) -> tuple[float, float]:
    """Returns the top-right corner as a tuple."""
    return (self.x_start + self.x_len, self.y_start)

  @property
  def bottom_right(self) -> tuple[float, float]:
    """Returns the bottom-right corner as a tuple."""
    return (self.x_start + self.x_len, self.y_start + self.y_len)

  @property
  def bottom_left(self) -> tuple[float, float]:
    """Returns the bottom-left corner as a tuple."""
    return (self.x_start, self.y_start + self.y_len)

  @property
  def x_end(self) -> float:
    """Returns the x-coordinate of the right edge of the rectangle."""
    return self.x_start + self.x_len

  @property
  def y_end(self) -> float:
    """Returns the y-coordinate of the bottom edge of the rectangle."""
    return self.y_start + self.y_len

  @property
  def area(self) -> float:
    """Returns the area of the rectangle."""
    return self.x_len * self.y_len

  def contains(self, other: 'Rectangle', eps: float = 1e-9) -> bool:
    """Checks if the rectangle contains another rectangle."""
    return (
        self.x_start <= other.x_start + eps
        and self.x_end >= other.x_end - eps
        and self.y_start <= other.y_start + eps
        and self.y_end >= other.y_end - eps
    )

  def treat_as_wide_on_border(self, grid_size: int) -> bool:
    """For a square on a border, determines if it should be treated as wide."""
    if not self.is_square:
      return self.is_wide

    on_left, on_right, on_top, on_bottom = self.is_center_on_border(grid_size)
    if (on_left or on_right) and (on_top or on_bottom):
      return False
    if on_left or on_right:
      return False
    if on_top or on_bottom:
      return True

    touch_left, touch_right, _, _ = self.get_borders_touched(grid_size)
    if touch_left or touch_right:
      return True

    return False

  def is_center_on_border(
      self, grid_size: int
  ) -> tuple[bool, bool, bool, bool]:
    """Checks if the center of the rectangle is on any of the four borders."""
    center_x, center_y = self.center
    on_left = math.isclose(center_x, 0, abs_tol=point_utils.DEFAULT_TOLERANCE)
    on_right = math.isclose(
        center_x, grid_size, abs_tol=point_utils.DEFAULT_TOLERANCE
    )
    on_top = math.isclose(center_y, 0, abs_tol=point_utils.DEFAULT_TOLERANCE)
    on_bottom = math.isclose(
        center_y, grid_size, abs_tol=point_utils.DEFAULT_TOLERANCE
    )
    return on_left, on_right, on_top, on_bottom

  def is_center_on_corner(self, grid_size: int) -> bool:
    """Checks if the center of the rectangle is on any of the four corners."""
    on_left, on_right, on_top, on_bottom = self.is_center_on_border(grid_size)
    return (on_left or on_right) and (on_top or on_bottom)

  def get_circle_centers(self, grid_size: int) -> Sequence[point_utils.Point]:
    """Computes circle centers for the rectangle."""
    if self.is_square:
      return (
          point_utils.Point(
              x=self.x_start + self.x_len / 2.0,
              y=self.y_start + self.y_len / 2.0,
              require_on_paper=False,
          ),
      )

    touch_left, touch_right, touch_top, touch_bottom = self.get_borders_touched(
        grid_size
    )

    if self.is_wide:
      center_y = self.y_start + self.y_len / 2.0
      c1_x = self.x_start + self.y_len / 2.0
      c2_x = self.x_start + self.x_len - self.y_len / 2.0
      if touch_left:
        c1_x = max(self.x_start, 0)
      if touch_right:
        c2_x = min(self.x_start + self.x_len, float(grid_size))
      p1 = point_utils.Point(x=c1_x, y=center_y, require_on_paper=False)
      p2 = point_utils.Point(x=c2_x, y=center_y, require_on_paper=False)
      return (p1, p2)
    elif self.is_tall:
      center_x = self.x_start + self.x_len / 2.0
      c1_y = self.y_start + self.x_len / 2.0
      c2_y = self.y_start + self.y_len - self.x_len / 2.0
      if touch_top:
        c1_y = max(self.y_start, 0)
      if touch_bottom:
        c2_y = min(self.y_start + self.y_len, float(grid_size))
      p1 = point_utils.Point(x=center_x, y=c1_y, require_on_paper=False)
      p2 = point_utils.Point(x=center_x, y=c2_y, require_on_paper=False)
      return (p1, p2)
    else:
      raise ValueError('Rectangle must be square, wide, or tall.')

  @classmethod
  def from_ridges_and_radius(
      cls,
      name: str,
      ridges: Sequence[crease_utils.Crease],
      radius: float | None,
  ) -> 'Rectangle':
    """Create a rectangle from a sequence of ridges and an (optional) radius."""
    merged_ridges = crease_utils.merge_creases(ridges)

    if any(ridge.is_horizontal or ridge.is_vertical for ridge in merged_ridges):
      horizontal_or_vertical_ridges = [
          ridge
          for ridge in merged_ridges
          if ridge.is_horizontal or ridge.is_vertical
      ]
      if len(horizontal_or_vertical_ridges) != 1:
        raise ValueError(
            'Wide/tall flaps must have exactly 1 horizontal/vertical ridge:'
            f' {merged_ridges}.'
        )
      horizontal_or_vertical_ridge = horizontal_or_vertical_ridges[0]
      circle_centers = [
          horizontal_or_vertical_ridge.p1,
          horizontal_or_vertical_ridge.p2,
      ]

      if radius is None:
        diagonals = [ridge for ridge in merged_ridges if ridge.is_diagonal]
        if len(diagonals) < 1:
          raise ValueError(
              f'Wide flaps must have 1 diagonal ridge: {merged_ridges}.'
          )
        radius = diagonals[0].length
    else:
      if not all(ridge.is_diagonal for ridge in merged_ridges):
        raise ValueError(
            f'All ridges for square must have slope 1 or -1: {merged_ridges}.'
        )
      if len(merged_ridges) != 2:
        raise ValueError(
            f'2 ridges are required to form a rectangle: {merged_ridges}.'
        )
      ridge_1, ridge_2 = merged_ridges
      center = ridge_1.vertex_with_crease(
          ridge_2,
          include_external=True,
          extend_lines=True,
      )
      if center is None:
        raise ValueError(
            f'No intersection found between ridges {merged_ridges[:2]}!'
        )
      circle_centers = [center]

      if radius is None:
        radius = max(
            point_utils.distance(center, p) for p in [ridge_1.p1, ridge_1.p2]
        )

    rectangle = cls.from_centers(
        circle_centers=circle_centers,
        radius=radius,
        name=name,
    )

    rect_ridges = rectangle.get_ridges()
    for ridge in merged_ridges:
      if not any(ridge.is_within(rect_ridge) for rect_ridge in rect_ridges):
        raise ValueError(
            f'(Merged) ridges {merged_ridges} do not match synthesized'
            f' rectangles ridges {rect_ridges}'
        )

    return rectangle

  @classmethod
  def from_ridges(
      cls, name: str, ridges: Sequence[crease_utils.Crease]
  ) -> 'Rectangle':
    """Create a rectangle from a sequence of ridges."""
    return cls.from_ridges_and_radius(name, ridges, radius=None)

  @classmethod
  def from_corners(
      cls,
      corners: Sequence[point_utils.Point],
      name: str,
      circle_centers: Sequence[point_utils.Point] | None = None,
      require_on_paper: bool = False,
  ) -> 'Rectangle':
    """Create a rectangle from its corners."""
    if len(corners) != 4:
      raise ValueError(f'Corners {corners} does not have 4 corners.')
    sorted_corners = sorted(
        corners, key=functools.cmp_to_key(rect_utils.compare_points)
    )
    rotated_corners = sorted_corners[:2] + sorted_corners[2:][::-1]
    if not (
        math.isclose(rotated_corners[0].y, rotated_corners[1].y, abs_tol=EPS)
        and math.isclose(
            rotated_corners[1].x, rotated_corners[2].x, abs_tol=EPS
        )
        and math.isclose(
            rotated_corners[2].y, rotated_corners[3].y, abs_tol=EPS
        )
        and math.isclose(
            rotated_corners[3].x, rotated_corners[0].x, abs_tol=EPS
        )
    ):
      raise ValueError(f'Corners {corners} do not form a rectangle.')
    return cls(
        name=name,
        x_start=rotated_corners[0].x,
        y_start=rotated_corners[0].y,
        x_len=rotated_corners[1].x - rotated_corners[0].x,
        y_len=rotated_corners[3].y - rotated_corners[0].y,
        circle_centers=circle_centers,
        require_on_paper=require_on_paper,
    )

  @classmethod
  def from_shaping_rect(
      cls,
      shaping_rect: rect_utils.Rectangle[float],
  ) -> 'Rectangle':
    """Create a rectangle from a shaping rectangle."""
    return cls.from_corners(
        shaping_rect.corners,
        name=shaping_rect.name,
    )

  @classmethod
  def from_clip_box(
      cls,
      name: str,
      clip_box: tuple[float, float, float, float],
      circle_centers: Sequence[point_utils.Point] | None = None,
      require_on_paper: bool = False,
  ) -> 'Rectangle':
    """Create a rectangle from a clip box."""
    x_min, y_min, x_max, y_max = clip_box
    if not (x_min < x_max and y_min < y_max):
      raise ValueError(f'Clip box {clip_box} is not a valid rectangle.')
    return cls(
        name=name,
        x_start=x_min,
        y_start=y_min,
        x_len=x_max - x_min,
        y_len=y_max - y_min,
        circle_centers=circle_centers,
        require_on_paper=require_on_paper,
    )

  @classmethod
  def from_centers(
      cls,
      circle_centers: Sequence[point_utils.Point],
      radius: float,
      name: str,
  ) -> 'Rectangle':
    """Create a rectangle from its centers."""
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
      return cls.from_corners(corners, name=name, circle_centers=circle_centers)
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
      )
    else:
      raise ValueError(
          f'Circle centers {circle_centers} must have 1 or 2 centers.'
      )

  @classmethod
  def shrinked_by_one(cls, rectangle: Self) -> Self:
    """Returns a rectangle that is shrunk by one grid unit."""
    return cls(
        name=rectangle.name,
        x_start=rectangle.x_start + 1,
        y_start=rectangle.y_start + 1,
        x_len=rectangle.x_len - 2,
        y_len=rectangle.y_len - 2,
        circle_centers=rectangle.circle_centers,
        require_on_paper=rectangle.require_on_paper,
    )

  def snap_to_grid(self, grid_size: int) -> 'Rectangle':
    """Snap rectangle to grid."""
    corners = [snap_point_to_grid(c, grid_size) for c in self.corners]
    return Rectangle.from_corners(
        corners,
        name=self.name,
        circle_centers=self.circle_centers,
        require_on_paper=self.require_on_paper,
    )

  @functools.cached_property
  def corners(
      self,
  ) -> tuple[
      point_utils.Point, point_utils.Point, point_utils.Point, point_utils.Point
  ]:
    """Get the corners of the rectangle."""
    return (
        point_utils.Point(
            x=self.x_start,
            y=self.y_start,
            require_on_paper=self.require_on_paper,
        ),
        point_utils.Point(
            x=self.x_start + self.x_len,
            y=self.y_start,
            require_on_paper=self.require_on_paper,
        ),
        point_utils.Point(
            x=self.x_start + self.x_len,
            y=self.y_start + self.y_len,
            require_on_paper=self.require_on_paper,
        ),
        point_utils.Point(
            x=self.x_start,
            y=self.y_start + self.y_len,
            require_on_paper=self.require_on_paper,
        ),
    )

  @functools.cached_property
  def edges(self) -> tuple[
      crease_utils.Crease,
      crease_utils.Crease,
      crease_utils.Crease,
      crease_utils.Crease,
  ]:
    """Get the edges of the rectangle."""
    corners = self.corners
    return tuple(  # pyrefly: ignore[bad-return]
        crease_utils.Crease(
            corners[i],
            corners[(i + 1) % 4],
            fold=crease_utils.Fold.HINGE,
        )
        for i in range(4)
    )

  def is_on_symmetry_line(self, grid_size: int) -> bool:
    """Checks if the rectangle has an edge that lies on a symmetry line."""
    return math.isclose(self.x_start, grid_size / 2) or math.isclose(
        self.x_start + self.x_len, grid_size / 2
    )

  def get_ridges(self) -> Sequence[crease_utils.Crease]:
    """Get the ridge creases of the rectangle."""
    if self.require_on_paper:
      raise ValueError(
          'Ridges for rectangles required to be on paper may extend off paper.'
      )
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

  def clip(
      self,
      xmin: float = 0.0,
      ymin: float = 0.0,
      xmax: float = 1.0,
      ymax: float = 1.0,
  ) -> Optional['Rectangle']:
    """Returns a new rectangle clipped to the given bounding box."""
    overlap_x_start = max(xmin, self.x_start)
    overlap_y_start = max(ymin, self.y_start)
    overlap_x_end = min(xmax, self.x_start + self.x_len)
    overlap_y_end = min(ymax, self.y_start + self.y_len)
    if overlap_x_end > overlap_x_start and overlap_y_end > overlap_y_start:
      return Rectangle(
          name=self.name,
          x_start=overlap_x_start,
          y_start=overlap_y_start,
          x_len=overlap_x_end - overlap_x_start,
          y_len=overlap_y_end - overlap_y_start,
          circle_centers=self.circle_centers,
          require_on_paper=self.require_on_paper,
      )
    return None

  def clip_to_paper(self) -> Optional['Rectangle']:
    """Clips the rectangle to the paper. Returns None if not on paper."""
    clipped_rect = self.clip(0, 0, 1, 1)
    if clipped_rect is None:
      return None
    return Rectangle(
        name=self.name,
        x_start=clipped_rect.x_start,
        y_start=clipped_rect.y_start,
        x_len=clipped_rect.x_len,
        y_len=clipped_rect.y_len,
        circle_centers=self.circle_centers,
        require_on_paper=True,
    )

  def is_corner_inside_or_on_border(self, width: int, height: int) -> bool:
    """Checks if at least one corner is within or on the border of (0,0,width,height)."""
    x_plus_w = self.x_start + self.x_len
    y_plus_h = self.y_start + self.y_len
    return (0 <= self.x_start <= width or 0 <= x_plus_w <= width) and (
        0 <= self.y_start <= height or 0 <= y_plus_h <= height
    )

  def overlaps(self, other: 'Rectangle') -> bool:
    """Checks if this rectangle overlaps with another rectangle."""
    is_separate = (
        (self.x_start + self.x_len <= other.x_start)
        or (self.x_start >= other.x_start + other.x_len)
        or (self.y_start + self.y_len <= other.y_start)
        or (self.y_start >= other.y_start + other.y_len)
    )
    return not is_separate

  def intersection_area(
      self, xmin: float, ymin: float, xmax: float, ymax: float
  ) -> float:
    """Calculates intersection area with a bounding box."""
    overlap_x_start = max(xmin, self.x_start)
    overlap_y_start = max(ymin, self.y_start)
    overlap_x_end = min(xmax, self.x_start + self.x_len)
    overlap_y_end = min(ymax, self.y_start + self.y_len)
    if overlap_x_end > overlap_x_start and overlap_y_end > overlap_y_start:
      return (overlap_x_end - overlap_x_start) * (
          overlap_y_end - overlap_y_start
      )
    return 0.0

  def get_unit_edge_vectors(self, grid_size: int) -> Sequence[vector.Vector]:
    """Get vectors pointing clockwise along rectangle edges with grid unit length."""
    grid_delta = 1 / grid_size
    return [
        grid_delta
        * (
            vector.Vector.from_point(self.corners[(i + 1) % 4])
            - vector.Vector.from_point(self.corners[i])
        ).normalize
        for i in range(4)
    ]

  def get_unit_depth_vectors(self, grid_size: int) -> Sequence[vector.Vector]:
    """Get vectors pointing into rectangle (orthogonal to respective edges) with grid unit length."""
    edge_vectors = self.get_unit_edge_vectors(grid_size)
    return [edge_vectors[(i + 1) % 4] for i in range(4)]

  def normalize_rectangle(
      self, grid_size: int, scale_factor: int = 1
  ) -> 'Rectangle':
    """Normalizes and rescales the rectangle attributes by the grid size and scale factor."""
    if grid_size <= 0:
      raise ValueError(f'Grid size {grid_size} must be positive.')

    if scale_factor < 1:
      raise ValueError(f'Scale factor {scale_factor} must be greater than 1.')

    if self.circle_centers:
      new_circle_centers = [
          point_utils.Point(
              x=(center.x / grid_size) * scale_factor,
              y=(center.y / grid_size) * scale_factor,
              require_on_paper=False,
          )
          for center in self.circle_centers
      ]
    else:
      new_circle_centers = None

    return Rectangle(
        name=self.name,
        x_start=(self.x_start / grid_size) * scale_factor,
        y_start=(self.y_start / grid_size) * scale_factor,
        x_len=(self.x_len / grid_size) * scale_factor,
        y_len=(self.y_len / grid_size) * scale_factor,
        circle_centers=new_circle_centers,
        require_on_paper=self.require_on_paper,
    )

  def flap_type(self, grid_size: int) -> FlapType:
    """Determines the type of flap based on its overlap with the grid borders."""
    clipped_rectangle = self.clip(
        xmin=0.0,
        ymin=0.0,
        xmax=float(grid_size),
        ymax=float(grid_size),
    )
    if not clipped_rectangle:
      raise ValueError('Rectangle is not on paper. Cannot determine flap type.')
    edges = [
        edge
        for edge in clipped_rectangle.normalize_rectangle(grid_size).edges
        if edge.is_overlapping_border
    ]
    overlapping_borders = len(edges)
    match overlapping_borders:
      case 0:
        return FlapType.INTERNAL
      case 1:
        return FlapType.EDGE
      case 2:
        if edges[0].intersect_at_tip(edges[1]):
          return FlapType.CORNER
        else:
          return FlapType.INTERNAL
      case 3:
        return FlapType.EDGE
      case 4:
        return FlapType.INTERNAL
      case _:
        raise ValueError(
            f'Overlapping borders {overlapping_borders} is not supported.'
        )

  def is_internal_point(
      self, p: point_utils.Point, strict: bool = False
  ) -> bool:
    """Checks if a point is inside the rectangle."""
    return point_utils.in_range(
        p.x, self.x_start, self.x_start + self.x_len, strict
    ) and point_utils.in_range(
        p.y, self.y_start, self.y_start + self.y_len, strict
    )

  def is_edge_point(
      self, p: point_utils.Point, extend_edges: bool = False
  ) -> bool:
    """Checks if a point is on an edge of the rectangle."""
    if extend_edges:
      x_end = self.x_start + self.x_len
      y_end = self.y_start + self.y_len
      return (
          math.isclose(p.x, self.x_start, abs_tol=EPS)
          or math.isclose(p.x, x_end, abs_tol=EPS)
          or math.isclose(p.y, self.y_start, abs_tol=EPS)
          or math.isclose(p.y, y_end, abs_tol=EPS)
      )

    return any(
        edge.point_status(p)
        in (crease_utils.PointStatus.TIP, crease_utils.PointStatus.INTERNAL)
        for edge in self.edges
    )

  def check_center_alignment(self) -> bool:
    """Check if the rectangle is centered around its circle centers."""
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

  def overlaps_border(self, crease: crease_utils.Crease) -> bool:
    """Checks if a crease overlaps the rectangle's border."""
    return any(crease.is_overlapping(edge) for edge in self.edges)

  @functools.cached_property
  def contains_central_crease(self) -> bool:
    """Checks if the rectangle contains the central crease (including boundary)."""
    central_crease_line_x = 0.5
    return point_utils.in_range(
        central_crease_line_x, self.x_start, self.x_start + self.x_len
    )

  @functools.cached_property
  def is_right_of_central_crease(self) -> bool:
    """Checks if the rectangle lies to the right of the central fold."""
    central_crease_line_x = 0.5
    return crease_utils.approx_lte(central_crease_line_x, self.x_start)

  @functools.cached_property
  def is_bottom_of_central_crease(self) -> bool:
    """Checks if the rectangle lies to the bottom of the central fold."""
    central_crease_line_y = 0.5
    return crease_utils.approx_gte(central_crease_line_y, self.y_start)

  def crease_is_within(
      self, crease: crease_utils.Crease, strict: bool = False
  ) -> bool:
    """Checks if a crease is within the rectangle."""
    return self.is_internal_point(crease.p1, strict) and self.is_internal_point(
        crease.p2, strict
    )

  def length_and_width_in_grid_units(self, grid_size: int) -> tuple[int, int]:
    """Returns the rectangle's length and width in grid units."""
    rect_length_in_grid_units = self.x_len * grid_size
    rect_width_in_grid_units = self.y_len * grid_size
    if not math.isclose(
        rect_length_in_grid_units, round(rect_length_in_grid_units), abs_tol=EPS
    ):
      raise ValueError(
          f'Rectangle length {rect_length_in_grid_units} is not integer!'
      )
    if not math.isclose(
        rect_width_in_grid_units, round(rect_width_in_grid_units), abs_tol=EPS
    ):
      raise ValueError(
          f'Rectangle width {rect_width_in_grid_units} is not integer!'
      )
    return round(rect_length_in_grid_units), round(rect_width_in_grid_units)

  def point_is_on_edge(self, p: point_utils.Point) -> bool:
    """Checks if a point is on the rectangle's border."""
    return any(
        edge.point_status(p) != crease_utils.PointStatus.EXTERNAL
        for edge in self.edges
    )

  def get_paper_border_overlapping_edges(self) -> list[crease_utils.Crease]:
    """Clips a rectangle to the paper's boundaries and returns border edges."""
    clipped_rect = self.clip_to_paper()
    if not clipped_rect:
      raise ValueError(
          f"Rectangle '{self.name}' is completely outside the paper."
      )
    return [edge for edge in clipped_rect.edges if edge.is_overlapping_border]


def overlaps_paper(
    x_start: float, y_start: float, x_len: float, y_len: float, grid_size: int
) -> bool:
  """Checks if a rectangle with unnormalized coordinates overlaps with paper."""
  return (
      x_start < grid_size
      and x_start + x_len > 0
      and y_start < grid_size
      and y_start + y_len > 0
  )


@dataclasses.dataclass
class River:
  pass


Flap = Rectangle
Element = Flap | River


def sign(x: float) -> float:
  """Returns the sign of a float."""
  if math.isclose(x, 0):
    return 0
  return 1 if x > 0 else -1


def get_sha256(string: str) -> str:
  """Returns the sha256 hash of a string."""
  return hashlib.sha256(string.encode('utf-8')).hexdigest()


def argmin(vals: Sequence[Any]) -> int | None:
  """Returns the index of the minimum value in a list."""
  return vals.index(min(vals)) if vals else None


def round_to_grid(num: float, grid_size: int, abs_tol: float = 1e-5) -> float:
  """Round float to multiple of 1 / grid_size."""
  closest_grid_point = round(num * grid_size) / grid_size
  if not math.isclose(num, closest_grid_point, abs_tol=abs_tol):
    raise ValueError(
        f'Num {num} is not close to {closest_grid_point} with tolerance'
        f' {abs_tol}!'
    )
  return closest_grid_point


def check_num_on_grid(
    num: float, grid_size: int, abs_tol: float = 1e-5
) -> bool:
  """Check if float is on the grid with grid size grid_size."""
  try:
    round_to_grid(num, grid_size, abs_tol)
    return True
  except ValueError:
    return False


def check_point_on_grid(
    p: point_utils.Point, grid_size: int, abs_tol: float = 1e-5
) -> bool:
  """Check if point is on the grid with grid size grid_size."""
  return check_num_on_grid(p.x, grid_size, abs_tol) and check_num_on_grid(
      p.y, grid_size, abs_tol
  )


def snap_point_to_grid(
    p: point_utils.Point,
    grid_size: int,
    snap_x: bool = True,
    snap_y: bool = True,
    abs_tol: float = 1e-5,
) -> point_utils.Point:
  """Round point to multiple of 1 / grid_size."""
  return point_utils.Point(
      x=round_to_grid(p.x, grid_size, abs_tol) if snap_x else p.x,
      y=round_to_grid(p.y, grid_size, abs_tol) if snap_y else p.y,
      require_on_paper=p.require_on_paper,
  )


def zigzag_angle_from_fold_angle(theta: float) -> float:
  """Computes zigzag angle from target fold angle."""
  if not (0 < theta < math.pi):
    raise ValueError(f'Theta {theta} is not in (0, pi).')
  return theta / 2


def zigzag_depth_from_fold_angle(theta: float) -> float:
  """Computes depth of zigzag from target fold angle (assuming unit base length)."""
  if not (0 < theta < math.pi):
    raise ValueError(f'Theta {theta} is not in (0, pi).')
  theta = zigzag_angle_from_fold_angle(theta)
  return float(math.sin(math.pi / 2 - theta) / math.sin(theta))


def fold_angle_from_zigzag_depth(depth: float) -> float:
  """Computes target fold angle from depth of zigzag (assuming unit base length)."""
  if depth < 0:
    raise ValueError(f'Depth {depth} is negative.')
  if math.isclose(depth, 0):
    return math.pi
  return 2 * math.atan(1 / depth)


def get_grid_unit_length(length: float, grid_size: int) -> float:
  """Returns the length of the crease in fractional grid units."""
  return length * grid_size


def get_integer_grid_unit_length(length: float, grid_size: int) -> int:
  """Returns the length of the crease in integer grid units."""
  grid_unit_length = round(length * grid_size)
  if not math.isclose(grid_unit_length / grid_size, length, abs_tol=1e-5):
    raise ValueError(f'Length {length} is not integer grid length!')
  return grid_unit_length


def integer_crease_grid_length(
    crease: crease_utils.Crease,
    grid_size: int,
) -> int:
  """Returns the length of the crease in integer grid units."""
  l = round(crease.length * grid_size)
  if not math.isclose(l / grid_size, crease.length, abs_tol=1e-5):
    raise ValueError(f'Crease {crease} is not integer grid length!')
  return l


def crease_grid_length(
    crease: crease_utils.Crease,
    grid_size: int,
) -> float:
  """Returns the length of the crease in integer grid units."""
  return crease.length * grid_size


def get_grid_nums_in_closed_interval(
    min_val: float,
    max_val: float,
    grid_size: int,
) -> Sequence[float]:
  """Returns the grid numbers in the interval [min_val, max_val]."""
  assert min_val < max_val
  min_val_grid = math.ceil(min_val * grid_size - EPS)
  max_val_grid = math.floor(max_val * grid_size + EPS)
  grid_nums = [
      i / grid_size for i in range(int(min_val_grid), int(max_val_grid) + 1)
  ]

  if not grid_nums:
    return grid_nums

  assert crease_utils.approx_lte(min_val, grid_nums[0]), (
      f'min_val {min_val} is not less than or equal to grid_nums[0]'
      f' {grid_nums[0]}!'
  )
  assert crease_utils.approx_lte(grid_nums[-1], max_val), (
      f'grid_nums[-1] {grid_nums[-1]} is not less than or equal to max_val'
      f' {max_val}!'
  )
  return grid_nums


def get_closest_grid_line_intersection_to_crease_point(
    crease: crease_utils.Crease,
    target_point: point_utils.Point,
    grid_size: int,
) -> point_utils.Point | None:
  """Returns the point on the crease intersecting a grid line closest to the target point."""
  assert (
      crease.point_status(target_point) != crease_utils.PointStatus.EXTERNAL
  ), f'Target point {target_point} is external to crease {crease}!'
  if crease.is_vertical:
    x = crease.p1.x
    min_y, max_y = sorted((crease.p1.y, crease.p2.y))
    y_grid_nums = get_grid_nums_in_closed_interval(min_y, max_y, grid_size)
    if not y_grid_nums:
      return None
    points = [
        point_utils.Point(x=x, y=y, require_on_paper=False) for y in y_grid_nums
    ]
    return min(points, key=lambda p: point_utils.distance(p, target_point))
  elif crease.is_horizontal:
    y = crease.p1.y
    min_x, max_x = sorted((crease.p1.x, crease.p2.x))
    x_grid_nums = get_grid_nums_in_closed_interval(min_x, max_x, grid_size)
    if not x_grid_nums:
      return None
    points = [
        point_utils.Point(x=x, y=y, require_on_paper=False) for x in x_grid_nums
    ]
    return min(points, key=lambda p: point_utils.distance(p, target_point))
  else:
    raise NotImplementedError(f'Crease {crease} is not vertical or horizontal!')


def compute_anchor_points_start(
    midpoint: point_utils.Point,
    grid_size: int,
    even_phase: bool,
    top_edge_vector: vector.Vector,
) -> point_utils.Point:
  """Compute anchor points walk starting point on rectangle top edge."""
  if check_num_on_grid(midpoint.x, grid_size):
    displacement = vector.Vector(x=0, y=0) if even_phase else top_edge_vector
    return displacement.point_add(midpoint)
  elif check_num_on_grid(midpoint.x, 2 * grid_size):
    displacement = (
        (1 / 2) * top_edge_vector if even_phase else -(1 / 2) * top_edge_vector
    )
    return displacement.point_add(midpoint)
  else:
    raise ValueError(f'Midpoint {midpoint} is not on primal or dual grids.')


def reflect_crease_over_ridge(
    crease: crease_utils.Crease,
    ridge: crease_utils.Crease,
    require_on_paper: bool = True,
    precomputed_intersection_point: point_utils.Point | None = None,
) -> vector.Vector:
  """Reflects a crease over a ridge crease."""
  if precomputed_intersection_point is None:
    intersection = ridge.vertex_with_crease(
        crease, include_external=not require_on_paper
    )
  else:
    intersection = precomputed_intersection_point

  if not intersection:
    raise ValueError(f'Zigzag {crease} does not intersect ridge {ridge}.')
  if crease.p1.isclose(intersection):
    crease_start = crease.p2
    crease_end = crease.p1
  elif crease.p2.isclose(intersection):
    crease_start = crease.p1
    crease_end = crease.p2
  else:
    raise ValueError(
        f'Zigzag {crease} does not intersect ridge {ridge} at tip.'
    )
  crease_vector = vector.Vector.from_point(
      crease_start
  ) - vector.Vector.from_point(crease_end)
  ridge_vector = vector.Vector(
      x=ridge.normed_vector[0], y=ridge.normed_vector[1]
  )
  ridge_component = crease_vector.dot(ridge_vector) * ridge_vector
  orthogonal_component = crease_vector - ridge_component
  reflection = crease_vector - 2 * orthogonal_component
  return reflection


def get_edge_index(
    p: point_utils.Point,
    corners: Sequence[point_utils.Point],
    edges: Sequence[crease_utils.Crease],
) -> int:
  """Get the index of the edge that a point is on."""
  is_corners = [p.isclose(c) for c in corners]
  if any(is_corners):
    return is_corners.index(True)
  point_status = [e.is_intermediate_point(p) for e in edges]
  if any(point_status):
    return point_status.index(True)
  raise ValueError(f'Point {p} is not on rectangle boundary.')


def check_corner_between_border_edge_and_interior_edge(
    corner: point_utils.Point,
    corners: Sequence[point_utils.Point],
    edges: Sequence[crease_utils.Crease],
) -> bool:
  """Check if a corner is incident between a paper border crease and an interior crease."""
  is_corners = [corner.isclose(c) for c in corners]
  if not any(is_corners):
    raise ValueError(f'Point {corner} is not a corner.')
  elif len(corners) != 4:
    raise ValueError(f'{corners} does not have 4 corners.')
  corner_index = is_corners.index(True)
  back_edge, front_edge = edges[(corner_index - 1) % 4], edges[corner_index]
  return back_edge.is_overlapping_border and not (
      front_edge.is_overlapping_border
  )


def get_crease_ridges_containment(
    start: point_utils.Point,
    end: point_utils.Point,
    ridges: Sequence[crease_utils.Crease],
) -> bool:
  """Check if a crease is contained entirely within a ridge."""
  crease = crease_utils.Crease(start, end, fold=crease_utils.Fold.MOUNTAIN)
  return any(crease.is_within(ridge) for ridge in ridges)


def preprocess_rect(
    rect: Rectangle,
    grid_size: int,
    target_flap_angle: float | None,
    target_zigzag_depth: float | None,
) -> tuple[
    Sequence[point_utils.Point],
    Sequence[crease_utils.Crease],
    Sequence[vector.Vector],
    Sequence[vector.Vector],
]:
  """Compute rectangle corners, rectangle edges, edge vectors, and depth vectors."""
  rect_corners = rect.corners
  rect_edges = rect.edges
  edge_vectors = rect.get_unit_edge_vectors(grid_size)
  if not (target_flap_angle is None or target_zigzag_depth is None):
    raise ValueError(
        'Only one of target_zigzag_angle or target_zigzag_depth can be'
        ' specified.'
    )
  if target_flap_angle is not None:
    depth = zigzag_depth_from_fold_angle(target_flap_angle)
  elif target_zigzag_depth is not None:
    if target_zigzag_depth <= 0:
      raise ValueError(
          f'Depth {target_zigzag_depth} must be at least 1 if target_flap_angle'
          ' is not specified.'
      )
    depth = target_zigzag_depth
  else:
    raise ValueError(
        'One of target_zigzag_angle or target_zigzag_depth must be specified.'
    )
  unit_depth_vectors = rect.get_unit_depth_vectors(grid_size)
  depth_vectors = [
      unit_depth_vector * depth for unit_depth_vector in unit_depth_vectors
  ]
  return rect_corners, rect_edges, edge_vectors, depth_vectors


def restrict_crease_to_rectangle(
    rect: Rectangle, crease: crease_utils.Crease
) -> crease_utils.Crease | None:
  """Restrict a single crease to the given rectangle."""
  border_creases = rect.edges
  if rect.is_internal_point(crease.p1) and rect.is_internal_point(crease.p2):
    return crease
  tip_candidates = [
      p for p in [crease.p1, crease.p2] if rect.is_internal_point(p)
  ]
  for border_crease in border_creases:
    overlap = crease.get_overlap(border_crease)
    if overlap is not None:
      tip_candidates = [overlap.p1, overlap.p2]
      break
    intersection = crease.vertex_with_crease(
        border_crease, include_external=True
    )
    if intersection is not None and not any(
        p.isclose(intersection) for p in tip_candidates
    ):
      tip_candidates.append(intersection)
  if len(tip_candidates) < 2:
    return None
  assert len(tip_candidates) == 2
  if tip_candidates[0].isclose(tip_candidates[1]):
    return None

  return dataclasses.replace(
      crease, p1_init=tip_candidates[0], p2_init=tip_candidates[1]
  )


def restrict_creases_to_rectangle(
    rect: Rectangle, creases: Sequence[crease_utils.Crease]
) -> list[crease_utils.Crease]:
  """Restrict each crease in sequence to the rectangle."""
  restricted_creases = [restrict_crease_to_rectangle(rect, c) for c in creases]
  filtered_creases = [
      crease for crease in restricted_creases if crease is not None
  ]
  return filtered_creases


def _merge_overlapping_creases(
    crease_1: crease_utils.Crease,
    crease_2: crease_utils.Crease,
) -> crease_utils.Crease:
  """Merges two overlapping creases by taking the longest possible crease."""
  if crease_1.isclose(crease_2):
    return crease_1
  return max(
      (
          crease_utils.Crease(
              p1,
              p2,
              fold=crease_1.fold,
              fold_percent=crease_1.fold_percent,
              elevation=crease_1.elevation,
              fold_step=crease_1.fold_step,
          )
          for p1, p2 in itertools.combinations(
              [crease_1.p1, crease_1.p2] + [crease_2.p1, crease_2.p2], 2
          )
          if not p1.isclose(p2)
      ),
      key=lambda c: c.length,
  )


def restrict_crease_to_rectangles(
    rects: Sequence[Rectangle], crease: crease_utils.Crease
) -> list[crease_utils.Crease]:
  """Restricts a crease to a list of rectangles and merges overlapping parts."""
  clipped_creases: list[crease_utils.Crease] = []
  for rect in rects:
    clipped = restrict_crease_to_rectangle(rect, crease)
    if clipped is not None:
      clipped_creases.append(clipped)

  if not clipped_creases:
    return []

  clipped_creases = sorted(
      clipped_creases, key=lambda c: (min(c.p1.x, c.p2.x), min(c.p1.y, c.p2.y))
  )

  merged_creases = []
  for crease in clipped_creases:
    merged = False
    for ind, merged_crease in enumerate(merged_creases):
      if merged_crease.intersect_at_tip(crease) or merged_crease.is_overlapping(
          crease
      ):
        merged_crease = _merge_overlapping_creases(merged_crease, crease)
        merged_creases[ind] = merged_crease
        merged = True
        break
    if not merged:
      merged_creases.append(crease)

  return merged_creases


def restrict_creases_to_rectangles(
    rects: Sequence[Rectangle], creases: Sequence[crease_utils.Crease]
) -> list[list[crease_utils.Crease]]:
  """Restricts a sequence of creases to a list of rectangles and merges overlapping parts."""
  clipped_creases: list[list[crease_utils.Crease]] = []
  for crease in creases:
    clipped_creases.append(restrict_crease_to_rectangles(rects, crease))
  return clipped_creases


def restrict_creases_to_paper(
    creases: Sequence[crease_utils.Crease],
) -> list[crease_utils.Crease]:
  """Restrict each crease in list to the paper ([0, 1]^2)."""
  return restrict_creases_to_rectangle(
      Rectangle(
          name='paper',
          x_start=0,
          y_start=0,
          x_len=1,
          y_len=1,
          require_on_paper=True,
      ),
      creases,
  )


def translate_creases(
    creases: Sequence[crease_utils.Crease], new_origin: point_utils.Point
) -> Sequence[crease_utils.Crease]:
  """Translate a crease pattern to a new origin."""
  new_origin = new_origin.relax_paper_requirement()
  new_creases = [
      crease_utils.Crease(
          crease.p1.relax_paper_requirement() - new_origin,
          crease.p2.relax_paper_requirement() - new_origin,
          fold=crease.fold,
          fold_percent=crease.fold_percent,
          fold_step=crease.fold_step,
      )
      for crease in creases
  ]
  return new_creases


def restrict_cp_to_rectangle(
    cp: cp_utils.CreasePattern, rect: Rectangle
) -> cp_utils.CreasePattern:
  """Restrict a crease pattern to a rectangle."""
  if not all(corner.on_paper for corner in rect.corners):
    raise ValueError('Rectangle must be on the paper.')
  restricted_creases = restrict_creases_to_rectangle(rect, cp.creases)
  creases = [
      crease
      for crease in restricted_creases
      if not rect.overlaps_border(crease)
  ]
  border_creases = [
      crease_utils.Crease(edge.p1, edge.p2, fold=crease_utils.Fold.BORDER)
      for edge in rect.edges
  ]
  return cp_utils.CreasePattern(creases=creases + border_creases)


def translate_cp(
    cp: cp_utils.CreasePattern, new_origin: point_utils.Point
) -> cp_utils.CreasePattern:
  """Translate a crease pattern to a new origin while clipping to the paper."""
  new_creases = translate_creases(cp.creases, new_origin)
  new_creases = [
      crease
      for crease in new_creases
      if crease.fold != crease_utils.Fold.BORDER
  ]
  paper = Rectangle(
      name='paper',
      x_start=0,
      y_start=0,
      x_len=1,
      y_len=1,
  )
  return restrict_cp_to_rectangle(
      cp_utils.CreasePattern(creases=new_creases), paper
  )


class FlapHierarchy:
  """A class to represent the hierarchy of flaps in a box pleating pattern."""

  def __init__(self, flaps: Sequence[Rectangle]):
    self.flaps_map: dict[str, Rectangle] = {f.name: f for f in flaps}
    self.parent_map: dict[str, Rectangle] = {}
    self._build_hierarchy(flaps)

  def _build_hierarchy(self, flaps):
    """Pre-calculates the largest parent (root container) for every flap."""
    for child in flaps:
      potential_parents = []

      for candidate in flaps:
        if child is candidate:
          continue
        if candidate.contains(child):
          potential_parents.append(candidate)

      if not potential_parents:
        self.parent_map[child.name] = child
      else:
        largest_parent = max(potential_parents, key=lambda r: r.area)
        self.parent_map[child.name] = largest_parent

  def get_root_flap_and_distance(self, name: str) -> tuple[str, float] | None:
    """Returns the root flap and the distance to it."""
    child = self.flaps_map.get(name)
    parent = self.parent_map.get(name)
    if not child or not parent:
      return None
    distance = (parent.radius - child.radius) / math.sqrt(2)  # pyrefly: ignore[unsupported-operation]
    return parent.name, distance
