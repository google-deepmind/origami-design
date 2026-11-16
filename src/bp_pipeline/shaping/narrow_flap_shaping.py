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

"""Functions for narrow flap shaping."""

import dataclasses
import enum
import logging
import math
import typing
from typing import Iterable, Optional, Sequence

from ..crease_pattern import angle_utils
from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import point as point_utils
from ..crease_pattern import rotation
from . import base_rectangle as rect_utils
from . import shaping_utils
from . import vector
from ..crease_pattern.parser import svg as parser
from .. import timeout as timeout_lib


class Orientation(enum.Enum):
  """Denotes which edge of the rectangle a zigzag crease is touching."""

  TOP = 'top'
  RIGHT = 'right'
  BOTTOM = 'bottom'
  LEFT = 'left'


@dataclasses.dataclass(kw_only=True)
class ZigZagInfo:
  """Information extracted from a zigzag crease, to be used when constructing a `ZigZag` instance.

  Attributes:
    border_point: The point on the rectangle border that the zigzag crease
      intersects.
    non_border_point: The other tip of the zigzag crease that is not the
      `border_point`.
    orientation: The edge of the rectangle the zigzag crease is touching.
  """

  border_point: point_utils.Point
  non_border_point: point_utils.Point
  orientation: Orientation


@dataclasses.dataclass(kw_only=True)
class ZigZag:
  """A zigzag crease that has at least one of its tips at the edge of a rectangle.

  NOTE: A zigzag crease can have both tips at the edge of two different
  rectangles if the zigzag crease is in the corner of the rectangle.

  NOTE: A zigzag crease is NOT considered TWO creases which meet at the tip, and
  whose other tips are at a rectangle edge (this would be two zigzags).

  Attributes:
    middle_point: The middle point of the zigzag crease.
    narrowing_crease_end_point: If you draw a crease from the middle_point to
      this point, it will form a narrowing crease in the correct direction,
      according to the angle of this ZigZag.
  """

  middle_point: point_utils.Point
  narrowing_crease_end_point: point_utils.Point

  @classmethod
  def from_zigzag_info(
      cls,
      zigzag_info: ZigZagInfo,
      grid_size: int,
  ) -> Optional['ZigZag']:
    """Creates a ZigZag from the border and non-border points and orientation."""
    border_point: point_utils.Point = zigzag_info.border_point
    non_border_point: point_utils.Point = zigzag_info.non_border_point
    orientation: Orientation = zigzag_info.orientation

    half_grid_length = 1 / (grid_size * 2)

    x_delta = abs(border_point.x - non_border_point.x)
    y_delta = abs(border_point.y - non_border_point.y)

    # A valid zigzag crease must be longer than half of the grid length in at
    # least one of the x or y components. This is because the narrowing creases
    # are drawn from the half grid length point of the zigzag crease.
    if (x_delta <= half_grid_length) and (y_delta <= half_grid_length):
      return None
    if math.isclose(x_delta, half_grid_length) and math.isclose(
        y_delta, half_grid_length
    ):
      return None

    # Calculate the middle point based on the orientation.
    match orientation:
      case Orientation.TOP:
        if non_border_point.x > border_point.x:
          middle_x = border_point.x + half_grid_length
          if (middle_x >= non_border_point.x) or math.isclose(
              middle_x, non_border_point.x
          ):
            # If the middle point is beyond the ZigZag, then we cannot form a
            # valid narrowing crease.
            return None
        else:
          middle_x = border_point.x - half_grid_length
          if (middle_x <= non_border_point.x) or math.isclose(
              middle_x, non_border_point.x
          ):
            return None
        middle_point = point_utils.Point(
            x=middle_x,
            y=point_utils.find_y_on_line(
                middle_x, border_point, non_border_point
            ),
        )
        narrowing_crease_end_point = point_utils.Point(
            x=middle_point.x, y=1.0  # narrowing crease extends downward
        )
      case Orientation.BOTTOM:
        if non_border_point.x > border_point.x:
          middle_x = border_point.x + half_grid_length
          if (middle_x >= non_border_point.x) or math.isclose(
              middle_x, non_border_point.x
          ):
            return None
        else:
          middle_x = border_point.x - half_grid_length
          if (middle_x <= non_border_point.x) or math.isclose(
              middle_x, non_border_point.x
          ):
            return None
        middle_point = point_utils.Point(
            x=middle_x,
            y=point_utils.find_y_on_line(
                middle_x, border_point, non_border_point
            ),
        )
        narrowing_crease_end_point = point_utils.Point(
            x=middle_point.x, y=0.0  # narrowing crease extends upward
        )
      case Orientation.RIGHT:
        if non_border_point.y > border_point.y:
          middle_y = border_point.y + half_grid_length
          if (middle_y >= non_border_point.y) or math.isclose(
              middle_y, non_border_point.y
          ):
            return None
        else:
          middle_y = border_point.y - half_grid_length
          if (middle_y <= non_border_point.y) or math.isclose(
              middle_y, non_border_point.y
          ):
            return None
        middle_point = point_utils.Point(
            x=point_utils.find_x_on_line(
                middle_y, border_point, non_border_point
            ),
            y=middle_y,
        )
        narrowing_crease_end_point = point_utils.Point(
            x=0.0, y=middle_point.y  # narrowing crease extends to the left
        )
      case Orientation.LEFT:
        if non_border_point.y > border_point.y:
          middle_y = border_point.y + half_grid_length
          if (middle_y >= non_border_point.y) or math.isclose(
              middle_y, non_border_point.y
          ):
            return None
        else:
          middle_y = border_point.y - half_grid_length
          if (middle_y <= non_border_point.y) or math.isclose(
              middle_y, non_border_point.y
          ):
            return None
        middle_point = point_utils.Point(
            x=point_utils.find_x_on_line(
                middle_y, border_point, non_border_point
            ),
            y=middle_y,
        )
        narrowing_crease_end_point = point_utils.Point(
            x=1.0, y=middle_point.y  # narrowing crease extends to the right
        )
      case _:
        typing.assert_never(orientation)

    return cls(
        middle_point=middle_point,
        narrowing_crease_end_point=narrowing_crease_end_point,
    )


@dataclasses.dataclass(kw_only=True)
class NarrowingPolygon:
  """A list of narrowing creases, forming a connected path.

  This path can either loop back on itself and form a polygon, or both ends of
  the path could finish at the edge of the rectangle.
  """

  creases: list[crease_utils.Crease]

  def __contains__(self, point: point_utils.Point) -> bool:
    """Checks if the point is a tip of any of the creases."""
    return any(
        crease.point_status(point) == crease_utils.PointStatus.TIP
        for crease in self.creases
    )

  def flip(self) -> 'NarrowingPolygon':
    """Returns a new NarrowingPolygon with the fold types of its creases flipped."""
    return NarrowingPolygon(
        creases=[
            dataclasses.replace(crease, fold=crease.fold.flip())
            for crease in self.creases
        ]
    )


def _update_double_border_crease_dicts(
    *,
    zigzag_creases: list[crease_utils.Crease],
    zigzag_info1: ZigZagInfo,
    zigzag_info2: ZigZagInfo,
    crease: crease_utils.Crease,
) -> crease_utils.Crease | None:
  """Updates double-border crease dicts with zigzag info and neighboring creases."""
  if zigzag_info1.border_point == zigzag_info2.border_point:
    # Each tip of the zigzag is at the same border edge of the rectangle.
    connecting_points = [zigzag_info1.non_border_point]
  else:
    invalid = False
    z0_border_point = zigzag_info1.border_point
    z1_border_point = zigzag_info2.border_point
    z0_non_border_point = zigzag_info1.non_border_point
    z1_non_border_point = zigzag_info2.non_border_point

    # Check to see if the two points for each zigzag are the same.
    if z0_border_point.isclose(z1_border_point):
      if not z0_non_border_point.isclose(z1_non_border_point):
        invalid = True
    elif z0_non_border_point.isclose(z1_border_point):
      if not z0_border_point.isclose(z1_non_border_point):
        invalid = True
    else:
      invalid = True

    # Each tip of the zigzag is at a different border edge of the rectangle.
    if invalid:
      raise ValueError(
          'Zigzag infos must have the same points:'
          f' {set([zigzag_info1.border_point, zigzag_info1.non_border_point])},'
          f'  {set([zigzag_info2.border_point, zigzag_info2.non_border_point])}'
      )

    connecting_points = [
        zigzag_info1.non_border_point,
        zigzag_info2.non_border_point,
    ]

  neighbor_crease = None
  for potential_neighbor_crease in zigzag_creases:
    if crease == potential_neighbor_crease:
      continue

    # connecting_point is the non-border intersection point of which this
    # zigzag crease's tip touches the tip of a neighboring zigzag crease.
    for connecting_point in connecting_points:
      if (
          potential_neighbor_crease.point_status(connecting_point)
          == crease_utils.PointStatus.TIP
      ):
        # There should only ever be one possible non-border tip intersection
        # point.
        neighbor_crease = potential_neighbor_crease
        break
    if neighbor_crease:
      break

  return neighbor_crease


def _get_zigzag_infos(
    rectangle: shaping_utils.Rectangle | rect_utils.Rectangle,
    crease: crease_utils.Crease,
    grid_size: int,
) -> list[ZigZagInfo]:
  """Returns zigzag metadata indicating which rectangle borders the zigzag touches."""
  grid_increment = 1 / grid_size

  zigzag_infos: list[ZigZagInfo] = []
  for edge, orientation in zip(rectangle.edges, Orientation):
    # The canonical edge order of rectangle.edges is 'top', 'right', 'bottom',
    # 'left'.
    intersection_point = crease.vertex_with_crease(edge)
    if intersection_point is None:
      continue

    border_point = intersection_point
    # Valid zigzag creases should have its border point on a grid point.
    x_on_grid_point = math.isclose(
        round(border_point.x / grid_increment),
        border_point.x / grid_increment,
        abs_tol=1e-3,
    )
    y_on_grid_point = math.isclose(
        round(border_point.y / grid_increment),
        border_point.y / grid_increment,
        abs_tol=1e-3,
    )
    if (not x_on_grid_point) or (not y_on_grid_point):
      continue

    non_border_point = (
        crease.p1 if (not crease.p1.isclose(intersection_point)) else crease.p2
    )
    zigzag_infos.append(
        ZigZagInfo(
            border_point=border_point,
            non_border_point=non_border_point,
            orientation=orientation,
        )
    )
  return zigzag_infos


def get_zigzags(
    zigzag_creases: list[crease_utils.Crease],
    rectangle: shaping_utils.Rectangle | rect_utils.Rectangle,
    grid_size: int,
) -> tuple[list[ZigZag], list[crease_utils.Crease]]:
  """Construct `ZigZag` instances for each zigzag crease.

  Each `ZigZag` instance has a `middle_point` attribute. Thus the `ZigZag` can
  be divided into two segments: the border-to-middle-point segment, and the
  middle-point-to-non-border segment.

  In addition to returning the constructed `ZigZag` instance, this function also
  returns all middle-point-to-non-border segments. These segments may be later
  replaced with a rotated version, in the edge case where a narrowing crease
  intersects with the rectangle border but not the paper border.

  Args:
    zigzag_creases: The creases to construct zigzags from.
    rectangle: The rectangle to construct zigzags from.
    grid_size: The size of the box-pleating grid.

  Returns:
    A tuple of lists:
      - The first list contains the constructed `ZigZag` instances.
      - The second list contains the middle-point-to-non-border segments.
  """
  # Dicts with crease hashes as keys.
  # Double border creases refer to zigzags that touch two separate rectangle
  # borders (one at each tip).
  double_border_crease_to_neighbor_crease: dict[int, crease_utils.Crease] = {}
  double_border_crease_to_zigzag_infos: dict[int, list[ZigZagInfo]] = {}
  crease_hash_to_orientation: dict[int, Orientation] = {}
  crease_hash_to_crease: dict[int, crease_utils.Crease] = {}

  zigzags: list[ZigZag] = []
  # Middle-point-to-non-border segments
  non_border_zigzag_creases: list[crease_utils.Crease] = []
  for crease in zigzag_creases:
    if not rectangle.crease_is_within(crease):
      # Only consider zigzag creases that are within the rectangle.
      continue

    crease_hash_to_crease[crease.hash_value] = crease

    zigzag_infos: list[ZigZagInfo] = _get_zigzag_infos(
        rectangle, crease, grid_size
    )

    if len(zigzag_infos) == 2:
      # Edge case where the zigzag touches two different rectangle borders.
      neighbor_crease = _update_double_border_crease_dicts(
          zigzag_creases=zigzag_creases,
          zigzag_info1=zigzag_infos[0],
          zigzag_info2=zigzag_infos[1],
          crease=crease,
      )
      if neighbor_crease:
        double_border_crease_to_neighbor_crease[crease.hash_value] = (
            neighbor_crease
        )
        double_border_crease_to_zigzag_infos[crease.hash_value] = zigzag_infos

    elif len(zigzag_infos) == 1:
      # zigzag_infos must be length 1, meaning that the zigzag crease is only
      # touching one rectangle border.
      zigzag_info = zigzag_infos[0]
      crease_hash_to_orientation[crease.hash_value] = zigzag_info.orientation

      zigzag = ZigZag.from_zigzag_info(
          zigzag_info=zigzag_info,
          grid_size=grid_size,
      )
      if zigzag is None:
        continue
      zigzags.append(zigzag)
      non_border_zigzag_creases.append(
          dataclasses.replace(
              crease,
              p1_init=zigzag.middle_point,
              p2_init=zigzag_info.non_border_point,
          )
      )

  # For a zigzag crease touching two different rectangle borders, its
  # orientation is ambiguous, so we find its neighbor crease and use that to
  # determine the orientation.
  for (
      double_border_crease_hash,
      neighbor_crease,
  ) in double_border_crease_to_neighbor_crease.items():
    timeout_lib.maybe_timeout_cooperatively()
    for zigzag_info in double_border_crease_to_zigzag_infos[
        double_border_crease_hash
    ]:
      if neighbor_crease.hash_value not in crease_hash_to_orientation:
        continue

      if (
          zigzag_info.orientation
          == crease_hash_to_orientation[neighbor_crease.hash_value]
      ):
        zigzag = ZigZag.from_zigzag_info(
            zigzag_info=zigzag_info,
            grid_size=grid_size,
        )
        if zigzag is None:
          continue
        zigzags.append(zigzag)
        non_border_zigzag_creases.append(
            dataclasses.replace(
                crease_hash_to_crease[double_border_crease_hash],
                p1_init=zigzag.middle_point,
                p2_init=zigzag_info.non_border_point,
            )
        )
        break

  return zigzags, non_border_zigzag_creases


def _get_crease_that_has_point_tip(
    point: point_utils.Point, reference_creases: list[crease_utils.Crease]
) -> crease_utils.Crease | None:
  """Returns the reference crease that has the given point as its tip.

  Raises an error if there are multiple reference creases that has the given
  point as its tip.

  Args:
    point: The point to find the reference crease for.
    reference_creases: The reference creases to search through.

  Returns:
    The reference crease that has the given point as its tip, or None if there
    is no such crease.
  """
  reference_crease = None
  for crease in reference_creases:
    if crease.is_tip(point):
      if reference_crease is None:
        reference_crease = crease
      else:
        raise ValueError(
            f'Multiple creases ({reference_crease}, {crease}) have the same'
            f' point {point} as their tip.'
        )
  if not reference_crease:
    logging.info('No creases found with point %s as their tip.', point)
    return None
  return reference_crease


def _get_paper_border_crease(
    point: point_utils.Point,
    cp: cp_utils.CreasePattern,
) -> crease_utils.Crease | None:
  """For a given point on the paper border, return the paper border crease that contains this point.

  Args:
    point: The point on the rectangle border.
    cp: The crease pattern to search through.

  Returns:
    The paper border crease that contains the given point, or None if there is
    no such crease.
  """
  for crease in cp.creases:
    if crease.fold != crease_utils.Fold.BORDER:
      continue

    if crease.point_status(point) in (
        crease_utils.PointStatus.TIP,
        crease_utils.PointStatus.INTERNAL,
    ):
      return crease
  return None


def _get_non_rect_border_crease(
    point: point_utils.Point,
    rectangle: shaping_utils.Rectangle | rect_utils.Rectangle,
    cp: cp_utils.CreasePattern,
) -> crease_utils.Crease | None:
  """Returns the non-border crease containing the given point on the rectangle border.

  We need this non-border crease since the narrowing crease will be reflected
  over this non-border crease. If there are multiple non-border creases, we
  return one of them, so long as all of them are collinear with each other
  (since the reflection angle would be the same with all of them). If they are
  not collinear, then we return None.

  Args:
    point: The point on the rectangle border.
    rectangle: The rectangle to search through.
    cp: The crease pattern to search through.

  Returns:
    The non-border crease that has the given point. If there are multiple
    non-border creases, we return one of them, so long as all of them are
    collinear with each other. If they are not collinear, then we return None.
    If we don't find any non-border creases, we return None.
  """
  if not rectangle.is_edge_point(point, extend_edges=True):
    raise ValueError(f'Point {point} is not on rectangle border {rectangle}.')

  non_border_crease: crease_utils.Crease | None = None
  for crease in cp.creases:
    if crease.point_status(point) not in (
        crease_utils.PointStatus.TIP,
        crease_utils.PointStatus.INTERNAL,
    ):
      continue
    if any((crease.is_overlapping(edge) for edge in rectangle.edges)):
      continue

    if non_border_crease is None:
      non_border_crease = crease
    elif not non_border_crease.is_collinear(crease):
      return None
  return non_border_crease


def _overlaps_existing_crease(
    crease: crease_utils.Crease, cp: cp_utils.CreasePattern
):
  """Returns whether the crease overlaps with any of the existing creases in the cp."""
  for existing_crease in cp.creases:
    if crease.is_overlapping(existing_crease):
      return True
  return False


def _get_closest_crease(
    start_point: point_utils.Point,
    incoming_crease: crease_utils.Crease,
    reference_creases: Iterable[crease_utils.Crease],
) -> tuple[crease_utils.Crease | None, point_utils.Point | None, float]:
  """Finds the closest crease intersecting `incoming_crease` away from `start_point`.

  NOTE: the start_point must be a tip of the incoming_crease.

  Args:
    start_point: The start point of the incoming crease.
    incoming_crease: The incoming crease to find the closest crease for.
    reference_creases: The reference creases to search through for the closest
      crease.

  Returns:
    A tuple of the closest crease and the intersection point of the incoming
    crease and the closest crease. If no intersection is found, the closest
    crease will be None.
  """
  if incoming_crease.point_status(start_point) != crease_utils.PointStatus.TIP:
    raise ValueError(
        f'The start point {start_point} of the incoming crease'
        f' {incoming_crease} is not a tip.'
    )

  closest_crease = None
  closest_distance = float('inf')
  closest_intersection_point = None
  for crease in reference_creases:
    intersection_point = incoming_crease.vertex_with_crease(crease)
    if (intersection_point is None) or (
        intersection_point.isclose(start_point)
    ):
      # We want the closest intersection that is not intersecting with the start
      # point of the incoming crease.
      continue
    distance = point_utils.distance(start_point, intersection_point)
    if distance < closest_distance:
      closest_crease = crease
      closest_distance = distance
      closest_intersection_point = intersection_point
  return closest_crease, closest_intersection_point, closest_distance


def _get_rotated_crease(
    crease: crease_utils.Crease,
    clockwise_angle_delta: float,
    start_point: point_utils.Point,
    cp: cp_utils.CreasePattern,
) -> crease_utils.Crease:
  """Rotates the crease and modifies the length of the rotated crease.

  How it works:
  - after rotation, the crease length is extended such that the end point is
  either at the paper border or beyond the paper border
  - the closest crease that intersects this rotated crease is found
  - the rotated crease is then truncated to end at this intersection point

  The result is the rotated crease that has its length perfectly adjusted to end
  at the closest intersection in the cp.

  Args:
    crease: The crease to rotate.
    clockwise_angle_delta: The clockwise angle delta to rotate the crease.
    start_point: The start point of the rotated crease. This point will be the
      point of rotation for the crease.
    cp: The CreasePattern to search through.

  Returns:
    The rotated crease.
  """
  rotated_crease = rotation.rotate_crease_clockwise(
      crease_to_rotate=crease,
      rotation_angle_degrees=clockwise_angle_delta,
      rotation_origin_tip=start_point,
      require_on_paper=False,
  )
  if rotated_crease.p1.isclose(start_point):
    other_point = rotated_crease.p2
  elif rotated_crease.p2.isclose(start_point):
    other_point = rotated_crease.p1
  else:
    raise ValueError(
        f'The rotated crease {rotated_crease} does not have'
        f' {start_point} as its tip.'
    )

  # Make the vector at least the maximum length of a line that can be drawn in a
  # square bounded by [0, 1].
  normalized_vector = vector.Vector.from_points(
      start_point, other_point
  ).normalize * math.sqrt(2)
  new_end_point = normalized_vector.point_add(
      dataclasses.replace(start_point, require_on_paper=False)
  )
  rotated_crease_extended = dataclasses.replace(
      rotated_crease, p1_init=start_point, p2_init=new_end_point
  )

  _, closest_intersection_point, _ = _get_closest_crease(
      start_point, rotated_crease_extended, cp.creases
  )
  if closest_intersection_point is None:
    raise ValueError(
        'No intersection found between the rotated crease and any existing'
        ' crease.'
    )
  return dataclasses.replace(
      crease, p1_init=start_point, p2_init=closest_intersection_point
  )


def draw_narrowing_crease(
    start_point: point_utils.Point,
    end_point: point_utils.Point,
    rectangle_border_points: Iterable[point_utils.Point],
    rectangle: shaping_utils.Rectangle | rect_utils.Rectangle,
    cp: cp_utils.CreasePattern,
    non_border_zigzag_creases: list[crease_utils.Crease],
    prev_clockwise_angle_delta: float | None = None,
    *,
    allow_paper_border_rotation: bool = True,
) -> (
    tuple[
        point_utils.Point,
        point_utils.Point,
        list[tuple[crease_utils.Crease, crease_utils.Crease]],
        float | None,
    ]
    | None
):
  """Computes the start and end points of the next narrowing crease segment.

  NOTE: The end_point could be beyond the [0, 1] boundaries of the paper, but it
  is merely used to denote the direction of the narrowing crease.

  If the narrowing crease intersects with the paper border, the angle of the
  narrowing crease may be altered which would also lead to changing the angle of
  the zigzag crease, so a dict containing information on zigzag replacements is
  also returned.

  If the previous narrowing crease had its angle altered, this information will
  be passed in as `prev_clockwise_angle_delta` so that this narrowing crease
  angle can be adjusted accordingly.

  Args:
    start_point: The start point of the incoming crease.
    end_point: The end point of the incoming crease.
    rectangle_border_points: The set of points that lie on the rectangle border
      that intersect with the crease pattern.
    rectangle: The rectangle to draw the narrowing crease for.
    cp: The CreasePattern to search through.
    non_border_zigzag_creases: The non-border zigzag creases that may need to be
      adjusted if the narrowing crease intersects with the paper border.
    prev_clockwise_angle_delta: The clockwise angle delta of the previous
      narrowing crease.
    allow_paper_border_rotation: Whether to allow rotating creases that meet the
      paper border.

  Returns:
    A tuple of the start point, end point, a list of zigzag replacements, and
    the clockwise angle delta of the narrowing crease. If the narrowing crease
    does not intersect with the paper border, then the clockwise angle delta
    will be None.
  """

  incoming_crease = crease_utils.Crease(
      start_point, end_point, fold=crease_utils.Fold.MOUNTAIN
  )

  # Maps the hash of the zigzag crease to be replaced to the new zigzag crease.
  zigzag_replacements: list[tuple[crease_utils.Crease, crease_utils.Crease]] = (
      []
  )

  # For the edge case where the narrowing crease hits the rectangle border,
  # sometimes the actual rectangle border crease does not exist in the original
  # crease pattern, so we have to try finding it by passing in the rectangle
  # border creases.
  cp_closest_crease, cp_closest_intersection_point, cp_closest_distance = (
      _get_closest_crease(start_point, incoming_crease, cp.creases)
  )
  (
      rect_closest_crease,
      rect_closest_intersection_point,
      rect_closest_distance,
  ) = _get_closest_crease(start_point, incoming_crease, rectangle.edges)
  if (cp_closest_distance <= rect_closest_distance) or math.isclose(
      cp_closest_distance, rect_closest_distance
  ):
    closest_crease = cp_closest_crease
    closest_intersection_point = cp_closest_intersection_point
  else:
    closest_crease = rect_closest_crease
    closest_intersection_point = rect_closest_intersection_point

  if (closest_intersection_point is None) or (closest_crease is None):
    logging.info(
        'No intersection found between the incoming crease and any existing'
        ' crease.'
    )
    return None

  new_clockwise_angle_delta = None
  # If the narrowing crease hits the border at a vertical or horizontal angle,
  # we shouldn't rotate or snap its end point to the nearest border point.
  if (
      rectangle.is_edge_point(closest_intersection_point)
      and (
          allow_paper_border_rotation
          or (not closest_intersection_point.is_border)
      )
      and (
          (prev_clockwise_angle_delta is None)
          or (not closest_intersection_point.is_border)
      )
      and (not incoming_crease.is_vertical)
      and (not incoming_crease.is_horizontal)
  ):
    # Edge case where the closest intersection point is on the rectangle border.
    # We have to change the angle of this incoming crease and move the
    # intersection point to the next closest vertex on the rectangle border.
    # If it's on the paper border, it's okay to just leave the incoming crease
    # unrotated since it's exiting the paper.
    if prev_clockwise_angle_delta is not None:
      raise ValueError(
          'prev_clockwise_angle_delta must be None for border-ending incoming'
          ' creases.'
      )

    old_clockwise_angle = angle_utils.find_angle_from_points(
        start_point, closest_intersection_point
    )

    # Sometimes when a narrowing crease hits the border and rotates, it overlaps
    # with an existing crease in the cp.
    candidate_closest_intersection_point = (
        closest_intersection_point.closest_point(rectangle_border_points)
    )
    candidate_narrowing_crease = crease_utils.Crease(
        start_point,
        candidate_closest_intersection_point,
        fold=crease_utils.Fold.MOUNTAIN,
    )
    # The rotated crease cannot overlap with any existing creases in the cp.
    if not _overlaps_existing_crease(candidate_narrowing_crease, cp):
      closest_intersection_point = candidate_closest_intersection_point

    if closest_intersection_point.is_border:
      # The border crease will be the reflection ridge.
      closest_crease = _get_paper_border_crease(closest_intersection_point, cp)
    else:
      # The crease that is not a rectangle border crease will be the reflection
      # ridge.
      closest_crease = _get_non_rect_border_crease(
          closest_intersection_point, rectangle, cp
      )
    if closest_crease is None:
      return None

    new_clockwise_angle = angle_utils.find_angle_from_points(
        start_point, closest_intersection_point
    )
    # convert radians to degrees
    new_clockwise_angle_delta = (
        (new_clockwise_angle - old_clockwise_angle) / math.pi * 180
    )

    zigzag_crease_to_replace = _get_crease_that_has_point_tip(
        start_point, non_border_zigzag_creases
    )
    # Edge case where the start point does not intersect with any non-border
    # zigzag creases at the tip, meaning we can't find a valid replacement for
    # the zigzag after rotation.
    if zigzag_crease_to_replace is None:
      return None
    new_zigzag_crease = _get_rotated_crease(
        zigzag_crease_to_replace, new_clockwise_angle_delta, start_point, cp
    )
    zigzag_replacements.append((zigzag_crease_to_replace, new_zigzag_crease))

  truncated_incoming_crease = crease_utils.Crease(
      start_point, closest_intersection_point, fold=crease_utils.Fold.MOUNTAIN
  )

  reflected_vector = shaping_utils.reflect_crease_over_ridge(
      crease=truncated_incoming_crease,
      ridge=closest_crease,
      precomputed_intersection_point=closest_intersection_point,
  )
  # Make the vector at least the maximum length of a line that can be drawn in a
  # square bounded by [0, 1]. Later on we will convert this to a crease and we
  # will be guaranteed to get an intersection within the boundaries of the paper
  # square, if it exists.
  reflected_vector = reflected_vector.normalize * math.sqrt(2)
  new_end_point = reflected_vector.point_add(
      dataclasses.replace(closest_intersection_point, require_on_paper=False)
  )

  if prev_clockwise_angle_delta is not None:
    # The previous narrowing crease was rotated `prev_clockwise_angle_delta`
    # degrees clockwise, meaning the reflected crease is rotated too far
    # `prev_clockwise_angle_delta` degrees clockwise. Therefore we should rotate
    # the reflected crease back the same amount of degrees in the other
    # direction, and also rotate the nonborder zigzag crease the same amount in
    # the same direction to maintain Kawasaki.
    new_end_point = rotation.rotate_point_clockwise(
        point_to_rotate=new_end_point,
        rotation_angle_degrees=-prev_clockwise_angle_delta,
        rotation_origin_tip=closest_intersection_point,
        require_on_paper=False,
    )

    zigzag_crease_to_replace = _get_crease_that_has_point_tip(
        closest_intersection_point, non_border_zigzag_creases
    )
    if zigzag_crease_to_replace is not None:
      new_zigzag_crease = _get_rotated_crease(
          zigzag_crease_to_replace,
          -prev_clockwise_angle_delta,
          closest_intersection_point,
          cp,
      )
      zigzag_replacements.append((zigzag_crease_to_replace, new_zigzag_crease))

  return (
      closest_intersection_point,
      new_end_point,
      zigzag_replacements,
      new_clockwise_angle_delta,
  )


def _get_narrowing_points(
    start_point: point_utils.Point,
    end_point: point_utils.Point,
    rectangle_border_points: set[point_utils.Point],
    rectangle: shaping_utils.Rectangle | rect_utils.Rectangle,
    cp: cp_utils.CreasePattern,
    non_border_zigzag_creases: list[crease_utils.Crease],
    *,
    max_num_creases: int = 1000,
    allow_paper_border_rotation: bool = True,
) -> (
    tuple[
        list[point_utils.Point],
        list[tuple[crease_utils.Crease, crease_utils.Crease]],
    ]
    | None
):
  """Traces the narrowing crease path until it loops back to start or hits a border.

  Args:
    start_point: The start point of the incoming crease.
    end_point: The end point of the incoming crease.
    rectangle_border_points: The set of points that lie on the rectangle border
      that intersect with the crease pattern.
    rectangle: The rectangle to draw the narrowing crease for.
    cp: The CreasePattern to search through.
    non_border_zigzag_creases: The non-border zigzag creases that may need to be
      adjusted if the narrowing crease intersects with the paper border.
    max_num_creases: The maximum number of narrowing creases allowed to be
      drawn, otherwise an error will be raised.
    allow_paper_border_rotation: Whether to allow rotating creases that meet the
      paper border.

  Returns:
    A tuple of:
    - the points that define the narrowing crease path,
    - a list of zigzag replacements,
    If narrowing is not valid, None is returned instead.
  """

  narrowing_points: list[point_utils.Point] = [start_point]
  close_loop_point = start_point

  zigzag_replacements: list[tuple[crease_utils.Crease, crease_utils.Crease]] = (
      []
  )

  # NOTE: Check if the narrowing crease intersects with itself
  # and prematurely exit if it does.
  clockwise_angle_delta = None
  num_creases_drawn = 0
  while True:
    timeout_lib.maybe_timeout_cooperatively()
    # NOTE: Remove this check once we can be certain that the
    # narrowing function does not cause infinite loops.
    if num_creases_drawn > max_num_creases:
      _narrowing_creases = []
      for j in range(max_num_creases - 1):
        _narrowing_creases.append(
            crease_utils.Crease(
                narrowing_points[j],
                narrowing_points[j + 1],
                # Purely for visualization debugging purposes; to have the
                # narrowing creases be a different color than M/V folds when
                # visualizing the SVG.
                fold=crease_utils.Fold.HINGE,
            )
        )
      _final_cp = cp_utils.CreasePattern(
          creases=cp.creases + _narrowing_creases
      )

      error_msg = f"""Too many narrowing creases drawn. Debug info:

Grid size: {cp.grid_size}
Original start point: {start_point}
Original end point: {end_point}
Rectangle: {rectangle}

Narrowed CP:
{parser.to_svg(_final_cp)}

Original CP:
{parser.to_svg(cp)}
"""
      raise ValueError(error_msg)

    result = draw_narrowing_crease(
        start_point=start_point,
        end_point=end_point,
        rectangle_border_points=rectangle_border_points,
        rectangle=rectangle,
        cp=cp,
        non_border_zigzag_creases=non_border_zigzag_creases,
        prev_clockwise_angle_delta=clockwise_angle_delta,
        allow_paper_border_rotation=allow_paper_border_rotation,
    )
    if result is None:
      return None
    start_point, end_point, child_zigzag_replacements, clockwise_angle_delta = (
        result
    )
    # NOTE: Fix narrowing code to handle this edge case.
    if any(start_point.isclose(p) for p in narrowing_points[1:]):
      # Narrowing creases looped back to a non-start point, meaning we're caught
      # in an infinite loop.
      return None
    # NOTE: Fix narrowing code to handle this edge case.
    if not (
        rectangle.is_internal_point(start_point)
        or rectangle.is_internal_point(narrowing_points[-1])
    ):
      # Narrowing creases that are strictly outside the rectangle but still
      # within the paper are invalid; i.e. narrowing creases that interfere with
      # other rectangles in the paper are not allowed.
      return None
    narrowing_points.append(start_point)
    zigzag_replacements.extend(child_zigzag_replacements)
    num_creases_drawn += 1

    if start_point.isclose(close_loop_point) or start_point.is_border:
      break

  return narrowing_points, zigzag_replacements


def _remove_zigzags(
    zigzags: list[ZigZag], points_to_remove: list[point_utils.Point]
):
  """Removes zigzags that have a middle point that is close to a point to remove."""
  for point_to_remove in points_to_remove:
    for zigzag in zigzags:
      if zigzag.middle_point.isclose(point_to_remove):
        zigzags.remove(zigzag)
        break


def _narrow_once(
    cp: cp_utils.CreasePattern,
    zigzag_creases: list[crease_utils.Crease],
    rectangles: Sequence[shaping_utils.Rectangle | rect_utils.Rectangle],
    grid_size: int,
    light_mode: bool,
    allow_paper_border_rotation: bool,
) -> cp_utils.CreasePattern:
  """Runs one narrowing pass with or without paper-border rotation."""
  narrowing_polygons: list[NarrowingPolygon] = []
  all_zigzag_replacements: list[
      tuple[crease_utils.Crease, crease_utils.Crease]
  ] = []

  for rectangle in rectangles:
    rectangle = rectangle.clip(xmin=0, ymin=0, xmax=1, ymax=1)
    if rectangle is None:
      raise ValueError(
          f'Clipped rectangle {rectangle} is outside the paper bounds.'
      )

    rectangle_border_points: set[point_utils.Point] = set()
    for crease in cp.creases:
      timeout_lib.maybe_timeout_cooperatively()
      for rect_edge in rectangle.edges:
        if crease.vertex_with_crease(rect_edge):
          intersection_point = crease.vertex_with_crease(rect_edge)
          if intersection_point is not None:
            rectangle_border_points.add(intersection_point)

    zigzags, non_border_zigzag_creases = get_zigzags(
        zigzag_creases, rectangle, grid_size
    )
    while zigzags:
      timeout_lib.maybe_timeout_cooperatively()
      zigzag: ZigZag = zigzags.pop(0)

      narrowing_creases: list[crease_utils.Crease] = []

      result = _get_narrowing_points(
          start_point=zigzag.middle_point,
          end_point=zigzag.narrowing_crease_end_point,
          rectangle_border_points=rectangle_border_points,
          rectangle=rectangle,
          cp=cp,
          non_border_zigzag_creases=non_border_zigzag_creases,
          allow_paper_border_rotation=allow_paper_border_rotation,
      )

      if result is None:
        continue
      narrowing_points, zigzag_replacements = result

      if len(narrowing_points) < 2:
        continue

      # Narrowing creases have looped back to the starting point.
      elif narrowing_points[-1].isclose(zigzag.middle_point):
        if len(narrowing_points) % 2 == 0:
          # Even number of points means odd number of creases, meaning we can't
          # successfully draw narrowing creases of alternating fold type that
          # loops
          continue

      # Narrowing creases have reached the border of the paper.
      elif narrowing_points[-1].is_border:

        if not rectangle.is_edge_point(narrowing_points[-1], extend_edges=True):
          # If the narrowing crease ends at the border, this border must be
          # within the bounds of the rectangle.
          continue

        # Draw narrowing creases in the reverse direction; it must end in a
        # border point.
        result = _get_narrowing_points(
            start_point=narrowing_points[1],
            end_point=narrowing_points[0],
            rectangle_border_points=rectangle_border_points,
            rectangle=rectangle,
            cp=cp,
            non_border_zigzag_creases=non_border_zigzag_creases,
            allow_paper_border_rotation=allow_paper_border_rotation,
        )
        if result is None:
          continue
        (
            narrowing_points_other_direction,
            zigzag_replacements_other_direction,
        ) = result

        if (
            (narrowing_points_other_direction is None)
            or (len(narrowing_points_other_direction) < 2)
            or (not narrowing_points_other_direction[-1].is_border)
            or (
                not rectangle.is_edge_point(
                    narrowing_points_other_direction[-1],
                    extend_edges=True,
                )
            )
        ):
          continue

        all_zigzag_replacements.extend(zigzag_replacements_other_direction)

        # We drew an extra crease in the beginning, so we should remove it.
        narrowing_points_other_direction = narrowing_points_other_direction[1:]

        # Start with the opposite fold type when drawing in the opposite
        # direction compared to the fold type when drawing in the original
        # direction. This ensures that the narrowing creases have alternating
        # fold types.
        curr_fold_type = crease_utils.Fold.VALLEY
        for i in range(len(narrowing_points_other_direction) - 1):
          narrowing_creases.append(
              crease_utils.Crease(
                  narrowing_points_other_direction[i],
                  narrowing_points_other_direction[i + 1],
                  fold=curr_fold_type,
              )
          )
          curr_fold_type = curr_fold_type.flip()

        _remove_zigzags(zigzags, narrowing_points_other_direction)

      else:
        raise ValueError(
            'End point of narrowing crease is not on border or close to the'
            ' starting point of the original loop.'
        )

      all_zigzag_replacements.extend(zigzag_replacements)

      curr_fold_type = crease_utils.Fold.MOUNTAIN
      for i in range(len(narrowing_points) - 1):
        narrowing_creases.append(
            crease_utils.Crease(
                narrowing_points[i],
                narrowing_points[i + 1],
                fold=curr_fold_type,
            )
        )
        curr_fold_type = curr_fold_type.flip()

      _remove_zigzags(zigzags, narrowing_points)

      if narrowing_creases:
        narrowing_polygons.append(NarrowingPolygon(creases=narrowing_creases))

  if not narrowing_polygons:
    logging.info(
        'Narrowing failed for cp:\n%s\n\nRectangles: %s',
        parser.to_svg(cp),
        rectangles,
    )
    return cp

  narrowing_creases: list[crease_utils.Crease] = []
  for narrowing_polygon in narrowing_polygons:
    narrowing_creases.extend(narrowing_polygon.creases)

  # Split the base creases into segments if they intersect with the narrowing
  # creases.
  tips, _ = cp_utils.get_tips_and_grid_creases(
      cp.creases + narrowing_creases, grid_size=-1
  )
  base_creases_split = cp_utils.split_creases(
      cp.creases, narrowing_creases, tips=tips
  )

  # Replace the zigzag lines in the base crease with the newly rotated zigzag
  # lines, where applicable.
  base_creases_replaced: list[crease_utils.Crease] = []
  any_replacement_done = False
  for crease in base_creases_split:
    timeout_lib.maybe_timeout_cooperatively()
    replacement_done = False
    for (
        zigzag_crease_to_replace,
        new_zigzag_crease,
    ) in all_zigzag_replacements:
      if crease.isclose(zigzag_crease_to_replace):
        base_creases_replaced.append(new_zigzag_crease)
        replacement_done = True
        any_replacement_done = True
        break
    if not replacement_done:
      base_creases_replaced.append(crease)

  if any_replacement_done:
    # Force light mode to be False since we might've adjusted the angle of some
    # of the zigzag creases.
    narrowed_cp = cp_utils.CreasePattern(
        creases=base_creases_replaced + narrowing_creases, light_mode=False
    )
  else:
    narrowed_cp = cp_utils.CreasePattern(
        creases=base_creases_replaced + narrowing_creases,
        light_mode=light_mode,
    )

  # For every non-flat-foldable vertex that intersects with a narrowing crease,
  # flip the crease assignments for all of the narrowing creases in that
  # narrowing polygon, and then retry testing for flat-foldability.
  unmodified_polygons: list[NarrowingPolygon] = narrowing_polygons.copy()
  modified_polygons: list[NarrowingPolygon] = []
  for v in narrowed_cp.vertices:
    timeout_lib.maybe_timeout_cooperatively()
    if not unmodified_polygons:
      break

    if not v.flat_foldability():
      for narrowing_polygon in unmodified_polygons:
        if v.point in narrowing_polygon:
          modified_polygons.append(narrowing_polygon.flip())
          unmodified_polygons.remove(narrowing_polygon)
          break

  if modified_polygons:
    all_creases = base_creases_replaced.copy()
    for polygon_list in [modified_polygons, unmodified_polygons]:
      for narrowing_polygon in polygon_list:
        all_creases.extend(narrowing_polygon.creases)

    if any_replacement_done:
      narrowed_cp = cp_utils.CreasePattern(
          creases=all_creases, light_mode=False
      )
    else:
      narrowed_cp = cp_utils.CreasePattern(
          creases=all_creases, light_mode=light_mode
      )

  return narrowed_cp


def narrow(
    cp: cp_utils.CreasePattern,
    zigzag_creases: list[crease_utils.Crease],
    rectangles: Sequence[shaping_utils.Rectangle | rect_utils.Rectangle],
    grid_size: int | None = None,
    light_mode: bool = False,
    raise_error_on_ff_violation: bool = False,
) -> cp_utils.CreasePattern:
  """Applies narrow flap shaping to a crease pattern.

  NOTE: this can only be applied after zigzag shaping has been applied.

  Args:
    cp: The CreasePattern to shape.
    zigzag_creases: The zigzag creases that resulted from zigzag shaping.
    rectangles: The rectangles to draw the narrowing creases for.
    grid_size: The grid size of the crease pattern, used to figure out where to
      draw the narrowing creases.
    light_mode: Whether to build the CreasePattern in light mode. NOTE: if
      zigzag replacements were made during narrowing, then the CreasePattern
      will be built in non-light mode, regardless of this flag.
    raise_error_on_ff_violation: Whether to raise an error if the crease pattern
      is not flat-foldable.

  Returns:
    The narrowed CreasePattern. Raises an error if narrowing fails.
  """
  if grid_size is None:
    grid_size = cp.grid_size

  narrowed_cp = _narrow_once(
      cp=cp,
      zigzag_creases=zigzag_creases,
      rectangles=rectangles,
      grid_size=grid_size,
      light_mode=light_mode,
      allow_paper_border_rotation=True,
  )
  if not narrowed_cp.local_flat_foldability():
    fallback_cp = _narrow_once(
        cp=cp,
        zigzag_creases=zigzag_creases,
        rectangles=rectangles,
        grid_size=grid_size,
        light_mode=light_mode,
        allow_paper_border_rotation=False,
    )
    if (
        len(fallback_cp.creases) > len(cp.creases)
        and fallback_cp.local_flat_foldability()
    ):
      narrowed_cp = fallback_cp

  if raise_error_on_ff_violation and (not narrowed_cp.local_flat_foldability()):
    raise ValueError('Narrowed CP is not flat-foldable.')

  return narrowed_cp
