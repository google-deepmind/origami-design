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

"""Heuristic util functions for 3D shaping.
"""

import collections
import dataclasses
import enum
import logging
import math
from typing import Any

import numpy as np

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import point as point_utils
from . import shaping_elements
from . import shaping_utils
from . import types as shaping_types
from . import zigzag_shaping
from ..stick_figures import stick as stick_lib
from ..stick_figures import stick_figure as stick_figure_lib
from ..stick_figures import traversal as tr


# The index of walk direction.
class Direction(enum.Enum):
  """The direction of the walk."""
  UP = 0
  RIGHT = 1
  DOWN = 2
  LEFT = 3

NDIGITS_OVERLAPPING = 8
FO_V = [crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN]
FO_M = [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY]


@dataclasses.dataclass(kw_only=True, frozen=True)
class XYAxisCrease:
  """A class to store the x and y axis creases for a flap.

  The x and y axis creases are creases on the 2D crease pattern that correspond
  to the x and y axis on the flap plane in 3D coordinate system, which will be
  used to compute the zigzag crease for a flap based on the local rotation axis.

  Attributes:
    x_axis_crease: The x axis crease.
    y_axis_crease: The y axis crease.
    x_axis_touched_pleat_assignment: The touched pleat assignment for the x axis
      crease.
  """

  x_axis_crease: crease_utils.Crease
  y_axis_crease: crease_utils.Crease
  x_axis_touched_pleat_assignment: crease_utils.Fold


def _find_ridge_edge_intersection(
    ridge: crease_utils.Crease,
    edges: tuple[
        crease_utils.Crease,
        crease_utils.Crease,
        crease_utils.Crease,
        crease_utils.Crease,
    ],
) -> tuple[point_utils.Point, crease_utils.Crease] | None:
  """Finds the first intersection point between a ridge and a list of edges.

  Args:
    ridge: The ridge crease to check for intersections.
    edges: A list of creases representing the edges to check against.

  Returns:
      A tuple of (intersection_point, intersected_edge) or None if no
      intersection.
  """
  for edge in edges:
    intersection = ridge.vertex_with_crease(edge)
    if intersection is not None:
      return intersection, edge
  return None


def _determine_ridge_tip(
    circle_centers: list[point_utils.Point] | None,
) -> point_utils.Point:
  """Determines the primary "tip" of the ridge based on circle centers.

  If two centers are provided, the tip is the one that is on the paper. If both
  are on the paper, the primary tip is the one that is not the paper corner.

  Args:
    circle_centers: A list of circle centers.
  Returns:
    The determined ridge tip.
  Raises:
    ValueError: If the circle centers are not on the paper.
  """
  if circle_centers is None:
    raise ValueError('Circle centers are None.')
  if len(circle_centers) == 1:
    return circle_centers[0]

  if len(circle_centers) == 2:
    c1, c2 = circle_centers
    on_paper_centers = [c for c in circle_centers if c.on_paper]
    if len(on_paper_centers) == 1:
      return on_paper_centers[0]
    else:
      if c1.is_border and c2.is_paper_corner:
        return c1
      elif c2.is_border and c1.is_paper_corner:
        return c2
      else:
        raise ValueError(
            f'Unexptected circle centers: {circle_centers}'
        )
  else:
    raise ValueError(
        f'Unexptected number of circle centers: {len(circle_centers)}'
    )


def _get_border_edges_for_ridge_corner_flap(
    rect: shaping_utils.Rectangle,
) -> list[crease_utils.Crease]:
  """Gets the border edges for a corner flap with one diagonal ridge.

  This is the case where the flap has a single diagonal ridge, and no
  non-diagonal ridge, typically when the flap is a corner flap.

  Args:
    rect: The rectangle to process.
  Returns:
    A list of creases representing the outermost border edges of the simple
    corner flap.
  """
  overlaps = rect.get_paper_border_overlapping_edges()
  assert len(overlaps) == 2

  # Make the overlaps directional.
  directional_overlaps = []
  for edge in overlaps:
    corner = [p for p in [edge.p1, edge.p2] if p.is_paper_corner][0]
    non_corner_point = edge.p1 if edge.p1 != corner else edge.p2
    directional_edge = crease_utils.Crease(
        non_corner_point, corner, fold=crease_utils.Fold.BORDER
    )
    directional_overlaps.append(directional_edge)
  return directional_overlaps


def _get_border_edges_for_half_y_shape_flap(
    rect: shaping_utils.Rectangle,
    diagonal_ridge: crease_utils.Crease,
    non_diagonal_ridge: crease_utils.Crease,
) -> list[crease_utils.Crease]:
  """Gets border edges for a half-Y flap (one diagonal and one non-diagonal ridge).

  This is the case where the flap has one diagonal ridge, and one non-diagonal
  ridge (usually for corner flaps). The non-diagonal ridge is typically
  horizontal or vertical. Based on this, we can determine two directional border
  creases by analyzing the intersection of the non-diagonal ridge with the
  rectangle edges. The first border edge is from the diagonal ridge's non-tip
  endpoint to the ridge tip. The second border edge is from the second border
  edge's end point to the corner point.

  Args:
    rect: The rectangle to process.
    diagonal_ridge: The diagonal ridge.
    non_diagonal_ridge: The non-diagonal ridge.
  Returns:
    A list of creases representing the outermost border edges of the half-Y
    shape.
  """
  circle_centers = rect.circle_centers
  assert circle_centers is not None
  ridge_tip = _determine_ridge_tip(circle_centers)
  assert rect.flap_type(1) == shaping_utils.FlapType.CORNER

  if non_diagonal_ridge.is_horizontal:
    diagonal_non_tip_point = [
        point for point in [diagonal_ridge.p1, diagonal_ridge.p2]
        if point != ridge_tip
    ][0]
    border_edge_1_end_point = point_utils.Point(
        x=diagonal_non_tip_point.x, y=ridge_tip.y)
    # The first border edge is from the diagonal ridge's non-tip endpoint to the
    # ridge tip.
    border_edge_1 = crease_utils.Crease(
        border_edge_1_end_point, ridge_tip, fold=crease_utils.Fold.BORDER)

    corner = [
        p for p in [non_diagonal_ridge.p1, non_diagonal_ridge.p2]
        if p.is_paper_corner
    ][0]
    # Determine the second border edge's end point by using corner's x
    # coordinate and the diagonal ridge's non-tip point's y coordinate.
    border_edge_2_end_point = point_utils.Point(
        x=corner.x, y=diagonal_non_tip_point.y)
    # The second border edge is from the second border edge's end point to the
    # corner point.
    border_edge_2 = crease_utils.Crease(
        border_edge_2_end_point, corner, fold=crease_utils.Fold.BORDER)
  elif non_diagonal_ridge.is_vertical:
    diagonal_non_tip_point = [
        point for point in [diagonal_ridge.p1, diagonal_ridge.p2]
        if point != ridge_tip
    ][0]
    border_edge_1_end_point = point_utils.Point(
        x=ridge_tip.x, y=diagonal_non_tip_point.y)
    border_edge_1 = crease_utils.Crease(
        border_edge_1_end_point, ridge_tip, fold=crease_utils.Fold.BORDER)

    corner = [
        p for p in [non_diagonal_ridge.p1, non_diagonal_ridge.p2]
        if p.is_paper_corner
    ][0]
    border_edge_2_end_point = point_utils.Point(
        x=diagonal_non_tip_point.x, y=corner.y)
    border_edge_2 = crease_utils.Crease(
        border_edge_2_end_point, corner, fold=crease_utils.Fold.BORDER)
  else:
    raise ValueError('Non-diagonal ridge in half-Y shape is neither horizontal '
                     'nor vertical.')

  outermost_borders = [border_edge_1, border_edge_2]
  return outermost_borders


def _get_border_edges_for_half_x_shape(
    rect: shaping_utils.Rectangle, ridge_tip: point_utils.Point
) -> list[crease_utils.Crease]:
  """Gets the border edges for a flap with two diagonal ridges, forming an 'X' shape.

  This is the case where the flap has two diagonal ridges, and no
  non-diagonal ridge. We find which edge the ridge tip lies on, and build two
  directional creases from the corners of that edge to the tip.

  Args:
    rect: The rectangle to process.
    ridge_tip: The ridge tip.
  Returns:
    A list of creases representing the outermost border edges of the half-X
    shape.
  """
  # The ridge tip lies on one of the rectangle's edges.
  # The outermost borders are the two segments from the corners of the edge to
  # the tips.
  for edge in rect.edges:
    if edge.is_intermediate_point(ridge_tip):
      return [
          crease_utils.Crease(
              edge.p1, ridge_tip, fold=crease_utils.Fold.BORDER),
          crease_utils.Crease(
              edge.p2, ridge_tip, fold=crease_utils.Fold.BORDER)
      ]
  raise ValueError('Ridge tip for X-shape is not on any rectangle edge.')


def _get_border_edges_for_y_shape(
    rect: shaping_utils.Rectangle,
    diagonal_ridges: list[crease_utils.Crease],
    non_diagonal_ridge: crease_utils.Crease,
    ridge_tip: point_utils.Point,
) -> list[tuple[crease_utils.Crease, ...]]:
  """Gets border edges for a Y-shaped flap (two diagonal and one non-diagonal ridge).

  Specifically, we handle two cases:
  1. Complete 'Y' shape where diagonals have the same length. For this case,
  we find the intersection of the non-diagonal ridge with the rectangle edges
  and build two directional creases between intersected edge's endpoints and the
  intersection point (pointing from the edge's endpoint to the intersection).
  2. Incomplete 'Y' shape where one diagonal is shorter. For this case, we
  will have three crease segments: two follow the same logic as in case 1, and
  one more is built to fill the gap created by the shorter diagonal.

  Args:
    rect: The rectangle to process.
    diagonal_ridges: The two diagonal ridges.
    non_diagonal_ridge: The non-diagonal ridge.
    ridge_tip: The ridge tip.

  Returns:
    A list of tuples of creases representing the outermost border edges of the
    'Y' shape.
  """
  # Case 1: Complete 'Y' shape where diagonals have the same length.
  if math.isclose(diagonal_ridges[0].length, diagonal_ridges[1].length):
    intersection_result = _find_ridge_edge_intersection(
        non_diagonal_ridge, rect.edges
    )
    if not intersection_result:
      raise ValueError(
          'Non-diagonal ridge in Y-shape does not intersect any rectangle edge.'
      )

    intersection_point, intersected_edge = intersection_result
    return [
        (
            crease_utils.Crease(
                intersected_edge.p1,
                intersection_point,
                fold=crease_utils.Fold.BORDER,
            ),
        ),
        (
            crease_utils.Crease(
                intersected_edge.p2,
                intersection_point,
                fold=crease_utils.Fold.BORDER,
            ),
        ),
    ]

  # Case 2: Incomplete 'Y' shape where one diagonal is shorter.
  else:
    diagonals = sorted(diagonal_ridges, key=lambda r: r.length)
    short_ridge, long_ridge = diagonals[0], diagonals[1]
    diagonal_non_tip_point_long_ridge = [
        point for point in [long_ridge.p1, long_ridge.p2]
        if point != ridge_tip
    ][0]

    # a) Find the small missing border segment created by the short ridge.
    intersection_result_short = _find_ridge_edge_intersection(
        short_ridge, rect.edges)
    if not intersection_result_short:
      raise ValueError(
          'Short diagonal ridge in incomplete-Y shape does not intersect any'
          ' edge.'
      )

    intersection_short, edge_short = intersection_result_short

    # Find which corner of the edge_short is the correct starting point of the
    # new border segment.
    start_point = None
    assert non_diagonal_ridge.is_horizontal or non_diagonal_ridge.is_vertical
    for corner in [edge_short.p1, edge_short.p2]:
      if (
          non_diagonal_ridge.is_horizontal
          and corner.x == diagonal_non_tip_point_long_ridge.x
      ):
        start_point = corner
        break
      elif (
          non_diagonal_ridge.is_vertical
          and corner.y == diagonal_non_tip_point_long_ridge.y
      ):
        start_point = corner
        break

    if start_point is None:
      raise ValueError(
          'Could not find the missing border segment for the incomplete-Y'
          ' shape.'
      )
    candidate_short_part_1 = crease_utils.Crease(
        start_point, intersection_short, fold=crease_utils.Fold.BORDER
    )
    # This is the other crease to fill the gap created by the short diagonal.

    # b) Find the two other border segments from the non-diagonal ridge.
    intersection_result_non_diag = _find_ridge_edge_intersection(
        non_diagonal_ridge, rect.edges
    )
    if not intersection_result_non_diag:
      raise ValueError(
          'Non-diagonal ridge in incomplete-Y shape does not intersect any'
          ' edge.'
      )

    intersection_non_diag, edge_non_diag = intersection_result_non_diag
    edge_non_diag_candidates = [
        crease_utils.Crease(
            edge_non_diag.p1,
            intersection_non_diag,
            fold=crease_utils.Fold.BORDER
        ),
        crease_utils.Crease(
            edge_non_diag.p2,
            intersection_non_diag,
            fold=crease_utils.Fold.BORDER
        )
    ]
    candidate_long, candidate_short_part_2 = sorted(
        edge_non_diag_candidates, key=lambda c: c.length, reverse=True
    )
    assert math.isclose(
        candidate_short_part_1.length + candidate_short_part_2.length,
        candidate_long.length,
        abs_tol=crease_utils.DEFAULT_TOLERANCE,
    ), (
        'Short border segments do not add up to the length of the'
        ' candidate_long.'
    )
    # Finally we order the candidates, the candidate_short_part_1 and
    # candidate_short_part_2 are the two sub-parts of the border so we group
    # them together.
    outermost_borders = [
        (candidate_long,),
        (candidate_short_part_1, candidate_short_part_2),
    ]
    return outermost_borders


def calculate_flap_border_directional_creases_on_2d_border(
    shaping_rectangle_dict: dict[str, shaping_utils.Rectangle]
) -> dict[str, list[tuple[crease_utils.Crease, ...]]]:
  """Determines 2D paper-border creases for a list of shaping rectangles.

  Segments are calculated based on their internal ridge and flap type. Note that
  the creases are directional, pointing from p1_init to p2_init.

  We calculate the border creases by analyzing the internal geometry of the
  rectangles, specifically their ridges and circle centers. We then employ
  heuristic rules to assign the outermost border creases based on the
  following criteria:
  1. One diagonal ridge, no non-diagonal ridge: Simple corner flap.
  2. One diagonal ridge, one non-diagonal ridge: Half-Y shape.
  3. Two diagonal ridges: Half-X shape.
  4. One non-diagonal ridge, two diagonal ridges: Y shape.

  Args:
      shaping_rectangle_dict: A dictionary of Rectangle objects to process.

  Returns:
      A dictionary mapping each rectangle's name to a list of its
      calculated outermost border creases.
  """
  boundary_dict = {}

  for rect_name, rect in shaping_rectangle_dict.items():
    # For now we are not handling central flaps well
    assert (
        rect.flap_type != shaping_utils.FlapType.INTERNAL
    ), f"Rectangle '{rect_name}' is an internal flap, which is not supported."

    ridges = rect.get_ridges()
    circle_centers = rect.circle_centers
    if (
        len(circle_centers) == 2
        and circle_centers[0].is_internal
        and circle_centers[1].is_internal
    ):
      raise ValueError(
          f"Rectangle '{rect_name}' has two internal circle centers, which is a"
          ' unsupported central flap.'
      )

    non_diagonal_ridge = None
    diagonal_ridges = []
    for r in ridges:
      if not r.is_diagonal:
        non_diagonal_ridge = shaping_utils.restrict_creases_to_paper([r])[0]
      elif r.p1.is_internal or r.p2.is_internal:
        diagonal_ridges.append(r)
      else:
        continue
    diagonal_ridges = shaping_utils.restrict_creases_to_paper(diagonal_ridges)

    rect = rect.clip_to_paper()
    assert rect is not None
    try:
      # Case 1: Simple corner flap, one diagonal ridge, no non-diagonal ridge.
      if non_diagonal_ridge is None and len(diagonal_ridges) == 1:
        outermost_borders = (
            _get_border_edges_for_ridge_corner_flap(rect)
        )
      # Case 2: Half-Y shape, one diagonal ridge, one non-diagonal ridge.
      elif non_diagonal_ridge is not None and len(diagonal_ridges) == 1:
        outermost_borders = _get_border_edges_for_half_y_shape_flap(
            rect, diagonal_ridges[0], non_diagonal_ridge
        )
      # Case 3: Half-X shape, two diagonal ridges, no non-diagonal ridge.
      elif non_diagonal_ridge is None and len(diagonal_ridges) == 2:
        ridge_tip = _determine_ridge_tip(circle_centers)
        outermost_borders = _get_border_edges_for_half_x_shape(rect, ridge_tip)
      # Case 4: Y shape, one non-diagonal ridge, two diagonal ridges.
      elif non_diagonal_ridge is not None and len(diagonal_ridges) == 2:
        ridge_tip = _determine_ridge_tip(circle_centers)
        outermost_borders = _get_border_edges_for_y_shape(
            rect, diagonal_ridges, non_diagonal_ridge, ridge_tip
        )

      else:
        # Optional: Log a warning for unhandled or ambiguous cases.
        logging.warning(
            "Warning: Skipping rectangle '%s' due to unhandled geometric"
            ' configuration.', rect.name
        )
        continue

      assert len(outermost_borders) == 2, (
          f'Rectangle {rect.name} has {len(outermost_borders)} border'
          ' segments.'
      )
      if not isinstance(outermost_borders[0], tuple):
        # For Simple corner flap, Half-Y shape, and Half-X shape.
        # So we align the type of outermost_borders to be the same for all
        # cases, i.e., a list of two tuples, where each tuple contains one or
        # more Crease objects.
        outermost_borders = [(outermost_borders[0],), (outermost_borders[1],)]
      boundary_dict[rect.name] = outermost_borders

    except ValueError as e:
      logging.warning("Error processing rectangle '%s': %s", rect.name, e)
      continue

  return boundary_dict


def get_stick_side_dict(
    stickfigure: stick_figure_lib.StickFigure,
    first_stick: stick_lib.Stick,
    root_direction_follow_stick_figure_root: bool,
) -> dict[str, bool]:
  """Gets the stick side dictionary for each flap.

  This function identifies which side each stick is on relative to a
  `first_stick`. The traversal starts from both ends of the `first_stick` and
  assigns a boolean value (True/False) to each stick based on which side of the
  `first_stick` it is reached from. E.g., if the `first_stick` is a top-side
  flap, all the sticks reached from the 'top' joint will be True, and those
  from the 'bottom' joint will be False.

  Args:
    stickfigure: The stick figure graph.
    first_stick: The initial stick from which to determine the sides.
    root_direction_follow_stick_figure_root: If True, the 'top' joint is
      `first_stick.end` and 'bottom' is `first_stick.start`. Otherwise, it's
      the reverse.

  Returns:
    A dictionary mapping each stick's name to a boolean indicating its side.
    Sticks reached from the 'top' joint are True, and those from the 'bottom'
    joint are False.
  """
  stick_side_dict = {}
  if root_direction_follow_stick_figure_root:
    top_joint_name = first_stick.end
    bot_joint_name = first_stick.start
  else:
    top_joint_name = first_stick.start
    bot_joint_name = first_stick.end

  for stick in tr.breadth_first_traversal(
      stickfigure,
      start_joint=top_joint_name,
      start_stick_name=first_stick.name,
      excluded_stick_names={first_stick.name},
  ):
    stick_side_dict[stick.name] = True

  for stick in tr.breadth_first_traversal(
      stickfigure,
      start_joint=bot_joint_name,
      start_stick_name=first_stick.name,
      excluded_stick_names={first_stick.name},
  ):
    stick_side_dict[stick.name] = False

  return stick_side_dict


def _find_intersection_with_bank(
    bank_creases: list[crease_utils.Crease],
    border_edge: crease_utils.Crease
) -> point_utils.Point | None:
  """Helper to find the intersection point between a river bank and a border edge.

  Args:
    bank_creases: The list of creases forming one side (bank) of the river.
    border_edge: One specific edge of the paper border.

  Returns:
    The intersection Point if found, otherwise None.
  """
  for crease in bank_creases:
    intersection = crease.vertex_with_crease(border_edge)
    if intersection:
      return intersection
  return None


def get_root_border_creases(
    root: shaping_elements.ShapingRiver | shaping_utils.Rectangle,
) -> list[crease_utils.Crease]:
  """Calculates the intersections between a root shaping element and the border.

  The points that lie on the same border form new creases. E.g., two points
  that lie on the top border will be converted to a single Crease object on the
  border.

  For a river root, we expect to have 4 intersection points, which we group into
  two valid pairs of points on two distinct borders. For a rectangle root, we
  directly get the border overlaps.

  Args:
    root: The root object, which is either a river or a rectangle.

  Returns:
    A list containing two new Crease objects. Each crease represents a
    segment on a border that the river crosses. For example:
    [Crease(p1, p2, fold=Fold.BORDER), Crease(p3, p4, fold=Fold.BORDER)],
    where p1 and p2 are on the same border.

  Raises:
    ValueError: If the number of unique intersection points is not 4,
                or if the 4 points cannot be grouped into two valid pairs.
  """
  if isinstance(root, shaping_elements.ShapingRiver):
    border = shaping_utils.Rectangle(
        name='paper', x_start=0, y_start=0, x_len=1, y_len=1
    )

    valid_segments = []

    # 3. Iterate through each of the 4 edges of the paper
    for border_edge in border.edges:
      # Find where Bank 1 intersects this specific border edge.
      p1 = _find_intersection_with_bank(list(root.bank_1), border_edge)

      # Find where Bank 2 intersects this specific border edge.
      p2 = _find_intersection_with_bank(list(root.bank_2), border_edge)

      # If both banks intersect this specific edge, the segment between them
      # is the root border crease we are looking for.
      # Note: If a bank hits a corner (e.g., 0,0), it will be detected on
      # both the Bottom edge loop and the Left edge loop, which is correct.
      if p1 and p2 and not p1.isclose(p2):
        valid_segments.append(
            crease_utils.Crease(p1, p2, fold=crease_utils.Fold.BORDER)
        )

    # 4. Validation: We expect exactly one entry and one exit for the river.
    if len(valid_segments) != 2:
      raise ValueError(
          f'Expected 2 border segments, found {len(valid_segments)}.'
      )
    border_segments = valid_segments
  elif isinstance(root, shaping_utils.Rectangle):
    border_segments = root.get_paper_border_overlapping_edges()
  else:
    raise ValueError(f'Unsupported root type: {type(root)}')
  return border_segments


def get_initial_point_from_border_creases(
    root_border_segments: list[crease_utils.Crease],
) -> point_utils.Point:
  """Returns the initial point of the river from the river border segments.

  This is used to determine the initial point of the border walk. For now we
  are returning the point with the largest x value.

  Args:
    root_border_segments: A list of river border segments.

  Returns:
    The initial point of the border walk.
  """
  p1 = root_border_segments[0].p1
  p2 = root_border_segments[0].p2
  p3 = root_border_segments[1].p1
  p4 = root_border_segments[1].p2
  start_point = max([p1, p2, p3, p4], key=lambda p: p.x)
  return start_point


def _determine_initial_direction(point: point_utils.Point) -> Direction:
  """Determines initial counter-clockwise traversal direction from the start point.

  Args:
      point: The starting Point object, which must be on the border.

  Returns:
      An integer representing the initial direction (Direction.UP,
      Direction.LEFT, Direction.DOWN, or Direction.RIGHT).

  Raises:
      ValueError: If the point is not located on any of the four grid borders.
  """
  # Check for corner positions first to handle them unambiguously.
  if point.isclose(point_utils.Point(x=1, y=0)):  # Top-right corner
    return Direction.LEFT
  if point.isclose(point_utils.Point(x=0, y=0)):  # Top-left corner
    return Direction.DOWN
  if point.isclose(point_utils.Point(x=0, y=1)):  # Bottom-left corner
    return Direction.RIGHT
  if point.isclose(point_utils.Point(x=1, y=1)):  # Bottom-right corner
    return Direction.UP

  # If not a corner, check which edge the point is on.
  if math.isclose(point.x, 1.0):  # Right edge
    return Direction.UP
  if math.isclose(point.y, 0.0):  # Top edge
    return Direction.LEFT
  if math.isclose(point.x, 0.0):  # Left edge
    return Direction.DOWN
  if math.isclose(point.y, 1.0):  # Bottom edge
    return Direction.RIGHT

  raise ValueError(f'Start point {point} is not on a grid border.')


def traverse_paper_border_counter_clockwise_to_select_border_creases(
    start_point: point_utils.Point,
    grid_size: int,
    # NOTE: Support river border.
    flap_border_creases_dict: dict[str, list[tuple[crease_utils.Crease, ...]]],
    stick_side_dict: dict[str, bool],
) -> dict[str, crease_utils.Crease]:
  """Traverses the paper border CCW to select the flap's 3D top-side border crease.

  In our 3D analysis, the flap's normal vector (0, 0, 1) defines the top (in 3D,
  not the stick figure's top) side of the flap in 3D. E.g., if you flip the flap
  which makes the bottom side of the flap up, the normal vector becomes (0, 0,
  -1). We care about the border crease on the top side of the flap. This
  function is used to find it.

  This function starts from any point on the border. It checks if the path of
  the traversal intersects with a flap border crease to detect transitions
  between shaping regions. Given the stick side info, we can determine which
  of the two creases on the flap border is the one on the top in 3D analysis.

  Specifically, we check how many times the traversal has passed a certain flap
  border crease. For example, if a flap border crease is intersected by the
  traversal 4 times (2 flap border creases and each crease has two tips), we
  consider it as visited and will not check it again. The stick side determines
  whether we choose the first (visit number = 1) or the second of the flap
  border crease (visit number = 3).

  Args:
    start_point: The starting Point object on the grid border.
    grid_size: The size of the grid, used to determine the step length.
    flap_border_creases_dict: A dictionary mapping a region's name (e.g.,
      'left_arm') to a list of its boundary Crease objects.
    stick_side_dict: A dictionary mapping a region's name to its stick side.

  Returns:
      A dictionary mapping each region's name to the boundary Crease object
      that was intersected during the traversal.

  Raises:
    ValueError: If the start_point is not on any known boundary crease.
    RuntimeError: If the traversal exceeds a maximum number of steps.
  """

  # --- 1. Initialization ---

  rect_zigzag_dict = {}
  visited_rect_counts = collections.Counter(
      {rect_name: 0 for rect_name in flap_border_creases_dict.keys()}
  )
  current_point = start_point

  # Use the helper function to set the initial traversal direction.
  direction = _determine_initial_direction(start_point)

  # --- 2. Main Traversal Loop ---

  grid_unit_length = 1.0 / grid_size
  max_steps = 4 * grid_size + 10  # A safe upper limit for steps

  for _ in range(max_steps):
    # Check for intersection with any unvisited boundary.
    for rect_name, boundary_creases in flap_border_creases_dict.items():
      if visited_rect_counts[rect_name] == 4:
        continue
      # We pick the first crease in each tuple
      boundary_creases = [boundary_creases[0][0], boundary_creases[1][0]]
      for boundary_crease in boundary_creases:
        if (
            boundary_crease.point_status(current_point)
            == crease_utils.PointStatus.TIP
        ):
          visited_rect_counts[rect_name] += 1
          if (
              stick_side_dict[rect_name]
              and visited_rect_counts[rect_name] == 1
          ) or (
              not stick_side_dict[rect_name]
              and visited_rect_counts[rect_name] == 3
          ):
            # 3 means touching the second candidate crease on the flap border
            # (because one flap has two sides so we need to determine the one
            # at the topside).
            rect_zigzag_dict[rect_name] = boundary_crease
            break

    previous_point = current_point

    # NOTE: Reuse shaping_boundary.walk method
    # Move one step to find the next point_utils.
    if direction == Direction.UP:
      # NOTE: check potential numerical issue for grid_unit_length
      next_pos_coords = (
          previous_point.x,
          previous_point.y - grid_unit_length,
      )
    elif direction == Direction.LEFT:
      next_pos_coords = (
          previous_point.x - grid_unit_length,
          previous_point.y,
      )
    elif direction == Direction.DOWN:
      next_pos_coords = (
          previous_point.x,
          previous_point.y + grid_unit_length,
      )
    elif direction == Direction.RIGHT:  # Direction.RIGHT
      next_pos_coords = (
          previous_point.x + grid_unit_length,
          previous_point.y,
      )
    else:
      raise ValueError(f'Unsupported direction: {direction}')

    current_point = point_utils.Point(
        x=next_pos_coords[0], y=next_pos_coords[1]
    )

    # --- 3. Corner Turning Logic ---

    # Check if the traversal has reached a corner of the grid to change
    # direction.
    # Top-right
    if current_point.isclose(point_utils.Point(x=1, y=0)):
      direction = Direction.LEFT
    # Top-left
    elif current_point.isclose(point_utils.Point(x=0, y=0)):
      direction = Direction.DOWN
    # Bottom-left
    elif current_point.isclose(point_utils.Point(x=0, y=1)):
      direction = Direction.RIGHT
    # Bottom-right
    elif current_point.isclose(point_utils.Point(x=1, y=1)):
      direction = Direction.UP

    if current_point.isclose(start_point):
      return rect_zigzag_dict

  raise RuntimeError(
      'Traversal failed to complete a full loop around the border.'
  )


def _get_assignment_for_touched_pleat(
    p: point_utils.Point, pleats: list[crease_utils.Crease]
) -> crease_utils.Fold:
  """Returns the fold for a given point that is on a pleat."""
  fold = None
  for pleat in pleats:
    if pleat.point_status(p) != crease_utils.PointStatus.EXTERNAL:
      fold = pleat.fold
      break
  if fold is None:
    raise ValueError('Point is not on a pleat')
  return fold


def _get_x_axis_crease_vector_heuristic(
    x_axis_point: point_utils.Point,
    y_axis_crease: crease_utils.Crease,
    pleats: list[crease_utils.Crease],
    stick_side: bool,
) -> tuple[crease_utils.Crease, crease_utils.Fold]:
  """Returns the x axis crease vector for a given y axis crease.

  According to the stick side and the touched pleat assignment, we develop a
  heuristic to determine the x axis crease vector and make sure the x_axis's
  p1_init is the origin point in the 3D analysis.

  Args:
    x_axis_point: The x axis point on the flap plane.
    y_axis_crease: The y axis crease.
    pleats: The list of pleats.
    stick_side: Whether the x_axis_point is on the top or bottom side.

  Returns:
    The x axis crease vector and the touched pleat assignment.
  Raises:
    ValueError: If the x_axis_point is not on a pleat.
  """
  touch_pleat_assignment = _get_assignment_for_touched_pleat(
      x_axis_point, pleats
  )
  match (touch_pleat_assignment, stick_side):
    case (crease_utils.Fold.MOUNTAIN, True):
      # x from M to V
      x_axis_crease = crease_utils.Crease(
          x_axis_point, y_axis_crease.p1_init, fold=crease_utils.Fold.MOUNTAIN
      )
    case (crease_utils.Fold.MOUNTAIN, False):
      x_axis_crease = crease_utils.Crease(
          y_axis_crease.p1_init, x_axis_point, fold=crease_utils.Fold.MOUNTAIN
      )
    case (crease_utils.Fold.VALLEY, True):
      x_axis_crease = crease_utils.Crease(
          y_axis_crease.p1_init, x_axis_point, fold=crease_utils.Fold.MOUNTAIN
      )
    case (crease_utils.Fold.VALLEY, False):
      x_axis_crease = crease_utils.Crease(
          x_axis_point, y_axis_crease.p1_init, fold=crease_utils.Fold.MOUNTAIN
      )
    case _:
      raise ValueError(
          'Unsupported touch_pleat_assignment and stick_side combination.'
      )
  return x_axis_crease, touch_pleat_assignment


# NOTE: Reuse elevation level to determine the x axis.
def get_directional_xy_axis_dict(
    shaping_rectangle_dict: dict[str, shaping_utils.Rectangle],
    flap_border_creases_dict: dict[str, crease_utils.Crease],
    pleats: list[crease_utils.Crease],
    stick_side_dict: dict[str, bool],
    grid_size: int,
) -> dict[str, XYAxisCrease]:
  """Gets the directional x and y axis creases for each flap.

  This function iterates through each flap, uses the flap border crease (which
  represents the y-axis crease for a flap in 3D analysis) and determines the
  corresponding x-axis crease. The x-axis crease is found by searching for a
  point on the edge of the flap rectangle that is one grid unit away from the
  y-axis crease's start point, and then constructing a crease between this point
  and the y-axis crease's start point.

  Note that both x and y axis creases are directional, pointing from p1_init to
  p2_init.

  Args:
    shaping_rectangle_dict: A dictionary mapping rectangle names to their
      shaping_utils.Rectangle objects.
    flap_border_creases_dict: A dictionary mapping rectangle names to the crease
      representing the y-axis for the flap.
    pleats: A list of all pleat creases in the pattern.
    stick_side_dict: A dictionary mapping rectangle names to a boolean
      indicating the stick side.
    grid_size: The size of the grid.

  Returns:
    A dictionary mapping rectangle names to an XYAxisCrease object, containing
    both the determined x-axis and the provided y-axis crease.
  """
  xy_axis_dict = {}
  grid_length = 1 / grid_size
  for rect_name, y_axis_crease in flap_border_creases_dict.items():
    flap_rect = shaping_rectangle_dict[rect_name].clip()
    assert flap_rect is not None
    x_axis_crease = None
    touch_pleat_assignment = None
    for walk_length in [-grid_length, grid_length]:
      if y_axis_crease.is_horizontal:
        # Try different walk, since the y_axis is horizontal, the x_axis must
        # be vertical so we keep x the same and only change y. The candidate
        # point must be on the border of the flap rectangle.
        candidate = point_utils.Point(
            x=y_axis_crease.p1_init.x,
            y=y_axis_crease.p1_init.y + walk_length,
            require_on_paper=False,
        )
      elif y_axis_crease.is_vertical:
        candidate = point_utils.Point(
            x=y_axis_crease.p1_init.x + walk_length,
            y=y_axis_crease.p1_init.y,
            require_on_paper=False,
        )
      else:
        raise ValueError('Unsupported y_axis_crease type')
      if flap_rect.is_edge_point(candidate):
        x_axis_crease, touch_pleat_assignment = (
            _get_x_axis_crease_vector_heuristic(
                candidate, y_axis_crease, pleats, stick_side_dict[rect_name]
            )
        )
        break
    # NOTE: Check if special case exists that both walks fail.
    assert x_axis_crease is not None and touch_pleat_assignment is not None
    xy_axis_dict[rect_name] = XYAxisCrease(
        x_axis_crease=x_axis_crease,
        y_axis_crease=y_axis_crease,
        x_axis_touched_pleat_assignment=touch_pleat_assignment,
    )
  return xy_axis_dict


def get_directorial_xy_axis_crease_on_flap(
    shaping_rectangle_dict: dict[str, shaping_utils.Rectangle],
    shaping_root: shaping_elements.ShapingRiver | shaping_utils.Rectangle,
    pleats: list[crease_utils.Crease],
    stick_side_dict: dict[str, bool],
    grid_size: int,
) -> dict[str, XYAxisCrease]:
  """Gets the directional x and y axis creases for each flap by orchestrating the helper functions.

  So we have the x,y axis creases on the flap, we can convert x,y axis in 3D
  space to 2D space.

  This function first finds the border segments of the root element, determines
  a starting point for traversal, and then walks the paper border to find the
  corresponding border creases on each flap (as y-axis creases). Finally, it
  computes the XYAxisCrease for each flap.

  Args:
    shaping_rectangle_dict: A dictionary mapping rectangle names to their
      shaping_utils.Rectangle objects.
    shaping_root: The root ShapingRiver or Rectangle element.
    pleats: A list of all pleat creases in the pattern.
    stick_side_dict: A dictionary mapping a flap's name to its stick side (top
      or bottom).
    grid_size: The size of the grid.

  Returns:
    A dictionary mapping rectangle names to their computed XYAxisCrease.
  """
  # First calculate potential flap's border creases inside the 2D paper border.
  # For each flap, there are two creases on the paper border that is on the
  # surface border of the flap. This function is used to find them.
  potential_flap_border_creases_dict = (
      calculate_flap_border_directional_creases_on_2d_border(
          shaping_rectangle_dict
      )
  )

  # First find the intersection points of the river borders with the paper
  # border, and use this as the starting point for the traversal.
  root_border = get_root_border_creases(shaping_root)
  start_point = get_initial_point_from_border_creases(root_border)

  # Given flap_border_creases_dict, now we identify the topside crease from the
  # two candidate creases (because one flap has two sides so we need to
  # determine the one at the topside in 3D analysis).

  # Here we walk the border of the paper counter-clockwisely to
  # determine that.
  # For top side flap, it is the first crease we met when we walk
  # counter-clockwisely; for bottom side flap, it is the third crease we met.
  flap_border_creases_dict = (
      traverse_paper_border_counter_clockwise_to_select_border_creases(
          start_point=start_point,
          grid_size=grid_size,
          flap_border_creases_dict=potential_flap_border_creases_dict,
          stick_side_dict=stick_side_dict,
      )
  )

  # The border crease is parallel to the y-axis crease on the flap (in 3D
  # analysis). So the next step is to identify the x-axis crease on the flap.
  # Then we have the transformation from 2D crease pattern space to 3D space.
  xy_axis_dict = get_directional_xy_axis_dict(
      shaping_rectangle_dict=shaping_rectangle_dict,
      flap_border_creases_dict=flap_border_creases_dict,
      pleats=pleats,
      stick_side_dict=stick_side_dict,
      grid_size=grid_size
  )
  return xy_axis_dict


def _get_fold_order_from_angle_and_touched_pleat_assignment(
    rotation_angle: float, touched_pleat_assignment: crease_utils.Fold
) -> crease_utils.Fold:
  """Returns the fold order and phase from the rotation angle and touched pleat assignment."""

  # flap_posture.rotation_angle > 0 represents a valley fold on the flap
  # surface, the touched_pleat_assignment can indicate which side of the paper
  # is on the top.
  # If the touched_pleat assignment is VALLEY, the other side of the paper is
  # up, so we need to flip the fold.

  # If np.float64, convert to float.
  if isinstance(rotation_angle, np.float64):
    rotation_angle = rotation_angle.item()
  match (
      rotation_angle > 0,
      touched_pleat_assignment == crease_utils.Fold.VALLEY
  ):
    case (True, False):
      fold = crease_utils.Fold.VALLEY
    case (True, True):
      fold = crease_utils.Fold.MOUNTAIN
    case (False, False):
      fold = crease_utils.Fold.MOUNTAIN
    case (False, True):
      fold = crease_utils.Fold.VALLEY
    case _:
      raise ValueError(
          f'Rotation angle {rotation_angle} and touched_pleat_assignment'
          f' {touched_pleat_assignment} is not supported.'
      )
  return fold


def get_crease_from_local_rotation_axis_and_xy_axis_dict(
    flap_post_dict: dict[str, Any],
    xy_axis_dict: dict[str, XYAxisCrease],
    grid_size: int,
) -> tuple[dict[str, crease_utils.Crease], dict[str, float]]:
  """Calculates a new 'zigzag' crease for each flap based on its local rotation axis.

  The local rotation axis provides the assignment and depth of the zigzag
  crease. The new crease is then formed by shifting one of the endpoints of the
  x-axis crease the depth distance along the direction of the y-axis crease.

  Args:
    flap_post_dict: A dictionary mapping rectangle names to FlapPosture objects.
    xy_axis_dict: A dictionary mapping rectangle names to objects containing
      directional x and y axis creases.
    grid_size: The size of the grid.

  Returns:
    A dictionary mapping rectangle names to the newly computed zigzag
    crease, and a dictionary mapping rectangle names to the depth of the
    rotation axis.
  """
  zigzag_creases = {}  # Initialize a dictionary to store all results
  depth_dict = {}
  for rect_name, flap_posture in flap_post_dict.items():
    # Skip flaps that don't have rotation
    if flap_posture.rotation_angle == 0:
      continue

    depth = flap_posture.rotation_axis_depth
    # NOTE: Add support for this case by using the inscribed polygon.
    if depth is None:
      continue
    depth_dict[rect_name] = abs(depth)

    # Depth is None if the rotation axis is parallel to the x-axis.
    # NOTE: Add support for this case by using the inscribed polygon
    if depth is None:
      continue

    xy_axes = xy_axis_dict[rect_name]
    x_axis_crease = xy_axes.x_axis_crease
    y_axis_crease = xy_axes.y_axis_crease
    touched_pleat_assignment = xy_axes.x_axis_touched_pleat_assignment

    # --- Calculation of the y-axis unit vector ---
    # 1. Get the vector for the y-axis crease by subtracting components
    y_vec_x = y_axis_crease.p2_init.x - y_axis_crease.p1_init.x
    y_vec_y = y_axis_crease.p2_init.y - y_axis_crease.p1_init.y
    y_vector_np = np.array([y_vec_x, y_vec_y])

    # 2. Calculate its norm (length)
    norm = np.linalg.norm(y_vector_np)

    # The direction of the y-axis crease
    y_unit_vector_np = y_vector_np / norm

    # Calculate the shift vector's components
    shift_vec_x = abs(depth) * 1 / grid_size * y_unit_vector_np[0]
    shift_vec_y = abs(depth) * 1 / grid_size * y_unit_vector_np[1]

    # p1_x is expected to be the origin point in 3D analysis.
    p1_x = x_axis_crease.p1_init
    p2_x = x_axis_crease.p2_init

    # Get the fold order from the rotation angle and touched pleat assignment
    fold = _get_fold_order_from_angle_and_touched_pleat_assignment(
        flap_posture.rotation_angle, touched_pleat_assignment
    )
    if depth > 0:
      # For positive ratio, shift p2 by adding components
      new_p2 = point_utils.Point(
          x=p2_x.x + shift_vec_x,
          y=p2_x.y + shift_vec_y
      )
      fold_percent = abs(flap_posture.rotation_angle) / 180
      new_crease = crease_utils.Crease(
          p1_x, new_p2, fold=fold, fold_percent=fold_percent)
    else:
      # For negative ratio, shift p1 by adding components, then create the
      # crease
      new_p1 = point_utils.Point(
          x=p1_x.x + shift_vec_x,
          y=p1_x.y + shift_vec_y
      )
      # Note: The original logic created the crease from p2_x to new_p1
      fold_percent = abs(flap_posture.rotation_angle) / 180
      new_crease = crease_utils.Crease(
          p2_x, new_p1, fold=fold, fold_percent=fold_percent)

    zigzag_creases[rect_name] = new_crease

  return zigzag_creases, depth_dict


def _check_overlapping_zigzag_crease(target_crease, zigzag_creases):
  """Checks if the target crease is present in the list of zigzag creases."""
  # Add a tolerance to the overlapping check to account for numerical errors.
  return any(
      target_crease.is_overlapping(zigzag_crease, ndigits=NDIGITS_OVERLAPPING)
      and zigzag_crease.fold == target_crease.fold
      for zigzag_crease in zigzag_creases
  )


def get_zigzag_fold_order_and_phase_by_sweeping(
    target_crease: crease_utils.Crease,
    grid_size: int,
    rect: shaping_utils.Rectangle,
    zigzag_depth: float,
):
  """Sweeps fold orders and phases to find parameters overlapping the target crease.

  Note that we only check overlapping and fold assignment, not direct identity
  matching. This is because the zigzag crease we calculate is the crease in 3D
  space, where we assume infinite paper surface. However, a flap has finite
  space and is composed of multiple layers of paper. So ideally we just need the
  real zigzag crease.

  Args:
    target_crease: The target crease to be matched.
    grid_size: The size of the grid.
    rect: The rectangle to be shaped.
    zigzag_depth: The depth of the zigzag.

  Returns:
    The zigzag parameters that result in a crease that overlaps with the target
    crease.
  """
  fold_order_final = None
  is_even_phase_final = None
  for fold_order in [FO_V, FO_M]:
    for even_phase in [True, False]:
      zigzag_element_params = [
          zigzag_shaping.ZigzagShapingParamsForElement(
              element_name=rect.name,
              zigzag_params=shaping_types.ZigzagParams(
                  target_flap_angle=None,
                  target_zigzag_depth=zigzag_depth,
                  fold_order=fold_order,
                  even_phase=even_phase,
                  fold_percent=target_crease.fold_percent,
              ),
          )
      ]
      zigzag_creases = zigzag_shaping.get_zigzag_shaping_creases(
          grid_size=grid_size,
          rectangles=[rect],
          element_shaping_params=zigzag_element_params,
          snap_to_grid=False,
          fold_step=1,
      )
      if _check_overlapping_zigzag_crease(target_crease, zigzag_creases):
        fold_order_final = fold_order
        is_even_phase_final = even_phase
        break
  assert fold_order_final is not None
  assert is_even_phase_final is not None
  return shaping_types.ZigzagParams(
      target_flap_angle=None,
      target_zigzag_depth=zigzag_depth,
      fold_order=fold_order_final,
      even_phase=is_even_phase_final,
      fold_percent=target_crease.fold_percent,
  )


def get_zigzag_params_from_crease_dict(
    crease_dict: dict[str, crease_utils.Crease],
    grid_size: int,
    rect_dict: dict[str, shaping_utils.Rectangle],
    depth_dict: dict[str, float],
) -> list[zigzag_shaping.ZigzagShapingParamsForElement]:
  """Calculates the full list of zigzag parameters for all rectangles.

  Specifically, we sweep through all possible fold orders and phases for each
  rectangle, and find the one that results in a crease that overlaps with the
  target crease.

  Args:
    crease_dict: A dictionary mapping rectangle names to their creases.
    grid_size: The size of the grid.
    rect_dict: A dictionary mapping rectangle names to their rectangles.
    depth_dict: A dictionary mapping rectangle names to their depths.
  Returns:
    A list of ZigzagShapingParamsForElement objects, one for each rectangle.
  """
  zigzag_params = []
  for rect_name, crease in crease_dict.items():
    rect = rect_dict[rect_name]
    zigzag_params.append(
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name=rect.name,
            zigzag_params=get_zigzag_fold_order_and_phase_by_sweeping(
                crease, grid_size, rect, depth_dict[rect_name]
            ),
        )
    )
  return zigzag_params
