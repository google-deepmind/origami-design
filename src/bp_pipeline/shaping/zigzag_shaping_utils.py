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

"""Utilities for converting relative angles and directions to zigzag inputs."""

import collections
import math

from absl import logging

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import point as point_utils
from . import heuristics
from . import shaping_elements
from . import shaping_utils
from . import types as shaping_types
from . import zigzag_shaping
from .. import timeout as timeout_lib


MIN_ANGLE = math.radians(20)
NINETY_DEGREE_THRESHOLD = math.pi / 16
HINGE_FF_SHIFT_DEPTH = 0.3
MOUNTAIN_TO_VALLEY = [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY]
VALLEY_TO_MOUNTAIN = [crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN]


# -------- Helper utils --------


def clip_angle_by_min_angle(
    theta: float,
    min_angle: float = MIN_ANGLE
) -> float:
  """Clips the shape angle to the minimum angle allowed."""
  return shaping_utils.sign(theta) * max(min_angle, abs(theta))


def _is_valid_and_nontrivial_angle(theta: float | None) -> bool:
  """Checks if the shape angle is valid.

  Returns False if angle is None or trivial (0).

  Args:
    theta: The shape angle.

  Raises:
    ValueError: If angle is not in (-pi, pi).

  Returns:
    True if the shape angle is valid, False otherwise.
  """
  if theta is None:
    return False
  if not point_utils.in_range(theta, -math.pi, math.pi, strict=True):
    raise ValueError(f'Theta {theta} is not in (-pi, pi).')
  return not math.isclose(theta, 0)


# NOTE: in future filter trees datasets themselves by shapability
def _clip_angle_by_depth(theta: float, depth: float) -> float:
  """Clips the shape angle to the minimum angle allowed by the depth."""
  min_theta = shaping_utils.fold_angle_from_zigzag_depth(depth)
  return shaping_utils.sign(theta) * max(min_theta, abs(theta))


# -------- Flap utils --------


def _get_creases_on_point(
    point: point_utils.Point,
    creases: list[crease_utils.Crease],
) -> list[crease_utils.Crease]:
  """Returns the creases containing a given point."""
  incident_creases = []
  for crease in creases:
    if (
        crease.point_status(point) != crease_utils.PointStatus.EXTERNAL
    ):
      incident_creases.append(crease)
  return incident_creases


def check_point_elevation(
    point: point_utils.Point,
    creases: list[crease_utils.Crease],
) -> int | None:
  """Checks if a point is on an elevation 1 or 0 pleat.

  Args:
    point: The point to check.
    creases: The list of creases used to check the point's elevation.

  Returns:
    The first valid integer elevation of an incident crease, or None if no valid
    elevation is found.
  """
  incident_creases = _get_creases_on_point(point, creases)
  return next(
      (c.elevation for c in incident_creases if c.elevation is not None), None
  )


def get_next_point_edge_index_and_elevation(
    point: point_utils.Point,
    edge_index: int,
    creases: list[crease_utils.Crease],
    boundary: shaping_elements.ShapingBoundary,
    grid_size: int,
) -> tuple[point_utils.Point, int, int | None]:
  """Returns the next point, edge index, and elevation (if any) for a given point."""
  next_point, next_edge_index = boundary.walk(point, edge_index, grid_size)
  next_elevation = check_point_elevation(next_point, creases)
  return next_point, next_edge_index, next_elevation


def partition_boundary_by_elevation(
    boundary: shaping_elements.ShapingBoundary,
    grid_size: int,
    creases: list[crease_utils.Crease],
) -> tuple[list[point_utils.Point], list[point_utils.Point]]:
  """Partitions flap boundary points into elevation 0 (Valley) and 1 (Mountain) lists.

  Partitioning is done by splitting adjacent grid points on the flap boundary
  into two separate classes. Each class is then assigned an elevation by
  estimating the elevation of each point in the class and taking the majority
  point elevation.

  Args:
    boundary: The boundary to partition.
    grid_size: The size of the grid.
    creases: The list of creases intersecting the boundary.

  Returns:
    A tuple of lists of points on elevation 0 (Valley) and elevation 1
    (Mountain).
  """
  # Initialize partition classes.
  elevation_classes = collections.defaultdict(list)

  # Initialize starting point for walk around the flap boundary.
  start, edge_index = boundary.creases[0].p1_init, 0
  elevation = check_point_elevation(start, creases)
  if elevation is not None:
    elevation_classes[elevation].append(start)

  # Walk around boundary.
  cur_p, edge_index, elevation = get_next_point_edge_index_and_elevation(
      start, edge_index, creases, boundary, grid_size
  )
  while not cur_p.isclose(start):
    timeout_lib.maybe_timeout_cooperatively()
    if elevation is not None:
      elevation_classes[elevation].append(cur_p)
    try:
      cur_p, edge_index, elevation = get_next_point_edge_index_and_elevation(
          cur_p, edge_index, creases, boundary, grid_size
      )
    # If value error, we've reached the end of the boundary.
    except ValueError:
      break

  unusual_elevations = elevation_classes.keys() - {0, 1}
  if unusual_elevations:
    logging.warning(
        'Ignoring unusual elevations found during zigzag shaping: %s',
        unusual_elevations,
    )

  # Return partition point classes
  return (
      elevation_classes[0],
      elevation_classes[1],
  )


def _get_distance_along_boundary(
    start_point: point_utils.Point,
    target_point: point_utils.Point,
    boundary: shaping_elements.ShapingBoundary,
    grid_size: int,
) -> int:
  """Returns the distance along the boundary between two points.

  Otherwise returns None if the target point cannot be reached from the start
  point.

  Args:
    start_point: The start point.
    target_point: The target point.
    boundary: The boundary to walk along.
    grid_size: The size of the grid.

  Returns:
    The distance along the boundary between the two points.
    None if the target point cannot be reached from the start point.
  """
  cur_p = start_point
  edge_index = boundary.get_edge_index(cur_p)
  distance = 0
  while not cur_p.isclose(target_point):
    timeout_lib.maybe_timeout_cooperatively()
    distance += 1
    try:
      cur_p, edge_index = boundary.walk(cur_p, edge_index, grid_size)
    # If value error, we've reached the end of the boundary.
    except ValueError as exc:
      raise RuntimeError(
          f'Target point {target_point} not reached from start point'
          f' {start_point} along boundary {boundary}.'
      ) from exc
  return distance


def _get_phase_from_anchor_at_paper_border_rect_corner(
    anchor_at_paper_border_rect_corner: bool,
    start_point: point_utils.Point,
    boundary: shaping_elements.ShapingBoundary,
    grid_size: int,
    creases: list[crease_utils.Crease],
) -> bool:
  """Returns the desired phase based on whether the zigzag is anchored at a paper border point.

  Args:
    anchor_at_paper_border_rect_corner: Whether zigzags are anchored at a
      point on the paper border.
    start_point: The point where zigzag drawing will start from if
      phase is even.
    boundary: The boundary to get the phase for.
    grid_size: The size of the grid.
    creases: The list of creases intersecting the boundary.

  Returns:
    The paper border rect corner anchor state (True/False).
  """
  # Get elevation partition classes of flap boundary.
  valley_points, mountain_points = partition_boundary_by_elevation(
      boundary, grid_size, creases,
  )

  # Choose target anchor point based on anchor_at_paper_border_rect_corner.
  # Set of points from which target can be chosen.
  target_set = (
      valley_points if anchor_at_paper_border_rect_corner else mountain_points
  )
  # Set of points adjacent to target set.
  adjacent_set = (
      mountain_points if anchor_at_paper_border_rect_corner else valley_points
  )
  # Choose point from target set if it exists.
  if target_set:
    target = target_set[0]
  # Otherwise choose adjacent point and take a step around the flap boundary.
  elif adjacent_set:
    target_adjacent = adjacent_set[0]
    target = boundary.walk(
        target_adjacent,
        boundary.get_edge_index(target_adjacent),
        grid_size,
    )
  else:
    raise ValueError(
        'No target point found for anchor_at_paper_border_rect_corner:'
        f'Boundary: {boundary}\n'
        f'Creases: {creases}\n'
    )

  start_target_perimeter_distance = _get_distance_along_boundary(
      start_point,
      target,
      boundary,
      grid_size,
  )

  # If even distance then target is anchor if and only if start is anchor.
  # Otherwise target is anchor if and only if start is not anchor.
  return start_target_perimeter_distance % 2 == 0


def _is_entering_paper_from_external_to_border(
    prev_point: point_utils.Point, cur_point: point_utils.Point
) -> bool:
  """Checks if the trajectory moves from an external point onto the paper border.

  Args:
    prev_point: The previous point.
    cur_point: The current point.

  Returns:
    True if the condition is met, False otherwise.
  """
  return prev_point.is_external and cur_point.is_border


def _is_internal_and_corner(
    prev_edge_index, cur_edge_index, cur_point: point_utils.Point
) -> bool:
  """Check whether the current point is an internal corner of the rectangle.

  Args:
    prev_edge_index: The previous edge index.
    cur_edge_index: The current edge index.
    cur_point: The current point.

  Returns:
    True if the condition is met, False otherwise.
  """
  return (
      prev_edge_index != cur_edge_index
      and cur_point.is_internal
  )


def get_potential_illegal_hinge_segment_coordinates_and_is_same_assignment(
    phase: bool, grid_size: int, rect: shaping_elements.ShapingFlap
) -> list[tuple[tuple[point_utils.Point, point_utils.Point], bool]]:
  """Get all the hinge segments and assignment from the phase that need check.

  We start with the top edge midpoint and go through the rect. Along the way,
  we identify key creases at the rectangle corner or when entering the paper as
  potential illegal hinge segment coordinates. We also record whether the fold
  assignment there is identical with the start's assignment.

  Args:
    phase: The phase of the zigzag.
    grid_size: The size of the box pleating grid.
    rect: The rectangle corresponding to the flap we are shaping.

  Returns:
    A list of tuples, where each tuple contains the coordinates of a hinge
    segment, and whether the fold assignment of the zigzag crease is identical
    with the start's assignment.
  """
  rect_corners, rect_edges, edge_vectors = (
      rect.corners,
      rect.edges,
      rect.get_unit_edge_vectors(grid_size),
  )
  top_edge_midpoint = point_utils.midpoint(rect_corners[0], rect_corners[1])

  potential_illegal_hinge_segment_coordinates_and_is_same_assignment = []
  start = shaping_utils.compute_anchor_points_start(
      top_edge_midpoint, grid_size, phase, edge_vectors[0]
  )
  is_same_assignment_with_start = True

  add_illegal_hinge_segment = False
  edge_index = shaping_utils.get_edge_index(start, rect_corners, rect_edges)
  boundary = shaping_elements.get_flap_boundary(rect, grid_size)
  cur_p, edge_index = boundary.walk(
      start, edge_index, grid_size, snap_to_grid=False
  )
  # current zigzag.
  if start.is_internal:
    illegal_hinge_segment_coordinates = (start, cur_p)
    potential_illegal_hinge_segment_coordinates_and_is_same_assignment.append(
        (illegal_hinge_segment_coordinates, is_same_assignment_with_start)
    )
  is_same_assignment_with_start = not is_same_assignment_with_start

  while not cur_p.isclose(start):
    timeout_lib.maybe_timeout_cooperatively()
    prev_edge_index = edge_index
    prev_p = cur_p
    cur_p, edge_index = boundary.walk(
        cur_p, edge_index, grid_size, snap_to_grid=False
    )
    if add_illegal_hinge_segment:
      illegal_hinge_segment_coordinates = (prev_p, cur_p)
      potential_illegal_hinge_segment_coordinates_and_is_same_assignment.append(
          (illegal_hinge_segment_coordinates, is_same_assignment_with_start)
      )
      add_illegal_hinge_segment = False

    is_same_assignment_with_start = not is_same_assignment_with_start

    # Check the CP at the corner CP or when entering the paper.
    if _is_internal_and_corner(
        prev_edge_index, edge_index, cur_p
    ) or _is_entering_paper_from_external_to_border(prev_p, cur_p):
      add_illegal_hinge_segment = True

  return potential_illegal_hinge_segment_coordinates_and_is_same_assignment


def _flip_phase_by_parent_shaping_history(
    desired_phase: bool,
    element: shaping_elements.ShapingElement,
    element_dict: dict[str, shaping_elements.ShapingElement],
    flap_info: dict[str, heuristics.Node],
) -> bool:
  """Flips the phase by the parent shaping history.

  When a parent element of a flap/river is shaped, this
  flips the flap/river's z-axis orientation. This will reverse the correct
  phase child element during shaping. We account for this here by flipping
  the phase of the child element for each shaping operation applied to
  each ancestor element.

  NOTE: This only supports 2D shaping.

  Args:
    desired_phase: The desired phase of the zigzag.
    element: The element to be shaped.
    element_dict: The dictionary of all elements.
    flap_info: The dictionary of all flaps.

  Returns:
    The flipped phase.
  """
  # Track z-rotation-angle: rotation relative to normal vector
  z_rotation_angle = 0
  for op in element.shaping_history:
    # Check that the shaping history is only 0 or 1 (in case of 2D shaping).
    assert math.isclose(op.fold_percent, 1) or math.isclose(op.fold_percent, 0)
    z_rotation_angle += round(180 * op.fold_percent)
  parent_name = flap_info[element.name].parent
  while parent_name is not None:
    timeout_lib.maybe_timeout_cooperatively()
    parent_shaping_history = element_dict[parent_name].shaping_history
    for op in parent_shaping_history:
      assert (
          math.isclose(op.fold_percent, 1) or math.isclose(op.fold_percent, 0)
      )
      z_rotation_angle += round(180 * op.fold_percent)
    parent_name = flap_info[parent_name].parent
  # NOTE: handle situation where root has been shaped
  # (in general it shouldn't be)
  is_flipped = (z_rotation_angle // 180) % 2 == 1
  desired_phase = not desired_phase if is_flipped else desired_phase
  return desired_phase


def convert_flap_angle_to_global_phase(
    theta: float,
    absolute_direction: heuristics.AbsFlapDirection,
    flap: shaping_elements.ShapingFlap,
) -> bool:
  """Converts an angle to a global phase (before parent shaping history adjustment)."""
  valley_anchored = (
      _get_anchor_from_abs_orientation_and_angle(absolute_direction, theta)
    )
  even_phase_start_point = shaping_utils.compute_anchor_points_start(
      point_utils.midpoint(flap.corners[0], flap.corners[1]),
      flap.grid_size,
      True,
      flap.get_unit_edge_vectors(flap.grid_size)[0],
  )
  boundary = shaping_elements.get_flap_boundary(flap, flap.grid_size)
  desired_phase = _get_phase_from_anchor_at_paper_border_rect_corner(
      valley_anchored,
      even_phase_start_point,
      boundary,
      flap.grid_size,
      flap.creases,
  )
  return desired_phase


def _get_flap_phase_from_heuristics(
    *,
    theta: float,
    absolute_direction: heuristics.AbsFlapDirection,
    rect: shaping_elements.ShapingFlap,
    elements: list[shaping_elements.ShapingElement],
    central_crease_removed: bool,
    flap_info: dict[str, heuristics.Node],
) -> bool:
  """Converts a zigzag direction to a phase.

  The phase choice is made by identifying the phase
  which yields the desired anchor_at_paper_border_rect_corner value.

  Args:
    theta: The angle of the zigzag.
    absolute_direction: The absolute direction of the zigzag.
    rect: The rectangle corresponding to the flap we are shaping.
    elements: The list of all elements in the box pleating model.
    central_crease_removed: Whether the central crease has been removed.
    flap_info: The dictionary of all flaps.

  Returns:
    The phase of the zigzag.
  """
  global_phase = convert_flap_angle_to_global_phase(
      theta, absolute_direction, rect
  )

  element_dict = {element.name: element for element in elements}
  local_phase = _flip_phase_by_parent_shaping_history(
      global_phase, rect, element_dict, flap_info
  )

  # If central fold is removed and rectangle is on the right half of paper,
  # flip the prediction.
  if central_crease_removed and rect.is_right_of_central_crease:
    local_phase = not local_phase

  return local_phase


def get_fold_order(
    is_even_phase: bool,
    grid_size: int,
    rect: shaping_elements.ShapingFlap,
    hinge_segments: list[crease_utils.Crease],
) -> tuple[list[crease_utils.Fold], bool]:
  """Returns the fold order of the zigzag.

  We get all the key creases assignment and hinge segments that need check.
  Then we try all fold orders to see potential zigzag crease and hinge
  segment assignment conflict. If no conflict found, we return that fold order.
  If both fold orders have conflicts, we return the default fold order and mark
  the fold order as invalid.

  Args:
    is_even_phase: Whether the phase of the zigzag is even.
    grid_size: The size of the box pleating grid.
    rect: The rectangle corresponding to the flap we are shaping.
    hinge_segments: The hinge segments of the flap.

  Returns:
    The fold order of the zigzag and a boolean indicating whether the fold order
    is valid.
  """

  potential_illegal_hinge_segment_coordinates_and_is_same_assignment = (
      get_potential_illegal_hinge_segment_coordinates_and_is_same_assignment(
          is_even_phase, grid_size, rect
      )
  )

  fold_order_trial = {
      'MV': MOUNTAIN_TO_VALLEY,
      'VM': VALLEY_TO_MOUNTAIN,
  }  # MV, VM
  fold_order_trial_success = {'MV': True, 'VM': True}

  # Try all fold orders to see potential zigzag crease and hinge segment
  # assignment conflict.
  for key, fold_order in fold_order_trial.items():
    for (
        data
    ) in potential_illegal_hinge_segment_coordinates_and_is_same_assignment:
      point_coordinates, is_same_assignment = data
      zigzag_assignment = fold_order[0] if is_same_assignment else fold_order[1]

      # Create potential illegal crease from coordinates and assignment.
      potential_illegal_crease = crease_utils.Crease(
          point_coordinates[0], point_coordinates[1], fold=zigzag_assignment
      )
      if any(
          potential_illegal_crease.eq_endpoints_and_fold(hinge_segment)
          for hinge_segment in hinge_segments
      ):
        fold_order_trial_success[key] = False
        break

  valid = True
  if fold_order_trial_success['MV']:
    fold_order = MOUNTAIN_TO_VALLEY
  elif fold_order_trial_success['VM']:
    fold_order = VALLEY_TO_MOUNTAIN
  else:
    logging.warning(
        'No valid fold order found for current hinge segments (%s), rectangle'
        ' (%s) and phase (%s). Use default fold order instead.',
        hinge_segments,
        rect,
        is_even_phase,
    )
    valid = False
    fold_order = MOUNTAIN_TO_VALLEY
  return fold_order, valid


def _get_anchor_from_abs_orientation_and_angle(
    abs_orientation: heuristics.AbsFlapDirection,
    angle: float,
) -> bool:
  """Decide whether or not to anchor zigzag at paper border."""
  is_down = abs_orientation == heuristics.AbsFlapDirection.DOWN
  is_counter_clockwise = angle > 0
  # anchor_at_paper_border_rect_corner tracks,
  # when proceeding clockwise from top edge of rect,
  # whether the first rectangle corner on the paper border we encounter
  # is a a zigzag anchor point (zigzag touches this point)
  # NOTE: this is equivalent to specifying the "Zigzag Direction" of a fold
  # (see docstring at top of file for more details)
  match (is_down, is_counter_clockwise):
    case (True, True):
      anchor_at_paper_border_rect_corner = False
    case (True, False):
      anchor_at_paper_border_rect_corner = True
    case (False, True):
      anchor_at_paper_border_rect_corner = True
    case _:
      anchor_at_paper_border_rect_corner = False
  return anchor_at_paper_border_rect_corner


def _get_phase_of_central_flap(
    angle: float,
    rect: shaping_elements.ShapingFlap,
    elements: list[shaping_elements.ShapingElement],
    central_crease_removed: bool,
    flap_info: dict[str, heuristics.Node],
) -> bool:
  """Gets the phase of a central flap.

  By default the phase is even for counter-clockwise rotations
  (angle > 0) and odd for clockwise rotations (angle < 0).
  We then further adjust the phase depending on the element's
  ancestors' shaping history as well as whether the central crease
  has been removed.

  Args:
    angle: The angle of the central flap.
    rect: The rectangle corresponding to the flap we are shaping.
    elements: All elements in the flap.
    central_crease_removed: Whether the central crease has been removed.
    flap_info: The dictionary of all flaps.

  Returns:
    The phase of the central flap.
  """
  if crease_utils.approx_lte(0, angle):
    is_even_phase = True
  else:
    is_even_phase = False

  element_dict = {element.name: element for element in elements}
  is_even_phase = _flip_phase_by_parent_shaping_history(
      is_even_phase, rect, element_dict, flap_info
  )

  # if central fold is removed and rectangle
  # is on the right half of paper,
  # flip the prediction
  if central_crease_removed and rect.is_right_of_central_crease:
    is_even_phase = not is_even_phase

  return is_even_phase


def _get_default_flap_zigzag_params() -> shaping_types.ZigzagParams:
  """Returns default flap zigzag params."""
  return shaping_types.ZigzagParams(
      target_flap_angle=0,
      even_phase=True,
      fold_order=MOUNTAIN_TO_VALLEY,
  )


def _get_flap_zigzag_params_from_heuristics(
    grid_size: int,
    rect: shaping_elements.ShapingFlap,
    flap_info: dict[str, heuristics.Node],
    hinge_segments: list[crease_utils.Crease],
    central_crease_removed: bool,
    elements: list[shaping_elements.ShapingElement],
    depth: float,
    resolve_hinge_ff_with_shifts: bool,
) -> zigzag_shaping.ZigzagShapingParamsForElement:
  """Converts a tree angle to a flap angle and phase.

  NOTE: This implementation assumes the flap is being viewed from a
  fixed camera angle (0, 0, 1) in which the paper underside
  is visible on the right.

  NOTE: implement for
  1. flaps with valley folds near boundary
  2. central flaps
  3. flaps spanning central crease when central crease is removed

  Args:
    grid_size: The size of the box pleating grid.
    rect: The rectangle corresponding to the flap we are shaping.
    flap_info: The dictionary of all flaps.
    hinge_segments: All hinge segments from the packing step.
    central_crease_removed: Whether the central fold has been removed.
    elements: The list of all elements in the box pleating model.
    depth: The depth into the flap of the zigzag.
    resolve_hinge_ff_with_shifts: Whether to resolve hinge ff violations
      by inscribing zigzagas by a small depth.

  Returns:
    The parameters for the zigzag shaping.
  """
  rect_name = rect.name
  theta = flap_info[rect_name].shape_angle
  absolute_direction = flap_info[rect_name].absolute_direction

  # Disallow shaping if flap is unit length
  if rect.grid_unit_radius <= 1.0 + point_utils.DEFAULT_TOLERANCE:
    return get_default_zigzag_params_for_element(rect)

  # Return trivial shaping params if angle is (almost) 0.
  if not _is_valid_and_nontrivial_angle(theta):
    return zigzag_shaping.ZigzagShapingParamsForElement(
        element_name=rect.name,
        zigzag_params=_get_default_flap_zigzag_params(),
    )
  assert theta is not None, 'Flap angle is None.'

  # Clip angle to min angle
  theta = clip_angle_by_min_angle(theta)

  # Round angles near 90 degrees to 90 degrees
  if abs(abs(theta) - math.pi / 2) < NINETY_DEGREE_THRESHOLD:
    theta = shaping_utils.sign(theta) * math.pi / 2

  # Ensure zigzag depth is not larger than max_shaping_depth if set
  if rect.max_shaping_depth is not None:
    theta = _clip_angle_by_depth(theta, rect.max_shaping_depth - depth)

  # NOTE: add support for flaps spanning
  # central fold when central fold is removed
  # for now just do not shape
  if central_crease_removed and rect.contains_central_crease:
    return get_default_zigzag_params_for_element(rect)

  # Shrink flap by shaping depth for phase heuristic calculation
  if depth > 0:
    rect = rect.shrink(depth)

  # orient central flaps by using x-values on the unfolded crease pattern
  if rect.is_central_flap:
    phase = _get_phase_of_central_flap(
        theta,
        rect,
        elements,
        central_crease_removed,
        flap_info,
    )
  # flap is on the border, we can use paper border to orient
  else:
    phase = _get_flap_phase_from_heuristics(
        theta=theta,
        absolute_direction=absolute_direction,
        rect=rect,
        elements=elements,
        central_crease_removed=central_crease_removed,
        flap_info=flap_info,
    )

  fold_order, ff_violation = get_fold_order(
      is_even_phase=phase,
      grid_size=grid_size,
      rect=rect,
      hinge_segments=hinge_segments,
  )

  if resolve_hinge_ff_with_shifts and ff_violation:
    logging.debug(
        'Hinge ff violation for %s, adding depth %f',
        rect.name,
        HINGE_FF_SHIFT_DEPTH,
    )
    depth += HINGE_FF_SHIFT_DEPTH
    if rect.max_shaping_depth is not None:
      theta = _clip_angle_by_depth(theta, rect.max_shaping_depth - depth)

  return zigzag_shaping.ZigzagShapingParamsForElement(
      element_name=rect.name,
      zigzag_params=shaping_types.ZigzagParams(
          target_flap_angle=abs(theta),
          even_phase=phase,
          fold_order=fold_order,
          depth=depth,
      ),
  )


# -------- River utils --------


def convert_river_angle_to_global_phase(
    theta: float,
    absolute_direction: heuristics.AbsFlapDirection,
    outer_bank_index: shaping_types.BankIndex,
    river: shaping_elements.ShapingRiver,
) -> bool:
  """Converts a zigzag direction to a phase."""
  anchor_at_paper_border = (
      _get_anchor_from_abs_orientation_and_angle(
          absolute_direction, theta
      )
  )
  # NOTE: assumes we are heuristically shaping the river's outer bank
  outer_bank_boundary = river.get_boundary_from_index(outer_bank_index)
  start_point = outer_bank_boundary.creases[0].p1_init
  phase = _get_phase_from_anchor_at_paper_border_rect_corner(
      anchor_at_paper_border,
      start_point,
      outer_bank_boundary,
      river.grid_size,
      river.creases,
  )
  return phase


def _get_river_phase_from_heuristics(
    river: shaping_elements.ShapingRiver,
    outer_bank_index: shaping_types.BankIndex,
    central_crease_removed: bool,
    elements: list[shaping_elements.ShapingElement],
    flap_info: dict[str, heuristics.Node],
) -> bool:
  """Converts a zigzag direction to a phase.

  If the zigzag is anchored at the border, the phase should be even.
  If the zigzag is not anchored at the border, the phase should be odd.

  Args:
    river: The river corresponding to the river node.
    outer_bank_index: The index of the outer bank of the river.
    central_crease_removed: Whether the central crease has been removed.
    elements: The list of all elements in the box pleating model.
    flap_info: The dictionary of all flaps.

  Returns:
    The phase of the zigzag.
  """
  river_node = flap_info[river.name]
  theta = river_node.shape_angle

  even_phase = convert_river_angle_to_global_phase(
      theta,
      river_node.absolute_direction,
      outer_bank_index,
      river,
  )

  element_dict = {element.name: element for element in elements}
  even_phase = _flip_phase_by_parent_shaping_history(
      even_phase, river, element_dict, flap_info
  )

  if central_crease_removed and river.is_right_of_central_crease:
    even_phase = not even_phase
  return even_phase


def _get_default_river_zigzag_params() -> shaping_types.RiverZigzagParams:
  """Returns default river zigzag params."""
  return shaping_types.RiverZigzagParams(
      target_flap_angle=0,
      even_phase=True,
      fold_order=MOUNTAIN_TO_VALLEY,
      bank_index='FIRST',
  )


def _get_river_zigzag_params_from_heuristics(
    river: shaping_elements.ShapingRiver,
    outer_bank_index: shaping_types.BankIndex,
    flap_info: dict[str, heuristics.Node],
    elements: list[shaping_elements.ShapingElement],
    central_crease_removed: bool,
    depth: float,
) -> zigzag_shaping.ZigzagShapingParamsForElement:
  """Converts a tree angle to a river angle and phase and bank.

  NOTE: This implementation assumes the river is being viewed from a
  fixed camera angle (0, 0, 1) in which the paper underside
  is visible on the right.

  Args:
    river: The river corresponding to the river node.
    outer_bank_index: The index of the outer bank of the river.
    flap_info: The dictionary of all flaps.
    elements: The list of all elements in the box pleating model.
    central_crease_removed: Whether the central crease has been removed.
    depth: The depth into the river of the zigzag.

  Returns:
    A tuple of the river angle and phase.
  """
  river_node = flap_info[river.name]
  theta = river_node.shape_angle

  # Return trivial shaping params if angle is (almost) 0.
  if not _is_valid_and_nontrivial_angle(theta):
    return zigzag_shaping.ZigzagShapingParamsForElement(
        element_name=river.name,
        zigzag_params=_get_default_river_zigzag_params(),
    )
  assert theta is not None, 'River angle is None.'

  # Clip angle to min angle
  theta = clip_angle_by_min_angle(theta)

  # Clip theta so zigzags lie within river width
  theta = _clip_angle_by_depth(theta, river.max_shaping_depth - depth)

  even_phase = _get_river_phase_from_heuristics(
      river,
      outer_bank_index,
      central_crease_removed,
      elements,
      flap_info,
  )

  return zigzag_shaping.ZigzagShapingParamsForElement(
      element_name=river.name,
      zigzag_params=shaping_types.RiverZigzagParams(
          target_flap_angle=abs(theta),
          fold_order=MOUNTAIN_TO_VALLEY,
          even_phase=even_phase,
          bank_index=outer_bank_index,
          depth=depth,
      ),
  )


# -------- Element utils --------


def get_default_zigzag_params_for_element(
    element: shaping_elements.ShapingElement,
) -> zigzag_shaping.ZigzagShapingParamsForElement:
  """Returns default zigzag params."""
  if isinstance(element, shaping_elements.ShapingFlap):
    zigzag_params = _get_default_flap_zigzag_params()
  else:
    zigzag_params = _get_default_river_zigzag_params()
  return zigzag_shaping.ZigzagShapingParamsForElement(
      element_name=element.name,
      zigzag_params=zigzag_params,
  )


def get_element_zigzag_params_from_heuristics(
    grid_size: int,
    element: shaping_elements.ShapingElement,
    outer_bank_indices: dict[str, shaping_types.BankIndex],
    flap_info: dict[str, heuristics.Node],
    hinge_segments: list[crease_utils.Crease],
    central_crease_removed: bool,
    elements: list[shaping_elements.ShapingElement],
    depth: int,
    resolve_hinge_ff_with_shifts: bool,
) -> zigzag_shaping.ZigzagShapingParamsForElement:
  """Calculates the zigzag parameters for an element."""
  if isinstance(element, shaping_elements.ShapingFlap):
    zigzag_params = _get_flap_zigzag_params_from_heuristics(
        grid_size,
        element,
        flap_info,
        hinge_segments,
        central_crease_removed,
        elements,
        depth,
        resolve_hinge_ff_with_shifts,
    )
  else:
    outer_bank_index = outer_bank_indices[element.name]
    zigzag_params = _get_river_zigzag_params_from_heuristics(
        element,
        outer_bank_index,
        flap_info,
        elements,
        central_crease_removed,
        depth,
    )
  return zigzag_params
