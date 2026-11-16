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

"""Methods implementing shaping operations in box pleating.

The following shaping operations will be supported:
- zigzag creases (Done).
- grid coarsening/refinenement (todo).
- fold angle adjustment (todo).

Zigzag shaping implementation pseudocode:
  1. Pre-compute rectangle corners, edges, edge vectors, and depth vectors.
    - corners: four corners of rectangle in clockwise order starting at top-left
    - edges: creases between corners in clockwise order starting with top crease
    - edge vectors: grid unit vectors pointing along edge in clockwise direction
    - depth vectors: grid unit vectors pointing into rectangle from edge
  2. Compute anchor points: all vertices on
  the rectangle border which must be incident to a zigzag crease.
    - Starting at midpoint of top edge, walk clockwise along rectangle border
    - Add corners as anchor points if they are
      - not a corner OR
      - A corner between a paper border crease and an interior crease (in that
      clockwise order)
  3. Draw zigzag creases using anchor points
    - While some anchor point has been unvisited:
      - Pop anchor point from unvisited anchor point
      - Draw clockwise zigzag crease from anchor point
      - If zigzag hits a ridge, reflect over ridge

"""

import dataclasses
import logging
import math
from typing import Sequence

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import point as point_utils
from . import narrow_flap_shaping
from . import shaping_elements
from . import shaping_utils
from . import types as shaping_types
from . import vector
from .. import timeout as timeout_lib

DEFAULT_TOLERANCE = 1e-5


# ---------- On-paper implementation ----------


def get_closest_ridge_intersection(
    start: point_utils.Point,
    end: point_utils.Point,
    ridges: Sequence[crease_utils.Crease],
    require_on_paper: bool = True,
    angle: float | None = None,
) -> tuple[point_utils.Point | None, int | None]:
  """Get the closest intersection point of a crease with ridges.

  Args:
    start: The start point of the crease.
    end: The end point of the crease.
    ridges: The ridges.
    require_on_paper: Whether to require the intersection to be on the paper.
    angle: The angle of the simple fold.

  Returns:
    end: The new end point of the crease.
    ridge_ind: The index of the ridge that the crease intersects.
  """
  # init crease from start, end. Give dummy fold for now.
  crease = crease_utils.Crease(start, end, fold=crease_utils.Fold.MOUNTAIN)
  # check for intersections with ridges
  intersections = []
  for ind, ridge in enumerate(ridges):
    intersection = ridge.vertex_with_crease(
        crease, include_external=not require_on_paper
    )
    # continue if no intersection
    if intersection is None:
      continue
    # continue if intersection is at crease start
    elif start.isclose(intersection):
      continue
    # NOTE: possibly should remove this condition
    # continue if intersection at ridge tip
    elif (
        intersection.isclose(ridge.p1) or intersection.isclose(ridge.p2)
    ) and (angle is None or angle < math.pi / 2):
      continue
    # continue if intersection at zigzag end and zigzag not orthogonal to ridge
    if intersection.isclose(end) and (
        not abs(
            abs(
                vector.Vector(
                    x=crease.normed_vector[0], y=crease.normed_vector[1]
                ).slope
            )
            - 1
        )
        < shaping_utils.EPS
    ):
      continue
    intersections.append((ind, intersection))
  # shrink crease to closest intersection point to start
  if intersections:
    dists = [
        (vector.Vector.from_point(i) - vector.Vector.from_point(start)).norm
        for _, i in intersections
    ]
    ridge_ind, end = intersections[shaping_utils.argmin(dists)]
  else:
    ridge_ind = None
    end = None
  return end, ridge_ind


def draw_zigzag_from_anchor(
    start: point_utils.Point,
    on_edge: bool,
    edge_index: int,
    edge_vectors: Sequence[vector.Vector],
    depth_vectors: Sequence[vector.Vector],
    zigzag_depth: float,
) -> point_utils.Point:
  """Find zigzag end for given start point."""
  # draw away from boundary
  if on_edge:
    end = (
        edge_vectors[edge_index] + zigzag_depth * depth_vectors[edge_index]
    ).point_add(start)
  # return to boundary
  else:
    end = (
        edge_vectors[edge_index] - zigzag_depth * depth_vectors[edge_index]
    ).point_add(start)
  return end


def _check_visited(
    p: point_utils.Point, visited_points: Sequence[point_utils.Point]
) -> bool:
  """Check if a point has been visited."""
  return any(p.isclose(v) for v in visited_points)


def _snap_to_grid(
    point: point_utils.Point,
    grid_size: int,
) -> point_utils.Point:
  """Snap a point to the grid lines."""
  try:
    point = shaping_utils.snap_point_to_grid(point, grid_size, snap_y=False)
  except ValueError:
    pass
  try:
    point = shaping_utils.snap_point_to_grid(point, grid_size, snap_x=False)
  except ValueError:
    pass
  return point


# NOTE: add an argument to check for flat foldability after
# applying zigzag shaping, and if it isn't flat foldable, call a function to
# solve the CP assignments to make it flat foldable, if possible.
def shape_boundary_with_zigzags(
    anchor_points: Sequence[tuple[point_utils.Point, int, int]],
    boundary: shaping_elements.ShapingBoundary,
    grid_size: int,
    target_flap_angle: float | None,
    target_zigzag_depth: float | None,
    fold_order: Sequence[crease_utils.Fold],
    even_phase: bool,
    fold_percent: float,
    ridges: Sequence[crease_utils.Crease],
    snap_to_grid: bool,
    require_on_paper: bool = True,
    fold_step: int = 0,
) -> list[crease_utils.Crease]:
  """Add zigzag creases along given rectangle.

  The implementation proceeds by first computing anchor points: all points on
  the rectangle border of the rectangle which must
  be incident to a zigzag crease. It then proceeds by drawing zigzags with
  target depth from the top-center anchor point. If a zigzag hits a ridge, it
  is reflected over the ridge. If a zigzag returns to the initial anchor point,
  the process restarts at the next unvisited anchor point. The process is
  repeated until all anchor points
  have been visited by a zigzag crease.

  Args:
    anchor_points: The anchor points which contain the point itself, the edge
      index of the edge it is on, and the grid point index it is on.
    boundary: The boundary to shape with zigzags.
    grid_size: The size of the box-pleating grid.
    target_flap_angle: The target flap angle. If None, target_zigzag_depth must
      be specified.
    target_zigzag_depth: The target zigzag depth. If None, target_flap_angle
      must be specified.
    fold_order: a sequence of folds detailing the fold order; for example a fold
      order of [MOUNTAIN, VALLEY] will have the zigzag crease starting with a
      mountain fold and then alternating valley and mountain orientations,
      whereas a fold order of [VALLEY] will have zigzag creases be all valley
      folds.
    even_phase: Whether the zigzag phase is even or odd. Zigzag phase determines
      the shift of zigzags.
    fold_percent: The percentage to fold the zigzag crease.
    ridges: The box pleating ridges.
    snap_to_grid: Whether to snap the zigzag creases to the grid.
    require_on_paper: Whether to require the zigzag creases to be on the paper.
    fold_step: The folding step for the creases.

  Returns:
    zigzags: The zigzag creases.
  """
  if target_zigzag_depth is None:
    target_zigzag_depth = shaping_utils.zigzag_depth_from_fold_angle(
        target_flap_angle
    )
  # now begin zigzag creation at start anchor point by walking along zigzag path
  # until we return to the start anchor point
  # if we return to the start anchor point, we move on to the next unvisited
  # anchor point
  zigzags = []
  visited_points = []
  # ----- Outer loop: while not every anchor is visited by a zigzag
  while anchor_points:
    timeout_lib.maybe_timeout_cooperatively()
    # draw zigzags until we return to start
    anchor, edge_index, grid_index = anchor_points[0]
    start = anchor
    # init vars to:
    # 1. track whether cur point is on a rectangle edge
    # 2. track current ridge being intersected
    # 3. track previous zigzag to check for backtracking
    # 4. track current fold (M/V)
    fold_index = grid_index % len(fold_order)
    on_edge, cur_ridge, prev_zigzag, cur_fold = (
        True,
        None,
        None,
        fold_order[fold_index],
    )
    # ----- Inner loop: draw zigzags until we return to start
    # or paper border (if require_on_paper is True)
    while prev_zigzag is None or (
        not _check_visited(start, visited_points)
        and not (start.is_border and require_on_paper)
    ):
      timeout_lib.maybe_timeout_cooperatively()
      logging.debug('start: %s', start)
      logging.debug('on_edge: %s', on_edge)
      logging.debug('edge_index: %s', edge_index)
      # ----- Try drawing zigzag
      # if starting on ridge, reflect prev_zigzag over ridge
      if cur_ridge is not None:
        reflection_vector = shaping_utils.reflect_crease_over_ridge(
            prev_zigzag,
            cur_ridge,
            require_on_paper=require_on_paper,
        )
        try:
          end = reflection_vector.point_add(start)
        except ValueError:
          # if reflection goes off paper,
          # just drop for now and try next anchor point
          break
      # otherwise draw zigzags as usual
      else:
        end = draw_zigzag_from_anchor(
            start,
            on_edge,
            edge_index,
            boundary.edge_unit_vectors,
            boundary.depth_unit_vectors,
            target_zigzag_depth,
        )

      # ----- Check for ridge intersections
      intersection_point, ridge_ind = get_closest_ridge_intersection(
          start,
          end,
          ridges,
          require_on_paper=require_on_paper,
          angle=target_flap_angle,
      )
      if intersection_point is not None:
        end = intersection_point
        cur_ridge = ridges[ridge_ind]
        logging.debug(
            'Ridge intersection point: %s',
            intersection_point,
        )
      else:
        cur_ridge = None
      # check for ridge overlaps
      # NOTE: this can only happen with 90o turns, depth = 1
      if shaping_utils.get_crease_ridges_containment(start, end, ridges):
        # if true, increment edge_index, flip fold, and try again
        edge_index = (edge_index + 1) % len(boundary.creases)
      else:
        # try snapping points to grid if requested
        if snap_to_grid:
          end = _snap_to_grid(end, grid_size)
        zigzag = crease_utils.Crease(
            start,
            end,
            fold=cur_fold,
            fold_percent=fold_percent,
            fold_step=fold_step,
        )
        # check we are not going backwards
        assert prev_zigzag is None or not prev_zigzag.isclose(
            zigzag, atol=DEFAULT_TOLERANCE
        ), 'Backtracking detected.'

        # ----- Update state
        logging.debug(
            'Drawing zigzag with start, end: %s, %s',
            zigzag.p1_init,
            zigzag.p2_init,
        )
        zigzags.append(zigzag)
        # mark start as visited if start on edge
        if on_edge:
          visited_points.append(start)
          anchor_points = [
              (p, ei, gi) for p, ei, gi in anchor_points if not p.isclose(start)
          ]
        prev_zigzag = zigzag
        start = end
        on_edge = not on_edge
        # update new edge index
        # If on edge and not reflecting over ridge,
        # then we can simply look-up the new edge index.
        if on_edge and cur_ridge is None:
          edge_index = boundary.get_edge_index(end)
          assert edge_index is not None, (
              f'Point {end} is not on boundary {boundary}.'
          )
        # If we are reflecting over a ridge we may not really be on the edge,
        # but we can just increment the edge index since we are traversing
        # clockwise along the boundary.
        elif on_edge and cur_ridge is not None:
          edge_index = (edge_index + 1) % len(boundary.creases)
        else:
          pass
      fold_index = (fold_index + 1) % len(fold_order)
      cur_fold = fold_order[fold_index]

    # ----- Add single counterclockwise edge if
    # boundary is not closed and phase is odd
    if not boundary.closed and not even_phase:
      end = draw_zigzag_from_anchor(
          anchor,
          True,
          0,
          [
              -1 * edge_vector for edge_vector in boundary.edge_unit_vectors
          ],  # reverse direction
          boundary.depth_unit_vectors,
          target_zigzag_depth,
      )
      zigzag = crease_utils.Crease(
          anchor,
          end,
          fold=fold_order[-1],
          fold_percent=fold_percent,
          fold_step=fold_step,
      )
      zigzags.append(zigzag)
  return zigzags


# ---------- Flap implementation ----------


def compute_flap_anchor_points(
    rect: shaping_elements.ShapingFlap | shaping_utils.Rectangle,
    grid_size: int,
    even_phase: bool,
) -> list[tuple[point_utils.Point, int, int]]:
  """Compute anchor points for rectangle.

  Anchor points are points on the rectangle boundary that must have a zigzag
  line incident to them in the clockwise direction. Proceeds by starting a walk
  along the rectangle boundary at the
  middle of the top edge. Every other point along the walk is an anchor point
  with the following exceptions:
  - points on the paper border
  - rectangle corners are excluded unless the corner
  is incident to a paper border and an interior edge (in clockwise order)

  Args:
    rect: The rectangle defining the flap.
    grid_size: The size of the box-pleating grid.
    even_phase: Determines whether zigzag is in even phase or odd phase.

  Returns:
    anchor_points: The anchor points which contain the point itself, the edge
    index of the edge it is on, and the grid point index it is on.
  """
  anchor_points = []
  # shift starting point depending on grid length and phase
  midpoint = point_utils.midpoint(rect.corners[0], rect.corners[1])
  start = shaping_utils.compute_anchor_points_start(
      midpoint, grid_size, even_phase, rect.get_unit_edge_vectors(grid_size)[0]
  )
  # walk along gridpoints on rectangle border,
  # adding anchor points every other point (with some exceptions)
  edge_index = shaping_utils.get_edge_index(start, rect.corners, rect.edges)
  cur_p = start
  # grid_point_index keeps track of the current grid point index. The anchor
  # points will lie on some of these grid points. The fold type of the zigzag
  # drawn from an anchor point at grid_point_index is equal to
  # fold_order[grid_point_index % len(fold_order)], where fold_order is the
  # sequence of folds detailing the fold order.
  grid_point_index = 0
  w = shaping_utils.crease_grid_length(rect.edges[0], grid_size)
  h = shaping_utils.crease_grid_length(rect.edges[1], grid_size)
  # Upper bound on number of anchor points.
  # Divide by 2 since we traverse two grid points per iteration.
  num_anchor_points_upper_bound = (2 * (int(w) + 1) + 2 * (int(h) + 1)) // 2 + 1
  boundary = shaping_elements.get_flap_boundary(rect, grid_size)
  for _ in range(num_anchor_points_upper_bound):
    timeout_lib.maybe_timeout_cooperatively()
    if any(cur_p.isclose(p[0]) for p in anchor_points):
      break
    is_rect_corner = cur_p.isclose(rect.corners[edge_index])
    # only disallow corners as anchor points
    if not is_rect_corner:
      anchor_points.append((cur_p, edge_index, grid_point_index))
    # take a double step to next candidate anchor point
    cur_p, edge_index = boundary.walk(
        cur_p, edge_index, grid_size
    )
    cur_p, edge_index = boundary.walk(
        cur_p, edge_index, grid_size
    )
    grid_point_index += 2

  logging.debug('anchor_points: %s', anchor_points)
  return anchor_points


def shape_flap_with_zigzags(
    *,
    rect: shaping_utils.Rectangle | shaping_elements.ShapingFlap,
    grid_size: int,
    target_flap_angle: float | None,
    target_zigzag_depth: float | None,
    fold_order: Sequence[crease_utils.Fold],
    even_phase: bool,
    fold_percent: float,
    snap_to_grid: bool,
    fold_step: int = 0,
) -> list[crease_utils.Crease]:
  """Shapes a flap with zigzag creases.

  Proceeds in the following steps:
  1. Draws zigzags using shape_rectangle_with_zigzags on full (off-paper)
  rectangle
  2. Restricts creases to paper using restrict_crease_to_paper

  Args:
    rect: The rectangle defining the flap.
    grid_size: The size of the box-pleating grid.
    target_flap_angle: The target flap angle. If None, target_zigzag_depth must
      be specified.
    target_zigzag_depth: The target zigzag depth. If None, target_flap_angle
      must be specified.
    fold_order: Determines whether a zigzag is in MV or VM orientation.
    even_phase: Whether the zigzag phase is even or odd. Zigzag phase determines
      the shift of zigzags.
    fold_percent: The percentage to fold the zigzag crease.
    snap_to_grid: Whether to snap the zigzag creases to the grid.
    fold_step: The folding step to use for the creases.

  Returns:
    The zigzag creases on the paper.
  """
  # ----- Step 0: ensure input is valid
  # ensure flap centers are present
  if rect.circle_centers is None or not rect.circle_centers:
    raise ValueError(
        f'Rectangle {rect} must have circle centers to apply shaping!'
    )
  # ensure flap centered around circle_centers
  if not rect.check_center_alignment():
    raise ValueError(
        f'Rectangle {rect} must be centered around its circle centers!'
    )

  # ----- Step 1: compute anchor points
  anchor_points = compute_flap_anchor_points(
      rect,
      grid_size,
      even_phase,
  )
  # if no anchor points, return empty list
  if not anchor_points:
    return []

  # ----- Step 2: draw zigzags on full rectangle
  shaping_boundary = shaping_elements.get_flap_boundary(rect, grid_size)
  zigzags = shape_boundary_with_zigzags(
      anchor_points,
      shaping_boundary,
      grid_size,
      target_flap_angle,
      target_zigzag_depth,
      fold_order,
      even_phase,
      fold_percent,
      rect.get_ridges(),
      snap_to_grid=snap_to_grid,
      require_on_paper=False,
      fold_step=fold_step,
  )

  # ----- Step 3: restrict zigzag creases to paper and return
  restricted_zigzags = shaping_utils.restrict_creases_to_paper(zigzags)
  return restricted_zigzags


# ---------- River implementation ----------


def compute_river_bank_anchor_points(
    boundary: shaping_elements.ShapingBoundary,
    grid_size: int,
    even_phase: bool,
) -> Sequence[tuple[point_utils.Point, int, int]]:
  """Compute anchor points for river bank boundary.

  Anchor points are points on the river bank from which
  zigzag lines will extend in the direction given by
  the boundary's edge vectors. Computed by walking along the river bank,
  adding every other point as an anchor point if the point is not a corner.

  Args:
    boundary: The river bank boundary.
    grid_size: The size of the box-pleating grid.
    even_phase: Determines whether zigzag is in even phase or odd phase.

  Returns:
    anchor_points: The anchor points which contain the point itself, the edge
    index of the edge it is on, and the grid point index it is on.
  """
  anchor_points = []
  # choose the first anchor point
  start = boundary.creases[0].p1_init
  # bump start forward if odd phase
  if not even_phase:
    start = boundary.edge_unit_vectors[0].point_add(start)
  edge_index = boundary.get_edge_index(start)
  if edge_index is None:
    raise ValueError(f'Start point {start} is not on element boundary.')
  # grid_point_index keeps track of the current grid point index. The anchor
  # points will lie on some of these grid points. The fold type of the zigzag
  # drawn from an anchor point at grid_point_index is equal to
  # fold_order[grid_point_index % len(fold_order)], where fold_order is the
  # sequence of folds detailing the fold order.
  grid_point_index = 0
  cur_p = start
  end = boundary.creases[-1].p2_init
  # Upper bound number of anchor points by flap edge lengths.
  num_anchor_points_upper_bound = sum(
      int(shaping_utils.get_grid_unit_length(edge.length, grid_size)) + 1
      for edge in boundary.creases
  )
  # walk along gridpoints on boundary,
  # adding anchor points every other point unless they are on corners
  # stopping at the end of the boundary
  for _ in range(num_anchor_points_upper_bound):
    timeout_lib.maybe_timeout_cooperatively()
    if anchor_points and cur_p.isclose(end):
      break
    is_corner = cur_p.isclose(boundary.tips[edge_index])
    # add points if they are not on boundary corners
    # and the end
    if grid_point_index % 2 == 0 and not is_corner:
      anchor_points.append((cur_p, edge_index, grid_point_index))
    # take a step along the boundary
    cur_p, edge_index = boundary.walk(cur_p, edge_index, grid_size)
    grid_point_index += 1
  # additionally add first point to anchor set if
  # the boundary is not closed
  if not boundary.closed:
    anchor_points.append((start, 0, 0))

  return anchor_points


def shape_river_with_zigzags(
    *,
    river: shaping_elements.ShapingRiver,
    grid_size: int,
    target_flap_angle: float | None,
    target_zigzag_depth: float | None,
    fold_order: Sequence[crease_utils.Fold],
    even_phase: bool,
    fold_percent: float,
    bank_index: shaping_types.BankIndex,
    snap_to_grid: bool,
    fold_step: int = 0,
) -> list[crease_utils.Crease]:
  """Shapes a river with zigzag creases.

  Args:
    river: The river to shape.
    grid_size: The size of the box-pleating grid.
    target_flap_angle: The target flap angle. If None, target_zigzag_depth must
      be specified.
    target_zigzag_depth: The target zigzag depth. If None, target_flap_angle
      must be specified.
    fold_order: Determines the sequence of fold assignments in zigzag creases.
    even_phase: Whether the zigzag phase is even or odd. Zigzag phase determines
      the shift of zigzags.
    fold_percent: The percentage to fold the zigzag crease.
    bank_index: The number of the river bank ('FIRST' or 'SECOND') to shape.
    snap_to_grid: Whether to snap the zigzag creases to the grid.
    fold_step: The folding step to use for the creases.

  Returns:
    The zigzag creases on the paper.
  """
  # ----- Step 0: ensure input is valid
  # raise error if river cannot be shaped with desired angle
  if target_zigzag_depth is None:
    target_zigzag_depth = shaping_utils.zigzag_depth_from_fold_angle(
        target_flap_angle
    )
  if target_zigzag_depth > river.width:
    raise ValueError(
        f'River {river} is too small to be shaped with angle'
        f' {target_flap_angle}, zigzags will be too big!'
    )

  # ----- Step 1: get river bank boundary and compute anchor points
  boundary = river.get_boundary_from_index(bank_index)
  anchor_points = compute_river_bank_anchor_points(
      boundary, grid_size, even_phase
  )
  # if no anchor points, return empty list
  if not anchor_points:
    return []

  # ----- Step 2: draw zigzags
  return shape_boundary_with_zigzags(
      anchor_points,
      boundary,
      grid_size,
      target_flap_angle,
      target_zigzag_depth,
      fold_order,
      even_phase,
      fold_percent,
      river.ridges,
      snap_to_grid=snap_to_grid,
      require_on_paper=True,
      fold_step=fold_step,
  )


# ---------- Wrappers ----------


@dataclasses.dataclass(frozen=True, kw_only=True)
class ZigzagShapingParamsForElement:
  """Shaping params for a single element (flap or river).

  Attributes:
    element_name: The name of the element being shaped.
    zigzag_params: The params determining the drawing of zigzags.
  """

  # NOTE: eventually allow for
  # a list of ZigzagParams with multiple depths
  element_name: str
  zigzag_params: shaping_types.ZigzagParams | shaping_types.RiverZigzagParams


def get_zigzag_shaping_creases(
    grid_size: int,
    rectangles: Sequence[
        shaping_utils.Rectangle | shaping_elements.ShapingFlap
    ],
    element_shaping_params: Sequence[ZigzagShapingParamsForElement],
    *,
    rivers: Sequence[shaping_elements.ShapingRiver] | None = None,
    snap_to_grid: bool = False,
    fold_step: int,
) -> list[crease_utils.Crease]:
  """Returns zigzag creases for shaping elements of a crease pattern.

  This function iterates through the shaping parameters, finds the matching
  rectangle for each element by name, and then calls `shape_flap_with_zigzags`
  to generate the zigzag creases with its specific parameters.

  Args:
    grid_size: The size of the crease pattern grid.
    rectangles: The rectangles available for shaping.
    element_shaping_params: The elements to shape and their specific params.
    rivers: The rivers available for shaping.
    snap_to_grid: Whether to snap the zigzag creases to the grid.
    fold_step: The current folding step.

  Returns:
    A list of `crease_utils.Crease` objects representing all the generated
    zigzag creases.

  Raises:
    ValueError: If a shaping parameter is provided for an element name that
      does not match any of the rectangles.
  """
  if rivers is None:
    rivers = []
  rectangle_dict = {rectangle.name: rectangle for rectangle in rectangles}
  river_dict = {river.name: river for river in rivers}
  element_dict = {**rectangle_dict, **river_dict}
  zigzag_creases = []
  for i, param in enumerate(element_shaping_params):
    timeout_lib.maybe_timeout_cooperatively()
    element = element_dict.get(param.element_name, None)
    if element is None:
      raise ValueError(
          f'Element rectangle {param.element_name} to be shaped not found'
          ' in CP.'
      )
    logging.debug(
        'Element %s to be shaped with params %s.',
        param.element_name,
        param.zigzag_params,
    )
    # skip flap shaping if neither target angle or zigzag depth specified
    # or is 0 (no zigzag)
    if not (
        param.zigzag_params.target_flap_angle
        or param.zigzag_params.target_zigzag_depth
    ):
      continue
    # sort into flap and river elements
    if isinstance(element, shaping_utils.Rectangle) or isinstance(
        element, shaping_elements.ShapingFlap
    ):
      if not isinstance(param.zigzag_params, shaping_types.ZigzagParams):
        raise ValueError(
            f'Element {element} is a rectangle but param {param} is not a'
            ' ZigzagParams!'
        )

      # Shrink rectangle if zigzags are inscribed
      if param.zigzag_params.depth > 0 and isinstance(
          element, shaping_elements.ShapingFlap
      ):
        logging.info(
            'Shrinking flap %s by %s',
            element,
            param.zigzag_params.depth,
        )
        element = element.shrink(param.zigzag_params.depth)
        logging.info('Shrunken flap %s', element)
      elif param.zigzag_params.depth > 0 and isinstance(
          element, shaping_utils.Rectangle
      ):
        raise ValueError(
            'Cannot inscribe zigzags into a (vanilla) rectangle.',
            'Soon rectangles will be deprecated. Instead using ShapingFlap.',
        )

      zigzag_creases.extend(
          shape_flap_with_zigzags(
              rect=element,
              grid_size=grid_size,
              target_flap_angle=param.zigzag_params.target_flap_angle,
              target_zigzag_depth=param.zigzag_params.target_zigzag_depth,
              fold_order=param.zigzag_params.fold_order,
              even_phase=param.zigzag_params.even_phase,
              fold_percent=param.zigzag_params.fold_percent,
              snap_to_grid=snap_to_grid,
              fold_step=fold_step + i,
          )
      )
    elif isinstance(element, shaping_elements.ShapingRiver):
      if not isinstance(param.zigzag_params, shaping_types.RiverZigzagParams):
        raise ValueError(
            f'Element {element} is a river but param {param} is not a'
            ' RiverZigzagParams!'
        )

      # NOTE: for now skip river shaping
      # if target depth is greater than river width
      if param.zigzag_params.target_zigzag_depth:
        target_depth = param.zigzag_params.target_zigzag_depth
      else:
        target_depth = shaping_utils.zigzag_depth_from_fold_angle(
            param.zigzag_params.target_flap_angle
        )
      if target_depth > element.width:
        logging.error(
            'River %s is too small (width %s) to be shaped with depth %s.'
            ' Skipping.',
            element.name,
            element.width,
            target_depth,
        )
        continue

      # Shrink river if zigzags are inscribed
      if param.zigzag_params.depth > 0:
        logging.info(
            'Shrinking river %s by %s',
            element,
            param.zigzag_params.depth,
        )
        element = element.shrink_along_bank(
            param.zigzag_params.bank_index, param.zigzag_params.depth
        )
        logging.info('Shrunken river %s', element)

      zigzag_creases.extend(
          shape_river_with_zigzags(
              river=element,
              grid_size=grid_size,
              target_flap_angle=param.zigzag_params.target_flap_angle,
              target_zigzag_depth=param.zigzag_params.target_zigzag_depth,
              fold_order=param.zigzag_params.fold_order,
              even_phase=param.zigzag_params.even_phase,
              fold_percent=param.zigzag_params.fold_percent,
              bank_index=param.zigzag_params.bank_index,
              snap_to_grid=snap_to_grid,
              fold_step=fold_step + i,
          )
      )
    else:
      raise ValueError(f'Element {element} is not a rectangle or river!')
  return zigzag_creases


def shape_cp_with_zigzags_pipeline_wrapper(
    cp: cp_utils.CreasePattern,
    grid_size: int,
    rectangles: Sequence[
        shaping_utils.Rectangle | shaping_elements.ShapingFlap
    ],
    element_shaping_params: Sequence[ZigzagShapingParamsForElement],
    *,
    rivers: Sequence[shaping_elements.ShapingRiver] | None = None,
    # NOTE: rename light mode.
    light_mode: bool = False,
    snap_to_grid: bool = False,
    narrow: bool = False,
    narrow_flap_names: Sequence[str] | None = None,
    raise_error_on_ff_violation: bool = False,
) -> cp_utils.CreasePattern:
  """Wrapper around zigzag shaping function for easier zigzag configuration.

  Args:
    cp: The crease pattern to shape.
    grid_size: The size of the box-pleating grid.
    rectangles: The rectangles available for shaping.
    element_shaping_params: The shaping parameters determining zigzag placement
      for each element (flap or river).
    rivers: The rivers available for shaping.
    light_mode: If light_mode is False, CreasePattern will be instantiated after
      splitting and merging creases that intersect with the zigzag lines. This
      is an expensive operation. If light_mode is True, this function will
      pre-emptively split the creases from the base `cp` that the zigzag lines
      intersect and reflect over. Then these pre-processed `cp` crease and
      zigzag lines will be passed to the CreasePattern and they will not be
      split and merged, which saves time.
    snap_to_grid: whether to snap the zigzag creases to the grid.
    narrow: whether to narrow the flaps after applying zigzag shaping.
    narrow_flap_names: The names of the flaps to narrow. If None, all flaps will
      be narrowed.
    raise_error_on_ff_violation: whether to raise an error if the shaped CP is
      not flat-foldable.

  Returns:
    A crease pattern shaped with zigzag creases.
  """
  # NOTE: Separate narrowing from zigzag shaping and just have the
  # narrowing function call get_zigzag_shaping_creases to get the zigzag
  # creases.
  zigzag_creases = get_zigzag_shaping_creases(
      grid_size,
      rectangles,
      element_shaping_params,
      rivers=rivers,
      snap_to_grid=snap_to_grid,
      fold_step=cp.num_fold_steps,
  )

  shaped_cp = cp.update_with_shaping_creases(
      zigzag_creases, split_shaping_by_shaping=not light_mode,
  )

  if raise_error_on_ff_violation and (not shaped_cp.local_flat_foldability()):
    raise ValueError('Zigzag shaped CP is not flat-foldable.')

  if narrow and zigzag_creases and rectangles:
    if narrow_flap_names is None:
      valid_rectangles = rectangles
    else:
      valid_rectangles = []
      for rect in rectangles:
        if any(flap_name in rect.name for flap_name in narrow_flap_names):
          valid_rectangles.append(rect)

    flap_only_params = [
        p
        for p in element_shaping_params
        if not isinstance(p.zigzag_params, shaping_types.RiverZigzagParams)
    ]
    if flap_only_params:
      flap_zigzag_creases = get_zigzag_shaping_creases(
          grid_size,
          rectangles,
          flap_only_params,
          snap_to_grid=snap_to_grid,
          fold_step=cp.num_fold_steps,
      )
    else:
      flap_zigzag_creases = []

    if flap_zigzag_creases:
      shaped_cp = narrow_flap_shaping.narrow(
          cp=shaped_cp,
          zigzag_creases=flap_zigzag_creases,
          rectangles=valid_rectangles,
          grid_size=grid_size,
          light_mode=light_mode,
          raise_error_on_ff_violation=raise_error_on_ff_violation,
      )

  return shaped_cp
