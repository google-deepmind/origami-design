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

"""Inscribed polygon methods implementing 3d shaping.

Implementation pseudocode:
- For each flap:
  - Get flap rectangle corners
  - For each inscribed rectangle in the flap:
    - Get the inscribed rectangle object by shrinking the flap rectangle
    around the flap's in
    - Find the starting point for a clockwise walk along
    the non-paper border boundary of the inscribed rectangle
    - Walk clockwise along the boundary of the inscribed rectangle,
    alternating M/V crease assignment
"""

import dataclasses
import itertools
import math
from typing import Sequence

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import point as point_utils
from . import shaping_elements
from . import shaping_utils
from . import types as shaping_types
from . import vector
from .. import timeout as timeout_lib

_CENTRAL_CREASE_X = 0.5  # x-coordinate of the central crease


# ------- Inscribed Rectangle shaping --------


@dataclasses.dataclass(kw_only=True, frozen=True)
class InscribedRectangleParams:
  """Params determining placement of an inscribed rectangle.

  Attributes:
    depth: How deep (in grid units) is the rectangle inside the flap.
    fold_order: The fold order (M/V) of the inscribed rectangle.
    fold_percent: The fold percent of the inscribed rectangle.
  """

  depth: int
  fold_order: Sequence[crease_utils.Fold]
  fold_percent: float


@dataclasses.dataclass(kw_only=True, frozen=True)
class InscribedRectangleFlapShapingParams:
  """Wrapper around flap name + list of inscribed rectangle params.

  Attributes:
    flap_name: The name of the flap to be shaped.
    inscribed_rectangles_params: The list of inscribed rectangle params
      determining the placement of the inscribed rectangles.
  """

  flap_name: str
  inscribed_rectangles_params: Sequence[InscribedRectangleParams]


def find_start_point(
    corners: Sequence[point_utils.Point],
    edges: Sequence[crease_utils.Crease],
) -> int:
  """Finds the start point of the inscribed polygon to begin a clockwise walk.

  Args:
    corners: The corners of the inscribed rectangle.
    edges: The edges of the inscribed rectangle.

  Returns:
    start_index: The index of the corner to start the clockwise walk from.
  """
  start_bools = [
      shaping_utils.check_corner_between_border_edge_and_interior_edge(
          corner, corners, edges
      )
      for corner in corners
  ]
  if any(start_bools):
    start_index = start_bools.index(True)
  # no border corners found, pick top left corner arbitrarily
  else:
    start_index = 0
  return start_index


def get_inscribed_rectangle(
    corners: Sequence[point_utils.Point],
    circle_centers: Sequence[point_utils.Point],
    grid_size: int,
    inscribe_depth: float,
) -> shaping_utils.Rectangle:
  """Inscribes a rectangle within the given corners with (grid unit) depth given by inscribe_depth.

  Args:
    corners: The corners of the parent rectangle.
    circle_centers: The center(s) of the circle(s) inscribed inside the parent
      rectangle.
    grid_size: The grid size of the box pleating crease pattern.
    inscribe_depth: The depth of the rectangle to be inscribed inside its
      parent.

  Returns:
    The inscribed rectangle.
  """
  if not circle_centers:
    raise ValueError("circle_centers must be non-empty.")
  grid_delta = 1 / grid_size
  # get inscribed rectangle corners
  new_corners = []
  for corner in corners:
    timeout_lib.maybe_timeout_cooperatively()
    # get closest circle center
    closest_center = min(
        circle_centers,
        key=lambda x, current_corner=corner: point_utils.distance(
            x, current_corner
        ),
    )
    # get vector pointing from corner to circle center
    center_vector = vector.Vector.from_point(
        closest_center
    ) - vector.Vector.from_point(corner)
    # rescale so that neither component is larger than grid unit length
    grid_unit_center_vector = grid_delta * center_vector.inf_normalize
    assert (
        grid_unit_center_vector.slope == float("inf")
        or math.isclose(abs(grid_unit_center_vector.slope), 0)
        or math.isclose(abs(grid_unit_center_vector.slope), 1)
    ), (
        f"Unexpected grid_unit_center_vector: {grid_unit_center_vector},"
        f" corner: {corner}, circle_center: {circle_centers}"
    )
    displacement_vector = inscribe_depth * grid_unit_center_vector
    if (
        displacement_vector.norm >= center_vector.norm - shaping_utils.EPS
        and not corner.isclose(closest_center)
    ):
      raise ValueError(
          "inscribe_depth is too large!",
          f"center_vector: {center_vector}",
          f"displacement_vector: {displacement_vector}",
      )
    new_corner = displacement_vector.point_add(corner)
    new_corners.append(new_corner)
  inscribed_rectangle = shaping_utils.Rectangle.from_corners(
      new_corners, "inscribed_rectangle", list(circle_centers)[:2]
  )
  return inscribed_rectangle


def get_inscribed_rectangle_creases(
    *,
    grid_size: int,
    start_index: int,
    corners: Sequence[point_utils.Point],
    fold_order: Sequence[crease_utils.Fold],
    fold_percent: float,
    central_crease_removed: bool = True,
    fold_step: int = 0,
) -> list[crease_utils.Crease]:
  """Returns a list of creases for along an inscribed rectangle.

  Walks clockwise along the boundary of the inscribed rectangle,
  drawing a unit grid length creases between each pair of points.

  Args:
    grid_size: The grid size of the box pleating crease pattern.
    start_index: The index of a corner on the inscribed rectangle boundary to
      start walking clockwise from.
    corners: The corners of the inscribedrectangle.
    fold_order: A sequence of folds detailing the fold order; for example a fold
      order of [MOUNTAIN, VALLEY] will have the inscribed rectangle crease
      starting with a mountain fold and then alternating valley and mountain
      orientations, whereas a fold order of [VALLEY] will have inscribed
      rectangle creases be all valley folds.
    fold_percent: The fold percent for all the creases.
    central_crease_removed: Whether the central crease is removed on the crease
      pattern.
    fold_step: The folding step for the creases.

  Returns:
    inscribed_polygon_creases: The inscribed polygon creases.
  """
  inscribed_polygon_creases = []
  start = corners[start_index]
  cur_point = start
  cur_edge_index = start_index

  fold_cycler = itertools.cycle(fold_order)
  boundary = shaping_elements.get_flap_boundary(
      shaping_utils.Rectangle.from_corners(corners, "inscribed_rectangle"),
      grid_size,
  )
  while not (start.isclose(cur_point) and inscribed_polygon_creases) and not (
      cur_point.is_border and inscribed_polygon_creases
  ):
    timeout_lib.maybe_timeout_cooperatively()
    # get next point and edge index
    next_point, next_edge_index = boundary.walk(
        cur_point, cur_edge_index, grid_size
    )
    # extend crease by an extra grid point if
    # along center and central flap removed
    # NOTE: assumes central flap is removed along x = 0.5
    if (
        central_crease_removed
        and next_edge_index == cur_edge_index
        and not math.isclose(
            cur_point.x, _CENTRAL_CREASE_X, abs_tol=shaping_utils.EPS
        )
        and math.isclose(
            next_point.x, _CENTRAL_CREASE_X, abs_tol=shaping_utils.EPS
        )
    ):
      next_point, next_edge_index = boundary.walk(
          next_point, next_edge_index, grid_size
      )
    # draw crease between cur and next point
    rectangle_crease = crease_utils.Crease(
        cur_point,
        next_point,
        fold=next(fold_cycler),
        fold_percent=fold_percent,
        fold_step=fold_step,
    )
    inscribed_polygon_creases.append(rectangle_crease)
    # update cur point
    cur_point = next_point
    cur_edge_index = next_edge_index
  return inscribed_polygon_creases


def shape_flap_with_inscribed_rectangles(
    flap_rectangle: shaping_utils.Rectangle | shaping_elements.ShapingFlap,
    grid_size: int,
    shaping_params: Sequence[InscribedRectangleParams],
    central_crease_removed: bool = True,
    fold_step: int = 0,
) -> tuple[list[crease_utils.Crease], list[shaping_utils.Rectangle]]:
  """Inscribes rectangles in a flap to generate shaping creases.

  Args:
    flap_rectangle: The flap to shape.
    grid_size: The grid size of the crease pattern.
    shaping_params: Parameters for the inscribed rectangles.
    central_crease_removed: Whether the central crease is removed.
    fold_step: The folding step for the creases.

  Returns:
    A tuple of the generated creases and the inscribed rectangles.
  """
  # NOTE: fully migrate to ShapingFlap and remove Rectangle type
  if isinstance(flap_rectangle, shaping_elements.ShapingFlap):
    flap_rectangle = flap_rectangle.to_legacy_rectangle()

  inscribed_rectangles = []
  inscribed_polygon_creases = []
  flap_corners, _, _, _ = shaping_utils.preprocess_rect(
      flap_rectangle, grid_size, None, 1
  )
  # now create inscribed rectangle creases
  # go from largest rectangle to smallest
  shaping_params = sorted(shaping_params, key=lambda x: x.depth)
  for rectangle_param in shaping_params:
    timeout_lib.maybe_timeout_cooperatively()
    # Skip if fold percent is 0
    if math.isclose(rectangle_param.fold_percent, 0):
      continue

    # get inscribed rectangle object
    inscribed_rectangle = get_inscribed_rectangle(
        flap_corners,
        flap_rectangle.circle_centers,
        grid_size,
        rectangle_param.depth,
    )
    # restrict inscribed rectangle to paper
    clipped_inscribed_rectangle = inscribed_rectangle.clip_to_paper()
    if clipped_inscribed_rectangle is None:
      continue
    inscribed_rectangles.append(inscribed_rectangle)
    # preprocess inscribed rectangle object
    inscribed_corners, inscribed_edges, _, _ = shaping_utils.preprocess_rect(
        clipped_inscribed_rectangle, grid_size, None, 1
    )
    # find clockwise walk starting corner index by looking
    # for corner between border and interior edge
    start_index = find_start_point(inscribed_corners, inscribed_edges)
    inscribed_polygon_creases.extend(
        get_inscribed_rectangle_creases(
            grid_size=grid_size,
            start_index=start_index,
            corners=inscribed_corners,
            fold_order=rectangle_param.fold_order,
            fold_percent=rectangle_param.fold_percent,
            central_crease_removed=central_crease_removed,
            fold_step=fold_step,
        )
    )

  return inscribed_polygon_creases, inscribed_rectangles


def get_inscribed_rectangle_shaping_creases_and_rectangles(
    grid_size: int,
    flap_rectangles: Sequence[
        shaping_utils.Rectangle | shaping_elements.ShapingFlap
    ],
    flap_shaping_params: Sequence[InscribedRectangleFlapShapingParams],
    *,
    fold_step: int,
    central_crease_removed: bool = True,
) -> tuple[list[crease_utils.Crease], list[shaping_utils.Rectangle]]:
  """Generates creases by inscribing rectangles within specified flaps.

  This function iterates through a list of flap shaping parameters. For each
  shaping parameter, it finds the corresponding flap rectangle and inscribes a
  rectangle based on those parameters. The resulting creases and the inscribed
  rectangles themselves are collected and returned. This is used to add 3D
  shaping to a box-pleated crease pattern.

  Args:
    grid_size: The grid size of the crease pattern.
    flap_rectangles: Rectangles or flaps defining each flap of the crease
      pattern.
    flap_shaping_params: Parameters defining the shaping of flaps.
    fold_step: The folding step for the creases.
    central_crease_removed: Whether the central crease is removed. This affects
      how creases along the centerline are handled.

  Returns:
    New shaping creases and the inscribed rectangles that were created.
  """
  rectangles_by_name = {rect.name: rect for rect in flap_rectangles}
  inscribed_polygon_creases = []
  inscribed_rectangles = []
  for i, flap_shaping_param in enumerate(flap_shaping_params):
    timeout_lib.maybe_timeout_cooperatively()
    flap_rectangle = rectangles_by_name.get(flap_shaping_param.flap_name)
    if flap_rectangle is None:
      raise ValueError(
          f"Flap rectangle {flap_shaping_param.flap_name} not found."
      )
    new_creases, inscribed_rectangle = shape_flap_with_inscribed_rectangles(
        flap_rectangle,
        grid_size,
        flap_shaping_param.inscribed_rectangles_params,
        central_crease_removed,
        fold_step=fold_step + i,
    )
    inscribed_polygon_creases.extend(new_creases)
    inscribed_rectangles.extend(inscribed_rectangle)
  return inscribed_polygon_creases, inscribed_rectangles


def shape_cp_with_inscribed_rectangles(
    cp: cp_utils.CreasePattern,
    grid_size: int,
    flap_rectangles: Sequence[shaping_utils.Rectangle],
    flap_shaping_params: Sequence[InscribedRectangleFlapShapingParams],
    *,
    central_crease_removed: bool = True,
    light_mode: bool = False,
) -> cp_utils.CreasePattern:
  """Inscribes sets of rectangles inside a list of flaps.

  Args:
    cp: The crease pattern to be shaped.
    grid_size: The grid size of the crease pattern.
    flap_rectangles: Rectangles containing each flap.
    flap_shaping_params: The shaping parameters determining inscribed rectangle
      placement.
    central_crease_removed: Whether the central crease is removed.
    light_mode: Whether the crease pattern is loaded in light mode.

  Returns:
    The inscribed rectangle shaped crease pattern and the inscribed rectangles.
  """
  inscribed_polygon_creases = (
      get_inscribed_rectangle_shaping_creases_and_rectangles(
          grid_size,
          flap_rectangles,
          flap_shaping_params,
          central_crease_removed=central_crease_removed,
          fold_step=cp.num_fold_steps,
      )[0]
  )

  # NOTE: prevent overlap with existing creases
  # this can happen if inscribing on boundary of flap
  # for now it is fine if overlapping creases come after originals
  # and light mode is False
  # NOTE: fix flat foldability issues
  return cp_utils.CreasePattern(
      creases=[*cp.creases, *inscribed_polygon_creases],
      light_mode=light_mode,
  )


# -------- Inscribed River shaping --------


@dataclasses.dataclass(kw_only=True, frozen=True)
class InscribedRiverParams:
  """Params determining placement of an inscribed river.

  Attributes:
    depth: How deep (in grid units) is the river inside the flap.
    fold_order: The fold order (M/V) of the inscribed river.
    fold_percent: The fold percent of the inscribed river.
    bank_index: The index of the bank to inscribe.
  """

  depth: int
  fold_order: Sequence[crease_utils.Fold]
  fold_percent: float
  bank_index: shaping_types.BankIndex


@dataclasses.dataclass(kw_only=True, frozen=True)
class InscribedRiverShapingParams:
  """Params determining placement of an inscribed river in a particular flap.

  Attributes:
    element_name: The name of the element to be shaped.
    inscribed_river_params: The list of inscribed river params determining the
      placement of the inscribed rivers.
  """

  element_name: str
  inscribed_river_params: Sequence[InscribedRiverParams]


def get_inscribed_river_creases(
    river: shaping_elements.ShapingRiver,
    inscribed_river_param: InscribedRiverParams,
    central_crease_removed: bool,
) -> Sequence[crease_utils.Crease]:
  """Returns a list of creases for an inscribed river bank."""
  inscribed_bank_creases_segments = river.get_inscribed_river_bank(
      inscribed_river_param.bank_index, inscribed_river_param.depth
  )
  fold_cycler = itertools.cycle(inscribed_river_param.fold_order)
  grid_size = river.grid_size
  edge_vectors = river.edge_unit_vectors
  new_creases = []
  for segment, edge_vector in zip(
      inscribed_bank_creases_segments, edge_vectors, strict=True
  ):
    timeout_lib.maybe_timeout_cooperatively()
    # NOTE: how to handle with non-integer crease length?
    # This occurs during non-integer depth requests.
    # One option is to handle such requests using zigzag
    # with target_zigzag_depth = 0.
    num_steps = shaping_utils.integer_crease_grid_length(segment, grid_size)
    start = segment.p1_init
    ind = 0
    while ind < num_steps:
      end = edge_vector.point_add(start)
      # Handle central crease specially
      if (
          central_crease_removed
          and not math.isclose(
              start.x, _CENTRAL_CREASE_X, abs_tol=shaping_utils.EPS
          )
          and math.isclose(end.x, _CENTRAL_CREASE_X, abs_tol=shaping_utils.EPS)
          and ind + 1 < num_steps
      ):
        end = edge_vector.point_add(end)
        ind += 1
      new_creases.append(
          crease_utils.Crease(
              start,
              end,
              fold=next(fold_cycler),
              fold_percent=inscribed_river_param.fold_percent,
          )
      )
      start = end
      ind += 1
  return new_creases


def shape_river_with_inscribed_rivers(
    river: shaping_elements.ShapingRiver,
    shaping_params: Sequence[InscribedRiverParams],
    central_crease_removed: bool,
) -> Sequence[crease_utils.Crease]:
  """Inscribes rivers in an element to generate shaping creases."""
  inscribed_river_creases = []
  shaping_params = sorted(shaping_params, key=lambda x: x.depth)
  for river_param in shaping_params:
    timeout_lib.maybe_timeout_cooperatively()
    # Skip if fold percent is 0
    if math.isclose(river_param.fold_percent, 0):
      continue

    inscribed_river_creases.extend(
        get_inscribed_river_creases(river, river_param, central_crease_removed)
    )
  return shaping_utils.restrict_creases_to_paper(inscribed_river_creases)


def get_inscribed_river_shaping_creases(
    rivers: Sequence[shaping_elements.ShapingRiver],
    river_shaping_params: Sequence[InscribedRiverShapingParams],
    central_crease_removed: bool,
) -> Sequence[crease_utils.Crease]:
  """Generates creases by inscribing rivers within specified elements.

  Args:
    rivers: Rivers to shape.
    river_shaping_params: The shaping parameters determining inscribed river
      placement.
    central_crease_removed: Whether the central crease is removed.

  Returns:
    The inscribed river shaping creases.
  """
  rivers_by_name = {river.name: river for river in rivers}
  inscribed_river_creases = []
  for shaping_param in river_shaping_params:
    timeout_lib.maybe_timeout_cooperatively()
    river = rivers_by_name.get(shaping_param.element_name)
    if river is None:
      raise ValueError(f"River {shaping_param.element_name} not found.")
    new_creases = shape_river_with_inscribed_rivers(
        river,
        shaping_param.inscribed_river_params,
        central_crease_removed,
    )
    inscribed_river_creases.extend(new_creases)
  return inscribed_river_creases


# -------- Combined flap/river inscribed shaping --------


@dataclasses.dataclass(kw_only=True, frozen=True)
class InscribedShapingParamsForElement:
  """Shaping params for a single element (flap or river).

  Attributes:
    element_name: The name of the element being shaped.
    inscribed_params: The params determining the drawing of inscribed creases.
  """

  element_name: str
  inscribed_params: (
      Sequence[InscribedRectangleParams] | Sequence[InscribedRiverParams]
  )


def get_inscribed_element_shaping_creases(
    grid_size: int,
    rectangles: Sequence[shaping_elements.ShapingFlap],
    rivers: Sequence[shaping_elements.ShapingRiver],
    element_shaping_params: Sequence[InscribedShapingParamsForElement],
    central_crease_removed: bool,
) -> Sequence[crease_utils.Crease]:
  """Returns inscribed creases for shaping elements of a crease pattern.

  Args:
    grid_size: The size of the crease pattern grid.
    rectangles: The rectangles available for shaping.
    rivers: The rivers available for shaping.
    element_shaping_params: The elements to shape and their specific params.
    central_crease_removed: Whether the central crease is removed.

  Returns:
    A list of `crease_utils.Crease` objects representing all the generated
    inscribed creases.
  """
  rectangle_dict = {rectangle.name: rectangle for rectangle in rectangles}
  river_dict = {river.name: river for river in rivers}
  element_dict = {**rectangle_dict, **river_dict}
  inscribed_creases = []
  for params in element_shaping_params:
    timeout_lib.maybe_timeout_cooperatively()
    element = element_dict.get(params.element_name, None)
    if element is None:
      raise ValueError(
          f"Element {params.element_name} to be shaped not found in CP."
      )
    # sort into flap and river elements
    if isinstance(element, shaping_elements.ShapingFlap):
      if not isinstance(params.inscribed_params[0], InscribedRectangleParams):
        raise ValueError(
            f"Element {element.name} is a rectangle but params"
            f" {params.inscribed_params} are not InscribedRectangleParams!"
        )

      new_creases, _ = shape_flap_with_inscribed_rectangles(
          element,
          grid_size,
          params.inscribed_params,
          central_crease_removed,
      )
      inscribed_creases.extend(new_creases)
    elif isinstance(element, shaping_elements.ShapingRiver):
      if not isinstance(params.inscribed_params[0], InscribedRiverParams):
        raise ValueError(
            f"Element {element.name} is a river but params"
            f" {params.inscribed_params} are not InscribedRiverParams!"
        )
      new_creases = shape_river_with_inscribed_rivers(
          element,
          params.inscribed_params,
          central_crease_removed,
      )
      inscribed_creases.extend(new_creases)
    else:
      raise ValueError(f"Element {element} is not a rectangle or river!")
  return inscribed_creases


def shape_cp_with_inscribed_elements(
    cp: cp_utils.CreasePattern,
    grid_size: int,
    rectangles: Sequence[shaping_elements.ShapingFlap],
    rivers: Sequence[shaping_elements.ShapingRiver],
    element_shaping_params: Sequence[InscribedShapingParamsForElement],
    central_crease_removed: bool,
    *,
    light_mode: bool = False,
    raise_error_on_ff_violation: bool = False,
) -> cp_utils.CreasePattern:
  """Wrapper around inscribed shaping function.

  Args:
    cp: The crease pattern to shape.
    grid_size: The size of the box-pleating grid.
    rectangles: The rectangles available for shaping.
    rivers: The rivers available for shaping.
    element_shaping_params: The shaping parameters determining inscribed crease
      placement for each element (flap or river).
    central_crease_removed: Whether the central crease is removed.
    light_mode: If light_mode is False, CreasePattern will be instantiated after
      splitting and merging creases that intersect with the inscribed lines.
      This is an expensive operation. If light_mode is True, creases will not be
      split and merged, which saves time.
    raise_error_on_ff_violation: Whether to raise an error if the crease pattern
      is not flat-foldable after shaping.

  Returns:
    A crease pattern shaped with inscribed creases.
  """
  inscribed_creases = get_inscribed_element_shaping_creases(
      grid_size,
      rectangles,
      rivers,
      element_shaping_params,
      central_crease_removed,
  )
  shaped_cp = cp.update_with_shaping_creases(
      inscribed_creases,
      split_shaping_by_shaping=not light_mode,
  )
  if raise_error_on_ff_violation and (not shaped_cp.local_flat_foldability()):
    raise ValueError("Inscribed rectangle shaped CP is not flat-foldable.")
  return shaped_cp
