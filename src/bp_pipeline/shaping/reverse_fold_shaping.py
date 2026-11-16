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

"""Reverse fold shaping methods."""

import dataclasses
import functools
import math
from typing import Literal, Sequence

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..crease_pattern import point
from . import inscribed_rectangle_shaping
from . import shaping_elements
from . import shaping_utils
from . import simple_fold_shaping
from . import types as shaping_types
from . import zigzag_shaping
from .. import timeout as timeout_lib


@dataclasses.dataclass(kw_only=True, frozen=True)
class ReverseFoldParams:
  """Params determining placement of a reverse fold.

  Attributes:
    flap_angle: The angle of the flap to be shaped.
    fold_order: A sequence of folds detailing the fold order; for example a fold
      order of [MOUNTAIN, VALLEY] will have the zigzag crease starting with a
      mountain fold and then alternating valley and mountain orientations,
      whereas a fold order of [VALLEY] will have zigzag creases be all valley
      folds.
    even_phase: Determines the phase shift of the zigzags.
    bank_index: The index of the river bank to shape.
    fold_percent: The percentage of the fold depth to use.
  """

  flap_angle: float
  fold_order: Sequence[crease_utils.Fold]
  even_phase: bool
  bank_index: shaping_types.BankIndex | None = None
  fold_percent: float = 1.0


@dataclasses.dataclass(kw_only=True, frozen=True)
class InscribedReverseFoldParams(ReverseFoldParams):
  """Params for reverse fold with inscribed logic (e.g. angle close to PI).

  Attributes:
    depth: The depth of the reverse fold (width of the pleat).
  """

  depth: int


@dataclasses.dataclass(kw_only=True, frozen=True)
class ReverseFoldShapingParams:
  """Wrapper around flap name + list of reverse fold params.

  Attributes:
    flap_name: The name of the flap to be shaped.
    reverse_fold_params: The list of reverse fold params determining the
      placement of the folds.
    orientation: The orientation of the reverse fold.
    fold_type: The orientation to be used for the reverse fold (i.e., inside or
      outside reverse fold)
  """

  flap_name: str
  reverse_fold_params: Sequence[ReverseFoldParams]
  orientation: Literal['vertical', 'horizontal'] = 'vertical'
  fold_type: crease_utils.Fold = crease_utils.Fold.VALLEY


@functools.cache
def make_reverse_fold_params(
    *,
    flap_angle: float,
    even_phase: bool,
    fold_order: Sequence[crease_utils.Fold],
    fold_percent: float,
    bank_index: shaping_types.BankIndex | None = None,
    depth: int | None = None,
) -> ReverseFoldParams:
  """Factory to create ReverseFoldParams or InscribedReverseFoldParams."""
  if math.isclose(
      abs(flap_angle),
      math.pi,
      abs_tol=simple_fold_shaping.PI_THRESHOLD,
  ):
    return InscribedReverseFoldParams(
        flap_angle=flap_angle,
        fold_order=fold_order,
        even_phase=even_phase,
        bank_index=bank_index,
        depth=depth or 0,
        fold_percent=fold_percent,
    )
  else:
    return ReverseFoldParams(
        flap_angle=flap_angle,
        fold_order=fold_order,
        even_phase=even_phase,
        bank_index=bank_index,
        fold_percent=fold_percent,
    )


@dataclasses.dataclass(frozen=True)
class ZigzagInstruction:
  """Instruction for creating a zigzag crease."""

  crease: crease_utils.Crease
  is_beak: bool
  original_index: int
  flap_rectangle: shaping_utils.Rectangle | shaping_elements.ShapingFlap | None


def _classify_zigzag_creases(
    flap_rectangle: shaping_utils.Rectangle | shaping_elements.ShapingFlap,
    zigzag_creases: Sequence[crease_utils.Crease],
) -> tuple[list[ZigzagInstruction], list[crease_utils.Crease]]:
  """Classifies zigzag creases into instructions and identifies beak creases."""
  instructions = []

  # Find the beak pair: two consecutive creases whose shared vertex is on
  # the boundary and closest to center.
  best_pair = None
  best_dist = float('inf')

  # Pre-calculate bounds for efficiency
  x_min = flap_rectangle.x_start
  y_min = flap_rectangle.y_start
  x_max = x_min + flap_rectangle.x_len
  y_max = y_min + flap_rectangle.y_len
  center = flap_rectangle.center  # Tuple (x, y)
  center_point = point.Point(x=center[0], y=center[1], require_on_paper=False)

  # Helper to check boundary
  def is_on_boundary(p: point.Point) -> bool:
    return (
        math.isclose(p.x, x_min, abs_tol=point.DEFAULT_TOLERANCE)
        or math.isclose(p.x, x_max, abs_tol=point.DEFAULT_TOLERANCE)
        or math.isclose(p.y, y_min, abs_tol=point.DEFAULT_TOLERANCE)
        or math.isclose(p.y, y_max, abs_tol=point.DEFAULT_TOLERANCE)
    )

  # Iterate through all pairs to find consecutive ones (sharing a vertex)
  # irrespective of list order.
  for i in range(len(zigzag_creases)):
    timeout_lib.maybe_timeout_cooperatively()
    for j in range(i + 1, len(zigzag_creases)):
      c1 = zigzag_creases[i]
      c2 = zigzag_creases[j]
      shared_vertex = c1.vertex_with_crease(c2)

      if shared_vertex and is_on_boundary(shared_vertex):
        dist = point.distance(shared_vertex, center_point)
        if dist < best_dist:
          best_dist = dist
          best_pair = (i, j)

  beak_creases = []
  beak_indices = set()
  if best_pair:
    beak_indices = set(best_pair)
    beak_creases = [zigzag_creases[best_pair[0]], zigzag_creases[best_pair[1]]]

  # Generate instructions
  for idx, crease in enumerate(zigzag_creases):
    if idx in beak_indices:
      instructions.append(
          ZigzagInstruction(
              crease=crease,
              is_beak=True,
              original_index=idx,
              flap_rectangle=None,
          )
      )
    else:
      instructions.append(
          ZigzagInstruction(
              crease=crease,
              is_beak=False,
              original_index=idx,
              flap_rectangle=flap_rectangle,
          )
      )

  return instructions, beak_creases


def _modify_base_creases(
    flap_rectangle: shaping_utils.Rectangle | shaping_elements.ShapingFlap,
    existing_creases: Sequence[crease_utils.Crease],
    beak_creases: Sequence[crease_utils.Crease],
    zigzag_instructions: Sequence[ZigzagInstruction],
) -> Sequence[crease_utils.Crease]:
  """Modifies base creases based on reverse fold geometry."""
  if len(beak_creases) < 2:
    raise ValueError(
        f'Expected at least two reverse fold creases, got {len(beak_creases)}.'
    )

  # Identify base creases to modification.
  potential_creases_to_split_and_flip = []
  other_creases = []
  for crease in existing_creases:
    if (
        crease.vertex_with_crease(beak_creases[0]) is not None
        and crease.vertex_with_crease(beak_creases[1]) is not None
    ):
      potential_creases_to_split_and_flip.append(crease)
    else:
      other_creases.append(crease)

  if not potential_creases_to_split_and_flip:
    return existing_creases

  # We need the final geometry of zigzags to filter split creases,
  # but for splitting purposes, the fold doesn't matter, just geometry.
  # We construct a temp list of "likely" zigzag creases for geometry check.
  temp_zigzag_creases = [instr.crease for instr in zigzag_instructions]

  temp_cp = cp_utils.CreasePattern(
      creases=temp_zigzag_creases + potential_creases_to_split_and_flip
  )

  modified_creases = list(other_creases)
  for crease in temp_cp.creases:
    timeout_lib.maybe_timeout_cooperatively()
    if any(crease.isclose(c) for c in temp_zigzag_creases):
      continue

    if (
        flap_rectangle.crease_is_within(crease)
        and crease.vertex_with_crease(beak_creases[0]) is not None
        and crease.vertex_with_crease(beak_creases[1]) is not None
        and crease.fold != crease_utils.Fold.BORDER
    ):
      modified_creases.append(crease.flip_fold())
    else:
      modified_creases.append(crease)

  return modified_creases


def _get_reverse_fold_instructions_for_flap(
    *,
    flap_rectangle: shaping_utils.Rectangle | shaping_elements.ShapingFlap,
    zigzag_crease: Sequence[crease_utils.Crease],
    existing_creases: Sequence[crease_utils.Crease],
) -> tuple[list[ZigzagInstruction], Sequence[crease_utils.Crease]]:
  """Generates instructions and modified base creases."""

  instructions, beak_creases = _classify_zigzag_creases(
      flap_rectangle, zigzag_crease
  )

  modified_creases = _modify_base_creases(
      flap_rectangle, existing_creases, beak_creases, instructions
  )

  return instructions, modified_creases


def _get_element(
    param: ReverseFoldShapingParams,
    rectangles_by_name: dict[
        str, shaping_utils.Rectangle | shaping_elements.ShapingFlap
    ],
    rivers_by_name: dict[str, shaping_elements.ShapingRiver],
) -> (
    shaping_utils.Rectangle
    | shaping_elements.ShapingFlap
    | shaping_elements.ShapingRiver
):
  """Retrieves the shaping element (Rectangle, Flap, or River) by name.

  Args:
    param: The ReverseFoldShapingParams containing the element name.
    rectangles_by_name: A dictionary mapping flap names to Rectangle or
      ShapingFlap objects.
    rivers_by_name: A dictionary mapping river names to ShapingRiver objects.

  Returns:
    The ShapingFlap, Rectangle, or ShapingRiver object corresponding to the
    flap_name in the param.

  Raises:
    ValueError: If the element specified in param.flap_name is not found in
      either rectangles_by_name or rivers_by_name.
  """
  element = rectangles_by_name.get(param.flap_name)
  if element is None:
    element = rivers_by_name.get(param.flap_name)

  if element is None:
    raise ValueError(f'Element to be shaped not found in CP: {param.flap_name}')
  return element


def _process_inscribed_logic(
    element: (
        shaping_utils.Rectangle
        | shaping_elements.ShapingFlap
        | shaping_elements.ShapingRiver
    ),
    reverse_fold_param: ReverseFoldParams,
    grid_size: int,
) -> tuple[Sequence[crease_utils.Crease], bool]:
  """Processes the inscribed logic for reverse fold shaping.

  This function generates zigzag creases based on inscribed rectangle/river
  shaping parameters, used when the flap angle is close to PI.

  Args:
    element: The shaping element (Rectangle, Flap, or River).
    reverse_fold_param: The parameters for the reverse fold, must be an instance
      of InscribedReverseFoldParams.
    grid_size: The size of the crease pattern grid.

  Returns:
    A tuple containing:
      - A sequence of generated zigzag creases.
      - A boolean indicating whether inscribed logic was used.

  Raises:
    ValueError: If reverse_fold_param is not an instance of
      InscribedReverseFoldParams.
  """
  if not isinstance(reverse_fold_param, InscribedReverseFoldParams):
    raise ValueError(
        'Reverse fold with angle close to PI must use'
        ' InscribedReverseFoldParams.'
    )
  if reverse_fold_param.bank_index is not None:
    inscribed_params = [
        inscribed_rectangle_shaping.InscribedRiverParams(
            depth=reverse_fold_param.depth,
            fold_order=reverse_fold_param.fold_order,
            fold_percent=reverse_fold_param.fold_percent,
            bank_index=reverse_fold_param.bank_index,
        )
    ]
  else:
    inscribed_params = [
        inscribed_rectangle_shaping.InscribedRectangleParams(
            depth=reverse_fold_param.depth,
            fold_order=reverse_fold_param.fold_order,
            fold_percent=reverse_fold_param.fold_percent,
        )
    ]
  inscribed_params_for_element = [
      inscribed_rectangle_shaping.InscribedShapingParamsForElement(
          element_name=element.name,
          inscribed_params=inscribed_params,
      )
  ]

  element_for_inscribed = element
  if isinstance(element, shaping_utils.Rectangle):
    element_for_inscribed = (
        shaping_elements.ShapingFlap.from_legacy_rectangle_direct(
            element, grid_size
        )
    )

  inscribed_rectangles = []
  inscribed_rivers = []
  if isinstance(element_for_inscribed, shaping_elements.ShapingFlap):
    inscribed_rectangles = [element_for_inscribed]
  elif isinstance(element_for_inscribed, shaping_elements.ShapingRiver):
    inscribed_rivers = [element_for_inscribed]

  zigzag_crease = inscribed_rectangle_shaping.get_inscribed_element_shaping_creases(
      grid_size=grid_size,
      rectangles=inscribed_rectangles,
      rivers=inscribed_rivers,
      element_shaping_params=inscribed_params_for_element,
      # Assuming True as default comparable to zigzag
      central_crease_removed=True,
  )
  return zigzag_crease, True


def _process_zigzag_logic(
    element: (
        shaping_utils.Rectangle
        | shaping_elements.ShapingFlap
        | shaping_elements.ShapingRiver
    ),
    reverse_fold_param: ReverseFoldParams,
    grid_size: int,
    snap_to_grid: bool,
    num_fold_steps: int,
) -> tuple[Sequence[crease_utils.Crease], bool]:
  """Processes the zigzag logic for reverse fold shaping.

  This function generates zigzag creases using the standard zigzag shaping
  logic based on the provided reverse fold parameters.

  Args:
    element: The shaping element (Rectangle, Flap, or River).
    reverse_fold_param: The parameters for the reverse fold.
    grid_size: The size of the crease pattern grid.
    snap_to_grid: Whether to snap the generated creases to the grid.
    num_fold_steps: The number of fold steps in the crease pattern.

  Returns:
    A tuple containing:
      - A sequence of generated zigzag creases.
      - A boolean, always False, indicating that inscribed logic was NOT used.
  """
  # Use placeholders for fold_order, we will overwrite later.
  if reverse_fold_param.bank_index is not None:
    zigzag_params = shaping_types.RiverZigzagParams(
        target_flap_angle=reverse_fold_param.flap_angle,
        target_zigzag_depth=None,
        fold_order=reverse_fold_param.fold_order,
        even_phase=reverse_fold_param.even_phase,
        bank_index=reverse_fold_param.bank_index,
    )
  else:
    zigzag_params = shaping_types.ZigzagParams(
        target_flap_angle=reverse_fold_param.flap_angle,
        target_zigzag_depth=None,
        fold_order=reverse_fold_param.fold_order,
        even_phase=reverse_fold_param.even_phase,
    )

  is_flap = isinstance(
      element, (shaping_utils.Rectangle, shaping_elements.ShapingFlap)
  )
  rectangles_arg = [element] if is_flap else []
  rivers_arg = [element] if not is_flap else []

  zigzag_crease = zigzag_shaping.get_zigzag_shaping_creases(
      grid_size=grid_size,
      rectangles=rectangles_arg,
      element_shaping_params=[
          zigzag_shaping.ZigzagShapingParamsForElement(
              element_name=element.name,
              zigzag_params=zigzag_params,
          )
      ],
      rivers=rivers_arg,
      snap_to_grid=snap_to_grid,
      fold_step=num_fold_steps,
  )
  return zigzag_crease, False


def get_reverse_fold_creases(
    cp: cp_utils.CreasePattern,
    grid_size: int,
    rectangles: Sequence[
        shaping_utils.Rectangle | shaping_elements.ShapingFlap
    ],
    element_shaping_params: Sequence[ReverseFoldShapingParams],
    *,
    rivers: Sequence[shaping_elements.ShapingRiver] | None = None,
    snap_to_grid: bool = False,
) -> tuple[list[crease_utils.Crease], list[ZigzagInstruction]]:
  """Generates base creases (modified) and instructions for zigzag creases.

  Args:
    cp: The crease pattern to be shaped.
    grid_size: The size of the crease pattern.
    rectangles: The rectangles defining the flaps in the crease pattern.
    element_shaping_params: The parameters for shaping the reverse folds.
    rivers: The rivers in the crease pattern.
    snap_to_grid: Whether to snap the zigzag creases to the grid.

  Returns:
    A tuple of (modified_base_creases, zigzag_instructions).
  """
  base_creases = cp.creases
  all_instructions = []

  rectangles_by_name = {rect.name: rect for rect in rectangles}
  if rivers is None:
    rivers = []
  rivers_by_name = {river.name: river for river in rivers}

  for param in element_shaping_params:
    element = _get_element(param, rectangles_by_name, rivers_by_name)
    for reverse_fold_param in param.reverse_fold_params:
      timeout_lib.maybe_timeout_cooperatively()
      if math.isclose(reverse_fold_param.flap_angle, 0):
        continue

      use_inscribed_logic = math.isclose(
          abs(reverse_fold_param.flap_angle),
          math.pi,
          abs_tol=simple_fold_shaping.PI_THRESHOLD,
      )

      if use_inscribed_logic:
        zigzag_crease, _ = _process_inscribed_logic(
            element, reverse_fold_param, grid_size
        )
      else:
        zigzag_crease, _ = _process_zigzag_logic(
            element,
            reverse_fold_param,
            grid_size,
            snap_to_grid,
            cp.num_fold_steps,
        )

      if not zigzag_crease:
        raise ValueError(
            f'Zigzag creases were not added to element: {param.flap_name},'
            f' param: {reverse_fold_param}'
        )

      if (
          isinstance(
              element, (shaping_utils.Rectangle, shaping_elements.ShapingFlap)
          )
          and not use_inscribed_logic
      ):
        instructions, base_creases = _get_reverse_fold_instructions_for_flap(
            flap_rectangle=element,
            zigzag_crease=zigzag_crease,
            existing_creases=base_creases,
        )
        all_instructions.extend(instructions)
      else:
        # If river or inscribed logic, just add zigzag creases as instructions.
        for idx, crease in enumerate(zigzag_crease):
          all_instructions.append(
              ZigzagInstruction(
                  crease=crease,
                  is_beak=False,
                  original_index=idx,
                  flap_rectangle=None,
              )
          )

  return list(base_creases), all_instructions


def assign_reverse_fold_orders(
    instructions: list[ZigzagInstruction],
    fold_type: crease_utils.Fold,
    orientation: str | None = None,
    flip_base_folds: bool = False,
) -> list[crease_utils.Crease]:
  """Assigns folds based on instructions and requested orders."""
  final_creases = []
  for instr in instructions:
    if instr.is_beak:
      final_creases.append(dataclasses.replace(instr.crease, fold=fold_type))
    else:
      # Use the fold already computed by zigzag_shaping as the base.
      base_fold = instr.crease.fold

      needs_flip = False
      if orientation and instr.flap_rectangle:
        crease = instr.crease
        flap_rectangle = instr.flap_rectangle
        if orientation == 'vertical':
          if flap_rectangle.is_bottom_of_central_crease:
            needs_flip = crease.top_point.y > flap_rectangle.center[1]
          else:
            needs_flip = crease.bottom_point.y < flap_rectangle.center[1]
        elif orientation == 'horizontal':
          if flap_rectangle.is_right_of_central_crease:
            needs_flip = crease.left_point.x > flap_rectangle.center[0]
          else:
            needs_flip = crease.right_point.x < flap_rectangle.center[0]

      if needs_flip:
        # Flip the assigned fold to match reverse fold structure.
        base_fold = base_fold.flip()

      if flip_base_folds:
        # Flip everything if requested
        # (e.g. searching for valid flat foldability).
        base_fold = base_fold.flip()

      final_creases.append(dataclasses.replace(instr.crease, fold=base_fold))
  return final_creases


def shape_cp_from_creases(
    base_creases: Sequence[crease_utils.Crease],
    zigzag_creases: Sequence[crease_utils.Crease],
) -> cp_utils.CreasePattern:
  """Creates CP from base and zigzag creases."""
  return cp_utils.CreasePattern(creases=[*base_creases, *zigzag_creases])


def shape_cp_with_reverse_fold(
    cp: cp_utils.CreasePattern,
    grid_size: int,
    rectangles: Sequence[
        shaping_utils.Rectangle | shaping_elements.ShapingFlap
    ],
    element_shaping_params: Sequence[ReverseFoldShapingParams],
    *,
    snap_to_grid: bool = False,
) -> cp_utils.CreasePattern:
  """Shapes a crease pattern with reverse folds.

  Reverse folds are a combination of an inscribed rectangle and zigzags.
  The reverse fold is applied to a single flap at a time.

  Args:
    cp: The crease pattern to be shaped.
    grid_size: The size of the crease pattern.
    rectangles: The rectangles defining the flaps in the crease pattern.
    element_shaping_params: The parameters for shaping the reverse folds.
    snap_to_grid: Whether to snap the zigzag creases to the grid.

  Returns:
    The crease pattern with the added reverse folds.
  """
  base_creases, instructions = get_reverse_fold_creases(
      cp=cp,
      grid_size=grid_size,
      rectangles=rectangles,
      element_shaping_params=element_shaping_params,
      snap_to_grid=snap_to_grid,
  )
  fold_type = crease_utils.Fold.VALLEY
  orientation = None
  if element_shaping_params:
    fold_type = element_shaping_params[0].fold_type
    orientation = element_shaping_params[0].orientation

  zigzag_creases = assign_reverse_fold_orders(
      instructions, fold_type, orientation=orientation, flip_base_folds=False
  )
  return shape_cp_from_creases(base_creases, zigzag_creases)
