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

"""Implements simple fold shaping via zigzag and inscribed polygon shaping."""

import itertools
import math
from typing import Sequence

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from . import heuristics
from . import heuristics_utils
from . import inscribed_rectangle_shaping
from . import shaping_elements
from . import shaping_utils
from . import types as shaping_types
from . import zigzag_shaping
from . import zigzag_shaping_utils
from .. import timeout as timeout_lib


PI_THRESHOLD = 1e-3


def _sort_shaping_requests(
    element_shaping_params: Sequence[
        zigzag_shaping.ZigzagShapingParamsForElement
    ],
) -> tuple[
    Sequence[zigzag_shaping.ZigzagShapingParamsForElement],
    Sequence[inscribed_rectangle_shaping.InscribedShapingParamsForElement],
]:
  """Sorts shaping requests into zigzag and inscribed rectangle requests."""
  # Split into zigzag and inscribed rectangle requests.
  zigzag_requests = []
  inscribed_rectangle_requests = []
  for shaping_param in element_shaping_params:
    timeout_lib.maybe_timeout_cooperatively()
    # Get the target angle.
    if shaping_param.zigzag_params.target_flap_angle is not None:
      target_angle = shaping_param.zigzag_params.target_flap_angle
    else:
      target_angle = shaping_utils.fold_angle_from_zigzag_depth(
          shaping_param.zigzag_params.target_zigzag_depth
      )

    # Sort zigzag and inscribed rectangle requests.
    if math.isclose(
        abs(target_angle),
        math.pi,
        abs_tol=PI_THRESHOLD,
    ) and isinstance(shaping_param.zigzag_params, shaping_types.ZigzagParams):
      inscribed_rectangle_requests.append(
          inscribed_rectangle_shaping.InscribedShapingParamsForElement(
              element_name=shaping_param.element_name,
              inscribed_params=[
                  inscribed_rectangle_shaping.InscribedRectangleParams(
                      depth=int(shaping_param.zigzag_params.depth),
                      fold_order=shaping_param.zigzag_params.fold_order,
                      fold_percent=shaping_param.zigzag_params.fold_percent,
                  )
              ],
          )
      )
    elif math.isclose(
        abs(target_angle),
        math.pi,
        abs_tol=PI_THRESHOLD,
    ) and isinstance(
        shaping_param.zigzag_params, shaping_types.RiverZigzagParams
    ):
      inscribed_rectangle_requests.append(
          inscribed_rectangle_shaping.InscribedShapingParamsForElement(
              element_name=shaping_param.element_name,
              inscribed_params=[
                  inscribed_rectangle_shaping.InscribedRiverParams(
                      depth=int(shaping_param.zigzag_params.depth),
                      fold_order=shaping_param.zigzag_params.fold_order,
                      fold_percent=shaping_param.zigzag_params.fold_percent,
                      bank_index=shaping_param.zigzag_params.bank_index,
                  )
              ],
          )
      )
    else:
      zigzag_requests.append(shaping_param)

  return zigzag_requests, inscribed_rectangle_requests


def get_simple_fold_shaping_creases(
    grid_size: int,
    flaps: Sequence[shaping_elements.ShapingFlap],
    rivers: Sequence[shaping_elements.ShapingRiver],
    element_shaping_params: Sequence[
        zigzag_shaping.ZigzagShapingParamsForElement
    ],
    central_crease_removed: bool,
) -> Sequence[crease_utils.Crease]:
  """Returns the creases that would be added by simple fold shaping."""
  zigzag_requests, inscribed_rectangle_requests = _sort_shaping_requests(
      element_shaping_params
  )

  # Call zigzag and inscribed rectangle shaping functions.
  creases = []
  if inscribed_rectangle_requests:
    inscribed_creases = (
        inscribed_rectangle_shaping.get_inscribed_element_shaping_creases(
            grid_size=grid_size,
            rectangles=flaps,
            rivers=rivers,
            element_shaping_params=inscribed_rectangle_requests,
            central_crease_removed=central_crease_removed,
        )
    )
    creases.extend(inscribed_creases)
  if zigzag_requests:
    # NOTE: Add support for passing flap names to select which
    zigzag_creases = zigzag_shaping.get_zigzag_shaping_creases(
        grid_size=grid_size,
        rectangles=flaps,
        rivers=rivers,
        element_shaping_params=zigzag_requests,
        fold_step=0,  # dummy fold_step argument.
    )
    creases.extend(zigzag_creases)

  return creases


def shape_cp_with_simple_folds(
    cp: cp_utils.CreasePattern,
    grid_size: int,
    flaps: Sequence[shaping_elements.ShapingFlap],
    rivers: Sequence[shaping_elements.ShapingRiver],
    element_shaping_params: Sequence[
        zigzag_shaping.ZigzagShapingParamsForElement
    ],
    central_crease_removed: bool,
    light_mode: bool = True,
    narrow: bool = False,
    raise_error_on_ff_violation: bool = False,
) -> cp_utils.CreasePattern:
  """Applies simple folds to list of flaps and rivers in a CreasePattern.

  If target_angle is close to pi, applies an inscribed rectangle shaping.
  Otherwise, applies zigzag shaping with target_angle.

  Args:
    cp: The crease pattern to shape.
    grid_size: The size of grid.
    flaps: The flaps to shape.
    rivers: The rivers to shape.
    element_shaping_params: The shaping params for each element.
    central_crease_removed: Whether the central crease is removed.
    light_mode: Whether to apply light mode.
    narrow: Whether to apply narrowing after simple folding.
    raise_error_on_ff_violation: Whether to raise an error if the crease pattern
      is not flat-foldable after shaping.

  Returns:
    The shaped crease pattern.
  """
  zigzag_requests, inscribed_rectangle_requests = _sort_shaping_requests(
      element_shaping_params
  )

  # Call zigzag and inscribed rectangle shaping functions.
  if inscribed_rectangle_requests:
    cp = inscribed_rectangle_shaping.shape_cp_with_inscribed_elements(
        cp=cp,
        grid_size=grid_size,
        rectangles=flaps,
        rivers=rivers,
        element_shaping_params=inscribed_rectangle_requests,
        central_crease_removed=central_crease_removed,
        light_mode=light_mode,
        raise_error_on_ff_violation=raise_error_on_ff_violation,
    )
  if zigzag_requests:
    # NOTE: Add support for passing flap names to select which
    # flaps to apply narrow shaping to.
    cp = zigzag_shaping.shape_cp_with_zigzags_pipeline_wrapper(
        cp=cp,
        grid_size=grid_size,
        rectangles=flaps,
        rivers=rivers,
        element_shaping_params=zigzag_requests,
        light_mode=light_mode,
        narrow=narrow,
        raise_error_on_ff_violation=raise_error_on_ff_violation,
    )

  cp.grid_size = grid_size
  return cp


def apply_heuristics_to_shaping_params(
    *,
    grid_size: int,
    flaps: Sequence[shaping_elements.ShapingFlap],
    rivers: Sequence[shaping_elements.ShapingRiver],
    element_orientations: dict[str, heuristics.AbsFlapDirection],
    hinge_segments: list[crease_utils.Crease],
    element_shaping_params: Sequence[
        zigzag_shaping.ZigzagShapingParamsForElement
    ],
    use_heuristics_for_fold_order: bool,
    use_heuristics_for_depth: bool,
    use_heuristics_for_phase: bool,
    convert_small_angles: bool,
) -> Sequence[zigzag_shaping.ZigzagShapingParamsForElement]:
  """Applies heuristics to shaping params to simplify tool learning.

    1. Estimate phase and fold order using heuristics.
    2. Convert small angles to two large angles, avoiding large zigzags.

  Args:
    grid_size: The size of the grid.
    flaps: The flaps to shape.
    rivers: The rivers to shape.
    element_orientations: The orientations of the elements.
    hinge_segments: The hinge segments of the stick figure.
    element_shaping_params: The shaping params for each element.
    use_heuristics_for_fold_order: Whether to use heuristics to determine the
      fold order.
    use_heuristics_for_depth: Whether to use heuristics to determine the fold
      depth.
    use_heuristics_for_phase: Whether to use heuristics to determine the fold
      phase.
    convert_small_angles: Whether to convert small angles to two large angles.

  Returns:
    The updated shaping params for each element.
  """
  new_element_shaping_params = []
  element_dict = {
      element.name: element for element in list(flaps) + list(rivers)
  }
  for shaping_params in element_shaping_params:
    timeout_lib.maybe_timeout_cooperatively()
    element_name = shaping_params.element_name
    element = element_dict[element_name]
    zigzag_params = shaping_params.zigzag_params
    if isinstance(zigzag_params, shaping_types.ZigzagParams):

      # Estimate phase.
      if use_heuristics_for_phase:
        phase = zigzag_shaping_utils.convert_flap_angle_to_global_phase(
            zigzag_params.target_flap_angle,
            element_orientations[element_name],
            element,
        )
      else:
        phase = zigzag_params.even_phase

      # Estimate fold order.
      if use_heuristics_for_fold_order:
        fold_order, is_valid = zigzag_shaping_utils.get_fold_order(
            is_even_phase=phase,
            grid_size=grid_size,
            rect=element,
            hinge_segments=hinge_segments,
        )
      else:
        fold_order = zigzag_params.fold_order
        is_valid = True

      request = zigzag_shaping.ZigzagShapingParamsForElement(
          element_name=element_name,
          zigzag_params=shaping_types.ZigzagParams(
              target_flap_angle=abs(zigzag_params.target_flap_angle),
              fold_order=fold_order,
              even_phase=phase,
              depth=zigzag_params.depth,
              fold_percent=zigzag_params.fold_percent,
          ),
      )

      # Convert small angles to two large angles.
      if convert_small_angles:
        requests = heuristics_utils.apply_small_angle_flip(
            request, inscribe_by_epsilon=False
        )
      else:
        requests = [request]

      # Try inscribing the fold by epsilon if neither fold order is valid.
      if use_heuristics_for_depth and not is_valid:
        requests = [heuristics_utils.apply_inscribe_trick(
            request
        ) for request in requests]

      new_element_shaping_params.extend(requests)
    elif isinstance(zigzag_params, shaping_types.RiverZigzagParams):

      # Compute phase of river.
      if use_heuristics_for_phase:
        phase = zigzag_shaping_utils.convert_river_angle_to_global_phase(
            zigzag_params.target_flap_angle,
            element_orientations[element_name],
            zigzag_params.bank_index,
            element,
        )
      else:
        phase = zigzag_params.even_phase

      request = zigzag_shaping.ZigzagShapingParamsForElement(
          element_name=element_name,
          zigzag_params=shaping_types.RiverZigzagParams(
              target_flap_angle=abs(zigzag_params.target_flap_angle),
              fold_order=zigzag_params.fold_order,
              even_phase=phase,
              depth=zigzag_params.depth,
              fold_percent=zigzag_params.fold_percent,
              bank_index=zigzag_params.bank_index,
          ),
      )

      # Convert small angles to two large angles.
      if convert_small_angles:
        requests = heuristics_utils.apply_small_angle_flip(
            request, inscribe_by_epsilon=False
        )
      else:
        requests = [request]

      new_element_shaping_params.extend(requests)
    else:
      raise ValueError(f'Unsupported zigzag params type: {type(zigzag_params)}')

  return new_element_shaping_params


# ---------------------------------------------------------------------------------
# Functions for inferring zigzag params from target creases.
# ------------------------------------------------------------------------------


def _are_matching_creases(
    c1: crease_utils.Crease, c2: crease_utils.Crease
) -> bool:
  return (
      c1.isclose(c2)
      and c1.fold == c2.fold
      and math.isclose(c1.fold_percent, c2.fold_percent, abs_tol=1e-3)
  )


def _find_crease_match(target_crease, candidate_creases) -> bool:
  return any(
      _are_matching_creases(target_crease, candidate_crease)
      for candidate_crease in candidate_creases
  )


def infer_fold_order_phase_from_creases(
    target_creases: Sequence[crease_utils.Crease],
    element: shaping_elements.ShapingElement,
    target_flap_angle: float,
    inscribe_depth: float,
    fold_percent: float,
    central_crease_removed: bool,
) -> zigzag_shaping.ZigzagShapingParamsForElement | None:
  """Infers the fold order and phase from the creases.

  Args:
      target_creases: The target creases to match.
      element: The element to infer the zigzag params for.
      target_flap_angle: The target flap angle to match.
      inscribe_depth: The depth to inscribe the zigzags.
      fold_percent: The fold percent to match.
      central_crease_removed: Whether the central crease has been removed.

  Returns:
      The zigzag shaping params for the element, or None if no match is found.
  """
  phases = [True, False]
  fold_orders = [
      [crease_utils.Fold.MOUNTAIN, crease_utils.Fold.VALLEY],
      [crease_utils.Fold.VALLEY, crease_utils.Fold.MOUNTAIN],
  ]

  best_params = None

  # Prepare common params for matching
  for even_phase, fold_order in itertools.product(phases, fold_orders):
    # Construct ZagzagParams/RiverZigzagParams
    if isinstance(element, shaping_elements.ShapingRiver):
      zigzag_params = shaping_types.RiverZigzagParams(
          target_flap_angle=target_flap_angle,
          fold_order=fold_order,
          even_phase=even_phase,
          bank_index=element.get_outer_bank_index(),
          fold_percent=fold_percent,
          depth=inscribe_depth,
      )
    else:  # ShapingFlap (Rectangle)
      zigzag_params = shaping_types.ZigzagParams(
          target_flap_angle=target_flap_angle,
          fold_order=fold_order,
          even_phase=even_phase,
          fold_percent=fold_percent,
          depth=inscribe_depth,
      )
    zigzag_params = zigzag_shaping.ZigzagShapingParamsForElement(
        element_name=element.name,
        zigzag_params=zigzag_params,
    )

    candidate_creases = get_simple_fold_shaping_creases(
        grid_size=element.grid_size,
        flaps=[element]
        if isinstance(element, shaping_elements.ShapingFlap)
        else [],
        rivers=[element]
        if isinstance(element, shaping_elements.ShapingRiver)
        else [],
        element_shaping_params=[zigzag_params],
        central_crease_removed=central_crease_removed,
    )

    if len(candidate_creases) != len(target_creases):
      continue

    is_same_crease_set = all(
        _find_crease_match(target_crease, candidate_creases)
        for target_crease in target_creases
    )
    if is_same_crease_set:
      best_params = zigzag_params
      break

  return best_params
