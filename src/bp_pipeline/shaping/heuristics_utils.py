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

"""Heuristic utilities for subdividing and preparing 3D shaping elements."""

import collections
from collections.abc import Sequence
import copy
import dataclasses
import json
import logging
import math
from typing import TypedDict

from .. import timeout as timeout_lib
from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as crease_pattern_lib
from ..packing import packing_output as packing_output_utils
from ..packing import types as packing_types
from ..solving import hinges as hinge_utils
from ..stick_figures import stick_figure as sf_utils
from ..stick_figures import traversal as tr
from . import heuristics
from . import inscribed_rectangle_shaping
from . import shaping_elements
from . import shaping_utils
from . import types as shaping_types
from . import zigzag_shaping
from . import zigzag_shaping_utils

MIN_ANGLE = math.radians(20)
# NOTE: 0.05 added to avoid numerical issues / duplicate creases.
LARGE_ANGLE = math.pi - MIN_ANGLE / 2 + 0.05


def _apply_small_angle_flip_float(target_flap_angle: float) -> Sequence[float]:
  if target_flap_angle < -crease_utils.DEFAULT_TOLERANCE:
    raise ValueError(
        'Target flap angle must be non-negative to apply small angle flip,'
        f' got {target_flap_angle}.'
    )
  if target_flap_angle >= (
      math.pi / 2
  ) - crease_utils.DEFAULT_TOLERANCE or math.isclose(
      target_flap_angle,
      0.0,
      abs_tol=crease_utils.DEFAULT_TOLERANCE,
  ):
    return [target_flap_angle]

  small_angle = max(MIN_ANGLE, target_flap_angle)
  remainder = (2 * math.pi - small_angle) - LARGE_ANGLE
  return [LARGE_ANGLE, remainder]


def _get_shaping_elements_for_unmerged_flaps(
    flaps: list[shaping_elements.ShapingFlap],
    rivers: list[shaping_elements.ShapingRiver],
    outer_bank_indices: dict[str, shaping_types.BankIndex],
    unmerged_flap_info: dict[str, heuristics.Node],
    merged_stick_name_dict: dict[str, list[str]],
) -> tuple[
    list[shaping_elements.ShapingFlap], list[shaping_elements.ShapingRiver]
]:
  """Creates new shaping elements for unmerged flaps and rivers."""
  unmerged_flaps, unmerged_rivers = [], []

  # Step 1: create new shaping elements for unmerged sticks in a flap
  for flap in flaps:
    merged_name = flap.name
    unmerged_names = merged_stick_name_dict.get(merged_name, None)

    # If flap name not found in merge dict, the flap was never merged
    if unmerged_names is None:
      unmerged_flaps.append(flap)
      continue

    logging.debug(
        'Subdividing flap %s with grid radius %s',
        merged_name,
        flap.grid_unit_radius,
    )

    for ind, unmerged_name in enumerate(unmerged_names):
      stick_length = unmerged_flap_info[unmerged_name].length
      logging.debug(
          'Creating unmerged flap %s with length %s',
          unmerged_name,
          stick_length,
      )
      if math.isclose(flap.grid_unit_radius, stick_length):
        unmerged_flap = flap.rename_and_copy(unmerged_name)
        assert (
            ind == len(unmerged_names) - 1
        ), 'Unmerged sticks too long or element too short!'
        unmerged_flaps.append(unmerged_flap)
      else:
        unmerged_flap = flap.rename_and_copy(
            unmerged_name, max_shaping_depth=stick_length
        )
        flap = flap.shrink(stick_length)
        unmerged_flaps.append(unmerged_flap)

  # Step 2: get unmerged rivers
  for river in rivers:
    merged_name = river.name
    unmerged_names = merged_stick_name_dict.get(merged_name, None)

    if unmerged_names is None:
      unmerged_rivers.append(river)
      continue

    logging.debug(
        'Subdividing river %s with grid radius %s',
        merged_name,
        river.width,
    )

    for ind, unmerged_name in enumerate(unmerged_names):
      stick_length = unmerged_flap_info[unmerged_name].length

      logging.debug(
          'Creating unmerged river %s with length %s',
          unmerged_name,
          stick_length,
      )

      if river.width == stick_length:
        unmerged_river = river.rename_and_copy(unmerged_name)
        assert (
            ind == len(unmerged_names) - 1
        ), 'Unmerged sticks too long or element too short!'
      else:
        outer_bank_index = outer_bank_indices[merged_name]
        unmerged_river = river.get_circumbscribed_river(
            stick_length, unmerged_name, outer_bank_index
        )
        river = river.shrink_along_bank(outer_bank_index, stick_length)
      unmerged_rivers.append(unmerged_river)

  return unmerged_flaps, unmerged_rivers


def get_outer_bank_indices(
    stick_figure: sf_utils.StickFigure,
    shaping_flaps: list[shaping_elements.ShapingFlap],
    shaping_rivers: list[shaping_elements.ShapingRiver],
) -> dict[str, shaping_types.BankIndex]:
  """Computes river outer bank indices."""
  outer_bank_indices: dict[str, shaping_types.BankIndex] = {}
  root_name = heuristics.get_first_stick_for_shaping(stick_figure).name
  for river in shaping_rivers:
    try:
      bank_index = river.get_outer_bank_index(
          stick_figure,
          root_name,
          shaping_flaps,
          shaping_rivers,
      )
    except ValueError as e:
      logging.warning(
          'Could not determine outer bank index for river %s: %s',
          river.name,
          e,
      )
      bank_index = river.longer_bank_index
    outer_bank_indices[river.name] = bank_index
  return outer_bank_indices


def _rename_shaping_elements_to_match_merge_converter(
    shaping_flaps: list[shaping_elements.ShapingFlap],
    shaping_rivers: list[shaping_elements.ShapingRiver],
    merge_converter: dict[str, str],
) -> tuple[
    list[shaping_elements.ShapingFlap], list[shaping_elements.ShapingRiver]
]:
  """Renames shaping elements to match `merge_converter`."""
  packing_name_to_tree_name = {v: k for k, v in merge_converter.items()}
  new_shaping_flaps = []
  new_shaping_rivers = []
  for packing_name, tree_name in packing_name_to_tree_name.items():
    shaping_element = shaping_elements.get_element_by_name(
        packing_name, shaping_flaps, shaping_rivers
    )
    if isinstance(shaping_element, shaping_elements.ShapingFlap):
      new_shaping_flaps.append(shaping_element.rename_and_copy(tree_name))
    elif isinstance(shaping_element, shaping_elements.ShapingRiver):
      new_shaping_rivers.append(shaping_element.rename_and_copy(tree_name))
    else:
      raise ValueError(
          f'Unsupported shaping element type: {type(shaping_element)}'
          f' for element {packing_name}.'
      )
  return new_shaping_flaps, new_shaping_rivers


def _order_stick_name_dict_values_by_root_proximity(
    unmerged_stick_figure: sf_utils.StickFigure,
    merged_stick_name_dict: dict[str, list[str]],
) -> dict[str, list[str]]:
  """Orders lists of unmerged sub-sticks by distance from the unmerged root."""
  unmerged_to_merged_dict = {}
  for merged_name, unmerged_names in merged_stick_name_dict.items():
    for unmerged_name in unmerged_names:
      unmerged_to_merged_dict[unmerged_name] = merged_name

  unmerged_root = heuristics.get_first_stick_for_shaping(unmerged_stick_figure)
  marked_merged_stick_names = set()
  for unmerged_stick in tr.breadth_first_traversal(
      unmerged_stick_figure, start_stick_name=unmerged_root.name
  ):
    if unmerged_stick.name in unmerged_to_merged_dict:
      merged_stick_name = unmerged_to_merged_dict[unmerged_stick.name]
      if merged_stick_name not in marked_merged_stick_names:
        unmerged_stick_names = merged_stick_name_dict[merged_stick_name]
        assert unmerged_stick.name == unmerged_stick_names[0] or (
            unmerged_stick.name == unmerged_stick_names[-1]
        ), (
            f'Unmerged stick {unmerged_stick.name} not at expected position'
            f' in unmerged_stick_names {unmerged_stick_names}!'
        )
        if unmerged_stick_names[-1] == unmerged_stick.name:
          merged_stick_name_dict[merged_stick_name] = unmerged_stick_names[::-1]

      marked_merged_stick_names.add(merged_stick_name)

  return merged_stick_name_dict


class ShapingData(TypedDict):
  """Shaping data used for 3D heuristics."""

  shaping_flaps: list[shaping_elements.ShapingFlap]
  shaping_rivers: list[shaping_elements.ShapingRiver]
  flap_info: dict[str, heuristics.Node]
  outer_bank_indices: dict[str, shaping_types.BankIndex]


def get_heuristics_shaping_data(
    stick_figure: sf_utils.StickFigure,
    shaping_flaps: list[shaping_elements.ShapingFlap],
    shaping_rivers: list[shaping_elements.ShapingRiver],
    hinge_info_list: list[packing_types.HingeInfo],
    central_crease_removed: bool,
    *,
    shape_with_unmerged_tree: bool = False,
    merge_converter: dict[str, str] | None = None,
    sacrificial_flaps: Sequence[packing_types.SacrificialFlap] = (),
) -> ShapingData:
  """Computes data necessary for 3D shaping heuristics."""
  unmerged_stick_figure = stick_figure
  stick_figure, merged_stick_name_dict = stick_figure.merge_linear_chains()
  merged_stick_name_dict = _order_stick_name_dict_values_by_root_proximity(
      unmerged_stick_figure, merged_stick_name_dict
  )
  stick_figure = stick_figure.align_root_stick_to_target_azimuth_elevation()
  unmerged_stick_figure = (
      unmerged_stick_figure.align_root_stick_to_target_azimuth_elevation()
  )
  if merge_converter:
    if sacrificial_flaps:
      raise NotImplementedError(
          'Sacrificial flaps are not supported when renaming shaping elements'
          ' to match merge_converter.'
      )
    shaping_flaps, shaping_rivers = (
        _rename_shaping_elements_to_match_merge_converter(
            shaping_flaps, shaping_rivers, merge_converter
        )
    )
    inverted_merge_converter = {v: k for k, v in merge_converter.items()}
    for hinge_info in hinge_info_list:
      element_names = hinge_info.flap_names
      new_names = tuple(
          inverted_merge_converter.get(name, name) for name in element_names
      )
      hinge_info.flap_names = new_names  # pyrefly: ignore[bad-assignment]

  root_stick_direction = (
      heuristics.compute_root_stick_direction_follow_stick_figure_root(
          stick_figure, shaping_flaps, shaping_rivers
      )
  )
  logging.debug('root_stick_direction: %r', root_stick_direction)

  sac_flaps_mapping = heuristics.build_sacrificial_flaps_by_base_flap(
      sacrificial_flaps
  )
  flap_info = heuristics.get_flap_info_dict_with_hinge_info(
      stick_figure,
      hinge_info_list,
      root_stick_direction,
      sac_flaps_mapping,
  )
  unmerged_flap_info = heuristics.get_flap_info_dict_for_unmerged_stick_figure(
      unmerged_stick_figure=unmerged_stick_figure,
      merged_stick_name_dict=merged_stick_name_dict,
      hinge_info_list=hinge_info_list,
      root_direction_follow_stick_figure_root=root_stick_direction,
      node_dict=flap_info,
      sacrificial_flaps_by_base_flap=sac_flaps_mapping,
  )

  outer_bank_indices = get_outer_bank_indices(
      stick_figure, shaping_flaps, shaping_rivers
  )

  if shape_with_unmerged_tree:
    shaping_flaps, shaping_rivers = _get_shaping_elements_for_unmerged_flaps(
        flaps=shaping_flaps,
        rivers=shaping_rivers,
        outer_bank_indices=outer_bank_indices,
        unmerged_flap_info=unmerged_flap_info,
        merged_stick_name_dict=merged_stick_name_dict,
    )
    flap_info = unmerged_flap_info
    outer_bank_indices = get_outer_bank_indices(
        unmerged_stick_figure, shaping_flaps, shaping_rivers
    )

    logging.debug('New flaps: %r', shaping_flaps)
    logging.debug('New rivers: %r', shaping_rivers)

  shaping_flaps, shaping_rivers = (
      shaping_elements.adjust_shaping_elements_for_central_crease_removal(
          shaping_flaps,
          shaping_rivers,
          outer_bank_indices,
          central_crease_removed,
      )
  )

  return ShapingData(
      shaping_flaps=shaping_flaps,
      shaping_rivers=shaping_rivers,
      flap_info=flap_info,
      outer_bank_indices=outer_bank_indices,
  )


def _apply_small_angle_flip_to_element(
    zigzag_params: zigzag_shaping.ZigzagShapingParamsForElement,
    inscribe_by_epsilon: bool = True,
) -> Sequence[zigzag_shaping.ZigzagShapingParamsForElement]:
  """Flips small folds to two large folds.

  Also optionally inscribes the folds by a small depth to avoid flat-foldability
  violations.

  Args:
    zigzag_params: zigzag shaping parameters for the element.
    inscribe_by_epsilon: Whether to inscribe the folds by a small depth.

  Returns:
    A list of zigzag shaping parameters for the element.
  """
  if zigzag_params.zigzag_params.target_flap_angle is None:
    raise ValueError(
        'Zigzag params must have a target flap angle to apply small angle flip.'
    )
  elif (
      zigzag_params.zigzag_params.target_flap_angle
      < -crease_utils.DEFAULT_TOLERANCE
  ):
    raise ValueError(
        'Zigzag params must have a positive target flap angle to apply small'
        ' angle flip.'
    )
  # Check if target angle is greater than pi / 2 or close to 0.
  elif (
      zigzag_params.zigzag_params.target_flap_angle
      >= (math.pi / 2) - crease_utils.DEFAULT_TOLERANCE
  ) or (
      math.isclose(
          zigzag_params.zigzag_params.target_flap_angle,
          0,
          abs_tol=crease_utils.DEFAULT_TOLERANCE,
      )
  ):
    return [zigzag_params]
  # Target angle is not None, positive, and less than pi/2.
  else:
    small_angle = zigzag_params.zigzag_params.target_flap_angle
    # Up-clip the small angle to avoid numerical issues.
    # NOTE: we up-clip to a little more than the minimum angle
    # to avoid duplicate creases.
    small_angle = zigzag_shaping_utils.clip_angle_by_min_angle(
        small_angle
    )
    remainder = (2 * math.pi - small_angle) - LARGE_ANGLE
    requests = []
    # Generate a large angle request with the same phase as the small angle.
    requests.append(
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name=zigzag_params.element_name,
            zigzag_params=dataclasses.replace(
                zigzag_params.zigzag_params,
                target_flap_angle=LARGE_ANGLE,
                even_phase=zigzag_params.zigzag_params.even_phase,
                depth=zigzag_params.zigzag_params.depth
                + zigzag_shaping_utils.HINGE_FF_SHIFT_DEPTH
                * inscribe_by_epsilon,
            ),
        )
    )
    # Generate a (slightly smaller) large angle request again
    # with the same phase but also opposite fold order.
    requests.append(
        zigzag_shaping.ZigzagShapingParamsForElement(
            element_name=zigzag_params.element_name,
            # Flip the fold order and add pi to the angle to get the correct
            # angle for the large fold.
            zigzag_params=dataclasses.replace(
                zigzag_params.zigzag_params,
                target_flap_angle=remainder,
                even_phase=zigzag_params.zigzag_params.even_phase,
                depth=zigzag_params.zigzag_params.depth
                + zigzag_shaping_utils.HINGE_FF_SHIFT_DEPTH
                * inscribe_by_epsilon,
                fold_order=_flip_fold_order(
                    list(zigzag_params.zigzag_params.fold_order)
                ),
            ),
        )
    )
    return requests


def apply_small_angle_flip(
    target_or_params: float | zigzag_shaping.ZigzagShapingParamsForElement,
    inscribe_by_epsilon: bool = True,
) -> Sequence[float] | Sequence[zigzag_shaping.ZigzagShapingParamsForElement]:
  """Splits an acute positive flap angle or zigzag parameter into shallow folds."""
  if isinstance(target_or_params, (int, float)):
    return _apply_small_angle_flip_float(float(target_or_params))
  return _apply_small_angle_flip_to_element(
      target_or_params, inscribe_by_epsilon=inscribe_by_epsilon
  )


def _flip_fold_order(
    fold_order: list[crease_utils.Fold],
) -> list[crease_utils.Fold]:
  """Flips the fold order."""
  assert len(fold_order) == 2, 'Fold order must have length 2 to be flipped.'
  return fold_order[::-1]


def apply_inscribe_trick(
    zigzag_params: zigzag_shaping.ZigzagShapingParamsForElement,
) -> zigzag_shaping.ZigzagShapingParamsForElement:
  """Inscribes shaping by a small depth to avoid flat-foldability violations.

  Args:
    zigzag_params: zigzag shaping parameters for the element.

  Returns:
    The updated zigzag shaping parameters for the element.
  """
  return dataclasses.replace(
      zigzag_params,
      zigzag_params=dataclasses.replace(
          zigzag_params.zigzag_params,
          depth=zigzag_params.zigzag_params.depth
          + zigzag_shaping_utils.HINGE_FF_SHIFT_DEPTH,
      ),
  )


def _get_default_inscribed_polygon_fold_order(
    shape_angle_elevation: float,
) -> list[crease_utils.Fold]:
  """Returns the default fold order for inscribed polygon shaping.

  Args:
    shape_angle_elevation: Quantitiy derived from an element's relative
      elevation angle with its parent determining how the element folds in 3d.

  Returns:
    M/V fold order if shape_angle_elevation is positive, V/M otherwise.
  """
  if shape_angle_elevation > 0:
    return [
        crease_utils.Fold.MOUNTAIN,
        crease_utils.Fold.VALLEY,
    ]
  else:
    return [
        crease_utils.Fold.VALLEY,
        crease_utils.Fold.MOUNTAIN,
    ]


def _get_inscribed_polygon_fold_order_for_central_flap(
    flap: shaping_elements.ShapingFlap,
    shape_angle_elevation: float,
    central_crease_removed: bool,
) -> list[crease_utils.Fold]:
  """Returns the fold order for inscribed polygon shaping for the central flap.

  First estimate fold order using shape_angle_elevation. Then test whether the
  fold order needs to be flipped based on the central crease position if the
  flap is central and the central crease is removed.

  Args:
    flap: The central flap.
    shape_angle_elevation: Quantitiy derived from an element's relative
      elevation angle with its parent determining how the element folds in 3d.
    central_crease_removed: Whether the central crease is removed.

  Returns:
    The fold order for inscribed polygon shaping for the central flap.
  """
  if not (central_crease_removed and flap.contains_central_crease):
    raise NotImplementedError(
        'Inscribed polygon shaping for the central flap is not implemented for'
        ' when the central crease is not removed.'
    )
  # Want crease crossing center to be M or V depending on shape_angle
  center_crease_target = _get_default_inscribed_polygon_fold_order(
      shape_angle_elevation
  )
  # Top-left corner
  start_point = flap.corners[0]
  grid_unit_central_dist = (
      shaping_utils.get_integer_grid_unit_length(
          shaping_elements.CENTRAL_CREASE_X - start_point.x, flap.grid_size
      )
      - 1
  )  # -1 to account for the deleted central crease
  if grid_unit_central_dist % 2 == 0:
    return center_crease_target
  else:
    return _flip_fold_order(center_crease_target)


def get_inscribed_polygon_fold_order(
    element: shaping_elements.ShapingElement,
    shape_angle_elevation: float,
) -> list[crease_utils.Fold]:
  """Returns the fold order for inscribed polygon shaping.

  Args:
    element: The element (flap or river) to be shaped.
    shape_angle_elevation: The shape angle elevation of the node.

  Returns:
    The fold order for inscribed polygon shaping.
  """
  if isinstance(element, shaping_elements.ShapingFlap):
    return _get_default_inscribed_polygon_fold_order(shape_angle_elevation)
  else:
    return _get_default_inscribed_polygon_fold_order(shape_angle_elevation)


def _get_default_inscribed_params_for_element(
    element: shaping_elements.ShapingElement,
) -> inscribed_rectangle_shaping.InscribedShapingParamsForElement:
  """Returns the default inscribed polygon parameters for an element."""
  return inscribed_rectangle_shaping.InscribedShapingParamsForElement(
      element_name=element.name,
      inscribed_params=[],
  )


def _get_element_inscribed_params_from_heuristics(
    element: shaping_elements.ShapingElement,
    outer_bank_indices: dict[str, shaping_types.BankIndex],
    flap_info: dict[str, heuristics.Node],
    *,
    apply_180_azimuthal_rotation: bool,
    shape_3d: bool,
) -> inscribed_rectangle_shaping.InscribedShapingParamsForElement:
  """Core logic determining the inscribed polygon parameters for an element.

  Fold all the way (fold_percent = 1) if apply_180_azimuthal_rotation is True.
  Otherwise, fold to the shape angle elevation if shape_3d is True.
  Compute the min and max depth for the shaping operation via
  the element's hinge assignment and radius/width.

  Args:
    element: The element (flap or river) to be shaped.
    outer_bank_indices: The dictionary of outer bank indices for rivers.
    flap_info: The dictionary of all flaps.
    apply_180_azimuthal_rotation: Whether to apply 180 azimuthal rotation.
    shape_3d: Whether to shape 3D.

  Returns:
    The parameters for inscribed rectangle shaping.
  """
  node_name = element.name
  node = flap_info[node_name]

  # Skip shaping if the node is a root node or if it is a node that does not
  # need to be shaped (shape_angle_elevation is 0)
  if node.is_root:
    return _get_default_inscribed_params_for_element(element)

  fold_percent = 1.0 if apply_180_azimuthal_rotation else 0.0

  # Set fold percent and fold order using shape_angle_elevation for 3D shaping.
  if shape_3d:
    fold_percent = abs(fold_percent - abs(node.shape_angle_elevation) / math.pi)

  fold_order = get_inscribed_polygon_fold_order(
      element, node.shape_angle_elevation
  )

  # Ensure no overlap with assigned hinge
  min_depth = int(node.has_assigned_hinge)
  # Ensure depth is not larger than the width of the element
  # and does not overlap with a hinge
  if element.max_shaping_depth is not None:
    max_depth = element.max_shaping_depth - 1
  else:
    max_depth = (
        element.grid_unit_radius - 1
        if isinstance(element, shaping_elements.ShapingFlap)
        else element.width - 1
    )
  if max_depth < min_depth:
    logging.warning(
        'Max depth %s is smaller than min depth %s for node %s, set min'
        ' depth to 1 though it will overlap with the zigzag creases.',
        max_depth,
        min_depth,
        node_name,
    )
  depth = min(min_depth, max_depth)

  if isinstance(element, shaping_elements.ShapingFlap):
    inscribed_params = [
        inscribed_rectangle_shaping.InscribedRectangleParams(
            depth=depth,
            fold_order=fold_order,
            fold_percent=fold_percent,
        )
    ]
  else:
    outer_bank_index = outer_bank_indices[node_name]
    inscribed_params = [
        inscribed_rectangle_shaping.InscribedRiverParams(
            depth=depth,
            fold_order=fold_order,
            fold_percent=fold_percent,
            bank_index=outer_bank_index,
        )
    ]

  return inscribed_rectangle_shaping.InscribedShapingParamsForElement(
      element_name=node_name,
      inscribed_params=inscribed_params,
  )


def _get_element_simple_fold_params_from_heuristics(
    grid_size: int,
    element: shaping_elements.ShapingElement,
    outer_bank_indices: dict[str, shaping_types.BankIndex],
    flap_info: dict[str, heuristics.Node],
    hinge_segments: list[crease_utils.Crease],
    *,
    central_crease_removed: bool,
    elements: list[shaping_elements.ShapingElement],
    apply_flip_to_acute_angles: bool,
    resolve_hinge_ff_with_shifts: bool,
    shape_3d: bool,
) -> tuple[
    zigzag_shaping.ZigzagShapingParamsForElement,
    inscribed_rectangle_shaping.InscribedShapingParamsForElement,
]:
  """Calculates the parameters for simple fold shaping for a single element (flap or river).

  First decide to apply 180 azimuthal rotation to the element if shape angle
  is close to 180 degrees.
  Then compute the parameters for inscribed element shaping.
  Finally, compute the parameters for zigzag shaping using the depth
  return by the inscribed element shaping function.

  Args:
    grid_size: The grid size of the box-pleating grid.
    element: The element (flap or river) to be shaped.
    outer_bank_indices: The dictionary of outer bank indices for rivers.
    flap_info: The dictionary of all flaps.
    hinge_segments: The hinge segments of the base crease pattern.
    central_crease_removed: Whether the central crease is removed.
    elements: The list of all elements (flaps and rivers).
    apply_flip_to_acute_angles: Whether to apply flip to acute zigzag angles
      (i.e., angles smaller than pi/2).
    resolve_hinge_ff_with_shifts: Whether to resolve hinge ff violations
      by inscribing zigzagas by a small depth.
    shape_3d: Whether to shape 3D.

  Returns:
    A tuple of the zigzag parameters and the inscribed polygon parameters.
  """
  logging.debug('Getting simple fold params for element %s', element.name)
  logging.debug(
      'Element info: %s',
      json.dumps(flap_info[element.name].to_serializable_dict(), indent=2),
  )

  rect_node = flap_info[element.name]
  shape_angle = rect_node.shape_angle

  # First decide to apply an inscribed element to rotate azimuth by 180 if
  # 1. using the flip trick (to convert acute angles to larger angles)
  # 2. abs(shape_angle) is 180
  apply_180_azimuthal_rotation = False
  if shape_angle is not None and (
      (
          apply_flip_to_acute_angles
          and (zigzag_shaping_utils.MIN_ANGLE < abs(shape_angle) < math.pi / 2)
      )
      or math.isclose(abs(shape_angle), math.pi, abs_tol=1e-3)
  ):
    rect_node.fold_state = heuristics.FoldState.FOLD_IN
    apply_180_azimuthal_rotation = True

  # Now do inscribed element shaping
  inscribed_params = _get_element_inscribed_params_from_heuristics(
      element,
      outer_bank_indices,
      flap_info,
      apply_180_azimuthal_rotation=apply_180_azimuthal_rotation,
      shape_3d=shape_3d,
  )

  # Append non-trivial shaping to element's shaping history
  if inscribed_params.inscribed_params:
    for param in inscribed_params.inscribed_params:
      if param.fold_percent > 0:
        element.shaping_history.append(param)

  # Now apply zigzag shaping conditioned on results of inscribed element shaping
  if rect_node.shape_angle is None or math.isclose(
      abs(rect_node.shape_angle), math.pi
  ):
    # If shape_angle is None or 0, return trivial params
    zigzag_params = zigzag_shaping_utils.get_default_zigzag_params_for_element(
        element
    )
  else:
    # Compute inscribed element's depth to ensure zigzags
    # are contained within the inscribed element
    inscribed_params_list = inscribed_params.inscribed_params
    # Case 1: inscribed params are trivial
    if not inscribed_params_list or math.isclose(
        inscribed_params_list[0].fold_percent, 0
    ):
      inscribe_depth = 0
    # Case 2: inscribed params are non-trivial
    else:
      inscribe_depth = inscribed_params_list[0].depth
    zigzag_params = (
        zigzag_shaping_utils.get_element_zigzag_params_from_heuristics(
            grid_size,
            element,
            outer_bank_indices,
            flap_info,
            hinge_segments,
            central_crease_removed,
            elements,
            inscribe_depth,  # Zigzags be contained by any inscribed rectangles
            resolve_hinge_ff_with_shifts,
        )
    )
  # Append non-trivial shaping to element's shaping history
  target_flap_angle = zigzag_params.zigzag_params.target_flap_angle
  if (target_flap_angle is not None and target_flap_angle > 0):
    element.shaping_history.append(zigzag_params.zigzag_params)

  logging.debug(
      'Simple fold params for element %s: %r \n %r',
      element.name,
      zigzag_params,
      inscribed_params,
  )

  return zigzag_params, inscribed_params


def get_all_simple_fold_params_from_heuristics(
    *,
    flap_info: dict[str, heuristics.Node],
    rectangles: list[shaping_elements.ShapingFlap],
    rivers: list[shaping_elements.ShapingRiver],
    outer_bank_indices: dict[str, shaping_types.BankIndex],
    grid_size: int,
    hinge_segments: list[crease_utils.Crease],
    apply_flip_to_acute_angles: bool,
    resolve_hinge_ff_with_shifts: bool,
    shape_rivers: bool,
    shape_3d: bool,
    central_crease_removed: bool,
) -> tuple[
    list[zigzag_shaping.ZigzagShapingParamsForElement],
    list[inscribed_rectangle_shaping.InscribedShapingParamsForElement],
]:
  """Calculates the inscribed polygon parameters using a heuristic approach.

  Args:
    flap_info: The flap info dictionary from the heuristics.
    rectangles: The rectangles to be shaped.
    rivers: The rivers to be shaped.
    outer_bank_indices: The dictionary of outer bank indices for rivers.
    grid_size: The grid size of the box-pleating grid.
    hinge_segments: The hinge segments of the base crease pattern.
    apply_flip_to_acute_angles: Whether to apply flip to acute zigzag angles
      (i.e., angles smaller than pi/2).
    resolve_hinge_ff_with_shifts: Whether to resolve hinge ff violations
      by inscribing zigzagas by a small depth.
    shape_rivers: Whether to shape rivers.
    shape_3d: Whether to shape 3D.
    central_crease_removed: Whether the central crease is removed.

  Returns:
    A tuple of the zigzag parameters, the inscribed polygon parameters, and
    a dictionary mapping flap names to whether the first side is up.
  """
  # Deepcopy flap_info, rectangles, and rivers to avoid modifying the original
  # data.
  flap_info = copy.deepcopy(flap_info)
  rectangles = copy.deepcopy(rectangles)
  rivers = copy.deepcopy(rivers)

  # List of flaps that need to be shaped by pi radian angle.
  all_zigzag_params = []
  all_inscribed_params = []

  # Construct element dictionary.
  elements = rectangles + rivers
  elements_dict = {element.name: element for element in elements}

  # Process elements (flaps and rivers) in BFS order from tree root,
  # since shaping operations on parent elements may affect
  # the phase prediction of child elements.
  root = [
      element_node
      for element_node in flap_info.values()
      if element_node.is_root
  ]
  assert len(root) == 1, 'There should be exactly one root element.'
  root = root[0]
  element_names = collections.deque([root.name])
  while element_names:
    timeout_lib.maybe_timeout_cooperatively()
    element_name = element_names.popleft()
    element = elements_dict.get(element_name, None)
    # continue if element is not present in element list
    # NOTE: this means to shape an element
    # we must also pass in all of its parents as elements
    if element is None:
      continue

    # Add child elements to queue for processing.
    element_names.extend(flap_info[element.name].children)

    if element.elevations - {0, 1}:
      logging.info(
          'Skipping element with elevations other than 0 and 1: %s',
          element.name,
      )
      continue

    # Get shaping parameters for flaps.
    zigzag_params, inscribed_params = (
        _get_element_simple_fold_params_from_heuristics(
            grid_size,
            element,
            outer_bank_indices,
            flap_info,
            hinge_segments,
            central_crease_removed=central_crease_removed,
            elements=elements,
            apply_flip_to_acute_angles=apply_flip_to_acute_angles,
            resolve_hinge_ff_with_shifts=resolve_hinge_ff_with_shifts,
            shape_3d=shape_3d,
        )
    )

    # Filter out zigzag params with:
    # 1. target_angle = 0
    # 2. shaped rivers (if shape_rivers is False)
    target_flap_angle = zigzag_params.zigzag_params.target_flap_angle
    if (
        target_flap_angle is not None
        and target_flap_angle > 0
        and (
            not isinstance(element, shaping_elements.ShapingRiver)
            or shape_rivers
        )
    ):
      all_zigzag_params.append(zigzag_params)

    # Filter out inscribed params with:
    # 1. empty inscribed_params lists
    if inscribed_params.inscribed_params and any(
        [param.fold_percent > 0 for param in inscribed_params.inscribed_params]
    ):
      all_inscribed_params.append(inscribed_params)

  return all_zigzag_params, all_inscribed_params


def get_simple_fold_shaping_elements_and_params_with_heuristics(
    stick_figure: sf_utils.StickFigure,
    grid_size: int,
    shaping_flaps: list[shaping_elements.ShapingFlap],
    shaping_rivers: list[shaping_elements.ShapingRiver],
    hinge_info_list: list[packing_types.HingeInfo],
    central_crease_removed: bool,
    *,
    apply_flip_to_acute_angles: bool,
    shape_rivers: bool,
    shape_with_unmerged_tree: bool,
    resolve_hinge_ff_with_shifts: bool,
    merge_converter: dict[str, str] | None,
    element_name_list: Sequence[str] | None = None,
    sacrificial_flaps: Sequence[packing_types.SacrificialFlap] = (),
) -> tuple[
    list[zigzag_shaping.ZigzagShapingParamsForElement],
    list[inscribed_rectangle_shaping.InscribedShapingParamsForElement],
    list[shaping_elements.ShapingFlap],
    list[shaping_elements.ShapingRiver],
]:
  """Shapes a crease pattern with stick figure heuristics.

  First computes flap info dicts determining element shaping angles for the
  merged and unmerged stick figures. Then computes the zigzag and inscribed
  polygon parameters from the flap info dicts with heuristics. Shaping elements
  are possibly adjusted for central crease removal and the unmerged tree.

  Args:
    stick_figure: The stick figure from which to infer the shaping parameters.
    grid_size: The grid size of the box-pleating grid.
    shaping_flaps: The list of all (merged) flaps.
    shaping_rivers: The list of all (merged) rivers.
    hinge_info_list: The list of all hinge info.
    central_crease_removed: Whether the central crease is removed.
    apply_flip_to_acute_angles: Whether to flip the flaps with acute zigzag
      angles.
    shape_rivers: Whether to shape the rivers.
    shape_with_unmerged_tree: Whether to shape with unmerged tree.
    resolve_hinge_ff_with_shifts: Whether to fix hinge ff violations by
      inscribing the simple fold by a small depth.
    merge_converter: Mapping from merged stick names in the tree to flap/river
      names in the packing.
    element_name_list: The list of elements to shape heuristically. If None, all
      elements will be shaped.
    sacrificial_flaps: The list of all sacrificial flaps.

  Returns:
    A tuple of the zigzag parameters, inscribed polygon parameters,
    shaping flaps and shaping rivers.
  """
  # Compute heuristics data necessary for shaping heuristics.
  shaping_data = (
      get_heuristics_shaping_data(
          stick_figure=stick_figure,
          shaping_flaps=shaping_flaps,
          shaping_rivers=shaping_rivers,
          hinge_info_list=hinge_info_list,
          central_crease_removed=central_crease_removed,
          shape_with_unmerged_tree=shape_with_unmerged_tree,
          merge_converter=merge_converter,
          sacrificial_flaps=sacrificial_flaps,
      )
  )
  shaping_flaps = shaping_data['shaping_flaps']
  shaping_rivers = shaping_data['shaping_rivers']
  flap_info = shaping_data['flap_info']
  outer_bank_indices = shaping_data['outer_bank_indices']

  # Compute zigzag and inscribed polygon parameters with heuristics data.
  hinge_segments = [hinge.coordinate for hinge in hinge_info_list]
  zigzag_params, inscribed_params = get_all_simple_fold_params_from_heuristics(
      flap_info=flap_info,
      rectangles=shaping_flaps,
      outer_bank_indices=outer_bank_indices,
      rivers=shaping_rivers,
      grid_size=grid_size,
      hinge_segments=hinge_segments,
      apply_flip_to_acute_angles=apply_flip_to_acute_angles,
      resolve_hinge_ff_with_shifts=resolve_hinge_ff_with_shifts,
      central_crease_removed=central_crease_removed,
      shape_rivers=shape_rivers,
      shape_3d=False,
  )

  # Filter out params for elements not in element_name_list (if provided).
  if element_name_list is not None:
    zigzag_params = [
        param
        for param in zigzag_params
        if param.element_name in element_name_list
    ]
    inscribed_params = [
        param
        for param in inscribed_params
        if param.element_name in element_name_list
    ]

  logging.debug('zigzag_params: %r', zigzag_params)
  logging.debug('inscribed_params: %r', inscribed_params)

  return zigzag_params, inscribed_params, shaping_flaps, shaping_rivers


def get_simple_fold_shaping_elements_and_params_with_heuristics_pipeline_wrapper(
    stick_figure: sf_utils.StickFigure,
    packing_output: packing_output_utils.PackingOutput,
    solving_output: hinge_utils.SolverResult,
    solving_grid_size: int,
    central_crease_removed: bool,
    shaping_flaps: list[shaping_elements.ShapingFlap],
    shaping_rivers: list[shaping_elements.ShapingRiver],
    *,
    apply_flip_to_acute_angles: bool = False,
    shape_rivers: bool = False,
    shape_with_unmerged_tree: bool = False,
    resolve_hinge_ff_with_shifts: bool = False,
    merge_converter: dict[str, str] | None = None,
    element_name_list: Sequence[str] | None = None,
) -> tuple[
    list[zigzag_shaping.ZigzagShapingParamsForElement],
    list[inscribed_rectangle_shaping.InscribedShapingParamsForElement],
    list[shaping_elements.ShapingFlap],
    list[shaping_elements.ShapingRiver],
]:
  """Pipeline wrapper around get_simple_fold_shaping_elements_and_params_with_heuristics."""
  # Get assigned hinge information.
  hinge_info_list = copy.deepcopy(packing_output.hinges)
  hinge_info_list = heuristics.update_hinge_info_from_solver_result(
      hinge_info_list, solving_output
  )

  # Get shaping elements and shaping parameters.
  return get_simple_fold_shaping_elements_and_params_with_heuristics(
      stick_figure=stick_figure,
      grid_size=solving_grid_size,
      shaping_flaps=shaping_flaps,
      shaping_rivers=shaping_rivers,
      hinge_info_list=hinge_info_list,
      central_crease_removed=central_crease_removed,
      apply_flip_to_acute_angles=apply_flip_to_acute_angles,
      shape_rivers=shape_rivers,
      shape_with_unmerged_tree=shape_with_unmerged_tree,
      resolve_hinge_ff_with_shifts=resolve_hinge_ff_with_shifts,
      merge_converter=merge_converter,
      element_name_list=element_name_list,
      sacrificial_flaps=packing_output.sacrificial_flaps or (),
  )


def shape_cp_with_heuristics(
    cp: crease_pattern_lib.CreasePattern,
    stick_figure: sf_utils.StickFigure,
    grid_size: int,
    shaping_flaps: list[shaping_elements.ShapingFlap],
    shaping_rivers: list[shaping_elements.ShapingRiver],
    hinge_info_list: list[packing_types.HingeInfo],
    central_crease_removed: bool,
    *,
    apply_flip_to_acute_angles: bool = False,
    shape_rivers: bool = False,
    shape_with_unmerged_tree: bool = False,
    resolve_hinge_ff_with_shifts: bool = False,
    merge_converter: dict[str, str] | None = None,
    narrow: bool = False,
    element_name_list: Sequence[str] | None = None,
) -> crease_pattern_lib.CreasePattern:
  """Shapes a crease pattern with stick figure heuristics.

  First calls `get_simple_fold_shaping_elements_and_params_with_heuristics` to
  get the zigzag and inscribed polygon parameters and the shaping elements. Then
  calls the zigzag and inscribed polygon shaping functions to shape the crease
  pattern.

  Args:
    cp: The crease pattern to be shaped.
    stick_figure: The stick figure from which to infer the shaping parameters.
    grid_size: The grid size of the box-pleating grid.
    shaping_flaps: The list of all flaps.
    shaping_rivers: The list of all rivers.
    hinge_info_list: The list of all hinge info.
    central_crease_removed: Whether the central crease is removed.
    apply_flip_to_acute_angles: Whether to flip the flaps with acute zigzag
      angles.
    shape_rivers: Whether to shape the rivers.
    shape_with_unmerged_tree: Whether to shape with unmerged tree.
    resolve_hinge_ff_with_shifts: Whether to fix hinge ff violations by
      inscribing the simple fold by a small depth.
    merge_converter: Mapping from merged stick names in the tree to flap/river
      names in the packing.
    narrow: Whether to narrow the crease pattern.
    element_name_list: The list of elements to shape heuristically. If None, all
      elements will be shaped.

  Returns:
    The shaped crease pattern.
  """

  # Step 1: Get shaping elements and shaping parameters.
  zigzag_params, inscribed_params, shaping_rectangles, shaping_rivers = (
      get_simple_fold_shaping_elements_and_params_with_heuristics(
          stick_figure=stick_figure,
          grid_size=grid_size,
          shaping_flaps=shaping_flaps,
          shaping_rivers=shaping_rivers,
          hinge_info_list=hinge_info_list,
          central_crease_removed=central_crease_removed,
          apply_flip_to_acute_angles=apply_flip_to_acute_angles,
          shape_rivers=shape_rivers,
          shape_with_unmerged_tree=shape_with_unmerged_tree,
          resolve_hinge_ff_with_shifts=resolve_hinge_ff_with_shifts,
          merge_converter=merge_converter,
          element_name_list=element_name_list,
      )
  )

  # Step 2: Shape the crease pattern.
  orig_num_creases = len(cp.creases)
  if zigzag_params:
    cp = zigzag_shaping.shape_cp_with_zigzags_pipeline_wrapper(
        cp=cp,
        grid_size=grid_size,
        rectangles=shaping_rectangles,
        rivers=shaping_rivers,
        element_shaping_params=zigzag_params,
        narrow=narrow,
    )
  if inscribed_params:
    cp = inscribed_rectangle_shaping.shape_cp_with_inscribed_elements(
        cp=cp,
        grid_size=grid_size,
        rectangles=shaping_rectangles,
        rivers=shaping_rivers,
        element_shaping_params=inscribed_params,
        central_crease_removed=central_crease_removed,
    )

  if len(cp.creases) == orig_num_creases:
    logging.warning(
        'No zigzag or inscribed polygon shaping creases added with heuristics!'
    )

  return cp


def shape_cp_with_heuristics_pipeline_wrapper(
    cp: crease_pattern_lib.CreasePattern,
    stick_figure: sf_utils.StickFigure,
    packing_output: packing_output_utils.PackingOutput,
    solving_output: hinge_utils.SolverResult,
    solving_grid_size: int,
    central_crease_removed: bool,
    shaping_flaps: list[shaping_elements.ShapingFlap],
    shaping_rivers: list[shaping_elements.ShapingRiver],
    *,
    apply_flip_to_acute_angles: bool = False,
    shape_rivers: bool = False,
    shape_with_unmerged_tree: bool = False,
    resolve_hinge_ff_with_shifts: bool = False,
    merge_converter: dict[str, str] | None = None,
    narrow: bool = False,
    element_name_list: Sequence[str] | None = None,
    raise_error_on_ff_violation: bool = False,
) -> crease_pattern_lib.CreasePattern:
  """Shapes a crease pattern with stick figure heuristics from packing and solving output."""
  # Step 1: Get shaping elements and shaping parameters.
  zigzag_params, inscribed_params, shaping_rectangles, shaping_rivers = (
      get_simple_fold_shaping_elements_and_params_with_heuristics_pipeline_wrapper(
          stick_figure=stick_figure,
          packing_output=packing_output,
          solving_output=solving_output,
          solving_grid_size=solving_grid_size,
          central_crease_removed=central_crease_removed,
          shaping_flaps=shaping_flaps,
          shaping_rivers=shaping_rivers,
          apply_flip_to_acute_angles=apply_flip_to_acute_angles,
          shape_rivers=shape_rivers,
          shape_with_unmerged_tree=shape_with_unmerged_tree,
          resolve_hinge_ff_with_shifts=resolve_hinge_ff_with_shifts,
          merge_converter=merge_converter,
          element_name_list=element_name_list,
      )
  )

  # Step 2: Shape the crease pattern.
  orig_num_creases = len(cp.creases)
  if zigzag_params:
    cp = zigzag_shaping.shape_cp_with_zigzags_pipeline_wrapper(
        cp=cp,
        grid_size=solving_grid_size,
        rectangles=shaping_rectangles,
        rivers=shaping_rivers,
        element_shaping_params=zigzag_params,
        narrow=narrow,
        raise_error_on_ff_violation=raise_error_on_ff_violation,
    )
  if inscribed_params:
    cp = inscribed_rectangle_shaping.shape_cp_with_inscribed_elements(
        cp=cp,
        grid_size=solving_grid_size,
        rectangles=shaping_rectangles,
        rivers=shaping_rivers,
        element_shaping_params=inscribed_params,
        central_crease_removed=central_crease_removed,
        raise_error_on_ff_violation=raise_error_on_ff_violation,
    )

  if len(cp.creases) == orig_num_creases:
    logging.warning(
        'No zigzag or inscribed polygon shaping creases added with heuristics!'
    )

  return cp
