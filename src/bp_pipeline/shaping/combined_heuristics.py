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

"""Shaping heuristics combining geometric folding and narrowing."""

import dataclasses
import itertools
import logging
import math
from typing import Sequence

from ..crease_pattern import crease as crease_utils
from ..crease_pattern import crease_pattern as cp_utils
from ..packing import packing_output as packing_output_utils
from ..packing import types as packing_types
from . import combined_heuristics_utils
from . import geometric_fold_3d_heuristics
from . import geometric_fold_3d_heuristics_utils
from . import geometric_simple_fold
from . import heuristics
from . import heuristics_3d
from . import heuristics_utils
from . import narrow_flap_shaping
from . import shaping_elements
from . import shaping_utils
from . import simple_fold_shaping
from ..solving import hinges as hinge_utils
from ..stick_figures import stick_figure as sf_utils


@dataclasses.dataclass(frozen=True)
class CombinedHeuristicsHParams:
  """Parameters for combined heuristics shaping.

  Attributes:
    geometric_fold_hparams: Parameters for the underlying 3D geometric folding.
    use_narrowing: Whether to apply narrowing to thin flaps.
  """

  geometric_fold_hparams: (
      geometric_fold_3d_heuristics.GeometricFolding3DHParams
  ) = geometric_fold_3d_heuristics.GeometricFolding3DHParams()
  use_narrowing: bool = True


def get_narrowed_zigzag_creases(
    shaping_results: Sequence[
        geometric_fold_3d_heuristics.ElementShapingResult
    ],
    central_crease_removed: bool,
) -> Sequence[crease_utils.Crease]:
  """Returns the narrowed zigzag creases for the narrowed flaps."""
  all_zigzag_params = []
  for result in shaping_results:
    # First get corresponding zigzag_shaping.py args.
    # Set inscribe_depth to 0 since result['shaping_element']
    # has already been inscribed by result['simple_fold_info']
    # w.r.t the original flap.
    simple_fold_info = dataclasses.replace(
        result['simple_fold_info'],
        inscribe_depth=0,
    )
    zigzag_params = (
        geometric_fold_3d_heuristics.get_zigzag_shaping_params_from_3d_creases(
            element=result['shaping_element'],
            simple_fold_info=simple_fold_info,
            shaping_creases_3d=result['shaping_creases'],
            central_crease_removed=central_crease_removed,
        )
    )
    if not zigzag_params:
      logging.warning(
          'Could not get zigzag params for element %s',
          result['shaping_element'].name,
      )
      continue
    all_zigzag_params.append(zigzag_params)

  # Double grid sizes and draw zigzags.
  flaps = [
      result['shaping_element']
      for result in shaping_results
      if isinstance(result['shaping_element'], shaping_elements.ShapingFlap)
  ]
  narrowed_zigzag_creases = []
  for zigzag_params in all_zigzag_params:
    # For each narrowed flap, get narrowed zigzag creases deeper in the flap.
    # Try to get narrowed zigzag creases using simple_fold_shaping.py.
    # If it fails, remove the fold entirely and continue to the next flap.
    try:
      creases = simple_fold_shaping.get_simple_fold_shaping_creases(
          grid_size=2 * flaps[0].grid_size,
          flaps=flaps,
          rivers=[],
          element_shaping_params=[zigzag_params],
          central_crease_removed=central_crease_removed,
      )
      narrowed_zigzag_creases.extend(creases)
    except AssertionError as e:
      logging.warning('Failed to get narrowed zigzag creases: %s', e)
      continue

  return narrowed_zigzag_creases


def _setup_shaping_elements(
    *,
    shaping_flaps: Sequence[shaping_elements.ShapingFlap] | None,
    shaping_rivers: Sequence[shaping_elements.ShapingRiver] | None,
    packing_output: packing_output_utils.PackingOutput | None,
    base_cp: cp_utils.CreasePattern,
    solving_grid_size: int,
    hinge_info_list: Sequence[packing_types.HingeInfo] | None,
    solving_output: hinge_utils.SolverResult | None,
) -> tuple[
    Sequence[shaping_elements.ShapingFlap],
    Sequence[shaping_elements.ShapingRiver],
    Sequence[packing_types.HingeInfo],
]:
  """Sets up shaping elements and hinge info."""
  if shaping_flaps is None or shaping_rivers is None:
    if packing_output is None:
      raise ValueError(
          'Must provide packing_output if shaping_flaps/rivers are None.'
      )
    packing_cp = packing_output.packing_cp
    assert packing_cp is not None, 'Packing CP is None.'
    shaping_flaps, shaping_rivers = (
        shaping_elements.convert_packing_elements_to_shaping_elements(
            packing_rectangles=packing_output.flap_rectangles,
            packing_rivers=packing_output.packing_rivers,
            packing_grid_size=packing_cp.grid_size,
            solving_grid_size=solving_grid_size,
            base_cp=base_cp,
        )
    )

  if hinge_info_list is None:
    if packing_output is None or solving_output is None:
      raise ValueError(
          'Must provide packing_output and solving_output if hinge_info_list is'
          ' None.'
      )
    hinge_info_list = packing_output.hinges
    hinge_info_list = heuristics.update_hinge_info_from_solver_result(
        hinge_info_list, solving_output
    )
  return shaping_flaps, shaping_rivers, hinge_info_list


def _compute_shaping_data(
    *,
    shaping_data: heuristics_utils.ShapingData | None,
    stick_figure: sf_utils.StickFigure,
    shaping_flaps: Sequence[shaping_elements.ShapingFlap],
    shaping_rivers: Sequence[shaping_elements.ShapingRiver],
    hinge_info_list: Sequence[packing_types.HingeInfo],
    central_crease_removed: bool,
    shape_with_unmerged_tree: bool,
    merge_converter: dict[str, str] | None,
) -> heuristics_utils.ShapingData:
  """Computes shaping data."""
  if shaping_data is None:
    shaping_data = heuristics_utils.get_heuristics_shaping_data(
        stick_figure=stick_figure,
        shaping_flaps=list(shaping_flaps),
        shaping_rivers=list(shaping_rivers),
        hinge_info_list=list(hinge_info_list),
        central_crease_removed=central_crease_removed,
        shape_with_unmerged_tree=shape_with_unmerged_tree,
        merge_converter=merge_converter,
    )
  return shaping_data


def _compute_narrowed_flaps(
    *,
    stick_figure: sf_utils.StickFigure,
    shaping_data: heuristics_utils.ShapingData,
    use_narrowing: bool,
) -> tuple[set[str], set[str], dict[str, heuristics_3d.FoldStrategy3D]]:
  """Computes which flaps get narrowed and their strategies."""
  narrowed_flap_parents = set()
  narrowed_flap_children = set()
  # Map names of flaps being narrowed to the ROTATE_BEND strategy in order
  # to accommodate narrowing.
  flap_fold_strategies = {}
  for stick in stick_figure.children:
    # NOTE: support river narrowing with clip patterns
    if use_narrowing and stick.width < 1.0 and stick.name in [
        flap.name for flap in shaping_data['shaping_flaps']
    ]:
      narrowed_flap_parents.add(stick.name)
      children = heuristics.get_descendants_names(
          shaping_data['flap_info'], stick.name
      )
      narrowed_flap_children.update(children)
      flap_fold_strategies.update({
          flap_name: heuristics_3d.FoldStrategy3D.ROTATE_BEND
          for flap_name in children + [stick.name]
      })
  narrowed_flap_parents = narrowed_flap_parents - narrowed_flap_children
  logging.info('narrowed_flap_parents: %s', narrowed_flap_parents)
  logging.info('narrowed_flap_children: %s', narrowed_flap_children)
  return narrowed_flap_parents, narrowed_flap_children, flap_fold_strategies


def _compute_3d_shaping_creases(
    *,
    base_cp: cp_utils.CreasePattern,
    stick_figure: sf_utils.StickFigure,
    shaping_flaps: Sequence[shaping_elements.ShapingFlap],
    shaping_rivers: Sequence[shaping_elements.ShapingRiver],
    hinge_info_list: Sequence[packing_types.HingeInfo],
    central_crease_removed: bool,
    shape_with_unmerged_tree: bool,
    merge_converter: dict[str, str] | None,
    hparams: geometric_fold_3d_heuristics.GeometricFolding3DHParams,
    graph_data: geometric_simple_fold.GraphData | None,
    shaping_data: heuristics_utils.ShapingData,
) -> dict[str, list[geometric_fold_3d_heuristics.ElementShapingResult]]:
  """Computes 3D shaping creases."""
  get_3d_creases = (
      geometric_fold_3d_heuristics.get_shaping_creases_with_3d_geometric_folding_heuristics
  )
  heuristics_3d_shaping_result = get_3d_creases(
      base_cp=base_cp,
      stick_figure=stick_figure,
      shaping_flaps=shaping_flaps,
      shaping_rivers=shaping_rivers,
      hinge_info_list=hinge_info_list,
      central_crease_removed=central_crease_removed,
      shape_with_unmerged_tree=shape_with_unmerged_tree,
      merge_converter=merge_converter,
      hparams=hparams,
      graph_data=graph_data,
      shaping_data=shaping_data,
  )
  return heuristics_3d_shaping_result['detailed_shaping_info']


def _partition_creases_by_narrowing(
    *,
    stick_figure: sf_utils.StickFigure,
    narrowed_flap_parents: set[str],
    narrowed_flap_children: set[str],
    detailed_shaping_info: dict[
        str, list[geometric_fold_3d_heuristics.ElementShapingResult]
    ],
) -> tuple[
    list[geometric_fold_3d_heuristics.ElementShapingResult],
    list[shaping_utils.Rectangle],
    list[geometric_fold_3d_heuristics.ElementShapingResult],
]:
  """Partitions creases into unnarrowed, narrowing rectangles, and narrowed."""
  unnarrowed_creases_and_info = []
  narrowed_creases_and_info = []
  narrowing_rectangles = []

  # First compute creases for all unnarrowed flaps.
  unnarrowed_names = [
      stick.name
      for stick in stick_figure.children
      if stick.name not in narrowed_flap_children
      and stick.name not in narrowed_flap_parents
  ]
  for unnarrowed_name in unnarrowed_names:
    if unnarrowed_name in detailed_shaping_info:
      unnarrowed_creases_and_info.extend(detailed_shaping_info[unnarrowed_name])

  # Then compute creases for all narrowed flap parents.
  # Also save narrowing flaps.
  for narrowed_parent_name in narrowed_flap_parents:
    logging.info(
        'Processing for narrowed flap with name: %s', narrowed_parent_name
    )
    if narrowed_parent_name in detailed_shaping_info:
      # Because 3d heuristics are computed with the SHALLOW_FOLDS strategy,
      # we expect the first simple fold to have rotation angle 180.0 (this
      # corresponds to a fold percent = 1.0 set of creases).
      # We will use this simple fold to facilitate narrowing, with all
      # subsequent simple folds residing in the narrowed region of the flap.
      initial_simple_fold = detailed_shaping_info[narrowed_parent_name][0]
      assert math.isclose(
          abs(initial_simple_fold['simple_fold_info'].rotation_angle),
          180.0,
          abs_tol=1e-3,
      ), (
          'First simple fold does not have rotation angle 180.0. Instead has'
          f' {initial_simple_fold["simple_fold_info"].rotation_angle}.'
      )
      narrowing_rectangle = initial_simple_fold['shaping_element']
      assert isinstance(
          narrowing_rectangle, shaping_elements.ShapingFlap
      ), 'Narrowing element is not a flap.'
      unnarrowed_creases_and_info.append(initial_simple_fold)
      narrowing_rectangles.append(narrowing_rectangle.to_legacy_rectangle())
      # Store remaining simple folds in the narrowed flap to be handled later.
      narrowed_creases_and_info.extend(
          detailed_shaping_info[narrowed_parent_name][1:]
      )

  # Finally, compute creases for narrowed flap descendants.
  for narrowed_child_name in narrowed_flap_children:
    if narrowed_child_name in detailed_shaping_info:
      narrowed_creases_and_info.extend(
          detailed_shaping_info[narrowed_child_name]
      )

  return (
      unnarrowed_creases_and_info,
      narrowing_rectangles,
      narrowed_creases_and_info,
  )


def _apply_narrowing_and_finalize(
    *,
    base_cp: cp_utils.CreasePattern,
    unnarrowed_creases_and_info: list[
        geometric_fold_3d_heuristics.ElementShapingResult
    ],
    narrowing_rectangles: list[shaping_utils.Rectangle],
    narrowed_creases_and_info: list[
        geometric_fold_3d_heuristics.ElementShapingResult
    ],
    flap_info: dict[str, heuristics.Node],
    central_crease_removed: bool,
) -> cp_utils.CreasePattern:
  """Applies narrowing and finalizes the CP."""
  # Add unnarrowed creases to the CP.
  unnarrowed_simple_fold_cp = (
      base_cp.update_with_shaping_creases(
          list(
              itertools.chain.from_iterable([
                  info['shaping_creases']
                  for info in unnarrowed_creases_and_info
              ])
          )
      )
  )

  # Map narrowed element names to all creases contained
  # in the element.
  narrowed_element_names = set(rect.name for rect in narrowing_rectangles)
  unnarrowed_creases_and_info_in_narrowed_flaps = [
      info
      for info in unnarrowed_creases_and_info
      if info['shaping_element'].name in narrowed_element_names
  ]
  element_name_to_unnarrowed_creases_and_info_map = (
      combined_heuristics_utils.sort_creases_info_by_narrowed_ancestor(
          unnarrowed_creases_and_info_in_narrowed_flaps,
          flap_info,
          narrowed_element_names,
      )
  )
  element_name_to_narrowed_crease_and_info_map = (
      combined_heuristics_utils.sort_creases_info_by_narrowed_ancestor(
          narrowed_creases_and_info,
          flap_info,
          narrowed_element_names,
      )
  )

  # Now iterate through narrowed flaps and apply narrowing.
  narrowed_cp = unnarrowed_simple_fold_cp
  successfully_narrowed_creases_and_info = []
  unssuccessfully_narrowed_creases_and_info = []
  for rectangle in narrowing_rectangles:
    try:
      unnarowed_creases_and_info = (
          element_name_to_unnarrowed_creases_and_info_map[rectangle.name]
      )
      assert len(unnarowed_creases_and_info) == 1, (
          'Expected one unnarrowed zigzag crease, but found'
          f' {len(unnarowed_creases_and_info)}.'
      )
      if not unnarowed_creases_and_info[0]['shaping_creases']:
        raise ValueError('No zigzag creases to narrow with.')
      if not math.isclose(
          unnarowed_creases_and_info[0]['shaping_creases'][0].fold_percent,
          1.0,
          abs_tol=1e-3,
      ):
        raise ValueError('Creases have fold percent < 1.0.')
      narrowed_cp = narrow_flap_shaping.narrow(
          cp=narrowed_cp,
          zigzag_creases=list(
              unnarowed_creases_and_info[0]['shaping_creases']
          ),
          rectangles=[rectangle],
          grid_size=base_cp.grid_size,
          raise_error_on_ff_violation=True,
      )
      successfully_narrowed_creases_and_info.extend(
          element_name_to_narrowed_crease_and_info_map[rectangle.name]
      )
    except ValueError as e:
      logging.warning('Failed to apply narrowing: %s', e)
      # If some narrowing fails, still apply simple folds in the same flap
      # but without narrowing.
      unssuccessfully_narrowed_creases_and_info.extend(
          element_name_to_narrowed_crease_and_info_map[rectangle.name]
      )
      unssuccessfully_narrowed_creases_and_info.extend(
          element_name_to_unnarrowed_creases_and_info_map[rectangle.name]
      )

  # 7. Apply remaining narrowed zigzag creases. These zigzags reside in the
  # narrowed regions, so we need to double their grid size when applying them.
  if successfully_narrowed_creases_and_info:
    # Filter out any requests that have no creases (perhaps failed to inscribe)
    filtered_narrowed_creases_and_info = [
        creases_and_info
        for creases_and_info in successfully_narrowed_creases_and_info
        if creases_and_info['shaping_creases']
    ]
    narrowed_zigzag_creases = get_narrowed_zigzag_creases(
        filtered_narrowed_creases_and_info,
        central_crease_removed=central_crease_removed,
    )
    shaped_cp = narrowed_cp.update_with_shaping_creases(
        narrowed_zigzag_creases,
    )
  else:
    shaped_cp = narrowed_cp

  # 8. Apply creases from unsuccessfully narrowed flaps.
  shaped_cp = shaped_cp.update_with_shaping_creases(
      list(
          itertools.chain.from_iterable(
              [
                  info['shaping_creases']
                  for info in unssuccessfully_narrowed_creases_and_info
              ]
          )
      )
  )

  return shaped_cp


def shape_cp_with_combined_heuristics(
    *,
    base_cp: cp_utils.CreasePattern,
    stick_figure: sf_utils.StickFigure,
    packing_output: packing_output_utils.PackingOutput | None = None,
    solving_output: hinge_utils.SolverResult | None = None,
    hinge_info_list: Sequence[packing_types.HingeInfo] | None = None,
    solving_grid_size: int,
    central_crease_removed: bool,
    shape_with_unmerged_tree: bool = True,
    merge_converter: dict[str, str] | None = None,
    hparams: CombinedHeuristicsHParams = CombinedHeuristicsHParams(),
    shaping_flaps: Sequence[shaping_elements.ShapingFlap] | None = None,
    shaping_rivers: Sequence[shaping_elements.ShapingRiver] | None = None,
    graph_data: geometric_simple_fold.GraphData | None = None,
    shaping_data: heuristics_utils.ShapingData | None = None,
) -> cp_utils.CreasePattern:
  """Shapes a crease pattern with 3D geometric folding heuristics and narrowing.

  This function first computes shaping creases using 3D geometric folding
  heuristics.
  Then, it applies narrowing to any flaps (sticks) that have a width less than 1
  grid unit.

  Args:
    base_cp: The base crease pattern.
    stick_figure: The stick figure.
    packing_output: The packing output.
    solving_output: The solver result.
    hinge_info_list: The list of hinge info.
    solving_grid_size: The grid size used for solving.
    central_crease_removed: Whether the central crease has been removed.
    shape_with_unmerged_tree: Whether to shape with unmerged tree.
    merge_converter: The merge converter.
    hparams: Hyperparameters controlling combined heuristic behavior.
    shaping_flaps: The shaping flaps. If None, derived from packing_output.
    shaping_rivers: The shaping rivers. If None, derived from packing_output.
    graph_data: Precomputed graph data for the crease pattern.
    shaping_data: Precomputed shaping data for the crease pattern.

  Returns:
    The shaped crease pattern.
  """
  geometric_fold_hparams = hparams.geometric_fold_hparams
  use_narrowing = hparams.use_narrowing

  shaping_flaps, shaping_rivers, hinge_info_list = _setup_shaping_elements(
      shaping_flaps=shaping_flaps,
      shaping_rivers=shaping_rivers,
      packing_output=packing_output,
      base_cp=base_cp,
      solving_grid_size=solving_grid_size,
      hinge_info_list=hinge_info_list,
      solving_output=solving_output,
  )
  logging.info('hinge_info_list: %s', hinge_info_list)

  shaping_data = _compute_shaping_data(
      shaping_data=shaping_data,
      stick_figure=stick_figure,
      shaping_flaps=shaping_flaps,
      shaping_rivers=shaping_rivers,
      hinge_info_list=hinge_info_list,
      central_crease_removed=central_crease_removed,
      shape_with_unmerged_tree=shape_with_unmerged_tree,
      merge_converter=merge_converter,
  )

  narrowed_flap_parents, narrowed_flap_children, flap_fold_strategies = (
      _compute_narrowed_flaps(
          stick_figure=stick_figure,
          shaping_data=shaping_data,
          use_narrowing=use_narrowing,
      )
  )

  per_flap_hinge_inscribe_depth = {
      name: (
          geometric_fold_3d_heuristics_utils.INSCRIBE_FRACTION
          if strategy != heuristics_3d.FoldStrategy3D.ROTATE_BEND
          else 1.0
      )
      for name, strategy in flap_fold_strategies.items()
  }
  geometric_fold_hparams = dataclasses.replace(
      geometric_fold_hparams,
      strategy=heuristics_3d.FoldStrategy3D.SHALLOW_FOLDS,
      per_flap_strategy=flap_fold_strategies,
      per_flap_hinge_inscribe_depth=per_flap_hinge_inscribe_depth,
  )

  detailed_shaping_info = _compute_3d_shaping_creases(
      base_cp=base_cp,
      stick_figure=stick_figure,
      shaping_flaps=shaping_flaps,
      shaping_rivers=shaping_rivers,
      hinge_info_list=hinge_info_list,
      central_crease_removed=central_crease_removed,
      shape_with_unmerged_tree=shape_with_unmerged_tree,
      merge_converter=merge_converter,
      hparams=geometric_fold_hparams,
      graph_data=graph_data,
      shaping_data=shaping_data,
  )

  (
      unnarrowed_creases_and_info,
      narrowing_rectangles,
      narrowed_creases_and_info,
  ) = _partition_creases_by_narrowing(
      stick_figure=stick_figure,
      narrowed_flap_parents=narrowed_flap_parents,
      narrowed_flap_children=narrowed_flap_children,
      detailed_shaping_info=detailed_shaping_info,
  )

  shaped_cp = _apply_narrowing_and_finalize(
      base_cp=base_cp,
      unnarrowed_creases_and_info=unnarrowed_creases_and_info,
      narrowing_rectangles=narrowing_rectangles,
      narrowed_creases_and_info=narrowed_creases_and_info,
      flap_info=shaping_data['flap_info'],
      central_crease_removed=central_crease_removed,
  )

  return shaped_cp
