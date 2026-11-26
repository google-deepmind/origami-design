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

"""Contains the core logic for the iterative backtracking solver for packing."""

from collections.abc import Callable, Generator, Iterator
import dataclasses
import logging
from typing import Sequence

from .. import utils_tree
from ..crease_pattern.parser import svg as parser
from ..stick_figures import packing_utils
from ..stick_figures import stick as stick_lib
from ..stick_figures import stick_figure as stick_figure_lib
from . import compute_ridges
from . import config as config_lib
from . import grid_size as grid_size_lib
from . import level_shifter
from . import packing_output
from . import scoring
from . import shaping_flap_templates
from . import tiling_solver
from . import types
from . import utils_packing_flaps
from . import utils_packing_svg
from . import utils_river_solver
from . import utils_tracker
from .debug import debug_types


def _compute_unfilled_area_from_placements(
    final_placement: utils_tracker.CreasePatternTracker,
    grid_size: int,
) -> int:
  """Computes unfilled area from a tiling solution.

  Args:
      final_placement: The final tiling solution containing packed polygons and
        rivers.
      grid_size: The grid size used for the tiling.

  Returns:
      The unfilled area.
  """
  polygon_placements = [
      (p.size, p.offset[0], p.offset[1])
      for p in final_placement.packed_polygons.values()
  ]
  river_placements = []
  for river in final_placement.packed_rivers.values():
    for rect in river.rectangles:
      river_placements.append((
          (int(rect.x_len), int(rect.y_len)),
          int(rect.x_start),
          int(rect.y_start),
      ))
  all_placements = polygon_placements + river_placements
  return utils_packing_flaps.compute_unfilled_area(
      grid_size, grid_size, all_placements
  )


def _create_error_packing_output(
    details: debug_types.TilingDetails,
    tracker: utils_tracker.CreasePatternTracker | None = None,
    config: config_lib.PackingConfig | None = None,
) -> packing_output.PackingOutput:
  """Creates a PackingOutput for a failed tiling.

  Args:
      details: Details about the tiling process.
      tracker: The crease pattern tracker state at time of error.
      config: The packing config.

  Returns:
      A PackingOutput object with success=False and error_message populated.
  """
  error_message = 'Tiling solver failed to find a perfect tiling solution.'
  unused_area = details['unused_area']
  grid_size = details['grid_size']
  if unused_area > 0:
    error_message += f' The best solution found has unused_area={unused_area}. '
    if details.get('num_delta_combinations') == 0:
      error_message += 'No delta combinations found to tile area.'
      details['reason'] = 'No delta combinations found'
    else:
      error_message += (
          'No delta combination could tile it perfectly after trying '
          f'{details.get("num_delta_combinations", "N/A")} combinations.'
      )
      details['reason'] = 'No perfect tiling with delta combinations.'

    if unused_area > grid_size / 2:
      error_message += '50% of the area is unused. Consider reducing grid_size.'
      details['suggestion'] = 'Consider reducing grid_size.'
  else:
    error_message += ' No solution could be found.'
    details['reason'] = 'Failed to find any valid tiling.'

  logging.error(error_message)
  return packing_output.PackingOutput.from_error(
      error_message=error_message,
      details=details,
      tracker=tracker,
      config=config,
  )


def _get_expected_adjacency(
    all_constraints: dict[str, dict[str, types.EdgeConstraint]],
) -> set[tuple[str, str]]:
  """Returns a set of flap pairs that are expected to be adjacent."""
  expected_adjacency = set()
  for stick_name, constraints in all_constraints.items():
    for other_stick, constraint in constraints.items():
      if (
          constraint['type'] == types.ConstraintType.EQUAL
          and constraint['value'] == 0
      ):
        expected_adjacency.add(tuple(sorted((stick_name, other_stick))))
  return expected_adjacency


def _check_proximity_constraints(
    tree: stick_figure_lib.StickFigure,
    hinges: Sequence[types.HingeInfo],
    details: debug_types.TilingDetails,
):
  """Checks proximity constraints and updates details."""
  all_constraints = utils_tree.compute_edge_constraints(tree)
  expected_adjacency = _get_expected_adjacency(all_constraints)

  actual_adjacency = set()
  for hinge in hinges:
    if len(hinge.flap_names) == 2:
      actual_adjacency.add(tuple(sorted(hinge.flap_names)))

  satisfied = []
  unsatisfied = []
  for pair in expected_adjacency:
    pair_str = f'{pair[0]},{pair[1]}'
    if pair in actual_adjacency:
      satisfied.append(pair_str)
    else:
      unsatisfied.append(pair_str)

  details['satisfied_constraints'] = satisfied
  details['unsatisfied_constraints'] = unsatisfied
  details['constraint_satisfaction'] = (
      f'{len(satisfied)} out of {len(expected_adjacency)} satisfied'
  )


def create_packing_output_from_tiling_solution(
    final_placement: utils_tracker.CreasePatternTracker,
    grid_size: int,
    tree: stick_figure_lib.StickFigure,
    details: debug_types.TilingDetails,
    config: config_lib.PackingConfig | None = None,
    unmerged_tree: stick_figure_lib.StickFigure | None = None,
) -> packing_output.PackingOutput:
  """Creates a PackingOutput from a tiling solution.

  Args:
      final_placement: The final tiling solution.
      grid_size: The grid size used for the tiling.
      tree: The stick figure used for the tiling.
      details: Details about the tiling process.
      config: The packing config.
      unmerged_tree: Optional unmerged stick figure before level-shifter merge.

  Returns:
      A PackingOutput object containing the tiling solution.
  """
  logging.info('Tiling successful.')
  hinges = utils_packing_svg.extract_hinges_info(final_placement, grid_size)

  # Check proximity constraints
  _check_proximity_constraints(tree, hinges, details)

  # 2. Build CreasePattern and flap rectangles directly
  logging.info('--- Building Crease Pattern ---')
  y_ridge_at_border = config.y_ridge_at_border if config is not None else True
  flap_shapes = compute_ridges.compute_flap_shapes(
      final_placement, y_ridge_at_border=y_ridge_at_border
  )
  for stick in tree.children:
    if stick.name in flap_shapes:
      stick.flap_tip_shape = flap_shapes[stick.name]
    else:
      stick.flap_tip_shape = stick_lib.FlapTipShape.NOT_APPLICABLE
  try:
    recovery = config.recovery if config is not None else False
    crease_pattern, flap_rectangles = (
        utils_packing_svg.build_packing_crease_pattern(
            grid_size,
            final_placement,
            y_ridge_at_border=y_ridge_at_border,
            recovery=recovery,
        )
    )
    logging.info('Crease pattern created successfully.')
    packed_rivers = final_placement.packed_rivers
    packing = packing_output.PackingOutput(
        success=True,
        packing_cp=crease_pattern,
        flap_rectangles=flap_rectangles,
        packing_rivers=packed_rivers,
        stick_figure=tree,
        unmerged_stick_figure=unmerged_tree,
        details=details,
        hinges=hinges,
        tracker=final_placement,
        config=config,
    )
    packing.validate()
    return packing
  except parser.SVGParseError as e:
    error_message = (
        f'Successfully found tiling, but failed to parse generated SVG: {e}'
    )
    logging.exception(error_message)
    details['reason'] = 'SVG parsing failed'
    return packing_output.PackingOutput.from_error(
        error_message=error_message,
        details=details,
        tracker=final_placement,
        config=config,
    )


def _pack_initial_pocket_for_hugging(
    default_traversal_plan: list[types.TraversalStep],
    grid_size: int,
    is_symmetric: bool,
    symmetry_axes: dict[str, bool] | None,
    debug: debug_types.DebugConfig,
    scoring_config: scoring.ScoringConfig,
    pockets_info: list[packing_utils.FlapPocket],
    config: config_lib.PackingConfig,
    base_square_dims_override: dict[str, tuple[int, int]] | None = None,
) -> Iterator[
    tuple[utils_tracker.CreasePatternTracker, dict[str, tuple[int, int]], int]
]:
  """Packs flaps in the first pocket for hugging river.

  When the first river is a hugging river, we need to pack the flaps in the
  first pocket before solving for the rest of the river. The first flap is
  packed in the top left cell of the grid.

  Args:
      default_traversal_plan: The traversal plan for packing.
      grid_size: The grid size to use for packing.
      is_symmetric: Whether symmetry is enforced.
      symmetry_axes: A dict indicating which stick names are on a symmetry axis.
      debug: Debug configuration object.
      scoring_config: The function used to score potential placements.
      pockets_info: Precomputed flap pockets.
      config: Configuration object that holds solver parameters.
      base_square_dims_override: If provided, overrides locally computed dims.

  Returns:
      An iterator of tuples containing the following:
      - placements_tracker: The tracker containing the packed placements.
      - base_square_dims: A dict mapping stick names to base square dimensions.
      - unused_area: The unused area in the grid.
  """
  stick_to_idx = {
      step['stick'].name: i for i, step in enumerate(default_traversal_plan)
  }
  pockets_info.sort(key=lambda p: min(stick_to_idx[f.name] for f in p['flaps']))
  pocket_to_pack = pockets_info[0]
  river_adjacent_cells = None
  pocket_flaps_to_pack = pocket_to_pack['flaps']
  pocket_flap_names = {f.name for f in pocket_flaps_to_pack}
  pocket_traversal = [
      s for s in default_traversal_plan if s['stick'].name in pocket_flap_names
  ]
  pocket_base_square_dims = {}
  for step in pocket_traversal:
    name = step['stick'].name
    if base_square_dims_override and name in base_square_dims_override:
      pocket_base_square_dims[name] = base_square_dims_override[name]
    else:
      pocket_base_square_dims[name] = (
          step['stick'].length * 2,
          step['stick'].length * 2,
      )
  return tiling_solver.pack_flaps_with_adjacency(
      pocket_traversal,
      pocket_base_square_dims,
      grid_size_for_tracker=grid_size,
      grid_width=grid_size,
      grid_height=grid_size,
      debug=debug,
      is_symmetric=is_symmetric,
      symmetry_axes=symmetry_axes,
      scoring_config=scoring_config,
      river_adjacent_cells=river_adjacent_cells,
      config=config,
  )


def _iter_river_solver(
    stick_figure_to_pack: stick_figure_lib.StickFigure,
    grid_size: int,
    is_symmetric: bool,
    symmetry_axes: dict[str, bool] | None,
    debug: debug_types.DebugConfig,
    scoring_config: scoring.ScoringConfig,
    config: config_lib.PackingConfig,
    default_traversal_plan: list[types.TraversalStep],
    pockets_info: list[packing_utils.FlapPocket],
    base_square_dims_override: dict[str, tuple[int, int]] | None = None,
    accept_fn: (
        Callable[[utils_tracker.CreasePatternTracker], bool] | None
    ) = None,
) -> Generator[utils_tracker.CreasePatternTracker, bool | None, None]:
  """Lazily yields candidate river trackers for downstream leaf-pocket tiling."""
  if not config.start_with_hugging_river:
    logging.info('Rivers found in stick figure, running river solver.')
    yield from utils_river_solver.iter_solve_rivers_iteratively(
        stick_figure_to_pack,
        grid_size=grid_size,
        is_symmetric=is_symmetric,
        symmetry_axes=symmetry_axes,
        debug=debug,
        scoring_config=scoring_config,
        max_solutions=None,
        config=config,
        pockets_info=pockets_info,
        default_traversal_plan=default_traversal_plan,
        base_square_dims_override=base_square_dims_override,
        accept_fn=accept_fn,
    )
    return

  packing_result_iterator = _pack_initial_pocket_for_hugging(
      default_traversal_plan,
      grid_size,
      is_symmetric,
      symmetry_axes,
      debug,
      scoring_config,
      pockets_info=pockets_info,
      config=config,
      base_square_dims_override=base_square_dims_override,
  )
  for placements_tracker, _, _ in packing_result_iterator:
    placements = placements_tracker.packed_polygons
    if not placements:
      continue
    initial_tracker_option = utils_tracker.CreasePatternTracker(grid_size)
    for flap_id, placement in placements.items():
      initial_tracker_option.add_polygon(flap_id, placement)

    logging.info(
        'Initial pocket packed for hugging river, running river solver for'
        ' remaining sticks.'
    )
    yield from utils_river_solver.iter_solve_rivers_iteratively(
        stick_figure_to_pack,
        grid_size=grid_size,
        is_symmetric=is_symmetric,
        symmetry_axes=symmetry_axes,
        debug=debug,
        scoring_config=scoring_config,
        max_solutions=None,
        config=config,
        initial_tracker=initial_tracker_option,
        pockets_info=pockets_info,
        accept_fn=accept_fn,
    )


def find_tiling_and_generate_crease_pattern(
    tree: stick_figure_lib.StickFigure,
    grid_size: int | None = None,
    is_symmetric: bool | None = None,
    symmetry_axes: dict[str, bool] | None = None,
    allow_paired_axis_placement: bool = False,
    debug: debug_types.DebugConfig | None = None,
    scoring_config: scoring.ScoringConfig = scoring.ScoringConfig(),
    max_solutions: int = 1,
    config: config_lib.PackingConfig = config_lib.PackingConfig(),
    unmerged_tree: stick_figure_lib.StickFigure | None = None,
    merge_converter: dict[str, str] | None = None,
    solution_callback: Callable[
        [packing_output.PackingOutput], packing_output.PackingOutput | None
    ] = lambda x: x,
) -> list[packing_output.PackingOutput]:
  """Runs the full pipeline from stick figure to a final crease pattern.

  This function orchestrates the entire process:
  1.  Solves for a perfect tiling based on the stick figure data.
  2.  Computes the straight skeletons for the resulting tiled polygons.
  3.  Generates a complete SVG representation of the tiling.
  4.  Parses the SVG to produce a final, structured CreasePattern object.

  Args:
      tree: The stick figure to pack.
      grid_size: The grid size to use for the tiling. If not provided, it's
        inferred from the stick figure.
      is_symmetric: If True or False, forces symmetry accordingly. If None,
        symmetry is inferred from stick_figure.is_symmetric_tree.
      symmetry_axes: A dict indicating which stick names are on a symmetry axis.
      allow_paired_axis_placement: If True, allow paired placement on axis.
      debug: If provided, enables debug mode with history recording and optional
        live plotting.
      scoring_config: The scoring config for non-easy_mode placement scoring.
        Ignored in easy_mode.
      max_solutions: Maximum number of solutions to return.
      config: Configuration object that holds solver parameters.
      unmerged_tree: The unmerged stick figure for level shifter handling.
      merge_converter: Mapping from unmerged stick name to merged stick name.
      solution_callback: A callback function for intermediate solutions. Called
        with each successful PackingOutput. Returning None from this callback
        will reject the solution.

  Returns:
      A PackingOutput object containing the packing output if successful, or
      error details if any step in the pipeline fails.
  """
  del merge_converter

  if debug is None:
    debug = debug_types.DebugConfig(live_plot=False, save_history=False)

  if is_symmetric is None:
    is_symmetric = tree.is_symmetric_tree

  if grid_size is None:
    grid_size = grid_size_lib.calculate_v4_grid_size(
        tree,
        use_level_shifter=config.use_level_shifter,
        is_symmetric=is_symmetric and not allow_paired_axis_placement,
        flap_dim_fn=utils_packing_flaps.get_flap_dim,
    )

  # Grid size must be even for symmetric packing.
  if is_symmetric and grid_size % 2 == 1:
    grid_size += 1

  if is_symmetric:
    stick_figure_to_pack, start_stick_name = (
        packing_utils.get_half_figure_and_start_stick_for_symmetric_packing(
            tree
        )
    )
    if start_stick_name:
      config = dataclasses.replace(config, start_stick_name=start_stick_name)
    if symmetry_axes is None:
      symmetry_axes = {
          stick.name: stick.is_on_symmetry_axis
          for stick in stick_figure_to_pack.children
      }
  else:
    stick_figure_to_pack = tree
  debug.is_symmetric = is_symmetric
  debug.stick_figure = stick_figure_to_pack

  if grid_size <= 0:
    raise ValueError('Grid size must be positive.')
  if is_symmetric and grid_size % 2 == 1:
    raise ValueError('Grid size must be even for symmetric packing.')
  if config.use_level_shifter:
    level_shifter.validate_supported_level_shifter_widths(tree)

  # 1. Solve for the tiling
  logging.info('--- Running Tiling Solver ---')
  try:
    logging.info('Computing constraints along traversal.')
    logging.info('Stick figure: %s', stick_figure_to_pack)
    if is_symmetric:
      pockets_info = packing_utils.precompute_flap_pockets_for_symmetric_half(
          half_figure=stick_figure_to_pack, full_figure=tree, config=config
      )
    else:
      pockets_info = packing_utils.precompute_flap_pockets(
          stick_figure_to_pack, config=config
      )
    default_traversal_plan = utils_tree.compute_constraints_along_traversal(
        stick_figure_to_pack,
        config=config,
        pockets_info=pockets_info,
    )
  except ValueError as e:
    return [
        packing_output.PackingOutput.from_error(
            error_message=f'Initial packing failed: {e}',
            details=debug_types.TilingDetails(
                grid_size=grid_size,
                is_symmetric=is_symmetric,
                perfect_tiling_found=False,
                unused_area=-1,
                reason='Initial packing failed',
                suggestion='Try increasing grid_size.',
            ),
            config=config,
        )
    ]
  base_square_dims = {}
  for step in default_traversal_plan:
    stick = step['stick']
    dim = utils_packing_flaps.get_flap_dim(stick, config.use_level_shifter)
    base_square_dims[stick.name] = (dim, dim)
  if not config.easy_mode:
    scoring_config = dataclasses.replace(
        scoring_config,
        scoring_fn=scoring.perimeter_score,
        context=scoring.PackingContext(base_square_dims=base_square_dims),
    )
  else:
    scoring_config = dataclasses.replace(
        scoring_config,
        context=scoring.PackingContext(base_square_dims=base_square_dims),
    )
  locked_long_side_flaps = {
      name for name, dims in base_square_dims.items() if dims[0] != dims[1]
  } or None

  # Pack shaping-flap templates when applicable.
  shaping_solutions_result = None
  has_shaping_flaps = any(
      stick.shaping_flap for stick in tree.children
  )
  if has_shaping_flaps:
    if config.easy_mode:
      raise ValueError(
          'Shaping-flap templates do not support easy_mode=True.'
      )
    logging.info('Shaping flaps found, running composable template solver.')
    shaping_solutions_result = (
        shaping_flap_templates.solve_trackers_with_shaping_templates(
            tree,
            stick_figure_to_pack,
            grid_size=grid_size,
            is_symmetric=is_symmetric,
            symmetry_axes=symmetry_axes,
            pockets_info=pockets_info,
            default_traversal_plan=default_traversal_plan,
            debug=debug,
            scoring_config=scoring_config,
            max_solutions=max_solutions,
            config=config,
        )
    )
    logging.info(
        'Shaping template solver finished: %s.',
        'success' if shaping_solutions_result else 'failure',
    )

  has_rivers = any(stick.is_river for stick in stick_figure_to_pack.children)
  if shaping_solutions_result is not None or has_rivers:
    if shaping_solutions_result is not None:
      source = 'Shaping'

      def _iter_shaping_solutions():
        for sol in shaping_solutions_result:
          yield sol

      river_solutions_gen = _iter_shaping_solutions()
    else:
      source = 'River'
      river_solutions_gen = _iter_river_solver(
          stick_figure_to_pack,
          grid_size,
          is_symmetric,
          symmetry_axes,
          debug,
          scoring_config,
          config,
          default_traversal_plan,
          pockets_info=pockets_info,
          base_square_dims_override=(
              base_square_dims if locked_long_side_flaps else None
          ),
      )
    outputs: list[packing_output.PackingOutput] = []
    best_error_output: packing_output.PackingOutput | None = None
    best_unused_area: int | None = None
    valid_tiling_count = 0
    rejected_river_solutions = 0
    river_candidates_tried = 0
    max_rejected_river_solutions = (
        max(16, max_solutions * 8)
        if config.river_backtracking
        else max_solutions
    )

    accepted: bool | None = None
    while True:
      if (
          not config.river_backtracking
          and river_candidates_tried >= max_solutions
      ):
        break
      try:
        final_placement = river_solutions_gen.send(accepted)
      except StopIteration:
        break
      river_candidates_tried += 1
      accepted = not config.river_backtracking
      if is_symmetric:
        utils_packing_svg.mirror_symmetric_placements(
            final_placement, grid_size, tree
        )
      packed_stick_names = set(final_placement.packed_polygons.keys()) | set(
          final_placement.packed_rivers.keys()
      )
      remaining_flaps = [
          s['stick']
          for s in default_traversal_plan
          if s['stick'].name not in packed_stick_names
          and not s['stick'].is_river
      ]
      if remaining_flaps:
        logging.info(
            'Still %d flaps to pack: %s, running tiling solver.',
            len(remaining_flaps),
            [f.name for f in remaining_flaps],
        )
        logging.info('Packed sticks: %s', packed_stick_names)
        remaining_traversal = [
            stick
            for stick in default_traversal_plan
            if any(f.name == stick['stick'].name for f in remaining_flaps)
        ]
        sacrificial_polygons = {
            name: placement
            for name, placement in final_placement.packed_polygons.items()
            if name.endswith(types.RIVER_SF1_SUFFIX)
            or name.endswith(types.RIVER_SF2_SUFFIX)
        }
        try:
          packed_flaps = tiling_solver.pack_flaps(
              tracker=final_placement,
              flaps_to_pack=remaining_flaps,
              traversal_plan=remaining_traversal,
              debug=debug,
              is_symmetric=is_symmetric,
              symmetry_axes=symmetry_axes,
              scoring_config=scoring_config,
              config=config,
              sacrificial_polygons=sacrificial_polygons or None,
              base_square_dims_override=(
                  base_square_dims if locked_long_side_flaps else None
              ),
          )
          if packed_flaps is not None:
            for flap_id, placement in packed_flaps.items():
              if flap_id in final_placement.packed_polygons:
                final_placement.remove_by_id(flap_id)
              final_placement.add_polygon(flap_id, placement)
          else:
            logging.warning('Could not pack remaining flaps.')
        except types.NoPackingFoundError as e:
          logging.warning('Could not pack remaining flaps: %s', e)
      for name, p in final_placement.packed_polygons.items():
        logging.debug('Polygon %s: offset=%s, size=%s', name, p.offset, p.size)
      for name, r in final_placement.packed_rivers.items():
        for rect in r.rectangles:
          logging.debug(
              'River %s: rect x_start=%s, y_start=%s, x_len=%s, y_len=%s',
              name,
              rect.x_start,
              rect.y_start,
              rect.x_len,
              rect.y_len,
          )
      if config.remove_central_crease and is_symmetric:
        tiling_solver.add_axis_filler_polygons(final_placement, grid_size)
      unused_area = _compute_unfilled_area_from_placements(
          final_placement, grid_size
      )
      details = debug_types.TilingDetails(
          grid_size=grid_size,
          is_symmetric=is_symmetric,
          perfect_tiling_found=unused_area == 0 or config.easy_mode,
          unused_area=unused_area,
          reason=f'Solved with {source} packer.',
      )
      if debug.save_history:
        details['debug_history'] = debug.history
      if unused_area == 0 or config.easy_mode:
        raw_result = create_packing_output_from_tiling_solution(
            final_placement,
            grid_size,
            tree,
            details,
            config=config,
            unmerged_tree=unmerged_tree,
        )
        if config.use_level_shifter and config.use_l_shaped_level_shifter:
          raw_result = level_shifter.apply_level_shifters(raw_result)
        if raw_result.success:
          valid_tiling_count += 1
          result = solution_callback(raw_result)
          if result is not None:
            accepted = True
            outputs.append(result)
            if len(outputs) >= max_solutions:
              break
          else:
            rejected_river_solutions += 1
            if rejected_river_solutions >= max_rejected_river_solutions:
              break
        else:
          if best_error_output is None:
            best_error_output = raw_result
          rejected_river_solutions += 1
          if rejected_river_solutions >= max_rejected_river_solutions:
            break
      else:
        if best_unused_area is None or unused_area < best_unused_area:
          best_unused_area = unused_area
          best_error_output = _create_error_packing_output(
              details, tracker=final_placement, config=config
          )
        rejected_river_solutions += 1
        if rejected_river_solutions >= max_rejected_river_solutions:
          break

    if valid_tiling_count > 0:
      return outputs
    if source == 'Shaping':
      template_desc = shaping_flap_templates.describe_shaping_templates(tree)
      reason = (
          f'Shaping-flap template packing failed for {template_desc} at'
          f' grid_size={grid_size}.'
      )
      details = debug_types.TilingDetails(
          grid_size=grid_size,
          is_symmetric=is_symmetric,
          perfect_tiling_found=False,
          unused_area=(
              best_unused_area if best_unused_area is not None else -1
          ),
          reason=reason,
      )
      if debug.save_history:
        details['debug_history'] = debug.history
      return [
          packing_output.PackingOutput.from_error(
              error_message=reason,
              details=details,
              tracker=None,
              config=config,
          )
      ]
    if best_error_output is not None:
      return [best_error_output]
    if config.start_with_hugging_river:
      reason = (
          'River packing failed to find a solution compatible with any'
          ' packing of the first pocket.'
      )
    else:
      reason = f'{source} packing failed to find a solution.'
    details = debug_types.TilingDetails(
        grid_size=grid_size,
        is_symmetric=is_symmetric,
        perfect_tiling_found=False,
        unused_area=-1,
        reason=reason,
    )
    if debug.save_history:
      details['debug_history'] = debug.history
    return [_create_error_packing_output(details, tracker=None, config=config)]
  else:  # River solver did not run, so run flap solver
    logging.info('No rivers found in stick figure, running flap tiling solver.')
    try:
      final_placement, details = tiling_solver.find_constrained_tiling(
          default_traversal_plan,
          base_square_dims,
          grid_size,
          grid_height=None,
          is_symmetric=is_symmetric,
          symmetry_axes=symmetry_axes,
          allow_paired_axis_placement=allow_paired_axis_placement,
          stick_figure_to_pack=stick_figure_to_pack,
          debug=debug,
          scoring_config=scoring_config,
          config=config,
          locked_long_side_flaps=locked_long_side_flaps,
      )
    except types.NoPackingFoundError as e:
      error_message = f'Initial packing failed: {e}'
      logging.error(error_message)
      details = debug_types.TilingDetails(
          grid_size=grid_size,
          is_symmetric=is_symmetric,
          perfect_tiling_found=False,
          unused_area=-1,
          reason='Initial packing failed',
          suggestion='Try increasing grid_size.',
      )
      if debug.save_history:
        details['debug_history'] = debug.history
      return [
          packing_output.PackingOutput.from_error(
              error_message=error_message, details=details, config=config
          )
      ]

  if not details['perfect_tiling_found']:
    outputs = [
        _create_error_packing_output(
            details, tracker=final_placement, config=config
        )
    ]
  else:
    raw_result = create_packing_output_from_tiling_solution(
        final_placement,
        grid_size,
        tree,
        details,
        config=config,
        unmerged_tree=unmerged_tree,
    )
    if config.use_level_shifter and config.use_l_shaped_level_shifter:
      raw_result = level_shifter.apply_level_shifters(raw_result)
    result = solution_callback(raw_result)
    outputs = [result] if result is not None else []

  return outputs
