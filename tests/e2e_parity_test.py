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

"""End-to-end standalone pipeline test for bp_pipeline."""

import copy
import dataclasses
import datetime
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from bp_pipeline import pipeline
from bp_pipeline import timeout
from bp_pipeline.crease_pattern import crease
from bp_pipeline.crease_pattern import crease_pattern
from bp_pipeline.crease_pattern import global_foldability
from bp_pipeline.crease_pattern import point
from bp_pipeline.crease_pattern import symmetry
from bp_pipeline.crease_pattern import vertex
from bp_pipeline.packing import compute_ridges
from bp_pipeline.packing import config as packing_config_lib
from bp_pipeline.packing import hugging_river_generator
from bp_pipeline.packing import packing_output as packing_output_lib
from bp_pipeline.packing import river as river_lib
from bp_pipeline.packing import utils_river_solver
from bp_pipeline.packing import utils_solver
from bp_pipeline.packing.debug import debug_types
from bp_pipeline.solving import hinges
from bp_pipeline.solving import partition as partition_utils
from bp_pipeline.solving import pleats as pleat_utils
from bp_pipeline.solving import ridges as ridge_utils
from bp_pipeline.solving import solvability
from bp_pipeline.stick_figures import stick_figure
from bp_pipeline.stick_figures.examples import example_trees


class EndToEndPipelineTest(parameterized.TestCase):

  @parameterized.parameters(('human',))
  def test_pack_and_solve_crease_pattern(self, tree_name: str):
    tree = example_trees.get_example_by_name(tree_name)
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    solve_cfg = pipeline.SolvingTaskConfig(
        max_number_of_cps=1,
        max_steps=1000,
        use_python_flat_folder=True,
    )
    packing_result, solve_result = pipeline.pack_and_solve_crease_pattern(
        tree,
        packing_config=pack_cfg,
        solving_config=solve_cfg,
    )
    self.assertNotEmpty(packing_result.packing_cps)
    self.assertGreater(solve_result.flat_foldable_cps_count, 0)
    solved_cp = solve_result.flat_foldable_cps[0]
    self.assertTrue(solved_cp.local_flat_foldability())
    self.assertTrue(
        global_foldability.is_globally_flat_foldable(
            solved_cp,
            scale=1,
            ndigits=3,
            grid=solved_cp.grid_size,
        )
    )

  def test_nested_cooperative_timeout_clamps_to_parent_deadline(self):
    with timeout.cooperative_timeout(datetime.timedelta(seconds=5.0)) as outer:
      with timeout.cooperative_timeout(
          datetime.timedelta(seconds=300.0)
      ) as inner:
        self.assertEqual(inner.deadline, outer.deadline)

  def test_solve_preserves_found_solutions_when_later_config_times_out(self):
    tree = example_trees.get_example_by_name('human')
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    packing_result = pipeline.pack_crease_pattern(tree, pack_cfg)
    real_solve_single = pipeline._solve_single_config
    call_configs = []

    def side_effect(config, packing_output):
      call_configs.append(config)
      if len(call_configs) == 1:
        return real_solve_single(config, packing_output)
      raise timeout.CooperativeTimeoutError(
          'Simulated timeout',
          expired_timeout=timeout.CooperativeTimeout.from_timedelta(
              datetime.timedelta(seconds=1.0)
          ),
      )

    with mock.patch.object(
        pipeline, '_solve_single_config', side_effect=side_effect
    ):
      solve_result = pipeline.solve_crease_pattern(
          packing_result,
          pipeline.SolvingTaskConfig.with_recovery(
              max_number_of_cps=1,
              max_steps=1000,
              stop_on_first_solve=False,
          ),
      )

    self.assertGreater(solve_result.flat_foldable_cps_count, 0)
    # Unguided configs (hinges_from_stick_figure=False) should be skipped once
    # a guided config succeeds.
    self.assertTrue(all(cfg.hinges_from_stick_figure for cfg in call_configs))

  def test_solve_lazy_repacks_when_initial_packing_unsolvable(self):
    tree = example_trees.get_example_by_name('human')
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    packing_result = pipeline.pack_crease_pattern(tree, pack_cfg)
    first_output = packing_result.packing_outputs[0]
    real_solve_single = pipeline._solve_single_config
    tried_outputs = []

    def side_effect(config, packing_output):
      if packing_output not in tried_outputs:
        tried_outputs.append(packing_output)
      if packing_output is first_output:
        return 'Simulated unsolvable first packing'
      return real_solve_single(config, packing_output)

    with mock.patch.object(
        pipeline, '_solve_single_config', side_effect=side_effect
    ):
      solve_result = pipeline.solve_crease_pattern(
          packing_result,
          pipeline.SolvingTaskConfig.with_recovery(
              max_number_of_cps=1, max_steps=1000
          ),
      )

    self.assertGreater(solve_result.flat_foldable_cps_count, 0)
    self.assertGreaterEqual(len(tried_outputs), 2)
    self.assertIsNot(solve_result.packing_outputs[0], first_output)
    self.assertIs(packing_result.packing_outputs[0], first_output)
    promoted = packing_result.with_promoted_packing(
        solve_result.packing_outputs[0]
    )
    self.assertIs(promoted.packing_outputs[0], solve_result.packing_outputs[0])

  def test_fallback_solver_sweep_runs_when_primary_guided_exhausted(self):
    tree = example_trees.get_example_by_name('human')
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    packing_result = pipeline.pack_crease_pattern(tree, pack_cfg)
    real_solve_single = pipeline._solve_single_config
    seen_fallback = []

    def side_effect(config, packing_output):
      if config.disallow_trapping_edge_flaps and config.scale_factor == 1:
        return 'Simulated primary exhaustion'
      seen_fallback.append(
          (config.disallow_trapping_edge_flaps, config.scale_factor)
      )
      return real_solve_single(config, packing_output)

    with mock.patch.object(
        pipeline, '_solve_single_config', side_effect=side_effect
    ):
      solve_result = pipeline.solve_crease_pattern(
          packing_result,
          pipeline.SolvingTaskConfig.with_recovery(
              max_number_of_cps=1, max_steps=1000
          ),
      )

    self.assertGreater(solve_result.flat_foldable_cps_count, 0)
    self.assertNotEmpty(seen_fallback)
    self.assertEqual(seen_fallback[0], (False, 1))

  def test_river_packing_backtracks_when_first_river_candidate_rejected(self):
    tree = example_trees.get_example_by_name('human')
    real_compute_unfilled = utils_solver._compute_unfilled_area_from_placements
    calls = []

    def simulated_unfilled(final_placement, grid_size):
      calls.append(grid_size)
      if len(calls) == 1:
        return 4
      return real_compute_unfilled(final_placement, grid_size)

    with mock.patch.object(
        utils_solver,
        '_compute_unfilled_area_from_placements',
        side_effect=simulated_unfilled,
    ):
      outputs = utils_solver.find_tiling_and_generate_crease_pattern(
          tree=tree,
          max_solutions=1,
          config=packing_config_lib.PackingConfig.with_recovery(
              use_level_shifter=False,
              enforce_proximity_constraints=False,
          ),
      )
    self.assertLen(outputs, 1)
    self.assertTrue(outputs[0].success)
    self.assertLen(calls, 2)

  def test_state_hash_and_symmetry_score_caching(self):
    tree = example_trees.get_example_by_name('human')
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    packing_result = pipeline.pack_crease_pattern(tree, pack_cfg)
    po = packing_result.packing_outputs[0]
    solver_cfg = hinges.SolverConfig.with_recovery(
        grid_size=po.packing_cp.grid_size,
        hinges_from_stick_figure=True,
    )
    solver = solver_cfg.make_from_cp(
        po.packing_cp,
        grid_size=po.packing_cp.grid_size,
        packing_information=po,
    )
    sol = next(solver.iter_solutions())
    next_state = sol.final_state
    h1 = hash(next_state)
    self.assertEqual(next_state._hash_cache, h1)
    s1 = hinges.hinge_symmetry_score(next_state)
    self.assertEqual(next_state._symmetry_score_cache, s1)
    cloned = next_state.copy_state()
    self.assertIsNone(cloned._hash_cache)
    self.assertIsNone(cloned._symmetry_score_cache)
    self.assertEqual(hash(cloned), h1)
    self.assertEqual(hinges.hinge_symmetry_score(cloned), s1)

  def test_iter_additional_packing_candidates_enforces_per_grid_quota(self):
    tree = example_trees.get_example_by_name('human')
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    packing_result = pipeline.pack_crease_pattern(tree, pack_cfg)
    first_grid = packing_result.packing_outputs[0].packing_cp.grid_size
    cands = list(
        pipeline.iter_additional_packing_candidates(
            packing_result,
            start_time=0.0,
            solve_timeout_seconds=None,
            max_extra_candidates=4,
            max_candidates_per_grid_size=2,
        )
    )
    self.assertNotEmpty(cands)
    cand_grids = [c.packing_cp.grid_size for c in cands]
    # Because `packing_result` already has 1 candidate at `first_grid`, at most
    # 1 additional candidate should come from `first_grid` before stepping up.
    self.assertLessEqual(cand_grids.count(first_grid), 1)
    self.assertIn(first_grid + 2, cand_grids)

  def test_iter_additional_packing_candidates_supports_none_max_grid_size(self):
    tree = example_trees.get_example_by_name('human')
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    sf_input = pipeline.process_stick_figure(tree)
    sf_input_none_grid = dataclasses.replace(sf_input, max_grid_size=None)
    packing_result = pipeline.pack_crease_pattern(sf_input_none_grid, pack_cfg)
    self.assertNotEmpty(packing_result.packing_outputs)
    self.assertIsNone(packing_result.stick_figure_input.max_grid_size)
    cands = list(
        pipeline.iter_additional_packing_candidates(
            packing_result,
            start_time=0.0,
            solve_timeout_seconds=None,
            max_extra_candidates=2,
            max_candidates_per_grid_size=1,
        )
    )
    self.assertNotEmpty(cands)

  def test_packing_two_pass_recovery_sweep_on_geometric_failure(self):
    tree = example_trees.get_example_by_name('human')
    calls = []
    real_attempt_packing = pipeline._attempt_packing

    def mock_attempt_packing(*args, **kwargs):
      calls.append(kwargs.get('config'))
      if len(calls) == 1:
        return [
            packing_output_lib.PackingOutput(
                success=False,
                packing_cp=None,
                flap_rectangles=None,
                packing_rivers={},
                stick_figure=None,
                error_message=(
                    'Tiling solver failed to find a perfect tiling solution.'
                    ' No solution could be found.'
                ),
                details=None,
                hinges=None,
                tracker=None,
                config=kwargs.get('config'),
            )
        ]
      return real_attempt_packing(*args, **kwargs)

    with mock.patch.object(
        pipeline, '_attempt_packing', side_effect=mock_attempt_packing
    ):
      res = pipeline.pack_crease_pattern(
          tree,
          config=pipeline.PackingTaskConfig.with_recovery(
              max_retries=1,
              max_number_of_cps=1,
          ),
      )
    self.assertLen(calls, 2)
    self.assertTrue(calls[0].star_hub_rivers)
    self.assertFalse(calls[0].symmetry_line_obstacles)
    self.assertFalse(calls[0].prune_partial_border_overhang)
    self.assertFalse(calls[0].stretch_intermediate_pocket_flaps)
    self.assertTrue(calls[1].star_hub_rivers)
    self.assertTrue(calls[1].symmetry_line_obstacles)
    self.assertTrue(calls[1].prune_partial_border_overhang)
    self.assertTrue(calls[1].stretch_intermediate_pocket_flaps)
    self.assertNotEmpty(res.packing_cps)

  def test_to_solver_config_honors_stop_on_first_solve(self):
    cfg_stop = pipeline.SolvingTaskConfig(
        max_number_of_cps=3,
        stop_on_first_solve=True,
    )
    sc_stop = cfg_stop.to_solver_config()
    self.assertEqual(sc_stop.max_number_of_cps, 1)
    self.assertEqual(sc_stop.max_attempts_per_cp, 300)
    cfg_multi = pipeline.SolvingTaskConfig(
        max_number_of_cps=3,
        stop_on_first_solve=False,
    )
    sc_multi = cfg_multi.to_solver_config()
    self.assertEqual(sc_multi.max_number_of_cps, 3)
    self.assertEqual(sc_multi.max_attempts_per_cp, 100)

  def test_partition_progress_advances_across_solved_partitions(self):
    tree = example_trees.get_example_by_name('human')
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    packing_result = pipeline.pack_crease_pattern(tree, pack_cfg)
    po = packing_result.packing_outputs[0]
    solver_cfg = hinges.SolverConfig.with_recovery(
        grid_size=po.packing_cp.grid_size,
        hinges_from_stick_figure=True,
    )
    solver = solver_cfg.make_from_cp(
        po.packing_cp,
        grid_size=po.packing_cp.grid_size,
        packing_information=po,
    )
    sol = next(solver.iter_solutions())
    self.assertLen(
        sol.final_state.partition_order.partitions,
        sol.final_state.partition_progress,
    )
    sol.final_state.partition_progress = 0
    sol.final_state.update_partition_progress()
    self.assertLen(
        sol.final_state.partition_order.partitions,
        sol.final_state.partition_progress,
    )

  def test_phase4_retries_timed_out_guided_config_with_extended_timeout(self):
    tree = example_trees.get_example_by_name('human')
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    packing_result = pipeline.pack_crease_pattern(tree, pack_cfg)
    real_solve_single = pipeline._solve_single_config
    seen_timeouts: list[float] = []
    seen_max_steps: list[int] = []

    def side_effect(config, packing_output):
      seen_timeouts.append(float(config.sweep_timeout_seconds or 0.0))
      seen_max_steps.append(config.max_steps)
      if (
          config.hinges_from_stick_figure
          and config.scale_factor == 1
          and (
              not config.disallow_trapping_edge_flaps
              or not config.use_partition_order
          )
      ):
        return 'Simulated fallback failure'
      if not config.hinges_from_stick_figure or config.scale_factor > 1:
        return 'Simulated Phase 3 failure'
      if (config.sweep_timeout_seconds or 0.0) <= 20.0:
        raise timeout.CooperativeTimeoutError('Simulated Phase 1 cap timeout')
      return real_solve_single(config, packing_output)

    with (
        mock.patch.object(
            pipeline, '_solve_single_config', side_effect=side_effect
        ),
        mock.patch.object(
            pipeline,
            'iter_additional_packing_candidates',
            return_value=iter(()),
        ),
    ):
      solve_result = pipeline.solve_crease_pattern(
          packing_result,
          pipeline.SolvingTaskConfig.with_recovery(
              max_number_of_cps=1,
              max_steps=1000,
              sweep_timeout_seconds=300,
              timeout_seconds=600,
          ),
      )

    self.assertGreater(solve_result.flat_foldable_cps_count, 0)
    self.assertIn(260.0, seen_timeouts)
    self.assertIn(1_000_000, seen_max_steps)

  def test_phase4_retries_handle_none_max_steps_and_escalate(self):
    tree = example_trees.get_example_by_name('human')
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    packing_result = pipeline.pack_crease_pattern(tree, pack_cfg)
    real_solve_single = pipeline._solve_single_config
    seen_max_steps: list[int] = []

    def side_effect(config, packing_output):
      seen_max_steps.append(config.max_steps)
      if (
          config.hinges_from_stick_figure
          and config.scale_factor == 1
          and (
              not config.disallow_trapping_edge_flaps
              or not config.use_partition_order
          )
      ):
        return 'Simulated fallback failure'
      if not config.hinges_from_stick_figure or config.scale_factor > 1:
        return 'Simulated Phase 3 failure'
      if (config.sweep_timeout_seconds or 0.0) <= 20.0:
        raise timeout.CooperativeTimeoutError('Simulated Phase 1 cap timeout')
      return real_solve_single(config, packing_output)

    with (
        mock.patch.object(
            pipeline, '_solve_single_config', side_effect=side_effect
        ),
        mock.patch.object(
            pipeline,
            'iter_additional_packing_candidates',
            return_value=iter(()),
        ),
    ):
      solve_result = pipeline.solve_crease_pattern(
          packing_result,
          pipeline.SolvingTaskConfig.with_recovery(
              max_number_of_cps=1,
              max_steps=None,
              sweep_timeout_seconds=300,
              timeout_seconds=600,
          ),
      )

    self.assertGreater(solve_result.flat_foldable_cps_count, 0)
    self.assertIn(1_000_000, seen_max_steps)

  def test_assign_pleats_disconnected_components(self):
    grid_size = 10
    p1 = crease.Crease(
        point.Point(x=0.0, y=0.1),
        point.Point(x=0.5, y=0.1),
        fold=crease.Fold.MOUNTAIN_ZEBRA,
    )
    p2 = crease.Crease(
        point.Point(x=0.0, y=0.3),
        point.Point(x=0.5, y=0.3),
        fold=crease.Fold.MOUNTAIN_ZEBRA,
    )
    p3 = crease.Crease(
        point.Point(x=0.0, y=0.6),
        point.Point(x=0.5, y=0.6),
        fold=crease.Fold.MOUNTAIN_ZEBRA,
    )
    self.assertLen(
        pleat_utils.assign_pleats(
            [p1, p2, p3], grid_size=grid_size, recovery=False
        ),
        1,
    )
    assigned = pleat_utils.assign_pleats(
        [p1, p2, p3], grid_size=grid_size, recovery=True
    )
    self.assertLen(assigned, 3)
    by_y = {round(c.p1.y, 2): c.fold for c in assigned}
    self.assertEqual(by_y[0.1], crease.Fold.MOUNTAIN)
    self.assertEqual(by_y[0.3], crease.Fold.MOUNTAIN)
    self.assertEqual(by_y[0.6], crease.Fold.VALLEY)

  def test_assign_ridges_internal_l_corner_without_y_vertex(self):
    grid_size = 10
    pl_h = crease.Crease(
        point.Point(x=0.2, y=0.5),
        point.Point(x=0.5, y=0.5),
        fold=crease.Fold.MOUNTAIN,
    )
    pl_v = crease.Crease(
        point.Point(x=0.5, y=0.2),
        point.Point(x=0.5, y=0.5),
        fold=crease.Fold.MOUNTAIN,
    )
    ridge = crease.Crease(
        point.Point(x=0.4, y=0.4),
        point.Point(x=0.6, y=0.6),
        fold=crease.Fold.TRIANGLE,
    )
    self.assertEmpty(
        ridge_utils.assign_ridges(
            [ridge], [pl_h, pl_v], grid_size=grid_size, recovery=False
        )
    )
    assigned = ridge_utils.assign_ridges(
        [ridge], [pl_h, pl_v], grid_size=grid_size, recovery=True
    )
    self.assertLen(assigned, 2)
    folds = {c.fold for c in assigned}
    self.assertEqual(folds, {crease.Fold.MOUNTAIN, crease.Fold.VALLEY})

  def test_additional_candidates_runs_exhaust_when_relaxed_grid_under_quota(
      self,
  ):
    tree = example_trees.get_example_by_name('human')
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    packing_result = pipeline.pack_crease_pattern(tree, pack_cfg)
    first_po = packing_result.packing_outputs[0]
    first_grid = first_po.packing_cp.grid_size

    real_find_tiling = utils_solver.find_tiling_and_generate_crease_pattern
    seen_calls: list[tuple[int, bool]] = []

    def fake_find_tiling(
        tree,
        grid_size,
        max_solutions,
        config,
        solution_callback,
        **kwargs,
    ):
      seen_calls.append((grid_size, config.exhaust_solutions))
      if grid_size == first_grid + 4:
        return []
      if grid_size > first_grid and not config.exhaust_solutions:
        return real_find_tiling(
            tree=tree,
            grid_size=grid_size,
            max_solutions=1,
            config=config,
            solution_callback=solution_callback,
            **kwargs,
        )
      return real_find_tiling(
          tree=tree,
          grid_size=grid_size,
          max_solutions=max_solutions,
          config=config,
          solution_callback=solution_callback,
          **kwargs,
      )

    with mock.patch.object(
        pipeline.utils_solver,
        'find_tiling_and_generate_crease_pattern',
        side_effect=fake_find_tiling,
    ):
      cands = list(
          pipeline.iter_additional_packing_candidates(
              packing_result,
              max_extra_candidates=6,
              max_candidates_per_grid_size=2,
              start_time=time.perf_counter(),
              solve_timeout_seconds=120.0,
          )
      )

    self.assertIn((first_grid + 2, False), seen_calls)
    self.assertIn((first_grid + 2, True), seen_calls)
    self.assertIn((first_grid + 4, False), seen_calls)
    self.assertNotIn((first_grid + 4, True), seen_calls)
    self.assertGreaterEqual(len(cands), 2)

  def test_generate_placements_from_start_cell_preserves_straight_before_lshape(
      self,
  ):
    placements = utils_river_solver.generate_placements_from_start_cell(
        start_cell_info=(0, 3),
        river_width=2,
        grid_size=10,
        include_straight_rivers=True,
        include_lshape_rivers=True,
    )
    self.assertNotEmpty(placements)
    self.assertIsInstance(placements[0], river_lib.River)
    seen_lshape = False
    for p in placements:
      if isinstance(p, river_lib.LShapedRiver):
        seen_lshape = True
      elif seen_lshape:
        self.fail('Straight river appeared after L-shaped river.')

  def test_iter_solve_rivers_iteratively_propagates_cooperative_timeout(self):
    tree = example_trees.get_example_by_name('human')
    debug = debug_types.DebugConfig(live_plot=False, save_history=False)
    with self.assertRaises(timeout.CooperativeTimeoutError):
      with timeout.cooperative_timeout(datetime.timedelta(microseconds=1)):
        time.sleep(0.002)
        _ = list(
            utils_river_solver.iter_solve_rivers_iteratively(
                tree,
                debug=debug,
                grid_size=14,
            )
        )

  def test_cross_partition_central_pleat_hinge_fallback(self):
    vp = point.Point(x=0.5, y=0.5)
    cp = crease.Crease(
        point.Point(x=0.25, y=0.5),
        vp,
        fold=crease.Fold.VALLEY,
    )
    h_cross = crease.Crease(
        point.Point(x=0.25, y=0.25),
        point.Point(x=0.25, y=0.75),
        fold=crease.Fold.HINGE,
    )
    r1 = crease.Crease(
        vp,
        point.Point(x=0.25, y=0.25),
        fold=crease.Fold.VALLEY,
    )
    r2 = crease.Crease(
        vp,
        point.Point(x=0.75, y=0.25),
        fold=crease.Fold.VALLEY,
    )
    r3 = crease.Crease(
        vp,
        point.Point(x=0.75, y=0.75),
        fold=crease.Fold.VALLEY,
    )
    r4 = crease.Crease(
        vp,
        point.Point(x=0.25, y=0.75),
        fold=crease.Fold.VALLEY,
    )
    cp2 = crease.Crease(
        vp,
        point.Point(x=0.75, y=0.5),
        fold=crease.Fold.VALLEY,
    )
    v_obj = vertex.Vertex([r1, r2, r3, r4, cp, cp2])
    vn = hinges.VertexNode.from_hinges_and_pleats(
        v_obj, [h_cross], [], [cp, cp2]
    )
    vn.score = 0
    meta = hinges.HingeMetaData(
        hinge=h_cross,
        segments=[
            crease.Crease(
                point.Point(x=0.25, y=0.25),
                point.Point(x=0.25, y=0.5),
                fold=crease.Fold.HINGE,
            ),
            crease.Crease(
                point.Point(x=0.25, y=0.5),
                point.Point(x=0.25, y=0.75),
                fold=crease.Fold.HINGE,
            ),
        ],
        central_pleats=[cp],
        solver_config=hinges.SolverConfig.with_recovery(grid_size=4),
    )

    class _FakePartitionOrder:
      partitions = ([vp], [h_cross.p1])

      def get_partition_id(self, p: point.Point) -> int:
        return 0 if p == vp else 1

    state = hinges.State(
        pleats=[cp, cp2],
        unassigned_pleats=[],
        hinges=[],
        unsplit_hinges=[],
        unvisited_hinges=[h_cross],
        vertices={vp: vn},
        state_score=0,
        ridges=[r1, r2, r3, r4],
        grid_points=[],
        expanded=False,
        solver_config=hinges.SolverConfig.with_recovery(grid_size=4),
        hinge_dict={h_cross.hash_value: meta},
        transform=symmetry.Transform.IDENTITY,
        partition_order=_FakePartitionOrder(),  # type: ignore[arg-type]
        partition_progress=0,
    )
    valid = state.get_valid_hinges()
    self.assertEqual(valid, [h_cross])

  def test_has_unsolveable_vertex_detects_odd_degree_without_changers(self):
    vp = point.Point(x=0.5, y=0.5)
    r1 = crease.Crease(
        vp,
        point.Point(x=0.25, y=0.25),
        fold=crease.Fold.VALLEY,
    )
    r2 = crease.Crease(
        vp,
        point.Point(x=0.75, y=0.25),
        fold=crease.Fold.VALLEY,
    )
    r3 = crease.Crease(
        vp,
        point.Point(x=0.75, y=0.75),
        fold=crease.Fold.MOUNTAIN,
    )
    r4 = crease.Crease(
        vp,
        point.Point(x=0.25, y=0.75),
        fold=crease.Fold.VALLEY,
    )
    cp = crease.Crease(
        point.Point(x=0.25, y=0.5),
        vp,
        fold=crease.Fold.MOUNTAIN,
    )
    v_obj = vertex.Vertex([r1, r2, r3, r4, cp])
    vn = hinges.VertexNode.from_hinges_and_pleats(v_obj, [], [], [cp])
    vn.score = 0
    state = hinges.State(
        pleats=[cp],
        unassigned_pleats=[],
        hinges=[],
        unsplit_hinges=[],
        unvisited_hinges=[],
        vertices={vp: vn},
        state_score=0,
        ridges=[r1, r2, r3, r4],
        grid_points=[],
        expanded=False,
        solver_config=hinges.SolverConfig.with_recovery(grid_size=4),
        hinge_dict={},
        transform=symmetry.Transform.IDENTITY,
        partition_order=None,
    )
    self.assertTrue(state._has_unsolveable_vertex())

  def test_ordered_partitions_maps_bridging_ridge_midpoint_to_earlier_partition(
      self,
  ):
    p1 = point.Point(x=0.0, y=1.0)
    p2 = point.Point(x=0.25, y=1.0)
    p3 = point.Point(x=0.5, y=1.0)
    p4 = point.Point(x=0.25, y=0.75)
    p5 = point.Point(x=0.5, y=0.75)
    h1 = crease.Crease(p1, p2, fold=crease.Fold.HINGE)
    h2 = crease.Crease(p2, p3, fold=crease.Fold.HINGE)
    h3 = crease.Crease(p4, p5, fold=crease.Fold.HINGE)
    bridging_ridge = crease.Crease(p2, p4, fold=crease.Fold.MOUNTAIN)
    part0 = partition_utils.HingePartition(
        hinges=crease.Creases.create_from_creases([h1, h2]),
        ridges=crease.Creases.create_from_creases([]),
        vertices=partition_utils.immutabledict.immutabledict({
            p1: vertex.Vertex([h1], vertex_point=p1),
            p2: vertex.Vertex([h1, h2], vertex_point=p2),
            p3: vertex.Vertex([h2], vertex_point=p3),
        }),
    )
    part1 = partition_utils.HingePartition(
        hinges=crease.Creases.create_from_creases([h3]),
        ridges=crease.Creases.create_from_creases([]),
        vertices=partition_utils.immutabledict.immutabledict({
            p4: vertex.Vertex([h3], vertex_point=p4),
            p5: vertex.Vertex([h3], vertex_point=p5),
        }),
    )
    ordered = partition_utils.OrderedPartitions(
        partitions=(part1, part0),
        bridging_creases=(bridging_ridge,),
    )
    bridging_midpoint = point.Point(x=0.25, y=0.875)
    self.assertEqual(ordered.get_partition_id(bridging_midpoint), 0)
    with self.assertRaises(ValueError):
      ordered.get_partition_id(point.Point(x=0.9, y=0.1))

  def test_with_relaxed_constraints_is_pure_and_updates_state_hash(self):
    vp = point.Point(x=0.5, y=0.5)
    r1 = crease.Crease(
        vp, point.Point(x=0.25, y=0.25), fold=crease.Fold.MOUNTAIN
    )
    r2 = crease.Crease(
        vp, point.Point(x=0.75, y=0.25), fold=crease.Fold.MOUNTAIN
    )
    r3 = crease.Crease(
        vp, point.Point(x=0.75, y=0.75), fold=crease.Fold.MOUNTAIN
    )
    r4 = crease.Crease(
        vp, point.Point(x=0.25, y=0.75), fold=crease.Fold.MOUNTAIN
    )
    h_flat1 = crease.Crease(
        vp, point.Point(x=0.5, y=0.0), fold=crease.Fold.HINGE
    )
    h_flat2 = crease.Crease(
        vp, point.Point(x=0.5, y=1.0), fold=crease.Fold.HINGE
    )
    crossing_pleat = crease.Crease(
        point.Point(x=0.0, y=0.25),
        point.Point(x=1.0, y=0.25),
        fold=crease.Fold.VALLEY,
    )
    v_obj = vertex.Vertex([r1, r2, r3, r4])
    vn = hinges.VertexNode.from_hinges_and_pleats(
        v_obj, [h_flat1, h_flat2], [], []
    )
    vn.score = 0
    cfg = hinges.SolverConfig.with_recovery(
        grid_size=4, hinges_from_stick_figure=True
    )

    class _FlatHingeMeta(hinges.HingeMetaData):

      @property
      def flat(self) -> bool:
        return not self.force_non_flat

    meta1 = _FlatHingeMeta(
        hinge=h_flat1,
        segments=[h_flat1],
        central_pleats=[],
        solver_config=cfg,
    )
    meta2 = _FlatHingeMeta(
        hinge=h_flat2,
        segments=[h_flat2],
        central_pleats=[],
        solver_config=cfg,
    )
    orig_dict = {h_flat1.hash_value: meta1, h_flat2.hash_value: meta2}
    state = hinges.State(
        pleats=[],
        unassigned_pleats=[crossing_pleat],
        hinges=[],
        unsplit_hinges=[],
        unvisited_hinges=[h_flat1, h_flat2],
        vertices={vp: vn},
        state_score=0,
        ridges=[r1, r2, r3, r4],
        grid_points=(),
        expanded=False,
        solver_config=cfg,
        hinge_dict=orig_dict,
        transform=symmetry.Transform.IDENTITY,
        partition_order=None,
    )
    self.assertTrue(state._has_unsolveable_vertex())
    self.assertIs(state.hinge_dict, orig_dict)
    # In parent state, h_flat1 is flat so it does not block crossing_pleat.
    self.assertTrue(hinges._is_assignable_pleat(crossing_pleat, state))
    relaxed = state.with_relaxed_constraints()
    self.assertIsNot(relaxed, state)
    self.assertFalse(relaxed._has_unsolveable_vertex())
    self.assertFalse(relaxed.hinge_dict[h_flat1.hash_value].flat)
    self.assertFalse(relaxed.hinge_dict[h_flat2.hash_value].flat)
    self.assertTrue(state.hinge_dict[h_flat1.hash_value].flat)
    # In relaxed child state, h_flat1 is now non-flat and blocks crossing_pleat
    # at intermediate midpoint (0.5, 0.25).
    self.assertFalse(hinges._is_assignable_pleat(crossing_pleat, relaxed))
    self.assertTrue(hinges._is_assignable_pleat(crossing_pleat, state))
    self.assertNotEqual(hash(relaxed), hash(state))
    self.assertNotEqual(relaxed, state)

  def test_solver_step_continues_expanding_when_validity_check_fails(self):
    p1 = point.Point(x=0.0, y=0.5)
    p2 = point.Point(x=1.0, y=0.5)
    h1 = crease.Crease(p1, p2, fold=crease.Fold.HINGE)
    cfg = hinges.SolverConfig.with_recovery(grid_size=2)
    meta = hinges.HingeMetaData(
        hinge=h1,
        segments=[h1],
        central_pleats=[],
        solver_config=cfg,
    )
    init_state = hinges.State(
        pleats=[],
        unassigned_pleats=[],
        hinges=[],
        unsplit_hinges=[],
        unvisited_hinges=[h1],
        vertices={},
        state_score=0,
        ridges=[],
        grid_points=(),
        expanded=False,
        solver_config=cfg,
        hinge_dict={h1.hash_value: meta},
        transform=symmetry.Transform.IDENTITY,
        partition_order=None,
    )
    solver = cfg.make_from_state(init_state, borders=[])
    solver.start_time = time.perf_counter()
    with mock.patch.object(
        hinges, 'add_solution_if_valid_state', return_value=None
    ):
      _, is_done = solver.step(0)
    self.assertFalse(is_done)
    self.assertGreater(len(solver.active_states), 1)

  @parameterized.parameters(True, False)
  def test_bucketed_get_state_to_expand_matches_linear_scan(
      self, tie_break_by_hinge_symmetry: bool
  ):
    tree = example_trees.get_example_by_name('human')
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    packing_result = pipeline.pack_crease_pattern(tree, pack_cfg)
    po = packing_result.packing_outputs[0]
    solver_cfg = hinges.SolverConfig(
        grid_size=po.packing_cp.grid_size,
        hinges_from_stick_figure=True,
        tie_break_by_hinge_symmetry=tie_break_by_hinge_symmetry,
    )
    solver = solver_cfg.make_from_cp(
        po.packing_cp,
        grid_size=po.packing_cp.grid_size,
        packing_information=po,
    )
    self.assertIsInstance(solver.active_states, hinges.ActiveStates)
    # Single initial state in bucket should not compute symmetry score eagerly.
    self.assertIsNone(solver.init_state._symmetry_score_cache)
    chosen0 = hinges.get_state_to_expand(
        solver.active_states,
        tie_break_by_hinge_symmetry=tie_break_by_hinge_symmetry,
    )
    self.assertIs(chosen0, solver.init_state)
    self.assertIsNone(solver.init_state._symmetry_score_cache)

    for step_idx in range(25):
      expected = hinges._linear_get_state_to_expand(
          dict(solver.active_states),
          tie_break_by_hinge_symmetry=tie_break_by_hinge_symmetry,
      )
      actual = hinges.get_state_to_expand(
          solver.active_states,
          tie_break_by_hinge_symmetry=tie_break_by_hinge_symmetry,
      )
      self.assertIs(actual, expected)
      _, is_done = solver.step(step_idx)
      if is_done:
        break

  def test_parent_state_immutability_and_cache_invalidation(self):
    tree = example_trees.get_example_by_name('human')
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    packing_result = pipeline.pack_crease_pattern(tree, pack_cfg)
    po = packing_result.packing_outputs[0]
    solver_cfg = hinges.SolverConfig.with_recovery(
        grid_size=po.packing_cp.grid_size,
        hinges_from_stick_figure=True,
    )
    solver = solver_cfg.make_from_cp(
        po.packing_cp,
        grid_size=po.packing_cp.grid_size,
        packing_information=po,
    )
    parent = solver.init_state
    orig_hash = hash(parent)
    orig_vertex_creases = {
        vp: list(vn.creases) for vp, vn in parent.vertices.items()
    }
    orig_vertex_scores = {vp: vn.score for vp, vn in parent.vertices.items()}
    orig_mv_counts = {vp: vn.mv_counts() for vp, vn in parent.vertices.items()}

    active = hinges.ActiveStates(tie_break_by_hinge_symmetry=True)
    active[orig_hash] = parent
    parent.get_next_states(active)

    self.assertGreater(len(active), 1)
    self.assertEqual(hash(parent), orig_hash)
    for vp, vn in parent.vertices.items():
      self.assertEqual(list(vn.creases), orig_vertex_creases[vp])
      self.assertEqual(vn.score, orig_vertex_scores[vp])
      self.assertEqual(vn.mv_counts(), orig_mv_counts[vp])

    # Mutating a cloned state with same-length pleat replacement must invalidate
    # caches via invalidate_caches().
    child = next(s for s in active.values() if s is not parent)
    child_hash_before = hash(child)
    child_sym_before = hinges.hinge_symmetry_score(child)
    self.assertIsNotNone(child._hash_cache)
    self.assertIsNotNone(child._symmetry_score_cache)
    child.invalidate_caches()
    self.assertIsNone(child._hash_cache)
    self.assertIsNone(child._symmetry_score_cache)
    self.assertEqual(hash(child), child_hash_before)
    self.assertEqual(hinges.hinge_symmetry_score(child), child_sym_before)

  def test_bucketed_active_states_synthetic_ties_and_throughput(self):
    tree = example_trees.get_example_by_name('human')
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    packing_result = pipeline.pack_crease_pattern(tree, pack_cfg)
    po = packing_result.packing_outputs[0]
    solver_cfg = hinges.SolverConfig.with_recovery(
        grid_size=po.packing_cp.grid_size,
        hinges_from_stick_figure=True,
    )
    solver = solver_cfg.make_from_cp(
        po.packing_cp,
        grid_size=po.packing_cp.grid_size,
        packing_information=po,
    )
    base = solver.init_state
    bucketed = hinges.ActiveStates(tie_break_by_hinge_symmetry=True)
    plain: dict[int, hinges.State] = {}
    for i in range(500):
      s = base.copy_state()
      s.expanded = i < 450
      s.partition_progress = i % 4
      s.state_score = i % 10
      s._symmetry_score_cache = float(i % 3)
      bucketed[i] = s
      plain[i] = s

    for tie_break in (True, False):
      expected = hinges._linear_get_state_to_expand(
          plain, tie_break_by_hinge_symmetry=tie_break
      )
      actual = hinges.get_state_to_expand(
          bucketed, tie_break_by_hinge_symmetry=tie_break
      )
      self.assertIs(actual, expected)

    # Verify deletion/pop stays in sync with plain dict.
    best_key = next(k for k, v in plain.items() if v is expected)
    del bucketed[best_key]
    del plain[best_key]
    self.assertIs(
        hinges.get_state_to_expand(bucketed, tie_break_by_hinge_symmetry=False),
        hinges._linear_get_state_to_expand(
            plain, tie_break_by_hinge_symmetry=False
        ),
    )
    bucketed[best_key] = expected
    plain[best_key] = expected

    t0 = time.perf_counter()
    for _ in range(200):
      hinges.get_state_to_expand(bucketed, tie_break_by_hinge_symmetry=True)
    bucketed_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _ in range(200):
      hinges._linear_get_state_to_expand(
          plain, tie_break_by_hinge_symmetry=True
      )
    linear_elapsed = time.perf_counter() - t1

    self.assertLess(bucketed_elapsed, linear_elapsed)

  def test_cp_signature_uses_crease_pattern_hash_value(self):
    tree = example_trees.get_example_by_name('human')
    pack_cfg = pipeline.PackingTaskConfig(
        max_number_of_cps=1,
        packing_config=packing_config_lib.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
    )
    packing_result = pipeline.pack_crease_pattern(tree, pack_cfg)
    cp = packing_result.packing_cps[0]
    self.assertIsNone(pipeline._cp_signature(None))
    self.assertEqual(pipeline._cp_signature(cp), cp.hash_value)
    self.assertIsNotNone(packing_result.stick_figure_input)
    self.assertIs(packing_result.packing_config, pack_cfg)
    self.assertIsNotNone(
        packing_result.packing_outputs[0].unmerged_stick_figure
    )

  def test_reference_configs_match_pinned_defaults(self):
    # Pinned field defaults matching the canonical baseline reference preset.
    pinned_reference_packing_config = {
        'enforce_proximity_constraints': True,
        'local_tiling_config': packing_config_lib.LocalTilingConfig(),
        'max_delta_combinations': 1_000_000,
        'traversal_mode': packing_config_lib.types.TraversalMode.RIVER_PRIORITY,
        'start_stick_name': None,
        'start_with_hugging_river': False,
        'use_level_shifter': False,
        'use_l_shaped_level_shifter': True,
        'exhaust_solutions': False,
        'easy_mode': False,
        'remove_central_crease': False,
        'include_straight_rivers': True,
        'include_lshape_rivers': True,
        'y_ridge_at_border': True,
        'river_backtracking': False,
        'legacy_iteration_order': True,
        'reject_degenerate_hugging_rivers': False,
        'star_hub_rivers': False,
        'symmetry_line_obstacles': False,
        'prune_partial_border_overhang': False,
        'stretch_intermediate_pocket_flaps': False,
        'memoize_pocket_tiling': False,
        'reject_provably_unsolvable_packings': False,
        'recovery': False,
    }
    pinned_reference_packing_task_config = {
        'timeout_seconds': None,
        'use_sweep': True,
        'sweep_hparams': {},
        'sweep_timeout_seconds': 119,
        'extra_grid_size': 0,
        'allow_paired_axis_placement': False,
        'max_cps_per_grid_size': 2,
        'max_number_of_cps': 1,
        'grid_mode': 'v2',
        'max_retries': 30,
        'packer': 'default',
        'packer_config': None,
    }
    pack_ref = pipeline.PackingTaskConfig.reference()
    actual_inner_pack = {
        f.name: getattr(pack_ref.packing_config, f.name)
        for f in dataclasses.fields(pack_ref.packing_config)
    }
    self.assertEqual(actual_inner_pack, pinned_reference_packing_config)
    actual_pack_task = {
        f.name: getattr(pack_ref, f.name)
        for f in dataclasses.fields(pack_ref)
        if f.name != 'packing_config'
    }
    self.assertEqual(actual_pack_task, pinned_reference_packing_task_config)
    self.assertEqual(pack_ref.extra_grid_sizes, range(0, 30))
    self.assertEqual(
        [c.extra_grid_size for c in pack_ref.config_iter()],
        list(range(0, 30)),
    )
    replaced_pack = dataclasses.replace(
        pipeline.PackingTaskConfig(), max_retries=5
    )
    self.assertEqual(
        [c.extra_grid_size for c in replaced_pack.config_iter()],
        [0, 1, 2, 3, 4],
    )

    pinned_reference_solving_task_config = {
        'timeout_seconds': None,
        'use_sweep': True,
        'sweep_hparams': {
            'hinges_from_stick_figure': [True, False],
            'remove_central_crease': [True, False],
        },
        'sweep_timeout_seconds': 300,
        'max_steps': 8000,
        'max_number_of_cps': 3,
        'check_local_flat_foldability': True,
        'check_global_flat_foldability': True,
        'use_python_flat_folder': True,
        'tie_break_by_hinge_symmetry': True,
        'remove_central_crease': False,
        'use_partition_order': True,
        'disallow_trapping_edge_flaps': True,
        'hinges_from_stick_figure': False,
        'scale_factor': 1,
        'allow_misaligned_symmetric_hinges': False,
        'up_down_method': 'flaps_down',
        'stop_on_first_solve': False,
        'enable_recovery': False,
        'enable_fallback_configs': False,
        'enable_lazy_repack': False,
        'enable_timeout_retries': False,
        'max_branching_factor': None,
        'solver': 'default',
        'num_workers': 1,
    }
    solve_ref = pipeline.SolvingTaskConfig.reference()
    actual_solve_task = {
        f.name: getattr(solve_ref, f.name)
        for f in dataclasses.fields(solve_ref)
    }
    self.assertEqual(actual_solve_task, pinned_reference_solving_task_config)
    self.assertEmpty(list(solve_ref.fallback_config_iter()))

    solver_cfg_ref = solve_ref.to_solver_config()
    self.assertFalse(solver_cfg_ref.recovery)
    self.assertFalse(solver_cfg_ref.allow_constraint_relaxation)
    self.assertEqual(solver_cfg_ref.max_steps, 8000)
    self.assertEqual(solver_cfg_ref.max_number_of_cps, 3)
    self.assertEqual(solver_cfg_ref.max_attempts_per_cp, 100)
    self.assertTrue(solver_cfg_ref.disallow_trapping_edge_flaps)
    self.assertEqual(solver_cfg_ref.up_down_method, 'flaps_down')

    pack_rec = pipeline.PackingTaskConfig.with_recovery()
    self.assertTrue(pack_rec.packing_config.river_backtracking)
    self.assertFalse(pack_rec.packing_config.legacy_iteration_order)
    self.assertTrue(pack_rec.packing_config.reject_degenerate_hugging_rivers)
    self.assertTrue(pack_rec.packing_config.star_hub_rivers)
    self.assertFalse(pack_rec.packing_config.symmetry_line_obstacles)
    self.assertFalse(pack_rec.packing_config.prune_partial_border_overhang)
    self.assertFalse(pack_rec.packing_config.stretch_intermediate_pocket_flaps)
    self.assertTrue(pack_rec.packing_config.memoize_pocket_tiling)
    self.assertFalse(
        pack_rec.packing_config.reject_provably_unsolvable_packings
    )
    self.assertTrue(pack_rec.packing_config.recovery)
    solve_rec = pipeline.SolvingTaskConfig.with_recovery()
    self.assertTrue(solve_rec.enable_recovery)
    self.assertTrue(solve_rec.to_solver_config().recovery)
    self.assertTrue(solve_rec.to_solver_config().allow_constraint_relaxation)

    # Verify that init_state under reference() uses the canonical
    # flat_foldability predicate (check_crimp_angles and check_kawasaki
    # with check_maekawa=False by default).
    tree = example_trees.get_example_by_name('human')
    packed_cp = pipeline.pack_crease_pattern(
        tree,
        pipeline.PackingTaskConfig(
            max_number_of_cps=1,
            packing_config=packing_config_lib.PackingConfig(
                use_level_shifter=False,
                enforce_proximity_constraints=False,
            ),
        ),
    ).packing_cps[0]
    solver = solver_cfg_ref.make_from_cp(packed_cp)
    borders = solver.borders
    init_state = solver.init_state
    bound_cfg = solver.config
    self.assertFalse(bound_cfg.recovery)
    init_cp = crease_pattern.CreasePattern(
        borders + init_state.pleats + init_state.ridges,
        grid_size=bound_cfg.grid_size,
    )
    ref_base_vertices = [
        v
        for v in init_cp.vertices
        if (not (not v.internal or (v.check_crimp_angles and v.check_kawasaki)))
        or hinges._four_ridges_vertex(v, init_state.ridges)
    ]
    ref_vertices = hinges._vertices_from_crease_intersections(
        init_cp, init_state.unsplit_hinges, ref_base_vertices
    )
    self.assertEqual(
        set(init_state.vertices.keys()),
        {v.point for v in ref_vertices},
    )
    for vp, vnode in init_state.vertices.items():
      v_test = vnode.vertex_.clone()
      for pleat in init_state.unassigned_pleats:
        if pleat.is_tip(vp):
          v_test.add_crease(pleat)
      ref_score = (not v_test.internal) or (
          v_test.check_crimp_angles and v_test.check_kawasaki
      )
      self.assertEqual(vnode.score, ref_score)

  def test_has_hairpin_or_separated_layers_detects_hairpin_and_layer_bypass(
      self,
  ):
    hairpin_path = [(20, 7), (20, 6), (19, 6), (19, 7)]
    self.assertTrue(hugging_river_generator.has_hairpin_turn([hairpin_path]))
    self.assertFalse(
        hugging_river_generator.has_separated_layers([hairpin_path])
    )
    self.assertTrue(
        hugging_river_generator.has_separated_layers(
            [hairpin_path, [(20, 16), (19, 16)]]
        )
    )

    normal_l_layers = [
        [(0, 4), (1, 4), (2, 4), (2, 5), (2, 6)],
        [(0, 5), (1, 5), (1, 6)],
    ]
    self.assertFalse(hugging_river_generator.has_hairpin_turn(normal_l_layers))
    self.assertFalse(
        hugging_river_generator.has_separated_layers(normal_l_layers)
    )
    self.assertFalse(
        hugging_river_generator.has_hairpin_or_separated_layers(normal_l_layers)
    )

  def test_has_kawasaki_impossible_vertex_detects_single_half_plane_vertex(
      self,
  ):
    def has_kawasaki_impossible_vertex(
        cp: crease_pattern.CreasePattern,
    ) -> bool:
      try:
        init_state = (
            hinges.SolverConfig.with_recovery(
                grid_size=cp.grid_size,
                use_partition_order=False,
            )
            .make_from_cp(cp)
            .init_state
        )
      except Exception:  # pylint: disable=broad-exception-caught
        return True
      for vp, vn in init_state.vertices.items():
        if vp.is_border:
          continue
        v_copy = vertex.Vertex(vn.vertex_.creases, vertex_point=vp)
        for meta in init_state.hinge_dict.values():
          for seg in meta.segments:
            if seg.is_tip(vp):
              v_copy.add_crease(seg)
        for p in init_state.unassigned_pleats:
          if p.is_tip(vp):
            v_copy.add_crease(p)
        if len(v_copy.creases) < 4 or any(
            theta >= 180.0 - 1e-6 for theta, _, _ in v_copy.sector_angles()
        ):
          return True
      return False

    vp = point.Point(x=0.25, y=0.375)
    borders = [
        crease.Crease(
            point.Point(x=0.0, y=0.0),
            point.Point(x=1.0, y=0.0),
            fold=crease.Fold.BORDER,
        ),
        crease.Crease(
            point.Point(x=1.0, y=0.0),
            point.Point(x=1.0, y=1.0),
            fold=crease.Fold.BORDER,
        ),
        crease.Crease(
            point.Point(x=1.0, y=1.0),
            point.Point(x=0.0, y=1.0),
            fold=crease.Fold.BORDER,
        ),
        crease.Crease(
            point.Point(x=0.0, y=1.0),
            point.Point(x=0.0, y=0.0),
            fold=crease.Fold.BORDER,
        ),
    ]
    r_45 = crease.Crease(
        vp,
        point.Point(x=0.5, y=0.625),
        fold=crease.Fold.MOUNTAIN,
    )
    r_135 = crease.Crease(
        vp,
        point.Point(x=0.0, y=0.625),
        fold=crease.Fold.MOUNTAIN,
    )
    h_0 = crease.Crease(
        vp,
        point.Point(x=1.0, y=0.375),
        fold=crease.Fold.HINGE,
    )
    h_90 = crease.Crease(
        vp,
        point.Point(x=0.25, y=1.0),
        fold=crease.Fold.HINGE,
    )
    h_180 = crease.Crease(
        point.Point(x=0.0, y=0.375),
        vp,
        fold=crease.Fold.HINGE,
    )
    h_lower = crease.Crease(
        point.Point(x=0.25, y=0.0),
        point.Point(x=0.25, y=0.25),
        fold=crease.Fold.HINGE,
    )
    impossible_cp = crease_pattern.CreasePattern(
        borders + [r_45, r_135, h_0, h_90, h_180, h_lower],
        grid_size=8,
        light_mode=True,
    )
    self.assertTrue(has_kawasaki_impossible_vertex(impossible_cp))

    tree = copy.deepcopy(example_trees.HUMAN_SIMPLE)
    valid_outputs = utils_solver.find_tiling_and_generate_crease_pattern(
        tree,
        grid_size=12,
        is_symmetric=True,
        max_solutions=1,
    )
    self.assertNotEmpty(valid_outputs)
    self.assertFalse(
        has_kawasaki_impossible_vertex(valid_outputs[0].packing_cp)
    )

  def test_solution_callback_rejection_triggers_river_backtracking(self):
    tree = copy.deepcopy(example_trees.HUMAN_SIMPLE)
    seen_outputs = []

    def reject_first_callback(output):
      seen_outputs.append(output)
      if len(seen_outputs) == 1:
        return None
      return output

    packing_outputs = utils_solver.find_tiling_and_generate_crease_pattern(
        tree,
        grid_size=16,
        is_symmetric=True,
        max_solutions=1,
        config=packing_config_lib.PackingConfig.with_recovery(
            exhaust_solutions=True
        ),
        solution_callback=reject_first_callback,
    )
    self.assertLen(packing_outputs, 1)
    self.assertLen(seen_outputs, 2)
    self.assertIs(packing_outputs[0], seen_outputs[1])
    self.assertIsNot(packing_outputs[0], seen_outputs[0])

  def test_two_cell_open_subriver_endpoint_ridge(self):
    # A 2-cell open river path [(32, 0), (33, 0)] has its first and last cells
    # 4-connected, so len(path) > 2 is required to distinguish open 2-cell
    # paths from closed loops and emit the endpoint turn ridge.
    river_obj = river_lib.HuggingRiver(
        grid_size=34,
        width=1,
        subriver_paths=[[(32, 0), (33, 0)]],
        hug_side=river_lib.HugSide.LEFT,
    )
    ridges = compute_ridges.get_river_ridge_lines(river_obj)
    self.assertIn(((0.0, 33.0), (1.0, 32.0)), ridges)

  @parameterized.parameters(
      (
          {
              'name': 'Swamp scorpion with a dark water-resistant shell',
              'category': 'diverse scorpions',
              'color_top': [25, 35, 30],
              'color_bottom': [20, 20, 20],
              'complexity': 2,
              'root': 'thorax',
              'children': [
                  {
                      'name': 'thorax',
                      'start': 'center',
                      'end': 'neck',
                      'azimuth_angle': 0,
                      'length': 3,
                      'width': 4,
                  },
                  {
                      'name': 'abdomen',
                      'start': 'center',
                      'end': 'base_tail',
                      'azimuth_angle': 180,
                      'length': 3,
                      'width': 4,
                  },
                  {
                      'name': 'leg_l1_inner',
                      'start': 'center',
                      'end': 'l1_knee',
                      'azimuth_angle': 60,
                      'elevation_angle': 30,
                      'length': 2,
                  },
                  {
                      'name': 'leg_l1_outer',
                      'start': 'l1_knee',
                      'end': 'l1_tip',
                      'azimuth_angle': 60,
                      'elevation_angle': -60,
                      'length': 2,
                  },
                  {
                      'name': 'leg_r1_inner',
                      'start': 'center',
                      'end': 'r1_knee',
                      'azimuth_angle': -60,
                      'elevation_angle': 30,
                      'length': 2,
                  },
                  {
                      'name': 'leg_r1_outer',
                      'start': 'r1_knee',
                      'end': 'r1_tip',
                      'azimuth_angle': -60,
                      'elevation_angle': -60,
                      'length': 2,
                  },
                  {
                      'name': 'leg_l2_inner',
                      'start': 'center',
                      'end': 'l2_knee',
                      'azimuth_angle': 100,
                      'elevation_angle': 30,
                      'length': 2,
                  },
                  {
                      'name': 'leg_l2_outer',
                      'start': 'l2_knee',
                      'end': 'l2_tip',
                      'azimuth_angle': 100,
                      'elevation_angle': -60,
                      'length': 2,
                  },
                  {
                      'name': 'leg_r2_inner',
                      'start': 'center',
                      'end': 'r2_knee',
                      'azimuth_angle': -100,
                      'elevation_angle': 30,
                      'length': 2,
                  },
                  {
                      'name': 'leg_r2_outer',
                      'start': 'r2_knee',
                      'end': 'r2_tip',
                      'azimuth_angle': -100,
                      'elevation_angle': -60,
                      'length': 2,
                  },
                  {
                      'name': 'leg_l3_inner',
                      'start': 'base_tail',
                      'end': 'l3_knee',
                      'azimuth_angle': 130,
                      'elevation_angle': 30,
                      'length': 2,
                  },
                  {
                      'name': 'leg_l3_outer',
                      'start': 'l3_knee',
                      'end': 'l3_tip',
                      'azimuth_angle': 130,
                      'elevation_angle': -60,
                      'length': 2,
                  },
                  {
                      'name': 'leg_r3_inner',
                      'start': 'base_tail',
                      'end': 'r3_knee',
                      'azimuth_angle': -130,
                      'elevation_angle': 30,
                      'length': 2,
                  },
                  {
                      'name': 'leg_r3_outer',
                      'start': 'r3_knee',
                      'end': 'r3_tip',
                      'azimuth_angle': -130,
                      'elevation_angle': -60,
                      'length': 2,
                  },
                  {
                      'name': 'leg_l4_inner',
                      'start': 'base_tail',
                      'end': 'l4_knee',
                      'azimuth_angle': 160,
                      'elevation_angle': 30,
                      'length': 2,
                  },
                  {
                      'name': 'leg_l4_outer',
                      'start': 'l4_knee',
                      'end': 'l4_tip',
                      'azimuth_angle': 160,
                      'elevation_angle': -60,
                      'length': 2,
                  },
                  {
                      'name': 'leg_r4_inner',
                      'start': 'base_tail',
                      'end': 'r4_knee',
                      'azimuth_angle': -160,
                      'elevation_angle': 30,
                      'length': 2,
                  },
                  {
                      'name': 'leg_r4_outer',
                      'start': 'r4_knee',
                      'end': 'r4_tip',
                      'azimuth_angle': -160,
                      'elevation_angle': -60,
                      'length': 2,
                  },
                  {
                      'name': 'pedipalp_l',
                      'start': 'neck',
                      'end': 'l_elbow',
                      'azimuth_angle': 40,
                      'elevation_angle': 10,
                      'length': 3,
                      'width': 2,
                  },
                  {
                      'name': 'claw_l1',
                      'start': 'l_elbow',
                      'end': 'l_p1',
                      'azimuth_angle': 20,
                      'length': 2,
                  },
                  {
                      'name': 'claw_l2',
                      'start': 'l_elbow',
                      'end': 'l_p2',
                      'azimuth_angle': 60,
                      'length': 2,
                  },
                  {
                      'name': 'pedipalp_r',
                      'start': 'neck',
                      'end': 'r_elbow',
                      'azimuth_angle': -40,
                      'elevation_angle': 10,
                      'length': 3,
                      'width': 2,
                  },
                  {
                      'name': 'claw_r1',
                      'start': 'r_elbow',
                      'end': 'r_p1',
                      'azimuth_angle': -20,
                      'length': 2,
                  },
                  {
                      'name': 'claw_r2',
                      'start': 'r_elbow',
                      'end': 'r_p2',
                      'azimuth_angle': -60,
                      'length': 2,
                  },
                  {
                      'name': 'tail_1',
                      'start': 'base_tail',
                      'end': 'tail_j1',
                      'azimuth_angle': 180,
                      'elevation_angle': 20,
                      'length': 2,
                      'width': 2,
                  },
                  {
                      'name': 'tail_2',
                      'start': 'tail_j1',
                      'end': 'tail_j2',
                      'azimuth_angle': 180,
                      'elevation_angle': 50,
                      'length': 2,
                      'width': 2,
                  },
                  {
                      'name': 'tail_3',
                      'start': 'tail_j2',
                      'end': 'tail_j3',
                      'azimuth_angle': 180,
                      'elevation_angle': 80,
                      'length': 2,
                  },
                  {
                      'name': 'stinger',
                      'start': 'tail_j3',
                      'end': 'sting_tip',
                      'azimuth_angle': 0,
                      'elevation_angle': 40,
                      'length': 2,
                  },
              ],
          },
          10,
      ),
  )
  def test_scorpion_trees_solve_globally_flat_foldable(
      self, tree_dict, extra_grid_size
  ):
    sf = stick_figure.StickFigure.from_dict(tree_dict)
    packing_res = pipeline.pack_crease_pattern(
        sf,
        config=pipeline.PackingTaskConfig(
            extra_grid_size=extra_grid_size,
            max_retries=1,
            max_number_of_cps=1,
            max_cps_per_grid_size=1,
            timeout_seconds=600,
            sweep_timeout_seconds=600,
            packing_config=packing_config_lib.PackingConfig(
                symmetry_line_obstacles=True,
                stretch_intermediate_pocket_flaps=True,
            ),
        ),
    )
    solving_res = pipeline.solve_crease_pattern(
        packing_res,
        config=pipeline.SolvingTaskConfig(
            timeout_seconds=1200,
            sweep_timeout_seconds=1200,
        ),
    )
    self.assertEqual(
        solving_res.sequence_status,
        'SUCCESS in SOLVE_FLAT_FOLDABLE_CREASE_PATTERN',
        msg=f'Failed to solve scorpion tree ({tree_dict["name"]})',
    )
    self.assertNotEmpty(solving_res.solver_outputs)
    self.assertTrue(solving_res.solver_outputs[0].is_globally_flat_foldable)

  def test_solver_frontier_retention_and_recovery_solve(self):
    sf = example_trees.get_example_by_name('human')
    pack_res = pipeline.pack_crease_pattern(
        sf,
        config=pipeline.PackingTaskConfig.with_recovery(
            max_retries=1,
            max_number_of_cps=1,
            timeout_seconds=60,
        ),
    )
    solve_res = pipeline.solve_crease_pattern(
        pack_res,
        config=pipeline.SolvingTaskConfig.with_recovery(
            max_steps=50,
            timeout_seconds=60,
        ),
    )
    self.assertEqual(
        solve_res.sequence_status,
        'SUCCESS in SOLVE_FLAT_FOLDABLE_CREASE_PATTERN',
    )

  def test_scale2_recovery_and_solving_with_none_grid_size(self):
    sf = example_trees.get_example_by_name('human')
    pack_res = pipeline.pack_crease_pattern(
        sf,
        config=pipeline.PackingTaskConfig(
            max_number_of_cps=1,
            timeout_seconds=30,
        ),
    )
    self.assertNotEmpty(pack_res.packing_outputs)
    orig_output = pack_res.packing_outputs[0]

    orig_grid = orig_output.packing_cp.grid_size
    mock_cp = crease_pattern.CreasePattern(list(orig_output.packing_cp.creases))
    mock_cp.grid_size = None
    mock_output = orig_output.model_copy(update={'packing_cp': mock_cp})

    # 1. Verify _safe_grid_size infers correct grid size from creases.
    self.assertIsNone(mock_cp.grid_size)
    inferred_grid = pipeline._safe_grid_size(mock_output)
    self.assertEqual(inferred_grid, orig_grid)

    # 2. Verify _build_scale2_guided_pairs constructs pairs without TypeError.
    recovery_ctx = pipeline._SolverRecoveryContext(
        all_tried_outputs=[mock_output],
        initial_outputs=[mock_output],
        primary_guided=[],
        primary_unguided=[],
        fallback_guided_scale1=[],
        fallback_guided_scale2=[
            pipeline.SolvingTaskConfig(scale_factor=2, max_steps=100)
        ],
        timed_out_guided_pairs=[],
        timed_out_unguided_pairs=[],
    )
    pairs = pipeline._build_scale2_guided_pairs(recovery_ctx)
    self.assertNotEmpty(pairs)
    self.assertEqual(pairs[0][0], mock_output)
    self.assertEqual(pairs[0][1].scale_factor, 2)

    # 3. Verify _solve_single_config scales correctly with mock_output.
    config = pipeline.SolvingTaskConfig(
        scale_factor=2,
        max_steps=50,
        use_python_flat_folder=True,
    )
    try:
      _ = pipeline._solve_single_config(config, mock_output)
    except hinges.NoSolutionError:
      pass

    # 4. Verify hinges make_from_cp works with grid_size=None.
    solver_cfg = config.to_solver_config()
    solver = solver_cfg.make_from_cp(mock_cp)
    self.assertEqual(solver.config.grid_size, inferred_grid)
    self.assertFalse(solver_cfg.is_valid_crease_pattern(mock_cp))

  def test_greedy_fallback_and_none_grid_size(self):
    pack_res = pipeline.packing_output_lib.PackingOutput(
        success=True,
        packing_cp=None,
        flap_rectangles=(),
        packing_rivers={},
        stick_figure=None,
        error_message=None,
        details=None,
        hinges=None,
        tracker=None,
        config=None,
    )
    ctx = pipeline._SolverRecoveryContext(
        initial_outputs=[pack_res],
        all_tried_outputs=[pack_res],
        primary_guided=[pipeline.SolvingTaskConfig(max_steps=1)],
        primary_unguided=[],
        fallback_guided_scale1=[],
        fallback_guided_scale2=[],
    )
    pairs_scale2 = pipeline._build_scale2_guided_pairs(ctx)
    self.assertEqual(pairs_scale2, [])

    pairs_greedy = pipeline._build_greedy_fallback_pairs(ctx)
    self.assertLen(pairs_greedy, 1)
    self.assertEqual(pairs_greedy[0][1].max_branching_factor, 1)

    # Verify resuming a cached solver preserves bound grid_size on solver.config
    borders = [
        crease.Crease(
            point.Point(x=0.0, y=0.0),
            point.Point(x=1.0, y=0.0),
            fold=crease.Fold.BORDER,
        ),
        crease.Crease(
            point.Point(x=1.0, y=0.0),
            point.Point(x=1.0, y=1.0),
            fold=crease.Fold.BORDER,
        ),
        crease.Crease(
            point.Point(x=1.0, y=1.0),
            point.Point(x=0.0, y=1.0),
            fold=crease.Fold.BORDER,
        ),
        crease.Crease(
            point.Point(x=0.0, y=1.0),
            point.Point(x=0.0, y=0.0),
            fold=crease.Fold.BORDER,
        ),
    ]
    valid_cp = crease_pattern.CreasePattern(borders, grid_size=16)
    pack_with_cp = pack_res.model_copy(update={'packing_cp': valid_cp})
    task_cfg = pipeline.SolvingTaskConfig(max_steps=1, enable_recovery=True)
    try:
      _ = pipeline._solve_single_config(
          task_cfg, pack_with_cp, recovery_ctx=ctx
      )
    except hinges.NoSolutionError:
      pass
    solver_key = pipeline._solver_instance_key(pack_with_cp, task_cfg)
    cached_solver = ctx.active_solvers[solver_key]
    self.assertEqual(cached_solver.config.grid_size, 16)
    greedy_cfg = dataclasses.replace(task_cfg, max_branching_factor=1)
    try:
      _ = pipeline._solve_single_config(
          greedy_cfg, pack_with_cp, recovery_ctx=ctx
      )
    except hinges.NoSolutionError:
      pass
    self.assertEqual(cached_solver.config.grid_size, 16)
    self.assertEqual(cached_solver.config.max_branching_factor, 1)

  def test_score_packing_solvability_and_opposing_pleat_detection(self):
    borders = [
        crease.Crease(
            point.Point(x=0.0, y=0.0),
            point.Point(x=1.0, y=0.0),
            fold=crease.Fold.BORDER,
        ),
        crease.Crease(
            point.Point(x=1.0, y=0.0),
            point.Point(x=1.0, y=1.0),
            fold=crease.Fold.BORDER,
        ),
        crease.Crease(
            point.Point(x=1.0, y=1.0),
            point.Point(x=0.0, y=1.0),
            fold=crease.Fold.BORDER,
        ),
        crease.Crease(
            point.Point(x=0.0, y=1.0),
            point.Point(x=0.0, y=0.0),
            fold=crease.Fold.BORDER,
        ),
    ]
    trapped_cp = crease_pattern.CreasePattern(
        borders
        + [
            crease.Crease(
                point.Point(x=0.5, y=0.0),
                point.Point(x=0.5, y=0.4),
                fold=crease.Fold.MOUNTAIN,
            ),
            crease.Crease(
                point.Point(x=0.5, y=0.6),
                point.Point(x=0.5, y=1.0),
                fold=crease.Fold.VALLEY,
            ),
        ],
        grid_size=10,
    )
    res_trapped = solvability.check_solvability(trapped_cp)
    self.assertFalse(res_trapped.is_solvable)
    self.assertIn('Opposing axial pleats', res_trapped.reason)

    separated_cp = crease_pattern.CreasePattern(
        borders
        + [
            crease.Crease(
                point.Point(x=0.5, y=0.0),
                point.Point(x=0.5, y=0.4),
                fold=crease.Fold.MOUNTAIN,
            ),
            crease.Crease(
                point.Point(x=0.5, y=0.6),
                point.Point(x=0.5, y=1.0),
                fold=crease.Fold.VALLEY,
            ),
            crease.Crease(
                point.Point(x=0.2, y=0.3),
                point.Point(x=0.8, y=0.7),
                fold=crease.Fold.MOUNTAIN,
            ),
        ],
        grid_size=10,
    )
    res_separated = solvability.check_solvability(separated_cp)
    self.assertTrue(res_separated.is_solvable)
    self.assertLess(
        solvability.score_packing_solvability(separated_cp),
        solvability.score_packing_solvability(trapped_cp),
    )


if __name__ == '__main__':
  absltest.main()
