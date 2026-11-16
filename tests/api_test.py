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

from collections.abc import Iterator
import importlib
import sys

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp


class ApiTest(parameterized.TestCase):

  def test_examples_and_prepare_variants(self):
    names = bp.list_examples()
    self.assertIn('cat', names)
    self.assertIn('human', names)
    self.assertIn('spider', names)

    cat_by_name = bp.load_example('cat')
    cat_by_const = bp.load_example('CAT_SIMPLE')
    self.assertEqual(cat_by_name, cat_by_const)
    cat_by_name.name = 'mutated_cat'
    self.assertEqual(bp.load_example('cat').name, 'cat')

    tree_from_name = bp.prepare('cat')
    tree_from_const = bp.prepare('CAT_SIMPLE')
    tree_from_sf = bp.prepare(cat_by_const)
    tree_from_dt = bp.prepare(tree_from_name)
    tree_from_raw_json = bp.prepare(cat_by_const.to_json())
    tree_from_codec_json = bp.prepare(bp.dumps(tree_from_name))

    self.assertEqual(tree_from_name, tree_from_const)
    self.assertEqual(tree_from_name, tree_from_sf)
    self.assertIs(tree_from_dt, tree_from_name)
    self.assertEqual(tree_from_name, tree_from_raw_json)
    self.assertEqual(tree_from_name, tree_from_codec_json)

  @parameterized.parameters('cat', 'CAT_SIMPLE', 'human')
  def test_tier1_fold_demo(self, example_id: str):
    result = bp.fold(example_id, preset='demo')
    self.assertIsInstance(result, bp.PipelineResult)
    self.assertIsInstance(result.design, bp.DesignTree)
    self.assertIsInstance(result.packed, bp.PackedCreasePattern)
    self.assertIsInstance(result.solved, bp.SolvedCreasePattern)
    self.assertNotEmpty(result.candidates)
    self.assertTrue(result.solved.creases.local_flat_foldability())
    self.assertTrue(result.solved.is_globally_flat_foldable)
    self.assertIn('<svg', result.packed.to_svg())
    self.assertIn('<svg', result.solved.to_svg())
    self.assertIn('vertices_coords', result.solved.flat_fold())

  def test_tier2_pack_and_streaming_solve(self):
    design = bp.prepare('human')
    fast_task_cfg = bp.PackingTaskConfig(
        packing_config=bp.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
        max_retries=5,
        max_cps_per_grid_size=1,
        max_number_of_cps=1,
        sweep_timeout_seconds=20.0,
    )

    packed = bp.pack(design, fast_task_cfg)
    self.assertGreater(packed.grid_size, 0)
    self.assertFalse(hasattr(packed, 'packer'))

    # Verify search packing handles subdivided stick figures via design.merged.
    subdivided_sf, _ = bp.load_example('human').subdivide_sticks(
        {'torso': [2, 2]}
    )
    subdivided_design = bp.prepare(subdivided_sf)
    packed_subdivided = bp.pack(subdivided_design, fast_task_cfg)
    self.assertIn('torso_0_torso_1', packed_subdivided.rivers)

    stream = bp.solve(
        packed,
        bp.SolverConfig(
            max_steps=1000,
            max_time_limit=20.0,
            max_number_of_cps=2,
            disallow_trapping_edge_flaps=True,
        ),
    )
    self.assertIsInstance(stream, Iterator)
    first_solved = next(stream)
    self.assertIsInstance(first_solved, bp.SolvedCreasePattern)
    self.assertTrue(first_solved.creases.local_flat_foldability())

    best_solved = bp.solve_best(
        packed,
        bp.SolverConfig(
            max_steps=1000,
            max_time_limit=20.0,
            max_number_of_cps=1,
            disallow_trapping_edge_flaps=True,
        ),
    )
    self.assertEqual(
        best_solved.creases.hash_value,
        first_solved.creases.hash_value,
    )

    sweep_solutions = bp.solve_sweep(
        packed,
        configs=bp.PRODUCTION_SWEEP[:2],
    )
    self.assertNotEmpty(sweep_solutions)

  def test_public_codec_round_trip_preserves_downstream_solving(self):
    design = bp.prepare('cat')
    packed = bp.pack(
        design,
        bp.PackingTaskConfig(
            packing_config=bp.PackingConfig(
                use_level_shifter=False,
                enforce_proximity_constraints=False,
            ),
            max_retries=5,
            max_cps_per_grid_size=1,
            max_number_of_cps=1,
            sweep_timeout_seconds=20.0,
        ),
    )
    decoded_packed = bp.loads(bp.dumps(packed))
    self.assertEqual(bp.to_dict(decoded_packed), bp.to_dict(packed))

    solver_cfg = bp.SolverConfig(
        max_steps=1000,
        max_time_limit=20.0,
        max_number_of_cps=1,
        disallow_trapping_edge_flaps=True,
        hinges_from_stick_figure=True,
        remove_central_crease=True,
        up_down_method='azimuth',
    )
    solved_orig = bp.solve_best(packed, solver_cfg)
    solved_decoded = bp.solve_best(decoded_packed, solver_cfg)
    self.assertEqual(
        solved_orig.creases.hash_value,
        solved_decoded.creases.hash_value,
    )

  def test_solve_remove_central_crease_when_not_applicable_returns_empty(self):
    # Construct an asymmetric PackedCreasePattern (Transform.IDENTITY), then
    # verify solve(..., remove_central_crease=True) catches NoSolutionError and
    # returns an empty iterator cleanly.
    result = bp.fold('human', preset='demo')
    asym_creases = [
        c
        for c in result.packed.creases.creases
        if not (c.fold == bp.Fold.HINGE and c.p1.x == 0.0)
    ]
    asym_cp = bp.CreasePattern(asym_creases, grid_size=result.packed.grid_size)
    asym_packed = bp.PackedCreasePattern.from_crease_pattern(asym_cp)
    solutions = list(
        bp.solve(
            asym_packed,
            bp.SolverConfig(
                max_steps=50,
                max_time_limit=10.0,
                remove_central_crease=True,
            ),
        )
    )
    self.assertEmpty(solutions)

  def test_render_packing_grid_bus_and_codec_and_eval_records(self):
    result = bp.fold('cat', preset='demo')
    html_out = bp.render_packing_grid(
        [
            result.packed,
            result.solved,
            bp.to_dict(result.packed),
            bp.to_dict(result.solved),
            {
                'stick_figure_name': 'cat',
                'pack_success': True,
                'solve_success': True,
                'packed': bp.to_dict(result.packed),
                'solved': bp.to_dict(result.solved),
            },
        ],
        title='Unit Test Contact Sheet',
    )
    self.assertIn('<!DOCTYPE html>', html_out)
    self.assertIn('Unit Test Contact Sheet', html_out)
    self.assertIn('bp-flap-rect', html_out)
    self.assertIn('bp-river-rect', html_out)
    self.assertIn('bp-hinge-line', html_out)
    self.assertIn('bp-flap-label', html_out)
    self.assertIn('non-scaling-stroke', html_out)
    self.assertIn('data-status="packed"', html_out)
    self.assertIn('data-status="ok"', html_out)

  def test_standalone_stack_utility(self):
    design = bp.prepare('human')
    stacked = bp.stack(design)
    self.assertIsInstance(stacked, bp.PackedCreasePattern)
    self.assertGreater(stacked.grid_size, 0)
    self.assertNotEmpty(stacked.flaps)

  def test_pipeline_config_presets(self):
    cfg_default = bp.PipelineConfig.default()
    self.assertIsInstance(cfg_default.packing, bp.PackingTaskConfig)
    self.assertIsInstance(cfg_default.solving, bp.SolvingTaskConfig)
    self.assertIsInstance(cfg_default.shaping, bp.ShapingTaskConfig)

    cfg_ref = bp.PipelineConfig.reference()
    self.assertFalse(cfg_ref.packing.packing_config.recovery)
    self.assertTrue(cfg_ref.packing.packing_config.legacy_iteration_order)
    self.assertFalse(cfg_ref.solving.enable_recovery)
    self.assertFalse(cfg_ref.shaping.enable_strategy_fallback)

    cfg_recovery = bp.PipelineConfig.with_recovery()
    self.assertTrue(cfg_recovery.packing.packing_config.recovery)
    self.assertFalse(cfg_recovery.packing.packing_config.legacy_iteration_order)
    self.assertTrue(cfg_recovery.solving.enable_recovery)
    self.assertTrue(cfg_recovery.shaping.use_sweep)

  def test_api_exports_parity_and_lazy_submodule_loading(self):
    api_mod = importlib.import_module('bp_pipeline.api')
    self.assertEqual(bp._API_EXPORTS, frozenset(api_mod.__all__))
    self.assertIn('PipelineConfig', api_mod.__all__)
    self.assertIn('PackerOptions', api_mod.__all__)
    self.assertNotIn('examples', dir(bp))
    self.assertIn('bps_types', dir(bp))

    saved_modules = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == 'bp_pipeline' or k.startswith('bp_pipeline.')
    }
    try:
      fresh_bp = importlib.import_module('bp_pipeline')
      self.assertNotIn('bp_pipeline.bps_types', sys.modules)
      self.assertNotIn('bp_pipeline.api', sys.modules)
      bps_mod = fresh_bp.bps_types
      self.assertIn('bp_pipeline.bps_types', sys.modules)
      self.assertTrue(hasattr(bps_mod, 'BoxPleatingTreeData'))
    finally:
      for k in list(sys.modules):
        if k == 'bp_pipeline' or k.startswith('bp_pipeline.'):
          sys.modules.pop(k, None)
      sys.modules.update(saved_modules)

  def test_pack_explicit_kwarg_precedence_overrides_config(self):
    plugins_mod = __import__('bp_pipeline.plugins', fromlist=['_PACKERS'])
    design = bp.prepare('cat')
    fast_base = bp.PackingTaskConfig(
        packing_config=bp.PackingConfig(
            use_level_shifter=False,
            enforce_proximity_constraints=False,
        ),
        max_retries=5,
        max_cps_per_grid_size=1,
        max_number_of_cps=1,
        sweep_timeout_seconds=20.0,
    )
    baseline_packed = bp.pack(design, fast_base)
    seen_calls: list[dict[str, object]] = []

    def _recording_packer(design_arg, **kwargs):
      del design_arg
      seen_calls.append(kwargs)
      return baseline_packed.to_packing_output()

    bp.register_packer('test_recording_packer', _recording_packer)
    try:
      cfg_with_custom = bp.PackingTaskConfig(
          packing_config=fast_base.packing_config,
          max_retries=5,
          max_cps_per_grid_size=4,
          max_number_of_cps=1,
          sweep_timeout_seconds=20.0,
          packer='test_recording_packer',
          packer_config={'source': 'config'},
      )
      bp.pack(design, cfg_with_custom)
      self.assertLen(seen_calls, 1)
      self.assertEqual(seen_calls[0]['packer_config'], {'source': 'config'})
      self.assertEqual(seen_calls[0]['max_solutions'], 4)
      self.assertEqual(seen_calls[0]['timeout_seconds'], 20.0)

      bp.pack(
          design,
          cfg_with_custom,
          packer_config={'source': 'kwarg'},
      )
      self.assertLen(seen_calls, 2)
      self.assertEqual(seen_calls[1]['packer_config'], {'source': 'kwarg'})

      typed_opts = bp.PackerOptions(
          timeout_seconds=12.5,
          per_grid_timeout_seconds=3.5,
          max_solutions=1,
          grid_size=14,
          num_workers=2,
      )
      bp.pack(
          design,
          cfg_with_custom,
          packer_options=typed_opts,
          packer_config={'plugin_only': 99},
      )
      self.assertLen(seen_calls, 3)
      self.assertEqual(seen_calls[2]['packer_config'], {'plugin_only': 99})
      self.assertEqual(seen_calls[2]['timeout_seconds'], 12.5)
      self.assertEqual(seen_calls[2]['per_grid_timeout_seconds'], 3.5)
      self.assertEqual(seen_calls[2]['max_solutions'], 1)
      self.assertEqual(seen_calls[2]['grid_size'], 14)
      self.assertEqual(seen_calls[2]['num_workers'], 2)

      # Single-grid offset config (`max_retries == 1` + `extra_grid_size != 0`)
      # forwards the shifted target grid size to registered plugins.
      single_grid_cfg = bp.PackingTaskConfig(
          packing_config=fast_base.packing_config,
          extra_grid_size=2,
          max_retries=1,
          packer='test_recording_packer',
      )
      bp.pack(design, single_grid_cfg)
      self.assertLen(seen_calls, 4)
      expected_grid = design.source.get_packing_gridsize() + (
          4 if design.source.is_symmetric_tree else 2
      )
      self.assertEqual(seen_calls[3]['grid_size'], expected_grid)

      # Explicit packer='default' must override config.packer.
      packed_default = bp.pack(design, cfg_with_custom, packer='default')
      self.assertLen(seen_calls, 4)
      self.assertGreater(packed_default.grid_size, 0)

      api_mod = importlib.import_module('bp_pipeline.api')
      resolved_default_cfg = api_mod._apply_packer_options_to_default_config(
          fast_base,
          bp.PackerOptions(
              timeout_seconds=30.0,
              per_grid_timeout_seconds=4.5,
              max_solutions=3,
              grid_size=expected_grid,
          ),
          design=design,
      )
      self.assertEqual(resolved_default_cfg.timeout_seconds, 30.0)
      self.assertEqual(resolved_default_cfg.sweep_timeout_seconds, 4.5)
      self.assertEqual(resolved_default_cfg.max_cps_per_grid_size, 3)
      self.assertEqual(resolved_default_cfg.max_number_of_cps, 3)
      self.assertEqual(resolved_default_cfg.extra_grid_size, 2)
      self.assertEqual(resolved_default_cfg.max_retries, 1)

      resolved_plugin_opts = api_mod._resolve_packer_options(
          None,
          task_cfg=bp.PackingTaskConfig(
              timeout_seconds=42.0, sweep_timeout_seconds=9.0
          ),
          design=design,
      )
      self.assertEqual(resolved_plugin_opts.timeout_seconds, 42.0)
    finally:
      plugins_mod._PACKERS.pop('test_recording_packer', None)

  def test_solver_plugin_and_river_recovery_capabilities(self):
    plugins_mod = __import__(
        'bp_pipeline.plugins',
        fromlist=['_SOLVERS', '_SOLVER_REQUIRES_RECOVERY'],
    )
    design = bp.prepare('cat')
    packed = bp.pack(
        design,
        bp.PackingTaskConfig(
            packing_config=bp.PackingConfig(
                use_level_shifter=False,
                enforce_proximity_constraints=False,
            ),
            max_retries=5,
            max_cps_per_grid_size=1,
            max_number_of_cps=1,
            sweep_timeout_seconds=20.0,
        ),
    )

    def _dummy_solver(init_state, borders, config, *, source_cp=None):
      del init_state, borders, config, source_cp
      return [], 0

    bp.register_solver(
        'test_req_rec_solver', _dummy_solver, requires_recovery=True
    )
    try:
      solver_instance = bp.SolverConfig(
          solver='test_req_rec_solver', recovery=False
      ).make_from_cp(packed.creases, grid_size=packed.grid_size)
      self.assertTrue(solver_instance.config.recovery)
    finally:
      plugins_mod._SOLVERS.pop('test_req_rec_solver', None)
      plugins_mod._SOLVER_REQUIRES_RECOVERY.discard('test_req_rec_solver')

    def _failing_solver(init_state, borders, config, *, source_cp=None):
      del init_state, borders, config, source_cp
      raise ValueError('simulated hinge invariant violation')

    bp.register_solver('test_failing_solver', _failing_solver)
    try:
      sweep_solutions = bp.solve_sweep(
          packed,
          configs=[
              bp.SolverConfig(
                  solver='test_failing_solver',
                  hinges_from_stick_figure=True,
                  max_steps=100,
              ),
              bp.PRODUCTION_SWEEP[0],
          ],
      )
      self.assertNotEmpty(sweep_solutions)
    finally:
      plugins_mod._SOLVERS.pop('test_failing_solver', None)

    class _DummyPackingInfo:

      def __init__(self, req: bool):
        self.packing_rivers = {
            'r0': type('_ExtRiver', (), {'requires_recovery': req})()
        }

    solver_with_ext = bp.SolverConfig(recovery=False).make_from_cp(
        packed.creases,
        grid_size=packed.grid_size,
        packing_information=_DummyPackingInfo(True),
    )
    self.assertTrue(solver_with_ext.config.recovery)
    solver_without_ext = bp.SolverConfig(recovery=False).make_from_cp(
        packed.creases,
        grid_size=packed.grid_size,
        packing_information=_DummyPackingInfo(False),
    )
    self.assertFalse(solver_without_ext.config.recovery)

  def test_shaped_to_fold_and_flat_fold_kinematic_fallback(self):
    res = bp.fold('cat', preset='demo')
    shaped = bp.shape(res.solved)
    self.assertIn('vertices_coords', bp.to_fold(shaped))
    self.assertIn('design', bp.to_bp_studio(shaped))
    folded_dict = shaped.flat_fold(strict=False)
    self.assertIn('vertices_coords', folded_dict)
    self.assertIn('edges_vertices', folded_dict)
    if folded_dict.get('bp_pipeline:kinematic_fallback'):
      with self.assertRaises(ValueError):
        shaped.flat_fold()

  def test_shaping_toolbox_and_pose_public_api(self):
    tools = bp.list_shaping_tools()
    self.assertLen(tools, 28)
    self.assertIn('simple_fold', tools)
    self.assertIn('clip_pattern_fold', tools)
    self.assertIn('squash_fold', tools)
    self.assertIn('open_flap', tools)
    self.assertIn('bend_to_direction', tools)

    res = bp.fold('human', preset='demo')
    posed, pose_report = bp.pose(
        res.solved, search=False, return_report=True
    )
    self.assertIsInstance(posed, bp.ShapedCreasePattern)
    self.assertTrue(posed.creases.local_flat_foldability())
    self.assertIn('fidelity', pose_report)

    state = bp.ShapingState.from_solved(res.solved)
    probe_out = bp.probe_shaping(state, ['left_leg'])
    self.assertIn('left_leg', probe_out)
    self.assertEmpty(state.history)

    next_state, results = bp.apply_shaping_tools(
        state,
        [
            {
                'name': 'simple_fold',
                'arguments': {
                    'element_name': 'left_leg',
                    'angle': 45.0,
                    'depth': 0.0,
                },
            },
            {
                'name': 'simple_fold',
                'arguments': {
                    'element_name': 'right_leg',
                    'angle': -45.0,
                    'depth': 0.0,
                },
            },
        ],
    )
    self.assertLen(results, 2)
    self.assertTrue(all(r.ok for r in results))
    shaped = next_state.to_shaped(res.solved)
    fidelity = bp.evaluate_fidelity(shaped)
    self.assertTrue(fidelity['clean_gate']['locally_flat_foldable'])


if __name__ == '__main__':
  absltest.main()
