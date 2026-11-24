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

from collections.abc import Callable
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline import pipeline
from bp_pipeline import plugins
from bp_pipeline.solving import hinges


class _FakeEntryPoint:
  """Minimal entry-point stub implementing `.load()`."""

  def __init__(self, loader: Callable[[], Callable[[], None]]):
    self._loader = loader
    self.load_calls = 0

  def load(self) -> Callable[[], None]:
    self.load_calls += 1
    return self._loader()


class PluginsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._saved_packers = dict(plugins._PACKERS)
    self._saved_grid_packers = dict(plugins._GRID_PACKERS)
    self._saved_solvers = dict(plugins._SOLVERS)
    self._saved_req_rec = set(plugins._SOLVER_REQUIRES_RECOVERY)
    self._saved_default_packer = plugins._DEFAULT_PACKER
    self._saved_default_solver = plugins._DEFAULT_SOLVER
    self._saved_loaded = plugins._PLUGINS_LOADED
    self._saved_in_progress = plugins._DISCOVERY_IN_PROGRESS
    self.addCleanup(self._restore_plugin_state)

  def _restore_plugin_state(self):
    plugins._PACKERS.clear()
    plugins._PACKERS.update(self._saved_packers)
    plugins._GRID_PACKERS.clear()
    plugins._GRID_PACKERS.update(self._saved_grid_packers)
    plugins._SOLVERS.clear()
    plugins._SOLVERS.update(self._saved_solvers)
    plugins._SOLVER_REQUIRES_RECOVERY.clear()
    plugins._SOLVER_REQUIRES_RECOVERY.update(self._saved_req_rec)
    plugins._DEFAULT_PACKER = self._saved_default_packer
    plugins._DEFAULT_SOLVER = self._saved_default_solver
    plugins._PLUGINS_LOADED = self._saved_loaded
    plugins._DISCOVERY_IN_PROGRESS = self._saved_in_progress

  def _reset_discovery(self):
    plugins._PACKERS.clear()
    plugins._GRID_PACKERS.clear()
    plugins._SOLVERS.clear()
    plugins._SOLVER_REQUIRES_RECOVERY.clear()
    plugins._DEFAULT_PACKER = None
    plugins._DEFAULT_SOLVER = None
    plugins._PLUGINS_LOADED = False
    plugins._DISCOVERY_IN_PROGRESS = False

  def _builtin_cat_result(self) -> bp.PipelineResult:
    return bp.fold('cat', preset='demo', packer='builtin', solver='builtin')

  def test_no_entry_points_resolves_to_builtin_and_preserves_default_config(
      self,
  ):
    self._reset_discovery()
    seen_groups: list[str] = []

    def _empty_entry_points(*, group: str):
      seen_groups.append(group)
      return ()

    with mock.patch.object(
        plugins.importlib_metadata,
        'entry_points',
        side_effect=_empty_entry_points,
    ):
      self.assertEqual(plugins.resolve_packer(None), 'builtin')
      self.assertEqual(plugins.resolve_packer('default'), 'builtin')
      self.assertEqual(plugins.resolve_packer('builtin'), 'builtin')
      self.assertEqual(plugins.resolve_solver(None), 'builtin')
      self.assertEqual(plugins.resolve_solver('default'), 'builtin')
      self.assertEqual(plugins.resolve_solver('builtin'), 'builtin')
      self.assertIn('default', bp.list_packers())
      self.assertIn('builtin', bp.list_packers())
      self.assertIn('default', bp.list_solvers())
      self.assertIn('builtin', bp.list_solvers())
      self.assertEqual(seen_groups, ['bp_pipeline.plugins'])

      folded = bp.fold('cat', preset='demo')
      self.assertEqual(folded.solved.config.solver, 'default')
      solved = bp.solve_best(
          folded.packed,
          bp.SolverConfig(max_steps=1000, max_time_limit=20.0),
      )
      self.assertEqual(solved.config.solver, 'default')
      self.assertEmpty(plugins._PACKERS)
      self.assertEmpty(plugins._SOLVERS)

  def test_reserved_names_rejected_and_reregistration_replaces(self):
    self._reset_discovery()
    plugins._PLUGINS_LOADED = True

    def _p1(design, **kwargs):
      del design, kwargs

    def _p2(design, **kwargs):
      del design, kwargs

    def _g1(**kwargs):
      del kwargs
      return []

    def _g2(**kwargs):
      del kwargs
      return []

    def _s1(init_state, borders, config, *, source_cp=None):
      del init_state, borders, config, source_cp
      return [], 0

    def _s2(init_state, borders, config, *, source_cp=None):
      del init_state, borders, config, source_cp
      return [], 0

    for reserved in ('default', 'builtin'):
      with self.assertRaisesRegex(ValueError, 'reserved'):
        plugins.register_packer(reserved, _p1)
      with self.assertRaisesRegex(ValueError, 'reserved'):
        plugins.register_solver(reserved, _s1)

    plugins.register_packer('custom_p', _p1, grid_fn=_g1)
    plugins.register_packer('custom_p', _p2, grid_fn=_g2)
    self.assertIs(plugins.get_packer('custom_p'), _p2)
    self.assertIs(plugins.get_grid_packer('custom_p'), _g2)

    plugins.register_solver('custom_s', _s1, requires_recovery=True)
    self.assertTrue(plugins.solver_requires_recovery('custom_s'))
    plugins.register_solver('custom_s', _s2, requires_recovery=False)
    self.assertIs(plugins.get_solver('custom_s'), _s2)
    self.assertFalse(plugins.solver_requires_recovery('custom_s'))

  def test_set_default_packer_and_solver_constraints(self):
    self._reset_discovery()
    plugins._PLUGINS_LOADED = True

    def _p(design, **kwargs):
      del design, kwargs

    def _s(init_state, borders, config, *, source_cp=None):
      del init_state, borders, config, source_cp
      return [], 0

    plugins.register_packer('p_a', _p)
    plugins.register_packer('p_b', _p)
    plugins.register_solver('s_a', _s)
    plugins.register_solver('s_b', _s)

    with self.assertRaisesRegex(ValueError, 'cannot be'):
      plugins.set_default_packer('default')
    with self.assertRaisesRegex(ValueError, 'unregistered'):
      plugins.set_default_packer('missing_p')
    with self.assertRaisesRegex(ValueError, 'cannot be'):
      plugins.set_default_solver('default')
    with self.assertRaisesRegex(ValueError, 'unregistered'):
      plugins.set_default_solver('missing_s')

    plugins.set_default_packer('p_a')
    plugins.set_default_packer('p_a')
    with self.assertRaisesRegex(ValueError, 'already set'):
      plugins.set_default_packer('p_b')

    plugins.set_default_solver('s_a')
    plugins.set_default_solver('s_a')
    with self.assertRaisesRegex(ValueError, 'already set'):
      plugins.set_default_solver('s_b')

  def test_entry_point_discovery_routes_api_while_inner_layers_stay_builtin(
      self,
  ):
    baseline = self._builtin_cat_result()
    self._reset_discovery()

    pack_out = baseline.packed.to_packing_output()
    snap = hinges.SolverStateSnapshot(
        pleats=list(baseline.solved.assigned_pleats),
        hinges=list(baseline.solved.assigned_hinges),
        ridges=[
            c
            for c in baseline.packed.creases.creases
            if c.fold == bp.Fold.MOUNTAIN
        ],
        solver_config=baseline.solved.config,
        transform=baseline.solved.symmetry,
        state_score=baseline.solved.stats.state_score,
        partition_progress=baseline.solved.stats.partition_progress,
    )
    solve_res = hinges.SolverResult(
        crease_pattern=baseline.solved.creases,
        final_state=snap,
        is_globally_flat_foldable=True,
        time_taken=0.01,
    )

    calls = {'register': 0, 'pack': 0, 'grid_pack': 0, 'solve': 0}

    def _pack(design, **kwargs):
      del design, kwargs
      calls['pack'] += 1
      return pack_out

    def _grid_pack(**kwargs):
      del kwargs
      calls['grid_pack'] += 1
      return [pack_out]

    def _solve(init_state, borders, config, *, source_cp=None):
      del init_state, borders, config, source_cp
      calls['solve'] += 1
      return [solve_res], 1

    def _register():
      calls['register'] += 1
      plugins.register_packer('example_packer', _pack, grid_fn=_grid_pack)
      plugins.register_solver(
          'example_solver', _solve, requires_recovery=True
      )
      plugins.set_default_packer('example_packer')
      plugins.set_default_solver('example_solver')
      # Re-entering registry listing during discovery must not loop.
      self.assertIn('example_packer', plugins.list_packers())

    fake_ep = _FakeEntryPoint(lambda: _register)
    seen_groups: list[str] = []

    def _fake_entry_points(*, group: str):
      seen_groups.append(group)
      return (fake_ep,)

    with mock.patch.object(
        plugins.importlib_metadata,
        'entry_points',
        side_effect=_fake_entry_points,
    ):
      self.assertEqual(plugins.resolve_packer('default'), 'example_packer')
      self.assertEqual(plugins.resolve_solver('default'), 'example_solver')
      self.assertEqual(seen_groups, ['bp_pipeline.plugins'])
      self.assertEqual(fake_ep.load_calls, 1)
      self.assertEqual(calls['register'], 1)

      # Raw getters do not resolve 'default'.
      self.assertIsNone(plugins.get_packer('default'))
      self.assertIsNone(plugins.get_grid_packer('default'))
      self.assertIsNone(plugins.get_solver('default'))
      self.assertFalse(plugins.solver_requires_recovery('default'))
      self.assertTrue(plugins.solver_requires_recovery('example_solver'))

      design = bp.prepare('cat')
      packed = bp.pack(design)
      self.assertEqual(calls['pack'], 1)
      self.assertEqual(packed.grid_size, baseline.packed.grid_size)

      # Explicit 'builtin' bypasses the registered default packer.
      packed_builtin = bp.pack(design, packer='builtin')
      self.assertEqual(calls['pack'], 1)
      self.assertEqual(
          packed_builtin.creases.hash_value, baseline.packed.creases.hash_value
      )

      solved = bp.solve_best(packed)
      self.assertEqual(calls['solve'], 1)
      self.assertEqual(
          solved.creases.hash_value, baseline.solved.creases.hash_value
      )

      sweep_sols = bp.solve_sweep(packed, configs=[bp.PRODUCTION_SWEEP[0]])
      self.assertEqual(calls['solve'], 2)
      self.assertLen(sweep_sols, 1)

      # Explicit 'builtin' bypasses the registered default solver.
      solved_builtin = bp.solve_best(
          packed,
          bp.SolverConfig(max_steps=1000, max_time_limit=20.0),
          solver='builtin',
      )
      self.assertEqual(calls['solve'], 2)
      self.assertTrue(solved_builtin.creases.local_flat_foldability())

      # bp.fold() with no engine named routes through default packer & solver.
      folded_default = bp.fold('cat', preset='demo')
      self.assertEqual(calls['pack'], 2)
      self.assertEqual(calls['solve'], 3)
      self.assertEqual(
          folded_default.packed.grid_size, baseline.packed.grid_size
      )

      # bp.fold() with explicit named packer no longer hits PackingResult error.
      folded_named = bp.fold(
          'cat',
          preset='demo',
          packer='example_packer',
          solver='example_solver',
      )
      self.assertEqual(calls['pack'], 3)
      self.assertEqual(calls['solve'], 4)
      self.assertTrue(folded_named.solved.is_globally_flat_foldable)

      # Entry points were loaded and invoked exactly once.
      self.assertEqual(fake_ep.load_calls, 1)
      self.assertEqual(calls['register'], 1)

      # Below api.py, both 'default' and 'builtin' stay on builtin engines.
      sf_input = pipeline.process_stick_figure(design)
      for engine in ('default', 'builtin'):
        grid_outs = pipeline._attempt_packing(
            stick_figure_input=sf_input,
            grid_size=baseline.packed.grid_size,
            is_symmetric=sf_input.is_symmetric_tree,
            allow_paired_axis_placement=False,
            max_solutions=1,
            config=bp.PackingConfig(
                use_level_shifter=False, enforce_proximity_constraints=False
            ),
            packer=engine,
        )
        self.assertEqual(calls['grid_pack'], 0)
        self.assertNotEmpty(grid_outs)

      direct_solver = hinges.SolverConfig(recovery=False).make_from_cp(
          packed.creases, grid_size=packed.grid_size
      )
      self.assertFalse(direct_solver.config.recovery)
      direct_sols = list(direct_solver.iter_solutions())
      self.assertEqual(calls['solve'], 4)
      self.assertNotEmpty(direct_sols)

  def test_fold_with_named_packer_without_grid_fn_does_not_repack(self):
    baseline = self._builtin_cat_result()
    self._reset_discovery()
    plugins._PLUGINS_LOADED = True

    def _pack_only(design, **kwargs):
      del design, kwargs
      return baseline.packed.to_packing_output()

    def _empty_solver(init_state, borders, config, *, source_cp=None):
      del init_state, borders, config, source_cp
      return [], 0

    plugins.register_packer('pack_only_no_grid', _pack_only)
    plugins.register_solver('empty_solver', _empty_solver)

    # Without grid_fn, a solving failure must raise SolvingFailedError directly
    # rather than crashing inside iter_additional_packing_candidates.
    with self.assertRaises(bp.SolvingFailedError):
      bp.fold(
          'cat',
          preset='demo',
          packer='pack_only_no_grid',
          solver='empty_solver',
      )

  def test_broken_entry_point_fails_loudly_without_fallback(self):
    self._reset_discovery()
    design = bp.prepare('cat')

    def _broken_loader():
      raise ModuleNotFoundError(
          "No module named '_missing_plugin_dep_for_test'",
          name='_missing_plugin_dep_for_test',
      )

    broken_ep = _FakeEntryPoint(_broken_loader)
    with mock.patch.object(
        plugins.importlib_metadata,
        'entry_points',
        return_value=(broken_ep,),
    ):
      # First and subsequent calls propagate loudly without falling back.
      with self.assertRaises(ModuleNotFoundError) as first_ctx:
        bp.pack(design)
      self.assertEqual(first_ctx.exception.name, '_missing_plugin_dep_for_test')

      with self.assertRaises(ModuleNotFoundError) as second_ctx:
        bp.pack(design)
      self.assertEqual(
          second_ctx.exception.name, '_missing_plugin_dep_for_test'
      )
      self.assertEqual(broken_ep.load_calls, 2)

      # Explicit 'builtin' remains usable without invoking entry points.
      packed_builtin = bp.pack(
          design,
          bp.PackingConfig(
              use_level_shifter=False, enforce_proximity_constraints=False
          ),
          packer='builtin',
      )
      self.assertGreater(packed_builtin.grid_size, 0)
      self.assertEqual(broken_ep.load_calls, 2)

    # A plugin registration function that raises is also never swallowed.
    self._reset_discovery()

    def _failing_register():
      raise RuntimeError('plugin initialization failed')

    failing_ep = _FakeEntryPoint(lambda: _failing_register)
    with mock.patch.object(
        plugins.importlib_metadata,
        'entry_points',
        return_value=(failing_ep,),
    ):
      with self.assertRaisesRegex(RuntimeError, 'plugin initialization failed'):
        bp.pack(design)
      with self.assertRaisesRegex(RuntimeError, 'plugin initialization failed'):
        bp.pack(design)


if __name__ == '__main__':
  absltest.main()
