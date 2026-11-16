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

"""Unit tests for `bp_pipeline.shaping.toolbox`."""

import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline.shaping import fidelity
from bp_pipeline.shaping import toolbox as shaping

_STATES: dict[str, shaping.ShapingState] = {}


def _build_subdivided_human() -> bp.StickFigure:
  """Builds a symmetric subdivided-limb human fixture for multi-segment tests."""
  base_dict = bp.load_example('HUMAN_SIMPLE').to_dict()
  for child in base_dict['children']:
    if child['name'] in ('left_leg', 'right_leg'):
      child['length'] = 6
  base = bp.StickFigure.from_dict(base_dict)
  sub, _ = base.subdivide_sticks({
      'left_leg': [3, 3],
      'right_leg': [3, 3],
      'left_arm': [2, 1],
      'right_arm': [2, 1],
  })
  return sub.update_tree(
      azimuth_updates={
          'left_leg_0': 90.0,
          'left_leg_1': 90.0,
          'right_leg_0': -90.0,
          'right_leg_1': -90.0,
      },
      elevation_updates={
          'left_leg_0': -45.0,
          'left_leg_1': -85.0,
          'right_leg_0': -45.0,
          'right_leg_1': -85.0,
      },
      width_updates={},
  )


def _solve_fixture(
    sf: bp.StickFigure,
    *,
    remove_central_crease: bool = False,
) -> bp.SolvedCreasePattern:
  """Packs and solves a StickFigure fixture deterministically."""
  design = bp.prepare(sf)
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
  for hinges_from_sf in (True, False):
    candidates = list(
        bp.solve(
            packed,
            bp.SolverConfig(
                max_steps=1500,
                max_time_limit=20.0,
                max_number_of_cps=1,
                disallow_trapping_edge_flaps=True,
                hinges_from_stick_figure=hinges_from_sf,
                remove_central_crease=remove_central_crease,
                up_down_method='flaps_down',
            ),
        )
    )
    if candidates:
      return candidates[0]
  return bp.fold(sf, preset='demo').solved


def _state(tree_name: str) -> shaping.ShapingState:
  """Returns a cached `ShapingState` for `tree_name`."""
  if tree_name not in _STATES:
    if tree_name == 'HUMAN_SUBDIVIDED':
      sf = _build_subdivided_human()
    else:
      sf = bp.load_example(tree_name)
    solved = _solve_fixture(sf, remove_central_crease=False)
    _STATES[tree_name] = shaping.ShapingState.from_solved(solved)
  return _STATES[tree_name]


def _crease_keys(state: shaping.ShapingState) -> set[tuple[float, ...]]:
  """Rounded endpoint and fold-direction keys of `state.pattern`."""
  keys = set()
  for crease in state.pattern.creases:
    ends = sorted([
        (round(crease.p1.x, 6), round(crease.p1.y, 6)),
        (round(crease.p2.x, 6), round(crease.p2.y, 6)),
    ])
    keys.add((*ends[0], *ends[1], crease.fold.value))
  return keys


def _segment(crease) -> tuple[float, float, float, float]:
  """Rounded, order-free endpoints of a crease."""
  ends = sorted([
      (round(crease.p1_init.x, 6), round(crease.p1_init.y, 6)),
      (round(crease.p2_init.x, 6), round(crease.p2_init.y, 6)),
  ])
  return (*ends[0], *ends[1])


def _with_fold_percent(pattern, segment, fold_percent):
  """Returns copy of `pattern` with crease on `segment` at `fold_percent`."""
  creases = [
      dataclasses.replace(c, fold_percent=fold_percent)
      if c.mountain_or_valley() and _segment(c) == segment
      else c
      for c in pattern.creases
  ]
  return shaping.crease_pattern_lib.CreasePattern(
      creases=creases, light_mode=True, tolerance=pattern.tolerance
  )


class ShapingToolsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='simple_fold',
          tree_name='EMBEDDED_ENGINE',
          tool='simple_fold',
          arguments={
              'element_name': 'nacelle_front',
              'angle': 90.0,
              'depth': 0.0,
          },
      ),
      dict(
          testcase_name='simple_fold_with_fold_percent',
          tree_name='EMBEDDED_ENGINE',
          tool='simple_fold',
          arguments={
              'element_name': 'nacelle_front',
              'angle': 90.0,
              'depth': 0.0,
              'fold_percent': 0.5,
          },
      ),
      dict(
          testcase_name='simple_fold_wrapped_angle',
          tree_name='EMBEDDED_ENGINE',
          tool='simple_fold',
          arguments={
              'element_name': 'nacelle_front',
              'angle': 190.0,
              'depth': 0.0,
          },
      ),
      dict(
          testcase_name='river_fold',
          tree_name='HUMAN_SIMPLE',
          tool='simple_fold',
          arguments={'element_name': 'torso', 'angle': 90.0, 'depth': 0.0},
      ),
      dict(
          testcase_name='simple_fold_prefixed_name',
          tree_name='HUMAN_SIMPLE',
          tool='shaping:simple_fold',
          arguments={'element_name': 'right_arm', 'angle': 90.0, 'depth': 0.0},
      ),
      dict(
          testcase_name='crimp_fold',
          tree_name='EMBEDDED_ENGINE',
          tool='crimp_fold',
          arguments={
              'element_name': 'nacelle_front',
              'angle': 90.0,
              'depth': 0.1,
          },
      ),
      dict(
          testcase_name='reverse_fold',
          tree_name='HUMAN_SIMPLE',
          tool='reverse_fold',
          arguments={
              'element_name': 'left_arm',
              'angle': 45.0,
              'orientation': 'horizontal',
          },
      ),
      dict(
          testcase_name='crimp_fold_leg',
          tree_name='HUMAN_SIMPLE',
          tool='crimp_fold',
          arguments={'element_name': 'right_leg', 'angle': 90.0, 'depth': 0.25},
      ),
      dict(
          testcase_name='squash_fold_leg',
          tree_name='HUMAN_SIMPLE',
          tool='squash_fold',
          arguments={'element_name': 'right_leg'},
      ),
      dict(
          testcase_name='squash_fold_leg_open_percent',
          tree_name='HUMAN_SIMPLE',
          tool='squash_fold',
          arguments={'element_name': 'right_leg', 'open_percent': 0.4},
      ),
      dict(
          testcase_name='clip_pattern_fold_straight_narrow',
          tree_name='HUMAN_SIMPLE',
          tool='clip_pattern_fold',
          arguments={
              'element_name': 'right_leg',
              'template_name': 'fold',
              'enable_narrow': True,
              'narrow_n_times': 2,
          },
      ),
      dict(
          testcase_name='clip_pattern_fold_bend_and_narrow',
          tree_name='HUMAN_SIMPLE',
          tool='clip_pattern_fold',
          arguments={
              'element_name': 'right_leg',
              'step': 1.0,
              'bend_angle': 90.0,
              'rotate_angle': -45.0,
              'enable_narrow': True,
              'narrow_n_times': 4,
          },
      ),
      dict(
          testcase_name='rotation_3d',
          tree_name='HUMAN_SIMPLE',
          tool='rotation_3d',
          arguments={
              'rotations': [
                  {'element_name': 'left_arm', 'el': 10.0, 'az': 45.0}
              ]
          },
      ),
      dict(
          testcase_name='shape_angle_width',
          tree_name='HUMAN_SIMPLE',
          tool='shape_angle_width',
          arguments={
              'updates': [{
                  'element_name': 'left_arm',
                  'el': 10.0,
                  'az': 45.0,
                  'narrow': True,
              }]
          },
      ),
      dict(
          testcase_name='fan_out_fold',
          tree_name='HUMAN_SIMPLE',
          tool='fan_out_fold',
          arguments={
              'element_name': 'left_leg',
              'orientation': 'horizontal',
              'move_delta': 0.5,
          },
      ),
      dict(
          testcase_name='fan_out_fold_open_percent',
          tree_name='HUMAN_SIMPLE',
          tool='fan_out_fold',
          arguments={
              'element_name': 'left_leg',
              'orientation': 'horizontal',
              'move_delta': 0.5,
              'open_percent': 0.5,
          },
      ),
      dict(
          testcase_name='fan_out_fold_heuristic',
          tree_name='HUMAN_SIMPLE',
          tool='fan_out_fold_heuristic',
          arguments={'element_name': 'left_leg', 'move_delta': 0.5},
      ),
      dict(
          testcase_name='geometric_simple_fold',
          tree_name='EMBEDDED_ENGINE',
          tool='geometric_simple_fold',
          arguments={
              'folds': [{
                  'element_names': ['nacelle_front'],
                  'start_x': 0.0,
                  'start_y': 0.0,
                  'end_x': 1.0,
                  'end_y': 1.0,
                  'fold_type': 'mountain',
              }]
          },
      ),
      dict(
          testcase_name='simple_fold_heuristic',
          tree_name='EMBEDDED_ENGINE',
          tool='simple_fold_heuristic',
          arguments={'element_name_list': ['nacelle_front']},
          expect_change=False,
      ),
  )
  def test_tool_succeeds_and_keeps_flat_foldability(
      self, tree_name, tool, arguments, expect_change=True
  ):
    state = _state(tree_name)
    new_state, result = shaping.apply_tool(state, tool, arguments)

    self.assertTrue(result.ok, result.message)
    if expect_change:
      self.assertIsNot(new_state.pattern, state.pattern)
    self.assertLen(new_state.history, 1)
    self.assertTrue(new_state.pattern.local_flat_foldability())
    self.assertEmpty(state.history)
    self.assertIs(state.pattern, state.base_pattern)

  @parameterized.named_parameters(
      dict(
          testcase_name='unknown_element',
          tool='simple_fold',
          arguments={'element_name': 'no_such_flap', 'angle': 30.0},
          expected_error='not found',
      ),
      dict(
          testcase_name='zero_angle',
          tool='simple_fold',
          arguments={'element_name': 'nacelle_front', 'angle': 0.0},
          expected_error='non-zero',
      ),
      dict(
          testcase_name='crimp_depth_zero',
          tool='crimp_fold',
          arguments={
              'element_name': 'nacelle_front',
              'angle': 90.0,
              'depth': 0.0,
          },
          expected_error='(0, 1)',
      ),
      dict(
          testcase_name='crimp_180',
          tool='crimp_fold',
          arguments={
              'element_name': 'nacelle_front',
              'angle': 180.0,
              'depth': 0.1,
          },
          expected_error='180',
      ),
      dict(
          testcase_name='unknown_tool',
          tool='origami:magic',
          arguments={},
          expected_error='Unknown tool',
      ),
  )
  def test_tool_failure_is_reported_without_changing_state(
      self, tool, arguments, expected_error
  ):
    state = _state('EMBEDDED_ENGINE')
    new_state, result = shaping.apply_tool(state, tool, arguments)

    self.assertFalse(result.ok)
    self.assertIn(expected_error, result.message)
    self.assertIs(new_state.pattern, state.pattern)
    self.assertLen(new_state.history, 1)
    self.assertFalse(new_state.history[0]['ok'])

  def test_from_solved_with_central_crease_removed(self):
    res = bp.fold('human', preset='demo')
    state = shaping.ShapingState.from_solved(res.solved)
    self.assertEqual(
        state.central_crease_removed, res.solved.central_crease_removed
    )
    next_state, result = shaping.apply_tool(
        state,
        'simple_fold',
        {'element_name': 'left_leg', 'angle': 45.0, 'depth': 0.0},
    )
    self.assertTrue(result.ok, result.message)
    self.assertTrue(next_state.pattern.local_flat_foldability())

  def test_narrowing_doubles_descendant_grid_size_and_reset_clears_it(self):
    state = _state('EMBEDDED_ENGINE')
    narrowed, result = shaping.apply_tool(
        state,
        'simple_fold',
        {'element_name': 'nacelle_front', 'angle': 90.0, 'narrow': True},
    )
    self.assertTrue(result.ok, result.message)
    self.assertEqual(narrowed.narrowed_elements, ('nacelle_front',))
    before = state.element('nacelle_front').grid_size
    after = narrowed.element('nacelle_front').grid_size
    self.assertEqual(after, 2 * before)

    reset = narrowed.reset()
    self.assertIs(reset.pattern, state.base_pattern)
    self.assertEmpty(reset.narrowed_elements)
    self.assertEmpty(reset.history)

  def test_knee_on_narrowed_leg_and_reverse_fold_on_narrowed_child(self):
    state = _state('HUMAN_SUBDIVIDED')
    narrowed, result = shaping.apply_tool(
        state,
        'simple_fold',
        {'element_name': 'left_leg_0', 'angle': 45.0, 'narrow': True},
    )
    self.assertTrue(result.ok, result.message)
    self.assertEqual(
        narrowed.element('left_leg_1').grid_size, 2 * state.solving_grid_size
    )
    for angle in (45.0, -45.0):
      with self.subTest(angle=angle):
        kneed, res_knee = shaping.apply_tool(
            narrowed,
            'simple_fold',
            {'element_name': 'left_leg_1', 'angle': angle},
        )
        self.assertTrue(res_knee.ok, res_knee.message)
        self.assertTrue(kneed.pattern.local_flat_foldability())

    reversed_state, res_rev = shaping.apply_tool(
        narrowed,
        'reverse_fold',
        {'element_name': 'left_leg_1', 'angle': 45.0},
    )
    self.assertTrue(res_rev.ok, res_rev.message)
    self.assertGreater(
        len(reversed_state.pattern.creases), len(narrowed.pattern.creases)
    )

  def test_knee_failure_on_length_one_segment_reports_both_variants(self):
    state = _state('HUMAN_SUBDIVIDED')
    narrowed, result = shaping.apply_tool(
        state,
        'simple_fold',
        {'element_name': 'left_arm_0', 'angle': 45.0, 'narrow': True},
    )
    self.assertTrue(result.ok, result.message)
    _, fail_res = shaping.apply_tool(
        narrowed, 'simple_fold', {'element_name': 'left_arm_1', 'angle': -45.0}
    )
    self.assertFalse(fail_res.ok)
    self.assertIn('small-angle conversion', fail_res.message)
    self.assertIn('plain zig-zag', fail_res.message)

  def test_simple_fold_order_picks_the_open_fold_side(self):
    state = _state('HUMAN_SUBDIVIDED')
    narrowed, result = shaping.apply_tool(
        state,
        'simple_fold',
        {'element_name': 'left_leg_0', 'angle': 45.0, 'narrow': True},
    )
    self.assertTrue(result.ok, result.message)
    arguments = {
        'element_name': 'left_leg_0',
        'angle': 180.0,
        'depth': 0.5,
        'fold_percent': 0.5,
    }
    shaped = {}
    for order in ('MV', 'VM'):
      bent, res_bend = shaping.apply_tool(
          narrowed, 'simple_fold', dict(arguments, fold_order=order)
      )
      self.assertTrue(res_bend.ok, res_bend.message)
      shaped[order] = _crease_keys(bent)
    self.assertNotEqual(shaped['MV'], shaped['VM'])
    _, bad_order = shaping.apply_tool(
        narrowed, 'simple_fold', dict(arguments, fold_order='XY')
    )
    self.assertFalse(bad_order.ok)
    self.assertIn('fold_order', bad_order.message)

  def test_rotation_3d_refuses_narrowed_and_reports_skipped_elements(self):
    state = _state('HUMAN_SIMPLE')
    narrowed, result = shaping.apply_tool(
        state,
        'simple_fold',
        {
            'element_name': 'right_arm',
            'angle': 45.0,
            'depth': 0.0,
            'narrow': True,
        },
    )
    self.assertTrue(result.ok, result.message)
    _, rot_res = shaping.apply_tool(
        narrowed,
        'rotation_3d',
        {'rotations': [{'element_name': 'right_arm', 'el': 30, 'az': 60}]},
    )
    self.assertFalse(rot_res.ok)
    self.assertIn('narrowed', rot_res.message)

    eng_state = _state('EMBEDDED_ENGINE')
    skipped_rot = {'element_name': 'nacelle_front', 'el': 0.0, 'az': 0.0}
    _, res_none = shaping.apply_tool(
        eng_state, 'rotation_3d', {'rotations': [skipped_rot]}
    )
    self.assertFalse(res_none.ok)
    self.assertIn('no creases', res_none.message)

    mixed = [
        skipped_rot,
        {'element_name': 'nacelle_rear', 'el': 35.0, 'az': 0.0},
    ]
    rotated, res_mixed = shaping.apply_tool(
        eng_state, 'rotation_3d', {'rotations': mixed}
    )
    self.assertTrue(res_mixed.ok, res_mixed.message)
    self.assertGreater(
        len(rotated.pattern.creases), len(eng_state.pattern.creases)
    )
    self.assertIn('left out: nacelle_front', res_mixed.message)

    _, strict = shaping.apply_tool(
        eng_state, 'rotation_3d', {'rotations': mixed, 'require_all': True}
    )
    self.assertFalse(strict.ok)
    self.assertIn('nacelle_front', strict.message)

  def test_apply_tools_and_context_helpers(self):
    state = _state('HUMAN_SIMPLE')
    calls = [
        {
            'name': 'simple_fold',
            'arguments': {'element_name': 'right_arm', 'angle': 90.0},
        },
        {'name': 'simple_fold', 'arguments': {'element_name': 'nope'}},
    ]
    new_state, results = shaping.apply_tools(state, calls)
    self.assertLen(results, 2)
    self.assertTrue(results[0].ok, results[0].message)
    self.assertFalse(results[1].ok)
    self.assertLen(new_state.history, 2)
    self.assertIn('simple_fold', shaping.history_json(new_state))

    table = shaping.element_table(state)
    self.assertStartsWith(table, 'central_fold_removed=False')
    self.assertIn('right_arm:', table)
    info = shaping.flap_orientation_info(state)
    self.assertIn('root_stick', info)
    self.assertNotEmpty(info['orientations'])
    description = shaping.describe(state)
    self.assertIn('right_arm', description['flaps'])
    self.assertEqual(description['num_tool_calls'], 0)
    self.assertIsInstance(shaping.level_shifter_positions(state), dict)

    _, ls_res = shaping.apply_tool(
        state,
        'level_shifter',
        {'element_name': 'right_arm', 'position_percent': 0.5},
    )
    self.assertFalse(ls_res.ok)
    self.assertIn('use_level_shifter=True', ls_res.message)


class Fold3dFallbackTest(absltest.TestCase):

  def test_reverse_fold_falls_back_to_reverse_fold_3d(self):
    state = _state('HUMAN_SIMPLE')
    arguments = {'element_name': 'left_arm', 'angle': 45.0}
    with mock.patch.object(
        shaping.reverse_fold_shaping,
        'assign_reverse_fold_orders',
        side_effect=ValueError('forced zig-zag failure'),
    ):
      folded, result = shaping.apply_tool(state, 'reverse_fold', arguments)
    self.assertTrue(result.ok, result.message)
    self.assertIn('applied reverse_fold_3d instead', result.message)
    self.assertTrue(folded.pattern.local_flat_foldability())

  def test_crimp_fold_falls_back_to_crimp_fold_3d(self):
    state = _state('HUMAN_SIMPLE')
    arguments = {'element_name': 'right_leg', 'angle': 45.0, 'depth': 0.4}
    with mock.patch.object(
        shaping.crimp_fold_shaping,
        'fix_flat_foldability',
        side_effect=ValueError('forced zig-zag failure'),
    ):
      folded, result = shaping.apply_tool(state, 'crimp_fold', arguments)
    self.assertTrue(result.ok, result.message)
    self.assertIn('applied crimp_fold_3d instead', result.message)
    self.assertTrue(folded.pattern.local_flat_foldability())


class ExtrasAndOverlayTest(absltest.TestCase):

  def test_probe_and_all_28_tools_registered(self):
    self.assertEqual(shaping.list_shaping_tools(), shaping.TOOL_NAMES)
    self.assertLen(shaping.TOOL_NAMES, 28)

    state = _state('HUMAN_SIMPLE')
    report = shaping.probe(state, element_names=['left_arm'])
    outcomes = report['left_arm']
    self.assertTrue(any('reverse_fold_3d' in k for k in outcomes))
    self.assertTrue(any('crimp_fold_3d' in k for k in outcomes))
    self.assertEmpty(state.history)

  def test_thin_limbs_3d_and_symmetric_dispatch(self):
    state = _state('HUMAN_SUBDIVIDED')
    errors = shaping.limb_direction_errors(
        state, [('left_leg_0', 'left_leg_1')]
    )
    self.assertCountEqual(errors, ['left_leg_0', 'left_leg_1'])

    calls = []

    def evaluate(candidate, pairs):
      calls.append(candidate)
      score = 10000.0 - len(candidate.pattern.creases)
      return {name: score for pair in pairs for name in pair}

    shaped, result = shaping.apply_tool(
        state,
        'thin_limbs_3d',
        {
            'limbs': [
                {'proximal': 'left_leg_0'},
                {'proximal': 'right_leg_0'},
            ],
            'evaluate': evaluate,
        },
    )
    self.assertTrue(result.ok, result.message)
    self.assertCountEqual(
        shaped.narrowed_elements, ['left_leg_0', 'right_leg_0']
    )
    self.assertTrue(shaped.pattern.local_flat_foldability())

    plans = result.arguments['limbs']
    calls.clear()
    replayed, replay_res = shaping.apply_tool(
        state, 'thin_limbs_3d', {'limbs': plans, 'evaluate': evaluate}
    )
    self.assertTrue(replay_res.ok, replay_res.message)
    self.assertLen(calls, 1)
    self.assertEqual(_crease_keys(replayed), _crease_keys(shaped))

    sym_partner = shaping.resolve_symmetric_element(state, 'left_leg_0')
    self.assertEqual(sym_partner, 'right_leg_0')

    st_sym, r_sym = shaping.apply_tool(
        state,
        'apply_symmetric',
        {
            'tool': 'simple_fold',
            'element_name': 'left_leg_0',
            'angle': 35.0,
            'depth': 0.3,
        },
    )
    self.assertTrue(r_sym.ok, r_sym.message)
    self.assertGreater(len(st_sym.pattern.creases), len(state.pattern.creases))
    replayed_sym, r_rep = shaping.apply_tool(
        state, 'apply_symmetric', dict(r_sym.arguments)
    )
    self.assertTrue(r_rep.ok, r_rep.message)
    self.assertEqual(_crease_keys(replayed_sym), _crease_keys(st_sym))

  def test_open_flap_and_3d_overlay_backbone_replay(self):
    human_state = _state('HUMAN_SIMPLE')
    _, r_spine = shaping.apply_tool(
        human_state,
        'fold_spine_3d',
        {'spine_fold_percent': 0.6, 'fold_type': 'valley'},
    )
    self.assertTrue(r_spine.ok, r_spine.message)

    eng_state = _state('EMBEDDED_ENGINE')
    opened, r_open = shaping.apply_tool(
        eng_state,
        'open_flap',
        {'element_name': 'nacelle_rear', 'fold_percent': 0.5},
    )
    self.assertTrue(r_open.ok, r_open.message)
    self.assertIsNotNone(opened.flat_backbone)
    assert opened.flat_backbone is not None
    self.assertTrue(opened.flat_backbone.local_flat_foldability())

    composed, r_comp = shaping.apply_tool(
        opened,
        'rotation_3d',
        {'rotations': [{'element_name': 'pylon', 'el': 35.0, 'az': 45.0}]},
    )
    self.assertTrue(r_comp.ok, r_comp.message)
    self.assertIsNotNone(composed.flat_backbone)
    self.assertTrue(any(c.fold_percent < 1.0 for c in composed.pattern.creases))
    model = fidelity.FoldedModel.from_pattern(composed.pattern)
    self.assertLessEqual(model.vertex_error, 1.0)

    real_from_pattern = fidelity.FoldedModel.from_pattern

    def high_error_from_pattern(pattern, *args, **kwargs):
      folded_model = real_from_pattern(pattern, *args, **kwargs)
      return mock.Mock(wraps=folded_model, vertex_error=5.5)

    with (
        mock.patch.object(
            fidelity,
            'replay_3d_overlay',
            side_effect=ValueError('forced replay failure'),
        ),
        mock.patch.object(
            fidelity.FoldedModel,
            'from_pattern',
            side_effect=high_error_from_pattern,
        ),
    ):
      _, r_bad = shaping.apply_tool(
          opened,
          'rotation_3d',
          {'rotations': [{'element_name': 'pylon', 'el': 35.0, 'az': 45.0}]},
      )
    self.assertFalse(r_bad.ok)
    self.assertIn('3-D overlay fold vertex error 5.50% > 1.0%', r_bad.message)

    target = next(
        c
        for c in opened.pattern.creases
        if c.mountain_or_valley() and c.fold_percent < 1.0
    )
    segment = _segment(target)
    seen_backbone: list[bool] = []

    @shaping._preserve_3d_overlay  # pylint: disable=protected-access
    def flare_only(st, *, verify_fold=True):
      del verify_fold
      seen_backbone.append(st.flat_backbone is None)
      return st.with_pattern(_with_fold_percent(st.pattern, segment, 0.75))

    out = flare_only(opened, verify_fold=False)
    self.assertEqual(seen_backbone, [True, True])
    self.assertTrue(
        any(
            _segment(c) == segment and abs(c.fold_percent - 0.75) < 1e-6
            for c in out.pattern.creases
        )
    )


if __name__ == '__main__':
  absltest.main()
