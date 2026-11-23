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

"""Unit tests for `bp_pipeline.shaping` state and actuator toolbox."""

import dataclasses
import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline import folder_3d
from bp_pipeline.crease_pattern import crease_pattern as crease_pattern_lib
from bp_pipeline.shaping import fan_squash
from bp_pipeline.shaping import fold_shaping
from bp_pipeline.shaping import heuristics
from bp_pipeline.shaping import overlay_3d
from bp_pipeline.shaping import state as state_lib
from bp_pipeline.shaping import toolbox

_SOLVED_CACHE: dict[str, bp.SolvedCreasePattern] = {}
_STATE_CACHE: dict[str, toolbox.ShapingState] = {}


def _build_subdivided_human() -> bp.StickFigure:
  """Builds a symmetric subdivided-limb human fixture for joint tests."""
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
  """Packs and solves `sf` deterministically with a small test budget."""
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


def _build_bilateral_tail_fixture() -> bp.StickFigure:
  """Builds a small symmetric bird-like tree with a long central tail flap."""
  return bp.StickFigure.from_dict({
      'name': 'bilateral_tail_fixture',
      'category': 'birds',
      'complexity': 1,
      'color_top': [200, 180, 150],
      'color_bottom': [200, 180, 150],
      'root': 'body',
      'children': [
          {
              'name': 'body',
              'start': 'hips',
              'end': 'shoulders',
              'azimuth_angle': 0.0,
              'elevation_angle': 0.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'tail',
              'start': 'hips',
              'end': 'tail_tip',
              'azimuth_angle': 180.0,
              'elevation_angle': 0.0,
              'length': 4,
              'width': 2,
              'shaping_flap': False,
          },
          {
              'name': 'leg_l',
              'start': 'hips',
              'end': 'foot_l',
              'azimuth_angle': 90.0,
              'elevation_angle': -60.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'leg_r',
              'start': 'hips',
              'end': 'foot_r',
              'azimuth_angle': -90.0,
              'elevation_angle': -60.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'head',
              'start': 'shoulders',
              'end': 'head_tip',
              'azimuth_angle': 0.0,
              'elevation_angle': 20.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'wing_l',
              'start': 'shoulders',
              'end': 'wing_l_tip',
              'azimuth_angle': 90.0,
              'elevation_angle': 0.0,
              'length': 3,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'wing_r',
              'start': 'shoulders',
              'end': 'wing_r_tip',
              'azimuth_angle': -90.0,
              'elevation_angle': 0.0,
              'length': 3,
              'width': 1,
              'shaping_flap': False,
          },
      ],
  })


def _solved(tree_name: str) -> bp.SolvedCreasePattern:
  """Returns a cached `SolvedCreasePattern` for `tree_name`."""
  if tree_name not in _SOLVED_CACHE:
    if tree_name == 'HUMAN_SUBDIVIDED':
      sf = _build_subdivided_human()
    elif tree_name == 'BILATERAL_TAIL':
      sf = _build_bilateral_tail_fixture()
    else:
      sf = bp.load_example(tree_name)
    _SOLVED_CACHE[tree_name] = _solve_fixture(sf)
  return _SOLVED_CACHE[tree_name]


def _state(tree_name: str) -> toolbox.ShapingState:
  """Returns a cached unshaped `ShapingState` for `tree_name`."""
  if tree_name not in _STATE_CACHE:
    _STATE_CACHE[tree_name] = toolbox.ShapingState.from_solved(
        _solved(tree_name)
    )
  return _STATE_CACHE[tree_name]


def _crease_keys(state: toolbox.ShapingState) -> set[tuple[object, ...]]:
  """Returns rounded endpoint and fold-type keys of `state.pattern`."""
  keys: set[tuple[object, ...]] = set()
  for crease in state.pattern.creases:
    ends = sorted([
        (round(crease.p1.x, 6), round(crease.p1.y, 6)),
        (round(crease.p2.x, 6), round(crease.p2.y, 6)),
    ])
    keys.add((*ends[0], *ends[1], crease.fold.value))
  return keys


class ShapingStateTest(absltest.TestCase):

  def test_from_solved_preserves_caller_tolerance_and_metadata_round_trip(self):
    solved = _solved('HUMAN_SIMPLE')
    orig_tolerance = solved.creases.tolerance
    custom_cfg = toolbox.ToolConfig(cp_tolerance=orig_tolerance * 2.0)
    state = toolbox.ShapingState.from_solved(solved, config=custom_cfg)
    self.assertEqual(solved.creases.tolerance, orig_tolerance)
    self.assertEqual(state.pattern.tolerance, orig_tolerance * 2.0)

    folded, res_fold = toolbox.apply_tool(
        state,
        'simple_fold',
        {'element_name': 'left_arm', 'angle': 45.0, 'narrow': True},
    )
    self.assertTrue(res_fold.ok, res_fold.message)
    spined, res_spine = toolbox.apply_tool(
        folded,
        'fold_spine',
        {'fold_percent': 0.7, 'verify_fold': False},
    )
    self.assertTrue(res_spine.ok, res_spine.message)
    self.assertIsNotNone(spined.flat_backbone)

    meta = spined.to_metadata_dict()
    self.assertNotIn('pattern_svg', meta)
    json.dumps(meta)  # Must be JSON-serializable.
    pattern_svg = toolbox.serialize_pattern_svg(spined.pattern)
    assert spined.flat_backbone is not None
    backbone_svg = toolbox.serialize_pattern_svg(spined.flat_backbone)
    restored = toolbox.ShapingState.from_metadata_dict(
        solved,
        meta,
        pattern_svg=pattern_svg,
        backbone_svg=backbone_svg,
    )
    self.assertEqual(restored.narrowed_elements, spined.narrowed_elements)
    self.assertEqual(restored.history, spined.history)
    self.assertIsNotNone(restored.flat_backbone)
    self.assertEqual(_crease_keys(restored), _crease_keys(spined))

    reset = restored.reset()
    self.assertIs(reset.pattern, restored.base_pattern)
    self.assertEmpty(reset.history)
    self.assertEmpty(reset.narrowed_elements)
    self.assertIsNone(reset.flat_backbone)

  def test_inspection_helpers_and_registry(self):
    self.assertEqual(toolbox.list_shaping_tools(), toolbox.TOOL_NAMES)
    self.assertLen(toolbox.TOOL_NAMES, 19)
    self.assertEqual(
        toolbox.normalize_tool_name('shaping:crimp_fold'), 'crimp_fold'
    )

    state = _state('HUMAN_SIMPLE')
    rows = toolbox.element_table(state)
    row_names = {row['name'] for row in rows}
    self.assertIn('right_arm', row_names)
    self.assertIn('torso', row_names)

    orient = toolbox.flap_orientation_info(state)
    self.assertIn('right_arm', orient)

    desc = toolbox.describe(state)
    self.assertIn('elements', desc)
    self.assertEqual(desc['history_length'], 0)

    self.assertEqual(
        toolbox.resolve_symmetric_element(state, 'left_leg'), 'right_leg'
    )
    self.assertTrue(state_lib.is_left_element(state, 'left_leg'))
    self.assertFalse(state_lib.is_left_element(state, 'right_leg'))

    report, dirs = toolbox.measured_directions(state)
    self.assertIn('fidelity', report)
    self.assertIn('left_leg', dirs)


class PrimitiveAndReliefToolsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='simple_fold_explicit',
          tree_name='EMBEDDED_ENGINE',
          tool='simple_fold',
          arguments={
              'element_name': 'nacelle_front',
              'angle': 90.0,
              'depth': 0.0,
          },
      ),
      dict(
          testcase_name='simple_fold_partial',
          tree_name='EMBEDDED_ENGINE',
          tool='simple_fold',
          arguments={
              'element_name': 'nacelle_front',
              'angle': 90.0,
              'depth': 0.0,
              'fold_percent': 0.5,
          },
          expect_flat=False,
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
          testcase_name='simple_fold_river',
          tree_name='HUMAN_SIMPLE',
          tool='simple_fold',
          arguments={'element_name': 'torso', 'angle': 90.0, 'depth': 0.0},
      ),
      dict(
          testcase_name='simple_fold_heuristic_angle',
          tree_name='HUMAN_SIMPLE',
          tool='simple_fold',
          arguments={'element_name': 'left_arm'},
      ),
      dict(
          testcase_name='simple_fold_prefixed',
          tree_name='HUMAN_SIMPLE',
          tool='shaping:simple_fold',
          arguments={'element_name': 'right_arm', 'angle': 90.0, 'depth': 0.0},
      ),
      dict(
          testcase_name='crimp_fold_2d',
          tree_name='EMBEDDED_ENGINE',
          tool='crimp_fold',
          arguments={
              'element_name': 'nacelle_front',
              'angle': 90.0,
              'depth': 0.1,
              'mode': '2d',
          },
      ),
      dict(
          testcase_name='crimp_fold_3d',
          tree_name='HUMAN_SIMPLE',
          tool='crimp_fold',
          arguments={
              'element_name': 'right_leg',
              'angle': 45.0,
              'depth': 0.4,
              'mode': '3d',
          },
      ),
      dict(
          testcase_name='reverse_fold_auto',
          tree_name='HUMAN_SIMPLE',
          tool='reverse_fold',
          arguments={
              'element_name': 'left_arm',
              'angle': 45.0,
              'orientation': 'horizontal',
          },
      ),
      dict(
          testcase_name='reverse_fold_2d_inscribed',
          tree_name='HUMAN_SIMPLE',
          tool='reverse_fold',
          arguments={
              'element_name': 'left_leg',
              'angle': 180.0,
              'depth': 0.25,
              'mode': '2d',
              'verify_fold': False,
          },
      ),
      dict(
          testcase_name='reverse_fold_3d',
          tree_name='HUMAN_SIMPLE',
          tool='reverse_fold',
          arguments={
              'element_name': 'left_arm',
              'angle': 45.0,
              'mode': '3d',
          },
      ),
      dict(
          testcase_name='geometric_simple_fold_single',
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
          testcase_name='clip_pattern_straight_narrow',
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
          testcase_name='clip_pattern_bend_and_narrow',
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
          testcase_name='narrow_central_flap_nacelle',
          tree_name='EMBEDDED_ENGINE',
          tool='narrow_central_flap',
          arguments={
              'element_name': 'nacelle_rear',
              'tip_width': 0.3,
              'verify_fold': False,
          },
      ),
      dict(
          testcase_name='fan_out_fold_explicit',
          tree_name='HUMAN_SIMPLE',
          tool='fan_out_fold',
          arguments={
              'element_name': 'left_leg',
              'orientation': 'horizontal',
              'move_delta': 0.5,
              'verify_fold': False,
          },
      ),
      dict(
          testcase_name='fan_out_fold_auto_orientation',
          tree_name='HUMAN_SIMPLE',
          tool='fan_out_fold',
          arguments={
              'element_name': 'left_leg',
              'move_delta': 0.5,
              'verify_fold': False,
          },
      ),
      dict(
          testcase_name='fan_out_fold_partial_open',
          tree_name='HUMAN_SIMPLE',
          tool='fan_out_fold',
          arguments={
              'element_name': 'left_leg',
              'orientation': 'horizontal',
              'move_delta': 0.5,
              'open_percent': 0.5,
          },
          expect_flat=False,
      ),
      dict(
          testcase_name='squash_fold_flat',
          tree_name='HUMAN_SIMPLE',
          tool='squash_fold',
          arguments={
              'element_name': 'right_leg',
              'open_percent': 1.0,
              'verify_fold': False,
          },
      ),
      # Note: open_percent=0.45 on HUMAN_SIMPLE right_leg cannot fold cleanly at
      # 0.45; under 155212 it silently relaxed to 0.65 on ridge 1 whose
      # non-rigid BFS vertex strain spans 0.47%..7.57% across its 42 anchor
      # faces (0.906% on Tom's face 0 vs 1.406% on folder_3d.py's face 0; both
      # fail clean_gate <= 0.85%). open_percent=0.65 squashes ridge 0 cleanly at
      # 0.696% (<= 0.85%) identically across both folders.
      dict(
          testcase_name='squash_fold_partial_3d',
          tree_name='HUMAN_SIMPLE',
          tool='squash_fold',
          arguments={'element_name': 'right_leg', 'open_percent': 0.65},
          expect_flat=False,
      ),
      dict(
          testcase_name='inscribed_flap_element',
          tree_name='HUMAN_SIMPLE',
          tool='inscribed_flap',
          arguments={
              'element_name': 'left_leg',
              'depths': [1],
              'verify_fold': False,
          },
      ),
      dict(
          testcase_name='inscribed_flap_river',
          tree_name='HUMAN_SIMPLE',
          tool='inscribed_flap',
          arguments={
              'element_name': 'torso',
              'depths': [1],
              'verify_fold': False,
          },
      ),
      dict(
          testcase_name='fold_spine_auto',
          tree_name='HUMAN_SIMPLE',
          tool='fold_spine',
          arguments={'fold_percent': 0.6, 'fold_type': 'valley'},
          expect_flat=False,
      ),
      dict(
          testcase_name='fold_spine_explicit_river',
          tree_name='HUMAN_SIMPLE',
          tool='fold_spine',
          arguments={
              'element_name': 'torso',
              'fold_percent': 0.7,
              'verify_fold': False,
          },
          expect_flat=False,
      ),
  )
  def test_tool_succeeds_without_mutating_input_state(
      self,
      tree_name: str,
      tool: str,
      arguments: dict[str, object],
      expect_flat: bool = True,
  ):
    state = _state(tree_name)
    new_state, result = toolbox.apply_tool(state, tool, arguments)
    self.assertTrue(result.ok, result.message)
    self.assertIsNot(new_state.pattern, state.pattern)
    self.assertLen(new_state.history, 1)
    if expect_flat:
      self.assertTrue(new_state.pattern.local_flat_foldability())
    else:
      self.assertTrue(overlay_3d.has_3d_overlay(new_state.pattern))
      ok, msg, _ = toolbox.verify_3d_fold(new_state.pattern)
      self.assertTrue(ok, msg)
    self.assertEmpty(state.history)
    self.assertIs(state.pattern, state.base_pattern)

  @parameterized.named_parameters(
      dict(
          testcase_name='unknown_element',
          tool='simple_fold',
          arguments={'element_name': 'no_such_flap', 'angle': 30.0},
          expected_error='no_such_flap',
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
          testcase_name='bad_fold_order',
          tool='simple_fold',
          arguments={
              'element_name': 'nacelle_front',
              'angle': 45.0,
              'fold_order': 'XY',
          },
          expected_error='fold_order',
      ),
      dict(
          testcase_name='unknown_tool',
          tool='origami:magic',
          arguments={},
          expected_error='Unknown tool',
      ),
  )
  def test_tool_failure_is_recorded_without_altering_pattern(
      self, tool: str, arguments: dict[str, object], expected_error: str
  ):
    state = _state('EMBEDDED_ENGINE')
    new_state, result = toolbox.apply_tool(state, tool, arguments)
    self.assertFalse(result.ok)
    self.assertIn(expected_error, result.message)
    self.assertIs(new_state.pattern, state.pattern)
    self.assertLen(new_state.history, 1)
    self.assertFalse(new_state.history[0]['ok'])

  def test_reverse_fold_preserves_explicit_fold_type_and_auto_fallbacks(self):
    state = _state('HUMAN_SIMPLE')
    m_state, m_res = toolbox.apply_tool(
        state,
        'reverse_fold',
        {
            'element_name': 'left_leg',
            'angle': 180.0,
            'depth': 0.25,
            'fold_type': 'mountain',
            'mode': '2d',
            'verify_fold': False,
        },
    )
    v_state, v_res = toolbox.apply_tool(
        state,
        'reverse_fold',
        {
            'element_name': 'left_leg',
            'angle': 180.0,
            'depth': 0.25,
            'fold_type': 'valley',
            'mode': '2d',
            'verify_fold': False,
        },
    )
    self.assertTrue(m_res.ok, m_res.message)
    self.assertTrue(v_res.ok, v_res.message)
    self.assertNotEqual(_crease_keys(m_state), _crease_keys(v_state))

    seen_first_fold: list[str] = []
    real_assign = fold_shaping.assign_reverse_fold_orders

    def _spy_assign(instructions, fold, *args, **kwargs):
      if not seen_first_fold:
        seen_first_fold.append(fold.name)
      return real_assign(instructions, fold, *args, **kwargs)

    with mock.patch.object(
        fold_shaping,
        'assign_reverse_fold_orders',
        side_effect=_spy_assign,
    ):
      toolbox.apply_tool(
          state,
          'reverse_fold',
          {
              'element_name': 'left_arm',
              'angle': 45.0,
              'fold_type': 'mountain',
              'mode': '2d',
          },
      )
    self.assertEqual(seen_first_fold, ['MOUNTAIN'])

    with mock.patch.object(
        fold_shaping,
        'assign_reverse_fold_orders',
        side_effect=ValueError('forced 2d reverse failure'),
    ):
      fb_rev, res_rev = toolbox.apply_tool(
          state, 'reverse_fold', {'element_name': 'left_arm', 'angle': 45.0}
      )
    self.assertTrue(res_rev.ok, res_rev.message)
    self.assertTrue(fb_rev.pattern.local_flat_foldability())
    self.assertIn('fallback from 2d', res_rev.message)
    self.assertIn('forced 2d reverse failure', res_rev.message)
    self.assertEqual(res_rev.arguments.get('mode'), '3d')

    with mock.patch.object(
        fold_shaping,
        'fix_crimp_flat_foldability',
        side_effect=ValueError('forced 2d crimp failure'),
    ):
      fb_crimp, res_crimp = toolbox.apply_tool(
          state,
          'crimp_fold',
          {'element_name': 'right_leg', 'angle': 45.0, 'depth': 0.4},
      )
    self.assertTrue(res_crimp.ok, res_crimp.message)
    self.assertTrue(fb_crimp.pattern.local_flat_foldability())
    self.assertIn('fallback from 2d', res_crimp.message)
    self.assertIn('forced 2d crimp failure', res_crimp.message)
    self.assertEqual(res_crimp.arguments.get('mode'), '3d')

    # Verify mode='2d' rejects gap instead of silently running 3d.
    _, res_gap_err = toolbox.apply_tool(
        state,
        'crimp_fold',
        {
            'element_name': 'left_leg',
            'angle': 45.0,
            'depth': 0.4,
            'mode': '2d',
            'gap': 0.1,
        },
    )
    self.assertFalse(res_gap_err.ok)
    self.assertIn("mode='2d'", res_gap_err.message)

    # Verify simple_fold(angle=None) converts shaping_angles radians to degrees.
    _, res_heur = toolbox.apply_tool(
        state,
        'simple_fold',
        {'element_name': 'left_arm'},
    )
    self.assertTrue(res_heur.ok, res_heur.message)
    self.assertAlmostEqual(abs(res_heur.arguments['angle']), 15.0, places=3)

  def test_narrowing_descendant_grid_doubling_and_fold_order_symmetry(self):
    state = _state('HUMAN_SUBDIVIDED')
    narrowed, res = toolbox.apply_tool(
        state,
        'simple_fold',
        {'element_name': 'left_leg_0', 'angle': 45.0, 'narrow': True},
    )
    self.assertTrue(res.ok, res.message)
    self.assertEqual(narrowed.narrowed_elements, ('left_leg_0',))
    self.assertEqual(
        narrowed.element('left_leg_1').grid_size, 2 * state.solving_grid_size
    )

    for angle in (45.0, -45.0):
      kneed, res_knee = toolbox.apply_tool(
          narrowed,
          'simple_fold',
          {'element_name': 'left_leg_1', 'angle': angle},
      )
      self.assertTrue(res_knee.ok, res_knee.message)
      self.assertTrue(kneed.pattern.local_flat_foldability())

    orders = {}
    for order in ('MV', 'VM'):
      bent, res_bend = toolbox.apply_tool(
          narrowed,
          'simple_fold',
          {
              'element_name': 'left_leg_0',
              'angle': 180.0,
              'depth': 0.5,
              'fold_percent': 0.5,
              'fold_order': order,
          },
      )
      self.assertTrue(res_bend.ok, res_bend.message)
      orders[order] = _crease_keys(bent)
    self.assertNotEqual(orders['MV'], orders['VM'])

  def test_open_flap_river_and_3d_overlay_backbone_replay(self):
    human = _state('HUMAN_SIMPLE')
    river_opened, r_riv = toolbox.apply_tool(
        human, 'open_flap', {'element_name': 'torso', 'fold_percent': 0.65}
    )
    self.assertTrue(r_riv.ok, r_riv.message)
    self.assertIsNotNone(river_opened.flat_backbone)

    eng = _state('EMBEDDED_ENGINE')
    opened, r_open = toolbox.apply_tool(
        eng, 'open_flap', {'element_name': 'nacelle_rear', 'fold_percent': 0.5}
    )
    self.assertTrue(r_open.ok, r_open.message)
    assert opened.flat_backbone is not None
    self.assertTrue(opened.flat_backbone.local_flat_foldability())

    composed, r_comp = toolbox.apply_tool(
        opened,
        'rotation_3d',
        {'rotations': [{'element_name': 'pylon', 'el': 35.0, 'az': 45.0}]},
    )
    self.assertTrue(r_comp.ok, r_comp.message)
    self.assertIsNotNone(composed.flat_backbone)
    self.assertTrue(any(c.fold_percent < 1.0 for c in composed.pattern.creases))
    model = folder_3d.fold_3d(composed.pattern)
    self.assertLessEqual(
        model.vertex_error, toolbox.MAX_RENDERABLE_VERTEX_ERROR
    )

    real_fold_3d = folder_3d.fold_3d

    def _high_strain(pattern, *args, **kwargs):
      folded = real_fold_3d(pattern, *args, **kwargs)
      return mock.Mock(wraps=folded, vertex_error=5.5)

    with (
        mock.patch.object(
            overlay_3d,
            'replay_3d_overlay',
            side_effect=ValueError('forced replay failure'),
        ),
        mock.patch.object(
            folder_3d,
            'fold_3d',
            side_effect=_high_strain,
        ),
    ):
      _, r_bad = toolbox.apply_tool(
          opened,
          'rotation_3d',
          {'rotations': [{'element_name': 'pylon', 'el': 35.0, 'az': 45.0}]},
      )
    self.assertFalse(r_bad.ok)
    self.assertIn('5.50%', r_bad.message)

    target_key = next(
        state_lib.crease_key(c)
        for c in opened.pattern.creases
        if c.mountain_or_valley() and c.fold_percent < 1.0
    )

    @toolbox.preserve_3d_overlay
    def _adjust_overlay(
        st: toolbox.ShapingState, *, verify_fold: bool = True
    ) -> toolbox.ShapingState:
      del verify_fold
      creases = [
          dataclasses.replace(c, fold_percent=0.75)
          if c.mountain_or_valley() and state_lib.crease_key(c) == target_key
          else c
          for c in st.pattern.creases
      ]
      updated = crease_pattern_lib.CreasePattern(
          creases=creases,
          grid_size=-1,
          light_mode=True,
          tolerance=st.pattern.tolerance,
      )
      return st.with_pattern(updated, allow_3d_vertices=True)

    replayed = _adjust_overlay(opened, verify_fold=False)
    self.assertTrue(
        any(
            state_lib.crease_key(c) == target_key
            and abs(c.fold_percent - 0.75) < 1e-6
            for c in replayed.pattern.creases
        )
    )


class CompositeAnd3DToolsTest(absltest.TestCase):

  def test_rotation_3d_narrow_guards_require_all_and_no_monkeypatch(self):
    orig_fn = heuristics.get_first_stick_for_shaping
    human = _state('HUMAN_SIMPLE')
    rot_narrow, res_rn = toolbox.apply_tool(
        human,
        'rotation_3d',
        {
            'rotations': [
                {
                    'element_name': 'left_arm',
                    'az': 45.0,
                    'el': 10.0,
                    'narrow': True,
                },
            ],
            'use_tree_root': True,
        },
    )
    self.assertTrue(res_rn.ok, res_rn.message)
    self.assertIs(heuristics.get_first_stick_for_shaping, orig_fn)
    self.assertIn('left_arm', rot_narrow.narrowed_elements)

    _, res_again = toolbox.apply_tool(
        rot_narrow,
        'rotation_3d',
        {'rotations': [{'element_name': 'left_arm', 'az': 60.0, 'el': 30.0}]},
    )
    self.assertFalse(res_again.ok)
    self.assertIn('narrowed', res_again.message)

    eng = _state('EMBEDDED_ENGINE')
    mixed = [
        {'element_name': 'nacelle_front', 'az': 0.0, 'el': 0.0},
        {'element_name': 'nacelle_rear', 'az': 0.0, 'el': 35.0},
    ]
    partial, res_partial = toolbox.apply_tool(
        eng, 'rotation_3d', {'rotations': mixed}
    )
    self.assertTrue(res_partial.ok, res_partial.message)
    self.assertGreater(len(partial.pattern.creases), len(eng.pattern.creases))

    _, res_strict = toolbox.apply_tool(
        eng, 'rotation_3d', {'rotations': mixed, 'require_all': True}
    )
    self.assertFalse(res_strict.ok)
    self.assertIn('nacelle_front', res_strict.message)

  def test_bend_to_direction_and_deterministic_replay(self):
    state = _state('HUMAN_SUBDIVIDED')
    bent, res = toolbox.apply_tool(
        state,
        'bend_to_direction',
        {
            'element_name': 'left_leg_1',
            'az': 60.0,
            'el': -30.0,
            'tolerance': 2.0,
        },
    )
    self.assertTrue(res.ok, res.message)
    chosen = res.arguments['chosen']
    replayed, rep_res = toolbox.apply_tool(
        state,
        'bend_to_direction',
        {
            'element_name': 'left_leg_1',
            'az': 60.0,
            'el': -30.0,
            'chosen': chosen,
        },
    )
    self.assertTrue(rep_res.ok, rep_res.message)
    self.assertEqual(_crease_keys(replayed), _crease_keys(bent))

  def test_thin_limbs_3d_and_apply_symmetric_with_replay(self):
    state = _state('HUMAN_SUBDIVIDED')
    errs = toolbox.limb_direction_errors(state)
    self.assertIn('left_leg_0', errs)
    self.assertIn('left_leg_1', errs)

    eval_calls: list[int] = []

    def _evaluate(candidate: toolbox.ShapingState) -> dict[str, float]:
      eval_calls.append(len(candidate.pattern.creases))
      score = float(len(candidate.pattern.creases))
      return {
          'left_leg_0': score,
          'left_leg_1': score,
          'right_leg_0': score,
          'right_leg_1': score,
      }

    shaped, res = toolbox.apply_tool(
        state,
        'thin_limbs_3d',
        {
            'limbs': [
                {'proximal': 'left_leg_0'},
                {'proximal': 'right_leg_0'},
            ],
            'evaluate': _evaluate,
        },
    )
    self.assertTrue(res.ok, res.message)
    self.assertCountEqual(
        shaped.narrowed_elements, ['left_leg_0', 'right_leg_0']
    )
    self.assertTrue(shaped.pattern.local_flat_foldability())

    resolved_limbs = res.arguments['limbs']
    eval_calls.clear()
    replayed, rep_res = toolbox.apply_tool(
        state,
        'thin_limbs_3d',
        {'limbs': resolved_limbs, 'evaluate': _evaluate},
    )
    self.assertTrue(rep_res.ok, rep_res.message)
    self.assertLessEqual(len(eval_calls), 1)
    self.assertEqual(_crease_keys(replayed), _crease_keys(shaped))

    sym_state, r_sym = toolbox.apply_tool(
        state,
        'apply_symmetric',
        {
            'tool': 'simple_fold',
            'element_name': 'left_leg_0',
            'arguments': {'angle': 35.0, 'depth': 0.3},
        },
    )
    self.assertTrue(r_sym.ok, r_sym.message)
    sym_replay, r_sym_rep = toolbox.apply_tool(
        state, 'apply_symmetric', dict(r_sym.arguments)
    )
    self.assertTrue(r_sym_rep.ok, r_sym_rep.message)
    self.assertEqual(_crease_keys(sym_replay), _crease_keys(sym_state))

    _, r_global = toolbox.apply_tool(
        state,
        'apply_symmetric',
        {'tool': 'fold_spine', 'element_name': 'left_leg_0'},
    )
    self.assertFalse(r_global.ok)

  def test_curve_flap_spread_fork_radial_fan_and_tree_edit(self):
    human = _state('HUMAN_SIMPLE')

    curved, r_curve = toolbox.apply_tool(
        human,
        'curve_flap',
        {
            'element_name': 'left_leg',
            'n_segments': 2,
            'total_bend_deg': 50.0,
            'direction': 'down',
            'verify_fold': False,
        },
    )
    self.assertTrue(r_curve.ok, r_curve.message)
    self.assertGreater(len(curved.pattern.creases), len(human.pattern.creases))

    forked, r_fork = toolbox.apply_tool(
        human,
        'spread_fork',
        {
            'element_a': 'left_leg',
            'element_b': 'right_leg',
            'spread_angle': 35.0,
            'depth': 0.2,
        },
    )
    self.assertTrue(r_fork.ok, r_fork.message)
    self.assertGreater(len(forked.pattern.creases), len(human.pattern.creases))

    fanned, r_fan = toolbox.apply_tool(
        human,
        'radial_fan',
        {
            'elements': ['left_arm', 'head', 'right_arm'],
            'arc_deg': 140.0,
            'dome_el': 20.0,
            'fan_wide': False,
        },
    )
    self.assertTrue(r_fan.ok, r_fan.message)
    self.assertGreater(len(fanned.pattern.creases), len(human.pattern.creases))

    edited, r_edit = toolbox.apply_tool(
        human,
        'tree_edit',
        {
            'subdivisions': {'left_leg': [2.0, 2.0], 'right_leg': [2.0, 2.0]},
            'elevations': {
                'left_leg_0': -35.0,
                'left_leg_1': -80.0,
                'right_leg_0': -35.0,
                'right_leg_1': -80.0,
            },
            'only_edited': True,
        },
    )
    self.assertTrue(r_edit.ok, r_edit.message)
    self.assertEqual(
        edited.tree['left_leg_1'].symmetric_stick_name, 'right_leg_1'
    )
    self.assertTrue(edited.pattern.local_flat_foldability())

    _, r_bad_edit = toolbox.apply_tool(
        human,
        'tree_edit',
        {'subdivisions': {'left_leg': [2.0, 5.0]}},
    )
    self.assertFalse(r_bad_edit.ok)

  def test_apply_tools_stop_on_error_and_probe(self):
    state = _state('HUMAN_SIMPLE')
    calls = [
        {'tool': 'simple_fold', 'arguments': {'element_name': 'missing'}},
        {
            'tool': 'simple_fold',
            'arguments': {'element_name': 'right_arm', 'angle': 45.0},
        },
    ]
    cont_state, cont_res = toolbox.apply_tools(state, calls)
    self.assertLen(cont_res, 2)
    self.assertFalse(cont_res[0].ok)
    self.assertTrue(cont_res[1].ok)
    self.assertIn('simple_fold', toolbox.history_json(cont_state))

    stopped_state, stopped_res = toolbox.apply_tools(
        state, calls, stop_on_error=True
    )
    self.assertLen(stopped_res, 1)
    self.assertIs(stopped_state.pattern, state.pattern)

    probed = toolbox.probe(state, element_names=['left_arm'])
    self.assertIn('left_arm', probed)
    self.assertIn('simple_fold', probed['left_arm']['available_tools'])
    self.assertEmpty(state.history)

  def test_squash_fold_bilateral_partial_uses_sequential_ridges(self):
    state = _state('BILATERAL_TAIL')
    flap = fan_squash._require_flap(state, 'tail')
    ridges = fan_squash.squash_ridges(flap)
    counts = fan_squash.ridge_segment_counts(state.pattern, flap)
    pairs = fan_squash._mirror_ridge_pairs(flap, ridges, counts)
    self.assertNotEmpty(pairs)

    real_open = fan_squash._open_squash_pattern
    real_accept = state_lib.accept_pattern
    kite_patterns: list[object] = []

    def _tag_kite(*args, **kwargs):
      out = real_open(*args, **kwargs)
      if kwargs.get('strategy') == 'kite_and_pleats':
        kite_patterns.append(out)
      return out

    def _reject_kite(st, shaped, *args, **kwargs):
      if any(shaped is p for p in kite_patterns):
        return None, 'simulated bilateral kite_and_pleats tear (>2%)', None
      return real_accept(st, shaped, *args, **kwargs)

    with (
        mock.patch.object(
            fan_squash, '_open_squash_pattern', side_effect=_tag_kite
        ),
        mock.patch.object(
            state_lib, 'accept_pattern', side_effect=_reject_kite
        ),
    ):
      squashed, res = toolbox.apply_tool(
          state,
          'squash_fold',
          {'element_name': 'tail', 'open_percent': 0.75},
      )

    self.assertTrue(res.ok, res.message)
    self.assertIn('mirror ridges', res.message)
    self.assertTrue(overlay_3d.has_3d_overlay(squashed.pattern))
    ok, msg, _ = toolbox.verify_3d_fold(squashed.pattern)
    self.assertTrue(ok, msg)

  def test_fan_out_fold_ranks_width_over_pleats(self):
    state = _state('HUMAN_SIMPLE')
    wider_fewer = fan_squash._FanCandidate(
        state=state,
        width_ratio=2.10,
        new_creases=19,
        pleats=2,
        label='horizontal/mountain',
        weak=False,
    )
    narrower_more = fan_squash._FanCandidate(
        state=state,
        width_ratio=1.14,
        new_creases=13,
        pleats=3,
        label='horizontal/mountain+valley',
        weak=False,
    )
    self.assertTrue(wider_fewer.beats(narrower_more))
    self.assertFalse(narrower_more.beats(wider_fewer))

    tied_fewer = fan_squash._FanCandidate(
        state=state,
        width_ratio=2.10,
        new_creases=19,
        pleats=2,
        label='horizontal/mountain',
        weak=False,
    )
    tied_more = fan_squash._FanCandidate(
        state=state,
        width_ratio=2.10,
        new_creases=19,
        pleats=4,
        label='horizontal/mountain',
        weak=False,
    )
    self.assertTrue(tied_more.beats(tied_fewer))

  def test_fan_out_fold_evaluates_opposite_delta(self):
    state = _state('HUMAN_SIMPLE')
    real_build = fan_squash._build_fan_plan
    seen_move_deltas: list[float] = []
    tagged_plans: list[tuple[object, float]] = []

    def _spy_build(*args, **kwargs):
      delta = float(kwargs['move_delta'])
      seen_move_deltas.append(delta)
      plan = real_build(*args, **kwargs)
      if plan is not None:
        tagged_plans.append((plan, delta))
      return plan

    def _fake_eval(st, element_name, plan, *args, **kwargs):
      del element_name, args, kwargs
      delta = next((d for p, d in tagged_plans if p is plan), 1.0)
      if delta > 0:
        return fan_squash._FanCandidate(
            state=st,
            width_ratio=1.50,
            new_creases=12,
            pleats=3,
            label=plan.label,
            weak=False,
        )
      return fan_squash._FanCandidate(
          state=st,
          width_ratio=2.53,
          new_creases=18,
          pleats=4,
          label=plan.label,
          weak=False,
      )

    with (
        mock.patch.object(
            fan_squash, '_build_fan_plan', side_effect=_spy_build
        ),
        mock.patch.object(fan_squash, '_evaluate_plan', side_effect=_fake_eval),
    ):
      _, res = toolbox.apply_tool(
          state,
          'fan_out_fold',
          {
              'element_name': 'left_leg',
              'orientation': 'horizontal',
              'move_delta': 1.0,
              'verify_fold': False,
          },
      )

    self.assertTrue(res.ok, res.message)
    self.assertIn(1.0, seen_move_deltas)
    self.assertIn(-1.0, seen_move_deltas)
    self.assertIn('2.53x', res.message)


if __name__ == '__main__':
  absltest.main()
