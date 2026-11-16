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

"""Unit tests for `bp_pipeline.shaping.recipes`."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline.shaping import fidelity
from bp_pipeline.shaping import recipes
from bp_pipeline.shaping import toolbox as shaping
from bp_pipeline.stick_figures import stick_figure as stick_figure_lib

_STATES: dict[str, shaping.ShapingState] = {}


def _build_quadruped_fixture() -> bp.StickFigure:
  """Builds a small symmetric quadruped with root `pelvis` (`az=180`)."""
  return bp.StickFigure.from_dict({
      'name': 'quadruped_fixture',
      'category': 'animals',
      'complexity': 2,
      'color_top': [200, 180, 150],
      'color_bottom': [200, 180, 150],
      'root': 'pelvis',
      'children': [
          {
              'name': 'pelvis',
              'start': 'chest',
              'end': 'hips',
              'azimuth_angle': 180.0,
              'elevation_angle': 0.0,
              'length': 3,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'head',
              'start': 'chest',
              'end': 'snout',
              'azimuth_angle': 0.0,
              'elevation_angle': 35.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'front_arm_l',
              'start': 'chest',
              'end': 'elbow_l',
              'azimuth_angle': 90.0,
              'elevation_angle': -45.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'front_shin_l',
              'start': 'elbow_l',
              'end': 'hoof_l',
              'azimuth_angle': 90.0,
              'elevation_angle': -85.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'front_arm_r',
              'start': 'chest',
              'end': 'elbow_r',
              'azimuth_angle': -90.0,
              'elevation_angle': -45.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'front_shin_r',
              'start': 'elbow_r',
              'end': 'hoof_r',
              'azimuth_angle': -90.0,
              'elevation_angle': -85.0,
              'length': 2,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'rear_leg_l',
              'start': 'hips',
              'end': 'rear_hoof_l',
              'azimuth_angle': 90.0,
              'elevation_angle': -75.0,
              'length': 3,
              'width': 1,
              'shaping_flap': False,
          },
          {
              'name': 'rear_leg_r',
              'start': 'hips',
              'end': 'rear_hoof_r',
              'azimuth_angle': -90.0,
              'elevation_angle': -75.0,
              'length': 3,
              'width': 1,
              'shaping_flap': False,
          },
      ],
  })


def _solve_fixture(sf: bp.StickFigure) -> bp.SolvedCreasePattern:
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
                remove_central_crease=False,
                up_down_method='flaps_down',
            ),
        )
    )
    if candidates:
      return candidates[0]
  return bp.fold(sf, preset='demo').solved


def _state(tree_name: str) -> shaping.ShapingState:
  if tree_name not in _STATES:
    if tree_name == 'quadruped_fixture':
      sf = _build_quadruped_fixture()
    else:
      sf = bp.load_example(tree_name.upper())
    _STATES[tree_name] = shaping.ShapingState.from_solved(_solve_fixture(sf))
  return _STATES[tree_name]


class PosePlanTest(parameterized.TestCase):

  def test_nudged_moves_angles_off_anchors(self):
    self.assertEqual(recipes._nudged(0.0), 18.0)
    self.assertEqual(recipes._nudged(-5.0), -18.0)
    self.assertEqual(recipes._nudged(90.0), 100.0)
    self.assertEqual(recipes._nudged(85.0), 80.0)
    self.assertEqual(recipes._nudged(-178.0), -170.0)
    self.assertEqual(recipes._nudged(45.0), 45.0)

  @parameterized.parameters('human_simple', 'spider_simple', 'embedded_engine')
  def test_plan_covers_every_element_once(self, tree_name):
    state = _state(tree_name)
    plans = recipes.pose_plan(state)
    names = [plan.name for plan in plans]
    self.assertCountEqual(names, [s.name for s in state.tree.children])
    roots = [plan for plan in plans if plan.role == recipes.ROLE_ROOT]
    self.assertLen(roots, 1)
    self.assertEqual(roots[0].name, state.tree.root)
    self.assertEmpty(roots[0].candidates)
    for plan in plans:
      if plan.role in (
          recipes.ROLE_ROOT,
          recipes.ROLE_RIVER,
          recipes.ROLE_SKIP,
      ):
        self.assertEmpty(plan.candidates)
      else:
        self.assertNotEmpty(plan.candidates)
        for candidate in plan.candidates:
          self.assertIn(candidate['name'], shaping.TOOL_NAMES)
          self.assertEqual(candidate['arguments']['element_name'], plan.name)

  @parameterized.parameters('human_simple', 'spider_simple', 'embedded_engine')
  def test_limb_roles_follow_width(self, tree_name):
    state = _state(tree_name)
    plans = {plan.name: plan for plan in recipes.pose_plan(state)}
    limbs = [
        plan
        for plan in plans.values()
        if plan.role in (recipes.ROLE_THIN_LIMB, recipes.ROLE_LIMB)
    ]
    self.assertNotEmpty(limbs)
    for plan in limbs:
      width = state.tree[plan.name].width
      first = plan.candidates[0]
      self.assertEqual(first['name'], 'simple_fold')
      if plan.role == recipes.ROLE_THIN_LIMB:
        self.assertLessEqual(width, 0.5)
        self.assertTrue(first['arguments']['narrow'])
      else:
        self.assertGreater(width, 0.5)
        self.assertNotIn('narrow', first['arguments'])

  def test_chain_tip_down_and_azimuth_flip_roles(self):
    state = _state('human_simple')
    tree_dict = state.tree.to_dict()
    tree_dict['children'].extend([
        {
            'name': 'tail_base',
            'start': 'hips',
            'end': 'tail_mid',
            'azimuth_angle': 180.0,
            'elevation_angle': 40.0,
            'length': 2,
            'width': 1.0,
            'shaping_flap': False,
        },
        {
            'name': 'tail_stinger',
            'start': 'tail_mid',
            'end': 'tail_tip',
            'azimuth_angle': 180.0,
            'elevation_angle': -20.0,
            'length': 2,
            'width': 1.0,
            'shaping_flap': False,
        },
    ])
    state_with_tail = state.replace(
        tree=stick_figure_lib.StickFigure.from_dict(tree_dict)
    )
    plans = {plan.name: plan for plan in recipes.pose_plan(state_with_tail)}
    self.assertEqual(plans['tail_stinger'].role, recipes.ROLE_TIP_DOWN)
    self.assertEqual(
        plans['tail_stinger'].candidates[0]['name'], 'reverse_fold'
    )

    arch_dict = state.tree.to_dict()
    arch_dict['children'].extend([
        {
            'name': 'tail_1',
            'start': 'hips',
            'end': 'tail_j1',
            'azimuth_angle': 180.0,
            'elevation_angle': 75.0,
            'length': 2,
            'width': 1.0,
            'shaping_flap': False,
        },
        {
            'name': 'tail_2',
            'start': 'tail_j1',
            'end': 'tail_j2',
            'azimuth_angle': 0.0,
            'elevation_angle': 70.0,
            'length': 2,
            'width': 1.0,
            'shaping_flap': False,
        },
    ])
    state_with_arch = state.replace(
        tree=stick_figure_lib.StickFigure.from_dict(arch_dict)
    )
    arch_plans = {
        plan.name: plan for plan in recipes.pose_plan(state_with_arch)
    }
    self.assertEqual(arch_plans['tail_2'].role, recipes.ROLE_SEGMENT)


class ApplyAndSearchPoseTest(parameterized.TestCase):

  @parameterized.parameters('human_simple', 'spider_simple')
  def test_pose_keeps_flat_foldability_and_reports(self, tree_name):
    state = _state(tree_name)
    posed, plans, report = recipes.pose(state)
    self.assertGreater(len(posed.pattern.creases), len(state.pattern.creases))
    self.assertTrue(posed.pattern.local_flat_foldability())
    self.assertLen(report, len(plans))
    applied = [entry for entry in report if entry['applied']]
    self.assertNotEmpty(applied)
    self.assertNotEmpty(posed.history)
    self.assertLessEqual(len(posed.history), len(applied))

  def test_search_pose_mock_and_budget_fallback(self):
    state = _state('human_simple')
    calls = []

    def mock_evaluate(candidate_state: shaping.ShapingState):
      calls.append(len(candidate_state.pattern.creases))
      if len(calls) == 1:
        return 2.0, {'fidelity': 2.0}
      if len(calls) == 2:
        return 5.5, {'fidelity': 5.5}
      if len(calls) == 3:
        return -1.0, {'error': 'fold failed'}
      return 4.0, {'fidelity': 4.0}

    posed, plans, report, summary = recipes.search_pose(
        state, time_budget_s=60.0, evaluate=mock_evaluate
    )
    self.assertGreater(
        len(posed.pattern.creases), len(state.base_pattern.creases)
    )
    self.assertLen(report, len(plans))
    self.assertEqual(summary['base_fidelity'], 2.0)
    self.assertEqual(summary['fidelity'], 5.5)
    self.assertEqual(summary['candidates_kept'], 1)
    self.assertFalse(summary['budget_exhausted'])

    fb_posed, _, _, fb_summary = recipes.search_pose(
        state,
        time_budget_s=-1.0,
        evaluate=lambda s: (3.0 + 0.001 * len(s.pattern.creases), {}),
    )
    self.assertTrue(fb_summary['budget_exhausted'])
    self.assertGreater(
        len(fb_posed.pattern.creases), len(state.base_pattern.creases)
    )

  def test_wide_tip_thin_limb_and_body_groups(self):
    state = _state('human_simple')
    tree_dict = state.tree.to_dict()
    tree_dict['children'].extend([
        {
            'name': 'arm_ext_L',
            'start': 'left_hand',
            'end': 'claw_L_tip',
            'azimuth_angle': 45.0,
            'elevation_angle': 10.0,
            'length': 2,
            'width': 2.0,
            'shaping_flap': True,
        },
        {
            'name': 'arm_ext_R',
            'start': 'right_hand',
            'end': 'claw_R_tip',
            'azimuth_angle': -45.0,
            'elevation_angle': 10.0,
            'length': 2,
            'width': 2.0,
            'shaping_flap': True,
        },
    ])
    wide_state = state.replace(
        tree=stick_figure_lib.StickFigure.from_dict(tree_dict)
    )
    plans = {p.name: p for p in recipes.pose_plan(wide_state)}
    self.assertEqual(plans['arm_ext_L'].role, recipes.ROLE_WIDE_TIP)
    groups = recipes.pose_groups(wide_state, list(plans.values()))
    pair_groups = [g for g in groups if 'arm_ext_L' in g.elements]
    self.assertLen(pair_groups, 1)
    labels = {c.label for c in pair_groups[0].candidates}
    self.assertIn('rot3d+squash', labels)
    self.assertIn('rot3d_all', labels)

  def test_search_pose_and_sculpt_model_improve_fidelity(self):
    state = _state('human_simple')
    posed, _, report, summary = recipes.search_pose(state, time_budget_s=20.0)
    self.assertTrue(posed.pattern.local_flat_foldability())
    self.assertGreaterEqual(summary['fidelity'], summary['base_fidelity'])
    self.assertTrue(any(entry['applied'] for entry in report))

    sculpted, sculpt_rep = recipes.sculpt_model(
        state,
        max_vertex_error=0.35,
        require_flat_foldable=True,
        time_budget_s=15.0,
    )
    self.assertTrue(sculpt_rep['clean_gate']['passed'], sculpt_rep)
    self.assertEqual(sculpt_rep['non_flat_foldable_vertices'], 0)
    self.assertGreater(sculpt_rep['fidelity'], sculpt_rep['base_fidelity'])
    self.assertTrue(sculpted.pattern.local_flat_foldability())


class FigurePoseTest(absltest.TestCase):

  def test_designer_root_3d_and_figure_pose_on_quadruped(self):
    state = _state('quadruped_fixture')
    original = shaping.heuristics.get_first_stick_for_shaping
    with shaping._designer_root_3d(state):  # pylint: disable=protected-access
      self.assertEqual(
          shaping.heuristics.get_first_stick_for_shaping(state.tree).name,
          state.tree.root,
      )
    self.assertIs(shaping.heuristics.get_first_stick_for_shaping, original)

    posed, summary = recipes.figure_pose(state, iterations=1)
    self.assertTrue(posed.pattern.local_flat_foldability())
    self.assertGreaterEqual(summary['components']['direction'], 0.70)
    for name, info in summary['directions'].items():
      self.assertIn('target', info, name)
      self.assertIn('realized', info, name)

  def test_bend_to_direction_and_replay(self):
    state = _state('quadruped_fixture').reset()
    _, before = recipes.realized_directions(state)
    bent, result = shaping.apply_tool(
        state, 'bend_to_direction', {'element_name': 'front_shin_l'}
    )
    self.assertTrue(result.ok, result.message)
    _, after = recipes.realized_directions(bent)
    self.assertLessEqual(
        after['front_shin_l']['error'], before['front_shin_l']['error']
    )
    chosen = bent.history[-1]['arguments']['chosen']
    replayed = shaping.bend_to_direction(
        state, element_name='front_shin_l', chosen=chosen
    )
    self.assertLen(replayed.pattern.creases, len(bent.pattern.creases))

  def test_nudges_and_side_crossing_fix_helpers(self):
    nudges = recipes._skip_nudges(-148.0, 0.0)
    self.assertNotIn((-148.0, 0.0), nudges)
    self.assertEqual(nudges[:2], [(-126.0, 0.0), (-161.0, 0.0)])

    az_l, el_l = recipes.lateral_bias(30.0, -60.0, 1.0, 0.4)
    az_r, el_r = recipes.lateral_bias(-30.0, -60.0, -1.0, 0.4)
    self.assertAlmostEqual(az_l, -az_r)
    self.assertAlmostEqual(el_l, el_r)

    state = _state('quadruped_fixture')
    tree = state.tree
    crossing = {'stick': 'front_shin_l', 'design_side': 'left'}
    requested = {stick.name: None for stick in tree.children}
    targets = recipes.side_fix_targets(tree, [crossing], requested)
    self.assertEqual(targets, {'front_shin_l': 1.0, 'front_arm_l': 1.0})

    original = {
        name: (
            float(tree[name].azimuth_angle),
            float(tree[name].elevation_angle),
        )
        for name in ('front_shin_l', 'front_shin_r')
    }
    calls = []

    def fake_apply_tool(current, tool, arguments):
      self.assertEqual(tool, 'rotation_3d')
      calls.append({
          r['element_name']: (r['az'], r['el']) for r in arguments['rotations']
      })
      return current, shaping.ToolResult(
          tool='rotation_3d',
          arguments=dict(arguments),
          ok=True,
          message='rotation_3d succeeded.',
          num_creases=0,
      )

    def fake_realized_directions(_):
      req = calls[-1]
      crossings = [
          {
              'stick': name,
              'point': 'tip',
              'kind': 'crossed',
              'design_side': 'left' if name.endswith('_l') else 'right',
              'design_offset': 1.0,
              'realized_offset': -0.5,
          }
          for name in original
          if req[name] == original[name]
      ]
      rep = {
          'fidelity': 5.0,
          'components': {'direction': 1.0, 'width': 0.5},
          'direction_error_deg': {name: 5.0 for name in req},
          'side_crossings': crossings,
          'pose_gate': {'passed': not crossings},
      }
      dirs = {
          name: {
              'target': list(req[name]),
              'realized': list(req[name]),
              'error': 5.0,
          }
          for name in req
      }
      return rep, dirs

    with mock.patch.object(shaping, 'apply_tool', fake_apply_tool):
      with mock.patch.object(
          recipes, 'realized_directions', fake_realized_directions
      ):
        _, fixed = recipes.figure_pose(state, iterations=0)
    self.assertEmpty(fixed['side_crossings'])
    self.assertTrue(fixed['pose_gate']['passed'])
    self.assertAlmostEqual(
        fixed['requested']['front_shin_l'][0],
        -fixed['requested']['front_shin_r'][0],
    )
    self.assertIsNotNone(fidelity.topology(tree))


if __name__ == '__main__':
  absltest.main()
