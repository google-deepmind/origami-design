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

import dataclasses
import json
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import bp_pipeline as bp
from bp_pipeline.shaping import heuristics
from bp_pipeline.shaping import recipes
from bp_pipeline.shaping import toolbox
from bp_pipeline.stick_figures import stick_figure as stick_figure_lib

_SOLVED_CACHE: dict[str, bp.SolvedCreasePattern] = {}
_STATE_CACHE: dict[str, toolbox.ShapingState] = {}


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
  """Packs and solves a small stick-figure fixture deterministically."""
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


def _solved(tree_name: str) -> bp.SolvedCreasePattern:
  if tree_name not in _SOLVED_CACHE:
    if tree_name == 'quadruped_fixture':
      sf = _build_quadruped_fixture()
    else:
      sf = bp.load_example(tree_name.upper())
    _SOLVED_CACHE[tree_name] = _solve_fixture(sf)
  return _SOLVED_CACHE[tree_name]


def _state(tree_name: str) -> toolbox.ShapingState:
  if tree_name not in _STATE_CACHE:
    _STATE_CACHE[tree_name] = toolbox.ShapingState.from_solved(
        _solved(tree_name)
    )
  return _STATE_CACHE[tree_name]


class PoseConfigAndPlanTest(parameterized.TestCase):

  def test_pose_config_defaults_and_validation(self):
    default_cfg = recipes.PoseConfig()
    self.assertEqual(default_cfg.mode, 'auto')
    self.assertFalse(default_cfg.narrow)
    self.assertEqual(default_cfg.time_budget_s, 60.0)
    self.assertEqual(default_cfg.feedback_rounds, 2)
    self.assertEqual(default_cfg.min_gain, 0.05)

    with self.assertRaises(ValueError):
      recipes.PoseConfig(mode='unknown')  # type: ignore[arg-type]
    with self.assertRaises(ValueError):
      recipes.PoseConfig(time_budget_s=0.0)
    with self.assertRaises(ValueError):
      recipes.PoseConfig(feedback_rounds=-1)
    with self.assertRaises(ValueError):
      recipes.PoseConfig(min_gain=-0.1)

  @parameterized.parameters('human_simple', 'spider_simple', 'embedded_engine')
  def test_pose_plan_covers_all_sticks_with_valid_tools(self, tree_name: str):
    state = _state(tree_name)
    plans = recipes.pose_plan(state)
    self.assertCountEqual(
        [p.name for p in plans], [s.name for s in state.tree.children]
    )
    roots = [p for p in plans if p.role == recipes.ROLE_ROOT]
    self.assertLen(roots, 1)
    self.assertEqual(roots[0].name, state.tree.root)
    self.assertLessEqual(set(roots[0].recommended_tools), {'fold_spine'})

    for plan in plans:
      json.dumps(plan.asdict())
      for tool_name in plan.recommended_tools:
        self.assertIn(tool_name, toolbox.TOOL_NAMES)

  def test_specialized_roles_and_symmetric_lateral_helpers(self):
    state = _state('human_simple')
    tree_dict = state.tree.to_dict()
    tree_dict['children'].extend([
        {
            'name': 'tail_base',
            'start': 'hips',
            'end': 'tail_mid',
            'azimuth_angle': 180.0,
            'elevation_angle': 45.0,
            'length': 2,
            'width': 1.0,
            'shaping_flap': False,
        },
        {
            'name': 'tail_stinger',
            'start': 'tail_mid',
            'end': 'tail_tip',
            'azimuth_angle': 180.0,
            'elevation_angle': -35.0,
            'length': 2,
            'width': 1.0,
            'shaping_flap': False,
        },
        {
            'name': 'claw_left',
            'start': 'left_hand',
            'end': 'claw_left_tip',
            'azimuth_angle': 45.0,
            'elevation_angle': 10.0,
            'length': 2,
            'width': 2.0,
            'shaping_flap': False,
        },
        {
            'name': 'claw_right',
            'start': 'right_hand',
            'end': 'claw_right_tip',
            'azimuth_angle': -45.0,
            'elevation_angle': 10.0,
            'length': 2,
            'width': 2.0,
            'shaping_flap': False,
        },
        {
            'name': 'dorsal_plate',
            'start': 'tail_mid',
            'end': 'dorsal_plate_tip',
            'azimuth_angle': 180.0,
            'elevation_angle': 80.0,
            'length': 1,
            'width': 1.0,
            'shaping_flap': True,
        },
    ])
    aug_tree = stick_figure_lib.StickFigure.from_dict(tree_dict)
    plans = {p.name: p for p in recipes.pose_plan(state.replace(tree=aug_tree))}
    self.assertEqual(plans['tail_stinger'].role, recipes.ROLE_TIP_DOWN)
    self.assertEqual(plans['claw_left'].role, recipes.ROLE_WIDE_TIP)
    self.assertEqual(plans['dorsal_plate'].role, recipes.ROLE_SKIP)
    self.assertEqual(
        plans['dorsal_plate'].recommended_tools, ('shape_flap_template',)
    )

    # `figure_pose` excludes `shaping_flap` sticks from `rotation_3d` targets.
    asym_knee = bp.StickFigure.from_dict({
        'name': 'asym_knee_pose_fixture',
        'category': 'animals',
        'complexity': 1,
        'root': 'hub_arm',
        'children': [
            {
                'name': 'hub_arm',
                'start': 'j_hub',
                'end': 'j_k',
                'length': 2,
                'width': 1,
                'azimuth_angle': 30.0,
                'elevation_angle': -10.0,
                'shaping_flap': False,
            },
            {
                'name': 'tip',
                'start': 'j_k',
                'end': 'j_tip',
                'length': 2,
                'width': 1,
                'azimuth_angle': 40.0,
                'elevation_angle': -45.0,
                'shaping_flap': False,
            },
            {
                'name': 'cap',
                'start': 'j_k',
                'end': 'j_cap',
                'length': 1,
                'width': 1,
                'azimuth_angle': 15.0,
                'elevation_angle': 35.0,
                'shaping_flap': True,
            },
            {
                'name': 'leg_a',
                'start': 'j_hub',
                'end': 'j_la',
                'length': 2,
                'width': 1,
                'azimuth_angle': 150.0,
                'elevation_angle': -50.0,
                'shaping_flap': False,
            },
            {
                'name': 'leg_b',
                'start': 'j_hub',
                'end': 'j_lb',
                'length': 2,
                'width': 1,
                'azimuth_angle': -130.0,
                'elevation_angle': -50.0,
                'shaping_flap': False,
            },
        ],
    })
    asym_packed = bp.pack(
        bp.prepare(asym_knee),
        bp.PackingTaskConfig(
            max_retries=4, max_cps_per_grid_size=1, max_number_of_cps=1
        ),
    )
    asym_solved = bp.solve_best(
        asym_packed,
        bp.SolverConfig(
            check_global_flat_foldability=True,
            max_number_of_cps=1,
            max_time_limit=10.0,
        ),
    )
    asym_posed, _ = recipes.figure_pose(
        toolbox.ShapingState.from_solved(asym_solved), feedback_rounds=0
    )
    rotated_elements: set[str] = set()
    for entry in asym_posed.history:
      if entry.get('tool') == 'rotation_3d':
        for rot in entry.get('arguments', {}).get('rotations', ()):
          rotated_elements.add(rot['element_name'])
    self.assertNotEmpty(rotated_elements)
    self.assertNotIn('cap', rotated_elements)

    quad = _state('quadruped_fixture')
    bias_tree = quad.tree.update_tree(
        azimuth_updates={'rear_leg_l': 25.0, 'rear_leg_r': -25.0},
        elevation_updates={'rear_leg_l': -75.0, 'rear_leg_r': -75.0},
        width_updates={},
    )
    biases = recipes.lateral_bias(bias_tree)
    self.assertIn('rear_leg_l', biases)
    self.assertAlmostEqual(biases['rear_leg_l'], -biases['rear_leg_r'])

    base_targets = {
        'front_shin_l': (10.0, -80.0),
        'front_shin_r': (-10.0, -80.0),
    }
    fixed = recipes.side_fix_targets(
        quad.tree, ['front_shin_l'], base_targets, step_deg=25.0
    )
    self.assertGreater(fixed['front_shin_l'][0], 10.0)
    self.assertAlmostEqual(fixed['front_shin_l'][0], -fixed['front_shin_r'][0])
    self.assertGreater(fixed['front_shin_l'][1], -80.0)


class PoseModesTest(parameterized.TestCase):

  @parameterized.parameters('human_simple', 'spider_simple')
  def test_greedy_pose_is_fast_and_flat_foldable(self, tree_name: str):
    state = _state(tree_name)
    posed, diag = recipes.greedy_pose(state, narrow=True)
    self.assertEqual(diag['mode'], 'greedy')
    self.assertGreater(len(posed.pattern.creases), len(state.pattern.creases))
    self.assertTrue(posed.pattern.local_flat_foldability())
    json.dumps(diag)

  def test_figure_pose_aligns_quadruped_without_monkeypatching(self):
    state = _state('quadruped_fixture')
    original_root_fn = heuristics.get_first_stick_for_shaping

    posed, diag = recipes.figure_pose(state, feedback_rounds=1, narrow=True)
    self.assertIs(heuristics.get_first_stick_for_shaping, original_root_fn)
    self.assertTrue(posed.pattern.local_flat_foldability())
    self.assertTrue(diag['clean_gate'], diag)
    self.assertGreaterEqual(diag['fidelity'], diag['base_fidelity'])
    rep, dirs = recipes.realized_directions(posed)
    self.assertGreaterEqual(rep['components']['direction'], 0.70)
    self.assertIn('front_shin_l', dirs)
    json.dumps(diag)

  def test_search_pose_custom_evaluator_and_real_fidelity(self):
    state = _state('human_simple')
    seen: list[int] = []

    def _mock_eval(candidate: toolbox.ShapingState) -> float:
      seen.append(len(candidate.pattern.creases))
      if len(seen) == 1:
        return 2.0
      if len(seen) == 2:
        return 5.5
      return 4.0

    searched, mock_diag = recipes.search_pose(
        state, time_budget_s=15.0, min_gain=0.1, evaluate=_mock_eval
    )
    self.assertGreater(
        len(searched.pattern.creases), len(state.base_pattern.creases)
    )
    self.assertEqual(mock_diag['base_fidelity'], 2.0)
    self.assertEqual(mock_diag['fidelity'], 5.5)
    self.assertEqual(mock_diag['accepted_steps'], 1)

    real_posed, real_diag = recipes.search_pose(state, time_budget_s=12.0)
    self.assertTrue(real_diag['clean_gate'], real_diag)
    self.assertGreaterEqual(real_diag['fidelity'], real_diag['base_fidelity'])
    self.assertGreaterEqual(
        len(real_posed.pattern.creases), len(state.base_pattern.creases)
    )
    json.dumps(real_diag)

  @parameterized.named_parameters(
      dict(
          testcase_name='greedy',
          config=recipes.PoseConfig(mode='greedy'),
          expected_strategy=bp.ShapingStrategy.POSE_GREEDY,
      ),
      dict(
          testcase_name='figure',
          config=recipes.PoseConfig(mode='figure', feedback_rounds=1),
          expected_strategy=bp.ShapingStrategy.POSE_FIGURE,
      ),
      dict(
          testcase_name='search',
          config=recipes.PoseConfig(mode='search', time_budget_s=6.0),
          expected_strategy=bp.ShapingStrategy.POSE_SEARCH,
      ),
      dict(
          testcase_name='auto',
          config=recipes.PoseConfig(
              mode='auto', feedback_rounds=1, time_budget_s=10.0
          ),
          expected_strategy=bp.ShapingStrategy.POSE_AUTO,
      ),
  )
  def test_pose_solved_all_modes(
      self,
      config: recipes.PoseConfig,
      expected_strategy: bp.ShapingStrategy,
  ):
    solved = _solved('human_simple')
    shaped, report = recipes.pose_solved(solved, config)
    self.assertIsInstance(shaped, bp.ShapedCreasePattern)
    self.assertEqual(shaped.strategy, expected_strategy)
    self.assertEqual(shaped.diagnostics['mode'], config.mode)
    self.assertEqual(report['mode'], config.mode)
    self.assertIn('time_taken', report)
    json.dumps(report)
    json.dumps(bp.to_dict(shaped))

  def test_figure_pose_falls_back_to_unbiased_when_bias_tears(self):
    quad = _state('quadruped_fixture')
    bias_tree = quad.tree.update_tree(
        azimuth_updates={'rear_leg_l': 25.0, 'rear_leg_r': -25.0},
        elevation_updates={'rear_leg_l': -75.0, 'rear_leg_r': -75.0},
        width_updates={},
    )
    biased_state = quad.replace(tree=bias_tree, base_tree=bias_tree)
    biases = recipes.lateral_bias(biased_state.tree)
    self.assertNotEmpty(biases)

    real_run_rotation = recipes._run_rotation
    real_score_state = recipes._score_state
    biased_patterns: list[object] = []

    def _tag_biased_rotation(base_state, targets, **kwargs):
      out = real_run_rotation(base_state, targets, **kwargs)
      if out is not None and any(
          abs(targets[k][0] - biased_state.tree[k].azimuth_angle) > 0.1
          for k in biases
          if k in targets
      ):
        biased_patterns.append(out.pattern)
      return out

    def _tear_on_biased(cand):
      if any(cand.pattern is p for p in biased_patterns):
        return recipes._ScoreSummary(
            score=4.5,
            clean_ok=False,
            pose_ok=False,
            direction=0.6,
            num_crossings=0,
            vertex_error=8.07,
            report={'error': 'simulated lateral_bias tear'},
        )
      return real_score_state(cand)

    with (
        mock.patch.object(
            recipes, '_run_rotation', side_effect=_tag_biased_rotation
        ),
        mock.patch.object(recipes, '_score_state', side_effect=_tear_on_biased),
    ):
      posed, diag = recipes.figure_pose(biased_state, feedback_rounds=0)
      auto_posed, auto_diag = recipes.pose_state(
          biased_state,
          recipes.PoseConfig(
              mode='auto', feedback_rounds=0, time_budget_s=1e-6
          ),
      )

    self.assertNotEmpty(biased_patterns)
    self.assertTrue(diag.get('unbiased_fallback'), diag)
    self.assertTrue(diag['clean_gate'], diag)
    self.assertGreater(
        len(posed.pattern.creases), len(biased_state.base_pattern.creases)
    )
    self.assertGreater(diag['fidelity'], diag['base_fidelity'])
    self.assertTrue(auto_diag.get('unbiased_fallback'), auto_diag)
    self.assertGreater(
        len(auto_posed.pattern.creases), len(biased_state.base_pattern.creases)
    )

  def test_pose_state_does_not_abort_figure_pose_on_time_budget(self):
    state = _state('quadruped_fixture')
    real_figure_pose = recipes.figure_pose
    seen_deadlines: list[float | None] = []

    def _spy_figure_pose(*args, **kwargs):
      seen_deadlines.append(kwargs.get('deadline'))
      return real_figure_pose(*args, **kwargs)

    with (
        mock.patch.object(recipes, 'figure_pose', side_effect=_spy_figure_pose),
        mock.patch.object(
            recipes, 'search_pose', wraps=recipes.search_pose
        ) as mock_search,
    ):
      fig_posed, fig_diag = recipes.pose_state(
          state,
          recipes.PoseConfig(
              mode='figure', feedback_rounds=1, time_budget_s=1e-6
          ),
      )
      auto_posed, auto_diag = recipes.pose_state(
          state,
          recipes.PoseConfig(
              mode='auto', feedback_rounds=1, time_budget_s=1e-6
          ),
      )

    self.assertEqual(seen_deadlines, [None, None])
    self.assertTrue(fig_diag['clean_gate'], fig_diag)
    self.assertGreater(
        len(fig_posed.pattern.creases), len(state.base_pattern.creases)
    )
    self.assertTrue(auto_diag['clean_gate'], auto_diag)
    self.assertGreater(
        len(auto_posed.pattern.creases), len(state.base_pattern.creases)
    )
    mock_search.assert_not_called()

  def test_pose_state_auto_skips_search_when_gates_pass_and_escalates_on_fail(
      self,
  ):
    state = _state('human_simple')
    with mock.patch.object(
        recipes, 'search_pose', wraps=recipes.search_pose
    ) as mock_search:
      _, pass_diag = recipes.pose_state(
          state,
          recipes.PoseConfig(
              mode='auto', feedback_rounds=1, time_budget_s=30.0
          ),
      )
      self.assertTrue(pass_diag['clean_gate'], pass_diag)
      self.assertTrue(pass_diag['pose_gate'], pass_diag)
      mock_search.assert_not_called()

    real_score_state = recipes._score_state

    def _fail_pose_gate(cand):
      return dataclasses.replace(real_score_state(cand), pose_ok=False)

    with (
        mock.patch.object(
            recipes, 'figure_pose', return_value=(state, {})
        ),
        mock.patch.object(
            recipes, '_score_state', side_effect=_fail_pose_gate
        ),
        mock.patch.object(
            recipes,
            'search_pose',
            return_value=(state, {'accepted_steps': 0}),
        ) as mock_search_escalate,
    ):
      recipes.pose_state(
          state,
          recipes.PoseConfig(
              mode='auto', feedback_rounds=0, time_budget_s=30.0
          ),
      )
      mock_search_escalate.assert_called_once()


if __name__ == '__main__':
  absltest.main()
